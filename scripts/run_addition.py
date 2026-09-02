#!/usr/bin/env python3
"""Run the focused addition experiment with resumable stages."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import torch
import yaml

from elk_generalization.utils import (
    DEVICE_CHOICES,
    DTYPE_CHOICES,
    get_quirky_model_name,
    resolve_device,
)

STAGES = (
    "validate",
    "extract",
    "probe",
    "summarize-replication",
    "intervene-matched",
    "intervene-cross",
    "summarize-extension",
)
PERSONAS = {"A": "Alice", "B": "Bob"}
EXTRACTION_FILES = (
    "hiddens.pt",
    "labels.pt",
    "alice_labels.pt",
    "bob_labels.pt",
    "lm_log_odds.pt",
)


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")
    required = {
        "dataset",
        "device",
        "dtype",
        "seed",
        "validation_examples",
        "test_examples",
        "intervention_examples",
        "layer_stride",
        "reporter",
        "templatization_method",
        "save_contrast_pairs",
        "run_root",
        "results_root",
        "cache_dir",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"Missing configuration fields: {sorted(missing)}")
    if config["dataset"] != "addition" or config["reporter"] != "mean-diff":
        raise ValueError("This runner supports addition with the mean-diff reporter only")
    if config.get("standardize_templates", False):
        raise ValueError("This focused configuration requires standardize_templates: false")
    if config.get("model_hub_user", "EleutherAI") != "EleutherAI":
        raise ValueError("This focused configuration requires model_hub_user: EleutherAI")
    if config["device"] not in DEVICE_CHOICES or config["dtype"] not in DTYPE_CHOICES:
        raise ValueError("device or dtype is not supported")
    if not isinstance(config["save_contrast_pairs"], bool):
        raise ValueError("save_contrast_pairs must be true or false")
    for field in (
        "seed",
        "validation_examples",
        "test_examples",
        "intervention_examples",
        "layer_stride",
    ):
        if (
            not isinstance(config[field], int)
            or isinstance(config[field], bool)
            or config[field] <= 0
        ):
            raise ValueError(f"{field} must be a positive integer")
    for field in ("run_root", "results_root", "cache_dir"):
        value = Path(config[field])
        if value.is_absolute() or ".." in value.parts:
            raise ValueError(f"{field} must be repository-relative")
    models = config.get("models")
    if models is None:
        models = [
            {
                "base_model": config["base_model"],
                "expected_layers": config["expected_layers"],
                "expected_hidden_size": config["expected_hidden_size"],
            }
        ]
    if not isinstance(models, list) or not models:
        raise ValueError("models must be a non-empty list")
    for model in models:
        if set(model) != {"base_model", "expected_layers", "expected_hidden_size"}:
            raise ValueError(
                "Each model needs base_model, expected_layers, and expected_hidden_size"
            )
        if model["expected_layers"] <= 0 or model["expected_hidden_size"] <= 0:
            raise ValueError("Expected model dimensions must be positive")
    config["models"] = models
    return config


def model_ids(config: dict[str, Any], model: dict[str, Any]) -> tuple[str, str]:
    return get_quirky_model_name(
        config["dataset"],
        model["base_model"],
        config["templatization_method"],
        config.get("standardize_templates", False),
        model_hub_user=config.get("model_hub_user", "EleutherAI"),
    )


def model_paths(config: dict[str, Any], model: dict[str, Any], repo: Path) -> dict[str, Path]:
    _, model_last = model_ids(config, model)
    run_root = repo / config["run_root"]
    return {
        "activations": run_root / "activations" / model_last,
        "interventions": run_root / "interventions" / model_last,
        "results": repo / config["results_root"] / model["base_model"].split("/")[-1],
    }


def check_extraction(
    root: Path, model: dict[str, Any], count: int, contrast_pairs: bool
) -> tuple[bool, str]:
    files = list(EXTRACTION_FILES) + (["ccs_hiddens.pt"] if contrast_pairs else [])
    missing = [name for name in files if not (root / name).is_file()]
    if missing:
        return False, f"missing {', '.join(missing)}"
    try:
        hiddens = torch.load(root / "hiddens.pt", map_location="cpu")
        shape = (count, model["expected_hidden_size"])
        if len(hiddens) != model["expected_layers"] or any(
            tuple(tensor.shape) != shape or not torch.isfinite(tensor).all()
            for tensor in hiddens
        ):
            return False, "hidden-state layers have unexpected shapes or values"
        for name in ("labels.pt", "alice_labels.pt", "bob_labels.pt", "lm_log_odds.pt"):
            tensor = torch.load(root / name, map_location="cpu")
            if tuple(tensor.shape) != (count,) or not torch.isfinite(tensor).all():
                return False, f"{name} has an unexpected shape or values"
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        return False, f"could not load artifacts: {error}"
    return True, "complete"


def check_probe(
    config: dict[str, Any],
    model: dict[str, Any],
    paths: dict[str, Path],
    source: str,
    targets: Sequence[str],
) -> tuple[bool, str]:
    reporter = paths["activations"] / source / "validation" / "mean-diff_reporters.pt"
    if not reporter.is_file():
        return False, "missing reporter weights"
    try:
        weights = torch.load(reporter, map_location="cpu")
        if len(weights) != model["expected_layers"] or any(
            weight.numel() != model["expected_hidden_size"]
            or not torch.isfinite(weight).all()
            for weight in weights
        ):
            return False, "reporter weights have unexpected shapes"
        for target in targets:
            output = (
                paths["activations"]
                / target
                / "test"
                / f"{source}_mean-diff_log_odds.pt"
            )
            scores = torch.load(output, map_location="cpu")
            if tuple(scores.shape) != (
                model["expected_layers"],
                config["test_examples"],
            ) or not torch.isfinite(scores).all():
                return False, f"{output.name} has an unexpected shape or values"
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as error:
        return False, f"could not load probe output: {error}"
    return True, "complete"


def intervention_dir(
    config: dict[str, Any], paths: dict[str, Path], source: str, target: str
) -> Path:
    name = f"mean-diff_{PERSONAS[source]}_to_{PERSONAS[target]}_0.0_1.0"
    return paths["interventions"] / name


def check_intervention(
    config: dict[str, Any],
    model: dict[str, Any],
    root: Path,
) -> tuple[bool, str]:
    if not (root / "summary.json").is_file() or not (root / "all_results.pt").is_file():
        return False, "missing summary.json or all_results.pt"
    try:
        with (root / "summary.json").open() as handle:
            summary = json.load(handle)
        expected_layers = list(
            range(model["expected_layers"] - 1, -1, -config["layer_stride"])
        )
        if [row["layer"] for row in summary] != expected_layers:
            return False, "summary has unexpected layers"
        results = torch.load(root / "all_results.pt", map_location="cpu")
        if len(results) != len(expected_layers) or any(
            len(row["clean_probs"]) != config["intervention_examples"] for row in results
        ):
            return False, "intervention results have unexpected sample counts"
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        return False, f"could not load intervention output: {error}"
    return True, "complete"


def run_command(command: list[str], config: dict[str, Any], repo: Path, dry_run: bool) -> None:
    print(f"Running: {shlex.join(command)}")
    if dry_run:
        return
    cache = repo / config["cache_dir"]
    cache.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["HF_HOME"] = str(cache)
    subprocess.run(command, cwd=repo, env=env, check=True)


def extraction_command(
    config: dict[str, Any],
    model: dict[str, Any],
    paths: dict[str, Path],
    repo: Path,
    persona: str,
    split: str,
    count: int,
) -> list[str]:
    model_id, _ = model_ids(config, model)
    command = [
        sys.executable,
        str(repo / "elk_generalization/elk/extract_hiddens.py"),
        "--model",
        model_id,
        "--dataset",
        f"EleutherAI/quirky_{config['dataset']}_raw",
        "--character",
        PERSONAS[persona],
        "--difficulty",
        "none",
        "--templatization-method",
        config["templatization_method"],
        "--save-path",
        str(paths["activations"] / persona),
        "--max-examples",
        str(count),
        "--splits",
        split,
        "--seed",
        str(config["seed"]),
        "--device",
        config["device"],
        "--dtype",
        config["dtype"],
    ]
    if not config["save_contrast_pairs"]:
        command.append("--skip-contrast-pairs")
    return command


def probe_command(
    config: dict[str, Any], paths: dict[str, Path], repo: Path, source: str, targets: Sequence[str]
) -> list[str]:
    return [
        sys.executable,
        str(repo / "elk_generalization/elk/transfer.py"),
        "--train-dir",
        str(paths["activations"] / source / "validation"),
        "--test-dirs",
        *(str(paths["activations"] / target / "test") for target in targets),
        "--reporter",
        "mean-diff",
        "--label-col",
        "labels",
        "--device",
        config["device"],
        "--verbose",
    ]


def intervention_command(
    config: dict[str, Any],
    model: dict[str, Any],
    paths: dict[str, Path],
    repo: Path,
    source: str,
    target: str,
) -> list[str]:
    return [
        sys.executable,
        str(repo / "elk_generalization/interventions/intervene.py"),
        "--ds_name",
        "addition",
        "--base_model_name",
        model["base_model"],
        "--probe_method",
        "mean-diff",
        "--probe_character",
        PERSONAS[source],
        "--probe_root_dir",
        str(paths["activations"].parent),
        "--output_dir",
        str(paths["interventions"].parent),
        "--test_character",
        PERSONAS[target],
        "--n_test",
        str(config["intervention_examples"]),
        "--layer_stride",
        str(config["layer_stride"]),
        "--templatization_method",
        config["templatization_method"],
        "--device",
        config["device"],
        "--dtype",
        config["dtype"],
    ]


def validate_model_metadata(model_id: str, model: dict[str, Any], cache: Path) -> None:
    from peft import PeftConfig
    from transformers import AutoConfig

    adapter = PeftConfig.from_pretrained(model_id, cache_dir=cache)
    base_model = adapter.base_model_name_or_path
    expected_base_model = model["base_model"]
    if base_model != expected_base_model:
        raise RuntimeError(
            f"{model_id} uses base model {base_model}; expected {expected_base_model}"
        )

    remote = AutoConfig.from_pretrained(base_model, cache_dir=cache)
    actual = (remote.num_hidden_layers, remote.hidden_size)
    expected = (model["expected_layers"], model["expected_hidden_size"])
    if actual != expected:
        raise RuntimeError(f"{base_model} shape is {actual}; expected {expected}")


def validate_remote(config: dict[str, Any], repo: Path, dry_run: bool) -> None:
    ids = [model_ids(config, model)[0] for model in config["models"]]
    dataset_id = f"EleutherAI/quirky_{config['dataset']}_raw"
    if dry_run:
        print(f"Would validate models {ids} and dataset {dataset_id}")
        return
    resolve_device(config["device"])
    from datasets import load_dataset

    cache = repo / config["cache_dir"]
    for model, model_id in zip(config["models"], ids):
        validate_model_metadata(model_id, model, cache)
    required = {"character", "label", "alice_label", "bob_label", "difficulty_quantile"}
    for split, count in (
        ("validation", config["validation_examples"]),
        ("test", max(config["test_examples"], config["intervention_examples"])),
    ):
        dataset = load_dataset(dataset_id, split=split, cache_dir=cache)
        missing = required - set(dataset.column_names)
        if missing:
            raise RuntimeError(f"{split} is missing columns: {sorted(missing)}")
        for character in PERSONAS.values():
            subset = dataset.filter(lambda row: row["character"] == character)
            if len(subset) < count:
                raise RuntimeError(
                    f"{split}/{character} has {len(subset)} examples; {count} are required"
                )
            subset = subset.shuffle(seed=config["seed"]).select(range(count))
            if set(subset["label"]) != {0, 1}:
                raise RuntimeError(f"{split}/{character} does not contain both label classes")
            if not any(
                alice != bob
                for alice, bob in zip(subset["alice_label"], subset["bob_label"])
            ):
                raise RuntimeError(f"{split}/{character} has no disagreement examples")
    print(f"Validated {dataset_id} and {', '.join(ids)}")


def require_complete(valid: bool, reason: str, path: Path) -> None:
    if not valid and path.exists() and any(path.iterdir()):
        raise RuntimeError(
            f"Incomplete output at {path}: {reason}. Move it aside before rerunning."
        )


def assert_valid(result: tuple[bool, str], context: str) -> None:
    valid, reason = result
    if not valid:
        raise RuntimeError(f"{context}: {reason}")


def run_stages(
    config: dict[str, Any], config_path: Path, repo: Path, stages: Sequence[str], dry_run: bool
) -> None:
    for stage in stages:
        print(f"\nStage: {stage}")
        if stage == "validate":
            validate_remote(config, repo, dry_run)
            continue
        if stage.startswith("summarize-"):
            script = (
                "summarize_replication.py"
                if stage == "summarize-replication"
                else "summarize_interventions.py"
            )
            run_command(
                [sys.executable, str(repo / "scripts" / script), "--config", str(config_path)],
                config,
                repo,
                dry_run,
            )
            continue
        for model in config["models"]:
            paths = model_paths(config, model, repo)
            if stage == "extract":
                for persona in PERSONAS:
                    for split, count in (
                        ("validation", config["validation_examples"]),
                        ("test", config["test_examples"]),
                    ):
                        root = paths["activations"] / persona / split
                        valid, reason = check_extraction(
                            root, model, count, config["save_contrast_pairs"]
                        )
                        if valid:
                            print(f"Skipping {persona}/{split}: complete")
                            continue
                        require_complete(valid, reason, root)
                        run_command(
                            extraction_command(config, model, paths, repo, persona, split, count),
                            config,
                            repo,
                            dry_run,
                        )
                        if not dry_run:
                            assert_valid(
                                check_extraction(
                                    root, model, count, config["save_contrast_pairs"]
                                ),
                                f"Extraction {persona}/{split} failed validation",
                            )
            elif stage == "probe":
                for source, targets in (("A", ("A", "B")), ("B", ("B",))):
                    valid, reason = check_probe(config, model, paths, source, targets)
                    if valid:
                        print(f"Skipping probe {source}: complete")
                        continue
                    output = paths["activations"] / source / "validation"
                    if not dry_run:
                        for persona in PERSONAS:
                            for split, count in (
                                ("validation", config["validation_examples"]),
                                ("test", config["test_examples"]),
                            ):
                                assert_valid(
                                    check_extraction(
                                        paths["activations"] / persona / split,
                                        model,
                                        count,
                                        config["save_contrast_pairs"],
                                    ),
                                    f"Extraction prerequisite {persona}/{split} is incomplete",
                                )
                    expected_outputs = [output / "mean-diff_reporters.pt"] + [
                        paths["activations"]
                        / target
                        / "test"
                        / f"{source}_mean-diff_log_odds.pt"
                        for target in targets
                    ]
                    if any(path.exists() for path in expected_outputs) and not valid:
                        raise RuntimeError(f"Incomplete probe output: {reason}")
                    run_command(
                        probe_command(config, paths, repo, source, targets),
                        config,
                        repo,
                        dry_run,
                    )
                    if not dry_run and not check_probe(config, model, paths, source, targets)[0]:
                        raise RuntimeError(f"Probe {source} failed post-run validation")
            elif stage in {"intervene-matched", "intervene-cross"}:
                pairs = (
                    (("A", "A"), ("B", "B"))
                    if stage.endswith("matched")
                    else (("A", "B"),)
                )
                for source, target in pairs:
                    root = intervention_dir(config, paths, source, target)
                    valid, reason = check_intervention(config, model, root)
                    if valid:
                        print(f"Skipping intervention {source}->{target}: complete")
                        continue
                    if root.exists():
                        raise RuntimeError(f"Incomplete intervention at {root}: {reason}")
                    run_command(
                        intervention_command(config, model, paths, repo, source, target),
                        config,
                        repo,
                        dry_run,
                    )
                    if not dry_run and not check_intervention(config, model, root)[0]:
                        raise RuntimeError(f"Intervention {source}->{target} failed validation")


def parse_stages(stage: str | None, stages: str | None) -> tuple[str, ...]:
    selected = [stage] if stage else stages.split(",") if stages else list(STAGES)
    selected = [item.strip() for item in selected if item and item.strip()]
    unknown = [item for item in selected if item not in STAGES]
    if unknown:
        raise ValueError(f"Unknown stages: {', '.join(unknown)}")
    return tuple(selected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--stage", choices=STAGES)
    group.add_argument("--stages", help="Comma-separated stages")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    run_stages(
        load_config(config_path),
        config_path,
        Path(__file__).resolve().parents[1],
        parse_stages(args.stage, args.stages),
        args.dry_run,
    )


if __name__ == "__main__":
    main()
