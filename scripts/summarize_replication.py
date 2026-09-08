#!/usr/bin/env python3
"""Write layerwise AUROC, Earliest Informative Layer, and PGR summaries."""

import argparse
import csv
import json
import subprocess
from pathlib import Path

import matplotlib
import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score

from elk_generalization.utils import get_quirky_model_name

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def models(config):
    return config.get("models") or [
        {
            "base_model": config["base_model"],
            "expected_layers": config["expected_layers"],
            "expected_hidden_size": config["expected_hidden_size"],
        }
    ]


def auroc(labels, scores):
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def earliest_layer(values, threshold=0.95):
    maximum = max(values)
    if maximum < 0.5:
        return len(values) // 2
    return next(
        i
        for i, value in enumerate(values)
        if value - 0.5 >= threshold * (maximum - 0.5)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    with args.config.open() as handle:
        config = yaml.safe_load(handle)

    for model in models(config):
        _, model_last = get_quirky_model_name(
            config["dataset"], model["base_model"], config["templatization_method"]
        )
        activation_root = repo / config["run_root"] / "activations" / model_last
        result_root = repo / config["results_root"] / model["base_model"].split("/")[-1]
        experiments = {
            "A-to-A": ("A", "A"),
            "A-to-B": ("A", "B"),
            "B-to-B": ("B", "B"),
        }
        rows, curves = [], {}
        for name, (source, target) in experiments.items():
            root = activation_root / target / "test"
            scores = torch.load(
                root / f"{source}_mean-diff_log_odds.pt", map_location="cpu"
            ).numpy()
            alice = torch.load(root / "alice_labels.pt", map_location="cpu").numpy()
            bob = torch.load(root / "bob_labels.pt", map_location="cpu").numpy()
            disagree = alice != bob
            curves[name] = []
            for index, layer_scores in enumerate(scores):
                score = auroc(alice, layer_scores)
                curves[name].append(score)
                rows.append(
                    {
                        "experiment": name,
                        "layer": index + 1,
                        "auroc_all": score,
                        "auroc_disagreement": auroc(alice[disagree], layer_scores[disagree]),
                    }
                )
        if any(not np.isfinite(curve).all() for curve in map(np.asarray, curves.values())):
            raise RuntimeError("An all-example layer AUROC is not finite")

        eil = earliest_layer(curves["A-to-A"])
        floor, ceiling = curves["B-to-B"][-1], curves["A-to-A"][-1]
        transfer = curves["A-to-B"][eil]
        pgr = None if np.isclose(ceiling, floor) else (transfer - floor) / (ceiling - floor)
        result_root.mkdir(parents=True, exist_ok=True)
        with (result_root / "metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        with (result_root / "run_metadata.json").open("w") as handle:
            json.dump(
                {
                    "project_commit": subprocess.run(
                        ["git", "rev-parse", "HEAD"],
                        cwd=repo,
                        check=True,
                        capture_output=True,
                        text=True,
                    ).stdout.strip(),
                    "upstream_reproduction_commit": "53bad05c4ac52b042f1b0255172f2f425124f670",
                    "model": model["base_model"],
                    "dataset": config["dataset"],
                    "device": config["device"],
                    "dtype": config["dtype"],
                    "seed": config["seed"],
                    "earliest_informative_layer": eil + 1,
                    "weak_floor": floor,
                    "strong_ceiling": ceiling,
                    "transfer_score": transfer,
                    "performance_gap_recovered": pgr,
                },
                handle,
                indent=2,
            )
            handle.write("\n")
        for name, curve in curves.items():
            plt.plot(range(1, len(curve) + 1), curve, label=name)
        plt.axvline(eil + 1, color="black", linestyle="--", linewidth=1)
        plt.axhline(0.5, color="gray", linestyle=":", linewidth=1)
        plt.xlabel("Layer")
        plt.ylabel("AUROC against Alice labels")
        plt.ylim(0, 1)
        plt.legend()
        plt.tight_layout()
        plt.savefig(result_root / "auroc_by_layer.png", dpi=160)
        plt.close()
        print(f"Wrote {result_root}")


if __name__ == "__main__":
    main()
