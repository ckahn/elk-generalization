#!/usr/bin/env python3
"""Write compact CSV and plot summaries for matched and cross-persona interventions."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import yaml

from elk_generalization.utils import get_quirky_model_name

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    with args.config.open() as handle:
        config = yaml.safe_load(handle)
    models = config.get("models") or [{"base_model": config["base_model"]}]
    experiments = {
        "A-to-A": ("Alice", "Alice"),
        "B-to-B": ("Bob", "Bob"),
        "A-to-B": ("Alice", "Bob"),
    }

    for model in models:
        _, model_last = get_quirky_model_name(
            config["dataset"], model["base_model"], config["templatization_method"]
        )
        root = repo / config["run_root"] / "interventions" / model_last
        result_root = repo / config["results_root"] / model["base_model"].split("/")[-1]
        rows = []
        for name, (source, target) in experiments.items():
            path = root / f"mean-diff_{source}_to_{target}_0.0_1.0" / "summary.json"
            with path.open() as handle:
                for result in json.load(handle):
                    rows.append({"experiment": name, **result})
        result_root.mkdir(parents=True, exist_ok=True)
        with (result_root / "intervention_metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        for name in experiments:
            selected = sorted(
                (row for row in rows if row["experiment"] == name),
                key=lambda row: row["layer"],
            )
            plt.plot(
                [row["layer"] + 1 for row in selected],
                [row["int_auroc_alice"] for row in selected],
                marker="o",
                label=name,
            )
        plt.axhline(0.5, color="gray", linestyle=":", linewidth=1)
        plt.xlabel("Layer")
        plt.ylabel("Intervened AUROC against Alice labels")
        plt.ylim(0, 1)
        plt.legend()
        plt.tight_layout()
        plt.savefig(result_root / "intervention_auroc_by_layer.png", dpi=160)
        plt.close()
        with (result_root / "intervention_metadata.json").open("w") as handle:
            json.dump(
                {
                    "model": model["base_model"],
                    "dataset": config["dataset"],
                    "intervention_examples": config["intervention_examples"],
                    "layer_stride": config["layer_stride"],
                },
                handle,
                indent=2,
            )
            handle.write("\n")
        print(f"Wrote {result_root}")


if __name__ == "__main__":
    main()
