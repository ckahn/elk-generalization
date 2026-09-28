#!/usr/bin/env python3
"""Overlay every model's layerwise probe and intervention AUROC on shared depth axes.

Reads only the small summaries in results/ (metrics.csv, intervention_metrics.csv,
run_metadata.json, intervention_metadata.json); no activations or models are loaded.
"""

import argparse
import csv
import json
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

EXPERIMENTS = ("A-to-A", "A-to-B", "B-to-B")
# Reference data-viz palette, slots 1-3 (validated all-pairs for CVD separation).
EXPERIMENT_COLORS = {"A-to-A": "#2a78d6", "A-to-B": "#eb6834", "B-to-B": "#1baf7a"}
EXPERIMENT_LABELS = {
    "A-to-A": "A-to-A: Alice probe, Alice context",
    "A-to-B": "A-to-B: Alice probe, Bob context",
    "B-to-B": "B-to-B: Bob probe, Bob context",
}
# Persona whose labels define "correct" in each experiment's test context.
CONTEXT_PERSONA = {"A-to-A": "alice", "A-to-B": "bob", "B-to-B": "bob"}
# Lightest (smallest model) to darkest (largest model) shade offsets.
SHADE_RANGE = (0.45, -0.3)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BAND = "#898781"
DEFAULT_ROOTS = ("results/replication", "results/extension")
OUTPUT_DIR = "results/combined"


@dataclass(frozen=True)
class ModelResults:
    name: str
    n_layers: int
    probe: dict  # experiment -> (depths, aurocs against Alice labels)
    intervention: dict  # experiment -> (depths, intervened aurocs vs context persona)
    clean: dict  # experiment -> clean output auroc vs context persona
    lm_output_vs_alice: dict  # context persona -> clean output auroc vs Alice labels
    pgr: float
    intervention_examples: int | None


def model_size(name):
    """Parameter count in billions parsed from a name like 'pythia-1.4b'."""
    match = re.search(r"(\d+(?:\.\d+)?)([bm])$", name.lower())
    if match is None:
        raise ValueError(f"Cannot parse model size from {name!r}")
    value = float(match.group(1))
    return value if match.group(2) == "b" else value / 1000


def display_name(name):
    family, _, size = name.partition("-")
    return f"{family.capitalize()}-{size.upper()}"


def intervention_column(experiment):
    if experiment not in CONTEXT_PERSONA:
        raise ValueError(f"Unknown experiment {experiment!r}")
    return f"int_auroc_{CONTEXT_PERSONA[experiment]}"


def clean_column(experiment):
    if experiment not in CONTEXT_PERSONA:
        raise ValueError(f"Unknown experiment {experiment!r}")
    return f"cl_auroc_{CONTEXT_PERSONA[experiment]}"


def shade(hex_color, amount):
    """Mix toward white (amount > 0) or black (amount < 0) by |amount|."""
    rgb = np.array([int(hex_color[i : i + 2], 16) for i in (1, 3, 5)], dtype=float)
    target = 255.0 if amount > 0 else 0.0
    mixed = rgb + (target - rgb) * abs(amount)
    return "#" + "".join(f"{int(round(channel)):02x}" for channel in mixed)


def read_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def curve(rows, experiment, column, n_layers, layer_offset):
    selected = sorted(
        (int(row["layer"]) + layer_offset, float(row[column]))
        for row in rows
        if row["experiment"] == experiment
    )
    if len(selected) != n_layers:
        raise ValueError(
            f"{experiment} has {len(selected)} layers for {column}, expected {n_layers}"
        )
    return (
        tuple(layer / n_layers for layer, _ in selected),
        tuple(value for _, value in selected),
    )


def single_value(rows, experiment, column):
    values = {float(row[column]) for row in rows if row["experiment"] == experiment}
    if len(values) != 1:
        raise ValueError(f"Expected one {column} value for {experiment}, got {values}")
    return values.pop()


def load_model_results(directory):
    directory = Path(directory)
    probe_rows = read_rows(directory / "metrics.csv")
    intervention_rows = read_rows(directory / "intervention_metrics.csv")
    metadata = json.loads((directory / "run_metadata.json").read_text())
    intervention_metadata_path = directory / "intervention_metadata.json"
    intervention_examples = (
        json.loads(intervention_metadata_path.read_text())["intervention_examples"]
        if intervention_metadata_path.exists()
        else None
    )
    n_layers = max(int(row["layer"]) for row in probe_rows)
    return ModelResults(
        name=directory.name,
        n_layers=n_layers,
        probe={
            name: curve(probe_rows, name, "auroc_all", n_layers, 0) for name in EXPERIMENTS
        },
        # Intervention summaries use 0-indexed layers; probe metrics are 1-indexed.
        intervention={
            name: curve(intervention_rows, name, intervention_column(name), n_layers, 1)
            for name in EXPERIMENTS
        },
        clean={
            name: single_value(intervention_rows, name, clean_column(name))
            for name in EXPERIMENTS
        },
        lm_output_vs_alice={
            "Alice": single_value(intervention_rows, "A-to-A", "cl_auroc_alice"),
            "Bob": single_value(intervention_rows, "B-to-B", "cl_auroc_alice"),
        },
        pgr=metadata["performance_gap_recovered"],
        intervention_examples=intervention_examples,
    )


def discover_model_dirs(roots):
    required = ("metrics.csv", "intervention_metrics.csv", "run_metadata.json")
    found = [
        path
        for root in map(Path, roots)
        if root.is_dir()
        for path in root.iterdir()
        if path.is_dir() and all((path / name).exists() for name in required)
    ]
    return sorted(found, key=lambda path: model_size(path.name))


def model_shades(count):
    if count == 1:
        return (0.0,)
    return tuple(np.linspace(*SHADE_RANGE, count))


def style_axes(ax, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Relative depth (layer / number of layers)", color=INK_SECONDARY)
    ax.set_ylabel(ylabel, color=INK_SECONDARY)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_MUTED, labelcolor=INK_SECONDARY)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.axhline(0.5, color=INK_MUTED, linestyle=":", linewidth=1.2, zorder=1)


def plot_curves(ax, models, attribute):
    for model, amount in zip(models, model_shades(len(models))):
        for name in EXPERIMENTS:
            depths, values = getattr(model, attribute)[name]
            ax.plot(
                depths,
                values,
                color=shade(EXPERIMENT_COLORS[name], amount),
                linewidth=2,
                solid_capstyle="round",
                zorder=3,
            )


def label_line_ends(ax, models, attribute):
    """Name each experiment at the right edge, beside the largest model's line."""
    ends = sorted(
        (getattr(models[-1], attribute)[name][1][-1], name) for name in EXPERIMENTS
    )
    min_gap = 0.06
    positions = []
    for value, name in ends:
        y = max(value, positions[-1][0] + min_gap) if positions else value
        positions.append((y, name))
    for y, name in positions:
        ax.annotate(
            name,
            xy=(1.0, y),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            color=INK,
            fontsize=10,
            fontweight="bold",
            annotation_clip=False,
        )


def band(ax, values, label, label_y_offset):
    low, high = min(values), max(values)
    ax.axhspan(low, high, color=BAND, alpha=0.18, linewidth=0, zorder=2)
    ax.text(
        0.015,
        high + label_y_offset if label_y_offset > 0 else low + label_y_offset,
        label,
        color=INK_SECONDARY,
        fontsize=9,
        va="bottom" if label_y_offset > 0 else "top",
    )


def legend_handles(models, reference_handles, rows):
    experiment_handles = [
        Line2D([], [], color=EXPERIMENT_COLORS[name], linewidth=2.5, label=EXPERIMENT_LABELS[name])
        for name in EXPERIMENTS
    ]
    model_handles = [
        Line2D(
            [],
            [],
            color=shade(INK_SECONDARY, amount),
            linewidth=2.5,
            label=f"{display_name(model.name)} ({model.n_layers} layers, PGR {model.pgr:.2f})",
        )
        for model, amount in zip(models, model_shades(len(models)))
    ]
    # Pad each group to a full column so experiments, models and references each
    # fill one legend column.
    padding = [Line2D([], [], linestyle="", label=" ")]
    return [
        handle
        for group in (experiment_handles, model_handles, reference_handles)
        for handle in group + padding * (rows - len(group))
    ]


def finish(fig, ax, models, reference_handles, title, footnote, path):
    rows = max(len(EXPERIMENTS), len(models), len(reference_handles))
    ax.set_title(title, loc="left", color=INK, fontsize=12, pad=12)
    fig.legend(
        handles=legend_handles(models, reference_handles, rows),
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=9,
        labelcolor=INK_SECONDARY,
        bbox_to_anchor=(0.5, 0.06),
    )
    fig.text(
        0.01,
        0.01,
        textwrap.fill(footnote, 150),
        color=INK_MUTED,
        fontsize=8,
        ha="left",
        va="bottom",
    )
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"Wrote {path}")


def new_figure():
    fig, ax = plt.subplots(figsize=(12, 7.5), facecolor=SURFACE)
    fig.subplots_adjust(left=0.07, right=0.9, top=0.93, bottom=0.29)
    return fig, ax


def plot_probe_auroc(models, path):
    fig, ax = new_figure()
    style_axes(ax, "Probe AUROC against Alice's labels (all test examples)")
    band(ax, [m.lm_output_vs_alice["Alice"] for m in models], "LM output, Alice context", -0.012)
    band(ax, [m.lm_output_vs_alice["Bob"] for m in models], "LM output, Bob context", 0.012)
    plot_curves(ax, models, "probe")
    label_line_ends(ax, models, "probe")
    reference = [
        Patch(color=BAND, alpha=0.35, label="LM output AUROC, range across models"),
        Line2D([], [], color=INK_MUTED, linestyle=":", label="Chance (0.5)"),
    ]
    finish(
        fig, ax, models, reference,
        "Diff-in-means probe AUROC by depth, all models",
        footnote_text(models, "LM output bands"),
        path,
    )


def plot_intervention_auroc(models, path):
    fig, ax = new_figure()
    style_axes(ax, "Intervened output AUROC against labels of the persona in context")
    band(ax, [value for m in models for value in m.clean.values()], "No intervention", -0.012)
    plot_curves(ax, models, "intervention")
    label_line_ends(ax, models, "intervention")
    reference = [
        Patch(color=BAND, alpha=0.35, label="Clean output AUROC, range across models"),
        Line2D([], [], color=INK_MUTED, linestyle=":", label="Chance (0.5)"),
    ]
    finish(
        fig, ax, models, reference,
        "Reflecting the probe direction at each depth, all models",
        footnote_text(models, "Clean-output values")
        + " A-to-A is scored against Alice's labels; A-to-B and B-to-B against Bob's.",
        path,
    )


def footnote_text(models, subject):
    counts = ", ".join(
        f"{display_name(m.name)}: {m.intervention_examples}"
        for m in models
        if m.intervention_examples is not None
    )
    return f"{subject} come from the intervention subset of test examples ({counts})."


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", nargs="+", default=list(DEFAULT_ROOTS))
    parser.add_argument("--output-dir", type=Path, default=Path(OUTPUT_DIR))
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    model_dirs = discover_model_dirs(repo / root for root in args.roots)
    if not model_dirs:
        raise SystemExit(f"No complete model results found under {args.roots}")
    models = [load_model_results(path) for path in model_dirs]
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_probe_auroc(models, output_dir / "auroc_by_layer.png")
    plot_intervention_auroc(models, output_dir / "intervention_auroc_by_layer.png")


if __name__ == "__main__":
    main()
