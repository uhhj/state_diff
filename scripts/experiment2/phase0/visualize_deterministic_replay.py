#!/usr/bin/env python3
"""Create plots for the Phase 0D fixed-step determinism audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _load_trace(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def plot_lengths(summary, output: Path, dpi: int) -> Path:
    labels = []
    free = []
    hidden = []
    for seed_result in summary["seed_results"]:
        seed = seed_result["seed"]
        for row in seed_result["repeats"]:
            labels.append(f"{seed}-r{row['repeat']}")
            free.append(row["free_trace_length"])
            hidden.append(row["hidden_trace_length"])
    positions = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(12, 6), dpi=dpi)
    ax.bar(positions - width / 2, free, width, label="free")
    ax.bar(positions + width / 2, hidden, width, label="hidden")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set(ylabel="trace frames", title="Fixed-step trace lengths")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    return output


def plot_errors(summary, output: Path, dpi: int) -> Path:
    labels = []
    errors = []
    for seed_result in summary["seed_results"]:
        for condition in ("free", "hidden"):
            key = f"{condition}_trace_comparisons"
            for row in seed_result[key]:
                labels.append(
                    f"{seed_result['seed']}-{condition}-r{row['candidate_repeat']}"
                )
                maxima = [
                    field["max_abs"]
                    for field in row["fields"].values()
                    if field.get("max_abs") is not None
                ]
                errors.append(max(maxima) if maxima else 0.0)
    fig, ax = plt.subplots(figsize=(12, 6), dpi=dpi)
    positions = np.arange(len(labels))
    ax.bar(positions, errors)
    ax.axhline(summary["numeric_atol"], linestyle="--", label="tolerance")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set(ylabel="maximum absolute error", title="Repeat trajectory errors")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    return output


def plot_metrics(summary, output: Path, dpi: int) -> Path:
    keys = (
        "preload_end_max_abs_xy",
        "contact_impulse_gap",
        "main_branch_ade",
        "main_branch_fde",
    )
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=dpi)
    for ax, key in zip(axes.flat, keys):
        labels = []
        values = []
        for seed_result in summary["seed_results"]:
            for row in seed_result["repeats"]:
                labels.append(f"{seed_result['seed']}-r{row['repeat']}")
                values.append(row["metrics"][key])
        positions = np.arange(len(values))
        ax.bar(positions, values)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
        ax.set_title(key)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Fixed-step repeat metrics")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    return output


def plot_final_shapes(summary_path: Path, summary, output: Path, dpi: int) -> Path:
    root = summary_path.parent
    seeds = [row["seed"] for row in summary["seed_results"]]
    fig, axes = plt.subplots(2, len(seeds), figsize=(5 * len(seeds), 8), dpi=dpi)
    if len(seeds) == 1:
        axes = np.asarray(axes).reshape(2, 1)
    for column, seed in enumerate(seeds):
        raw = root / "raw" / f"seed_{seed:06d}"
        for repeat in range(summary["repeats"]):
            free = _load_trace(raw / f"repeat_{repeat}_free.npz")
            hidden = _load_trace(
                raw / f"repeat_{repeat}_hidden_high_friction.npz"
            )
            for row_index, trace in enumerate((free, hidden)):
                beads = trace["bead_positions"][-1]
                axes[row_index, column].plot(
                    beads[:, 0],
                    beads[:, 1],
                    marker="o",
                    markersize=2,
                    label=f"repeat {repeat}",
                )
        axes[0, column].set_title(f"seed {seed} | free")
        axes[1, column].set_title(f"seed {seed} | hidden")
        for row_index in range(2):
            axes[row_index, column].set_aspect("equal", adjustable="box")
            axes[row_index, column].set_xlabel("x")
            axes[row_index, column].set_ylabel("y")
            axes[row_index, column].legend(fontsize=7)
            axes[row_index, column].grid(alpha=0.2)
    fig.suptitle("Final cable shapes across deterministic repeats")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=(
            "reports/experiment2/phase0_hidden_friction/determinism/"
            "determinism_summary.json"
        ),
    )
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    summary_path = Path(args.input).resolve()
    summary = _load_json(summary_path)
    plots = summary_path.parent / "plots"
    print(plot_lengths(summary, plots / "trace_lengths.png", args.dpi))
    print(plot_errors(summary, plots / "repeat_errors.png", args.dpi))
    print(plot_metrics(summary, plots / "repeat_metrics.png", args.dpi))
    print(
        plot_final_shapes(
            summary_path,
            summary,
            plots / "final_shapes.png",
            args.dpi,
        )
    )


if __name__ == "__main__":
    main()
