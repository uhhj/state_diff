#!/usr/bin/env python3
"""Visualize Phase 0C hidden-friction calibration trade-offs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def make_tradeoff(
    summary: Dict[str, Any],
    output: Path,
    dpi: int,
) -> Path:
    rows = summary["ranked_candidates"]
    fig, ax = plt.subplots(figsize=(10, 7), dpi=dpi)
    for row in rows:
        median = row["median"]
        size = 80.0 + 500.0 * min(
            float(median["contact_impulse_gap"]), 0.5
        )
        marker = "o" if row["eligible"] else "x"
        ax.scatter(
            median["preload_end_max_abs_xy"],
            median["main_branch_fde"],
            s=size,
            marker=marker,
        )
        ax.annotate(
            row["candidate_id"],
            (
                median["preload_end_max_abs_xy"],
                median["main_branch_fde"],
            ),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
        )

    targets = summary["targets"]
    ax.axvline(
        targets["max_median_preload_visible_difference"],
        linestyle="--",
        label="preload target",
    )
    ax.axhline(
        targets["min_median_main_branch_fde"],
        linestyle=":",
        label="FDE target",
    )
    ax.set(
        xlabel=(
            "median preload visible max difference "
            "(m; lower is better)"
        ),
        ylabel="median main-pull FDE (m; higher is better)",
        title=(
            "Phase 0C candidate trade-off\n"
            "marker size = median contact impulse gap"
        ),
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    return output


def make_scores(
    summary: Dict[str, Any],
    output: Path,
    dpi: int,
) -> Path:
    rows = summary["ranked_candidates"]
    labels = [row["candidate_id"] for row in rows]
    values = [float(row["score"]) for row in rows]
    fig, ax = plt.subplots(figsize=(11, 6), dpi=dpi)
    positions = np.arange(len(rows))
    ax.bar(positions, values)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set(
        ylabel="calibration score",
        title="Phase 0C candidate ranking",
    )
    for index, row in enumerate(rows):
        ax.text(
            index,
            values[index],
            "eligible" if row["eligible"] else "review",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    return output


def make_selected_seed_metrics(
    summary: Dict[str, Any],
    output: Path,
    dpi: int,
) -> Path:
    selected_id = summary["selected_candidate_id"]
    selected = next(
        row
        for row in summary["ranked_candidates"]
        if row["candidate_id"] == selected_id
    )
    pairs = selected["pairs"]
    labels = [row["group_id"] for row in pairs]
    keys = (
        "preload_end_max_abs_xy",
        "contact_impulse_gap",
        "main_branch_ade",
        "main_branch_fde",
    )
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=dpi)
    for ax, key in zip(axes.flat, keys):
        values = [float(row[key]) for row in pairs]
        positions = np.arange(len(values))
        ax.bar(positions, values)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_title(key)
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle(f"Selected candidate: {selected_id}")
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
            "reports/experiment2/phase0_hidden_friction/"
            "calibration/calibration_summary.json"
        ),
    )
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    summary_path = Path(args.input).resolve()
    summary = _load(summary_path)
    plots = summary_path.parent / "plots"

    print(
        make_tradeoff(
            summary,
            plots / "candidate_tradeoff.png",
            args.dpi,
        )
    )
    print(
        make_scores(
            summary,
            plots / "candidate_scores.png",
            args.dpi,
        )
    )
    print(
        make_selected_seed_metrics(
            summary,
            plots / "selected_candidate_seed_metrics.png",
            args.dpi,
        )
    )


if __name__ == "__main__":
    main()
