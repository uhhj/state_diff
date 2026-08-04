#!/usr/bin/env python3
"""Visualize the bounded Phase 0E recalibration summary."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_figure(figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(str(path), dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="reports/experiment2/phase0_hidden_friction/phase0e/summary.json",
    )
    args = parser.parse_args()
    input_path = Path(args.input).resolve()
    summary = json.loads(input_path.read_text(encoding="utf-8"))
    ranked = summary["ranked_candidates"]
    targets = summary["targets"]
    output = input_path.parent / "plots"

    figure, axis = plt.subplots(figsize=(9, 6))
    for row in ranked:
        median = row["median"]
        protocol = row["candidate"]["action"].get("protocol", "direct")
        marker = "o" if protocol == "direct" else "s"
        size = 40.0 + 260.0 * max(float(median["contact_impulse_gap"]), 0.0)
        axis.scatter(
            median["preload_end_max_abs_xy"],
            median["main_branch_fde"],
            s=size,
            marker=marker,
            alpha=0.75,
        )
        axis.annotate(
            row["candidate_id"],
            (median["preload_end_max_abs_xy"], median["main_branch_fde"]),
            fontsize=7,
            xytext=(4, 3),
            textcoords="offset points",
        )
    axis.axvline(
        targets["max_median_preload_visible_difference"],
        color="tab:red",
        linestyle="--",
        label="max preload visibility",
    )
    axis.axhline(
        targets["min_median_main_branch_fde"],
        color="tab:green",
        linestyle="--",
        label="min main FDE",
    )
    axis.set_xlabel("Median preload visible difference")
    axis.set_ylabel("Median main FDE")
    axis.set_title("Phase 0E preload visibility versus future divergence")
    axis.legend(loc="best")
    axis.grid(alpha=0.25)
    save_figure(figure, output / "preload_vs_fde.png")

    labels = [row["candidate_id"] for row in ranked]
    positions = np.arange(len(labels), dtype=np.float64)
    width = 0.25
    figure, axis = plt.subplots(figsize=(12, 6))
    axis.bar(
        positions - width,
        [row["classifiers"]["vision_raw"]["accuracy"] for row in ranked],
        width,
        label="vision raw",
    )
    axis.bar(
        positions,
        [row["classifiers"]["vision_delta"]["accuracy"] for row in ranked],
        width,
        label="vision delta",
    )
    axis.bar(
        positions + width,
        [row["classifiers"]["contact"]["accuracy"] for row in ranked],
        width,
        label="contact",
    )
    axis.axhline(0.5, color="black", linestyle="--", label="random")
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("Grouped leave-one-seed-out accuracy")
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
    axis.set_title("Phase 0E grouped classifier screen")
    axis.legend(loc="best")
    axis.grid(axis="y", alpha=0.25)
    save_figure(figure, output / "classifier_accuracy.png")

    score_labels = [
        "{} ({})".format(
            row["candidate_id"], "eligible" if row["eligible"] else "review"
        )
        for row in ranked
    ]
    figure, axis = plt.subplots(figsize=(10, 6))
    y = np.arange(len(ranked), dtype=np.float64)
    axis.barh(y, [row["score"] for row in ranked])
    axis.set_yticks(y)
    axis.set_yticklabels(score_labels, fontsize=8)
    axis.invert_yaxis()
    axis.set_xlabel("Final score")
    axis.set_title("Phase 0E candidate ranking")
    axis.grid(axis="x", alpha=0.25)
    save_figure(figure, output / "candidate_scores.png")

    selected = ranked[0]
    pair_rows = selected["pairs"]
    seeds = [str(row["seed"]) for row in pair_rows]
    metrics = [
        ("preload visibility", "preload_end_max_abs_xy"),
        ("contact impulse gap", "contact_impulse_gap"),
        ("main ADE", "main_branch_ade"),
        ("main FDE", "main_branch_fde"),
        ("branch amplification", "branch_amplification"),
    ]
    figure, axes = plt.subplots(len(metrics), 1, figsize=(10, 10), sharex=True)
    for axis, (label, key) in zip(axes, metrics):
        axis.plot(seeds, [row[key] for row in pair_rows], marker="o")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    axes[0].set_title(
        "Top candidate per-seed metrics: {}".format(selected["candidate_id"])
    )
    axes[-1].set_xlabel("Seed")
    save_figure(figure, output / "selected_per_seed.png")


if __name__ == "__main__":
    main()
