#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="reports/experiment2/phase0_hidden_hook/phase0g/summary.json",
    )
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()
    path = Path(args.input).resolve()
    summary = load(path)
    rows = summary["ranked_candidates"]
    plots = path.parent / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=args.dpi)
    for row in rows:
        median = row["median"]
        size = 80 + 500 * float(row["classifiers"]["formal_sensor"]["accuracy"])
        ax.scatter(
            median["preload_end_max_abs_xy"],
            median["main_branch_fde"],
            s=size,
        )
        ax.annotate(
            row["candidate_id"],
            (median["preload_end_max_abs_xy"], median["main_branch_fde"]),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=9,
        )
    ax.axvline(summary["targets"]["max_median_preload_visible_difference"], linestyle="--")
    ax.axhline(summary["targets"]["min_median_main_branch_fde"], linestyle=":")
    ax.set(
        xlabel="median preload visible difference (m)",
        ylabel="median main-pull FDE (m)",
        title="Phase 0G preload vs FDE\nmarker size = formal sensor accuracy",
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots / "preload_vs_fde.png")
    plt.close(fig)

    labels = [row["candidate_id"] for row in rows]
    x = np.arange(len(rows))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 6), dpi=args.dpi)
    ax.bar(
        x - width,
        [row["classifiers"]["vision_delta"]["accuracy"] for row in rows],
        width,
        label="vision delta",
    )
    ax.bar(
        x,
        [row["classifiers"]["formal_sensor"]["accuracy"] for row in rows],
        width,
        label="formal sensor",
    )
    ax.bar(
        x + width,
        [row["classifiers"]["oracle_contact"]["accuracy"] for row in rows],
        width,
        label="Oracle (privileged)",
    )
    ax.axhline(0.5, linestyle=":", label="random")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set(ylabel="leave-one-seed-out accuracy", title="Phase 0G modality comparison")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots / "modality_accuracy.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6), dpi=args.dpi)
    values = [row["median"]["mean_cable_progress_gap"] for row in rows]
    ax.bar(labels, values)
    ax.axhline(summary["targets"]["min_median_progress_gap"], linestyle="--")
    ax.set(
        ylabel="median free-hidden mean cable progress gap (m)",
        title="Phase 0G routing outcome gap",
    )
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots / "outcome_progress_gap.png")
    plt.close(fig)

    top = rows[0]
    metrics = (
        ("preload_end_max_abs_xy", "preload difference"),
        ("main_branch_fde", "main FDE"),
        ("branch_amplification", "amplification"),
        ("mean_cable_progress_gap", "progress gap"),
        ("hidden_hook_engagement_fraction", "engagement fraction"),
    )
    fig, axes = plt.subplots(3, 2, figsize=(12, 12), dpi=args.dpi)
    pair_rows = sorted(top["pairs"], key=lambda row: int(row["seed"]))
    seed_labels = [str(row["group_id"]) for row in pair_rows]
    for ax, (key, title) in zip(axes.flat, metrics):
        ax.bar(seed_labels, [row[key] for row in pair_rows])
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.25)
    axes.flat[-1].axis("off")
    fig.suptitle(f"Top candidate: {top['candidate_id']}", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(plots / "top_candidate_per_seed.png")
    plt.close(fig)

    for filename in (
        "preload_vs_fde.png",
        "modality_accuracy.png",
        "outcome_progress_gap.png",
        "top_candidate_per_seed.png",
    ):
        print(plots / filename)


if __name__ == "__main__":
    main()
