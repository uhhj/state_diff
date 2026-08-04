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


def marker(row):
    mechanism = row["candidate"]["friction"].get("mechanism")
    protocol = row["candidate"]["action"].get("protocol")
    if mechanism == "native_segment":
        return "s"
    if protocol == "planar_microprobe":
        return "^"
    return "o"


def annotation_offset(candidate_id):
    """Keep labels readable when candidates share the same plotted point."""
    offsets = {
        "legacy_micro_p05": (5, 5),
        "legacy_micro_p10": (5, -10),
        "native_micro_p05_low": (5, 31),
        "native_micro_p10_low": (5, 19),
        "native_micro_p05_mid": (5, 7),
        "native_micro_p10_mid": (5, -5),
        "native_micro_p05_high": (5, -17),
        "native_micro_p10_high": (5, -29),
    }
    return offsets.get(candidate_id, (5, 4))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=(
            "reports/experiment2/phase0_hidden_friction/phase0f/summary.json"
        ),
    )
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    path = Path(args.input).resolve()
    summary = load(path)
    rows = summary["ranked_candidates"]
    plots = path.parent / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 7), dpi=args.dpi)
    for row in rows:
        median = row["median"]
        size = 80 + 500 * float(
            row["classifiers"]["formal_sensor"]["accuracy"]
        )
        ax.scatter(
            median["preload_end_max_abs_xy"],
            median["main_branch_fde"],
            s=size,
            marker=marker(row),
        )
        ax.annotate(
            row["candidate_id"],
            (
                median["preload_end_max_abs_xy"],
                median["main_branch_fde"],
            ),
            xytext=annotation_offset(row["candidate_id"]),
            textcoords="offset points",
            fontsize=8,
        )
    ax.axvline(
        summary["targets"]["max_median_preload_visible_difference"],
        linestyle="--",
    )
    ax.axhline(
        summary["targets"]["min_median_main_branch_fde"],
        linestyle=":",
    )
    ax.set(
        xlabel="median preload visible difference (m)",
        ylabel="median main-pull FDE (m)",
        title=(
            "Phase 0F mechanism trade-off\n"
            "marker size = formal sensor accuracy"
        ),
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots / "mechanism_tradeoff.png")
    plt.close(fig)

    labels = [row["candidate_id"] for row in rows]
    x = np.arange(len(rows))
    width = 0.2
    fig, ax = plt.subplots(figsize=(14, 7), dpi=args.dpi)
    ax.bar(
        x - 1.5 * width,
        [r["classifiers"]["vision_raw"]["accuracy"] for r in rows],
        width,
        label="vision raw",
    )
    ax.bar(
        x - 0.5 * width,
        [r["classifiers"]["vision_delta"]["accuracy"] for r in rows],
        width,
        label="vision delta",
    )
    ax.bar(
        x + 0.5 * width,
        [r["classifiers"]["formal_sensor"]["accuracy"] for r in rows],
        width,
        label="formal sensor",
    )
    ax.bar(
        x + 1.5 * width,
        [r["classifiers"]["oracle_contact"]["accuracy"] for r in rows],
        width,
        label="Oracle",
    )
    ax.axhline(0.5, linestyle=":", label="random")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set(
        ylabel="leave-one-seed-out accuracy",
        title="Phase 0F modality comparison",
    )
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "modality_accuracy.png")
    plt.close(fig)

    values = [float(row["score"]) for row in rows]
    fig, ax = plt.subplots(figsize=(12, 6), dpi=args.dpi)
    ax.bar(x, values)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set(ylabel="engineering score", title="Phase 0F candidate ranking")
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
    fig.savefig(plots / "candidate_scores.png")
    plt.close(fig)

    top = rows[0]
    pairs = top["pairs"]
    pair_labels = [row["group_id"] for row in pairs]
    keys = (
        "preload_end_max_abs_xy",
        "contact_impulse_gap",
        "main_branch_ade",
        "main_branch_fde",
        "branch_amplification",
    )
    fig, axes = plt.subplots(3, 2, figsize=(12, 11), dpi=args.dpi)
    for ax, key in zip(axes.flat, keys):
        values = [float(row[key]) for row in pairs]
        positions = np.arange(len(values))
        ax.bar(positions, values)
        ax.set_xticks(positions)
        ax.set_xticklabels(pair_labels, rotation=25, ha="right")
        ax.set_title(key)
        ax.grid(axis="y", alpha=0.25)
    axes.flat[-1].axis("off")
    fig.suptitle(f"Top candidate: {top['candidate_id']}")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(plots / "top_candidate_per_seed.png")
    plt.close(fig)

    for file in sorted(plots.glob("*.png")):
        print(file)


if __name__ == "__main__":
    main()
