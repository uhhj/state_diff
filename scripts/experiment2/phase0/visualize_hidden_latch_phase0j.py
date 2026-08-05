#!/usr/bin/env python3
"""Visualize fixed Phase 0J wide-stop outcome diagnostics."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_values(summary):
    topology = summary["topology"]
    return {
        "preload (m)": topology["median"]["preload_end_max_abs_xy"],
        "FDE (m)": topology["median"]["main_branch_fde"],
        "amplification": topology["median"]["branch_amplification"],
        "progress gap (m)": topology["median"]["mean_cable_progress_gap"],
        "formal sensor": topology["classifiers"]["formal_sensor"]["accuracy"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", default="reports/experiment2/phase0_hidden_hook/phase0j"
    )
    parser.add_argument(
        "--baseline", default="reports/experiment2/phase0_hidden_hook/phase0i"
    )
    args = parser.parse_args()
    input_root = (REPO_ROOT / args.input).resolve()
    baseline_root = (REPO_ROOT / args.baseline).resolve()
    plots = input_root / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    current = load_json(input_root / "summary.json")
    baseline = load_json(baseline_root / "summary.json")
    audit = load_json(input_root / "outcome_audit.json")

    old_values = _metric_values(baseline)
    new_values = _metric_values(current)
    thresholds = {
        "preload (m)": (0.004, "max"),
        "FDE (m)": (0.015, "min"),
        "progress gap (m)": (0.01, "min"),
    }
    fig, axes = plt.subplots(1, 5, figsize=(18, 4.5))
    for axis, name in zip(axes, old_values):
        axis.bar(["Phase 0I", "Phase 0J"], [old_values[name], new_values[name]])
        axis.set_title(name)
        axis.grid(axis="y", alpha=0.25)
        if name in thresholds:
            value, direction = thresholds[name]
            axis.axhline(value, linestyle="--", color="tab:red", label=f"{direction} gate")
            axis.legend(fontsize=8)
    fig.suptitle("Phase 0I vs Phase 0J — fixed wide-stop comparison")
    fig.tight_layout()
    fig.savefig(plots / "phase0i_vs_phase0j.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 5))
    for row in audit["rows"]:
        bead_rows = row["per_bead"]
        axis.plot(
            [bead["bead_index"] for bead in bead_rows],
            [bead["progress_gap"] for bead in bead_rows],
            marker="o", markersize=3, label=str(row["seed"]),
        )
    axis.axhline(0.01, linestyle="--", color="tab:red", label="official mean gate 0.01 m")
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xlabel("ordered bead index")
    axis.set_ylabel("free − hidden progress (m)")
    axis.set_title("Phase 0J per-bead progress gaps (diagnostic only)")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(plots / "per_bead_progress_gap.png", dpi=160)
    plt.close(fig)

    segment_names = ["blocked", "pulled_side", "trailing_side"]
    x = np.arange(len(audit["rows"]), dtype=np.float64)
    width = 0.24
    fig, axis = plt.subplots(figsize=(9, 5))
    for offset, name in enumerate(segment_names):
        values = [
            row["segments"].get(name, {}).get("progress_gap", np.nan)
            for row in audit["rows"]
        ]
        positions = x + (offset - 1) * width
        axis.bar(
            positions,
            values,
            width=width,
            label=name,
        )
        for position, value in zip(positions, values):
            if not np.isfinite(value):
                axis.text(
                    position, 0.00015, "N/A", ha="center", va="bottom",
                    rotation=90, fontsize=8, color="0.35",
                )
    axis.axhline(0.01, linestyle="--", color="tab:red", label="official all-bead gate 0.01 m")
    axis.set_xticks(x)
    axis.set_xticklabels([str(row["seed"]) for row in audit["rows"]])
    axis.set_xlabel("seed")
    axis.set_ylabel("segment mean progress gap (m)")
    axis.set_title("Wide-stop segment localization (diagnostic only)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(plots / "segment_progress_gap.png", dpi=160)
    plt.close(fig)

    pairs = current["topology"]["pairs"]
    fig, axis = plt.subplots(figsize=(7, 5))
    for pair in pairs:
        axis.scatter(
            pair["preload_end_max_abs_xy"], pair["main_branch_fde"], s=65
        )
        axis.annotate(
            str(pair["seed"]),
            (pair["preload_end_max_abs_xy"], pair["main_branch_fde"]),
            xytext=(5, 5), textcoords="offset points",
        )
    axis.axvline(0.004, linestyle="--", color="tab:blue", label="preload max 0.004 m")
    axis.axhline(0.015, linestyle=":", color="tab:red", label="FDE min 0.015 m")
    axis.set_xlabel("preload visible difference (m)")
    axis.set_ylabel("main FDE (m)")
    axis.set_title("Phase 0J fixed wide-stop: preload vs FDE")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(plots / "preload_vs_fde.png", dpi=160)
    plt.close(fig)
    print(plots)


if __name__ == "__main__":
    main()
