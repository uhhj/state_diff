#!/usr/bin/env python3
"""Visualize the fixed Phase 0K tension-extension audit."""
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


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", default="reports/experiment2/phase0_hidden_hook/phase0k"
    )
    parser.add_argument(
        "--baseline", default="reports/experiment2/phase0_hidden_hook/phase0i"
    )
    args = parser.parse_args()
    root = (REPO_ROOT / args.input).resolve()
    baseline_root = (REPO_ROOT / args.baseline).resolve()
    summary = load(root / "summary.json")
    audit = load(root / "tension_outcome_audit.json")
    baseline = load(baseline_root / "candidate_delayed_z_latch_v1.json")
    current = summary["topology"]

    names = [
        "preload", "ADE", "FDE", "amplification",
        "official gap", "RGB delta", "formal sensor",
    ]
    old = [
        baseline["median"]["preload_end_max_abs_xy"],
        baseline["median"]["main_branch_ade"],
        baseline["median"]["main_branch_fde"],
        baseline["median"]["branch_amplification"],
        baseline["median"]["mean_cable_progress_gap"],
        baseline["classifiers"]["vision_delta"]["accuracy"],
        baseline["classifiers"]["formal_sensor"]["accuracy"],
    ]
    new = [
        current["median"]["preload_end_max_abs_xy"],
        current["median"]["main_branch_ade"],
        current["median"]["main_branch_fde"],
        current["median"]["branch_amplification"],
        current["median"]["mean_cable_progress_gap"],
        current["classifiers"]["vision_delta"]["accuracy"],
        current["classifiers"]["formal_sensor"]["accuracy"],
    ]
    fig, axes = plt.subplots(1, len(names), figsize=(22, 4.5))
    gates = {"preload": 0.004, "FDE": 0.015, "official gap": 0.01}
    for axis, name, old_value, new_value in zip(axes, names, old, new):
        axis.bar(["Phase 0I", "Phase 0K"], [old_value, new_value])
        axis.set_title(name)
        axis.grid(axis="y", alpha=0.25)
        if name in gates:
            axis.axhline(gates[name], linestyle="--", color="tab:red", label="official gate")
            axis.legend(fontsize=7)
    fig.suptitle("Phase 0I vs Phase 0K — fixed same-end tension extension")
    fig.tight_layout()
    fig.savefig(root / "phase0i_vs_phase0k.png", dpi=160)
    plt.close(fig)

    median = audit["median"]
    stages = ["stage 1", "stage 2", "post-release final"]
    gaps = [
        median["stage1_progress_gap"],
        median["stage2_progress_gap"],
        median["final_progress_gap"],
    ]
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.bar(stages, gaps)
    axis.axhline(0.01, linestyle="--", color="tab:red", label="official final gate 0.01 m")
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_ylabel("mean cable progress gap (m)")
    axis.set_title("Phase 0K stage progress gaps (diagnostic only)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(root / "stage_progress_gap.png", dpi=160)
    plt.close(fig)

    distances = [
        median["stage1_branch_distance"],
        median["stage2_branch_distance"],
        current["median"]["main_branch_fde"],
    ]
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.bar(["stage 1 mean", "stage 2 mean", "final FDE"], distances)
    axis.set_ylabel("branch distance (m)")
    axis.set_title("Phase 0K branch distance by tension stage")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(root / "stage_branch_distance.png", dpi=160)
    plt.close(fig)

    x = np.arange(len(audit["rows"]), dtype=np.float64)
    width = 0.24
    fig, axis = plt.subplots(figsize=(9, 5))
    for offset, (label, key) in enumerate((
        ("stage 1", "stage1"),
        ("stage 2", "stage2"),
        ("final official", "final"),
    )):
        axis.bar(
            x + (offset - 1) * width,
            [row[key]["mean_progress_gap"] for row in audit["rows"]],
            width=width,
            label=label,
        )
    axis.axhline(0.01, linestyle="--", color="tab:red", label="official final gate 0.01 m")
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(x)
    axis.set_xticklabels([str(row["seed"]) for row in audit["rows"]])
    axis.set_xlabel("seed")
    axis.set_ylabel("mean cable progress gap (m)")
    axis.set_title("Per-seed fixed tension-extension progress")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(root / "per_seed_progress_gap.png", dpi=160)
    plt.close(fig)
    print(root)


if __name__ == "__main__":
    main()
