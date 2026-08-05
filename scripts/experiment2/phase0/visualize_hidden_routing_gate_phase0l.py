#!/usr/bin/env python3
"""Render the four fixed Phase 0L routing-gate audit plots."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="reports/experiment2/phase0_hidden_routing_gate/phase0l",
    )
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()
    root = Path(args.input).resolve()
    if root.is_file():
        root = root.parent

    summary = load_json(root / "summary.json")
    outcome = load_json(root / "routing_outcome_audit.json")
    candidate_report = str(summary.get(
        "candidate_report",
        "candidate_hidden_routing_gate_v1.json",
    ))
    topology = load_json(root / candidate_report)
    stage_name = str(summary.get("stage", "Experiment2 Phase 0L"))
    rows = sorted(outcome["rows"], key=lambda row: int(row["seed"]))
    labels = [str(row["seed"]) for row in rows]
    x = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(9, 5), dpi=args.dpi)
    width = 0.25
    ax.bar(x - width, [int(row["free_endpoint_target_success"]) for row in rows], width,
           label="free success")
    ax.bar(x, [int(row["hidden_endpoint_target_success"]) for row in rows], width,
           label="hidden success")
    ax.bar(x + width, [row["routing_success_gap"] for row in rows], width,
           label="free - hidden gap")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label="official gate")
    ax.set(xticks=x, xticklabels=labels, xlabel="seed", ylabel="binary outcome",
           title=f"{stage_name} endpoint routing success by seed")
    ax.set_ylim(-1.1, 1.25)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / "routing_success_by_seed.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5), dpi=args.dpi)
    ax.plot(x, [row["free_endpoint_target_margin"] for row in rows], "o-",
            label="free target margin")
    ax.plot(x, [row["hidden_endpoint_target_margin"] for row in rows], "s-",
            label="hidden target margin")
    ax.axhline(0.0, color="black", linestyle="--", label="target plane")
    ax.set(xticks=x, xticklabels=labels, xlabel="seed", ylabel="normal margin (m)",
           title=f"{stage_name} pulled-endpoint target margin")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / "endpoint_target_margin_by_seed.png")
    plt.close(fig)

    med = topology["median"]
    physical_labels = ["preload", "ADE", "FDE", "amplification", "engagement"]
    physical_values = [
        med["preload_end_max_abs_xy"],
        med["main_branch_ade"],
        med["main_branch_fde"],
        med["branch_amplification"],
        med["hidden_gate_engagement_fraction"],
    ]
    fig, ax = plt.subplots(figsize=(9, 5), dpi=args.dpi)
    ax.bar(physical_labels, physical_values)
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.set(ylabel="median value (symlog)",
           title=f"{stage_name} physical observability metrics")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(root / "physical_observability_metrics.png")
    plt.close(fig)

    classifiers = topology["classifiers"]
    classifier_labels = ["RGB raw", "RGB delta", "formal sensor", "Oracle"]
    classifier_values = [
        classifiers["vision_raw"]["accuracy"],
        classifiers["vision_delta"]["accuracy"],
        classifiers["formal_sensor"]["accuracy"],
        classifiers["oracle_contact"]["accuracy"],
    ]
    fig, ax = plt.subplots(figsize=(9, 5), dpi=args.dpi)
    ax.bar(classifier_labels, classifier_values)
    ax.set_ylim(0, 1.05)
    ax.set(ylabel="grouped accuracy", title=f"{stage_name} classifier audit")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(root / "classifier_accuracy.png")
    plt.close(fig)

    for name in (
        "routing_success_by_seed.png",
        "endpoint_target_margin_by_seed.png",
        "physical_observability_metrics.png",
        "classifier_accuracy.png",
    ):
        print(root / name)


if __name__ == "__main__":
    main()
