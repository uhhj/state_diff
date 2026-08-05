#!/usr/bin/env python3
"""Render the four bounded Phase 0I Z-latch audit plots."""
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

from scripts.experiment2.phase0.phase0i_sensor_features import (
    LOAD_STAGES,
    UNLOAD_STAGES,
    motion_stage_mask,
)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_trace(path):
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def scalar_signals(trace):
    reaction = np.asarray(trace["sensor_joint_reaction_force_torque"])
    return {
        "motor": np.linalg.norm(trace["sensor_joint_motor_torque"], axis=-1),
        "joint reaction force": np.linalg.norm(
            np.linalg.norm(reaction[..., :3], axis=-1), axis=-1
        ),
        "joint reaction torque": np.linalg.norm(
            np.linalg.norm(reaction[..., 3:], axis=-1), axis=-1
        ),
        "suction force": np.linalg.norm(trace["sensor_suction_force_xyz"], axis=-1),
        "suction torque": np.linalg.norm(trace["sensor_suction_torque_xyz"], axis=-1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", default="reports/experiment2/phase0_hidden_hook/phase0i"
    )
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()
    root = Path(args.input).resolve()
    if root.is_file():
        root = root.parent
    summary = load_json(root / "summary.json")
    topology = summary["topology"]
    pairs = sorted(topology["pairs"], key=lambda row: int(row["seed"]))
    plots = root / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=args.dpi)
    for row in pairs:
        ax.scatter(row["preload_end_max_abs_xy"], row["main_branch_fde"], s=90)
        ax.annotate(str(row["seed"]), (
            row["preload_end_max_abs_xy"], row["main_branch_fde"]
        ), xytext=(6, 5), textcoords="offset points")
    ax.axvline(summary["targets"]["max_median_preload_visible_difference"], linestyle="--")
    ax.axhline(summary["targets"]["min_median_main_branch_fde"], linestyle=":")
    ax.set(xlabel="preload visible difference (m)", ylabel="main FDE (m)",
           title="Phase 0I fixed Z-latch: preload vs FDE")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots / "preload_vs_fde.png")
    plt.close(fig)

    topology_id = summary["topology"]["candidate_id"]
    records = []
    for row in pairs:
        group = root / "raw" / topology_id / row["group_id"]
        pair = load_json(group / "pair.json")
        for condition, meta_key in (("free", "free_metadata"), ("hidden_hook", "hidden_metadata")):
            trace = load_trace(group / f"{condition}.npz")
            steps = trace["physics_step"]
            events = pair[meta_key]["motion_events"]
            masks = {
                "load": motion_stage_mask(steps, events, LOAD_STAGES),
                "unload": motion_stage_mask(steps, events, UNLOAD_STAGES),
            }
            signals = scalar_signals(trace)
            for window, mask in masks.items():
                records.append({
                    "condition": condition, "window": window,
                    "values": {name: float(np.mean(value[mask])) for name, value in signals.items()},
                })
    names = list(records[0]["values"])
    labels = ["free load", "hidden load", "free unload", "hidden unload"]
    means = []
    for condition, window in (("free", "load"), ("hidden_hook", "load"),
                              ("free", "unload"), ("hidden_hook", "unload")):
        subset = [row for row in records if row["condition"] == condition and row["window"] == window]
        means.append([np.mean([row["values"][name] for row in subset]) for name in names])
    x = np.arange(len(names))
    width = 0.2
    fig, ax = plt.subplots(figsize=(12, 6), dpi=args.dpi)
    for index, (label, values) in enumerate(zip(labels, means)):
        ax.bar(x + (index - 1.5) * width, values, width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_ylabel("mean robot-observable signal (symlog)")
    ax.set_title("Phase 0I frozen formal sensor windows")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots / "formal_sensor_window_compare.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=args.dpi)
    ax.bar([str(row["seed"]) for row in pairs], [row["mean_cable_progress_gap"] for row in pairs])
    ax.axhline(summary["targets"]["min_median_progress_gap"], linestyle="--")
    ax.set(xlabel="seed", ylabel="free-hidden mean cable progress gap (m)",
           title="Phase 0I progress gap by seed")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots / "progress_gap_by_seed.png")
    plt.close(fig)

    mechanism = load_json(root / "mechanism_audit.json")["rows"]
    labels = ["free preload", "hidden preload", "hidden main", "hidden post-release"]
    x = np.arange(len(mechanism))
    width = 0.2
    keys = [
        "free_preload_contact_fraction_privileged",
        "hidden_preload_contact_fraction_privileged",
        "hidden_main_contact_fraction_privileged",
        "hidden_post_release_contact_fraction_privileged",
    ]
    fig, ax = plt.subplots(figsize=(10, 5), dpi=args.dpi)
    for index, (label, key) in enumerate(zip(labels, keys)):
        ax.bar(x + (index - 1.5) * width, [row[key] for row in mechanism], width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([str(row["seed"]) for row in mechanism])
    ax.set_ylim(0, 1.05)
    ax.set(xlabel="seed", ylabel="contact sample fraction",
           title="Z-latch contact timing — Oracle privileged audit only")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots / "contact_timing_by_seed.png")
    plt.close(fig)

    for path in sorted(plots.glob("*.png")):
        print(path)


if __name__ == "__main__":
    main()
