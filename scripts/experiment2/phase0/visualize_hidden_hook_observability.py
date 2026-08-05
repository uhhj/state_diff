#!/usr/bin/env python3
"""Render Phase 0H formal-sensor and Oracle-alignment audit plots."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_timeline(repo_root: Path, relative: str):
    with np.load(repo_root / relative, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="reports/experiment2/phase0_hidden_hook/phase0h/summary.json",
    )
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()
    summary_path = Path(args.input).resolve()
    output_root = summary_path.parent
    repo_root = output_root.parents[3]
    summary = load_json(summary_path)
    audit = load_json(output_root / summary["sensor_audit"])
    alignment = load_json(output_root / summary["contact_alignment"])
    plots = output_root / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    trace_rows = [row for row in audit["rows"] if row["condition"] != "free_hidden_difference"]
    by_group = {}
    for row in trace_rows:
        by_group.setdefault(row["group_id"], {})[row["condition"]] = row

    for group_id, conditions in sorted(by_group.items()):
        fig, axes = plt.subplots(4, 1, figsize=(12, 10), dpi=args.dpi, sharex=True)
        for condition, color in (("free", "tab:blue"), ("hidden_hook", "tab:orange")):
            data = load_timeline(repo_root, conditions[condition]["timeline"])
            step = data["physics_step"]
            axes[0].plot(step, np.linalg.norm(data["joint_motor_torque"], axis=1), color=color, label=condition)
            reaction = data["joint_reaction_force_torque"]
            axes[1].plot(step, np.linalg.norm(reaction.reshape(reaction.shape[0], -1), axis=1), color=color)
            axes[2].plot(step, np.linalg.norm(data["ee_constraint_force_xyz"], axis=1), color=color)
            axes[3].step(step, data["probe_phase_index"], where="post", color=color)
            if condition == "hidden_hook":
                active = data["oracle_contact_active_beads_privileged"] > 0
                axes[3].fill_between(step, 0, 4, where=active, alpha=0.16, color="red", label="Oracle contact (privileged)")
        axes[0].set_ylabel("motor torque norm")
        axes[1].set_ylabel("joint reaction norm")
        axes[2].set_ylabel("EE constraint force")
        axes[3].set_ylabel("probe phase index")
        axes[3].set_xlabel("physics step")
        axes[0].legend()
        axes[3].legend(loc="upper right")
        for ax in axes:
            ax.grid(alpha=0.25)
        fig.suptitle(f"Phase 0H raw sensor timeline: {group_id}")
        fig.tight_layout()
        fig.savefig(plots / f"sensor_timeline_{group_id}.png")
        plt.close(fig)

    difference_rows = [row for row in audit["rows"] if row["condition"] == "free_hidden_difference"]
    signal_names = list(difference_rows[0]["signals"])
    x = np.arange(len(signal_names))
    width = 0.8 / len(difference_rows)
    fig, ax = plt.subplots(figsize=(13, 6), dpi=args.dpi)
    for index, row in enumerate(difference_rows):
        values = [row["signals"][key]["root_mean_square_difference"] for key in signal_names]
        ax.bar(x - 0.4 + width / 2 + index * width, values, width, label=str(row["seed"]))
    ax.set_xticks(x)
    ax.set_xticklabels([name.replace("sensor_", "") for name in signal_names], rotation=30, ha="right")
    ax.set_ylabel("aligned free-hidden RMS difference")
    ax.set_title("Phase 0H formal sensor comparison")
    ax.legend(title="seed")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots / "formal_sensor_free_hidden_comparison.png")
    plt.close(fig)

    hidden_alignment = [row for row in alignment["rows"] if row["condition"] == "hidden_hook"]
    peak_names = list(hidden_alignment[0]["formal_sensor_peak_alignment"])
    fig, ax = plt.subplots(figsize=(12, 6), dpi=args.dpi)
    for index, row in enumerate(hidden_alignment):
        lags = [row["formal_sensor_peak_alignment"][name]["peak_lag_from_oracle_contact_steps"] for name in peak_names]
        ax.plot(peak_names, lags, marker="o", label=str(row["seed"]))
    ax.axhline(0, color="red", linestyle="--", label="Oracle contact onset (privileged)")
    ax.set_ylabel("formal sensor peak lag (physics steps)")
    ax.set_title("Phase 0H contact alignment audit")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(title="seed")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots / "contact_alignment.png")
    plt.close(fig)

    topology = summary["topology"]
    classifiers = topology["classifiers"]
    labels = ["RGB raw", "RGB delta", "formal sensor", "Oracle (privileged)"]
    values = [
        classifiers["vision_raw"]["accuracy"],
        classifiers["vision_delta"]["accuracy"],
        classifiers["formal_sensor"]["accuracy"],
        classifiers["oracle_contact"]["accuracy"],
    ]
    fig, ax = plt.subplots(figsize=(9, 5), dpi=args.dpi)
    ax.bar(labels, values)
    ax.axhline(0.5, linestyle=":", color="black", label="random")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("leave-one-seed-out accuracy")
    ax.set_title("Phase 0H observability screen")
    ax.tick_params(axis="x", rotation=20)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots / "modality_accuracy.png")
    plt.close(fig)

    for path in sorted(plots.glob("*.png")):
        print(path)


if __name__ == "__main__":
    main()
