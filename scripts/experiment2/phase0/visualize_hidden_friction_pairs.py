#!/usr/bin/env python3
"""Create non-privileged paired plots and optional GIFs."""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiment2.phase0.common import (  # noqa: E402
    ordered_bead_distance,
    phase_slice,
    resample_indices,
)


def _load_trace(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def _load_group(root: Path, group_id: str):
    raw = root / "raw"
    free = _load_trace(raw / f"{group_id}_free.npz")
    hidden = _load_trace(raw / f"{group_id}_hidden_high_friction.npz")
    free_meta = json.loads((raw / f"{group_id}_free.json").read_text(encoding="utf-8"))
    hidden_meta = json.loads(
        (raw / f"{group_id}_hidden_high_friction.json").read_text(encoding="utf-8")
    )
    pair = json.loads((raw / f"{group_id}_pair.json").read_text(encoding="utf-8"))
    return free, hidden, free_meta, hidden_meta, pair


def _phase_frame(trace: Dict[str, np.ndarray], phase: str, last: bool = True) -> np.ndarray:
    frames = phase_slice(trace, phase)["bead_positions"]
    if frames.shape[0] == 0:
        raise ValueError(f"phase {phase!r} has no frames")
    return frames[-1 if last else 0]


def _fixed_limits(free: np.ndarray, hidden: np.ndarray) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    values = np.concatenate([free[..., :2].reshape(-1, 2), hidden[..., :2].reshape(-1, 2)])
    lower = np.min(values, axis=0)
    upper = np.max(values, axis=0)
    pad = max(float(np.max(upper - lower)) * 0.08, 0.01)
    return (float(lower[0] - pad), float(upper[0] + pad)), (
        float(lower[1] - pad),
        float(upper[1] + pad),
    )


def make_overview(
    root: Path,
    group_id: str,
    show_privileged: bool,
    dpi: int,
) -> Path:
    free, hidden, _, hidden_meta, pair = _load_group(root, group_id)
    metrics = pair["metrics"]
    xlim, ylim = _fixed_limits(free["bead_positions"], hidden["bead_positions"])
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=dpi)

    shape_ax = axes[0, 0]
    frames = [
        ("initial free", free["bead_positions"][0], "C0", "--"),
        ("initial hidden", hidden["bead_positions"][0], "C1", "--"),
        ("preload free", _phase_frame(free, "preload"), "C0", ":"),
        ("preload hidden", _phase_frame(hidden, "preload"), "C1", ":"),
        ("main free", _phase_frame(free, "main_pull"), "C0", "-"),
        ("main hidden", _phase_frame(hidden, "main_pull"), "C1", "-"),
    ]
    for label, beads, color, style in frames:
        shape_ax.plot(beads[:, 0], beads[:, 1], marker="o", ms=2, color=color, ls=style, label=label)
    if show_privileged:
        privileged = hidden_meta["privileged_state"]
        center = privileged["patch_center_xy"]
        radius = privileged["friction_model"]["config"]["patch_radius"]
        shape_ax.add_patch(
            plt.Circle(center, radius, fill=False, ls="--", color="black", label="audit only")
        )
    shape_ax.set(xlim=xlim, ylim=ylim, title="Ordered cable shapes", xlabel="x", ylabel="y")
    shape_ax.set_aspect("equal", adjustable="box")
    shape_ax.legend(fontsize=7)

    contact_ax = axes[0, 1]
    for label, trace, color in (("free", free, "C0"), ("hidden", hidden, "C1")):
        steps = trace["physics_step"]
        contact_ax.plot(steps, trace["contact_force_norm"], label=label, color=color)
        for phase, alpha in (("preload", 0.08), ("main_pull", 0.05)):
            sliced = phase_slice(trace, phase)
            if sliced["physics_step"].size:
                contact_ax.axvspan(
                    sliced["physics_step"][0], sliced["physics_step"][-1], color=color, alpha=alpha
                )
    contact_ax.set(title="Hidden contact resistance history", xlabel="physics step", ylabel="sum force norm")
    contact_ax.legend()

    branch_ax = axes[1, 0]
    free_main = phase_slice(free, "main_pull")["bead_positions"]
    hidden_main = phase_slice(hidden, "main_pull")["bead_positions"]
    target = min(free_main.shape[0], hidden_main.shape[0], 128)
    free_aligned = free_main[resample_indices(free_main.shape[0], target)]
    hidden_aligned = hidden_main[resample_indices(hidden_main.shape[0], target)]
    distance = ordered_bead_distance(free_aligned, hidden_aligned)
    branch_ax.plot(np.linspace(0.0, 1.0, target), distance, color="C3")
    branch_ax.set(title="Main-pull future branch distance", xlabel="normalized main-pull time", ylabel="mean ordered-bead XY distance")

    text_ax = axes[1, 1]
    text_ax.axis("off")
    fields = [
        ("action hash match", metrics["action_hash_match"]),
        ("preload visible max", metrics["preload_end_max_abs_xy"]),
        ("contact impulse gap", metrics["contact_impulse_gap"]),
        ("main ADE", metrics["main_branch_ade"]),
        ("main FDE", metrics["main_branch_fde"]),
        ("free no-action drift", metrics["free_no_action_drift"]),
        ("hidden no-action drift", metrics["hidden_no_action_drift"]),
    ]
    text_ax.text(0.02, 0.98, "\n".join(f"{name}: {value}" for name, value in fields), va="top", family="monospace")
    fig.suptitle(f"Hidden-Friction Cable pair {group_id} (pilot only)")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    path = root / "plots" / f"pair_{group_id}_overview.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def make_gif(root: Path, group_id: str, frames_count: int, dpi: int) -> Path:
    free, hidden, _, _, _ = _load_group(root, group_id)
    free_beads = free["bead_positions"]
    hidden_beads = hidden["bead_positions"]
    free_indices = resample_indices(free_beads.shape[0], frames_count)
    hidden_indices = resample_indices(hidden_beads.shape[0], frames_count)
    xlim, ylim = _fixed_limits(free_beads, hidden_beads)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=dpi)
    lines = []
    for ax, title, color in zip(axes, ("free", "hidden_high_friction"), ("C0", "C1")):
        (line,) = ax.plot([], [], "o-", ms=3, color=color)
        ax.set(xlim=xlim, ylim=ylim, xlabel="x", ylabel="y")
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title)
        lines.append(line)

    def update(frame_index):
        free_index = free_indices[frame_index]
        hidden_index = hidden_indices[frame_index]
        for line, beads in zip(lines, (free_beads[free_index], hidden_beads[hidden_index])):
            line.set_data(beads[:, 0], beads[:, 1])
        axes[0].set_title(f"free | {free['phase'][free_index]}")
        axes[1].set_title(f"hidden_high_friction | {hidden['phase'][hidden_index]}")
        return lines

    movie = animation.FuncAnimation(fig, update, frames=frames_count, interval=50, blit=False)
    path = root / "plots" / f"pair_{group_id}.gif"
    try:
        movie.save(path, writer=animation.PillowWriter(fps=20))
    except Exception as exc:
        plt.close(fig)
        raise RuntimeError(f"Pillow GIF generation failed: {exc}") from exc
    plt.close(fig)
    return path


def make_aggregate(root: Path, metrics: Dict[str, Any], dpi: int) -> Path:
    keys = ("preload_end_max_abs_xy", "contact_impulse_gap", "main_branch_ade", "main_branch_fde")
    groups = metrics["groups"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=dpi)
    for ax, key in zip(axes.flat, keys):
        values = [group["metrics"][key] for group in groups]
        labels = [group["group_id"] for group in groups]
        ax.bar(np.arange(len(values)), values)
        ax.set_xticks(np.arange(len(values)))
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_title(key)
    fig.suptitle("Phase 0A pilot groups (not a significance claim)")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    path = root / "plots" / "aggregate_metrics.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--group-id")
    parser.add_argument("--show-privileged", action="store_true")
    parser.add_argument("--make-gif", action="store_true")
    args = parser.parse_args()
    root = Path(args.input).resolve()
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    config = json.loads((REPO_ROOT / metrics["config_path"]).read_text(encoding="utf-8"))
    dpi = int(config["visualization"]["dpi"])
    frames_count = int(config["visualization"]["resample_frames"])
    group_ids = [args.group_id] if args.group_id else [item["group_id"] for item in metrics["groups"]]
    for group_id in group_ids:
        print(make_overview(root, group_id, args.show_privileged, dpi))
        if args.make_gif:
            try:
                print(make_gif(root, group_id, frames_count, dpi))
            except RuntimeError as exc:
                warnings.warn(str(exc))
    print(make_aggregate(root, metrics, dpi))


if __name__ == "__main__":
    main()
