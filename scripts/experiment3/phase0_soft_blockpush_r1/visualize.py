"""Generate R1 coupon-independent Pair visual evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def _plot(path: Path, x: np.ndarray, values: dict, ylabel: str,
          title: str, test_end: int = -1) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, series in values.items():
        ax.plot(x, series, label=label)
    if test_end >= 0:
        ax.axvline(test_end, color="black", linestyle="--", label="test_end")
    ax.set(xlabel="physics step", ylabel=ylabel, title=title)
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(path, dpi=150); plt.close(fig)


def _combine(left_path: Path, right_path: Path, output: Path, fps: float) -> None:
    left, right = cv2.VideoCapture(str(left_path)), cv2.VideoCapture(str(right_path))
    width, height = int(left.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
        left.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (2 * width, height))
    if not writer.isOpened():
        raise RuntimeError("unable to create comparison video")
    while True:
        ok_l, frame_l = left.read(); ok_r, frame_r = right.read()
        if not ok_l or not ok_r:
            break
        cv2.putText(frame_l, "Free", (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    .6, (255, 255, 255), 1)
        cv2.putText(frame_r, "High", (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    .6, (255, 255, 255), 1)
        writer.write(np.concatenate([frame_l, frame_r], axis=1))
    writer.release(); left.release(); right.release()


def visualize(pair_dir: Path, report_dir: Path) -> None:
    """Write all prescribed R1 Pair plots and official comparison video."""
    pair_dir, report_dir = Path(pair_dir), Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    meta = json.loads((pair_dir / "metadata.json").read_text(encoding="utf-8"))
    free = _load(pair_dir / "uniform_low" / "trajectory.npz")
    high = _load(pair_dir / "right_local_high" / "trajectory.npz")
    ts = _load(report_dir / "timeseries.npz")
    steps = ts["physics_step"]
    test_indices = np.flatnonzero(ts["phase"].astype(str) == "test")
    test_end = int(steps[test_indices[-1]]) if len(test_indices) else -1
    _plot(report_dir / "physics_rate_gap.png", steps,
          {"formal fused": ts["physics_formal_gap"],
           "visible m": ts["visible_gap"]}, "scaled gap / m",
          "Physics-rate formal and visual gaps", test_end)
    _plot(report_dir / "policy_rate_gap.png", ts["policy_step"],
          {"formal fused": ts["policy_formal_gap"],
           "macro visible m": ts["policy_visible_gap"]}, "scaled gap / m",
          "10 Hz policy-rate gaps", test_end)
    _plot(report_dir / "local_anchor.png", steps,
          {"patch slip reduction": ts["patch_slip_reduction"],
           "local-gradient advantage": ts["local_gradient_advantage"]},
          "displacement advantage (m)", "Local anchoring", test_end)
    _plot(report_dir / "deformation_branch.png", steps,
          {"visible": ts["visible_gap"], "full": ts["full_gap"],
           "rigid-aligned": ts["rigid_gap"]}, "RMSE (m)",
          "Global and non-rigid branch", test_end)
    _plot(report_dir / "target_progress.png", steps,
          {"Free": ts["free_progress"], "High": ts["high_progress"]},
          "target progress (m)", "Test-end target progress", test_end)
    _plot(report_dir / "spring_stability.png", steps,
          {"Free energy": ts["free_energy"], "High energy": ts["high_energy"],
           "Free max strain": np.max(np.abs(ts["free_strain"]), axis=1),
           "High max strain": np.max(np.abs(ts["high_strain"]), axis=1)},
          "energy / strain", "Spring energy and edge strain", test_end)
    base = _load(pair_dir / "base_state.npz")["node_positions"]
    inside = np.asarray(meta["initial_patch_bottom_indices"], dtype=int)
    outside = np.asarray(meta["initial_outside_bottom_indices"], dtype=int)
    bounds = meta["config"]["floor"]
    cx, cy = bounds["patch_center_xy"]; sx, sy = bounds["patch_size_xy"]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(base[outside, 0], base[outside, 1], label="outside")
    ax.scatter(base[inside, 0], base[inside, 1], label="inside")
    rect = plt.Rectangle((cx - sx / 2, cy - sy / 2), sx, sy,
                         fill=False, color="red", linewidth=2,
                         label="ORACLE_ONLY patch")
    ax.add_patch(rect); ax.axis("equal"); ax.grid(alpha=.25); ax.legend()
    ax.set_title("ORACLE_ONLY initial patch membership"); fig.tight_layout()
    fig.savefig(report_dir / "oracle_patch_membership.png", dpi=150); plt.close(fig)
    _combine(pair_dir / "uniform_low" / "video.mp4",
             pair_dir / "right_local_high" / "video.mp4",
             report_dir / "free_vs_high.mp4",
             float(meta["config"]["camera"]["fps"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-dir", required=True)
    parser.add_argument("--report-dir", required=True)
    args = parser.parse_args()
    visualize(Path(args.pair_dir), Path(args.report_dir))


if __name__ == "__main__":
    main()
