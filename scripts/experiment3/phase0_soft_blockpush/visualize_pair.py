"""Create the required Phase 0B plots and side-by-side branch video."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _npz(path: Path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _plot(path: Path, x, series, ylabel: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, values in series.items():
        ax.plot(x, values, label=label, linewidth=1.2)
    ax.set(xlabel="absolute physics step", ylabel=ylabel, title=title)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def combine_video(free_path: Path, high_path: Path, output: Path,
                  fps: float) -> None:
    left, right = cv2.VideoCapture(str(free_path)), cv2.VideoCapture(str(high_path))
    width = int(left.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(left.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width * 2, height))
    if not writer.isOpened():
        raise RuntimeError("unable to create comparison video")
    while True:
        ok_l, frame_l = left.read()
        ok_r, frame_r = right.read()
        if not ok_l or not ok_r:
            break
        cv2.putText(frame_l, "Free / uniform low", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
        cv2.putText(frame_r, "High / local patch", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
        writer.write(np.concatenate([frame_l, frame_r], axis=1))
    writer.release()
    left.release()
    right.release()


def visualize(pair_dir: Path, report_dir: Path) -> None:
    pair_dir, report_dir = Path(pair_dir), Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    meta = json.loads((pair_dir / "metadata.json").read_text(encoding="utf-8"))
    free = _npz(pair_dir / "uniform_low" / "trajectory.npz")
    high = _npz(pair_dir / "right_local_high" / "trajectory.npz")
    ts = _npz(report_dir / "timeseries.npz")
    steps = ts["physics_step"]
    formal = {key[len("formal_"):]: value for key, value in ts.items()
              if key.startswith("formal_")}
    formal["fused"] = ts["fused_gap"]
    _plot(report_dir / "sensor_gap.png", steps, formal,
          "standardized RMS branch gap", "Formal sensor branch gaps")
    _plot(report_dir / "visible_gap.png", steps, {
        "visible": ts["visible_gap"], "full": ts["full_gap"],
        "rigid-aligned": ts["rigid_gap"]}, "RMSE (m)", "Geometric branch gaps")
    _plot(report_dir / "oracle_friction.png", steps, {
        "Free tangential": free["oracle_patch_tangential_force"],
        "High tangential": high["oracle_patch_tangential_force"],
        "oracle composite gap": ts["oracle_gap"]}, "force (N) / composite",
        "ORACLE_ONLY patch friction response")
    _plot(report_dir / "progress.png", steps, {
        "Free": ts["free_progress"], "High": ts["high_progress"]},
        "target progress (m)", "Center-of-mass target progress")
    _plot(report_dir / "strain.png", steps, {
        "Free max |strain|": np.max(np.abs(ts["free_strain"]), axis=1),
        "High max |strain|": np.max(np.abs(ts["high_strain"]), axis=1)},
        "absolute structural strain", "Structural edge strain")
    _plot(report_dir / "com_trajectory.png", ts["free_com"][:, 0], {
        "Free y": ts["free_com"][:, 1]}, "COM y (m)", "Free COM XY trajectory")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(ts["free_com"][:, 0], ts["free_com"][:, 1], label="Free")
    ax.plot(ts["high_com"][:, 0], ts["high_com"][:, 1], label="High")
    ax.set(xlabel="COM x (m)", ylabel="COM y (m)", title="COM trajectories")
    ax.axis("equal"); ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(report_dir / "com_trajectory.png", dpi=150); plt.close(fig)
    top = np.asarray(meta["top_indices"], dtype=int)
    fig, ax = plt.subplots(figsize=(7, 6))
    for label, values, color in (("Free", free["node_positions"][-1, top], "C0"),
                                 ("High", high["node_positions"][-1, top], "C1")):
        ax.scatter(values[:, 0], values[:, 1], label=label, s=20, alpha=.8, c=color)
    ax.set(xlabel="x (m)", ylabel="y (m)", title="Final top-keypoint overlay")
    ax.axis("equal"); ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(report_dir / "keypoint_branch.png", dpi=150); plt.close(fig)
    _plot(report_dir / "deformation_branch.png", steps, {
        "visible": ts["visible_gap"], "rigid-aligned": ts["rigid_gap"]},
        "RMSE (m)", "Absolute versus non-rigid branch")
    combine_video(pair_dir / "uniform_low" / "video.mp4",
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
