from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    import imageio.v2 as imageio
except ImportError as exc:  # pragma: no cover
    raise ImportError("imageio is required to save rollout videos. Install it with `pip install imageio imageio-ffmpeg`.") from exc

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MUJOCO_GL", "egl")

from state_diff.env.ccda_hose.config import FREE_INSERT, RIGHT_HIDDEN_JAM, HoseEnvConfig
from state_diff.env.ccda_hose.env_transparent import transparent_scripted_rollout


def _frame_diff(frames) -> float:
    if len(frames) < 2:
        return 0.0
    a = frames[0].astype(np.float32)
    b = frames[-1].astype(np.float32)
    return float(np.mean(np.abs(a - b)))


def _save_video(frames, path: Path, fps: int = 20) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        raise RuntimeError("No frames were recorded.")
    imageio.mimsave(path, frames, fps=fps)


def _plot_hose_trajectories(rollouts, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    colors = {
        FREE_INSERT: "tab:blue",
        RIGHT_HIDDEN_JAM: "tab:red",
    }
    labels = {
        0: "plug",
        1: "hose_0",
        2: "hose_1",
        3: "hose_2",
        4: "hose_3",
    }

    for plane, axes, ylabel, out_name in [
        ("XY", (0, 1), "y", "hose_keypoint_trajectory_xy.png"),
        ("XZ", (0, 2), "z", "hose_keypoint_trajectory_xz.png"),
    ]:
        plt.figure(figsize=(8, 5))
        for rollout in rollouts:
            keypoints = rollout["trace"]["hose_keypoints"]
            cond = rollout["condition"]
            n_keypoints = min(5, keypoints.shape[1])
            for i in range(n_keypoints):
                alpha = 0.9 if i == 0 else 0.35
                linestyle = "-" if i == 0 else "--"
                label = f"{cond} / {labels.get(i, f'kp_{i}')}"
                plt.plot(
                    keypoints[:, i, axes[0]],
                    keypoints[:, i, axes[1]],
                    color=colors.get(cond),
                    alpha=alpha,
                    linestyle=linestyle,
                    linewidth=1.8 if i == 0 else 1.0,
                    label=label,
                )
        plt.xlabel("x")
        plt.ylabel(ylabel)
        plt.title(f"Hose keypoint trajectory ({plane})")
        plt.legend(fontsize=8)
        plt.axis("equal")
        plt.tight_layout()
        plt.savefig(out_dir / out_name, dpi=170)
        plt.close()


def _write_plug_csv(rollouts, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "condition",
                "step",
                "x",
                "y",
                "z",
                "insertion_depth",
                "lateral_offset",
                "max_curvature",
            ]
        )
        for rollout in rollouts:
            trace = rollout["trace"]
            plug = trace["plug_pos"]
            for step in range(plug.shape[0]):
                writer.writerow(
                    [
                        rollout["condition"],
                        step,
                        float(plug[step, 0]),
                        float(plug[step, 1]),
                        float(plug[step, 2]),
                        float(trace["insertion_depth"][step, 0]),
                        float(trace["lateral_offset"][step, 0]),
                        float(trace["max_curvature"][step, 0]),
                    ]
                )


def _diagnostic_line(rollout, diff: float) -> str:
    trace = rollout["trace"]
    initial_depth = float(trace["insertion_depth"][0, 0])
    final_depth = float(trace["insertion_depth"][-1, 0])
    initial_plug = trace["plug_pos"][0].tolist()
    final_plug = trace["plug_pos"][-1].tolist()
    return (
        f"* `{rollout['condition']}`: frame_diff={diff:.3f}, "
        f"initial_depth={initial_depth:.6f}, final_depth={final_depth:.6f}, "
        f"initial_plug_pos={initial_plug}, final_plug_pos={final_plug}, "
        f"final_branch=`{rollout['final_branch']}`, final_success={rollout['final_success']}"
    )


def _write_report(out_dir: Path, camera: str, diagnostics) -> None:
    md = f"""# Stage 1 Transparent Socket Visual Debug

Camera: `{camera}`

This transparent socket environment is only for debug visualization. It should not be used as formal model input and does not change the CCDA data definition.

## Debug Scope

* Socket wall geoms keep collision enabled; only their material alpha changes.
* The hidden jam block is semi-transparent red to show the right-side jam location.
* The transparent visual view can reveal hidden contact and should not be used for official CCDA audit inputs.
* The `side_top` camera is the recommended debug view for socket, plug, hose-front, insertion, and lateral-jam motion.
* If a video still looks static, first inspect plug x and insertion depth over time.

## Generated Files

* `free_insert_{camera}_transparent.mp4`
* `right_hidden_jam_{camera}_transparent.mp4`
* `hose_keypoint_trajectory_xy.png`
* `hose_keypoint_trajectory_xz.png`
* `plug_trajectory_xyz.csv`

## Frame Diagnostics

{chr(10).join(diagnostics)}
"""
    (out_dir / "transparent_visual_report.md").write_text(md, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--camera", type=str, default="side_top", choices=["front", "top", "side", "side_top"])
    parser.add_argument("--out-dir", type=str, default="reports/ccda_hose_stage1/transparent_visual")
    parser.add_argument("--show-occluder", action="store_true")
    args = parser.parse_args()

    cfg = HoseEnvConfig()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rollouts = [
        transparent_scripted_rollout(
            FREE_INSERT,
            seed=args.seed,
            config=cfg,
            record_frames=True,
            camera_name=args.camera,
            show_occluder=args.show_occluder,
        ),
        transparent_scripted_rollout(
            RIGHT_HIDDEN_JAM,
            seed=args.seed + 1,
            config=cfg,
            record_frames=True,
            camera_name=args.camera,
            show_occluder=args.show_occluder,
        ),
    ]

    diagnostics = []
    for rollout in rollouts:
        video_path = out_dir / f"{rollout['condition']}_{args.camera}_transparent.mp4"
        _save_video(rollout["frames"], video_path)
        diff = _frame_diff(rollout["frames"])
        line = _diagnostic_line(rollout, diff)
        diagnostics.append(line)
        print(f"Saved {video_path}; {line.lstrip('* ')}")
        if diff < 1.0:
            print(
                "WARNING: rendered frames have very small pixel difference; video may look static. "
                "Check gripper/plug trajectory and camera view."
            )

    _plot_hose_trajectories(rollouts, out_dir)
    _write_plug_csv(rollouts, out_dir / "plug_trajectory_xyz.csv")
    _write_report(out_dir, args.camera, diagnostics)
    print(f"Saved transparent visual report to {out_dir / 'transparent_visual_report.md'}")


if __name__ == "__main__":
    main()
