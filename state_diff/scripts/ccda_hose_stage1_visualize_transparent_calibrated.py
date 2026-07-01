from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import fields
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

try:
    import imageio.v2 as imageio
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "imageio is required to save rollout videos. Install it with "
        "`pip install imageio imageio-ffmpeg`."
    ) from exc

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MUJOCO_GL", "egl")

from state_diff.env.ccda_hose.config import FREE_INSERT, RIGHT_HIDDEN_JAM, HoseEnvConfig
from state_diff.env.ccda_hose.env_transparent import transparent_scripted_rollout


DEFAULT_CONFIG_JSON = "reports/ccda_hose_stage1/calibration_v1/best_config.json"
DEFAULT_OUT_DIR = "reports/ccda_hose_stage1/transparent_calib_v1"


def _load_config(path: Path) -> HoseEnvConfig:
    if not path.exists():
        raise FileNotFoundError(
            f"Calibration config not found: {path}. Run ccda_hose_stage1_calibrate.py "
            "or pass --config-json to an existing HoseEnvConfig JSON file."
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    allowed = {f.name for f in fields(HoseEnvConfig)}
    kwargs = {k: v for k, v in raw.items() if k in allowed}
    if "jam_block_size" in kwargs:
        kwargs["jam_block_size"] = tuple(kwargs["jam_block_size"])
    if "jam_friction" in kwargs:
        kwargs["jam_friction"] = tuple(kwargs["jam_friction"])
    return HoseEnvConfig(**kwargs)


def _frame_diff(frames: List[np.ndarray]) -> float:
    if len(frames) < 2:
        return 0.0
    a = frames[0].astype(np.float32)
    b = frames[-1].astype(np.float32)
    return float(np.mean(np.abs(a - b)))


def _save_video(frames: List[np.ndarray], path: Path, fps: int = 20) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        raise RuntimeError("No frames were recorded.")
    imageio.mimsave(path, frames, fps=fps)


def _plot_plug_trajectories(rollouts: List[Dict[str, object]], out_dir: Path, marker_stride: int) -> None:
    colors = {FREE_INSERT: "tab:blue", RIGHT_HIDDEN_JAM: "tab:red"}
    for plane, axes, ylabel, out_name in [
        ("XY", (0, 1), "y", "plug_trajectory_xy.png"),
        ("XZ", (0, 2), "z", "plug_trajectory_xz.png"),
    ]:
        plt.figure(figsize=(8.0, 5.2))
        for rollout in rollouts:
            trace = rollout["trace"]
            plug = trace["plug_pos"]
            cond = rollout["condition"]
            color = colors.get(cond, "tab:gray")
            plt.plot(
                plug[:, axes[0]],
                plug[:, axes[1]],
                color=color,
                linewidth=2.0,
                label=f"{cond} plug",
            )
            samples = plug[::marker_stride]
            plt.scatter(samples[:, axes[0]], samples[:, axes[1]], color=color, s=18, alpha=0.7)
        plt.xlabel("x")
        plt.ylabel(ylabel)
        plt.title(f"Calibrated transparent plug trajectory ({plane})")
        plt.legend(fontsize=8)
        plt.axis("equal")
        plt.tight_layout()
        plt.savefig(out_dir / out_name, dpi=180)
        plt.close()


def _plot_hose_front_keypoints(rollouts: List[Dict[str, object]], out_dir: Path, marker_stride: int) -> None:
    colors = {FREE_INSERT: "tab:blue", RIGHT_HIDDEN_JAM: "tab:red"}
    names = {0: "plug", 1: "hose_0", 2: "hose_1", 3: "hose_2", 4: "hose_3"}
    for plane, axes, ylabel, out_name in [
        ("XY", (0, 1), "y", "hose_front_keypoints_xy.png"),
        ("XZ", (0, 2), "z", "hose_front_keypoints_xz.png"),
    ]:
        plt.figure(figsize=(8.0, 5.2))
        for rollout in rollouts:
            trace = rollout["trace"]
            keypoints = trace["hose_keypoints"]
            cond = rollout["condition"]
            color = colors.get(cond, "tab:gray")
            for kp_idx in range(min(5, keypoints.shape[1])):
                alpha = 0.95 if kp_idx == 0 else 0.42
                linestyle = "-" if kp_idx == 0 else "--"
                width = 2.0 if kp_idx == 0 else 1.0
                label = f"{cond} / {names.get(kp_idx, f'kp_{kp_idx}')}"
                plt.plot(
                    keypoints[:, kp_idx, axes[0]],
                    keypoints[:, kp_idx, axes[1]],
                    color=color,
                    alpha=alpha,
                    linestyle=linestyle,
                    linewidth=width,
                    label=label,
                )
                samples = keypoints[::marker_stride, kp_idx]
                plt.scatter(
                    samples[:, axes[0]],
                    samples[:, axes[1]],
                    color=color,
                    s=12 if kp_idx == 0 else 7,
                    alpha=0.45,
                )
        plt.xlabel("x")
        plt.ylabel(ylabel)
        plt.title(f"Calibrated transparent front-hose trajectories ({plane})")
        plt.legend(fontsize=7)
        plt.axis("equal")
        plt.tight_layout()
        plt.savefig(out_dir / out_name, dpi=180)
        plt.close()


def _write_plug_csv(rollouts: List[Dict[str, object]], path: Path) -> None:
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
                "jam_contact_force",
                "lateral_contact_force",
            ]
        )
        for rollout in rollouts:
            trace = rollout["trace"]
            plug = trace["plug_pos"]
            contacts = trace["privileged_contact"]
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
                        float(contacts[step, 2]),
                        float(contacts[step, 3]),
                    ]
                )


def _diagnostics(rollout: Dict[str, object], frame_diff: float) -> Dict[str, object]:
    trace = rollout["trace"]
    return {
        "condition": rollout["condition"],
        "frame_diff": frame_diff,
        "initial_insertion_depth": float(trace["insertion_depth"][0, 0]),
        "final_insertion_depth": float(trace["insertion_depth"][-1, 0]),
        "initial_plug_pos": [float(x) for x in trace["plug_pos"][0]],
        "final_plug_pos": [float(x) for x in trace["plug_pos"][-1]],
        "final_branch": str(rollout["final_branch"]),
        "final_success": bool(rollout["final_success"]),
        "max_jam_contact_force": float(np.max(trace["privileged_contact"][:, 2])),
        "max_lateral_contact_force": float(np.max(trace["privileged_contact"][:, 3])),
        "final_lateral_offset": float(trace["lateral_offset"][-1, 0]),
        "final_max_curvature": float(trace["max_curvature"][-1, 0]),
    }


def _write_report(
    out_dir: Path,
    camera: str,
    config_json: Path,
    cfg: HoseEnvConfig,
    show_occluder: bool,
    rows: List[Dict[str, object]],
) -> None:
    cfg_summary = {
        "hose_joint_stiffness": cfg.hose_joint_stiffness,
        "max_delta_per_env_step": cfg.max_delta_per_env_step,
        "push_distance": cfg.push_distance,
        "push_steps": cfg.push_steps,
        "jam_block_x": cfg.jam_block_x,
        "jam_block_y": cfg.jam_block_y,
        "jam_block_size": cfg.jam_block_size,
        "jam_friction": cfg.jam_friction,
        "lateral_offset_threshold": cfg.lateral_offset_threshold,
    }
    diag_md = []
    for row in rows:
        diag_md.append(
            f"| `{row['condition']}` | {row['frame_diff']:.3f} | "
            f"{row['initial_insertion_depth']:.6f} | {row['final_insertion_depth']:.6f} | "
            f"`{row['initial_plug_pos']}` | `{row['final_plug_pos']}` | "
            f"`{row['final_branch']}` | {row['final_success']} | "
            f"{row['max_jam_contact_force']:.6f} | {row['max_lateral_contact_force']:.6f} | "
            f"{row['final_lateral_offset']:.6f} | {row['final_max_curvature']:.6f} |"
        )

    md = f"""# Stage 1.5 Calibrated Transparent Socket Visual Debug

## Scope

This script uses the calibrated `HoseEnvConfig` from `{config_json}` and the transparent socket debug environment.
It is only for inspecting socket-internal plug and hose motion. It is not formal model input and does not change the CCDA data definition.

Previous unqualified transparent visual outputs were removed from:

* `reports/ccda_hose_stage1/transparent_debug_close`
* `reports/ccda_hose_stage1/transparent_socket_v2`

## Runtime

* camera: `{camera}`
* transparent_socket: `True`
* show_occluder: `{show_occluder}`

## Calibrated Config Summary

```json
{json.dumps(cfg_summary, indent=2)}
```

## Generated Files

* `free_insert_{camera}_transparent_calibrated.mp4`
* `right_hidden_jam_{camera}_transparent_calibrated.mp4`
* `plug_trajectory_xy.png`
* `plug_trajectory_xz.png`
* `hose_front_keypoints_xy.png`
* `hose_front_keypoints_xz.png`
* `plug_trajectory_xyz.csv`
* `transparent_calibrated_visual_report.md`

## Diagnostics

| Condition | frame_diff | initial depth | final depth | initial plug | final plug | final branch | final success | max jam force | max lateral force | final lateral offset | final curvature |
| --- | ---: | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(diag_md)}

If a video looks static, inspect `plug_trajectory_xyz.csv` and the frame_diff values above first.
"""
    (out_dir / "transparent_calibrated_visual_report.md").write_text(md, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--camera",
        type=str,
        default="debug_close",
        choices=["front", "top", "side", "side_top", "debug_close"],
    )
    parser.add_argument("--config-json", type=str, default=DEFAULT_CONFIG_JSON)
    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument("--show-occluder", action="store_true")
    parser.add_argument("--marker-stride", type=int, default=12)
    args = parser.parse_args()

    config_json = Path(args.config_json)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = _load_config(config_json)
    rollouts = []
    for offset, condition in enumerate([FREE_INSERT, RIGHT_HIDDEN_JAM]):
        rollouts.append(
            transparent_scripted_rollout(
                condition,
                seed=args.seed + offset,
                config=cfg,
                record_frames=True,
                camera_name=args.camera,
                show_occluder=args.show_occluder,
            )
        )

    diagnostics = []
    for rollout in rollouts:
        video_path = out_dir / f"{rollout['condition']}_{args.camera}_transparent_calibrated.mp4"
        _save_video(rollout["frames"], video_path)
        diff = _frame_diff(rollout["frames"])
        row = _diagnostics(rollout, diff)
        diagnostics.append(row)
        print(
            f"Saved {video_path}; frame_diff={diff:.3f}; "
            f"initial_depth={row['initial_insertion_depth']:.6f}; "
            f"final_depth={row['final_insertion_depth']:.6f}; "
            f"initial_plug_pos={row['initial_plug_pos']}; "
            f"final_plug_pos={row['final_plug_pos']}; "
            f"final_branch={row['final_branch']}; final_success={row['final_success']}; "
            f"max_jam_contact_force={row['max_jam_contact_force']:.6f}; "
            f"max_lateral_contact_force={row['max_lateral_contact_force']:.6f}"
        )
        if diff < 1.0:
            print(
                "WARNING: rendered frames have very small pixel difference; video may look static. "
                "Check gripper/plug trajectory and camera view."
            )

    _plot_plug_trajectories(rollouts, out_dir, marker_stride=args.marker_stride)
    _plot_hose_front_keypoints(rollouts, out_dir, marker_stride=args.marker_stride)
    _write_plug_csv(rollouts, out_dir / "plug_trajectory_xyz.csv")
    _write_report(out_dir, args.camera, config_json, cfg, args.show_occluder, diagnostics)
    print(f"Saved calibrated transparent visual report to {out_dir / 'transparent_calibrated_visual_report.md'}")


if __name__ == "__main__":
    main()
