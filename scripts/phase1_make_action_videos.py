#!/usr/bin/env python3
"""Make Phase1 action-step rollout videos from saved DeformableRavens RGB observations.

This is a dataset replay video. It does not rerun PyBullet and does not record
continuous internal primitive substeps. Each video frame corresponds to one
saved observation step, with last_color appended as the final frame.
"""

import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np


CONDITIONS = ["free", "hidden_pin", "hidden_high_friction", "hidden_side_jam"]


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def get_extras(info: Dict) -> Dict:
    if isinstance(info, dict):
        return info.get("extras", {})
    return {}


def episode_files(cond_dir: Path) -> List[Path]:
    color_dir = cond_dir / "color"
    if not color_dir.exists():
        return []
    return sorted(color_dir.glob("*.pkl"))


def load_first_info(cond_dir: Path, fname: str) -> Dict:
    info_path = cond_dir / "info" / fname
    info_list = load_pickle(info_path)
    if isinstance(info_list, list) and len(info_list) > 0:
        return info_list[0]
    return {}


def get_seed_for_episode(cond_dir: Path, color_file: Path) -> str:
    info = load_first_info(cond_dir, color_file.name)
    extras = get_extras(info)
    seed = extras.get("ccda_visible_seed", None)
    if seed is None:
        return color_file.stem
    return str(seed)


def group_by_seed(data_root: Path) -> Dict[str, Dict[str, Path]]:
    out: Dict[str, Dict[str, Path]] = {}
    for cond in CONDITIONS:
        cond_dir = data_root / cond
        for color_file in episode_files(cond_dir):
            seed = get_seed_for_episode(cond_dir, color_file)
            out.setdefault(seed, {})[cond] = color_file
    return out


def select_color_frame(color_obj: Any, timestep: int, camera_index: int) -> np.ndarray:
    arr = np.asarray(color_obj)

    if arr.ndim == 5:
        # (T, C, H, W, 3)
        t = min(max(timestep, 0), arr.shape[0] - 1)
        c = min(max(camera_index, 0), arr.shape[1] - 1)
        frame = arr[t, c]
    elif arr.ndim == 4:
        if arr.shape[-1] == 3 and arr.shape[0] <= 8:
            # (C, H, W, 3), e.g. last_color
            c = min(max(camera_index, 0), arr.shape[0] - 1)
            frame = arr[c]
        elif arr.shape[-1] == 3:
            # (T, H, W, 3)
            t = min(max(timestep, 0), arr.shape[0] - 1)
            frame = arr[t]
        else:
            raise ValueError(f"Unsupported color shape: {arr.shape}")
    elif arr.ndim == 3 and arr.shape[-1] == 3:
        frame = arr
    else:
        raise ValueError(f"Unsupported color shape: {arr.shape}")

    frame = np.asarray(frame)
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def load_rgb_sequence(cond_dir: Path, fname: str, camera_index: int) -> List[np.ndarray]:
    color = load_pickle(cond_dir / "color" / fname)
    color_arr = np.asarray(color)

    if color_arr.ndim == 5:
        T = color_arr.shape[0]
    elif color_arr.ndim == 4 and color_arr.shape[-1] == 3 and color_arr.shape[0] > 8:
        T = color_arr.shape[0]
    else:
        T = 1

    frames = [select_color_frame(color, t, camera_index) for t in range(T)]

    last_color_path = cond_dir / "last_color" / fname
    if last_color_path.exists():
        last_color = load_pickle(last_color_path)
        frames.append(select_color_frame(last_color, 0, camera_index))

    return frames


def parse_episode_len(path: Path) -> int:
    try:
        return int(path.stem.split("-")[-1])
    except Exception:
        return -1


def resize_rgb(rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    rgb = np.asarray(rgb)
    return cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)


def make_tile(
    rgb: Optional[np.ndarray],
    title: str,
    step: int,
    total_steps: int,
    tile_w: int,
    tile_h: int,
) -> np.ndarray:
    if rgb is None:
        tile = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
    else:
        tile = resize_rgb(rgb, tile_w, tile_h)

    # Convert RGB tile to BGR for cv2 annotation.
    bgr = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)

    cv2.rectangle(bgr, (0, 0), (tile_w, 50), (0, 0, 0), -1)
    cv2.putText(
        bgr,
        title,
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        bgr,
        f"step {step + 1}/{total_steps}",
        (10, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return bgr


def get_frame_at(seq: List[np.ndarray], idx: int) -> Optional[np.ndarray]:
    if not seq:
        return None
    idx = min(max(idx, 0), len(seq) - 1)
    return seq[idx]


def make_video_for_seed(
    seed: str,
    group: Dict[str, Path],
    data_root: Path,
    camera_index: int,
    outdir: Path,
    fps: float,
    tile_width: int,
    tile_height: int,
    repeat_final: int,
) -> Dict:
    seqs: Dict[str, List[np.ndarray]] = {}

    for cond in CONDITIONS:
        if cond not in group:
            seqs[cond] = []
            continue
        cond_dir = data_root / cond
        seqs[cond] = load_rgb_sequence(cond_dir, group[cond].name, camera_index)

    max_len = max([len(v) for v in seqs.values()] + [0])
    if max_len <= 0:
        return {
            "visible_seed": seed,
            "status": "skipped",
            "reason": "no frames",
        }

    total_frames = max_len + max(0, repeat_final)
    out_path = outdir / f"seed_{seed}_action_steps.mp4"

    frame_w = tile_width * 2
    frame_h = tile_height * 2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (frame_w, frame_h))

    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {out_path}")

    for step in range(total_frames):
        src_step = min(step, max_len - 1)

        tiles = []
        for cond in CONDITIONS:
            rgb = get_frame_at(seqs[cond], src_step)
            tile = make_tile(
                rgb=rgb,
                title=cond,
                step=src_step,
                total_steps=max_len,
                tile_w=tile_width,
                tile_h=tile_height,
            )
            tiles.append(tile)

        top = np.concatenate([tiles[0], tiles[1]], axis=1)
        bottom = np.concatenate([tiles[2], tiles[3]], axis=1)
        frame = np.concatenate([top, bottom], axis=0)

        cv2.putText(
            frame,
            f"Phase1 hidden-contact cable | visible_seed={seed} | camera={camera_index}",
            (10, frame_h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        writer.write(frame)

    writer.release()

    return {
        "visible_seed": seed,
        "status": "written",
        "path": str(out_path),
        "camera_index": camera_index,
        "fps": fps,
        "tile_width": tile_width,
        "tile_height": tile_height,
        "num_video_frames": total_frames,
        "max_observation_steps": max_len,
        "condition_frame_counts": {cond: len(seqs[cond]) for cond in CONDITIONS},
        "condition_files": {cond: group[cond].name if cond in group else None for cond in CONDITIONS},
    }


def write_report(path: Path, summary: Dict) -> None:
    lines = []
    lines.append("# Phase1 Action Execution Video Report")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append("This report summarizes action-step observation videos generated from saved DeformableRavens RGB observations.")
    lines.append("Each MP4 shows four hidden-contact conditions in a 2x2 grid for the same visible seed.")
    lines.append("")
    lines.append("## Important Note")
    lines.append("")
    lines.append("These are dataset replay videos, not continuous PyBullet substep recordings.")
    lines.append("Each frame corresponds to a saved observation before or after an action step.")
    lines.append("")
    lines.append("## Videos")
    lines.append("")
    lines.append("| Seed | Status | Frames | Max Obs Steps | Path |")
    lines.append("|---|---|---:|---:|---|")

    for item in summary["videos"]:
        lines.append(
            "| {} | {} | {} | {} | `{}` |".format(
                item.get("visible_seed"),
                item.get("status"),
                item.get("num_video_frames", "NA"),
                item.get("max_observation_steps", "NA"),
                item.get("path", item.get("reason", "")),
            )
        )

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- Use these videos to inspect whether the same visible seed produces different action-step outcomes under different hidden-contact conditions.")
    lines.append("- For formal CCDA proof, combine this video check with RGB-D difference metrics, bead trajectory divergence, success difference, and later threshold-based pair mining.")
    lines.append("")

    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--task", default="hidden-contact-cable-line")
    parser.add_argument("--camera_index", type=int, default=0)
    parser.add_argument("--max_groups", type=int, default=20)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--tile_width", type=int, default=426)
    parser.add_argument("--tile_height", type=int, default=320)
    parser.add_argument("--repeat_final", type=int, default=4)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    data_root = root / "external" / "deformable-ravens" / "data" / args.task
    outdir = root / "reports" / "phase1_action_videos"
    outdir.mkdir(parents=True, exist_ok=True)

    groups = group_by_seed(data_root)
    if not groups:
        raise SystemExit(f"No groups found under {data_root}")

    videos = []
    for i, (seed, group) in enumerate(sorted(groups.items(), key=lambda kv: str(kv[0]))):
        if i >= args.max_groups:
            break
        item = make_video_for_seed(
            seed=seed,
            group=group,
            data_root=data_root,
            camera_index=args.camera_index,
            outdir=outdir,
            fps=args.fps,
            tile_width=args.tile_width,
            tile_height=args.tile_height,
            repeat_final=args.repeat_final,
        )
        videos.append(item)
        print("[Phase1] video:", item)

    summary = {
        "task": args.task,
        "data_root": str(data_root),
        "output_dir": str(outdir),
        "camera_index": args.camera_index,
        "fps": args.fps,
        "max_groups": args.max_groups,
        "videos": videos,
    }

    json_path = root / "reports" / "phase1_action_video_summary.json"
    md_path = root / "reports" / "phase1_action_video_report.md"

    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    write_report(md_path, summary)

    print(f"[Phase1] wrote {json_path}")
    print(f"[Phase1] wrote {md_path}")
    print(f"[Phase1] wrote videos to {outdir}")

    written = [v for v in videos if v.get("status") == "written"]
    if not written:
        raise SystemExit("[Phase1][FAIL] no videos written")


if __name__ == "__main__":
    main()
