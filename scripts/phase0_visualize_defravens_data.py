#!/usr/bin/env python3
import argparse
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASK = os.environ.get("TASK", "cable-line-notarget")
DEFAULT_DEFRAVENS_ROOT = Path(
    os.environ.get("DEFRAVENS_ROOT", ROOT / "external" / "deformable-ravens")
).resolve()


def warn(message: str) -> None:
    print(f"[Phase0][visualize][warning] {message}")


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def choose_camera(arr: np.ndarray, camera_index: int, field: str) -> np.ndarray:
    if arr.ndim >= 1 and arr.shape[0] <= 8:
        idx = min(max(camera_index, 0), arr.shape[0] - 1)
        return arr[idx]
    warn(f"{field}: cannot identify camera axis for shape {arr.shape}; using input as-is")
    return arr


def as_uint8_rgb(frame: np.ndarray) -> Optional[np.ndarray]:
    arr = np.asarray(frame)
    if arr.size == 0:
        return None
    arr = np.squeeze(arr)

    if arr.ndim == 2:
        arr = normalize_depth_frame(arr)
        return np.repeat(arr[..., None], 3, axis=-1)

    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        arr = arr[..., :3]
    elif arr.ndim == 3 and arr.shape[0] in (3, 4):
        arr = np.moveaxis(arr[:3], 0, -1)
    else:
        return None

    if arr.dtype == np.uint8:
        return np.ascontiguousarray(arr)

    arr = arr.astype(np.float32)
    if np.nanmax(arr) <= 1.0:
        arr = arr * 255.0
    arr = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def normalize_depth_frame(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame, dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros(arr.shape[-2:], dtype=np.uint8)
    valid = arr[finite]
    lo, hi = np.percentile(valid, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(valid.min()), float(valid.max())
    if hi <= lo:
        return np.zeros(arr.shape[-2:], dtype=np.uint8)
    out = (arr - lo) / (hi - lo)
    out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def split_array_frames(obj: np.ndarray, field: str, camera_index: int) -> Optional[List[np.ndarray]]:
    arr = np.asarray(obj)
    if arr.size == 0:
        warn(f"{field}: empty ndarray")
        return None

    if field == "color":
        if arr.ndim == 5 and arr.shape[-1] in (3, 4):
            return [as_uint8_rgb(frame) for frame in arr[:, min(camera_index, arr.shape[1] - 1)]]
        if arr.ndim == 4 and arr.shape[-1] in (3, 4):
            if arr.shape[0] <= 8 and arr.shape[1] > 32 and arr.shape[2] > 32:
                return [as_uint8_rgb(choose_camera(arr, camera_index, field))]
            return [as_uint8_rgb(frame) for frame in arr]
        if arr.ndim == 3:
            return [as_uint8_rgb(arr)]

    if field == "depth":
        if arr.ndim == 4:
            if arr.shape[-1] == 1:
                return [as_uint8_rgb(frame[..., 0]) for frame in arr]
            return [as_uint8_rgb(frame) for frame in arr[:, min(camera_index, arr.shape[1] - 1)]]
        if arr.ndim == 3:
            if arr.shape[0] <= 8 and arr.shape[1] > 32 and arr.shape[2] > 32:
                return [as_uint8_rgb(choose_camera(arr, camera_index, field))]
            return [as_uint8_rgb(frame) for frame in arr]
        if arr.ndim == 2:
            return [as_uint8_rgb(arr)]

    warn(f"{field}: unsupported ndarray shape {arr.shape}")
    return None


def flatten_frames(obj: Any, field: str, camera_index: int) -> List[np.ndarray]:
    if isinstance(obj, np.ndarray):
        frames = split_array_frames(obj, field, camera_index)
    elif isinstance(obj, (list, tuple)):
        frames = []
        for i, item in enumerate(obj):
            if isinstance(item, np.ndarray):
                sub = split_array_frames(item, field, camera_index)
                if sub:
                    frames.extend(sub)
            elif isinstance(item, (list, tuple)):
                sub = flatten_frames(item, field, camera_index)
                if sub:
                    frames.extend(sub)
            else:
                warn(f"{field}: skipping list item {i} of type {type(item).__name__}")
    else:
        warn(f"{field}: unsupported object type {type(obj).__name__}")
        return []

    clean = [f for f in (frames or []) if f is not None and f.ndim == 3 and f.shape[-1] == 3]
    if not clean:
        warn(f"{field}: no renderable frames decoded")
    return clean


def load_frames(path: Path, field: str, camera_index: int) -> List[np.ndarray]:
    try:
        obj = load_pickle(path)
    except Exception as exc:
        warn(f"{field}: failed to read {path}: {exc!r}")
        return []
    try:
        return flatten_frames(obj, field, camera_index)
    except Exception as exc:
        warn(f"{field}: failed to decode {path}: {exc!r}")
        return []


def write_video(frames: Sequence[np.ndarray], path: Path, fps: int) -> bool:
    if not frames:
        warn(f"cannot write empty video: {path}")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import cv2

        h, w = frames[0].shape[:2]
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if writer.isOpened():
            for frame in frames:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.release()
            return True
        warn(f"cv2 could not open writer: {path}")
    except Exception as cv2_exc:
        warn(f"cv2 failed for {path}: {cv2_exc!r}")

    try:
        import imageio.v2 as imageio

        imageio.mimsave(path, list(frames), fps=fps, macro_block_size=1)
        return True
    except Exception as imageio_exc:
        warn(f"imageio failed for {path}: {imageio_exc!r}")
    return False


def resize_frame(frame: np.ndarray, width: int) -> np.ndarray:
    try:
        from PIL import Image

        img = Image.fromarray(frame)
        h = max(1, int(frame.shape[0] * (width / frame.shape[1])))
        return np.asarray(img.resize((width, h)))
    except Exception:
        return frame


def write_contact_sheet(samples: Sequence[Tuple[str, Sequence[np.ndarray]]], path: Path) -> bool:
    if not samples:
        warn("no samples available for contact sheet")
        return False
    try:
        from PIL import Image, ImageDraw
    except Exception as exc:
        warn(f"Pillow unavailable, cannot write contact sheet: {exc!r}")
        return False

    tile_w = 240
    label_h = 28
    gap = 8
    rows = []
    labels = ("start", "middle", "end")

    for episode_id, frames in samples:
        if not frames:
            continue
        idxs = [0, len(frames) // 2, len(frames) - 1]
        tiles = []
        for label, idx in zip(labels, idxs):
            frame = resize_frame(frames[idx], tile_w)
            img = Image.fromarray(frame)
            canvas = Image.new("RGB", (img.width, img.height + label_h), "white")
            canvas.paste(img, (0, label_h))
            draw = ImageDraw.Draw(canvas)
            draw.text((6, 6), f"demo_{episode_id} {label}", fill=(0, 0, 0))
            tiles.append(canvas)
        row_h = max(t.height for t in tiles)
        row = Image.new("RGB", (len(tiles) * tile_w + (len(tiles) - 1) * gap, row_h), "white")
        x = 0
        for tile in tiles:
            row.paste(tile, (x, 0))
            x += tile_w + gap
        rows.append(row)

    if not rows:
        warn("no rows available for contact sheet")
        return False

    sheet_w = max(r.width for r in rows)
    sheet_h = sum(r.height for r in rows) + (len(rows) - 1) * gap
    sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
    y = 0
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.height + gap
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    return True


def replace_visualization_section(report_path: Path, lines: Sequence[str]) -> None:
    marker = "## Visualization Outputs"
    section = [marker, "", *lines, ""]
    if report_path.exists():
        text = report_path.read_text()
        head = text.split(marker, 1)[0].rstrip()
        report_path.write_text(head + "\n\n" + "\n".join(section))
    else:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("# Phase 0 DeformableRavens Setup Report\n\n" + "\n".join(section))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Phase0 DeformableRavens demo videos and contact sheet.")
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--num-episodes", type=int, default=int(os.environ.get("NUM_EPISODES", "3")))
    parser.add_argument("--fps", type=int, default=int(os.environ.get("FPS", "6")))
    parser.add_argument("--camera-index", type=int, default=int(os.environ.get("CAMERA_INDEX", "0")))
    parser.add_argument("--defravens-root", type=Path, default=DEFAULT_DEFRAVENS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.defravens_root / "data" / args.task
    out_dir = args.output_dir or (ROOT / "reports" / "phase0_visualizations" / args.task)
    report_path = ROOT / "reports" / "phase0_defravens_setup.md"

    color_dir = data_dir / "color"
    depth_dir = data_dir / "depth"
    if not color_dir.exists():
        warn(f"missing color directory: {color_dir}")
        return
    if not depth_dir.exists():
        warn(f"missing depth directory: {depth_dir}")

    color_files = sorted(color_dir.glob("*.pkl"))[: max(args.num_episodes, 0)]
    if not color_files:
        warn(f"no color pickle files found in {color_dir}")
        return

    outputs: Dict[str, List[Path]] = {"color": [], "depth": [], "overview": []}
    overview_samples: List[Tuple[str, Sequence[np.ndarray]]] = []

    for color_path in color_files:
        episode_id = color_path.stem.split("-")[0]
        depth_path = depth_dir / color_path.name

        color_frames = load_frames(color_path, "color", args.camera_index)
        if color_frames:
            color_out = out_dir / f"demo_{episode_id}_color.mp4"
            if write_video(color_frames, color_out, args.fps):
                outputs["color"].append(color_out)
                overview_samples.append((episode_id, color_frames))

        if not depth_path.exists():
            warn(f"depth file missing for demo_{episode_id}: {depth_path}")
            continue
        depth_frames = load_frames(depth_path, "depth", args.camera_index)
        if depth_frames:
            depth_out = out_dir / f"demo_{episode_id}_depth.mp4"
            if write_video(depth_frames, depth_out, args.fps):
                outputs["depth"].append(depth_out)

    overview_path = out_dir / "overview_contact_sheet.png"
    if write_contact_sheet(overview_samples, overview_path):
        outputs["overview"].append(overview_path)

    report_lines = [
        f"- Output directory: `{out_dir}`",
        f"- Episodes requested: `{args.num_episodes}`",
        f"- Camera index: `{args.camera_index}`",
        f"- Color videos: `{len(outputs['color'])}`",
        f"- Depth videos: `{len(outputs['depth'])}`",
        f"- Overview PNG: `{overview_path}`",
        "",
        "### Files",
    ]
    for key in ("color", "depth", "overview"):
        for path in outputs[key]:
            report_lines.append(f"- `{path}`")

    replace_visualization_section(report_path, report_lines)

    print("[Phase0] visualization outputs:")
    for key in ("color", "depth", "overview"):
        for path in outputs[key]:
            print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
