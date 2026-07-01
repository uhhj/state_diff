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

WARNINGS: List[str] = []
SHORT_EPISODE_WARNED = set()


def warn(message: str) -> None:
    WARNINGS.append(message)
    print(f"[Phase0][visualize][warning] {message}")


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def parse_episode_len(path: Path) -> Optional[int]:
    try:
        return int(path.stem.split("-")[-1])
    except Exception:
        return None


def clamp_camera_idx(num_cameras: int, camera_idx: int, context: str) -> int:
    if num_cameras <= 0:
        return 0
    if camera_idx < 0 or camera_idx >= num_cameras:
        idx = min(max(camera_idx, 0), num_cameras - 1)
        warn(f"{context}: camera_idx={camera_idx} out of range for {num_cameras} cameras; using {idx}")
        return idx
    return camera_idx


def color_frame_to_rgb(frame: Any, context: str) -> Optional[np.ndarray]:
    arr = np.asarray(frame)
    if arr.size == 0:
        warn(f"{context}: empty color frame")
        return None
    arr = np.squeeze(arr)
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        arr = arr[..., :3]
    elif arr.ndim == 3 and arr.shape[0] in (3, 4):
        arr = np.moveaxis(arr[:3], 0, -1)
    else:
        warn(f"{context}: unsupported color frame shape {arr.shape}")
        return None

    if arr.dtype == np.uint8:
        return np.ascontiguousarray(arr)
    arr = arr.astype(np.float32)
    finite = np.isfinite(arr)
    if finite.any() and float(np.nanmax(arr)) <= 1.0:
        arr = arr * 255.0
    arr = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)
    return np.ascontiguousarray(np.clip(arr, 0, 255).astype(np.uint8))


def depth_frame_to_float(frame: Any, context: str) -> Optional[np.ndarray]:
    arr = np.asarray(frame)
    if arr.size == 0:
        warn(f"{context}: empty depth frame")
        return None
    arr = np.squeeze(arr)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 2:
        warn(f"{context}: unsupported depth frame shape {arr.shape}")
        return None
    return np.asarray(arr, dtype=np.float32)


def parse_color_timestep(item: Any, camera_idx: int, context: str) -> Optional[np.ndarray]:
    if isinstance(item, (list, tuple)):
        if not item:
            warn(f"{context}: empty color camera list")
            return None
        idx = clamp_camera_idx(len(item), camera_idx, context)
        return parse_color_timestep(item[idx], camera_idx, f"{context}[camera {idx}]")

    arr = np.asarray(item)
    if arr.ndim == 5 and arr.shape[0] == 1:
        return parse_color_timestep(arr[0], camera_idx, f"{context}[0]")
    if arr.ndim == 4 and arr.shape[-1] in (3, 4):
        idx = clamp_camera_idx(arr.shape[0], camera_idx, context)
        return color_frame_to_rgb(arr[idx], f"{context}[camera {idx}]")
    return color_frame_to_rgb(arr, context)


def parse_depth_timestep(item: Any, camera_idx: int, context: str) -> Optional[np.ndarray]:
    if isinstance(item, (list, tuple)):
        if not item:
            warn(f"{context}: empty depth camera list")
            return None
        idx = clamp_camera_idx(len(item), camera_idx, context)
        return parse_depth_timestep(item[idx], camera_idx, f"{context}[camera {idx}]")

    arr = np.asarray(item)
    if arr.ndim == 4 and arr.shape[0] == 1:
        return parse_depth_timestep(arr[0], camera_idx, f"{context}[0]")
    if arr.ndim == 3 and arr.shape[0] <= 16 and arr.shape[1] > 32 and arr.shape[2] > 32:
        idx = clamp_camera_idx(arr.shape[0], camera_idx, context)
        return depth_frame_to_float(arr[idx], f"{context}[camera {idx}]")
    return depth_frame_to_float(arr, context)


def decode_color_episode(obj: Any, camera_idx: int, expected_len: Optional[int], episode_id: str) -> List[np.ndarray]:
    frames: List[np.ndarray] = []

    if isinstance(obj, (list, tuple)):
        for t, item in enumerate(obj):
            frame = parse_color_timestep(item, camera_idx, f"demo_{episode_id} color timestep {t}")
            if frame is not None:
                frames.append(frame)
        return frames

    arr = np.asarray(obj)
    if arr.ndim == 5 and arr.shape[-1] in (3, 4):
        for t in range(arr.shape[0]):
            frame = parse_color_timestep(arr[t], camera_idx, f"demo_{episode_id} color timestep {t}")
            if frame is not None:
                frames.append(frame)
        return frames

    if arr.ndim == 4 and arr.shape[-1] in (3, 4):
        if expected_len is not None and arr.shape[0] == expected_len:
            for t in range(arr.shape[0]):
                frame = color_frame_to_rgb(arr[t], f"demo_{episode_id} color timestep {t}")
                if frame is not None:
                    frames.append(frame)
        else:
            frame = parse_color_timestep(arr, camera_idx, f"demo_{episode_id} color single timestep")
            if frame is not None:
                frames.append(frame)
        return frames

    frame = color_frame_to_rgb(arr, f"demo_{episode_id} color single frame")
    if frame is not None:
        frames.append(frame)
    return frames


def decode_depth_episode(obj: Any, camera_idx: int, expected_len: Optional[int], episode_id: str) -> List[np.ndarray]:
    frames: List[np.ndarray] = []

    if isinstance(obj, (list, tuple)):
        for t, item in enumerate(obj):
            frame = parse_depth_timestep(item, camera_idx, f"demo_{episode_id} depth timestep {t}")
            if frame is not None:
                frames.append(frame)
        return frames

    arr = np.asarray(obj)
    if arr.ndim == 4:
        for t in range(arr.shape[0]):
            frame = parse_depth_timestep(arr[t], camera_idx, f"demo_{episode_id} depth timestep {t}")
            if frame is not None:
                frames.append(frame)
        return frames

    if arr.ndim == 3:
        if expected_len is not None and arr.shape[0] == expected_len:
            for t in range(arr.shape[0]):
                frame = depth_frame_to_float(arr[t], f"demo_{episode_id} depth timestep {t}")
                if frame is not None:
                    frames.append(frame)
        else:
            frame = parse_depth_timestep(arr, camera_idx, f"demo_{episode_id} depth single timestep")
            if frame is not None:
                frames.append(frame)
        return frames

    frame = depth_frame_to_float(arr, f"demo_{episode_id} depth single frame")
    if frame is not None:
        frames.append(frame)
    return frames


def load_episode_frames(path: Path, field: str, camera_idx: int, episode_id: str) -> List[np.ndarray]:
    try:
        obj = load_pickle(path)
    except Exception as exc:
        warn(f"demo_{episode_id} {field}: failed to read {path}: {exc!r}")
        return []

    expected_len = parse_episode_len(path)
    try:
        if field == "color":
            frames = decode_color_episode(obj, camera_idx, expected_len, episode_id)
        else:
            frames = decode_depth_episode(obj, camera_idx, expected_len, episode_id)
    except Exception as exc:
        warn(f"demo_{episode_id} {field}: failed to decode {path}: {exc!r}")
        return []

    if not frames:
        warn(f"demo_{episode_id} {field}: no renderable timesteps decoded")
    return frames


def normalize_depth_episode(frames: Sequence[np.ndarray], episode_id: str) -> List[np.ndarray]:
    if not frames:
        return []
    finite_values = [frame[np.isfinite(frame)] for frame in frames if np.isfinite(frame).any()]
    if not finite_values:
        warn(f"demo_{episode_id} depth: no finite values; rendering black frames")
        return [np.zeros((*frame.shape, 3), dtype=np.uint8) for frame in frames]

    values = np.concatenate([x.reshape(-1) for x in finite_values])
    lo = float(values.min())
    hi = float(values.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        warn(f"demo_{episode_id} depth: invalid global min/max ({lo}, {hi}); rendering black frames")
        return [np.zeros((*frame.shape, 3), dtype=np.uint8) for frame in frames]

    rgb_frames = []
    for frame in frames:
        arr = (frame - lo) / (hi - lo)
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
        gray = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        rgb_frames.append(np.repeat(gray[..., None], 3, axis=-1))
    return rgb_frames


def annotate_frame(
    frame: np.ndarray,
    episode_id: str,
    timestep_idx: int,
    camera_idx: int,
    total_timesteps: int,
) -> np.ndarray:
    try:
        from PIL import Image, ImageDraw

        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        text = f"demo {episode_id} | timestep {timestep_idx} | camera {camera_idx} | total {total_timesteps}"
        try:
            box = draw.textbbox((0, 0), text)
            tw, th = box[2] - box[0], box[3] - box[1]
        except Exception:
            tw, th = draw.textsize(text)
        draw.rectangle((0, 0, tw + 10, th + 10), fill=(0, 0, 0))
        draw.text((5, 5), text, fill=(255, 255, 255))
        return np.asarray(img)
    except Exception:
        return frame


def expand_with_hold(
    frames: Sequence[np.ndarray],
    episode_id: str,
    camera_idx: int,
    fps: int,
    hold_sec: float,
    min_duration_sec: float,
) -> List[np.ndarray]:
    if not frames:
        return []
    total = len(frames)
    if 1 <= total <= 3 and episode_id not in SHORT_EPISODE_WARNED:
        SHORT_EPISODE_WARNED.add(episode_id)
        warn(
            f"demo_{episode_id}: source episode has only {total} high-level timesteps; "
            "video is padded with frame holding, but source data is short."
        )

    repeat = max(1, int(fps * hold_sec))
    expanded: List[np.ndarray] = []
    for t, frame in enumerate(frames):
        annotated = annotate_frame(frame, episode_id, t, camera_idx, total)
        expanded.extend([annotated] * repeat)

    min_frames = max(1, int(np.ceil(fps * min_duration_sec)))
    if len(expanded) < min_frames:
        expanded.extend([expanded[-1]] * (min_frames - len(expanded)))
    return expanded


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


def write_contact_sheet(
    samples: Sequence[Tuple[str, Sequence[np.ndarray], int]],
    path: Path,
    camera_idx: int,
) -> bool:
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

    for episode_id, frames, total in samples:
        if not frames:
            continue
        idxs = [0, len(frames) // 2, len(frames) - 1]
        tiles = []
        for label, idx in zip(labels, idxs):
            frame = annotate_frame(frames[idx], episode_id, idx, camera_idx, total)
            frame = resize_frame(frame, tile_w)
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
    parser.add_argument("--fps", type=int, default=int(os.environ.get("FPS", "12")))
    parser.add_argument("--hold-sec", type=float, default=float(os.environ.get("HOLD_SEC", "0.6")))
    parser.add_argument("--min-duration-sec", type=float, default=float(os.environ.get("MIN_DURATION_SEC", "5.0")))
    parser.add_argument("--camera_idx", "--camera-index", dest="camera_idx", type=int, default=int(os.environ.get("CAMERA_IDX", os.environ.get("CAMERA_INDEX", "0"))))
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
    overview_samples: List[Tuple[str, Sequence[np.ndarray], int]] = []
    episode_lengths: Dict[str, int] = {}

    for color_path in color_files:
        episode_id = color_path.stem.split("-")[0]
        depth_path = depth_dir / color_path.name

        color_steps = load_episode_frames(color_path, "color", args.camera_idx, episode_id)
        if color_steps:
            episode_lengths[episode_id] = len(color_steps)
            color_video_frames = expand_with_hold(
                color_steps,
                episode_id,
                args.camera_idx,
                args.fps,
                args.hold_sec,
                args.min_duration_sec,
            )
            color_out = out_dir / f"demo_{episode_id}_color.mp4"
            if write_video(color_video_frames, color_out, args.fps):
                outputs["color"].append(color_out)
                overview_samples.append((episode_id, color_steps, len(color_steps)))

        if not depth_path.exists():
            warn(f"depth file missing for demo_{episode_id}: {depth_path}")
            continue
        depth_steps = load_episode_frames(depth_path, "depth", args.camera_idx, episode_id)
        if depth_steps:
            total_steps = len(depth_steps)
            episode_lengths.setdefault(episode_id, total_steps)
            depth_rgb_steps = normalize_depth_episode(depth_steps, episode_id)
            depth_video_frames = expand_with_hold(
                depth_rgb_steps,
                episode_id,
                args.camera_idx,
                args.fps,
                args.hold_sec,
                args.min_duration_sec,
            )
            depth_out = out_dir / f"demo_{episode_id}_depth.mp4"
            if write_video(depth_video_frames, depth_out, args.fps):
                outputs["depth"].append(depth_out)

    overview_path = out_dir / "overview_contact_sheet.png"
    if write_contact_sheet(overview_samples, overview_path, args.camera_idx):
        outputs["overview"].append(overview_path)

    short_episodes = {episode_id: n for episode_id, n in episode_lengths.items() if 1 <= n <= 3}
    report_lines = [
        f"- Output directory: `{out_dir}`",
        f"- Episodes requested: `{args.num_episodes}`",
        f"- Camera index: `{args.camera_idx}`",
        f"- FPS: `{args.fps}`",
        f"- Hold seconds per high-level timestep: `{args.hold_sec}`",
        f"- Minimum video duration seconds: `{args.min_duration_sec}`",
        f"- Color videos: `{len(outputs['color'])}`",
        f"- Depth videos: `{len(outputs['depth'])}`",
        f"- Overview PNG: `{overview_path}`",
        "",
        "### Warnings",
    ]
    if short_episodes:
        for episode_id, n in sorted(short_episodes.items()):
            report_lines.append(
                f"- `demo_{episode_id}` has only `{n}` high-level timestep(s); "
                "frame holding/padding keeps video length >= 5 seconds, but the source data is short."
            )
    else:
        report_lines.append("- None for selected episodes.")

    if WARNINGS:
        report_lines.append("")
        report_lines.append("### Runtime Warnings")
        for message in WARNINGS:
            report_lines.append(f"- {message}")

    report_lines.extend(["", "### Files"])
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
