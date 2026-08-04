"""Utilities for composing real CCDA simulation videos."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np


def letterbox(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    value = np.asarray(frame, dtype=np.uint8)
    if value.ndim != 3 or value.shape[2] != 3:
        raise ValueError(f"expected BGR frame [H,W,3], got {value.shape}")
    if width <= 0 or height <= 0:
        raise ValueError("target width and height must be positive")
    scale = min(float(width) / value.shape[1], float(height) / value.shape[0])
    new_width = max(1, int(round(value.shape[1] * scale)))
    new_height = max(1, int(round(value.shape[0] * scale)))
    resized = cv2.resize(value, (new_width, new_height), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x0 = (width - new_width) // 2
    y0 = (height - new_height) // 2
    canvas[y0:y0 + new_height, x0:x0 + new_width] = resized
    return canvas


def compose_frame(
    free_frame: np.ndarray,
    hidden_frame: np.ndarray,
    panel_width: int,
    panel_height: int,
    group_id: str,
    action_hash: str,
) -> np.ndarray:
    left = letterbox(free_frame, panel_width, panel_height)
    right = letterbox(hidden_frame, panel_width, panel_height)
    header_height = 42
    output = np.zeros((panel_height + header_height, panel_width * 2, 3), dtype=np.uint8)
    output[header_height:, :panel_width] = left
    output[header_height:, panel_width:] = right
    cv2.line(output, (panel_width, header_height), (panel_width, output.shape[0] - 1), (220, 220, 220), 2)
    cv2.putText(output, f"{group_id} | identical action hash: {action_hash[:16]}", (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.63, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(output, "free", (panel_width - 58, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(output, "hidden_high_friction", (panel_width * 2 - 208, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def compose_side_by_side_video(
    free_path: Path,
    hidden_path: Path,
    output_path: Path,
    group_id: str,
    action_hash: str,
    fps: Optional[float] = None,
) -> Dict[str, object]:
    free_path = Path(free_path).resolve()
    hidden_path = Path(hidden_path).resolve()
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    free_cap = cv2.VideoCapture(str(free_path))
    hidden_cap = cv2.VideoCapture(str(hidden_path))
    if not free_cap.isOpened() or not hidden_cap.isOpened():
        free_cap.release()
        hidden_cap.release()
        raise RuntimeError("could not open one or both condition videos")

    panel_width = max(
        int(free_cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(hidden_cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    )
    panel_height = max(
        int(free_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        int(hidden_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    free_fps = float(free_cap.get(cv2.CAP_PROP_FPS) or 20.0)
    hidden_fps = float(hidden_cap.get(cv2.CAP_PROP_FPS) or 20.0)
    output_fps = float(fps if fps is not None else min(free_fps, hidden_fps))

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        (panel_width * 2, panel_height + 42),
    )
    if not writer.isOpened():
        free_cap.release()
        hidden_cap.release()
        raise RuntimeError("could not open side-by-side MP4 writer")

    last_free = None
    last_hidden = None
    frames_written = 0
    try:
        while True:
            ok_free, free_frame = free_cap.read()
            ok_hidden, hidden_frame = hidden_cap.read()
            if ok_free:
                last_free = free_frame
            if ok_hidden:
                last_hidden = hidden_frame
            if last_free is None or last_hidden is None:
                if not ok_free and not ok_hidden:
                    break
                continue
            if not ok_free and not ok_hidden:
                break
            writer.write(
                compose_frame(
                    last_free,
                    last_hidden,
                    panel_width,
                    panel_height,
                    group_id,
                    action_hash,
                )
            )
            frames_written += 1
    finally:
        free_cap.release()
        hidden_cap.release()
        writer.release()

    if frames_written <= 0:
        raise RuntimeError("side-by-side composition produced no frames")

    cap = cv2.VideoCapture(str(output_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    selected = [0, max(0, total // 2), max(0, total - 1)]
    contact_frames = []
    for index in selected:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if ok:
            cv2.putText(frame, f"frame {index}/{max(0, total - 1)}", (12, frame.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            contact_frames.append(frame)
    cap.release()

    contact_sheet_path = output_path.with_name(output_path.stem + "_contact_sheet.png")
    if contact_frames:
        target_height = min(frame.shape[0] for frame in contact_frames)
        resized = [
            cv2.resize(
                frame,
                (int(round(frame.shape[1] * target_height / frame.shape[0])), target_height),
                interpolation=cv2.INTER_AREA,
            )
            for frame in contact_frames
        ]
        cv2.imwrite(str(contact_sheet_path), np.concatenate(resized, axis=1))

    return {
        "free_path": str(free_path),
        "hidden_path": str(hidden_path),
        "output_path": str(output_path),
        "contact_sheet_path": str(contact_sheet_path),
        "fps": output_fps,
        "frames_written": frames_written,
        "panel_width": panel_width,
        "panel_height": panel_height,
    }
