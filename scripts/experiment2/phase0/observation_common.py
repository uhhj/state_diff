"""Real-image observation and grouped classifier helpers for Phase 0E."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Sequence

import cv2
import numpy as np
import pybullet as p


def render_fixed_rgb(config: Dict[str, Any]) -> np.ndarray:
    width = int(config["width"])
    height = int(config["height"])
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")

    eye = [float(v) for v in config["camera_eye"]]
    target = [float(v) for v in config["camera_target"]]
    up = [float(v) for v in config.get("camera_up", [0, 0, 1])]
    view = p.computeViewMatrix(eye, target, up)
    projection = p.computeProjectionMatrixFOV(
        fov=float(config.get("fov", 50.0)),
        aspect=float(width) / float(height),
        nearVal=float(config.get("near", 0.01)),
        farVal=float(config.get("far", 3.0)),
    )
    image = p.getCameraImage(
        width=width,
        height=height,
        viewMatrix=view,
        projectionMatrix=projection,
        shadow=1,
        renderer=p.ER_TINY_RENDERER,
    )
    rgba = np.asarray(image[2], dtype=np.uint8).reshape(height, width, 4)
    return rgba[:, :, :3].copy()


def save_rgb(path: Path, rgb: np.ndarray) -> Dict[str, Any]:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    value = np.asarray(rgb, dtype=np.uint8)
    if value.ndim != 3 or value.shape[2] != 3:
        raise ValueError(f"expected RGB [H,W,3], got {value.shape}")
    ok = cv2.imwrite(str(path), cv2.cvtColor(value, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError(f"failed to write {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "path": str(path),
        "sha256": digest,
        "shape": list(value.shape),
    }


def load_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"failed to read RGB image {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def image_feature(
    rgb: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    value = np.asarray(rgb, dtype=np.uint8)
    resized = cv2.resize(
        value,
        (int(width), int(height)),
        interpolation=cv2.INTER_AREA,
    )
    return resized.astype(np.float64).reshape(-1) / 255.0


def delta_image_feature(
    before_rgb: np.ndarray,
    after_rgb: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    before = cv2.resize(
        np.asarray(before_rgb, dtype=np.uint8),
        (int(width), int(height)),
        interpolation=cv2.INTER_AREA,
    ).astype(np.float64)
    after = cv2.resize(
        np.asarray(after_rgb, dtype=np.uint8),
        (int(width), int(height)),
        interpolation=cv2.INTER_AREA,
    ).astype(np.float64)
    return ((after - before) / 255.0).reshape(-1)


def grouped_ridge_accuracy(
    samples: Sequence[Dict[str, Any]],
    feature_key: str,
    l2: float,
) -> Dict[str, Any]:
    if l2 <= 0:
        raise ValueError("ridge l2 must be positive")
    groups = sorted({str(sample["group_id"]) for sample in samples})
    if len(groups) < 3:
        raise ValueError("at least three groups are required")

    predictions = []
    for held_out in groups:
        train = [s for s in samples if str(s["group_id"]) != held_out]
        test = [s for s in samples if str(s["group_id"]) == held_out]
        x_train = np.stack(
            [np.asarray(s[feature_key], dtype=np.float64) for s in train]
        )
        y_train = np.asarray([int(s["label"]) for s in train], dtype=np.float64)
        x_test = np.stack(
            [np.asarray(s[feature_key], dtype=np.float64) for s in test]
        )

        mean = np.mean(x_train, axis=0)
        std = np.std(x_train, axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        z_train = (x_train - mean) / std
        z_test = (x_test - mean) / std

        gram = z_train @ z_train.T
        alpha = np.linalg.solve(
            gram + float(l2) * np.eye(gram.shape[0]),
            y_train,
        )
        weight = z_train.T @ alpha
        scores = z_test @ weight
        predicted = np.where(scores >= 0.0, 1, -1)

        for sample, score, label in zip(test, scores, predicted):
            predictions.append(
                {
                    "group_id": held_out,
                    "condition": sample["condition"],
                    "truth": int(sample["label"]),
                    "prediction": int(label),
                    "score": float(score),
                }
            )

    accuracy = float(
        np.mean(
            [row["truth"] == row["prediction"] for row in predictions]
        )
    )
    return {
        "accuracy": accuracy,
        "groups": groups,
        "predictions": predictions,
        "feature_key": feature_key,
        "l2": float(l2),
    }
