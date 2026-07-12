"""Phase3.14a immutable-cache and deterministic-baseline contract."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .schema_v2 import (
    ACTION_DIM,
    DEFAULT_TF,
    DEFAULT_TH,
    FORMAL_CONDITIONS,
    PAPER_X_DIM,
    STATE_ACTION_X_DIM,
    STATE_DIM,
)


CACHE_VERSION = "phase3_14a_training_cache_v1"
CACHE_ROWS = 4256
N_BEADS = 24
BEAD_XY_DIM = N_BEADS * 2
ROBOT_PROXY_DIM = STATE_DIM - BEAD_XY_DIM
ROBOT_QUATERNION_SLICE = slice(83, 87)

FORMAL_REQUIRED_KEYS = (
    "paper_x",
    "state_action_x",
    "y_state",
    "y_final_state",
    "y_action",
    "condition_name",
    "visible_seed",
    "split_name",
    "source_file",
    "pair_group",
    "window_t",
    "success",
    "final_fraction",
    "engagement_step",
    "release_step",
    "pre_engagement",
)


def _jsonable(value: Any, path: str = "root") -> Any:
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value), path)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        if value.dtype.kind in "fc" and not np.all(np.isfinite(value)):
            raise ValueError(f"{path} contains NaN or Inf")
        return _jsonable(value.tolist(), path)
    if isinstance(value, np.generic):
        return _jsonable(value.item(), path)
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item, f"{path}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _jsonable(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError(f"{path} is non-finite")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"{path} has unsupported type {type(value)!r}")


def strict_json_dump(path: Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        _jsonable(dict(payload)),
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)


def strict_json_load(path: Path) -> Dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object")
    _jsonable(value)
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(json.dumps(list(array.shape)).encode("utf-8"))
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def load_npz_no_pickle(path: Path) -> Dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    object_keys = [
        key for key, value in arrays.items() if value.dtype.kind == "O"
    ]
    if object_keys:
        raise ValueError(f"object arrays are forbidden: {object_keys}")
    return arrays


def validate_formal_windows(
    arrays: Mapping[str, np.ndarray],
    *,
    expected_rows: int = CACHE_ROWS,
) -> Dict[str, Any]:
    missing = sorted(set(FORMAL_REQUIRED_KEYS).difference(arrays))
    if missing:
        raise ValueError(f"missing formal arrays: {missing}")

    rows = int(np.asarray(arrays["paper_x"]).shape[0])
    if rows != int(expected_rows):
        raise ValueError(f"row count {rows} != {expected_rows}")

    required_shapes = {
        "paper_x": (rows, PAPER_X_DIM),
        "state_action_x": (rows, STATE_ACTION_X_DIM),
        "y_state": (rows, DEFAULT_TF, STATE_DIM),
        "y_final_state": (rows, STATE_DIM),
        "y_action": (rows, ACTION_DIM),
    }
    for key, expected in required_shapes.items():
        value = np.asarray(arrays[key])
        if value.shape != expected:
            raise ValueError(f"{key} shape {value.shape} != {expected}")
        if value.dtype != np.float32:
            raise ValueError(f"{key} must be float32")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{key} contains NaN or Inf")

    for key in FORMAL_REQUIRED_KEYS[5:]:
        if np.asarray(arrays[key]).shape != (rows,):
            raise ValueError(f"{key} must have shape ({rows},)")

    conditions = set(
        np.asarray(arrays["condition_name"]).astype(str).tolist()
    )
    if conditions != set(FORMAL_CONDITIONS):
        raise ValueError(f"condition set mismatch: {conditions}")

    splits = set(np.asarray(arrays["split_name"]).astype(str).tolist())
    if splits != {"train", "val", "test"}:
        raise ValueError(f"split set mismatch: {splits}")

    prefix_error = float(
        np.max(
            np.abs(
                np.asarray(arrays["state_action_x"])[:, :PAPER_X_DIM]
                - np.asarray(arrays["paper_x"])
            )
        )
    )
    if prefix_error != 0.0:
        raise ValueError(
            f"state_action_x/paper_x prefix error is {prefix_error}"
        )

    return {
        "rows": rows,
        "conditions": sorted(conditions),
        "splits": sorted(splits),
        "paper_prefix_max_abs": prefix_error,
    }


def state_history_valid_mask(
    current_index: int,
    th: int = DEFAULT_TH,
) -> np.ndarray:
    current = int(current_index)
    if current < 0:
        raise ValueError("current_index must be non-negative")
    valid = min(int(th), current + 1)
    mask = np.zeros(int(th), dtype=np.bool_)
    mask[-valid:] = True
    return mask


def action_history_valid_mask(
    current_index: int,
    th: int = DEFAULT_TH,
) -> np.ndarray:
    current = int(current_index)
    if current < 0:
        raise ValueError("current_index must be non-negative")
    valid = min(int(th), current)
    mask = np.zeros(int(th), dtype=np.bool_)
    if valid:
        mask[-valid:] = True
    return mask


def future_valid_mask(
    action_count: int,
    current_index: int,
    tf: int = DEFAULT_TF,
) -> np.ndarray:
    actions = int(action_count)
    current = int(current_index)
    if actions <= 0:
        raise ValueError("action_count must be positive")
    if current < 0 or current >= actions:
        raise ValueError(
            f"current_index {current} outside action count {actions}"
        )
    available = actions - current
    valid = min(int(tf), available)
    mask = np.zeros(int(tf), dtype=np.bool_)
    mask[:valid] = True
    return mask


def current_state_from_paper_x(paper_x: np.ndarray) -> np.ndarray:
    value = np.asarray(paper_x, dtype=np.float32)
    if value.shape[-1] != PAPER_X_DIM:
        raise ValueError("paper_x dimension mismatch")
    return value[..., -STATE_DIM:].copy()


def previous_state_from_paper_x(
    paper_x: np.ndarray,
    state_history_mask: np.ndarray,
) -> np.ndarray:
    value = np.asarray(paper_x, dtype=np.float32)
    mask = np.asarray(state_history_mask, dtype=np.bool_)
    if value.shape[-1] != PAPER_X_DIM:
        raise ValueError("paper_x dimension mismatch")
    if mask.shape[-1] != DEFAULT_TH:
        raise ValueError("state history mask dimension mismatch")
    history = value.reshape(value.shape[:-1] + (DEFAULT_TH, STATE_DIM))
    if history.ndim == 2:
        return history[-2].copy() if mask[-2] else history[-1].copy()

    output = history[..., -1, :].copy()
    has_previous = mask[..., -2]
    output[has_previous] = history[..., -2, :][has_previous]
    return output


def canonical_source_path(
    formal_root: Path,
    *,
    split: str,
    condition: str,
    recorded_source: str,
) -> Tuple[Path, bool]:
    """Resolve promoted data even when source_file stored a staging absolute path."""
    formal = Path(formal_root).resolve()
    recorded = Path(str(recorded_source))
    if recorded.is_file():
        return recorded.resolve(), False

    candidate = (
        formal
        / "raw"
        / str(split)
        / str(condition)
        / recorded.name
    ).resolve()
    raw_root = (formal / "raw").resolve()
    if raw_root not in candidate.parents:
        raise ValueError("canonical source path escapes raw root")
    if not candidate.is_file():
        raise FileNotFoundError(
            f"recorded and canonical source paths are missing: "
            f"{recorded_source!r}, {candidate}"
        )
    return candidate, True


def load_raw_action_count(path: Path) -> int:
    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)
    actions = list(payload.get("actions", []))
    infos = list(payload.get("infos", []))
    if not actions or len(infos) != len(actions):
        raise ValueError(
            f"raw episode action/info mismatch in {path}: "
            f"{len(actions)} actions, {len(infos)} infos"
        )
    if payload.get("last_info") is None:
        raise ValueError(f"raw episode has no last_info: {path}")
    return len(actions)


def verify_source_sidecar(
    source_path: Path,
    *,
    split: str,
    condition: str,
    visible_seed: int,
    pair_group: str,
) -> Dict[str, Any]:
    sidecar = source_path.with_suffix(".manifest.json")
    if not sidecar.is_file():
        raise FileNotFoundError(sidecar)
    manifest = strict_json_load(sidecar)
    expected = {
        "split": str(split),
        "condition": str(condition),
        "visible_seed": int(visible_seed),
        "pair_group": str(pair_group),
        "contains_simulator_bead_velocity": False,
        "input_canonicalization": False,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(
                f"sidecar {sidecar} field {key}: "
                f"{manifest.get(key)!r} != {value!r}"
            )
    return manifest


@dataclass(frozen=True)
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray
    active: np.ndarray
    raw_std: np.ndarray

    def transform(self, value: np.ndarray) -> np.ndarray:
        result = (
            np.asarray(value, dtype=np.float32)
            - np.asarray(self.mean, dtype=np.float32)
        ) / np.asarray(self.scale, dtype=np.float32)
        if not np.all(np.isfinite(result)):
            raise ValueError("standardized values are non-finite")
        return result.astype(np.float32)

    def inverse(self, value: np.ndarray) -> np.ndarray:
        result = (
            np.asarray(value, dtype=np.float32)
            * np.asarray(self.scale, dtype=np.float32)
            + np.asarray(self.mean, dtype=np.float32)
        )
        if not np.all(np.isfinite(result)):
            raise ValueError("inverse-standardized values are non-finite")
        return result.astype(np.float32)

    def to_npz(self, prefix: str) -> Dict[str, np.ndarray]:
        return {
            f"{prefix}_mean": np.asarray(self.mean, dtype=np.float32),
            f"{prefix}_scale": np.asarray(self.scale, dtype=np.float32),
            f"{prefix}_active": np.asarray(self.active, dtype=np.bool_),
            f"{prefix}_raw_std": np.asarray(
                self.raw_std,
                dtype=np.float32,
            ),
        }


def fit_standardizer(
    value: np.ndarray,
    *,
    variance_floor: float = 1e-8,
) -> Standardizer:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or not np.all(np.isfinite(array)):
        raise ValueError("standardizer input must be finite [N,D]")
    mean = np.mean(array, axis=0, dtype=np.float64).astype(np.float32)
    raw_std = np.std(array, axis=0, dtype=np.float64).astype(np.float32)
    active = raw_std >= float(variance_floor)
    scale = raw_std.copy()
    scale[~active] = 1.0
    return Standardizer(
        mean=mean,
        scale=scale.astype(np.float32),
        active=active.astype(np.bool_),
        raw_std=raw_std,
    )


def fit_masked_future_standardizer(
    future: np.ndarray,
    valid_mask: np.ndarray,
    *,
    variance_floor: float = 1e-8,
) -> Standardizer:
    values = np.asarray(future, dtype=np.float32)
    mask = np.asarray(valid_mask, dtype=np.bool_)
    if values.ndim != 3 or values.shape[1:] != (DEFAULT_TF, STATE_DIM):
        raise ValueError("future must be [N,4,87]")
    if mask.shape != values.shape[:2]:
        raise ValueError("future mask must be [N,4]")

    mean = np.zeros((DEFAULT_TF, STATE_DIM), dtype=np.float32)
    raw_std = np.zeros_like(mean)
    active = np.zeros_like(mean, dtype=np.bool_)
    scale = np.ones_like(mean)
    for horizon in range(DEFAULT_TF):
        selected = values[mask[:, horizon], horizon, :]
        if selected.shape[0] < 2:
            raise ValueError(
                f"future horizon {horizon} has fewer than two train rows"
            )
        mean[horizon] = np.mean(
            selected,
            axis=0,
            dtype=np.float64,
        ).astype(np.float32)
        raw_std[horizon] = np.std(
            selected,
            axis=0,
            dtype=np.float64,
        ).astype(np.float32)
        active[horizon] = raw_std[horizon] >= float(variance_floor)
        scale[horizon, active[horizon]] = raw_std[
            horizon,
            active[horizon],
        ]
    return Standardizer(
        mean=mean,
        scale=scale,
        active=active,
        raw_std=raw_std,
    )


def validate_state_quaternions(
    states: np.ndarray,
    *,
    tolerance: float = 1e-3,
) -> Dict[str, float]:
    value = np.asarray(states, dtype=np.float32)
    if value.shape[-1] != STATE_DIM:
        raise ValueError("state dimension mismatch")
    quat = value[..., ROBOT_QUATERNION_SLICE]
    norms = np.linalg.norm(quat, axis=-1)
    if not np.all(np.isfinite(norms)):
        raise ValueError("non-finite quaternion norm")
    error = float(np.max(np.abs(norms - 1.0)))
    if error > float(tolerance):
        raise ValueError(
            f"quaternion max norm error {error} > {tolerance}"
        )
    return {
        "min_norm": float(np.min(norms)),
        "max_norm": float(np.max(norms)),
        "max_abs_norm_error": error,
    }
