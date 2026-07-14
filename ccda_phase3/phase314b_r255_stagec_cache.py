"""Phase3.14b-r2.5.5 Stage C state-v3 immutable cache.

This module is additive.  It consumes the byte-audited Stage-B state-v3
windows, creates a new cache under a new path/version, and emits a separate
train-only attribution view.  It never mutates the legacy state-v2 cache or
the Stage-B dataset.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import struct
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from ccda_phase3.schema_v3 import (
    ACTION_DIM,
    CABLE_DIM,
    DEFAULT_TF,
    DEFAULT_TH,
    EE_QUATERNION_SLICE,
    PAPER_X_DIM,
    ROBOT_PROXY_DIM,
    SCHEMA_VERSION,
    STATE_ACTION_X_DIM,
    STATE_DIM,
    SchemaV3Manifest,
)

PHASE = "Phase3.14b-r2.5.5 Stage C"
PHASE_ID = "phase314b_r255_stagec"
BASE_EVIDENCE_COMMIT = "83d8aa4bf6895a91c264c8350d95cb5878cebc9b"
EXPECTED_SUBMODULE_COMMIT = "633a88752445cf5d6776ed374fdbbdb35f93050c"
EXPECTED_LEGACY_CACHE_SHA256 = (
    "3cc512f650557c81c3b81f4f128a5b55360b77d3f664fdd6607666936285fbe8"
)
EXPECTED_STAGE_B_TEST_GATE_SHA256 = (
    "f8580964ab6f55c4bd6f9bcd888491d10eda38722f7b7fc58b7b030a7342ae94"
)
EXPECTED_STAGE_B_SUMMARY_SHA256 = (
    "af7dec74a81f65f8b351df3f4333063cc25b7723ff65fc25256e40768389447e"
)
EXPECTED_STAGE_B_REPORT_SHA256 = (
    "acf1a04718c45c4cb2f2855126bbdede25a30543706a49b9922ca597485a41fe"
)
EXPECTED_STAGE_B_MANIFEST_SHA256 = (
    "a61daf1e168fcf30fa52359b65fc1b93439e6cf284bdadb1f7006ad096b1af8e"
)
EXPECTED_STAGE_B_ACTION_TEMPLATE_SHA256 = (
    "4275b76fd321322dcfc9d715f92e496e268a584513340f4d0b4d4a1963c5d495"
)
EXPECTED_STAGE_B_DATASET_SHA256 = (
    "17a872d72dceadca1151821dbb5fd62982bf5ff1585852c05688b14d49d78b69"
)
EXPECTED_STAGE_B_WINDOWS_SHA256 = (
    "0e55ae7e777536838e8ac70d33d830abc83b1940a42d97fd9b74081fce187319"
)
EXPECTED_ROWS = 4256
EXPECTED_TRAIN_ROWS = 2440
EXPECTED_PAIR_KEYS = 2128
EXPECTED_EPISODES = 896

STAGE_B_ROOT = "data/phase3_state_v3_slack"
STAGE_B_MANIFEST = f"{STAGE_B_ROOT}/manifest.json"
STAGE_B_ACTION_TEMPLATE = f"{STAGE_B_ROOT}/action_template.pkl"
STAGE_B_DATASET = (
    f"{STAGE_B_ROOT}/migrated/phase3_14b_r255_stageb_dataset.npz"
)
STAGE_B_WINDOWS = (
    f"{STAGE_B_ROOT}/windows/phase3_14b_r255_stageb_windows.npz"
)

CACHE_VERSION = "phase314b_r255_stagec_state_v3_cache_v1"
CACHE_FILE = "phase3_14b_r255_stagec_state_v3_training_cache.npz"
TRAIN_VIEW_FILE = "phase3_14b_r255_stagec_train_attribution_view.npz"
CACHE_MANIFEST_FILE = "phase3_14b_r255_stagec_cache_manifest.json"

SOURCE_FILES = (
    "ccda_phase3/schema_v3.py",
    "ccda_phase3/phase314b_r255_stageb_dataset.py",
    "ccda_phase3/phase314b_r255_stagec_cache.py",
    "ccda_phase3/phase314b_r255_stagec_attribution.py",
    "scripts/phase3_14b_r255_stagec_worker.py",
    "scripts/phase3_14b_r255_stagec_attribution_worker.py",
    "scripts/phase3_14b_r255_stagec_build_and_audit.py",
    "scripts/phase3_14b_r255_stagec_test_gate.py",
    "scripts/phase3_14b_r255_stagec_blocked.py",
    "scripts/phase3_14b_r255_stagec_run.sh",
    "tests/test_phase3_14b_r255_stagec_cache_attribution.py",
)

REQUIRED_WINDOW_KEYS = (
    "paper_x",
    "state_action_x",
    "y_state",
    "y_final_state",
    "y_action",
    "state_history_valid_mask",
    "future_valid_mask",
    "action_history_valid_mask",
    "condition_name",
    "visible_seed",
    "split_name",
    "source_file",
    "source_pickle_sha256",
    "pair_group",
    "episode_index",
    "window_t",
    "success",
    "final_fraction",
    "engagement_step",
    "release_step",
    "pre_engagement",
)


class StageCCacheError(RuntimeError):
    """Raised when the Stage-C cache contract cannot be satisfied."""


@dataclass(frozen=True)
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray
    active: np.ndarray

    def validate(self, *, tail_shape: Tuple[int, ...]) -> None:
        if self.mean.shape != tail_shape:
            raise ValueError(f"mean shape {self.mean.shape} != {tail_shape}")
        if self.scale.shape != tail_shape or self.active.shape != tail_shape:
            raise ValueError("standardizer shapes do not match")
        if self.mean.dtype != np.float32 or self.scale.dtype != np.float32:
            raise ValueError("standardizer mean/scale must be float32")
        if self.active.dtype != np.bool_:
            raise ValueError("standardizer active mask must be bool")
        if not np.all(np.isfinite(self.mean)) or not np.all(np.isfinite(self.scale)):
            raise ValueError("standardizer is non-finite")
        if np.any(self.scale <= 0.0):
            raise ValueError("standardizer scale must be positive")
        if not np.all(self.scale[~self.active] == 1.0):
            raise ValueError("inactive dimensions must have unit scale")

    def to_arrays(self, prefix: str) -> Dict[str, np.ndarray]:
        return {
            f"{prefix}_mean": self.mean.copy(),
            f"{prefix}_scale": self.scale.copy(),
            f"{prefix}_active": self.active.copy(),
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(json.dumps(list(array.shape)).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        if value.dtype.kind in "fc" and not np.all(np.isfinite(value)):
            raise ValueError("non-finite array cannot be serialized")
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("non-finite float cannot be serialized")
        return value
    raise TypeError(f"unsupported JSON value: {type(value)!r}")


def stable_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            jsonable(dict(payload)),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def atomic_write_once(path: Path, payload: bytes, *, mode: int = 0o644) -> None:
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite write-once artifact: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.{os.getpid()}.tmp"
    if temporary.exists():
        temporary.unlink()
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, target)
        directory_fd = os.open(str(target.parent), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def deterministic_npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    """Return a byte-stable compressed NPZ.

    ``numpy.savez_compressed`` stores current ZIP timestamps.  This writer fixes
    entry order, timestamps, permissions and compression settings so two
    independent processes produce identical file bytes.
    """
    output = io.BytesIO()
    with zipfile.ZipFile(
        output,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        strict_timestamps=True,
    ) as archive:
        for key in sorted(arrays):
            if not key or "/" in key or "\\" in key:
                raise ValueError(f"invalid NPZ key: {key!r}")
            array = np.asarray(arrays[key])
            if array.dtype.kind == "O":
                raise ValueError(f"object dtype is forbidden: {key}")
            if array.dtype.kind in "fc" and not np.all(np.isfinite(array)):
                raise ValueError(f"non-finite array is forbidden: {key}")
            npy = io.BytesIO()
            np.lib.format.write_array(
                npy,
                np.ascontiguousarray(array),
                allow_pickle=False,
            )
            info = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.flag_bits = 0
            archive.writestr(info, npy.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return output.getvalue()


def write_npz_once(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    atomic_write_once(path, deterministic_npz_bytes(arrays))


def load_npz_strict(path: Path, *, keys: Optional[Sequence[str]] = None) -> Dict[str, np.ndarray]:
    selected = None if keys is None else set(str(key) for key in keys)
    result: Dict[str, np.ndarray] = {}
    with np.load(Path(path), allow_pickle=False) as archive:
        available = set(archive.files)
        if selected is not None and not selected.issubset(available):
            raise StageCCacheError(
                f"NPZ is missing keys: {sorted(selected - available)}"
            )
        for key in archive.files:
            if selected is not None and key not in selected:
                continue
            value = np.asarray(archive[key])
            if value.dtype.kind == "O":
                raise StageCCacheError(f"object array is forbidden: {key}")
            if value.dtype.kind in "fc" and not np.all(np.isfinite(value)):
                raise StageCCacheError(f"non-finite array: {key}")
            result[key] = value.copy()
    return result


def fit_standardizer(value: np.ndarray, *, epsilon: float = 1.0e-12) -> Standardizer:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim < 2 or array.shape[0] < 2:
        raise ValueError("standardizer requires at least two rows")
    if not np.all(np.isfinite(array)):
        raise ValueError("standardizer input is non-finite")
    mean = np.mean(array, axis=0)
    std = np.std(array, axis=0)
    active = std > float(epsilon)
    scale = np.where(active, std, 1.0)
    result = Standardizer(
        mean=mean.astype(np.float32),
        scale=scale.astype(np.float32),
        active=active.astype(np.bool_),
    )
    result.validate(tail_shape=array.shape[1:])
    return result


def fit_masked_future_standardizer(
    future: np.ndarray,
    valid_mask: np.ndarray,
    *,
    epsilon: float = 1.0e-12,
) -> Standardizer:
    values = np.asarray(future, dtype=np.float64)
    mask = np.asarray(valid_mask, dtype=np.bool_)
    if values.ndim != 3 or values.shape[1:] != (DEFAULT_TF, STATE_DIM):
        raise ValueError("future must be [N,4,67]")
    if mask.shape != values.shape[:2]:
        raise ValueError("future mask must be [N,4]")
    mean = np.zeros((DEFAULT_TF, STATE_DIM), dtype=np.float64)
    scale = np.ones((DEFAULT_TF, STATE_DIM), dtype=np.float64)
    active = np.zeros((DEFAULT_TF, STATE_DIM), dtype=np.bool_)
    for horizon in range(DEFAULT_TF):
        selected = values[mask[:, horizon], horizon]
        if selected.shape[0] < 2:
            raise ValueError(f"insufficient valid rows at horizon {horizon}")
        mean[horizon] = np.mean(selected, axis=0)
        std = np.std(selected, axis=0)
        active[horizon] = std > float(epsilon)
        scale[horizon] = np.where(active[horizon], std, 1.0)
    result = Standardizer(
        mean=mean.astype(np.float32),
        scale=scale.astype(np.float32),
        active=active,
    )
    result.validate(tail_shape=(DEFAULT_TF, STATE_DIM))
    return result


def _require_shape(
    arrays: Mapping[str, np.ndarray],
    key: str,
    shape: Tuple[Optional[int], ...],
    dtype: Optional[np.dtype] = None,
) -> np.ndarray:
    if key not in arrays:
        raise StageCCacheError(f"missing Stage-B window key: {key}")
    value = np.asarray(arrays[key])
    if value.ndim != len(shape):
        raise StageCCacheError(f"{key} rank {value.ndim} != {len(shape)}")
    for observed, expected in zip(value.shape, shape):
        if expected is not None and observed != expected:
            raise StageCCacheError(f"{key} shape {value.shape} != {shape}")
    if dtype is not None and value.dtype != np.dtype(dtype):
        raise StageCCacheError(f"{key} dtype {value.dtype} != {np.dtype(dtype)}")
    if value.dtype.kind in "fc" and not np.all(np.isfinite(value)):
        raise StageCCacheError(f"{key} is non-finite")
    return value


def _validate_quaternion_batch(states: np.ndarray, *, name: str) -> None:
    array = np.asarray(states, dtype=np.float32)
    if array.shape[-1] != STATE_DIM:
        raise StageCCacheError(f"{name} is not state-v3")
    quaternion = array[..., EE_QUATERNION_SLICE].astype(np.float64)
    norm_error = np.abs(np.linalg.norm(quaternion, axis=-1) - 1.0)
    if float(np.max(norm_error)) > 1.0e-5:
        raise StageCCacheError(
            f"{name} quaternion max norm error {float(np.max(norm_error))}"
        )
    if np.any(quaternion[..., 3] < -1.0e-12):
        raise StageCCacheError(f"{name} quaternion hemisphere is not canonical")
    tie = np.abs(quaternion[..., 3]) <= 1.0e-12
    if np.any(tie):
        flat = quaternion[tie]
        for row in flat:
            for component in row[:3]:
                if abs(float(component)) > 1.0e-12:
                    if component < 0.0:
                        raise StageCCacheError(
                            f"{name} quaternion tie-break is not canonical"
                        )
                    break


def validate_stage_b_windows(arrays: Mapping[str, np.ndarray]) -> Dict[str, Any]:
    missing = sorted(set(REQUIRED_WINDOW_KEYS) - set(arrays))
    if missing:
        raise StageCCacheError(f"Stage-B windows missing keys: {missing}")
    paper = _require_shape(arrays, "paper_x", (EXPECTED_ROWS, PAPER_X_DIM), np.float32)
    state_action = _require_shape(
        arrays, "state_action_x", (EXPECTED_ROWS, STATE_ACTION_X_DIM), np.float32
    )
    future = _require_shape(
        arrays, "y_state", (EXPECTED_ROWS, DEFAULT_TF, STATE_DIM), np.float32
    )
    final = _require_shape(arrays, "y_final_state", (EXPECTED_ROWS, STATE_DIM), np.float32)
    action = _require_shape(arrays, "y_action", (EXPECTED_ROWS, ACTION_DIM), np.float32)
    future_mask = _require_shape(
        arrays, "future_valid_mask", (EXPECTED_ROWS, DEFAULT_TF), np.bool_
    )
    _require_shape(
        arrays, "state_history_valid_mask", (EXPECTED_ROWS, DEFAULT_TH), np.bool_
    )
    _require_shape(
        arrays, "action_history_valid_mask", (EXPECTED_ROWS, DEFAULT_TH), np.bool_
    )
    if not np.array_equal(final, future[:, -1]):
        raise StageCCacheError("y_final_state differs from y_state[:, -1]")
    if not np.array_equal(state_action[:, :PAPER_X_DIM], paper):
        raise StageCCacheError("state_action_x does not begin with paper_x")
    history = paper.reshape(EXPECTED_ROWS, DEFAULT_TH, STATE_DIM)
    _validate_quaternion_batch(history, name="history")
    _validate_quaternion_batch(future, name="future")
    if not np.all(future_mask[:, 0]):
        raise StageCCacheError("first future horizon must be valid for every row")
    state_history_mask = np.asarray(
        arrays["state_history_valid_mask"], dtype=np.bool_
    )
    action_history_mask = np.asarray(
        arrays["action_history_valid_mask"], dtype=np.bool_
    )
    # Validity masks must be monotone: left-padded histories are false->true,
    # and right-padded futures are true->false.
    if np.any(np.diff(state_history_mask.astype(np.int8), axis=1) < 0):
        raise StageCCacheError("state history validity is not right-aligned")
    if np.any(np.diff(action_history_mask.astype(np.int8), axis=1) < 0):
        raise StageCCacheError("action history validity is not right-aligned")
    if np.any(np.diff(future_mask.astype(np.int8), axis=1) > 0):
        raise StageCCacheError("future validity is not left-aligned")
    action_history = state_action[:, PAPER_X_DIM:].reshape(
        EXPECTED_ROWS, DEFAULT_TH, ACTION_DIM
    )
    if np.any(np.abs(action_history[~action_history_mask]) > 0.0):
        raise StageCCacheError("invalid past-action slots are not exact zero")
    for row_index in range(EXPECTED_ROWS):
        valid_history = np.flatnonzero(state_history_mask[row_index])
        if valid_history.size == 0:
            raise StageCCacheError("state history has no valid element")
        first_valid = int(valid_history[0])
        if first_valid > 0 and not np.array_equal(
            history[row_index, :first_valid],
            np.repeat(
                history[row_index, first_valid:first_valid + 1],
                first_valid,
                axis=0,
            ),
        ):
            raise StageCCacheError("left-padded state history changed")
        valid_future = np.flatnonzero(future_mask[row_index])
        last_valid = int(valid_future[-1])
        if last_valid + 1 < DEFAULT_TF and not np.array_equal(
            future[row_index, last_valid + 1:],
            np.repeat(
                future[row_index, last_valid:last_valid + 1],
                DEFAULT_TF - last_valid - 1,
                axis=0,
            ),
        ):
            raise StageCCacheError("right-padded future state changed")

    splits = np.asarray(arrays["split_name"]).astype(str)
    conditions = np.asarray(arrays["condition_name"]).astype(str)
    seeds = np.asarray(arrays["visible_seed"], dtype=np.int64)
    groups = np.asarray(arrays["pair_group"]).astype(str)
    times = np.asarray(arrays["window_t"], dtype=np.int64)
    source_files = np.asarray(arrays["source_file"]).astype(str)
    source_sha = np.asarray(arrays["source_pickle_sha256"]).astype(str)
    episode_index = np.asarray(arrays["episode_index"], dtype=np.int64)
    success = np.asarray(arrays["success"], dtype=np.bool_)
    final_fraction = np.asarray(arrays["final_fraction"], dtype=np.float64)
    engagement = np.asarray(arrays["engagement_step"], dtype=np.int64)
    release = np.asarray(arrays["release_step"], dtype=np.int64)
    pre_engagement = np.asarray(arrays["pre_engagement"], dtype=np.bool_)
    metadata = {
        "split_name": splits,
        "condition_name": conditions,
        "visible_seed": seeds,
        "pair_group": groups,
        "window_t": times,
        "source_file": source_files,
        "source_pickle_sha256": source_sha,
        "episode_index": episode_index,
        "success": success,
        "final_fraction": final_fraction,
        "engagement_step": engagement,
        "release_step": release,
        "pre_engagement": pre_engagement,
    }
    for name, value in metadata.items():
        if value.shape != (EXPECTED_ROWS,):
            raise StageCCacheError(
                f"metadata {name} shape {value.shape} != ({EXPECTED_ROWS},)"
            )
    if np.any(~np.isfinite(final_fraction)):
        raise StageCCacheError("final_fraction is non-finite")
    if np.any((final_fraction < 0.0) | (final_fraction > 1.0)):
        raise StageCCacheError("final_fraction is outside [0,1]")
    if np.any(seeds < 0) or np.any(times < 0) or np.any(episode_index < 0):
        raise StageCCacheError("seed/time/episode metadata is negative")
    if np.any(engagement < -1) or np.any(release < -1):
        raise StageCCacheError("engagement/release metadata is below -1")
    if any(not value for value in groups.tolist()):
        raise StageCCacheError("pair_group contains an empty value")
    if any(not value for value in source_files.tolist()):
        raise StageCCacheError("source_file contains an empty value")
    hexadecimal = set("0123456789abcdef")
    if any(
        len(value) != 64 or any(character not in hexadecimal for character in value)
        for value in source_sha.tolist()
    ):
        raise StageCCacheError("source_pickle_sha256 is not lowercase SHA256")
    train_rows = int(np.sum(splits == "train"))
    if train_rows != EXPECTED_TRAIN_ROWS:
        raise StageCCacheError(f"train rows {train_rows} != {EXPECTED_TRAIN_ROWS}")
    if set(splits.tolist()) != {"train", "val", "test"}:
        raise StageCCacheError("split names changed")
    expected_conditions = {"free", "hidden_slack_breakaway_pin_v2"}
    if set(conditions.tolist()) != expected_conditions:
        raise StageCCacheError("condition names changed")

    pair_members: MutableMapping[str, list] = {}
    for split, seed, group, time, condition in zip(
        splits, seeds, groups, times, conditions
    ):
        key = f"{split}|{int(seed)}|{group}|{int(time)}"
        pair_members.setdefault(key, []).append(condition)
    invalid = {
        key: sorted(value)
        for key, value in pair_members.items()
        if sorted(value) != sorted(expected_conditions)
    }
    if invalid:
        raise StageCCacheError(
            f"invalid paired window multiplicity: {list(invalid.items())[:5]}"
        )
    if len(pair_members) != EXPECTED_PAIR_KEYS:
        raise StageCCacheError(
            f"pair keys {len(pair_members)} != {EXPECTED_PAIR_KEYS}"
        )
    pair_rows: MutableMapping[str, list] = {}
    for row_index, key in enumerate(
        f"{split}|{int(seed)}|{group}|{int(time)}"
        for split, seed, group, time in zip(splits, seeds, groups, times)
    ):
        pair_rows.setdefault(key, []).append(row_index)
    for key, row_indices in pair_rows.items():
        if len(row_indices) != 2:
            raise StageCCacheError(f"pair {key} does not have two rows")
        left_row, right_row = row_indices
        if not np.array_equal(action[left_row], action[right_row]):
            raise StageCCacheError(f"paired target action changed: {key}")
        if not np.array_equal(
            action_history[left_row], action_history[right_row]
        ):
            raise StageCCacheError(f"paired action history changed: {key}")

    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        for name, values in (
            ("visible_seed", seeds),
            ("pair_group", groups),
            ("source_file", source_files),
        ):
            overlap = set(values[splits == left].tolist()).intersection(
                set(values[splits == right].tolist())
            )
            if overlap:
                raise StageCCacheError(
                    f"{left}/{right} {name} overlap: {len(overlap)}"
                )
    return {
        "rows": EXPECTED_ROWS,
        "train_rows": train_rows,
        "pair_keys": len(pair_members),
        "state_dim": STATE_DIM,
        "paper_x_dim": PAPER_X_DIM,
        "state_action_x_dim": STATE_ACTION_X_DIM,
        "future_shape": [DEFAULT_TF, STATE_DIM],
        "action_dim": ACTION_DIM,
        "quaternion_contract_pass": True,
        "split_isolation_pass": True,
        "pair_contract_pass": True,
    }


def make_pair_key(arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    values = [
        f"{split}|{int(seed)}|{group}|{int(time)}"
        for split, seed, group, time in zip(
            np.asarray(arrays["split_name"]).astype(str),
            np.asarray(arrays["visible_seed"], dtype=np.int64),
            np.asarray(arrays["pair_group"]).astype(str),
            np.asarray(arrays["window_t"], dtype=np.int64),
        )
    ]
    return np.asarray(values, dtype="<U160")


def make_episode_group_key(arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    values = [
        f"{split}|{int(seed)}|{group}"
        for split, seed, group in zip(
            np.asarray(arrays["split_name"]).astype(str),
            np.asarray(arrays["visible_seed"], dtype=np.int64),
            np.asarray(arrays["pair_group"]).astype(str),
        )
    ]
    return np.asarray(values, dtype="<U144")


def build_cache_arrays(
    windows: Mapping[str, np.ndarray],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, Any]]:
    validation = validate_stage_b_windows(windows)
    arrays: Dict[str, np.ndarray] = {
        key: np.asarray(value).copy() for key, value in windows.items()
    }
    rows = EXPECTED_ROWS
    history = arrays["paper_x"].reshape(rows, DEFAULT_TH, STATE_DIM)
    current_state = history[:, -1].copy()
    next_state = arrays["y_state"][:, 0].copy()
    pair_key = make_pair_key(arrays)
    episode_group_key = make_episode_group_key(arrays)
    train_mask = arrays["split_name"].astype(str) == "train"
    train_row_index = np.flatnonzero(train_mask).astype(np.int64)

    cache_arrays: Dict[str, np.ndarray] = {
        **arrays,
        "row_index": np.arange(rows, dtype=np.int64),
        "pair_key": pair_key,
        "episode_group_key": episode_group_key,
        "current_state": current_state.astype(np.float32),
        "next_state": next_state.astype(np.float32),
        "robot_history": history[..., CABLE_DIM:].astype(np.float32),
        "robot_future": arrays["y_state"][..., CABLE_DIM:].astype(np.float32),
        "idm_pair_x": np.concatenate(
            [current_state, next_state], axis=1
        ).astype(np.float32),
        "idm_trajectory_x": np.concatenate(
            [arrays["paper_x"], arrays["y_state"].reshape(rows, -1)],
            axis=1,
        ).astype(np.float32),
        "direct_action_x": arrays["state_action_x"].copy(),
        "train_row_index": train_row_index,
    }

    paper_standardizer = fit_standardizer(arrays["paper_x"][train_mask])
    state_action_standardizer = fit_standardizer(
        arrays["state_action_x"][train_mask]
    )
    future_standardizer = fit_masked_future_standardizer(
        arrays["y_state"][train_mask],
        arrays["future_valid_mask"][train_mask],
    )
    action_standardizer = fit_standardizer(arrays["y_action"][train_mask])
    cache_arrays.update(paper_standardizer.to_arrays("paper_x"))
    cache_arrays.update(state_action_standardizer.to_arrays("state_action_x"))
    cache_arrays.update(future_standardizer.to_arrays("future"))
    cache_arrays.update(action_standardizer.to_arrays("action"))

    train_keys = (
        "paper_x",
        "state_action_x",
        "y_state",
        "y_action",
        "future_valid_mask",
        "state_history_valid_mask",
        "action_history_valid_mask",
        "condition_name",
        "split_name",
        "visible_seed",
        "pair_group",
        "source_file",
        "episode_index",
        "window_t",
        "pre_engagement",
    )
    train_view: Dict[str, np.ndarray] = {
        key: arrays[key][train_mask].copy() for key in train_keys
    }
    train_view.update(
        {
            "source_row_index": train_row_index,
            "pair_key": pair_key[train_mask].copy(),
            "episode_group_key": episode_group_key[train_mask].copy(),
            "robot_history": history[train_mask, :, CABLE_DIM:].astype(np.float32),
            "robot_future": arrays["y_state"][
                train_mask, :, CABLE_DIM:
            ].astype(np.float32),
        }
    )
    for prefix in ("paper_x", "state_action_x", "future", "action"):
        for suffix in ("mean", "scale", "active"):
            key = f"{prefix}_{suffix}"
            train_view[key] = cache_arrays[key].copy()

    if train_view["paper_x"].shape[0] != EXPECTED_TRAIN_ROWS:
        raise StageCCacheError("train-only view row count changed")
    if set(train_view["split_name"].astype(str).tolist()) != {"train"}:
        raise StageCCacheError("train-only view contains non-train rows")
    for name, value in cache_arrays.items():
        if value.dtype.kind == "O":
            raise StageCCacheError(f"cache contains object array: {name}")
        if value.dtype.kind in "fc" and not np.all(np.isfinite(value)):
            raise StageCCacheError(f"cache contains non-finite array: {name}")
    return cache_arrays, train_view, validation


def _load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise StageCCacheError(f"JSON root is not an object: {path}")
    return value


def validate_stage_b_artifacts(root: Path) -> Dict[str, Any]:
    root = Path(root).resolve()
    expected_files = {
        "reports/phase3_14b_r255_stageb_test_gate_summary.json":
            EXPECTED_STAGE_B_TEST_GATE_SHA256,
        "reports/phase3_14b_r255_stageb_summary.json":
            EXPECTED_STAGE_B_SUMMARY_SHA256,
        "reports/phase3_14b_r255_stageb_report.md":
            EXPECTED_STAGE_B_REPORT_SHA256,
        STAGE_B_MANIFEST: EXPECTED_STAGE_B_MANIFEST_SHA256,
        STAGE_B_ACTION_TEMPLATE: EXPECTED_STAGE_B_ACTION_TEMPLATE_SHA256,
        STAGE_B_DATASET: EXPECTED_STAGE_B_DATASET_SHA256,
        STAGE_B_WINDOWS: EXPECTED_STAGE_B_WINDOWS_SHA256,
        "data/phase3_14_cache/phase3_14a_training_cache.npz":
            EXPECTED_LEGACY_CACHE_SHA256,
    }
    for relative, expected in expected_files.items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = sha256_file(path)
        if observed != expected:
            raise StageCCacheError(
                f"immutable input SHA changed: {relative}: {observed}"
            )

    summary = _load_json(root / "reports/phase3_14b_r255_stageb_summary.json")
    if summary.get("verdict") != "PASS":
        raise StageCCacheError("Stage B verdict is not PASS")
    if summary.get("root_cause") != (
        "phase314b_r255_stageb_state_v3_dataset_materialized_and_byte_reproducible"
    ):
        raise StageCCacheError("unexpected Stage B root cause")
    if summary.get("required_next_path") != (
        "BUILD_NEW_STATE_V3_CACHE_AND_RERUN_ROBOT_PROXY_ATTRIBUTION"
    ):
        raise StageCCacheError("unexpected Stage B next path")
    if summary.get("robot_proxy_attribution_interpretable") is not False:
        raise StageCCacheError("Stage B attribution boundary changed")

    manifest = _load_json(root / STAGE_B_MANIFEST)
    if manifest.get("dataset_sha256") != EXPECTED_STAGE_B_DATASET_SHA256:
        raise StageCCacheError("Stage B dataset manifest binding changed")
    if manifest.get("windows_sha256") != EXPECTED_STAGE_B_WINDOWS_SHA256:
        raise StageCCacheError("Stage B windows manifest binding changed")
    if manifest.get("action_template_sha256") != EXPECTED_STAGE_B_ACTION_TEMPLATE_SHA256:
        raise StageCCacheError("Stage B action-template binding changed")
    if int(manifest.get("episode_count", -1)) != EXPECTED_EPISODES:
        raise StageCCacheError("Stage B episode count changed")
    if int(manifest.get("window_count", -1)) != EXPECTED_ROWS:
        raise StageCCacheError("Stage B window count changed")
    schema = manifest.get("state_schema", {})
    if schema.get("schema_version") != SCHEMA_VERSION:
        raise StageCCacheError("Stage B state schema changed")
    if int(schema.get("state_dim", -1)) != STATE_DIM:
        raise StageCCacheError("Stage B state dimension changed")
    return {
        "summary": summary,
        "manifest": manifest,
        "file_sha256": expected_files,
    }


def build_cache_artifacts(
    *,
    root: Path,
    output_root: Path,
) -> Dict[str, Any]:
    root = Path(root).resolve()
    output_root = Path(output_root).resolve()
    if output_root.exists():
        raise FileExistsError(f"worker output already exists: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)

    inputs = validate_stage_b_artifacts(root)
    windows = load_npz_strict(root / STAGE_B_WINDOWS)
    cache_arrays, train_view, validation = build_cache_arrays(windows)

    cache_path = output_root / CACHE_FILE
    train_view_path = output_root / TRAIN_VIEW_FILE
    manifest_path = output_root / CACHE_MANIFEST_FILE
    write_npz_once(cache_path, cache_arrays)
    write_npz_once(train_view_path, train_view)

    # Re-open every generated array without pickle before sealing the manifest.
    checked_cache = load_npz_strict(cache_path)
    checked_train = load_npz_strict(train_view_path)
    if set(checked_cache) != set(cache_arrays):
        raise StageCCacheError("cache key set changed after serialization")
    if set(checked_train) != set(train_view):
        raise StageCCacheError("train-view key set changed after serialization")
    for key in cache_arrays:
        if not np.array_equal(checked_cache[key], cache_arrays[key]):
            raise StageCCacheError(f"cache array changed after serialization: {key}")
    for key in train_view:
        if not np.array_equal(checked_train[key], train_view[key]):
            raise StageCCacheError(
                f"train-view array changed after serialization: {key}"
            )

    schema = SchemaV3Manifest()
    schema.validate()
    source_sha = {
        relative: sha256_file(root / relative)
        for relative in SOURCE_FILES
        if (root / relative).is_file()
    }
    manifest_payload: Dict[str, Any] = {
        "phase": PHASE,
        "schema": "phase314b_r255_stagec_cache_manifest_v1",
        "cache_version": CACHE_VERSION,
        "state_schema": schema.to_dict(),
        "source_stage_b_manifest": STAGE_B_MANIFEST,
        "source_stage_b_manifest_sha256": EXPECTED_STAGE_B_MANIFEST_SHA256,
        "source_stage_b_dataset_sha256": EXPECTED_STAGE_B_DATASET_SHA256,
        "source_stage_b_windows_sha256": EXPECTED_STAGE_B_WINDOWS_SHA256,
        "source_action_template_sha256":
            EXPECTED_STAGE_B_ACTION_TEMPLATE_SHA256,
        "source_legacy_cache_sha256": EXPECTED_LEGACY_CACHE_SHA256,
        "cache_file": CACHE_FILE,
        "cache_sha256": sha256_file(cache_path),
        "cache_size_bytes": int(cache_path.stat().st_size),
        "train_attribution_view_file": TRAIN_VIEW_FILE,
        "train_attribution_view_sha256": sha256_file(train_view_path),
        "train_attribution_view_size_bytes": int(train_view_path.stat().st_size),
        "rows": EXPECTED_ROWS,
        "train_rows": EXPECTED_TRAIN_ROWS,
        "pair_keys": EXPECTED_PAIR_KEYS,
        "validation": validation,
        "array_sha256": {
            key: sha256_array(value) for key, value in cache_arrays.items()
        },
        "train_view_array_sha256": {
            key: sha256_array(value) for key, value in train_view.items()
        },
        "source_code_sha256": source_sha,
        "deterministic_npz": True,
        "write_once": True,
        "legacy_cache_modified": False,
        "stage_b_artifacts_modified": False,
        "validation_targets_used": False,
        "formal_test_targets_used": False,
        "diffusion_training": False,
        "reverse_sampling": False,
        "idm_training": False,
        "candidate_execution": False,
        "phase4": False,
        "cps": False,
    }
    atomic_write_once(manifest_path, stable_json_bytes(manifest_payload))
    return manifest_payload


def directory_manifest(root: Path) -> Dict[str, str]:
    base = Path(root)
    return {
        path.relative_to(base).as_posix(): sha256_file(path)
        for path in sorted(base.rglob("*"))
        if path.is_file()
    }


def compare_build_directories(left: Path, right: Path) -> Dict[str, Any]:
    left_manifest = directory_manifest(left)
    right_manifest = directory_manifest(right)
    exact = left_manifest == right_manifest
    return {
        "exact": exact,
        "file_count": len(left_manifest),
        "left": left_manifest,
        "right": right_manifest,
        "only_left": sorted(set(left_manifest) - set(right_manifest)),
        "only_right": sorted(set(right_manifest) - set(left_manifest)),
        "different": sorted(
            key
            for key in set(left_manifest).intersection(right_manifest)
            if left_manifest[key] != right_manifest[key]
        ),
    }
