"""Phase3.14b-r2.5.6 Stage A contracts and train-only audits.

This stage performs two logically separate operations:

1. Materialize a cable-only diffusion target while retaining deployable
   state/action history as conditioning.
2. Audit whether the existing scripted dataset can identify a deployable IDM
   mapping from history plus desired cable motion to the next action.

It does not train a diffusion model or a formal IDM.  Historical state-v2/v3
artifacts remain immutable.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from ccda_phase3.schema_v3 import (
    ACTION_DIM,
    CABLE_DIM,
    DEFAULT_TF,
    DEFAULT_TH,
    PAPER_X_DIM,
    ROBOT_PROXY_DIM,
    STATE_ACTION_X_DIM,
    STATE_DIM,
)

PHASE = "Phase3.14b-r2.5.6 Stage A"
PHASE_ID = "phase314b_r256_stagea"
BASE_EVIDENCE_COMMIT = "5ba05f4c6a3ae38bc8869fccced29583dec86670"
EXPECTED_SUBMODULE_COMMIT = "633a88752445cf5d6776ed374fdbbdb35f93050c"

EXPECTED_STAGEC_TEST_GATE_SHA256 = (
    "fb5115e4c368727a474e80230962f0cefa08ddab525806f0413a13f2c011bfcd"
)
EXPECTED_STAGEC_PREFLIGHT_SHA256 = (
    "42c2444a77c83b0dfd997e8701b766352e021c44e41c033fb21f7e24b2891bbd"
)
EXPECTED_STAGEC_CACHE_SHA256 = (
    "c342284cfcfcadb9a1412cf3ec93c27887eb2560bf8be4bf4e6936f037797c17"
)
EXPECTED_STAGEC_TRAIN_VIEW_SHA256 = (
    "18bf6148f9e8041c6e0d937c750d3fc750f87ae371be750d6956aa7c59558b73"
)
EXPECTED_STAGEC_CACHE_MANIFEST_SHA256 = (
    "c47715af4554a412f8059cc1e53087d93cfdad9bda6ea086afc746f226e6f59f"
)
EXPECTED_STAGEC_ATTRIBUTION_SHA256 = (
    "70794e8837d8795346689b6ee021388e83184a0f811374e35ded70b043c0380a"
)
EXPECTED_STAGEC_SUMMARY_SHA256 = (
    "1499ba284a0d8e01d27ca1b76499de3efe721c86733cfc01f908eb108b882d4b"
)
EXPECTED_STAGEC_REPORT_SHA256 = (
    "2e3e7c2e47ae7063731917d22c1b15b778ebd6d07b97b8cae5d62157c791918e"
)

EXPECTED_ROWS = 4256
EXPECTED_TRAIN_ROWS = 2440
EXPECTED_PAIR_KEYS = 2128

STAGEC_ROOT = "data/phase3_14_cache_v3_r255_stagec_resume1"
STAGEC_CACHE = (
    f"{STAGEC_ROOT}/phase3_14b_r255_stagec_state_v3_training_cache.npz"
)
STAGEC_TRAIN_VIEW = (
    f"{STAGEC_ROOT}/phase3_14b_r255_stagec_train_attribution_view.npz"
)
STAGEC_CACHE_MANIFEST = (
    f"{STAGEC_ROOT}/phase3_14b_r255_stagec_cache_manifest.json"
)

CONTRACT_VERSION = "phase314b_r256_stagea_cable_diffusion_idm_contract_v1"
CONTRACT_FILE = "phase3_14b_r256_stagea_cable_idm_contract.npz"
TRAIN_VIEW_FILE = "phase3_14b_r256_stagea_train_audit_view.npz"
MANIFEST_FILE = "phase3_14b_r256_stagea_manifest.json"

DIFFUSION_CONDITION_DIM = STATE_ACTION_X_DIM
DIFFUSION_TARGET_DIM = CABLE_DIM
IDM_HISTORY_DIM = STATE_ACTION_X_DIM
IDM_NEXT_CABLE_DIM = STATE_ACTION_X_DIM + CABLE_DIM
IDM_CABLE_TRAJECTORY_DIM = (
    STATE_ACTION_X_DIM + DEFAULT_TF * CABLE_DIM
)

REQUIRED_CACHE_KEYS = (
    "paper_x",
    "state_action_x",
    "y_state",
    "y_final_state",
    "y_action",
    "future_valid_mask",
    "state_history_valid_mask",
    "action_history_valid_mask",
    "condition_name",
    "split_name",
    "visible_seed",
    "pair_group",
    "pair_key",
    "episode_group_key",
    "source_file",
    "source_pickle_sha256",
    "episode_index",
    "window_t",
    "pre_engagement",
    "current_state",
    "row_index",
)

FORBIDDEN_MODEL_METADATA_KEYS = (
    "condition_name",
    "split_name",
    "visible_seed",
    "pair_group",
    "pair_key",
    "episode_group_key",
    "source_file",
    "source_pickle_sha256",
    "episode_index",
    "window_t",
    "pre_engagement",
    "success",
    "final_fraction",
    "engagement_step",
    "release_step",
)

SOURCE_AUDIT_FILES = (
    "ccda_phase3/phase314b_models.py",
    "ccda_phase3/phase314b_diffusion.py",
    "ccda_phase3/phase314b_r2_diffusion.py",
    "ccda_phase3/phase314b_r255_stagec_cache.py",
    "ccda_phase3/phase314b_r255_stagec_attribution.py",
)


class StageAContractError(RuntimeError):
    """Raised when the r2.5.6 Stage-A contract cannot be established."""


@dataclass(frozen=True)
class AuditSpec:
    folds: int = 8
    ridge_regularization: float = 1.0e-3
    variance_epsilon: float = 1.0e-10
    action_equality_epsilon: float = 1.0e-8
    cable_incremental_gain_min: float = 0.05
    gain_over_permuted_min: float = 0.02
    minimum_active_action_dims: int = 2
    minimum_unique_action_vectors: int = 16
    minimum_paired_action_diverse_fraction: float = 0.05
    permutation_shifts: Tuple[int, ...] = (1, 3, 7, 13)

    def validate(self) -> None:
        if self.folds < 2:
            raise ValueError("fold count must be at least two")
        if self.ridge_regularization <= 0.0:
            raise ValueError("ridge regularization must be positive")
        if self.minimum_active_action_dims < 1:
            raise ValueError("minimum active action dimensions must be positive")
        if self.minimum_unique_action_vectors < self.folds:
            raise ValueError("minimum unique actions must be at least fold count")
        if not self.permutation_shifts:
            raise ValueError("at least one negative-control shift is required")


@dataclass(frozen=True)
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray
    active: np.ndarray

    def validate(self, tail_shape: Tuple[int, ...]) -> None:
        if self.mean.shape != tail_shape:
            raise ValueError(f"mean shape {self.mean.shape} != {tail_shape}")
        if self.scale.shape != tail_shape or self.active.shape != tail_shape:
            raise ValueError("standardizer shape mismatch")
        if self.mean.dtype != np.float32 or self.scale.dtype != np.float32:
            raise ValueError("mean and scale must be float32")
        if self.active.dtype != np.bool_:
            raise ValueError("active mask must be bool")
        if not np.all(np.isfinite(self.mean)):
            raise ValueError("mean is non-finite")
        if not np.all(np.isfinite(self.scale)) or np.any(self.scale <= 0.0):
            raise ValueError("scale is invalid")
        if not np.all(self.scale[~self.active] == 1.0):
            raise ValueError("inactive dimensions must have scale one")

    def arrays(self, prefix: str) -> Dict[str, np.ndarray]:
        return {
            f"{prefix}_mean": self.mean.copy(),
            f"{prefix}_scale": self.scale.copy(),
            f"{prefix}_active": self.active.copy(),
        }


def git_output(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=str(root),
        text=True,
    ).strip()


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
            raise ValueError("non-finite scalar cannot be serialized")
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
        raise FileExistsError(f"refusing to overwrite: {target}")
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
            value = np.asarray(arrays[key])
            if value.dtype.kind == "O":
                raise ValueError(f"object dtype is forbidden: {key}")
            if value.dtype.kind in "fc" and not np.all(np.isfinite(value)):
                raise ValueError(f"non-finite array is forbidden: {key}")
            payload = io.BytesIO()
            np.lib.format.write_array(
                payload,
                np.ascontiguousarray(value),
                allow_pickle=False,
            )
            info = zipfile.ZipInfo(
                f"{key}.npy",
                date_time=(1980, 1, 1, 0, 0, 0),
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.flag_bits = 0
            archive.writestr(
                info,
                payload.getvalue(),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )
    return output.getvalue()


def write_npz_once(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    atomic_write_once(path, deterministic_npz_bytes(arrays))


def load_npz_strict(
    path: Path,
    *,
    keys: Optional[Sequence[str]] = None,
) -> Dict[str, np.ndarray]:
    selected = None if keys is None else set(str(key) for key in keys)
    result: Dict[str, np.ndarray] = {}
    with np.load(Path(path), allow_pickle=False) as archive:
        available = set(archive.files)
        if selected is not None and not selected.issubset(available):
            raise StageAContractError(
                f"NPZ is missing keys: {sorted(selected - available)}"
            )
        for key in archive.files:
            if selected is not None and key not in selected:
                continue
            value = np.asarray(archive[key])
            if value.dtype.kind == "O":
                raise StageAContractError(f"object array is forbidden: {key}")
            if value.dtype.kind in "fc" and not np.all(np.isfinite(value)):
                raise StageAContractError(f"non-finite array: {key}")
            result[key] = value.copy()
    return result


def fit_standardizer(
    values: np.ndarray,
    *,
    epsilon: float = 1.0e-12,
) -> Standardizer:
    array = np.asarray(values, dtype=np.float64)
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
    result.validate(array.shape[1:])
    return result


def fit_masked_cable_standardizer(
    cable_future: np.ndarray,
    valid_mask: np.ndarray,
    *,
    epsilon: float = 1.0e-12,
) -> Standardizer:
    future = np.asarray(cable_future, dtype=np.float64)
    mask = np.asarray(valid_mask, dtype=np.bool_)
    if future.ndim != 3 or future.shape[1:] != (DEFAULT_TF, CABLE_DIM):
        raise ValueError("cable future must be [N,4,48]")
    if mask.shape != future.shape[:2]:
        raise ValueError("future mask must be [N,4]")
    mean = np.zeros((DEFAULT_TF, CABLE_DIM), dtype=np.float64)
    scale = np.ones((DEFAULT_TF, CABLE_DIM), dtype=np.float64)
    active = np.zeros((DEFAULT_TF, CABLE_DIM), dtype=np.bool_)
    for horizon in range(DEFAULT_TF):
        selected = future[mask[:, horizon], horizon]
        if selected.shape[0] < 2:
            raise ValueError(f"horizon {horizon} has insufficient valid rows")
        mean[horizon] = np.mean(selected, axis=0)
        std = np.std(selected, axis=0)
        active[horizon] = std > float(epsilon)
        scale[horizon] = np.where(active[horizon], std, 1.0)
    result = Standardizer(
        mean=mean.astype(np.float32),
        scale=scale.astype(np.float32),
        active=active,
    )
    result.validate((DEFAULT_TF, CABLE_DIM))
    return result


def load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise StageAContractError(f"JSON root is not an object: {path}")
    return value


def validate_immutable_inputs(root: Path) -> Dict[str, Any]:
    expected = {
        "reports/phase3_14b_r255_stagec_test_gate_summary.json":
            EXPECTED_STAGEC_TEST_GATE_SHA256,
        "reports/phase3_14b_r255_stagec_resume1_correction_preflight.json":
            EXPECTED_STAGEC_PREFLIGHT_SHA256,
        STAGEC_CACHE: EXPECTED_STAGEC_CACHE_SHA256,
        STAGEC_TRAIN_VIEW: EXPECTED_STAGEC_TRAIN_VIEW_SHA256,
        STAGEC_CACHE_MANIFEST: EXPECTED_STAGEC_CACHE_MANIFEST_SHA256,
        "reports/phase3_14b_r255_stagec_resume1_attribution.json":
            EXPECTED_STAGEC_ATTRIBUTION_SHA256,
        "reports/phase3_14b_r255_stagec_resume1_summary.json":
            EXPECTED_STAGEC_SUMMARY_SHA256,
        "reports/phase3_14b_r255_stagec_resume1_report.md":
            EXPECTED_STAGEC_REPORT_SHA256,
    }
    for relative, expected_sha in expected.items():
        path = Path(root) / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = sha256_file(path)
        if observed != expected_sha:
            raise StageAContractError(
                f"immutable input changed: {relative}: {observed}"
            )
    summary = load_json(
        Path(root) / "reports/phase3_14b_r255_stagec_resume1_summary.json"
    )
    if summary.get("verdict") != "PASS":
        raise StageAContractError("Stage C Resume1 verdict is not PASS")
    if summary.get("required_next_path") != (
        "REMOVE_ROBOT_FUTURE_FROM_DIFFUSION_TARGET_AND_REDEFINE_IDM_INPUT"
    ):
        raise StageAContractError("Stage C next path changed")
    if summary.get("scientific_status") != "BLOCKED":
        raise StageAContractError("Stage C scientific boundary changed")
    if summary.get("train_only_recommendation") is not None:
        raise StageAContractError("Stage C selected a recommendation")
    if summary.get("selected_configuration") is not None:
        raise StageAContractError("Stage C selected a configuration")
    return {
        "file_sha256": expected,
        "stagec_summary": summary,
    }


def source_contract_audit(root: Path) -> Dict[str, Any]:
    texts = {
        relative: (Path(root) / relative).read_text(encoding="utf-8")
        for relative in SOURCE_AUDIT_FILES
    }
    checks = {
        "historical_models_bind_schema_v2": (
            "from ccda_phase3.schema_v2 import DEFAULT_TF, STATE_DIM"
            in texts["ccda_phase3/phase314b_models.py"]
        ),
        "historical_diffusion_binds_schema_v2": (
            "from ccda_phase3.schema_v2 import DEFAULT_TF, STATE_DIM"
            in texts["ccda_phase3/phase314b_diffusion.py"]
        ),
        "historical_r2_diffusion_binds_schema_v2": (
            "from ccda_phase3.schema_v2 import DEFAULT_TF, STATE_DIM"
            in texts["ccda_phase3/phase314b_r2_diffusion.py"]
        ),
        "historical_r2_error_mentions_87": (
            'raise ValueError("active mask must be [4,87]")'
            in texts["ccda_phase3/phase314b_r2_diffusion.py"]
        ),
        "stagec_cache_contains_full_state_idm_pair": (
            '"idm_pair_x": np.concatenate('
            in texts["ccda_phase3/phase314b_r255_stagec_cache.py"]
        ),
        "stagec_cache_contains_full_state_idm_trajectory": (
            '"idm_trajectory_x": np.concatenate('
            in texts["ccda_phase3/phase314b_r255_stagec_cache.py"]
        ),
        "stagec_action_probe_uses_future_robot": (
            '"history_plus_next_robot"'
            in texts["ccda_phase3/phase314b_r255_stagec_attribution.py"]
        ),
        "stagec_root_name_uses_ambiguous_or": (
            "robot_future_not_deployably_predictable_"
            in texts["ccda_phase3/phase314b_r255_stagec_attribution.py"]
            and "or_idm_material" in
            texts["ccda_phase3/phase314b_r255_stagec_attribution.py"]
        ),
    }
    return {
        "checks": checks,
        "all_expected_legacy_bindings_confirmed": bool(all(checks.values())),
        "source_sha256": {
            relative: sha256_file(Path(root) / relative)
            for relative in SOURCE_AUDIT_FILES
        },
    }


def _require_shape(
    arrays: Mapping[str, np.ndarray],
    key: str,
    shape: Tuple[Optional[int], ...],
    dtype: Optional[np.dtype] = None,
) -> np.ndarray:
    if key not in arrays:
        raise StageAContractError(f"missing cache key: {key}")
    value = np.asarray(arrays[key])
    if value.ndim != len(shape):
        raise StageAContractError(f"{key} rank mismatch: {value.shape}")
    for observed, expected in zip(value.shape, shape):
        if expected is not None and observed != expected:
            raise StageAContractError(
                f"{key} shape {value.shape} != {shape}"
            )
    if dtype is not None and value.dtype != np.dtype(dtype):
        raise StageAContractError(
            f"{key} dtype {value.dtype} != {np.dtype(dtype)}"
        )
    if value.dtype.kind in "fc" and not np.all(np.isfinite(value)):
        raise StageAContractError(f"{key} contains NaN or Inf")
    return value


def validate_stagec_cache(arrays: Mapping[str, np.ndarray]) -> Dict[str, Any]:
    missing = sorted(set(REQUIRED_CACHE_KEYS) - set(arrays))
    if missing:
        raise StageAContractError(f"Stage C cache missing keys: {missing}")
    paper_x = _require_shape(
        arrays, "paper_x", (EXPECTED_ROWS, PAPER_X_DIM), np.float32
    )
    state_action_x = _require_shape(
        arrays,
        "state_action_x",
        (EXPECTED_ROWS, STATE_ACTION_X_DIM),
        np.float32,
    )
    y_state = _require_shape(
        arrays,
        "y_state",
        (EXPECTED_ROWS, DEFAULT_TF, STATE_DIM),
        np.float32,
    )
    y_action = _require_shape(
        arrays, "y_action", (EXPECTED_ROWS, ACTION_DIM), np.float32
    )
    current_state = _require_shape(
        arrays, "current_state", (EXPECTED_ROWS, STATE_DIM), np.float32
    )
    future_mask = _require_shape(
        arrays,
        "future_valid_mask",
        (EXPECTED_ROWS, DEFAULT_TF),
        np.bool_,
    )
    if not np.array_equal(
        state_action_x[:, :PAPER_X_DIM],
        paper_x,
    ):
        raise StageAContractError("state_action_x prefix differs from paper_x")
    history = paper_x.reshape(EXPECTED_ROWS, DEFAULT_TH, STATE_DIM)
    if not np.array_equal(history[:, -1], current_state):
        raise StageAContractError("current_state differs from history tail")
    if not np.all(future_mask[:, 0]):
        raise StageAContractError("first future horizon is not universally valid")
    split = np.asarray(arrays["split_name"]).astype(str)
    if int(np.sum(split == "train")) != EXPECTED_TRAIN_ROWS:
        raise StageAContractError("train row count changed")
    pair_key = np.asarray(arrays["pair_key"]).astype(str)
    if len(set(pair_key.tolist())) != EXPECTED_PAIR_KEYS:
        raise StageAContractError("paired key count changed")
    if CABLE_DIM + ROBOT_PROXY_DIM != STATE_DIM:
        raise StageAContractError("state-v3 partition changed")
    return {
        "rows": EXPECTED_ROWS,
        "train_rows": EXPECTED_TRAIN_ROWS,
        "pair_keys": EXPECTED_PAIR_KEYS,
        "state_dim": STATE_DIM,
        "robot_proxy_dim": ROBOT_PROXY_DIM,
        "condition_dim": STATE_ACTION_X_DIM,
        "diffusion_target_shape": [DEFAULT_TF, CABLE_DIM],
        "action_dim": ACTION_DIM,
        "full_horizon_rows": int(np.sum(np.all(future_mask, axis=1))),
    }


def build_contract_arrays(
    arrays: Mapping[str, np.ndarray],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, Any]]:
    validation = validate_stagec_cache(arrays)
    condition = np.asarray(arrays["state_action_x"], dtype=np.float32)
    full_future = np.asarray(arrays["y_state"], dtype=np.float32)
    cable_future = full_future[..., :CABLE_DIM].copy()
    robot_future = full_future[..., CABLE_DIM:]
    current_state = np.asarray(arrays["current_state"], dtype=np.float32)
    current_cable = current_state[:, :CABLE_DIM].copy()
    cable_delta = cable_future - current_cable[:, None, :]
    action_target = np.asarray(arrays["y_action"], dtype=np.float32)
    future_mask = np.asarray(arrays["future_valid_mask"], dtype=np.bool_)
    split = np.asarray(arrays["split_name"]).astype(str)
    train_mask = split == "train"
    train_full_mask = train_mask & np.all(future_mask, axis=1)

    if robot_future.shape != (EXPECTED_ROWS, DEFAULT_TF, ROBOT_PROXY_DIM):
        raise StageAContractError("robot-future shape changed")
    if not np.array_equal(cable_future, full_future[..., :CABLE_DIM]):
        raise StageAContractError("cable slice is not byte-exact")
    if np.shares_memory(cable_future, full_future):
        raise StageAContractError("cable target unexpectedly aliases source")
    if int(np.sum(train_full_mask)) < 2:
        raise StageAContractError("insufficient train full-horizon rows")

    idm_next = np.concatenate(
        [condition, cable_future[:, 0]],
        axis=1,
    ).astype(np.float32)
    idm_next_delta = np.concatenate(
        [condition, cable_delta[:, 0]],
        axis=1,
    ).astype(np.float32)
    idm_trajectory = np.concatenate(
        [condition, cable_future.reshape(EXPECTED_ROWS, -1)],
        axis=1,
    ).astype(np.float32)
    idm_delta_trajectory = np.concatenate(
        [condition, cable_delta.reshape(EXPECTED_ROWS, -1)],
        axis=1,
    ).astype(np.float32)

    contract_arrays: Dict[str, np.ndarray] = {
        "row_index": np.asarray(arrays["row_index"], dtype=np.int64).copy(),
        "diffusion_condition_x": condition.copy(),
        "diffusion_target_cable": cable_future,
        "diffusion_target_cable_delta": cable_delta.astype(np.float32),
        "diffusion_target_final_cable": cable_future[:, -1].copy(),
        "current_cable": current_cable,
        "future_valid_mask": future_mask.copy(),
        "action_target": action_target.copy(),
        "idm_history_x": condition.copy(),
        "idm_next_cable_x": idm_next,
        "idm_next_cable_delta_x": idm_next_delta,
        "idm_cable_trajectory_x": idm_trajectory,
        "idm_cable_delta_trajectory_x": idm_delta_trajectory,
        "train_row_index": np.flatnonzero(train_mask).astype(np.int64),
        "train_full_horizon_row_index":
            np.flatnonzero(train_full_mask).astype(np.int64),
    }
    for key in (
        "condition_name",
        "split_name",
        "visible_seed",
        "pair_group",
        "pair_key",
        "episode_group_key",
        "source_file",
        "source_pickle_sha256",
        "episode_index",
        "window_t",
        "pre_engagement",
        "state_history_valid_mask",
        "action_history_valid_mask",
    ):
        contract_arrays[key] = np.asarray(arrays[key]).copy()

    standardizers = {
        "condition": fit_standardizer(condition[train_mask]),
        "cable_future": fit_masked_cable_standardizer(
            cable_future[train_mask],
            future_mask[train_mask],
        ),
        "action": fit_standardizer(action_target[train_mask]),
        "idm_next_cable": fit_standardizer(idm_next[train_full_mask]),
        "idm_next_cable_delta": fit_standardizer(
            idm_next_delta[train_full_mask]
        ),
        "idm_cable_trajectory": fit_standardizer(
            idm_trajectory[train_full_mask]
        ),
        "idm_cable_delta_trajectory": fit_standardizer(
            idm_delta_trajectory[train_full_mask]
        ),
    }
    for prefix, standardizer in standardizers.items():
        contract_arrays.update(standardizer.arrays(prefix))

    train_keys = (
        "diffusion_condition_x",
        "diffusion_target_cable",
        "diffusion_target_cable_delta",
        "future_valid_mask",
        "action_target",
        "idm_history_x",
        "idm_next_cable_x",
        "idm_next_cable_delta_x",
        "idm_cable_trajectory_x",
        "idm_cable_delta_trajectory_x",
        "condition_name",
        "split_name",
        "visible_seed",
        "pair_group",
        "pair_key",
        "episode_group_key",
        "window_t",
    )
    train_view = {
        key: contract_arrays[key][train_full_mask].copy()
        for key in train_keys
    }
    train_view["source_row_index"] = np.flatnonzero(
        train_full_mask
    ).astype(np.int64)
    for prefix in standardizers:
        for suffix in ("mean", "scale", "active"):
            key = f"{prefix}_{suffix}"
            train_view[key] = contract_arrays[key].copy()

    model_input_keys = (
        "diffusion_condition_x",
        "idm_history_x",
        "idm_next_cable_x",
        "idm_next_cable_delta_x",
        "idm_cable_trajectory_x",
        "idm_cable_delta_trajectory_x",
    )
    forbidden_key_overlap = sorted(
        set(model_input_keys).intersection(FORBIDDEN_MODEL_METADATA_KEYS)
    )
    if forbidden_key_overlap:
        raise StageAContractError(
            f"metadata entered model key set: {forbidden_key_overlap}"
        )
    expected_shapes = {
        "diffusion_condition_x": (EXPECTED_ROWS, DIFFUSION_CONDITION_DIM),
        "diffusion_target_cable":
            (EXPECTED_ROWS, DEFAULT_TF, DIFFUSION_TARGET_DIM),
        "idm_history_x": (EXPECTED_ROWS, IDM_HISTORY_DIM),
        "idm_next_cable_x": (EXPECTED_ROWS, IDM_NEXT_CABLE_DIM),
        "idm_next_cable_delta_x":
            (EXPECTED_ROWS, IDM_NEXT_CABLE_DIM),
        "idm_cable_trajectory_x":
            (EXPECTED_ROWS, IDM_CABLE_TRAJECTORY_DIM),
        "idm_cable_delta_trajectory_x":
            (EXPECTED_ROWS, IDM_CABLE_TRAJECTORY_DIM),
    }
    for key, expected_shape in expected_shapes.items():
        if contract_arrays[key].shape != expected_shape:
            raise StageAContractError(
                f"{key} shape {contract_arrays[key].shape} != {expected_shape}"
            )
    for name, value in contract_arrays.items():
        if value.dtype.kind == "O":
            raise StageAContractError(f"object array in contract: {name}")
        if value.dtype.kind in "fc" and not np.all(np.isfinite(value)):
            raise StageAContractError(f"non-finite array in contract: {name}")
    validation.update(
        {
            "train_full_horizon_rows": int(np.sum(train_full_mask)),
            "cable_target_exact": True,
            "robot_future_excluded_from_diffusion_target": True,
            "robot_future_excluded_from_idm_inputs": True,
            "target_action_excluded_from_model_inputs": True,
            "metadata_excluded_from_model_inputs": True,
            "historical_stagec_idm_fields_reused": False,
        }
    )
    return contract_arrays, train_view, validation


def group_fold_assignment(
    groups: Sequence[Any],
    *,
    folds: int,
) -> Tuple[np.ndarray, Dict[str, int]]:
    values = np.asarray(groups).astype(str)
    unique = sorted(set(values.tolist()))
    if len(unique) < folds:
        raise StageAContractError(
            f"group count {len(unique)} < fold count {folds}"
        )
    mapping = {group: index % folds for index, group in enumerate(unique)}
    assignment = np.asarray(
        [mapping[group] for group in values],
        dtype=np.int64,
    )
    for group in unique:
        if len(set(assignment[values == group].tolist())) != 1:
            raise AssertionError("episode group crossed folds")
    return assignment, mapping


def ridge_fit_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    eval_x: np.ndarray,
    *,
    regularization: float,
) -> np.ndarray:
    x_train = np.asarray(train_x, dtype=np.float64)
    y_train = np.asarray(train_y, dtype=np.float64)
    x_eval = np.asarray(eval_x, dtype=np.float64)
    if (
        x_train.ndim != 2
        or y_train.ndim != 2
        or x_eval.ndim != 2
        or x_train.shape[0] != y_train.shape[0]
        or x_train.shape[1] != x_eval.shape[1]
    ):
        raise StageAContractError("ridge input shape mismatch")
    x_mean = np.mean(x_train, axis=0, keepdims=True)
    x_std = np.std(x_train, axis=0, keepdims=True)
    x_active = x_std[0] > 1.0e-12
    y_mean = np.mean(y_train, axis=0, keepdims=True)
    y_std = np.std(y_train, axis=0, keepdims=True)
    y_active = y_std[0] > 1.0e-12
    prediction = np.repeat(y_mean, x_eval.shape[0], axis=0)
    if not np.any(x_active) or not np.any(y_active):
        return prediction
    x_scale = np.where(x_std > 1.0e-12, x_std, 1.0)
    y_scale = np.where(y_std > 1.0e-12, y_std, 1.0)
    xz = (
        x_train[:, x_active] - x_mean[:, x_active]
    ) / x_scale[:, x_active]
    ez = (
        x_eval[:, x_active] - x_mean[:, x_active]
    ) / x_scale[:, x_active]
    yz = (
        y_train[:, y_active] - y_mean[:, y_active]
    ) / y_scale[:, y_active]
    gram = xz.T @ xz
    gram.flat[:: gram.shape[0] + 1] += float(regularization)
    weights = np.linalg.solve(gram, xz.T @ yz)
    prediction[:, y_active] = (
        ez @ weights
    ) * y_scale[:, y_active] + y_mean[:, y_active]
    return prediction


def grouped_oof_prediction(
    features: np.ndarray,
    target: np.ndarray,
    groups: Sequence[Any],
    *,
    spec: AuditSpec,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise StageAContractError("OOF feature/target mismatch")
    assignment, mapping = group_fold_assignment(groups, folds=spec.folds)
    prediction = np.full(y.shape, np.nan, dtype=np.float64)
    fold_records = []
    group_values = np.asarray(groups).astype(str)
    for fold in range(spec.folds):
        evaluation = assignment == fold
        training = assignment != fold
        if not np.any(evaluation) or int(np.sum(training)) < 2:
            raise StageAContractError(f"invalid fold {fold}")
        prediction[evaluation] = ridge_fit_predict(
            x[training],
            y[training],
            x[evaluation],
            regularization=spec.ridge_regularization,
        )
        fold_records.append(
            {
                "fold": fold,
                "training_rows": int(np.sum(training)),
                "evaluation_rows": int(np.sum(evaluation)),
                "evaluation_groups": len(
                    set(group_values[evaluation].tolist())
                ),
            }
        )
    if not np.all(np.isfinite(prediction)):
        raise StageAContractError("OOF prediction is incomplete")
    return prediction, {
        "folds": spec.folds,
        "group_count": len(mapping),
        "group_to_fold": mapping,
        "fold_records": fold_records,
        "group_integrity_pass": True,
    }


def action_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    variance_epsilon: float,
) -> Dict[str, Any]:
    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    variance = np.var(truth, axis=0)
    active = variance > float(variance_epsilon)
    if not np.any(active):
        raise StageAContractError("action target has no active dimensions")
    mse_per_dim = np.mean((pred - truth) ** 2, axis=0)
    normalized = mse_per_dim[active] / variance[active]
    return {
        "normalized_mse": float(np.mean(normalized)),
        "raw_mse": float(np.mean(mse_per_dim[active])),
        "active_action_dimensions": int(np.sum(active)),
        "active_mask": active.tolist(),
        "per_dimension_variance": variance.tolist(),
        "per_dimension_mse": mse_per_dim.tolist(),
    }


def _gain(baseline_nmse: float, candidate_nmse: float) -> float:
    denominator = max(abs(float(baseline_nmse)), 1.0e-12)
    return float((baseline_nmse - candidate_nmse) / denominator)


def deterministic_stratified_permutation(
    cable_features: np.ndarray,
    *,
    condition_name: Sequence[Any],
    window_t: Sequence[Any],
    episode_group_key: Sequence[Any],
    shift: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    features = np.asarray(cable_features)
    condition = np.asarray(condition_name).astype(str)
    time = np.asarray(window_t, dtype=np.int64)
    group = np.asarray(episode_group_key).astype(str)
    if (
        features.shape[0] != condition.shape[0]
        or condition.shape != time.shape
        or time.shape != group.shape
    ):
        raise StageAContractError("permutation metadata shape mismatch")
    result = np.empty_like(features)
    strata: MutableMapping[Tuple[str, int], List[int]] = {}
    for index, key in enumerate(zip(condition, time)):
        strata.setdefault((str(key[0]), int(key[1])), []).append(index)
    singleton = 0
    for key in sorted(strata):
        indices = sorted(strata[key], key=lambda index: group[index])
        if len(indices) <= 1:
            result[indices] = features[indices]
            singleton += len(indices)
            continue
        offset = int(shift) % len(indices)
        if offset == 0:
            offset = 1
        source = indices[offset:] + indices[:offset]
        result[np.asarray(indices)] = features[np.asarray(source)]
    return result, {
        "shift": int(shift),
        "strata": len(strata),
        "singleton_rows": singleton,
        "condition_preserved": True,
        "window_t_preserved": True,
    }


def paired_action_intervention_audit(
    pair_key: Sequence[Any],
    action_target: np.ndarray,
    cable_future: np.ndarray,
    *,
    epsilon: float,
) -> Dict[str, Any]:
    keys = np.asarray(pair_key).astype(str)
    action = np.asarray(action_target, dtype=np.float64)
    cable = np.asarray(cable_future, dtype=np.float64)
    groups: MutableMapping[str, List[int]] = {}
    for index, key in enumerate(keys):
        groups.setdefault(key, []).append(index)
    invalid = [key for key, indices in groups.items() if len(indices) != 2]
    if invalid:
        raise StageAContractError(
            f"pair keys without two rows: {invalid[:5]}"
        )
    action_distance = []
    cable_distance = []
    diverse = 0
    cable_divergent = 0
    for key in sorted(groups):
        left, right = groups[key]
        action_delta = float(np.linalg.norm(action[left] - action[right]))
        cable_delta = float(
            np.linalg.norm(cable[left].reshape(-1) - cable[right].reshape(-1))
        )
        action_distance.append(action_delta)
        cable_distance.append(cable_delta)
        if action_delta > float(epsilon):
            diverse += 1
        if cable_delta > float(epsilon):
            cable_divergent += 1
    total = len(groups)
    return {
        "pair_keys": total,
        "paired_action_diverse_keys": diverse,
        "paired_action_diverse_fraction": float(diverse / max(total, 1)),
        "paired_cable_divergent_keys": cable_divergent,
        "paired_cable_divergent_fraction": float(
            cable_divergent / max(total, 1)
        ),
        "action_distance_max": float(max(action_distance) if action_distance else 0.0),
        "action_distance_mean": float(
            np.mean(action_distance) if action_distance else 0.0
        ),
        "cable_distance_mean": float(
            np.mean(cable_distance) if cable_distance else 0.0
        ),
        "same_action_pair_contract": bool(
            max(action_distance) <= float(epsilon)
            if action_distance
            else False
        ),
        "supports_paired_action_intervention": bool(diverse > 0),
    }


def unique_action_count(action_target: np.ndarray) -> int:
    rounded = np.round(
        np.asarray(action_target, dtype=np.float64),
        decimals=6,
    )
    return int(np.unique(rounded, axis=0).shape[0])


def run_train_only_idm_audit(
    train_arrays: Mapping[str, np.ndarray],
    *,
    spec: Optional[AuditSpec] = None,
) -> Dict[str, Any]:
    active_spec = AuditSpec() if spec is None else spec
    active_spec.validate()
    action = np.asarray(train_arrays["action_target"], dtype=np.float64)
    groups = np.asarray(
        train_arrays["episode_group_key"]
    ).astype(str)
    condition = np.asarray(train_arrays["condition_name"]).astype(str)
    window_t = np.asarray(train_arrays["window_t"], dtype=np.int64)
    pair_key = np.asarray(train_arrays["pair_key"]).astype(str)
    cable_future = np.asarray(
        train_arrays["diffusion_target_cable"],
        dtype=np.float64,
    )
    feature_sets = {
        "history_only": np.asarray(
            train_arrays["idm_history_x"], dtype=np.float64
        ),
        "history_plus_next_cable": np.asarray(
            train_arrays["idm_next_cable_x"], dtype=np.float64
        ),
        "history_plus_next_cable_delta": np.asarray(
            train_arrays["idm_next_cable_delta_x"], dtype=np.float64
        ),
        "history_plus_cable_trajectory": np.asarray(
            train_arrays["idm_cable_trajectory_x"], dtype=np.float64
        ),
        "history_plus_cable_delta_trajectory": np.asarray(
            train_arrays["idm_cable_delta_trajectory_x"],
            dtype=np.float64,
        ),
    }
    metrics: Dict[str, Any] = {}
    fold_contract = None
    for name, features in feature_sets.items():
        prediction, contract = grouped_oof_prediction(
            features,
            action,
            groups,
            spec=active_spec,
        )
        metrics[name] = action_metrics(
            prediction,
            action,
            variance_epsilon=active_spec.variance_epsilon,
        )
        if fold_contract is None:
            fold_contract = contract

    baseline_nmse = metrics["history_only"]["normalized_mse"]
    cable_names = tuple(name for name in feature_sets if name != "history_only")
    best_name = min(
        cable_names,
        key=lambda name: metrics[name]["normalized_mse"],
    )
    best_nmse = float(metrics[best_name]["normalized_mse"])
    original_gain = _gain(baseline_nmse, best_nmse)

    # Negative controls use the same candidate layout but replace only its
    # cable suffix with a deterministic condition/time-stratified permutation.
    base_history = feature_sets["history_only"]
    permutation_records = []
    shuffled_gains = []
    for shift in active_spec.permutation_shifts:
        candidate_records = {}
        for name in cable_names:
            original = feature_sets[name]
            suffix = original[:, base_history.shape[1]:]
            shuffled_suffix, permutation_contract = (
                deterministic_stratified_permutation(
                    suffix,
                    condition_name=condition,
                    window_t=window_t,
                    episode_group_key=groups,
                    shift=shift,
                )
            )
            shuffled = np.concatenate(
                [base_history, shuffled_suffix],
                axis=1,
            )
            prediction, _ = grouped_oof_prediction(
                shuffled,
                action,
                groups,
                spec=active_spec,
            )
            shuffled_metrics = action_metrics(
                prediction,
                action,
                variance_epsilon=active_spec.variance_epsilon,
            )
            shuffled_gain = _gain(
                baseline_nmse,
                shuffled_metrics["normalized_mse"],
            )
            candidate_records[name] = {
                "normalized_mse": shuffled_metrics["normalized_mse"],
                "gain": shuffled_gain,
                "permutation_contract": permutation_contract,
            }
            shuffled_gains.append(shuffled_gain)
        permutation_records.append(
            {
                "shift": int(shift),
                "candidates": candidate_records,
            }
        )

    max_shuffled_gain = float(max(shuffled_gains))
    gain_over_permuted = float(original_gain - max_shuffled_gain)
    active_dims = int(
        metrics["history_only"]["active_action_dimensions"]
    )
    unique_actions = unique_action_count(action)
    paired = paired_action_intervention_audit(
        pair_key,
        action,
        cable_future,
        epsilon=active_spec.action_equality_epsilon,
    )
    predictive_support = bool(
        active_dims >= active_spec.minimum_active_action_dims
        and unique_actions >= active_spec.minimum_unique_action_vectors
        and original_gain >= active_spec.cable_incremental_gain_min
        and gain_over_permuted >= active_spec.gain_over_permuted_min
    )
    paired_intervention_support = bool(
        paired["paired_action_diverse_fraction"]
        >= active_spec.minimum_paired_action_diverse_fraction
    )
    formal_idm_data_ready = bool(
        predictive_support and paired_intervention_support
    )
    if not paired_intervention_support:
        root = (
            "phase314b_r256_stagea_cable_only_contract_supported_"
            "paired_action_intervention_missing"
        )
        next_path = (
            "RUN_CABLE_ONLY_DIFFUSION_DIAGNOSTIC_AND_COLLECT_"
            "ACTION_DIVERSE_IDM_DATA"
        )
    elif not predictive_support:
        root = (
            "phase314b_r256_stagea_cable_only_contract_supported_"
            "idm_predictive_support_insufficient"
        )
        next_path = (
            "RUN_CABLE_ONLY_DIFFUSION_DIAGNOSTIC_AND_REDESIGN_"
            "IDM_DATA_SUPPORT"
        )
    else:
        root = (
            "phase314b_r256_stagea_cable_only_and_deployable_idm_"
            "contracts_supported"
        )
        next_path = (
            "RUN_TRAIN_ONLY_CABLE_DIFFUSION_AND_DEPLOYABLE_IDM_DIAGNOSTICS"
        )
    return {
        "schema": "phase314b_r256_stagea_idm_identifiability_audit_v1",
        "spec": asdict(active_spec),
        "train_full_horizon_rows": int(action.shape[0]),
        "paired_episode_groups": len(set(groups.tolist())),
        "feature_metrics": metrics,
        "best_cable_feature": best_name,
        "baseline_nmse": float(baseline_nmse),
        "best_cable_nmse": best_nmse,
        "cable_incremental_gain": original_gain,
        "negative_controls": permutation_records,
        "maximum_permuted_gain": max_shuffled_gain,
        "gain_over_permuted": gain_over_permuted,
        "active_action_dimensions": active_dims,
        "unique_action_vectors_rounded_1e6": unique_actions,
        "paired_action_intervention": paired,
        "predictive_support": predictive_support,
        "paired_intervention_support": paired_intervention_support,
        "formal_idm_data_ready": formal_idm_data_ready,
        "fold_contract": fold_contract,
        "root_cause": root,
        "required_next_path": next_path,
        "future_robot_used": False,
        "target_action_used_as_input": False,
        "metadata_used_as_input": False,
        "diagnostic_only": True,
        "formal_idm_training": False,
    }


def build_artifacts(
    *,
    root: Path,
    output_root: Path,
) -> Dict[str, Any]:
    repository_root = Path(root).resolve()
    destination = Path(output_root).resolve()
    if destination.exists():
        raise FileExistsError(f"worker output exists: {destination}")
    destination.mkdir(parents=True, exist_ok=False)
    immutable = validate_immutable_inputs(repository_root)
    source_audit = source_contract_audit(repository_root)
    if not source_audit["all_expected_legacy_bindings_confirmed"]:
        raise StageAContractError("historical source assumptions changed")
    source_cache = load_npz_strict(repository_root / STAGEC_CACHE)
    contract_arrays, train_view, validation = build_contract_arrays(
        source_cache
    )
    audit = run_train_only_idm_audit(train_view)

    contract_path = destination / CONTRACT_FILE
    train_view_path = destination / TRAIN_VIEW_FILE
    manifest_path = destination / MANIFEST_FILE
    write_npz_once(contract_path, contract_arrays)
    write_npz_once(train_view_path, train_view)
    reloaded_contract = load_npz_strict(contract_path)
    reloaded_train = load_npz_strict(train_view_path)
    if set(reloaded_contract) != set(contract_arrays):
        raise StageAContractError("contract key set changed after serialization")
    if set(reloaded_train) != set(train_view):
        raise StageAContractError("train-view key set changed after serialization")
    for key in contract_arrays:
        if not np.array_equal(reloaded_contract[key], contract_arrays[key]):
            raise StageAContractError(
                f"contract array changed after serialization: {key}"
            )
    for key in train_view:
        if not np.array_equal(reloaded_train[key], train_view[key]):
            raise StageAContractError(
                f"train-view array changed after serialization: {key}"
            )

    manifest = {
        "phase": PHASE,
        "phase_id": PHASE_ID,
        "schema": "phase314b_r256_stagea_manifest_v1",
        "contract_version": CONTRACT_VERSION,
        "base_evidence_commit": BASE_EVIDENCE_COMMIT,
        "source_stagec_cache": STAGEC_CACHE,
        "source_stagec_cache_sha256": EXPECTED_STAGEC_CACHE_SHA256,
        "source_stagec_train_view_sha256":
            EXPECTED_STAGEC_TRAIN_VIEW_SHA256,
        "source_stagec_manifest_sha256":
            EXPECTED_STAGEC_CACHE_MANIFEST_SHA256,
        "contract_file": CONTRACT_FILE,
        "contract_sha256": sha256_file(contract_path),
        "train_view_file": TRAIN_VIEW_FILE,
        "train_view_sha256": sha256_file(train_view_path),
        "validation": validation,
        "source_contract_audit": source_audit,
        "idm_identifiability_audit": audit,
        "array_sha256": {
            key: sha256_array(value)
            for key, value in contract_arrays.items()
        },
        "train_view_array_sha256": {
            key: sha256_array(value)
            for key, value in train_view.items()
        },
        "diffusion_contract": {
            "condition": "state_v3_history_plus_past_action_history",
            "condition_dim": DIFFUSION_CONDITION_DIM,
            "target": "ordered_cable_xy_future_only",
            "target_shape": [DEFAULT_TF, DIFFUSION_TARGET_DIM],
            "robot_history_retained_in_condition": True,
            "robot_future_removed_from_target": True,
        },
        "idm_contract_candidates": {
            "history_only_dim": IDM_HISTORY_DIM,
            "next_cable_dim": IDM_NEXT_CABLE_DIM,
            "cable_trajectory_dim": IDM_CABLE_TRAJECTORY_DIM,
            "robot_future_removed": True,
            "current_action_removed": True,
        },
        "immutable_inputs": immutable["file_sha256"],
        "write_once": True,
        "deterministic_npz": True,
        "legacy_artifacts_modified": False,
        "diffusion_training": False,
        "reverse_sampling": False,
        "formal_idm_training": False,
        "candidate_execution": False,
        "phase4": False,
        "cps": False,
    }
    atomic_write_once(manifest_path, stable_json_bytes(manifest))
    return manifest


def directory_manifest(root: Path) -> Dict[str, str]:
    base = Path(root)
    return {
        path.relative_to(base).as_posix(): sha256_file(path)
        for path in sorted(base.rglob("*"))
        if path.is_file()
    }


def compare_directories(left: Path, right: Path) -> Dict[str, Any]:
    left_manifest = directory_manifest(left)
    right_manifest = directory_manifest(right)
    return {
        "exact": left_manifest == right_manifest,
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
