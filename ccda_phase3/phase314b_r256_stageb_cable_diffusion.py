"""Phase3.14b-r2.5.6 Stage B cable-only diffusion diagnostic.

The implementation is additive and independent of the historical 87-dimensional
state-v2 diffusion modules.  It trains only on the write-once r2.5.6 Stage-A
train/full-horizon view:

    condition: state-v3 history + past action history, [N, 243]
    target: ordered cable XY future only, [N, 4, 48]

No validation or formal-test target is opened.  No checkpoint or prediction
tensor is persisted.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

PHASE = "Phase3.14b-r2.5.6 Stage B"
PHASE_ID = "phase314b_r256_stageb"
BASE_EVIDENCE_COMMIT = "7ab3779a2c9be066916615051cd9c4313393939d"
EXPECTED_SUBMODULE_COMMIT = "633a88752445cf5d6776ed374fdbbdb35f93050c"

EXPECTED_STAGEA_CONTRACT_SHA256 = (
    "3bdcb24fc0bee2846d50c310e111a7c2c7f5dc6b7cf45391b7e5b7386a1aa272"
)
EXPECTED_STAGEA_TRAIN_VIEW_SHA256 = (
    "7acac27de1ced12c038d8cef3bd588c94b6ab2300bcd6d8e1f28492e73275633"
)
EXPECTED_STAGEA_MANIFEST_SHA256 = (
    "1ed831fbbf5915f2e8264695d2c3845aab1bf8cb7abd98f571320dabc6255015"
)
EXPECTED_STAGEA_SUMMARY_SHA256 = (
    "baa3760f75d640c64d433e63c6698abb1791788f142cb1e2857fdbe3bc3b1ccb"
)
EXPECTED_STAGEA_REPORT_SHA256 = (
    "8e82a7c1e0d52e75f50eeecd09b945f6d1a3ccad3d402813cd181e78f32cd745"
)

STAGEA_ROOT = "data/phase3_14_cache_v3_r256_stagea"
STAGEA_CONTRACT = (
    f"{STAGEA_ROOT}/phase3_14b_r256_stagea_cable_idm_contract.npz"
)
STAGEA_TRAIN_VIEW = (
    f"{STAGEA_ROOT}/phase3_14b_r256_stagea_train_audit_view.npz"
)
STAGEA_MANIFEST = f"{STAGEA_ROOT}/phase3_14b_r256_stagea_manifest.json"

CONDITION_DIM = 243
CABLE_DIM = 48
FUTURE_STEPS = 4
TARGET_FLAT_DIM = FUTURE_STEPS * CABLE_DIM
STATE_DIM = 67
HISTORY_STEPS = 3
BEADS = 24
ACTION_DIM = 14

REQUIRED_TRAIN_KEYS = (
    "diffusion_condition_x",
    "diffusion_target_cable",
    "diffusion_target_cable_delta",
    "future_valid_mask",
    "condition_name",
    "pair_key",
    "episode_group_key",
    "visible_seed",
    "window_t",
    "split_name",
)

HISTORICAL_SOURCE_FILES = (
    "ccda_phase3/phase314b_models.py",
    "ccda_phase3/phase314b_diffusion.py",
    "ccda_phase3/phase314b_r2_diffusion.py",
)


class CableDiffusionDiagnosticError(RuntimeError):
    """Raised when the cable-only diagnostic contract is violated."""


@dataclass(frozen=True)
class DiagnosticSpec:
    train_timesteps: int = 100
    train_steps: int = 8000
    batch_size: int = 64
    learning_rate: float = 1.0e-3
    weight_decay: float = 0.0
    width: int = 512
    residual_blocks: int = 4
    time_embedding_dim: int = 32
    seed: int = 104000
    probe_fold: int = 0
    group_folds: int = 8
    one_step_timesteps: Tuple[int, ...] = (10, 25, 50)
    reverse_candidates: int = 8
    reverse_steps: int = 100
    train_control_rows: int = 64
    train_control_nmse_max: float = 0.10
    probe_improvement_min: float = 0.05
    branch_support_min: float = 0.75
    physical_candidate_rate_min: float = 0.95
    beta_start: float = 1.0e-4
    beta_end: float = 2.0e-2

    def validate(self) -> None:
        if self.train_timesteps != 100 or self.reverse_steps != 100:
            raise ValueError("diagnostic freezes a 100-step diffusion process")
        if self.train_steps <= 0 or self.batch_size <= 0:
            raise ValueError("training settings must be positive")
        if self.width <= 0 or self.residual_blocks <= 0:
            raise ValueError("model capacity must be positive")
        if not 0 <= self.probe_fold < self.group_folds:
            raise ValueError("probe fold is outside grouped split")
        if self.reverse_candidates <= 0:
            raise ValueError("reverse candidate count must be positive")
        if tuple(sorted(set(self.one_step_timesteps))) != self.one_step_timesteps:
            raise ValueError("one-step timesteps must be ordered and unique")
        if self.one_step_timesteps[-1] >= self.train_timesteps:
            raise ValueError("one-step timestep is outside scheduler")


@dataclass(frozen=True)
class ArrayStandardizer:
    mean: np.ndarray
    scale: np.ndarray
    active: np.ndarray

    def validate(self, shape: Tuple[int, ...]) -> None:
        if self.mean.shape != shape:
            raise ValueError(f"mean shape {self.mean.shape} != {shape}")
        if self.scale.shape != shape or self.active.shape != shape:
            raise ValueError("standardizer shape mismatch")
        if self.mean.dtype != np.float32 or self.scale.dtype != np.float32:
            raise ValueError("standardizer mean/scale must be float32")
        if self.active.dtype != np.bool_:
            raise ValueError("standardizer active mask must be bool")
        if not np.all(np.isfinite(self.mean)):
            raise ValueError("standardizer mean is non-finite")
        if not np.all(np.isfinite(self.scale)) or np.any(self.scale <= 0.0):
            raise ValueError("standardizer scale is invalid")
        if not np.all(self.scale[~self.active] == 1.0):
            raise ValueError("inactive dimensions must have scale one")

    def normalize(self, value: np.ndarray) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32)
        return ((array - self.mean) / self.scale).astype(np.float32)

    def denormalize(self, value: np.ndarray) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32)
        return (array * self.scale + self.mean).astype(np.float32)


@dataclass(frozen=True)
class GeometryContract:
    segment_lower: np.ndarray
    segment_upper: np.ndarray
    coordinate_abs_max: float
    target_intersection_max: int

    def validate(self) -> None:
        if self.segment_lower.shape != (FUTURE_STEPS, BEADS - 1):
            raise ValueError("segment lower-bound shape mismatch")
        if self.segment_upper.shape != self.segment_lower.shape:
            raise ValueError("segment upper-bound shape mismatch")
        if not np.all(np.isfinite(self.segment_lower)):
            raise ValueError("segment lower bounds are non-finite")
        if not np.all(np.isfinite(self.segment_upper)):
            raise ValueError("segment upper bounds are non-finite")
        if np.any(self.segment_lower <= 0.0):
            raise ValueError("segment lower bounds must be positive")
        if np.any(self.segment_upper <= self.segment_lower):
            raise ValueError("segment upper bounds are invalid")
        if not np.isfinite(self.coordinate_abs_max) or self.coordinate_abs_max <= 0.0:
            raise ValueError("coordinate bound is invalid")
        if self.target_intersection_max < 0:
            raise ValueError("intersection bound is invalid")


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


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


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


def load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CableDiffusionDiagnosticError(
            f"JSON root is not an object: {path}"
        )
    return value


def load_npz_strict(
    path: Path,
    *,
    required_keys: Sequence[str],
) -> Dict[str, np.ndarray]:
    result: Dict[str, np.ndarray] = {}
    with np.load(Path(path), allow_pickle=False) as archive:
        available = set(archive.files)
        missing = sorted(set(required_keys) - available)
        if missing:
            raise CableDiffusionDiagnosticError(
                f"NPZ missing keys: {missing}"
            )
        for key in required_keys:
            value = np.asarray(archive[key])
            if value.dtype.kind == "O":
                raise CableDiffusionDiagnosticError(
                    f"object array is forbidden: {key}"
                )
            if value.dtype.kind in "fc" and not np.all(np.isfinite(value)):
                raise CableDiffusionDiagnosticError(
                    f"non-finite array: {key}"
                )
            result[key] = value.copy()
    return result


def validate_immutable_inputs(root: Path) -> Dict[str, Any]:
    expected = {
        STAGEA_CONTRACT: EXPECTED_STAGEA_CONTRACT_SHA256,
        STAGEA_TRAIN_VIEW: EXPECTED_STAGEA_TRAIN_VIEW_SHA256,
        STAGEA_MANIFEST: EXPECTED_STAGEA_MANIFEST_SHA256,
        "reports/phase3_14b_r256_stagea_summary.json":
            EXPECTED_STAGEA_SUMMARY_SHA256,
        "reports/phase3_14b_r256_stagea_report.md":
            EXPECTED_STAGEA_REPORT_SHA256,
    }
    for relative, expected_sha in expected.items():
        path = Path(root) / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = sha256_file(path)
        if observed != expected_sha:
            raise CableDiffusionDiagnosticError(
                f"immutable input changed: {relative}: {observed}"
            )
    summary = load_json(
        Path(root) / "reports/phase3_14b_r256_stagea_summary.json"
    )
    if summary.get("verdict") != "PASS":
        raise CableDiffusionDiagnosticError(
            "r2.5.6 Stage-A verdict is not PASS"
        )
    if summary.get("scientific_status") != "BLOCKED":
        raise CableDiffusionDiagnosticError(
            "r2.5.6 Stage-A scientific boundary changed"
        )
    if summary.get("required_next_path") != (
        "RUN_CABLE_ONLY_DIFFUSION_DIAGNOSTIC_AND_COLLECT_ACTION_DIVERSE_IDM_DATA"
    ):
        raise CableDiffusionDiagnosticError(
            "r2.5.6 Stage-A next path changed"
        )
    if summary.get("formal_idm_data_ready") is not False:
        raise CableDiffusionDiagnosticError(
            "formal IDM data-readiness boundary changed"
        )
    return {"file_sha256": expected, "summary": summary}


def source_contract_audit(root: Path) -> Dict[str, Any]:
    texts = {
        relative: (Path(root) / relative).read_text(encoding="utf-8")
        for relative in HISTORICAL_SOURCE_FILES
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
        "historical_r2_mentions_4_by_87": (
            "[4,87]" in texts["ccda_phase3/phase314b_r2_diffusion.py"]
        ),
    }
    return {
        "checks": checks,
        "all_confirmed": bool(all(checks.values())),
        "source_sha256": {
            relative: sha256_file(Path(root) / relative)
            for relative in HISTORICAL_SOURCE_FILES
        },
    }


def validate_train_view(arrays: Mapping[str, np.ndarray]) -> Dict[str, Any]:
    missing = sorted(set(REQUIRED_TRAIN_KEYS) - set(arrays))
    if missing:
        raise CableDiffusionDiagnosticError(
            f"train view missing keys: {missing}"
        )
    condition = np.asarray(arrays["diffusion_condition_x"])
    target = np.asarray(arrays["diffusion_target_cable"])
    target_delta = np.asarray(arrays["diffusion_target_cable_delta"])
    valid = np.asarray(arrays["future_valid_mask"])
    rows = condition.shape[0]
    if condition.shape != (rows, CONDITION_DIM):
        raise CableDiffusionDiagnosticError(
            f"condition shape changed: {condition.shape}"
        )
    if target.shape != (rows, FUTURE_STEPS, CABLE_DIM):
        raise CableDiffusionDiagnosticError(
            f"target shape changed: {target.shape}"
        )
    if target_delta.shape != target.shape:
        raise CableDiffusionDiagnosticError(
            "cable-delta target shape changed"
        )
    if valid.shape != (rows, FUTURE_STEPS) or valid.dtype != np.bool_:
        raise CableDiffusionDiagnosticError(
            "future-valid mask shape/dtype changed"
        )
    if not np.all(valid):
        raise CableDiffusionDiagnosticError(
            "Stage-B diagnostic requires the full-horizon train view"
        )
    if condition.dtype != np.float32 or target.dtype != np.float32:
        raise CableDiffusionDiagnosticError(
            "condition and target must be float32"
        )
    split_values = np.asarray(arrays.get("split_name", np.asarray(["train"]))).astype(str)
    if set(split_values.tolist()) != {"train"}:
        raise CableDiffusionDiagnosticError(
            "train-only view contains non-train rows"
        )
    pair_key = np.asarray(arrays["pair_key"]).astype(str)
    condition_name = np.asarray(arrays["condition_name"]).astype(str)
    pair_members: MutableMapping[str, List[str]] = {}
    for key, name in zip(pair_key, condition_name):
        pair_members.setdefault(key, []).append(name)
    expected_conditions = {"free", "hidden_slack_breakaway_pin_v2"}
    invalid = {
        key: sorted(value)
        for key, value in pair_members.items()
        if set(value) != expected_conditions or len(value) != 2
    }
    if invalid:
        raise CableDiffusionDiagnosticError(
            f"paired train view is incomplete: {list(invalid.items())[:5]}"
        )
    return {
        "rows": rows,
        "pair_keys": len(pair_members),
        "condition_dim": CONDITION_DIM,
        "target_shape": [FUTURE_STEPS, CABLE_DIM],
        "robot_future_in_target": False,
        "all_horizons_valid": True,
        "pair_contract_pass": True,
    }


def deterministic_group_split(
    groups: Sequence[Any],
    *,
    folds: int,
    probe_fold: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    values = np.asarray(groups).astype(str)
    unique = sorted(set(values.tolist()))
    if len(unique) < folds:
        raise CableDiffusionDiagnosticError(
            f"group count {len(unique)} is smaller than {folds}"
        )
    mapping = {group: index % folds for index, group in enumerate(unique)}
    assignment = np.asarray([mapping[group] for group in values], dtype=np.int64)
    train_mask = assignment != int(probe_fold)
    probe_mask = assignment == int(probe_fold)
    if not np.any(train_mask) or not np.any(probe_mask):
        raise CableDiffusionDiagnosticError(
            "grouped diagnostic split is empty"
        )
    for group in unique:
        observed = set(assignment[values == group].tolist())
        if len(observed) != 1:
            raise AssertionError("an episode group crossed diagnostic split")
    return train_mask, probe_mask, mapping


def fit_standardizer(
    value: np.ndarray,
    *,
    epsilon: float = 1.0e-12,
) -> ArrayStandardizer:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim < 2 or array.shape[0] < 2:
        raise ValueError("standardizer requires at least two rows")
    if not np.all(np.isfinite(array)):
        raise ValueError("standardizer input is non-finite")
    mean = np.mean(array, axis=0)
    std = np.std(array, axis=0)
    active = std > float(epsilon)
    scale = np.where(active, std, 1.0)
    result = ArrayStandardizer(
        mean=mean.astype(np.float32),
        scale=scale.astype(np.float32),
        active=active.astype(np.bool_),
    )
    result.validate(array.shape[1:])
    return result


def linear_beta_schedule(
    timesteps: int,
    *,
    beta_start: float,
    beta_end: float,
) -> np.ndarray:
    if timesteps <= 1:
        raise ValueError("scheduler requires more than one timestep")
    betas = np.linspace(
        float(beta_start),
        float(beta_end),
        int(timesteps),
        dtype=np.float64,
    )
    if np.any(betas <= 0.0) or np.any(betas >= 1.0):
        raise ValueError("beta schedule is outside (0,1)")
    return betas


def scheduler_arrays(spec: DiagnosticSpec) -> Dict[str, np.ndarray]:
    betas = linear_beta_schedule(
        spec.train_timesteps,
        beta_start=spec.beta_start,
        beta_end=spec.beta_end,
    )
    alphas = 1.0 - betas
    alpha_bar = np.cumprod(alphas)
    return {
        "betas": betas.astype(np.float32),
        "alphas": alphas.astype(np.float32),
        "alpha_bar": alpha_bar.astype(np.float32),
        "sqrt_alpha_bar": np.sqrt(alpha_bar).astype(np.float32),
        "sqrt_one_minus_alpha_bar":
            np.sqrt(1.0 - alpha_bar).astype(np.float32),
    }


def q_sample_numpy(
    x0: np.ndarray,
    noise: np.ndarray,
    timestep: np.ndarray,
    alpha_bar: np.ndarray,
) -> np.ndarray:
    clean = np.asarray(x0, dtype=np.float32)
    epsilon = np.asarray(noise, dtype=np.float32)
    t = np.asarray(timestep, dtype=np.int64)
    if clean.shape != epsilon.shape:
        raise ValueError("clean/noise shape mismatch")
    if t.shape != (clean.shape[0],):
        raise ValueError("timestep batch shape mismatch")
    sqrt_alpha = np.sqrt(alpha_bar[t]).astype(np.float32)
    sqrt_one_minus = np.sqrt(1.0 - alpha_bar[t]).astype(np.float32)
    expand = (slice(None),) + (None,) * (clean.ndim - 1)
    return (
        sqrt_alpha[expand] * clean
        + sqrt_one_minus[expand] * epsilon
    ).astype(np.float32)


def current_cable_from_condition(condition: np.ndarray) -> np.ndarray:
    value = np.asarray(condition, dtype=np.float32)
    if value.ndim != 2 or value.shape[1] != CONDITION_DIM:
        raise ValueError("condition must be [N,243]")
    state_history = value[:, : HISTORY_STEPS * STATE_DIM].reshape(
        value.shape[0],
        HISTORY_STEPS,
        STATE_DIM,
    )
    return state_history[:, -1, :CABLE_DIM].copy()


def cable_history_from_condition(condition: np.ndarray) -> np.ndarray:
    value = np.asarray(condition, dtype=np.float32)
    state_history = value[:, : HISTORY_STEPS * STATE_DIM].reshape(
        value.shape[0],
        HISTORY_STEPS,
        STATE_DIM,
    )
    return state_history[..., :CABLE_DIM].copy()


def last_state_baseline(condition: np.ndarray) -> np.ndarray:
    current = current_cable_from_condition(condition)
    return np.repeat(current[:, None, :], FUTURE_STEPS, axis=1)


def constant_velocity_baseline(condition: np.ndarray) -> np.ndarray:
    history = cable_history_from_condition(condition)
    current = history[:, -1]
    delta = history[:, -1] - history[:, -2]
    result = np.empty(
        (history.shape[0], FUTURE_STEPS, CABLE_DIM),
        dtype=np.float32,
    )
    for horizon in range(FUTURE_STEPS):
        result[:, horizon] = current + float(horizon + 1) * delta
    return result


def cable_points(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape[-1] != CABLE_DIM:
        raise ValueError("cable last dimension must be 48")
    return array.reshape(*array.shape[:-1], BEADS, 2)


def segment_lengths(value: np.ndarray) -> np.ndarray:
    points = cable_points(value)
    delta = points[..., 1:, :] - points[..., :-1, :]
    return np.linalg.norm(delta.astype(np.float64), axis=-1)


def _orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float(
        (b[0] - a[0]) * (c[1] - a[1])
        - (b[1] - a[1]) * (c[0] - a[0])
    )


def _strict_segment_intersection(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    *,
    epsilon: float = 1.0e-12,
) -> bool:
    o1 = _orientation(a, b, c)
    o2 = _orientation(a, b, d)
    o3 = _orientation(c, d, a)
    o4 = _orientation(c, d, b)
    return bool(
        o1 * o2 < -epsilon
        and o3 * o4 < -epsilon
    )


def self_intersection_count_single(cable: np.ndarray) -> int:
    points = cable_points(np.asarray(cable)).reshape(BEADS, 2)
    count = 0
    for left in range(BEADS - 1):
        for right in range(left + 2, BEADS - 1):
            if right == left + 1:
                continue
            if _strict_segment_intersection(
                points[left],
                points[left + 1],
                points[right],
                points[right + 1],
            ):
                count += 1
    return count


def fit_geometry_contract(target: np.ndarray) -> GeometryContract:
    future = np.asarray(target, dtype=np.float32)
    if future.ndim != 3 or future.shape[1:] != (FUTURE_STEPS, CABLE_DIM):
        raise ValueError("geometry target must be [N,4,48]")
    lengths = segment_lengths(future)
    lower = np.percentile(lengths, 0.5, axis=0)
    upper = np.percentile(lengths, 99.5, axis=0)
    lower = np.maximum(lower * 0.5, 1.0e-6)
    upper = np.maximum(upper * 1.5, lower + 1.0e-6)
    coordinate_abs_max = float(np.max(np.abs(future))) * 1.5 + 1.0e-6
    intersections = [
        self_intersection_count_single(future[row, horizon])
        for row in range(future.shape[0])
        for horizon in range(FUTURE_STEPS)
    ]
    result = GeometryContract(
        segment_lower=lower.astype(np.float32),
        segment_upper=upper.astype(np.float32),
        coordinate_abs_max=coordinate_abs_max,
        target_intersection_max=int(max(intersections) if intersections else 0),
    )
    result.validate()
    return result


def physical_validity(
    candidates: np.ndarray,
    contract: GeometryContract,
) -> Dict[str, Any]:
    contract.validate()
    value = np.asarray(candidates, dtype=np.float32)
    if value.ndim != 4 or value.shape[2:] != (FUTURE_STEPS, CABLE_DIM):
        raise ValueError("candidates must be [N,K,4,48]")
    finite = np.all(np.isfinite(value), axis=(2, 3))
    coordinate = np.max(np.abs(value), axis=(2, 3)) <= contract.coordinate_abs_max
    lengths = segment_lengths(value)
    segment = np.all(
        (lengths >= contract.segment_lower[None, None])
        & (lengths <= contract.segment_upper[None, None]),
        axis=(2, 3),
    )
    intersections = np.zeros(value.shape[:2], dtype=np.int64)
    topology = np.ones(value.shape[:2], dtype=np.bool_)
    for row in range(value.shape[0]):
        for candidate in range(value.shape[1]):
            maximum = 0
            for horizon in range(FUTURE_STEPS):
                maximum = max(
                    maximum,
                    self_intersection_count_single(
                        value[row, candidate, horizon]
                    ),
                )
            intersections[row, candidate] = maximum
            topology[row, candidate] = (
                maximum <= contract.target_intersection_max + 1
            )
    valid = finite & coordinate & segment & topology
    return {
        "valid": valid,
        "finite": finite,
        "coordinate": coordinate,
        "segment": segment,
        "topology": topology,
        "intersection_max": intersections,
        "candidate_rate": float(np.mean(valid)),
        "row_any_rate": float(np.mean(np.any(valid, axis=1))),
    }


def normalized_mse(
    prediction: np.ndarray,
    target: np.ndarray,
    standardizer: ArrayStandardizer,
) -> float:
    pred = standardizer.normalize(prediction)
    truth = standardizer.normalize(target)
    active = standardizer.active
    squared = (pred - truth) ** 2
    active_flat = active.reshape(-1)
    if not np.any(active_flat):
        return 0.0
    squared_flat = squared.reshape(*squared.shape[:-2], -1)
    return float(np.mean(squared_flat[..., active_flat]))


def rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(
        target,
        dtype=np.float64,
    )
    return float(np.sqrt(np.mean(error ** 2)))


def best_of_k_nmse(
    candidates: np.ndarray,
    target: np.ndarray,
    standardizer: ArrayStandardizer,
) -> Tuple[float, np.ndarray]:
    pred = standardizer.normalize(candidates)
    truth = standardizer.normalize(target)[:, None]
    active = standardizer.active
    squared = (pred - truth) ** 2
    active_flat = active.reshape(-1)
    if not np.any(active_flat):
        raise ValueError("target standardizer has no active dimensions")
    squared_flat = squared.reshape(
        squared.shape[0],
        squared.shape[1],
        -1,
    )
    per_candidate = np.mean(
        squared_flat[..., active_flat],
        axis=-1,
    )
    best_index = np.argmin(per_candidate, axis=1)
    best = candidates[np.arange(candidates.shape[0]), best_index]
    return normalized_mse(best, target, standardizer), best_index


def branch_metrics(
    *,
    pair_key: Sequence[Any],
    condition_name: Sequence[Any],
    target: np.ndarray,
    candidates: np.ndarray,
) -> Dict[str, Any]:
    keys = np.asarray(pair_key).astype(str)
    names = np.asarray(condition_name).astype(str)
    truth = np.asarray(target, dtype=np.float64)
    pred = np.asarray(candidates, dtype=np.float64)
    groups: MutableMapping[str, List[int]] = {}
    for index, key in enumerate(keys):
        groups.setdefault(key, []).append(index)
    support = []
    mean_correct = []
    pair_records = []
    for key in sorted(groups):
        indices = groups[key]
        if len(indices) != 2:
            raise CableDiffusionDiagnosticError(
                f"probe pair {key} does not have two rows"
            )
        left, right = indices
        if names[left] == names[right]:
            raise CableDiffusionDiagnosticError(
                f"probe pair {key} has duplicate condition"
            )
        pair_support = []
        pair_mean = []
        for own, other in ((left, right), (right, left)):
            own_error = np.mean(
                (pred[own] - truth[own][None]) ** 2,
                axis=(1, 2),
            )
            other_error = np.mean(
                (pred[own] - truth[other][None]) ** 2,
                axis=(1, 2),
            )
            pair_support.append(bool(np.any(own_error < other_error)))
            mean_prediction = np.mean(pred[own], axis=0)
            mean_own = float(np.mean((mean_prediction - truth[own]) ** 2))
            mean_other = float(np.mean((mean_prediction - truth[other]) ** 2))
            pair_mean.append(bool(mean_own < mean_other))
        support.extend(pair_support)
        mean_correct.extend(pair_mean)
        pair_records.append(
            {
                "pair_key": key,
                "conditions": [names[left], names[right]],
                "own_branch_support": pair_support,
                "mean_prediction_correct": pair_mean,
            }
        )
    return {
        "pair_count": len(groups),
        "row_own_branch_support_rate": float(np.mean(support)),
        "row_mean_prediction_correct_rate": float(np.mean(mean_correct)),
        "pairs": pair_records,
    }


def _torch_imports() -> Tuple[Any, Any]:
    import torch
    import torch.nn as nn
    return torch, nn


def set_deterministic_runtime(seed: int) -> Dict[str, Any]:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch, _ = _torch_imports()
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if not torch.cuda.is_available():
        raise CableDiffusionDiagnosticError(
            "cable diffusion diagnostic requires CUDA"
        )
    torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    return {
        "python_seed": int(seed),
        "numpy_seed": int(seed),
        "torch_seed": int(seed),
        "cuda_device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "deterministic_algorithms": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "cublas_workspace_config":
            os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def sinusoidal_timestep_embedding(
    timestep: Any,
    dimension: int,
) -> Any:
    torch, _ = _torch_imports()
    if dimension % 2 != 0:
        raise ValueError("time embedding dimension must be even")
    half = dimension // 2
    exponent = torch.arange(
        half,
        device=timestep.device,
        dtype=torch.float32,
    )
    exponent = -math.log(10000.0) * exponent / max(half - 1, 1)
    frequencies = torch.exp(exponent)
    angles = timestep.float()[:, None] * frequencies[None]
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)


def make_model(spec: DiagnosticSpec) -> Any:
    torch, nn = _torch_imports()

    class ResidualBlock(nn.Module):
        def __init__(self, width: int):
            super().__init__()
            self.norm = nn.LayerNorm(width)
            self.fc1 = nn.Linear(width, width * 2)
            self.fc2 = nn.Linear(width * 2, width)
            self.activation = nn.SiLU()

        def forward(self, value: Any) -> Any:
            residual = value
            value = self.norm(value)
            value = self.activation(self.fc1(value))
            value = self.fc2(value)
            return residual + value

    class CableX0Denoiser(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.noisy_projection = nn.Linear(TARGET_FLAT_DIM, spec.width)
            self.condition_projection = nn.Linear(CONDITION_DIM, spec.width)
            self.time_projection = nn.Sequential(
                nn.Linear(spec.time_embedding_dim, spec.width),
                nn.SiLU(),
                nn.Linear(spec.width, spec.width),
            )
            self.blocks = nn.ModuleList(
                [ResidualBlock(spec.width) for _ in range(spec.residual_blocks)]
            )
            self.output_norm = nn.LayerNorm(spec.width)
            self.output = nn.Linear(spec.width, TARGET_FLAT_DIM)

        def forward(
            self,
            noisy: Any,
            timestep: Any,
            condition: Any,
        ) -> Any:
            flat = noisy.reshape(noisy.shape[0], TARGET_FLAT_DIM)
            time = sinusoidal_timestep_embedding(
                timestep,
                spec.time_embedding_dim,
            )
            hidden = (
                self.noisy_projection(flat)
                + self.condition_projection(condition)
                + self.time_projection(time)
            )
            for block in self.blocks:
                hidden = block(hidden)
            predicted = flat + self.output(self.output_norm(hidden))
            return predicted.reshape(
                noisy.shape[0],
                FUTURE_STEPS,
                CABLE_DIM,
            )

    return CableX0Denoiser()


def tensor_state_sha256(model: Any) -> str:
    torch, _ = _torch_imports()
    digest = hashlib.sha256()
    state = model.state_dict()
    for key in sorted(state):
        value = state[key].detach().to(
            device="cpu",
            dtype=torch.float32,
        ).contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def optimizer_state_sha256(optimizer: Any) -> str:
    torch, _ = _torch_imports()
    digest = hashlib.sha256()
    state = optimizer.state_dict()
    digest.update(json.dumps(state["param_groups"], sort_keys=True).encode("utf-8"))
    for parameter_id in sorted(state["state"]):
        digest.update(str(parameter_id).encode("utf-8"))
        values = state["state"][parameter_id]
        for key in sorted(values):
            value = values[key]
            digest.update(key.encode("utf-8"))
            if torch.is_tensor(value):
                tensor = value.detach().to(
                    device="cpu",
                    dtype=torch.float32,
                ).contiguous()
                digest.update(str(tuple(tensor.shape)).encode("utf-8"))
                digest.update(tensor.numpy().tobytes())
            else:
                digest.update(repr(value).encode("utf-8"))
    return digest.hexdigest()


def _fixed_noise(
    shape: Tuple[int, ...],
    *,
    seed: int,
) -> np.ndarray:
    rng = np.random.RandomState(int(seed))
    return rng.standard_normal(shape).astype(np.float32)


def train_model(
    *,
    condition: np.ndarray,
    target: np.ndarray,
    spec: DiagnosticSpec,
    condition_standardizer: ArrayStandardizer,
    target_standardizer: ArrayStandardizer,
) -> Tuple[Any, Dict[str, Any], Dict[str, np.ndarray]]:
    torch, _ = _torch_imports()
    device = torch.device("cuda:0")
    model = make_model(spec).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=spec.learning_rate,
        weight_decay=spec.weight_decay,
    )
    scheduler = scheduler_arrays(spec)
    alpha_bar = torch.as_tensor(
        scheduler["alpha_bar"],
        dtype=torch.float32,
        device=device,
    )
    condition_z = torch.as_tensor(
        condition_standardizer.normalize(condition),
        dtype=torch.float32,
        device=device,
    )
    target_z = torch.as_tensor(
        target_standardizer.normalize(target),
        dtype=torch.float32,
        device=device,
    )
    initial_model_sha = tensor_state_sha256(model)
    initial_optimizer_sha = optimizer_state_sha256(optimizer)

    generator = torch.Generator(device=device)
    generator.manual_seed(spec.seed + 17)
    loss_values: List[float] = []
    gradient_values: List[float] = []
    exposure = np.zeros(condition.shape[0], dtype=np.int64)
    model.train()
    for step in range(spec.train_steps):
        indices = torch.randint(
            low=0,
            high=condition.shape[0],
            size=(spec.batch_size,),
            generator=generator,
            device=device,
        )
        timestep = torch.randint(
            low=0,
            high=spec.train_timesteps,
            size=(spec.batch_size,),
            generator=generator,
            device=device,
        )
        noise = torch.randn(
            (spec.batch_size, FUTURE_STEPS, CABLE_DIM),
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        clean = target_z[indices]
        alpha = alpha_bar[timestep].reshape(-1, 1, 1)
        noisy = torch.sqrt(alpha) * clean + torch.sqrt(1.0 - alpha) * noise
        predicted = model(noisy, timestep, condition_z[indices])
        loss = torch.mean((predicted - clean) ** 2)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=10.0,
        )
        optimizer.step()
        loss_values.append(float(loss.detach().cpu()))
        gradient_values.append(float(gradient_norm.detach().cpu()))
        np.add.at(exposure, indices.detach().cpu().numpy(), 1)

    model.eval()
    records = {
        "initial_model_sha256": initial_model_sha,
        "final_model_sha256": tensor_state_sha256(model),
        "initial_optimizer_sha256": initial_optimizer_sha,
        "final_optimizer_sha256": optimizer_state_sha256(optimizer),
        "loss_history_sha256": sha256_array(
            np.asarray(loss_values, dtype=np.float64)
        ),
        "gradient_history_sha256": sha256_array(
            np.asarray(gradient_values, dtype=np.float64)
        ),
        "source_exposure_sha256": sha256_array(exposure),
        "loss_first": float(loss_values[0]),
        "loss_final": float(loss_values[-1]),
        "loss_tail_mean": float(np.mean(loss_values[-100:])),
        "gradient_tail_mean": float(np.mean(gradient_values[-100:])),
        "training_rows": int(condition.shape[0]),
        "training_steps": spec.train_steps,
    }
    histories = {
        "loss": np.asarray(loss_values, dtype=np.float64),
        "gradient": np.asarray(gradient_values, dtype=np.float64),
        "exposure": exposure,
    }
    return model, records, histories


def predict_x0(
    *,
    model: Any,
    noisy: np.ndarray,
    timestep: np.ndarray,
    condition: np.ndarray,
    condition_standardizer: ArrayStandardizer,
    target_standardizer: ArrayStandardizer,
) -> np.ndarray:
    torch, _ = _torch_imports()
    device = next(model.parameters()).device
    noisy_z = target_standardizer.normalize(noisy)
    condition_z = condition_standardizer.normalize(condition)
    with torch.no_grad():
        prediction_z = model(
            torch.as_tensor(noisy_z, dtype=torch.float32, device=device),
            torch.as_tensor(timestep, dtype=torch.long, device=device),
            torch.as_tensor(condition_z, dtype=torch.float32, device=device),
        )
    return target_standardizer.denormalize(
        prediction_z.detach().cpu().numpy()
    )


def one_step_metrics(
    *,
    model: Any,
    condition: np.ndarray,
    target: np.ndarray,
    spec: DiagnosticSpec,
    condition_standardizer: ArrayStandardizer,
    target_standardizer: ArrayStandardizer,
    noise_seed: int,
) -> Dict[str, Any]:
    scheduler = scheduler_arrays(spec)
    target_z = target_standardizer.normalize(target)
    noise = _fixed_noise(target.shape, seed=noise_seed)
    records = {}
    prediction_digests = hashlib.sha256()
    for timestep_value in spec.one_step_timesteps:
        timestep = np.full(
            target.shape[0],
            timestep_value,
            dtype=np.int64,
        )
        noisy_z = q_sample_numpy(
            target_z,
            noise,
            timestep,
            scheduler["alpha_bar"],
        )
        noisy = target_standardizer.denormalize(noisy_z)
        prediction = predict_x0(
            model=model,
            noisy=noisy,
            timestep=timestep,
            condition=condition,
            condition_standardizer=condition_standardizer,
            target_standardizer=target_standardizer,
        )
        prediction_digests.update(
            np.ascontiguousarray(prediction, dtype=np.float32).tobytes()
        )
        records[str(timestep_value)] = {
            "normalized_mse": normalized_mse(
                prediction,
                target,
                target_standardizer,
            ),
            "raw_rmse": rmse(prediction, target),
        }
    return {
        "timesteps": records,
        "prediction_sha256": prediction_digests.hexdigest(),
    }


def reverse_sample(
    *,
    model: Any,
    condition: np.ndarray,
    spec: DiagnosticSpec,
    condition_standardizer: ArrayStandardizer,
    target_standardizer: ArrayStandardizer,
    noise_seed: int,
) -> np.ndarray:
    torch, _ = _torch_imports()
    device = next(model.parameters()).device
    scheduler = scheduler_arrays(spec)
    alpha_bar = scheduler["alpha_bar"].astype(np.float64)
    rows = condition.shape[0]
    candidates = spec.reverse_candidates
    initial = _fixed_noise(
        (rows, candidates, FUTURE_STEPS, CABLE_DIM),
        seed=noise_seed,
    )
    value = torch.as_tensor(
        initial.reshape(rows * candidates, FUTURE_STEPS, CABLE_DIM),
        dtype=torch.float32,
        device=device,
    )
    condition_z = condition_standardizer.normalize(condition)
    condition_repeated = np.repeat(
        condition_z[:, None, :],
        candidates,
        axis=1,
    ).reshape(rows * candidates, CONDITION_DIM)
    condition_tensor = torch.as_tensor(
        condition_repeated,
        dtype=torch.float32,
        device=device,
    )
    model.eval()
    with torch.no_grad():
        for timestep_value in reversed(range(spec.reverse_steps)):
            timestep = torch.full(
                (rows * candidates,),
                timestep_value,
                dtype=torch.long,
                device=device,
            )
            predicted_x0 = model(value, timestep, condition_tensor)
            current_alpha = float(alpha_bar[timestep_value])
            if timestep_value == 0:
                value = predicted_x0
                continue
            previous_alpha = float(alpha_bar[timestep_value - 1])
            epsilon = (
                value - math.sqrt(current_alpha) * predicted_x0
            ) / math.sqrt(max(1.0 - current_alpha, 1.0e-12))
            value = (
                math.sqrt(previous_alpha) * predicted_x0
                + math.sqrt(max(1.0 - previous_alpha, 0.0)) * epsilon
            )
    prediction_z = value.detach().cpu().numpy().reshape(
        rows,
        candidates,
        FUTURE_STEPS,
        CABLE_DIM,
    )
    return target_standardizer.denormalize(prediction_z)


def evaluate_diagnostic(
    *,
    model: Any,
    train_condition: np.ndarray,
    train_target: np.ndarray,
    probe_condition: np.ndarray,
    probe_target: np.ndarray,
    probe_pair_key: Sequence[Any],
    probe_condition_name: Sequence[Any],
    spec: DiagnosticSpec,
    condition_standardizer: ArrayStandardizer,
    target_standardizer: ArrayStandardizer,
    geometry: GeometryContract,
) -> Dict[str, Any]:
    scheduler = scheduler_arrays(spec)
    control_rows = min(spec.train_control_rows, train_condition.shape[0])
    train_condition_control = train_condition[:control_rows]
    train_target_control = train_target[:control_rows]
    target_control_z = target_standardizer.normalize(train_target_control)
    control_timestep = np.full(control_rows, 25, dtype=np.int64)
    control_noise = _fixed_noise(
        train_target_control.shape,
        seed=spec.seed + 1001,
    )
    control_noisy_z = q_sample_numpy(
        target_control_z,
        control_noise,
        control_timestep,
        scheduler["alpha_bar"],
    )
    control_prediction = predict_x0(
        model=model,
        noisy=target_standardizer.denormalize(control_noisy_z),
        timestep=control_timestep,
        condition=train_condition_control,
        condition_standardizer=condition_standardizer,
        target_standardizer=target_standardizer,
    )
    train_control_nmse = normalized_mse(
        control_prediction,
        train_target_control,
        target_standardizer,
    )

    probe_one_step = one_step_metrics(
        model=model,
        condition=probe_condition,
        target=probe_target,
        spec=spec,
        condition_standardizer=condition_standardizer,
        target_standardizer=target_standardizer,
        noise_seed=spec.seed + 2001,
    )
    candidates = reverse_sample(
        model=model,
        condition=probe_condition,
        spec=spec,
        condition_standardizer=condition_standardizer,
        target_standardizer=target_standardizer,
        noise_seed=spec.seed + 3001,
    )
    candidate_sha = sha256_array(candidates)
    best_nmse, best_indices = best_of_k_nmse(
        candidates,
        probe_target,
        target_standardizer,
    )
    mean_prediction = np.mean(candidates, axis=1)
    mean_nmse = normalized_mse(
        mean_prediction,
        probe_target,
        target_standardizer,
    )
    last = last_state_baseline(probe_condition)
    velocity = constant_velocity_baseline(probe_condition)
    baseline_metrics = {
        "last_state": {
            "normalized_mse":
                normalized_mse(last, probe_target, target_standardizer),
            "raw_rmse": rmse(last, probe_target),
        },
        "constant_velocity": {
            "normalized_mse":
                normalized_mse(velocity, probe_target, target_standardizer),
            "raw_rmse": rmse(velocity, probe_target),
        },
    }
    best_baseline_name = min(
        baseline_metrics,
        key=lambda name: baseline_metrics[name]["normalized_mse"],
    )
    best_baseline_nmse = float(
        baseline_metrics[best_baseline_name]["normalized_mse"]
    )
    improvement = float(
        (best_baseline_nmse - best_nmse)
        / max(abs(best_baseline_nmse), 1.0e-12)
    )
    physical = physical_validity(candidates, geometry)
    branch = branch_metrics(
        pair_key=probe_pair_key,
        condition_name=probe_condition_name,
        target=probe_target,
        candidates=candidates,
    )

    train_pass = bool(train_control_nmse <= spec.train_control_nmse_max)
    generalization_pass = bool(improvement >= spec.probe_improvement_min)
    branch_pass = bool(
        branch["row_own_branch_support_rate"] >= spec.branch_support_min
    )
    physical_pass = bool(
        physical["candidate_rate"] >= spec.physical_candidate_rate_min
    )

    if not train_pass:
        root = (
            "phase314b_r256_stageb_cable_only_denoiser_train_control_failed"
        )
        next_path = "REPAIR_CABLE_ONLY_DENOISER_OPTIMIZATION"
    elif not generalization_pass:
        root = (
            "phase314b_r256_stageb_cable_only_probe_generalization_failed"
        )
        next_path = (
            "REPAIR_CABLE_ONLY_CONDITIONING_OR_GENERALIZATION_BEFORE_FORMAL_PILOT"
        )
    elif not branch_pass:
        root = (
            "phase314b_r256_stageb_cable_only_branch_support_failed"
        )
        next_path = (
            "REPAIR_CABLE_ONLY_BRANCH_TRANSPORT_BEFORE_FORMAL_PILOT"
        )
    elif not physical_pass:
        root = (
            "phase314b_r256_stageb_cable_only_physical_validity_failed"
        )
        next_path = (
            "REPAIR_CABLE_ONLY_REVERSE_PHYSICAL_VALIDITY_BEFORE_FORMAL_PILOT"
        )
    else:
        root = (
            "phase314b_r256_stageb_train_only_cable_diffusion_"
            "diagnostic_supported"
        )
        next_path = (
            "RUN_FROZEN_CABLE_ONLY_FORMAL_PILOT_AND_COLLECT_"
            "ACTION_DIVERSE_IDM_DATA"
        )
    return {
        "train_control": {
            "rows": control_rows,
            "timestep": 25,
            "normalized_mse": train_control_nmse,
            "threshold": spec.train_control_nmse_max,
            "pass": train_pass,
            "prediction_sha256": sha256_array(control_prediction),
        },
        "probe_one_step": probe_one_step,
        "probe_reverse": {
            "candidate_count": spec.reverse_candidates,
            "candidate_sha256": candidate_sha,
            "best_of_k_normalized_mse": best_nmse,
            "mean_prediction_normalized_mse": mean_nmse,
            "best_candidate_index_sha256": sha256_array(best_indices),
            "baseline_metrics": baseline_metrics,
            "best_baseline": best_baseline_name,
            "best_baseline_normalized_mse": best_baseline_nmse,
            "relative_improvement": improvement,
            "required_improvement": spec.probe_improvement_min,
            "generalization_pass": generalization_pass,
        },
        "branch": {
            **branch,
            "required_support_rate": spec.branch_support_min,
            "pass": branch_pass,
        },
        "physical": {
            "candidate_rate": physical["candidate_rate"],
            "row_any_rate": physical["row_any_rate"],
            "required_candidate_rate":
                spec.physical_candidate_rate_min,
            "pass": physical_pass,
            "finite_rate": float(np.mean(physical["finite"])),
            "coordinate_rate": float(np.mean(physical["coordinate"])),
            "segment_rate": float(np.mean(physical["segment"])),
            "topology_rate": float(np.mean(physical["topology"])),
            "maximum_intersections":
                int(np.max(physical["intersection_max"])),
        },
        "root_cause": root,
        "required_next_path": next_path,
        "diagnostic_pass": bool(
            train_pass
            and generalization_pass
            and branch_pass
            and physical_pass
        ),
    }


def run_diagnostic(
    *,
    root: Path,
    spec: Optional[DiagnosticSpec] = None,
) -> Dict[str, Any]:
    active_spec = DiagnosticSpec() if spec is None else spec
    active_spec.validate()
    repository_root = Path(root).resolve()
    immutable = validate_immutable_inputs(repository_root)
    source_audit = source_contract_audit(repository_root)
    if not source_audit["all_confirmed"]:
        raise CableDiffusionDiagnosticError(
            "historical source assumptions changed"
        )
    arrays = load_npz_strict(
        repository_root / STAGEA_TRAIN_VIEW,
        required_keys=REQUIRED_TRAIN_KEYS,
    )
    validation = validate_train_view(arrays)
    runtime = set_deterministic_runtime(active_spec.seed)

    condition = np.asarray(
        arrays["diffusion_condition_x"],
        dtype=np.float32,
    )
    target = np.asarray(
        arrays["diffusion_target_cable"],
        dtype=np.float32,
    )
    groups = np.asarray(arrays["episode_group_key"]).astype(str)
    train_mask, probe_mask, group_mapping = deterministic_group_split(
        groups,
        folds=active_spec.group_folds,
        probe_fold=active_spec.probe_fold,
    )
    condition_standardizer = fit_standardizer(condition[train_mask])
    target_standardizer = fit_standardizer(target[train_mask])
    geometry = fit_geometry_contract(target[train_mask])

    model, training, histories = train_model(
        condition=condition[train_mask],
        target=target[train_mask],
        spec=active_spec,
        condition_standardizer=condition_standardizer,
        target_standardizer=target_standardizer,
    )
    evaluation = evaluate_diagnostic(
        model=model,
        train_condition=condition[train_mask],
        train_target=target[train_mask],
        probe_condition=condition[probe_mask],
        probe_target=target[probe_mask],
        probe_pair_key=np.asarray(arrays["pair_key"]).astype(str)[probe_mask],
        probe_condition_name=
            np.asarray(arrays["condition_name"]).astype(str)[probe_mask],
        spec=active_spec,
        condition_standardizer=condition_standardizer,
        target_standardizer=target_standardizer,
        geometry=geometry,
    )
    identity = {
        "final_model_sha256": training["final_model_sha256"],
        "final_optimizer_sha256": training["final_optimizer_sha256"],
        "loss_history_sha256": training["loss_history_sha256"],
        "gradient_history_sha256": training["gradient_history_sha256"],
        "source_exposure_sha256": training["source_exposure_sha256"],
        "train_control_prediction_sha256":
            evaluation["train_control"]["prediction_sha256"],
        "probe_one_step_prediction_sha256":
            evaluation["probe_one_step"]["prediction_sha256"],
        "probe_reverse_candidate_sha256":
            evaluation["probe_reverse"]["candidate_sha256"],
    }
    return {
        "phase": PHASE,
        "phase_id": PHASE_ID,
        "schema": "phase314b_r256_stageb_worker_result_v1",
        "verdict": "PASS",
        "scientific_status": "BLOCKED",
        "root_cause": evaluation["root_cause"],
        "required_next_path": evaluation["required_next_path"],
        "spec": asdict(active_spec),
        "runtime": runtime,
        "immutable_inputs": immutable["file_sha256"],
        "source_contract_audit": source_audit,
        "train_view_validation": validation,
        "split": {
            "group_folds": active_spec.group_folds,
            "probe_fold": active_spec.probe_fold,
            "total_groups": len(group_mapping),
            "training_rows": int(np.sum(train_mask)),
            "probe_rows": int(np.sum(probe_mask)),
            "training_groups": int(
                len(set(groups[train_mask].tolist()))
            ),
            "probe_groups": int(
                len(set(groups[probe_mask].tolist()))
            ),
            "group_integrity_pass": True,
        },
        "standardizers": {
            "condition_mean_sha256":
                sha256_array(condition_standardizer.mean),
            "condition_scale_sha256":
                sha256_array(condition_standardizer.scale),
            "target_mean_sha256":
                sha256_array(target_standardizer.mean),
            "target_scale_sha256":
                sha256_array(target_standardizer.scale),
        },
        "geometry_contract": {
            "segment_lower_sha256":
                sha256_array(geometry.segment_lower),
            "segment_upper_sha256":
                sha256_array(geometry.segment_upper),
            "coordinate_abs_max": geometry.coordinate_abs_max,
            "target_intersection_max":
                geometry.target_intersection_max,
        },
        "training": training,
        "evaluation": evaluation,
        "identity": identity,
        "same_device_determinism_required": True,
        "train_only": True,
        "validation_targets_used": False,
        "formal_test_targets_used": False,
        "checkpoint_saved": False,
        "weights_persisted": False,
        "prediction_tensor_persisted": False,
        "diffusion_training": True,
        "formal_diffusion_training": False,
        "reverse_sampling": True,
        "formal_reverse_sampling": False,
        "formal_idm_training": False,
        "action_diverse_data_collection": False,
        "candidate_execution": False,
        "phase4": False,
        "cps": False,
        "train_only_recommendation": None,
        "selected_configuration": None,
    }


def identity_projection(result: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "root_cause": result["root_cause"],
        "required_next_path": result["required_next_path"],
        "identity": result["identity"],
        "training": {
            "loss_first": result["training"]["loss_first"],
            "loss_final": result["training"]["loss_final"],
            "loss_tail_mean": result["training"]["loss_tail_mean"],
            "gradient_tail_mean":
                result["training"]["gradient_tail_mean"],
        },
        "evaluation": result["evaluation"],
        "split": result["split"],
        "standardizers": result["standardizers"],
        "geometry_contract": result["geometry_contract"],
    }


def compare_worker_results(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> Dict[str, Any]:
    left_projection = identity_projection(left)
    right_projection = identity_projection(right)
    left_bytes = stable_json_bytes(left_projection)
    right_bytes = stable_json_bytes(right_projection)
    return {
        "exact": left_bytes == right_bytes,
        "left_sha256": sha256_bytes(left_bytes),
        "right_sha256": sha256_bytes(right_bytes),
        "identity_categories_exact": {
            key: left["identity"][key] == right["identity"][key]
            for key in sorted(left["identity"])
        },
    }
