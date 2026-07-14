"""Train-only robot-proxy schema, predictability and action-sensitivity audit.

This module is intentionally independent of the diffusion training loop.  The
r2.5.4 runner reuses the frozen r2.5.3 training implementation and calls these
helpers only on train rows and in-memory predictions.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
import subprocess
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import torch

from ccda_phase3.schema_v2 import N_BEADS, ROBOT_PROXY_DIM, STATE_DIM

PHASE = "phase3_14b_r254"
BASE_REPORT_COMMIT = "c35c374f61f9b4dc2ca4cdaa76b12ddda71be994"
BASE_IMPLEMENTATION_COMMIT = "1e7873f5981093f1d79f547892357d35ad54e87b"
EXPECTED_SUBMODULE_COMMIT = "633a88752445cf5d6776ed374fdbbdb35f93050c"
EXPECTED_CACHE_SHA256 = (
    "3cc512f650557c81c3b81f4f128a5b55360b77d3f664fdd6607666936285fbe8"
)
EXPECTED_CONTRACT_SHA256 = (
    "fa2725ca40da2499008360f13d291d2ce8694e6393910b0522fe800f4f37cdcc"
)
EXPECTED_R253_ROOT_CAUSE = (
    "phase314b_r253_robot_proxy_reconstruction_conflation_supported"
)
EXPECTED_R253_MECHANISMS = (
    "exact_reconstruction_vs_branch_identity_separation",
    "intermediate_horizon_fidelity_gap",
    "exact_cable_fidelity_tail_gap",
    "all_source_tail_failure",
    "robot_proxy_reconstruction_conflation",
)
EXPECTED_R253_RECOMMENDATION = None
EXPECTED_R253_PILOT_SHA256 = (
    "a6a8aa2faf739e80159af7e70cb57b0e2a4c852ab5af5c2d7ecb29ff31116c86"
)
EXPECTED_SHARED_PAIRED_PRIOR_SHA256 = (
    "083278d6b0710ddc3863d822978973f29842bc757edf73ff03a93bb154f5665d"
)
EXPECTED_PAIRED_ROWS = (
    6, 1226, 10, 1230, 19, 1239, 25, 1245,
    31, 1251, 36, 1256, 43, 1263, 49, 1269,
)
DIAGNOSTIC_OBJECTIVE_NAMES = (
    "v_only_frozen_control",
    "ordered_mean_raw_g100",
    "ordered_cvar_contract_g010",
)
PAIR_TIMESTEPS = (10, 25, 50)
EVALUATION_NOISE_SEEDS = tuple(99000 + index for index in range(8))
COMMON_PAIRED_PRIOR_SEED = 102000
COMMON_PAIRED_TRAINING_SEED = 102000

CABLE_DIM = N_BEADS * 2
ROBOT_JOINT_POSITION_SLICE = slice(0, 16)
ROBOT_JOINT_VELOCITY_SLICE = slice(16, 32)
ROBOT_EE_POSITION_SLICE = slice(32, 35)
ROBOT_EE_ORIENTATION_SLICE = slice(35, 39)

ROBOT_PROXY_SCHEMA = "phase314b_r254_robot_proxy_schema_audit_v1"
ROBOT_BASELINE_SCHEMA = "phase314b_r254_robot_predictability_baselines_v1"
ROBOT_MODEL_ERROR_SCHEMA = "phase314b_r254_model_robot_error_v1"
ROBOT_ACTION_SCHEMA = "phase314b_r254_action_sensitivity_probe_v1"
ROBOT_HYBRID_SCHEMA = "phase314b_r254_hybrid_counterfactual_v1"
ROBOT_ATTRIBUTION_SCHEMA = "phase314b_r254_robot_proxy_attribution_v1"

SOURCE_PATHS = (
    "ccda_phase3/phase314b_r254_robot_proxy_attribution.py",
    "scripts/phase3_14b_r254_preflight.py",
    "scripts/phase3_14b_r254_run_pilot.py",
    "scripts/phase3_14b_r254_finalize.py",
    "scripts/phase3_14b_r254_run.sh",
    "tests/test_phase314b_r254_robot_proxy_attribution.py",
)
DEPENDENCY_PATHS = (
    "ccda_phase3/data_io.py",
    "ccda_phase3/schema_v2.py",
    "ccda_phase3/phase314b_r252_transport_attribution.py",
    "ccda_phase3/phase314b_r253_gate_separation.py",
    "scripts/phase3_14b_r253_run_pilot.py",
    "reports/phase3_14b_r253_pilot_summary.json",
    "reports/phase3_14b_r253_summary.json",
    "reports/phase3_14b_r253_report.md",
)


@dataclass(frozen=True)
class RobotProxyAuditSpec:
    variance_epsilon: float = 1.0e-10
    zero_epsilon: float = 1.0e-8
    structural_padding_zero_fraction_min: float = 0.999
    intermittent_zero_fraction_min: float = 0.05
    intermittent_zero_fraction_max: float = 0.95
    quaternion_norm_error_p95_max: float = 5.0e-3
    quaternion_sign_ambiguity_fraction_min: float = 0.01
    quaternion_sign_invariant_gain_min: float = 0.20
    simple_baseline_sufficiency_ratio: float = 1.05
    action_conditioned_gain_min: float = 0.20
    action_robot_incremental_gain_min: float = 0.05
    model_robot_nmse_max: float = 0.50
    ridge_regularization: float = 1.0e-3
    reproduction_absolute_tolerance: float = 2.0e-6
    minimum_models_for_mechanism: int = 2


def robot_proxy_groups() -> Dict[str, Tuple[int, int]]:
    return {
        "joint_position": (0, 16),
        "joint_velocity": (16, 32),
        "ee_position": (32, 35),
        "ee_orientation": (35, 39),
    }


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_paths(root: Path, paths: Sequence[str]) -> Dict[str, str]:
    return {path: sha256_file(root / path) for path in paths}


def source_sha256(root: Path) -> Dict[str, str]:
    return _hash_paths(root, SOURCE_PATHS)


def dependency_sha256(root: Path) -> Dict[str, str]:
    return _hash_paths(root, DEPENDENCY_PATHS)


def git_output(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=root, text=True, stderr=subprocess.STDOUT
    ).strip()


def assert_only_allowed_worktree_paths(root: Path, allowed: Sequence[str]) -> None:
    observed = {
        line[3:].strip()
        for line in git_output(root, "status", "--short").splitlines()
        if line.strip()
    }
    allowed_set = {str(value) for value in allowed}
    unexpected = sorted(observed - allowed_set)
    if unexpected:
        raise RuntimeError(f"unexpected worktree paths: {unexpected}")


def _as_numpy(value: Any, *, name: str, dtype: Any = np.float64) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    array = array.astype(dtype, copy=False)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _require_state_history(value: Any, *, name: str) -> np.ndarray:
    array = _as_numpy(value, name=name)
    if array.ndim == 2:
        if array.shape[1] % STATE_DIM != 0:
            raise ValueError(f"{name} flattened width must be divisible by {STATE_DIM}")
        array = array.reshape(array.shape[0], array.shape[1] // STATE_DIM, STATE_DIM)
    if array.ndim != 3 or array.shape[-1] != STATE_DIM:
        raise ValueError(f"{name} must be [N,H,{STATE_DIM}], got {array.shape}")
    if array.shape[1] < 2:
        raise ValueError(f"{name} requires at least two history states")
    return array


def _require_future(value: Any, *, name: str) -> np.ndarray:
    array = _as_numpy(value, name=name)
    if array.ndim != 3 or array.shape[-1] != STATE_DIM:
        raise ValueError(f"{name} must be [N,T,{STATE_DIM}], got {array.shape}")
    return array


def _require_robot(value: Any, *, name: str) -> np.ndarray:
    array = _as_numpy(value, name=name)
    if array.ndim != 3 or array.shape[-1] != ROBOT_PROXY_DIM:
        raise ValueError(
            f"{name} must be [N,T,{ROBOT_PROXY_DIM}], got {array.shape}"
        )
    return array


def _stats(values: Any) -> Dict[str, float]:
    array = _as_numpy(values, name="values").reshape(-1)
    if array.size == 0:
        return {key: 0.0 for key in ("mean", "std", "p05", "p50", "p95", "min", "max")}
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "p05": float(np.percentile(array, 5.0)),
        "p50": float(np.percentile(array, 50.0)),
        "p95": float(np.percentile(array, 95.0)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / max(abs(denominator), 1.0e-12))


def _nmse(prediction: np.ndarray, target: np.ndarray, active: np.ndarray) -> float:
    mask = np.asarray(active, dtype=bool)
    if mask.ndim == 1:
        mask = np.broadcast_to(mask[None, :], target.shape[1:])
    if mask.shape != target.shape[1:]:
        raise ValueError(f"active shape mismatch: {mask.shape} vs {target.shape[1:]}")
    selected_prediction = prediction[:, mask]
    selected_target = target[:, mask]
    mse = float(np.mean((selected_prediction - selected_target) ** 2))
    variance = float(np.mean((selected_target - np.mean(selected_target, axis=0)) ** 2))
    return _safe_ratio(mse, variance)


def normalize_quaternion(value: np.ndarray) -> np.ndarray:
    array = _as_numpy(value, name="quaternion")
    if array.shape[-1] != 4:
        raise ValueError("quaternion last dimension must be 4")
    norm = np.linalg.norm(array, axis=-1, keepdims=True)
    fallback = np.zeros_like(array)
    fallback[..., 3] = 1.0
    return np.where(norm > 1.0e-12, array / np.maximum(norm, 1.0e-12), fallback)


def canonicalize_quaternion_trajectory(
    quaternion: Any,
    *,
    reference: Any | None = None,
) -> np.ndarray:
    values = normalize_quaternion(_as_numpy(quaternion, name="quaternion"))
    if values.ndim != 3 or values.shape[-1] != 4:
        raise ValueError("quaternion must be [N,T,4]")
    output = values.copy()
    if reference is None:
        previous = output[:, 0].copy()
        start = 1
    else:
        previous = normalize_quaternion(_as_numpy(reference, name="reference"))
        if previous.shape != (values.shape[0], 4):
            raise ValueError("reference must be [N,4]")
        start = 0
    for timestep in range(start, output.shape[1]):
        flip = np.sum(previous * output[:, timestep], axis=-1) < 0.0
        output[flip, timestep] *= -1.0
        previous = output[:, timestep].copy()
    return output


def quaternion_error_metrics(
    prediction: Any,
    target: Any,
    *,
    reference: Any | None = None,
) -> Dict[str, Any]:
    pred = normalize_quaternion(_as_numpy(prediction, name="prediction_quaternion"))
    truth = normalize_quaternion(_as_numpy(target, name="target_quaternion"))
    if pred.shape != truth.shape or pred.ndim != 3:
        raise ValueError("quaternion prediction/target must match [N,T,4]")
    direct_sq = np.mean((pred - truth) ** 2, axis=-1)
    negated_sq = np.mean((pred + truth) ** 2, axis=-1)
    sign_invariant_sq = np.minimum(direct_sq, negated_sq)
    dot = np.clip(np.abs(np.sum(pred * truth, axis=-1)), 0.0, 1.0)
    angle = 2.0 * np.arccos(dot)
    canonical_pred = canonicalize_quaternion_trajectory(pred, reference=reference)
    canonical_truth = canonicalize_quaternion_trajectory(truth, reference=reference)
    canonical_sq = np.mean((canonical_pred - canonical_truth) ** 2, axis=-1)
    target_dot = np.sum(truth[:, 1:] * truth[:, :-1], axis=-1)
    flip_fraction = float(np.mean(target_dot < 0.0)) if target_dot.size else 0.0
    direct_mse = float(np.mean(direct_sq))
    invariant_mse = float(np.mean(sign_invariant_sq))
    return {
        "direct_mse": direct_mse,
        "sign_invariant_mse": invariant_mse,
        "canonicalized_mse": float(np.mean(canonical_sq)),
        "sign_invariant_gain": _safe_ratio(direct_mse - invariant_mse, direct_mse),
        "geodesic_angle_rad": _stats(angle),
        "target_consecutive_sign_flip_fraction": flip_fraction,
        "prediction_norm_error": _stats(np.abs(np.linalg.norm(prediction, axis=-1) - 1.0)),
        "target_norm_error": _stats(np.abs(np.linalg.norm(target, axis=-1) - 1.0)),
    }


def robot_proxy_schema_audit(
    *,
    history_raw: Any,
    future_raw: Any,
    future_active: Any,
    spec: RobotProxyAuditSpec,
) -> Dict[str, Any]:
    history = _require_state_history(history_raw, name="history_raw")
    future = _require_future(future_raw, name="future_raw")
    if history.shape[0] != future.shape[0]:
        raise ValueError("history/future row count mismatch")
    active = np.asarray(future_active, dtype=bool)
    if active.shape != future.shape[1:]:
        raise ValueError("future_active shape mismatch")
    robot_history = history[..., CABLE_DIM:]
    robot_future = future[..., CABLE_DIM:]
    robot_active = active[:, CABLE_DIM:]
    if robot_history.shape[-1] != ROBOT_PROXY_DIM:
        raise ValueError("robot history dimension mismatch")

    combined = np.concatenate([robot_history.reshape(-1, ROBOT_PROXY_DIM), robot_future.reshape(-1, ROBOT_PROXY_DIM)], axis=0)
    dimension_records: List[Dict[str, Any]] = []
    active_any = np.any(robot_active, axis=0)
    for dimension in range(ROBOT_PROXY_DIM):
        values = combined[:, dimension]
        variance = float(np.var(values))
        zero_fraction = float(np.mean(np.abs(values) <= spec.zero_epsilon))
        structural_padding = bool(
            zero_fraction >= spec.structural_padding_zero_fraction_min
            and variance <= spec.variance_epsilon
        )
        intermittent_zero = bool(
            spec.intermittent_zero_fraction_min <= zero_fraction
            <= spec.intermittent_zero_fraction_max
        )
        dimension_records.append({
            "robot_dimension": dimension,
            "active": bool(active_any[dimension]),
            "variance": variance,
            "zero_fraction": zero_fraction,
            "structural_padding_candidate": structural_padding,
            "intermittent_zero_candidate": intermittent_zero,
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
        })

    group_records: Dict[str, Any] = {}
    for name, (start, end) in robot_proxy_groups().items():
        values = combined[:, start:end]
        group_records[name] = {
            "slice": [start, end],
            "dimension_count": end - start,
            "active_dimension_count": int(np.count_nonzero(active_any[start:end])),
            "variance": _stats(np.var(values, axis=0)),
            "zero_fraction": _stats(np.mean(np.abs(values) <= spec.zero_epsilon, axis=0)),
        }

    history_last_q = robot_history[:, -1, ROBOT_EE_ORIENTATION_SLICE]
    quaternion = quaternion_error_metrics(
        robot_future[..., ROBOT_EE_ORIENTATION_SLICE],
        robot_future[..., ROBOT_EE_ORIENTATION_SLICE],
        reference=history_last_q,
    )
    target_q = robot_future[..., ROBOT_EE_ORIENTATION_SLICE]
    consecutive = canonicalize_quaternion_trajectory(target_q, reference=history_last_q)
    direct_target_change = np.mean((normalize_quaternion(target_q) - consecutive) ** 2)
    target_norm_error = np.abs(np.linalg.norm(target_q, axis=-1) - 1.0)
    quaternion.update({
        "target_self_canonicalization_mse": float(direct_target_change),
        "target_norm_error": _stats(target_norm_error),
    })

    structural = [
        item["robot_dimension"]
        for item in dimension_records
        if item["structural_padding_candidate"]
    ]
    intermittent = [
        item["robot_dimension"]
        for item in dimension_records
        if item["intermittent_zero_candidate"]
    ]
    near_zero_variance = [
        item["robot_dimension"]
        for item in dimension_records
        if item["variance"] <= spec.variance_epsilon
    ]
    representation_warnings = []
    if structural:
        representation_warnings.append("structural_zero_padding_present")
    if intermittent:
        representation_warnings.append("intermittent_zero_without_presence_mask")
    if quaternion["target_norm_error"]["p95"] > spec.quaternion_norm_error_p95_max:
        representation_warnings.append("non_unit_quaternion_targets")
    if quaternion["target_consecutive_sign_flip_fraction"] >= spec.quaternion_sign_ambiguity_fraction_min:
        representation_warnings.append("quaternion_sign_ambiguity_present")
    return {
        "schema": ROBOT_PROXY_SCHEMA,
        "state_dim": int(STATE_DIM),
        "cable_dim": int(CABLE_DIM),
        "robot_proxy_dim": int(ROBOT_PROXY_DIM),
        "layout": robot_proxy_groups(),
        "dimension_records": dimension_records,
        "groups": group_records,
        "structural_padding_dimensions": structural,
        "intermittent_zero_dimensions": intermittent,
        "near_zero_variance_dimensions": near_zero_variance,
        "quaternion": quaternion,
        "representation_warnings": representation_warnings,
        "schema_contract_pass": bool(
            STATE_DIM == CABLE_DIM + ROBOT_PROXY_DIM
            and ROBOT_PROXY_DIM == 39
            and active.shape[-1] == STATE_DIM
        ),
    }


def last_observation_robot_baseline(history_raw: Any, future_steps: int) -> np.ndarray:
    history = _require_state_history(history_raw, name="history_raw")
    last = history[:, -1, CABLE_DIM:]
    return np.repeat(last[:, None, :], int(future_steps), axis=1)


def constant_velocity_robot_baseline(history_raw: Any, future_steps: int) -> np.ndarray:
    history = _require_state_history(history_raw, name="history_raw")
    robot = history[..., CABLE_DIM:]
    last = robot[:, -1]
    previous = robot[:, -2]
    delta = last - previous
    result = np.repeat(last[:, None, :], int(future_steps), axis=1)
    for horizon in range(int(future_steps)):
        step = float(horizon + 1)
        result[:, horizon, ROBOT_JOINT_POSITION_SLICE] = (
            last[:, ROBOT_JOINT_POSITION_SLICE]
            + step * delta[:, ROBOT_JOINT_POSITION_SLICE]
        )
        result[:, horizon, ROBOT_JOINT_VELOCITY_SLICE] = last[:, ROBOT_JOINT_VELOCITY_SLICE]
        result[:, horizon, ROBOT_EE_POSITION_SLICE] = (
            last[:, ROBOT_EE_POSITION_SLICE]
            + step * delta[:, ROBOT_EE_POSITION_SLICE]
        )
        result[:, horizon, ROBOT_EE_ORIENTATION_SLICE] = last[:, ROBOT_EE_ORIENTATION_SLICE]
    result[..., ROBOT_EE_ORIENTATION_SLICE] = normalize_quaternion(
        result[..., ROBOT_EE_ORIENTATION_SLICE]
    )
    return result


def standardize_future_raw(
    future_raw: Any,
    *,
    future_mean: Any,
    future_scale: Any,
    future_active: Any,
) -> np.ndarray:
    future = _require_future(future_raw, name="future_raw")
    mean = _as_numpy(future_mean, name="future_mean")
    scale = _as_numpy(future_scale, name="future_scale")
    active = np.asarray(future_active, dtype=bool)
    if mean.shape != future.shape[1:] or scale.shape != future.shape[1:]:
        raise ValueError("future standardizer shape mismatch")
    if active.shape != future.shape[1:]:
        raise ValueError("future_active shape mismatch")
    output = np.zeros_like(future, dtype=np.float64)
    safe = np.where(active, np.maximum(scale, 1.0e-12), 1.0)
    output[:, active] = ((future - mean)[..., :])[:, active] / safe[active]
    return output


def ridge_fit_predict(
    train_x: Any,
    train_y: Any,
    test_x: Any,
    *,
    regularization: float,
) -> np.ndarray:
    x = _as_numpy(train_x, name="train_x")
    y = _as_numpy(train_y, name="train_y")
    test = _as_numpy(test_x, name="test_x")
    if x.ndim != 2 or y.ndim != 2 or test.ndim != 2:
        raise ValueError("ridge arrays must be two-dimensional")
    if x.shape[0] != y.shape[0] or x.shape[1] != test.shape[1]:
        raise ValueError("ridge shape mismatch")
    x_mean = np.mean(x, axis=0, keepdims=True)
    y_mean = np.mean(y, axis=0, keepdims=True)
    x_centered = x - x_mean
    y_centered = y - y_mean
    scale = np.std(x_centered, axis=0, keepdims=True)
    scale = np.where(scale > 1.0e-12, scale, 1.0)
    x_scaled = x_centered / scale
    test_scaled = (test - x_mean) / scale
    gram = x_scaled.T @ x_scaled
    regularized = gram + float(regularization) * np.eye(gram.shape[0])
    weights = np.linalg.solve(regularized, x_scaled.T @ y_centered)
    return test_scaled @ weights + y_mean


def grouped_holdout_ridge_predictions(
    *,
    features: Any,
    targets: Any,
    groups: Sequence[Any],
    fit_rows: Sequence[int],
    evaluation_rows: Sequence[int],
    regularization: float,
) -> np.ndarray:
    x = _as_numpy(features, name="features")
    y = _as_numpy(targets, name="targets")
    group_array = np.asarray(groups)
    fit_index = np.asarray(fit_rows, dtype=np.int64)
    eval_index = np.asarray(evaluation_rows, dtype=np.int64)
    if x.shape[0] != y.shape[0] or x.shape[0] != group_array.shape[0]:
        raise ValueError("row count mismatch")
    output = np.empty((len(eval_index), y.shape[1]), dtype=np.float64)
    for group in np.unique(group_array[eval_index]):
        eval_local = np.where(group_array[eval_index] == group)[0]
        train_rows = fit_index[group_array[fit_index] != group]
        if train_rows.size < 2:
            raise ValueError(f"insufficient holdout training rows for group {group!r}")
        output[eval_local] = ridge_fit_predict(
            x[train_rows], y[train_rows], x[eval_index[eval_local]],
            regularization=regularization,
        )
    return output


def _robot_metric_payload(
    prediction_z: np.ndarray,
    target_z: np.ndarray,
    active_robot: np.ndarray,
) -> Dict[str, Any]:
    if prediction_z.shape != target_z.shape:
        raise ValueError("robot metric shape mismatch")
    squared = (prediction_z - target_z) ** 2
    selected = squared[:, active_robot]
    by_horizon = {}
    for horizon in range(target_z.shape[1]):
        mask = active_robot[horizon]
        values = squared[:, horizon, mask]
        by_horizon[str(horizon)] = {
            "z_mse": float(np.mean(values)) if values.size else 0.0,
            "z_rmse": float(np.sqrt(np.mean(values))) if values.size else 0.0,
        }
    by_group = {}
    for name, (start, end) in robot_proxy_groups().items():
        mask = active_robot[:, start:end]
        values = squared[:, :, start:end][:, mask]
        by_group[name] = {
            "z_mse": float(np.mean(values)) if values.size else 0.0,
            "z_rmse": float(np.sqrt(np.mean(values))) if values.size else 0.0,
        }
    return {
        "z_mse": float(np.mean(selected)) if selected.size else 0.0,
        "z_rmse": float(np.sqrt(np.mean(selected))) if selected.size else 0.0,
        "nmse": _nmse(prediction_z, target_z, active_robot),
        "by_horizon": by_horizon,
        "by_group": by_group,
    }


def robot_predictability_baselines(
    *,
    history_raw: Any,
    condition_features: Any,
    future_raw: Any,
    future_z: Any,
    future_mean: Any,
    future_scale: Any,
    future_active: Any,
    action_target: Any,
    groups: Sequence[Any],
    fit_rows: Sequence[int],
    evaluation_rows: Sequence[int],
    spec: RobotProxyAuditSpec,
) -> Dict[str, Any]:
    history = _require_state_history(history_raw, name="history_raw")
    future = _require_future(future_raw, name="future_raw")
    future_z_array = _require_future(future_z, name="future_z")
    condition = _as_numpy(condition_features, name="condition_features")
    action = _as_numpy(action_target, name="action_target")
    eval_index = np.asarray(evaluation_rows, dtype=np.int64)
    active = np.asarray(future_active, dtype=bool)
    active_robot = active[:, CABLE_DIM:]
    target_robot_z = future_z_array[eval_index, :, CABLE_DIM:]
    target_robot_raw = future[eval_index, :, CABLE_DIM:]

    baseline_raw = {
        "last_observation": last_observation_robot_baseline(history[eval_index], future.shape[1]),
        "constant_velocity": constant_velocity_robot_baseline(history[eval_index], future.shape[1]),
    }
    predictions: Dict[str, np.ndarray] = {}
    for name, robot_raw in baseline_raw.items():
        full = future[eval_index].copy()
        full[..., CABLE_DIM:] = robot_raw
        predictions[name] = standardize_future_raw(
            full,
            future_mean=future_mean,
            future_scale=future_scale,
            future_active=future_active,
        )[..., CABLE_DIM:]

    flat_target = future_z_array[..., CABLE_DIM:].reshape(future.shape[0], -1)
    history_ridge = grouped_holdout_ridge_predictions(
        features=condition,
        targets=flat_target,
        groups=groups,
        fit_rows=fit_rows,
        evaluation_rows=eval_index,
        regularization=spec.ridge_regularization,
    ).reshape(len(eval_index), future.shape[1], ROBOT_PROXY_DIM)
    predictions["history_ridge"] = history_ridge

    action_mean = np.mean(action[np.asarray(fit_rows, dtype=np.int64)], axis=0, keepdims=True)
    action_scale = np.std(action[np.asarray(fit_rows, dtype=np.int64)], axis=0, keepdims=True)
    action_scale = np.where(action_scale > 1.0e-12, action_scale, 1.0)
    action_standardized = (action - action_mean) / action_scale
    action_features = np.concatenate([condition, action_standardized], axis=1)
    history_action = grouped_holdout_ridge_predictions(
        features=action_features,
        targets=flat_target,
        groups=groups,
        fit_rows=fit_rows,
        evaluation_rows=eval_index,
        regularization=spec.ridge_regularization,
    ).reshape(len(eval_index), future.shape[1], ROBOT_PROXY_DIM)
    predictions["history_plus_current_action_ridge_nondeployable"] = history_action

    metrics = {
        name: _robot_metric_payload(prediction, target_robot_z, active_robot)
        for name, prediction in predictions.items()
    }
    baseline_best_name = min(
        ("last_observation", "constant_velocity", "history_ridge"),
        key=lambda name: metrics[name]["z_mse"],
    )
    deployable_best = metrics[baseline_best_name]["z_mse"]
    action_mse = metrics["history_plus_current_action_ridge_nondeployable"]["z_mse"]
    return {
        "schema": ROBOT_BASELINE_SCHEMA,
        "evaluation_row_count": int(len(eval_index)),
        "group_holdout": True,
        "action_conditioned_baseline_deployable": False,
        "baselines": metrics,
        "best_deployable_baseline": baseline_best_name,
        "action_conditioned_gain": _safe_ratio(deployable_best - action_mse, deployable_best),
        "action_conditioned_underdetermination_supported": bool(
            _safe_ratio(deployable_best - action_mse, deployable_best)
            >= spec.action_conditioned_gain_min
        ),
        "_prediction_z": predictions,
        "_target_robot_z": target_robot_z,
    }


def model_robot_error_audit(
    *,
    prediction_z: Any,
    prediction_raw: Any,
    target_z: Any,
    target_raw: Any,
    future_active: Any,
    history_raw: Any,
) -> Dict[str, Any]:
    pred_z = _require_future(prediction_z, name="prediction_z")
    pred_raw = _require_future(prediction_raw, name="prediction_raw")
    truth_z = _require_future(target_z, name="target_z")
    truth_raw = _require_future(target_raw, name="target_raw")
    history = _require_state_history(history_raw, name="history_raw")
    if not (pred_z.shape == pred_raw.shape == truth_z.shape == truth_raw.shape):
        raise ValueError("prediction/target shape mismatch")
    active = np.asarray(future_active, dtype=bool)
    active_robot = active[:, CABLE_DIM:]
    metrics = _robot_metric_payload(
        pred_z[..., CABLE_DIM:], truth_z[..., CABLE_DIM:], active_robot
    )
    raw_squared = (pred_raw[..., CABLE_DIM:] - truth_raw[..., CABLE_DIM:]) ** 2
    metrics["raw_mse"] = float(np.mean(raw_squared[:, active_robot]))
    metrics["raw_rmse"] = float(np.sqrt(metrics["raw_mse"]))
    reference = history[:, -1, CABLE_DIM + 35:CABLE_DIM + 39]
    metrics["quaternion"] = quaternion_error_metrics(
        pred_raw[..., CABLE_DIM + 35:CABLE_DIM + 39],
        truth_raw[..., CABLE_DIM + 35:CABLE_DIM + 39],
        reference=reference,
    )
    per_dimension = []
    error = (pred_z[..., CABLE_DIM:] - truth_z[..., CABLE_DIM:]) ** 2
    for dimension in range(ROBOT_PROXY_DIM):
        mask = active_robot[:, dimension]
        values = error[:, mask, dimension]
        per_dimension.append({
            "robot_dimension": dimension,
            "z_mse": float(np.mean(values)) if values.size else 0.0,
            "z_rmse": float(np.sqrt(np.mean(values))) if values.size else 0.0,
        })
    metrics["per_dimension"] = per_dimension
    metrics["schema"] = ROBOT_MODEL_ERROR_SCHEMA
    return metrics


def hybrid_counterfactual_audit(
    *,
    prediction_z: Any,
    target_z: Any,
    baseline_robot_z: Any,
    future_active: Any,
    z_mse_threshold: float,
) -> Dict[str, Any]:
    prediction = _require_future(prediction_z, name="prediction_z")
    target = _require_future(target_z, name="target_z")
    baseline = _require_robot(baseline_robot_z, name="baseline_robot_z")
    if prediction.shape != target.shape or baseline.shape[:2] != target.shape[:2]:
        raise ValueError("hybrid shape mismatch")
    active = np.asarray(future_active, dtype=bool)
    variants = {
        "model_full": prediction.copy(),
        "model_cable_oracle_robot": prediction.copy(),
        "oracle_cable_model_robot": target.copy(),
        "model_cable_baseline_robot": prediction.copy(),
    }
    variants["model_cable_oracle_robot"][..., CABLE_DIM:] = target[..., CABLE_DIM:]
    variants["oracle_cable_model_robot"][..., CABLE_DIM:] = prediction[..., CABLE_DIM:]
    variants["model_cable_baseline_robot"][..., CABLE_DIM:] = baseline
    payload = {}
    for name, value in variants.items():
        mse = float(np.mean((value[:, active] - target[:, active]) ** 2))
        payload[name] = {
            "z_mse": mse,
            "exact_z_gate_pass": bool(mse <= float(z_mse_threshold)),
        }
    full = payload["model_full"]["z_mse"]
    oracle_robot = payload["model_cable_oracle_robot"]["z_mse"]
    return {
        "schema": ROBOT_HYBRID_SCHEMA,
        "variants": payload,
        "oracle_robot_error_reduction_fraction": _safe_ratio(full - oracle_robot, full),
        "robot_proxy_drives_full_z_failure": bool(
            not payload["model_full"]["exact_z_gate_pass"]
            and payload["model_cable_oracle_robot"]["exact_z_gate_pass"]
        ),
    }


def grouped_holdout_cross_feature_predictions(
    *,
    train_features: Any,
    evaluation_features: Any,
    targets: Any,
    groups: Sequence[Any],
    fit_rows: Sequence[int],
    evaluation_rows: Sequence[int],
    regularization: float,
) -> np.ndarray:
    train_x = _as_numpy(train_features, name="train_features")
    eval_x = _as_numpy(evaluation_features, name="evaluation_features")
    y = _as_numpy(targets, name="targets")
    group_array = np.asarray(groups)
    fit_index = np.asarray(fit_rows, dtype=np.int64)
    eval_index = np.asarray(evaluation_rows, dtype=np.int64)
    if train_x.shape != eval_x.shape or train_x.shape[0] != y.shape[0]:
        raise ValueError("cross-feature row/shape mismatch")
    output = np.empty((len(eval_index), y.shape[1]), dtype=np.float64)
    for group in np.unique(group_array[eval_index]):
        eval_local = np.where(group_array[eval_index] == group)[0]
        train_rows = fit_index[group_array[fit_index] != group]
        if train_rows.size < 2:
            raise ValueError(f"insufficient holdout rows for group {group!r}")
        output[eval_local] = ridge_fit_predict(
            train_x[train_rows],
            y[train_rows],
            eval_x[eval_index[eval_local]],
            regularization=regularization,
        )
    return output


def action_sensitivity_probe(
    *,
    condition_features: Any,
    oracle_future_z: Any,
    model_future_z_for_evaluation: Any,
    action_target: Any,
    groups: Sequence[Any],
    fit_rows: Sequence[int],
    evaluation_rows: Sequence[int],
    spec: RobotProxyAuditSpec,
) -> Dict[str, Any]:
    condition = _as_numpy(condition_features, name="condition_features")
    oracle = _require_future(oracle_future_z, name="oracle_future_z")
    model_eval = _require_future(
        model_future_z_for_evaluation, name="model_future_z_for_evaluation"
    )
    eval_index = np.asarray(evaluation_rows, dtype=np.int64)
    if model_eval.shape[0] != len(eval_index) or model_eval.shape[1:] != oracle.shape[1:]:
        raise ValueError("model evaluation futures must be [len(evaluation_rows),T,87]")
    if condition.shape[0] != oracle.shape[0]:
        raise ValueError("condition/oracle row count mismatch")
    action = _as_numpy(action_target, name="action_target")
    fit_index = np.asarray(fit_rows, dtype=np.int64)
    action_mean = np.mean(action[fit_index], axis=0, keepdims=True)
    action_scale = np.std(action[fit_index], axis=0, keepdims=True)
    action_scale = np.where(action_scale > 1.0e-12, action_scale, 1.0)
    action_z = (action - action_mean) / action_scale

    oracle_first = oracle[:, 0]
    model_first = model_eval[:, 0]
    oracle_feature_sets = {
        "history_only": condition,
        "history_plus_oracle_cable": np.concatenate(
            [condition, oracle_first[:, :CABLE_DIM]], axis=1
        ),
        "history_plus_oracle_robot": np.concatenate(
            [condition, oracle_first[:, CABLE_DIM:]], axis=1
        ),
        "history_plus_oracle_full": np.concatenate([condition, oracle_first], axis=1),
    }
    model_test_sets = {
        "history_only": condition[eval_index],
        "history_plus_oracle_cable": np.concatenate(
            [condition[eval_index], model_first[:, :CABLE_DIM]], axis=1
        ),
        "history_plus_oracle_robot": np.concatenate(
            [condition[eval_index], model_first[:, CABLE_DIM:]], axis=1
        ),
        "history_plus_oracle_full": np.concatenate(
            [condition[eval_index], model_first], axis=1
        ),
    }
    metrics: Dict[str, Any] = {}
    for name, train_features in oracle_feature_sets.items():
        oracle_prediction = grouped_holdout_cross_feature_predictions(
            train_features=train_features,
            evaluation_features=train_features,
            targets=action_z,
            groups=groups,
            fit_rows=fit_rows,
            evaluation_rows=eval_index,
            regularization=spec.ridge_regularization,
        )
        # Build an all-row evaluation matrix only to satisfy the shared fold helper;
        # rows outside evaluation_rows are never read from this matrix.
        model_features_all = train_features.copy()
        model_features_all[eval_index] = model_test_sets[name]
        model_prediction = grouped_holdout_cross_feature_predictions(
            train_features=train_features,
            evaluation_features=model_features_all,
            targets=action_z,
            groups=groups,
            fit_rows=fit_rows,
            evaluation_rows=eval_index,
            regularization=spec.ridge_regularization,
        )
        truth = action_z[eval_index]
        metrics[name] = {
            "oracle_feature_normalized_mse": float(
                np.mean((oracle_prediction - truth) ** 2)
            ),
            "model_feature_normalized_mse": float(
                np.mean((model_prediction - truth) ** 2)
            ),
        }
    history = metrics["history_only"]["oracle_feature_normalized_mse"]
    oracle_robot = metrics["history_plus_oracle_robot"][
        "oracle_feature_normalized_mse"
    ]
    model_robot = metrics["history_plus_oracle_robot"][
        "model_feature_normalized_mse"
    ]
    oracle_cable = metrics["history_plus_oracle_cable"][
        "oracle_feature_normalized_mse"
    ]
    denominator = history - oracle_robot
    return {
        "schema": ROBOT_ACTION_SCHEMA,
        "diagnostic_only": True,
        "not_formal_idm": True,
        "decoder_trained_on_oracle_train_features": True,
        "features_use_first_future_horizon": True,
        "metrics": metrics,
        "oracle_robot_incremental_gain": _safe_ratio(history - oracle_robot, history),
        "oracle_cable_incremental_gain": _safe_ratio(history - oracle_cable, history),
        "model_robot_realized_fraction": _safe_ratio(history - model_robot, denominator),
        "robot_proxy_material_for_action_probe": bool(
            _safe_ratio(history - oracle_robot, history)
            >= spec.action_robot_incremental_gain_min
        ),
    }

def evaluate_synthetic_controls(spec: RobotProxyAuditSpec) -> Dict[str, Any]:
    rng = np.random.default_rng(1234)
    target = rng.normal(size=(6, 4, ROBOT_PROXY_DIM))
    target[..., ROBOT_EE_ORIENTATION_SLICE] = normalize_quaternion(
        target[..., ROBOT_EE_ORIENTATION_SLICE]
    )
    sign_flip = target.copy()
    sign_flip[:, 1::2, ROBOT_EE_ORIENTATION_SLICE] *= -1.0
    quaternion = quaternion_error_metrics(
        sign_flip[..., ROBOT_EE_ORIENTATION_SLICE],
        target[..., ROBOT_EE_ORIENTATION_SLICE],
    )
    padding = np.zeros((6, 3, STATE_DIM))
    future = np.zeros((6, 4, STATE_DIM))
    future[..., CABLE_DIM + 0] = np.linspace(0.0, 1.0, 24).reshape(6, 4)
    active = np.ones((4, STATE_DIM), dtype=bool)
    schema = robot_proxy_schema_audit(
        history_raw=padding,
        future_raw=future,
        future_active=active,
        spec=spec,
    )
    return {
        "quaternion_sign_flip": {
            "direct_mse_positive": bool(quaternion["direct_mse"] > 0.1),
            "sign_invariant_mse_zero": bool(quaternion["sign_invariant_mse"] < 1.0e-12),
            "pass": bool(
                quaternion["direct_mse"] > 0.1
                and quaternion["sign_invariant_mse"] < 1.0e-12
            ),
        },
        "structural_padding": {
            "detected": bool(len(schema["structural_padding_dimensions"]) >= 38),
            "pass": bool(len(schema["structural_padding_dimensions"]) >= 38),
        },
        "pass": bool(
            quaternion["direct_mse"] > 0.1
            and quaternion["sign_invariant_mse"] < 1.0e-12
            and len(schema["structural_padding_dimensions"]) >= 38
        ),
    }


def _variant_mechanisms(value: Mapping[str, Any], spec: RobotProxyAuditSpec) -> List[str]:
    mechanisms: List[str] = []
    schema = value.get("schema_audit", {})
    model = value.get("source_mean_robot_error", value.get("robot_error", {}))
    baselines = value.get("predictability_baselines", {})
    hybrid = value.get("hybrid_counterfactual", {})
    action = value.get("action_sensitivity", {})
    if schema.get("structural_padding_dimensions") or schema.get("intermittent_zero_dimensions"):
        mechanisms.append("zero_padding_or_presence_mask_ambiguity")
    quaternion = model.get("quaternion", {})
    if (
        float(quaternion.get("target_consecutive_sign_flip_fraction", 0.0))
        >= spec.quaternion_sign_ambiguity_fraction_min
        and float(quaternion.get("sign_invariant_gain", 0.0))
        >= spec.quaternion_sign_invariant_gain_min
    ):
        mechanisms.append("quaternion_double_cover_error_inflation")
    if bool(baselines.get("action_conditioned_underdetermination_supported")):
        mechanisms.append("action_conditioned_robot_future_underdetermination")
    baseline_name = baselines.get("best_deployable_baseline")
    baseline = baselines.get("baselines", {}).get(baseline_name, {}) if baseline_name else {}
    if baseline and float(baseline.get("z_mse", np.inf)) <= spec.simple_baseline_sufficiency_ratio * float(model.get("z_mse", np.inf)):
        mechanisms.append("simple_robot_baseline_matches_diffusion")
    if bool(hybrid.get("robot_proxy_drives_full_z_failure")):
        mechanisms.append("robot_proxy_drives_full_state_gate_failure")
    if bool(action.get("robot_proxy_material_for_action_probe")):
        mechanisms.append("robot_proxy_material_for_action_sensitivity")
    return mechanisms


def classify_robot_proxy_attribution(report: Mapping[str, Any]) -> Dict[str, Any]:
    spec_payload = report.get("spec", {})
    spec = RobotProxyAuditSpec(**{
        key: spec_payload.get(key, getattr(RobotProxyAuditSpec(), key))
        for key in asdict(RobotProxyAuditSpec())
    })
    variants = report.get("variants")
    if not isinstance(variants, Mapping) or set(variants) != set(DIAGNOSTIC_OBJECTIVE_NAMES):
        root = "phase314b_r254_diagnostic_matrix_incomplete"
        return _classification(root, [], "repair diagnostic matrix")
    if not all(bool(value.get("training_completed")) for value in variants.values()):
        root = "phase314b_r254_training_reproduction_failed"
        return _classification(root, [], "repair deterministic training reproduction")
    if not all(bool(value.get("r253_reproduction_pass")) for value in variants.values()):
        root = "phase314b_r254_r253_contract_reproduction_failed"
        return _classification(root, [], "repair r2.5.3 contract reproduction")
    controls = report.get("synthetic_controls", {})
    if not bool(controls.get("pass")):
        root = "phase314b_r254_robot_proxy_controls_failed"
        return _classification(root, [], "repair robot-proxy synthetic controls")
    schema = report.get("schema_audit", {})
    if not bool(schema.get("schema_contract_pass")):
        root = "phase314b_r254_robot_proxy_schema_invalid"
        return _classification(root, [], "repair robot-proxy schema contract")

    per_variant = {
        name: _variant_mechanisms(value, spec)
        for name, value in variants.items()
    }
    counts: Dict[str, int] = {}
    for mechanisms in per_variant.values():
        for mechanism in mechanisms:
            counts[mechanism] = counts.get(mechanism, 0) + 1
    supported = sorted(
        mechanism
        for mechanism, count in counts.items()
        if count >= spec.minimum_models_for_mechanism
    )
    if "quaternion_double_cover_error_inflation" in supported:
        root = "phase314b_r254_quaternion_representation_ambiguity_supported"
        next_stage = "canonicalize quaternion targets and re-audit robot-only fidelity without changing cable contracts"
    elif "zero_padding_or_presence_mask_ambiguity" in supported:
        root = "phase314b_r254_robot_proxy_presence_contract_deficient"
        next_stage = "add an explicit robot-proxy presence/schema audit before any robot objective"
    elif "action_conditioned_robot_future_underdetermination" in supported:
        root = "phase314b_r254_action_conditioned_robot_future_supported"
        next_stage = "audit robot-future conditioning on deployable action/history inputs without changing cable training"
    elif "simple_robot_baseline_matches_diffusion" in supported:
        root = "phase314b_r254_simple_robot_baseline_sufficient"
        next_stage = "separate robot proxy from diffusion target and retain a deterministic robot baseline candidate"
    elif "robot_proxy_material_for_action_sensitivity" in supported:
        root = "phase314b_r254_robot_proxy_action_value_model_gap_supported"
        next_stage = "audit a robot-only trajectory head under the frozen cable branch contract"
    elif "robot_proxy_drives_full_state_gate_failure" in supported:
        root = "phase314b_r254_robot_proxy_model_fidelity_gap_supported"
        next_stage = "audit a robot-only trajectory head under the frozen cable branch contract"
    else:
        root = "phase314b_r254_robot_proxy_failure_unattributed"
        next_stage = "inspect field-level robot-proxy data provenance before changing the model"
    result = _classification(root, supported, next_stage)
    result["variant_supported_mechanisms"] = per_variant
    result["mechanism_counts"] = counts
    return result


def _classification(root: str, mechanisms: Sequence[str], next_stage: str) -> Dict[str, Any]:
    return {
        "root_cause": root,
        "supported_mechanisms": list(mechanisms),
        "next_stage": next_stage,
        "train_only_recommendation": None,
        "selected_configuration": None,
    }


def strip_runtime_objects(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        raise TypeError("runtime tensor must not be serialized")
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        output: MutableMapping[str, Any] = {}
        for key, item in value.items():
            if str(key).startswith("_"):
                continue
            output[str(key)] = strip_runtime_objects(item)
        return dict(output)
    if isinstance(value, (list, tuple)):
        return [strip_runtime_objects(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"unsupported serializable type: {type(value)!r}")
