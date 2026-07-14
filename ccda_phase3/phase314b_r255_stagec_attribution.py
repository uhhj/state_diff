"""Train-only state-v3 robot-future predictability and IDM-materiality audit.

This is a data-side audit.  It does not train or evaluate diffusion models.
All supervised predictions are deterministic out-of-fold ridge regressions with
entire paired episode groups kept in one fold.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from ccda_phase3.schema_v3 import (
    ACTION_DIM,
    CABLE_DIM,
    DEFAULT_TF,
    DEFAULT_TH,
    EE_POSITION_SLICE,
    EE_QUATERNION_SLICE,
    JOINT_POSITION_SLICE,
    JOINT_VELOCITY_SLICE,
    ROBOT_EE_POSITION_SLICE,
    ROBOT_EE_QUATERNION_SLICE,
    ROBOT_JOINT_POSITION_SLICE,
    ROBOT_JOINT_VELOCITY_SLICE,
    ROBOT_PROXY_DIM,
    STATE_ACTION_X_DIM,
    STATE_DIM,
)

ATTRIBUTION_SCHEMA = "phase314b_r255_stagec_state_v3_attribution_v1"


class StageCAttributionError(RuntimeError):
    """Raised when the train-only attribution contract cannot be evaluated."""


@dataclass(frozen=True)
class AttributionSpec:
    folds: int = 8
    ridge_regularization: float = 1.0e-3
    variance_epsilon: float = 1.0e-10
    zero_epsilon: float = 1.0e-8
    deployable_nmse_max: float = 0.50
    action_conditioned_gain_min: float = 0.20
    robot_action_incremental_gain_min: float = 0.05
    simple_baseline_ratio: float = 1.05
    quaternion_angle_p95_max_rad: float = 0.35

    def validate(self) -> None:
        if self.folds < 2:
            raise ValueError("attribution requires at least two folds")
        if self.ridge_regularization <= 0.0:
            raise ValueError("ridge regularization must be positive")
        if not 0.0 < self.deployable_nmse_max:
            raise ValueError("deployable NMSE threshold must be positive")
        if not 0.0 <= self.robot_action_incremental_gain_min < 1.0:
            raise ValueError("robot action-gain threshold is invalid")


def _finite(value: Any, *, name: str, dtype: Any = np.float64) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if not np.all(np.isfinite(array)):
        raise StageCAttributionError(f"{name} contains NaN or Inf")
    return array


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / max(abs(float(denominator)), 1.0e-12))


def _stats(value: Any) -> Dict[str, float]:
    array = _finite(value, name="statistics input").reshape(-1)
    if array.size == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "p05": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "p05": float(np.percentile(array, 5.0)),
        "p50": float(np.percentile(array, 50.0)),
        "p95": float(np.percentile(array, 95.0)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def normalize_quaternion(value: Any) -> np.ndarray:
    array = _finite(value, name="quaternion")
    if array.shape[-1] != 4:
        raise StageCAttributionError("quaternion last dimension must be four")
    norm = np.linalg.norm(array, axis=-1, keepdims=True)
    fallback = np.zeros_like(array)
    fallback[..., 3] = 1.0
    result = np.where(
        norm > 1.0e-12,
        array / np.maximum(norm, 1.0e-12),
        fallback,
    )
    flip = result[..., 3] < 0.0
    result[flip] *= -1.0
    tie = np.abs(result[..., 3]) <= 1.0e-12
    for index in np.argwhere(tie):
        row = result[tuple(index)]
        for component in row[:3]:
            if abs(float(component)) > 1.0e-12:
                if component < 0.0:
                    result[tuple(index)] *= -1.0
                break
    return result


def group_fold_assignment(
    groups: Sequence[Any],
    *,
    folds: int,
) -> Tuple[np.ndarray, Dict[str, int]]:
    group_array = np.asarray(groups).astype(str)
    unique = sorted(set(group_array.tolist()))
    if len(unique) < folds:
        raise StageCAttributionError(
            f"group count {len(unique)} is smaller than fold count {folds}"
        )
    counts = {
        group: int(np.sum(group_array == group)) for group in unique
    }
    fold_rows = [0 for _ in range(int(folds))]
    mapping: Dict[str, int] = {}
    for group in sorted(unique, key=lambda item: (-counts[item], item)):
        fold = min(range(int(folds)), key=lambda index: (fold_rows[index], index))
        mapping[group] = int(fold)
        fold_rows[fold] += counts[group]
    assignment = np.asarray([mapping[group] for group in group_array], dtype=np.int64)
    for group in unique:
        observed = set(assignment[group_array == group].tolist())
        if len(observed) != 1:
            raise AssertionError("a paired episode group was split across folds")
    return assignment, mapping


def ridge_fit_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    eval_x: np.ndarray,
    *,
    regularization: float,
) -> np.ndarray:
    x_train = _finite(train_x, name="ridge train_x")
    y_train = _finite(train_y, name="ridge train_y")
    x_eval = _finite(eval_x, name="ridge eval_x")
    if x_train.ndim != 2 or y_train.ndim != 2 or x_eval.ndim != 2:
        raise StageCAttributionError("ridge arrays must be rank two")
    if x_train.shape[0] != y_train.shape[0]:
        raise StageCAttributionError("ridge train row count mismatch")
    if x_train.shape[1] != x_eval.shape[1]:
        raise StageCAttributionError("ridge feature width mismatch")
    if x_train.shape[0] < 2:
        raise StageCAttributionError("ridge requires at least two training rows")

    x_mean = np.mean(x_train, axis=0, keepdims=True)
    x_scale = np.std(x_train, axis=0, keepdims=True)
    x_active = x_scale[0] > 1.0e-12
    if not np.any(x_active):
        return np.repeat(np.mean(y_train, axis=0, keepdims=True), x_eval.shape[0], axis=0)
    x_scale = np.where(x_scale > 1.0e-12, x_scale, 1.0)
    xz = (x_train[:, x_active] - x_mean[:, x_active]) / x_scale[:, x_active]
    ez = (x_eval[:, x_active] - x_mean[:, x_active]) / x_scale[:, x_active]

    y_mean = np.mean(y_train, axis=0, keepdims=True)
    y_scale = np.std(y_train, axis=0, keepdims=True)
    y_active = y_scale[0] > 1.0e-12
    y_scale = np.where(y_scale > 1.0e-12, y_scale, 1.0)
    yz = (y_train - y_mean) / y_scale

    gram = xz.T @ xz
    gram.flat[:: gram.shape[0] + 1] += float(regularization)
    weights = np.linalg.solve(gram, xz.T @ yz)
    prediction_z = ez @ weights
    prediction = prediction_z * y_scale + y_mean
    prediction[:, ~y_active] = y_mean[:, ~y_active]
    return prediction


def grouped_oof_regression(
    *,
    features: np.ndarray,
    targets: np.ndarray,
    groups: Sequence[Any],
    folds: int,
    regularization: float,
    valid_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    x = _finite(features, name="OOF features")
    y = _finite(targets, name="OOF targets")
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise StageCAttributionError("OOF feature/target shape mismatch")
    assignment, mapping = group_fold_assignment(groups, folds=folds)
    valid = (
        np.ones(x.shape[0], dtype=np.bool_)
        if valid_mask is None
        else np.asarray(valid_mask, dtype=np.bool_)
    )
    if valid.shape != (x.shape[0],):
        raise StageCAttributionError("OOF valid mask shape mismatch")
    prediction = np.full(y.shape, np.nan, dtype=np.float64)
    fold_records: List[Dict[str, Any]] = []
    for fold in range(int(folds)):
        evaluation = (assignment == fold) & valid
        training = (assignment != fold) & valid
        if int(np.sum(evaluation)) == 0:
            raise StageCAttributionError(f"fold {fold} has no evaluation rows")
        if int(np.sum(training)) < 2:
            raise StageCAttributionError(f"fold {fold} has insufficient training rows")
        prediction[evaluation] = ridge_fit_predict(
            x[training],
            y[training],
            x[evaluation],
            regularization=regularization,
        )
        fold_records.append(
            {
                "fold": fold,
                "training_rows": int(np.sum(training)),
                "evaluation_rows": int(np.sum(evaluation)),
                "evaluation_groups": int(
                    len(set(np.asarray(groups).astype(str)[evaluation].tolist()))
                ),
            }
        )
    if np.any(~np.isfinite(prediction[valid])):
        raise StageCAttributionError("OOF predictions are incomplete")
    return prediction, {
        "folds": int(folds),
        "group_count": len(mapping),
        "group_to_fold": mapping,
        "fold_records": fold_records,
        "group_integrity_pass": True,
    }


def grouped_oof_future_robot(
    *,
    features: np.ndarray,
    robot_future: np.ndarray,
    future_valid_mask: np.ndarray,
    groups: Sequence[Any],
    spec: AttributionSpec,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    x = _finite(features, name="future features")
    target = _finite(robot_future, name="robot future")
    mask = np.asarray(future_valid_mask, dtype=np.bool_)
    if target.ndim != 3 or target.shape[1:] != (DEFAULT_TF, ROBOT_PROXY_DIM):
        raise StageCAttributionError("robot future must be [N,4,19]")
    if mask.shape != target.shape[:2]:
        raise StageCAttributionError("future valid mask shape mismatch")
    prediction = np.full(target.shape, np.nan, dtype=np.float64)
    horizon_records = []
    fold_reference: Optional[Dict[str, Any]] = None
    for horizon in range(DEFAULT_TF):
        horizon_prediction, fold_record = grouped_oof_regression(
            features=x,
            targets=target[:, horizon],
            groups=groups,
            folds=spec.folds,
            regularization=spec.ridge_regularization,
            valid_mask=mask[:, horizon],
        )
        prediction[:, horizon] = horizon_prediction
        horizon_records.append(
            {
                "horizon": horizon,
                "valid_rows": int(np.sum(mask[:, horizon])),
                "folds": fold_record["fold_records"],
            }
        )
        if fold_reference is None:
            fold_reference = fold_record
    # Padded future rows are excluded by ``future_valid_mask``.  Fill them
    # with their stored repeated target before enforcing quaternion geometry so
    # no NaN enters the serialized diagnostic state.
    prediction[~mask] = target[~mask]
    prediction[..., ROBOT_EE_QUATERNION_SLICE] = normalize_quaternion(
        prediction[..., ROBOT_EE_QUATERNION_SLICE]
    )
    return prediction, {
        "group_count": int(fold_reference["group_count"] if fold_reference else 0),
        "folds": spec.folds,
        "horizons": horizon_records,
        "group_integrity_pass": True,
    }


def last_observation_baseline(
    robot_history: np.ndarray,
    *,
    future_steps: int = DEFAULT_TF,
) -> np.ndarray:
    history = _finite(robot_history, name="robot history")
    if history.ndim != 3 or history.shape[1:] != (DEFAULT_TH, ROBOT_PROXY_DIM):
        raise StageCAttributionError("robot history must be [N,3,19]")
    return np.repeat(history[:, -1:, :], int(future_steps), axis=1)


def constant_velocity_baseline(
    robot_history: np.ndarray,
    *,
    future_steps: int = DEFAULT_TF,
) -> np.ndarray:
    history = _finite(robot_history, name="robot history")
    if history.ndim != 3 or history.shape[1:] != (DEFAULT_TH, ROBOT_PROXY_DIM):
        raise StageCAttributionError("robot history must be [N,3,19]")
    last = history[:, -1]
    previous = history[:, -2]
    delta = last - previous
    result = np.repeat(last[:, None, :], int(future_steps), axis=1)
    for horizon in range(int(future_steps)):
        step = float(horizon + 1)
        result[:, horizon, ROBOT_JOINT_POSITION_SLICE] = (
            last[:, ROBOT_JOINT_POSITION_SLICE]
            + step * delta[:, ROBOT_JOINT_POSITION_SLICE]
        )
        result[:, horizon, ROBOT_JOINT_VELOCITY_SLICE] = (
            last[:, ROBOT_JOINT_VELOCITY_SLICE]
        )
        result[:, horizon, ROBOT_EE_POSITION_SLICE] = (
            last[:, ROBOT_EE_POSITION_SLICE]
            + step * delta[:, ROBOT_EE_POSITION_SLICE]
        )
        result[:, horizon, ROBOT_EE_QUATERNION_SLICE] = (
            last[:, ROBOT_EE_QUATERNION_SLICE]
        )
    result[..., ROBOT_EE_QUATERNION_SLICE] = normalize_quaternion(
        result[..., ROBOT_EE_QUATERNION_SLICE]
    )
    return result


def robot_target_variance(
    target: np.ndarray,
    valid_mask: np.ndarray,
    *,
    epsilon: float,
) -> Tuple[np.ndarray, np.ndarray]:
    future = _finite(target, name="robot target")
    mask = np.asarray(valid_mask, dtype=np.bool_)
    variance = np.zeros((DEFAULT_TF, ROBOT_PROXY_DIM), dtype=np.float64)
    active = np.zeros((DEFAULT_TF, ROBOT_PROXY_DIM), dtype=np.bool_)
    for horizon in range(DEFAULT_TF):
        selected = future[mask[:, horizon], horizon]
        variance[horizon] = np.var(selected, axis=0)
        active[horizon] = variance[horizon] > float(epsilon)
    return variance, active


def quaternion_error_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    valid_mask: np.ndarray,
) -> Dict[str, Any]:
    pred = normalize_quaternion(prediction)
    truth = normalize_quaternion(target)
    mask = np.asarray(valid_mask, dtype=np.bool_)
    pred_q = pred[mask]
    truth_q = truth[mask]
    dot = np.sum(pred_q * truth_q, axis=-1)
    dot_abs = np.clip(np.abs(dot), 0.0, 1.0)
    angle = 2.0 * np.arccos(dot_abs)
    direct = np.mean((pred_q - truth_q) ** 2, axis=-1)
    invariant = np.minimum(
        np.mean((pred_q - truth_q) ** 2, axis=-1),
        np.mean((pred_q + truth_q) ** 2, axis=-1),
    )
    return {
        "geodesic_angle_rad": _stats(angle),
        "direct_mse": float(np.mean(direct)),
        "sign_invariant_mse": float(np.mean(invariant)),
        "prediction_norm_error": _stats(
            np.abs(np.linalg.norm(pred_q, axis=-1) - 1.0)
        ),
        "target_norm_error": _stats(
            np.abs(np.linalg.norm(truth_q, axis=-1) - 1.0)
        ),
    }


def robot_prediction_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    valid_mask: np.ndarray,
    *,
    variance_epsilon: float,
) -> Dict[str, Any]:
    pred = _finite(prediction, name="robot prediction")
    truth = _finite(target, name="robot truth")
    mask = np.asarray(valid_mask, dtype=np.bool_)
    if pred.shape != truth.shape or pred.shape[1:] != (DEFAULT_TF, ROBOT_PROXY_DIM):
        raise StageCAttributionError("robot prediction shape mismatch")
    variance, active = robot_target_variance(
        truth,
        mask,
        epsilon=variance_epsilon,
    )
    squared = (pred - truth) ** 2
    normalized_values = []
    per_dimension = []
    for horizon in range(DEFAULT_TF):
        horizon_valid = mask[:, horizon]
        for dimension in range(ROBOT_PROXY_DIM):
            values = squared[horizon_valid, horizon, dimension]
            mse = float(np.mean(values))
            is_active = bool(active[horizon, dimension])
            if is_active:
                normalized_values.append(mse / variance[horizon, dimension])
            per_dimension.append(
                {
                    "horizon": horizon,
                    "robot_dimension": dimension,
                    "active": is_active,
                    "variance": float(variance[horizon, dimension]),
                    "raw_mse": mse,
                    "normalized_mse": (
                        float(mse / variance[horizon, dimension])
                        if is_active
                        else 0.0
                    ),
                }
            )
    valid_squared = squared[mask]
    group_slices = {
        "joint_position": ROBOT_JOINT_POSITION_SLICE,
        "joint_velocity": ROBOT_JOINT_VELOCITY_SLICE,
        "ee_position": ROBOT_EE_POSITION_SLICE,
    }
    groups = {}
    for name, group_slice in group_slices.items():
        group_error = squared[..., group_slice][mask]
        groups[name] = {
            "raw_mse": float(np.mean(group_error)),
            "raw_rmse": float(np.sqrt(np.mean(group_error))),
        }
    quaternion = quaternion_error_metrics(
        pred[..., ROBOT_EE_QUATERNION_SLICE],
        truth[..., ROBOT_EE_QUATERNION_SLICE],
        mask,
    )
    return {
        "normalized_mse": (
            float(np.mean(normalized_values))
            if normalized_values
            else 0.0
        ),
        "raw_mse": float(np.mean(valid_squared)),
        "raw_rmse": float(np.sqrt(np.mean(valid_squared))),
        "active_horizon_dimensions": int(np.sum(active)),
        "total_horizon_dimensions": int(active.size),
        "groups": groups,
        "quaternion": quaternion,
        "per_dimension": per_dimension,
    }


def schema_audit(
    robot_history: np.ndarray,
    robot_future: np.ndarray,
    future_valid_mask: np.ndarray,
    *,
    spec: AttributionSpec,
) -> Dict[str, Any]:
    history = _finite(robot_history, name="robot history")
    future = _finite(robot_future, name="robot future")
    mask = np.asarray(future_valid_mask, dtype=np.bool_)
    if history.shape[1:] != (DEFAULT_TH, ROBOT_PROXY_DIM):
        raise StageCAttributionError("robot history schema mismatch")
    if future.shape[1:] != (DEFAULT_TF, ROBOT_PROXY_DIM):
        raise StageCAttributionError("robot future schema mismatch")
    combined = np.concatenate(
        [history.reshape(-1, ROBOT_PROXY_DIM), future[mask]],
        axis=0,
    )
    dimensions = []
    near_zero_variance = []
    for dimension in range(ROBOT_PROXY_DIM):
        values = combined[:, dimension]
        variance = float(np.var(values))
        zero_fraction = float(np.mean(np.abs(values) <= spec.zero_epsilon))
        if variance <= spec.variance_epsilon:
            near_zero_variance.append(dimension)
        dimensions.append(
            {
                "robot_dimension": dimension,
                "variance": variance,
                "zero_fraction": zero_fraction,
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
            }
        )
    quaternion = np.concatenate(
        [
            history[..., ROBOT_EE_QUATERNION_SLICE].reshape(-1, 4),
            future[..., ROBOT_EE_QUATERNION_SLICE][mask],
        ],
        axis=0,
    )
    norm_error = np.abs(np.linalg.norm(quaternion, axis=-1) - 1.0)
    canonical = bool(
        np.all(quaternion[:, 3] >= -1.0e-12)
        and float(np.max(norm_error)) <= 1.0e-5
    )
    return {
        "schema": "phase314b_r255_stagec_state_v3_schema_audit_v1",
        "state_dim": STATE_DIM,
        "cable_dim": CABLE_DIM,
        "robot_proxy_dim": ROBOT_PROXY_DIM,
        "layout": {
            "joint_position": [0, 6],
            "joint_velocity": [6, 12],
            "ee_position": [12, 15],
            "ee_quaternion": [15, 19],
        },
        "dimension_records": dimensions,
        "near_zero_variance_dimensions": near_zero_variance,
        "quaternion_norm_error": _stats(norm_error),
        "quaternion_canonical": canonical,
        "padding_values_used": False,
        "missing_values_used": False,
        "schema_contract_pass": bool(
            STATE_DIM == 67
            and CABLE_DIM == 48
            and ROBOT_PROXY_DIM == 19
            and canonical
        ),
    }


def action_prediction_probe(
    *,
    paper_x: np.ndarray,
    state_action_x: np.ndarray,
    robot_history: np.ndarray,
    robot_future: np.ndarray,
    cable_future: np.ndarray,
    action_target: np.ndarray,
    groups: Sequence[Any],
    spec: AttributionSpec,
) -> Dict[str, Any]:
    history = _finite(paper_x, name="paper_x")
    deployable_history = _finite(
        state_action_x, name="state_action_x"
    )
    robot_hist = _finite(robot_history, name="robot_history")
    robot = _finite(robot_future, name="robot_future")
    cable = _finite(cable_future, name="cable_future")
    action = _finite(action_target, name="action_target")
    if history.shape[1] != DEFAULT_TH * STATE_DIM:
        raise StageCAttributionError("paper_x width changed")
    if deployable_history.shape[1] != STATE_ACTION_X_DIM:
        raise StageCAttributionError("state_action_x width changed")
    current_robot = robot_hist[:, -1]
    current_cable = history.reshape(-1, DEFAULT_TH, STATE_DIM)[:, -1, :CABLE_DIM]
    next_robot = robot[:, 0]
    next_cable = cable[:, 0]
    feature_sets = {
        "deployable_history": deployable_history,
        "history_plus_next_robot": np.concatenate(
            [deployable_history, next_robot], axis=1
        ),
        "history_plus_robot_delta": np.concatenate(
            [deployable_history, next_robot - current_robot], axis=1
        ),
        "history_plus_next_cable": np.concatenate(
            [deployable_history, next_cable], axis=1
        ),
        "history_plus_cable_delta": np.concatenate(
            [deployable_history, next_cable - current_cable], axis=1
        ),
        "history_plus_next_full": np.concatenate(
            [deployable_history, next_cable, next_robot], axis=1
        ),
        "history_plus_full_delta": np.concatenate(
            [
                deployable_history,
                next_cable - current_cable,
                next_robot - current_robot,
            ],
            axis=1,
        ),
    }
    target_variance = np.var(action, axis=0)
    active = target_variance > spec.variance_epsilon
    if not np.any(active):
        raise StageCAttributionError("action target has no active dimensions")
    metrics: Dict[str, Any] = {}
    fold_contract = None
    for name, features in feature_sets.items():
        prediction, contract = grouped_oof_regression(
            features=features,
            targets=action,
            groups=groups,
            folds=spec.folds,
            regularization=spec.ridge_regularization,
        )
        error = np.mean((prediction[:, active] - action[:, active]) ** 2, axis=0)
        nmse = float(np.mean(error / target_variance[active]))
        metrics[name] = {
            "normalized_mse": nmse,
            "raw_mse": float(
                np.mean((prediction[:, active] - action[:, active]) ** 2)
            ),
        }
        if fold_contract is None:
            fold_contract = contract
    history_nmse = metrics["deployable_history"]["normalized_mse"]
    robot_best_name = min(
        ("history_plus_next_robot", "history_plus_robot_delta"),
        key=lambda name: metrics[name]["normalized_mse"],
    )
    cable_best_name = min(
        ("history_plus_next_cable", "history_plus_cable_delta"),
        key=lambda name: metrics[name]["normalized_mse"],
    )
    full_best_name = min(
        ("history_plus_next_full", "history_plus_full_delta"),
        key=lambda name: metrics[name]["normalized_mse"],
    )
    robot_gain = _safe_ratio(
        history_nmse - metrics[robot_best_name]["normalized_mse"],
        history_nmse,
    )
    cable_gain = _safe_ratio(
        history_nmse - metrics[cable_best_name]["normalized_mse"],
        history_nmse,
    )
    return {
        "schema": "phase314b_r255_stagec_action_materiality_probe_v1",
        "diagnostic_only": True,
        "not_formal_idm": True,
        "features_use_first_future_horizon": True,
        "metrics": metrics,
        "best_robot_feature": robot_best_name,
        "best_cable_feature": cable_best_name,
        "best_full_feature": full_best_name,
        "robot_incremental_gain": robot_gain,
        "cable_incremental_gain": cable_gain,
        "robot_proxy_material_for_action_probe": bool(
            robot_gain >= spec.robot_action_incremental_gain_min
        ),
        "fold_contract": fold_contract,
    }


def classify_attribution(
    *,
    schema: Mapping[str, Any],
    predictability: Mapping[str, Any],
    action_probe: Mapping[str, Any],
    spec: AttributionSpec,
) -> Dict[str, Any]:
    if not bool(schema.get("schema_contract_pass")):
        return {
            "root_cause": "phase314b_r255_stagec_state_v3_schema_invalid",
            "required_next_path": "REPAIR_STATE_V3_CACHE_OR_SCHEMA",
        }

    models = predictability["models"]
    deployable_names = (
        "last_observation",
        "constant_velocity",
        "cable_history_ridge",
        "robot_history_ridge",
        "full_state_history_ridge",
        "full_state_plus_past_actions_ridge",
    )
    best_deployable = min(
        deployable_names,
        key=lambda name: models[name]["normalized_mse"],
    )
    best_deployable_nmse = float(models[best_deployable]["normalized_mse"])
    action_nmse = float(
        models["full_state_past_actions_plus_current_action_ridge_nondeployable"][
            "normalized_mse"
        ]
    )
    action_conditioned_gain = _safe_ratio(
        best_deployable_nmse - action_nmse,
        best_deployable_nmse,
    )
    robot_action_gain = float(action_probe["robot_incremental_gain"])
    simple_best = min(
        ("last_observation", "constant_velocity"),
        key=lambda name: models[name]["normalized_mse"],
    )
    simple_nmse = float(models[simple_best]["normalized_mse"])
    simple_matches = bool(
        simple_nmse
        <= spec.simple_baseline_ratio * best_deployable_nmse
    )
    best_deployable_quaternion_p95 = float(
        models[best_deployable]["quaternion"]["geodesic_angle_rad"]["p95"]
    )
    action_quaternion_p95 = float(
        models["full_state_past_actions_plus_current_action_ridge_nondeployable"]
        ["quaternion"]["geodesic_angle_rad"]["p95"]
    )
    deployably_predictable = bool(
        best_deployable_nmse <= spec.deployable_nmse_max
        and best_deployable_quaternion_p95
        <= spec.quaternion_angle_p95_max_rad
    )
    action_conditioned = bool(
        action_conditioned_gain >= spec.action_conditioned_gain_min
        and action_quaternion_p95 <= spec.quaternion_angle_p95_max_rad
    )
    action_material = bool(
        robot_action_gain >= spec.robot_action_incremental_gain_min
    )

    if action_conditioned and action_material:
        root = (
            "phase314b_r255_stagec_robot_future_action_conditioned_and_idm_material"
        )
        next_path = (
            "DESIGN_ACTION_CONDITIONED_ROBOT_TRANSITION_OR_REMOVE_ROBOT_"
            "FROM_STATE_DIFFUSION_TARGET"
        )
    elif action_conditioned and not action_material:
        root = (
            "phase314b_r255_stagec_robot_future_action_conditioned_but_idm_immaterial"
        )
        next_path = (
            "REMOVE_ROBOT_FUTURE_FROM_DIFFUSION_TARGET_KEEP_ROBOT_HISTORY_"
            "CONDITIONING"
        )
    elif deployably_predictable and action_material:
        root = (
            "phase314b_r255_stagec_robot_future_deployably_predictable_and_"
            "idm_material"
        )
        next_path = "AUDIT_ROBOT_ONLY_HEAD_WITH_FROZEN_CABLE_CONTRACT"
    elif simple_matches and not action_material:
        root = "phase314b_r255_stagec_simple_robot_baseline_sufficient"
        next_path = (
            "SEPARATE_ROBOT_FUTURE_FROM_DIFFUSION_TARGET_RETAIN_"
            "DETERMINISTIC_BASELINE"
        )
    else:
        root = (
            "phase314b_r255_stagec_robot_future_not_deployably_predictable_"
            "or_idm_material"
        )
        next_path = (
            "REMOVE_ROBOT_FUTURE_FROM_DIFFUSION_TARGET_AND_REDEFINE_IDM_INPUT"
        )
    return {
        "root_cause": root,
        "required_next_path": next_path,
        "best_deployable_model": best_deployable,
        "best_deployable_nmse": best_deployable_nmse,
        "best_deployable_quaternion_angle_p95_rad": (
            best_deployable_quaternion_p95
        ),
        "best_simple_model": simple_best,
        "best_simple_nmse": simple_nmse,
        "action_conditioned_nondeployable_nmse": action_nmse,
        "action_conditioned_quaternion_angle_p95_rad": action_quaternion_p95,
        "action_conditioned_gain": action_conditioned_gain,
        "robot_action_incremental_gain": robot_action_gain,
        "deployably_predictable": deployably_predictable,
        "action_conditioned_underdetermination_supported": action_conditioned,
        "robot_proxy_material_for_action_probe": action_material,
        "simple_baseline_matches_best_deployable": simple_matches,
    }


def run_train_only_attribution(
    train_arrays: Mapping[str, np.ndarray],
    *,
    spec: Optional[AttributionSpec] = None,
) -> Dict[str, Any]:
    active_spec = AttributionSpec() if spec is None else spec
    active_spec.validate()
    paper_x = _finite(train_arrays["paper_x"], name="paper_x")
    state_action_x = _finite(
        train_arrays["state_action_x"], name="state_action_x"
    )
    robot_history = _finite(
        train_arrays["robot_history"], name="robot_history"
    )
    robot_future = _finite(
        train_arrays["robot_future"], name="robot_future"
    )
    full_future = _finite(train_arrays["y_state"], name="y_state")
    action = _finite(train_arrays["y_action"], name="y_action")
    future_mask = np.asarray(train_arrays["future_valid_mask"], dtype=np.bool_)
    groups = np.asarray(train_arrays["episode_group_key"]).astype(str)
    if paper_x.shape[1] != DEFAULT_TH * STATE_DIM:
        raise StageCAttributionError("paper_x is not state-v3")
    if state_action_x.shape[1] != STATE_ACTION_X_DIM:
        raise StageCAttributionError("state_action_x is not state-v3")
    if set(np.asarray(train_arrays["split_name"]).astype(str).tolist()) != {"train"}:
        raise StageCAttributionError("attribution view contains non-train rows")
    if robot_history.shape[0] != paper_x.shape[0]:
        raise StageCAttributionError("train row count mismatch")

    history = paper_x.reshape(-1, DEFAULT_TH, STATE_DIM)
    feature_sets = {
        "cable_history_ridge": history[..., :CABLE_DIM].reshape(
            history.shape[0], -1
        ),
        "robot_history_ridge": history[..., CABLE_DIM:].reshape(
            history.shape[0], -1
        ),
        "full_state_history_ridge": paper_x,
        "full_state_plus_past_actions_ridge": state_action_x,
        "full_state_past_actions_plus_current_action_ridge_nondeployable": np.concatenate(
            [state_action_x, action], axis=1
        ),
    }

    predictions: Dict[str, np.ndarray] = {
        "last_observation": last_observation_baseline(robot_history),
        "constant_velocity": constant_velocity_baseline(robot_history),
    }
    regression_contracts: Dict[str, Any] = {}
    for name, features in feature_sets.items():
        prediction, contract = grouped_oof_future_robot(
            features=features,
            robot_future=robot_future,
            future_valid_mask=future_mask,
            groups=groups,
            spec=active_spec,
        )
        predictions[name] = prediction
        regression_contracts[name] = contract

    metrics = {
        name: robot_prediction_metrics(
            prediction,
            robot_future,
            future_mask,
            variance_epsilon=active_spec.variance_epsilon,
        )
        for name, prediction in predictions.items()
    }
    action_probe = action_prediction_probe(
        paper_x=paper_x,
        state_action_x=state_action_x,
        robot_history=robot_history,
        robot_future=robot_future,
        cable_future=full_future[..., :CABLE_DIM],
        action_target=action,
        groups=groups,
        spec=active_spec,
    )
    schema = schema_audit(
        robot_history,
        robot_future,
        future_mask,
        spec=active_spec,
    )
    predictability = {
        "schema": "phase314b_r255_stagec_robot_predictability_v1",
        "models": metrics,
        "deployable_models": [
            "last_observation",
            "constant_velocity",
            "cable_history_ridge",
            "robot_history_ridge",
            "full_state_history_ridge",
            "full_state_plus_past_actions_ridge",
        ],
        "nondeployable_models": [
            "full_state_past_actions_plus_current_action_ridge_nondeployable"
        ],
        "regression_contracts": regression_contracts,
    }
    classification = classify_attribution(
        schema=schema,
        predictability=predictability,
        action_probe=action_probe,
        spec=active_spec,
    )
    return {
        "schema": ATTRIBUTION_SCHEMA,
        "verdict": "PASS",
        "scientific_status": "BLOCKED",
        "spec": asdict(active_spec),
        "train_rows": int(paper_x.shape[0]),
        "paired_episode_groups": int(len(set(groups.tolist()))),
        "train_only": True,
        "validation_targets_used": False,
        "formal_test_targets_used": False,
        "cache_arrays_materialized_full": False,
        "train_only_view_used": True,
        "schema_audit": schema,
        "predictability": predictability,
        "action_materiality": action_probe,
        "classification": classification,
        "root_cause": classification["root_cause"],
        "required_next_path": classification["required_next_path"],
        "state_v3_robot_proxy_attribution_interpretable": True,
        "legacy_state_v2_robot_proxy_attribution_interpretable": False,
        "model_attribution_performed": False,
        "diffusion_training": False,
        "reverse_sampling": False,
        "idm_training": False,
        "candidate_execution": False,
        "phase4": False,
        "cps": False,
        "train_only_recommendation": None,
        "selected_configuration": None,
    }
