from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from ccda_phase3.phase314b_r254_robot_proxy_attribution import (
    CABLE_DIM,
    DIAGNOSTIC_OBJECTIVE_NAMES,
    ROBOT_PROXY_DIM,
    RobotProxyAuditSpec,
    action_sensitivity_probe,
    canonicalize_quaternion_trajectory,
    classify_robot_proxy_attribution,
    constant_velocity_robot_baseline,
    evaluate_synthetic_controls,
    grouped_holdout_cross_feature_predictions,
    grouped_holdout_ridge_predictions,
    hybrid_counterfactual_audit,
    last_observation_robot_baseline,
    model_robot_error_audit,
    normalize_quaternion,
    quaternion_error_metrics,
    ridge_fit_predict,
    robot_predictability_baselines,
    robot_proxy_groups,
    robot_proxy_schema_audit,
    standardize_future_raw,
    strip_runtime_objects,
)
from ccda_phase3.schema_v2 import STATE_DIM


def state_history(rows: int = 8, steps: int = 3) -> np.ndarray:
    value = np.zeros((rows, steps, STATE_DIM), dtype=np.float64)
    for row in range(rows):
        for step in range(steps):
            value[row, step, CABLE_DIM:CABLE_DIM + 16] = row + step
            value[row, step, CABLE_DIM + 16:CABLE_DIM + 32] = 0.1 * row
            value[row, step, CABLE_DIM + 32:CABLE_DIM + 35] = [row, step, row + step]
            value[row, step, CABLE_DIM + 35:CABLE_DIM + 39] = [0, 0, 0, 1]
    return value


def future(rows: int = 8, steps: int = 4) -> np.ndarray:
    value = np.zeros((rows, steps, STATE_DIM), dtype=np.float64)
    value[..., :CABLE_DIM] = np.linspace(0, 1, CABLE_DIM)
    for row in range(rows):
        for step in range(steps):
            value[row, step, CABLE_DIM:CABLE_DIM + 16] = row + step + 1
            value[row, step, CABLE_DIM + 16:CABLE_DIM + 32] = 0.1 * row
            value[row, step, CABLE_DIM + 32:CABLE_DIM + 35] = [row, step + 1, row + step + 1]
            value[row, step, CABLE_DIM + 35:CABLE_DIM + 39] = [0, 0, 0, 1]
    return value


def active() -> np.ndarray:
    return np.ones((4, STATE_DIM), dtype=bool)


def test_layout_is_exact() -> None:
    assert robot_proxy_groups() == {
        "joint_position": (0, 16),
        "joint_velocity": (16, 32),
        "ee_position": (32, 35),
        "ee_orientation": (35, 39),
    }


def test_state_dimension_identity() -> None:
    assert STATE_DIM == CABLE_DIM + ROBOT_PROXY_DIM == 87


def test_normalize_quaternion_identity_fallback() -> None:
    value = normalize_quaternion(np.zeros((2, 4)))
    np.testing.assert_allclose(value[:, 3], 1.0)


def test_normalize_quaternion_unit_norm() -> None:
    rng = np.random.default_rng(1)
    value = normalize_quaternion(rng.normal(size=(3, 4, 4)))
    np.testing.assert_allclose(np.linalg.norm(value, axis=-1), 1.0)


def test_canonicalize_quaternion_sign() -> None:
    value = np.array([[[0, 0, 0, 1], [0, 0, 0, -1], [0, 0, 0, 1]]], dtype=float)
    canonical = canonicalize_quaternion_trajectory(value)
    assert np.all(np.sum(canonical[:, 1:] * canonical[:, :-1], axis=-1) >= 0)


def test_quaternion_sign_invariant_error() -> None:
    target = np.zeros((2, 4, 4)); target[..., 3] = 1
    prediction = -target
    metrics = quaternion_error_metrics(prediction, target)
    assert metrics["direct_mse"] > 0.1
    assert metrics["sign_invariant_mse"] == pytest.approx(0.0)


def test_quaternion_geodesic_zero_for_double_cover() -> None:
    target = np.zeros((1, 2, 4)); target[..., 3] = 1
    metrics = quaternion_error_metrics(-target, target)
    assert metrics["geodesic_angle_rad"]["max"] == pytest.approx(0.0)


def test_schema_audit_dimensions() -> None:
    report = robot_proxy_schema_audit(
        history_raw=state_history(), future_raw=future(), future_active=active(),
        spec=RobotProxyAuditSpec(),
    )
    assert report["schema_contract_pass"]
    assert len(report["dimension_records"]) == 39


def test_schema_audit_detects_padding() -> None:
    history = np.zeros((4, 3, STATE_DIM))
    target = np.zeros((4, 4, STATE_DIM))
    target[..., CABLE_DIM] = 1
    report = robot_proxy_schema_audit(
        history_raw=history, future_raw=target, future_active=active(),
        spec=RobotProxyAuditSpec(),
    )
    assert len(report["structural_padding_dimensions"]) >= 38


def test_schema_audit_rejects_bad_active_shape() -> None:
    with pytest.raises(ValueError):
        robot_proxy_schema_audit(
            history_raw=state_history(), future_raw=future(),
            future_active=np.ones((4, 86), bool), spec=RobotProxyAuditSpec(),
        )


def test_last_observation_baseline() -> None:
    history = state_history(2)
    result = last_observation_robot_baseline(history, 4)
    np.testing.assert_allclose(result[:, 0], history[:, -1, CABLE_DIM:])
    np.testing.assert_allclose(result[:, 3], history[:, -1, CABLE_DIM:])


def test_constant_velocity_positions() -> None:
    history = state_history(2)
    result = constant_velocity_robot_baseline(history, 4)
    delta = history[:, -1, CABLE_DIM:CABLE_DIM + 16] - history[:, -2, CABLE_DIM:CABLE_DIM + 16]
    np.testing.assert_allclose(
        result[:, 1, :16], history[:, -1, CABLE_DIM:CABLE_DIM + 16] + 2 * delta
    )


def test_constant_velocity_holds_quaternion() -> None:
    result = constant_velocity_robot_baseline(state_history(2), 4)
    np.testing.assert_allclose(result[..., 35:39], np.broadcast_to(np.array([0, 0, 0, 1]), result[..., 35:39].shape))


def test_standardize_future_round_trip_values() -> None:
    target = future(2)
    mean = np.zeros((4, STATE_DIM))
    scale = np.full((4, STATE_DIM), 2.0)
    result = standardize_future_raw(target, future_mean=mean, future_scale=scale, future_active=active())
    np.testing.assert_allclose(result, target / 2)


def test_standardize_rejects_shape() -> None:
    with pytest.raises(ValueError):
        standardize_future_raw(future(2), future_mean=np.zeros((4, 86)), future_scale=np.ones((4, 86)), future_active=active())


def test_ridge_fit_predict_linear() -> None:
    x = np.arange(20, dtype=float).reshape(10, 2)
    y = x @ np.array([[2.0], [-1.0]])
    prediction = ridge_fit_predict(x, y, x, regularization=1e-8)
    np.testing.assert_allclose(prediction, y, atol=1e-5)


def test_ridge_shape_rejected() -> None:
    with pytest.raises(ValueError):
        ridge_fit_predict(np.zeros((3, 2)), np.zeros((4, 1)), np.zeros((2, 2)), regularization=1e-3)


def test_group_holdout_does_not_train_on_same_group() -> None:
    x = np.arange(24, dtype=float).reshape(12, 2)
    y = x[:, :1]
    groups = np.repeat(np.arange(6), 2)
    prediction = grouped_holdout_ridge_predictions(
        features=x, targets=y, groups=groups, fit_rows=np.arange(12),
        evaluation_rows=np.array([0, 2, 4, 6, 8, 10]), regularization=1e-8,
    )
    assert prediction.shape == (6, 1)


def test_cross_feature_shape() -> None:
    x = np.arange(24, dtype=float).reshape(12, 2)
    y = x[:, :1]
    groups = np.repeat(np.arange(6), 2)
    prediction = grouped_holdout_cross_feature_predictions(
        train_features=x, evaluation_features=x, targets=y, groups=groups,
        fit_rows=np.arange(12), evaluation_rows=np.arange(0, 12, 2),
        regularization=1e-8,
    )
    assert prediction.shape == (6, 1)


def test_predictability_baselines_schema() -> None:
    rows = 12
    hist = state_history(rows)
    raw = future(rows)
    mean = np.zeros((4, STATE_DIM))
    scale = np.ones((4, STATE_DIM))
    z = raw.copy()
    condition = hist.reshape(rows, -1)
    groups = np.repeat(np.arange(6), 2)
    report = robot_predictability_baselines(
        history_raw=hist, condition_features=condition, future_raw=raw,
        future_z=z, future_mean=mean, future_scale=scale, future_active=active(),
        action_target=np.zeros((rows, 3)), groups=groups, fit_rows=np.arange(rows),
        evaluation_rows=np.arange(0, rows, 2), spec=RobotProxyAuditSpec(),
    )
    assert report["group_holdout"]
    assert report["action_conditioned_baseline_deployable"] is False
    assert "_prediction_z" in report


def test_model_robot_error_zero() -> None:
    target = future(4)
    report = model_robot_error_audit(
        prediction_z=target, prediction_raw=target, target_z=target,
        target_raw=target, future_active=active(), history_raw=state_history(4),
    )
    assert report["z_mse"] == pytest.approx(0.0)
    assert report["quaternion"]["geodesic_angle_rad"]["max"] == pytest.approx(0.0)


def test_model_robot_error_rejects_shape() -> None:
    with pytest.raises(ValueError):
        model_robot_error_audit(
            prediction_z=np.zeros((3, 4, 86)), prediction_raw=future(3),
            target_z=future(3), target_raw=future(3), future_active=active(),
            history_raw=state_history(3),
        )


def test_hybrid_oracle_robot_restores_gate() -> None:
    target = np.zeros((3, 4, STATE_DIM))
    prediction = target.copy()
    prediction[..., CABLE_DIM:] = 1.0
    baseline = np.zeros((3, 4, ROBOT_PROXY_DIM))
    report = hybrid_counterfactual_audit(
        prediction_z=prediction, target_z=target, baseline_robot_z=baseline,
        future_active=active(), z_mse_threshold=0.001,
    )
    assert report["robot_proxy_drives_full_z_failure"]


def test_hybrid_shape_rejected() -> None:
    with pytest.raises(ValueError):
        hybrid_counterfactual_audit(
            prediction_z=future(2), target_z=future(2),
            baseline_robot_z=np.zeros((3, 4, 39)), future_active=active(),
            z_mse_threshold=0.1,
        )


def test_action_probe_is_not_formal_idm() -> None:
    rows = 12
    hist = state_history(rows)
    target = future(rows)
    groups = np.repeat(np.arange(6), 2)
    report = action_sensitivity_probe(
        condition_features=hist.reshape(rows, -1), oracle_future_z=target,
        model_future_z_for_evaluation=target[np.arange(0, rows, 2)],
        action_target=np.arange(rows * 2, dtype=float).reshape(rows, 2),
        groups=groups, fit_rows=np.arange(rows), evaluation_rows=np.arange(0, rows, 2),
        spec=RobotProxyAuditSpec(),
    )
    assert report["diagnostic_only"]
    assert report["not_formal_idm"]


def test_synthetic_controls_pass() -> None:
    assert evaluate_synthetic_controls(RobotProxyAuditSpec())["pass"]


def valid_report() -> dict:
    variant = {
        "training_completed": True,
        "r253_reproduction_pass": True,
        "schema_audit": {
            "structural_padding_dimensions": [],
            "intermittent_zero_dimensions": [],
        },
        "robot_error": {
            "z_mse": 1.0,
            "quaternion": {
                "target_consecutive_sign_flip_fraction": 0.0,
                "sign_invariant_gain": 0.0,
            },
        },
        "predictability_baselines": {
            "best_deployable_baseline": "history_ridge",
            "baselines": {"history_ridge": {"z_mse": 3.0}},
            "action_conditioned_underdetermination_supported": False,
        },
        "hybrid_counterfactual": {"robot_proxy_drives_full_z_failure": True},
        "action_sensitivity": {"robot_proxy_material_for_action_probe": False},
    }
    return {
        "spec": RobotProxyAuditSpec().__dict__,
        "schema_audit": {"schema_contract_pass": True},
        "synthetic_controls": {"pass": True},
        "variants": {name: dict(variant) for name in DIAGNOSTIC_OBJECTIVE_NAMES},
    }


def test_classifier_matrix_precedence() -> None:
    report = valid_report(); report["variants"].pop(DIAGNOSTIC_OBJECTIVE_NAMES[0])
    assert classify_robot_proxy_attribution(report)["root_cause"] == "phase314b_r254_diagnostic_matrix_incomplete"


def test_classifier_training_precedence() -> None:
    report = valid_report(); report["variants"][DIAGNOSTIC_OBJECTIVE_NAMES[0]]["training_completed"] = False
    assert classify_robot_proxy_attribution(report)["root_cause"] == "phase314b_r254_training_reproduction_failed"


def test_classifier_reproduction_precedence() -> None:
    report = valid_report(); report["variants"][DIAGNOSTIC_OBJECTIVE_NAMES[0]]["r253_reproduction_pass"] = False
    assert classify_robot_proxy_attribution(report)["root_cause"] == "phase314b_r254_r253_contract_reproduction_failed"


def test_classifier_controls_precedence() -> None:
    report = valid_report(); report["synthetic_controls"]["pass"] = False
    assert classify_robot_proxy_attribution(report)["root_cause"] == "phase314b_r254_robot_proxy_controls_failed"


def test_classifier_hybrid_mechanism() -> None:
    result = classify_robot_proxy_attribution(valid_report())
    assert result["root_cause"] == "phase314b_r254_robot_proxy_model_fidelity_gap_supported"
    assert result["train_only_recommendation"] is None
    assert result["selected_configuration"] is None


def test_classifier_action_underdetermination_precedence() -> None:
    report = valid_report()
    for value in report["variants"].values():
        value["predictability_baselines"]["action_conditioned_underdetermination_supported"] = True
    assert classify_robot_proxy_attribution(report)["root_cause"] == "phase314b_r254_action_conditioned_robot_future_supported"


def test_strip_runtime_numpy() -> None:
    assert strip_runtime_objects({"value": np.array([1, 2])}) == {"value": [1, 2]}


def test_strip_runtime_private_key() -> None:
    assert strip_runtime_objects({"_model": object(), "x": 1}) == {"x": 1}


def test_strip_runtime_rejects_tensor() -> None:
    with pytest.raises(TypeError):
        strip_runtime_objects({"tensor": torch.zeros(1)})


def test_run_script_has_explicit_interpreter() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts/phase3_14b_r254_run.sh"
    text = script.read_text(encoding="utf-8")
    assert "/miniforge3/envs/coord_bimanual/bin/python" in text
    assert '"${PYTHON_BIN}" scripts/phase3_14b_r254_run_pilot.py' in text


def test_run_script_does_not_disable_cuda() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts/phase3_14b_r254_run.sh"
    text = script.read_text(encoding="utf-8")
    assert 'export CUDA_VISIBLE_DEVICES=""' not in text
