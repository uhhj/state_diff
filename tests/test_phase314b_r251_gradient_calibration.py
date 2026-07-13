from __future__ import annotations

import numpy as np
import pytest

import ccda_phase3.phase314b_r251_gradient_calibration as r251

from ccda_phase3.phase314b_r251_gradient_calibration import (
    CALIBRATED_GEOMETRY_OBJECTIVES,
    TARGET_GRADIENT_RATIOS,
    CalibratedGeometryObjective,
    GradientCalibrationSpec,
    calibrated_multiplier_from_gradient_norms,
    classify_pilot,
    compare_calibrated_to_control,
    gradient_tracking_gate,
    paired_reverse_pool_metrics_with_inversion,
    strip_runtime_objects,
    tensor_state_sha256,
)


def test_objective_matrix_is_complete_and_fixed() -> None:
    names = [value.name for value in CALIBRATED_GEOMETRY_OBJECTIVES]
    assert names == [
        "v_only_frozen_control",
        "ordered_mean_raw_g010",
        "ordered_mean_raw_g050",
        "ordered_mean_raw_g100",
        "ordered_cvar_raw_g010",
        "ordered_cvar_raw_g050",
        "ordered_cvar_raw_g100",
        "ordered_cvar_contract_g010",
        "ordered_cvar_contract_g050",
        "ordered_cvar_contract_g100",
    ]
    assert TARGET_GRADIENT_RATIOS == (0.10, 0.50, 1.00)
    for value in CALIBRATED_GEOMETRY_OBJECTIVES:
        value.validate()


def test_control_cannot_have_geometry_weight() -> None:
    with pytest.raises(ValueError, match="control"):
        CalibratedGeometryObjective(
            name="bad",
            family="v_only",
            target_gradient_ratio=0.0,
            mean_weight=1.0,
            cvar_weight=0.0,
            contract_weight=0.0,
            selectable=False,
        ).validate()


def test_multiplier_uses_actual_gradient_ratio() -> None:
    spec = GradientCalibrationSpec(batch_count=4)
    result = calibrated_multiplier_from_gradient_norms(
        target_ratio=0.5,
        v_gradient_norms=[2.0, 2.0, 2.0, 2.0],
        geometry_gradient_norms=[20.0, 20.0, 20.0, 20.0],
        spec=spec,
    )
    assert result["raw_ratio_median"] == pytest.approx(10.0)
    assert result["multiplier"] == pytest.approx(0.05)
    assert result["weighted_ratio_median"] == pytest.approx(0.5)
    assert result["pass"]


def test_multiplier_is_not_loss_magnitude_balancing() -> None:
    spec = GradientCalibrationSpec(batch_count=3)
    result = calibrated_multiplier_from_gradient_norms(
        target_ratio=0.1,
        v_gradient_norms=[1.0, 2.0, 4.0],
        geometry_gradient_norms=[10.0, 10.0, 10.0],
        spec=spec,
    )
    assert result["raw_geometry_to_v_ratios"] == pytest.approx([10.0, 5.0, 2.5])
    assert result["multiplier"] == pytest.approx(0.02)


def test_multiplier_rejects_zero_gradient() -> None:
    with pytest.raises(ValueError, match="positive"):
        calibrated_multiplier_from_gradient_norms(
            target_ratio=0.5,
            v_gradient_norms=[1.0, 1.0],
            geometry_gradient_norms=[1.0, 0.0],
            spec=GradientCalibrationSpec(batch_count=2),
        )


def test_multiplier_rejects_nonfinite_gradient() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        calibrated_multiplier_from_gradient_norms(
            target_ratio=0.5,
            v_gradient_norms=[1.0, np.nan],
            geometry_gradient_norms=[1.0, 1.0],
            spec=GradientCalibrationSpec(batch_count=2),
        )


def test_gradient_tracking_accepts_target_neighborhood() -> None:
    result = gradient_tracking_gate(
        target_ratio=0.5,
        observed_ratios=[0.3, 0.4, 0.6, 0.8],
        spec=GradientCalibrationSpec(),
    )
    assert result["target_tracked"]
    assert result["safe_gradient_gate"]
    assert result["pass"]


def test_gradient_tracking_rejects_r25_scale_explosion() -> None:
    result = gradient_tracking_gate(
        target_ratio=0.5,
        observed_ratios=[28.8, 30.2, 67.4],
        spec=GradientCalibrationSpec(),
    )
    assert not result["safe_gradient_gate"]
    assert not result["pass"]


def test_gradient_tracking_rejects_ineffective_geometry() -> None:
    result = gradient_tracking_gate(
        target_ratio=0.1,
        observed_ratios=[0.001, 0.002, 0.003],
        spec=GradientCalibrationSpec(),
    )
    assert not result["target_tracked"]
    assert not result["pass"]


def test_tensor_state_hash_is_order_independent_and_value_sensitive() -> None:
    first = {
        "b": r251.torch.tensor([2.0]),
        "a": r251.torch.tensor([1.0]),
    }
    second = {
        "a": r251.torch.tensor([1.0]),
        "b": r251.torch.tensor([2.0]),
    }
    changed = {
        "a": r251.torch.tensor([1.0]),
        "b": r251.torch.tensor([3.0]),
    }
    assert tensor_state_sha256(first) == tensor_state_sha256(second)
    assert tensor_state_sha256(first) != tensor_state_sha256(changed)


def test_runtime_strip_removes_nested_private_objects() -> None:
    value = {
        "public": {"value": 1, "_tensor": object()},
        "_model": object(),
        "items": [{"ok": True, "_raw": object()}],
    }
    assert strip_runtime_objects(value) == {
        "public": {"value": 1},
        "items": [{"ok": True}],
    }


def test_compare_requires_inversion_preservation() -> None:
    control = {
        "reverse_metrics": {
            "calibrated": {
                "segment_score_p95": 10.0,
                "sample_validity_rate": 0.5,
            },
            "best_ordered_rmse_mean": 0.01,
            "nearest_inversion_p95": 2.0,
        }
    }
    candidate = {
        "reverse_metrics": {
            "calibrated": {
                "segment_score_p95": 8.0,
                "sample_validity_rate": 0.6,
            },
            "best_ordered_rmse_mean": 0.0105,
            "nearest_inversion_p95": 5.0,
        }
    }
    result = compare_calibrated_to_control(candidate=candidate, control=control)
    assert result["tail_improved"]
    assert result["ordered_preserved"]
    assert not result["inversion_preserved"]
    assert not result["pass"]


def test_paired_reverse_metric_emits_nearest_inversion(monkeypatch) -> None:
    pool = np.zeros((2, 1, 4, 87), dtype=np.float32)
    targets = np.zeros((1, 2, 4, 87), dtype=np.float32)
    targets[:, 1, :, 1] = 0.1
    monkeypatch.setattr(
        r251,
        "torch_inverse_standardize",
        lambda value, mean, scale: value,
    )
    monkeypatch.setattr(
        r251,
        "calibrated_validity",
        lambda raw, contract: {
            "sample_valid_mask": np.ones(raw.shape[:2], dtype=bool),
            "sample_validity_rate": 1.0,
            "query_has_valid_candidate_rate": 1.0,
            "segment_score_p95": 1.0,
        },
    )
    monkeypatch.setattr(
        r251,
        "nearest_index_metrics",
        lambda sample, target: {"nearest_inversion": 2.0},
    )
    result = paired_reverse_pool_metrics_with_inversion(
        pool_z=r251.torch.from_numpy(pool),
        paired_target_raw=r251.torch.from_numpy(targets),
        future_mean=r251.torch.zeros(4, 87),
        future_scale=r251.torch.ones(4, 87),
        physical_contract=object(),
    )
    assert result["nearest_inversion_mean"] == pytest.approx(2.0)
    assert result["nearest_inversion_p95"] == pytest.approx(2.0)
    assert result["nearest_inversion_max"] == pytest.approx(2.0)


def _unique(pass_value: bool = True) -> dict:
    return {
        "pass": pass_value,
        "calibration": {"pass": True},
        "gradient_tracking": {"pass": True},
    }


def _paired(
    *,
    one_step: bool = True,
    comparison: bool = True,
    branch: bool = True,
) -> dict:
    return {
        "one_step_pass": one_step,
        "reverse_metrics": {
            "calibrated": {
                "query_has_valid_candidate_rate": 1.0,
                "sample_validity_rate": 0.9,
            },
            "best_ordered_rmse_mean": 0.01,
            "nearest_inversion_p95": 1.0,
        },
        "full_reverse_branch_support": {
            "pass": branch,
            "both_branch_support_rate": 1.0,
            "two_branch_occupancy_rate": 1.0,
        },
        "comparison_to_control": {"pass": comparison},
    }


def test_classifier_can_recommend_only_calibrated_geometry() -> None:
    report = {
        "unique_free_variants": {
            "v_only_frozen_control": _unique(),
            "ordered_mean_raw_g050": _unique(),
        },
        "paired_variants": {
            "v_only_frozen_control": _paired(one_step=False),
            "ordered_mean_raw_g050": _paired(),
        },
    }
    result = classify_pilot(report)
    assert result["root_cause"] == "phase314b_r251_train_only_gradient_calibration_supported"
    assert result["train_only_recommendation"] == "ordered_mean_raw_g050"


def test_classifier_never_recommends_control() -> None:
    report = {
        "unique_free_variants": {
            "v_only_frozen_control": _unique(),
        },
        "paired_variants": {
            "v_only_frozen_control": _paired(),
        },
    }
    result = classify_pilot(report)
    assert result["train_only_recommendation"] is None


def test_classifier_does_not_mislabel_mixed_reverse_failures_as_collapse() -> None:
    report = {
        "unique_free_variants": {
            "v_only_frozen_control": _unique(),
            "ordered_mean_raw_g010": _unique(),
            "ordered_mean_raw_g050": _unique(),
        },
        "paired_variants": {
            "v_only_frozen_control": _paired(one_step=False),
            "ordered_mean_raw_g010": _paired(branch=False),
            "ordered_mean_raw_g050": _paired(comparison=False, branch=True),
        },
    }
    result = classify_pilot(report)
    assert result["root_cause"] == "phase314b_r251_reverse_geometry_gain_not_supported"


def test_classifier_reports_nonstationary_gradient() -> None:
    report = {
        "unique_free_variants": {
            "v_only_frozen_control": _unique(),
            "ordered_mean_raw_g050": {
                "pass": False,
                "calibration": {"pass": True},
                "gradient_tracking": {"pass": False},
            },
        },
        "paired_variants": {},
    }
    result = classify_pilot(report)
    assert result["root_cause"] == "phase314b_r251_geometry_gradient_nonstationarity_supported"
