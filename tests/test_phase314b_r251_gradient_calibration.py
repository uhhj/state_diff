from __future__ import annotations

import numpy as np
import pytest

import ccda_phase3.phase314b_r251_gradient_calibration as r251

from ccda_phase3.phase314b_r251_gradient_calibration import (
    CALIBRATED_GEOMETRY_OBJECTIVES,
    EXPECTED_PAIRED_ROWS,
    PAIRED_INVERSION_SCHEMA,
    TARGET_GRADIENT_RATIOS,
    CalibratedGeometryObjective,
    GradientCalibrationSpec,
    assert_canonical_paired_row_contract,
    calibrated_multiplier_from_gradient_norms,
    classify_pilot,
    compare_calibrated_to_control,
    gradient_tracking_gate,
    paired_nearest_index_audit,
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


def _identity_reverse_dependencies(monkeypatch) -> None:
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


def test_paired_reverse_metric_batches_nearest_index_inputs(monkeypatch) -> None:
    pool = np.zeros((2, 1, 4, 87), dtype=np.float32)
    targets = np.zeros((1, 2, 4, 87), dtype=np.float32)
    targets[:, 1, :, :48] = 0.1
    _identity_reverse_dependencies(monkeypatch)
    observed = {}

    def fake_nearest(sample, target):
        observed["sample_shape"] = sample.shape
        observed["target_shape"] = target.shape
        count = sample.shape[0]
        return {
            "nearest_inversion": np.full(count, 0.25, dtype=np.float64),
            "nearest_unique_fraction": np.full(count, 0.75, dtype=np.float64),
        }

    monkeypatch.setattr(r251, "nearest_index_metrics", fake_nearest)
    result = paired_reverse_pool_metrics_with_inversion(
        pool_z=r251.torch.from_numpy(pool),
        paired_target_raw=r251.torch.from_numpy(targets),
        future_mean=r251.torch.zeros(4, 87),
        future_scale=r251.torch.ones(4, 87),
        physical_contract=object(),
    )
    assert observed["sample_shape"] == (2, 4, 87)
    assert observed["target_shape"] == (2, 4, 87)
    assert result["schema"] == PAIRED_INVERSION_SCHEMA
    assert result["batched_item_count"] == 2
    assert result["nearest_inversion_mean"] == pytest.approx(0.25)
    assert result["nearest_inversion_p95"] == pytest.approx(0.25)
    assert result["nearest_inversion_max"] == pytest.approx(0.25)
    assert result["nearest_unique_fraction_mean"] == pytest.approx(0.75)
    assert result["nearest_unique_fraction_p05"] == pytest.approx(0.75)


def test_paired_reverse_metric_real_nearest_index_integration(monkeypatch) -> None:
    pool = np.zeros((2, 2, 4, 87), dtype=np.float32)
    targets = np.zeros((2, 2, 4, 87), dtype=np.float32)
    targets[:, 1, :, :48] = 0.1
    _identity_reverse_dependencies(monkeypatch)
    result = paired_reverse_pool_metrics_with_inversion(
        pool_z=r251.torch.from_numpy(pool),
        paired_target_raw=r251.torch.from_numpy(targets),
        future_mean=r251.torch.zeros(4, 87),
        future_scale=r251.torch.ones(4, 87),
        physical_contract=object(),
    )
    assert result["batched_item_count"] == 4
    assert 0.0 <= result["nearest_inversion_mean"] <= 1.0
    assert 0.0 <= result["nearest_inversion_p95"] <= 1.0
    assert 0.0 <= result["nearest_unique_fraction_min"] <= 1.0


def test_paired_nearest_index_audit_selects_one_target_per_candidate(monkeypatch) -> None:
    raw = np.zeros((2, 2, 4, 87), dtype=np.float32)
    targets = np.zeros((2, 2, 4, 87), dtype=np.float32)
    targets[:, 1] = 3.0
    branch = np.asarray([[0, 1], [1, 0]], dtype=np.int64)
    captured = {}

    def fake_nearest(sample, target):
        captured["target"] = target.copy()
        count = sample.shape[0]
        return {
            "nearest_inversion": np.zeros(count, dtype=np.float64),
            "nearest_unique_fraction": np.ones(count, dtype=np.float64),
        }

    monkeypatch.setattr(r251, "nearest_index_metrics", fake_nearest)
    result = paired_nearest_index_audit(
        pool_raw=raw,
        paired_target_raw=targets,
        nearest_branch=branch,
    )
    assert captured["target"].shape == (4, 4, 87)
    assert np.all(captured["target"][[0, 3]] == 0.0)
    assert np.all(captured["target"][[1, 2]] == 3.0)
    assert result["batched_item_count"] == 4


def test_paired_nearest_index_audit_rejects_bad_branch_shape() -> None:
    with pytest.raises(ValueError, match="nearest_branch"):
        paired_nearest_index_audit(
            pool_raw=np.zeros((2, 1, 4, 87), dtype=np.float32),
            paired_target_raw=np.zeros((1, 2, 4, 87), dtype=np.float32),
            nearest_branch=np.zeros((2, 2), dtype=np.int64),
        )


def test_as_future_batch_accepts_singleton_and_rejects_invalid() -> None:
    singleton = r251._as_future_batch(
        np.zeros((4, 87), dtype=np.float32),
        name="singleton",
    )
    assert singleton.shape == (1, 4, 87)
    with pytest.raises(ValueError, match="must be"):
        r251._as_future_batch(np.zeros((4, 86), dtype=np.float32), name="bad")
    invalid = np.zeros((4, 87), dtype=np.float32)
    invalid[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        r251._as_future_batch(invalid, name="nan")


def _paired_contract_arrays() -> dict:
    size = max(EXPECTED_PAIRED_ROWS) + 1
    condition = np.full(size, "other", dtype=object)
    visible_seed = np.full(size, -1, dtype=np.int64)
    pair_key = np.full(size, "", dtype=object)
    for pair_index in range(8):
        free_row = EXPECTED_PAIRED_ROWS[2 * pair_index]
        hidden_row = EXPECTED_PAIRED_ROWS[2 * pair_index + 1]
        condition[free_row] = "free"
        condition[hidden_row] = "hidden_slack_breakaway_pin_v2"
        visible_seed[[free_row, hidden_row]] = 400000 + pair_index
        pair_key[[free_row, hidden_row]] = f"pair-{pair_index}"
    return {
        "condition_name": condition,
        "visible_seed": visible_seed,
        "pair_key": pair_key,
    }


def test_canonical_paired_row_contract_accepts_1256_row() -> None:
    result = assert_canonical_paired_row_contract(
        _paired_contract_arrays(),
        EXPECTED_PAIRED_ROWS,
    )
    assert result["pass"]
    assert result["observed_rows"][11] == 1256
    assert result["distinct_row_count"] == 16
    assert result["distinct_visible_seed_count"] == 8


def test_canonical_paired_row_contract_rejects_reported_duplicate_1263() -> None:
    wrong = list(EXPECTED_PAIRED_ROWS)
    wrong[11] = 1263
    with pytest.raises(RuntimeError, match="identity mismatch"):
        assert_canonical_paired_row_contract(_paired_contract_arrays(), wrong)


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
