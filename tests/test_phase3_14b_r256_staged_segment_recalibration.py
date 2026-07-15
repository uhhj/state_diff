from __future__ import annotations

import numpy as np
import pytest
from typing import Optional

from ccda_phase3 import phase314b_r256_stageb_cable_diffusion as stageb
from ccda_phase3.phase314b_r256_staged_segment_recalibration import (
    RecalibrationSpec,
    SegmentGateContract,
    SegmentReference,
    _conformal_quantile,
    _group_acceptance,
    _higher_quantile,
    classify_recalibration,
    deterministic_eligible_branch_bootstrap,
    deterministic_gate_split,
    fit_segment_gate,
    fit_segment_reference,
    physical_branch_prefix_curve,
    public_population_record,
    recalibrated_physical_decomposition,
    segment_scores,
)


def straight_cable(*, spacing: float = 0.04, y: float = 0.0) -> np.ndarray:
    x = np.arange(stageb.BEADS, dtype=np.float32) * float(spacing)
    y_value = np.full(stageb.BEADS, float(y), dtype=np.float32)
    return np.stack([x, y_value], axis=1).reshape(stageb.CABLE_DIM)


def trajectory(*, spacing: float = 0.04) -> np.ndarray:
    return np.stack(
        [
            straight_cable(spacing=spacing, y=0.01 * horizon)
            for horizon in range(stageb.FUTURE_STEPS)
        ],
        axis=0,
    )


def synthetic_targets(rows: int = 40) -> np.ndarray:
    values = np.empty(
        (rows, stageb.FUTURE_STEPS, stageb.CABLE_DIM),
        dtype=np.float32,
    )
    for row in range(rows):
        spacing = 0.04 * (1.0 + 0.002 * ((row % 7) - 3))
        values[row] = trajectory(spacing=spacing)
    return values


def historical_geometry(target: np.ndarray) -> stageb.GeometryContract:
    return stageb.fit_geometry_contract(target)


def simple_contract(target: np.ndarray) -> SegmentGateContract:
    spec = RecalibrationSpec()
    split = {
        "fit_group_count": 10,
        "calibration_group_count": 10,
        "fit_group_sha256": "a" * 64,
        "calibration_group_sha256": "b" * 64,
    }
    groups = np.asarray([f"g{index // 2}" for index in range(20)])
    gate, _ = fit_segment_gate(
        fit_target=target[:20],
        calibration_target=target[20:40],
        calibration_groups=groups,
        split_record=split,
        spec=spec,
    )
    return gate


def population_record(rate: float) -> dict:
    return {
        "segment": {
            "row_any_rate": rate,
            "group_acceptance": {
                "group_acceptance_rate": rate,
            },
            "condition_row_any_rate": {
                "free": rate,
                "hidden_slack_breakaway_pin_v2": rate,
            },
        },
        "combined": {
            "row_any_rate": rate,
        },
    }


def physical_branch_record(
    *,
    eligible_rate: float,
    support: Optional[float],
) -> dict:
    return {
        "eligible_row_rate": eligible_rate,
        "support_rate_among_eligible": support,
    }


def test_spec_validates_registered_contract():
    RecalibrationSpec().validate()


def test_spec_rejects_overlapping_fit_and_calibration():
    with pytest.raises(ValueError):
        RecalibrationSpec(
            fit_remainder=0,
            calibration_remainder=0,
        ).validate()


def test_higher_quantile_is_deterministic_order_statistic():
    value = np.asarray([4.0, 1.0, 3.0, 2.0])
    assert _higher_quantile(value, 0.50) == 2.0
    assert _higher_quantile(value, 1.00) == 4.0


def test_conformal_quantile_records_empirical_acceptance():
    value = np.arange(1.0, 21.0)
    result = _conformal_quantile(value, 0.90)
    assert result["sample_count"] == 20
    assert result["empirical_acceptance"] >= 0.90


def test_deterministic_gate_split_preserves_whole_groups():
    groups = np.asarray(
        [f"g{index // 2}" for index in range(32)]
    )
    train = np.ones(32, dtype=np.bool_)
    fit, calibration, record = deterministic_gate_split(
        groups,
        train,
        spec=RecalibrationSpec(),
    )
    assert not np.any(fit & calibration)
    assert np.array_equal(fit | calibration, train)
    for group in sorted(set(groups)):
        selected = groups == group
        assert bool(np.all(fit[selected])) != bool(
            np.all(calibration[selected])
        )
    assert record["all_stageb_training_rows_assigned"]


def test_segment_reference_has_registered_shape_and_positive_scale():
    reference = fit_segment_reference(
        synthetic_targets(20),
        scale_floor=1.0e-3,
    )
    reference.validate()
    assert reference.center_log.shape == (4, 23)
    assert np.all(reference.scale_log >= 1.0e-3)


def test_segment_scores_are_zero_at_reference_center():
    center_length = 0.04
    target = np.repeat(
        trajectory(spacing=center_length)[None],
        20,
        axis=0,
    )
    reference = fit_segment_reference(
        target,
        scale_floor=1.0e-3,
    )
    scores = segment_scores(target[:2], reference)
    np.testing.assert_allclose(scores["joint"], 0.0, atol=2.0e-4)



def test_segment_scores_treat_zero_length_as_finite_rejection_score():
    target = synthetic_targets(20)
    reference = fit_segment_reference(
        target,
        scale_floor=1.0e-3,
    )
    candidate = target[:1].copy().reshape(1, 4, stageb.BEADS, 2)
    candidate[:, :, 1] = candidate[:, :, 0]
    scores = segment_scores(candidate.reshape(1, 4, 48), reference)
    assert np.all(np.isfinite(scores["joint"]))
    assert int(np.sum(scores["nonpositive"])) > 0
    assert float(scores["joint"][0, 0]) > 1000.0

def test_fit_segment_gate_is_group_calibrated():
    target = synthetic_targets(40)
    groups = np.asarray([f"g{index // 2}" for index in range(20)])
    split = {
        "fit_group_count": 10,
        "calibration_group_count": 10,
        "fit_group_sha256": "a" * 64,
        "calibration_group_sha256": "b" * 64,
    }
    gate, fit = fit_segment_gate(
        fit_target=target[:20],
        calibration_target=target[20:],
        calibration_groups=groups,
        split_record=split,
        spec=RecalibrationSpec(),
    )
    gate.validate()
    assert fit["joint"]["empirical_acceptance"] >= 0.95


def test_group_acceptance_uses_registered_within_group_coverage():
    passed = np.asarray(
        [True] * 19 + [False] + [True] * 10 + [False] * 10
    )
    groups = np.asarray(["a"] * 20 + ["b"] * 20)
    result = _group_acceptance(
        passed,
        groups,
        within_group_coverage=0.95,
    )
    assert result["accepted_group_count"] == 1
    assert result["group_acceptance_rate"] == 0.5


def test_recalibrated_gate_accepts_training_like_ground_truth():
    target = synthetic_targets(40)
    gate = simple_contract(target)
    result = recalibrated_physical_decomposition(
        value=target[:10],
        contract=gate,
        historical_geometry=historical_geometry(target[:20]),
        groups=np.asarray([f"g{index // 2}" for index in range(10)]),
        condition_name=np.asarray(
            ["free", "hidden_slack_breakaway_pin_v2"] * 5
        ),
    )
    assert result["segment"]["row_any_rate"] > 0.0
    assert result["segment"]["accepted_candidate_count"] > 0


def test_recalibrated_gate_rejects_large_segment_distortion():
    target = synthetic_targets(40)
    gate = simple_contract(target)
    distorted = np.repeat(trajectory(spacing=0.4)[None], 4, axis=0)
    result = recalibrated_physical_decomposition(
        value=distorted,
        contract=gate,
        historical_geometry=historical_geometry(target[:20]),
        groups=np.asarray(["g0", "g0", "g1", "g1"]),
        condition_name=np.asarray(
            ["free", "hidden_slack_breakaway_pin_v2"] * 2
        ),
    )
    assert result["segment"]["candidate_rate"] == 0.0


def test_public_population_record_removes_masks():
    target = synthetic_targets(40)
    gate = simple_contract(target)
    result = recalibrated_physical_decomposition(
        value=target[:4],
        contract=gate,
        historical_geometry=historical_geometry(target[:20]),
        groups=np.asarray(["g0", "g0", "g1", "g1"]),
        condition_name=np.asarray(
            ["free", "hidden_slack_breakaway_pin_v2"] * 2
        ),
    )
    public = public_population_record(result)
    assert "segment_valid_mask" not in public
    assert "combined_valid_mask" not in public


def paired_branch_data():
    target = np.zeros((4, 4, 48), dtype=np.float32)
    target[1] = 2.0
    target[2] = 4.0
    target[3] = 6.0
    candidates = np.zeros((4, 4, 4, 48), dtype=np.float32)
    candidates[0, 0] = target[0]
    candidates[1, 0] = target[1]
    candidates[2, 1] = target[2]
    candidates[3, 1] = target[3]
    valid = np.zeros((4, 4), dtype=np.bool_)
    valid[:, :2] = True
    return (
        ["p0", "p0", "p1", "p1"],
        ["free", "hidden", "free", "hidden"],
        target,
        candidates,
        valid,
    )


def test_physical_branch_prefix_curve_reports_each_k():
    pair_key, condition, target, candidates, valid = paired_branch_data()
    result = physical_branch_prefix_curve(
        pair_key=pair_key,
        condition_name=condition,
        target=target,
        candidates=candidates,
        combined_valid_mask=valid,
        prefix_k=(1, 2, 4),
    )
    assert set(result) == {"1", "2", "4"}


def test_eligible_branch_bootstrap_reports_unavailable():
    result = deterministic_eligible_branch_bootstrap(
        {"rows": [{"eligible": False, "supported": None}]},
        resamples=128,
        seed=7,
    )
    assert not result["available"]
    assert result["ci95"] is None


def test_eligible_branch_bootstrap_is_deterministic():
    physical = {
        "rows": [
            {"eligible": True, "supported": True},
            {"eligible": True, "supported": False},
            {"eligible": True, "supported": True},
        ]
    }
    left = deterministic_eligible_branch_bootstrap(
        physical,
        resamples=128,
        seed=7,
    )
    right = deterministic_eligible_branch_bootstrap(
        physical,
        resamples=128,
        seed=7,
    )
    assert left == right
    assert left["available"]


def test_classification_prioritizes_internal_calibration_failure():
    result = classify_recalibration(
        spec=RecalibrationSpec(),
        calibration_ground_truth=population_record(0.50),
        probe_ground_truth=population_record(1.0),
        one_step={"10": population_record(1.0)},
        reverse_final=population_record(1.0),
        physical_branch=physical_branch_record(
            eligible_rate=1.0,
            support=1.0,
        ),
    )
    assert result["primary_failure_locus"] == "gate_implementation"


def test_classification_detects_probe_generalization_failure():
    result = classify_recalibration(
        spec=RecalibrationSpec(),
        calibration_ground_truth=population_record(1.0),
        probe_ground_truth=population_record(0.50),
        one_step={"10": population_record(1.0)},
        reverse_final=population_record(1.0),
        physical_branch=physical_branch_record(
            eligible_rate=1.0,
            support=1.0,
        ),
    )
    assert result["primary_failure_locus"] == "gate_generalization"


def test_classification_detects_x0_failure():
    result = classify_recalibration(
        spec=RecalibrationSpec(),
        calibration_ground_truth=population_record(1.0),
        probe_ground_truth=population_record(1.0),
        one_step={"10": population_record(0.20)},
        reverse_final=population_record(1.0),
        physical_branch=physical_branch_record(
            eligible_rate=1.0,
            support=1.0,
        ),
    )
    assert result["primary_failure_locus"] == "x0_denoiser"


def test_classification_detects_reverse_segment_failure():
    result = classify_recalibration(
        spec=RecalibrationSpec(),
        calibration_ground_truth=population_record(1.0),
        probe_ground_truth=population_record(1.0),
        one_step={"10": population_record(1.0)},
        reverse_final=population_record(0.20),
        physical_branch=physical_branch_record(
            eligible_rate=0.20,
            support=1.0,
        ),
    )
    assert result["primary_failure_locus"] == "reverse_segment_transport"


def test_classification_reaches_branch_after_physical_coverage():
    result = classify_recalibration(
        spec=RecalibrationSpec(),
        calibration_ground_truth=population_record(1.0),
        probe_ground_truth=population_record(1.0),
        one_step={"10": population_record(1.0)},
        reverse_final=population_record(1.0),
        physical_branch=physical_branch_record(
            eligible_rate=1.0,
            support=0.50,
        ),
    )
    assert result["primary_failure_locus"] == "branch_transport"


def test_classification_can_reach_supported_path():
    result = classify_recalibration(
        spec=RecalibrationSpec(),
        calibration_ground_truth=population_record(1.0),
        probe_ground_truth=population_record(1.0),
        one_step={"10": population_record(1.0)},
        reverse_final=population_record(1.0),
        physical_branch=physical_branch_record(
            eligible_rate=1.0,
            support=1.0,
        ),
    )
    assert result["primary_failure_locus"] == "none"
