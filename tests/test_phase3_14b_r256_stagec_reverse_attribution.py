from __future__ import annotations

import numpy as np
import pytest

from ccda_phase3 import phase314b_r256_stageb_cable_diffusion as stageb
from ccda_phase3.phase314b_r256_stagec_reverse_attribution import (
    AttributionSpec,
    as_candidates,
    branch_margin_records,
    branch_prefix_curve,
    classify_attribution,
    compare_worker_results,
    deterministic_pair_bootstrap_ci,
    physical_decomposition,
    physical_valid_branch_metrics,
)


def straight_cable(
    *,
    scale: float = 1.0,
    y: float = 0.0,
) -> np.ndarray:
    x = np.linspace(0.0, scale, stageb.BEADS, dtype=np.float32)
    y_values = np.full(stageb.BEADS, y, dtype=np.float32)
    return np.stack([x, y_values], axis=1).reshape(stageb.CABLE_DIM)


def trajectory(*, scale: float = 1.0) -> np.ndarray:
    return np.stack(
        [
            straight_cable(scale=scale, y=0.01 * horizon)
            for horizon in range(stageb.FUTURE_STEPS)
        ],
        axis=0,
    )


def contract(
    *,
    lower: float = 0.01,
    upper: float = 0.10,
) -> stageb.GeometryContract:
    return stageb.GeometryContract(
        segment_lower=np.full(
            (stageb.FUTURE_STEPS, stageb.BEADS - 1),
            lower,
            dtype=np.float32,
        ),
        segment_upper=np.full(
            (stageb.FUTURE_STEPS, stageb.BEADS - 1),
            upper,
            dtype=np.float32,
        ),
        coordinate_abs_max=2.0,
        target_intersection_max=0,
    )


def physical_record(candidate_rate: float) -> dict:
    return {
        "segment": {
            "candidate_all_pass_rate": candidate_rate,
        },
        "combined_candidate_rate": candidate_rate,
    }


def test_as_candidates_adds_candidate_axis():
    value = np.zeros((3, 4, 48), dtype=np.float32)
    assert as_candidates(value).shape == (3, 1, 4, 48)


def test_as_candidates_preserves_candidate_axis():
    value = np.zeros((3, 8, 4, 48), dtype=np.float32)
    assert as_candidates(value).shape == value.shape


def test_as_candidates_rejects_wrong_shape():
    with pytest.raises(ValueError):
        as_candidates(np.zeros((3, 48), dtype=np.float32))


def test_physical_decomposition_accepts_straight_trajectory():
    value = np.repeat(trajectory()[None, None], 4, axis=0)
    result = physical_decomposition(value, contract())
    assert result["combined_candidate_rate"] == 1.0
    assert result["segment"]["candidate_all_pass_rate"] == 1.0
    assert result["segment"]["element_pass_rate"] == 1.0


def test_physical_decomposition_detects_lower_violation():
    value = np.repeat(trajectory()[None, None], 2, axis=0)
    points = value.reshape(2, 1, 4, stageb.BEADS, 2)
    points[:, :, :, 1] = points[:, :, :, 0]
    result = physical_decomposition(value, contract())
    assert result["segment"]["lower_violation_count"] > 0
    assert result["segment"]["candidate_all_pass_rate"] == 0.0


def test_physical_decomposition_detects_upper_violation():
    value = np.repeat(trajectory(scale=10.0)[None, None], 2, axis=0)
    result = physical_decomposition(value, contract())
    assert result["segment"]["upper_violation_count"] > 0
    assert result["segment"]["candidate_all_pass_rate"] == 0.0


def test_all_segment_gate_can_be_zero_with_high_element_rate():
    value = np.repeat(trajectory()[None, None], 10, axis=0)
    points = value.reshape(10, 1, 4, stageb.BEADS, 2)
    for row in range(10):
        horizon = row % 4
        bead = 1 + (row % (stageb.BEADS - 1))
        points[row, 0, horizon, bead] = points[row, 0, horizon, bead - 1]
    result = physical_decomposition(value, contract())
    assert result["segment"]["element_pass_rate"] > 0.98
    assert result["segment"]["candidate_all_pass_rate"] == 0.0


def paired_branch_data():
    target = np.zeros((4, 4, 48), dtype=np.float32)
    target[1] = 2.0
    target[2] = 4.0
    target[3] = 6.0
    candidates = np.zeros((4, 4, 4, 48), dtype=np.float32)
    candidates[0, 0] = target[1]
    candidates[0, 1] = target[0]
    candidates[1, 0] = target[0]
    candidates[1, 1] = target[1]
    candidates[2, 0] = target[3]
    candidates[2, 1] = target[2]
    candidates[3, 0] = target[2]
    candidates[3, 1] = target[3]
    return (
        ["p0", "p0", "p1", "p1"],
        ["free", "hidden", "free", "hidden"],
        target,
        candidates,
    )


def test_branch_prefix_curve_is_non_decreasing_for_any_support():
    pair_key, condition_name, target, candidates = paired_branch_data()
    curve = branch_prefix_curve(
        pair_key=pair_key,
        condition_name=condition_name,
        target=target,
        candidates=candidates,
        prefix_k=(1, 2, 4),
    )
    values = [
        curve[str(k)]["row_own_branch_support_rate"]
        for k in (1, 2, 4)
    ]
    assert values[0] <= values[1] <= values[2]
    assert values[0] == 0.0
    assert values[1] == 1.0


def test_branch_margin_records_preserves_conditions():
    pair_key, condition_name, target, candidates = paired_branch_data()
    result = branch_margin_records(
        pair_key=pair_key,
        condition_name=condition_name,
        target=target,
        candidates=candidates,
    )
    assert result["row_count"] == 4
    assert result["support_rate"] == 1.0
    assert set(result["condition_support_rate"]) == {"free", "hidden"}


def test_physical_valid_branch_metrics_reports_unavailable():
    pair_key, condition_name, target, candidates = paired_branch_data()
    result = physical_valid_branch_metrics(
        pair_key=pair_key,
        condition_name=condition_name,
        target=target,
        candidates=candidates,
        valid_mask=np.zeros(candidates.shape[:2], dtype=np.bool_),
    )
    assert not result["available"]
    assert result["eligible_rows"] == 0
    assert result["physical_valid_candidate_count"] == 0


def test_pair_bootstrap_is_deterministic():
    pair_key, condition_name, target, candidates = paired_branch_data()
    left = deterministic_pair_bootstrap_ci(
        pair_key=pair_key,
        condition_name=condition_name,
        target=target,
        candidates=candidates,
        resamples=128,
        seed=7,
    )
    right = deterministic_pair_bootstrap_ci(
        pair_key=pair_key,
        condition_name=condition_name,
        target=target,
        candidates=candidates,
        resamples=128,
        seed=7,
    )
    assert left == right
    assert left["point_estimate"] == 1.0


def test_classification_prioritizes_gate_self_calibration():
    spec = AttributionSpec()
    result = classify_attribution(
        spec=spec,
        training_target=physical_record(0.50),
        probe_target=physical_record(1.0),
        one_step={"10": physical_record(1.0)},
        reverse=physical_record(0.0),
        historical_branch_support=0.70,
    )
    assert result["primary_failure_locus"] == "physical_gate_contract"
    assert "self_calibration_failed" in result["root_cause"]


def test_classification_detects_probe_distribution_shift():
    spec = AttributionSpec()
    result = classify_attribution(
        spec=spec,
        training_target=physical_record(1.0),
        probe_target=physical_record(0.50),
        one_step={"10": physical_record(1.0)},
        reverse=physical_record(0.0),
        historical_branch_support=0.70,
    )
    assert result["primary_failure_locus"] == "probe_distribution_or_gate"


def test_classification_detects_x0_geometry_failure():
    spec = AttributionSpec()
    result = classify_attribution(
        spec=spec,
        training_target=physical_record(1.0),
        probe_target=physical_record(1.0),
        one_step={"10": physical_record(0.20)},
        reverse=physical_record(0.0),
        historical_branch_support=0.70,
    )
    assert result["primary_failure_locus"] == "x0_denoiser"


def test_classification_detects_reverse_transport_failure():
    spec = AttributionSpec()
    result = classify_attribution(
        spec=spec,
        training_target=physical_record(1.0),
        probe_target=physical_record(1.0),
        one_step={"10": physical_record(1.0)},
        reverse=physical_record(0.0),
        historical_branch_support=0.70,
    )
    assert result["primary_failure_locus"] == "reverse_transport"
    assert result["historical_branch_failure_masked_physical_failure"]


def test_classification_reaches_branch_only_after_physical_pass():
    spec = AttributionSpec()
    result = classify_attribution(
        spec=spec,
        training_target=physical_record(1.0),
        probe_target=physical_record(1.0),
        one_step={"10": physical_record(1.0)},
        reverse=physical_record(1.0),
        historical_branch_support=0.70,
    )
    assert result["primary_failure_locus"] == "branch_transport"


def test_compare_worker_results_detects_exact_projection():
    base = {
        "root_cause": "root",
        "required_next_path": "next",
        "frozen_replay": {
            "identity": {"model": "a"},
            "value": 1,
        },
        "physical_attribution": {"a": 1},
        "branch_attribution": {"b": 2},
        "classification": {"c": 3},
        "split": {"d": 4},
    }
    result = compare_worker_results(base, dict(base))
    assert result["exact"]
    assert result["left_sha256"] == result["right_sha256"]
