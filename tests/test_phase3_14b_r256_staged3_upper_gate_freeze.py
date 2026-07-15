from __future__ import annotations

import numpy as np
import pytest

from ccda_phase3 import phase314b_r256_stageb_cable_diffusion as stageb
from ccda_phase3 import phase314b_r256_staged_segment_recalibration as staged
from ccda_phase3.phase314b_r256_staged3_upper_gate_freeze import (
    OneSidedUpperContract,
    UpperGateSpec,
    branch_prefix_curve,
    classify_result,
    compare_worker_results,
    float64_bits,
    public_decomposition,
    recompute_upper_contract,
    sha256_array,
    upper_only_decomposition,
)


def straight_cable(
    *,
    spacing: float = 0.04,
    y: float = 0.0,
) -> np.ndarray:
    x = np.arange(stageb.BEADS, dtype=np.float32) * float(spacing)
    y_value = np.full(stageb.BEADS, float(y), dtype=np.float32)
    return np.stack([x, y_value], axis=1).reshape(stageb.CABLE_DIM)


def trajectory(spacing: float = 0.04) -> np.ndarray:
    return np.stack(
        [
            straight_cable(
                spacing=spacing,
                y=0.01 * horizon,
            )
            for horizon in range(stageb.FUTURE_STEPS)
        ],
        axis=0,
    )


def targets(rows: int = 40) -> np.ndarray:
    value = np.empty(
        (rows, stageb.FUTURE_STEPS, stageb.CABLE_DIM),
        dtype=np.float32,
    )
    for row in range(rows):
        value[row] = trajectory(
            spacing=0.04 * (1.0 + 0.001 * ((row % 5) - 2))
        )
    return value


def gate_inputs():
    value = targets(40)
    fit_mask = np.zeros(40, dtype=np.bool_)
    fit_mask[:20] = True
    calibration_mask = ~fit_mask
    groups = np.asarray(
        [f"fit_{index // 2}" for index in range(20)]
        + [f"cal_{index // 2}" for index in range(20)]
    )
    reference = staged.fit_segment_reference(
        value[fit_mask],
        scale_floor=1.0e-3,
    )
    scores = staged.segment_scores(
        value[calibration_mask],
        reference,
    )
    group_upper, _ = staged._group_score_distribution(
        scores["upper"][:, 0],
        groups[calibration_mask],
        within_group_coverage=0.95,
    )
    quantile = staged._conformal_quantile(group_upper, 0.95)
    stage_d = staged.SegmentGateContract(
        reference=reference,
        joint_threshold=100.0,
        lower_threshold=100.0,
        upper_threshold=float(quantile["threshold"]),
        within_group_row_coverage=0.95,
        conformal_group_coverage=0.95,
        fit_group_count=10,
        calibration_group_count=10,
        fit_group_sha256="a" * 64,
        calibration_group_sha256="b" * 64,
    )
    payload = stage_d.to_dict()
    payload["contract_sha256"] = "c" * 64
    split = {
        "fit_group_count": 10,
        "calibration_group_count": 10,
        "fit_group_sha256": "a" * 64,
        "calibration_group_sha256": "b" * 64,
    }
    immutable = {
        "stage_d2_raw_xyz_sha256": "d" * 64,
        "stage_d2_projection_artifact_fraction": 1.0,
    }
    return (
        value,
        groups,
        fit_mask,
        calibration_mask,
        stage_d,
        payload,
        split,
        immutable,
    )


def contract(
    *,
    upper_threshold: float = 3.0,
) -> OneSidedUpperContract:
    reference = staged.SegmentReference(
        center_log=np.full(
            (stageb.FUTURE_STEPS, stageb.BEADS - 1),
            np.log(0.04),
            dtype=np.float32,
        ),
        scale_log=np.full(
            (stageb.FUTURE_STEPS, stageb.BEADS - 1),
            0.01,
            dtype=np.float32,
        ),
        mad_scale=np.full(
            (stageb.FUTURE_STEPS, stageb.BEADS - 1),
            0.01,
            dtype=np.float32,
        ),
        iqr_scale=np.full(
            (stageb.FUTURE_STEPS, stageb.BEADS - 1),
            0.01,
            dtype=np.float32,
        ),
        scale_floor=0.01,
    )
    value = OneSidedUpperContract(
        reference=reference,
        upper_threshold=upper_threshold,
        within_group_row_coverage=0.95,
        conformal_group_coverage=0.95,
        fit_group_count=10,
        calibration_group_count=10,
        fit_group_sha256="a" * 64,
        calibration_group_sha256="b" * 64,
        calibration_group_score_sha256="c" * 64,
        stage_d_internal_contract_sha256="d" * 64,
        stage_d2_raw_xyz_sha256="e" * 64,
        stage_d2_projection_artifact_fraction=1.0,
    )
    value.validate()
    return value


def stage_d_contract() -> staged.SegmentGateContract:
    result = staged.SegmentGateContract(
        reference=contract().reference,
        joint_threshold=100.0,
        lower_threshold=100.0,
        upper_threshold=3.0,
        within_group_row_coverage=0.95,
        conformal_group_coverage=0.95,
        fit_group_count=10,
        calibration_group_count=10,
        fit_group_sha256="a" * 64,
        calibration_group_sha256="b" * 64,
    )
    result.validate()
    return result


def population(rate: float) -> dict:
    return {
        "upper_segment": {
            "row_any_rate": rate,
            "group_acceptance": {
                "group_acceptance_rate": rate,
            },
            "condition_row_any_rate": {
                "free": rate,
                "hidden": rate,
            },
        },
        "combined": {"row_any_rate": rate},
    }


def branch(eligible: float, support: float | None) -> dict:
    return {
        "eligible_row_rate": eligible,
        "support_rate_among_eligible": support,
    }


def test_spec_validates():
    UpperGateSpec().validate()


def test_spec_rejects_unsorted_k():
    with pytest.raises(ValueError):
        UpperGateSpec(branch_prefix_k=(1, 4, 2, 8)).validate()


def test_contract_validates():
    contract().validate()


def test_contract_rejects_non_projection_provenance():
    value = contract()
    with pytest.raises(ValueError):
        OneSidedUpperContract(
            reference=value.reference,
            upper_threshold=value.upper_threshold,
            within_group_row_coverage=0.95,
            conformal_group_coverage=0.95,
            fit_group_count=10,
            calibration_group_count=10,
            fit_group_sha256="a" * 64,
            calibration_group_sha256="b" * 64,
            calibration_group_score_sha256="c" * 64,
            stage_d_internal_contract_sha256="d" * 64,
            stage_d2_raw_xyz_sha256="e" * 64,
            stage_d2_projection_artifact_fraction=0.99,
        ).validate()


def test_contract_declares_upper_only_primary_gate():
    payload = contract().to_dict()
    assert payload["primary_gate"] == (
        "upper_score <= upper_threshold"
    )
    assert payload["lower_score_role"] == "diagnostic_only"
    assert not payload["uses_lower_score_for_validity"]


def test_float64_bits_distinguishes_adjacent_values():
    value = np.float64(1.0)
    adjacent = np.nextafter(value, np.float64(2.0))
    assert float64_bits(value) != float64_bits(adjacent)


def test_recompute_upper_contract_is_bit_exact():
    (
        value,
        groups,
        fit_mask,
        calibration_mask,
        stage_d,
        payload,
        split,
        immutable,
    ) = gate_inputs()
    frozen, record = recompute_upper_contract(
        target=value,
        groups=groups,
        fit_mask=fit_mask,
        calibration_mask=calibration_mask,
        gate_split=split,
        stage_d_contract=stage_d,
        stage_d_contract_payload=payload,
        immutable=immutable,
        spec=UpperGateSpec(),
    )
    assert record["threshold_bit_exact"]
    assert float64_bits(frozen.upper_threshold) == float64_bits(
        stage_d.upper_threshold
    )


def test_recompute_upper_contract_rejects_group_mismatch():
    (
        value,
        groups,
        fit_mask,
        calibration_mask,
        stage_d,
        payload,
        split,
        immutable,
    ) = gate_inputs()
    bad = dict(split)
    bad["fit_group_sha256"] = "f" * 64
    with pytest.raises(Exception):
        recompute_upper_contract(
            target=value,
            groups=groups,
            fit_mask=fit_mask,
            calibration_mask=calibration_mask,
            gate_split=bad,
            stage_d_contract=stage_d,
            stage_d_contract_payload=payload,
            immutable=immutable,
            spec=UpperGateSpec(),
        )


def evaluate(value: np.ndarray):
    target = targets(20)
    return upper_only_decomposition(
        value=value,
        contract=contract(),
        stage_d_contract=stage_d_contract(),
        historical_geometry=stageb.fit_geometry_contract(target),
        groups=np.asarray(
            [f"g{index}" for index in range(value.shape[0])]
        ),
        condition_name=np.asarray(
            [
                "free" if index % 2 == 0 else "hidden"
                for index in range(value.shape[0])
            ]
        ),
    )


def test_upper_gate_accepts_reference_geometry():
    result = evaluate(
        np.repeat(trajectory()[None], 4, axis=0)
    )
    assert result["upper_segment"]["row_any_rate"] == 1.0


def test_upper_gate_rejects_large_expansion():
    result = evaluate(
        np.repeat(
            trajectory(spacing=0.08)[None],
            4,
            axis=0,
        )
    )
    assert result["upper_segment"]["row_any_rate"] == 0.0


def test_upper_gate_ignores_xy_contraction():
    result = evaluate(
        np.repeat(
            trajectory(spacing=0.001)[None],
            4,
            axis=0,
        )
    )
    assert result["upper_segment"]["row_any_rate"] == 1.0
    assert not result["lower_diagnostic"]["used_for_validity"]


def test_upper_gate_reports_lower_diagnostic():
    result = evaluate(
        np.repeat(
            trajectory(spacing=0.001)[None],
            2,
            axis=0,
        )
    )
    assert result["lower_diagnostic"][
        "lower_only_diagnostic_invalid_count"
    ] >= 0
    assert "lower_score" in result["lower_diagnostic"]


def test_upper_gate_combines_unchanged_subgates():
    result = evaluate(
        np.repeat(trajectory()[None], 2, axis=0)
    )
    assert result["unchanged_subgates"][
        "finite_candidate_rate"
    ] == 1.0
    assert result["combined"]["row_any_rate"] == 1.0


def test_public_decomposition_removes_masks():
    result = public_decomposition(
        {
            "x": 1,
            "upper_valid_mask": np.ones((1, 1), dtype=np.bool_),
            "combined_valid_mask": np.ones((1, 1), dtype=np.bool_),
        }
    )
    assert result == {"x": 1}


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
    valid = np.ones((4, 4), dtype=np.bool_)
    return (
        ["p0", "p0", "p1", "p1"],
        ["free", "hidden", "free", "hidden"],
        target,
        candidates,
        valid,
    )


def test_branch_prefix_curve_reports_all_k():
    pair_key, condition_name, target, candidates, valid = (
        paired_branch_data()
    )
    result = branch_prefix_curve(
        pair_key=pair_key,
        condition_name=condition_name,
        target=target,
        candidates=candidates,
        valid_mask=valid,
        prefix_k=(1, 2, 4),
    )
    assert set(result) == {"1", "2", "4"}


def test_classification_detects_gate_calibration_failure():
    result = classify_result(
        spec=UpperGateSpec(),
        calibration_gt=population(0.5),
        probe_gt=population(1.0),
        one_step={"10": population(1.0)},
        reverse_final=population(1.0),
        branch_k8=branch(1.0, 1.0),
    )
    assert result["primary_failure_locus"] == (
        "upper_gate_calibration"
    )


def test_classification_detects_probe_failure():
    result = classify_result(
        spec=UpperGateSpec(),
        calibration_gt=population(1.0),
        probe_gt=population(0.5),
        one_step={"10": population(1.0)},
        reverse_final=population(1.0),
        branch_k8=branch(1.0, 1.0),
    )
    assert result["primary_failure_locus"] == (
        "upper_gate_generalization"
    )


def test_classification_detects_x0_failure():
    result = classify_result(
        spec=UpperGateSpec(),
        calibration_gt=population(1.0),
        probe_gt=population(1.0),
        one_step={"10": population(0.5)},
        reverse_final=population(1.0),
        branch_k8=branch(1.0, 1.0),
    )
    assert result["primary_failure_locus"] == (
        "x0_upper_expansion"
    )


def test_classification_detects_reverse_transport_failure():
    result = classify_result(
        spec=UpperGateSpec(),
        calibration_gt=population(1.0),
        probe_gt=population(1.0),
        one_step={"10": population(1.0)},
        reverse_final=population(0.5),
        branch_k8=branch(0.5, 1.0),
    )
    assert result["primary_failure_locus"] == (
        "reverse_upper_transport"
    )


def test_classification_detects_branch_eligibility_failure():
    reverse = population(1.0)
    result = classify_result(
        spec=UpperGateSpec(),
        calibration_gt=population(1.0),
        probe_gt=population(1.0),
        one_step={"10": population(1.0)},
        reverse_final=reverse,
        branch_k8=branch(0.5, 1.0),
    )
    assert result["primary_failure_locus"] == (
        "branch_eligibility"
    )


def test_classification_detects_branch_support_failure():
    result = classify_result(
        spec=UpperGateSpec(),
        calibration_gt=population(1.0),
        probe_gt=population(1.0),
        one_step={"10": population(1.0)},
        reverse_final=population(1.0),
        branch_k8=branch(1.0, 0.5),
    )
    assert result["primary_failure_locus"] == (
        "branch_transport"
    )


def test_classification_can_reach_supported_path():
    result = classify_result(
        spec=UpperGateSpec(),
        calibration_gt=population(1.0),
        probe_gt=population(1.0),
        one_step={"10": population(1.0)},
        reverse_final=population(1.0),
        branch_k8=branch(1.0, 1.0),
    )
    assert result["primary_failure_locus"] == "none"


def test_compare_worker_results_is_exact():
    base = {
        "root_cause": "r",
        "required_next_path": "n",
        "frozen_replay": {"identity": {"x": "a"}},
        "upper_gate_contract": {"x": 1},
        "upper_gate_fit": {"x": 1},
        "ground_truth_evaluation": {"x": 1},
        "frozen_prediction_evaluation": {"x": 1},
        "physical_branch_attribution": {"x": 1},
        "classification": {"x": 1},
        "split": {"x": 1},
    }
    assert compare_worker_results(base, dict(base))["exact"]


def test_sha256_array_changes_with_value():
    left = sha256_array(np.asarray([1.0], dtype=np.float32))
    right = sha256_array(np.asarray([2.0], dtype=np.float32))
    assert left != right
