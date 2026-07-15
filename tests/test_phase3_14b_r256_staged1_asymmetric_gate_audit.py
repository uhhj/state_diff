from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ccda_phase3 import phase314b_r256_stageb_cable_diffusion as stageb
from ccda_phase3 import phase314b_r256_staged_segment_recalibration as staged
from ccda_phase3.phase314b_r256_staged1_asymmetric_gate_audit import (
    AuditSpec,
    DirectionalThresholds,
    apply_directional_thresholds,
    attach_source_provenance,
    bonferroni_thresholds,
    classify_audit,
    deterministic_eligible_bootstrap,
    directional_element_pass,
    evaluate_gate_mask,
    gate_difference,
    load_stage_d_gate,
    physical_branch_curve,
    public_evaluation,
    ratio_audit,
    sha256_bytes,
    stable_json_bytes,
    stage_d_masks,
    threshold_driver_audit,
)


def straight_cable(spacing: float = 0.04, y: float = 0.0) -> np.ndarray:
    x = np.arange(stageb.BEADS, dtype=np.float32) * float(spacing)
    y_value = np.full(stageb.BEADS, float(y), dtype=np.float32)
    return np.stack([x, y_value], axis=1).reshape(stageb.CABLE_DIM)


def trajectory(spacing: float = 0.04) -> np.ndarray:
    return np.stack(
        [
            straight_cable(spacing=spacing, y=0.01 * horizon)
            for horizon in range(stageb.FUTURE_STEPS)
        ],
        axis=0,
    )


def targets(rows: int = 40) -> np.ndarray:
    result = np.empty(
        (rows, stageb.FUTURE_STEPS, stageb.CABLE_DIM),
        dtype=np.float32,
    )
    for row in range(rows):
        result[row] = trajectory(
            spacing=0.04 * (1.0 + 0.001 * ((row % 5) - 2))
        )
    return result


def reference(scale: float = 0.01) -> staged.SegmentReference:
    shape = (stageb.FUTURE_STEPS, stageb.BEADS - 1)
    value = staged.SegmentReference(
        center_log=np.full(shape, np.log(0.04), dtype=np.float32),
        scale_log=np.full(shape, scale, dtype=np.float32),
        mad_scale=np.full(shape, scale, dtype=np.float32),
        iqr_scale=np.full(shape, scale, dtype=np.float32),
        scale_floor=float(scale),
    )
    value.validate()
    return value


def contract(
    *,
    joint: float = 100.0,
    lower: float = 100.0,
    upper: float = 3.0,
) -> staged.SegmentGateContract:
    value = staged.SegmentGateContract(
        reference=reference(),
        joint_threshold=joint,
        lower_threshold=lower,
        upper_threshold=upper,
        within_group_row_coverage=0.95,
        conformal_group_coverage=0.95,
        fit_group_count=10,
        calibration_group_count=10,
        fit_group_sha256="a" * 64,
        calibration_group_sha256="b" * 64,
    )
    value.validate()
    return value


def population(rate: float) -> dict:
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
        "combined": {"row_any_rate": rate},
    }


def branch_record(eligible: float, support: float) -> dict:
    return {
        "eligible_row_rate": eligible,
        "support_rate_among_eligible": support,
    }


def classification_inputs(
    *,
    calibration: float = 1.0,
    probe: float = 1.0,
    reverse: float = 1.0,
    eligible: float = 1.0,
    support: float = 1.0,
):
    return {
        "calibration_results": {
            "bonferroni_asymmetric": population(calibration)
        },
        "probe_results": {
            "bonferroni_asymmetric": population(probe)
        },
        "reverse_results": {
            "bonferroni_asymmetric": population(reverse)
        },
        "branch_results": {
            "bonferroni_asymmetric": {
                "k8": branch_record(eligible, support)
            }
        },
    }


def test_audit_spec_validates():
    AuditSpec().validate()


def test_audit_spec_rejects_wrong_bonferroni_coverage():
    with pytest.raises(ValueError):
        AuditSpec(one_sided_bonferroni_coverage=0.95).validate()


def test_directional_thresholds_validate():
    DirectionalThresholds(
        name="x",
        lower=10.0,
        upper=2.0,
        calibration_coverage=0.95,
        source="test",
    ).validate()


def test_stage_d_joint_gate_relaxes_upper_side():
    ref = reference(scale=0.01)
    candidate = np.repeat(trajectory(spacing=0.05)[None], 1, axis=0)
    scores = staged.segment_scores(candidate, ref)
    masks = stage_d_masks(scores, contract())
    assert bool(masks["stage_d_joint"][0, 0])
    assert not bool(masks["stage_d_naive_asymmetric"][0, 0])


def test_apply_directional_thresholds_uses_both_sides():
    ref = reference(scale=0.01)
    candidate = np.repeat(trajectory(spacing=0.05)[None], 1, axis=0)
    scores = staged.segment_scores(candidate, ref)
    thresholds = DirectionalThresholds(
        name="asym",
        lower=100.0,
        upper=3.0,
        calibration_coverage=0.95,
        source="test",
    )
    assert not bool(apply_directional_thresholds(scores, thresholds)[0, 0])


def test_directional_element_pass_uses_separate_thresholds():
    ref = reference(scale=0.01)
    candidate = np.repeat(trajectory(spacing=0.05)[None], 1, axis=0)
    scores = staged.segment_scores(candidate, ref)
    mask = directional_element_pass(
        scores,
        lower_threshold=100.0,
        upper_threshold=3.0,
    )
    assert mask.shape == (1, 1, 4, 23)
    assert not np.all(mask)


def test_bonferroni_thresholds_are_finite():
    value = targets(20)
    ref = staged.fit_segment_reference(value[:10], scale_floor=1.0e-3)
    scores = staged.segment_scores(value[10:], ref)
    groups = np.asarray([f"g{index}" for index in range(10)])
    threshold, record = bonferroni_thresholds(
        calibration_scores=scores,
        calibration_groups=groups,
        contract=staged.SegmentGateContract(
            reference=ref,
            joint_threshold=10.0,
            lower_threshold=10.0,
            upper_threshold=10.0,
            within_group_row_coverage=0.95,
            conformal_group_coverage=0.95,
            fit_group_count=10,
            calibration_group_count=10,
            fit_group_sha256="a" * 64,
            calibration_group_sha256="b" * 64,
        ),
        spec=AuditSpec(),
    )
    assert np.isfinite(threshold.lower)
    assert np.isfinite(threshold.upper)
    assert record["one_sided_coverage"] == 0.975


def test_gate_difference_counts_directional_disagreement():
    left = np.asarray([[True, True], [False, False]])
    right = np.asarray([[True, False], [True, False]])
    result = gate_difference(left, right)
    assert result["different"] == 2
    assert result["left_only"] == 1
    assert result["right_only"] == 1


def test_evaluate_gate_mask_reports_row_any():
    value = np.repeat(trajectory()[None, None], 4, axis=0)
    value = np.repeat(value, 2, axis=1)
    candidate_mask = np.ones((4, 2), dtype=np.bool_)
    element_mask = np.ones((4, 2, 4, 23), dtype=np.bool_)
    result = evaluate_gate_mask(
        value=value,
        segment_mask=candidate_mask,
        element_mask=element_mask,
        historical_geometry=stageb.fit_geometry_contract(targets(20)),
        groups=np.asarray(["g0", "g0", "g1", "g1"]),
        condition_name=np.asarray(
            ["free", "hidden_slack_breakaway_pin_v2"] * 2
        ),
    )
    assert result["segment"]["row_any_rate"] == 1.0
    assert result["combined"]["row_any_rate"] == 1.0


def test_public_evaluation_removes_masks():
    public = public_evaluation(
        {
            "x": 1,
            "segment_valid_mask": np.ones((1, 1), dtype=np.bool_),
            "combined_valid_mask": np.ones((1, 1), dtype=np.bool_),
        }
    )
    assert public == {"x": 1}


def test_ratio_audit_detects_severe_collapse():
    value = targets(4)
    collapsed = value.reshape(4, 4, 24, 2)
    collapsed[0, 0, 1] = collapsed[0, 0, 0] + np.asarray(
        [0.001, 0.0],
        dtype=np.float32,
    )
    result = ratio_audit(
        target=value,
        reference=reference(),
        metadata={
            "episode_group_key": np.asarray(["a", "b", "c", "d"]),
        },
        spec=AuditSpec(),
    )
    assert result["row_count_with_severe_collapse"] >= 1
    assert result["lowest_rows"][0]["minimum_ratio"] < 0.25


def test_threshold_driver_detects_directional_coupling():
    value = targets(8)
    reshaped = value.reshape(8, 4, 24, 2)
    reshaped[0, 0, 1] = reshaped[0, 0, 0] + np.asarray(
        [0.001, 0.0],
        dtype=np.float32,
    )
    result = threshold_driver_audit(
        calibration_target=value,
        calibration_groups=np.asarray(
            ["g0", "g0", "g1", "g1", "g2", "g2", "g3", "g3"]
        ),
        calibration_metadata={
            "condition_name": np.asarray(["free", "hidden"] * 4),
        },
        contract=contract(joint=300.0, lower=300.0, upper=3.0),
        spec=AuditSpec(),
    )
    assert result["directional_thresholds"]["directionally_coupled"]
    assert result["severe_collapse_drives_lower_tail"]


def test_attach_source_provenance_maps_source_rows():
    train = {"source_row_index": np.asarray([2, 0], dtype=np.int64)}
    full = {
        "source_file": np.asarray(["a", "b", "c"]),
        "row_index": np.asarray([10, 11, 12]),
    }
    result = attach_source_provenance(
        train_arrays=train,
        full_contract=full,
    )
    assert result["source_file"].tolist() == ["c", "a"]
    assert result["row_index"].tolist() == [12, 10]


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


def test_physical_branch_curve_reports_all_k():
    pair_key, condition_name, target, candidates, valid = (
        paired_branch_data()
    )
    result = physical_branch_curve(
        pair_key=pair_key,
        condition_name=condition_name,
        target=target,
        candidates=candidates,
        valid_mask=valid,
        prefix_k=(1, 2, 4),
    )
    assert set(result) == {"1", "2", "4"}


def test_bootstrap_reports_unavailable_without_eligible_rows():
    result = deterministic_eligible_bootstrap(
        {"rows": [{"eligible": False, "supported": None}]},
        resamples=128,
        seed=4,
    )
    assert not result["available"]
    assert result["ci95"] is None


def test_bootstrap_is_deterministic():
    branch = {
        "rows": [
            {"eligible": True, "supported": True},
            {"eligible": True, "supported": False},
            {"eligible": True, "supported": True},
        ]
    }
    left = deterministic_eligible_bootstrap(
        branch,
        resamples=128,
        seed=4,
    )
    right = deterministic_eligible_bootstrap(
        branch,
        resamples=128,
        seed=4,
    )
    assert left == right
    assert left["available"]


def test_classification_prioritizes_collapse_provenance():
    inputs = classification_inputs()
    result = classify_audit(
        spec=AuditSpec(),
        threshold_driver={
            "directional_thresholds": {"directionally_coupled": True},
            "severe_collapse_drives_lower_tail": True,
        },
        **inputs,
    )
    assert result["primary_failure_locus"] == (
        "ground_truth_segment_collapse_provenance"
    )


def test_classification_detects_bonferroni_calibration_failure():
    inputs = classification_inputs(calibration=0.5)
    result = classify_audit(
        spec=AuditSpec(),
        threshold_driver={
            "directional_thresholds": {"directionally_coupled": True},
            "severe_collapse_drives_lower_tail": False,
        },
        **inputs,
    )
    assert result["primary_failure_locus"] == (
        "directional_gate_calibration"
    )


def test_classification_detects_probe_failure():
    inputs = classification_inputs(probe=0.5)
    result = classify_audit(
        spec=AuditSpec(),
        threshold_driver={
            "directional_thresholds": {"directionally_coupled": True},
            "severe_collapse_drives_lower_tail": False,
        },
        **inputs,
    )
    assert result["primary_failure_locus"] == (
        "directional_gate_generalization"
    )


def test_classification_detects_physical_coverage_failure():
    inputs = classification_inputs(reverse=0.5)
    result = classify_audit(
        spec=AuditSpec(),
        threshold_driver={
            "directional_thresholds": {"directionally_coupled": False},
            "severe_collapse_drives_lower_tail": False,
        },
        **inputs,
    )
    assert result["primary_failure_locus"] == (
        "physical_candidate_coverage"
    )


def test_classification_detects_branch_failure():
    inputs = classification_inputs(support=0.5)
    result = classify_audit(
        spec=AuditSpec(),
        threshold_driver={
            "directional_thresholds": {"directionally_coupled": False},
            "severe_collapse_drives_lower_tail": False,
        },
        **inputs,
    )
    assert result["primary_failure_locus"] == "branch_transport"


def test_classification_can_reach_supported_path():
    inputs = classification_inputs()
    result = classify_audit(
        spec=AuditSpec(),
        threshold_driver={
            "directional_thresholds": {"directionally_coupled": False},
            "severe_collapse_drives_lower_tail": False,
        },
        **inputs,
    )
    assert result["primary_failure_locus"] == "none"


def test_load_stage_d_gate_verifies_internal_sha(tmp_path: Path):
    value = contract()
    payload = value.to_dict()
    payload["contract_sha256"] = sha256_bytes(
        stable_json_bytes(payload)
    )
    payload.update(
        {
            "phase": "x",
            "base_evidence_commit": "y",
            "frozen_model_sha256": "m",
            "frozen_reverse_candidate_sha256": "r",
        }
    )
    path = tmp_path / "gate.json"
    path.write_text(
        json.dumps(payload, sort_keys=True),
        encoding="utf-8",
    )
    loaded, raw = load_stage_d_gate(path)
    assert loaded.joint_threshold == value.joint_threshold
    assert raw["contract_sha256"] == payload["contract_sha256"]
