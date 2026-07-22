from __future__ import annotations

from types import SimpleNamespace
from typing import Optional

import numpy as np
import pytest

from ccda_phase3 import (
    phase314b_r258_stager_tolerance_aligned_oof_post_upper_audit as stager,
)


def attempt(
    *,
    active: int = 10,
    accepted: int = 0,
    predicate: str = "direction_retention",
    failure: Optional[int] = None,
):
    count = active - accepted if failure is None else failure
    failures = {name: 0 for name in stager.PREDICATE_ORDER}
    failures[predicate] = count
    return {
        "shadow_active_row_count": active,
        "shadow_accepted_count": accepted,
        "shadow_first_failed_counts": failures,
    }


def assembly_for(
    predicate: str,
    *,
    active: int = 10,
    accepted: int = 0,
):
    return stager.aggregate_attempt_records(
        [attempt(active=active, accepted=accepted, predicate=predicate)]
    )


def synthetic_cell(
    backbone: str,
    timestep: int,
    *,
    acceptance: float = 0.0,
    discriminator: Optional[str] = "direction_retention",
    mismatch: int = 0,
    regression: int = 0,
    aligned_upper: int = 0,
    agreement: bool = True,
):
    raw_predicate = discriminator
    projected_predicate = discriminator if agreement else "displacement"
    comparison = lambda predicate: {"discriminator_predicate": predicate}
    return {
        "base_direction_id": backbone,
        "timestep": timestep,
        "aligned_acceptance_rate": acceptance,
        "length_log_z_element_mismatch_count": mismatch,
        "strict_pass_aligned_fail_row_count": regression,
        "aligned_upper_element_failure_count": aligned_upper,
        "comparator_discriminator_predicate": discriminator,
        "dual_oracle_discriminator_predicate": discriminator if agreement else None,
        "oracle_discriminator_agreement": agreement,
        "raw_oracle_control": {"comparison": comparison(raw_predicate)},
        "projected_oracle_control": {
            "comparison": comparison(projected_predicate)
        },
    }


def full_cells(**kwargs):
    return [
        synthetic_cell(f"b{backbone}", timestep, **kwargs)
        for backbone in range(9)
        for timestep in stager.EXPECTED_TIMESTEPS
    ]


def test_phase_and_schema():
    assert stager.PHASE.endswith("Stage R")
    assert stager.SCHEMA.endswith("_v1")


def test_starting_commit_is_stageq_resume1_evidence():
    assert stager.BASE_EVIDENCE_COMMIT == (
        "b47d2d848ce1a4f4109f5937d9f224897e1301cb"
    )


def test_base_report_sha_is_frozen():
    assert stager.EXPECTED_BASE_REPORT_SHA256 == (
        "0f5383ba53f415525f69875591165c0cbfafe26469c3aba846b4a8b3e8c60f24"
    )


def test_stagel_report_sha_is_frozen():
    assert stager.EXPECTED_STAGEL_REPORT_SHA256 == (
        "120cf31f649342589e74e3a5b8ce97bb50f4aac33f865fae54ebebef02f79eed"
    )


def test_population_contract():
    assert stager.EXPECTED_BACKBONE_COUNT == 9
    assert stager.EXPECTED_CELL_COUNT == 27
    assert stager.EXPECTED_CALLBACK_PAIR_COUNT == 27
    assert stager.EXPECTED_INTERNAL_ATTEMPT_COUNT == 189


def test_implementation_population_is_add_only():
    assert len(stager.IMPLEMENTATION_PATHS) == 3
    assert all(status == "A" for status, _ in stager.IMPLEMENTATION_PATHS)


def test_predicate_order_matches_frozen_callback():
    assert stager.PREDICATE_ORDER[0] == "finite_state"
    assert stager.PREDICATE_ORDER[1] == "upper_segment_geometry"
    assert stager.PREDICATE_ORDER[-1] == "topology"
    assert len(stager.PREDICATE_ORDER) == 10


def test_spec_defaults_validate():
    stager.StageRSpec().validate()


@pytest.mark.parametrize(
    "field,value",
    [
        ("external_multiplier", 0.5),
        ("grouped_cv_folds", 5),
        ("stable_cell_support_min", 23),
        ("sparse_other_cell_support_max", 2),
        ("stable_backbone_coverage_min", 7),
        ("reconstruction_tolerance_factor", 4.0),
        ("binary_tolerance", 0.0),
        ("candidate_motion_epsilon", 0.0),
    ],
)
def test_spec_rejects_contract_change(field, value):
    values = stager.asdict(stager.StageRSpec())
    values[field] = value
    with pytest.raises(stager.StageRError):
        stager.StageRSpec(**values).validate()


@pytest.mark.parametrize(
    "field,value",
    [
        ("oracle_acceptance_rate_min", -0.1),
        ("oof_rejection_rate_max", 1.1),
        ("oracle_conditional_pass_rate_min", float("nan")),
        ("oof_conditional_pass_rate_max", -1.0),
        ("oof_rejection_mass_min", 2.0),
        ("oracle_rejection_mass_max", -0.1),
    ],
)
def test_spec_rejects_invalid_rate(field, value):
    values = stager.asdict(stager.StageRSpec())
    values[field] = value
    with pytest.raises(stager.StageRError):
        stager.StageRSpec(**values).validate()


def test_selected_scale_histogram_closes():
    result = stager.selected_scale_histogram(
        np.asarray([0.0, 0.25, 0.25, 1.0], dtype=np.float64)
    )
    assert result == {"0": 1, "0.25": 2, "1": 1}


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
def test_selected_scale_histogram_rejects_invalid(bad):
    with pytest.raises(stager.StageRError):
        stager.selected_scale_histogram(np.asarray([0.0, bad]))


def test_aggregate_attempt_records_closes():
    result = stager.aggregate_attempt_records(
        [attempt(active=10, accepted=2, predicate="direction_retention")]
    )
    assert result["active_row_attempt_count"] == 10
    assert result["accepted_row_attempt_count"] == 2
    assert result["rejected_row_attempt_count"] == 8
    assert result["dominant_failure_predicate"] == "direction_retention"
    assert result["sequential_population_closed"] is True


def test_aggregate_attempt_records_multiple_scales():
    result = stager.aggregate_attempt_records(
        [
            attempt(active=10, accepted=2, predicate="direction_retention"),
            attempt(active=8, accepted=3, predicate="displacement"),
        ]
    )
    assert result["attempt_count"] == 2
    assert result["active_row_attempt_count"] == 18
    assert result["accepted_row_attempt_count"] == 5
    assert result["first_failed_counts"]["direction_retention"] == 8
    assert result["first_failed_counts"]["displacement"] == 5


def test_aggregate_attempt_records_rejects_empty():
    with pytest.raises(stager.StageRError):
        stager.aggregate_attempt_records([])


def test_aggregate_attempt_records_rejects_overaccepted():
    value = attempt(active=2, accepted=3, predicate="direction_retention", failure=0)
    with pytest.raises(stager.StageRError):
        stager.aggregate_attempt_records([value])


def test_aggregate_attempt_records_rejects_nonclosing_population():
    value = attempt(active=10, accepted=1, predicate="direction_retention", failure=8)
    with pytest.raises(stager.StageRError):
        stager.aggregate_attempt_records([value])


def test_compare_with_oracle_identifies_conditional_discriminator():
    spec = stager.StageRSpec()
    oof = assembly_for("direction_retention")
    oracle = assembly_for("direction_retention", accepted=10)
    result = stager.compare_with_oracle(oof=oof, oracle=oracle, spec=spec)
    assert result["discriminator_predicate"] == "direction_retention"
    assert result["discriminator_mode"] in {
        "conditional_pass",
        "conditional_and_rejection_mass",
    }


def test_compare_with_oracle_returns_none_when_surfaces_match():
    spec = stager.StageRSpec()
    value = assembly_for("direction_retention")
    result = stager.compare_with_oracle(oof=value, oracle=value, spec=spec)
    assert result["discriminator_predicate"] is None


def test_compare_with_oracle_uses_predicate_order():
    spec = stager.StageRSpec()
    oof_records = [attempt(predicate="lower_segment_geometry")]
    oracle_records = [attempt(accepted=10, predicate="lower_segment_geometry")]
    result = stager.compare_with_oracle(
        oof=stager.aggregate_attempt_records(oof_records),
        oracle=stager.aggregate_attempt_records(oracle_records),
        spec=spec,
    )
    assert result["discriminator_predicate"] == "lower_segment_geometry"


def test_classify_mask_mismatch_has_priority():
    cells = full_cells(mismatch=1)
    result = stager.classify_oof_post_upper(cells)
    assert result["required_next_path"] == (
        "AUDIT_LENGTH_LOG_Z_UPPER_TOLERANCE_NUMERICS_ON_OOF"
    )


def test_classify_regression_has_priority_after_equivalence():
    cells = full_cells(regression=1)
    result = stager.classify_oof_post_upper(cells)
    assert result["primary_failure_locus"] == "strict_pass_aligned_fail_regression"


def test_classify_aligned_upper_failure():
    cells = full_cells(aligned_upper=1)
    result = stager.classify_oof_post_upper(cells)
    assert result["primary_failure_locus"] == "aligned_upper_violation_remains"


def test_classify_all_oracle_like():
    cells = full_cells(acceptance=0.96, discriminator=None)
    result = stager.classify_oof_post_upper(cells)
    assert result["required_next_path"] == (
        "FREEZE_TOLERANCE_ALIGNED_GATE_AND_CONFIRM_OOF_CANDIDATE_MATRIX"
    )
    assert result["oracle_like_admission_cell_count"] == 27


def test_classify_broad_support():
    cells = full_cells(acceptance=0.10, discriminator=None)
    result = stager.classify_oof_post_upper(cells)
    assert result["primary_failure_locus"] == "broad_oof_support_restored"
    assert result["meaningful_support_cell_count"] == 27


@pytest.mark.parametrize(
    "predicate,next_path",
    [
        (
            "finite_state",
            "AUDIT_OOF_FINITE_STATE_AFTER_UPPER_TOLERANCE_ALIGNMENT",
        ),
        (
            "lower_segment_geometry",
            "AUDIT_OOF_LOWER_SEGMENT_GATE_AFTER_UPPER_TOLERANCE_ALIGNMENT",
        ),
        (
            "coordinate_recenter",
            "AUDIT_OOF_LOCAL_FRAME_AFTER_UPPER_TOLERANCE_ALIGNMENT",
        ),
        (
            "coordinate_geometry",
            "AUDIT_OOF_LOCAL_FRAME_AFTER_UPPER_TOLERANCE_ALIGNMENT",
        ),
        (
            "reconstruction_bounds",
            "AUDIT_OOF_RECONSTRUCTION_BOUND_CLOSURE_AFTER_UPPER_TOLERANCE_ALIGNMENT",
        ),
        (
            "segment_geometry",
            "ALIGN_HISTORICAL_SEGMENT_BOUND_TOLERANCE_WITH_RECONSTRUCTION_CONTRACT",
        ),
        (
            "direction_retention",
            "CALIBRATE_INTEGRATOR_COMPATIBLE_DIRECTION_BASIS_ON_OBJECTIVE_TRAIN_ONLY",
        ),
        (
            "displacement",
            "CALIBRATE_EXECUTABLE_DIRECTION_MAGNITUDE_PARAMETERIZATION",
        ),
        (
            "topology",
            "CALIBRATE_TOPOLOGY_COMPATIBLE_DIRECTION_SURROGATE_ON_OBJECTIVE_TRAIN_ONLY",
        ),
    ],
)
def test_classify_stable_post_upper_predicate(predicate, next_path):
    cells = full_cells(discriminator=predicate)
    result = stager.classify_oof_post_upper(cells)
    assert result["required_next_path"] == next_path
    assert result["dominant_dual_oracle_discriminator"] == predicate
    assert result["dominant_discriminator_support_count"] == 27
    assert result["dominant_discriminator_backbone_coverage"] == 9
    assert result["dominant_discriminator_timestep_coverage"] == 3


def test_classify_oracle_source_dependent():
    cells = full_cells(agreement=False)
    result = stager.classify_oof_post_upper(cells)
    assert result["required_next_path"] == (
        "STRATIFY_TOLERANCE_ALIGNED_OOF_REJECTION_BY_ORACLE_SOURCE"
    )


def test_classify_heterogeneous_by_backbone():
    cells = full_cells(discriminator="direction_retention")
    for index in range(6):
        cells[index] = synthetic_cell(
            str(cells[index]["base_direction_id"]),
            int(cells[index]["timestep"]),
            discriminator="displacement",
        )
    result = stager.classify_oof_post_upper(cells)
    assert result["primary_failure_locus"] == "heterogeneous_post_upper_rejection"


def test_classify_scalar_surface_insufficient():
    cells = full_cells(discriminator=None)
    result = stager.classify_oof_post_upper(cells)
    assert result["primary_failure_locus"] == "post_upper_scalar_surface_insufficient"


def test_classify_rejects_wrong_cell_count():
    with pytest.raises(stager.StageRError):
        stager.classify_oof_post_upper(full_cells()[:-1])


def test_classify_rejects_wrong_backbone_population():
    cells = full_cells()
    for cell in cells:
        cell["base_direction_id"] = "same"
    with pytest.raises(stager.StageRError):
        stager.classify_oof_post_upper(cells)


def test_oracle_control_map_accepts_six_admitted_cells():
    cells = []
    for source in stager.ORACLE_SOURCES:
        for timestep in stager.EXPECTED_TIMESTEPS:
            cells.append(
                {
                    "source_id": source,
                    "timestep": timestep,
                    "aligned_oracle_admitted": True,
                    "aligned_oracle_acceptance_rate": 0.99,
                }
            )
    result = stager.oracle_control_map(
        {"tolerance_aligned_upper_gate_shadow": {"cell_records": cells}}
    )
    assert len(result) == 6


def test_oracle_control_map_rejects_nonadmitted_cell():
    cells = []
    for source in stager.ORACLE_SOURCES:
        for timestep in stager.EXPECTED_TIMESTEPS:
            cells.append(
                {
                    "source_id": source,
                    "timestep": timestep,
                    "aligned_oracle_admitted": source != "raw_oracle" or timestep != 10,
                    "aligned_oracle_acceptance_rate": 0.99,
                }
            )
    with pytest.raises(stager.StageRError):
        stager.oracle_control_map(
            {"tolerance_aligned_upper_gate_shadow": {"cell_records": cells}}
        )


def test_stagel_oof_record_map_accepts_27_records():
    records = [
        {"base_direction_id": f"b{backbone}", "timestep": timestep}
        for backbone in range(9)
        for timestep in stager.EXPECTED_TIMESTEPS
    ]
    scientific = {"predicate_assembly_audit": {"oof_records": records}}
    result = stager.stagel_oof_record_map(scientific)
    assert len(result) == 27


def test_stagel_oof_record_map_rejects_duplicate():
    records = [
        {"base_direction_id": "b0", "timestep": 10}
        for _ in range(27)
    ]
    scientific = {"predicate_assembly_audit": {"oof_records": records}}
    with pytest.raises(stager.StageRError):
        stager.stagel_oof_record_map(scientific)


def test_persisted_multiplier_record_selects_025():
    result = stager._persisted_multiplier_record(
        {
            "multiplier_records": [
                {"scale_multiplier": 0.25, "id": "low"},
                {"scale_multiplier": 0.5, "id": "high"},
            ]
        },
        0.25,
    )
    assert result["id"] == "low"


def test_persisted_multiplier_record_rejects_duplicate():
    with pytest.raises(stager.StageRError):
        stager._persisted_multiplier_record(
            {
                "multiplier_records": [
                    {"scale_multiplier": 0.25},
                    {"scale_multiplier": 0.25},
                ]
            },
            0.25,
        )


def test_validate_legacy_oof_identity_accepts_correct_hash_domains():
    control = np.zeros((2, 1, 2), dtype=np.float32)
    candidate = np.ones_like(control)
    selected = np.asarray([0.25, 0.0], dtype=np.float64)
    proposed = np.ones_like(control, dtype=np.float64) * 0.25
    events = [
        {
            "event_type": "scale_attempt",
            "predicate_order": stager.PREDICATE_ORDER,
            "active_row_count": 2,
            "accepted_count": 1,
            "first_failed_counts": {
                **{name: 0 for name in stager.PREDICATE_ORDER},
                "direction_retention": 1,
            },
            "predicate_pass_counts": {
                name: 2 for name in stager.PREDICATE_ORDER
            },
            "topology_checked_count": 1,
        }
    ]

    def array_hash(value):
        return stager.sha256_array(value)

    def aggregate(values):
        return {"events": len(values)}

    stagel = SimpleNamespace(
        sha256_array=array_hash,
        aggregate_sequential_attempts=aggregate,
    )
    stagek = SimpleNamespace(
        _row_norm=lambda value: np.sqrt(
            np.sum(np.asarray(value).reshape(value.shape[0], -1) ** 2, axis=1)
        )
    )
    motion = stagek._row_norm(candidate.astype(np.float64))
    selected_positive = selected > 1.0e-12
    motion_positive = motion > 1.0e-12
    # Match selected/motion observables for this synthetic fixture.
    candidate[1] = 0.0
    motion = stagek._row_norm(candidate.astype(np.float64))
    motion_positive = motion > 1.0e-12
    assert np.array_equal(selected_positive, motion_positive)
    persisted_multiplier = {
        "scale_multiplier": 0.25,
        "proposed_direction_sha256": array_hash(proposed),
        "selected_scale_positive_rate": float(np.mean(selected_positive)),
        "selected_scale_sha256": array_hash(selected),
        "candidate_motion_positive_rate": float(np.mean(motion_positive)),
        "candidate_sha256": array_hash(candidate),
        "callback_capture_sha256": "capture",
        "callback_result_bit_exact": True,
        "assembly": aggregate(events),
    }
    persisted_record = {
        "base_direction_id": "b0",
        "timestep": 10,
        "feature_sha256": "feature",
        "oof_prediction_sha256": "prediction",
        "oof_predicate_assembly": {
            "multiplier_records": [persisted_multiplier]
        },
    }
    integration = {"selected_scale": selected, "candidate": candidate}
    capture = {
        "events": events,
        "events_sha256": "capture",
        "returned_result_bit_exact": True,
    }
    result = stager.validate_legacy_oof_identity(
        base_direction_id="b0",
        timestep=10,
        feature_sha256="feature",
        prediction_sha256="prediction",
        proposed_direction=proposed,
        integration=integration,
        capture=capture,
        persisted_record=persisted_record,
        stagel=stagel,
        stagek=stagek,
        spec=stager.StageRSpec(),
        control=control,
    )
    assert result["all_exact"] is True


def test_blocked_report_is_fail_closed():
    payload = stager.blocked_report(
        repository={"head": "x"}, error=RuntimeError("synthetic")
    )
    assert payload["execution_verdict"] == "BLOCKED"
    assert payload["scientific_status"] == "BLOCKED"
    assert payload["selected_configuration"] is None
    assert payload["train_only_recommendation"] is None
    assert payload["oof_surrogate_weights_persisted"] is False


def test_false_boundary_population_is_unique():
    assert len(stager.FALSE_BOUNDARIES) == len(set(stager.FALSE_BOUNDARIES))


def test_expected_environment_contract_is_complete():
    assert stager.EXPECTED_ENV["PYTHONHASHSEED"] == "0"
    assert stager.EXPECTED_ENV["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert len(stager.EXPECTED_ENV) == 9


def test_frozen_source_population_contains_hash_sensitive_modules():
    keys = set(stager.FROZEN_SOURCE_SHA256)
    assert any("stagee_constrained_integrator" in key for key in keys)
    assert any("stageh_candidate_descriptor" in key for key in keys)
    assert any("stagel_predicate_assembly" in key for key in keys)
    assert any("stageq_tolerance_aligned" in key for key in keys)
    assert any("stageq_resume1_candidate_hash" in key for key in keys)
