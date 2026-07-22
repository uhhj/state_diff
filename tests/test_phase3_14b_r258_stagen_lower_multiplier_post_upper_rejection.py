from __future__ import annotations

import copy
import math
from pathlib import Path

import pytest

from ccda_phase3 import phase314b_r258_stagen_lower_multiplier_post_upper_rejection as stagen


def comparison(predicate: str, *, discriminate: bool = False):
    return {
        "predicate": predicate,
        "oof_conditional_pass_rate": 0.0 if discriminate else 1.0,
        "oracle_conditional_pass_rate": 1.0,
        "oof_rejection_mass_rate": 1.0 if discriminate else 0.0,
        "oracle_rejection_mass_rate": 0.0,
        "conditional_discriminator": discriminate,
        "rejection_mass_discriminator": discriminate,
    }


def upper_metrics(*, pass_rate: float = 1.0, mass: float = 0.0):
    return {
        "active_row_attempt_count": 10,
        "accepted_row_attempt_count": 0 if mass else 10,
        "sequential_reach_count": 10,
        "first_failed_count": int(round(10 * mass)),
        "conditional_failure_rate": 1.0 - pass_rate,
        "rejection_mass_rate": mass,
    }


def cell(
    multiplier: float,
    *,
    local_predicate=None,
    post_predicate=None,
    oracle_acceptance: float = 1.0,
    oof_acceptance: float = 0.0,
    upper_pass: float = 1.0,
    upper_mass: float = 0.0,
):
    comparisons = [
        comparison(name, discriminate=(name == post_predicate))
        for name in stagen.PREDICATE_ORDER
    ]
    upper = next(item for item in comparisons if item["predicate"] == "upper_segment_geometry")
    upper["oof_conditional_pass_rate"] = upper_pass
    upper["oof_rejection_mass_rate"] = upper_mass
    upper["conditional_discriminator"] = local_predicate == "upper_segment_geometry"
    upper["rejection_mass_discriminator"] = local_predicate == "upper_segment_geometry"
    return {
        "external_multiplier": multiplier,
        "stratum_key": stagen._scale_key(multiplier),
        "comparator_source": "projected_oracle",
        "oof_acceptance_rate": oof_acceptance,
        "comparator_oracle_acceptance_rate": oracle_acceptance,
        "raw_oracle_acceptance_rate": oracle_acceptance,
        "projected_oracle_acceptance_rate": oracle_acceptance,
        "oracle_locally_admitted": oracle_acceptance >= 0.95,
        "oof_locally_rejected": oof_acceptance <= 0.05,
        "locally_eligible": oracle_acceptance >= 0.95 and oof_acceptance <= 0.05,
        "stage_l_stored_discriminator_predicate": local_predicate,
        "recomputed_ungated_discriminator_predicate": local_predicate,
        "local_discriminator_predicate": local_predicate,
        "locus": "x",
        "upper_segment_oof": upper_metrics(pass_rate=upper_pass, mass=upper_mass),
        "upper_segment_oracle": upper_metrics(),
        "predicate_comparisons": comparisons,
    }


def source_record(base: str, timestep: int, *, post_predicate="direction_retention", **lower_kwargs):
    return {
        "base_direction_id": base,
        "timestep": timestep,
        "feature_mode": "paper_x",
        "strata": [
            cell(0.25, post_predicate=post_predicate, **lower_kwargs),
            cell(
                0.5,
                local_predicate="upper_segment_geometry",
                upper_pass=0.0,
                upper_mass=1.0,
            ),
            cell(1.0),
            cell(2.0),
        ],
    }


def records(*, post_predicate="direction_retention", **lower_kwargs):
    return [
        source_record(f"b{index}", timestep, post_predicate=post_predicate, **lower_kwargs)
        for index in range(9)
        for timestep in stagen.EXPECTED_TIMESTEPS
    ]


def stage_m_report(*, post_predicate="direction_retention", **lower_kwargs):
    false_values = {key: False for key in stagen.FALSE_BOUNDARIES}
    return {
        "execution_verdict": "PASS",
        "scientific_status": "BLOCKED",
        "root_cause": stagen.EXPECTED_STAGEM_ROOT_CAUSE,
        "required_next_path": stagen.EXPECTED_STAGEM_NEXT_PATH,
        "selected_configuration": None,
        "train_only_recommendation": None,
        "classification": {
            "record_count": 27,
            "stratum_count": 108,
            "dominant_upper_multiplier": 0.5,
            "dominant_upper_support_count": 27,
            "maximum_other_multiplier_upper_support_count": 0,
            "upper_failure_rate_monotone_record_count": 27,
            "upper_rejection_mass_monotone_record_count": 27,
        },
        "immutable_inputs": {
            "callback_pairs_rerun": False,
            "stage_l_science_rerun": False,
            "historical_tests_rerun": False,
        },
        "mechanism_boundary": {
            "report_only_stratification": True,
            "stagee_modified": False,
            "stagek_modified": False,
            "stagel_modified": False,
            "callback_schema_modified": False,
            "integrator_thresholds_changed": False,
            "integrator_scale_search_changed": False,
            "external_scale_bank_changed": False,
            "direction_model_refit": False,
            "new_ranker_fitted": False,
        },
        "record_strata": records(post_predicate=post_predicate, **lower_kwargs),
        **false_values,
    }


def built_records(*, post_predicate="direction_retention", **lower_kwargs):
    return [
        stagen.build_record_audit(value)
        for value in records(post_predicate=post_predicate, **lower_kwargs)
    ]


def test_phase_and_constants():
    assert stagen.PHASE == "Phase3.14b-r2.5.8 Stage N"
    assert stagen.BASE_EVIDENCE_COMMIT == "2803d0131f4bc398abdc340f48cee8bdee769ae4"
    assert stagen.LOWER_MULTIPLIER == 0.25
    assert stagen.UPPER_BOUNDARY_MULTIPLIER == 0.5


def test_spec_validates():
    stagen.StageNSpec().validate()


def test_spec_rejects_bad_rate():
    with pytest.raises(stagen.StageNError):
        stagen.StageNSpec(oracle_local_acceptance_min=1.1).validate()


def test_spec_rejects_changed_support():
    with pytest.raises(stagen.StageNError):
        stagen.StageNSpec(stable_record_support_min=24).validate()


def test_stable_json_sorted_and_newline():
    assert stagen.stable_json_bytes({"b": 1, "a": 2}) == b'{"a":2,"b":1}\n'


def test_sha256_bytes_is_stable():
    assert stagen.sha256_bytes(b"x") == stagen.sha256_bytes(b"x")


def test_scale_key():
    assert stagen._scale_key(0.25) == "0.25"


def test_finite_rate_rejects_nan():
    with pytest.raises(stagen.StageNError):
        stagen._finite_rate(math.nan, "x")


def test_finite_rate_rejects_out_of_range():
    with pytest.raises(stagen.StageNError):
        stagen._finite_rate(-0.1, "x")


def test_nonnegative_int_rejects_negative():
    with pytest.raises(stagen.StageNError):
        stagen._nonnegative_int(-1, "x")


def test_validate_stage_m_report():
    result = stagen.validate_stage_m_report(stage_m_report())
    assert result["dominant_upper_multiplier"] == 0.5


def test_validate_stage_m_report_rejects_root_change():
    value = stage_m_report()
    value["root_cause"] = "wrong"
    with pytest.raises(stagen.StageNError):
        stagen.validate_stage_m_report(value)


def test_validate_stage_m_report_rejects_boundary_change():
    value = stage_m_report()
    value["frozen_probe_accessed"] = True
    with pytest.raises(stagen.StageNError):
        stagen.validate_stage_m_report(value)


def test_strata_map_requires_complete_bank():
    value = source_record("b", 10)
    value["strata"].pop()
    with pytest.raises(stagen.StageNError):
        stagen._strata_map(value)


def test_comparison_map_requires_order():
    value = cell(0.25)
    value["predicate_comparisons"] = list(reversed(value["predicate_comparisons"]))
    with pytest.raises(stagen.StageNError):
        stagen._comparison_map(value)


def test_predicate_candidates_selects_first_post_upper():
    value = cell(0.25, post_predicate="direction_retention")
    candidates, predicate, mode = stagen._predicate_candidates(stagen._comparison_map(value))
    assert predicate == "direction_retention"
    assert mode == "both"
    assert candidates[0]["predicate"] == "direction_retention"


def test_build_record_direction_retention():
    result = stagen.build_record_audit(source_record("b", 10))
    assert result["lower_multiplier_upper_gate_cleared"] is True
    assert result["post_upper_discriminator_predicate"] == "direction_retention"


def test_build_record_rejects_changed_boundary():
    value = source_record("b", 10)
    value["strata"][1]["local_discriminator_predicate"] = None
    with pytest.raises(stagen.StageNError):
        stagen.build_record_audit(value)


def test_build_record_oracle_not_admitted():
    result = stagen.build_record_audit(
        source_record("b", 10, oracle_acceptance=0.0)
    )
    assert result["locus"] == "lower_multiplier_oracle_not_admitted"


def test_build_record_oof_accepted():
    result = stagen.build_record_audit(
        source_record("b", 10, oof_acceptance=1.0)
    )
    assert result["locus"] == "lower_multiplier_oof_not_rejected"


def test_build_record_upper_not_cleared():
    result = stagen.build_record_audit(
        source_record("b", 10, upper_pass=0.5, upper_mass=0.5)
    )
    assert result["locus"] == "lower_multiplier_does_not_clear_upper_segment_gate"


def test_build_record_no_post_upper_predicate():
    result = stagen.build_record_audit(
        source_record("b", 10, post_predicate=None)
    )
    assert result["locus"] == "lower_multiplier_has_no_post_upper_discriminator"


def test_aggregate_records_counts_predicate():
    result = stagen.aggregate_records(built_records())
    assert result["post_upper_discriminator_counts"] == {"direction_retention": 27}
    assert result["upper_gate_cleared_record_count"] == 27


def test_classify_direction_retention():
    values = built_records()
    result = stagen.classify(values, stagen.aggregate_records(values))
    assert result["root_cause"] == "phase314b_r258_stagen_direction_retention_is_post_upper_boundary"
    assert result["required_next_path"] == "CALIBRATE_INTEGRATOR_COMPATIBLE_DIRECTION_BASIS_ON_OBJECTIVE_TRAIN_ONLY"


@pytest.mark.parametrize(
    "predicate,expected_path",
    [
        ("lower_segment_geometry", "AUDIT_LOWER_SEGMENT_GEOMETRY_AT_LOWER_EXTERNAL_MULTIPLIER"),
        ("coordinate_recenter", "AUDIT_LOCAL_FRAME_RECENTERING_AT_LOWER_EXTERNAL_MULTIPLIER"),
        ("coordinate_geometry", "AUDIT_LOCAL_FRAME_COMPATIBILITY_AT_LOWER_EXTERNAL_MULTIPLIER"),
        ("reconstruction_bounds", "AUDIT_SEGMENT_RECONSTRUCTION_BOUNDS_AT_LOWER_EXTERNAL_MULTIPLIER"),
        ("segment_geometry", "AUDIT_SEGMENT_VECTOR_COMPATIBILITY_AT_LOWER_EXTERNAL_MULTIPLIER"),
        ("displacement", "AUDIT_MINIMUM_DISPLACEMENT_AT_LOWER_EXTERNAL_MULTIPLIER"),
        ("topology", "CALIBRATE_TOPOLOGY_COMPATIBLE_DIRECTION_SURROGATE_ON_OBJECTIVE_TRAIN_ONLY"),
    ],
)
def test_classify_shared_post_upper_predicates(predicate, expected_path):
    values = built_records(post_predicate=predicate)
    result = stagen.classify(values, stagen.aggregate_records(values))
    assert result["required_next_path"] == expected_path


def test_classify_oracle_not_admitted():
    values = built_records(oracle_acceptance=0.0)
    result = stagen.classify(values, stagen.aggregate_records(values))
    assert result["primary_failure_locus"] == "lower_multiplier_oracle_admission"


def test_classify_acceptance_emergence():
    values = built_records(oof_acceptance=1.0)
    result = stagen.classify(values, stagen.aggregate_records(values))
    assert result["primary_failure_locus"] == "lower_multiplier_acceptance_emergence"


def test_classify_upper_not_cleared():
    values = built_records(upper_pass=0.5, upper_mass=0.5)
    result = stagen.classify(values, stagen.aggregate_records(values))
    assert result["primary_failure_locus"] == "lower_multiplier_upper_gate_clearance"


def test_classify_heterogeneous():
    values = built_records()
    for index, value in enumerate(values[:10]):
        value["post_upper_discriminator_predicate"] = "topology"
        value["locus"] = "lower_multiplier_post_upper::topology"
    aggregate = stagen.aggregate_records(values)
    result = stagen.classify(values, aggregate)
    assert result["primary_failure_locus"] == "heterogeneous_post_upper_rejection"


def test_classify_no_predicate():
    values = built_records(post_predicate=None)
    result = stagen.classify(values, stagen.aggregate_records(values))
    assert result["primary_failure_locus"] == "post_upper_scalar_summary_insufficient"


def test_build_report_population_and_boundaries():
    result = stagen.build_report(repository={"head": "x"}, stage_m_report=stage_m_report())
    assert result["execution_verdict"] == "PASS"
    assert result["scientific_status"] == "BLOCKED"
    assert len(result["record_audits"]) == 27
    assert result["mechanism_boundary"]["report_only_post_upper_audit"] is True
    assert result["frozen_probe_accessed"] is False


def test_build_report_rejects_incomplete_population():
    value = stage_m_report()
    value["record_strata"].pop()
    value["classification"]["record_count"] = 26
    with pytest.raises(stagen.StageNError):
        stagen.build_report(repository={}, stage_m_report=value)


def test_blocked_report_is_fail_closed():
    result = stagen.blocked_report(repository={"head": "x"}, error=RuntimeError("x"))
    assert result["execution_verdict"] == "BLOCKED"
    assert result["formal_training_run"] is False
    assert result["selected_configuration"] is None


def test_write_once_refuses_overwrite(tmp_path: Path):
    path = tmp_path / "x.json"
    stagen.write_once(path, b"x")
    with pytest.raises(FileExistsError):
        stagen.write_once(path, b"y")
