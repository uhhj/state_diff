from __future__ import annotations

import copy
import math
from pathlib import Path

import pytest

from ccda_phase3 import phase314b_r258_stagem_external_multiplier_predicate_stratification as stagem


def assembly(*, upper_pass: float = 1.0, upper_mass: float = 0.0):
    conditional_pass = {name: 1.0 for name in stagem.PREDICATE_ORDER}
    conditional_failure = {name: 0.0 for name in stagem.PREDICATE_ORDER}
    rejection_mass = {name: 0.0 for name in stagem.PREDICATE_ORDER}
    reached = {name: 10 for name in stagem.PREDICATE_ORDER}
    first_failed = {name: 0 for name in stagem.PREDICATE_ORDER}
    conditional_pass["upper_segment_geometry"] = upper_pass
    conditional_failure["upper_segment_geometry"] = 1.0 - upper_pass
    rejection_mass["upper_segment_geometry"] = upper_mass
    first_failed["upper_segment_geometry"] = int(round(10 * upper_mass))
    return {
        "active_row_attempt_count": 10,
        "accepted_row_attempt_count": 0 if upper_mass else 10,
        "sequential_reach_counts": reached,
        "first_failed_counts": first_failed,
        "conditional_pass_rates": conditional_pass,
        "conditional_failure_rates": conditional_failure,
        "rejection_mass_rates": rejection_mass,
    }


def source_audit(*, acceptance_by_multiplier=None, upper_at=None):
    acceptance_by_multiplier = acceptance_by_multiplier or {
        0.25: 1.0,
        0.5: 1.0,
        1.0: 1.0,
        2.0: 1.0,
    }
    upper_at = set() if upper_at is None else set(upper_at)
    records = []
    assemblies = {}
    for multiplier in stagem.EXPECTED_EXTERNAL_MULTIPLIERS:
        key = stagem._scale_key(multiplier)
        records.append(
            {
                "scale_multiplier": multiplier,
                "selected_scale_positive_rate": acceptance_by_multiplier[multiplier],
            }
        )
        assemblies[key] = (
            assembly(upper_pass=0.0, upper_mass=1.0)
            if multiplier in upper_at
            else assembly()
        )
    return {
        "multiplier_records": records,
        "assembly_by_external_multiplier": assemblies,
    }


def stored_external(*, upper_at=None):
    upper_at = set() if upper_at is None else set(upper_at)
    return {
        "records": [
            {
                "stratum": stagem._scale_key(multiplier),
                "discriminator_predicate": (
                    "upper_segment_geometry" if multiplier in upper_at else None
                ),
            }
            for multiplier in stagem.EXPECTED_EXTERNAL_MULTIPLIERS
        ]
    }


def record(base="b0", timestep=10, *, upper_at=(2.0,), comparator="projected_oracle"):
    return {
        "base_direction_id": base,
        "timestep": timestep,
        "feature_mode": "paper_x",
        "oof_predicate_assembly": source_audit(
            acceptance_by_multiplier={0.25: 0.0, 0.5: 0.0, 1.0: 0.0, 2.0: 0.0},
            upper_at=upper_at,
        ),
        "assembly_comparison": {
            "comparator_source": comparator,
            "external_multiplier_strata": stored_external(upper_at=upper_at),
        },
    }


def controls(*, oracle_acceptance=None):
    oracle_acceptance = oracle_acceptance or {
        0.25: 1.0,
        0.5: 1.0,
        1.0: 1.0,
        2.0: 1.0,
    }
    result = {}
    for timestep in stagem.EXPECTED_TIMESTEPS:
        oracle = source_audit(acceptance_by_multiplier=oracle_acceptance)
        result[str(timestep)] = {
            "raw_oracle": copy.deepcopy(oracle),
            "projected_oracle": copy.deepcopy(oracle),
        }
    return result


def complete_records(*, upper_at=(2.0,)):
    output = []
    for base_index in range(9):
        for timestep in stagem.EXPECTED_TIMESTEPS:
            output.append(record(f"b{base_index}", timestep, upper_at=upper_at))
    return output


def scientific_result(records=None, control_map=None):
    false_values = {key: False for key in stagem.FALSE_BOUNDARIES}
    return {
        "selected_configuration": None,
        "train_only_recommendation": None,
        **false_values,
        "predicate_assembly_audit": {
            "callback_off_on_pair_count": 132,
            "all_callback_results_bit_exact": True,
            "callback_events_persisted": False,
            "control_assembly_by_timestep": control_map or controls(),
            "oof_records": records or complete_records(),
        },
    }


def resume1_report(records=None, control_map=None):
    return {
        "execution_verdict": "PASS",
        "scientific_status": "BLOCKED",
        "recovery_contract": {
            "underlying_stage_l_single_run_result_sha256": stagem.EXPECTED_STAGEL_SINGLE_RUN_SHA256,
            "underlying_callback_pair_count": 132,
            "stage_l_science_modified": False,
            "existing_test_gates_rerun": False,
        },
        "stage_l_result": {
            "execution_verdict": "PASS",
            "scientific_result": scientific_result(records, control_map),
        },
    }


def resume2_report():
    return {
        "execution_verdict": "PASS",
        "scientific_status": "BLOCKED",
        "root_cause": "phase314b_r258_stagel_rejection_signal_is_scale_stratified_and_heterogeneous",
        "required_next_path": "STRATIFY_OOF_REJECTION_BY_CALLBACK_SCALE_AND_PREDICATE",
        "corrected_classification": {
            "record_count": 27,
            "external_heterogeneous_record_count": 27,
            "internal_heterogeneous_record_count": 0,
            "external_stratum_discriminator_counts": {"upper_segment_geometry": 27},
        },
        "correction_contract": {
            "stage_l_science_rerun": False,
            "callback_pairs_rerun": False,
            "existing_tests_rerun": False,
            "base_report_modified": False,
        },
    }


def built_records(*, upper_at=(2.0,), oracle_acceptance=None):
    control_map = controls(oracle_acceptance=oracle_acceptance)
    return [
        stagem.build_record_strata(
            record=value,
            controls=control_map,
            spec=stagem.StageMSpec(),
        )
        for value in complete_records(upper_at=upper_at)
    ]


def test_spec_validates():
    stagem.StageMSpec().validate()


def test_spec_rejects_invalid_rate():
    with pytest.raises(stagem.StageMError):
        stagem.StageMSpec(oracle_local_acceptance_min=1.1).validate()


def test_spec_rejects_changed_support():
    with pytest.raises(stagem.StageMError):
        stagem.StageMSpec(stable_record_support_min=24).validate()


def test_stable_json_is_sorted_and_newline_terminated():
    assert stagem.stable_json_bytes({"b": 1, "a": 2}) == b'{"a":2,"b":1}\n'


def test_scale_key_matches_stage_l_format():
    assert stagem._scale_key(0.25) == "0.25"


def test_finite_rate_rejects_nan():
    with pytest.raises(stagem.StageMError):
        stagem._finite_rate(math.nan, "x")


def test_finite_rate_rejects_out_of_range():
    with pytest.raises(stagem.StageMError):
        stagem._finite_rate(-0.1, "x")


def test_nonnegative_int_rejects_negative():
    with pytest.raises(stagem.StageMError):
        stagem._nonnegative_int(-1, "x")


def test_predicate_discriminator_uses_conditional_gate():
    predicate, _ = stagem._predicate_discriminator(
        oof=assembly(upper_pass=0.0),
        oracle=assembly(upper_pass=1.0),
        spec=stagem.StageMSpec(),
    )
    assert predicate == "upper_segment_geometry"


def test_predicate_discriminator_uses_rejection_mass_gate():
    predicate, _ = stagem._predicate_discriminator(
        oof=assembly(upper_pass=0.5, upper_mass=1.0),
        oracle=assembly(upper_pass=0.5, upper_mass=0.0),
        spec=stagem.StageMSpec(),
    )
    assert predicate == "upper_segment_geometry"


def test_predicate_discriminator_none_when_no_signal():
    predicate, comparisons = stagem._predicate_discriminator(
        oof=assembly(), oracle=assembly(), spec=stagem.StageMSpec()
    )
    assert predicate is None
    assert len(comparisons) == len(stagem.PREDICATE_ORDER)


def test_predicate_discriminator_respects_order():
    oof = assembly(upper_pass=0.0, upper_mass=1.0)
    oracle = assembly()
    oof["conditional_pass_rates"]["finite_state"] = 0.0
    oof["rejection_mass_rates"]["finite_state"] = 1.0
    predicate, _ = stagem._predicate_discriminator(
        oof=oof, oracle=oracle, spec=stagem.StageMSpec()
    )
    assert predicate == "finite_state"


def test_assembly_metrics_extracts_upper_values():
    result = stagem._assembly_metrics(
        assembly(upper_pass=0.0, upper_mass=1.0), "upper_segment_geometry"
    )
    assert result["first_failed_count"] == 10
    assert result["conditional_failure_rate"] == 1.0


def test_multiplier_record_map_accepts_exact_bank():
    result = stagem._multiplier_record_map(source_audit(), "x")
    assert tuple(sorted(result, key=float)) == ("0.25", "0.5", "1", "2")


def test_multiplier_record_map_rejects_duplicate():
    value = source_audit()
    value["multiplier_records"].append(copy.deepcopy(value["multiplier_records"][0]))
    with pytest.raises(stagem.StageMError):
        stagem._multiplier_record_map(value, "x")


def test_multiplier_record_map_rejects_missing_multiplier():
    value = source_audit()
    value["multiplier_records"] = value["multiplier_records"][:-1]
    with pytest.raises(stagem.StageMError):
        stagem._multiplier_record_map(value, "x")


def test_stored_stratum_map_accepts_exact_bank():
    comparison = {"external_multiplier_strata": stored_external(upper_at=(2.0,))}
    result = stagem._stored_stratum_map(comparison, "x")
    assert result["2"]["discriminator_predicate"] == "upper_segment_geometry"


def test_build_record_strata_identifies_local_upper_predicate():
    result = stagem.build_record_strata(
        record=record(), controls=controls(), spec=stagem.StageMSpec()
    )
    cell = next(item for item in result["strata"] if item["external_multiplier"] == 2.0)
    assert cell["local_discriminator_predicate"] == "upper_segment_geometry"
    assert cell["locally_eligible"] is True


def test_build_record_strata_local_oracle_gate_invalidates_predicate():
    result = stagem.build_record_strata(
        record=record(),
        controls=controls(oracle_acceptance={0.25: 1.0, 0.5: 1.0, 1.0: 1.0, 2.0: 0.5}),
        spec=stagem.StageMSpec(),
    )
    cell = next(item for item in result["strata"] if item["external_multiplier"] == 2.0)
    assert cell["local_discriminator_predicate"] is None
    assert result["prior_discriminator_not_locally_eligible_count"] == 1


def test_build_record_strata_detects_stored_mismatch():
    value = record()
    value["assembly_comparison"]["external_multiplier_strata"] = stored_external()
    result = stagem.build_record_strata(
        record=value, controls=controls(), spec=stagem.StageMSpec()
    )
    assert result["stored_stratum_discriminator_mismatch_count"] == 1


def test_build_record_strata_reports_monotonic_upper_failure():
    result = stagem.build_record_strata(
        record=record(), controls=controls(), spec=stagem.StageMSpec()
    )
    assert result["upper_failure_rate_nondecreasing_with_multiplier"] is True
    assert result["upper_rejection_mass_nondecreasing_with_multiplier"] is True


def test_aggregate_by_multiplier_counts_upper_support():
    records = built_records()
    result = stagem.aggregate_by_multiplier(records, stagem.StageMSpec())
    assert result["2"]["local_upper_segment_geometry_count"] == 27
    assert result["1"]["local_upper_segment_geometry_count"] == 0


def test_classification_detects_persisted_mismatch():
    records = built_records()
    records[0] = dict(records[0])
    records[0]["stored_stratum_discriminator_mismatch_count"] = 1
    aggregate = stagem.aggregate_by_multiplier(records, stagem.StageMSpec())
    result = stagem.classify_stratification(
        records=records, by_multiplier=aggregate, spec=stagem.StageMSpec()
    )
    assert result["primary_failure_locus"] == "stratum_summary_identity"


def test_classification_detects_local_oracle_admission_failure():
    records = built_records(
        oracle_acceptance={0.25: 1.0, 0.5: 1.0, 1.0: 1.0, 2.0: 0.5}
    )
    aggregate = stagem.aggregate_by_multiplier(records, stagem.StageMSpec())
    result = stagem.classify_stratification(
        records=records, by_multiplier=aggregate, spec=stagem.StageMSpec()
    )
    assert result["primary_failure_locus"] == "stratum_local_oracle_admission"


def test_classification_detects_oof_acceptance_emergence():
    records = built_records()
    for record_value in records:
        cell = next(item for item in record_value["strata"] if item["external_multiplier"] == 0.25)
        cell["oof_acceptance_rate"] = 0.5
    aggregate = stagem.aggregate_by_multiplier(records, stagem.StageMSpec())
    result = stagem.classify_stratification(
        records=records, by_multiplier=aggregate, spec=stagem.StageMSpec()
    )
    assert result["primary_failure_locus"] == "oof_acceptance_emergence"


def test_classification_detects_single_multiplier_boundary():
    records = built_records(upper_at=(2.0,))
    aggregate = stagem.aggregate_by_multiplier(records, stagem.StageMSpec())
    result = stagem.classify_stratification(
        records=records, by_multiplier=aggregate, spec=stagem.StageMSpec()
    )
    assert result["primary_failure_locus"] == "single_external_multiplier_upper_segment_boundary"
    assert result["dominant_upper_multiplier"] == 2.0


def test_classification_detects_monotone_multi_multiplier_boundary():
    records = built_records(upper_at=(1.0, 2.0))
    aggregate = stagem.aggregate_by_multiplier(records, stagem.StageMSpec())
    result = stagem.classify_stratification(
        records=records, by_multiplier=aggregate, spec=stagem.StageMSpec()
    )
    assert result["primary_failure_locus"] == "monotone_external_magnitude_upper_segment_boundary"


def test_classification_detects_backbone_or_timestep_dependence():
    records = built_records(upper_at=(2.0,))
    # Preserve total support >=25 but break one timestep's 8/9 coverage.
    changed = 0
    for record_value in records:
        if record_value["timestep"] == 10 and changed < 2:
            cell = next(item for item in record_value["strata"] if item["external_multiplier"] == 2.0)
            cell["local_discriminator_predicate"] = None
            changed += 1
    aggregate = stagem.aggregate_by_multiplier(records, stagem.StageMSpec())
    result = stagem.classify_stratification(
        records=records, by_multiplier=aggregate, spec=stagem.StageMSpec()
    )
    assert result["primary_failure_locus"] == "backbone_or_timestep_dependent_upper_segment_boundary"


def test_classification_detects_no_stable_predicate():
    records = built_records(upper_at=())
    aggregate = stagem.aggregate_by_multiplier(records, stagem.StageMSpec())
    result = stagem.classify_stratification(
        records=records, by_multiplier=aggregate, spec=stagem.StageMSpec()
    )
    assert result["primary_failure_locus"] == "external_multiplier_stratification_insufficient"


def test_validate_resume2_report_accepts_expected_contract():
    corrected = stagem.validate_resume2_report(resume2_report())
    assert corrected["external_heterogeneous_record_count"] == 27


def test_validate_resume2_report_rejects_changed_discriminator():
    value = resume2_report()
    value["corrected_classification"]["external_stratum_discriminator_counts"] = {}
    with pytest.raises(stagem.StageMError):
        stagem.validate_resume2_report(value)


def test_build_report_is_report_only_and_complete():
    result = stagem.build_report(
        repository={"head": "x"},
        resume1_report=resume1_report(),
        resume2_report=resume2_report(),
    )
    assert result["execution_verdict"] == "PASS"
    assert result["scientific_status"] == "BLOCKED"
    assert len(result["record_strata"]) == 27
    assert result["classification"]["stratum_count"] == 108
    assert result["mechanism_boundary"]["report_only_stratification"] is True
    assert result["immutable_inputs"]["callback_pairs_rerun"] is False
    assert result["source_support_limitations"]["no_missing_value_was_inferred"] is True
    assert result["selected_configuration"] is None
    assert result["train_only_recommendation"] is None
