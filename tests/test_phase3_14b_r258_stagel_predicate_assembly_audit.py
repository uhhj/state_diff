from __future__ import annotations

from pathlib import Path

import pytest

from ccda_phase3 import phase314b_r258_stagel_predicate_assembly_audit as stagel


def event(*, active=10, accepted=10, failure_predicate=None, failure_count=None, contamination=True, scale=1.0):
    failures = {name: 0 for name in stagel.PREDICATE_ORDER}
    if failure_predicate is not None:
        failures[failure_predicate] = active if failure_count is None else failure_count
    unresolved = active
    sequential_survivors = {}
    for name in stagel.PREDICATE_ORDER:
        unresolved -= failures[name]
        sequential_survivors[name] = unresolved
    assert unresolved == accepted
    passes = {}
    topology_reach = active - sum(
        failures[name] for name in stagel.PREDICATE_ORDER[:-1]
    )
    for name in stagel.PREDICATE_ORDER:
        if name == "topology":
            passes[name] = accepted
        elif contamination:
            passes[name] = active - failures[name]
        else:
            passes[name] = sequential_survivors[name]
    return {
        "event_type": "scale_attempt",
        "attempted_scale": scale,
        "predicate_order": list(stagel.PREDICATE_ORDER),
        "active_row_count": active,
        "accepted_count": accepted,
        "topology_checked_count": topology_reach,
        "predicate_pass_counts": passes,
        "first_failed_counts": failures,
    }


def source(assembly, acceptance):
    return {
        "scale_bank_acceptance": {
            "rows_with_any_accepted_multiplier_rate": acceptance,
        },
        "assembly_overall": assembly,
        "assembly_by_external_multiplier": {"1": assembly, "2": assembly},
        "assembly_by_internal_scale": {"0.5": assembly, "1": assembly},
    }


def record(comparison):
    return {"assembly_comparison": comparison}


def test_phase_and_base_constants():
    assert stagel.PHASE == "Phase3.14b-r2.5.8 Stage L"
    assert stagel.BASE_EVIDENCE_COMMIT == "70f55e62aea557194737528f942bfde697f984b5"


def test_frozen_populations():
    spec = stagel.PredicateAssemblyAuditSpec()
    spec.validate()
    assert spec.timesteps == (10, 25, 50)
    assert spec.scale_multipliers == (0.25, 0.5, 1.0, 2.0)
    assert spec.grouped_cv_folds == 6


def test_scale_key_is_stable():
    assert stagel._scale_key(0.25) == "0.25"
    assert stagel._scale_key(2.0) == "2"


def test_empty_event_population_rejected():
    with pytest.raises(stagel.PredicateAssemblyAuditError):
        stagel.aggregate_sequential_attempts([])


def test_all_accepted_population_closes():
    value = stagel.aggregate_sequential_attempts([event()])
    assert value["accepted_row_attempt_count"] == 10
    assert value["rejected_row_attempt_count"] == 0
    assert value["sequential_population_closed"] is True


def test_first_failure_reduces_later_reach():
    value = stagel.aggregate_sequential_attempts(
        [event(active=10, accepted=0, failure_predicate="direction_retention")]
    )
    assert value["sequential_reach_counts"]["direction_retention"] == 10
    assert value["sequential_reach_counts"]["displacement"] == 0
    assert value["sequential_reach_counts"]["topology"] == 0


def test_conditional_pass_identifies_failure():
    value = stagel.aggregate_sequential_attempts(
        [event(active=10, accepted=0, failure_predicate="direction_retention")]
    )
    assert value["conditional_pass_rates"]["direction_retention"] == 0.0
    assert value["conditional_failure_rates"]["direction_retention"] == 1.0


def test_rejection_mass_normalizes_over_rejections():
    value = stagel.aggregate_sequential_attempts(
        [event(active=10, accepted=5, failure_predicate="direction_retention", failure_count=5)]
    )
    assert value["rejection_mass_rates"]["direction_retention"] == 1.0
    assert value["global_active_first_failure_rates"]["direction_retention"] == 0.5


def test_unreachable_pass_contamination_is_measured():
    value = stagel.aggregate_sequential_attempts(
        [event(active=10, accepted=0, failure_predicate="direction_retention", contamination=True)]
    )
    assert value["unreachable_pass_contamination_counts"]["displacement"] == 10
    assert value["unreachable_pass_contamination_total"] > 0


def test_no_contamination_when_pass_counts_are_sequential():
    value = stagel.aggregate_sequential_attempts(
        [event(active=10, accepted=0, failure_predicate="direction_retention", contamination=False)]
    )
    assert value["unreachable_pass_contamination_total"] == 0


def test_dominant_failure_predicate():
    value = stagel.aggregate_sequential_attempts(
        [event(active=10, accepted=0, failure_predicate="segment_geometry")]
    )
    assert value["dominant_failure_predicate"] == "segment_geometry"
    assert value["dominant_failure_mass"] == 1.0


def test_predicate_order_change_rejected():
    value = event()
    value["predicate_order"] = list(reversed(stagel.PREDICATE_ORDER))
    with pytest.raises(stagel.PredicateAssemblyAuditError):
        stagel.aggregate_sequential_attempts([value])


def test_invalid_first_failure_population_rejected():
    value = event()
    value["first_failed_counts"]["finite_state"] = 11
    with pytest.raises(stagel.PredicateAssemblyAuditError):
        stagel.aggregate_sequential_attempts([value])


def test_topology_population_uses_sequential_reach():
    value = stagel.aggregate_sequential_attempts(
        [event(active=10, accepted=6, failure_predicate="topology", failure_count=4)]
    )
    assert value["sequential_reach_counts"]["topology"] == 10
    assert value["conditional_pass_rates"]["topology"] == 0.6


def test_compare_identifies_conditional_discriminator():
    spec = stagel.PredicateAssemblyAuditSpec()
    oracle = stagel.aggregate_sequential_attempts([event()])
    oof = stagel.aggregate_sequential_attempts(
        [event(active=10, accepted=0, failure_predicate="direction_retention")]
    )
    result = stagel.compare_assembly_surfaces(
        oof=source(oof, 0.0), raw=source(oracle, 1.0), projected=source(oracle, 1.0), spec=spec
    )
    assert result["corrected_discriminator_predicate"] == "direction_retention"


def test_compare_reports_unconditional_denominator_contamination():
    spec = stagel.PredicateAssemblyAuditSpec()
    oracle = stagel.aggregate_sequential_attempts([event()])
    oof = stagel.aggregate_sequential_attempts(
        [event(active=10, accepted=0, failure_predicate="direction_retention", contamination=True)]
    )
    result = stagel.compare_assembly_surfaces(
        oof=source(oof, 0.0), raw=source(oracle, 1.0), projected=source(oracle, 1.0), spec=spec
    )
    assert result["assembly_failure_mode"] in {
        "unconditional_pass_denominator_contamination",
        "sequential_assembly_recovers_discriminator",
    }


def test_compare_requires_oracle_admission():
    spec = stagel.PredicateAssemblyAuditSpec()
    failed = stagel.aggregate_sequential_attempts(
        [event(active=10, accepted=0, failure_predicate="finite_state")]
    )
    result = stagel.compare_assembly_surfaces(
        oof=source(failed, 0.0), raw=source(failed, 0.0), projected=source(failed, 0.0), spec=spec
    )
    assert result["locus"] == "oracle_control_not_admitted"


def test_compare_requires_oof_rejection_reproduction():
    spec = stagel.PredicateAssemblyAuditSpec()
    accepted = stagel.aggregate_sequential_attempts([event()])
    result = stagel.compare_assembly_surfaces(
        oof=source(accepted, 1.0), raw=source(accepted, 1.0), projected=source(accepted, 1.0), spec=spec
    )
    assert result["locus"] == "stageh_rejection_not_reproduced"


def test_stratum_shared_discriminator():
    spec = stagel.PredicateAssemblyAuditSpec()
    oracle = stagel.aggregate_sequential_attempts([event()])
    oof = stagel.aggregate_sequential_attempts(
        [event(active=10, accepted=0, failure_predicate="segment_geometry")]
    )
    result = stagel._stratum_discriminators(
        oof={"a": oof, "b": oof}, oracle={"a": oracle, "b": oracle}, spec=spec
    )
    assert result["shared_discriminator_predicate"] == "segment_geometry"


def test_classify_shared_corrected_predicate():
    comparison = {
        "locus": "corrected_sequential_assembly_identifies_direction_retention",
        "corrected_discriminator_predicate": "direction_retention",
        "assembly_failure_mode": "unconditional_pass_denominator_contamination",
    }
    result = stagel.classify([record(comparison) for _ in range(27)])
    assert result["primary_failure_locus"] == "predicate_assembly::direction_retention"
    assert "CORRECT_STAGEK_PREDICATE_AGGREGATION" in result["required_next_path"]


def test_classify_heterogeneous_corrected_predicates():
    left = {
        "locus": "corrected_sequential_assembly_identifies_direction_retention",
        "corrected_discriminator_predicate": "direction_retention",
        "assembly_failure_mode": None,
    }
    right = {
        "locus": "corrected_sequential_assembly_identifies_topology",
        "corrected_discriminator_predicate": "topology",
        "assembly_failure_mode": None,
    }
    result = stagel.classify([record(left), record(right)])
    assert result["primary_failure_locus"] == "heterogeneous_corrected_predicates"


def test_classify_external_scale_signal():
    comparison = {
        "locus": "external_multiplier_stratification_identifies_displacement",
        "corrected_discriminator_predicate": None,
        "assembly_failure_mode": None,
        "external_multiplier_strata": {"shared_discriminator_predicate": "displacement"},
        "internal_scale_strata": {"shared_discriminator_predicate": None},
    }
    result = stagel.classify([record(comparison) for _ in range(27)])
    assert result["primary_failure_locus"] == "scale_stratified_predicate_assembly"


def test_classify_internal_scale_signal():
    comparison = {
        "locus": "internal_scale_stratification_identifies_topology",
        "corrected_discriminator_predicate": None,
        "assembly_failure_mode": None,
        "external_multiplier_strata": {"shared_discriminator_predicate": None},
        "internal_scale_strata": {"shared_discriminator_predicate": "topology"},
    }
    result = stagel.classify([record(comparison) for _ in range(27)])
    assert result["primary_failure_locus"] == "scale_stratified_predicate_assembly"


def test_classify_row_linkage_insufficiency():
    comparison = {
        "locus": "scalar_callback_cannot_link_rows_across_scale_search",
        "corrected_discriminator_predicate": None,
        "assembly_failure_mode": None,
    }
    result = stagel.classify([record(comparison) for _ in range(27)])
    assert result["primary_failure_locus"] == "cross_scale_row_linkage"
    assert result["required_next_path"] == "ADD_READ_ONLY_ROW_COHORT_HASHES_TO_EXPLICIT_CALLBACK"


def test_classify_predicate_mask_semantics():
    comparison = {
        "locus": "callback_predicate_masks_do_not_discriminate",
        "corrected_discriminator_predicate": None,
        "assembly_failure_mode": None,
    }
    result = stagel.classify([record(comparison) for _ in range(27)])
    assert result["primary_failure_locus"] == "predicate_mask_semantics"


def test_classify_rejects_empty_records():
    with pytest.raises(stagel.PredicateAssemblyAuditError):
        stagel.classify([])


def test_stage_l_source_does_not_define_integrator_callback():
    source = Path(stagel.__file__).read_text(encoding="utf-8")
    assert "def integrate_rowwise" not in source
    assert "PREDICATE_CALLBACK_SCHEMA =" not in source


def test_runner_targets_stage_l_only():
    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts/phase3_14b_r258_stagef_consolidated_e2e.py").read_text(
        encoding="utf-8"
    )
    assert "phase314b_r258_stagel_predicate_assembly_audit" in source
    assert "Stage L: audit explicit callback predicate assembly" in source
    assert "stagee_observability_surface_modified\": False" in source
