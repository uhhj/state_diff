from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ccda_phase3 import phase314b_r258_stageo_dual_oracle_lower_multiplier_admission as stageo


def event(*, scale=0.25, active=4, failures=None, accepted=0):
    values = {name: 0 for name in stageo.PREDICATE_ORDER}
    if failures:
        values.update(failures)
    return {
        "event_type": "scale_attempt",
        "attempted_scale": scale,
        "predicate_order": stageo.PREDICATE_ORDER,
        "active_row_count": active,
        "accepted_count": accepted,
        "first_failed_counts": values,
    }


def cell(source, timestep, *, admitted=False, predicate="displacement", mass=0.9):
    return {
        "source_id": source,
        "timestep": timestep,
        "oracle_admitted": admitted,
        "assembly_overall": {
            "dominant_failure_predicate": None if admitted else predicate,
            "dominant_failure_mass": 0.0 if admitted else mass,
        },
    }


def six_cells(*, admitted_sources=(), partial=None, predicates=None):
    output = []
    partial = partial or {}
    predicates = predicates or {}
    for source in stageo.ORACLE_SOURCES:
        for timestep in stageo.EXPECTED_TIMESTEPS:
            admitted = source in admitted_sources or timestep in partial.get(source, ())
            predicate = predicates.get((source, timestep), "displacement")
            output.append(cell(source, timestep, admitted=admitted, predicate=predicate))
    return output


def valid_stage_n_report():
    return {
        "execution_verdict": "PASS",
        "scientific_status": "BLOCKED",
        "root_cause": "phase314b_r258_stagen_lower_multiplier_oracle_not_admitted",
        "required_next_path": "AUDIT_ORACLE_ADMISSION_AT_LOWER_EXTERNAL_MULTIPLIER",
        "selected_configuration": None,
        "train_only_recommendation": None,
        "classification": {
            "record_count": 27,
            "oracle_admitted_record_count": 0,
            "upper_gate_cleared_record_count": 0,
        },
        "record_audits": [
            {
                "base_direction_id": f"b{index % 9}",
                "timestep": stageo.EXPECTED_TIMESTEPS[index // 9],
            }
            for index in range(27)
        ],
        **{key: False for key in stageo.FALSE_BOUNDARIES},
    }


def valid_stage_m_report():
    records = []
    for timestep in stageo.EXPECTED_TIMESTEPS:
        for index in range(9):
            records.append(
                {
                    "base_direction_id": f"b{index}",
                    "timestep": timestep,
                    "comparator_source": (
                        "raw_oracle" if timestep != 25 else "projected_oracle"
                    ),
                }
            )
    return {
        "execution_verdict": "PASS",
        "scientific_status": "BLOCKED",
        "root_cause": "phase314b_r258_stagem_upper_segment_geometry_is_single_multiplier_boundary",
        "required_next_path": "AUDIT_LOWER_MULTIPLIER_REJECTION_AFTER_UPPER_SEGMENT_GATE",
        "record_strata": records,
        **{key: False for key in stageo.FALSE_BOUNDARIES},
    }


def test_phase_and_schema_names():
    assert stageo.PHASE.endswith("Stage O")
    assert "dual_oracle" in stageo.SCHEMA


def test_fixed_population_constants():
    assert stageo.EXPECTED_TIMESTEPS == (10, 25, 50)
    assert stageo.ORACLE_SOURCES == ("raw_oracle", "projected_oracle")
    assert stageo.EXPECTED_REPLAY_PAIR_COUNT == 6


def test_lower_multiplier_is_frozen():
    assert stageo.LOWER_MULTIPLIER == 0.25
    stageo.StageOSpec().validate()


def test_spec_rejects_multiplier_change():
    with pytest.raises(stageo.StageOError):
        stageo.StageOSpec(lower_multiplier=0.5).validate()


@pytest.mark.parametrize("field", ["oracle_admission_rate_min", "dominant_failure_mass_min"])
def test_spec_rejects_rate_outside_unit_interval(field):
    kwargs = {field: 1.1}
    with pytest.raises(stageo.StageOError):
        stageo.StageOSpec(**kwargs).validate()


def test_stable_json_is_order_invariant():
    assert stageo.stable_json_bytes({"b": 2, "a": 1}) == stageo.stable_json_bytes(
        {"a": 1, "b": 2}
    )


def test_sha256_array_binds_dtype_and_shape():
    a = np.asarray([1, 2], dtype=np.float32)
    b = np.asarray([1, 2], dtype=np.float64)
    c = np.asarray([[1, 2]], dtype=np.float32)
    assert stageo.sha256_array(a) != stageo.sha256_array(b)
    assert stageo.sha256_array(a) != stageo.sha256_array(c)


def test_mapping_guard():
    with pytest.raises(stageo.StageOError):
        stageo._mapping([], "x")


def test_sequence_guard_rejects_string():
    with pytest.raises(stageo.StageOError):
        stageo._sequence("abc", "x")


@pytest.mark.parametrize("value", [-0.1, 1.1, float("inf"), float("nan")])
def test_finite_rate_rejects_invalid(value):
    with pytest.raises(stageo.StageOError):
        stageo._finite_rate(value, "x")


def test_nonnegative_int_rejects_boolean():
    with pytest.raises(stageo.StageOError):
        stageo._nonnegative_int(True, "x")


def test_scale_key_is_stable():
    assert stageo._scale_key(0.25) == "0.25"


def test_aggregate_single_predicate_failure():
    result = stageo.aggregate_sequential_attempts(
        [event(failures={"displacement": 4})]
    )
    assert result["first_failed_counts"]["displacement"] == 4
    assert result["dominant_failure_predicate"] == "displacement"
    assert result["dominant_failure_mass"] == 1.0
    assert result["accepted_row_attempt_count"] == 0


def test_aggregate_acceptance_closes():
    result = stageo.aggregate_sequential_attempts([event(active=4, accepted=4)])
    assert result["accepted_row_attempt_count"] == 4
    assert result["rejected_row_attempt_count"] == 0
    assert result["dominant_failure_predicate"] is None


def test_aggregate_rejects_empty_population():
    with pytest.raises(stageo.StageOError):
        stageo.aggregate_sequential_attempts([])


def test_aggregate_rejects_changed_order():
    value = event()
    value["predicate_order"] = tuple(reversed(stageo.PREDICATE_ORDER))
    with pytest.raises(stageo.StageOError):
        stageo.aggregate_sequential_attempts([value])


def test_aggregate_rejects_nonclosing_population():
    with pytest.raises(stageo.StageOError):
        stageo.aggregate_sequential_attempts(
            [event(active=4, failures={"displacement": 1}, accepted=0)]
        )


def test_group_internal_scale_preserves_order():
    grouped = stageo.group_attempts_by_internal_scale(
        [event(scale=0.25), event(scale=0.5), event(scale=1.0)]
    )
    assert list(grouped) == ["0.25", "0.5", "1"]


def test_group_internal_scale_rejects_duplicate_or_decreasing_scale():
    with pytest.raises(stageo.StageOError):
        stageo.group_attempts_by_internal_scale(
            [event(scale=0.5), event(scale=0.25)]
        )


def test_scalar_stats_are_finite():
    result = stageo.scalar_stats(np.asarray([1.0, 2.0, 3.0]))
    assert result["count"] == 3
    assert result["mean"] == 2.0
    assert result["median"] == 2.0


def test_scalar_stats_reject_empty():
    with pytest.raises(stageo.StageOError):
        stageo.scalar_stats(np.asarray([]))


def test_selected_scale_histogram_closes():
    result = stageo.selected_scale_histogram(np.asarray([0.0, 0.25, 0.25, 1.0]))
    assert result == {"0": 1, "0.25": 2, "1": 1}


def test_selected_scale_histogram_rejects_negative():
    with pytest.raises(stageo.StageOError):
        stageo.selected_scale_histogram(np.asarray([-1.0]))


def test_stage_n_validator_accepts_frozen_result():
    result = stageo.validate_stage_n_report(valid_stage_n_report())
    assert result["record_count"] == 27


def test_stage_n_validator_rejects_changed_root():
    report = valid_stage_n_report()
    report["root_cause"] = "changed"
    with pytest.raises(stageo.StageOError):
        stageo.validate_stage_n_report(report)


def test_stage_m_validator_accepts_frozen_result():
    stageo.validate_stage_m_report(valid_stage_m_report())


def test_unique_stage_n_cells_detects_replication():
    result = stageo.unique_stage_n_comparator_cells(
        valid_stage_n_report(), valid_stage_m_report()
    )
    assert result["stage_n_record_count"] == 27
    assert result["unique_comparator_oracle_cell_count"] == 3
    assert result["backbone_replication_factor"] == 9


def test_unique_stage_n_cells_rejects_backbone_population_change():
    report = valid_stage_m_report()
    report["record_strata"] = report["record_strata"][:-1]
    with pytest.raises(stageo.StageOError):
        stageo.unique_stage_n_comparator_cells(valid_stage_n_report(), report)


def test_classify_both_sources_admitted():
    result = stageo.classify_oracle_admission(
        six_cells(admitted_sources=stageo.ORACLE_SOURCES)
    )
    assert result["root_cause"].endswith("both_lower_multiplier_oracle_sources_admitted")


def test_classify_raw_source_specific_admission():
    result = stageo.classify_oracle_admission(
        six_cells(admitted_sources=("raw_oracle",))
    )
    assert result["primary_failure_locus"] == "admitted_oracle_source::raw_oracle"


def test_classify_projected_source_specific_admission():
    result = stageo.classify_oracle_admission(
        six_cells(admitted_sources=("projected_oracle",))
    )
    assert result["primary_failure_locus"] == "admitted_oracle_source::projected_oracle"


def test_classify_partial_admission():
    result = stageo.classify_oracle_admission(
        six_cells(partial={"raw_oracle": (10,)})
    )
    assert result["primary_failure_locus"] == "heterogeneous_oracle_admission"


@pytest.mark.parametrize("predicate", stageo.PREDICATE_ORDER)
def test_classify_stable_predicate_failure(predicate):
    result = stageo.classify_oracle_admission(six_cells(predicates={
        (source, timestep): predicate
        for source in stageo.ORACLE_SOURCES
        for timestep in stageo.EXPECTED_TIMESTEPS
    }))
    assert result["dominant_failure_predicate"] == predicate
    assert result["dominant_failure_support_count"] == 6
    assert result["primary_failure_locus"] == f"oracle_rejection::{predicate}"


def test_classify_heterogeneous_failure():
    predicates = {}
    values = ("displacement", "direction_retention")
    index = 0
    for source in stageo.ORACLE_SOURCES:
        for timestep in stageo.EXPECTED_TIMESTEPS:
            predicates[(source, timestep)] = values[index % 2]
            index += 1
    result = stageo.classify_oracle_admission(six_cells(predicates=predicates))
    assert result["primary_failure_locus"] == "heterogeneous_oracle_rejection"


def test_classify_insufficient_scalar_surface():
    cells = six_cells()
    for value in cells:
        value["assembly_overall"] = {
            "dominant_failure_predicate": None,
            "dominant_failure_mass": 0.0,
        }
    result = stageo.classify_oracle_admission(cells)
    assert result["primary_failure_locus"] == "oracle_scalar_surface_insufficient"


def test_classify_rejects_wrong_cell_count():
    with pytest.raises(stageo.StageOError):
        stageo.classify_oracle_admission(six_cells()[:-1])


def test_blocked_report_is_fail_closed():
    report = stageo.blocked_report(repository={"head": "x"}, error=RuntimeError("boom"))
    assert report["execution_verdict"] == "BLOCKED"
    assert report["scientific_status"] == "BLOCKED"
    assert report["selected_configuration"] is None
    assert report["train_only_recommendation"] is None
    assert report["formal_training_run"] is False


def test_false_boundary_population_contains_holdout_and_probe():
    assert "selection_holdout_evaluated" in stageo.FALSE_BOUNDARIES
    assert "frozen_probe_accessed" in stageo.FALSE_BOUNDARIES


def test_frozen_source_population_contains_stagee_k_l_m_n():
    names = "\n".join(stageo.FROZEN_SOURCE_SHA256)
    for stage in ("stagee", "stagek", "stagel", "stagem", "stagen"):
        assert stage in names


def test_implementation_is_add_only_three_paths():
    assert len(stageo.IMPLEMENTATION_PATHS) == 3
    assert all(status == "A" for status, _ in stageo.IMPLEMENTATION_PATHS)


def test_success_and_blocked_reports_are_distinct():
    assert stageo.SUCCESS_REPORT != stageo.BLOCKED_REPORT


def test_expected_environment_has_all_thread_controls():
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        assert stageo.EXPECTED_ENV[key] == "1"


def test_source_does_not_authorize_oof_fit_or_holdout():
    source = Path(stageo.__file__).read_text(encoding="utf-8")
    assert '"oof_surrogate_refit": False' in source
    assert '"selection_holdout_evaluated"' in source
    assert "fit_oof_base_direction(" not in source


def test_report_json_serialization_accepts_classification():
    result = stageo.classify_oracle_admission(six_cells())
    json.loads(stageo.stable_json_bytes(result).decode("utf-8"))


def test_classify_internal_scale_heterogeneity_precedes_global_predicate():
    cells = six_cells()
    cells[0]["internal_scale_failure_heterogeneous"] = True
    cells[0]["stable_internal_scale_failure_predicate"] = None
    for value in cells[1:]:
        value["internal_scale_failure_heterogeneous"] = False
        value["stable_internal_scale_failure_predicate"] = "displacement"
    result = stageo.classify_oracle_admission(cells)
    assert result["primary_failure_locus"] == "internal_scale_dependent_oracle_rejection"


def test_audit_oracle_direction_records_internal_scale_surface():
    control = np.zeros((4, 2), dtype=np.float32)
    direction = np.ones((4, 2), dtype=np.float64)
    attempt = event(scale=0.25, active=4, failures={"displacement": 4}, accepted=0)
    stage_l_assembly = {"exact": "assembly"}

    class FakeStageK:
        @staticmethod
        def callback_integrate_rowwise(**kwargs):
            integration = {
                "selected_scale": np.zeros(4, dtype=np.float64),
                "candidate": control.copy(),
            }
            capture = {
                "events": (attempt,),
                "events_sha256": "events-sha",
                "returned_result_bit_exact": True,
            }
            return integration, capture

        @staticmethod
        def _row_norm(value):
            return np.linalg.norm(np.asarray(value), axis=1)

    class FakeStageL:
        @staticmethod
        def aggregate_sequential_attempts(events):
            return stage_l_assembly

    selected = np.zeros(4, dtype=np.float64)
    candidate = control.copy()
    proposed = direction * stageo.LOWER_MULTIPLIER
    persisted = {
        "multiplier_records": [
            {
                "scale_multiplier": stageo.LOWER_MULTIPLIER,
                "selected_scale_positive_rate": 0.0,
                "selected_scale_sha256": stageo.sha256_array(selected),
                "candidate_motion_positive_rate": 0.0,
                "candidate_sha256": stageo.sha256_array(candidate),
                "callback_capture_sha256": "events-sha",
                "callback_result_bit_exact": True,
                "assembly": stage_l_assembly,
            }
        ]
    }
    result = stageo.audit_oracle_direction(
        source_id="raw_oracle",
        timestep=10,
        control=control,
        base_direction=direction,
        context={},
        persisted=persisted,
        runtime={
            "stagek": FakeStageK,
            "stagel": FakeStageL,
            "integrator_spec": object(),
            "callback_spec": object(),
        },
        spec=stageo.StageOSpec(),
    )
    assert result["oracle_admitted"] is False
    assert result["dominant_failure_predicate_by_internal_scale"] == {
        "0.25": "displacement"
    }
    assert result["stable_internal_scale_failure_predicate"] == "displacement"
    assert result["stage_l_identity_check"]["matched_stage_l_025_multiplier_record"] is True
    assert result["proposal_row_norm_stats"]["mean"] == pytest.approx(
        np.linalg.norm(proposed[0])
    )
