from __future__ import annotations

import inspect
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

from ccda_phase3 import phase314b_r256_stageb_cable_diffusion as stageb
from ccda_phase3 import phase314b_r256_staged_segment_recalibration as staged
from ccda_phase3 import phase314b_r258_stagee_constrained_integrator as stagee
from ccda_phase3 import phase314b_r258_stagej_readonly_integrator_telemetry as stagej
from ccda_phase3 import phase314b_r258_stagek_explicit_integrator_predicate_callback as stagek


def make_cable(rows: int = 4, length: float = 0.10) -> np.ndarray:
    x = (np.arange(stageb.BEADS, dtype=np.float32) - 11.5) * float(length)
    points = np.zeros((rows, stageb.FUTURE_STEPS, stageb.BEADS, 2), dtype=np.float32)
    points[..., 0] = x[None, None]
    for row in range(rows):
        points[row, :, :, 1] = 0.01 * row
    return points.reshape(rows, stageb.FUTURE_STEPS, stageb.CABLE_DIM)


def make_reference(length: float = 0.10, scale: float = 0.25):
    shape = (stageb.FUTURE_STEPS, stageb.BEADS - 1)
    center = np.full(shape, np.log(length), dtype=np.float64)
    scale_array = np.full(shape, scale, dtype=np.float64)
    return staged.SegmentReference(
        center_log=center,
        scale_log=scale_array,
        mad_scale=scale_array.copy(),
        iqr_scale=scale_array.copy(),
        scale_floor=1.0e-6,
    )


def make_context() -> dict:
    shape = (stageb.FUTURE_STEPS, stageb.BEADS - 1)
    historical = stageb.GeometryContract(
        segment_lower=np.full(shape, 0.05, dtype=np.float32),
        segment_upper=np.full(shape, 0.20, dtype=np.float32),
        coordinate_abs_max=5.0,
        target_intersection_max=0,
    )
    return {
        "historical_geometry": historical,
        "stage_d_contract": SimpleNamespace(
            reference=make_reference(),
            lower_threshold=4.0,
        ),
        "upper_gate": SimpleNamespace(upper_threshold=4.0),
    }


def definition() -> stagee.IntegratorDefinition:
    return next(
        item
        for item in stagee.INTEGRATOR_DEFINITIONS
        if item.candidate_id == "centered_a10_z4_m1_r25"
    )


def integrate(callback=None):
    control = make_cable()
    direction = np.full_like(control, 0.001, dtype=np.float64)
    return stagee.integrate_rowwise(
        control=control,
        direction=direction,
        definition=definition(),
        context=make_context(),
        spec=stagee.ConstrainedIntegratorSpec(),
        predicate_callback=callback,
    )


def captured():
    events = []
    result = integrate(events.append)
    return result, events


def test_phase_and_base_constants():
    assert stagek.PHASE == "Phase3.14b-r2.5.8 Stage K"
    assert stagek.BASE_EVIDENCE_COMMIT == "90375c4a21e4a798539bc0fabea79787f2011966"


def test_spec_validates_frozen_populations():
    spec = stagek.ExplicitPredicateCallbackSpec()
    spec.validate()
    assert spec.scale_multipliers == (0.25, 0.5, 1.0, 2.0)


def test_callback_signature_is_optional_and_default_disabled():
    parameter = inspect.signature(stagee.integrate_rowwise).parameters["predicate_callback"]
    assert parameter.default is None
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


def test_predicate_order_is_complete_and_stable():
    assert stagek.PREDICATE_ORDER == stagee.PREDICATE_ORDER
    assert tuple(stagek.PREDICATE_OUTCOMES) == stagek.PREDICATE_ORDER


def test_callback_none_matches_legacy_call():
    control = make_cable()
    direction = np.full_like(control, 0.001, dtype=np.float64)
    legacy = stagee.integrate_rowwise(
        control=control,
        direction=direction,
        definition=definition(),
        context=make_context(),
        spec=stagee.ConstrainedIntegratorSpec(),
    )
    explicit_none = integrate(None)
    stagej._exact_identity(legacy, explicit_none, "$.none")


def test_noop_callback_is_byte_exact():
    baseline = integrate(None)
    observed = integrate(lambda _event: None)
    stagej._exact_identity(baseline, observed, "$.noop")


def test_recording_callback_is_byte_exact():
    baseline = integrate(None)
    observed, _events = captured()
    stagej._exact_identity(baseline, observed, "$.recording")


def test_callback_emits_attempts_then_one_final_summary():
    _result, events = captured()
    assert [event["event_type"] for event in events] == [
        "scale_attempt", "scale_attempt", "scale_attempt", "scale_attempt",
        "final_summary",
    ]


def test_attempted_scales_follow_real_descending_search_order():
    _result, events = captured()
    attempts = [event for event in events if event["event_type"] == "scale_attempt"]
    assert [event["attempted_scale"] for event in attempts] == [1.0, 0.75, 0.5, 0.25]


def test_callback_top_level_mapping_is_immutable():
    _result, events = captured()
    assert isinstance(events[0], MappingProxyType)
    with pytest.raises(TypeError):
        events[0]["x"] = 1


def test_callback_nested_mappings_are_immutable():
    _result, events = captured()
    assert isinstance(events[0]["predicate_pass_rates"], MappingProxyType)
    with pytest.raises(TypeError):
        events[0]["predicate_pass_rates"]["finite_state"] = 0.0


def test_callback_payload_is_scalar_only():
    _result, events = captured()
    for event in events:
        stagek._assert_scalar_tree(event)


def test_callback_payload_does_not_contain_target_or_arrays():
    _result, events = captured()
    text = repr(events)
    assert "target" in text  # only the explicit target_used=false contract
    assert "target_used': False" in text
    assert "array(" not in text


def test_attempt_population_closes_exactly():
    _result, events = captured()
    for event in events:
        if event["event_type"] != "scale_attempt":
            continue
        assert sum(event["first_failed_counts"].values()) + event["accepted_count"] == event["active_row_count"]
        assert event["all_active_rows_accounted_for"] is True


def test_final_event_matches_returned_selection():
    result, events = captured()
    final = events[-1]
    assert final["selected_count"] == int(np.count_nonzero(result["selected_scale"]))
    assert final["fallback_count"] == int(np.count_nonzero(result["selected_scale"] == 0.0))


def test_callback_exception_propagates_fail_closed():
    def fail(_event):
        raise RuntimeError("observer failed")
    with pytest.raises(RuntimeError, match="observer failed"):
        integrate(fail)


def test_collector_rejects_mutable_event():
    collector = stagek.PredicateEventCollector(stagek.ExplicitPredicateCallbackSpec())
    with pytest.raises(stagek.ExplicitPredicateCallbackError, match="mutable"):
        collector({"schema": stagee.PREDICATE_CALLBACK_SCHEMA})


def test_collector_accepts_real_events_and_finalizes():
    collector = stagek.PredicateEventCollector(stagek.ExplicitPredicateCallbackSpec())
    result = integrate(collector)
    capture = collector.finalize()
    assert result["fallback_rate"] == 0.0
    assert capture["attempt_count"] == 4
    assert capture["attempt_count"] == 4


def test_aggregate_attempt_telemetry_has_all_predicates():
    collector = stagek.PredicateEventCollector(stagek.ExplicitPredicateCallbackSpec())
    integrate(collector)
    capture = collector.finalize()
    aggregate = stagek.aggregate_attempt_telemetry([capture])
    assert tuple(aggregate["predicate_pass_rates"]) == stagek.PREDICATE_ORDER
    assert aggregate["attempt_count"] == 4


def synthetic_audit(*, acceptance: float, predicate: str, pass_rate: float, first_rate: float):
    return {
        "scale_bank_acceptance": {"rows_with_any_accepted_multiplier_rate": acceptance},
        "predicate_aggregate": {
            "predicate_pass_rates": {
                name: (pass_rate if name == predicate else 1.0)
                for name in stagek.PREDICATE_ORDER
            },
            "first_failed_rates": {
                name: (first_rate if name == predicate else 0.0)
                for name in stagek.PREDICATE_ORDER
            },
        },
    }


def test_compare_identifies_direction_retention():
    spec = stagek.ExplicitPredicateCallbackSpec()
    oof = synthetic_audit(acceptance=0.0, predicate="direction_retention", pass_rate=0.0, first_rate=1.0)
    oracle = synthetic_audit(acceptance=1.0, predicate="direction_retention", pass_rate=1.0, first_rate=0.0)
    result = stagek.compare_predicate_surfaces(oof=oof, raw=oracle, projected=oracle, spec=spec)
    assert result["discriminator_predicate"] == "direction_retention"


def test_compare_requires_oracle_admission():
    spec = stagek.ExplicitPredicateCallbackSpec()
    value = synthetic_audit(acceptance=0.0, predicate="direction_retention", pass_rate=0.0, first_rate=1.0)
    result = stagek.compare_predicate_surfaces(oof=value, raw=value, projected=value, spec=spec)
    assert result["locus"] == "oracle_control_not_admitted"


def test_compare_rejects_nonreproduced_oof_failure():
    spec = stagek.ExplicitPredicateCallbackSpec()
    oracle = synthetic_audit(acceptance=1.0, predicate="direction_retention", pass_rate=1.0, first_rate=0.0)
    result = stagek.compare_predicate_surfaces(oof=oracle, raw=oracle, projected=oracle, spec=spec)
    assert result["locus"] == "stageh_rejection_not_reproduced"


def record(predicate):
    return {"predicate_comparison": {"locus": "oof_first_diverges_at_{}".format(predicate), "discriminator_predicate": predicate}}


def test_classify_shared_direction_retention():
    result = stagek.classify([record("direction_retention") for _ in range(27)])
    assert result["primary_failure_locus"] == "direction_retention"
    assert result["required_next_path"] == "CALIBRATE_INTEGRATOR_COMPATIBLE_DIRECTION_BASIS_ON_OBJECTIVE_TRAIN_ONLY"


def test_classify_shared_topology():
    result = stagek.classify([record("topology") for _ in range(27)])
    assert result["primary_failure_locus"] == "topology"


def test_classify_heterogeneous_predicates():
    result = stagek.classify([record("direction_retention"), record("topology")])
    assert result["primary_failure_locus"] == "heterogeneous_rejection_predicates"


def test_classify_empty_callback_surface():
    records = [{"predicate_comparison": {"locus": "explicit_callback_does_not_identify_rejection", "discriminator_predicate": None}}]
    result = stagek.classify(records)
    assert result["primary_failure_locus"] == "callback_observability"


def test_normalized_current_source_matches_parent_blob():
    root = Path(__file__).resolve().parents[1]
    if not (root / ".git").exists():
        pytest.skip("requires the real repository commit graph")
    parent = stagek._git(root, "show", "{}:{}".format(stagek.BASE_EVIDENCE_COMMIT, stagek.STAGEE_RELATIVE_PATH)).decode("utf-8")
    current = (root / stagek.STAGEE_RELATIVE_PATH).read_text(encoding="utf-8")
    assert stagek.normalized_integrator_ast(current) == stagek.normalized_integrator_ast(parent)


def test_integrator_callback_annotation_contains_no_target():
    assert "target" not in stagee.integrate_rowwise.__annotations__


def test_stagek_does_not_access_holdout_or_probe_objects():
    names = set(stagek.run_calibration.__code__.co_names)
    assert not any("holdout" in name.lower() for name in names)
    assert not any("probe" in name.lower() for name in names)
