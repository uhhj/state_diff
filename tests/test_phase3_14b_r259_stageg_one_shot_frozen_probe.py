from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from ccda_phase3 import phase314b_r259_stageg_one_shot_frozen_probe as g


@pytest.mark.parametrize(
    "name,expected",
    [
        ("LOCKED_TIMESTEPS", (10, 25, 50)),
        ("OUTER_FOLDS", 6),
        ("EXPECTED_OBJECTIVE_ROWS", 638),
        ("EXPECTED_OBJECTIVE_GROUPS", 126),
        ("EXPECTED_FROZEN_PROBE_ROWS", 126),
        ("EXPECTED_INTERNAL_SCALE_COUNT", 7),
        ("FROZEN_PROBE_NOISE_SEED_OFFSET", 9901),
        ("CANONICAL_DESCRIPTOR", "compact_v1"),
        ("CANONICAL_T10_RISK_L2", 4.0),
        ("LEGACY_CONTROL_RISK_L2", 1.0),
        ("LOCKED_SHRINKAGE", 1.0),
        ("LOCKED_RISK_THRESHOLD", 0.5),
        ("LOCKED_TEMPERATURE", 1.0),
    ],
)
def test_frozen_constants(name, expected):
    assert getattr(g, name) == expected


@pytest.mark.parametrize(
    "value",
    [g.BASE_STAGEF_IMPLEMENTATION, g.BASE_STAGEF_EVIDENCE, g.EXPECTED_SUBMODULE],
)
def test_commit_values_are_sha1(value):
    assert len(value) == 40
    int(value, 16)


def test_canonical_recipe_id_exact():
    assert g.CANONICAL_RECIPE_ID == (
        "desc_compact_v1__l2_4.00__temp_1.00__shrink_1.00__risk_0.50"
    )


def test_implementation_is_add_only_four_files():
    assert len(g.IMPLEMENTATION_PATHS) == 4
    assert {status for status, _ in g.IMPLEMENTATION_PATHS} == {"A"}


def test_expected_counts_exact():
    assert g.EXPECTED_EXECUTION_COUNTS == {
        "objective_direction_fit_count": 21,
        "candidate_generation_count": 21,
        "objective_risk_fit_count": 3,
        "descriptor_build_count": 6,
        "internal_scale_attempt_count": 147,
        "frozen_probe_control_prediction_count": 1,
        "frozen_probe_joint_evaluation_count": 1,
    }


def test_spec_accepts_frozen_defaults():
    g.FrozenProbeSpec().validate()


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("timesteps", (10, 25), "timestep"),
        ("frozen_probe_noise_seed_offset", 1, "noise"),
        ("required_acceptance_rate", 0.4, "acceptance"),
        ("required_positive_reduction_rate", 0.4, "positive"),
        ("adverse_sse_reduction_fraction", 0.2, "adverse"),
        ("group_cvar_fraction", 0.1, "CVaR"),
    ],
)
def test_spec_rejects_changes(field, value, match):
    values = g.FrozenProbeSpec().__dict__.copy()
    values[field] = value
    with pytest.raises(g.StageGError, match=match):
        g.FrozenProbeSpec(**values).validate()


def test_stable_json_order_independent():
    assert g.stable_json_bytes({"b": 1, "a": 2}) == g.stable_json_bytes({"a": 2, "b": 1})


def test_sha256_bytes_known():
    assert g.sha256_bytes(b"abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_sha256_array_includes_shape_and_dtype():
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    b = np.arange(6, dtype=np.float64).reshape(2, 3)
    c = np.arange(6, dtype=np.float32).reshape(3, 2)
    assert g.sha256_array(a) != g.sha256_array(b)
    assert g.sha256_array(a) != g.sha256_array(c)


def test_atomic_write_once(tmp_path):
    path = tmp_path / "x.json"
    g.atomic_write_once(path, b"x")
    assert path.read_bytes() == b"x"


def test_atomic_write_refuses_overwrite(tmp_path):
    path = tmp_path / "x.json"
    path.write_bytes(b"old")
    with pytest.raises(g.StageGError, match="write-once"):
        g.atomic_write_once(path, b"new")


def test_promote_write_ahead(tmp_path):
    source = tmp_path / "wa.json"
    target = tmp_path / "repo" / "out.json"
    source.write_text('{"x": 1}', encoding="utf-8")
    digest = g.promote_write_ahead(source, target, lambda payload: payload)
    assert target.read_bytes() == source.read_bytes()
    assert digest == g.sha256_bytes(source.read_bytes())


def test_promote_requires_source(tmp_path):
    with pytest.raises(g.StageGError, match="missing"):
        g.promote_write_ahead(tmp_path / "none", tmp_path / "out", lambda payload: payload)


def stagef_payload():
    return {
        "schema": "phase314b_r259_stagef_operational_family_canonical_lock_v1",
        "execution_verdict": "PASS",
        "scientific_status": "READY",
        "summary_sha256": g.STAGEF_SELF_SHA256,
        "lock": {
            "canonical_lock_ready": True,
            "canonicalization": {
                "canonical_recipe_id": g.CANONICAL_RECIPE_ID,
                "performance_metrics_used": False,
            },
        },
        "train_only_recommendation": {
            "backbone_id": "segment_target_rr64_feasible",
            "timesteps": [10, 25, 50],
            "timestep_policy": "joint_all_timesteps_no_cherry_pick",
            "t10_descriptor_id": "compact_v1",
            "t10_descriptor_dimension": 17,
            "t10_risk_l2": 4.0,
            "t10_prevalence_centered_temperature": 1.0,
            "t10_direction_shrinkage": 1.0,
            "t10_adverse_risk_threshold": 0.5,
            "t25_t50_risk_path": "frozen_stagec_resume1_legacy_control",
            "frozen_probe_access_authorized_by_this_report": False,
        },
    }


def test_validate_stagef_report_accepts(monkeypatch, tmp_path):
    path = tmp_path / "stagef.json"
    path.write_bytes(g.stable_json_bytes(stagef_payload()))
    monkeypatch.setattr(g, "STAGEF_REPORT_SHA256", g.sha256_file(path))
    assert g.validate_stagef_report(path)["scientific_status"] == "READY"


@pytest.mark.parametrize(
    "mutator,match",
    [
        (lambda p: p.update(schema="bad"), "schema"),
        (lambda p: p.update(scientific_status="BLOCKED"), "not READY"),
        (lambda p: p["lock"].update(canonical_lock_ready=False), "lock"),
        (lambda p: p["lock"]["canonicalization"].update(canonical_recipe_id="bad"), "recipe"),
        (lambda p: p["lock"]["canonicalization"].update(performance_metrics_used=True), "performance"),
        (lambda p: p["train_only_recommendation"].update(t10_risk_l2=1.0), "recommendation"),
        (lambda p: p["train_only_recommendation"].update(frozen_probe_access_authorized_by_this_report=True), "authorized"),
    ],
)
def test_validate_stagef_report_rejects(monkeypatch, tmp_path, mutator, match):
    payload = stagef_payload()
    mutator(payload)
    path = tmp_path / "stagef.json"
    path.write_bytes(g.stable_json_bytes(payload))
    monkeypatch.setattr(g, "STAGEF_REPORT_SHA256", g.sha256_file(path))
    with pytest.raises(g.StageGError, match=match):
        g.validate_stagef_report(path)


def environment_probe():
    payload = {
        "phase": g.PHASE,
        "schema": g.PROBE_SCHEMA,
        "execution_verdict": "PASS",
        "process_id": 101,
        "environment": {
            "compatibility_sha256": "x",
            "compatibility_pass": True,
            "compatibility": {"required_operation_pass": True},
        },
        "portable_environment_audit": {"portable_contract_passed": True},
        "required_operation_dry_run": {"pass": True},
        "durable_write_ahead_evidence": True,
        "frozen_probe_accessed": False,
    }
    payload["probe_evidence_sha256"] = g.sha256_bytes(g.stable_json_bytes(payload))
    return payload


def test_validate_environment_probe_accepts():
    assert g.validate_environment_probe(environment_probe())["compatibility_sha256"] == "x"


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("schema", "bad", "schema"),
        ("process_id", 0, "PID"),
        ("environment", None, "environment"),
        ("durable_write_ahead_evidence", False, "durable"),
        ("frozen_probe_accessed", True, "accessed"),
    ],
)
def test_validate_environment_probe_rejects(field, value, match):
    payload = environment_probe()
    payload[field] = value
    payload["probe_evidence_sha256"] = g.sha256_bytes(
        g.stable_json_bytes({k: v for k, v in payload.items() if k != "probe_evidence_sha256"})
    )
    with pytest.raises(g.StageGError, match=match):
        g.validate_environment_probe(payload)


def test_risk_l2_contract():
    assert g._risk_l2_for_timestep(10) == 4.0
    assert g._risk_l2_for_timestep(25) == 1.0
    assert g._risk_l2_for_timestep(50) == 1.0


def metric_record(pass_value=True):
    return {
        "eligibility": {"pass": pass_value, "checks": {"acceptance": pass_value}},
        "gate_counts": {
            "length_log_z_element_mismatch_count": 0,
            "aligned_upper_element_failure_count": 0,
            "strict_pass_aligned_fail_row_count": 0,
        },
    }


def worker_payload(ready=True):
    records = {str(t): metric_record(ready) for t in g.LOCKED_TIMESTEPS}
    selected = {
        "backbone_id": "segment_target_rr64_feasible",
        "timesteps": [10, 25, 50],
        "frozen_probe_validation_passed": True,
    } if ready else None
    payload = {
        "schema": g.WORKER_SCHEMA,
        "execution_verdict": "PASS",
        "scientific_status": "READY" if ready else "BLOCKED",
        "preregistration_contract": {
            "one_shot_frozen_probe": True,
            "joint_all_timesteps_no_cherry_pick": True,
            "fallback_forbidden": True,
            "post_probe_refit_forbidden": True,
            "post_probe_threshold_change_forbidden": True,
            "rerun_forbidden": True,
        },
        "execution_counts": dict(g.EXPECTED_EXECUTION_COUNTS),
        "frozen_probe_timestep_records": records,
        "all_timestep_metric_eligibility_pass": ready,
        "all_timesteps_pass": ready,
        "procedure_gate_totals": {
            "length_log_z_element_mismatch_count": 0,
            "aligned_upper_element_failure_count": 0,
            "strict_pass_aligned_fail_row_count": 0,
        },
        "selected_configuration": selected,
        "train_only_recommendation": selected,
        "selection_holdout_evaluation_count_added": 0,
        "cumulative_selection_holdout_evaluation_count": 1,
        "frozen_probe_evaluation_count_added": 1,
        "cumulative_frozen_probe_evaluation_count": 1,
        "frozen_probe_accessed": True,
        "rerun_authorized": False,
    }
    for key in g.FALSE_BOUNDARIES:
        payload[key] = False
    payload["worker_result_sha256"] = g.sha256_bytes(g.stable_json_bytes(payload))
    return payload


@pytest.mark.parametrize("ready", [True, False])
def test_worker_validation_accepts(ready):
    g.validate_worker_evidence(worker_payload(ready))


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("execution_counts", {}, "counts"),
        ("selection_holdout_evaluation_count_added", 1, "selection holdout"),
        ("cumulative_selection_holdout_evaluation_count", 2, "holdout count"),
        ("frozen_probe_evaluation_count_added", 0, "frozen-probe count"),
        ("cumulative_frozen_probe_evaluation_count", 2, "cumulative"),
        ("frozen_probe_accessed", False, "did not record"),
        ("rerun_authorized", True, "rerun"),
    ],
)
def test_worker_validation_rejects_boundaries(field, value, match):
    payload = worker_payload(True)
    payload[field] = value
    payload["worker_result_sha256"] = g.sha256_bytes(
        g.stable_json_bytes({k: v for k, v in payload.items() if k != "worker_result_sha256"})
    )
    with pytest.raises(g.StageGError, match=match):
        g.validate_worker_evidence(payload)


def test_worker_blocked_cannot_select():
    payload = worker_payload(False)
    payload["selected_configuration"] = {"x": 1}
    payload["worker_result_sha256"] = g.sha256_bytes(
        g.stable_json_bytes({k: v for k, v in payload.items() if k != "worker_result_sha256"})
    )
    with pytest.raises(g.StageGError, match="selected"):
        g.validate_worker_evidence(payload)


def summary_payload(ready=True):
    payload = {
        "schema": g.SCHEMA,
        "execution_verdict": "PASS",
        "scientific_status": "READY" if ready else "BLOCKED",
        "frozen_probe_timestep_records": {str(t): metric_record(ready) for t in g.LOCKED_TIMESTEPS},
        "all_timesteps_pass": ready,
        "selected_configuration": {"x": 1} if ready else None,
        "selection_holdout_evaluation_count_added": 0,
        "cumulative_selection_holdout_evaluation_count": 1,
        "frozen_probe_evaluation_count_added": 1,
        "cumulative_frozen_probe_evaluation_count": 1,
        "frozen_probe_accessed": True,
        "rerun_authorized": False,
    }
    for key in g.FALSE_BOUNDARIES:
        payload[key] = False
    payload["summary_sha256"] = g.sha256_bytes(g.stable_json_bytes(payload))
    return payload


@pytest.mark.parametrize("ready", [True, False])
def test_summary_validation_accepts(ready):
    g.validate_summary(summary_payload(ready))


def test_summary_rejects_bad_hash():
    payload = summary_payload(True)
    payload["summary_sha256"] = "x" * 64
    with pytest.raises(g.StageGError, match="self-hash"):
        g.validate_summary(payload)


def test_summary_rejects_blocked_selection():
    payload = summary_payload(False)
    payload["selected_configuration"] = {"x": 1}
    payload["summary_sha256"] = g.sha256_bytes(
        g.stable_json_bytes({k: v for k, v in payload.items() if k != "summary_sha256"})
    )
    with pytest.raises(g.StageGError, match="selected"):
        g.validate_summary(payload)


def test_blocked_before_worker_does_not_claim_probe_access(tmp_path):
    payload = g.blocked_report(
        repository=None,
        error=RuntimeError("x"),
        probe_path=tmp_path / "probe",
        worker_path=tmp_path / "worker",
        worker_started_path=tmp_path / "started",
        access_marker_path=tmp_path / "access",
    )
    assert payload["one_shot_attempt_consumed"] is False
    assert payload["frozen_probe_accessed"] is False
    assert payload["science_reexecution_authorized"] is False


def test_blocked_after_worker_start_consumes_attempt(tmp_path):
    started = tmp_path / "started"
    started.write_text("{}", encoding="utf-8")
    payload = g.blocked_report(
        repository=None,
        error=RuntimeError("x"),
        probe_path=tmp_path / "probe",
        worker_path=tmp_path / "worker",
        worker_started_path=started,
        access_marker_path=tmp_path / "access",
    )
    assert payload["one_shot_attempt_consumed"] is True
    assert payload["rerun_authorized"] is False


def test_blocked_after_access_records_one_evaluation(tmp_path):
    started = tmp_path / "started"
    access = tmp_path / "access"
    started.write_text("{}", encoding="utf-8")
    access.write_text("{}", encoding="utf-8")
    payload = g.blocked_report(
        repository=None,
        error=RuntimeError("x"),
        probe_path=tmp_path / "probe",
        worker_path=tmp_path / "worker",
        worker_started_path=started,
        access_marker_path=access,
    )
    assert payload["frozen_probe_evaluation_count_added"] == 1
    assert payload["frozen_probe_accessed"] is True
    assert payload["selected_configuration"] is None
