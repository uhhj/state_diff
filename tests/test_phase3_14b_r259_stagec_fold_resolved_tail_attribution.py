from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import numpy as np
import pytest

from ccda_phase3 import phase314b_r259_stagec_fold_resolved_tail_attribution as c


def spec():
    return c.stagex.StageXSpec()


def basic_arrays(rows: int = 6):
    control = np.zeros((rows, 1, 1), dtype=np.float32)
    target = np.ones_like(control)
    candidate = np.full_like(control, 0.5)
    scale = np.ones(rows, dtype=np.float64)
    risk = np.linspace(0.0, 1.0, rows, dtype=np.float64)
    constant = np.full(rows, 0.5, dtype=np.float64)
    groups = np.asarray(["g{}".format(index // 2) for index in range(rows)])
    conditions = np.asarray(["free" if index % 2 == 0 else "hidden" for index in range(rows)])
    return control, target, candidate, scale, risk, constant, groups, conditions


def make_fold_record(fold: int, acceptance: float, raw_support: float, *, lost=2, avoided=1):
    rows = 100
    return {
        "outer_fold": fold,
        "timestep": 10,
        "raw_candidate": {"support_rate": raw_support},
        "final_policy_metrics": {"acceptance_rate": acceptance},
        "decision_partition": {
            "rejected_would_improve_count": lost,
            "rejected_adverse_count": avoided,
            "rejected_lost_beneficial_sse_mass": float(lost) / rows,
            "rejected_avoided_adverse_sse_mass": float(avoided) / rows,
        },
    }


def six_t10_records(failing):
    records = []
    for fold in range(6):
        records.append(
            make_fold_record(
                fold,
                0.40 if fold in failing else 0.60,
                0.70,
                lost=3,
                avoided=1,
            )
        )
    return records


def synthetic_reproduction_pair():
    fields = {
        "locked_scientific_base": {"a": 1},
        "stagex_spec": {"b": 2},
        "policy_population": [{"id": "p"}],
        "population": {"rows": 638},
        "outer_fold_selections": [{"outer_fold": i} for i in range(6)],
        "inner_selection_modal_policy_diagnostic": {"policy_id": "p"},
        "outer_crossfit_fold_selected_policy_records": {"10": {"x": 1}},
        "outer_crossfit_baseline_records": {"10": {"x": 2}},
        "full_objective_oof_fixed_policy_selection": {"selected_policy_id": "p"},
        "full_objective_oof_fixed_policy_records": {"p": {"10": {"x": 3}}},
        "procedure_gate_totals": {"10": {"x": 0}},
        "execution_counts": dict(c.EXPECTED_EXECUTION_COUNTS),
    }
    kernel_payload = dict(fields)
    kernel_payload["scientific_status"] = "BLOCKED"
    kernel_payload["primary_failure_locus"] = "outer_crossfit_tail_fidelity_failure"
    stagea_worker = dict(fields)
    stagea_worker["scientific_status"] = "BLOCKED"
    stagea_worker["scientific_kernel"] = {
        "kernel_primary_failure_locus": "outer_crossfit_tail_fidelity_failure"
    }
    return kernel_payload, stagea_worker


def synthetic_worker_payload():
    dummy_record = {
        "outer_fold": 0,
        "timestep": 10,
        "risk_calibration": {"bins": [{"count": 0}] * 5},
    }
    records = []
    for fold in range(6):
        for timestep in c.LOCKED_TIMESTEPS:
            value = copy.deepcopy(dummy_record)
            value["outer_fold"] = fold
            value["timestep"] = timestep
            records.append(value)
    payload = {
        "schema": c.WORKER_SCHEMA,
        "execution_verdict": "PASS",
        "scientific_status": "BLOCKED",
        "selected_configuration": None,
        "train_only_recommendation": None,
        "selection_holdout_evaluation_count_added": 0,
        "cumulative_selection_holdout_evaluation_count": 1,
        "frozen_probe_accessed": False,
        "rerun_authorized": False,
        "execution_counts": dict(c.EXPECTED_EXECUTION_COUNTS),
        "stagea_reproduction_identity": {"all_exact": True},
        "instrumentation_identity": {
            "stitch_call_count": 1,
            "outer_outputs_unchanged": True,
            "outer_outputs_before_sha256": "a" * 64,
            "outer_outputs_after_sha256": "a" * 64,
        },
        "fold_resolved_attribution": {
            "fold_timestep_record_count": 18,
            "fold_timestep_records": records,
        },
        "durable_write_ahead_evidence": True,
        "write_ahead_location_outside_git_worktree": True,
    }
    payload.update({key: False for key in c.FALSE_BOUNDARIES})
    payload["worker_result_sha256"] = c.sha256_bytes(c.stable_json_bytes(payload))
    return payload


def synthetic_summary_payload():
    payload = {
        "schema": c.SCHEMA,
        "execution_verdict": "PASS",
        "scientific_status": "BLOCKED",
        "selected_configuration": None,
        "train_only_recommendation": None,
        "selection_holdout_evaluation_count_added": 0,
        "cumulative_selection_holdout_evaluation_count": 1,
        "frozen_probe_accessed": False,
        "rerun_authorized": False,
        "durable_evidence_protocol": {
            "worker_result_persisted_before_controller_decoration": True,
            "external_write_ahead_outside_git_worktree": True,
            "repository_evidence_promoted_byte_exact": True,
            "worker_result_sha_recorded_before_cleanup": True,
            "write_ahead_files_write_once": True,
            "controller_recomputed_worker_science": False,
        },
        "process_topology": {"processes_distinct": True},
        "stagea_reproduction_identity": {"all_exact": True},
    }
    payload.update({key: False for key in c.FALSE_BOUNDARIES})
    payload["summary_sha256"] = c.sha256_bytes(c.stable_json_bytes(payload))
    return payload


def synthetic_outer_outputs(rows: int = 12):
    outputs = {}
    fold_rows = rows // 6
    for fold in range(6):
        indices = np.arange(fold * fold_rows, (fold + 1) * fold_rows)
        outputs[fold] = {}
        for timestep in c.LOCKED_TIMESTEPS:
            outputs[fold][timestep] = {}
            for shrinkage in c.DIRECTION_SHRINKAGES:
                control = np.zeros((fold_rows, 1, 1), dtype=np.float32)
                candidate = np.full_like(control, 0.5 * shrinkage)
                outputs[fold][timestep][shrinkage] = {
                    "indices": indices,
                    "control": control,
                    "direction": np.full_like(control, shrinkage),
                    "candidate": candidate,
                    "selected_scale": np.ones(fold_rows, dtype=np.float64),
                    "risk_probability": np.full(fold_rows, 0.1, dtype=np.float64),
                    "risk_constant_probability": np.full(fold_rows, 0.5, dtype=np.float64),
                    "gate_counts": {
                        "length_log_z_element_mismatch_count": 0,
                        "aligned_upper_element_failure_count": 0,
                        "strict_pass_aligned_fail_row_count": 0,
                    },
                }
    return outputs


def synthetic_selections():
    policy = c.kernel.TailPolicy(shrinkage=1.0, risk_threshold=0.5)
    return [
        {
            "outer_fold": fold,
            "selected_policy": {"shrinkage": 1.0, "risk_threshold": 0.5},
            "selected_policy_id": policy.policy_id,
            "selected_policy_inner_eligible": fold not in (1, 4),
            "diagnostic_fallback_used": fold in (1, 4),
        }
        for fold in range(6)
    ]


def test_phase_name():
    assert c.PHASE == "Phase3.14b-r2.5.9 Stage C"


def test_starting_commit_is_frozen():
    assert c.BASE_STAGEB_RESUME1_EVIDENCE_COMMIT == "4f8a73fb5cc9b60c92c440b6e61bc1765704f3e5"
    assert c.EXPECTED_REMOTE == "4f8a73fb5cc9b60c92c440b6e61bc1765704f3e5"


def test_stageb_report_hash_is_frozen():
    assert c.EXPECTED_STAGEB_RESUME1_REPORT_SHA256.startswith("b1a4729b")


def test_stagea_worker_hash_is_frozen():
    assert c.EXPECTED_STAGEA_WORKER_SHA256.startswith("f9524bf1")


def test_policy_bank_is_frozen():
    assert c.DIRECTION_SHRINKAGES == (0.5, 0.75, 1.0)


def test_timesteps_are_frozen():
    assert c.LOCKED_TIMESTEPS == (10, 25, 50)


def test_stable_json_is_sorted():
    assert c.stable_json_bytes({"b": 1, "a": 2}).startswith(b'{\n  "a"')


def test_compact_json_is_compact():
    assert c.compact_json_bytes({"b": 1, "a": 2}) == b'{"a":2,"b":1}'


def test_anonymous_group_hash_is_deterministic():
    assert c.anonymous_group_hash("group-a") == c.anonymous_group_hash("group-a")


def test_anonymous_group_hash_changes():
    assert c.anonymous_group_hash("group-a") != c.anonymous_group_hash("group-b")


def test_anonymous_group_hash_does_not_contain_raw_value():
    value = c.anonymous_group_hash("secret-group")
    assert "secret-group" not in value
    assert len(value) == 64


def test_array_stats_empty():
    assert c._array_stats(np.asarray([]))["count"] == 0


def test_array_stats_nonempty():
    result = c._array_stats(np.asarray([1.0, 2.0, 3.0]))
    assert result["count"] == 3
    assert result["mean"] == 2.0


def test_calibration_bins_cover_support():
    risk = np.asarray([0.0, 0.2, 0.4, 0.8, 1.0])
    adverse = np.asarray([False, False, True, True, True])
    support = np.ones(5, dtype=bool)
    result = c.calibration_bins(risk, adverse, support)
    assert sum(item["count"] for item in result) == 5


def test_calibration_bins_accept_one_in_last_bin():
    result = c.calibration_bins(
        np.asarray([1.0]), np.asarray([True]), np.asarray([True])
    )
    assert result[-1]["count"] == 1


def test_calibration_bins_ignore_unsupported():
    result = c.calibration_bins(
        np.asarray([0.1, 0.9]),
        np.asarray([False, True]),
        np.asarray([True, False]),
    )
    assert sum(item["count"] for item in result) == 1


def test_calibration_bins_reject_shape():
    with pytest.raises(c.StageCError):
        c.calibration_bins(np.zeros(2), np.zeros(3), np.ones(2, dtype=bool))


def test_calibration_bins_reject_out_of_range():
    with pytest.raises(c.StageCError):
        c.calibration_bins(
            np.asarray([1.1]), np.asarray([False]), np.asarray([True])
        )


def test_extract_probe_process_id():
    assert c.extract_probe_process_id({"process_id": 42}) == 42


def test_extract_probe_process_id_rejects_nested():
    with pytest.raises(c.StageCError):
        c.extract_probe_process_id({"probe": {"process_id": 1}, "process_id": 2})


def test_extract_probe_process_id_rejects_bool():
    with pytest.raises(c.StageCError):
        c.extract_probe_process_id({"process_id": True})


def test_child_environment_prepends_root(monkeypatch, tmp_path):
    monkeypatch.setenv("PYTHONPATH", "/x")
    result = c.child_environment(tmp_path)
    assert result["PYTHONPATH"].split(os.pathsep)[0] == str(tmp_path.resolve())


def test_atomic_write_once(tmp_path):
    path = tmp_path / "evidence.json"
    c.atomic_write_once(path, b"{}\n")
    assert path.read_bytes() == b"{}\n"
    with pytest.raises(c.StageCError):
        c.atomic_write_once(path, b"again")


def test_fold_policy_attribution_basic():
    args = basic_arrays()
    result = c.fold_policy_attribution(
        control=args[0],
        raw_candidate=args[2],
        raw_scale=args[3],
        risk_probability=args[4],
        risk_constant_probability=args[5],
        threshold=0.5,
        target=args[1],
        groups=args[6],
        conditions=args[7],
        spec=spec(),
        policy_id="p",
        outer_fold=0,
        selected_policy_inner_eligible=True,
        diagnostic_fallback_used=False,
    )
    assert result["row_count"] == 6
    assert result["group_count"] == 3
    assert result["condition_count"] == 2


def test_fold_policy_partition_covers_raw_support():
    args = basic_arrays()
    result = c.fold_policy_attribution(
        control=args[0], raw_candidate=args[2], raw_scale=args[3],
        risk_probability=args[4], risk_constant_probability=args[5],
        threshold=0.5, target=args[1], groups=args[6], conditions=args[7],
        spec=spec(), policy_id="p", outer_fold=0,
        selected_policy_inner_eligible=True, diagnostic_fallback_used=False,
    )
    partition = result["decision_partition"]
    total = sum(partition[key] for key in (
        "accepted_improving_count", "accepted_adverse_count", "accepted_neutral_count",
        "rejected_would_improve_count", "rejected_adverse_count", "rejected_neutral_count",
    ))
    assert total == result["raw_candidate"]["support_count"]


def test_fold_policy_does_not_persist_raw_group_names():
    args = basic_arrays()
    result = c.fold_policy_attribution(
        control=args[0], raw_candidate=args[2], raw_scale=args[3],
        risk_probability=args[4], risk_constant_probability=args[5],
        threshold=0.5, target=args[1], groups=args[6], conditions=args[7],
        spec=spec(), policy_id="p", outer_fold=0,
        selected_policy_inner_eligible=True, diagnostic_fallback_used=False,
    )
    text = json.dumps(result["anonymous_group_aggregates"], sort_keys=True)
    assert "g0" not in text
    assert "g1" not in text


def test_fold_policy_persists_condition_aggregates():
    args = basic_arrays()
    result = c.fold_policy_attribution(
        control=args[0], raw_candidate=args[2], raw_scale=args[3],
        risk_probability=args[4], risk_constant_probability=args[5],
        threshold=0.5, target=args[1], groups=args[6], conditions=args[7],
        spec=spec(), policy_id="p", outer_fold=0,
        selected_policy_inner_eligible=True, diagnostic_fallback_used=False,
    )
    assert {item["condition_name"] for item in result["condition_aggregates"]} == {"free", "hidden"}


def test_fold_policy_rejects_bad_threshold():
    args = basic_arrays()
    with pytest.raises(c.StageCError):
        c.fold_policy_attribution(
            control=args[0], raw_candidate=args[2], raw_scale=args[3],
            risk_probability=args[4], risk_constant_probability=args[5],
            threshold=1.1, target=args[1], groups=args[6], conditions=args[7],
            spec=spec(), policy_id="p", outer_fold=0,
            selected_policy_inner_eligible=True, diagnostic_fallback_used=False,
        )


def test_outer_outputs_fingerprint_deterministic():
    outputs = synthetic_outer_outputs()
    assert c.outer_outputs_fingerprint(outputs) == c.outer_outputs_fingerprint(outputs)


def test_outer_outputs_fingerprint_detects_change():
    outputs = synthetic_outer_outputs()
    before = c.outer_outputs_fingerprint(outputs)
    outputs[0][10][1.0]["candidate"][0, 0, 0] += 0.1
    assert c.outer_outputs_fingerprint(outputs) != before


def test_build_fold_resolved_records(monkeypatch):
    monkeypatch.setattr(c, "EXPECTED_ROWS", 12)
    context = {
        "objective_condition_name": np.asarray(["free", "hidden"] * 6),
    }
    target = np.ones((12, 1, 1), dtype=np.float32)
    groups = np.asarray(["g{}".format(i) for i in range(12)])
    result = c.build_fold_resolved_records(
        context=context,
        outer_outputs=synthetic_outer_outputs(),
        outer_selections=synthetic_selections(),
        target=target,
        groups=groups,
        spec=spec(),
    )
    assert result["fold_timestep_record_count"] == 18
    assert result["risk_calibration_bin_count"] == 90


def test_build_fold_resolved_records_rejects_overlap(monkeypatch):
    monkeypatch.setattr(c, "EXPECTED_ROWS", 12)
    outputs = synthetic_outer_outputs()
    for timestep in c.LOCKED_TIMESTEPS:
        for shrinkage in c.DIRECTION_SHRINKAGES:
            outputs[1][timestep][shrinkage]["indices"] = outputs[0][timestep][shrinkage]["indices"]
    with pytest.raises(c.StageCError):
        c.build_fold_resolved_records(
            context={"objective_condition_name": np.asarray(["x"] * 12)},
            outer_outputs=outputs,
            outer_selections=synthetic_selections(),
            target=np.ones((12, 1, 1), dtype=np.float32),
            groups=np.asarray(["g{}".format(i) for i in range(12)]),
            spec=spec(),
        )


def test_compare_stagea_reproduction_exact():
    current, old = synthetic_reproduction_pair()
    assert c.compare_stagea_reproduction(current, old)["all_exact"] is True


def test_compare_stagea_reproduction_rejects_mismatch():
    current, old = synthetic_reproduction_pair()
    current["execution_counts"] = {"x": 1}
    with pytest.raises(c.StageCError):
        c.compare_stagea_reproduction(current, old)


def test_classify_candidate_support_limited():
    records = six_t10_records([1, 4])
    for item in records:
        if item["outer_fold"] in (1, 4):
            item["raw_candidate"]["support_rate"] = 0.45
    result = c.classify_t10_fold_attribution(records, [1, 4])
    assert result["primary_failure_locus"] == "t10_candidate_support_limited_by_raw_candidate_availability"


def test_classify_beneficial_overrejection():
    result = c.classify_t10_fold_attribution(six_t10_records([1, 4]), [1, 4])
    assert result["primary_failure_locus"] == "t10_risk_abstention_overrejects_beneficial_candidates"


def test_classify_risk_mixed_quality():
    records = six_t10_records([1, 4])
    for item in records:
        if item["outer_fold"] in (1, 4):
            item["decision_partition"]["rejected_would_improve_count"] = 1
            item["decision_partition"]["rejected_adverse_count"] = 2
            item["decision_partition"]["rejected_lost_beneficial_sse_mass"] = 0.01
            item["decision_partition"]["rejected_avoided_adverse_sse_mass"] = 0.02
    result = c.classify_t10_fold_attribution(records, [1, 4])
    assert result["primary_failure_locus"] == "t10_risk_abstention_limited_with_mixed_candidate_quality"


def test_classify_mixed_support_and_abstention():
    records = six_t10_records([1, 4])
    records[1]["raw_candidate"]["support_rate"] = 0.40
    result = c.classify_t10_fold_attribution(records, [1, 4])
    assert result["primary_failure_locus"] == "t10_mixed_candidate_support_and_risk_abstention_failure"


def test_classify_records_fallback_overlap():
    result = c.classify_t10_fold_attribution(six_t10_records([1, 4]), [1, 4])
    assert result["failed_fold_fallback_overlap"] == [1, 4]
    assert result["all_failed_folds_are_fallback_folds"] is True


def test_classify_does_not_authorize_threshold_change():
    result = c.classify_t10_fold_attribution(six_t10_records([1]), [1, 4])
    assert result["threshold_relaxation_authorized"] is False


def test_classify_rejects_no_failure():
    with pytest.raises(c.StageCError):
        c.classify_t10_fold_attribution(six_t10_records([]), [1, 4])


def test_classify_rejects_wrong_population():
    with pytest.raises(c.StageCError):
        c.classify_t10_fold_attribution(six_t10_records([1])[:5], [1, 4])


def test_validate_worker_payload():
    c.validate_worker_evidence(synthetic_worker_payload())


def test_validate_worker_rejects_ready():
    payload = synthetic_worker_payload()
    payload["scientific_status"] = "READY"
    payload["worker_result_sha256"] = c.sha256_bytes(
        c.stable_json_bytes({k: v for k, v in payload.items() if k != "worker_result_sha256"})
    )
    with pytest.raises(c.StageCError):
        c.validate_worker_evidence(payload)


def test_validate_worker_rejects_mutation():
    payload = synthetic_worker_payload()
    payload["instrumentation_identity"]["outer_outputs_after_sha256"] = "b" * 64
    payload["worker_result_sha256"] = c.sha256_bytes(
        c.stable_json_bytes({k: v for k, v in payload.items() if k != "worker_result_sha256"})
    )
    with pytest.raises(c.StageCError):
        c.validate_worker_evidence(payload)


def test_validate_worker_rejects_holdout_access():
    payload = synthetic_worker_payload()
    payload["selection_holdout_evaluation_count_added"] = 1
    payload["worker_result_sha256"] = c.sha256_bytes(
        c.stable_json_bytes({k: v for k, v in payload.items() if k != "worker_result_sha256"})
    )
    with pytest.raises(c.StageCError):
        c.validate_worker_evidence(payload)


def test_validate_summary():
    c.validate_summary(synthetic_summary_payload())


def test_validate_summary_rejects_controller_recompute():
    payload = synthetic_summary_payload()
    payload["durable_evidence_protocol"]["controller_recomputed_worker_science"] = True
    payload["summary_sha256"] = c.sha256_bytes(
        c.stable_json_bytes({k: v for k, v in payload.items() if k != "summary_sha256"})
    )
    with pytest.raises(c.StageCError):
        c.validate_summary(payload)


def test_validate_summary_rejects_recommendation():
    payload = synthetic_summary_payload()
    payload["train_only_recommendation"] = {"x": 1}
    payload["summary_sha256"] = c.sha256_bytes(
        c.stable_json_bytes({k: v for k, v in payload.items() if k != "summary_sha256"})
    )
    with pytest.raises(c.StageCError):
        c.validate_summary(payload)


def test_blocked_report_before_worker():
    result = c.blocked_report(repository=None, error=RuntimeError("x"))
    assert result["durable_worker_evidence_available"] is False
    assert result["rerun_authorized"] is False


def test_blocked_report_with_worker(tmp_path, monkeypatch):
    path = tmp_path / "worker.json"
    path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(c, "validate_worker_evidence", lambda value: None)
    result = c.blocked_report(repository=None, error=RuntimeError("x"), worker_path=path)
    assert result["durable_worker_evidence_available"] is True
    assert "FINALIZE_R259_STAGEC" in result["required_next_path"]


def test_false_boundaries_are_present():
    payload = synthetic_worker_payload()
    assert all(payload[key] is False for key in c.FALSE_BOUNDARIES)


def test_implementation_paths_are_add_only():
    assert len(c.IMPLEMENTATION_PATHS) == 4
    assert all(status == "A" for status, _ in c.IMPLEMENTATION_PATHS)


def test_no_row_index_field_in_group_aggregate():
    args = basic_arrays()
    result = c.fold_policy_attribution(
        control=args[0], raw_candidate=args[2], raw_scale=args[3],
        risk_probability=args[4], risk_constant_probability=args[5],
        threshold=0.5, target=args[1], groups=args[6], conditions=args[7],
        spec=spec(), policy_id="p", outer_fold=0,
        selected_policy_inner_eligible=True, diagnostic_fallback_used=False,
    )
    assert all("row_index" not in item for item in result["anonymous_group_aggregates"])


def test_source_does_not_persist_raw_group_field():
    source = Path(c.__file__).read_text(encoding="utf-8")
    assert '"raw_group"' not in source
    assert '"group_id"' not in source


def test_expected_diagnostic_counts():
    assert c.EXPECTED_DIAGNOSTIC_COUNTS == {
        "fold_timestep_record_count": 18,
        "fold_resolved_policy_metric_evaluation_count": 36,
        "risk_calibration_bin_count": 90,
    }
