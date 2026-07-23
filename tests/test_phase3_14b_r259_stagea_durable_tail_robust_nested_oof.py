from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ccda_phase3 import phase314b_r259_stagea_durable_tail_robust_nested_oof as stagea


def base_adjudication():
    value = {
        "schema": "phase314b_r258_stagex_resume2_lost_worker_result_adjudication_v1",
        "execution_verdict": "PASS",
        "scientific_status": "BLOCKED",
        "primary_failure_locus": "completed_worker_result_not_persisted",
        "root_cause": (
            "phase314b_r258_stagex_resume2_worker_completed_but_result_"
            "irrecoverably_lost_before_repository_evidence"
        ),
        "required_next_path": (
            "PREREGISTER_NEW_OBJECTIVE_TRAIN_ONLY_STAGE_WITH_DURABLE_WRITE_AHEAD_"
            "WORKER_EVIDENCE_AND_NO_HOLDOUT_OR_FROZEN_PROBE_ACCESS"
        ),
        "rerun_authorized": False,
        "selected_configuration": None,
        "train_only_recommendation": None,
        "selection_holdout_evaluation_count_added": 0,
        "cumulative_selection_holdout_evaluation_count": 1,
        "replay_governance": {
            "stagex_resume1_science_replay_authorized": False,
            "environment_probe_reaccess_authorized": False,
            "science_worker_reexecution_authorized": False,
        },
        "future_controller_contract": {
            "durable_worker_result_before_controller_decoration_required": True,
            "worker_result_sha_must_be_recorded_before_temporary_cleanup": True,
            "write_ahead_result_must_be_write_once": True,
            "fresh_stage_must_be_preregistered_before_any_new_science_process": True,
        },
    }
    value["adjudication_sha256"] = stagea.sha256_bytes(stagea.stable_json_bytes(value))
    return value


def probe_payload(pid=101):
    environment = {"compatibility_pass": True, "x": 1}
    value = {
        "phase": stagea.PHASE,
        "schema": stagea.PROBE_SCHEMA,
        "execution_verdict": "PASS",
        "process_id": pid,
        "source_probe_schema": "source",
        "environment": environment,
        "environment_sha256": stagea.resume2a.sha256_bytes(stagea.resume2a.stable_json_bytes(environment)),
        "source_probe_payload_sha256": "a" * 64,
        "cuda_initialization_allowed_in_this_process": True,
        "process_disposable": True,
        "durable_write_ahead_evidence": True,
        "write_ahead_location_outside_git_worktree": True,
    }
    value["probe_evidence_sha256"] = stagea.sha256_bytes(stagea.stable_json_bytes(value))
    return value


def worker_payload(status="BLOCKED", pid=202):
    value = {
        "phase": stagea.PHASE,
        "schema": stagea.WORKER_SCHEMA,
        "execution_verdict": "PASS",
        "scientific_status": status,
        "root_cause": "root",
        "required_next_path": "next",
        "primary_failure_locus": "locus",
        "process_id": pid,
        "repository_head": "implementation",
        "environment_sha256": "b" * 64,
        "cold_cuda_precheck": {"torch_cuda_is_initialized": False},
        "preregistration_contract": {},
        "scientific_kernel": {},
        "locked_scientific_base": {},
        "stagex_spec": {},
        "policy_population": [],
        "population": {},
        "outer_fold_selections": [],
        "inner_selection_modal_policy_diagnostic": {},
        "outer_crossfit_fold_selected_policy_records": {"10": {}, "25": {}, "50": {}},
        "outer_crossfit_baseline_records": {},
        "full_objective_oof_fixed_policy_selection": {},
        "full_objective_oof_fixed_policy_records": {},
        "procedure_gate_totals": {},
        "execution_counts": dict(stagea.EXPECTED_EXECUTION_COUNTS),
        "selected_configuration": None,
        "train_only_recommendation": {"policy": 1} if status == "READY" else None,
        "selection_holdout_evaluation_count_added": 0,
        "cumulative_selection_holdout_evaluation_count": 1,
        "selection_holdout_evaluated_in_r259_stagea": False,
        "rerun_authorized": False,
        "durable_write_ahead_evidence": True,
        "write_ahead_location_outside_git_worktree": True,
    }
    value.update({key: False for key in stagea.FALSE_BOUNDARIES})
    value["worker_result_sha256"] = stagea.sha256_bytes(stagea.stable_json_bytes(value))
    return value


def kernel_payload(status="BLOCKED", locus="outer_crossfit_tail_fidelity_failure"):
    value = {
        "scientific_status": status,
        "root_cause": "old-root",
        "required_next_path": "old-next",
        "primary_failure_locus": locus,
        "process_id": 303,
        "environment_sha256": "c" * 64,
        "cold_cuda_precheck": {"torch_cuda_is_initialized": False},
        "scientific_result_sha256": "d" * 64,
        "locked_scientific_base": {"backbone_id": "segment_target_rr64_feasible"},
        "stagex_spec": {},
        "policy_population": [],
        "population": {},
        "outer_fold_selections": [],
        "inner_selection_modal_policy_diagnostic": {},
        "outer_crossfit_fold_selected_policy_records": {"10": {}, "25": {}, "50": {}},
        "outer_crossfit_baseline_records": {},
        "full_objective_oof_fixed_policy_selection": {},
        "full_objective_oof_fixed_policy_records": {},
        "procedure_gate_totals": {},
        "execution_counts": dict(stagea.EXPECTED_EXECUTION_COUNTS),
        "train_only_recommendation": {"policy": 1} if status == "READY" else None,
    }
    return value


def test_phase_is_new_preregistered_stage():
    assert stagea.PHASE == "Phase3.14b-r2.5.9 Stage A"


def test_base_is_resume2_evidence():
    assert stagea.BASE_RESUME2_EVIDENCE_COMMIT == "34011592bb762b350d4c7489d598a22ee9175042"


def test_implementation_is_add_only_four_files():
    assert len(stagea.IMPLEMENTATION_PATHS) == 4
    assert all(status == "A" for status, _ in stagea.IMPLEMENTATION_PATHS)


def test_output_population_has_durable_probe_worker_and_terminal_reports():
    assert stagea.OUTPUT_PATHS == (
        stagea.PROBE_EVIDENCE,
        stagea.WORKER_EVIDENCE,
        stagea.SUCCESS_REPORT,
        stagea.BLOCKED_REPORT,
    )


def test_expected_execution_counts_are_frozen():
    assert stagea.EXPECTED_EXECUTION_COUNTS["direction_fit_count"] == 108
    assert stagea.EXPECTED_EXECUTION_COUNTS["internal_scale_attempt_count"] == 2268
    assert stagea.EXPECTED_EXECUTION_COUNTS["nonconverged_risk_fit_count"] == 0


def test_stable_json_is_sorted_and_indented():
    assert stagea.stable_json_bytes({"b": 1, "a": 2}) == b'{\n  "a": 2,\n  "b": 1\n}'


def test_sha256_bytes_matches_hashlib():
    import hashlib
    assert stagea.sha256_bytes(b"abc") == hashlib.sha256(b"abc").hexdigest()



def test_promote_write_ahead_is_byte_exact(tmp_path: Path):
    source = tmp_path / "outside.json"
    target = tmp_path / "repo" / "inside.json"
    payload = probe_payload()
    source.write_bytes(stagea.stable_json_bytes(payload) + b"\n")
    observed = stagea.promote_write_ahead(source, target, stagea.validate_probe_evidence)
    assert observed == stagea.sha256_file(source)
    assert target.read_bytes() == source.read_bytes()
    assert stagea.promote_write_ahead(source, target, stagea.validate_probe_evidence) == observed

def test_atomic_write_once_and_refuses_overwrite(tmp_path: Path):
    path = tmp_path / "report.json"
    stagea.atomic_write_once(path, b"{}\n")
    assert path.read_bytes() == b"{}\n"
    with pytest.raises(stagea.StageAError, match="write-once"):
        stagea.atomic_write_once(path, b"again")


def test_atomic_write_leaves_no_temporary(tmp_path: Path):
    path = tmp_path / "report.json"
    stagea.atomic_write_once(path, b"x")
    assert list(tmp_path.glob("*.tmp.*")) == []


def test_extract_probe_pid_accepts_top_level():
    assert stagea.extract_probe_process_id({"process_id": 42}) == 42


@pytest.mark.parametrize("value", [None, 0, -1, True, "1"])
def test_extract_probe_pid_rejects_invalid(value):
    with pytest.raises(stagea.StageAError):
        stagea.extract_probe_process_id({"process_id": value})


def test_extract_probe_pid_rejects_nested_wrapper():
    with pytest.raises(stagea.StageAError, match="nested"):
        stagea.extract_probe_process_id({"process_id": 1, "probe": {}})


def test_validate_base_adjudication_accepts_bound_shape(monkeypatch):
    value = base_adjudication()
    monkeypatch.setattr(stagea, "EXPECTED_BASE_ADJUDICATION_SHA256", value["adjudication_sha256"])
    assert stagea.validate_base_adjudication(value) is value


def test_validate_base_adjudication_rejects_replay_authorization(monkeypatch):
    value = base_adjudication()
    value["replay_governance"]["science_worker_reexecution_authorized"] = True
    value["adjudication_sha256"] = stagea.sha256_bytes(
        stagea.stable_json_bytes({k: v for k, v in value.items() if k != "adjudication_sha256"})
    )
    monkeypatch.setattr(stagea, "EXPECTED_BASE_ADJUDICATION_SHA256", value["adjudication_sha256"])
    with pytest.raises(stagea.StageAError, match="governance"):
        stagea.validate_base_adjudication(value)


def test_child_environment_puts_root_first(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("PYTHONPATH", "/x:/y")
    env = stagea.child_environment(tmp_path)
    assert env["PYTHONPATH"].split(":")[0] == str(tmp_path.resolve())


def test_validate_probe_evidence_accepts_valid_payload():
    assert stagea.validate_probe_evidence(probe_payload())["x"] == 1


def test_validate_probe_evidence_rejects_bad_self_hash():
    value = probe_payload()
    value["probe_evidence_sha256"] = "bad"
    with pytest.raises(stagea.StageAError, match="self-hash"):
        stagea.validate_probe_evidence(value)


def test_make_probe_evidence_uses_top_level_pid(monkeypatch, tmp_path: Path):
    source = {
        "schema": "source",
        "mode": "environment_probe",
        "execution_verdict": "PASS",
        "process_id": 77,
        "environment": {"compatibility_pass": True},
        "environment_sha256": stagea.resume2a.sha256_bytes(
            stagea.resume2a.stable_json_bytes({"compatibility_pass": True})
        ),
    }
    monkeypatch.setattr(stagea.resume2a, "environment_probe_payload", lambda root: source)
    monkeypatch.setattr(
        stagea.resume2a, "validate_probe_payload", lambda payload: payload["environment"]
    )
    result = stagea.make_probe_evidence(tmp_path)
    assert result["process_id"] == 77
    assert "probe" not in result


@pytest.mark.parametrize(
    "status,locus,expected_status,expected_locus",
    [
        ("READY", "tail_robust_nested_selection_procedure_ready", "READY", "durable_tail_robust_nested_oof_ready"),
        ("BLOCKED", "outer_crossfit_tail_fidelity_failure", "BLOCKED", "outer_crossfit_tail_fidelity_failure"),
        ("BLOCKED", "inner_selection_instability", "BLOCKED", "inner_selection_instability"),
        ("BLOCKED", "policy_stability_failure", "BLOCKED", "policy_stability_failure"),
        ("BLOCKED", "full_objective_oof_policy_failure", "BLOCKED", "full_objective_oof_policy_failure"),
    ],
)
def test_new_classification(status, locus, expected_status, expected_locus):
    result = stagea._new_classification({"scientific_status": status, "primary_failure_locus": locus})
    assert result["scientific_status"] == expected_status
    assert result["primary_failure_locus"] == expected_locus


def test_validate_worker_accepts_blocked():
    stagea.validate_worker_evidence(worker_payload())


def test_validate_worker_accepts_ready_with_recommendation():
    stagea.validate_worker_evidence(worker_payload("READY"))


def test_validate_worker_rejects_wrong_counts():
    value = worker_payload()
    value["execution_counts"]["direction_fit_count"] = 107
    value["worker_result_sha256"] = stagea.sha256_bytes(
        stagea.stable_json_bytes({k: v for k, v in value.items() if k != "worker_result_sha256"})
    )
    with pytest.raises(stagea.StageAError, match="counts"):
        stagea.validate_worker_evidence(value)


def test_validate_worker_rejects_blocked_recommendation():
    value = worker_payload()
    value["train_only_recommendation"] = {"bad": True}
    value["worker_result_sha256"] = stagea.sha256_bytes(
        stagea.stable_json_bytes({k: v for k, v in value.items() if k != "worker_result_sha256"})
    )
    with pytest.raises(stagea.StageAError, match="retained"):
        stagea.validate_worker_evidence(value)


def test_make_worker_evidence_delegates_to_frozen_kernel(monkeypatch, tmp_path: Path):
    source = kernel_payload()
    seen = {}
    def fake_run(**kwargs):
        seen.update(kwargs)
        return copy.deepcopy(source)
    monkeypatch.setattr(stagea.kernel, "run_nested_oof", fake_run)
    monkeypatch.setattr(stagea.kernel, "validate_worker_payload", lambda payload: None)
    probe = probe_payload()
    result = stagea.make_worker_evidence(
        root=tmp_path,
        probe_payload=probe,
        repository_head="new-head",
        stageu_contract={"aligned_gate_lock": {}},
    )
    assert seen["repository_head"] == "new-head"
    assert result["scientific_kernel"]["kernel_scientific_result_sha256"] == "d" * 64
    assert result["preregistration_contract"]["stagex_resume1_stage_reexecuted"] is False
    assert result["preregistration_contract"]["frozen_scientific_kernel_reused"] is True
    assert seen["probe_payload"]["process_disposable"] is True
    assert seen["probe_payload"]["cuda_initialization_allowed_in_this_process"] is True
    assert result["worker_result_sha256"]


def test_make_worker_evidence_ready_preserves_train_only_recommendation(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(stagea.kernel, "run_nested_oof", lambda **kwargs: kernel_payload("READY", "tail_robust_nested_selection_procedure_ready"))
    monkeypatch.setattr(stagea.kernel, "validate_worker_payload", lambda payload: None)
    result = stagea.make_worker_evidence(
        root=tmp_path,
        probe_payload=probe_payload(),
        repository_head="head",
        stageu_contract={},
    )
    assert result["scientific_status"] == "READY"
    assert result["train_only_recommendation"] == {"policy": 1}


def test_build_summary_references_durable_files(tmp_path: Path):
    probe_path = tmp_path / "probe.json"
    worker_path = tmp_path / "worker.json"
    probe_path.write_bytes(stagea.stable_json_bytes(probe_payload()) + b"\n")
    worker_path.write_bytes(stagea.stable_json_bytes(worker_payload()) + b"\n")
    result = stagea.build_summary(
        repository={"head": "implementation"},
        probe_path=probe_path,
        worker_path=worker_path,
    )
    protocol = result["durable_evidence_protocol"]
    assert protocol["worker_result_persisted_before_controller_decoration"] is True
    assert protocol["temporary_worker_payload_used"] is False
    assert result["process_topology"]["processes_distinct"] is True


def test_build_summary_rejects_same_pid(tmp_path: Path):
    probe_path = tmp_path / "probe.json"
    worker_path = tmp_path / "worker.json"
    probe_path.write_bytes(stagea.stable_json_bytes(probe_payload(5)) + b"\n")
    worker_path.write_bytes(stagea.stable_json_bytes(worker_payload(pid=5)) + b"\n")
    with pytest.raises(stagea.StageAError, match="distinct"):
        stagea.build_summary(repository={}, probe_path=probe_path, worker_path=worker_path)


def test_validate_summary_accepts_valid_summary(tmp_path: Path):
    probe_path = tmp_path / "probe.json"
    worker_path = tmp_path / "worker.json"
    probe_path.write_bytes(stagea.stable_json_bytes(probe_payload()) + b"\n")
    worker_path.write_bytes(stagea.stable_json_bytes(worker_payload()) + b"\n")
    result = stagea.build_summary(repository={}, probe_path=probe_path, worker_path=worker_path)
    stagea.validate_summary(result)


def test_blocked_report_before_worker_does_not_authorize_reexecution(tmp_path: Path):
    result = stagea.blocked_report(
        repository=None,
        error=RuntimeError("x"),
        probe_path=tmp_path / "probe",
        worker_path=tmp_path / "worker",
    )
    assert result["durable_worker_result_preserved"] is False
    assert result["science_reexecution_authorized"] is False
    assert result["rerun_authorized"] is False


def test_blocked_report_after_worker_preserves_reference(tmp_path: Path):
    probe_path = tmp_path / "probe"
    worker_path = tmp_path / "worker"
    probe_path.write_bytes(stagea.stable_json_bytes(probe_payload()) + b"\n")
    worker_path.write_bytes(stagea.stable_json_bytes(worker_payload()) + b"\n")
    result = stagea.blocked_report(
        repository={"head": "x"},
        error=RuntimeError("decorate"),
        probe_path=probe_path,
        worker_path=worker_path,
    )
    assert result["durable_worker_result_preserved"] is True
    assert result["durable_worker_evidence"]["worker_result_sha256"]
    assert "WITHOUT_SCIENCE_REEXECUTION" in result["required_next_path"]


def test_false_boundaries_are_unique():
    assert len(stagea.FALSE_BOUNDARIES) == len(set(stagea.FALSE_BOUNDARIES))


def test_scientific_sources_are_frozen():
    assert set(stagea.FROZEN_SCIENCE_SOURCE_SHA256) == {
        "ccda_phase3/phase314b_r258_stagex_tail_robust_nested_oof.py",
        "ccda_phase3/phase314b_r258_stagex_resume1_import_crossfit_recovery.py",
    }


def test_worker_script_bootstraps_before_project_import():
    path = Path(__file__).resolve().parents[1] / "scripts/phase3_14b_r259_stagea_worker.py"
    text = path.read_text(encoding="utf-8")
    assert text.index("sys.path.insert") < text.index("from ccda_phase3")
    assert "atomic_write_once" in text


def test_controller_script_bootstraps_before_project_import():
    path = Path(__file__).resolve().parents[1] / "scripts/phase3_14b_r259_stagea_execute.py"
    text = path.read_text(encoding="utf-8")
    assert text.index("sys.path.insert") < text.index("from ccda_phase3")


def test_controller_has_no_temporary_directory():
    path = Path(__file__).resolve().parents[1] / "scripts/phase3_14b_r259_stagea_execute.py"
    text = path.read_text(encoding="utf-8")
    assert "TemporaryDirectory" not in text
    assert "--finalize-existing-evidence" in text


def test_controller_orders_probe_before_science():
    path = Path(__file__).resolve().parents[1] / "scripts/phase3_14b_r259_stagea_execute.py"
    text = path.read_text(encoding="utf-8")
    assert text.index('"environment-probe"') < text.index('"science"')


def test_stage_does_not_define_selection_holdout_path():
    source = Path(stagea.__file__).read_text(encoding="utf-8")
    assert "holdout_target" not in source
    assert "frozen_probe_accessed" in stagea.FALSE_BOUNDARIES
