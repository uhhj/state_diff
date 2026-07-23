from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ccda_phase3 import phase314b_r259_stageb_outer_crossfit_tail_failure_audit as stageb
from ccda_phase3 import phase314b_r259_stageb_resume1_postcommit_evidence_audit_recovery as resume1


def original_blocked_payload():
    payload = {
        "phase": stageb.PHASE,
        "schema": stageb.BLOCKED_SCHEMA,
        "execution_verdict": "BLOCKED",
        "scientific_status": "BLOCKED",
        "root_cause": "phase314b_r259_stageb_evidence_audit_execution_failed",
        "required_next_path": (
            "RESTORE_R259_STAGEB_EVIDENCE_ONLY_OUTER_CROSSFIT_AUDIT_WITHOUT_"
            "SCIENCE_REEXECUTION"
        ),
        "primary_failure_locus": "evidence_audit_execution_contract",
        "repository": None,
        "error_type": "StageBError",
        "error_message": "Stage-B starting HEAD changed",
        "selected_configuration": None,
        "train_only_recommendation": None,
        "selection_holdout_evaluation_count_added": 0,
        "cumulative_selection_holdout_evaluation_count": 1,
        "rerun_authorized": False,
    }
    payload.update({key: False for key in resume1.FALSE_BOUNDARIES})
    return payload


def valid_result(commit="1" * 40):
    payload = {
        "phase": resume1.PHASE,
        "schema": resume1.SCHEMA,
        "execution_verdict": "PASS",
        "scientific_status": "BLOCKED",
        "root_cause": (
            "phase314b_r259_stageb_aggregate_evidence_localizes_t10_"
            "acceptance_coverage_failure_but_outer_test_fold_attribution_is_unavailable"
        ),
        "required_next_path": (
            "PREREGISTER_R259_STAGEC_OBJECTIVE_TRAIN_ONLY_FOLD_RESOLVED_"
            "TAIL_ATTRIBUTION_WITH_DURABLE_AGGREGATE_EVIDENCE"
        ),
        "primary_failure_locus": (
            "t10_acceptance_coverage_with_fold_resolved_attribution_unavailable"
        ),
        "repository": {"head": commit},
        "recovery_contract": {
            "evidence_only": True,
            "original_stageb_failure_before_stagea_evidence_read": True,
            "original_stageb_scientific_audit_completed": False,
            "stagea_science_replayed": False,
            "stagea_policy_or_threshold_changed": False,
            "dynamic_resume1_implementation_commit_bound": True,
            "repository_validator_invocation_count": 1,
            "original_stageb_controller_reexecuted": False,
            "new_fit_count": 0,
            "new_candidate_generation_count": 0,
            "new_risk_fit_count": 0,
            "new_internal_scale_attempt_count": 0,
            "new_policy_evaluation_count": 0,
        },
        "selected_configuration": None,
        "train_only_recommendation": None,
        "selection_holdout_evaluation_count_added": 0,
        "cumulative_selection_holdout_evaluation_count": 1,
        "rerun_authorized": False,
    }
    payload.update({key: False for key in resume1.FALSE_BOUNDARIES})
    payload["audit_sha256"] = resume1.sha256_bytes(resume1.stable_json_bytes(payload))
    return payload


def test_phase_name():
    assert resume1.PHASE.endswith("Stage B Resume1")


def test_schema_name():
    assert resume1.SCHEMA.endswith("_v1")


def test_blocked_schema_name():
    assert resume1.BLOCKED_SCHEMA.endswith("_blocked_v1")


def test_base_chain_constants():
    assert resume1.BASE_STAGEB_IMPLEMENTATION_COMMIT.startswith("aa1bfb4a")
    assert resume1.BASE_STAGEB_BLOCKED_EVIDENCE_COMMIT.startswith("2d956888")
    assert resume1.BASE_STAGEB_PARENT.startswith("3100a877")


def test_remote_and_submodule_constants():
    assert resume1.EXPECTED_REMOTE == "6758ea7ad800667a436b0243d3b1f6c63256d854"
    assert resume1.EXPECTED_SUBMODULE == "633a88752445cf5d6776ed374fdbbdb35f93050c"


def test_implementation_is_add_only_three_paths():
    assert len(resume1.IMPLEMENTATION_PATHS) == 3
    assert all(status == "A" for status, _ in resume1.IMPLEMENTATION_PATHS)


def test_original_stageb_source_population_bound():
    assert set(resume1.BASE_STAGEB_IMPLEMENTATION_SHA256) == {
        path for _, path in resume1.BASE_STAGEB_IMPLEMENTATION_PATHS
    }


def test_original_blocked_path_population():
    assert resume1.BASE_STAGEB_BLOCKED_PATHS == (("A", resume1.BASE_STAGEB_BLOCKED_REPORT),)


def test_success_and_blocked_paths_distinct():
    assert resume1.SUCCESS_REPORT != resume1.BLOCKED_REPORT


def test_stable_json_is_sorted():
    assert resume1.stable_json_bytes({"b": 1, "a": 2}).startswith(b'{\n  "a"')


def test_stable_json_rejects_nan():
    with pytest.raises(ValueError):
        resume1.stable_json_bytes({"x": float("nan")})


def test_sha256_bytes():
    assert resume1.sha256_bytes(b"abc") == hashlib.sha256(b"abc").hexdigest()


def test_sha256_file(tmp_path: Path):
    path = tmp_path / "x"
    path.write_bytes(b"abc")
    assert resume1.sha256_file(path) == hashlib.sha256(b"abc").hexdigest()


def test_load_json_accepts_mapping(tmp_path: Path):
    path = tmp_path / "x.json"
    path.write_text('{"a": 1}', encoding="utf-8")
    assert resume1.load_json(path) == {"a": 1}


def test_load_json_rejects_sequence(tmp_path: Path):
    path = tmp_path / "x.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(resume1.StageBResume1Error):
        resume1.load_json(path)


def test_write_once(tmp_path: Path):
    path = tmp_path / "x.json"
    resume1.write_once(path, b"{}")
    assert path.read_bytes() == b"{}"
    with pytest.raises(resume1.StageBResume1Error):
        resume1.write_once(path, b"again")


def test_commit_name_status_parsing(monkeypatch):
    monkeypatch.setattr(resume1, "_git", lambda *args: "A\tb\nM\ta")
    assert resume1.commit_name_status(Path("/tmp"), "x") == (("A", "b"), ("M", "a"))


def test_commit_name_status_rejects_bad_line(monkeypatch):
    monkeypatch.setattr(resume1, "_git", lambda *args: "bad")
    with pytest.raises(resume1.StageBResume1Error):
        resume1.commit_name_status(Path("/tmp"), "x")


def test_assert_clean_worktree_accepts_empty(monkeypatch):
    monkeypatch.setattr(resume1, "_git", lambda *args: "")
    resume1.assert_clean_worktree(Path("/tmp"), "x")


def test_assert_clean_worktree_rejects_dirty(monkeypatch):
    monkeypatch.setattr(resume1, "_git", lambda *args: "?? x")
    with pytest.raises(resume1.StageBResume1Error):
        resume1.assert_clean_worktree(Path("/tmp"), "x")


def test_validate_original_blocked_accepts_exact(monkeypatch, tmp_path: Path):
    path = tmp_path / "blocked.json"
    path.write_text(json.dumps(original_blocked_payload()), encoding="utf-8")
    monkeypatch.setattr(resume1, "BASE_STAGEB_BLOCKED_REPORT_SHA256", resume1.sha256_file(path))
    assert resume1.validate_original_blocked_report(path)["error_message"] == "Stage-B starting HEAD changed"


@pytest.mark.parametrize(
    "field,value",
    [
        ("execution_verdict", "PASS"),
        ("scientific_status", "READY"),
        ("root_cause", "x"),
        ("required_next_path", "x"),
        ("primary_failure_locus", "x"),
        ("error_type", "KeyError"),
        ("error_message", "x"),
        ("selected_configuration", {}),
        ("train_only_recommendation", {}),
        ("selection_holdout_evaluation_count_added", 1),
        ("cumulative_selection_holdout_evaluation_count", 2),
        ("rerun_authorized", True),
    ],
)
def test_validate_original_blocked_rejects_changed_field(monkeypatch, tmp_path: Path, field, value):
    payload = original_blocked_payload()
    payload[field] = value
    path = tmp_path / "blocked.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(resume1, "BASE_STAGEB_BLOCKED_REPORT_SHA256", resume1.sha256_file(path))
    with pytest.raises(resume1.StageBResume1Error):
        resume1.validate_original_blocked_report(path)


def test_validate_original_blocked_rejects_repository(monkeypatch, tmp_path: Path):
    payload = original_blocked_payload()
    payload["repository"] = {"head": "x"}
    path = tmp_path / "blocked.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(resume1, "BASE_STAGEB_BLOCKED_REPORT_SHA256", resume1.sha256_file(path))
    with pytest.raises(resume1.StageBResume1Error):
        resume1.validate_original_blocked_report(path)


def test_validate_original_blocked_rejects_boundary(monkeypatch, tmp_path: Path):
    payload = original_blocked_payload()
    payload[resume1.FALSE_BOUNDARIES[0]] = True
    path = tmp_path / "blocked.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(resume1, "BASE_STAGEB_BLOCKED_REPORT_SHA256", resume1.sha256_file(path))
    with pytest.raises(resume1.StageBResume1Error):
        resume1.validate_original_blocked_report(path)


def test_validate_recovery_result_accepts_exact():
    commit = "1" * 40
    resume1.validate_recovery_result(valid_result(commit), commit)


@pytest.mark.parametrize(
    "field,value",
    [
        ("execution_verdict", "BLOCKED"),
        ("scientific_status", "READY"),
        ("root_cause", "x"),
        ("required_next_path", "x"),
        ("primary_failure_locus", "x"),
        ("selected_configuration", {}),
        ("train_only_recommendation", {}),
        ("selection_holdout_evaluation_count_added", 1),
        ("cumulative_selection_holdout_evaluation_count", 2),
        ("rerun_authorized", True),
    ],
)
def test_validate_recovery_result_rejects_changed_field(field, value):
    commit = "1" * 40
    payload = valid_result(commit)
    payload[field] = value
    payload["audit_sha256"] = resume1.sha256_bytes(
        resume1.stable_json_bytes({k: v for k, v in payload.items() if k != "audit_sha256"})
    )
    with pytest.raises(resume1.StageBResume1Error):
        resume1.validate_recovery_result(payload, commit)


def test_validate_recovery_result_rejects_wrong_commit():
    payload = valid_result("1" * 40)
    with pytest.raises(resume1.StageBResume1Error):
        resume1.validate_recovery_result(payload, "2" * 40)


def test_validate_recovery_result_rejects_science_count():
    commit = "1" * 40
    payload = valid_result(commit)
    payload["recovery_contract"]["new_fit_count"] = 1
    payload["audit_sha256"] = resume1.sha256_bytes(
        resume1.stable_json_bytes({k: v for k, v in payload.items() if k != "audit_sha256"})
    )
    with pytest.raises(resume1.StageBResume1Error):
        resume1.validate_recovery_result(payload, commit)


def test_validate_recovery_result_rejects_boundary():
    commit = "1" * 40
    payload = valid_result(commit)
    payload[resume1.FALSE_BOUNDARIES[0]] = True
    payload["audit_sha256"] = resume1.sha256_bytes(
        resume1.stable_json_bytes({k: v for k, v in payload.items() if k != "audit_sha256"})
    )
    with pytest.raises(resume1.StageBResume1Error):
        resume1.validate_recovery_result(payload, commit)


def test_validate_recovery_result_rejects_hash():
    commit = "1" * 40
    payload = valid_result(commit)
    payload["audit_sha256"] = "0" * 64
    with pytest.raises(resume1.StageBResume1Error):
        resume1.validate_recovery_result(payload, commit)


def test_blocked_report_contract():
    payload = resume1.blocked_report(None, RuntimeError("x"))
    assert payload["execution_verdict"] == "BLOCKED"
    assert payload["scientific_status"] == "BLOCKED"
    assert payload["repository"] is None
    assert payload["rerun_authorized"] is False


def test_blocked_report_preserves_repository():
    payload = resume1.blocked_report({"head": "x"}, RuntimeError("x"))
    assert payload["repository"] == {"head": "x"}


def test_run_recovery_uses_original_pure_audit_functions(monkeypatch, tmp_path: Path):
    commit = "1" * 40
    calls = []
    monkeypatch.setattr(
        resume1,
        "validate_repository",
        lambda root, implementation_commit: {
            "root": str(root),
            "head": implementation_commit,
        },
    )
    monkeypatch.setattr(resume1, "_validate_stagea_inputs", lambda root: ({}, {"worker": True}, {}))
    monkeypatch.setattr(stageb, "reconstruct_outer_procedure_gates", lambda worker: calls.append("gates") or {"single_failed_timestep": 10, "single_failed_gate": "acceptance"})
    monkeypatch.setattr(stageb, "audit_outer_selections", lambda worker: calls.append("selections") or {"fallback_folds": [1, 4]})
    monkeypatch.setattr(stageb, "audit_attribution_surface", lambda worker: calls.append("surface") or {"outer_test_metrics_by_fold_available": False})
    monkeypatch.setattr(
        stageb,
        "classify_audit",
        lambda gates, selections, surface: calls.append("classify") or {
            "scientific_status": "BLOCKED",
            "root_cause": (
                "phase314b_r259_stageb_aggregate_evidence_localizes_t10_"
                "acceptance_coverage_failure_but_outer_test_fold_attribution_is_unavailable"
            ),
            "required_next_path": (
                "PREREGISTER_R259_STAGEC_OBJECTIVE_TRAIN_ONLY_FOLD_RESOLVED_"
                "TAIL_ATTRIBUTION_WITH_DURABLE_AGGREGATE_EVIDENCE"
            ),
            "primary_failure_locus": (
                "t10_acceptance_coverage_with_fold_resolved_attribution_unavailable"
            ),
        },
    )
    result = resume1.run_recovery(tmp_path, commit)
    assert calls == ["gates", "selections", "surface", "classify"]
    assert result["repository"]["head"] == commit
    assert result["recovery_contract"]["new_fit_count"] == 0


def test_validate_repository_rejects_short_commit(tmp_path: Path):
    with pytest.raises(resume1.StageBResume1Error, match="implementation commit"):
        resume1.validate_repository(tmp_path, "short")


def test_execute_source_requires_dynamic_commit():
    path = Path(__file__).resolve().parents[1] / "scripts/phase3_14b_r259_stageb_resume1_execute.py"
    text = path.read_text(encoding="utf-8")
    assert 'parser.add_argument("--implementation-commit")' in text
    assert "run_recovery(root, args.implementation_commit)" in text


def test_execute_source_has_single_success_path_validator():
    path = Path(__file__).resolve().parents[1] / "scripts/phase3_14b_r259_stageb_resume1_execute.py"
    text = path.read_text(encoding="utf-8")
    assert text.count("run_recovery(root, args.implementation_commit)") == 1
    assert "validate_repository(root" not in text


def test_no_worker_file_in_resume1_paths():
    assert all("worker" not in path for _, path in resume1.IMPLEMENTATION_PATHS)


def test_false_boundaries_are_unique():
    assert len(resume1.FALSE_BOUNDARIES) == len(set(resume1.FALSE_BOUNDARIES))
