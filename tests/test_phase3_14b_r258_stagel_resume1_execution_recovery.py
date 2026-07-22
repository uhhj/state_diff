from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

from ccda_phase3 import phase314b_r258_stagel_resume1_execution_recovery as resume1


def blocked_payload(tmp_path: Path) -> dict:
    return {
        "phase": "Phase3.14b-r2.5.8 Stage L Predicate Assembly Audit",
        "schema": "phase314b_r258_stagel_consolidated_blocked_v1_predicate_assembly",
        "execution_verdict": "BLOCKED",
        "scientific_status": "BLOCKED",
        "root_cause": "phase314b_r258_stagel_consolidated_execution_failed",
        "required_next_path": "INSPECT_THE_SINGLE_CONSOLIDATED_ENTRYPOINT_WITHOUT_ADDING_A_WRAPPER",
        "error_type": "ExecutionError",
        "error": "deterministic environment changed: {}",
        "selected_configuration": None,
        "train_only_recommendation": None,
        **{key: False for key in resume1.FALSE_BOUNDARIES},
    }


def test_phase_identity():
    assert resume1.PHASE == "Phase3.14b-r2.5.8 Stage L Resume1"


def test_commit_constants():
    assert resume1.STAGEL_IMPLEMENTATION_COMMIT == "cd7fe38b1013b13ca84086bedd64dfa5eaacff47"
    assert resume1.STAGEK_EVIDENCE_COMMIT == "70f55e62aea557194737528f942bfde697f984b5"


def test_blocked_sha_constant():
    assert resume1.EXPECTED_STAGEL_BLOCKED_SHA256 == "eaa9aedcc801fbf96175a68e0e0d958b6481a91e51ea2e5bbe610b212dc3f088"


def test_resume1_paths_are_add_only():
    assert len(resume1.RESUME1_IMPLEMENTATION_PATHS) == 3
    assert all(status == "A" for status, _ in resume1.RESUME1_IMPLEMENTATION_PATHS)


def test_stage_l_paths_are_frozen():
    assert resume1.STAGEL_IMPLEMENTATION_PATHS == (
        ("A", "ccda_phase3/phase314b_r258_stagel_predicate_assembly_audit.py"),
        ("M", "scripts/phase3_14b_r258_stagef_consolidated_e2e.py"),
        ("A", "tests/test_phase3_14b_r258_stagel_predicate_assembly_audit.py"),
    )


def test_prior_counts_exact():
    assert resume1.PRIOR_TEST_COUNTS == (28, 29, 81, 24, 21, 18, 15, 15, 138)


def test_environment_population():
    assert len(resume1.EXPECTED_ENV) == 9
    assert resume1.EXPECTED_ENV["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_validate_environment_accepts_exact_mapping():
    assert resume1.validate_environment(dict(resume1.EXPECTED_ENV)) == dict(resume1.EXPECTED_ENV)


def test_validate_environment_rejects_missing_value():
    value = dict(resume1.EXPECTED_ENV)
    value.pop("PYTHONHASHSEED")
    with pytest.raises(resume1.StageLResume1Error, match="deterministic environment changed"):
        resume1.validate_environment(value)


def test_stable_json_is_sorted_and_terminal_newline():
    assert resume1.stable_json_bytes({"b": 1, "a": 2}) == b'{\n  "a": 2,\n  "b": 1\n}\n'


def test_sha256_bytes():
    assert resume1.sha256_bytes(b"abc") == hashlib.sha256(b"abc").hexdigest()


def test_write_once(tmp_path: Path):
    path = tmp_path / "x.json"
    resume1.write_once(path, b"{}\n")
    assert path.read_bytes() == b"{}\n"
    with pytest.raises(resume1.StageLResume1Error, match="overwrite"):
        resume1.write_once(path, b"again")


def test_require_equal():
    resume1.require_equal("x", 1, 1)
    with pytest.raises(resume1.StageLResume1Error, match="x changed"):
        resume1.require_equal("x", 1, 2)


def test_blocked_report_validator_rejects_wrong_sha(tmp_path: Path):
    path = tmp_path / "blocked.json"
    path.write_text(json.dumps(blocked_payload(tmp_path)), encoding="utf-8")
    with pytest.raises(resume1.StageLResume1Error, match="SHA256"):
        resume1.validate_original_blocked_report(path)


def test_blocked_report_validator_accepts_bound_payload(monkeypatch, tmp_path: Path):
    path = tmp_path / "blocked.json"
    path.write_text(json.dumps(blocked_payload(tmp_path)), encoding="utf-8")
    monkeypatch.setattr(resume1, "EXPECTED_STAGEL_BLOCKED_SHA256", resume1.sha256_path(path))
    result = resume1.validate_original_blocked_report(path)
    assert result["root_cause"] == "phase314b_r258_stagel_consolidated_execution_failed"


def test_blocked_report_rejects_scientific_result(monkeypatch, tmp_path: Path):
    value = blocked_payload(tmp_path)
    value["scientific_result"] = {}
    path = tmp_path / "blocked.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    monkeypatch.setattr(resume1, "EXPECTED_STAGEL_BLOCKED_SHA256", resume1.sha256_path(path))
    with pytest.raises(resume1.StageLResume1Error, match="scientific result"):
        resume1.validate_original_blocked_report(path)


def test_false_boundary_population_contains_callback_event():
    assert "callback_event_persisted" in resume1.FALSE_BOUNDARIES


def test_parse_args_default():
    assert resume1.parse_args([]).root == "/data/state_diff2"


def test_blocked_payload_is_fail_closed():
    value = resume1.blocked_payload(RuntimeError("x"), {"head": "y"})
    assert value["execution_verdict"] == "BLOCKED"
    assert value["scientific_status"] == "BLOCKED"
    assert value["selected_configuration"] is None
    assert value["existing_test_gates_rerun"] is False


def test_validate_stage_l_summary_accepts_minimal_valid():
    value = {
        "execution_verdict": "PASS",
        "scientific_status": "BLOCKED",
        "root_cause": "r",
        "required_next_path": "n",
        "selected_configuration": None,
        "train_only_recommendation": None,
        "execution": {
            "single_run_result_sha256": "a" * 64,
            "python_process_count": 1,
            "child_python_process_count": 0,
        },
        "scientific_result": {
            "primary_failure_locus": "p",
            "predicate_assembly_audit_completed": True,
            "predicate_assembly_audit": {
                "callback_off_on_pair_count": 132,
                "all_callback_results_bit_exact": True,
                "callback_events_persisted": False,
            },
            **{key: False for key in resume1.FALSE_BOUNDARIES},
        },
        "boundaries": {key: False for key in resume1.FALSE_BOUNDARIES},
    }
    result = resume1.validate_stage_l_summary(value)
    assert result["single_run_result_sha256"] == "a" * 64


def test_validate_stage_l_summary_rejects_child_python():
    value = {
        "execution_verdict": "PASS",
        "scientific_status": "BLOCKED",
        "selected_configuration": None,
        "train_only_recommendation": None,
        "execution": {
            "single_run_result_sha256": "a" * 64,
            "python_process_count": 1,
            "child_python_process_count": 1,
        },
    }
    with pytest.raises(resume1.StageLResume1Error):
        resume1.validate_stage_l_summary(value)


def test_commit_name_status_parses_nul_records(tmp_path: Path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    (tmp_path / "base.txt").write_text("base", encoding="utf-8")
    subprocess.run(["git", "add", "base.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "base"], cwd=tmp_path, check=True)
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    subprocess.run(["git", "add", "a.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "a"], cwd=tmp_path, check=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True).strip()
    assert resume1.commit_name_status(tmp_path, commit) == (("A", "a.txt"),)


def test_require_ancestor(tmp_path: Path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    (tmp_path / "a").write_text("1", encoding="utf-8")
    subprocess.run(["git", "add", "a"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "one"], cwd=tmp_path, check=True)
    first = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True).strip()
    (tmp_path / "a").write_text("2", encoding="utf-8")
    subprocess.run(["git", "commit", "-qam", "two"], cwd=tmp_path, check=True)
    second = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True).strip()
    resume1.require_ancestor(tmp_path, first, second)


def test_execute_script_is_thin_entrypoint():
    root = Path(__file__).resolve().parents[1]
    text = (root / "scripts/phase3_14b_r258_stagel_resume1_execute.py").read_text(encoding="utf-8")
    assert "phase314b_r258_stagel_resume1_execution_recovery import main" in text
    assert "run_calibration" not in text


def test_resume1_module_does_not_import_stage_l_science_directly():
    source = Path(resume1.__file__).read_text(encoding="utf-8")
    assert "phase314b_r258_stagel_predicate_assembly_audit import" not in source
    assert "runner.run_once(root, underlying_repository)" in source
