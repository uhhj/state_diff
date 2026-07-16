from __future__ import annotations

import json
from pathlib import Path

import pytest

from ccda_phase3 import phase314b_r258_stagea_resume1_output_lifecycle as resume1


def blocked_payload():
    return {
        "verdict": "BLOCKED",
        "scientific_status": "BLOCKED",
        "root_cause":
            "phase314b_r258_stagea_execution_failed_before_completion",
        "failed_command":
            '"${PYTHON_BIN}" scripts/phase3_14b_r258_stagea_worker.py '
            '--mode environment-probe',
        "new_hyperparameter_candidate_run": False,
        "new_objective_variant_run": False,
        "control_replay_exact": False,
        "frozen_probe_accessed": False,
        "reverse_sampling_run": False,
        "full_stageb_repaired_model_trained": False,
        "formal_pilot_run": False,
        "candidate_execution": False,
        "deformable_ravens_executed": False,
        "phase4": False,
        "cps": False,
    }


def test_base_constants_are_frozen():
    assert resume1.BASE_EVIDENCE_COMMIT == (
        "f0a5bec1f89f625e74150e55e2da884413d72eb9"
    )
    assert resume1.EXPECTED_FAILED_BLOCKED_SHA256 == (
        "832626d9cf6d146fe43cd76c7efada14b86658369ef1b49171cbe2d78a672768"
    )


def test_original_file_population_is_seven():
    assert len(resume1.ORIGINAL_PORTABLE_FILES) == 7
    assert (
        "scripts/phase3_14b_r258_stagea_run.sh"
        in resume1.ORIGINAL_PORTABLE_FILES
    )


def test_original_run_sha_is_frozen():
    assert resume1.ORIGINAL_PORTABLE_FILES[
        "scripts/phase3_14b_r258_stagea_run.sh"
    ] == (
        "5bb70651f4cd58dda89657eb364a936a1507b532f16105ff34ab8789f5bb3e9f"
    )


def test_make_absent_child_returns_nonexistent_path(tmp_path):
    output = resume1.make_absent_child(
        tmp_path,
        "environment.json",
    )
    assert output == tmp_path / "environment.json"
    assert not output.exists()


def test_make_absent_child_rejects_existing_path(tmp_path):
    path = tmp_path / "environment.json"
    path.write_text("x", encoding="utf-8")
    with pytest.raises(FileExistsError):
        resume1.make_absent_child(
            tmp_path,
            "environment.json",
        )


def test_make_absent_child_rejects_nested_filename(tmp_path):
    with pytest.raises(resume1.Resume1LifecycleError):
        resume1.make_absent_child(
            tmp_path,
            "nested/environment.json",
        )


def test_atomic_write_once_accepts_absent_child(tmp_path):
    output = resume1.make_absent_child(
        tmp_path,
        "environment.json",
    )
    resume1.atomic_write_once(
        output,
        b'{"pass": true}\n',
    )
    assert json.loads(
        output.read_text(encoding="utf-8")
    )["pass"] is True


def test_atomic_write_once_rejects_existing_file(tmp_path):
    output = tmp_path / "environment.json"
    output.write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError):
        resume1.atomic_write_once(
            output,
            b"{}\n",
        )


def test_validate_failed_blocked_payload_passes():
    resume1.validate_failed_blocked_payload(
        blocked_payload()
    )


def test_validate_failed_blocked_payload_rejects_candidate_run():
    payload = blocked_payload()
    payload["new_hyperparameter_candidate_run"] = True
    with pytest.raises(resume1.Resume1LifecycleError):
        resume1.validate_failed_blocked_payload(
            payload
        )


def test_correction_record_does_not_change_science():
    record = resume1.correction_record(
        {"failed_blocked_sha256": "a" * 64}
    )
    assert record["namespace"] == (
        "phase3_14b_r258_stagea_resume1"
    )
    assert record["scientific_contract_changed"] is False
    assert record["candidate_population_changed"] is False
    assert record["selection_contract_changed"] is False


def test_stable_json_is_deterministic():
    left = resume1.stable_json_bytes(
        {"b": 2, "a": 1}
    )
    right = resume1.stable_json_bytes(
        {"a": 1, "b": 2}
    )
    assert left == right
