from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from ccda_phase3 import phase314b_r258_stagec_resume1_commit_recovery as resume


def valid_test_gate():
    manifest = {
        "tests/test_{:02d}.py".format(index):
            "{:064x}".format(index + 1)
        for index in range(50)
    }
    return {
        "phase":
            "Phase3.14b-r2.5.8 Stage C",
        "verdict": "PASS",
        "base_evidence_commit":
            resume.REMOTE_BASE_COMMIT,
        "base_passed": 959,
        "stagec_new_passed": 64,
        "passed_test_count": 1023,
        "test_file_count": 50,
        "test_manifest_sha256":
            manifest,
    }


def valid_blocked():
    return {
        "phase":
            "Phase3.14b-r2.5.8 Stage C",
        "verdict": "BLOCKED",
        "scientific_status": "BLOCKED",
        "root_cause": (
            "phase314b_r258_stagec_"
            "execution_failed_before_completion"
        ),
        "exit_code": 141,
        "base_evidence_commit":
            resume.REMOTE_BASE_COMMIT,
        "new_model_candidate_trained":
            False,
        "direct_x0_tensor_optimization_run":
            False,
        "balanced_objective_calibration_run":
            False,
        "control_replay_exact":
            False,
        "selected_configuration":
            None,
        "train_only_recommendation":
            None,
        "frozen_probe_accessed":
            False,
        "reverse_sampling_run":
            False,
        "full_stageb_repaired_model_trained":
            False,
        "formal_pilot_run":
            False,
        "checkpoint_saved":
            False,
        "weights_persisted":
            False,
        "prediction_tensor_persisted":
            False,
        "oracle_tensor_persisted":
            False,
        "candidate_tensor_persisted":
            False,
        "npz_saved":
            False,
        "cache_saved":
            False,
        "formal_diffusion_training":
            False,
        "formal_reverse_sampling":
            False,
        "formal_idm_training":
            False,
        "action_diverse_data_collection":
            False,
        "candidate_execution":
            False,
        "deformable_ravens_executed":
            False,
        "phase4":
            False,
        "cps":
            False,
    }


def test_phase_constant():
    assert resume.PHASE == (
        "Phase3.14b-r2.5.8 Stage C Resume1"
    )


def test_remote_base_constant():
    assert resume.REMOTE_BASE_COMMIT == (
        "174d4428f835bb7fa76d9bdd497c9ff63452122f"
    )


def test_implementation_commit_constant():
    assert resume.IMPLEMENTATION_COMMIT == (
        "0692a5ba794e1b7477d56874f7d003ec887eafc1"
    )


def test_submodule_constant():
    assert resume.EXPECTED_SUBMODULE_COMMIT == (
        "633a88752445cf5d6776ed374fdbbdb35f93050c"
    )


def test_original_implementation_population():
    assert len(
        resume.IMPLEMENTATION_PATH_SHA256
    ) == 7


def test_original_sha_values_are_sha256():
    assert all(
        len(value) == 64
        for value
        in resume.IMPLEMENTATION_PATH_SHA256.values()
    )


def test_resume_implementation_population():
    assert len(
        resume.RESUME_IMPLEMENTATION_PATHS
    ) == 7


def test_resume_success_population():
    assert len(
        resume.RESUME_SUCCESS_PATHS
    ) == 5


def test_first_provenance_population():
    assert resume.FIRST_PROVENANCE_PATHS == (
        resume.FIRST_TEST_GATE,
        resume.FIRST_BLOCKED,
    )


def test_commit_messages_are_distinct():
    messages = {
        resume.IMPLEMENTATION_MESSAGE,
        resume.PROVENANCE_MESSAGE,
        resume.RESUME_IMPLEMENTATION_MESSAGE,
        resume.EVIDENCE_MESSAGE,
    }
    assert len(messages) == 4


def test_stable_json_deterministic():
    assert resume.stable_json_bytes(
        {"b": 2, "a": 1}
    ) == resume.stable_json_bytes(
        {"a": 1, "b": 2}
    )


def test_atomic_write_once(tmp_path):
    path = tmp_path / "value.json"
    resume.atomic_write_once(
        path,
        b"{}\n",
    )
    assert path.read_bytes() == b"{}\n"
    with pytest.raises(FileExistsError):
        resume.atomic_write_once(
            path,
            b"{}\n",
        )


def test_valid_test_gate():
    resume.validate_first_test_gate(
        valid_test_gate()
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("phase", "wrong"),
        ("verdict", "FAIL"),
        ("base_evidence_commit", "0" * 40),
        ("base_passed", 958),
        ("stagec_new_passed", 63),
        ("passed_test_count", 1022),
        ("test_file_count", 49),
    ],
)
def test_test_gate_rejects_changed_field(
    field,
    value,
):
    payload = valid_test_gate()
    payload[field] = value
    with pytest.raises(
        resume.CommitRecoveryError
    ):
        resume.validate_first_test_gate(
            payload
        )


def test_test_gate_rejects_manifest_size():
    payload = valid_test_gate()
    payload[
        "test_manifest_sha256"
    ].pop(next(iter(
        payload["test_manifest_sha256"]
    )))
    with pytest.raises(
        resume.CommitRecoveryError
    ):
        resume.validate_first_test_gate(
            payload
        )


def test_valid_blocked():
    resume.validate_first_blocked(
        valid_blocked()
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("phase", "wrong"),
        ("verdict", "PASS"),
        ("scientific_status", "READY"),
        ("root_cause", "wrong"),
        ("exit_code", 1),
        ("base_evidence_commit", "0" * 40),
        ("selected_configuration", {}),
        ("train_only_recommendation", {}),
    ],
)
def test_blocked_rejects_changed_field(
    field,
    value,
):
    payload = valid_blocked()
    payload[field] = value
    with pytest.raises(
        resume.CommitRecoveryError
    ):
        resume.validate_first_blocked(
            payload
        )


def test_blocked_rejects_scientific_run():
    payload = valid_blocked()
    payload[
        "balanced_objective_calibration_run"
    ] = True
    with pytest.raises(
        resume.CommitRecoveryError
    ):
        resume.validate_first_blocked(
            payload
        )


def test_validate_first_attempt_reports(
    tmp_path,
):
    test_path = (
        tmp_path
        / resume.FIRST_TEST_GATE
    )
    blocked_path = (
        tmp_path
        / resume.FIRST_BLOCKED
    )
    test_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    test_path.write_text(
        json.dumps(valid_test_gate()),
        encoding="utf-8",
    )
    blocked_path.write_text(
        json.dumps(valid_blocked()),
        encoding="utf-8",
    )
    result = (
        resume.validate_first_attempt_reports(
            tmp_path
        )
    )
    assert result[
        "commit_command_status"
    ] == 141
    assert result[
        "commit_transaction_completed"
    ] is True
    assert result[
        "scientific_calibration_started"
    ] is False
    assert len(
        result["test_gate_sha256"]
    ) == 64
    assert len(
        result["blocked_sha256"]
    ) == 64


def test_validate_first_attempt_missing_file(
    tmp_path,
):
    with pytest.raises(
        FileNotFoundError
    ):
        resume.validate_first_attempt_reports(
            tmp_path
        )


def test_status_paths_empty_git_repo(
    tmp_path,
):
    import subprocess

    subprocess.run(
        ["git", "init", "-q"],
        cwd=str(tmp_path),
        check=True,
    )
    assert resume.status_paths(
        tmp_path
    ) == ()


def test_status_paths_untracked(
    tmp_path,
):
    import subprocess

    subprocess.run(
        ["git", "init", "-q"],
        cwd=str(tmp_path),
        check=True,
    )
    (tmp_path / "b.txt").write_text(
        "b",
        encoding="utf-8",
    )
    (tmp_path / "a.txt").write_text(
        "a",
        encoding="utf-8",
    )
    assert resume.status_paths(
        tmp_path
    ) == (
        "a.txt",
        "b.txt",
    )


def test_commit_helpers(
    tmp_path,
):
    import subprocess

    subprocess.run(
        ["git", "init", "-q"],
        cwd=str(tmp_path),
        check=True,
    )
    subprocess.run(
        [
            "git",
            "config",
            "user.email",
            "test@example.invalid",
        ],
        cwd=str(tmp_path),
        check=True,
    )
    subprocess.run(
        [
            "git",
            "config",
            "user.name",
            "Test",
        ],
        cwd=str(tmp_path),
        check=True,
    )
    (tmp_path / "a.txt").write_text(
        "a",
        encoding="utf-8",
    )
    subprocess.run(
        ["git", "add", "a.txt"],
        cwd=str(tmp_path),
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-q", "-m", "first"],
        cwd=str(tmp_path),
        check=True,
    )
    first = resume.git_output(
        tmp_path,
        "rev-parse",
        "HEAD",
    )
    (tmp_path / "b.txt").write_text(
        "b",
        encoding="utf-8",
    )
    subprocess.run(
        ["git", "add", "b.txt"],
        cwd=str(tmp_path),
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-q", "-m", "second"],
        cwd=str(tmp_path),
        check=True,
    )
    second = resume.git_output(
        tmp_path,
        "rev-parse",
        "HEAD",
    )
    assert resume.commit_subject(
        tmp_path,
        second,
    ) == "second"
    assert resume.commit_parents(
        tmp_path,
        second,
    ) == (first,)
    assert resume.commit_paths(
        tmp_path,
        second,
    ) == ("b.txt",)


def test_assert_commit_shape_accepts(
    tmp_path,
):
    import subprocess

    subprocess.run(
        ["git", "init", "-q"],
        cwd=str(tmp_path),
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "a@b.invalid"],
        cwd=str(tmp_path),
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "T"],
        cwd=str(tmp_path),
        check=True,
    )
    subprocess.run(
        [
            "git",
            "commit",
            "--allow-empty",
            "-q",
            "-m",
            "base",
        ],
        cwd=str(tmp_path),
        check=True,
    )
    parent = resume.git_output(
        tmp_path,
        "rev-parse",
        "HEAD",
    )
    (tmp_path / "x").write_text(
        "x",
        encoding="utf-8",
    )
    subprocess.run(
        ["git", "add", "x"],
        cwd=str(tmp_path),
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-q", "-m", "child"],
        cwd=str(tmp_path),
        check=True,
    )
    child = resume.git_output(
        tmp_path,
        "rev-parse",
        "HEAD",
    )
    resume.assert_commit_shape(
        tmp_path,
        commit=child,
        parent=parent,
        subject="child",
        paths=("x",),
    )


def test_assert_commit_shape_rejects_subject(
    tmp_path,
):
    import subprocess

    subprocess.run(
        ["git", "init", "-q"],
        cwd=str(tmp_path),
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "a@b.invalid"],
        cwd=str(tmp_path),
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "T"],
        cwd=str(tmp_path),
        check=True,
    )
    subprocess.run(
        [
            "git",
            "commit",
            "--allow-empty",
            "-q",
            "-m",
            "base",
        ],
        cwd=str(tmp_path),
        check=True,
    )
    parent = resume.git_output(
        tmp_path,
        "rev-parse",
        "HEAD",
    )
    (tmp_path / "x").write_text(
        "x",
        encoding="utf-8",
    )
    subprocess.run(
        ["git", "add", "x"],
        cwd=str(tmp_path),
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-q", "-m", "child"],
        cwd=str(tmp_path),
        check=True,
    )
    child = resume.git_output(
        tmp_path,
        "rev-parse",
        "HEAD",
    )
    with pytest.raises(
        resume.CommitRecoveryError
    ):
        resume.assert_commit_shape(
            tmp_path,
            commit=child,
            parent=parent,
            subject="wrong",
            paths=("x",),
        )


def test_compare_workers_delegates(
    monkeypatch,
):
    expected = {"exact": True}

    def fake(left, right):
        assert left == {"a": 1}
        assert right == {"a": 1}
        return expected

    monkeypatch.setattr(
        resume.stagec,
        "compare_worker_results",
        fake,
    )
    assert resume.compare_worker_results(
        {"a": 1},
        {"a": 1},
    ) is expected


def test_run_calibration_delegates(
    monkeypatch,
    tmp_path,
):
    expected = {
        "phase":
            "Phase3.14b-r2.5.8 Stage C",
    }

    def fake(*, root, environment):
        assert root == tmp_path.resolve()
        assert environment == {"x": 1}
        return expected

    monkeypatch.setattr(
        resume.stagec,
        "run_calibration",
        fake,
    )
    assert resume.run_calibration(
        root=tmp_path,
        environment={"x": 1},
    ) is expected


def test_run_calibration_rejects_phase(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        resume.stagec,
        "run_calibration",
        lambda **kwargs: {
            "phase": "wrong"
        },
    )
    with pytest.raises(
        resume.CommitRecoveryError
    ):
        resume.run_calibration(
            root=tmp_path,
            environment={},
        )
