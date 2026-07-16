"""Phase3.14b-r2.5.8 Stage C Resume1 commit-outcome recovery.

The first Stage-C invocation passed the 50-file/1023-test gate and created the
implementation commit, but the interactive SSH/output channel caused
``git commit`` to return 141 (SIGPIPE).  The shell treated the nonzero process
status as a failed commit even though the Git transaction had completed.

This module freezes that provenance and separates command status from
transaction outcome.  The original Stage-C implementation, test-gate report,
and blocked report are never modified.  Resume1 uses a new write-once report
namespace and runs the scientifically unstarted balanced-objective calibration.

No scientific threshold, candidate, objective, split, environment contract, or
forbidden-stage boundary is changed.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from ccda_phase3 import phase314b_r258_stagec_balanced_geometry as stagec

PHASE = "Phase3.14b-r2.5.8 Stage C Resume1"
PHASE_ID = "phase314b_r258_stagec_resume1"

REMOTE_BASE_COMMIT = (
    "174d4428f835bb7fa76d9bdd497c9ff63452122f"
)
IMPLEMENTATION_COMMIT = (
    "0692a5ba794e1b7477d56874f7d003ec887eafc1"
)
IMPLEMENTATION_MESSAGE = (
    "Add Phase3.14b-r2.5.8 balanced segment objective calibration"
)
PROVENANCE_MESSAGE = (
    "Record blocked Phase3.14b-r2.5.8 Stage-C commit-outcome execution"
)
RESUME_IMPLEMENTATION_MESSAGE = (
    "Add Phase3.14b-r2.5.8 Stage-C Resume1 commit-outcome recovery"
)
EVIDENCE_MESSAGE = (
    "Record Phase3.14b-r2.5.8 Stage-C Resume1 evidence"
)
EXPECTED_SUBMODULE_COMMIT = (
    "633a88752445cf5d6776ed374fdbbdb35f93050c"
)

FIRST_TEST_GATE = (
    "reports/phase3_14b_r258_stagec_test_gate_summary.json"
)
FIRST_BLOCKED = (
    "reports/phase3_14b_r258_stagec_blocked_summary.json"
)
FIRST_PROVENANCE_PATHS = (
    FIRST_TEST_GATE,
    FIRST_BLOCKED,
)

IMPLEMENTATION_PATH_SHA256 = {
    "ccda_phase3/phase314b_r258_stagec_balanced_geometry.py":
        "65e3f41a1788e90dcd87de1bfd0f373d67fa34bc3c2b6f7fa595f5f2968b77c6",
    "scripts/phase3_14b_r258_stagec_worker.py":
        "bc5a44006270ee740fca30bedb93b6ed5f2a1f3b5ffdc434ec734070b711a8f8",
    "scripts/phase3_14b_r258_stagec_run_calibration.py":
        "f978d741e80c22f18143962fa79d9ecfddfd377e54803c49d71f31a5c10648b1",
    "scripts/phase3_14b_r258_stagec_test_gate.py":
        "ec317f107a9e102de0df5e2ae88b9b84e5c40d1c63695effc500eb77895a2be0",
    "scripts/phase3_14b_r258_stagec_blocked.py":
        "e783d652debca1e1a8960dc84ca527d2e1f65de9dd60375a792efdd88213dacd",
    "scripts/phase3_14b_r258_stagec_run.sh":
        "2a4e3737608618cd895ae804d73b444ce02ed8cc2111c75e2259b358c3f5a37c",
    "tests/test_phase3_14b_r258_stagec_balanced_geometry.py":
        "6b28cd921e48d56b38987b780b8890d9f934c98b6cff63ad0710d1142d8577b5",
}

RESUME_IMPLEMENTATION_PATHS = (
    "ccda_phase3/phase314b_r258_stagec_resume1_commit_recovery.py",
    "scripts/phase3_14b_r258_stagec_resume1_worker.py",
    "scripts/phase3_14b_r258_stagec_resume1_run_calibration.py",
    "scripts/phase3_14b_r258_stagec_resume1_test_gate.py",
    "scripts/phase3_14b_r258_stagec_resume1_blocked.py",
    "scripts/phase3_14b_r258_stagec_resume1_run.sh",
    "tests/test_phase3_14b_r258_stagec_resume1_commit_recovery.py",
)

RESUME_TEST_GATE = (
    "reports/phase3_14b_r258_stagec_resume1_test_gate_summary.json"
)
RESUME_CONTRACT = (
    "reports/phase3_14b_r258_stagec_resume1_contract.json"
)
RESUME_WORKER = (
    "reports/phase3_14b_r258_stagec_resume1_worker_evidence.json"
)
RESUME_SUMMARY = (
    "reports/phase3_14b_r258_stagec_resume1_summary.json"
)
RESUME_REPORT = (
    "reports/phase3_14b_r258_stagec_resume1_report.md"
)
RESUME_BLOCKED = (
    "reports/phase3_14b_r258_stagec_resume1_blocked_summary.json"
)
RESUME_SUCCESS_PATHS = (
    RESUME_TEST_GATE,
    RESUME_CONTRACT,
    RESUME_WORKER,
    RESUME_SUMMARY,
    RESUME_REPORT,
)


class CommitRecoveryError(RuntimeError):
    """Raised when Git outcome or first-attempt provenance is inconsistent."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return stagec.stable_json_bytes(payload)


def atomic_write_once(
    path: Path,
    payload: bytes,
    *,
    mode: int = 0o644,
) -> None:
    stagec.atomic_write_once(
        path,
        payload,
        mode=mode,
    )


def load_json(path: Path) -> Dict[str, Any]:
    return stagec.load_json(path)


def git_output(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=str(root),
        text=True,
    ).strip()


def commit_subject(root: Path, commit: str) -> str:
    return git_output(
        root,
        "show",
        "-s",
        "--format=%s",
        commit,
    )


def commit_parents(root: Path, commit: str) -> Tuple[str, ...]:
    value = git_output(
        root,
        "show",
        "-s",
        "--format=%P",
        commit,
    )
    return tuple(
        item
        for item in value.split()
        if item
    )


def commit_paths(root: Path, commit: str) -> Tuple[str, ...]:
    output = git_output(
        root,
        "diff-tree",
        "--no-commit-id",
        "--name-only",
        "-r",
        commit,
    )
    return tuple(
        sorted(
            line.strip()
            for line in output.splitlines()
            if line.strip()
        )
    )


def assert_commit_exists(root: Path, commit: str) -> None:
    completed = subprocess.run(
        ["git", "cat-file", "-e", "{}^{{commit}}".format(commit)],
        cwd=str(root),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if completed.returncode != 0:
        raise CommitRecoveryError(
            "required commit is missing: {}".format(commit)
        )


def assert_commit_shape(
    root: Path,
    *,
    commit: str,
    parent: str,
    subject: str,
    paths: Sequence[str],
) -> None:
    assert_commit_exists(root, commit)
    if commit_parents(root, commit) != (parent,):
        raise CommitRecoveryError(
            "commit parent changed: {}".format(commit)
        )
    if commit_subject(root, commit) != subject:
        raise CommitRecoveryError(
            "commit subject changed: {}".format(commit)
        )
    if commit_paths(root, commit) != tuple(sorted(paths)):
        raise CommitRecoveryError(
            "commit path population changed: {}".format(commit)
        )


def validate_implementation_commit(root: Path) -> Dict[str, Any]:
    repository_root = Path(root).resolve()
    assert_commit_shape(
        repository_root,
        commit=IMPLEMENTATION_COMMIT,
        parent=REMOTE_BASE_COMMIT,
        subject=IMPLEMENTATION_MESSAGE,
        paths=tuple(IMPLEMENTATION_PATH_SHA256),
    )
    file_sha = {}
    for relative, expected in IMPLEMENTATION_PATH_SHA256.items():
        path = repository_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        committed = subprocess.check_output(
            [
                "git",
                "show",
                "{}:{}".format(
                    IMPLEMENTATION_COMMIT,
                    relative,
                ),
            ],
            cwd=str(repository_root),
        )
        if path.read_bytes() != committed:
            raise CommitRecoveryError(
                "implementation worktree file differs: {}".format(relative)
            )
        observed = sha256_bytes(committed)
        if observed != expected:
            raise CommitRecoveryError(
                "implementation file SHA changed: {}".format(relative)
            )
        file_sha[relative] = observed

    submodule = repository_root / "external/deformable-ravens"
    if git_output(
        submodule,
        "rev-parse",
        "HEAD",
    ) != EXPECTED_SUBMODULE_COMMIT:
        raise CommitRecoveryError(
            "DeformableRavens commit changed"
        )
    if git_output(
        submodule,
        "status",
        "--porcelain",
        "--untracked-files=all",
    ):
        raise CommitRecoveryError(
            "DeformableRavens worktree is dirty"
        )
    return {
        "implementation_commit": IMPLEMENTATION_COMMIT,
        "implementation_parent": REMOTE_BASE_COMMIT,
        "implementation_message": IMPLEMENTATION_MESSAGE,
        "implementation_file_sha256": file_sha,
        "submodule_commit": EXPECTED_SUBMODULE_COMMIT,
    }


def validate_first_test_gate(payload: Mapping[str, Any]) -> None:
    if payload.get("phase") != "Phase3.14b-r2.5.8 Stage C":
        raise CommitRecoveryError(
            "first Stage-C test-gate phase changed"
        )
    if payload.get("verdict") != "PASS":
        raise CommitRecoveryError(
            "first Stage-C test gate is not PASS"
        )
    if payload.get("base_evidence_commit") != REMOTE_BASE_COMMIT:
        raise CommitRecoveryError(
            "first Stage-C test-gate base changed"
        )
    if int(payload.get("base_passed", -1)) != 959:
        raise CommitRecoveryError(
            "first Stage-C base pass count changed"
        )
    if int(payload.get("stagec_new_passed", -1)) != 64:
        raise CommitRecoveryError(
            "first Stage-C new pass count changed"
        )
    if int(payload.get("passed_test_count", -1)) != 1023:
        raise CommitRecoveryError(
            "first Stage-C total pass count changed"
        )
    if int(payload.get("test_file_count", -1)) != 50:
        raise CommitRecoveryError(
            "first Stage-C test-file count changed"
        )
    manifest = payload.get("test_manifest_sha256")
    if not isinstance(manifest, dict) or len(manifest) != 50:
        raise CommitRecoveryError(
            "first Stage-C test manifest changed"
        )


def validate_first_blocked(payload: Mapping[str, Any]) -> None:
    if payload.get("phase") != "Phase3.14b-r2.5.8 Stage C":
        raise CommitRecoveryError(
            "first Stage-C blocked phase changed"
        )
    if payload.get("verdict") != "BLOCKED":
        raise CommitRecoveryError(
            "first Stage-C blocked verdict changed"
        )
    if payload.get("scientific_status") != "BLOCKED":
        raise CommitRecoveryError(
            "first Stage-C scientific status changed"
        )
    if payload.get("root_cause") != (
        "phase314b_r258_stagec_execution_failed_before_completion"
    ):
        raise CommitRecoveryError(
            "first Stage-C blocked root cause changed"
        )
    if int(payload.get("exit_code", -1)) != 141:
        raise CommitRecoveryError(
            "first Stage-C blocked exit code is not SIGPIPE/141"
        )
    if payload.get("base_evidence_commit") != REMOTE_BASE_COMMIT:
        raise CommitRecoveryError(
            "first Stage-C blocked base changed"
        )
    required_false = (
        "new_model_candidate_trained",
        "direct_x0_tensor_optimization_run",
        "balanced_objective_calibration_run",
        "control_replay_exact",
        "frozen_probe_accessed",
        "reverse_sampling_run",
        "full_stageb_repaired_model_trained",
        "formal_pilot_run",
        "checkpoint_saved",
        "weights_persisted",
        "prediction_tensor_persisted",
        "oracle_tensor_persisted",
        "candidate_tensor_persisted",
        "npz_saved",
        "cache_saved",
        "formal_diffusion_training",
        "formal_reverse_sampling",
        "formal_idm_training",
        "action_diverse_data_collection",
        "candidate_execution",
        "deformable_ravens_executed",
        "phase4",
        "cps",
    )
    for key in required_false:
        if payload.get(key) is not False:
            raise CommitRecoveryError(
                "first Stage-C blocked boundary changed: {}".format(key)
            )
    if payload.get("selected_configuration") is not None:
        raise CommitRecoveryError(
            "first Stage-C blocked selected a configuration"
        )
    if payload.get("train_only_recommendation") is not None:
        raise CommitRecoveryError(
            "first Stage-C blocked emitted a recommendation"
        )


def validate_first_attempt_reports(root: Path) -> Dict[str, Any]:
    repository_root = Path(root).resolve()
    test_path = repository_root / FIRST_TEST_GATE
    blocked_path = repository_root / FIRST_BLOCKED
    if not test_path.is_file():
        raise FileNotFoundError(test_path)
    if not blocked_path.is_file():
        raise FileNotFoundError(blocked_path)
    test_payload = load_json(test_path)
    blocked_payload = load_json(blocked_path)
    validate_first_test_gate(test_payload)
    validate_first_blocked(blocked_payload)
    return {
        "test_gate_path": FIRST_TEST_GATE,
        "test_gate_sha256": sha256_file(test_path),
        "test_gate": test_payload,
        "blocked_path": FIRST_BLOCKED,
        "blocked_sha256": sha256_file(blocked_path),
        "blocked": blocked_payload,
        "commit_command_status": 141,
        "commit_transaction_completed": True,
        "scientific_calibration_started": False,
    }


def validate_provenance_commit(
    root: Path,
    commit: str = "HEAD",
) -> Dict[str, Any]:
    repository_root = Path(root).resolve()
    resolved = git_output(
        repository_root,
        "rev-parse",
        commit,
    )
    assert_commit_shape(
        repository_root,
        commit=resolved,
        parent=IMPLEMENTATION_COMMIT,
        subject=PROVENANCE_MESSAGE,
        paths=FIRST_PROVENANCE_PATHS,
    )
    reports = validate_first_attempt_reports(
        repository_root
    )
    for relative in FIRST_PROVENANCE_PATHS:
        committed = subprocess.check_output(
            [
                "git",
                "show",
                "{}:{}".format(
                    resolved,
                    relative,
                ),
            ],
            cwd=str(repository_root),
        )
        if committed != (
            repository_root / relative
        ).read_bytes():
            raise CommitRecoveryError(
                "provenance report differs from commit: {}".format(relative)
            )
    return {
        "provenance_commit": resolved,
        "provenance_parent": IMPLEMENTATION_COMMIT,
        "provenance_message": PROVENANCE_MESSAGE,
        "reports": reports,
    }


def validate_remote_before_resume(root: Path) -> str:
    repository_root = Path(root).resolve()
    remote = git_output(
        repository_root,
        "rev-parse",
        "origin/Experiment1",
    )
    if remote != REMOTE_BASE_COMMIT:
        raise CommitRecoveryError(
            "origin/Experiment1 changed before Resume1: {}".format(remote)
        )
    return remote


def status_paths(root: Path) -> Tuple[str, ...]:
    output = git_output(
        root,
        "status",
        "--porcelain",
        "--untracked-files=all",
    )
    paths = []
    for line in output.splitlines():
        if not line.strip():
            continue
        value = line[3:].strip()
        if " -> " in value:
            value = value.split(" -> ", 1)[1]
        paths.append(value)
    return tuple(sorted(paths))


def validate_initial_resume_worktree(root: Path) -> Dict[str, Any]:
    repository_root = Path(root).resolve()
    if git_output(
        repository_root,
        "branch",
        "--show-current",
    ) != "Experiment1":
        raise CommitRecoveryError(
            "Resume1 requires Experiment1"
        )
    if git_output(
        repository_root,
        "rev-parse",
        "HEAD",
    ) != IMPLEMENTATION_COMMIT:
        raise CommitRecoveryError(
            "Resume1 initial HEAD changed"
        )
    remote = validate_remote_before_resume(
        repository_root
    )
    implementation = validate_implementation_commit(
        repository_root
    )
    reports = validate_first_attempt_reports(
        repository_root
    )
    expected = tuple(
        sorted(
            FIRST_PROVENANCE_PATHS
            + RESUME_IMPLEMENTATION_PATHS
        )
    )
    observed = status_paths(
        repository_root
    )
    if observed != expected:
        raise CommitRecoveryError(
            "unexpected initial Resume1 worktree paths: {}".format(observed)
        )
    return {
        "head": IMPLEMENTATION_COMMIT,
        "remote": remote,
        "implementation": implementation,
        "reports": reports,
        "expected_untracked_paths": list(expected),
    }


def validate_resume_implementation_commit(
    root: Path,
    commit: str = "HEAD",
) -> Dict[str, Any]:
    repository_root = Path(root).resolve()
    resolved = git_output(
        repository_root,
        "rev-parse",
        commit,
    )
    parent = git_output(
        repository_root,
        "rev-parse",
        "{}^".format(resolved),
    )
    provenance = validate_provenance_commit(
        repository_root,
        parent,
    )
    assert_commit_shape(
        repository_root,
        commit=resolved,
        parent=parent,
        subject=RESUME_IMPLEMENTATION_MESSAGE,
        paths=RESUME_IMPLEMENTATION_PATHS,
    )
    return {
        "resume_implementation_commit": resolved,
        "resume_parent": parent,
        "resume_message": RESUME_IMPLEMENTATION_MESSAGE,
        "provenance": provenance,
    }


def run_calibration(
    *,
    root: Path,
    environment: Mapping[str, Any],
) -> Dict[str, Any]:
    """Run the unchanged Stage-C scientific calibration."""
    result = stagec.run_calibration(
        root=Path(root).resolve(),
        environment=environment,
    )
    if result.get("phase") != "Phase3.14b-r2.5.8 Stage C":
        raise CommitRecoveryError(
            "underlying Stage-C phase changed"
        )
    return result


def compare_worker_results(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> Dict[str, Any]:
    return stagec.compare_worker_results(
        left,
        right,
    )
