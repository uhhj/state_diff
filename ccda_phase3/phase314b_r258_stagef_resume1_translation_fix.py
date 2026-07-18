"""Stage-F Resume1 provenance and translation-fix execution contract."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from ccda_phase3 import phase314b_r258_stagef_constraint_aware_surrogate as stagef

PHASE = "Phase3.14b-r2.5.8 Stage F Resume1"
PHASE_ID = "phase314b_r258_stagef_resume1"

BASE_EVIDENCE_COMMIT = "6758ea7ad800667a436b0243d3b1f6c63256d854"
ORIGINAL_IMPLEMENTATION_COMMIT = "519531f411c43b17a668c3c6c1a43b46a94f40a7"
EXPECTED_INITIAL_HEAD = ORIGINAL_IMPLEMENTATION_COMMIT
EXPECTED_INITIAL_REMOTE = BASE_EVIDENCE_COMMIT
EXPECTED_SUBMODULE_COMMIT = "633a88752445cf5d6776ed374fdbbdb35f93050c"

ORIGINAL_IMPLEMENTATION_MESSAGE = (
    "Add Phase3.14b-r2.5.8 constraint-aware projected-oracle direction surrogate"
)
BLOCKED_PROVENANCE_MESSAGE = (
    "Record blocked Phase3.14b-r2.5.8 Stage-F translation-invariance execution"
)
RESUME1_IMPLEMENTATION_MESSAGE = (
    "Add Phase3.14b-r2.5.8 Stage-F Resume1 stable translation features"
)
RESUME1_EVIDENCE_MESSAGE = (
    "Record Phase3.14b-r2.5.8 Stage-F Resume1 evidence"
)

ORIGINAL_IMPLEMENTATION_PATH_SHA256 = {
    "ccda_phase3/phase314b_r258_stagef_constraint_aware_surrogate.py": (
        "3443ae27d7688a5646c9c5c95e33e916ae9dbf3c123c8ef41d097671be078d1b"
    ),
    "scripts/phase3_14b_r258_stagef_worker.py": (
        "b4a02a8547b76b3f55d1e5e3c87d3803cb4b8b18495af01e807623d6f02879cf"
    ),
    "scripts/phase3_14b_r258_stagef_run_calibration.py": (
        "b38ed39a7a6229634eb9c7ea98f455e6cca7fc1a9f499d2732b7a6549cbc5048"
    ),
    "scripts/phase3_14b_r258_stagef_test_gate.py": (
        "6891dcb1b1eb7c9071b76df40e78443e6849fec81dd590392cda4b68851ce8d9"
    ),
    "scripts/phase3_14b_r258_stagef_blocked.py": (
        "7ce0884a9332b9ad6dbdde90079f9b20be0f1682b7e4d50207ec0546c45181b2"
    ),
    "scripts/phase3_14b_r258_stagef_run.sh": (
        "a5fe633ca6642485ca73c2a49689cb1e6d48983e8f645cf16a9b416fd1b517d9"
    ),
    "tests/test_phase3_14b_r258_stagef_constraint_aware_surrogate.py": (
        "ac4f6a2e0d9b6b1bce08622d735fa2fe4333728729d0051497cf2bad859e2aab"
    ),
}
ORIGINAL_IMPLEMENTATION_PATHS = tuple(ORIGINAL_IMPLEMENTATION_PATH_SHA256)

BLOCKED_PROVENANCE_PATHS = (
    "reports/phase3_14b_r258_stagef_test_gate_summary.json",
    "reports/phase3_14b_r258_stagef_blocked_summary.json",
)

RESUME1_IMPLEMENTATION_PATHS = (
    "ccda_phase3/phase314b_r258_stagef_constraint_aware_surrogate.py",
    "ccda_phase3/phase314b_r258_stagef_resume1_translation_fix.py",
    "scripts/phase3_14b_r258_stagef_resume1_worker.py",
    "scripts/phase3_14b_r258_stagef_resume1_test_gate.py",
    "scripts/phase3_14b_r258_stagef_resume1_run_calibration.py",
    "scripts/phase3_14b_r258_stagef_resume1_blocked.py",
    "scripts/phase3_14b_r258_stagef_resume1_run.sh",
    "tests/test_phase3_14b_r258_stagef_resume1_translation_fix.py",
)

RESUME1_TEST_GATE = (
    "reports/phase3_14b_r258_stagef_resume1_test_gate_summary.json"
)
RESUME1_CONTRACT = "reports/phase3_14b_r258_stagef_resume1_contract.json"
RESUME1_WORKER = "reports/phase3_14b_r258_stagef_resume1_worker_evidence.json"
RESUME1_SUMMARY = "reports/phase3_14b_r258_stagef_resume1_summary.json"
RESUME1_REPORT = "reports/phase3_14b_r258_stagef_resume1_report.md"
RESUME1_BLOCKED = "reports/phase3_14b_r258_stagef_resume1_blocked_summary.json"
RESUME1_SUCCESS_PATHS = (
    RESUME1_TEST_GATE,
    RESUME1_CONTRACT,
    RESUME1_WORKER,
    RESUME1_SUMMARY,
    RESUME1_REPORT,
)

ORIGINAL_STAGEF_TEST = (
    "tests/test_phase3_14b_r258_stagef_constraint_aware_surrogate.py"
)
RESUME1_TEST = (
    "tests/test_phase3_14b_r258_stagef_resume1_translation_fix.py"
)


class StageFResume1Error(RuntimeError):
    """Raised when Resume1 provenance or execution invariants fail."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            stagef.jsonable(dict(payload)),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def atomic_write_once(path: Path, payload: bytes, *, mode: int = 0o644) -> None:
    stagef.atomic_write_once(path, payload, mode=mode)


def load_json(path: Path) -> Dict[str, Any]:
    return stagef.load_json(path)


def git_output(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=str(root),
        text=True,
    ).strip()


def status_paths(root: Path) -> Tuple[str, ...]:
    output = git_output(root, "status", "--porcelain", "--untracked-files=all")
    paths = []
    for line in output.splitlines():
        if not line.strip():
            continue
        value = line[3:].strip()
        if " -> " in value:
            value = value.split(" -> ", 1)[1]
        paths.append(value)
    return tuple(sorted(paths))


def commit_parent(root: Path, commit: str) -> str:
    return git_output(root, "rev-parse", "{}^".format(commit))


def commit_subject(root: Path, commit: str) -> str:
    return git_output(root, "show", "-s", "--format=%s", commit)


def commit_paths(root: Path, commit: str) -> Tuple[str, ...]:
    output = git_output(
        root,
        "diff-tree",
        "--no-commit-id",
        "--name-only",
        "-r",
        commit,
    )
    return tuple(sorted(line.strip() for line in output.splitlines() if line.strip()))


def assert_commit_shape(
    root: Path,
    *,
    commit: str,
    parent: str,
    subject: str,
    paths: Sequence[str],
) -> None:
    if commit_parent(root, commit) != parent:
        raise StageFResume1Error("commit parent changed: {}".format(commit))
    if commit_subject(root, commit) != subject:
        raise StageFResume1Error("commit subject changed: {}".format(commit))
    if commit_paths(root, commit) != tuple(sorted(str(path) for path in paths)):
        raise StageFResume1Error("commit path population changed: {}".format(commit))


def assert_commit_file_sha_map(
    root: Path,
    *,
    commit: str,
    expected: Mapping[str, str],
) -> Dict[str, str]:
    observed: Dict[str, str] = {}
    for relative, expected_sha in expected.items():
        payload = subprocess.check_output(
            ["git", "show", "{}:{}".format(commit, relative)],
            cwd=str(root),
        )
        actual = sha256_bytes(payload)
        if actual != expected_sha:
            raise StageFResume1Error(
                "committed Stage-F file changed: {}".format(relative)
            )
        observed[str(relative)] = actual
    return observed


def validate_original_implementation_commit(root: Path) -> Dict[str, Any]:
    repository_root = Path(root).resolve()
    assert_commit_shape(
        repository_root,
        commit=ORIGINAL_IMPLEMENTATION_COMMIT,
        parent=BASE_EVIDENCE_COMMIT,
        subject=ORIGINAL_IMPLEMENTATION_MESSAGE,
        paths=ORIGINAL_IMPLEMENTATION_PATHS,
    )
    blob_sha = assert_commit_file_sha_map(
        repository_root,
        commit=ORIGINAL_IMPLEMENTATION_COMMIT,
        expected=ORIGINAL_IMPLEMENTATION_PATH_SHA256,
    )
    return {
        "base_evidence_commit": BASE_EVIDENCE_COMMIT,
        "original_implementation_commit": ORIGINAL_IMPLEMENTATION_COMMIT,
        "original_blob_sha256": blob_sha,
    }


def validate_original_blocked_reports(root: Path) -> Dict[str, Any]:
    repository_root = Path(root).resolve()
    gate_path = repository_root / BLOCKED_PROVENANCE_PATHS[0]
    blocked_path = repository_root / BLOCKED_PROVENANCE_PATHS[1]
    if not gate_path.is_file() or not blocked_path.is_file():
        raise FileNotFoundError("Stage-F write-once failure evidence is incomplete")
    gate = load_json(gate_path)
    blocked = load_json(blocked_path)
    if gate.get("phase") != "Phase3.14b-r2.5.8 Stage F":
        raise StageFResume1Error("Stage-F test-gate phase changed")
    if gate.get("verdict") != "PASS":
        raise StageFResume1Error("Stage-F test gate is not PASS")
    if int(gate.get("test_file_count", -1)) != 56:
        raise StageFResume1Error("Stage-F test file count changed")
    if int(gate.get("passed_test_count", -1)) != 1408:
        raise StageFResume1Error("Stage-F pass count changed")
    if gate.get("scientific_calibration_run") is not False:
        raise StageFResume1Error("Stage-F gate claims calibration ran")
    expected_blocked = {
        "phase": "Phase3.14b-r2.5.8 Stage F",
        "verdict": "BLOCKED",
        "scientific_status": "BLOCKED",
        "root_cause": "phase314b_r258_stagef_execution_failed_before_completion",
        "base_evidence_commit": BASE_EVIDENCE_COMMIT,
        "scientific_result_sealed": False,
        "success_evidence_commit_created": False,
        "push_completed": False,
        "selected_configuration": None,
        "train_only_recommendation": None,
        "selection_holdout_evaluated": False,
        "frozen_probe_accessed": False,
        "new_diffusion_model_candidate_trained": False,
        "reverse_sampling_run": False,
        "formal_training_run": False,
        "idm_run": False,
        "candidate_execution": False,
        "deformable_ravens_executed": False,
        "phase4": False,
        "cps": False,
    }
    for key, value in expected_blocked.items():
        if blocked.get(key) != value:
            raise StageFResume1Error(
                "Stage-F blocked report field changed: {}".format(key)
            )
    if blocked.get("head") != ORIGINAL_IMPLEMENTATION_COMMIT:
        raise StageFResume1Error("Stage-F blocked report HEAD changed")
    if blocked.get("origin_experiment1") != BASE_EVIDENCE_COMMIT:
        raise StageFResume1Error("Stage-F blocked report remote changed")
    failed_command = str(blocked.get("failed_command", ""))
    if "stagef" not in failed_command.lower() and "worker" not in failed_command.lower():
        raise StageFResume1Error("Stage-F blocked command provenance is missing")
    return {
        "test_gate": gate,
        "blocked": blocked,
        "test_gate_sha256": sha256_file(gate_path),
        "blocked_sha256": sha256_file(blocked_path),
    }


def validate_initial_worktree(root: Path) -> Dict[str, Any]:
    repository_root = Path(root).resolve()
    if git_output(repository_root, "branch", "--show-current") != "Experiment1":
        raise StageFResume1Error("Resume1 requires Experiment1")
    head = git_output(repository_root, "rev-parse", "HEAD")
    remote = git_output(repository_root, "rev-parse", "origin/Experiment1")
    if head != EXPECTED_INITIAL_HEAD:
        raise StageFResume1Error("initial local HEAD changed: {}".format(head))
    if remote != EXPECTED_INITIAL_REMOTE:
        raise StageFResume1Error("initial remote changed: {}".format(remote))
    implementation = validate_original_implementation_commit(repository_root)
    failure = validate_original_blocked_reports(repository_root)
    for relative, expected_sha in ORIGINAL_IMPLEMENTATION_PATH_SHA256.items():
        if relative == "ccda_phase3/phase314b_r258_stagef_constraint_aware_surrogate.py":
            continue
        path = repository_root / relative
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise StageFResume1Error("unchanged Stage-F file changed: {}".format(relative))
    expected_status = tuple(
        sorted((*BLOCKED_PROVENANCE_PATHS, *RESUME1_IMPLEMENTATION_PATHS))
    )
    observed = status_paths(repository_root)
    if observed != expected_status:
        raise StageFResume1Error(
            "unexpected Resume1 initial paths: {}".format(observed)
        )
    submodule = repository_root / "external/deformable-ravens"
    if git_output(submodule, "rev-parse", "HEAD") != EXPECTED_SUBMODULE_COMMIT:
        raise StageFResume1Error("DeformableRavens commit changed")
    if status_paths(submodule):
        raise StageFResume1Error("DeformableRavens worktree is dirty")
    return {
        "branch": "Experiment1",
        "head": head,
        "origin_experiment1": remote,
        "implementation": implementation,
        "failure_evidence": {
            "test_gate_sha256": failure["test_gate_sha256"],
            "blocked_sha256": failure["blocked_sha256"],
        },
        "expected_status_paths": list(expected_status),
        "submodule_commit": EXPECTED_SUBMODULE_COMMIT,
    }


def validate_blocked_provenance_commit(root: Path) -> Dict[str, Any]:
    repository_root = Path(root).resolve()
    commit = git_output(repository_root, "rev-parse", "HEAD")
    assert_commit_shape(
        repository_root,
        commit=commit,
        parent=ORIGINAL_IMPLEMENTATION_COMMIT,
        subject=BLOCKED_PROVENANCE_MESSAGE,
        paths=BLOCKED_PROVENANCE_PATHS,
    )
    failure = validate_original_blocked_reports(repository_root)
    for relative, key in (
        (BLOCKED_PROVENANCE_PATHS[0], "test_gate_sha256"),
        (BLOCKED_PROVENANCE_PATHS[1], "blocked_sha256"),
    ):
        payload = subprocess.check_output(
            ["git", "show", "{}:{}".format(commit, relative)],
            cwd=str(repository_root),
        )
        if sha256_bytes(payload) != failure[key]:
            raise StageFResume1Error("blocked provenance blob changed: {}".format(relative))
    return {
        **validate_original_implementation_commit(repository_root),
        "blocked_provenance_commit": commit,
        "test_gate_sha256": failure["test_gate_sha256"],
        "blocked_sha256": failure["blocked_sha256"],
    }


def validate_resume1_implementation_commit(root: Path) -> Dict[str, Any]:
    repository_root = Path(root).resolve()
    commit = git_output(repository_root, "rev-parse", "HEAD")
    blocked_commit = commit_parent(repository_root, commit)
    assert_commit_shape(
        repository_root,
        commit=commit,
        parent=blocked_commit,
        subject=RESUME1_IMPLEMENTATION_MESSAGE,
        paths=RESUME1_IMPLEMENTATION_PATHS,
    )
    assert_commit_shape(
        repository_root,
        commit=blocked_commit,
        parent=ORIGINAL_IMPLEMENTATION_COMMIT,
        subject=BLOCKED_PROVENANCE_MESSAGE,
        paths=BLOCKED_PROVENANCE_PATHS,
    )
    chain = validate_blocked_provenance_commit_at(repository_root, blocked_commit)
    corrected_module = (
        repository_root
        / "ccda_phase3/phase314b_r258_stagef_constraint_aware_surrogate.py"
    )
    committed = subprocess.check_output(
        [
            "git",
            "show",
            "{}:{}".format(
                commit,
                "ccda_phase3/phase314b_r258_stagef_constraint_aware_surrogate.py",
            ),
        ],
        cwd=str(repository_root),
    )
    if committed != corrected_module.read_bytes():
        raise StageFResume1Error("corrected Stage-F module differs from commit")
    return {
        **chain,
        "resume1_implementation_commit": commit,
        "corrected_module_sha256": sha256_file(corrected_module),
    }


def validate_blocked_provenance_commit_at(
    root: Path,
    commit: str,
) -> Dict[str, Any]:
    repository_root = Path(root).resolve()
    assert_commit_shape(
        repository_root,
        commit=commit,
        parent=ORIGINAL_IMPLEMENTATION_COMMIT,
        subject=BLOCKED_PROVENANCE_MESSAGE,
        paths=BLOCKED_PROVENANCE_PATHS,
    )
    original = validate_original_implementation_commit(repository_root)
    sha_map = {}
    for relative in BLOCKED_PROVENANCE_PATHS:
        payload = subprocess.check_output(
            ["git", "show", "{}:{}".format(commit, relative)],
            cwd=str(repository_root),
        )
        sha_map[relative] = sha256_bytes(payload)
    return {
        **original,
        "blocked_provenance_commit": commit,
        "blocked_provenance_blob_sha256": sha_map,
    }
