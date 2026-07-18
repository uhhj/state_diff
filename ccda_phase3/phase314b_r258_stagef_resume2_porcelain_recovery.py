"""Stage-F Resume2 robust porcelain-path recovery contract.

Resume1 stopped before creating any commit because its helper called
``git status --porcelain`` through a function that stripped the complete
stdout string.  A first record with status `` M`` therefore lost its leading
space before the fixed ``line[3:]`` slice, dropping the first path character.

Resume2 parses ``git status --porcelain=v1 -z`` as bytes.  NUL-delimited
records preserve leading status columns and arbitrary path bytes.  The
scientific Stage-F translation fix, candidate matrix, integrator, thresholds,
folds, and evidence boundaries are unchanged.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

from ccda_phase3 import phase314b_r258_stagef_resume1_translation_fix as resume1
from ccda_phase3 import phase314b_r258_stagef_constraint_aware_surrogate as stagef

PHASE = "Phase3.14b-r2.5.8 Stage F Resume2"
PHASE_ID = "phase314b_r258_stagef_resume2"

BASE_EVIDENCE_COMMIT = "6758ea7ad800667a436b0243d3b1f6c63256d854"
ORIGINAL_IMPLEMENTATION_COMMIT = "519531f411c43b17a668c3c6c1a43b46a94f40a7"
EXPECTED_INITIAL_HEAD = ORIGINAL_IMPLEMENTATION_COMMIT
EXPECTED_INITIAL_REMOTE = BASE_EVIDENCE_COMMIT
EXPECTED_SUBMODULE_COMMIT = "633a88752445cf5d6776ed374fdbbdb35f93050c"

STAGEF_BLOCKED_PROVENANCE_MESSAGE = (
    "Record blocked Phase3.14b-r2.5.8 Stage-F translation-invariance execution"
)
RESUME1_IMPLEMENTATION_MESSAGE = (
    "Add Phase3.14b-r2.5.8 Stage-F Resume1 stable translation features"
)
RESUME1_BLOCKED_PROVENANCE_MESSAGE = (
    "Record blocked Phase3.14b-r2.5.8 Stage-F Resume1 porcelain-path execution"
)
RESUME2_IMPLEMENTATION_MESSAGE = (
    "Add Phase3.14b-r2.5.8 Stage-F Resume2 robust porcelain path recovery"
)
RESUME2_EVIDENCE_MESSAGE = (
    "Record Phase3.14b-r2.5.8 Stage-F Resume2 evidence"
)

STAGEF_TEST_GATE_PATH = "reports/phase3_14b_r258_stagef_test_gate_summary.json"
STAGEF_BLOCKED_PATH = "reports/phase3_14b_r258_stagef_blocked_summary.json"
RESUME1_BLOCKED_PATH = (
    "reports/phase3_14b_r258_stagef_resume1_blocked_summary.json"
)
STAGEF_TEST_GATE_SHA256 = (
    "7d3eadbd063e2248a17c0d26fcd3d47897c46dc434b98a6692c17c214a9a5bd2"
)
STAGEF_BLOCKED_SHA256 = (
    "3a970c1c36998da92312eb78833e2f21401e8556414c87f4502be253a69f5886"
)
RESUME1_BLOCKED_SHA256 = (
    "b8a11c88aefd9e70f65d2a0d2afbf88b8feabad5f6c97784e41f4d813513c80f"
)

RESUME1_IMPLEMENTATION_PATH_SHA256: Dict[str, str] = {
    "ccda_phase3/phase314b_r258_stagef_constraint_aware_surrogate.py": (
        "b8b8a346a32f20d716e197c32b81a745a4d2730b4ce52beb2f4825ec46f3f14a"
    ),
    "ccda_phase3/phase314b_r258_stagef_resume1_translation_fix.py": (
        "2b52b92cd87ecb0bbf3adaed2c9f299ed251085a63c7415ac3117fb59c9b5f80"
    ),
    "scripts/phase3_14b_r258_stagef_resume1_blocked.py": (
        "bee83e7b9b53f10316bb718ebd7d8d6826d67196a2975f3d9607c98de78a19c6"
    ),
    "scripts/phase3_14b_r258_stagef_resume1_run.sh": (
        "fdf710203c74cead595f0f747f81dba32d30a859fb37e03161f0b81e1ba3529c"
    ),
    "scripts/phase3_14b_r258_stagef_resume1_run_calibration.py": (
        "faae8d8662874229ff848f8611da10adcf0720445c69ea538180e266c6b2e180"
    ),
    "scripts/phase3_14b_r258_stagef_resume1_test_gate.py": (
        "2581656d9b0485243448e19f4a6a70d0d5c7c916bbb2759fe8d3f66ae9db207a"
    ),
    "scripts/phase3_14b_r258_stagef_resume1_worker.py": (
        "d838f8881e70a94a38f5e4e1edbd7448db5e03ee295682e431a77280fb06d70e"
    ),
    "tests/test_phase3_14b_r258_stagef_resume1_translation_fix.py": (
        "0d2470fc138ea467b238e8459029cec35ec078f093711bb5787821ed0ef95a07"
    ),
}
RESUME1_IMPLEMENTATION_PATHS = tuple(RESUME1_IMPLEMENTATION_PATH_SHA256)

RESUME2_IMPLEMENTATION_PATHS = (
    "ccda_phase3/phase314b_r258_stagef_resume2_porcelain_recovery.py",
    "scripts/phase3_14b_r258_stagef_resume2_worker.py",
    "scripts/phase3_14b_r258_stagef_resume2_test_gate.py",
    "scripts/phase3_14b_r258_stagef_resume2_run_calibration.py",
    "scripts/phase3_14b_r258_stagef_resume2_blocked.py",
    "scripts/phase3_14b_r258_stagef_resume2_run.sh",
    "tests/test_phase3_14b_r258_stagef_resume2_porcelain_recovery.py",
)

RESUME2_TEST_GATE = (
    "reports/phase3_14b_r258_stagef_resume2_test_gate_summary.json"
)
RESUME2_CONTRACT = "reports/phase3_14b_r258_stagef_resume2_contract.json"
RESUME2_WORKER = "reports/phase3_14b_r258_stagef_resume2_worker_evidence.json"
RESUME2_SUMMARY = "reports/phase3_14b_r258_stagef_resume2_summary.json"
RESUME2_REPORT = "reports/phase3_14b_r258_stagef_resume2_report.md"
RESUME2_BLOCKED = "reports/phase3_14b_r258_stagef_resume2_blocked_summary.json"
RESUME2_SUCCESS_PATHS = (
    RESUME2_TEST_GATE,
    RESUME2_CONTRACT,
    RESUME2_WORKER,
    RESUME2_SUMMARY,
    RESUME2_REPORT,
)
RESUME2_TEST = "tests/test_phase3_14b_r258_stagef_resume2_porcelain_recovery.py"


class StageFResume2Error(RuntimeError):
    """Raised when Resume2 provenance or porcelain parsing gates fail."""


@dataclass(frozen=True)
class PorcelainEntry:
    index_status: str
    worktree_status: str
    path: str
    original_path: str | None = None

    @property
    def status(self) -> str:
        return self.index_status + self.worktree_status


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


def parse_porcelain_v1_z(payload: bytes) -> Tuple[PorcelainEntry, ...]:
    """Parse ``git status --porcelain=v1 -z`` without trimming status columns."""
    if not isinstance(payload, (bytes, bytearray)):
        raise TypeError("porcelain payload must be bytes")
    data = bytes(payload)
    if not data:
        return ()
    if not data.endswith(b"\0"):
        raise StageFResume2Error("porcelain -z payload is not NUL terminated")
    records = data[:-1].split(b"\0")
    entries = []
    index = 0
    while index < len(records):
        record = records[index]
        if len(record) < 4 or record[2:3] != b" ":
            raise StageFResume2Error(
                "malformed porcelain record at {}: {!r}".format(index, record)
            )
        try:
            status = record[:2].decode("ascii")
        except UnicodeDecodeError as exc:
            raise StageFResume2Error("non-ASCII porcelain status") from exc
        destination = os.fsdecode(record[3:])
        if not destination:
            raise StageFResume2Error("empty porcelain path")
        original = None
        if "R" in status or "C" in status:
            index += 1
            if index >= len(records):
                raise StageFResume2Error("rename/copy record is missing original path")
            original = os.fsdecode(records[index])
            if not original:
                raise StageFResume2Error("rename/copy original path is empty")
        entries.append(
            PorcelainEntry(
                index_status=status[0],
                worktree_status=status[1],
                path=destination,
                original_path=original,
            )
        )
        index += 1
    return tuple(entries)


def status_entries(root: Path) -> Tuple[PorcelainEntry, ...]:
    payload = subprocess.check_output(
        [
            "git",
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
        ],
        cwd=str(root),
    )
    return parse_porcelain_v1_z(payload)


def status_paths(root: Path) -> Tuple[str, ...]:
    return tuple(sorted(entry.path for entry in status_entries(root)))


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
    return tuple(sorted(line for line in output.splitlines() if line))


def assert_commit_shape(
    root: Path,
    *,
    commit: str,
    parent: str,
    subject: str,
    paths: Sequence[str],
) -> None:
    if commit_parent(root, commit) != parent:
        raise StageFResume2Error("commit parent changed: {}".format(commit))
    if commit_subject(root, commit) != subject:
        raise StageFResume2Error("commit subject changed: {}".format(commit))
    if commit_paths(root, commit) != tuple(sorted(paths)):
        raise StageFResume2Error("commit paths changed: {}".format(commit))


def assert_file_sha_map(root: Path, expected: Mapping[str, str]) -> Dict[str, str]:
    observed: Dict[str, str] = {}
    for relative, expected_sha in expected.items():
        path = Path(root) / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != expected_sha:
            raise StageFResume2Error("file changed: {}".format(relative))
        observed[relative] = actual
    return observed


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
            raise StageFResume2Error(
                "committed file changed: {}:{}".format(commit, relative)
            )
        observed[relative] = actual
    return observed


def validate_report_sha(path: Path, expected_sha: str) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    if sha256_file(path) != expected_sha:
        raise StageFResume2Error("write-once report SHA changed: {}".format(path))
    return load_json(path)


def validate_failure_reports(root: Path) -> Dict[str, Any]:
    repository_root = Path(root).resolve()
    stagef_gate = validate_report_sha(
        repository_root / STAGEF_TEST_GATE_PATH,
        STAGEF_TEST_GATE_SHA256,
    )
    stagef_blocked = validate_report_sha(
        repository_root / STAGEF_BLOCKED_PATH,
        STAGEF_BLOCKED_SHA256,
    )
    resume1_blocked = validate_report_sha(
        repository_root / RESUME1_BLOCKED_PATH,
        RESUME1_BLOCKED_SHA256,
    )
    if stagef_gate.get("verdict") != "PASS":
        raise StageFResume2Error("Stage-F test gate is not PASS")
    if int(stagef_gate.get("test_file_count", -1)) != 56:
        raise StageFResume2Error("Stage-F test file count changed")
    if int(stagef_gate.get("passed_test_count", -1)) != 1408:
        raise StageFResume2Error("Stage-F pass count changed")
    if stagef_gate.get("scientific_calibration_run") is not False:
        raise StageFResume2Error("Stage-F gate claims science ran")
    for report, phase in (
        (stagef_blocked, "Phase3.14b-r2.5.8 Stage F"),
        (resume1_blocked, "Phase3.14b-r2.5.8 Stage F Resume1"),
    ):
        if report.get("phase") != phase:
            raise StageFResume2Error("blocked phase changed: {}".format(phase))
        if report.get("verdict") != "BLOCKED":
            raise StageFResume2Error("blocked verdict changed: {}".format(phase))
        if report.get("scientific_status") != "BLOCKED":
            raise StageFResume2Error("blocked scientific status changed")
        if report.get("scientific_result_sealed") is not False:
            raise StageFResume2Error("blocked report sealed a scientific result")
        if report.get("push_completed") is not False:
            raise StageFResume2Error("blocked report claims push")
        if report.get("selected_configuration") is not None:
            raise StageFResume2Error("blocked report selected a configuration")
        if report.get("train_only_recommendation") is not None:
            raise StageFResume2Error("blocked report emitted a recommendation")
    if resume1_blocked.get("head") != ORIGINAL_IMPLEMENTATION_COMMIT:
        raise StageFResume2Error("Resume1 blocked HEAD changed")
    if resume1_blocked.get("origin_experiment1") != BASE_EVIDENCE_COMMIT:
        raise StageFResume2Error("Resume1 blocked remote changed")
    failed = str(resume1_blocked.get("failed_command", ""))
    if "validate_initial_worktree" not in failed and "python" not in failed.lower():
        raise StageFResume2Error("Resume1 blocked command provenance changed")
    return {
        "stagef_test_gate": stagef_gate,
        "stagef_blocked": stagef_blocked,
        "resume1_blocked": resume1_blocked,
        "sha256": {
            STAGEF_TEST_GATE_PATH: STAGEF_TEST_GATE_SHA256,
            STAGEF_BLOCKED_PATH: STAGEF_BLOCKED_SHA256,
            RESUME1_BLOCKED_PATH: RESUME1_BLOCKED_SHA256,
        },
    }


def expected_initial_status_paths() -> Tuple[str, ...]:
    return tuple(
        sorted(
            (
                STAGEF_TEST_GATE_PATH,
                STAGEF_BLOCKED_PATH,
                RESUME1_BLOCKED_PATH,
                *RESUME1_IMPLEMENTATION_PATHS,
                *RESUME2_IMPLEMENTATION_PATHS,
            )
        )
    )


def validate_initial_worktree(root: Path) -> Dict[str, Any]:
    repository_root = Path(root).resolve()
    if git_output(repository_root, "branch", "--show-current") != "Experiment1":
        raise StageFResume2Error("Resume2 requires Experiment1")
    head = git_output(repository_root, "rev-parse", "HEAD")
    remote = git_output(repository_root, "rev-parse", "origin/Experiment1")
    if head != EXPECTED_INITIAL_HEAD:
        raise StageFResume2Error("initial local HEAD changed: {}".format(head))
    if remote != EXPECTED_INITIAL_REMOTE:
        raise StageFResume2Error("initial remote changed: {}".format(remote))
    original = resume1.validate_original_implementation_commit(repository_root)
    failure = validate_failure_reports(repository_root)
    resume1_files = assert_file_sha_map(
        repository_root,
        RESUME1_IMPLEMENTATION_PATH_SHA256,
    )
    observed = status_paths(repository_root)
    expected = expected_initial_status_paths()
    if observed != expected:
        raise StageFResume2Error(
            "unexpected Resume2 initial paths: observed={!r}, expected={!r}".format(
                observed,
                expected,
            )
        )
    submodule = repository_root / "external/deformable-ravens"
    if git_output(submodule, "rev-parse", "HEAD") != EXPECTED_SUBMODULE_COMMIT:
        raise StageFResume2Error("DeformableRavens commit changed")
    if status_paths(submodule):
        raise StageFResume2Error("DeformableRavens worktree is dirty")
    return {
        "phase": PHASE,
        "branch": "Experiment1",
        "head": head,
        "origin_experiment1": remote,
        "original_implementation": original,
        "failure_evidence": failure["sha256"],
        "resume1_file_sha256": resume1_files,
        "expected_status_paths": list(expected),
        "porcelain_parser": {
            "format": "v1-z",
            "stdout_trimmed": False,
            "nul_delimited": True,
            "rename_copy_supported": True,
        },
        "submodule_commit": EXPECTED_SUBMODULE_COMMIT,
    }


def validate_stagef_blocked_provenance_commit(root: Path) -> Dict[str, Any]:
    repository_root = Path(root).resolve()
    commit = git_output(repository_root, "rev-parse", "HEAD")
    assert_commit_shape(
        repository_root,
        commit=commit,
        parent=ORIGINAL_IMPLEMENTATION_COMMIT,
        subject=STAGEF_BLOCKED_PROVENANCE_MESSAGE,
        paths=(STAGEF_TEST_GATE_PATH, STAGEF_BLOCKED_PATH),
    )
    for relative, expected in (
        (STAGEF_TEST_GATE_PATH, STAGEF_TEST_GATE_SHA256),
        (STAGEF_BLOCKED_PATH, STAGEF_BLOCKED_SHA256),
    ):
        payload = subprocess.check_output(
            ["git", "show", "{}:{}".format(commit, relative)],
            cwd=str(repository_root),
        )
        if sha256_bytes(payload) != expected:
            raise StageFResume2Error("Stage-F provenance blob changed")
    return {
        "original_implementation_commit": ORIGINAL_IMPLEMENTATION_COMMIT,
        "stagef_blocked_provenance_commit": commit,
    }


def validate_resume1_implementation_commit(root: Path) -> Dict[str, Any]:
    repository_root = Path(root).resolve()
    commit = git_output(repository_root, "rev-parse", "HEAD")
    parent = commit_parent(repository_root, commit)
    assert_commit_shape(
        repository_root,
        commit=commit,
        parent=parent,
        subject=RESUME1_IMPLEMENTATION_MESSAGE,
        paths=RESUME1_IMPLEMENTATION_PATHS,
    )
    assert_commit_shape(
        repository_root,
        commit=parent,
        parent=ORIGINAL_IMPLEMENTATION_COMMIT,
        subject=STAGEF_BLOCKED_PROVENANCE_MESSAGE,
        paths=(STAGEF_TEST_GATE_PATH, STAGEF_BLOCKED_PATH),
    )
    committed_sha = assert_commit_file_sha_map(
        repository_root,
        commit=commit,
        expected=RESUME1_IMPLEMENTATION_PATH_SHA256,
    )
    return {
        "original_implementation_commit": ORIGINAL_IMPLEMENTATION_COMMIT,
        "stagef_blocked_provenance_commit": parent,
        "resume1_implementation_commit": commit,
        "resume1_blob_sha256": committed_sha,
    }


def validate_resume1_blocked_provenance_commit(root: Path) -> Dict[str, Any]:
    repository_root = Path(root).resolve()
    commit = git_output(repository_root, "rev-parse", "HEAD")
    resume1_commit = commit_parent(repository_root, commit)
    assert_commit_shape(
        repository_root,
        commit=commit,
        parent=resume1_commit,
        subject=RESUME1_BLOCKED_PROVENANCE_MESSAGE,
        paths=(RESUME1_BLOCKED_PATH,),
    )
    payload = subprocess.check_output(
        ["git", "show", "{}:{}".format(commit, RESUME1_BLOCKED_PATH)],
        cwd=str(repository_root),
    )
    if sha256_bytes(payload) != RESUME1_BLOCKED_SHA256:
        raise StageFResume2Error("Resume1 blocked provenance blob changed")
    resume1_chain = validate_resume1_implementation_commit_at(
        repository_root,
        resume1_commit,
    )
    return {
        **resume1_chain,
        "resume1_blocked_provenance_commit": commit,
    }


def validate_resume1_implementation_commit_at(
    root: Path,
    commit: str,
) -> Dict[str, Any]:
    repository_root = Path(root).resolve()
    parent = commit_parent(repository_root, commit)
    assert_commit_shape(
        repository_root,
        commit=commit,
        parent=parent,
        subject=RESUME1_IMPLEMENTATION_MESSAGE,
        paths=RESUME1_IMPLEMENTATION_PATHS,
    )
    assert_commit_shape(
        repository_root,
        commit=parent,
        parent=ORIGINAL_IMPLEMENTATION_COMMIT,
        subject=STAGEF_BLOCKED_PROVENANCE_MESSAGE,
        paths=(STAGEF_TEST_GATE_PATH, STAGEF_BLOCKED_PATH),
    )
    assert_commit_file_sha_map(
        repository_root,
        commit=commit,
        expected=RESUME1_IMPLEMENTATION_PATH_SHA256,
    )
    return {
        "original_implementation_commit": ORIGINAL_IMPLEMENTATION_COMMIT,
        "stagef_blocked_provenance_commit": parent,
        "resume1_implementation_commit": commit,
    }


def validate_resume2_implementation_commit(root: Path) -> Dict[str, Any]:
    repository_root = Path(root).resolve()
    commit = git_output(repository_root, "rev-parse", "HEAD")
    blocked_commit = commit_parent(repository_root, commit)
    assert_commit_shape(
        repository_root,
        commit=commit,
        parent=blocked_commit,
        subject=RESUME2_IMPLEMENTATION_MESSAGE,
        paths=RESUME2_IMPLEMENTATION_PATHS,
    )
    assert_commit_shape(
        repository_root,
        commit=blocked_commit,
        parent=commit_parent(repository_root, blocked_commit),
        subject=RESUME1_BLOCKED_PROVENANCE_MESSAGE,
        paths=(RESUME1_BLOCKED_PATH,),
    )
    chain = validate_resume1_blocked_provenance_commit_at(
        repository_root,
        blocked_commit,
    )
    return {
        **chain,
        "resume2_implementation_commit": commit,
    }


def validate_resume1_blocked_provenance_commit_at(
    root: Path,
    commit: str,
) -> Dict[str, Any]:
    repository_root = Path(root).resolve()
    resume1_commit = commit_parent(repository_root, commit)
    assert_commit_shape(
        repository_root,
        commit=commit,
        parent=resume1_commit,
        subject=RESUME1_BLOCKED_PROVENANCE_MESSAGE,
        paths=(RESUME1_BLOCKED_PATH,),
    )
    payload = subprocess.check_output(
        ["git", "show", "{}:{}".format(commit, RESUME1_BLOCKED_PATH)],
        cwd=str(repository_root),
    )
    if sha256_bytes(payload) != RESUME1_BLOCKED_SHA256:
        raise StageFResume2Error("Resume1 blocked provenance blob changed")
    return {
        **validate_resume1_implementation_commit_at(repository_root, resume1_commit),
        "resume1_blocked_provenance_commit": commit,
    }
