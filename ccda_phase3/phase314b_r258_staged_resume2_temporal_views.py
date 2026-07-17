"""Phase3.14b-r2.5.8 Stage-D Resume2 temporal test-view isolation.

Resume1 correctly moved frozen tests out of the current worktree, but imposed an
obsolete assumption that the Stage-C Resume1 commit itself contained only the
27 Phase3 r2.x tests known to the r2.5.4 Resume4 audit.  In reality, the
immutable 6866507 evidence view legitimately includes later r2.5.5--r2.5.8
tests.  The historical Resume4 test is the sole test in the frozen 51-file
manifest whose assertion depends on a repository-wide glob.

Resume2 runs:
  * the other 50 frozen tests in the 6866507 evidence view;
  * the historical Resume4 test in its introduction commit 3b30ad6..., where
    its 27-file closed-world contract is true;
  * Stage-D, Resume1, and Resume2 tests explicitly in the current view.

The two historical populations must sum to the immutable 51 files / 1064
passes.  No test implementation is modified, skipped, ignored, deselected,
renamed, or copied into a synthetic source tree.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

PHASE = "Phase3.14b-r2.5.8 Stage D Resume2"
PHASE_ID = "phase314b_r258_staged_resume2"

BASE_EVIDENCE_COMMIT = (
    "6866507c42b9bc9d2d271becd1a9423f61710405"
)
ORIGINAL_IMPLEMENTATION_COMMIT = (
    "539521776dd515fac11a84396d3664ce44f31017"
)
ORIGINAL_BLOCKED_PROVENANCE_COMMIT = (
    "4888d9fa85e7b9411899152015acc6dad4f15223"
)
RESUME1_IMPLEMENTATION_COMMIT = (
    "8b692966d9f332ad2557c755c1b3657fc4a94943"
)
EXPECTED_INITIAL_HEAD = RESUME1_IMPLEMENTATION_COMMIT
EXPECTED_INITIAL_REMOTE = BASE_EVIDENCE_COMMIT

HISTORICAL_CLOSED_WORLD_COMMIT = (
    "3b30ad610bf35244839c2f78d5fed1921d98c6bc"
)
HISTORICAL_CLOSED_WORLD_TEST = (
    "tests/test_phase314b_r254_resume4_reproduction_audit.py"
)
HISTORICAL_CLOSED_WORLD_MODULE = (
    "ccda_phase3/phase314b_r254_resume4_reproduction_audit.py"
)
EXPECTED_HISTORICAL_SCOPE_COUNT = 27
EXPECTED_HISTORICAL_TEST_PASSED = 8

EXPECTED_SUBMODULE_COMMIT = (
    "633a88752445cf5d6776ed374fdbbdb35f93050c"
)

ORIGINAL_IMPLEMENTATION_MESSAGE = (
    "Add Phase3.14b-r2.5.8 objective-train direction surrogate calibration"
)
ORIGINAL_BLOCKED_MESSAGE = (
    "Record blocked Phase3.14b-r2.5.8 Stage-D test-view execution"
)
RESUME1_IMPLEMENTATION_MESSAGE = (
    "Add Phase3.14b-r2.5.8 Stage-D Resume1 frozen test-view isolation"
)
RESUME1_BLOCKED_MESSAGE = (
    "Record blocked Phase3.14b-r2.5.8 Stage-D Resume1 test-view execution"
)
RESUME2_IMPLEMENTATION_MESSAGE = (
    "Add Phase3.14b-r2.5.8 Stage-D Resume2 temporal test-view isolation"
)
RESUME2_EVIDENCE_MESSAGE = (
    "Record Phase3.14b-r2.5.8 Stage-D Resume2 evidence"
)

ORIGINAL_IMPLEMENTATION_PATH_SHA256 = {
    "ccda_phase3/phase314b_r258_staged_direction_surrogate.py": (
        "946fca9d84b54e4affd35ab890970ba1"
        "f17154aa8b925f66746994ffe484bbb9"
    ),
    "scripts/phase3_14b_r258_staged_worker.py": (
        "1dead7186e43664b78b2865b06616c345"
        "2a7263bde8f701a34000458ed26afbe"
    ),
    "scripts/phase3_14b_r258_staged_run_calibration.py": (
        "29da739c1cb4973bd965b5e51353a610a"
        "7c02b675a5c3f03706cf08e401525eb"
    ),
    "scripts/phase3_14b_r258_staged_test_gate.py": (
        "4275469c6b872a4c3180255a4306d925"
        "e571bf65d8ef75d044e0649587c2b755"
    ),
    "scripts/phase3_14b_r258_staged_blocked.py": (
        "cdb4a895407d6aae1512429caa00687da"
        "2b9bd70340d114ceb5781337835278c"
    ),
    "scripts/phase3_14b_r258_staged_run.sh": (
        "213c9d4a99712a9519ea7ccb0035540b2"
        "2517b09971c498b0d736732d7d5ada4"
    ),
    "tests/test_phase3_14b_r258_staged_direction_surrogate.py": (
        "be782b988059fe3b1ce282933acbffd37"
        "268ac40ec2437f082540c9a60d61dd1"
    ),
}
ORIGINAL_IMPLEMENTATION_PATHS = tuple(
    ORIGINAL_IMPLEMENTATION_PATH_SHA256
)

RESUME1_IMPLEMENTATION_PATH_SHA256 = {
    "ccda_phase3/phase314b_r258_staged_resume1_test_view.py": (
        "1b0349bd3d0605cc25f011d2e92bd0b9"
        "cd46f7ca801fc0e5d96950aaa3e51fbf"
    ),
    "scripts/phase3_14b_r258_staged_resume1_test_gate.py": (
        "93e59dbaab0d97552db2e19b9d05eb4e"
        "2d476ad280e5bfd824765ef1113af6b4"
    ),
    "scripts/phase3_14b_r258_staged_resume1_run_calibration.py": (
        "d449eea452c214ca447f9263f4d3b0415"
        "a9467643c50ba2a942d51cdf2e2da50"
    ),
    "scripts/phase3_14b_r258_staged_resume1_blocked.py": (
        "3a34d0addb24f0838289d37bee9031cb6b"
        "bbe35b87be3acae66c58516b2ac6ef"
    ),
    "scripts/phase3_14b_r258_staged_resume1_run.sh": (
        "6a112c8e1e45bbd02da6e685d5f0d6a1"
        "ddd39d6b2e8f9d5f95790435fc279ea4"
    ),
    "tests/test_phase3_14b_r258_staged_resume1_test_view.py": (
        "51d590d2693304a74acb7988ab30d6e48"
        "4ce5ee88563f7deb35d5c166917e329"
    ),
}
RESUME1_IMPLEMENTATION_PATHS = tuple(
    RESUME1_IMPLEMENTATION_PATH_SHA256
)

ORIGINAL_BLOCKED_PATH = (
    "reports/phase3_14b_r258_staged_blocked_summary.json"
)
ORIGINAL_BLOCKED_SHA256 = (
    "c6176abe55cdbf63506765b5ec67745a6"
    "828a582d6eda70fd95b122e2e012bdc"
)
RESUME1_BLOCKED_PATH = (
    "reports/phase3_14b_r258_staged_resume1_blocked_summary.json"
)
RESUME1_BLOCKED_SHA256 = (
    "f14373370a56a269722dbdb6a3474ef5b"
    "e4c9983d0cd44b61f56e1dd4d46ddeb"
)

RESUME2_IMPLEMENTATION_PATHS = (
    "ccda_phase3/phase314b_r258_staged_resume2_temporal_views.py",
    "scripts/phase3_14b_r258_staged_resume2_test_gate.py",
    "scripts/phase3_14b_r258_staged_resume2_run_calibration.py",
    "scripts/phase3_14b_r258_staged_resume2_blocked.py",
    "scripts/phase3_14b_r258_staged_resume2_run.sh",
    "tests/test_phase3_14b_r258_staged_resume2_temporal_views.py",
)

BASE_TEST_GATE = (
    "reports/phase3_14b_r258_stagec_resume1_test_gate_summary.json"
)
CURRENT_STAGE_D_TEST = (
    "tests/test_phase3_14b_r258_staged_direction_surrogate.py"
)
CURRENT_RESUME1_TEST = (
    "tests/test_phase3_14b_r258_staged_resume1_test_view.py"
)
CURRENT_RESUME2_TEST = (
    "tests/test_phase3_14b_r258_staged_resume2_temporal_views.py"
)

RESUME2_TEST_GATE = (
    "reports/phase3_14b_r258_staged_resume2_test_gate_summary.json"
)
RESUME2_CONTRACT = (
    "reports/phase3_14b_r258_staged_resume2_contract.json"
)
RESUME2_WORKER = (
    "reports/phase3_14b_r258_staged_resume2_worker_evidence.json"
)
RESUME2_SUMMARY = (
    "reports/phase3_14b_r258_staged_resume2_summary.json"
)
RESUME2_REPORT = (
    "reports/phase3_14b_r258_staged_resume2_report.md"
)
RESUME2_BLOCKED = (
    "reports/phase3_14b_r258_staged_resume2_blocked_summary.json"
)
RESUME2_SUCCESS_PATHS = (
    RESUME2_TEST_GATE,
    RESUME2_CONTRACT,
    RESUME2_WORKER,
    RESUME2_SUMMARY,
    RESUME2_REPORT,
)

EXPECTED_BASE_TEST_FILES = 51
EXPECTED_BASE_PASSED = 1064
EXPECTED_BASE_REGULAR_FILES = 50
EXPECTED_BASE_REGULAR_PASSED = (
    EXPECTED_BASE_PASSED
    - EXPECTED_HISTORICAL_TEST_PASSED
)
EXPECTED_STAGE_D_PASSED = 84
EXPECTED_RESUME1_PASSED = 40
# Updated after the Resume2 unit test is frozen.
EXPECTED_RESUME2_PASSED = 55


class TemporalTestViewError(RuntimeError):
    """Raised when commit, temporal view, or test-population gates fail."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): jsonable(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if value is None or isinstance(
        value,
        (str, int, bool),
    ):
        return value
    if isinstance(value, float):
        if not (
            value == value
            and value not in (
                float("inf"),
                float("-inf"),
            )
        ):
            raise ValueError(
                "non-finite float cannot be serialized"
            )
        return value
    raise TypeError(
        "unsupported JSON type: {!r}".format(
            type(value)
        )
    )


def stable_json_bytes(
    payload: Mapping[str, Any],
) -> bytes:
    return (
        json.dumps(
            jsonable(dict(payload)),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def atomic_write_once(
    path: Path,
    payload: bytes,
    *,
    mode: int = 0o644,
) -> None:
    target = Path(path)
    if target.exists():
        raise FileExistsError(
            "refusing to overwrite write-once output: {}".format(
                target
            )
        )
    target.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temporary = target.parent / (
        ".{}.{}.tmp".format(
            target.name,
            os.getpid(),
        )
    )
    if temporary.exists():
        temporary.unlink()
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, target)
        directory_fd = os.open(
            str(target.parent),
            os.O_RDONLY,
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(
        Path(path).read_text(
            encoding="utf-8"
        )
    )
    if not isinstance(value, dict):
        raise TemporalTestViewError(
            "JSON root is not an object: {}".format(
                path
            )
        )
    return value


def git_output(
    root: Path,
    *args: str,
) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=str(root),
        text=True,
    ).strip()


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
            value = value.split(
                " -> ",
                1,
            )[1]
        paths.append(value)
    return tuple(sorted(paths))


def commit_subject(
    root: Path,
    commit: str,
) -> str:
    return git_output(
        root,
        "show",
        "-s",
        "--format=%s",
        commit,
    )


def commit_parent(
    root: Path,
    commit: str,
) -> str:
    return git_output(
        root,
        "rev-parse",
        "{}^".format(commit),
    )


def commit_paths(
    root: Path,
    commit: str,
) -> Tuple[str, ...]:
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


def assert_commit_shape(
    root: Path,
    *,
    commit: str,
    parent: str,
    subject: str,
    paths: Sequence[str],
) -> None:
    if commit_parent(
        root,
        commit,
    ) != parent:
        raise TemporalTestViewError(
            "commit parent changed: {}".format(
                commit
            )
        )
    if commit_subject(
        root,
        commit,
    ) != subject:
        raise TemporalTestViewError(
            "commit subject changed: {}".format(
                commit
            )
        )
    expected = tuple(
        sorted(str(path) for path in paths)
    )
    if commit_paths(
        root,
        commit,
    ) != expected:
        raise TemporalTestViewError(
            "commit path population changed: {}".format(
                commit
            )
        )


def assert_file_sha_map(
    root: Path,
    expected: Mapping[str, str],
) -> Dict[str, str]:
    repository_root = Path(root).resolve()
    observed = {}
    for relative, expected_sha in expected.items():
        path = repository_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != expected_sha:
            raise TemporalTestViewError(
                "file changed: {}".format(relative)
            )
        observed[str(relative)] = actual
    return observed


def assert_commit_file_sha_map(
    root: Path,
    *,
    commit: str,
    expected: Mapping[str, str],
) -> Dict[str, str]:
    repository_root = Path(root).resolve()
    observed = {}
    for relative, expected_sha in expected.items():
        payload = subprocess.check_output(
            [
                "git",
                "show",
                "{}:{}".format(
                    commit,
                    relative,
                ),
            ],
            cwd=str(repository_root),
        )
        actual = sha256_bytes(payload)
        if actual != expected_sha:
            raise TemporalTestViewError(
                "committed file changed: {}:{}".format(
                    commit,
                    relative,
                )
            )
        observed[str(relative)] = actual
    return observed


def validate_existing_commit_chain(
    root: Path,
) -> Dict[str, str]:
    repository_root = Path(root).resolve()
    assert_commit_shape(
        repository_root,
        commit=ORIGINAL_IMPLEMENTATION_COMMIT,
        parent=BASE_EVIDENCE_COMMIT,
        subject=ORIGINAL_IMPLEMENTATION_MESSAGE,
        paths=ORIGINAL_IMPLEMENTATION_PATHS,
    )
    assert_commit_shape(
        repository_root,
        commit=ORIGINAL_BLOCKED_PROVENANCE_COMMIT,
        parent=ORIGINAL_IMPLEMENTATION_COMMIT,
        subject=ORIGINAL_BLOCKED_MESSAGE,
        paths=(ORIGINAL_BLOCKED_PATH,),
    )
    assert_commit_shape(
        repository_root,
        commit=RESUME1_IMPLEMENTATION_COMMIT,
        parent=ORIGINAL_BLOCKED_PROVENANCE_COMMIT,
        subject=RESUME1_IMPLEMENTATION_MESSAGE,
        paths=RESUME1_IMPLEMENTATION_PATHS,
    )
    assert_commit_file_sha_map(
        repository_root,
        commit=ORIGINAL_IMPLEMENTATION_COMMIT,
        expected=ORIGINAL_IMPLEMENTATION_PATH_SHA256,
    )
    assert_commit_file_sha_map(
        repository_root,
        commit=RESUME1_IMPLEMENTATION_COMMIT,
        expected=RESUME1_IMPLEMENTATION_PATH_SHA256,
    )
    original_blocked_blob = subprocess.check_output(
        [
            "git",
            "show",
            "{}:{}".format(
                ORIGINAL_BLOCKED_PROVENANCE_COMMIT,
                ORIGINAL_BLOCKED_PATH,
            ),
        ],
        cwd=str(repository_root),
    )
    if sha256_bytes(
        original_blocked_blob
    ) != ORIGINAL_BLOCKED_SHA256:
        raise TemporalTestViewError(
            "original blocked provenance blob changed"
        )
    return {
        "base_evidence_commit":
            BASE_EVIDENCE_COMMIT,
        "original_implementation_commit":
            ORIGINAL_IMPLEMENTATION_COMMIT,
        "original_blocked_provenance_commit":
            ORIGINAL_BLOCKED_PROVENANCE_COMMIT,
        "resume1_implementation_commit":
            RESUME1_IMPLEMENTATION_COMMIT,
    }


def validate_blocked_report(
    path: Path,
    *,
    expected_sha256: str,
    expected_phase: str,
) -> Dict[str, Any]:
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(target)
    if sha256_file(target) != expected_sha256:
        raise TemporalTestViewError(
            "blocked report SHA changed: {}".format(
                target
            )
        )
    report = load_json(target)
    if report.get("phase") != expected_phase:
        raise TemporalTestViewError(
            "blocked report phase changed"
        )
    if report.get("verdict") != "BLOCKED":
        raise TemporalTestViewError(
            "blocked report verdict changed"
        )
    if report.get("scientific_status") != "BLOCKED":
        raise TemporalTestViewError(
            "blocked report scientific status changed"
        )
    if report.get("scientific_result_sealed") is not False:
        raise TemporalTestViewError(
            "blocked report sealed a scientific result"
        )
    if report.get("push_completed") is not False:
        raise TemporalTestViewError(
            "blocked report claims a push"
        )
    if report.get("selected_configuration") is not None:
        raise TemporalTestViewError(
            "blocked report selected a configuration"
        )
    if report.get("train_only_recommendation") is not None:
        raise TemporalTestViewError(
            "blocked report emitted a recommendation"
        )
    return report


def validate_base_test_gate(
    root: Path,
) -> Dict[str, Any]:
    repository_root = Path(root).resolve()
    path = repository_root / BASE_TEST_GATE
    if not path.is_file():
        raise FileNotFoundError(path)
    committed = subprocess.check_output(
        [
            "git",
            "show",
            "{}:{}".format(
                BASE_EVIDENCE_COMMIT,
                BASE_TEST_GATE,
            ),
        ],
        cwd=str(repository_root),
    )
    if committed != path.read_bytes():
        raise TemporalTestViewError(
            "base test gate differs from base evidence commit"
        )
    report = load_json(path)
    if report.get("verdict") != "PASS":
        raise TemporalTestViewError(
            "base test gate is not PASS"
        )
    if int(
        report.get("test_file_count", -1)
    ) != EXPECTED_BASE_TEST_FILES:
        raise TemporalTestViewError(
            "base test-file count changed"
        )
    if int(
        report.get("passed_test_count", -1)
    ) != EXPECTED_BASE_PASSED:
        raise TemporalTestViewError(
            "base pass count changed"
        )
    manifest = report.get(
        "test_manifest_sha256"
    )
    if (
        not isinstance(manifest, dict)
        or len(manifest)
        != EXPECTED_BASE_TEST_FILES
    ):
        raise TemporalTestViewError(
            "base test manifest changed"
        )
    if HISTORICAL_CLOSED_WORLD_TEST not in manifest:
        raise TemporalTestViewError(
            "historical closed-world test is not in base manifest"
        )
    return report


def validate_initial_worktree(
    root: Path,
) -> Dict[str, Any]:
    repository_root = Path(root).resolve()
    if git_output(
        repository_root,
        "branch",
        "--show-current",
    ) != "Experiment1":
        raise TemporalTestViewError(
            "Resume2 requires Experiment1"
        )
    head = git_output(
        repository_root,
        "rev-parse",
        "HEAD",
    )
    remote = git_output(
        repository_root,
        "rev-parse",
        "origin/Experiment1",
    )
    if head != EXPECTED_INITIAL_HEAD:
        raise TemporalTestViewError(
            "initial local HEAD changed: {}".format(
                head
            )
        )
    if remote != EXPECTED_INITIAL_REMOTE:
        raise TemporalTestViewError(
            "initial remote changed: {}".format(
                remote
            )
        )
    commits = validate_existing_commit_chain(
        repository_root
    )
    original_sha = assert_file_sha_map(
        repository_root,
        ORIGINAL_IMPLEMENTATION_PATH_SHA256,
    )
    resume1_sha = assert_file_sha_map(
        repository_root,
        RESUME1_IMPLEMENTATION_PATH_SHA256,
    )
    original_blocked = validate_blocked_report(
        repository_root / ORIGINAL_BLOCKED_PATH,
        expected_sha256=ORIGINAL_BLOCKED_SHA256,
        expected_phase=(
            "Phase3.14b-r2.5.8 Stage D"
        ),
    )
    resume1_blocked = validate_blocked_report(
        repository_root / RESUME1_BLOCKED_PATH,
        expected_sha256=RESUME1_BLOCKED_SHA256,
        expected_phase=(
            "Phase3.14b-r2.5.8 Stage D Resume1"
        ),
    )
    validate_base_test_gate(
        repository_root
    )

    expected_untracked = tuple(
        sorted(
            (
                RESUME1_BLOCKED_PATH,
                *RESUME2_IMPLEMENTATION_PATHS,
            )
        )
    )
    observed = status_paths(
        repository_root
    )
    if observed != expected_untracked:
        raise TemporalTestViewError(
            "unexpected initial worktree paths: {}".format(
                observed
            )
        )

    submodule = (
        repository_root
        / "external/deformable-ravens"
    )
    if git_output(
        submodule,
        "rev-parse",
        "HEAD",
    ) != EXPECTED_SUBMODULE_COMMIT:
        raise TemporalTestViewError(
            "DeformableRavens commit changed"
        )
    if status_paths(submodule):
        raise TemporalTestViewError(
            "DeformableRavens worktree is dirty"
        )
    return {
        "head": head,
        "origin_experiment1": remote,
        "commit_chain": commits,
        "original_file_sha256": original_sha,
        "resume1_file_sha256": resume1_sha,
        "original_blocked": original_blocked,
        "resume1_blocked": resume1_blocked,
        "expected_untracked_paths":
            list(expected_untracked),
        "submodule_commit":
            EXPECTED_SUBMODULE_COMMIT,
    }


def validate_resume1_blocked_provenance_commit(
    root: Path,
) -> Dict[str, Any]:
    repository_root = Path(root).resolve()
    commit = git_output(
        repository_root,
        "rev-parse",
        "HEAD",
    )
    assert_commit_shape(
        repository_root,
        commit=commit,
        parent=RESUME1_IMPLEMENTATION_COMMIT,
        subject=RESUME1_BLOCKED_MESSAGE,
        paths=(RESUME1_BLOCKED_PATH,),
    )
    committed = subprocess.check_output(
        [
            "git",
            "show",
            "{}:{}".format(
                commit,
                RESUME1_BLOCKED_PATH,
            ),
        ],
        cwd=str(repository_root),
    )
    if sha256_bytes(
        committed
    ) != RESUME1_BLOCKED_SHA256:
        raise TemporalTestViewError(
            "Resume1 blocked provenance blob changed"
        )
    return {
        **validate_existing_commit_chain(
            repository_root
        ),
        "resume1_blocked_provenance_commit":
            commit,
    }


def validate_resume2_implementation_commit(
    root: Path,
) -> Dict[str, Any]:
    repository_root = Path(root).resolve()
    commit = git_output(
        repository_root,
        "rev-parse",
        "HEAD",
    )
    parent = commit_parent(
        repository_root,
        commit,
    )
    assert_commit_shape(
        repository_root,
        commit=commit,
        parent=parent,
        subject=RESUME2_IMPLEMENTATION_MESSAGE,
        paths=RESUME2_IMPLEMENTATION_PATHS,
    )
    assert_commit_shape(
        repository_root,
        commit=parent,
        parent=RESUME1_IMPLEMENTATION_COMMIT,
        subject=RESUME1_BLOCKED_MESSAGE,
        paths=(RESUME1_BLOCKED_PATH,),
    )
    committed_blocked = subprocess.check_output(
        [
            "git",
            "show",
            "{}:{}".format(
                parent,
                RESUME1_BLOCKED_PATH,
            ),
        ],
        cwd=str(repository_root),
    )
    if sha256_bytes(
        committed_blocked
    ) != RESUME1_BLOCKED_SHA256:
        raise TemporalTestViewError(
            "Resume1 blocked provenance changed"
        )
    return {
        **validate_existing_commit_chain(
            repository_root
        ),
        "resume1_blocked_provenance_commit":
            parent,
        "resume2_implementation_commit":
            commit,
    }


def parse_pytest_pass_count(
    output: str,
) -> int:
    summaries = []
    for line in str(output).splitlines():
        if " passed" not in line:
            continue
        match = re.search(
            r"(?:^|\s)([0-9]+) passed(?:,|\s|$)",
            line,
        )
        if match:
            summaries.append(
                (
                    line,
                    int(match.group(1)),
                )
            )
    if len(summaries) != 1:
        raise ValueError(
            "expected exactly one pytest pass summary"
        )
    line, count = summaries[0]
    forbidden = (
        " failed",
        " error",
        " skipped",
        " xfailed",
        " xpassed",
        " deselected",
    )
    if any(
        token in line
        for token in forbidden
    ):
        raise ValueError(
            "pytest summary is not all-pass: {}".format(
                line
            )
        )
    return count


def gitlink_commit(
    root: Path,
    commit: str,
    path: str = "external/deformable-ravens",
) -> str:
    output = git_output(
        root,
        "ls-tree",
        commit,
        path,
    )
    parts = output.split()
    if (
        len(parts) < 4
        or parts[0] != "160000"
        or parts[1] != "commit"
        or parts[3] != path
    ):
        raise TemporalTestViewError(
            "submodule gitlink changed at {}".format(
                commit
            )
        )
    return parts[2]


def _set_remote_url_from_source(
    source: Path,
    target: Path,
) -> None:
    completed = subprocess.run(
        [
            "git",
            "config",
            "--get",
            "remote.origin.url",
        ],
        cwd=str(source),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if (
        completed.returncode == 0
        and completed.stdout.strip()
    ):
        subprocess.run(
            [
                "git",
                "remote",
                "set-url",
                "origin",
                completed.stdout.strip(),
            ],
            cwd=str(target),
            check=True,
        )


def create_temporal_clone(
    *,
    source_root: Path,
    destination: Path,
    commit: str,
) -> Dict[str, Any]:
    source = Path(source_root).resolve()
    target = Path(destination).resolve()
    if target.exists():
        raise FileExistsError(
            "temporal clone destination exists: {}".format(
                target
            )
        )
    subprocess.run(
        [
            "git",
            "cat-file",
            "-e",
            "{}^{{commit}}".format(commit),
        ],
        cwd=str(source),
        check=True,
    )
    source_submodule = (
        source
        / "external/deformable-ravens"
    )
    if not source_submodule.is_dir():
        raise FileNotFoundError(
            source_submodule
        )
    submodule_commit = gitlink_commit(
        source,
        commit,
    )
    subprocess.run(
        [
            "git",
            "cat-file",
            "-e",
            "{}^{{commit}}".format(
                submodule_commit
            ),
        ],
        cwd=str(source_submodule),
        check=True,
    )

    subprocess.run(
        [
            "git",
            "clone",
            "--shared",
            "--quiet",
            str(source),
            str(target),
        ],
        check=True,
    )
    try:
        subprocess.run(
            [
                "git",
                "checkout",
                "-B",
                "Experiment1",
                commit,
            ],
            cwd=str(target),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        subprocess.run(
            [
                "git",
                "reset",
                "--hard",
                commit,
            ],
            cwd=str(target),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        subprocess.run(
            [
                "git",
                "update-ref",
                "refs/remotes/origin/Experiment1",
                commit,
            ],
            cwd=str(target),
            check=True,
        )
        _set_remote_url_from_source(
            source,
            target,
        )
        subprocess.run(
            [
                "git",
                "clean",
                "-ffd",
            ],
            cwd=str(target),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )

        target_submodule = (
            target
            / "external/deformable-ravens"
        )
        if target_submodule.exists():
            if target_submodule.is_dir():
                shutil.rmtree(
                    target_submodule
                )
            else:
                target_submodule.unlink()
        target_submodule.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        subprocess.run(
            [
                "git",
                "clone",
                "--shared",
                "--quiet",
                str(source_submodule),
                str(target_submodule),
            ],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "checkout",
                "--detach",
                submodule_commit,
            ],
            cwd=str(target_submodule),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        subprocess.run(
            [
                "git",
                "reset",
                "--hard",
                submodule_commit,
            ],
            cwd=str(target_submodule),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        _set_remote_url_from_source(
            source_submodule,
            target_submodule,
        )

        if git_output(
            target,
            "branch",
            "--show-current",
        ) != "Experiment1":
            raise TemporalTestViewError(
                "temporal clone branch changed"
            )
        if git_output(
            target,
            "rev-parse",
            "HEAD",
        ) != commit:
            raise TemporalTestViewError(
                "temporal clone HEAD changed"
            )
        if git_output(
            target,
            "rev-parse",
            "origin/Experiment1",
        ) != commit:
            raise TemporalTestViewError(
                "temporal clone remote ref changed"
            )
        if status_paths(target):
            raise TemporalTestViewError(
                "temporal clone is dirty"
            )
        if git_output(
            target_submodule,
            "rev-parse",
            "HEAD",
        ) != submodule_commit:
            raise TemporalTestViewError(
                "temporal clone submodule changed"
            )
        if status_paths(target_submodule):
            raise TemporalTestViewError(
                "temporal clone submodule is dirty"
            )
        return {
            "source_root": str(source),
            "clone_root": str(target),
            "clone_branch": "Experiment1",
            "clone_head": commit,
            "clone_origin_experiment1":
                commit,
            "clone_status_clean": True,
            "clone_submodule_commit":
                submodule_commit,
            "clone_submodule_clean": True,
            "uses_network": False,
            "uses_ignore": False,
            "uses_k_expression": False,
            "uses_deselection": False,
            "renames_current_tests": False,
            "copies_or_patches_test_sources":
                False,
            "mutates_current_worktree": False,
        }
    except Exception:
        shutil.rmtree(
            target,
            ignore_errors=True,
        )
        raise


def validate_manifest_files(
    *,
    root: Path,
    manifest: Mapping[str, str],
) -> Dict[str, str]:
    base = Path(root).resolve()
    observed = {}
    for relative, expected in manifest.items():
        path = base / str(relative)
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != expected:
            raise TemporalTestViewError(
                "manifest file changed: {}".format(
                    relative
                )
            )
        observed[str(relative)] = actual
    return observed


def split_base_manifest(
    manifest: Mapping[str, str],
) -> Dict[str, Any]:
    if len(manifest) != EXPECTED_BASE_TEST_FILES:
        raise TemporalTestViewError(
            "base manifest size changed"
        )
    if HISTORICAL_CLOSED_WORLD_TEST not in manifest:
        raise TemporalTestViewError(
            "historical test missing from manifest"
        )
    regular = {
        str(relative): str(expected)
        for relative, expected in manifest.items()
        if relative != HISTORICAL_CLOSED_WORLD_TEST
    }
    historical = {
        HISTORICAL_CLOSED_WORLD_TEST:
            str(
                manifest[
                    HISTORICAL_CLOSED_WORLD_TEST
                ]
            )
    }
    if len(regular) != EXPECTED_BASE_REGULAR_FILES:
        raise TemporalTestViewError(
            "regular base manifest size changed"
        )
    return {
        "regular_manifest": regular,
        "historical_manifest": historical,
        "regular_count": len(regular),
        "historical_count": 1,
    }


def historical_test_blob_identity(
    root: Path,
) -> Dict[str, Any]:
    repository_root = Path(root).resolve()
    records = {}
    for relative in (
        HISTORICAL_CLOSED_WORLD_TEST,
        HISTORICAL_CLOSED_WORLD_MODULE,
    ):
        base_blob = subprocess.check_output(
            [
                "git",
                "show",
                "{}:{}".format(
                    BASE_EVIDENCE_COMMIT,
                    relative,
                ),
            ],
            cwd=str(repository_root),
        )
        historical_blob = subprocess.check_output(
            [
                "git",
                "show",
                "{}:{}".format(
                    HISTORICAL_CLOSED_WORLD_COMMIT,
                    relative,
                ),
            ],
            cwd=str(repository_root),
        )
        if base_blob != historical_blob:
            raise TemporalTestViewError(
                "historical closed-world contract changed after "
                "introduction: {}".format(relative)
            )
        records[relative] = {
            "base_blob_sha256":
                sha256_bytes(base_blob),
            "historical_blob_sha256":
                sha256_bytes(historical_blob),
            "byte_exact":
                True,
        }
    return {
        "records": records,
        "byte_exact": True,
    }


def inspect_historical_scope(
    *,
    python_bin: str,
    clone_root: Path,
) -> Dict[str, Any]:
    script = r"""
import json
from pathlib import Path
from ccda_phase3.phase314b_r254_resume4_reproduction_audit import (
    PHASE3_R2_TEST_PATHS,
    phase3_r2_test_manifest,
)
root = Path.cwd()
observed = tuple(phase3_r2_test_manifest(root))
expected = tuple(sorted(PHASE3_R2_TEST_PATHS))
print(json.dumps({
    "observed": list(observed),
    "expected": list(expected),
    "observed_count": len(observed),
    "expected_count": len(expected),
    "exact": observed == expected,
}, sort_keys=True))
"""
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONNOUSERSITE": "1",
            "PYTHONHASHSEED": "0",
        }
    )
    completed = subprocess.run(
        [
            str(python_bin),
            "-c",
            script,
        ],
        cwd=str(clone_root),
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if completed.returncode != 0:
        raise TemporalTestViewError(
            "historical scope inspection failed:\n{}".format(
                completed.stdout
            )
        )
    lines = [
        line
        for line in completed.stdout.splitlines()
        if line.strip().startswith("{")
    ]
    if len(lines) != 1:
        raise TemporalTestViewError(
            "historical scope inspection output changed"
        )
    record = json.loads(lines[0])
    if record.get("exact") is not True:
        raise TemporalTestViewError(
            "historical scope is not exact"
        )
    if int(
        record.get("observed_count", -1)
    ) != EXPECTED_HISTORICAL_SCOPE_COUNT:
        raise TemporalTestViewError(
            "historical scope count changed"
        )
    return record
