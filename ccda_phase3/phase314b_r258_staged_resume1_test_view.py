"""Phase3.14b-r2.5.8 Stage D Resume1 test-view isolation.

The first Stage-D execution stopped before any implementation commit or
scientific calibration.  The failure was caused by a historical test whose
contract intentionally globbed the complete Phase3 r2.x test namespace.  The
new Stage-D test file entered that glob and violated the historical closed-world
manifest.

Resume1 does not modify any Stage-D scientific code, candidate, threshold,
feature, split, worker, or output contract.  It executes the frozen 51-file
population in a local clone reset to the immutable Stage-C Resume1 evidence
commit.  The original Stage-D tests and the Resume1 isolation tests execute in
the current implementation view as separate, explicit populations.

No --ignore, -k, deselection, temporary renaming, or mutation of the current
worktree is used.
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

PHASE = "Phase3.14b-r2.5.8 Stage D Resume1"
PHASE_ID = "phase314b_r258_staged_resume1"

BASE_EVIDENCE_COMMIT = (
    "6866507c42b9bc9d2d271becd1a9423f61710405"
)
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
RESUME1_EVIDENCE_MESSAGE = (
    "Record Phase3.14b-r2.5.8 Stage-D Resume1 evidence"
)

ORIGINAL_IMPLEMENTATION_PATH_SHA256 = {
    (
        "ccda_phase3/"
        "phase314b_r258_staged_direction_surrogate.py"
    ): (
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
    (
        "tests/"
        "test_phase3_14b_r258_staged_direction_surrogate.py"
    ): (
        "be782b988059fe3b1ce282933acbffd37"
        "268ac40ec2437f082540c9a60d61dd1"
    ),
}
ORIGINAL_IMPLEMENTATION_PATHS = tuple(
    ORIGINAL_IMPLEMENTATION_PATH_SHA256
)

ORIGINAL_BLOCKED_PATH = (
    "reports/phase3_14b_r258_staged_blocked_summary.json"
)
ORIGINAL_BLOCKED_SHA256 = (
    "c6176abe55cdbf63506765b5ec67745a6"
    "828a582d6eda70fd95b122e2e012bdc"
)

RESUME1_IMPLEMENTATION_PATHS = (
    (
        "ccda_phase3/"
        "phase314b_r258_staged_resume1_test_view.py"
    ),
    (
        "scripts/"
        "phase3_14b_r258_staged_resume1_test_gate.py"
    ),
    (
        "scripts/"
        "phase3_14b_r258_staged_resume1_run_calibration.py"
    ),
    (
        "scripts/"
        "phase3_14b_r258_staged_resume1_blocked.py"
    ),
    (
        "scripts/"
        "phase3_14b_r258_staged_resume1_run.sh"
    ),
    (
        "tests/"
        "test_phase3_14b_r258_staged_resume1_test_view.py"
    ),
)

RESUME1_TEST_GATE = (
    "reports/"
    "phase3_14b_r258_staged_resume1_test_gate_summary.json"
)
RESUME1_CONTRACT = (
    "reports/"
    "phase3_14b_r258_staged_resume1_contract.json"
)
RESUME1_WORKER = (
    "reports/"
    "phase3_14b_r258_staged_resume1_worker_evidence.json"
)
RESUME1_SUMMARY = (
    "reports/"
    "phase3_14b_r258_staged_resume1_summary.json"
)
RESUME1_REPORT = (
    "reports/"
    "phase3_14b_r258_staged_resume1_report.md"
)
RESUME1_BLOCKED = (
    "reports/"
    "phase3_14b_r258_staged_resume1_blocked_summary.json"
)
RESUME1_SUCCESS_PATHS = (
    RESUME1_TEST_GATE,
    RESUME1_CONTRACT,
    RESUME1_WORKER,
    RESUME1_SUMMARY,
    RESUME1_REPORT,
)

BASE_TEST_GATE = (
    "reports/"
    "phase3_14b_r258_stagec_resume1_test_gate_summary.json"
)
ORIGINAL_STAGE_D_TEST = (
    "tests/"
    "test_phase3_14b_r258_staged_direction_surrogate.py"
)
RESUME1_TEST = (
    "tests/"
    "test_phase3_14b_r258_staged_resume1_test_view.py"
)

EXPECTED_BASE_TEST_FILES = 51
EXPECTED_BASE_PASSED = 1064
EXPECTED_STAGE_D_PASSED = 84
EXPECTED_RESUME1_PASSED = 40
EXPECTED_TOTAL_FILES = 53
EXPECTED_TOTAL_PASSED = (
    EXPECTED_BASE_PASSED
    + EXPECTED_STAGE_D_PASSED
    + EXPECTED_RESUME1_PASSED
)

PHASE3_R2_PATTERNS = (
    "tests/test_phase3_14b_r2*.py",
    "tests/test_phase314b_r2*.py",
)
EXPECTED_PHASE3_R2_TEST_PATHS = (
    "tests/test_phase3_14b_r21_contract.py",
    "tests/test_phase3_14b_r21_geometry.py",
    "tests/test_phase3_14b_r2_contract.py",
    "tests/test_phase3_14b_r2_diffusion.py",
    "tests/test_phase3_14b_r2_metrics.py",
    "tests/test_phase314b_r22_contract.py",
    "tests/test_phase314b_r22_gates.py",
    "tests/test_phase314b_r22_geometry.py",
    "tests/test_phase314b_r22_loss.py",
    "tests/test_phase314b_r231_controls.py",
    "tests/test_phase314b_r232_controls.py",
    "tests/test_phase314b_r23_diagnostics.py",
    "tests/test_phase314b_r241_multirow.py",
    "tests/test_phase314b_r242_frozen_prior.py",
    "tests/test_phase314b_r24_noisy_skip.py",
    "tests/test_phase314b_r251_gradient_calibration.py",
    "tests/test_phase314b_r252_transport_attribution.py",
    "tests/test_phase314b_r253_gate_separation.py",
    "tests/test_phase314b_r253_resume1_finalizer.py",
    "tests/test_phase314b_r253_resume2_finalizer.py",
    "tests/test_phase314b_r253_resume3_finalizer.py",
    "tests/test_phase314b_r254_resume1_prior_determinism.py",
    "tests/test_phase314b_r254_resume2_functional_prior.py",
    "tests/test_phase314b_r254_resume3_prediction_adapter.py",
    "tests/test_phase314b_r254_resume4_reproduction_audit.py",
    "tests/test_phase314b_r254_robot_proxy_attribution.py",
    "tests/test_phase314b_r25_ordered_geometry.py",
)


class TestViewIsolationError(RuntimeError):
    """Raised when Resume1 provenance or isolation invariants fail."""


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
        raise TestViewIsolationError(
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
        raise TestViewIsolationError(
            "commit parent changed: {}".format(
                commit
            )
        )
    if commit_subject(
        root,
        commit,
    ) != subject:
        raise TestViewIsolationError(
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
        raise TestViewIsolationError(
            "commit path population changed: {}".format(
                commit
            )
        )


def validate_original_files(
    root: Path,
) -> Dict[str, str]:
    repository_root = Path(root).resolve()
    observed = {}
    for relative, expected in (
        ORIGINAL_IMPLEMENTATION_PATH_SHA256.items()
    ):
        path = repository_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != expected:
            raise TestViewIsolationError(
                "original Stage-D file changed: {}".format(
                    relative
                )
            )
        observed[relative] = actual
    return observed


def validate_original_blocked(
    root: Path,
) -> Dict[str, Any]:
    path = Path(root).resolve() / ORIGINAL_BLOCKED_PATH
    if not path.is_file():
        raise FileNotFoundError(path)
    if sha256_file(path) != ORIGINAL_BLOCKED_SHA256:
        raise TestViewIsolationError(
            "original Stage-D blocked report changed"
        )
    report = load_json(path)
    expected = {
        "phase":
            "Phase3.14b-r2.5.8 Stage D",
        "verdict":
            "BLOCKED",
        "scientific_status":
            "BLOCKED",
        "root_cause": (
            "phase314b_r258_staged_"
            "execution_failed_before_completion"
        ),
        "base_evidence_commit":
            BASE_EVIDENCE_COMMIT,
        "scientific_result_sealed":
            False,
        "success_evidence_commit_created":
            False,
        "push_completed":
            False,
        "selected_configuration":
            None,
        "train_only_recommendation":
            None,
        "new_diffusion_model_candidate_trained":
            False,
        "frozen_probe_accessed":
            False,
        "reverse_sampling_run":
            False,
        "formal_diffusion_training":
            False,
        "formal_reverse_sampling":
            False,
        "formal_idm_training":
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
    for key, value in expected.items():
        if report.get(key) != value:
            raise TestViewIsolationError(
                "blocked report field changed: {}".format(
                    key
                )
            )
    return report


def discover_phase3_r2_tests(
    root: Path,
) -> Tuple[str, ...]:
    base = Path(root).resolve()
    discovered = {
        path.relative_to(base).as_posix()
        for pattern in PHASE3_R2_PATTERNS
        for path in base.glob(pattern)
    }
    return tuple(sorted(discovered))


def validate_closed_world_test_scope(
    root: Path,
) -> Dict[str, Any]:
    discovered = discover_phase3_r2_tests(
        root
    )
    expected = tuple(
        sorted(
            EXPECTED_PHASE3_R2_TEST_PATHS
        )
    )
    if discovered != expected:
        missing = tuple(
            sorted(
                set(expected)
                - set(discovered)
            )
        )
        extra = tuple(
            sorted(
                set(discovered)
                - set(expected)
            )
        )
        raise TestViewIsolationError(
            "frozen Phase3 r2.x scope changed; "
            "missing={!r}, extra={!r}".format(
                missing,
                extra,
            )
        )
    return {
        "patterns":
            list(PHASE3_R2_PATTERNS),
        "discovered":
            list(discovered),
        "count":
            len(discovered),
        "stage_d_test_visible":
            ORIGINAL_STAGE_D_TEST
            in discovered,
        "resume1_test_visible":
            RESUME1_TEST
            in discovered,
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
        raise TestViewIsolationError(
            "base test gate differs from base evidence commit"
        )
    report = load_json(path)
    if report.get("verdict") != "PASS":
        raise TestViewIsolationError(
            "base test gate is not PASS"
        )
    if int(
        report.get(
            "test_file_count",
            -1,
        )
    ) != EXPECTED_BASE_TEST_FILES:
        raise TestViewIsolationError(
            "base test-file count changed"
        )
    if int(
        report.get(
            "passed_test_count",
            -1,
        )
    ) != EXPECTED_BASE_PASSED:
        raise TestViewIsolationError(
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
        raise TestViewIsolationError(
            "base test manifest changed"
        )
    for relative, expected in manifest.items():
        path = repository_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256_file(path) != expected:
            raise TestViewIsolationError(
                "historical test changed: {}".format(
                    relative
                )
            )
    return report


def validate_initial_worktree(
    root: Path,
) -> Dict[str, Any]:
    repository_root = Path(root).resolve()
    branch = git_output(
        repository_root,
        "branch",
        "--show-current",
    )
    if branch != "Experiment1":
        raise TestViewIsolationError(
            "Resume1 requires Experiment1"
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
    if head != BASE_EVIDENCE_COMMIT:
        raise TestViewIsolationError(
            "initial local HEAD changed: {}".format(
                head
            )
        )
    if remote != BASE_EVIDENCE_COMMIT:
        raise TestViewIsolationError(
            "initial remote HEAD changed: {}".format(
                remote
            )
        )

    original_sha = validate_original_files(
        repository_root
    )
    blocked = validate_original_blocked(
        repository_root
    )
    validate_base_test_gate(
        repository_root
    )

    expected = tuple(
        sorted(
            (
                *ORIGINAL_IMPLEMENTATION_PATHS,
                ORIGINAL_BLOCKED_PATH,
                *RESUME1_IMPLEMENTATION_PATHS,
            )
        )
    )
    observed = status_paths(
        repository_root
    )
    if observed != expected:
        raise TestViewIsolationError(
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
        raise TestViewIsolationError(
            "DeformableRavens commit changed"
        )
    if status_paths(submodule):
        raise TestViewIsolationError(
            "DeformableRavens worktree is dirty"
        )
    return {
        "branch": branch,
        "head": head,
        "origin_experiment1": remote,
        "original_file_sha256":
            original_sha,
        "original_blocked_sha256":
            ORIGINAL_BLOCKED_SHA256,
        "original_blocked":
            blocked,
        "expected_untracked_paths":
            list(expected),
        "submodule_commit":
            EXPECTED_SUBMODULE_COMMIT,
    }


def validate_original_implementation_commit(
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
        parent=BASE_EVIDENCE_COMMIT,
        subject=ORIGINAL_IMPLEMENTATION_MESSAGE,
        paths=ORIGINAL_IMPLEMENTATION_PATHS,
    )
    validate_original_files(
        repository_root
    )
    return {
        "original_implementation_commit":
            commit,
        "parent":
            BASE_EVIDENCE_COMMIT,
    }


def validate_blocked_provenance_commit(
    root: Path,
) -> Dict[str, Any]:
    repository_root = Path(root).resolve()
    commit = git_output(
        repository_root,
        "rev-parse",
        "HEAD",
    )
    implementation = commit_parent(
        repository_root,
        commit,
    )
    assert_commit_shape(
        repository_root,
        commit=commit,
        parent=implementation,
        subject=ORIGINAL_BLOCKED_MESSAGE,
        paths=(ORIGINAL_BLOCKED_PATH,),
    )
    assert_commit_shape(
        repository_root,
        commit=implementation,
        parent=BASE_EVIDENCE_COMMIT,
        subject=ORIGINAL_IMPLEMENTATION_MESSAGE,
        paths=ORIGINAL_IMPLEMENTATION_PATHS,
    )
    committed = subprocess.check_output(
        [
            "git",
            "show",
            "{}:{}".format(
                commit,
                ORIGINAL_BLOCKED_PATH,
            ),
        ],
        cwd=str(repository_root),
    )
    if sha256_bytes(
        committed
    ) != ORIGINAL_BLOCKED_SHA256:
        raise TestViewIsolationError(
            "blocked provenance blob changed"
        )
    validate_original_blocked(
        repository_root
    )
    return {
        "original_implementation_commit":
            implementation,
        "blocked_provenance_commit":
            commit,
    }


def validate_resume1_implementation_commit(
    root: Path,
) -> Dict[str, Any]:
    repository_root = Path(root).resolve()
    commit = git_output(
        repository_root,
        "rev-parse",
        "HEAD",
    )
    blocked_commit = commit_parent(
        repository_root,
        commit,
    )
    assert_commit_shape(
        repository_root,
        commit=commit,
        parent=blocked_commit,
        subject=RESUME1_IMPLEMENTATION_MESSAGE,
        paths=RESUME1_IMPLEMENTATION_PATHS,
    )
    provenance = validate_blocked_provenance_commit_at(
        repository_root,
        blocked_commit,
    )
    return {
        **provenance,
        "resume1_implementation_commit":
            commit,
    }


def validate_blocked_provenance_commit_at(
    root: Path,
    commit: str,
) -> Dict[str, Any]:
    repository_root = Path(root).resolve()
    implementation = commit_parent(
        repository_root,
        commit,
    )
    assert_commit_shape(
        repository_root,
        commit=commit,
        parent=implementation,
        subject=ORIGINAL_BLOCKED_MESSAGE,
        paths=(ORIGINAL_BLOCKED_PATH,),
    )
    assert_commit_shape(
        repository_root,
        commit=implementation,
        parent=BASE_EVIDENCE_COMMIT,
        subject=ORIGINAL_IMPLEMENTATION_MESSAGE,
        paths=ORIGINAL_IMPLEMENTATION_PATHS,
    )
    committed = subprocess.check_output(
        [
            "git",
            "show",
            "{}:{}".format(
                commit,
                ORIGINAL_BLOCKED_PATH,
            ),
        ],
        cwd=str(repository_root),
    )
    if sha256_bytes(
        committed
    ) != ORIGINAL_BLOCKED_SHA256:
        raise TestViewIsolationError(
            "blocked provenance blob changed"
        )
    return {
        "original_implementation_commit":
            implementation,
        "blocked_provenance_commit":
            commit,
    }


def create_frozen_clone(
    *,
    source_root: Path,
    destination: Path,
    base_commit: str = BASE_EVIDENCE_COMMIT,
    submodule_commit: str = EXPECTED_SUBMODULE_COMMIT,
) -> Dict[str, Any]:
    source = Path(source_root).resolve()
    target = Path(destination).resolve()
    if target.exists():
        raise FileExistsError(
            "frozen clone destination exists: {}".format(
                target
            )
        )
    source_submodule = (
        source
        / "external/deformable-ravens"
    )
    if not source_submodule.is_dir():
        raise FileNotFoundError(
            source_submodule
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
                base_commit,
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
                base_commit,
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
                base_commit,
            ],
            cwd=str(target),
            check=True,
        )
        source_origin = subprocess.run(
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
            source_origin.returncode == 0
            and source_origin.stdout.strip()
        ):
            subprocess.run(
                [
                    "git",
                    "remote",
                    "set-url",
                    "origin",
                    source_origin.stdout.strip(),
                ],
                cwd=str(target),
                check=True,
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
                "-B",
                "ccda-cable",
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
        subprocess.run(
            [
                "git",
                "update-ref",
                "refs/remotes/origin/ccda-cable",
                submodule_commit,
            ],
            cwd=str(target_submodule),
            check=True,
        )
        source_submodule_origin = subprocess.run(
            [
                "git",
                "config",
                "--get",
                "remote.origin.url",
            ],
            cwd=str(source_submodule),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        if (
            source_submodule_origin.returncode == 0
            and source_submodule_origin.stdout.strip()
        ):
            subprocess.run(
                [
                    "git",
                    "remote",
                    "set-url",
                    "origin",
                    source_submodule_origin.stdout.strip(),
                ],
                cwd=str(target_submodule),
                check=True,
            )

        if git_output(
            target,
            "branch",
            "--show-current",
        ) != "Experiment1":
            raise TestViewIsolationError(
                "frozen clone branch changed"
            )
        if git_output(
            target,
            "rev-parse",
            "HEAD",
        ) != base_commit:
            raise TestViewIsolationError(
                "frozen clone HEAD changed"
            )
        if git_output(
            target,
            "rev-parse",
            "origin/Experiment1",
        ) != base_commit:
            raise TestViewIsolationError(
                "frozen clone origin/Experiment1 changed"
            )
        if status_paths(target):
            raise TestViewIsolationError(
                "frozen clone is dirty"
            )
        if git_output(
            target_submodule,
            "rev-parse",
            "HEAD",
        ) != submodule_commit:
            raise TestViewIsolationError(
                "frozen clone submodule changed"
            )
        if status_paths(
            target_submodule
        ):
            raise TestViewIsolationError(
                "frozen clone submodule is dirty"
            )

        scope = (
            validate_closed_world_test_scope(
                target
            )
        )
        if scope[
            "stage_d_test_visible"
        ]:
            raise TestViewIsolationError(
                "Stage-D test entered frozen clone"
            )
        if scope[
            "resume1_test_visible"
        ]:
            raise TestViewIsolationError(
                "Resume1 test entered frozen clone"
            )
        return {
            "source_root":
                str(source),
            "clone_root":
                str(target),
            "clone_branch":
                "Experiment1",
            "clone_head":
                base_commit,
            "clone_origin_experiment1":
                base_commit,
            "clone_status_clean":
                True,
            "clone_submodule_commit":
                submodule_commit,
            "clone_submodule_clean":
                True,
            "closed_world_scope":
                scope,
            "uses_network":
                False,
            "uses_ignore":
                False,
            "uses_k_expression":
                False,
            "uses_deselection":
                False,
            "renames_current_tests":
                False,
            "mutates_current_worktree":
                False,
        }
    except Exception:
        shutil.rmtree(
            target,
            ignore_errors=True,
        )
        raise


def validate_frozen_manifest(
    *,
    clone_root: Path,
    manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    root = Path(clone_root).resolve()
    if len(manifest) != EXPECTED_BASE_TEST_FILES:
        raise TestViewIsolationError(
            "frozen manifest size changed"
        )
    observed = {}
    for relative, expected in manifest.items():
        path = root / str(relative)
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != expected:
            raise TestViewIsolationError(
                "frozen clone test changed: {}".format(
                    relative
                )
            )
        observed[str(relative)] = actual
    scope = validate_closed_world_test_scope(
        root
    )
    return {
        "manifest_sha256":
            observed,
        "manifest_count":
            len(observed),
        "closed_world_scope":
            scope,
    }
