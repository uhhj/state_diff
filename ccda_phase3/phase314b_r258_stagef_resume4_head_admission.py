"""Versioned implementation-head admission for Stage F Resume4.

The validator intentionally does not hard-code the Resume4 implementation SHA.
Instead, it proves that the current single-parent HEAD is an add-only child of the
frozen Resume3 blocked-provenance commit and that the exact implementation path
population, remote base, submodule identity, reports, generated-file manifest,
and worktree cleanliness remain valid.

No model, feature, fold, integrator, threshold, holdout, probe, or scientific
selection logic is defined here.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple


REPO_ROOT = Path("/data/state_diff2")

EXPECTED_BRANCH = "Experiment1"
EXPECTED_REMOTE_HEAD = "6758ea7ad800667a436b0243d3b1f6c63256d854"
EXPECTED_SUBMODULE_HEAD = "633a88752445cf5d6776ed374fdbbdb35f93050c"
EXPECTED_IMPLEMENTATION_PARENT = "2cab275171bdcc671f40f86092f19a357226dd18"
EXPECTED_IMPLEMENTATION_SUBJECT = (
    "Phase3.14b-r2.5.8 Stage F Resume4: recover versioned head admission"
)

REQUIRED_ANCESTORS: Tuple[str, ...] = (
    "578708d9cd10d1abd1b052ee338dae7ee9cae011",
    "abe6b8020bfea9d72c1e54bf2e866ee5c285d632",
    "2cab275171bdcc671f40f86092f19a357226dd18",
)

IMMUTABLE_REPORT_SHA256: Mapping[str, str] = {
    "reports/phase3_14b_r258_stagef_test_gate_summary.json":
        "7d3eadbd063e2248a17c0d26fcd3d47897c46dc434b98a6692c17c214a9a5bd2",
    "reports/phase3_14b_r258_stagef_blocked_summary.json":
        "3a970c1c36998da92312eb78833e2f21401e8556414c87f4502be253a69f5886",
    "reports/phase3_14b_r258_stagef_resume1_blocked_summary.json":
        "b8a11c88aefd9e70f65d2a0d2afbf88b8feabad5f6c97784e41f4d813513c80f",
    "reports/phase3_14b_r258_stagef_resume2_blocked_summary.json":
        "a143a48cba9ccec0987989d5b589d8bd18960d15ee09aa107ef7525ee41cdb70",
}

RESUME3_BLOCKED_REPORT = (
    "reports/phase3_14b_r258_stagef_resume3_blocked_summary.json"
)
MATERIALIZATION_MANIFEST = (
    "reports/phase3_14b_r258_stagef_resume4_materialization.json"
)

GENERATED_WRAPPER_PATHS: Tuple[str, ...] = (
    "scripts/phase3_14b_r258_stagef_resume4_test_gate.py",
    "scripts/phase3_14b_r258_stagef_resume4_worker.py",
    "scripts/phase3_14b_r258_stagef_resume4_run_calibration.py",
    "scripts/phase3_14b_r258_stagef_resume4_blocked.py",
    "scripts/phase3_14b_r258_stagef_resume4_run.sh",
)

ALLOWED_IMPLEMENTATION_PATHS = frozenset(
    GENERATED_WRAPPER_PATHS
    + (
        "ccda_phase3/phase314b_r258_stagef_resume4_head_admission.py",
        "ccda_phase3/phase314b_r258_stagef_resume4_preflight_evidence.py",
        "scripts/phase3_14b_r258_stagef_resume4_materialize.py",
        "scripts/phase3_14b_r258_stagef_resume4_preflight.py",
        "tests/test_phase3_14b_r258_stagef_resume4_head_admission.py",
        MATERIALIZATION_MANIFEST,
    )
)

LEGACY_VALIDATOR_TOKEN = "validate_initial_worktree"
NEW_VALIDATOR_TOKEN = "validate_resume4_initial_state"


class Resume4AdmissionError(RuntimeError):
    """Raised when the Resume4 implementation boundary is not exact."""


@dataclass(frozen=True)
class Resume4AdmissionEvidence:
    branch: str
    head: str
    parent: str
    subject: str
    remote_head: str
    submodule_gitlink: str
    submodule_worktree_head: str
    implementation_paths: Tuple[str, ...]
    resume3_blocked_report_sha256: str
    generated_file_sha256: Mapping[str, str]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "branch": self.branch,
            "head": self.head,
            "parent": self.parent,
            "subject": self.subject,
            "remote_head": self.remote_head,
            "submodule_gitlink": self.submodule_gitlink,
            "submodule_worktree_head": self.submodule_worktree_head,
            "implementation_paths": list(self.implementation_paths),
            "resume3_blocked_report_sha256": self.resume3_blocked_report_sha256,
            "generated_file_sha256": dict(self.generated_file_sha256),
        }


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_path(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _run_git_bytes(repo: Path, *args: str, check: bool = True) -> bytes:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(repo),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and proc.returncode != 0:
        raise Resume4AdmissionError(
            "git command failed: "
            f"git {' '.join(args)}; returncode={proc.returncode}; "
            f"stderr={proc.stderr.decode('utf-8', 'replace')}"
        )
    return proc.stdout


def _run_git_text(repo: Path, *args: str, check: bool = True) -> str:
    return _run_git_bytes(repo, *args, check=check).decode("utf-8", "strict").rstrip("\n")


def _require_equal(label: str, actual: str, expected: str) -> None:
    if actual != expected:
        raise Resume4AdmissionError(
            f"{label} changed: expected={expected!r}, actual={actual!r}"
        )


def _require_clean(label: str, status: bytes) -> None:
    if status:
        raise Resume4AdmissionError(
            f"{label} is not clean: {status.decode('utf-8', 'replace')!r}"
        )


def _parse_nul_name_status(raw: bytes) -> Tuple[Tuple[str, str], ...]:
    fields = raw.split(b"\0")
    if fields and fields[-1] == b"":
        fields.pop()
    if len(fields) % 2 != 0:
        raise Resume4AdmissionError(
            f"invalid NUL name-status field count: {len(fields)}"
        )
    records = []
    for index in range(0, len(fields), 2):
        status = fields[index].decode("ascii", "strict")
        path = fields[index + 1].decode("utf-8", "strict")
        records.append((status, path))
    return tuple(records)


def _require_single_parent(repo: Path, head: str) -> str:
    parents = _run_git_text(repo, "show", "-s", "--format=%P", head).split()
    if len(parents) != 1:
        raise Resume4AdmissionError(
            f"Resume4 implementation must have exactly one parent, got {parents!r}"
        )
    return parents[0]


def _require_ancestor(repo: Path, ancestor: str, descendant: str) -> None:
    proc = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=str(repo),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise Resume4AdmissionError(
            f"required ancestor missing: ancestor={ancestor}, descendant={descendant}"
        )


def _require_blob_equal_to_commit(repo: Path, commit: str, relpath: str) -> str:
    path = repo / relpath
    if not path.is_file():
        raise Resume4AdmissionError(f"required immutable file is missing: {relpath}")
    committed = _run_git_bytes(repo, "show", f"{commit}:{relpath}")
    current = path.read_bytes()
    if current != committed:
        raise Resume4AdmissionError(
            f"immutable file differs from {commit}: {relpath}"
        )
    return _sha256_bytes(current)


def _load_manifest(repo: Path) -> Mapping[str, Any]:
    path = repo / MATERIALIZATION_MANIFEST
    if not path.is_file():
        raise Resume4AdmissionError(
            f"Resume4 materialization manifest is missing: {MATERIALIZATION_MANIFEST}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise Resume4AdmissionError(
            f"invalid Resume4 materialization manifest: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise Resume4AdmissionError("Resume4 materialization manifest must be an object")
    return payload


def _validate_manifest(repo: Path, manifest: Mapping[str, Any]) -> Mapping[str, str]:
    if manifest.get("phase") != "Phase3.14b-r2.5.8 Stage F Resume4":
        raise Resume4AdmissionError("Resume4 materialization phase changed")
    if manifest.get("operation") != "add_only_versioned_head_admission_recovery":
        raise Resume4AdmissionError("Resume4 materialization operation changed")
    if manifest.get("input_head") != EXPECTED_IMPLEMENTATION_PARENT:
        raise Resume4AdmissionError("Resume4 materialization input HEAD changed")
    if manifest.get("scientific_logic_changed") is not False:
        raise Resume4AdmissionError("Resume4 manifest claims scientific logic changed")
    if manifest.get("frozen_temporal_files") != 58:
        raise Resume4AdmissionError("frozen temporal file count changed")
    if manifest.get("frozen_temporal_passed") != 1503:
        raise Resume4AdmissionError("frozen temporal passed count changed")

    raw_hashes = manifest.get("generated_sha256")
    if not isinstance(raw_hashes, dict):
        raise Resume4AdmissionError("generated_sha256 must be an object")

    expected_hashed_paths = ALLOWED_IMPLEMENTATION_PATHS - {MATERIALIZATION_MANIFEST}
    observed_paths = frozenset(str(path) for path in raw_hashes)
    if observed_paths != expected_hashed_paths:
        raise Resume4AdmissionError(
            "materialization generated path population changed: "
            f"expected={sorted(expected_hashed_paths)!r}, "
            f"actual={sorted(observed_paths)!r}"
        )

    verified: Dict[str, str] = {}
    for relpath in sorted(expected_hashed_paths):
        expected = raw_hashes.get(relpath)
        if not isinstance(expected, str) or len(expected) != 64:
            raise Resume4AdmissionError(
                f"invalid generated SHA256 entry: {relpath}={expected!r}"
            )
        path = repo / relpath
        if not path.is_file():
            raise Resume4AdmissionError(f"generated implementation file missing: {relpath}")
        actual = _sha256_path(path)
        if actual != expected:
            raise Resume4AdmissionError(
                f"generated implementation file changed: {relpath}: "
                f"{actual} != {expected}"
            )
        verified[relpath] = actual
    return verified


def _validate_implementation_changes(
    changes: Sequence[Tuple[str, str]],
) -> Tuple[str, ...]:
    non_additions = tuple(record for record in changes if record[0] != "A")
    if non_additions:
        raise Resume4AdmissionError(
            f"Resume4 implementation is not add-only: {non_additions!r}"
        )
    changed_paths = frozenset(path for _, path in changes)
    if changed_paths != ALLOWED_IMPLEMENTATION_PATHS:
        raise Resume4AdmissionError(
            "Resume4 implementation path population changed: "
            f"expected={sorted(ALLOWED_IMPLEMENTATION_PATHS)!r}, "
            f"actual={sorted(changed_paths)!r}"
        )
    return tuple(sorted(changed_paths))


def _validate_generated_wrappers(repo: Path) -> None:
    total_new_calls = 0
    for relpath in GENERATED_WRAPPER_PATHS:
        path = repo / relpath
        if not path.is_file():
            raise Resume4AdmissionError(f"generated wrapper missing: {relpath}")
        text = path.read_text(encoding="utf-8")
        if LEGACY_VALIDATOR_TOKEN in text:
            raise Resume4AdmissionError(
                f"legacy initial-worktree validator remains in {relpath}"
            )
        total_new_calls += text.count(NEW_VALIDATOR_TOKEN)
    if total_new_calls < 1:
        raise Resume4AdmissionError(
            "no generated Resume4 wrapper invokes the new versioned validator"
        )


def validate_resume4_initial_state(
    repo: Path = REPO_ROOT,
) -> Resume4AdmissionEvidence:
    """Validate the committed Resume4 implementation before any test/science run."""

    repo = repo.resolve()
    top = _run_git_text(repo, "rev-parse", "--show-toplevel")
    _require_equal("repository root", top, str(repo))

    branch = _run_git_text(repo, "branch", "--show-current")
    _require_equal("branch", branch, EXPECTED_BRANCH)

    head = _run_git_text(repo, "rev-parse", "HEAD")
    parent = _require_single_parent(repo, head)
    _require_equal("Resume4 implementation parent", parent, EXPECTED_IMPLEMENTATION_PARENT)

    subject = _run_git_text(repo, "show", "-s", "--format=%s", head)
    _require_equal("Resume4 implementation subject", subject, EXPECTED_IMPLEMENTATION_SUBJECT)

    for ancestor in REQUIRED_ANCESTORS:
        _require_ancestor(repo, ancestor, head)

    remote_head = _run_git_text(repo, "rev-parse", "refs/remotes/origin/Experiment1")
    _require_equal("origin/Experiment1", remote_head, EXPECTED_REMOTE_HEAD)

    _require_clean(
        "main worktree",
        _run_git_bytes(repo, "status", "--porcelain=v1", "-z", "--untracked-files=all"),
    )

    submodule_rel = "external/deformable-ravens"
    submodule_gitlink = _run_git_text(repo, "rev-parse", f"HEAD:{submodule_rel}")
    _require_equal("DeformableRavens gitlink", submodule_gitlink, EXPECTED_SUBMODULE_HEAD)

    submodule = repo / submodule_rel
    if not submodule.is_dir():
        raise Resume4AdmissionError("DeformableRavens submodule directory is missing")
    submodule_head = _run_git_text(submodule, "rev-parse", "HEAD")
    _require_equal("DeformableRavens worktree HEAD", submodule_head, EXPECTED_SUBMODULE_HEAD)
    _require_clean(
        "DeformableRavens worktree",
        _run_git_bytes(
            submodule,
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
        ),
    )

    changes = _parse_nul_name_status(
        _run_git_bytes(
            repo,
            "diff-tree",
            "--no-commit-id",
            "--name-status",
            "-r",
            "-z",
            head,
        )
    )
    changed_paths = _validate_implementation_changes(changes)

    for relpath, expected_sha in IMMUTABLE_REPORT_SHA256.items():
        path = repo / relpath
        if not path.is_file():
            raise Resume4AdmissionError(f"immutable report missing: {relpath}")
        actual_sha = _sha256_path(path)
        if actual_sha != expected_sha:
            raise Resume4AdmissionError(
                f"immutable report changed: {relpath}: {actual_sha} != {expected_sha}"
            )

    resume3_blocked_sha = _require_blob_equal_to_commit(
        repo,
        EXPECTED_IMPLEMENTATION_PARENT,
        RESUME3_BLOCKED_REPORT,
    )

    manifest = _load_manifest(repo)
    generated_hashes = _validate_manifest(repo, manifest)
    _validate_generated_wrappers(repo)

    return Resume4AdmissionEvidence(
        branch=branch,
        head=head,
        parent=parent,
        subject=subject,
        remote_head=remote_head,
        submodule_gitlink=submodule_gitlink,
        submodule_worktree_head=submodule_head,
        implementation_paths=changed_paths,
        resume3_blocked_report_sha256=resume3_blocked_sha,
        generated_file_sha256=generated_hashes,
    )


__all__ = [
    "ALLOWED_IMPLEMENTATION_PATHS",
    "EXPECTED_IMPLEMENTATION_PARENT",
    "EXPECTED_IMPLEMENTATION_SUBJECT",
    "EXPECTED_REMOTE_HEAD",
    "EXPECTED_SUBMODULE_HEAD",
    "Resume4AdmissionError",
    "Resume4AdmissionEvidence",
    "validate_resume4_initial_state",
]
