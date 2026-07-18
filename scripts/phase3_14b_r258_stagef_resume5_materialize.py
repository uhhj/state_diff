#!/usr/bin/env python3
"""Materialize add-only Stage-F Resume5 wrappers from committed Resume4 sources.

Resume5 repairs only Python import bootstrap and versioned admission.  It copies
five committed Resume4 execution wrappers into new Resume5 paths, replaces the
Resume4 admission call with Resume5 admission, inserts a repository-root
``sys.path`` bootstrap before every ``ccda_phase3`` import in generated Python
entrypoints and Python heredocs, and exports ``PYTHONPATH`` from the generated
shell wrapper.  Previous files are never edited.

Any ambiguous heredoc, source mutation, destination collision, unexpected Git
state, missing bootstrap, compile failure, or shell syntax failure causes a
fail-closed rollback of every newly written path.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple


# This script is itself a directly executable Python entrypoint.  Keep its
# bootstrap before any future ccda_phase3 import is ever added.
_RESUME5_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_RESUME5_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_RESUME5_REPO_ROOT))

EXPECTED_INPUT_HEAD = "23c49349c842a96aefadb91a2c9eac4ea78c131f"
EXPECTED_REMOTE_HEAD = "6758ea7ad800667a436b0243d3b1f6c63256d854"
EXPECTED_SUBMODULE = "633a88752445cf5d6776ed374fdbbdb35f93050c"
REQUIRED_ANCESTORS: Tuple[str, ...] = (
    "578708d9cd10d1abd1b052ee338dae7ee9cae011",
    "abe6b8020bfea9d72c1e54bf2e866ee5c285d632",
    "2cab275171bdcc671f40f86092f19a357226dd18",
    "ce88ac14159c7073ab65a4004e49c748d51fab8d",
    "23c49349c842a96aefadb91a2c9eac4ea78c131f",
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
RESUME3_BLOCKED_REPORT = "reports/phase3_14b_r258_stagef_resume3_blocked_summary.json"
RESUME4_BLOCKED_REPORT = "reports/phase3_14b_r258_stagef_resume4_blocked_summary.json"
RESUME3_BLOCKED_COMMIT = "2cab275171bdcc671f40f86092f19a357226dd18"
RESUME4_BLOCKED_COMMIT = EXPECTED_INPUT_HEAD
MATERIALIZATION_MANIFEST = "reports/phase3_14b_r258_stagef_resume5_materialization.json"

OLD_VALIDATOR_NAME = "validate_resume4_initial_state"
NEW_VALIDATOR_NAME = "validate_resume5_initial_state"
OLD_ADMISSION_MODULE = "ccda_phase3.phase314b_r258_stagef_resume4_head_admission"
NEW_ADMISSION_MODULE = "ccda_phase3.phase314b_r258_stagef_resume5_head_admission"
PYTHON_BOOTSTRAP_MARKER = "_RESUME5_REPO_ROOT"
SHELL_BOOTSTRAP_MARKER = "CCDA_REPO_ROOT"

FILE_BOOTSTRAP = '''\n# Resume5 direct-entrypoint import bootstrap.\nimport sys as _resume5_sys\nfrom pathlib import Path as _Resume5Path\n_RESUME5_REPO_ROOT = _Resume5Path(__file__).resolve().parents[1]\nif str(_RESUME5_REPO_ROOT) not in _resume5_sys.path:\n    _resume5_sys.path.insert(0, str(_RESUME5_REPO_ROOT))\n'''

HEREDOC_BOOTSTRAP = '''\n# Resume5 heredoc import bootstrap.\nimport os as _resume5_os\nimport sys as _resume5_sys\n_RESUME5_REPO_ROOT = _resume5_os.environ.get("CCDA_REPO_ROOT", "/data/state_diff2")\nif _RESUME5_REPO_ROOT not in _resume5_sys.path:\n    _resume5_sys.path.insert(0, _RESUME5_REPO_ROOT)\n'''

SHELL_ENV_BOOTSTRAP = '''\n# Resume5 subprocess import bootstrap.\nexport CCDA_REPO_ROOT="/data/state_diff2"\nexport PYTHONPATH="${CCDA_REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"\nexport PYTHONDONTWRITEBYTECODE=1\n'''


@dataclass(frozen=True)
class CopySpec:
    source: str
    destination: str


COPY_SPECS: Tuple[CopySpec, ...] = (
    CopySpec(
        "scripts/phase3_14b_r258_stagef_resume4_test_gate.py",
        "scripts/phase3_14b_r258_stagef_resume5_test_gate.py",
    ),
    CopySpec(
        "scripts/phase3_14b_r258_stagef_resume4_worker.py",
        "scripts/phase3_14b_r258_stagef_resume5_worker.py",
    ),
    CopySpec(
        "scripts/phase3_14b_r258_stagef_resume4_run_calibration.py",
        "scripts/phase3_14b_r258_stagef_resume5_run_calibration.py",
    ),
    CopySpec(
        "scripts/phase3_14b_r258_stagef_resume4_blocked.py",
        "scripts/phase3_14b_r258_stagef_resume5_blocked.py",
    ),
    CopySpec(
        "scripts/phase3_14b_r258_stagef_resume4_run.sh",
        "scripts/phase3_14b_r258_stagef_resume5_run.sh",
    ),
)

SUPPORT_PATHS: Tuple[str, ...] = (
    "ccda_phase3/phase314b_r258_stagef_resume5_head_admission.py",
    "ccda_phase3/phase314b_r258_stagef_resume5_preflight_evidence.py",
    "scripts/phase3_14b_r258_stagef_resume5_materialize.py",
    "scripts/phase3_14b_r258_stagef_resume5_preflight.py",
    "tests/test_phase3_14b_r258_stagef_resume5_import_bootstrap.py",
)

ALL_IMPLEMENTATION_PATHS = frozenset(
    tuple(spec.destination for spec in COPY_SPECS)
    + SUPPORT_PATHS
    + (MATERIALIZATION_MANIFEST,)
)

PYTHON_ENTRYPOINT_PATHS: Tuple[str, ...] = (
    "scripts/phase3_14b_r258_stagef_resume5_test_gate.py",
    "scripts/phase3_14b_r258_stagef_resume5_worker.py",
    "scripts/phase3_14b_r258_stagef_resume5_run_calibration.py",
    "scripts/phase3_14b_r258_stagef_resume5_blocked.py",
    "scripts/phase3_14b_r258_stagef_resume5_materialize.py",
    "scripts/phase3_14b_r258_stagef_resume5_preflight.py",
)

HEREDOC_START_RE = re.compile(
    r"<<(?P<strip_tabs>-?)(?P<quote>['\"]?)(?P<delimiter>[A-Za-z_][A-Za-z0-9_]*)(?P=quote)"
)


class MaterializationError(RuntimeError):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def run_git_bytes(repo: Path, *args: str, check: bool = True) -> bytes:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(repo),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and proc.returncode != 0:
        raise MaterializationError(
            "git command failed: "
            f"git {' '.join(args)}; returncode={proc.returncode}; "
            f"stderr={proc.stderr.decode('utf-8', 'replace')}"
        )
    return proc.stdout


def run_git(repo: Path, *args: str, check: bool = True) -> str:
    return run_git_bytes(repo, *args, check=check).decode("utf-8", "strict").rstrip("\n")


def require_ancestor(repo: Path, ancestor: str, descendant: str) -> None:
    proc = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=str(repo),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise MaterializationError(
            f"required ancestor missing: ancestor={ancestor}, descendant={descendant}"
        )


def _insertion_line_after_preamble(source: str) -> int:
    """Return a zero-based line index after docstring and future imports."""

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise MaterializationError(f"Python source is not parseable: {exc}") from exc

    insertion_line = 0
    body = list(tree.body)
    index = 0
    if body and isinstance(body[0], ast.Expr):
        value = body[0].value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            insertion_line = int(getattr(body[0], "end_lineno", body[0].lineno))
            index = 1
    while index < len(body):
        node = body[index]
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            insertion_line = int(getattr(node, "end_lineno", node.lineno))
            index += 1
            continue
        break
    return insertion_line


def ensure_python_bootstrap(source: str, *, heredoc: bool = False) -> Tuple[str, bool]:
    if PYTHON_BOOTSTRAP_MARKER in source:
        return source, False
    snippet = HEREDOC_BOOTSTRAP if heredoc else FILE_BOOTSTRAP
    insertion_line = _insertion_line_after_preamble(source)
    lines = source.splitlines(keepends=True)
    lines.insert(insertion_line, snippet)
    transformed = "".join(lines)
    try:
        ast.parse(transformed)
    except SyntaxError as exc:
        raise MaterializationError(
            f"bootstrap insertion produced invalid Python: {exc}"
        ) from exc
    first_ccda = transformed.find("from ccda_phase3")
    if first_ccda < 0:
        first_ccda = transformed.find("import ccda_phase3")
    marker = transformed.find(PYTHON_BOOTSTRAP_MARKER)
    if first_ccda >= 0 and marker > first_ccda:
        raise MaterializationError("bootstrap was inserted after ccda_phase3 import")
    return transformed, True


def replace_stage_literals(text: str) -> str:
    replacements = (
        (
            "phase3_14b_r258_stagef_resume4",
            "phase3_14b_r258_stagef_resume5",
        ),
        (
            "phase314b_r258_stagef_resume4",
            "phase314b_r258_stagef_resume5",
        ),
        (
            "Phase3.14b-r2.5.8 Stage F Resume4",
            "Phase3.14b-r2.5.8 Stage F Resume5",
        ),
        (
            "Versioned Implementation-Head Admission Recovery and Scientific Continuation",
            "Script Import Bootstrap Recovery and Scientific Continuation",
        ),
    )
    result = text
    for old, new in replacements:
        result = result.replace(old, new)
    return result


def replace_validator(text: str) -> Tuple[str, int]:
    old_count = text.count(OLD_VALIDATOR_NAME)
    module_count = text.count(OLD_ADMISSION_MODULE)
    transformed = text.replace(OLD_ADMISSION_MODULE, NEW_ADMISSION_MODULE)
    transformed = transformed.replace(OLD_VALIDATOR_NAME, NEW_VALIDATOR_NAME)
    if OLD_VALIDATOR_NAME in transformed or OLD_ADMISSION_MODULE in transformed:
        raise MaterializationError("Resume4 admission token remains after replacement")
    if old_count < 1:
        return transformed, 0
    if NEW_VALIDATOR_NAME not in transformed:
        raise MaterializationError("Resume5 validator token missing after replacement")
    if module_count > 0 and NEW_ADMISSION_MODULE not in transformed:
        raise MaterializationError("Resume5 admission module missing after replacement")
    return transformed, old_count


def transform_python_source(source_rel: str, source: str) -> Tuple[str, int, bool]:
    transformed = replace_stage_literals(source)
    transformed, calls = replace_validator(transformed)
    transformed, inserted = ensure_python_bootstrap(transformed, heredoc=False)
    try:
        ast.parse(transformed)
    except SyntaxError as exc:
        raise MaterializationError(
            f"transformed Python is invalid for {source_rel}: {exc}"
        ) from exc
    return transformed, calls, inserted


def _heredoc_terminator_matches(line: str, delimiter: str, *, strip_tabs: bool) -> bool:
    candidate = line.rstrip("\r\n")
    if strip_tabs:
        candidate = candidate.lstrip("\t")
    return candidate == delimiter


def _runtime_heredoc_body(lines: Sequence[str], *, strip_tabs: bool) -> str:
    if not strip_tabs:
        return "".join(lines)
    return "".join(line.lstrip("\t") for line in lines)


def patch_shell_python_heredocs(source: str) -> Tuple[str, int, int]:
    lines = source.splitlines(keepends=True)
    output: List[str] = []
    call_count = 0
    bootstrap_count = 0
    index = 0

    while index < len(lines):
        line = lines[index]
        matches = list(HEREDOC_START_RE.finditer(line))
        if len(matches) > 1:
            raise MaterializationError(
                "multiple heredoc operators on one shell line are unsupported"
            )
        output.append(line)
        if not matches:
            index += 1
            continue

        match = matches[0]
        delimiter = match.group("delimiter")
        strip_tabs = match.group("strip_tabs") == "-"
        terminator_index = index + 1
        while terminator_index < len(lines):
            if _heredoc_terminator_matches(
                lines[terminator_index], delimiter, strip_tabs=strip_tabs
            ):
                break
            terminator_index += 1
        if terminator_index >= len(lines):
            raise MaterializationError(
                f"unterminated shell heredoc delimiter: {delimiter!r}"
            )

        body_lines = lines[index + 1 : terminator_index]
        runtime_body = _runtime_heredoc_body(body_lines, strip_tabs=strip_tabs)
        needs_python_transform = (
            OLD_VALIDATOR_NAME in runtime_body
            or OLD_ADMISSION_MODULE in runtime_body
            or "ccda_phase3" in runtime_body
        )
        if needs_python_transform:
            try:
                ast.parse(runtime_body)
            except SyntaxError as exc:
                raise MaterializationError(
                    f"heredoc {delimiter!r} containing ccda/admission tokens is not Python: {exc}"
                ) from exc
            transformed = replace_stage_literals(runtime_body)
            transformed, calls = replace_validator(transformed)
            transformed, inserted = ensure_python_bootstrap(transformed, heredoc=True)
            call_count += calls
            bootstrap_count += int(inserted)
            output.append(transformed)
        else:
            output.extend(body_lines)

        output.append(lines[terminator_index])
        index = terminator_index + 1

    transformed_shell = "".join(output)
    if OLD_VALIDATOR_NAME in transformed_shell or OLD_ADMISSION_MODULE in transformed_shell:
        raise MaterializationError(
            "Resume4 admission token remains outside a safely patchable Python heredoc"
        )
    return transformed_shell, call_count, bootstrap_count


def ensure_shell_environment_bootstrap(source: str) -> Tuple[str, bool]:
    if SHELL_BOOTSTRAP_MARKER in source and "PYTHONPATH" in source:
        return source, False
    lines = source.splitlines(keepends=True)
    insertion = 1 if lines and lines[0].startswith("#!") else 0
    lines.insert(insertion, SHELL_ENV_BOOTSTRAP)
    return "".join(lines), True


def transform_source(source_rel: str, text: str) -> Tuple[str, int, int, int]:
    if source_rel.endswith(".py"):
        transformed, calls, inserted = transform_python_source(source_rel, text)
        return transformed, calls, int(inserted), 0
    if source_rel.endswith(".sh"):
        transformed = replace_stage_literals(text)
        transformed, calls, heredoc_bootstraps = patch_shell_python_heredocs(transformed)
        transformed, shell_inserted = ensure_shell_environment_bootstrap(transformed)
        return transformed, calls, heredoc_bootstraps, int(shell_inserted)
    raise MaterializationError(f"unsupported wrapper source type: {source_rel}")


def validate_initial_repo(repo: Path) -> str:
    if run_git(repo, "rev-parse", "--show-toplevel") != str(repo.resolve()):
        raise MaterializationError("--repo must be the Git top-level directory")
    if run_git(repo, "branch", "--show-current") != "Experiment1":
        raise MaterializationError("expected branch Experiment1")
    head = run_git(repo, "rev-parse", "HEAD")
    if head != EXPECTED_INPUT_HEAD:
        raise MaterializationError(
            f"expected Resume4 blocked-provenance HEAD {EXPECTED_INPUT_HEAD}, got {head}"
        )
    for ancestor in REQUIRED_ANCESTORS:
        require_ancestor(repo, ancestor, head)
    remote = run_git(repo, "rev-parse", "refs/remotes/origin/Experiment1")
    if remote != EXPECTED_REMOTE_HEAD:
        raise MaterializationError(
            f"expected origin/Experiment1 {EXPECTED_REMOTE_HEAD}, got {remote}"
        )
    if run_git_bytes(
        repo, "status", "--porcelain=v1", "-z", "--untracked-files=all"
    ):
        raise MaterializationError("main worktree must be clean before materialization")

    gitlink = run_git(repo, "rev-parse", "HEAD:external/deformable-ravens")
    if gitlink != EXPECTED_SUBMODULE:
        raise MaterializationError(
            f"expected submodule gitlink {EXPECTED_SUBMODULE}, got {gitlink}"
        )
    submodule = repo / "external/deformable-ravens"
    if not submodule.is_dir():
        raise MaterializationError("DeformableRavens submodule directory is missing")
    if run_git(submodule, "rev-parse", "HEAD") != EXPECTED_SUBMODULE:
        raise MaterializationError("DeformableRavens worktree HEAD changed")
    if run_git_bytes(
        submodule, "status", "--porcelain=v1", "-z", "--untracked-files=all"
    ):
        raise MaterializationError("DeformableRavens worktree is not clean")
    return head


def validate_prior_reports(repo: Path) -> Tuple[str, str]:
    for relpath, expected in IMMUTABLE_REPORT_SHA256.items():
        path = repo / relpath
        if not path.is_file():
            raise MaterializationError(f"immutable report missing: {relpath}")
        actual = sha256_path(path)
        if actual != expected:
            raise MaterializationError(
                f"immutable report changed: {relpath}: {actual} != {expected}"
            )

    hashes = []
    for commit, relpath in (
        (RESUME3_BLOCKED_COMMIT, RESUME3_BLOCKED_REPORT),
        (RESUME4_BLOCKED_COMMIT, RESUME4_BLOCKED_REPORT),
    ):
        report = repo / relpath
        if not report.is_file():
            raise MaterializationError(f"blocked report missing: {relpath}")
        committed = run_git_bytes(repo, "show", f"{commit}:{relpath}")
        current = report.read_bytes()
        if current != committed:
            raise MaterializationError(
                f"blocked report differs from {commit}: {relpath}"
            )
        hashes.append(sha256_bytes(current))
    return hashes[0], hashes[1]


def validate_source_blobs(repo: Path) -> Mapping[str, bytes]:
    result: MutableMapping[str, bytes] = {}
    for spec in COPY_SPECS:
        source = repo / spec.source
        destination = repo / spec.destination
        if not source.is_file():
            raise MaterializationError(f"required Resume4 source missing: {spec.source}")
        if destination.exists():
            raise MaterializationError(
                f"refusing to overwrite existing Resume5 path: {spec.destination}"
            )
        current = source.read_bytes()
        committed = run_git_bytes(repo, "show", f"{EXPECTED_INPUT_HEAD}:{spec.source}")
        if current != committed:
            raise MaterializationError(
                f"Resume4 source differs from {EXPECTED_INPUT_HEAD}: {spec.source}"
            )
        result[spec.source] = current
    return result


def copy_support_files(repo: Path, package_root: Path) -> List[Path]:
    written: List[Path] = []
    for relpath in SUPPORT_PATHS:
        source = package_root / relpath
        destination = repo / relpath
        if not source.is_file():
            raise MaterializationError(f"package support file missing: {relpath}")
        if destination.exists():
            raise MaterializationError(f"refusing to overwrite existing path: {relpath}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        if os.access(source, os.X_OK):
            destination.chmod(source.stat().st_mode)
        written.append(destination)
    return written


def _validate_written_bootstraps(repo: Path) -> Tuple[int, int]:
    python_count = 0
    for relpath in PYTHON_ENTRYPOINT_PATHS:
        path = repo / relpath
        if not path.is_file():
            raise MaterializationError(f"Python entrypoint missing: {relpath}")
        text = path.read_text(encoding="utf-8")
        if PYTHON_BOOTSTRAP_MARKER not in text:
            raise MaterializationError(f"Python entrypoint lacks bootstrap: {relpath}")
        first_ccda = text.find("from ccda_phase3")
        if first_ccda < 0:
            first_ccda = text.find("import ccda_phase3")
        marker = text.find(PYTHON_BOOTSTRAP_MARKER)
        if first_ccda >= 0 and marker > first_ccda:
            raise MaterializationError(
                f"Python bootstrap occurs after ccda_phase3 import: {relpath}"
            )
        python_count += 1

    shell_path = repo / "scripts/phase3_14b_r258_stagef_resume5_run.sh"
    shell_text = shell_path.read_text(encoding="utf-8")
    if SHELL_BOOTSTRAP_MARKER not in shell_text or "PYTHONPATH" not in shell_text:
        raise MaterializationError("Resume5 shell wrapper lacks PYTHONPATH bootstrap")
    return python_count, 1


def materialize(repo: Path, package_root: Path) -> Mapping[str, object]:
    repo = repo.resolve()
    package_root = package_root.resolve()
    input_head = validate_initial_repo(repo)
    resume3_sha, resume4_sha = validate_prior_reports(repo)
    source_bytes = validate_source_blobs(repo)

    written: List[Path] = []
    source_hashes: Dict[str, str] = {
        path: sha256_bytes(data) for path, data in source_bytes.items()
    }
    validator_call_count = 0
    generated_python_bootstraps = 0
    shell_heredoc_bootstraps = 0
    shell_environment_bootstraps = 0

    try:
        for spec in COPY_SPECS:
            try:
                source_text = source_bytes[spec.source].decode("utf-8")
            except UnicodeDecodeError as exc:
                raise MaterializationError(
                    f"Resume4 source is not UTF-8: {spec.source}"
                ) from exc
            transformed, calls, py_bootstraps, shell_bootstraps = transform_source(
                spec.source, source_text
            )
            validator_call_count += calls
            if spec.source.endswith(".py"):
                generated_python_bootstraps += py_bootstraps
            else:
                shell_heredoc_bootstraps += py_bootstraps
                shell_environment_bootstraps += shell_bootstraps
            destination = repo / spec.destination
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("w", encoding="utf-8", newline="") as handle:
                handle.write(transformed)
            if os.access(repo / spec.source, os.X_OK):
                destination.chmod((repo / spec.source).stat().st_mode)
            written.append(destination)

        if validator_call_count < 1:
            raise MaterializationError(
                "no Resume4 validate_resume4_initial_state call was found"
            )
        if shell_environment_bootstraps != 1:
            raise MaterializationError(
                "Resume5 shell environment bootstrap was not inserted exactly once"
            )

        written.extend(copy_support_files(repo, package_root))

        for path in written:
            if path.suffix == ".py":
                compile(path.read_text(encoding="utf-8"), str(path), "exec")

        shell_path = repo / "scripts/phase3_14b_r258_stagef_resume5_run.sh"
        shell_check = subprocess.run(
            ["bash", "-n", str(shell_path)],
            cwd=str(repo),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if shell_check.returncode != 0:
            raise MaterializationError(
                "Resume5 shell wrapper failed bash -n: " + shell_check.stderr
            )

        python_bootstrap_count, shell_bootstrap_count = _validate_written_bootstraps(repo)

        generated_hashes = {
            str(path.relative_to(repo)): sha256_path(path)
            for path in sorted(written)
        }
        expected_hashed_paths = ALL_IMPLEMENTATION_PATHS - {MATERIALIZATION_MANIFEST}
        if frozenset(generated_hashes) != expected_hashed_paths:
            raise MaterializationError(
                "generated path population changed before manifest: "
                f"expected={sorted(expected_hashed_paths)!r}, "
                f"actual={sorted(generated_hashes)!r}"
            )

        manifest_payload: Dict[str, object] = {
            "phase": "Phase3.14b-r2.5.8 Stage F Resume5",
            "title": "Script Import Bootstrap Recovery and Scientific Continuation",
            "operation": "add_only_script_import_bootstrap_recovery",
            "input_head": input_head,
            "remote_head": EXPECTED_REMOTE_HEAD,
            "submodule_head": EXPECTED_SUBMODULE,
            "resume3_blocked_report_sha256": resume3_sha,
            "resume4_blocked_report_sha256": resume4_sha,
            "source_sha256": source_hashes,
            "generated_sha256": generated_hashes,
            "validator_calls_replaced": validator_call_count,
            "generated_python_bootstraps_inserted": generated_python_bootstraps,
            "shell_python_heredoc_bootstraps_inserted": shell_heredoc_bootstraps,
            "shell_environment_bootstraps_inserted": shell_environment_bootstraps,
            "python_entrypoints_bootstrapped": python_bootstrap_count,
            "shell_wrappers_bootstrapped": shell_bootstrap_count,
            "scientific_logic_changed": False,
            "frozen_temporal_files": 58,
            "frozen_temporal_passed": 1503,
        }
        manifest_path = repo / MATERIALIZATION_MANIFEST
        if manifest_path.exists():
            raise MaterializationError(
                f"refusing to overwrite manifest: {MATERIALIZATION_MANIFEST}"
            )
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        written.append(manifest_path)

        status = run_git_bytes(
            repo, "status", "--porcelain=v1", "-z", "--untracked-files=all"
        )
        fields = [field for field in status.split(b"\0") if field]
        observed_paths = set()
        for field in fields:
            if len(field) < 4:
                raise MaterializationError(f"invalid porcelain record: {field!r}")
            observed_paths.add(field[3:].decode("utf-8", "strict"))
        if observed_paths != ALL_IMPLEMENTATION_PATHS:
            raise MaterializationError(
                "materialized worktree population changed: "
                f"expected={sorted(ALL_IMPLEMENTATION_PATHS)!r}, "
                f"actual={sorted(observed_paths)!r}"
            )

        result = dict(manifest_payload)
        result["manifest_path"] = str(manifest_path)
        result["manifest_sha256"] = sha256_path(manifest_path)
        return result
    except Exception:
        for path in reversed(written):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        raise


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("/data/state_diff2"))
    parser.add_argument(
        "--package-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        result = materialize(args.repo, args.package_root)
    except MaterializationError as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
