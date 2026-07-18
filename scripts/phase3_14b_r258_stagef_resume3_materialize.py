#!/usr/bin/env python3
"""Materialize add-only Stage-F Resume3 wrappers from the local Resume2 sources.

The public remote does not contain the user's unpushed Stage-F Resume2 files.
This generator therefore copies the exact local wrappers and applies only:
  * new Resume3 wrapper/report names;
  * the historical/current Resume2 count namespace correction.

It never edits a Resume2 source file and refuses ambiguous source layouts.
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
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


EXPECTED_LOCAL_HEAD = "ebb3ddf9f84a4d41e25430dda417679a2900f92b"
EXPECTED_REMOTE_HEAD = "6758ea7ad800667a436b0243d3b1f6c63256d854"
EXPECTED_SUBMODULE = "633a88752445cf5d6776ed374fdbbdb35f93050c"

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


@dataclass(frozen=True)
class CopySpec:
    source: str
    destination: str


COPY_SPECS: Tuple[CopySpec, ...] = (
    CopySpec(
        "scripts/phase3_14b_r258_stagef_resume2_test_gate.py",
        "scripts/phase3_14b_r258_stagef_resume3_test_gate.py",
    ),
    CopySpec(
        "scripts/phase3_14b_r258_stagef_resume2_worker.py",
        "scripts/phase3_14b_r258_stagef_resume3_worker.py",
    ),
    CopySpec(
        "scripts/phase3_14b_r258_stagef_resume2_run_calibration.py",
        "scripts/phase3_14b_r258_stagef_resume3_run_calibration.py",
    ),
    CopySpec(
        "scripts/phase3_14b_r258_stagef_resume2_blocked.py",
        "scripts/phase3_14b_r258_stagef_resume3_blocked.py",
    ),
    CopySpec(
        "scripts/phase3_14b_r258_stagef_resume2_run.sh",
        "scripts/phase3_14b_r258_stagef_resume3_run.sh",
    ),
)

WRAPPER_NAME_REPLACEMENTS: Tuple[Tuple[str, str], ...] = (
    (
        "phase3_14b_r258_stagef_resume2_test_gate.py",
        "phase3_14b_r258_stagef_resume3_test_gate.py",
    ),
    (
        "phase3_14b_r258_stagef_resume2_worker.py",
        "phase3_14b_r258_stagef_resume3_worker.py",
    ),
    (
        "phase3_14b_r258_stagef_resume2_run_calibration.py",
        "phase3_14b_r258_stagef_resume3_run_calibration.py",
    ),
    (
        "phase3_14b_r258_stagef_resume2_blocked.py",
        "phase3_14b_r258_stagef_resume3_blocked.py",
    ),
    (
        "phase3_14b_r258_stagef_resume2_run.sh",
        "phase3_14b_r258_stagef_resume3_run.sh",
    ),
)

REPORT_REPLACEMENTS: Tuple[Tuple[str, str], ...] = tuple(
    (
        f"phase3_14b_r258_stagef_resume2_{suffix}",
        f"phase3_14b_r258_stagef_resume3_{suffix}",
    )
    for suffix in (
        "test_gate_summary.json",
        "blocked_summary.json",
        "summary.json",
        "contract.json",
        "worker_evidence.json",
        "report.md",
    )
)

PHASE_LABEL_REPLACEMENTS: Tuple[Tuple[str, str], ...] = (
    ("Phase3.14b-r2.5.8 Stage F Resume2", "Phase3.14b-r2.5.8 Stage F Resume3"),
    ("Stage F Resume2 —", "Stage F Resume3 —"),
    ("Stage-F Resume2 wrapper", "Stage-F Resume3 wrapper"),
)


class MaterializationError(RuntimeError):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def run_git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(repo),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise MaterializationError(
            f"git {' '.join(args)} failed ({proc.returncode}): "
            f"{proc.stderr.decode('utf-8', 'replace')}"
        )
    return proc.stdout.decode("utf-8", "strict").strip()


def require_exact_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise MaterializationError(
            f"{label}: expected exactly one occurrence of {old!r}, found {count}"
        )
    return text.replace(old, new, 1)


def replace_optional(text: str, replacements: Iterable[Tuple[str, str]]) -> str:
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def correct_test_count_namespace(text: str) -> str:
    # Import alias: historical Stage-D Resume2 count remains 55.
    import_pattern = re.compile(
        r"(?m)^(?P<indent>\s*)EXPECTED_RESUME2_PASSED,\s*$"
    )
    matches = list(import_pattern.finditer(text))
    if len(matches) != 1:
        raise MaterializationError(
            "test gate must contain exactly one multiline import entry named "
            f"EXPECTED_RESUME2_PASSED; found {len(matches)}"
        )
    match = matches[0]
    replacement = (
        f"{match.group('indent')}EXPECTED_RESUME2_PASSED "
        "as EXPECTED_STAGED_RESUME2_PASSED,"
    )
    text = text[: match.start()] + replacement + text[match.end() :]

    text = require_exact_once(
        text,
        "EXPECTED_RESUME2_PASSED = 48",
        "EXPECTED_STAGEF_RESUME2_PASSED = 48",
        "current Stage-F Resume2 count declaration",
    )
    text = require_exact_once(
        text,
        "CURRENT_RESUME2_TEST: EXPECTED_RESUME2_PASSED,",
        "CURRENT_RESUME2_TEST: EXPECTED_STAGED_RESUME2_PASSED,",
        "historical Stage-D Resume2 mapping",
    )

    assertion_patterns = (
        (
            'resume2_result["passed"] != EXPECTED_RESUME2_PASSED',
            'resume2_result["passed"] != EXPECTED_STAGEF_RESUME2_PASSED',
        ),
        (
            "resume2_result['passed'] != EXPECTED_RESUME2_PASSED",
            "resume2_result['passed'] != EXPECTED_STAGEF_RESUME2_PASSED",
        ),
    )
    found = [(old, new) for old, new in assertion_patterns if old in text]
    if len(found) != 1:
        raise MaterializationError(
            "current Stage-F Resume2 assertion must match exactly one supported form"
        )
    old, new = found[0]
    text = require_exact_once(text, old, new, "current Stage-F Resume2 assertion")
    return text


def transform_source(source_rel: str, text: str) -> str:
    transformed = text
    transformed = replace_optional(transformed, WRAPPER_NAME_REPLACEMENTS)
    transformed = replace_optional(transformed, REPORT_REPLACEMENTS)
    transformed = replace_optional(transformed, PHASE_LABEL_REPLACEMENTS)
    if source_rel.endswith("_test_gate.py"):
        transformed = correct_test_count_namespace(transformed)
    return transformed


def validate_test_gate_ast(text: str) -> None:
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        raise MaterializationError(f"generated test gate is invalid Python: {exc}") from exc

    assigned: Dict[str, object] = {}
    imported_aliases: List[Tuple[str, str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    value: object = None
                    if isinstance(node.value, ast.Constant):
                        value = node.value.value
                    assigned[target.id] = value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            value = node.value.value if isinstance(node.value, ast.Constant) else None
            assigned[node.target.id] = value
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                imported_aliases.append((module, alias.name, alias.asname or alias.name))

    if "EXPECTED_RESUME2_PASSED" in assigned:
        raise MaterializationError(
            "generated test gate still assigns EXPECTED_RESUME2_PASSED"
        )
    if assigned.get("EXPECTED_STAGEF_RESUME2_PASSED") != 48:
        raise MaterializationError(
            "generated test gate does not bind EXPECTED_STAGEF_RESUME2_PASSED=48"
        )
    expected_alias = (
        "ccda_phase3.phase314b_r258_staged_resume2_temporal_views",
        "EXPECTED_RESUME2_PASSED",
        "EXPECTED_STAGED_RESUME2_PASSED",
    )
    if expected_alias not in imported_aliases:
        raise MaterializationError(
            "generated test gate lacks explicit historical Resume2 import alias"
        )
    if "CURRENT_RESUME2_TEST: EXPECTED_STAGED_RESUME2_PASSED," not in text:
        raise MaterializationError("historical mapping is not bound to the 55-count alias")
    if "!= EXPECTED_STAGEF_RESUME2_PASSED" not in text:
        raise MaterializationError("current porcelain assertion is not bound to 48")
    if "EXPECTED_TOTAL_PASSED = 1503" not in text:
        raise MaterializationError("frozen temporal total 1503 is missing")


def validate_repo_state(repo: Path, strict_head: bool) -> None:
    if run_git(repo, "rev-parse", "--show-toplevel") != str(repo.resolve()):
        raise MaterializationError("--repo must be the Git top-level directory")
    branch = run_git(repo, "branch", "--show-current")
    if branch != "Experiment1":
        raise MaterializationError(f"expected branch Experiment1, got {branch!r}")
    if strict_head:
        head = run_git(repo, "rev-parse", "HEAD")
        if head != EXPECTED_LOCAL_HEAD:
            raise MaterializationError(
                f"expected local HEAD {EXPECTED_LOCAL_HEAD}, got {head}"
            )
    remote = run_git(repo, "rev-parse", "origin/Experiment1")
    if remote != EXPECTED_REMOTE_HEAD:
        raise MaterializationError(
            f"expected origin/Experiment1 {EXPECTED_REMOTE_HEAD}, got {remote}"
        )
    gitlink = run_git(repo, "rev-parse", "HEAD:external/deformable-ravens")
    if gitlink != EXPECTED_SUBMODULE:
        raise MaterializationError(
            f"expected submodule gitlink {EXPECTED_SUBMODULE}, got {gitlink}"
        )
    submodule = repo / "external/deformable-ravens"
    if not submodule.is_dir():
        raise MaterializationError("DeformableRavens submodule directory is missing")
    sub_head = run_git(submodule, "rev-parse", "HEAD")
    if sub_head != EXPECTED_SUBMODULE:
        raise MaterializationError(
            f"expected submodule worktree {EXPECTED_SUBMODULE}, got {sub_head}"
        )
    if run_git(submodule, "status", "--porcelain=v1", "--untracked-files=all"):
        raise MaterializationError("DeformableRavens submodule is not clean")


def validate_immutable_reports(repo: Path) -> None:
    for rel, expected in IMMUTABLE_REPORT_SHA256.items():
        path = repo / rel
        if not path.is_file():
            raise MaterializationError(f"immutable report missing: {rel}")
        actual = sha256_path(path)
        if actual != expected:
            raise MaterializationError(
                f"immutable report changed: {rel}: {actual} != {expected}"
            )


def copy_support_files(repo: Path, package_root: Path) -> List[Path]:
    support = (
        "ccda_phase3/phase314b_r258_stagef_resume3_test_count_namespace.py",
        "tests/test_phase3_14b_r258_stagef_resume3_count_namespace.py",
    )
    written: List[Path] = []
    for rel in support:
        src = package_root / rel
        dst = repo / rel
        if not src.is_file():
            raise MaterializationError(f"package support file missing: {rel}")
        if dst.exists():
            raise MaterializationError(f"refusing to overwrite existing path: {rel}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())
        written.append(dst)
    return written


def materialize(repo: Path, package_root: Path, strict_head: bool) -> Mapping[str, object]:
    validate_repo_state(repo, strict_head=strict_head)
    validate_immutable_reports(repo)

    source_hashes: MutableMapping[str, str] = {}
    source_bytes: MutableMapping[str, bytes] = {}
    for spec in COPY_SPECS:
        src = repo / spec.source
        dst = repo / spec.destination
        if not src.is_file():
            raise MaterializationError(f"required Resume2 source missing: {spec.source}")
        if dst.exists():
            raise MaterializationError(
                f"refusing to overwrite existing Resume3 path: {spec.destination}"
            )
        data = src.read_bytes()
        source_bytes[spec.source] = data
        source_hashes[spec.source] = sha256_bytes(data)

    written: List[Path] = []
    try:
        for spec in COPY_SPECS:
            src_data = source_bytes[spec.source]
            try:
                src_text = src_data.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise MaterializationError(
                    f"source is not UTF-8: {spec.source}"
                ) from exc
            out_text = transform_source(spec.source, src_text)
            dst = repo / spec.destination
            dst.parent.mkdir(parents=True, exist_ok=True)
            with dst.open("w", encoding="utf-8", newline="") as handle:
                handle.write(out_text)
            if os.access(repo / spec.source, os.X_OK):
                dst.chmod((repo / spec.source).stat().st_mode)
            written.append(dst)

        written.extend(copy_support_files(repo, package_root))

        gate_path = repo / "scripts/phase3_14b_r258_stagef_resume3_test_gate.py"
        validate_test_gate_ast(gate_path.read_text(encoding="utf-8"))
        for path in written:
            if path.suffix == ".py":
                compile(path.read_text(encoding="utf-8"), str(path), "exec")
        shell_path = repo / "scripts/phase3_14b_r258_stagef_resume3_run.sh"
        shell_check = subprocess.run(
            ["bash", "-n", str(shell_path)],
            cwd=str(repo),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if shell_check.returncode != 0:
            raise MaterializationError(
                "generated Resume3 shell is invalid: "
                + shell_check.stderr.decode("utf-8", "replace")
            )

        for spec in COPY_SPECS:
            current = sha256_path(repo / spec.source)
            if current != source_hashes[spec.source]:
                raise MaterializationError(
                    f"Resume2 source mutated during materialization: {spec.source}"
                )

        manifest = {
            "phase": "Phase3.14b-r2.5.8 Stage F Resume3",
            "operation": "add_only_namespace_recovery_materialization",
            "source_sha256": dict(source_hashes),
            "generated_sha256": {
                str(path.relative_to(repo)): sha256_path(path) for path in written
            },
            "historical_stage_d_resume2_passed": 55,
            "current_stage_f_resume2_passed": 48,
            "frozen_temporal_files": 58,
            "frozen_temporal_passed": 1503,
            "scientific_logic_changed": False,
        }
        manifest_path = repo / "reports/phase3_14b_r258_stagef_resume3_materialization.json"
        if manifest_path.exists():
            raise MaterializationError(
                "refusing to overwrite Resume3 materialization manifest"
            )
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        written.append(manifest_path)
        return manifest
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
    parser.add_argument(
        "--allow-head-after-provenance-commit",
        action="store_true",
        help=(
            "Allow HEAD to differ from ebb3ddf after the existing Resume2 blocked "
            "report has been committed; remote/submodule/report hashes remain strict."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    repo = args.repo.resolve()
    package_root = args.package_root.resolve()
    try:
        manifest = materialize(
            repo,
            package_root,
            strict_head=not args.allow_head_after_provenance_commit,
        )
    except MaterializationError as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
