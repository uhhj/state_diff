#!/usr/bin/env python3
"""Materialize add-only Stage-F Resume4 wrappers from committed Resume3 sources.

The generator runs only from the frozen Resume3 blocked-provenance HEAD. It
copies the exact Resume3 execution wrappers, changes only wrapper/report/phase
names, and replaces every legacy ``validate_initial_worktree`` invocation with
``validate_resume4_initial_state`` from the new versioned admission module. For
the shell run wrapper, the replacement is applied only inside safely delimited
Python heredoc bodies; all surrounding shell text is preserved.

The generator never edits prior files. Ambiguous source structure, missing
legacy validation, unexpected worktree state, source mutation, or destination
collision causes a fail-closed stop and rolls back newly written files.
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


EXPECTED_INPUT_HEAD = "2cab275171bdcc671f40f86092f19a357226dd18"
EXPECTED_REMOTE_HEAD = "6758ea7ad800667a436b0243d3b1f6c63256d854"
EXPECTED_SUBMODULE = "633a88752445cf5d6776ed374fdbbdb35f93050c"
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

RESUME3_BLOCKED_REPORT = "reports/phase3_14b_r258_stagef_resume3_blocked_summary.json"
MANIFEST_PATH = "reports/phase3_14b_r258_stagef_resume4_materialization.json"


@dataclass(frozen=True)
class CopySpec:
    source: str
    destination: str


COPY_SPECS: Tuple[CopySpec, ...] = (
    CopySpec(
        "scripts/phase3_14b_r258_stagef_resume3_test_gate.py",
        "scripts/phase3_14b_r258_stagef_resume4_test_gate.py",
    ),
    CopySpec(
        "scripts/phase3_14b_r258_stagef_resume3_worker.py",
        "scripts/phase3_14b_r258_stagef_resume4_worker.py",
    ),
    CopySpec(
        "scripts/phase3_14b_r258_stagef_resume3_run_calibration.py",
        "scripts/phase3_14b_r258_stagef_resume4_run_calibration.py",
    ),
    CopySpec(
        "scripts/phase3_14b_r258_stagef_resume3_blocked.py",
        "scripts/phase3_14b_r258_stagef_resume4_blocked.py",
    ),
    CopySpec(
        "scripts/phase3_14b_r258_stagef_resume3_run.sh",
        "scripts/phase3_14b_r258_stagef_resume4_run.sh",
    ),
)

SUPPORT_PATHS: Tuple[str, ...] = (
    "ccda_phase3/phase314b_r258_stagef_resume4_head_admission.py",
    "ccda_phase3/phase314b_r258_stagef_resume4_preflight_evidence.py",
    "scripts/phase3_14b_r258_stagef_resume4_materialize.py",
    "scripts/phase3_14b_r258_stagef_resume4_preflight.py",
    "tests/test_phase3_14b_r258_stagef_resume4_head_admission.py",
)

WRAPPER_FILENAME_REPLACEMENTS: Tuple[Tuple[str, str], ...] = tuple(
    pair
    for spec in COPY_SPECS
    for pair in (
        (Path(spec.source).name, Path(spec.destination).name),
        (Path(spec.source).stem, Path(spec.destination).stem),
    )
)

REPORT_REPLACEMENTS: Tuple[Tuple[str, str], ...] = tuple(
    (
        f"phase3_14b_r258_stagef_resume3_{suffix}",
        f"phase3_14b_r258_stagef_resume4_{suffix}",
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
    ("Phase3.14b-r2.5.8 Stage F Resume3", "Phase3.14b-r2.5.8 Stage F Resume4"),
    ("Stage F Resume3 —", "Stage F Resume4 —"),
    ("Stage-F Resume3 wrapper", "Stage-F Resume4 wrapper"),
)

NEW_IMPORT = (
    "from ccda_phase3.phase314b_r258_stagef_resume4_head_admission "
    "import validate_resume4_initial_state"
)
LEGACY_NAME = "validate_initial_worktree"
NEW_NAME = "validate_resume4_initial_state"

HEREDOC_START_RE = re.compile(
    r"<<(?!<)(?P<strip_tabs>-)?[ \t]*(?P<quote>['\"]?)"
    r"(?P<delimiter>[A-Za-z_][A-Za-z0-9_]*)(?P=quote)"
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
            f"git {' '.join(args)} failed ({proc.returncode}): "
            f"{proc.stderr.decode('utf-8', 'replace')}"
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


def replace_literals(text: str, replacements: Iterable[Tuple[str, str]]) -> str:
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def _line_offsets(text: str) -> List[int]:
    offsets = [0]
    for match in re.finditer("\n", text):
        offsets.append(match.end())
    return offsets


def _node_span(text: str, node: ast.AST) -> Tuple[int, int]:
    if not hasattr(node, "lineno") or not hasattr(node, "end_lineno"):
        raise MaterializationError("Python AST lacks source positions")
    offsets = _line_offsets(text)
    start = offsets[node.lineno - 1] + node.col_offset  # type: ignore[attr-defined]
    end = offsets[node.end_lineno - 1] + node.end_col_offset  # type: ignore[attr-defined]
    return start, end


def _format_import_from(node: ast.ImportFrom, names: Sequence[ast.alias]) -> str:
    module = "." * node.level + (node.module or "")
    indent = " " * node.col_offset
    rendered = [
        alias.name + (f" as {alias.asname}" if alias.asname else "")
        for alias in names
    ]
    if len(rendered) == 1:
        return f"{indent}from {module} import {rendered[0]}"
    body = "\n".join(f"{indent}    {name}," for name in rendered)
    return f"{indent}from {module} import (\n{body}\n{indent})"


def _apply_replacements(text: str, replacements: Sequence[Tuple[int, int, str]]) -> str:
    result = text
    for start, end, replacement in sorted(replacements, reverse=True):
        result = result[:start] + replacement + result[end:]
    return result


def _insert_import(text: str, import_line: str) -> str:
    tree = ast.parse(text)
    insertion_line = 0
    body = list(tree.body)
    if body and isinstance(body[0], ast.Expr):
        value = body[0].value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            insertion_line = int(body[0].end_lineno or body[0].lineno)
    for node in body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            insertion_line = max(insertion_line, int(node.end_lineno or node.lineno))

    lines = text.splitlines(keepends=True)
    insertion = import_line + "\n"
    if insertion_line > 0:
        if insertion_line < len(lines) and lines[insertion_line].strip():
            insertion += "\n"
        lines.insert(insertion_line, insertion)
    else:
        lines.insert(0, insertion + "\n")
    return "".join(lines)


def patch_legacy_validator(text: str) -> Tuple[str, int]:
    """Replace old validator imports/calls while preserving all other semantics."""

    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        raise MaterializationError(f"source wrapper is invalid Python: {exc}") from exc

    replacements: List[Tuple[int, int, str]] = []
    legacy_imports = 0
    new_import_injected = NEW_IMPORT in text
    legacy_import_nodes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and any(alias.name == LEGACY_NAME for alias in node.names)
    ]
    legacy_import_nodes.sort(key=lambda node: (node.lineno, node.col_offset))
    for node in legacy_import_nodes:
        kept = [alias for alias in node.names if alias.name != LEGACY_NAME]
        legacy_imports += len(node.names) - len(kept)
        start, end = _node_span(text, node)
        replacement = _format_import_from(node, kept) if kept else ""
        if not new_import_injected:
            in_place_import = " " * node.col_offset + NEW_IMPORT
            replacement = (
                replacement + "\n" + in_place_import
                if replacement
                else in_place_import
            )
            new_import_injected = True
        replacements.append((start, end, replacement))

    call_replacements = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        is_legacy = (
            isinstance(func, ast.Name) and func.id == LEGACY_NAME
        ) or (
            isinstance(func, ast.Attribute) and func.attr == LEGACY_NAME
        )
        if not is_legacy:
            continue
        start, end = _node_span(text, func)
        replacements.append((start, end, NEW_NAME))
        call_replacements += 1

    if call_replacements == 0 and legacy_imports == 0:
        return text, 0
    if call_replacements == 0:
        raise MaterializationError(
            "legacy validator import exists without a corresponding call"
        )

    transformed = _apply_replacements(text, replacements)
    if not new_import_injected:
        transformed = _insert_import(transformed, NEW_IMPORT)

    try:
        transformed_tree = ast.parse(transformed)
    except SyntaxError as exc:
        raise MaterializationError(
            f"validator-patched wrapper is invalid Python: {exc}"
        ) from exc

    for node in ast.walk(transformed_tree):
        if isinstance(node, ast.ImportFrom):
            if any(alias.name == LEGACY_NAME for alias in node.names):
                raise MaterializationError("legacy validator import remains after patch")
        if isinstance(node, ast.Call):
            func = node.func
            if (
                isinstance(func, ast.Name) and func.id == LEGACY_NAME
            ) or (
                isinstance(func, ast.Attribute) and func.attr == LEGACY_NAME
            ):
                raise MaterializationError("legacy validator call remains after patch")

    if transformed.count(NEW_NAME) < call_replacements + 1:
        raise MaterializationError("new Resume4 validator binding is incomplete")
    return transformed, call_replacements


def _heredoc_terminator_matches(
    line: str,
    delimiter: str,
    *,
    strip_tabs: bool,
) -> bool:
    candidate = line.rstrip("\r\n")
    if strip_tabs:
        candidate = candidate.lstrip("\t")
    return candidate == delimiter


def _runtime_heredoc_body(body_lines: Sequence[str], *, strip_tabs: bool) -> str:
    if not strip_tabs:
        return "".join(body_lines)
    # ``<<-`` removes leading TAB characters before passing the body to Python.
    # Materialize the exact runtime body rather than attempting AST offsets on
    # source-only indentation that the shell itself discards.
    return "".join(line.lstrip("\t") for line in body_lines)


def patch_shell_python_heredocs(text: str) -> Tuple[str, int]:
    """Patch legacy validator calls inside shell-embedded Python only.

    Resume3's run wrapper performs its initial repository validation in a
    Python heredoc.  Shell text is not Python, so applying ``ast.parse`` to the
    entire file is invalid.  This parser preserves every shell line and every
    heredoc delimiter verbatim, extracts only heredoc bodies containing the
    exact legacy token, requires those bodies to be valid Python, and delegates
    their semantic transformation to :func:`patch_legacy_validator`.

    A legacy token outside a uniquely delimited, valid Python heredoc is an
    ambiguous layout and therefore blocks materialization.
    """

    lines = text.splitlines(keepends=True)
    output: List[str] = []
    call_count = 0
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
                lines[terminator_index],
                delimiter,
                strip_tabs=strip_tabs,
            ):
                break
            terminator_index += 1
        if terminator_index >= len(lines):
            raise MaterializationError(
                f"unterminated shell heredoc delimiter: {delimiter!r}"
            )

        body_lines = lines[index + 1 : terminator_index]
        source_body = "".join(body_lines)
        if LEGACY_NAME in source_body:
            runtime_body = _runtime_heredoc_body(
                body_lines,
                strip_tabs=strip_tabs,
            )
            try:
                transformed_body, replaced = patch_legacy_validator(runtime_body)
            except MaterializationError as exc:
                raise MaterializationError(
                    f"legacy validator heredoc {delimiter!r} is not safely "
                    f"patchable Python: {exc}"
                ) from exc
            if replaced < 1:
                raise MaterializationError(
                    f"legacy validator token in heredoc {delimiter!r} is not an "
                    "executable Python call"
                )
            if LEGACY_NAME in transformed_body:
                raise MaterializationError(
                    f"legacy validator token remains in heredoc {delimiter!r}"
                )
            output.append(transformed_body)
            call_count += replaced
        else:
            output.extend(body_lines)

        output.append(lines[terminator_index])
        index = terminator_index + 1

    transformed = "".join(output)
    if LEGACY_NAME in transformed:
        raise MaterializationError(
            "legacy validate_initial_worktree token remains outside a safely "
            "patchable Python heredoc"
        )
    return transformed, call_count


def transform_source(source_rel: str, text: str) -> Tuple[str, int]:
    transformed = replace_literals(text, WRAPPER_FILENAME_REPLACEMENTS)
    transformed = replace_literals(transformed, REPORT_REPLACEMENTS)
    transformed = replace_literals(transformed, PHASE_LABEL_REPLACEMENTS)
    if source_rel.endswith(".py"):
        transformed, call_count = patch_legacy_validator(transformed)
    elif source_rel.endswith(".sh"):
        transformed, call_count = patch_shell_python_heredocs(transformed)
    else:
        if LEGACY_NAME in transformed:
            raise MaterializationError(
                f"legacy validator found in unsupported source type: {source_rel}"
            )
        call_count = 0
    return transformed, call_count


def validate_initial_repo(repo: Path) -> str:
    if run_git(repo, "rev-parse", "--show-toplevel") != str(repo.resolve()):
        raise MaterializationError("--repo must be the Git top-level directory")
    if run_git(repo, "branch", "--show-current") != "Experiment1":
        raise MaterializationError("expected branch Experiment1")
    head = run_git(repo, "rev-parse", "HEAD")
    if head != EXPECTED_INPUT_HEAD:
        raise MaterializationError(
            f"expected Resume3 blocked-provenance HEAD {EXPECTED_INPUT_HEAD}, got {head}"
        )
    for ancestor in REQUIRED_ANCESTORS:
        require_ancestor(repo, ancestor, head)
    remote = run_git(repo, "rev-parse", "refs/remotes/origin/Experiment1")
    if remote != EXPECTED_REMOTE_HEAD:
        raise MaterializationError(
            f"expected origin/Experiment1 {EXPECTED_REMOTE_HEAD}, got {remote}"
        )
    if run_git_bytes(
        repo,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
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
        submodule,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
    ):
        raise MaterializationError("DeformableRavens worktree is not clean")
    return head


def validate_prior_reports(repo: Path) -> str:
    for relpath, expected in IMMUTABLE_REPORT_SHA256.items():
        path = repo / relpath
        if not path.is_file():
            raise MaterializationError(f"immutable report missing: {relpath}")
        actual = sha256_path(path)
        if actual != expected:
            raise MaterializationError(
                f"immutable report changed: {relpath}: {actual} != {expected}"
            )

    report = repo / RESUME3_BLOCKED_REPORT
    if not report.is_file():
        raise MaterializationError(f"Resume3 blocked report missing: {RESUME3_BLOCKED_REPORT}")
    committed = run_git_bytes(repo, "show", f"{EXPECTED_INPUT_HEAD}:{RESUME3_BLOCKED_REPORT}")
    current = report.read_bytes()
    if current != committed:
        raise MaterializationError(
            "Resume3 blocked report differs from blocked-provenance commit"
        )
    return sha256_bytes(current)


def validate_source_blobs(repo: Path) -> Mapping[str, bytes]:
    result: MutableMapping[str, bytes] = {}
    for spec in COPY_SPECS:
        source = repo / spec.source
        destination = repo / spec.destination
        if not source.is_file():
            raise MaterializationError(f"required Resume3 source missing: {spec.source}")
        if destination.exists():
            raise MaterializationError(
                f"refusing to overwrite existing Resume4 path: {spec.destination}"
            )
        current = source.read_bytes()
        committed = run_git_bytes(repo, "show", f"{EXPECTED_INPUT_HEAD}:{spec.source}")
        if current != committed:
            raise MaterializationError(
                f"Resume3 source differs from {EXPECTED_INPUT_HEAD}: {spec.source}"
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


def materialize(repo: Path, package_root: Path) -> Mapping[str, object]:
    repo = repo.resolve()
    package_root = package_root.resolve()
    input_head = validate_initial_repo(repo)
    resume3_blocked_sha = validate_prior_reports(repo)
    source_bytes = validate_source_blobs(repo)

    written: List[Path] = []
    source_hashes: Dict[str, str] = {
        path: sha256_bytes(data) for path, data in source_bytes.items()
    }
    validator_call_count = 0
    python_validator_call_count = 0
    shell_heredoc_validator_call_count = 0
    try:
        for spec in COPY_SPECS:
            try:
                source_text = source_bytes[spec.source].decode("utf-8")
            except UnicodeDecodeError as exc:
                raise MaterializationError(
                    f"Resume3 source is not UTF-8: {spec.source}"
                ) from exc
            transformed, calls = transform_source(spec.source, source_text)
            validator_call_count += calls
            if spec.source.endswith(".sh"):
                shell_heredoc_validator_call_count += calls
            else:
                python_validator_call_count += calls
            destination = repo / spec.destination
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("w", encoding="utf-8", newline="") as handle:
                handle.write(transformed)
            if os.access(repo / spec.source, os.X_OK):
                destination.chmod((repo / spec.source).stat().st_mode)
            written.append(destination)

        if validator_call_count < 1:
            raise MaterializationError(
                "no legacy validate_initial_worktree call was found in Resume3 wrappers"
            )
        if shell_heredoc_validator_call_count < 1:
            raise MaterializationError(
                "the confirmed Resume3 shell-heredoc legacy validator call was not replaced"
            )

        written.extend(copy_support_files(repo, package_root))

        for path in written:
            if path.suffix == ".py":
                compile(path.read_text(encoding="utf-8"), str(path), "exec")
        shell_path = repo / "scripts/phase3_14b_r258_stagef_resume4_run.sh"
        shell_check = subprocess.run(
            ["bash", "-n", str(shell_path)],
            cwd=str(repo),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if shell_check.returncode != 0:
            raise MaterializationError(
                "generated Resume4 shell is invalid: "
                + shell_check.stderr.decode("utf-8", "replace")
            )

        for spec in COPY_SPECS:
            if sha256_path(repo / spec.source) != source_hashes[spec.source]:
                raise MaterializationError(
                    f"Resume3 source mutated during materialization: {spec.source}"
                )

        generated_hashes = {
            str(path.relative_to(repo)): sha256_path(path)
            for path in sorted(written)
        }
        expected_generated = {
            spec.destination for spec in COPY_SPECS
        } | set(SUPPORT_PATHS)
        if set(generated_hashes) != expected_generated:
            raise MaterializationError(
                "generated implementation population changed: "
                f"expected={sorted(expected_generated)!r}, "
                f"actual={sorted(generated_hashes)!r}"
            )

        manifest: Dict[str, object] = {
            "phase": "Phase3.14b-r2.5.8 Stage F Resume4",
            "operation": "add_only_versioned_head_admission_recovery",
            "input_head": input_head,
            "remote_head": EXPECTED_REMOTE_HEAD,
            "submodule_head": EXPECTED_SUBMODULE,
            "required_ancestors": list(REQUIRED_ANCESTORS),
            "resume3_blocked_report_sha256": resume3_blocked_sha,
            "source_sha256": source_hashes,
            "generated_sha256": generated_hashes,
            "legacy_validator_calls_replaced": validator_call_count,
            "python_validator_calls_replaced": python_validator_call_count,
            "shell_python_heredoc_validator_calls_replaced": (
                shell_heredoc_validator_call_count
            ),
            "frozen_temporal_files": 58,
            "frozen_temporal_passed": 1503,
            "scientific_logic_changed": False,
        }
        manifest_path = repo / MANIFEST_PATH
        if manifest_path.exists():
            raise MaterializationError(
                f"refusing to overwrite Resume4 manifest: {MANIFEST_PATH}"
            )
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
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
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        manifest = materialize(args.repo, args.package_root)
    except MaterializationError as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
