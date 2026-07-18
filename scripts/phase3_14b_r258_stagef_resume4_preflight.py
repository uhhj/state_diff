#!/usr/bin/env python3
"""Run Resume4 admission/regression gates, then delegate to the copied run wrapper."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# The admission predicate checks an exactly clean worktree. Prevent imports in
# this process and all delegated Python processes from creating repository-local
# __pycache__ files before that predicate runs.
sys.dont_write_bytecode = True

from ccda_phase3.phase314b_r258_stagef_resume4_head_admission import (
    Resume4AdmissionError,
    validate_resume4_initial_state,
)
from ccda_phase3.phase314b_r258_stagef_resume4_preflight_evidence import (
    BLOCKED_REPORT,
    Resume4BlockedReportError,
    write_resume4_preflight_blocked_report,
)


REPO_ROOT = Path("/data/state_diff2")
EXPECTED_REGRESSION_PASSED = 24
REGRESSION_TEST = "tests/test_phase3_14b_r258_stagef_resume4_head_admission.py"
RUN_WRAPPER = "scripts/phase3_14b_r258_stagef_resume4_run.sh"

COMPILE_PATHS: Tuple[str, ...] = (
    "ccda_phase3/phase314b_r258_stagef_resume4_head_admission.py",
    "ccda_phase3/phase314b_r258_stagef_resume4_preflight_evidence.py",
    "scripts/phase3_14b_r258_stagef_resume4_materialize.py",
    "scripts/phase3_14b_r258_stagef_resume4_preflight.py",
    "scripts/phase3_14b_r258_stagef_resume4_test_gate.py",
    "scripts/phase3_14b_r258_stagef_resume4_worker.py",
    "scripts/phase3_14b_r258_stagef_resume4_run_calibration.py",
    "scripts/phase3_14b_r258_stagef_resume4_blocked.py",
    REGRESSION_TEST,
)


class Resume4PreflightError(RuntimeError):
    def __init__(
        self,
        stage: str,
        root_cause: str,
        detail: str,
        *,
        command: Optional[Sequence[str]] = None,
        returncode: Optional[int] = None,
        stdout: Optional[str] = None,
        stderr: Optional[str] = None,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(detail)
        self.stage = stage
        self.root_cause = root_cause
        self.detail = detail
        self.command = list(command) if command is not None else None
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.extra = dict(extra or {})


def _run(
    command: Sequence[str],
    *,
    repo: Path,
    env: Optional[Mapping[str, str]] = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=str(repo),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        env=dict(env) if env is not None else None,
    )


def _parse_pytest_passed(output: str) -> Optional[int]:
    matches = re.findall(r"(?<!\d)(\d+) passed(?:,|\s|$)", output)
    if not matches:
        return None
    return int(matches[-1])


def _run_regression(repo: Path) -> Mapping[str, Any]:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        REGRESSION_TEST,
    ]
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    result = _run(command, repo=repo, env=env)
    combined = result.stdout + "\n" + result.stderr
    passed = _parse_pytest_passed(combined)
    if result.returncode != 0:
        raise Resume4PreflightError(
            "resume4_regression",
            "phase314b_r258_stagef_resume4_regression_failed",
            "Resume4 regression test command failed",
            command=command,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )
    if passed != EXPECTED_REGRESSION_PASSED:
        raise Resume4PreflightError(
            "resume4_regression",
            "phase314b_r258_stagef_resume4_regression_count_changed",
            f"expected {EXPECTED_REGRESSION_PASSED} passed, observed {passed!r}",
            command=command,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )
    return {
        "command": command,
        "returncode": result.returncode,
        "passed": passed,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def _run_compile_gate(repo: Path) -> Mapping[str, Any]:
    compiled: List[str] = []
    for relpath in COMPILE_PATHS:
        path = repo / relpath
        if not path.is_file():
            raise Resume4PreflightError(
                "resume4_compile",
                "phase314b_r258_stagef_resume4_compile_path_missing",
                f"compile path missing: {relpath}",
            )
        try:
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
        except (OSError, SyntaxError, UnicodeError) as exc:
            raise Resume4PreflightError(
                "resume4_compile",
                "phase314b_r258_stagef_resume4_python_compile_failed",
                f"Python compile failed for {relpath}: {exc}",
            ) from exc
        compiled.append(relpath)

    command = ["bash", "-n", RUN_WRAPPER]
    result = _run(command, repo=repo)
    if result.returncode != 0:
        raise Resume4PreflightError(
            "resume4_compile",
            "phase314b_r258_stagef_resume4_shell_syntax_failed",
            "Resume4 run wrapper failed bash -n",
            command=command,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )
    return {
        "python_compiled": compiled,
        "shell_command": command,
        "shell_returncode": result.returncode,
    }


def _write_preflight_blocked(repo: Path, error: Resume4PreflightError) -> None:
    try:
        result = write_resume4_preflight_blocked_report(
            failure_stage=error.stage,
            root_cause=error.root_cause,
            detail=error.detail,
            command=error.command,
            returncode=error.returncode,
            stdout=error.stdout,
            stderr=error.stderr,
            repo=repo,
            extra=error.extra,
        )
    except Resume4BlockedReportError as report_error:
        print(f"BLOCKED REPORT ERROR: {report_error}", file=sys.stderr)
        return
    print(json.dumps(result, indent=2, sort_keys=True), file=sys.stderr)


def run_preflight(repo: Path) -> int:
    repo = repo.resolve()
    try:
        admission = validate_resume4_initial_state(repo)
    except Resume4AdmissionError as exc:
        error = Resume4PreflightError(
            "resume4_head_admission",
            "phase314b_r258_stagef_resume4_versioned_head_admission_failed",
            str(exc),
        )
        _write_preflight_blocked(repo, error)
        return 2

    try:
        regression = _run_regression(repo)
        compile_gate = _run_compile_gate(repo)
    except Resume4PreflightError as exc:
        exc.extra.setdefault("admission", admission.as_dict())
        _write_preflight_blocked(repo, exc)
        return 2

    run_command = ["bash", RUN_WRAPPER]
    run_env = os.environ.copy()
    run_env["PYTHONDONTWRITEBYTECODE"] = "1"
    result = _run(run_command, repo=repo, env=run_env)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)

    if result.returncode != 0:
        report_path = repo / BLOCKED_REPORT.relative_to(REPO_ROOT)
        if not report_path.exists():
            error = Resume4PreflightError(
                "resume4_wrapped_execution",
                "phase314b_r258_stagef_resume4_wrapped_execution_failed_without_report",
                "Resume4 copied run wrapper failed without creating blocked evidence",
                command=run_command,
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                extra={
                    "admission": admission.as_dict(),
                    "regression": regression,
                    "compile_gate": compile_gate,
                },
            )
            _write_preflight_blocked(repo, error)
        return result.returncode

    return 0


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    return run_preflight(args.repo)


if __name__ == "__main__":
    raise SystemExit(main())
