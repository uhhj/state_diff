#!/usr/bin/env python3
"""Run Resume5 import-bootstrap, admission, regression, and compile gates.

This file must remain directly executable from any current working directory and
with PYTHONPATH unset.  Therefore the repository root is inserted into
``sys.path`` before importing any ``ccda_phase3`` module.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


# Do not move this block below a ccda_phase3 import.
sys.dont_write_bytecode = True
_RESUME5_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_RESUME5_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_RESUME5_REPO_ROOT))

from ccda_phase3.phase314b_r258_stagef_resume5_head_admission import (  # noqa: E402
    Resume5AdmissionError,
    validate_resume5_initial_state,
)
from ccda_phase3.phase314b_r258_stagef_resume5_preflight_evidence import (  # noqa: E402
    BLOCKED_REPORT,
    Resume5BlockedReportError,
    write_resume5_preflight_blocked_report,
)


REPO_ROOT = Path("/data/state_diff2")
EXPECTED_REGRESSION_PASSED = 30
REGRESSION_TEST = "tests/test_phase3_14b_r258_stagef_resume5_import_bootstrap.py"
RUN_WRAPPER = "scripts/phase3_14b_r258_stagef_resume5_run.sh"

COMPILE_PATHS: Tuple[str, ...] = (
    "ccda_phase3/phase314b_r258_stagef_resume5_head_admission.py",
    "ccda_phase3/phase314b_r258_stagef_resume5_preflight_evidence.py",
    "scripts/phase3_14b_r258_stagef_resume5_materialize.py",
    "scripts/phase3_14b_r258_stagef_resume5_preflight.py",
    "scripts/phase3_14b_r258_stagef_resume5_test_gate.py",
    "scripts/phase3_14b_r258_stagef_resume5_worker.py",
    "scripts/phase3_14b_r258_stagef_resume5_run_calibration.py",
    "scripts/phase3_14b_r258_stagef_resume5_blocked.py",
    REGRESSION_TEST,
)


class Resume5PreflightError(RuntimeError):
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


def _execution_env(repo: Path) -> Dict[str, str]:
    env = os.environ.copy()
    repo_text = str(repo.resolve())
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = repo_text if not existing else repo_text + os.pathsep + existing
    env["CCDA_REPO_ROOT"] = repo_text
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _run(
    command: Sequence[str],
    *,
    repo: Path,
    env: Optional[Mapping[str, str]] = None,
    cwd: Optional[Path] = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=str(cwd if cwd is not None else repo),
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
    result = _run(command, repo=repo, env=_execution_env(repo))
    combined = result.stdout + "\n" + result.stderr
    passed = _parse_pytest_passed(combined)
    if result.returncode != 0:
        raise Resume5PreflightError(
            "resume5_regression",
            "phase314b_r258_stagef_resume5_regression_failed",
            "Resume5 regression test command failed",
            command=command,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )
    if passed != EXPECTED_REGRESSION_PASSED:
        raise Resume5PreflightError(
            "resume5_regression",
            "phase314b_r258_stagef_resume5_regression_count_changed",
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
            raise Resume5PreflightError(
                "resume5_compile",
                "phase314b_r258_stagef_resume5_compile_path_missing",
                f"compile path missing: {relpath}",
            )
        try:
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
        except (OSError, SyntaxError, UnicodeError) as exc:
            raise Resume5PreflightError(
                "resume5_compile",
                "phase314b_r258_stagef_resume5_python_compile_failed",
                f"Python compile failed for {relpath}: {exc}",
            ) from exc
        compiled.append(relpath)

    command = ["bash", "-n", RUN_WRAPPER]
    result = _run(command, repo=repo, env=_execution_env(repo))
    if result.returncode != 0:
        raise Resume5PreflightError(
            "resume5_compile",
            "phase314b_r258_stagef_resume5_shell_syntax_failed",
            "Resume5 run wrapper failed bash -n",
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


def _write_preflight_blocked(repo: Path, error: Resume5PreflightError) -> None:
    try:
        result = write_resume5_preflight_blocked_report(
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
    except Resume5BlockedReportError as report_error:
        print(f"BLOCKED REPORT ERROR: {report_error}", file=sys.stderr)
        return
    print(json.dumps(result, indent=2, sort_keys=True), file=sys.stderr)


def bootstrap_probe_payload() -> Mapping[str, Any]:
    root_text = str(_RESUME5_REPO_ROOT)
    return {
        "status": "PASS",
        "repo_root": root_text,
        "repo_root_in_sys_path": root_text in sys.path,
        "ccda_admission_imported": callable(validate_resume5_initial_state),
        "python": sys.executable,
        "cwd": os.getcwd(),
    }


def run_preflight(repo: Path) -> int:
    repo = repo.resolve()
    try:
        admission = validate_resume5_initial_state(repo)
    except Resume5AdmissionError as exc:
        error = Resume5PreflightError(
            "resume5_head_admission",
            "phase314b_r258_stagef_resume5_versioned_head_admission_failed",
            str(exc),
        )
        _write_preflight_blocked(repo, error)
        return 2

    try:
        regression = _run_regression(repo)
        compile_gate = _run_compile_gate(repo)
    except Resume5PreflightError as exc:
        exc.extra.setdefault("admission", admission.as_dict())
        _write_preflight_blocked(repo, exc)
        return 2

    run_command = ["bash", RUN_WRAPPER]
    run_env = _execution_env(repo)
    result = _run(run_command, repo=repo, env=run_env)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)

    if result.returncode != 0:
        report_path = repo / BLOCKED_REPORT.relative_to(REPO_ROOT)
        if not report_path.exists():
            error = Resume5PreflightError(
                "resume5_wrapped_execution",
                "phase314b_r258_stagef_resume5_wrapped_execution_failed_without_report",
                "Resume5 copied run wrapper failed without creating blocked evidence",
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
    parser.add_argument("--bootstrap-probe", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.bootstrap_probe:
        payload = bootstrap_probe_payload()
        print(json.dumps(payload, sort_keys=True))
        return 0 if payload["repo_root_in_sys_path"] else 2
    return run_preflight(args.repo)


if __name__ == "__main__":
    raise SystemExit(main())
