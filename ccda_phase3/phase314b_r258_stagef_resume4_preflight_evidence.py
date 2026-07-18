"""Write-once preflight failure evidence for Stage F Resume4."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence


REPO_ROOT = Path("/data/state_diff2")
BLOCKED_REPORT = REPO_ROOT / "reports/phase3_14b_r258_stagef_resume4_blocked_summary.json"


class Resume4BlockedReportError(RuntimeError):
    pass


def _git(repo: Path, *args: str) -> Optional[str]:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(repo),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        text=True,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout.rstrip("\n")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_repo_state(repo: Path) -> Mapping[str, Any]:
    submodule = repo / "external/deformable-ravens"
    return {
        "branch": _git(repo, "branch", "--show-current"),
        "head": _git(repo, "rev-parse", "HEAD"),
        "origin_experiment1": _git(
            repo,
            "rev-parse",
            "refs/remotes/origin/Experiment1",
        ),
        "main_worktree_porcelain": _git(
            repo,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ),
        "submodule_gitlink": _git(
            repo,
            "rev-parse",
            "HEAD:external/deformable-ravens",
        ),
        "submodule_head": _git(submodule, "rev-parse", "HEAD") if submodule.is_dir() else None,
        "submodule_worktree_porcelain": (
            _git(
                submodule,
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            )
            if submodule.is_dir()
            else None
        ),
    }


def write_resume4_preflight_blocked_report(
    *,
    failure_stage: str,
    root_cause: str,
    detail: str,
    command: Optional[Sequence[str]] = None,
    returncode: Optional[int] = None,
    stdout: Optional[str] = None,
    stderr: Optional[str] = None,
    repo: Path = REPO_ROOT,
    extra: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, Any]:
    """Create the Resume4 blocked report exactly once using O_EXCL."""

    repo = repo.resolve()
    report_path = repo / BLOCKED_REPORT.relative_to(REPO_ROOT)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "phase": "Phase3.14b-r2.5.8 Stage F Resume4",
        "title": "Versioned Implementation-Head Admission Recovery and Scientific Continuation",
        "status": "BLOCKED",
        "scientific_status": "BLOCKED",
        "failure_stage": str(failure_stage),
        "root_cause": str(root_cause),
        "detail": str(detail),
        "command": list(command) if command is not None else None,
        "returncode": returncode,
        "stdout": stdout,
        "stderr": stderr,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repository": dict(_safe_repo_state(repo)),
        "scientific_calibration_run": False,
        "selected_configuration": None,
        "train_only_recommendation": None,
        "frozen_probe_accessed": False,
        "holdout_evaluated": False,
        "worker_exact_sha": None,
        "weights_generated": False,
        "checkpoint_generated": False,
        "npz_generated": False,
        "image_generated": False,
        "video_generated": False,
        "push_performed": False,
    }
    if extra:
        payload["extra"] = dict(extra)

    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    try:
        fd = os.open(
            str(report_path),
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
        )
    except FileExistsError as exc:
        raise Resume4BlockedReportError(
            f"refusing to overwrite existing Resume4 blocked report: {report_path}"
        ) from exc

    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            report_path.unlink()
        except FileNotFoundError:
            pass
        raise

    result = dict(payload)
    result["report_path"] = str(report_path)
    result["report_sha256"] = _sha256_bytes(encoded)
    return result


__all__ = [
    "BLOCKED_REPORT",
    "Resume4BlockedReportError",
    "write_resume4_preflight_blocked_report",
]
