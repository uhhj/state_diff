#!/usr/bin/env python3
"""Controller for Stage-D objective-train candidate-signal benchmark."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Optional


def _bootstrap() -> Path:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


_bootstrap()
from ccda_phase3 import phase314b_r260_staged_candidate_signal_benchmark as staged  # noqa: E402


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    staged.atomic_write_once(path, staged.stable_json_bytes(payload))


def execute(root: Path, implementation_commit: str, python_bin: Path) -> Mapping[str, Any]:
    repo = Path(root).resolve()
    repository = staged.validate_repository(repo, implementation_commit)
    staged.metadata_preflight(repo)
    write_ahead = staged.write_ahead_root(repo)
    if write_ahead.exists():
        raise staged.StageDError("Stage-D write-ahead already exists")
    write_ahead.mkdir(parents=True, exist_ok=False)
    marker = {
        "phase": staged.PHASE,
        "schema": staged.SCHEMA + "_attempt_started_v1",
        "implementation_commit": implementation_commit,
        "objective_train_open_authorized": True,
        "selection_holdout_open_authorized": False,
        "frozen_probe_open_authorized": False,
        "final_evaluation_open_authorized": False,
        "rerun_authorized": False,
        "resume_authorized": False,
    }
    _write_json(write_ahead / staged.ATTEMPT_MARKER_NAME, marker)
    worker_path = write_ahead / staged.WORKER_EVIDENCE_NAME
    staged.atomic_write_once(
        write_ahead / staged.WORKER_STARTED_NAME,
        staged.stable_json_bytes({
            "phase": staged.PHASE,
            "schema": staged.SCHEMA + "_worker_started_v1",
            "implementation_commit": implementation_commit,
        }),
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(repo), str(repo / "scripts"), str(repo / "external/deformable-ravens"), env.get("PYTHONPATH", "")]
    )
    subprocess.run(
        [
            str(python_bin),
            str(repo / "scripts/phase3_14b_r260_staged_worker.py"),
            "--root",
            str(repo),
            "--output",
            str(worker_path),
        ],
        cwd=str(repo),
        env=env,
        check=True,
    )
    worker = staged.load_json(worker_path)
    staged.validate_worker_result(worker)
    summary = staged.build_summary(
        repository=repository,
        worker=worker,
        worker_file_sha256=staged.sha256_file(worker_path),
    )
    _write_json(repo / staged.SUCCESS_REPORT, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--implementation-commit")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--metadata-preflight", action="store_true")
    parser.add_argument("--import-smoke", action="store_true")
    args = parser.parse_args()
    if args.import_smoke:
        print(json.dumps({"import_smoke": True, "phase": staged.PHASE}, sort_keys=True))
        return 0
    repo = Path(args.root).resolve()
    if args.metadata_preflight:
        print(json.dumps(staged.metadata_preflight(repo), indent=2, sort_keys=True, allow_nan=False))
        return 0
    if not args.implementation_commit:
        raise SystemExit("--implementation-commit is required")
    repository: Optional[Mapping[str, Any]] = None
    worker_started = False
    try:
        summary = execute(repo, str(args.implementation_commit), Path(args.python_bin).resolve())
        print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except BaseException as error:
        worker_started = (staged.write_ahead_root(repo) / staged.WORKER_STARTED_NAME).is_file()
        try:
            if repo.is_dir():
                repository = {
                    "root": str(repo),
                    "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo), text=True).strip(),
                    "submodule_commit": subprocess.check_output(
                        ["git", "rev-parse", "HEAD"],
                        cwd=str(repo / "external/deformable-ravens"),
                        text=True,
                    ).strip(),
                }
        except BaseException:
            repository = None
        blocked = staged.blocked_report(
            repository=repository,
            error=error,
            worker_started=worker_started,
        )
        blocked_path = repo / staged.BLOCKED_REPORT
        if not blocked_path.exists():
            staged.atomic_write_once(blocked_path, staged.stable_json_bytes(blocked))
        print(json.dumps(blocked, indent=2, sort_keys=True, allow_nan=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
