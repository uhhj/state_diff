#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Optional

ROOT_FROM_FILE = Path(__file__).resolve().parents[1]
if str(ROOT_FROM_FILE) not in sys.path:
    sys.path.insert(0, str(ROOT_FROM_FILE))

from ccda_phase3 import phase314b_r260_stagec_objective_train_baseline as stagec


def _run_worker(root: Path, output: Path, python_bin: Path) -> None:
    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = "0"
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["OMP_NUM_THREADS"] = "1"
    environment["OPENBLAS_NUM_THREADS"] = "1"
    environment["MKL_NUM_THREADS"] = "1"
    environment["NUMEXPR_NUM_THREADS"] = "1"
    environment["VECLIB_MAXIMUM_THREADS"] = "1"
    environment["PYTHONPATH"] = os.pathsep.join(
        [
            str(root),
            str(root / "scripts"),
            str(root / "external/deformable-ravens"),
            environment.get("PYTHONPATH", ""),
        ]
    )
    subprocess.run(
        [
            str(python_bin),
            str(root / "scripts/phase3_14b_r260_stagec_worker.py"),
            "--root",
            str(root),
            "--output",
            str(output),
        ],
        cwd=str(root),
        env=environment,
        check=True,
    )


def execute(root: Path, implementation_commit: str, python_bin: Path) -> Mapping[str, Any]:
    repository = stagec.validate_repository(root, implementation_commit)
    # Metadata and source-file SHA checks do not parse any role NPZ.
    stagec.validate_source_metadata(root, open_objective_npz=False)
    write_ahead = stagec.write_ahead_root(root)
    write_ahead.mkdir(parents=True, exist_ok=False)
    marker = {
        "phase": stagec.PHASE,
        "schema": stagec.SCHEMA + "_attempt_started_v1",
        "implementation_commit": implementation_commit,
        "objective_train_open_authorized": True,
        "selection_holdout_open_authorized": False,
        "frozen_probe_open_authorized": False,
        "final_evaluation_open_authorized": False,
        "rerun_authorized": False,
        "resume_authorized": False,
    }
    stagec.atomic_write_once(
        write_ahead / stagec.ATTEMPT_MARKER_NAME,
        stagec.stable_json_bytes(marker),
    )
    worker_path = write_ahead / stagec.WORKER_EVIDENCE_NAME
    _run_worker(root, worker_path, python_bin)
    worker = stagec.load_json(worker_path)
    stagec.validate_worker_result(worker, root)
    summary = stagec.build_summary(
        repository=repository,
        worker=worker,
        worker_file_sha256=stagec.sha256_file(worker_path),
    )
    stagec.atomic_write_once(root / stagec.SUCCESS_REPORT, stagec.stable_json_bytes(summary))
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
        print(stagec.SCHEMA)
        return 0
    root = Path(args.root).resolve()
    if args.metadata_preflight:
        result = stagec.validate_source_metadata(root, open_objective_npz=False)
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0
    if not args.implementation_commit:
        raise SystemExit("--implementation-commit is required")
    repository: Optional[Mapping[str, Any]] = None
    try:
        summary = execute(
            root,
            str(args.implementation_commit),
            Path(args.python_bin).resolve(),
        )
        print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except BaseException as error:
        worker_started = (stagec.write_ahead_root(root) / stagec.ATTEMPT_MARKER_NAME).is_file()
        try:
            repository = {
                "root": str(root),
                "head": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=str(root), text=True
                ).strip(),
                "submodule_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"],
                    cwd=str(root / "external/deformable-ravens"),
                    text=True,
                ).strip(),
            }
        except BaseException:
            repository = None
        blocked = stagec.blocked_report(
            repository=repository,
            error=error,
            worker_started=worker_started,
        )
        blocked_path = root / stagec.BLOCKED_REPORT
        if not blocked_path.exists():
            stagec.atomic_write_once(blocked_path, stagec.stable_json_bytes(blocked))
        print(json.dumps(blocked, indent=2, sort_keys=True, allow_nan=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
