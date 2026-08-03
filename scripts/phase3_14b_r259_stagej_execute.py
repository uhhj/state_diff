#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Optional


def _bootstrap(root: Path) -> None:
    value = str(Path(root).resolve())
    if value not in sys.path:
        sys.path.insert(0, value)


def _run(command: list[str], *, root: Path, label: str) -> None:
    result = subprocess.run(command, cwd=str(root), check=False)
    if result.returncode != 0:
        raise RuntimeError("{} failed with return code {}".format(label, result.returncode))


def controller(
    *, root: Path, python_bin: Path, implementation_commit: str, write_ahead: Path
) -> int:
    repo = Path(root).resolve()
    _bootstrap(repo)
    from ccda_phase3 import phase314b_r259_stagej_prevalence_anchored_risk_repair as stagej

    repository: Optional[Mapping[str, Any]] = None
    probe_repo = repo / stagej.PROBE_EVIDENCE
    worker_repo = repo / stagej.WORKER_EVIDENCE
    success_repo = repo / stagej.SUCCESS_REPORT
    blocked_repo = repo / stagej.BLOCKED_REPORT
    probe_wa = write_ahead / "environment_probe_evidence.json"
    worker_wa = write_ahead / "prevalence_anchored_worker_evidence.json"
    worker_started = write_ahead / "worker_started.json"
    try:
        repository = stagej.validate_repository(repo, implementation_commit)
        write_ahead.mkdir(parents=True, exist_ok=False)
        probe_command = [
            str(python_bin),
            "-c",
            (
                "from pathlib import Path; "
                "from ccda_phase3 import phase314b_r259_stagej_prevalence_anchored_risk_repair as s; "
                "p=s.make_environment_probe(Path(r'{}')); "
                "s.atomic_write_once(Path(r'{}'), s.stable_json_bytes(p))"
            ).format(repo, probe_wa),
        ]
        _run(probe_command, root=repo, label="environment probe")
        stagej.promote_write_ahead(probe_wa, probe_repo, stagej.validate_environment_probe)
        stagej.atomic_write_once(
            worker_started,
            stagej.stable_json_bytes(
                {
                    "phase": stagej.PHASE,
                    "worker_started": True,
                    "science_attempt_consumed": True,
                    "selection_holdout_access_authorized": False,
                    "frozen_probe_access_authorized": False,
                    "fresh_evaluation_access_authorized": False,
                    "rerun_authorized": False,
                }
            ),
        )
        worker_command = [
            str(python_bin),
            str(repo / "scripts/phase3_14b_r259_stagej_worker.py"),
            "--root",
            str(repo),
            "--probe",
            str(probe_wa),
            "--repository-head",
            implementation_commit,
            "--output",
            str(worker_wa),
        ]
        _run(worker_command, root=repo, label="objective-train science worker")
        stagej.promote_write_ahead(worker_wa, worker_repo, stagej.validate_worker)
        summary = stagej.build_summary(
            repository=repository, probe_path=probe_repo, worker_path=worker_repo
        )
        stagej.atomic_write_once(success_repo, stagej.stable_json_bytes(summary))
        return 0
    except BaseException as error:
        if success_repo.exists() or blocked_repo.exists():
            raise
        blocked = stagej.blocked_report(
            repository, error, worker_started=worker_started.is_file()
        )
        stagej.atomic_write_once(blocked_repo, stagej.stable_json_bytes(blocked))
        print("BLOCKED: {}".format(error), file=sys.stderr)
        return 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/data/state_diff2"))
    parser.add_argument("--python-bin", type=Path)
    parser.add_argument("--implementation-commit")
    parser.add_argument("--write-ahead-dir", type=Path)
    parser.add_argument("--import-smoke", action="store_true")
    args = parser.parse_args()
    _bootstrap(args.root)
    from ccda_phase3 import phase314b_r259_stagej_prevalence_anchored_risk_repair as stagej

    if args.import_smoke:
        print(stagej.SCHEMA)
        return 0
    if args.python_bin is None or args.implementation_commit is None or args.write_ahead_dir is None:
        raise SystemExit("python-bin, implementation-commit and write-ahead-dir are required")
    return controller(
        root=args.root,
        python_bin=args.python_bin,
        implementation_commit=args.implementation_commit,
        write_ahead=args.write_ahead_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())
