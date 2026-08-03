#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
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


def _write_marker(stageg: Any, path: Path, payload: Mapping[str, Any]) -> None:
    stageg.atomic_write_once(path, stageg.stable_json_bytes(dict(payload)))


def controller(
    *, root: Path, python_bin: Path, implementation_commit: str, write_ahead: Path
) -> int:
    repo = Path(root).resolve()
    _bootstrap(repo)
    from ccda_phase3 import phase314b_r259_stageg_one_shot_frozen_probe as stageg

    repository: Optional[Mapping[str, Any]] = None
    probe_repo = repo / stageg.PROBE_EVIDENCE
    worker_repo = repo / stageg.WORKER_EVIDENCE
    success_repo = repo / stageg.SUCCESS_REPORT
    blocked_repo = repo / stageg.BLOCKED_REPORT
    probe_wa = write_ahead / "environment_probe_evidence.json"
    worker_wa = write_ahead / "frozen_probe_worker_evidence.json"
    worker_started = write_ahead / "worker_started.json"
    access_marker = write_ahead / "frozen_probe_access_begun.json"
    try:
        repository = stageg.validate_repository(repo, implementation_commit)
        write_ahead.mkdir(parents=True, exist_ok=False)
        probe_command = [
            str(python_bin), "-c",
            (
                "from pathlib import Path; "
                "from ccda_phase3 import phase314b_r259_stageg_one_shot_frozen_probe as s; "
                "p=s.make_environment_probe(Path(r'{}')); "
                "s.atomic_write_once(Path(r'{}'), s.stable_json_bytes(p))"
            ).format(repo, probe_wa),
        ]
        _run(probe_command, root=repo, label="environment probe")
        stageg.promote_write_ahead(probe_wa, probe_repo, stageg.validate_environment_probe)
        _write_marker(
            stageg,
            worker_started,
            {
                "phase": stageg.PHASE,
                "worker_started": True,
                "one_shot_attempt_consumed": True,
                "rerun_authorized": False,
            },
        )
        worker_script = repo / "scripts/phase3_14b_r259_stageg_worker.py"
        worker_command = [
            str(python_bin), str(worker_script),
            "--root", str(repo),
            "--probe", str(probe_wa),
            "--repository-head", implementation_commit,
            "--access-marker", str(access_marker),
            "--output", str(worker_wa),
        ]
        _run(worker_command, root=repo, label="one-shot frozen-probe worker")
        stageg.promote_write_ahead(worker_wa, worker_repo, stageg.validate_worker_evidence)
        summary = stageg.build_summary(
            repository=repository,
            probe_path=probe_repo,
            worker_path=worker_repo,
        )
        stageg.atomic_write_once(success_repo, stageg.stable_json_bytes(summary))
        return 0
    except BaseException as error:
        if success_repo.exists() or blocked_repo.exists():
            raise
        blocked = stageg.blocked_report(
            repository=repository,
            error=error,
            probe_path=probe_repo,
            worker_path=worker_repo,
            worker_started_path=worker_started,
            access_marker_path=access_marker,
        )
        stageg.atomic_write_once(blocked_repo, stageg.stable_json_bytes(blocked))
        print("BLOCKED: {}".format(error), file=sys.stderr)
        return 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("controller", "probe"), default="controller")
    parser.add_argument("--root", type=Path, default=Path("/data/state_diff2"))
    parser.add_argument("--python-bin", type=Path)
    parser.add_argument("--implementation-commit")
    parser.add_argument("--write-ahead-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--import-smoke", action="store_true")
    args = parser.parse_args()
    _bootstrap(args.root)
    from ccda_phase3 import phase314b_r259_stageg_one_shot_frozen_probe as stageg

    if args.import_smoke:
        print(stageg.SCHEMA)
        return 0
    if args.mode == "probe":
        if args.output is None:
            raise SystemExit("--output required for probe mode")
        payload = stageg.make_environment_probe(args.root)
        stageg.atomic_write_once(args.output, stageg.stable_json_bytes(payload))
        return 0
    if args.python_bin is None or args.implementation_commit is None or args.write_ahead_dir is None:
        raise SystemExit("controller mode requires python-bin, implementation-commit and write-ahead-dir")
    return controller(
        root=args.root,
        python_bin=args.python_bin,
        implementation_commit=args.implementation_commit,
        write_ahead=args.write_ahead_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())
