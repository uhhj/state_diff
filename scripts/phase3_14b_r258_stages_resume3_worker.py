#!/usr/bin/env python3
"""Run one fresh cold Stage-S Resume3 science worker."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ccda_phase3.phase314b_r258_stages_resume3_cold_worker_oof_confirmation import (
    StageSResume3Error,
    cold_science_worker_payload,
    load_json,
    stable_json_bytes,
    validate_environment_variables,
)


def write_once(path: Path, payload: bytes) -> None:
    target = Path(path)
    if target.exists():
        raise StageSResume3Error(f"write-once worker output exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
    temporary.replace(target)


def _git_head(root: Path) -> str:
    import subprocess

    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(root),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise StageSResume3Error(
            f"cannot read repository HEAD: {completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        validate_environment_variables()
        root = Path(args.root).resolve()
        result = cold_science_worker_payload(
            worker_id=args.worker_id,
            root=root,
            probe_payload=load_json(Path(args.probe).resolve()),
            repository_head=_git_head(root),
        )
        output = Path(args.output).resolve()
        write_once(output, stable_json_bytes(result))
        print(
            json.dumps(
                {
                    "worker_id": args.worker_id,
                    "execution_verdict": "PASS",
                    "process_id": result["process_id"],
                    "functional_projection_sha256": result[
                        "functional_projection_sha256"
                    ],
                    "current_fit_projection_sha256": result[
                        "current_fit_projection_sha256"
                    ],
                    "output": str(output),
                },
                sort_keys=True,
            )
        )
        return 0
    except BaseException as error:
        print(f"BLOCKED worker {args.worker_id}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
