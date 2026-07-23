#!/usr/bin/env python3
"""Execute the one authorized Stage-V Resume1 recovery."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ccda_phase3.phase314b_r258_stagev_resume1_preholdout_cuda_recovery import (
    RESUME1_BLOCKED_REPORT,
    RESUME1_SUCCESS_REPORT,
    blocked_report,
    run_recovery,
    stable_json_bytes,
    validate_cuda_admission,
    validate_environment_variables,
    validate_repository,
    write_once,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--python-bin", default=sys.executable)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    success = root / RESUME1_SUCCESS_REPORT
    blocked = root / RESUME1_BLOCKED_REPORT
    repository = None

    if success.exists() or blocked.exists():
        print(
            "BLOCKED: Stage-V Resume1 output already exists; rerun is forbidden",
            file=sys.stderr,
        )
        return 2

    try:
        validate_environment_variables()
        repository = validate_repository(root)
        cuda_admission = validate_cuda_admission()
        summary = run_recovery(
            root=root,
            repository=repository,
            python_bin=args.python_bin,
            cuda_admission=cuda_admission,
        )
        write_once(success, stable_json_bytes(summary))
        print(
            json.dumps(
                {
                    "execution_verdict": summary["execution_verdict"],
                    "resume1_execution_verdict": summary[
                        "resume1_execution_verdict"
                    ],
                    "scientific_status": summary["scientific_status"],
                    "root_cause": summary["root_cause"],
                    "required_next_path": summary["required_next_path"],
                    "failed_timesteps": summary["failed_timesteps"],
                    "scientific_result_sha256": summary[
                        "scientific_result_sha256"
                    ],
                    "output": str(success),
                },
                sort_keys=True,
            )
        )
        return 0
    except BaseException as error:
        payload = blocked_report(repository=repository, error=error)
        try:
            write_once(blocked, stable_json_bytes(payload))
        except BaseException as write_error:
            print(
                f"BLOCKED: {error}; blocked-evidence write failed: {write_error}",
                file=sys.stderr,
            )
            return 2
        print(f"BLOCKED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
