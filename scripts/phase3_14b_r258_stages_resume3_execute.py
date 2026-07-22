#!/usr/bin/env python3
"""Execute Stage-S Resume3 exactly once."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ccda_phase3.phase314b_r258_stages_resume3_cold_worker_oof_confirmation import (
    BLOCKED_REPORT,
    SUCCESS_REPORT,
    StageSResume3Error,
    blocked_report,
    run_confirmation,
    stable_json_bytes,
    validate_environment_variables,
    validate_repository,
)


def write_once(path: Path, payload: bytes) -> None:
    target = Path(path)
    if target.exists():
        raise StageSResume3Error(f"write-once output exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
    temporary.replace(target)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--python-bin", default=sys.executable)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    success = root / SUCCESS_REPORT
    blocked = root / BLOCKED_REPORT
    if success.exists() or blocked.exists():
        print("BLOCKED: Stage-S Resume3 output already exists", file=sys.stderr)
        return 2
    repository = None
    try:
        validate_environment_variables()
        repository = validate_repository(root)
        result = run_confirmation(
            root=root,
            repository=repository,
            python_bin=args.python_bin,
        )
        write_once(success, stable_json_bytes(result))
        execution = result["confirmation_execution"]
        summary = result["confirmation_summary"]
        topology = result["process_topology"]
        print(
            json.dumps(
                {
                    "execution_verdict": result["execution_verdict"],
                    "scientific_status": result["scientific_status"],
                    "root_cause": result["root_cause"],
                    "required_next_path": result["required_next_path"],
                    "scientific_result_sha256": result[
                        "scientific_result_sha256"
                    ],
                    "environment_probe_process_count": topology[
                        "environment_probe_process_count"
                    ],
                    "cold_science_worker_process_count": topology[
                        "cold_science_worker_process_count"
                    ],
                    "total_oof_fit_count": execution["total_oof_fit_count"],
                    "total_callback_pair_count": execution[
                        "total_callback_pair_count"
                    ],
                    "total_internal_scale_attempt_count": execution[
                        "total_internal_scale_attempt_count"
                    ],
                    "nonzero_support_cell_count": summary[
                        "nonzero_support_cell_count"
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
                f"BLOCKED: {error}; additionally failed to write blocked "
                f"evidence: {write_error}",
                file=sys.stderr,
            )
            return 2
        print(f"BLOCKED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
