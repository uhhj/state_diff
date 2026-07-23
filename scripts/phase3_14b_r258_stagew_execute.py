#!/usr/bin/env python3
"""Execute the Stage-W evidence-only transfer-failure audit once."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ccda_phase3.phase314b_r258_stagew_selection_holdout_transfer_failure_audit import (
    BLOCKED_REPORT,
    OUTPUT_REPORT,
    blocked_report,
    execute_audit,
    stable_json_bytes,
    validate_environment_variables,
    validate_repository,
    write_once,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    output = root / OUTPUT_REPORT
    blocked = root / BLOCKED_REPORT
    repository = None

    if output.exists() or blocked.exists():
        print("BLOCKED: Stage-W output already exists", file=sys.stderr)
        return 2

    try:
        validate_environment_variables()
        repository = validate_repository(root)
        result = execute_audit(root=root, repository=repository)
        write_once(output, stable_json_bytes(result))
        print(
            json.dumps(
                {
                    "execution_verdict": result["execution_verdict"],
                    "scientific_status": result["scientific_status"],
                    "root_cause": result["root_cause"],
                    "required_next_path": result["required_next_path"],
                    "diagnostic_closest_to_threshold_timestep": result[
                        "cross_timestep_diagnostic"
                    ]["diagnostic_closest_to_threshold_timestep"],
                    "audit_result_sha256": result["audit_result_sha256"],
                    "output": str(output),
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
