#!/usr/bin/env python3
"""Execute the Phase3.14b-r2.5.9 Stage-B evidence-only audit."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ccda_phase3.phase314b_r259_stageb_outer_crossfit_tail_failure_audit import (
    BLOCKED_REPORT,
    SUCCESS_REPORT,
    blocked_report,
    run_audit,
    stable_json_bytes,
    validate_repository,
    write_once,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--import-smoke", action="store_true")
    args = parser.parse_args()
    if args.import_smoke:
        print(json.dumps({"import_smoke": "PASS", "repository_root": str(REPOSITORY_ROOT)}))
        return 0
    root = Path(args.root).resolve()
    success = root / SUCCESS_REPORT
    blocked = root / BLOCKED_REPORT
    if success.exists() or blocked.exists():
        print("BLOCKED: Stage-B terminal output already exists", file=sys.stderr)
        return 2
    repository = None
    try:
        repository = validate_repository(root)
        result = run_audit(root)
        write_once(success, stable_json_bytes(result))
        print(
            json.dumps(
                {
                    "execution_verdict": result["execution_verdict"],
                    "scientific_status": result["scientific_status"],
                    "root_cause": result["root_cause"],
                    "primary_failure_locus": result["primary_failure_locus"],
                    "required_next_path": result["required_next_path"],
                    "audit_sha256": result["audit_sha256"],
                    "output": str(success),
                },
                sort_keys=True,
            )
        )
        return 0
    except BaseException as error:
        payload = blocked_report(repository, error)
        try:
            write_once(blocked, stable_json_bytes(payload))
        except BaseException as write_error:
            print(
                "BLOCKED: {}; failed to write blocked evidence: {}".format(
                    error, write_error
                ),
                file=sys.stderr,
            )
            return 2
        print("BLOCKED: {}".format(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
