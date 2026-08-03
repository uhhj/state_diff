#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ccda_phase3 import phase314b_r259_stageh_frozen_probe_transfer_audit as stageh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--implementation-commit")
    parser.add_argument("--import-smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.import_smoke:
        print(stageh.SCHEMA)
        return 0
    if not args.implementation_commit:
        raise SystemExit("--implementation-commit is required")
    root = Path(args.root).resolve()
    success = root / stageh.SUCCESS_REPORT
    blocked = root / stageh.BLOCKED_REPORT
    repository = None
    try:
        repository = stageh.validate_repository(root, args.implementation_commit)
        probe, worker, source_summary = stageh.validate_source_evidence(root)
        payload = stageh.build_summary(repository, probe, worker, source_summary)
        stageh.atomic_write_once(success, stageh.stable_json_bytes(payload))
        print(
            json.dumps(
                {
                    "execution_verdict": payload["execution_verdict"],
                    "scientific_status": payload["scientific_status"],
                    "classification": payload["audit"]["classification"],
                    "required_next_path": payload["required_next_path"],
                    "report": str(success),
                },
                sort_keys=True,
            )
        )
        return 0
    except Exception as error:
        if success.exists() or blocked.exists():
            raise
        payload = stageh.blocked_report(repository, error)
        stageh.atomic_write_once(blocked, stageh.stable_json_bytes(payload))
        print(
            json.dumps(
                {
                    "execution_verdict": "BLOCKED",
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "report": str(blocked),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
