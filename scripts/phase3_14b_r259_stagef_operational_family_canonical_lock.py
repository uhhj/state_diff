#!/usr/bin/env python3
"""Single-controller Stage-F evidence-only canonical lock."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--import-smoke", action="store_true")
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--implementation-commit")
    args = parser.parse_args()

    if args.import_smoke:
        print(json.dumps({"import_smoke": "PASS", "repository_root": str(REPOSITORY_ROOT)}))
        return 0
    if not args.implementation_commit:
        parser.error("--implementation-commit is required")

    from ccda_phase3.phase314b_r259_stagef_operational_family_canonical_lock import (
        BLOCKED_REPORT,
        SUCCESS_REPORT,
        atomic_write_once,
        blocked_report,
        build_summary,
        stable_json_bytes,
        validate_repository,
        validate_source_evidence,
    )

    root = Path(args.root).resolve()
    success = root / SUCCESS_REPORT
    blocked = root / BLOCKED_REPORT
    repository = None
    try:
        repository = validate_repository(root, str(args.implementation_commit))
        stagee, worker = validate_source_evidence(root)
        summary = build_summary(repository, stagee, worker)
        atomic_write_once(success, stable_json_bytes(summary) + b"\n")
        print(
            json.dumps(
                {
                    "execution_verdict": summary["execution_verdict"],
                    "scientific_status": summary["scientific_status"],
                    "canonical_recipe_id": summary["lock"]["canonicalization"]["canonical_recipe_id"],
                    "root_cause": summary["root_cause"],
                    "required_next_path": summary["required_next_path"],
                    "summary_sha256": summary["summary_sha256"],
                    "output": str(success),
                },
                sort_keys=True,
            )
        )
        return 0
    except BaseException as error:
        try:
            payload = blocked_report(repository, error)
            atomic_write_once(blocked, stable_json_bytes(payload) + b"\n")
        except BaseException as nested:
            print(
                "BLOCKED: {}; failed to persist blocked report: {}".format(error, nested),
                file=sys.stderr,
            )
            return 2
        print("BLOCKED: {}".format(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
