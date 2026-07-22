#!/usr/bin/env python3
"""Write-once report-only finalizer for Phase3.14b-r2.5.8 Stage N."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ccda_phase3.phase314b_r258_stagen_lower_multiplier_post_upper_rejection import (
    BLOCKED_REPORT,
    STAGEM_REPORT,
    SUCCESS_REPORT,
    blocked_report,
    build_report,
    load_json,
    stable_json_bytes,
    validate_implementation_commit,
    write_once,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    success = root / SUCCESS_REPORT
    blocked = root / BLOCKED_REPORT
    if success.exists() or blocked.exists():
        print("BLOCKED: Stage-N output already exists", file=sys.stderr)
        return 2

    repository = None
    try:
        repository = validate_implementation_commit(root)
        stage_m_path = root / STAGEM_REPORT
        before = stage_m_path.read_bytes()
        report = build_report(
            repository=repository,
            stage_m_report=load_json(stage_m_path),
        )
        if stage_m_path.read_bytes() != before:
            raise RuntimeError("Stage-M report changed during Stage-N finalization")
        write_once(success, stable_json_bytes(report))
        print(
            json.dumps(
                {
                    "execution_verdict": report["execution_verdict"],
                    "scientific_status": report["scientific_status"],
                    "root_cause": report["root_cause"],
                    "required_next_path": report["required_next_path"],
                    "dominant_post_upper_predicate": report["classification"][
                        "dominant_post_upper_predicate"
                    ],
                    "report": str(success),
                },
                sort_keys=True,
            )
        )
        return 0
    except BaseException as error:
        try:
            write_once(blocked, stable_json_bytes(blocked_report(repository=repository, error=error)))
        except BaseException as write_error:
            print(f"BLOCKED: {error}; blocked report failed: {write_error}", file=sys.stderr)
            return 3
        print(f"BLOCKED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
