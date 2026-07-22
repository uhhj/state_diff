#!/usr/bin/env python3
"""Finalize report-only Stage-M external-multiplier stratification."""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Optional, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ccda_phase3.phase314b_r258_stagem_external_multiplier_predicate_stratification import (
    BLOCKED_REPORT,
    STAGEL_RESUME1_REPORT,
    STAGEL_RESUME2_REPORT,
    SUCCESS_REPORT,
    blocked_payload,
    build_report,
    load_json,
    stable_json_bytes,
    validate_repository,
    write_once,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    root = Path(args.root).resolve()
    success = root / SUCCESS_REPORT
    blocked = root / BLOCKED_REPORT
    if success.exists() or blocked.exists():
        print("BLOCKED: Stage-M output already exists", file=sys.stderr)
        return 2
    repository = None
    try:
        repository = validate_repository(root)
        resume1 = load_json(root / STAGEL_RESUME1_REPORT)
        resume2 = load_json(root / STAGEL_RESUME2_REPORT)
        result = build_report(
            repository=repository,
            resume1_report=resume1,
            resume2_report=resume2,
        )
        write_once(success, stable_json_bytes(result))
        print(
            json.dumps(
                {
                    "execution_verdict": result["execution_verdict"],
                    "scientific_status": result["scientific_status"],
                    "root_cause": result["root_cause"],
                    "required_next_path": result["required_next_path"],
                    "dominant_upper_multiplier": result["classification"][
                        "dominant_upper_multiplier"
                    ],
                    "output": str(success),
                },
                sort_keys=True,
            )
        )
        return 0
    except BaseException as error:
        payload = dict(blocked_payload(error, repository))
        payload["traceback"] = traceback.format_exc()
        write_once(blocked, stable_json_bytes(payload))
        print(f"BLOCKED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
