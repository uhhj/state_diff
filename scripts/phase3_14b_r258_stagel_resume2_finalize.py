#!/usr/bin/env python3
"""Finalize Stage-L Resume2 classification repair without scientific replay."""
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

from ccda_phase3.phase314b_r258_stagel_resume2_stratified_classifier_repair import (
    BASE_REPORT,
    BLOCKED_REPORT,
    SUCCESS_REPORT,
    blocked_payload,
    build_corrected_report,
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
        print("BLOCKED: Stage-L Resume2 output already exists", file=sys.stderr)
        return 2
    repository = None
    try:
        repository = validate_repository(root)
        base = load_json(root / BASE_REPORT)
        result = build_corrected_report(root=root, repository=repository, base_report=base)
        write_once(success, stable_json_bytes(result))
        print(
            json.dumps(
                {
                    "execution_verdict": result["execution_verdict"],
                    "scientific_status": result["scientific_status"],
                    "root_cause": result["root_cause"],
                    "required_next_path": result["required_next_path"],
                    "output": str(success),
                },
                sort_keys=True,
            )
        )
        return 0
    except BaseException as error:
        payload = blocked_payload(error, repository)
        payload = dict(payload)
        payload["traceback"] = traceback.format_exc()
        write_once(blocked, stable_json_bytes(payload))
        print(f"BLOCKED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
