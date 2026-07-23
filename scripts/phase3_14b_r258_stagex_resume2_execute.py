#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ccda_phase3.phase314b_r258_stagex_resume2_lost_worker_result_adjudication import (
    BLOCKED_REPORT,
    SUCCESS_REPORT,
    blocked_report,
    build_adjudication,
    stable_json_bytes,
    validate_adjudication,
    validate_repository,
    write_once,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    repository = None
    try:
        repository = validate_repository(root)
        result = build_adjudication(root, repository)
        validate_adjudication(result)
        write_once(root / SUCCESS_REPORT, stable_json_bytes(result) + b"\n")
        return 0
    except BaseException as error:
        try:
            write_once(
                root / BLOCKED_REPORT,
                stable_json_bytes(blocked_report(repository, error)) + b"\n",
            )
        except BaseException:
            pass
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
