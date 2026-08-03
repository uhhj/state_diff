#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_FROM_FILE = Path(__file__).resolve().parents[1]
if str(ROOT_FROM_FILE) not in sys.path:
    sys.path.insert(0, str(ROOT_FROM_FILE))

from ccda_phase3 import phase314b_r260_stagec_objective_train_baseline as stagec


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--output")
    parser.add_argument("--import-smoke", action="store_true")
    args = parser.parse_args()
    if args.import_smoke:
        print(stagec.SCHEMA)
        return 0
    if not args.output:
        raise SystemExit("--output is required")
    root = Path(args.root).resolve()
    output = Path(args.output).resolve()
    result = stagec.run_objective_train_analysis(root)
    stagec.atomic_write_once(output, stagec.stable_json_bytes(result))
    print(json.dumps({"worker_result_sha256": result["worker_result_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
