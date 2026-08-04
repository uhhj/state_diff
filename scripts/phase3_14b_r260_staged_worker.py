#!/usr/bin/env python3
"""Stage-D objective-train candidate-signal worker."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _bootstrap() -> Path:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


_bootstrap()
from ccda_phase3 import phase314b_r260_staged_candidate_signal_benchmark as staged  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--output")
    parser.add_argument("--import-smoke", action="store_true")
    args = parser.parse_args()
    if args.import_smoke:
        print(json.dumps({"import_smoke": True, "phase": staged.PHASE}, sort_keys=True))
        return 0
    if not args.output:
        raise SystemExit("--output is required")
    result = staged.run_candidate_benchmark(Path(args.root).resolve())
    staged.atomic_write_once(Path(args.output).resolve(), staged.stable_json_bytes(result))
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
