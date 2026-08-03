#!/usr/bin/env python3
"""Single-pair worker for r2.6.0 Stage-A data genesis."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ccda_phase3 import phase314b_r260_stagea_data_universe_genesis as stagea  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--write-ahead", required=True)
    parser.add_argument("--role", required=True, choices=sorted(stagea.ROLE_BY_NAME))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--import-smoke", action="store_true")
    args = parser.parse_args()
    if args.import_smoke:
        print(stagea.SCHEMA)
        return 0
    result = stagea.generate_pair_job(
        {
            "root": str(Path(args.root).resolve()),
            "dataset_root": str(Path(args.dataset_root).resolve()),
            "write_ahead": str(Path(args.write_ahead).resolve()),
            "role": args.role,
            "seed": int(args.seed),
            "implementation_commit": args.implementation_commit,
        }
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
