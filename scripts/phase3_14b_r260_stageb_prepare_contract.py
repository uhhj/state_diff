#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_FROM_FILE = Path(__file__).resolve().parents[1]
if str(ROOT_FROM_FILE) not in sys.path:
    sys.path.insert(0, str(ROOT_FROM_FILE))

from ccda_phase3 import phase314b_r260_stageb_external_backup_attestation as stageb


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--output")
    parser.add_argument("--import-smoke", action="store_true")
    args = parser.parse_args()
    if args.import_smoke:
        print(stageb.CONTRACT_SCHEMA)
        return 0
    if not args.output:
        raise SystemExit("--output is required")
    payload = stageb.build_backup_contract(Path(args.root).resolve())
    output = Path(args.output).resolve()
    stageb.atomic_write_once(output, stageb.stable_json_bytes(payload))
    print(
        json.dumps(
            {
                "contract": str(output),
                "contract_sha256": payload["contract_sha256"],
                "required_file_count": payload["required_file_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
