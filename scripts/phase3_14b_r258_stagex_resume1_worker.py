#!/usr/bin/env python3
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
    parser.add_argument("--root")
    parser.add_argument("--probe")
    parser.add_argument("--repository-head")
    parser.add_argument("--stageu-contract")
    parser.add_argument("--output")
    args = parser.parse_args()
    from ccda_phase3.phase314b_r258_stagex_resume1_import_crossfit_recovery import (
        run_nested_oof,
        stable_json_bytes,
    )
    if args.import_smoke:
        return 0
    required = {
        "root": args.root,
        "probe": args.probe,
        "repository_head": args.repository_head,
        "stageu_contract": args.stageu_contract,
        "output": args.output,
    }
    missing = sorted(key for key, value in required.items() if not value)
    if missing:
        parser.error("missing required arguments: {}".format(", ".join(missing)))
    probe = json.loads(Path(args.probe).read_text(encoding="utf-8"))
    contract = json.loads(Path(args.stageu_contract).read_text(encoding="utf-8"))
    payload = run_nested_oof(
        root=Path(args.root),
        probe_payload=probe,
        repository_head=str(args.repository_head),
        stageu_contract=contract,
    )
    Path(args.output).write_bytes(stable_json_bytes(payload) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
