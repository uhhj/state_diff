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
    parser.add_argument("--mode", choices=("environment-probe", "science"))
    parser.add_argument("--worker-slot", choices=("A", "B"))
    parser.add_argument("--root")
    parser.add_argument("--probe")
    parser.add_argument("--repository-head")
    parser.add_argument("--stageu-contract")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.import_smoke:
        return 0
    from ccda_phase3.phase314b_r259_stagec_resume1_portable_gpu_compatibility_recovery import (
        atomic_write_once,
        load_json,
        make_probe_evidence,
        make_worker_evidence,
        stable_json_bytes,
    )
    if not args.mode or not args.root or not args.output:
        parser.error("--mode, --root and --output are required")
    try:
        root = Path(args.root).resolve()
        if args.mode == "environment-probe":
            if any((args.worker_slot, args.probe, args.repository_head, args.stageu_contract)):
                parser.error("environment-probe accepts no science arguments")
            payload = make_probe_evidence(root)
        else:
            missing = [
                name
                for name, value in (
                    ("worker-slot", args.worker_slot),
                    ("probe", args.probe),
                    ("repository-head", args.repository_head),
                    ("stageu-contract", args.stageu_contract),
                )
                if not value
            ]
            if missing:
                parser.error("science missing: {}".format(", ".join(missing)))
            payload = make_worker_evidence(
                root=root,
                probe_payload=load_json(Path(args.probe).resolve()),
                repository_head=str(args.repository_head),
                stageu_contract=load_json(Path(args.stageu_contract).resolve()),
                worker_slot=str(args.worker_slot),
            )
        output = Path(args.output).resolve()
        atomic_write_once(output, stable_json_bytes(payload) + b"\n")
        print(
            json.dumps(
                {
                    "mode": args.mode,
                    "worker_slot": args.worker_slot,
                    "execution_verdict": "PASS",
                    "process_id": payload["process_id"],
                    "output": str(output),
                },
                sort_keys=True,
            )
        )
        return 0
    except BaseException as error:
        print("BLOCKED {}: {}".format(args.mode, error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
