#!/usr/bin/env python3
"""Disposable Stage-D environment-probe and cold science-worker entry point."""
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
        print(json.dumps({"import_smoke": "PASS", "repository_root": str(REPOSITORY_ROOT)}))
        return 0

    from ccda_phase3.phase314b_r259_staged_t10_risk_descriptor_calibration_repair import (
        atomic_write_once,
        load_json,
        make_probe_evidence,
        run_stage_d,
        sha256_bytes,
        stable_json_bytes,
        validate_worker_evidence,
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
            payload = dict(
                run_stage_d(
                    root=root,
                    probe_payload=load_json(Path(args.probe).resolve()),
                    repository_head=str(args.repository_head),
                    stageu_contract=load_json(Path(args.stageu_contract).resolve()),
                )
            )
            payload.pop("worker_result_sha256", None)
            payload["worker_slot"] = str(args.worker_slot)
            payload["worker_result_sha256"] = sha256_bytes(stable_json_bytes(payload))
            validate_worker_evidence(payload)
        output = Path(args.output).resolve()
        atomic_write_once(output, stable_json_bytes(payload) + b"\n")
        print(
            json.dumps(
                {
                    "mode": args.mode,
                    "worker_slot": args.worker_slot,
                    "execution_verdict": "PASS",
                    "scientific_status": payload.get("scientific_status"),
                    "process_id": payload["process_id"],
                    "worker_result_sha256": payload.get("worker_result_sha256"),
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
