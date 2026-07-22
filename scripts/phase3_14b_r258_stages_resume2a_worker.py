#!/usr/bin/env python3
"""Disposable worker for Stage-S Resume2A process-boundary smoke."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ccda_phase3.phase314b_r258_stages_resume2a_cold_cuda_process_boundary_smoke import (
    StageSResume2AError,
    cold_control_smoke_payload,
    environment_probe_payload,
    load_json,
    stable_json_bytes,
    validate_environment_variables,
)


def write_once(path: Path, payload: bytes) -> None:
    target = Path(path)
    if target.exists():
        raise StageSResume2AError(f"write-once worker output exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
    temporary.replace(target)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        required=True,
        choices=("environment-probe", "cold-control-smoke"),
    )
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--probe")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        validate_environment_variables()
        root = Path(args.root).resolve()
        if args.mode == "environment-probe":
            if args.probe is not None:
                raise StageSResume2AError("probe mode does not accept --probe")
            result = environment_probe_payload(root)
        else:
            if not args.probe:
                raise StageSResume2AError("cold-control-smoke requires --probe")
            result = cold_control_smoke_payload(
                root=root,
                probe_payload=load_json(Path(args.probe).resolve()),
            )
        output = Path(args.output).resolve()
        write_once(output, stable_json_bytes(result))
        print(
            json.dumps(
                {
                    "mode": args.mode,
                    "execution_verdict": "PASS",
                    "process_id": result["process_id"],
                    "output": str(output),
                },
                sort_keys=True,
            )
        )
        return 0
    except BaseException as error:
        print(f"BLOCKED {args.mode}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
