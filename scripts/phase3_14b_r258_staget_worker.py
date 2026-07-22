#!/usr/bin/env python3
"""Run one cold Stage-T candidate-matrix worker."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ccda_phase3.phase314b_r258_staget_aligned_gate_candidate_matrix import (
    StageTError,
    load_json,
    stable_json_bytes,
    validate_environment_variables,
    worker_payload,
)


def write_once(path: Path, payload: bytes) -> None:
    target = Path(path)
    if target.exists():
        raise StageTError(f"worker output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
    temporary.replace(target)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--repository-head", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        validate_environment_variables()
        result = worker_payload(
            worker_id=args.worker_id,
            root=Path(args.root).resolve(),
            probe_payload=load_json(Path(args.probe).resolve()),
            repository_head=args.repository_head,
        )
        output = Path(args.output).resolve()
        write_once(output, stable_json_bytes(result))
        print(json.dumps({
            "worker_id": args.worker_id,
            "execution_verdict": "PASS",
            "process_id": result["process_id"],
            "candidate_matrix_sha256": result["candidate_matrix_sha256"],
            "gate_contract_sha256": result["gate_contract_sha256"],
            "output": str(output),
        }, sort_keys=True))
        return 0
    except BaseException as error:
        print(f"BLOCKED worker {args.worker_id}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
