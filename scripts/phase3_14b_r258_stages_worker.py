#!/usr/bin/env python3
"""Isolated Stage-S confirmation worker."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ccda_phase3.phase314b_r258_stages_oof_support_confirmation import (
    StageSError,
    stable_json_bytes,
    validate_repository,
    worker_payload,
)


def write_once(path: Path, payload: bytes) -> None:
    target = Path(path)
    if target.exists():
        raise StageSError(f"worker output already exists: {target}")
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
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    try:
        repository = validate_repository(root)
        result = worker_payload(
            worker_id=args.worker_id,
            root=root,
            repository=repository,
        )
        write_once(Path(args.output).resolve(), stable_json_bytes(result))
        print(json.dumps({
            "worker_id": args.worker_id,
            "execution_verdict": "PASS",
            "functional_projection_sha256": result["functional_projection_sha256"],
            "current_fit_projection_sha256": result["current_fit_projection_sha256"],
            "output": str(Path(args.output).resolve()),
        }, sort_keys=True))
        return 0
    except BaseException as error:
        print(f"BLOCKED worker {args.worker_id}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
