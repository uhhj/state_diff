#!/usr/bin/env python3
"""Run two isolated Stage-S workers and seal confirmation evidence."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ccda_phase3.phase314b_r258_stages_oof_support_confirmation import (
    BLOCKED_REPORT,
    SUCCESS_REPORT,
    StageSError,
    blocked_report,
    run_confirmation,
    stable_json_bytes,
    validate_environment_variables,
    validate_repository,
)


def write_once(path: Path, payload: bytes) -> None:
    target = Path(path)
    if target.exists():
        raise StageSError(f"write-once output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
    temporary.replace(target)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--python-bin", default=sys.executable)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    success = root / SUCCESS_REPORT
    blocked = root / BLOCKED_REPORT
    if success.exists() or blocked.exists():
        print("BLOCKED: Stage-S output already exists", file=sys.stderr)
        return 2
    repository = None
    try:
        validate_environment_variables()
        repository = validate_repository(root)
        result = run_confirmation(
            root=root,
            repository=repository,
            python_bin=args.python_bin,
        )
        write_once(success, stable_json_bytes(result))
        print(json.dumps({
            "execution_verdict": result["execution_verdict"],
            "scientific_status": result["scientific_status"],
            "root_cause": result["root_cause"],
            "required_next_path": result["required_next_path"],
            "scientific_result_sha256": result["scientific_result_sha256"],
            "worker_count": result["confirmation_execution"]["worker_count"],
            "total_oof_fit_count": result["confirmation_execution"]["total_oof_fit_count"],
            "total_callback_pair_count": result["confirmation_execution"]["total_callback_pair_count"],
            "nonzero_support_cell_count": result["confirmation_summary"]["nonzero_support_cell_count"],
            "output": str(success),
        }, sort_keys=True))
        return 0
    except BaseException as error:
        payload = blocked_report(repository=repository, error=error)
        try:
            write_once(blocked, stable_json_bytes(payload))
        except BaseException as write_error:
            print(
                f"BLOCKED: {error}; additionally failed to write blocked evidence: {write_error}",
                file=sys.stderr,
            )
            return 2
        print(f"BLOCKED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
