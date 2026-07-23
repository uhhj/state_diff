#!/usr/bin/env python3
"""Execute Stage-U candidate-frontier lock exactly once."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Mapping

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ccda_phase3.phase314b_r258_stageu_candidate_frontier_lock import (
    BLOCKED_REPORT,
    CONTRACT_REPORT,
    SUCCESS_REPORT,
    StageUError,
    blocked_report,
    execute_lock,
    stable_json_bytes,
    validate_environment_variables,
    validate_repository,
)


def write_once(path: Path, payload: bytes) -> None:
    target = Path(path)
    if target.exists():
        raise StageUError(f"write-once output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(target)


def write_success_bundle(
    root: Path,
    summary: Mapping[str, object],
    contract: Mapping[str, object],
) -> None:
    summary_path = root / SUCCESS_REPORT
    contract_path = root / CONTRACT_REPORT
    if summary_path.exists() or contract_path.exists():
        raise StageUError("Stage-U success bundle already exists")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_tmp = summary_path.with_suffix(summary_path.suffix + ".tmp")
    contract_tmp = contract_path.with_suffix(contract_path.suffix + ".tmp")
    try:
        with summary_tmp.open("xb") as handle:
            handle.write(stable_json_bytes(summary))
            handle.flush()
            os.fsync(handle.fileno())
        with contract_tmp.open("xb") as handle:
            handle.write(stable_json_bytes(contract))
            handle.flush()
            os.fsync(handle.fileno())
        contract_tmp.replace(contract_path)
        summary_tmp.replace(summary_path)
    finally:
        summary_tmp.unlink(missing_ok=True)
        contract_tmp.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    success = root / SUCCESS_REPORT
    contract = root / CONTRACT_REPORT
    blocked = root / BLOCKED_REPORT
    if success.exists() or contract.exists() or blocked.exists():
        print("BLOCKED: Stage-U output already exists", file=sys.stderr)
        return 2
    repository = None
    try:
        validate_environment_variables()
        repository = validate_repository(root)
        result, lock_contract = execute_lock(root=root, repository=repository)
        write_success_bundle(root, result, lock_contract)
        print(
            json.dumps(
                {
                    "execution_verdict": result["execution_verdict"],
                    "scientific_status": result["scientific_status"],
                    "root_cause": result["root_cause"],
                    "required_next_path": result["required_next_path"],
                    "scientific_result_sha256": result[
                        "scientific_result_sha256"
                    ],
                    "frontier_contract_sha256": result[
                        "frontier_contract_sha256"
                    ],
                    "frontier_backbone": result["frontier_lock_summary"][
                        "unique_frontier_backbone"
                    ],
                    "locked_timesteps": result["frontier_lock_summary"][
                        "locked_timesteps"
                    ],
                    "stageu_oof_fit_count": result["stageu_execution"][
                        "oof_fit_count"
                    ],
                    "selection_holdout_evaluated": result[
                        "selection_holdout_evaluated"
                    ],
                    "output": str(success),
                },
                sort_keys=True,
            )
        )
        return 0
    except BaseException as error:
        payload = blocked_report(repository=repository, error=error)
        try:
            write_once(blocked, stable_json_bytes(payload))
        except BaseException as write_error:
            print(
                f"BLOCKED: {error}; additionally failed to write blocked "
                f"evidence: {write_error}",
                file=sys.stderr,
            )
            return 2
        print(f"BLOCKED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
