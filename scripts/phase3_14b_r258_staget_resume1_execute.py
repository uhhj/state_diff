#!/usr/bin/env python3
"""Execute Stage-T Resume1 process-topology schema recovery exactly once."""
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

from ccda_phase3.phase314b_r258_staget_resume1_process_topology_schema_recovery import (
    BLOCKED_REPORT,
    GATE_CONTRACT_REPORT,
    SUCCESS_REPORT,
    StageTResume1Error,
    blocked_report,
    execute_recovery,
    stable_json_bytes,
    validate_environment_variables,
    validate_repository,
)


def write_once(path: Path, payload: bytes) -> None:
    target = Path(path)
    if target.exists():
        raise StageTResume1Error(f"write-once output already exists: {target}")
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
    gate: Mapping[str, object],
) -> None:
    summary_path = root / SUCCESS_REPORT
    gate_path = root / GATE_CONTRACT_REPORT
    if summary_path.exists() or gate_path.exists():
        raise StageTResume1Error("Stage-T Resume1 success bundle already exists")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_tmp = summary_path.with_suffix(summary_path.suffix + ".tmp")
    gate_tmp = gate_path.with_suffix(gate_path.suffix + ".tmp")
    try:
        with summary_tmp.open("xb") as handle:
            handle.write(stable_json_bytes(summary))
            handle.flush()
            os.fsync(handle.fileno())
        with gate_tmp.open("xb") as handle:
            handle.write(stable_json_bytes(gate))
            handle.flush()
            os.fsync(handle.fileno())
        gate_tmp.replace(gate_path)
        summary_tmp.replace(summary_path)
    finally:
        summary_tmp.unlink(missing_ok=True)
        gate_tmp.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--python-bin", default=sys.executable)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    success = root / SUCCESS_REPORT
    gate = root / GATE_CONTRACT_REPORT
    blocked = root / BLOCKED_REPORT
    if success.exists() or gate.exists() or blocked.exists():
        print("BLOCKED: Stage-T Resume1 output already exists", file=sys.stderr)
        return 2
    repository = None
    try:
        validate_environment_variables()
        repository = validate_repository(root)
        result, gate_result = execute_recovery(
            root=root,
            repository=repository,
            python_bin=args.python_bin,
        )
        write_success_bundle(root, result, gate_result)
        recovered = result["recovered_stage_t_result"]
        matrix = recovered["candidate_matrix"]
        execution = recovered["confirmation_execution"]
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
                    "recovered_stage_t_result_sha256": result[
                        "recovered_stage_t_result_sha256"
                    ],
                    "recovered_stage_t_gate_contract_sha256": result[
                        "recovered_stage_t_gate_contract_sha256"
                    ],
                    "total_oof_fit_count": execution["total_oof_fit_count"],
                    "total_callback_pair_count": execution[
                        "total_callback_pair_count"
                    ],
                    "mechanism_eligible_cells": matrix[
                        "mechanism_eligible_cell_count"
                    ],
                    "fidelity_eligible_cells": matrix[
                        "fidelity_eligible_cell_count"
                    ],
                    "matrix_eligible_backbones": matrix[
                        "matrix_eligible_backbone_count"
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
