#!/usr/bin/env python3
"""Write-once blocked evidence for r2.5.6 Stage D.2."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--exit-code", required=True, type=int)
    parser.add_argument("--failed-command", default="")
    parser.add_argument("--failed-line", default="")
    parser.add_argument(
        "--output",
        default="reports/phase3_14b_r256_staged2_blocked_summary.json",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output = (root / args.output).resolve()
    if output.exists():
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    payload = {
        "phase": "Phase3.14b-r2.5.6 Stage D.2",
        "schema": "phase314b_r256_staged2_blocked_v1",
        "verdict": "BLOCKED",
        "scientific_status": "BLOCKED",
        "root_cause":
            "phase314b_r256_staged2_execution_failed_before_completion",
        "exit_code": int(args.exit_code),
        "failed_command": str(args.failed_command),
        "failed_line": str(args.failed_line),
        "source_pickle_loaded_read_only": True,
        "deformable_ravens_executed": False,
        "deformable_ravens_modified": False,
        "gate_selected": False,
        "threshold_changed": False,
        "model_or_branch_repaired": False,
        "diffusion_training": False,
        "reverse_sampling": False,
        "formal_pilot_run": False,
        "checkpoint_saved": False,
        "prediction_tensor_persisted": False,
        "candidate_tensor_persisted": False,
        "npz_saved": False,
        "formal_idm_training": False,
        "action_diverse_data_collection": False,
        "candidate_execution": False,
        "phase4": False,
        "cps": False,
        "train_only_recommendation": None,
        "selected_configuration": None,
    }
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)


if __name__ == "__main__":
    main()
