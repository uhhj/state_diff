#!/usr/bin/env python3
"""Write-once blocked evidence for Stage-D Resume4 RTX 3090."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="/data/state_diff2",
    )
    parser.add_argument(
        "--exit-code",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--failed-command",
        default="",
    )
    parser.add_argument(
        "--failed-line",
        default="",
    )
    parser.add_argument(
        "--output",
        default=(
            "reports/"
            "phase3_14b_r257_staged_resume4_3090_"
            "blocked_summary.json"
        ),
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output = (root / args.output).resolve()
    if output.exists():
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(
        output.suffix + ".tmp"
    )
    payload = {
        "phase":
            "Phase3.14b-r2.5.7 Stage D Resume4 RTX 3090",
        "schema":
            "phase314b_r257_staged_resume4_3090_blocked_v1",
        "verdict": "BLOCKED",
        "scientific_status": "BLOCKED",
        "root_cause": (
            "phase314b_r257_staged_resume4_3090_"
            "execution_failed_before_completion"
        ),
        "exit_code": int(args.exit_code),
        "failed_command": str(args.failed_command),
        "failed_line": str(args.failed_line),
        "resume3_implementation_commit":
            "ef5eebb964ac7a8198506962646e048ccf0f6605",
        "resume3_blocked_sha256":
            "c769fa1f0af69dfe24612cf9bdb7f43c8ce03fb93f039c069b84b07d088a89cd",
        "resume3_test_gate_sha256":
            "c1cf8558bc0d5a44018812b034e4ce5d3498c290c68c20d0ef8a249938b05adc",
        "new_write_once_namespace":
            "phase3_14b_r257_staged_resume4_3090",
        "operator_selected_gpu":
            "NVIDIA GeForce RTX 3090",
        "resume4_3090_correction_applied": False,
        "resume3_files_modified": False,
        "resume3_blocked_modified": False,
        "resume3_test_gate_modified": False,
        "control_replay_exact": False,
        "selected_configuration": None,
        "train_only_recommendation": None,
        "new_hyperparameter_candidate_run": False,
        "new_objective_variant_run": False,
        "frozen_probe_accessed": False,
        "reverse_sampling_run": False,
        "full_stageb_repaired_model_trained": False,
        "formal_pilot_run": False,
        "checkpoint_saved": False,
        "weights_persisted": False,
        "prediction_tensor_persisted": False,
        "candidate_tensor_persisted": False,
        "npz_saved": False,
        "cache_saved": False,
        "formal_diffusion_training": False,
        "formal_reverse_sampling": False,
        "formal_idm_training": False,
        "action_diverse_data_collection": False,
        "candidate_execution": False,
        "deformable_ravens_executed": False,
        "phase4": False,
        "cps": False,
    }
    with temporary.open(
        "x",
        encoding="utf-8",
    ) as handle:
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
