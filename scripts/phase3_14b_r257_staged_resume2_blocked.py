#!/usr/bin/env python3
"""Write-once blocked evidence for r2.5.7 Stage-D Resume2."""
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
            "phase3_14b_r257_staged_resume2_blocked_summary.json"
        ),
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output = (root / args.output).resolve()
    if output.exists():
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    payload = {
        "phase":
            "Phase3.14b-r2.5.7 Stage D Resume2",
        "schema":
            "phase314b_r257_staged_resume2_blocked_v1",
        "verdict": "BLOCKED",
        "scientific_status": "BLOCKED",
        "root_cause": (
            "phase314b_r257_staged_resume2_"
            "execution_failed_before_completion"
        ),
        "exit_code": int(args.exit_code),
        "failed_command": str(args.failed_command),
        "failed_line": str(args.failed_line),
        "resume1_implementation_commit":
            "d34838bfebbde2376d128dff398dadcc97877191",
        "resume1_blocked_sha256":
            "ab1ec6cd75c7164b29da29e296925fb9ed9897bcc552fd835060e0b1e1c4eb24",
        "resume1_test_gate_sha256":
            "d31a882153134304b37f3fd2b16b853fdec8c177ef49c3ee879813a43ea06298",
        "resume2_correction_applied": False,
        "resume1_files_modified": False,
        "resume1_blocked_modified": False,
        "resume1_test_gate_modified": False,
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
