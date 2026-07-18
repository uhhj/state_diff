#!/usr/bin/env python3
"""Write-once blocked evidence for Stage-F Resume1."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


def git_value(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=str(root),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    return completed.stdout.strip() if completed.returncode == 0 else ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--exit-code", required=True, type=int)
    parser.add_argument("--failed-command", default="")
    parser.add_argument("--failed-line", default="")
    parser.add_argument(
        "--output",
        default="reports/phase3_14b_r258_stagef_resume1_blocked_summary.json",
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()
    output = (
        Path(args.output).resolve()
        if Path(args.output).is_absolute()
        else (root / args.output).resolve()
    )
    if output.exists():
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    payload = {
        "phase": "Phase3.14b-r2.5.8 Stage F Resume1",
        "schema": "phase314b_r258_stagef_resume1_blocked_v1",
        "verdict": "BLOCKED",
        "scientific_status": "BLOCKED",
        "root_cause": (
            "phase314b_r258_stagef_resume1_execution_failed_before_completion"
        ),
        "exit_code": int(args.exit_code),
        "failed_command": str(args.failed_command),
        "failed_line": str(args.failed_line),
        "head": git_value(root, "rev-parse", "HEAD"),
        "origin_experiment1": git_value(root, "rev-parse", "origin/Experiment1"),
        "base_evidence_commit": "6758ea7ad800667a436b0243d3b1f6c63256d854",
        "original_stagef_implementation_commit": (
            "519531f411c43b17a668c3c6c1a43b46a94f40a7"
        ),
        "translation_fix_completed": False,
        "scientific_result_sealed": False,
        "success_evidence_commit_created": False,
        "push_completed": False,
        "selected_configuration": None,
        "train_only_recommendation": None,
        "selection_holdout_evaluated": False,
        "frozen_probe_accessed": False,
        "new_diffusion_model_candidate_trained": False,
        "reverse_sampling_run": False,
        "formal_training_run": False,
        "idm_run": False,
        "candidate_execution": False,
        "deformable_ravens_executed": False,
        "phase4": False,
        "cps": False,
        "checkpoint_saved": False,
        "weights_persisted": False,
        "surrogate_weights_persisted": False,
        "prediction_tensor_persisted": False,
        "candidate_tensor_persisted": False,
        "npz_saved": False,
        "cache_saved": False,
        "image_saved": False,
        "video_saved": False,
    }
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)


if __name__ == "__main__":
    main()
