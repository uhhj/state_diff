#!/usr/bin/env python3
"""Write-once blocked evidence for Stage-D Resume1."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


def git_value(
    root: Path,
    *args: str,
) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=str(root),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    return (
        completed.stdout.strip()
        if completed.returncode == 0
        else ""
    )


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
            "phase3_14b_r258_staged_resume1_"
            "blocked_summary.json"
        ),
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
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temporary = output.with_suffix(
        output.suffix + ".tmp"
    )
    payload = {
        "phase":
            "Phase3.14b-r2.5.8 Stage D Resume1",
        "schema":
            "phase314b_r258_staged_resume1_blocked_v1",
        "verdict":
            "BLOCKED",
        "scientific_status":
            "BLOCKED",
        "root_cause": (
            "phase314b_r258_staged_resume1_"
            "execution_failed_before_completion"
        ),
        "exit_code":
            int(args.exit_code),
        "failed_command":
            str(args.failed_command),
        "failed_line":
            str(args.failed_line),
        "head":
            git_value(
                root,
                "rev-parse",
                "HEAD",
            ),
        "origin_experiment1":
            git_value(
                root,
                "rev-parse",
                "origin/Experiment1",
            ),
        "base_evidence_commit":
            "6866507c42b9bc9d2d271becd1a9423f61710405",
        "original_blocked_sha256":
            (
                "c6176abe55cdbf63506765b5ec67745a6"
                "828a582d6eda70fd95b122e2e012bdc"
            ),
        "test_view_isolation_completed":
            False,
        "scientific_result_sealed":
            False,
        "success_evidence_commit_created":
            False,
        "push_completed":
            False,
        "selected_configuration":
            None,
        "train_only_recommendation":
            None,
        "new_diffusion_model_candidate_trained":
            False,
        "frozen_probe_accessed":
            False,
        "reverse_sampling_run":
            False,
        "full_stageb_repaired_model_trained":
            False,
        "formal_pilot_run":
            False,
        "checkpoint_saved":
            False,
        "weights_persisted":
            False,
        "surrogate_weights_persisted":
            False,
        "prediction_tensor_persisted":
            False,
        "oracle_tensor_persisted":
            False,
        "candidate_tensor_persisted":
            False,
        "npz_saved":
            False,
        "cache_saved":
            False,
        "formal_diffusion_training":
            False,
        "formal_reverse_sampling":
            False,
        "formal_idm_training":
            False,
        "action_diverse_data_collection":
            False,
        "candidate_execution":
            False,
        "deformable_ravens_executed":
            False,
        "phase4":
            False,
        "cps":
            False,
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
    os.replace(
        temporary,
        output,
    )


if __name__ == "__main__":
    main()
