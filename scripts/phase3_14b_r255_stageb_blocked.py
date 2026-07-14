#!/usr/bin/env python3
"""Write-once blocked evidence for Stage B execution failures."""
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
        default="reports/phase3_14b_r255_stageb_blocked_summary.json",
    )
    args = parser.parse_args()
    output = (Path(args.root).resolve() / args.output).resolve()
    if output.exists():
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".tmp.{os.getpid()}")
    root = Path(args.root).resolve()
    final_data_root = root / "data/phase3_state_v3_slack"
    payload = {
        "phase": "Phase3.14b-r2.5.5 Stage B",
        "schema": "phase314b_r255_stageb_blocked_v1",
        "verdict": "BLOCKED",
        "scientific_status": "BLOCKED",
        "root_cause": "phase314b_r255_stageb_execution_failed_before_materialization_completion",
        "exit_code": int(args.exit_code),
        "failed_command": str(args.failed_command),
        "failed_line": str(args.failed_line),
        "robot_proxy_attribution_interpretable": False,
        "train_only_recommendation": None,
        "selected_configuration": None,
        "new_state_v3_data_root_present": final_data_root.exists(),
        "new_state_v3_manifest_present": (
            final_data_root / "manifest.json"
        ).is_file(),
        "new_state_v3_dataset_present": (
            final_data_root
            / "migrated/phase3_14b_r255_stageb_dataset.npz"
        ).is_file(),
        "new_state_v3_windows_present": (
            final_data_root
            / "windows/phase3_14b_r255_stageb_windows.npz"
        ).is_file(),
        "new_cache_written": False,
        "diffusion_training": False,
        "reverse_sampling": False,
        "idm": False,
        "candidate_execution": False,
        "phase4": False,
        "cps": False,
    }
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)


if __name__ == "__main__":
    main()
