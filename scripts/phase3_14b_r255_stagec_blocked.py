#!/usr/bin/env python3
"""Best-effort write-once blocked evidence for Stage C."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--exit-code", type=int, required=True)
    parser.add_argument("--failed-command", default="")
    parser.add_argument("--failed-line", default="")
    parser.add_argument(
        "--output",
        default="reports/phase3_14b_r255_stagec_blocked_summary.json",
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()
    output = (root / args.output).resolve()
    if output.exists():
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.{os.getpid()}.tmp"
    payload = {
        "phase": "Phase3.14b-r2.5.5 Stage C",
        "schema": "phase314b_r255_stagec_blocked_v1",
        "verdict": "BLOCKED",
        "scientific_status": "BLOCKED",
        "root_cause": "phase314b_r255_stagec_execution_failed_before_completion",
        "exit_code": int(args.exit_code),
        "failed_command": str(args.failed_command),
        "failed_line": str(args.failed_line),
        "legacy_state_v2_robot_proxy_attribution_interpretable": False,
        "state_v3_robot_proxy_attribution_interpretable": False,
        "train_only_recommendation": None,
        "selected_configuration": None,
        "diffusion_training": False,
        "reverse_sampling": False,
        "idm_training": False,
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
