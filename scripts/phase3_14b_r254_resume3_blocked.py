#!/usr/bin/env python3
"""Write-once blocked evidence for r2.5.4 Resume3."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ccda_phase3.phase314b_r23_diagnostics import write_json_once
from ccda_phase3.phase314b_r254_resume3_prediction_adapter import (
    RESUME3_BLOCKED_REPORT_PATH,
    RESUME3_BLOCKED_SUMMARY_PATH,
    RESUME3_PILOT_PATH,
    RESUME3_PREFLIGHT_PATH,
    sha256_file,
)


def write_text_once(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(content)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--exit-code", type=int, required=True)
    parser.add_argument("--failed-line", required=True)
    parser.add_argument("--failed-command", required=True)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    summary_path = root / RESUME3_BLOCKED_SUMMARY_PATH
    report_path = root / RESUME3_BLOCKED_REPORT_PATH
    if summary_path.exists() or report_path.exists():
        raise RuntimeError("refusing to overwrite Resume3 blocked evidence")

    preflight = root / RESUME3_PREFLIGHT_PATH
    pilot = root / RESUME3_PILOT_PATH
    payload: Dict[str, Any] = {
        "phase": "Phase3.14b-r2.5.4 Resume3",
        "resume_generation": 3,
        "verdict": "BLOCKED",
        "root_cause": "phase314b_r254_resume3_execution_failed",
        "exit_code": int(args.exit_code),
        "failed_line": str(args.failed_line),
        "failed_command": str(args.failed_command),
        "preflight_report": preflight.relative_to(root).as_posix() if preflight.is_file() else None,
        "preflight_sha256": sha256_file(preflight) if preflight.is_file() else None,
        "pilot_report": pilot.relative_to(root).as_posix() if pilot.is_file() else None,
        "pilot_sha256": sha256_file(pilot) if pilot.is_file() else None,
        "functional_prior_contract_required": True,
        "prediction_reconstruction_contract_required": True,
        "prediction_tensor_persisted": False,
        "robot_proxy_attribution_interpretable": False,
        "train_only_recommendation": None,
        "selected_configuration": None,
        "reverse_sampling_rerun": False,
        "validation_targets_used": False,
        "formal_test_read": False,
        "formal_training": False,
        "idm": False,
        "candidate_execution": False,
        "phase4": False,
        "cps": False,
    }
    write_json_once(summary_path, payload)
    lines = [
        "# Phase3.14b-r2.5.4 Resume3 Blocked",
        "",
        "- Verdict: `BLOCKED`",
        f"- Failed line: `{args.failed_line}`",
        f"- Failed command: `{args.failed_command}`",
        f"- Exit code: `{args.exit_code}`",
        f"- Preflight SHA256: `{payload['preflight_sha256']}`",
        f"- Pilot SHA256: `{payload['pilot_sha256']}`",
        "- Prediction tensor persisted: `False`",
        "- Robot-proxy attribution interpretable: `False`",
        "- Train-only recommendation: `None`",
        "- Selected configuration: `None`",
        "- Reverse sampling rerun: `False`",
        "",
    ]
    write_text_once(report_path, "\n".join(lines))
    print(json.dumps({"verdict": "BLOCKED", "exit_code": args.exit_code}, sort_keys=True))


if __name__ == "__main__":
    main()
