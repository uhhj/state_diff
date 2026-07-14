#!/usr/bin/env python3
"""Write operational failure evidence if Resume5 cannot complete."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ccda_phase3.phase314b_r254_resume5_residual_determinism import (
    sha256_file,
    write_json_once,
    write_text_once,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--exit-code", type=int, required=True)
    parser.add_argument("--failed-line", required=True)
    parser.add_argument("--failed-command", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    summary = root / "reports/phase3_14b_r254_resume5_blocked_summary.json"
    markdown = root / "reports/phase3_14b_r254_resume5_blocked_report.md"
    final_summary = root / "reports/phase3_14b_r254_resume5_summary.json"
    if final_summary.exists() or summary.exists() or markdown.exists():
        return
    evidence = {}
    for relative in (
        "reports/phase3_14b_r254_resume5_test_gate_summary.json",
        "reports/phase3_14b_r254_resume5_preflight_summary.json",
        "reports/phase3_14b_r254_resume5_determinism_audit_summary.json",
    ):
        path = root / relative
        if path.is_file():
            evidence[relative] = sha256_file(path)
    payload = {
        "phase": "Phase3.14b-r2.5.4 Resume5",
        "verdict": "BLOCKED",
        "root_cause": "phase314b_r254_resume5_execution_failed",
        "exit_code": int(args.exit_code),
        "failed_line": str(args.failed_line),
        "failed_command": str(args.failed_command),
        "available_evidence_sha256": evidence,
        "robot_proxy_attribution_interpretable": False,
        "validation_targets_used": False,
        "formal_test_read": False,
        "formal_training": False,
        "reverse_sampling_rerun": False,
        "checkpoint_saved": False,
        "weights_persisted": False,
        "prediction_tensor_persisted": False,
        "candidate_execution": False,
        "idm": False,
        "phase4": False,
        "cps": False,
        "train_only_recommendation": None,
        "selected_configuration": None,
    }
    text = (
        "# Phase3.14b-r2.5.4 Resume5 Blocked\n\n"
        f"- Root cause: `{payload['root_cause']}`\n"
        f"- Exit code: `{payload['exit_code']}`\n"
        f"- Failed line: `{payload['failed_line']}`\n"
        f"- Failed command: `{payload['failed_command']}`\n\n"
        "No robot-proxy metric is interpreted and no model is selected.\n"
    )
    write_text_once(markdown, text)
    write_json_once(summary, payload)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
