#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/data/state_diff2}"
cd "$ROOT"

blocked_report() {
  local status="$1"
  local command="$2"
  python - "$ROOT" "$status" "$command" <<'PY'
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
status = int(sys.argv[2])
command = sys.argv[3]
json_path = root / "reports/phase3_14b_r232_blocked_summary.json"
md_path = root / "reports/phase3_14b_r232_blocked_report.md"
payload = {
    "phase": "phase3_14b_r232",
    "verdict": "BLOCKED",
    "root_cause": "phase314b_r232_runtime_step_failed",
    "exit_status": status,
    "failing_command": command,
    "formal_test_read": False,
    "formal_training": False,
    "candidate_eligible": False,
    "checkpoint_saved": False,
    "idm": False,
    "candidate_execution": False,
    "phase4": False,
    "cps": False,
}
if not json_path.exists():
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
if not md_path.exists():
    md_path.write_text(
        "# Phase3.14b-r2.3.2 Blocked Report\n\n"
        f"- Exit status: `{status}`\n"
        f"- Failing command: `{command}`\n"
        "- Formal test read: `False`\n"
        "- Formal training / IDM / execution / Phase4 / CPS: `False`\n"
    )
PY
}

trap 'status=$?; blocked_report "$status" "$BASH_COMMAND"; exit "$status"' ERR

python scripts/phase3_14b_r232_preflight.py --root "$ROOT"
python scripts/phase3_14b_r232_run_controls.py --root "$ROOT"
python scripts/phase3_14b_r232_finalize.py --root "$ROOT"

trap - ERR
