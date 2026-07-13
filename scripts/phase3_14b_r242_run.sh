#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/data/state_diff2}"
cd "${ROOT}"

BLOCKED_JSON="reports/phase3_14b_r242_blocked_summary.json"
BLOCKED_MD="reports/phase3_14b_r242_blocked_report.md"

on_error() {
  local exit_code="$?"
  local line_number="$1"
  local command_text="$2"
  trap - ERR

  python - "${ROOT}" "${BLOCKED_JSON}" "${BLOCKED_MD}" \
    "${exit_code}" "${line_number}" "${command_text}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
json_path = root / sys.argv[2]
md_path = root / sys.argv[3]
exit_code = int(sys.argv[4])
line_number = int(sys.argv[5])
command_text = sys.argv[6]

payload = {
    "phase": "phase3_14b_r242",
    "verdict": "BLOCKED",
    "root_cause": "phase314b_r242_execution_failed",
    "exit_code": exit_code,
    "line_number": line_number,
    "failing_command": command_text,
    "validation_targets_used": False,
    "formal_test_read": False,
    "formal_training": False,
    "candidate_eligible": False,
    "selected_configuration": None,
    "checkpoint_saved": False,
    "idm": False,
    "candidate_execution": False,
    "phase4": False,
    "cps": False,
}
if not json_path.exists():
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
if not md_path.exists():
    md_path.write_text(
        "\n".join(
            [
                "# Phase3.14b-r2.4.2 Blocked Report",
                "",
                "- Verdict: `BLOCKED`",
                "- Root cause: `phase314b_r242_execution_failed`",
                f"- Exit code: `{exit_code}`",
                f"- Line: `{line_number}`",
                "",
                "```text",
                command_text,
                "```",
                "",
                "Formal test, formal training, IDM, candidate execution, "
                "Phase4 and CPS were not authorized.",
                "",
            ]
        ),
        encoding="utf-8",
    )
PY
  exit "${exit_code}"
}

trap 'on_error "${LINENO}" "${BASH_COMMAND}"' ERR

python scripts/phase3_14b_r242_preflight.py \
  --root "${ROOT}"

python scripts/phase3_14b_r242_run_pilot.py \
  --root "${ROOT}" \
  --prior-steps 5000 \
  --residual-steps 8000 \
  --paired-residual-steps 8000 \
  --batch-size 64 \
  --prior-learning-rate 1e-3 \
  --residual-learning-rate 1e-3 \
  --decoupled-prior-learning-rate 1e-4 \
  --evaluation-noises 8

python scripts/phase3_14b_r242_finalize.py \
  --root "${ROOT}"
