#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/data/state_diff2}"
PREFLIGHT="reports/phase3_14b_r25_preflight_summary.json"
PILOT="reports/phase3_14b_r25_pilot_summary.json"
SUMMARY="reports/phase3_14b_r25_summary.json"
REPORT="reports/phase3_14b_r25_report.md"
BLOCKED_JSON="reports/phase3_14b_r25_blocked_summary.json"
BLOCKED_MD="reports/phase3_14b_r25_blocked_report.md"

cd "${ROOT}"

write_blocked() {
  local exit_code="$1"
  local line_number="$2"
  local command_text="$3"
  python - "${ROOT}" "${exit_code}" "${line_number}" "${command_text}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
exit_code = int(sys.argv[2])
line_number = int(sys.argv[3])
command_text = sys.argv[4]
json_path = root / "reports/phase3_14b_r25_blocked_summary.json"
md_path = root / "reports/phase3_14b_r25_blocked_report.md"
if json_path.exists() or md_path.exists():
    raise SystemExit("refusing to overwrite existing r2.5 blocked report")
report = {
    "phase": "phase3_14b_r25",
    "verdict": "BLOCKED",
    "root_cause": "phase314b_r25_execution_failed",
    "exit_code": exit_code,
    "line": line_number,
    "command": command_text,
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
json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
md_path.write_text(
    "# Phase3.14b-r2.5 Blocked Report\n\n"
    f"- Verdict: `BLOCKED`\n"
    f"- Root cause: `phase314b_r25_execution_failed`\n"
    f"- Exit code: `{exit_code}`\n"
    f"- Line: `{line_number}`\n\n"
    "```text\n" + command_text + "\n```\n\n"
    "Formal test, formal training, IDM, candidate execution, Phase4 and CPS were not authorized.\n",
    encoding="utf-8",
)
PY
}

trap 'code=$?; write_blocked "$code" "$LINENO" "$BASH_COMMAND"; exit "$code"' ERR

for path in "${PREFLIGHT}" "${PILOT}" "${SUMMARY}" "${REPORT}" "${BLOCKED_JSON}" "${BLOCKED_MD}"; do
  if [[ -e "${path}" ]]; then
    echo "Refusing to overwrite existing artifact: ${path}" >&2
    exit 1
  fi
done

python scripts/phase3_14b_r25_preflight.py \
  --root "${ROOT}" \
  --output "${PREFLIGHT}"

python scripts/phase3_14b_r25_run_pilot.py \
  --root "${ROOT}" \
  --preflight-report "${PREFLIGHT}" \
  --prior-steps 5000 \
  --residual-steps 8000 \
  --batch-size 64 \
  --prior-learning-rate 1e-3 \
  --residual-learning-rate 1e-3 \
  --evaluation-noises 8 \
  --reverse-samples 16

python scripts/phase3_14b_r25_finalize.py \
  --root "${ROOT}" \
  --preflight-report "${PREFLIGHT}" \
  --pilot-report "${PILOT}"

trap - ERR
