#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/data/state_diff2}"
cd "${ROOT}"

PREFLIGHT="reports/phase3_14b_r253_resume_preflight_summary.json"
BLOCKED_JSON="reports/phase3_14b_r253_resume_blocked_summary.json"
BLOCKED_MD="reports/phase3_14b_r253_resume_blocked_report.md"
PILOT="reports/phase3_14b_r253_pilot_summary.json"
PILOT_SHA="a6a8aa2faf739e80159af7e70cb57b0e2a4c852ab5af5c2d7ecb29ff31116c86"

if [[ -e "${PREFLIGHT}" || -e "${BLOCKED_JSON}" || -e "${BLOCKED_MD}" ]]; then
  echo "Resume1 output already exists; refusing to overwrite" >&2
  exit 2
fi

write_blocked() {
  local exit_code="$1"
  local failed_line="$2"
  python - "${ROOT}" "${exit_code}" "${failed_line}" "${PREFLIGHT}" "${PILOT}" "${PILOT_SHA}" <<'PY'
import hashlib
import json
import sys
import traceback
from pathlib import Path

root = Path(sys.argv[1]).resolve()
exit_code = int(sys.argv[2])
failed_line = sys.argv[3]
preflight = sys.argv[4]
pilot_relative = sys.argv[5]
expected_sha = sys.argv[6]
pilot = root / pilot_relative
actual_sha = hashlib.sha256(pilot.read_bytes()).hexdigest() if pilot.is_file() else None
final_summary = root / "reports/phase3_14b_r253_summary.json"
final_report = root / "reports/phase3_14b_r253_report.md"
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
payload = {
    "phase": "Phase3.14b-r2.5.3-Resume1",
    "verdict": "BLOCKED",
    "root_cause": "phase314b_r253_resume1_finalization_failed",
    "exit_code": exit_code,
    "failed_line": failed_line,
    "finalizer_only": True,
    "gpu_pilot_rerun": False,
    "preflight_report": preflight if (root / preflight).exists() else None,
    "pilot_summary": pilot_relative,
    "pilot_summary_sha256": actual_sha,
    "expected_pilot_summary_sha256": expected_sha,
    "pilot_summary_unchanged": actual_sha == expected_sha,
    "final_summary_sha256": digest(final_summary),
    "final_report_sha256": digest(final_report),
    "train_only_recommendation": None,
    "selected_configuration": None,
    "validation_targets_used": False,
    "formal_test_read": False,
    "formal_training": False,
    "idm": False,
    "candidate_execution": False,
    "phase4": False,
    "cps": False,
}
json_path = root / "reports/phase3_14b_r253_resume_blocked_summary.json"
md_path = root / "reports/phase3_14b_r253_resume_blocked_report.md"
json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
md_path.write_text(
    "# Phase3.14b-r2.5.3 Resume1 Blocked Report\n\n"
    f"- Exit code: `{exit_code}`\n"
    f"- Failed line: `{failed_line}`\n"
    "- GPU pilot rerun: `False`\n"
    f"- Pilot SHA unchanged: `{payload['pilot_summary_unchanged']}`\n"
    "- Train-only recommendation: `None`\n"
    "- Selected configuration: `None`\n",
    encoding="utf-8",
)
PY
}

trap 'code=$?; line=${BASH_LINENO[0]:-unknown}; write_blocked "$code" "$line"; exit "$code"' ERR

python scripts/phase3_14b_r253_resume1_preflight.py \
  --root "${ROOT}" \
  --output "${PREFLIGHT}"

python scripts/phase3_14b_r253_resume1_finalize.py \
  --root "${ROOT}" \
  --preflight-report "${PREFLIGHT}" \
  --pilot-report "${PILOT}"

trap - ERR

echo "Phase3.14b-r2.5.3 Resume1 finalizer completed without GPU retraining."
