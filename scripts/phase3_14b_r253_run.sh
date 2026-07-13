#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/data/state_diff2}"
cd "${ROOT}"

PREFLIGHT_JSON="reports/phase3_14b_r253_preflight_summary.json"
PILOT_JSON="reports/phase3_14b_r253_pilot_summary.json"
SUMMARY_JSON="reports/phase3_14b_r253_summary.json"
REPORT_MD="reports/phase3_14b_r253_report.md"
BLOCKED_JSON="reports/phase3_14b_r253_blocked_summary.json"
BLOCKED_MD="reports/phase3_14b_r253_blocked_report.md"

for path in \
  "${PREFLIGHT_JSON}" "${PILOT_JSON}" "${SUMMARY_JSON}" \
  "${REPORT_MD}" "${BLOCKED_JSON}" "${BLOCKED_MD}"; do
  if [[ -e "${path}" ]]; then
    echo "Refusing to overwrite existing artifact: ${path}" >&2
    exit 2
  fi
done

CURRENT_LINE="initialization"
write_blocked() {
  local exit_code="$1"
  local traceback_file="${2:-}"
  python - "${ROOT}" "${exit_code}" "${CURRENT_LINE}" "${traceback_file}" <<'PY'
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
exit_code = int(sys.argv[2])
line = sys.argv[3]
traceback_path = Path(sys.argv[4]) if sys.argv[4] else None


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

traceback_text = ""
if traceback_path is not None and traceback_path.exists():
    traceback_text = traceback_path.read_text(encoding="utf-8", errors="replace")
    traceback_path.unlink()

pilot = root / "reports/phase3_14b_r253_pilot_summary.json"
payload = {
    "phase": "phase3_14b_r253",
    "verdict": "BLOCKED",
    "root_cause": "phase314b_r253_execution_failed",
    "exit_code": exit_code,
    "line": line,
    "exact_exception": traceback_text[-20000:],
    "pilot_summary_exists": pilot.exists(),
    "pilot_summary_sha256": sha256(pilot) if pilot.exists() else None,
    "train_only_recommendation": None,
    "selected_configuration": None,
    "validation_targets_used": False,
    "formal_test_read": False,
    "formal_training": False,
    "candidate_selected": False,
    "checkpoint_saved": False,
    "idm": False,
    "candidate_execution": False,
    "phase4": False,
    "cps": False,
}
json_path = root / "reports/phase3_14b_r253_blocked_summary.json"
md_path = root / "reports/phase3_14b_r253_blocked_report.md"
if json_path.exists() or md_path.exists():
    raise RuntimeError("refusing to overwrite blocked evidence")
json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
md_path.write_text(
    "# Phase3.14b-r2.5.3 Blocked Report\n\n"
    f"- Verdict: `BLOCKED`\n"
    f"- Root cause: `phase314b_r253_execution_failed`\n"
    f"- Exit code: `{exit_code}`\n"
    f"- Line: `{line}`\n"
    f"- Pilot summary exists: `{pilot.exists()}`\n"
    f"- Pilot summary SHA256: `{payload['pilot_summary_sha256']}`\n\n"
    "## Exact Exception\n\n```text\n"
    + traceback_text[-20000:]
    + "\n```\n\n"
    "Validation/formal-test/formal-training/IDM/candidate execution/Phase4/CPS "
    "were not authorized.\n",
    encoding="utf-8",
)
PY
}

run_step() {
  CURRENT_LINE="$1"
  shift
  local trace
  trace="$(mktemp)"
  set +e
  "$@" 2> >(tee "${trace}" >&2)
  local status=$?
  set -e
  if [[ ${status} -ne 0 ]]; then
    write_blocked "${status}" "${trace}"
    exit "${status}"
  fi
  rm -f "${trace}"
}

run_step "preflight" \
  python scripts/phase3_14b_r253_preflight.py \
    --root "${ROOT}" \
    --output "${PREFLIGHT_JSON}"

run_step "paired deterministic retraining and gate decomposition" \
  python scripts/phase3_14b_r253_run_pilot.py \
    --root "${ROOT}" \
    --preflight-report "${PREFLIGHT_JSON}" \
    --prior-steps 5000 \
    --residual-steps 8000 \
    --batch-size 64 \
    --prior-learning-rate 1e-3 \
    --residual-learning-rate 1e-3 \
    --calibration-batches 8

run_step "finalization" \
  python scripts/phase3_14b_r253_finalize.py \
    --root "${ROOT}" \
    --preflight-report "${PREFLIGHT_JSON}" \
    --pilot-report "${PILOT_JSON}"

for path in "${PREFLIGHT_JSON}" "${PILOT_JSON}" "${SUMMARY_JSON}" "${REPORT_MD}"; do
  test -f "${path}"
done

test ! -e "${BLOCKED_JSON}"
test ! -e "${BLOCKED_MD}"

echo "Phase3.14b-r2.5.3 completed."
