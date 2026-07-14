#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/data/state_diff2}"
EXPECTED_PYTHON="/miniforge3/envs/coord_bimanual/bin/python"
PYTHON_INPUT="${PHASE314B_PYTHON:-${EXPECTED_PYTHON}}"
PYTHON_BIN="$(readlink -m "${PYTHON_INPUT}")"
EXPECTED_RESOLVED="$(readlink -m "${EXPECTED_PYTHON}")"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Phase3.14b-r2.5.4 Python is not executable: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ "${PYTHON_BIN}" != "${EXPECTED_RESOLVED}" ]]; then
  echo "Unexpected Phase3.14b-r2.5.4 Python: ${PYTHON_BIN}" >&2
  exit 1
fi

cd "${ROOT}"
export PYTHONPATH="${ROOT}"
export PYTHONNOUSERSITE="1"

PREFLIGHT="reports/phase3_14b_r254_preflight_summary.json"
PILOT="reports/phase3_14b_r254_pilot_summary.json"
SUMMARY="reports/phase3_14b_r254_summary.json"
REPORT="reports/phase3_14b_r254_report.md"
BLOCKED_JSON="reports/phase3_14b_r254_blocked_summary.json"
BLOCKED_MD="reports/phase3_14b_r254_blocked_report.md"
LOG_FILE="$(mktemp /tmp/phase314b_r254.XXXXXX.log)"
CURRENT_STAGE="interpreter_probe"

cleanup() {
  rm -f "${LOG_FILE}"
}
trap cleanup EXIT

write_blocked() {
  local exit_code="$1"
  local failed_line="$2"
  if [[ -e "${BLOCKED_JSON}" || -e "${BLOCKED_MD}" ]]; then
    echo "Refusing to overwrite r2.5.4 blocked evidence" >&2
    return 1
  fi
  PHASE314B_STAGE="${CURRENT_STAGE}" \
  PHASE314B_EXIT_CODE="${exit_code}" \
  PHASE314B_FAILED_LINE="${failed_line}" \
  PHASE314B_LOG_FILE="${LOG_FILE}" \
  PHASE314B_PILOT="${PILOT}" \
  PHASE314B_BLOCKED_JSON="${BLOCKED_JSON}" \
  PHASE314B_BLOCKED_MD="${BLOCKED_MD}" \
  "${PYTHON_BIN}" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

stage = os.environ["PHASE314B_STAGE"]
exit_code = int(os.environ["PHASE314B_EXIT_CODE"])
failed_line = os.environ["PHASE314B_FAILED_LINE"]
log_path = Path(os.environ["PHASE314B_LOG_FILE"])
pilot = Path(os.environ["PHASE314B_PILOT"])
json_path = Path(os.environ["PHASE314B_BLOCKED_JSON"])
md_path = Path(os.environ["PHASE314B_BLOCKED_MD"])
log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
tail = "\n".join(log.splitlines()[-80:])
pilot_sha = None
if pilot.exists():
    pilot_sha = hashlib.sha256(pilot.read_bytes()).hexdigest()
payload = {
    "phase": "phase3_14b_r254",
    "verdict": "BLOCKED",
    "root_cause": "phase314b_r254_execution_failed",
    "failed_stage": stage,
    "exit_code": exit_code,
    "failed_line": failed_line,
    "traceback_tail": tail,
    "pilot_summary_exists": pilot.exists(),
    "pilot_summary_sha256": pilot_sha,
    "train_only_recommendation": None,
    "selected_configuration": None,
    "validation_targets_used": False,
    "formal_test_read": False,
    "formal_training": False,
    "checkpoint_saved": False,
    "idm": False,
    "candidate_execution": False,
    "phase4": False,
    "cps": False,
}
json_path.parent.mkdir(parents=True, exist_ok=True)
with json_path.open("x", encoding="utf-8") as stream:
    json.dump(payload, stream, indent=2, sort_keys=True)
    stream.write("\n")
with md_path.open("x", encoding="utf-8") as stream:
    stream.write("# Phase3.14b-r2.5.4 BLOCKED\n\n")
    stream.write(f"- Failed stage: `{stage}`\n")
    stream.write(f"- Exit code: `{exit_code}`\n")
    stream.write(f"- Failed line: `{failed_line}`\n")
    stream.write(f"- Pilot summary SHA256: `{pilot_sha}`\n\n")
    stream.write("```text\n")
    stream.write(tail)
    stream.write("\n```\n")
PY
}

on_error() {
  local exit_code=$?
  local failed_line="${BASH_LINENO[0]:-unknown}"
  trap - ERR
  write_blocked "${exit_code}" "${failed_line}" || true
  exit "${exit_code}"
}
trap on_error ERR

if [[ -e "${PREFLIGHT}" || -e "${PILOT}" || -e "${SUMMARY}" || -e "${REPORT}" || -e "${BLOCKED_JSON}" || -e "${BLOCKED_MD}" ]]; then
  echo "Refusing to overwrite existing Phase3.14b-r2.5.4 artifacts" >&2
  exit 1
fi

"${PYTHON_BIN}" - <<'PY' 2>&1 | tee -a "${LOG_FILE}"
import json
import os
import sys
import numpy
import torch
expected = os.path.realpath("/miniforge3/envs/coord_bimanual/bin/python")
observed = os.path.realpath(sys.executable)
if observed != expected:
    raise RuntimeError(f"interpreter mismatch: expected={expected}, observed={observed}")
if sys.version_info[:3] != (3, 9, 15):
    raise RuntimeError(f"Python mismatch: {sys.version}")
if numpy.__version__ != "1.23.3":
    raise RuntimeError(f"NumPy mismatch: {numpy.__version__}")
if torch.__version__ != "1.12.1.post200":
    raise RuntimeError(f"PyTorch mismatch: {torch.__version__}")
if str(torch.version.cuda) != "11.2":
    raise RuntimeError(f"CUDA runtime mismatch: {torch.version.cuda}")
if os.environ.get("PYTHONNOUSERSITE") != "1" or not sys.flags.no_user_site:
    raise RuntimeError("user-site isolation contract failed")
if not torch.cuda.is_available():
    raise RuntimeError("r2.5.4 deterministic retraining requires CUDA")
print(json.dumps({
    "interpreter": observed,
    "python": sys.version.split()[0],
    "numpy": numpy.__version__,
    "torch": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "gpu": torch.cuda.get_device_name(0),
}, sort_keys=True))
PY

CURRENT_STAGE="repository_import_probe"
"${PYTHON_BIN}" -c 'import ccda_phase3; from ccda_phase3.phase314b_r254_robot_proxy_attribution import PHASE; print(PHASE)' 2>&1 | tee -a "${LOG_FILE}"

CURRENT_STAGE="preflight"
"${PYTHON_BIN}" scripts/phase3_14b_r254_preflight.py \
  --root "${ROOT}" --output "${PREFLIGHT}" 2>&1 | tee -a "${LOG_FILE}"

CURRENT_STAGE="pilot"
"${PYTHON_BIN}" scripts/phase3_14b_r254_run_pilot.py \
  --root "${ROOT}" --preflight-report "${PREFLIGHT}" 2>&1 | tee -a "${LOG_FILE}"

CURRENT_STAGE="finalization"
"${PYTHON_BIN}" scripts/phase3_14b_r254_finalize.py \
  --root "${ROOT}" \
  --preflight-report "${PREFLIGHT}" \
  --pilot-report "${PILOT}" \
  --summary-output "${SUMMARY}" \
  --markdown-output "${REPORT}" 2>&1 | tee -a "${LOG_FILE}"

trap - ERR
echo "Phase3.14b-r2.5.4 completed"
