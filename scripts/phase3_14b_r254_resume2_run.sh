#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${1:-/data/state_diff2}"
EXPECTED_PYTHON="/miniforge3/envs/coord_bimanual/bin/python"
PYTHON_BIN_INPUT="${PHASE314B_PYTHON:-${EXPECTED_PYTHON}}"
PYTHON_BIN="$(readlink -m "${PYTHON_BIN_INPUT}")"
EXPECTED_PYTHON_RESOLVED="$(readlink -m "${EXPECTED_PYTHON}")"

cd "${ROOT}"
export PYTHONPATH="${ROOT}"
export PYTHONNOUSERSITE="1"
unset CUDA_VISIBLE_DEVICES || true

BLOCKED_JSON="reports/phase3_14b_r254_resume2_blocked_summary.json"
BLOCKED_MD="reports/phase3_14b_r254_resume2_blocked_report.md"

write_blocked() {
  local exit_code="$1"
  local failed_line="$2"
  local failed_command="$3"
  if [[ -e "${BLOCKED_JSON}" || -e "${BLOCKED_MD}" ]]; then
    printf 'Resume2 blocked evidence already exists; refusing overwrite.\n' >&2
    return 0
  fi
  if [[ -x "${PYTHON_BIN}" ]]; then
    "${PYTHON_BIN}" scripts/phase3_14b_r254_resume2_blocked.py \
      --root "${ROOT}" \
      --exit-code "${exit_code}" \
      --failed-line "${failed_line}" \
      --failed-command "${failed_command}" || true
  fi
}

on_error() {
  local exit_code="$?"
  local failed_line="${BASH_LINENO[0]:-0}"
  local failed_command="${BASH_COMMAND:-unknown}"
  trap - ERR
  write_blocked "${exit_code}" "${failed_line}" "${failed_command}"
  exit "${exit_code}"
}
trap on_error ERR

if [[ ! -x "${PYTHON_BIN}" ]]; then
  printf 'Required interpreter is not executable: %s\n' "${PYTHON_BIN}" >&2
  exit 1
fi
if [[ "${PYTHON_BIN}" != "${EXPECTED_PYTHON_RESOLVED}" ]]; then
  printf 'Unexpected interpreter: expected=%s observed=%s\n' \
    "${EXPECTED_PYTHON_RESOLVED}" "${PYTHON_BIN}" >&2
  exit 1
fi

"${PYTHON_BIN}" - <<'PY'
import os
import sys
import numpy
import torch

expected = "/miniforge3/envs/coord_bimanual/bin/python3.9"
if os.path.realpath(sys.executable) != os.path.realpath(expected):
    raise RuntimeError((sys.executable, expected))
if sys.version.split()[0] != "3.9.15":
    raise RuntimeError(sys.version)
if numpy.__version__ != "1.23.3":
    raise RuntimeError(numpy.__version__)
if torch.__version__ != "1.12.1.post200":
    raise RuntimeError(torch.__version__)
if str(torch.version.cuda) != "11.2":
    raise RuntimeError(torch.version.cuda)
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required")
if torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 4090":
    raise RuntimeError(torch.cuda.get_device_name(0))
if os.environ.get("PYTHONNOUSERSITE") != "1" or not sys.flags.no_user_site:
    raise RuntimeError("user-site isolation failed")
from ccda_phase3.phase314b_r254_resume2_functional_prior import (
    CURRENT_DEVICE_PRIOR_STATE_SHA256,
)
print("Resume2 interpreter, CUDA, repository import, and functional-prior adapter: PASS")
print(CURRENT_DEVICE_PRIOR_STATE_SHA256)
PY

"${PYTHON_BIN}" scripts/phase3_14b_r254_resume2_preflight.py \
  --root "${ROOT}" \
  --output reports/phase3_14b_r254_resume2_preflight_summary.json

"${PYTHON_BIN}" scripts/phase3_14b_r254_resume2_run_pilot.py \
  --root "${ROOT}" \
  --preflight-report reports/phase3_14b_r254_resume2_preflight_summary.json \
  --output reports/phase3_14b_r254_resume2_pilot_summary.json

"${PYTHON_BIN}" scripts/phase3_14b_r254_resume2_finalize.py \
  --root "${ROOT}" \
  --preflight-report reports/phase3_14b_r254_resume2_preflight_summary.json \
  --pilot-report reports/phase3_14b_r254_resume2_pilot_summary.json \
  --summary-output reports/phase3_14b_r254_summary.json \
  --markdown-output reports/phase3_14b_r254_report.md

trap - ERR
printf 'Phase3.14b-r2.5.4 Resume2 completed.\n'
