#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${1:-/data/state_diff2}"
PYTHON_BIN="${PYTHON_BIN:-/miniforge3/envs/coord_bimanual/bin/python}"
PREFLIGHT="reports/phase3_14b_r255_stagec_resume1_correction_preflight.json"
BLOCKED="reports/phase3_14b_r255_stagec_resume1_blocked_summary.json"
CACHE_ROOT="data/phase3_14_cache_v3_r255_stagec_resume1"
ATTRIBUTION="reports/phase3_14b_r255_stagec_resume1_attribution.json"
SUMMARY="reports/phase3_14b_r255_stagec_resume1_summary.json"
MARKDOWN="reports/phase3_14b_r255_stagec_resume1_report.md"

export PYTHONNOUSERSITE=1
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

on_error() {
  local code="$?"
  "${PYTHON_BIN}" "${ROOT}/scripts/phase3_14b_r255_stagec_blocked.py" \
    --root "${ROOT}" \
    --output "${BLOCKED}" \
    --exit-code "${code}" \
    --failed-line "${BASH_LINENO[0]:-unknown}" \
    --failed-command "${BASH_COMMAND:-unknown}" || true
  exit "${code}"
}
trap on_error ERR

cd "${ROOT}"

"${PYTHON_BIN}" - <<'PY'
import sys
import numpy as np
if sys.version_info[:3] != (3, 9, 15):
    raise RuntimeError(f"Python contract changed: {sys.version}")
if np.__version__ != "1.23.3":
    raise RuntimeError(f"NumPy contract changed: {np.__version__}")
print({"python": sys.version.split()[0], "numpy": np.__version__})
PY

# Deliberately do not rerun the completed Stage-C pytest gate.
"${PYTHON_BIN}" scripts/phase3_14b_r255_stagec_resume1_preflight.py \
  --root "${ROOT}" \
  --output "${PREFLIGHT}"

"${PYTHON_BIN}" scripts/phase3_14b_r255_stagec_build_and_audit.py \
  --root "${ROOT}" \
  --python-bin "${PYTHON_BIN}" \
  --cache-root "${CACHE_ROOT}" \
  --attribution-report "${ATTRIBUTION}" \
  --summary-report "${SUMMARY}" \
  --markdown-report "${MARKDOWN}" \
  --allowed-untracked-path "${PREFLIGHT}"
