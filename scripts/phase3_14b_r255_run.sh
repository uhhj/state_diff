#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${1:-/data/state_diff2}"
PYTHON_BIN="${PYTHON_BIN:-/miniforge3/envs/coord_bimanual/bin/python}"
export PYTHONNOUSERSITE=1

on_error() {
  local code="$?"
  "${PYTHON_BIN}" "${ROOT}/scripts/phase3_14b_r255_blocked.py" \
    --root "${ROOT}" \
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
import pybullet
if sys.version_info[:3] != (3, 9, 15):
    raise RuntimeError(f"Python contract changed: {sys.version}")
if np.__version__ != "1.23.3":
    raise RuntimeError(f"NumPy contract changed: {np.__version__}")
print({
    "python": sys.version.split()[0],
    "numpy": np.__version__,
    "pybullet_api": getattr(pybullet, "__version__", "module-import-ok"),
})
PY

"${PYTHON_BIN}" scripts/phase3_14b_r255_test_gate.py --root "${ROOT}"
"${PYTHON_BIN}" scripts/phase3_14b_r255_audit.py --root "${ROOT}"
