#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${1:-/data/state_diff2}"
PYTHON_BIN="${PYTHON_BIN:-/miniforge3/envs/coord_bimanual/bin/python}"
export PYTHONNOUSERSITE=1
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

on_error() {
  local code="$?"
  "${PYTHON_BIN}" "${ROOT}/scripts/phase3_14b_r256_stagea_blocked.py" \
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
if sys.version_info[:3] != (3, 9, 15):
    raise RuntimeError(f"Python contract changed: {sys.version}")
if np.__version__ != "1.23.3":
    raise RuntimeError(f"NumPy contract changed: {np.__version__}")
print({"python": sys.version.split()[0], "numpy": np.__version__})
PY

"${PYTHON_BIN}" scripts/phase3_14b_r256_stagea_test_gate.py --root "${ROOT}"

# Create the implementation commit after the write-once test gate is sealed.
"${PYTHON_BIN}" - <<'PY'
import subprocess
allowed = {
    "ccda_phase3/phase314b_r256_stagea_contract.py",
    "scripts/phase3_14b_r256_stagea_worker.py",
    "scripts/phase3_14b_r256_stagea_build_and_audit.py",
    "scripts/phase3_14b_r256_stagea_test_gate.py",
    "scripts/phase3_14b_r256_stagea_blocked.py",
    "scripts/phase3_14b_r256_stagea_run.sh",
    "tests/test_phase3_14b_r256_stagea_contract.py",
    "reports/phase3_14b_r256_stagea_test_gate_summary.json",
}
unexpected = []
for line in subprocess.check_output(
    ["git", "status", "--porcelain", "--untracked-files=all"],
    text=True,
).splitlines():
    path = line[3:].strip()
    if " -> " in path:
        path = path.split(" -> ", 1)[1]
    if path not in allowed:
        unexpected.append(line)
if unexpected:
    raise RuntimeError(f"unexpected worktree paths: {unexpected}")
PY

git add \
  ccda_phase3/phase314b_r256_stagea_contract.py \
  scripts/phase3_14b_r256_stagea_worker.py \
  scripts/phase3_14b_r256_stagea_build_and_audit.py \
  scripts/phase3_14b_r256_stagea_test_gate.py \
  scripts/phase3_14b_r256_stagea_blocked.py \
  scripts/phase3_14b_r256_stagea_run.sh \
  tests/test_phase3_14b_r256_stagea_contract.py

git commit -m "Add Phase3.14b-r2.5.6 cable-only and IDM contract audit"

"${PYTHON_BIN}" scripts/phase3_14b_r256_stagea_build_and_audit.py \
  --root "${ROOT}" \
  --python-bin "${PYTHON_BIN}"
