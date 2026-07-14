#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${1:-/data/state_diff2}"
PYTHON_BIN="${PYTHON_BIN:-/miniforge3/envs/coord_bimanual/bin/python}"
export PYTHONNOUSERSITE=1

on_error() {
  local code="$?"
  "${PYTHON_BIN}" "${ROOT}/scripts/phase3_14b_r254_resume5_blocked.py" \
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
import torch
if sys.version_info[:3] != (3, 9, 15):
    raise RuntimeError(f"Python contract changed: {sys.version}")
if np.__version__ != "1.23.3":
    raise RuntimeError(f"NumPy contract changed: {np.__version__}")
if torch.__version__ != "1.12.1.post200":
    raise RuntimeError(f"PyTorch contract changed: {torch.__version__}")
if torch.version.cuda != "11.2":
    raise RuntimeError(f"CUDA runtime contract changed: {torch.version.cuda}")
if not torch.cuda.is_available():
    raise RuntimeError("fixed experiment CUDA device is unavailable")
if torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 4090":
    raise RuntimeError(f"fixed GPU changed: {torch.cuda.get_device_name(0)}")
print({
    "python": sys.version.split()[0],
    "numpy": np.__version__,
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0),
})
PY

"${PYTHON_BIN}" scripts/phase3_14b_r254_resume5_test_gate.py --root "${ROOT}"
"${PYTHON_BIN}" scripts/phase3_14b_r254_resume5_preflight.py --root "${ROOT}"
"${PYTHON_BIN}" scripts/phase3_14b_r254_resume5_run_audit.py --root "${ROOT}"
"${PYTHON_BIN}" scripts/phase3_14b_r254_resume5_finalize.py --root "${ROOT}"
