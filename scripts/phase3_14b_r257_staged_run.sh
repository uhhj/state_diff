#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${1:-/data/state_diff2}"
PYTHON_BIN="${PYTHON_BIN:-/miniforge3/envs/coord_bimanual/bin/python}"
BASE_COMMIT="a002246558e66029bdd6279e09c95e1303e2356c"
SUBMODULE_COMMIT="633a88752445cf5d6776ed374fdbbdb35f93050c"

export PYTHONNOUSERSITE=1
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

on_error() {
  local code="$?"
  "${PYTHON_BIN}" \
    "${ROOT}/scripts/phase3_14b_r257_staged_blocked.py" \
    --root "${ROOT}" \
    --exit-code "${code}" \
    --failed-line "${BASH_LINENO[0]:-unknown}" \
    --failed-command "${BASH_COMMAND:-unknown}" || true
  exit "${code}"
}
trap on_error ERR

cd "${ROOT}"

if [[ "$(git branch --show-current)" != "Experiment1" ]]; then
  echo "Stage D requires Experiment1" >&2
  exit 1
fi
if [[ "$(git rev-parse HEAD)" != "${BASE_COMMIT}" ]]; then
  echo "Local HEAD is not the frozen Stage-C evidence commit" >&2
  exit 1
fi
if [[ "$(git rev-parse origin/Experiment1)" != "${BASE_COMMIT}" ]]; then
  echo "origin/Experiment1 is not the frozen Stage-C evidence commit" >&2
  exit 1
fi
if [[ "$(git -C external/deformable-ravens rev-parse HEAD)" != "${SUBMODULE_COMMIT}" ]]; then
  echo "DeformableRavens commit changed" >&2
  exit 1
fi
if [[ -n "$(git -C external/deformable-ravens status --porcelain --untracked-files=all)" ]]; then
  echo "DeformableRavens worktree is dirty" >&2
  exit 1
fi

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
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable")
print({
    "python": sys.version.split()[0],
    "numpy": np.__version__,
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0),
})
PY

"${PYTHON_BIN}" \
  scripts/phase3_14b_r257_staged_test_gate.py \
  --root "${ROOT}"

IMPLEMENTATION_PATHS=(
  ccda_phase3/phase314b_r257_staged_timestep_gate.py
  scripts/phase3_14b_r257_staged_worker.py
  scripts/phase3_14b_r257_staged_run_calibration.py
  scripts/phase3_14b_r257_staged_test_gate.py
  scripts/phase3_14b_r257_staged_blocked.py
  scripts/phase3_14b_r257_staged_run.sh
  tests/test_phase3_14b_r257_staged_timestep_gate.py
)

git add "${IMPLEMENTATION_PATHS[@]}"

EXPECTED_STAGED="$(printf '%s\n' "${IMPLEMENTATION_PATHS[@]}" | sort)"
OBSERVED_STAGED="$(git diff --cached --name-only | sort)"
if [[ "${OBSERVED_STAGED}" != "${EXPECTED_STAGED}" ]]; then
  echo "Unexpected staged r2.5.7 Stage-D paths" >&2
  printf 'Expected:\n%s\nObserved:\n%s\n' \
    "${EXPECTED_STAGED}" "${OBSERVED_STAGED}" >&2
  exit 1
fi

UNEXPECTED="$(
  git status --porcelain --untracked-files=all |
  grep -v '^?? reports/phase3_14b_r257_staged_test_gate_summary.json$' |
  grep -v '^A  ccda_phase3/phase314b_r257_staged_timestep_gate.py$' |
  grep -v '^A  scripts/phase3_14b_r257_staged_worker.py$' |
  grep -v '^A  scripts/phase3_14b_r257_staged_run_calibration.py$' |
  grep -v '^A  scripts/phase3_14b_r257_staged_test_gate.py$' |
  grep -v '^A  scripts/phase3_14b_r257_staged_blocked.py$' |
  grep -v '^A  scripts/phase3_14b_r257_staged_run.sh$' |
  grep -v '^A  tests/test_phase3_14b_r257_staged_timestep_gate.py$' || true
)"
if [[ -n "${UNEXPECTED}" ]]; then
  echo "Unexpected worktree changes before Stage-D implementation commit" >&2
  printf '%s\n' "${UNEXPECTED}" >&2
  exit 1
fi

git commit -m \
  "Add Phase3.14b-r2.5.7 timestep-gated K16 calibration"

"${PYTHON_BIN}" \
  scripts/phase3_14b_r257_staged_run_calibration.py \
  --root "${ROOT}" \
  --python-bin "${PYTHON_BIN}"
