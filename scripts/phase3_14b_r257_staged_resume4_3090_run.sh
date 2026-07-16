#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${1:-/data/state_diff2}"
PYTHON_BIN="${PYTHON_BIN:-/miniforge3/envs/coord_bimanual/bin/python}"

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
    "${ROOT}/scripts/phase3_14b_r257_staged_resume4_3090_blocked.py" \
    --root "${ROOT}" \
    --exit-code "${code}" \
    --failed-line "${BASH_LINENO[0]:-unknown}" \
    --failed-command "${BASH_COMMAND:-unknown}" || true
  exit "${code}"
}
trap on_error ERR

cd "${ROOT}"

PREFLIGHT="$(mktemp /tmp/phase314b_r257_resume4_3090_env.XXXXXX.json)"
rm -f "${PREFLIGHT}"
"${PYTHON_BIN}" \
  scripts/phase3_14b_r257_staged_resume4_3090_worker.py \
  --root "${ROOT}" \
  --mode environment-probe \
  --output "${PREFLIGHT}"
rm -f "${PREFLIGHT}"

"${PYTHON_BIN}" \
  scripts/phase3_14b_r257_staged_resume4_3090_test_gate.py \
  --root "${ROOT}"

IMPLEMENTATION_PATHS=(
  ccda_phase3/phase314b_r257_staged_resume4_3090.py
  scripts/phase3_14b_r257_staged_resume4_3090_worker.py
  scripts/phase3_14b_r257_staged_resume4_3090_run.py
  scripts/phase3_14b_r257_staged_resume4_3090_test_gate.py
  scripts/phase3_14b_r257_staged_resume4_3090_blocked.py
  scripts/phase3_14b_r257_staged_resume4_3090_run.sh
  tests/test_phase3_14b_r257_staged_resume4_3090.py
)

git add "${IMPLEMENTATION_PATHS[@]}"

EXPECTED_STAGED="$(
  printf '%s\n' "${IMPLEMENTATION_PATHS[@]}" |
  sort
)"
OBSERVED_STAGED="$(
  git diff --cached --name-only |
  sort
)"
if [[ "${OBSERVED_STAGED}" != "${EXPECTED_STAGED}" ]]; then
  echo "Unexpected staged Resume4 implementation paths" >&2
  printf 'Expected:\n%s\nObserved:\n%s\n' \
    "${EXPECTED_STAGED}" \
    "${OBSERVED_STAGED}" >&2
  exit 1
fi

UNEXPECTED="$(
  git status --porcelain --untracked-files=all |
  grep -v '^?? reports/phase3_14b_r257_staged_resume4_3090_test_gate_summary.json$' |
  grep -v '^A  ccda_phase3/phase314b_r257_staged_resume4_3090.py$' |
  grep -v '^A  scripts/phase3_14b_r257_staged_resume4_3090_worker.py$' |
  grep -v '^A  scripts/phase3_14b_r257_staged_resume4_3090_run.py$' |
  grep -v '^A  scripts/phase3_14b_r257_staged_resume4_3090_test_gate.py$' |
  grep -v '^A  scripts/phase3_14b_r257_staged_resume4_3090_blocked.py$' |
  grep -v '^A  scripts/phase3_14b_r257_staged_resume4_3090_run.sh$' |
  grep -v '^A  tests/test_phase3_14b_r257_staged_resume4_3090.py$' || true
)"
if [[ -n "${UNEXPECTED}" ]]; then
  echo "Unexpected worktree changes before Resume4 implementation commit" >&2
  printf '%s\n' "${UNEXPECTED}" >&2
  exit 1
fi

git commit -m \
  "Add Phase3.14b-r2.5.7 Stage-D Resume4 RTX-3090 replay"

"${PYTHON_BIN}" \
  scripts/phase3_14b_r257_staged_resume4_3090_run.py \
  --root "${ROOT}" \
  --python-bin "${PYTHON_BIN}"
