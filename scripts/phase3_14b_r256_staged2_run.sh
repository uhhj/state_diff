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
  "${PYTHON_BIN}" \
    "${ROOT}/scripts/phase3_14b_r256_staged2_blocked.py" \
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
print({
    "python": sys.version.split()[0],
    "numpy": np.__version__,
    "audit_backend": "cpu_source_pickle_read_only",
})
PY

"${PYTHON_BIN}" scripts/phase3_14b_r256_staged2_test_gate.py \
  --root "${ROOT}"

IMPLEMENTATION_PATHS=(
  ccda_phase3/phase314b_r256_staged2_collapse_provenance.py
  scripts/phase3_14b_r256_staged2_worker.py
  scripts/phase3_14b_r256_staged2_run_audit.py
  scripts/phase3_14b_r256_staged2_test_gate.py
  scripts/phase3_14b_r256_staged2_blocked.py
  scripts/phase3_14b_r256_staged2_run.sh
  tests/test_phase3_14b_r256_staged2_collapse_provenance.py
)

git add "${IMPLEMENTATION_PATHS[@]}"

EXPECTED_STAGED="$(printf '%s\n' "${IMPLEMENTATION_PATHS[@]}" | sort)"
OBSERVED_STAGED="$(git diff --cached --name-only | sort)"
if [[ "${OBSERVED_STAGED}" != "${EXPECTED_STAGED}" ]]; then
  echo "Unexpected staged implementation paths" >&2
  printf 'Expected:\n%s\nObserved:\n%s\n' \
    "${EXPECTED_STAGED}" "${OBSERVED_STAGED}" >&2
  exit 1
fi

UNEXPECTED="$(
  git status --porcelain --untracked-files=all |
  grep -v '^?? reports/phase3_14b_r256_staged2_test_gate_summary.json$' |
  grep -v '^A  ccda_phase3/phase314b_r256_staged2_collapse_provenance.py$' |
  grep -v '^A  scripts/phase3_14b_r256_staged2_worker.py$' |
  grep -v '^A  scripts/phase3_14b_r256_staged2_run_audit.py$' |
  grep -v '^A  scripts/phase3_14b_r256_staged2_test_gate.py$' |
  grep -v '^A  scripts/phase3_14b_r256_staged2_blocked.py$' |
  grep -v '^A  scripts/phase3_14b_r256_staged2_run.sh$' |
  grep -v '^A  tests/test_phase3_14b_r256_staged2_collapse_provenance.py$' || true
)"
if [[ -n "${UNEXPECTED}" ]]; then
  echo "Unexpected worktree changes before implementation commit" >&2
  printf '%s\n' "${UNEXPECTED}" >&2
  exit 1
fi

git commit -m "Add Phase3.14b-r2.5.6 cable-XY collapse provenance audit"

"${PYTHON_BIN}" scripts/phase3_14b_r256_staged2_run_audit.py \
  --root "${ROOT}" \
  --python-bin "${PYTHON_BIN}"
