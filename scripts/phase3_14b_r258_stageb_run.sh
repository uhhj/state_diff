#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${1:-/data/state_diff2}"
PYTHON_BIN="${PYTHON_BIN:-/miniforge3/envs/coord_bimanual/bin/python}"
PREFLIGHT_DIR=""

export PYTHONNOUSERSITE=1
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

cleanup_tmp() {
  if [[ -n "${PREFLIGHT_DIR}" && -d "${PREFLIGHT_DIR}" ]]; then
    rm -rf "${PREFLIGHT_DIR}"
  fi
}

on_error() {
  local code="$?"
  cleanup_tmp
  "${PYTHON_BIN}" \
    "${ROOT}/scripts/phase3_14b_r258_stageb_blocked.py" \
    --root "${ROOT}" \
    --exit-code "${code}" \
    --failed-line "${BASH_LINENO[0]:-unknown}" \
    --failed-command "${BASH_COMMAND:-unknown}" || true
  exit "${code}"
}
trap on_error ERR
trap cleanup_tmp EXIT

cd "${ROOT}"

"${PYTHON_BIN}" - "${ROOT}" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
from ccda_phase3.phase314b_r258_stageb_direct_x0_reachability import (
    validate_base_evidence,
)

result = validate_base_evidence(root)
print({
    "base_evidence_commit":
        result["base_evidence_commit"],
    "base_worker_exact":
        result["worker"]["workers_exact"],
    "base_test_count":
        result["test_gate"]["passed_test_count"],
})
PY

PREFLIGHT_DIR="$(
  mktemp -d \
    /tmp/phase314b_r258_stageb_environment.XXXXXX
)"
PREFLIGHT="${PREFLIGHT_DIR}/environment.json"
if [[ -e "${PREFLIGHT}" ]]; then
  echo "Stage-B preflight output unexpectedly exists" >&2
  exit 1
fi

"${PYTHON_BIN}" \
  scripts/phase3_14b_r258_stageb_worker.py \
  --root "${ROOT}" \
  --mode environment-probe \
  --output "${PREFLIGHT}"

"${PYTHON_BIN}" - "${PREFLIGHT}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
value = json.loads(
    path.read_text(encoding="utf-8")
)
if value.get("compatibility_pass") is not True:
    raise RuntimeError(
        "portable compatibility gate failed"
    )
compatibility = value.get(
    "compatibility",
    {},
)
dry_run = value.get(
    "required_operation_dry_run",
    {},
)
observation = value.get(
    "hardware_observation",
    {},
)
if compatibility.get(
    "required_operation_pass"
) is not True:
    raise RuntimeError(
        "required-operation compatibility failed"
    )
if dry_run.get("pass") is not True:
    raise RuntimeError(
        "required-operation dry run failed"
    )
if not observation.get(
    "torch_device_name"
):
    raise RuntimeError(
        "GPU observation is missing"
    )
print({
    "gpu":
        observation.get(
            "torch_device_name"
        ),
    "compute_capability":
        observation.get(
            "compute_capability"
        ),
    "compatibility_sha256":
        value.get(
            "compatibility_sha256"
        ),
    "observation_sha256":
        value.get(
            "observation_sha256"
        ),
    "dry_run_peak_reserved_bytes":
        dry_run.get(
            "peak_memory_reserved_bytes"
        ),
})
PY
rm -rf "${PREFLIGHT_DIR}"
PREFLIGHT_DIR=""

"${PYTHON_BIN}" \
  scripts/phase3_14b_r258_stageb_test_gate.py \
  --root "${ROOT}"

IMPLEMENTATION_PATHS=(
  ccda_phase3/phase314b_r258_stageb_direct_x0_reachability.py
  scripts/phase3_14b_r258_stageb_worker.py
  scripts/phase3_14b_r258_stageb_run_audit.py
  scripts/phase3_14b_r258_stageb_test_gate.py
  scripts/phase3_14b_r258_stageb_blocked.py
  scripts/phase3_14b_r258_stageb_run.sh
  tests/test_phase3_14b_r258_stageb_direct_x0_reachability.py
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
  echo "Unexpected staged Stage-B implementation paths" >&2
  printf 'Expected:\n%s\nObserved:\n%s\n' \
    "${EXPECTED_STAGED}" \
    "${OBSERVED_STAGED}" >&2
  exit 1
fi

UNEXPECTED="$(
  git status --porcelain --untracked-files=all |
  grep -v '^?? reports/phase3_14b_r258_stageb_test_gate_summary.json$' |
  grep -v '^A  ccda_phase3/phase314b_r258_stageb_direct_x0_reachability.py$' |
  grep -v '^A  scripts/phase3_14b_r258_stageb_worker.py$' |
  grep -v '^A  scripts/phase3_14b_r258_stageb_run_audit.py$' |
  grep -v '^A  scripts/phase3_14b_r258_stageb_test_gate.py$' |
  grep -v '^A  scripts/phase3_14b_r258_stageb_blocked.py$' |
  grep -v '^A  scripts/phase3_14b_r258_stageb_run.sh$' |
  grep -v '^A  tests/test_phase3_14b_r258_stageb_direct_x0_reachability.py$' || true
)"
if [[ -n "${UNEXPECTED}" ]]; then
  echo "Unexpected worktree changes before Stage-B implementation commit" >&2
  printf '%s\n' "${UNEXPECTED}" >&2
  exit 1
fi

git commit -m \
  "Add Phase3.14b-r2.5.8 direct-x0 reachability audit"

"${PYTHON_BIN}" \
  scripts/phase3_14b_r258_stageb_run_audit.py \
  --root "${ROOT}" \
  --python-bin "${PYTHON_BIN}"
