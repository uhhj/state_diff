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
    "${ROOT}/scripts/phase3_14b_r258_stagea_resume1_blocked.py" \
    --root "${ROOT}" \
    --exit-code "${code}" \
    --failed-line "${BASH_LINENO[0]:-unknown}" \
    --failed-command "${BASH_COMMAND:-unknown}" || true
  exit "${code}"
}
trap on_error ERR
trap cleanup_tmp EXIT

cd "${ROOT}"

ORIGINAL_PATHS=(
  ccda_phase3/phase314b_r258_stagea_conflict_projected_k16.py
  scripts/phase3_14b_r258_stagea_worker.py
  scripts/phase3_14b_r258_stagea_run_calibration.py
  scripts/phase3_14b_r258_stagea_test_gate.py
  scripts/phase3_14b_r258_stagea_blocked.py
  scripts/phase3_14b_r258_stagea_run.sh
  tests/test_phase3_14b_r258_stagea_conflict_projected_k16.py
)
FAILED_BLOCKED=(
  reports/phase3_14b_r258_stagea_blocked_summary.json
)
RESUME1_PATHS=(
  ccda_phase3/phase314b_r258_stagea_resume1_output_lifecycle.py
  scripts/phase3_14b_r258_stagea_resume1_test_gate.py
  scripts/phase3_14b_r258_stagea_resume1_run_calibration.py
  scripts/phase3_14b_r258_stagea_resume1_blocked.py
  scripts/phase3_14b_r258_stagea_resume1_run.sh
  tests/test_phase3_14b_r258_stagea_resume1_output_lifecycle.py
)

"${PYTHON_BIN}" - "${ROOT}" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
from ccda_phase3.phase314b_r258_stagea_resume1_output_lifecycle import (
    validate_failed_files,
)

result = validate_failed_files(
    root,
    require_tracked=False,
)
print({
    "failed_blocked_sha256": result["failed_blocked_sha256"],
    "original_file_count": len(
        result["original_portable_file_sha256"]
    ),
    "submodule_commit": result["submodule_commit"],
})
PY

for path in "${ORIGINAL_PATHS[@]}" "${FAILED_BLOCKED[@]}"; do
  if git ls-files --error-unmatch "${path}" >/dev/null 2>&1; then
    echo "Failed-attempt path is unexpectedly already tracked: ${path}" >&2
    exit 1
  fi
done

git add "${ORIGINAL_PATHS[@]}"
EXPECTED_ORIGINAL="$(
  printf '%s\n' "${ORIGINAL_PATHS[@]}" | sort
)"
OBSERVED_ORIGINAL="$(
  git diff --cached --name-only | sort
)"
if [[ "${OBSERVED_ORIGINAL}" != "${EXPECTED_ORIGINAL}" ]]; then
  echo "Unexpected staged original portable files" >&2
  printf 'Expected:\n%s\nObserved:\n%s\n' \
    "${EXPECTED_ORIGINAL}" \
    "${OBSERVED_ORIGINAL}" >&2
  exit 1
fi
git commit -m \
  "Add Phase3.14b-r2.5.8 portable conflict-projected K16 calibration"

git add "${FAILED_BLOCKED[@]}"
EXPECTED_BLOCKED="$(
  printf '%s\n' "${FAILED_BLOCKED[@]}" | sort
)"
OBSERVED_BLOCKED="$(
  git diff --cached --name-only | sort
)"
if [[ "${OBSERVED_BLOCKED}" != "${EXPECTED_BLOCKED}" ]]; then
  echo "Unexpected staged first-attempt blocked evidence" >&2
  printf 'Expected:\n%s\nObserved:\n%s\n' \
    "${EXPECTED_BLOCKED}" \
    "${OBSERVED_BLOCKED}" >&2
  exit 1
fi
git commit -m \
  "Record blocked Phase3.14b-r2.5.8 Stage-A portable preflight execution"

"${PYTHON_BIN}" - "${ROOT}" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
from ccda_phase3.phase314b_r258_stagea_resume1_output_lifecycle import (
    validate_failed_files,
)

validate_failed_files(
    root,
    require_tracked=True,
)
print("failed-attempt provenance is tracked and exact")
PY

PREFLIGHT_DIR="$(
  mktemp -d \
    /tmp/phase314b_r258_stagea_resume1_environment.XXXXXX
)"
PREFLIGHT="${PREFLIGHT_DIR}/environment.json"
if [[ -e "${PREFLIGHT}" ]]; then
  echo "Resume1 preflight output unexpectedly exists" >&2
  exit 1
fi

"${PYTHON_BIN}" \
  scripts/phase3_14b_r258_stagea_worker.py \
  --root "${ROOT}" \
  --mode environment-probe \
  --output "${PREFLIGHT}"

"${PYTHON_BIN}" - "${PREFLIGHT}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
value = json.loads(path.read_text(encoding="utf-8"))
if value.get("compatibility_pass") is not True:
    raise RuntimeError(
        "portable environment compatibility failed"
    )
compatibility = value.get("compatibility", {})
dry_run = value.get("required_operation_dry_run", {})
observation = value.get("hardware_observation", {})
if compatibility.get("required_operation_pass") is not True:
    raise RuntimeError(
        "required-operation compatibility failed"
    )
if dry_run.get("pass") is not True:
    raise RuntimeError(
        "fixed-batch CUDA dry run failed"
    )
if not observation.get("torch_device_name"):
    raise RuntimeError(
        "GPU observation is missing"
    )
print({
    "gpu": observation.get("torch_device_name"),
    "compute_capability":
        observation.get("compute_capability"),
    "compatibility_sha256":
        value.get("compatibility_sha256"),
    "observation_sha256":
        value.get("observation_sha256"),
    "dry_run_peak_reserved_bytes":
        dry_run.get("peak_memory_reserved_bytes"),
})
PY
rm -rf "${PREFLIGHT_DIR}"
PREFLIGHT_DIR=""

"${PYTHON_BIN}" \
  scripts/phase3_14b_r258_stagea_resume1_test_gate.py \
  --root "${ROOT}"

git add "${RESUME1_PATHS[@]}"
EXPECTED_RESUME1="$(
  printf '%s\n' "${RESUME1_PATHS[@]}" | sort
)"
OBSERVED_RESUME1="$(
  git diff --cached --name-only | sort
)"
if [[ "${OBSERVED_RESUME1}" != "${EXPECTED_RESUME1}" ]]; then
  echo "Unexpected staged Resume1 implementation paths" >&2
  printf 'Expected:\n%s\nObserved:\n%s\n' \
    "${EXPECTED_RESUME1}" \
    "${OBSERVED_RESUME1}" >&2
  exit 1
fi

UNEXPECTED="$(
  git status --porcelain --untracked-files=all |
  grep -v '^?? reports/phase3_14b_r258_stagea_resume1_test_gate_summary.json$' |
  grep -v '^A  ccda_phase3/phase314b_r258_stagea_resume1_output_lifecycle.py$' |
  grep -v '^A  scripts/phase3_14b_r258_stagea_resume1_test_gate.py$' |
  grep -v '^A  scripts/phase3_14b_r258_stagea_resume1_run_calibration.py$' |
  grep -v '^A  scripts/phase3_14b_r258_stagea_resume1_blocked.py$' |
  grep -v '^A  scripts/phase3_14b_r258_stagea_resume1_run.sh$' |
  grep -v '^A  tests/test_phase3_14b_r258_stagea_resume1_output_lifecycle.py$' || true
)"
if [[ -n "${UNEXPECTED}" ]]; then
  echo "Unexpected worktree changes before Resume1 commit" >&2
  printf '%s\n' "${UNEXPECTED}" >&2
  exit 1
fi

git commit -m \
  "Add Phase3.14b-r2.5.8 Stage-A portable Resume1 output-lifecycle correction"

"${PYTHON_BIN}" \
  scripts/phase3_14b_r258_stagea_resume1_run_calibration.py \
  --root "${ROOT}" \
  --python-bin "${PYTHON_BIN}"
