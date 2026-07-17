#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${1:-/data/state_diff2}"
PYTHON_BIN="${PYTHON_BIN:-/miniforge3/envs/coord_bimanual/bin/python}"
PREFLIGHT_DIR=""
PUSH_COMPLETED="false"

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
  if [[ "${PUSH_COMPLETED}" != "true" ]]; then
    "${PYTHON_BIN}" \
      "${ROOT}/scripts/phase3_14b_r258_stagee_blocked.py" \
      --root "${ROOT}" \
      --exit-code "${code}" \
      --failed-line "${BASH_LINENO[0]:-unknown}" \
      --failed-command "${BASH_COMMAND:-unknown}" || true
  fi
  exit "${code}"
}
trap on_error ERR
trap cleanup_tmp EXIT

sorted_lines() {
  printf '%s\n' "$@" | LC_ALL=C sort
}

assert_cached_paths() {
  local -a expected=("$@")
  local expected_text observed_text
  expected_text="$(sorted_lines "${expected[@]}")"
  observed_text="$(git diff --cached --name-only | LC_ALL=C sort)"
  if [[ "${observed_text}" != "${expected_text}" ]]; then
    printf 'Unexpected staged paths.\nExpected:\n%s\nObserved:\n%s\n' \
      "${expected_text}" "${observed_text}" >&2
    return 1
  fi
}

commit_with_outcome() {
  local message="$1"
  shift
  local -a expected_paths=("$@")
  local pre_head post_head parent subject actual_paths expected_text
  local command_status log_path

  assert_cached_paths "${expected_paths[@]}"
  pre_head="$(git rev-parse HEAD)"
  log_path="$(mktemp /tmp/phase314b_r258_stagee_commit.XXXXXX.log)"

  set +e
  git commit -m "${message}" >"${log_path}" 2>&1
  command_status="$?"
  set -e

  post_head="$(git rev-parse HEAD)"
  if [[ "${post_head}" == "${pre_head}" ]]; then
    sed -n '1,200p' "${log_path}" >&2 || true
    rm -f "${log_path}"
    printf 'Git transaction did not create a commit; status=%s\n' \
      "${command_status}" >&2
    return 1
  fi
  parent="$(git rev-parse "${post_head}^")"
  subject="$(git show -s --format=%s "${post_head}")"
  actual_paths="$(
    git diff-tree --no-commit-id --name-only -r "${post_head}" |
    LC_ALL=C sort
  )"
  expected_text="$(sorted_lines "${expected_paths[@]}")"
  if [[ "${parent}" != "${pre_head}" ]]; then
    printf 'Commit parent mismatch\n' >&2
    return 1
  fi
  if [[ "${subject}" != "${message}" ]]; then
    printf 'Commit subject mismatch\n' >&2
    return 1
  fi
  if [[ "${actual_paths}" != "${expected_text}" ]]; then
    printf 'Commit path mismatch.\nExpected:\n%s\nObserved:\n%s\n' \
      "${expected_text}" "${actual_paths}" >&2
    return 1
  fi
  if ! git diff --cached --quiet; then
    printf 'Index is not clean after commit\n' >&2
    return 1
  fi
  rm -f "${log_path}"
  printf 'Verified commit transaction: status=%s sha=%s\n' \
    "${command_status}" "${post_head}"
}

push_with_outcome() {
  local local_head command_status log_path remote_head local_tracking attempt
  local_head="$(git rev-parse HEAD)"
  log_path="$(mktemp /tmp/phase314b_r258_stagee_push.XXXXXX.log)"

  set +e
  git push origin Experiment1 >"${log_path}" 2>&1
  command_status="$?"
  set -e

  remote_head=""
  for attempt in 1 2 3; do
    set +e
    remote_head="$(
      git ls-remote origin refs/heads/Experiment1 2>/dev/null |
      awk '{print $1}'
    )"
    set -e
    if [[ "${remote_head}" == "${local_head}" ]]; then
      break
    fi
    sleep 2
  done
  if [[ "${remote_head}" != "${local_head}" ]]; then
    local_tracking="$(git rev-parse origin/Experiment1 2>/dev/null || true)"
    if [[ "${command_status}" == "0" && "${local_tracking}" == "${local_head}" ]]; then
      remote_head="${local_head}"
    else
      sed -n '1,200p' "${log_path}" >&2 || true
      rm -f "${log_path}"
      printf 'Push outcome not verified: status=%s local=%s remote=%s\n' \
        "${command_status}" "${local_head}" "${remote_head}" >&2
      return 1
    fi
  fi
  git update-ref refs/remotes/origin/Experiment1 "${local_head}"
  rm -f "${log_path}"
  printf 'Verified push transaction: status=%s remote=%s\n' \
    "${command_status}" "${remote_head}"
}

cd "${ROOT}"

"${PYTHON_BIN}" - "${ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
from ccda_phase3.phase314b_r258_stagee_constrained_integrator import (
    validate_initial_worktree,
)
print(json.dumps(validate_initial_worktree(root), sort_keys=True))
PY

git diff --check

"${PYTHON_BIN}" -m py_compile \
  ccda_phase3/phase314b_r258_stagee_constrained_integrator.py \
  scripts/phase3_14b_r258_stagee_worker.py \
  scripts/phase3_14b_r258_stagee_run_calibration.py \
  scripts/phase3_14b_r258_stagee_test_gate.py \
  scripts/phase3_14b_r258_stagee_blocked.py
bash -n scripts/phase3_14b_r258_stagee_run.sh

PREFLIGHT_DIR="$(mktemp -d /tmp/phase314b_r258_stagee_environment.XXXXXX)"
PREFLIGHT="${PREFLIGHT_DIR}/environment.json"
if [[ -e "${PREFLIGHT}" ]]; then
  printf 'Stage-E preflight output unexpectedly exists\n' >&2
  exit 1
fi
"${PYTHON_BIN}" \
  scripts/phase3_14b_r258_stagee_worker.py \
  --root "${ROOT}" \
  --mode environment-probe \
  --output "${PREFLIGHT}"
"${PYTHON_BIN}" - "${PREFLIGHT}" <<'PY'
import json
import sys
from pathlib import Path
value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if value.get("compatibility_pass") is not True:
    raise RuntimeError("portable compatibility gate failed")
if value.get("compatibility", {}).get("required_operation_pass") is not True:
    raise RuntimeError("required-operation compatibility failed")
if value.get("required_operation_dry_run", {}).get("pass") is not True:
    raise RuntimeError("required-operation dry run failed")
print({
    "gpu": value["hardware_observation"]["torch_device_name"],
    "compute_capability": value["hardware_observation"]["compute_capability"],
    "compatibility_sha256": value["compatibility_sha256"],
    "observation_sha256": value["observation_sha256"],
    "peak_reserved_bytes": value["required_operation_dry_run"]["peak_memory_reserved_bytes"],
})
PY
rm -rf "${PREFLIGHT_DIR}"
PREFLIGHT_DIR=""

IMPLEMENTATION_PATHS=(
  ccda_phase3/phase314b_r258_stagee_constrained_integrator.py
  scripts/phase3_14b_r258_stagee_worker.py
  scripts/phase3_14b_r258_stagee_run_calibration.py
  scripts/phase3_14b_r258_stagee_test_gate.py
  scripts/phase3_14b_r258_stagee_blocked.py
  scripts/phase3_14b_r258_stagee_run.sh
  tests/test_phase3_14b_r258_stagee_constrained_integrator.py
)
git add "${IMPLEMENTATION_PATHS[@]}"
commit_with_outcome \
  "Add Phase3.14b-r2.5.8 surrogate-guided constrained direct-x0 integrator" \
  "${IMPLEMENTATION_PATHS[@]}"

"${PYTHON_BIN}" \
  scripts/phase3_14b_r258_stagee_test_gate.py \
  --root "${ROOT}" \
  --python-bin "${PYTHON_BIN}"

"${PYTHON_BIN}" \
  scripts/phase3_14b_r258_stagee_run_calibration.py \
  --root "${ROOT}" \
  --python-bin "${PYTHON_BIN}"

SUCCESS_PATHS=(
  reports/phase3_14b_r258_stagee_test_gate_summary.json
  reports/phase3_14b_r258_stagee_contract.json
  reports/phase3_14b_r258_stagee_worker_evidence.json
  reports/phase3_14b_r258_stagee_summary.json
  reports/phase3_14b_r258_stagee_report.md
)
git add "${SUCCESS_PATHS[@]}"
commit_with_outcome \
  "Record Phase3.14b-r2.5.8 constrained integrator evidence" \
  "${SUCCESS_PATHS[@]}"

push_with_outcome
PUSH_COMPLETED="true"

LOCAL_HEAD="$(git rev-parse HEAD)"
REMOTE_HEAD="$(git rev-parse origin/Experiment1)"
if [[ "${LOCAL_HEAD}" != "${REMOTE_HEAD}" ]]; then
  printf 'Final local/remote mismatch\n' >&2
  exit 1
fi
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  git status --short >&2
  printf 'Main worktree is not clean\n' >&2
  exit 1
fi
if [[ "$(git -C external/deformable-ravens rev-parse HEAD)" != \
  "633a88752445cf5d6776ed374fdbbdb35f93050c" ]]; then
  printf 'Submodule commit changed\n' >&2
  exit 1
fi
if [[ -n "$(git -C external/deformable-ravens status --porcelain --untracked-files=all)" ]]; then
  printf 'Submodule is dirty\n' >&2
  exit 1
fi

trap - ERR
printf 'Stage E complete and pushed: %s\n' "${LOCAL_HEAD}"
