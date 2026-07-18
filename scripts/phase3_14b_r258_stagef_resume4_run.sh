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
      "${ROOT}/scripts/phase3_14b_r258_stagef_resume4_blocked.py" \
      --root "${ROOT}" \
      --exit-code "${code}" \
      --failed-line "${BASH_LINENO[0]:-unknown}" \
      --failed-command "${BASH_COMMAND:-unknown}" || true
  fi
  exit "${code}"
}
trap on_error ERR
trap cleanup_tmp EXIT

sorted_lines() { printf '%s\n' "$@" | LC_ALL=C sort; }

assert_cached_paths() {
  local -a expected=("$@")
  local expected_text observed_text
  expected_text="$(sorted_lines "${expected[@]}")"
  observed_text="$(git diff --cached --name-only | LC_ALL=C sort)"
  [[ "${observed_text}" == "${expected_text}" ]] || {
    printf 'Unexpected staged paths.\nExpected:\n%s\nObserved:\n%s\n' \
      "${expected_text}" "${observed_text}" >&2
    return 1
  }
}

commit_with_outcome() {
  local message="$1"; shift
  local -a expected_paths=("$@")
  local pre_head post_head parent subject actual_paths expected_text status log_path
  assert_cached_paths "${expected_paths[@]}"
  pre_head="$(git rev-parse HEAD)"
  log_path="$(mktemp /tmp/phase314b_r258_stagef_resume2_commit.XXXXXX.log)"
  set +e
  git commit -m "${message}" >"${log_path}" 2>&1
  status="$?"
  set -e
  post_head="$(git rev-parse HEAD)"
  [[ "${post_head}" != "${pre_head}" ]] || {
    sed -n '1,200p' "${log_path}" >&2 || true
    rm -f "${log_path}"
    printf 'Git transaction did not create a commit; status=%s\n' "${status}" >&2
    return 1
  }
  parent="$(git rev-parse "${post_head}^")"
  subject="$(git show -s --format=%s "${post_head}")"
  actual_paths="$(git diff-tree --no-commit-id --name-only -r "${post_head}" | LC_ALL=C sort)"
  expected_text="$(sorted_lines "${expected_paths[@]}")"
  [[ "${parent}" == "${pre_head}" ]] || return 1
  [[ "${subject}" == "${message}" ]] || return 1
  [[ "${actual_paths}" == "${expected_text}" ]] || return 1
  git diff --cached --quiet || return 1
  rm -f "${log_path}"
  printf 'Verified commit transaction: status=%s sha=%s\n' "${status}" "${post_head}"
}

push_with_outcome() {
  local local_head status log_path remote_head attempt
  local_head="$(git rev-parse HEAD)"
  log_path="$(mktemp /tmp/phase314b_r258_stagef_resume2_push.XXXXXX.log)"
  set +e
  git push origin Experiment1 >"${log_path}" 2>&1
  status="$?"
  set -e
  remote_head=""
  for attempt in 1 2 3; do
    remote_head="$(git ls-remote origin refs/heads/Experiment1 2>/dev/null | awk '{print $1}')"
    [[ "${remote_head}" == "${local_head}" ]] && break
    sleep 2
  done
  [[ "${remote_head}" == "${local_head}" ]] || {
    sed -n '1,200p' "${log_path}" >&2 || true
    rm -f "${log_path}"
    printf 'Push outcome not verified: status=%s local=%s remote=%s\n' \
      "${status}" "${local_head}" "${remote_head}" >&2
    return 1
  }
  git update-ref refs/remotes/origin/Experiment1 "${local_head}"
  rm -f "${log_path}"
  printf 'Verified push transaction: status=%s remote=%s\n' "${status}" "${remote_head}"
}

cd "${ROOT}"

"${PYTHON_BIN}" - "${ROOT}" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1]).resolve(); sys.path.insert(0, str(root))
from ccda_phase3.phase314b_r258_stagef_resume4_head_admission import validate_resume4_initial_state
print(json.dumps(validate_resume4_initial_state(root), sort_keys=True))
PY

git diff --check

STAGEF_BLOCKED_PATHS=(
  reports/phase3_14b_r258_stagef_test_gate_summary.json
  reports/phase3_14b_r258_stagef_blocked_summary.json
)
git add "${STAGEF_BLOCKED_PATHS[@]}"
commit_with_outcome \
  "Record blocked Phase3.14b-r2.5.8 Stage-F translation-invariance execution" \
  "${STAGEF_BLOCKED_PATHS[@]}"

"${PYTHON_BIN}" - "${ROOT}" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1]).resolve(); sys.path.insert(0, str(root))
from ccda_phase3.phase314b_r258_stagef_resume2_porcelain_recovery import validate_stagef_blocked_provenance_commit
print(json.dumps(validate_stagef_blocked_provenance_commit(root), sort_keys=True))
PY

RESUME1_IMPLEMENTATION_PATHS=(
  ccda_phase3/phase314b_r258_stagef_constraint_aware_surrogate.py
  ccda_phase3/phase314b_r258_stagef_resume1_translation_fix.py
  scripts/phase3_14b_r258_stagef_resume1_worker.py
  scripts/phase3_14b_r258_stagef_resume1_test_gate.py
  scripts/phase3_14b_r258_stagef_resume1_run_calibration.py
  scripts/phase3_14b_r258_stagef_resume1_blocked.py
  scripts/phase3_14b_r258_stagef_resume1_run.sh
  tests/test_phase3_14b_r258_stagef_resume1_translation_fix.py
)
git add "${RESUME1_IMPLEMENTATION_PATHS[@]}"
commit_with_outcome \
  "Add Phase3.14b-r2.5.8 Stage-F Resume1 stable translation features" \
  "${RESUME1_IMPLEMENTATION_PATHS[@]}"

"${PYTHON_BIN}" - "${ROOT}" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1]).resolve(); sys.path.insert(0, str(root))
from ccda_phase3.phase314b_r258_stagef_resume2_porcelain_recovery import validate_resume1_implementation_commit
print(json.dumps(validate_resume1_implementation_commit(root), sort_keys=True))
PY

RESUME1_BLOCKED_PATHS=(
  reports/phase3_14b_r258_stagef_resume1_blocked_summary.json
)
git add "${RESUME1_BLOCKED_PATHS[@]}"
commit_with_outcome \
  "Record blocked Phase3.14b-r2.5.8 Stage-F Resume1 porcelain-path execution" \
  "${RESUME1_BLOCKED_PATHS[@]}"

"${PYTHON_BIN}" - "${ROOT}" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1]).resolve(); sys.path.insert(0, str(root))
from ccda_phase3.phase314b_r258_stagef_resume2_porcelain_recovery import validate_resume1_blocked_provenance_commit
print(json.dumps(validate_resume1_blocked_provenance_commit(root), sort_keys=True))
PY

RESUME2_IMPLEMENTATION_PATHS=(
  ccda_phase3/phase314b_r258_stagef_resume2_porcelain_recovery.py
  scripts/phase3_14b_r258_stagef_resume4_worker.py
  scripts/phase3_14b_r258_stagef_resume4_test_gate.py
  scripts/phase3_14b_r258_stagef_resume4_run_calibration.py
  scripts/phase3_14b_r258_stagef_resume4_blocked.py
  scripts/phase3_14b_r258_stagef_resume4_run.sh
  tests/test_phase3_14b_r258_stagef_resume2_porcelain_recovery.py
)
git add "${RESUME2_IMPLEMENTATION_PATHS[@]}"
commit_with_outcome \
  "Add Phase3.14b-r2.5.8 Stage-F Resume2 robust porcelain path recovery" \
  "${RESUME2_IMPLEMENTATION_PATHS[@]}"

"${PYTHON_BIN}" - "${ROOT}" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1]).resolve(); sys.path.insert(0, str(root))
from ccda_phase3.phase314b_r258_stagef_resume2_porcelain_recovery import validate_resume2_implementation_commit
print(json.dumps(validate_resume2_implementation_commit(root), sort_keys=True))
PY

PREFLIGHT_DIR="$(mktemp -d /tmp/phase314b_r258_stagef_resume2_environment.XXXXXX)"
PREFLIGHT="${PREFLIGHT_DIR}/environment.json"
[[ ! -e "${PREFLIGHT}" ]]
"${PYTHON_BIN}" scripts/phase3_14b_r258_stagef_resume4_worker.py \
  --root "${ROOT}" --mode environment-probe --output "${PREFLIGHT}"
"${PYTHON_BIN}" - "${PREFLIGHT}" <<'PY'
import json, sys
from pathlib import Path
value = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
if value.get('compatibility_pass') is not True:
    raise RuntimeError('portable compatibility gate failed')
if value.get('required_operation_dry_run', {}).get('pass') is not True:
    raise RuntimeError('required-operation dry run failed')
print({
    'gpu': value['hardware_observation']['torch_device_name'],
    'compute_capability': value['hardware_observation']['compute_capability'],
    'compatibility_sha256': value['compatibility_sha256'],
    'observation_sha256': value['observation_sha256'],
})
PY
rm -rf "${PREFLIGHT_DIR}"
PREFLIGHT_DIR=""

"${PYTHON_BIN}" scripts/phase3_14b_r258_stagef_resume4_test_gate.py \
  --root "${ROOT}" --python-bin "${PYTHON_BIN}"
"${PYTHON_BIN}" scripts/phase3_14b_r258_stagef_resume4_run_calibration.py \
  --root "${ROOT}" --python-bin "${PYTHON_BIN}"

SUCCESS_PATHS=(
  reports/phase3_14b_r258_stagef_resume4_test_gate_summary.json
  reports/phase3_14b_r258_stagef_resume4_contract.json
  reports/phase3_14b_r258_stagef_resume4_worker_evidence.json
  reports/phase3_14b_r258_stagef_resume4_summary.json
  reports/phase3_14b_r258_stagef_resume4_report.md
)
git add "${SUCCESS_PATHS[@]}"
commit_with_outcome \
  "Record Phase3.14b-r2.5.8 Stage-F Resume2 evidence" \
  "${SUCCESS_PATHS[@]}"

push_with_outcome
PUSH_COMPLETED="true"

LOCAL_HEAD="$(git rev-parse HEAD)"
REMOTE_HEAD="$(git rev-parse origin/Experiment1)"
[[ "${LOCAL_HEAD}" == "${REMOTE_HEAD}" ]]
[[ -z "$(git status --porcelain=v1 --untracked-files=all)" ]]
[[ "$(git -C external/deformable-ravens rev-parse HEAD)" == \
   "633a88752445cf5d6776ed374fdbbdb35f93050c" ]]
[[ -z "$(git -C external/deformable-ravens status --porcelain=v1 --untracked-files=all)" ]]

trap - ERR
printf 'Stage-F Resume2 complete and pushed: %s\n' "${LOCAL_HEAD}"
