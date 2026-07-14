#!/usr/bin/env bash
set -euo pipefail

ROOT_INPUT="${1:-/data/state_diff2}"
ROOT="$(cd "${ROOT_INPUT}" && pwd -P)"
cd "${ROOT}"
mkdir -p reports

EXPECTED_PYTHON="/miniforge3/envs/coord_bimanual/bin/python"
PYTHON_BIN_INPUT="${PHASE314B_PYTHON:-${EXPECTED_PYTHON}}"
PYTHON_BIN="$(readlink -m "${PYTHON_BIN_INPUT}")"
EXPECTED_PYTHON="$(readlink -m "${EXPECTED_PYTHON}")"

PREFLIGHT="reports/phase3_14b_r253_resume3_preflight_summary.json"
BLOCKED_JSON="reports/phase3_14b_r253_resume3_blocked_summary.json"
BLOCKED_MD="reports/phase3_14b_r253_resume3_blocked_report.md"
PILOT="reports/phase3_14b_r253_pilot_summary.json"
PILOT_SHA="a6a8aa2faf739e80159af7e70cb57b0e2a4c852ab5af5c2d7ecb29ff31116c86"
FINAL_SUMMARY="reports/phase3_14b_r253_summary.json"
FINAL_REPORT="reports/phase3_14b_r253_report.md"

for path in \
  "${PREFLIGHT}" \
  "${BLOCKED_JSON}" \
  "${BLOCKED_MD}" \
  "${FINAL_SUMMARY}" \
  "${FINAL_REPORT}"; do
  if [[ -e "${path}" ]]; then
    echo "Resume3/final output already exists; refusing to overwrite: ${path}" >&2
    exit 2
  fi
done

write_shell_interpreter_blocked() {
  local reason="$1"
  local actual="${2:-unresolved}"
  local pilot_actual="missing"
  if [[ -f "${PILOT}" ]]; then
    pilot_actual="$(sha256sum "${PILOT}" | awk '{print $1}')"
  fi
  cat > "${BLOCKED_JSON}" <<JSON
{
  "phase": "Phase3.14b-r2.5.3-Resume3",
  "verdict": "BLOCKED",
  "root_cause": "phase314b_r253_resume3_interpreter_contract_failed",
  "failure_reason": "${reason}",
  "expected_python": "${EXPECTED_PYTHON}",
  "observed_python": "${actual}",
  "finalizer_only": true,
  "gpu_pilot_rerun": false,
  "training_rerun": false,
  "one_step_rerun": false,
  "reverse_rerun": false,
  "pilot_summary_sha256": "${pilot_actual}",
  "expected_pilot_summary_sha256": "${PILOT_SHA}",
  "pilot_summary_unchanged": $([[ "${pilot_actual}" == "${PILOT_SHA}" ]] && echo true || echo false),
  "train_only_recommendation": null,
  "selected_configuration": null,
  "validation_targets_used": false,
  "formal_test_read": false,
  "formal_training": false,
  "idm": false,
  "candidate_execution": false,
  "phase4": false,
  "cps": false
}
JSON
  local pilot_unchanged="False"
  if [[ "${pilot_actual}" == "${PILOT_SHA}" ]]; then
    pilot_unchanged="True"
  fi
  printf '%s\n' \
    '# Phase3.14b-r2.5.3 Resume3 Blocked Report' \
    '' \
    "- Failure: \`${reason}\`" \
    "- Expected Python: \`${EXPECTED_PYTHON}\`" \
    "- Observed Python: \`${actual}\`" \
    '- GPU pilot rerun: `False`' \
    '- Training rerun: `False`' \
    '- One-step rerun: `False`' \
    '- Reverse rerun: `False`' \
    "- Pilot SHA unchanged: \`${pilot_unchanged}\`" \
    '- Train-only recommendation: `None`' \
    '- Selected configuration: `None`' \
    > "${BLOCKED_MD}"
}

if [[ ! -x "${PYTHON_BIN}" ]]; then
  write_shell_interpreter_blocked "python_not_executable" "${PYTHON_BIN}"
  exit 1
fi
if [[ "${PYTHON_BIN}" != "${EXPECTED_PYTHON}" ]]; then
  write_shell_interpreter_blocked "python_executable_mismatch" "${PYTHON_BIN}"
  exit 1
fi

export PYTHONPATH="${ROOT}"
export PYTHONNOUSERSITE="1"
export CUDA_VISIBLE_DEVICES=""
export PHASE314B_PYTHON_RESOLVED="${PYTHON_BIN}"

write_blocked() {
  local exit_code="$1"
  local failed_line="$2"
  local failed_command="$3"
  "${PYTHON_BIN}" - \
    "${ROOT}" \
    "${exit_code}" \
    "${failed_line}" \
    "${failed_command}" \
    "${PREFLIGHT}" \
    "${PILOT}" \
    "${PILOT_SHA}" \
    "${PYTHON_BIN}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
exit_code = int(sys.argv[2])
failed_line = sys.argv[3]
failed_command = sys.argv[4]
preflight = sys.argv[5]
pilot_relative = sys.argv[6]
expected_sha = sys.argv[7]
python_bin = str(Path(sys.argv[8]).resolve())
pilot = root / pilot_relative
actual_sha = hashlib.sha256(pilot.read_bytes()).hexdigest() if pilot.is_file() else None


def digest(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


payload = {
    "phase": "Phase3.14b-r2.5.3-Resume3",
    "verdict": "BLOCKED",
    "root_cause": "phase314b_r253_resume3_finalization_failed",
    "exit_code": exit_code,
    "failed_line": failed_line,
    "failed_command": failed_command,
    "finalizer_only": True,
    "gpu_pilot_rerun": False,
    "training_rerun": False,
    "one_step_rerun": False,
    "reverse_rerun": False,
    "python_executable": python_bin,
    "pythonpath": sys.path,
    "preflight_report": preflight if (root / preflight).exists() else None,
    "pilot_summary": pilot_relative,
    "pilot_summary_sha256": actual_sha,
    "expected_pilot_summary_sha256": expected_sha,
    "pilot_summary_unchanged": actual_sha == expected_sha,
    "final_summary_sha256": digest(root / "reports/phase3_14b_r253_summary.json"),
    "final_report_sha256": digest(root / "reports/phase3_14b_r253_report.md"),
    "train_only_recommendation": None,
    "selected_configuration": None,
    "validation_targets_used": False,
    "formal_test_read": False,
    "formal_training": False,
    "idm": False,
    "candidate_execution": False,
    "phase4": False,
    "cps": False,
}
json_path = root / "reports/phase3_14b_r253_resume3_blocked_summary.json"
md_path = root / "reports/phase3_14b_r253_resume3_blocked_report.md"
json_path.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
md_path.write_text(
    "# Phase3.14b-r2.5.3 Resume3 Blocked Report\n\n"
    f"- Exit code: `{exit_code}`\n"
    f"- Failed line: `{failed_line}`\n"
    f"- Failed command: `{failed_command}`\n"
    f"- Python executable: `{python_bin}`\n"
    "- GPU pilot rerun: `False`\n"
    "- Training rerun: `False`\n"
    "- One-step rerun: `False`\n"
    "- Reverse rerun: `False`\n"
    f"- Pilot SHA unchanged: `{payload['pilot_summary_unchanged']}`\n"
    "- Train-only recommendation: `None`\n"
    "- Selected configuration: `None`\n",
    encoding="utf-8",
)
PY
}

trap 'code=$?; line=${BASH_LINENO[0]:-unknown}; command=${BASH_COMMAND:-unknown}; trap - ERR; write_blocked "$code" "$line" "$command"; exit "$code"' ERR

"${PYTHON_BIN}" - <<'PY'
import json
import os
import platform
import sys

import numpy
import torch

expected = {
    "python_executable": os.path.realpath(
        os.environ["PHASE314B_PYTHON_RESOLVED"]
    ),
    "python_version": "3.9.15",
    "numpy_version": "1.23.3",
    "torch_version": "1.12.1.post200",
    "torch_cuda_version": "11.2",
    "cuda_available": False,
    "cuda_visible_devices": "",
    "python_no_user_site": "1",
    "no_user_site_flag": True,
}
observed = {
    "python_executable": os.path.realpath(sys.executable),
    "python_version": platform.python_version(),
    "numpy_version": numpy.__version__,
    "torch_version": torch.__version__,
    "torch_cuda_version": str(torch.version.cuda),
    "cuda_available": torch.cuda.is_available(),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "python_no_user_site": os.environ.get("PYTHONNOUSERSITE"),
    "no_user_site_flag": bool(sys.flags.no_user_site),
}
if observed != expected:
    raise RuntimeError(
        "coord_bimanual interpreter contract mismatch: "
        f"expected={expected}, observed={observed}"
    )
import ccda_phase3
from ccda_phase3.phase314b_r253_resume1_finalizer import corrected_pilot_view
print("interpreter and repository import contract: PASS")
print(json.dumps(observed, sort_keys=True))
PY

"${PYTHON_BIN}" scripts/phase3_14b_r253_resume3_preflight.py \
  --root "${ROOT}" \
  --output "${PREFLIGHT}"

"${PYTHON_BIN}" scripts/phase3_14b_r253_resume3_finalize.py \
  --root "${ROOT}" \
  --preflight-report "${PREFLIGHT}" \
  --pilot-report "${PILOT}"

trap - ERR
echo "Phase3.14b-r2.5.3 Resume3 finalizer completed without GPU, training, one-step, or reverse rerun."
