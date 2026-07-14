#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/data/state_diff2}"
EXPECTED_PYTHON="/miniforge3/envs/coord_bimanual/bin/python"
PYTHON_INPUT="${PHASE314B_PYTHON:-${EXPECTED_PYTHON}}"
PYTHON_BIN="$(readlink -m "${PYTHON_INPUT}")"
EXPECTED_RESOLVED="$(readlink -m "${EXPECTED_PYTHON}")"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Phase3.14b-r2.5.4 Resume1 Python is not executable: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ "${PYTHON_BIN}" != "${EXPECTED_RESOLVED}" ]]; then
  echo "Unexpected Phase3.14b-r2.5.4 Resume1 Python: ${PYTHON_BIN}" >&2
  exit 1
fi

cd "${ROOT}"
export PYTHONPATH="${ROOT}"
export PYTHONNOUSERSITE="1"

PREFLIGHT="reports/phase3_14b_r254_resume1_preflight_summary.json"
EVIDENCE="reports/phase3_14b_r254_resume1_prior_repeat_evidence.json"
AUDIT="reports/phase3_14b_r254_resume1_prior_audit_summary.json"
SUMMARY="reports/phase3_14b_r254_resume1_summary.json"
REPORT="reports/phase3_14b_r254_resume1_report.md"
BLOCKED_JSON="reports/phase3_14b_r254_resume1_blocked_summary.json"
BLOCKED_MD="reports/phase3_14b_r254_resume1_blocked_report.md"
LOG_FILE="$(mktemp /tmp/phase314b_r254_resume1.XXXXXX.log)"
CURRENT_STAGE="interpreter_probe"

cleanup() {
  rm -f "${LOG_FILE}"
}
trap cleanup EXIT

write_blocked() {
  local exit_code="$1"
  local failed_line="$2"
  if [[ -e "${BLOCKED_JSON}" || -e "${BLOCKED_MD}" ]]; then
    echo "Refusing to overwrite r2.5.4 Resume1 blocked evidence" >&2
    return 1
  fi
  PHASE314B_STAGE="${CURRENT_STAGE}" \
  PHASE314B_EXIT_CODE="${exit_code}" \
  PHASE314B_FAILED_LINE="${failed_line}" \
  PHASE314B_LOG_FILE="${LOG_FILE}" \
  PHASE314B_PREFLIGHT="${PREFLIGHT}" \
  PHASE314B_EVIDENCE="${EVIDENCE}" \
  PHASE314B_AUDIT="${AUDIT}" \
  PHASE314B_BLOCKED_JSON="${BLOCKED_JSON}" \
  PHASE314B_BLOCKED_MD="${BLOCKED_MD}" \
  "${PYTHON_BIN}" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

stage = os.environ["PHASE314B_STAGE"]
exit_code = int(os.environ["PHASE314B_EXIT_CODE"])
failed_line = os.environ["PHASE314B_FAILED_LINE"]
log_path = Path(os.environ["PHASE314B_LOG_FILE"])
preflight_path = Path(os.environ["PHASE314B_PREFLIGHT"])
evidence_path = Path(os.environ["PHASE314B_EVIDENCE"])
audit_path = Path(os.environ["PHASE314B_AUDIT"])
json_path = Path(os.environ["PHASE314B_BLOCKED_JSON"])
md_path = Path(os.environ["PHASE314B_BLOCKED_MD"])
log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
tail = "\n".join(log.splitlines()[-120:])

def file_sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None

evidence = None
if evidence_path.exists():
    try:
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    except Exception:
        evidence = None
audit = None
if audit_path.exists():
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except Exception:
        audit = None
source = audit if isinstance(audit, dict) else evidence
observed = []
if isinstance(source, dict):
    for row in source.get("runs", []):
        if isinstance(row, dict) and row.get("prior_state_sha256"):
            observed.append(row["prior_state_sha256"])

payload = {
    "phase": "phase3_14b_r254_resume1",
    "verdict": "BLOCKED",
    "root_cause": "phase314b_r254_resume1_execution_failed",
    "failed_stage": stage,
    "exit_code": exit_code,
    "failed_line": failed_line,
    "traceback_tail": tail,
    "preflight_exists": preflight_path.exists(),
    "preflight_sha256": file_sha(preflight_path),
    "repeat_evidence_exists": evidence_path.exists(),
    "repeat_evidence_sha256": file_sha(evidence_path),
    "prior_audit_exists": audit_path.exists(),
    "prior_audit_sha256": file_sha(audit_path),
    "observed_prior_state_sha256": observed,
    "observed_sha_persisted_before_gate": bool(observed),
    "robot_proxy_attribution_run": False,
    "reverse_sampling_rerun": False,
    "train_only_recommendation": None,
    "selected_configuration": None,
    "validation_targets_used": False,
    "formal_test_read": False,
    "formal_training": False,
    "checkpoint_saved": False,
    "idm": False,
    "candidate_execution": False,
    "phase4": False,
    "cps": False,
}
json_path.parent.mkdir(parents=True, exist_ok=True)
with json_path.open("x", encoding="utf-8") as stream:
    json.dump(payload, stream, indent=2, sort_keys=True)
    stream.write("\n")
with md_path.open("x", encoding="utf-8") as stream:
    stream.write("# Phase3.14b-r2.5.4 Resume1 BLOCKED\n\n")
    stream.write(f"- Failed stage: `{stage}`\n")
    stream.write(f"- Exit code: `{exit_code}`\n")
    stream.write(f"- Failed line: `{failed_line}`\n")
    stream.write(f"- Repeat evidence SHA256: `{file_sha(evidence_path)}`\n")
    stream.write(f"- Prior audit SHA256: `{file_sha(audit_path)}`\n")
    stream.write(f"- Observed prior SHA values: `{observed}`\n\n")
    stream.write("```text\n")
    stream.write(tail)
    stream.write("\n```\n")
PY
}

on_error() {
  local exit_code=$?
  local failed_line="${BASH_LINENO[0]:-unknown}"
  trap - ERR
  write_blocked "${exit_code}" "${failed_line}" || true
  exit "${exit_code}"
}
trap on_error ERR

if [[ -e "${PREFLIGHT}" || -e "${EVIDENCE}" || -e "${AUDIT}" || -e "${SUMMARY}" || -e "${REPORT}" || -e "${BLOCKED_JSON}" || -e "${BLOCKED_MD}" ]]; then
  echo "Refusing to overwrite existing Phase3.14b-r2.5.4 Resume1 artifacts" >&2
  exit 1
fi

"${PYTHON_BIN}" - <<'PY' 2>&1 | tee -a "${LOG_FILE}"
import json
import os
import sys
import numpy
import torch
expected = os.path.realpath("/miniforge3/envs/coord_bimanual/bin/python")
observed = os.path.realpath(sys.executable)
if observed != expected:
    raise RuntimeError(f"interpreter mismatch: expected={expected}, observed={observed}")
if sys.version_info[:3] != (3, 9, 15):
    raise RuntimeError(f"Python mismatch: {sys.version}")
if numpy.__version__ != "1.23.3":
    raise RuntimeError(f"NumPy mismatch: {numpy.__version__}")
if torch.__version__ != "1.12.1.post200":
    raise RuntimeError(f"PyTorch mismatch: {torch.__version__}")
if str(torch.version.cuda) != "11.2":
    raise RuntimeError(f"CUDA runtime mismatch: {torch.version.cuda}")
if os.environ.get("PYTHONNOUSERSITE") != "1" or not sys.flags.no_user_site:
    raise RuntimeError("user-site isolation contract failed")
if not torch.cuda.is_available():
    raise RuntimeError("prior determinism audit requires CUDA")
capability = torch.cuda.get_device_capability(0)
print(json.dumps({
    "interpreter": observed,
    "python": sys.version.split()[0],
    "numpy": numpy.__version__,
    "torch": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "gpu": torch.cuda.get_device_name(0),
    "gpu_capability": list(capability),
    "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    "cudnn_deterministic": torch.backends.cudnn.deterministic,
    "cudnn_benchmark": torch.backends.cudnn.benchmark,
    "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
    "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
    "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
}, sort_keys=True))
PY

CURRENT_STAGE="repository_import_probe"
"${PYTHON_BIN}" -c 'import ccda_phase3; from ccda_phase3.phase314b_r254_resume1_prior_determinism import PHASE; print(PHASE)' 2>&1 | tee -a "${LOG_FILE}"

CURRENT_STAGE="preflight"
"${PYTHON_BIN}" scripts/phase3_14b_r254_resume1_preflight.py \
  --root "${ROOT}" --output "${PREFLIGHT}" 2>&1 | tee -a "${LOG_FILE}"

CURRENT_STAGE="prior_repeat_audit"
"${PYTHON_BIN}" scripts/phase3_14b_r254_resume1_run_audit.py \
  --root "${ROOT}" \
  --preflight-report "${PREFLIGHT}" \
  --evidence-output "${EVIDENCE}" \
  --output "${AUDIT}" 2>&1 | tee -a "${LOG_FILE}"

CURRENT_STAGE="finalization"
"${PYTHON_BIN}" scripts/phase3_14b_r254_resume1_finalize.py \
  --root "${ROOT}" \
  --preflight-report "${PREFLIGHT}" \
  --evidence-report "${EVIDENCE}" \
  --audit-report "${AUDIT}" \
  --summary-output "${SUMMARY}" \
  --markdown-output "${REPORT}" 2>&1 | tee -a "${LOG_FILE}"

trap - ERR
echo "Phase3.14b-r2.5.4 Resume1 completed"
