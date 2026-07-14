#!/usr/bin/env bash
set -euo pipefail

ROOT_INPUT="${1:-/data/state_diff2}"
ROOT="$(cd "${ROOT_INPUT}" && pwd -P)"
cd "${ROOT}"

# Python invoked as `python scripts/file.py` receives scripts/ as sys.path[0].
# Explicitly prepend the repository root before any project import.
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
# Finalizer-only recovery must never acquire a GPU context.
export CUDA_VISIBLE_DEVICES=""

PREFLIGHT="reports/phase3_14b_r253_resume2_preflight_summary.json"
BLOCKED_JSON="reports/phase3_14b_r253_resume2_blocked_summary.json"
BLOCKED_MD="reports/phase3_14b_r253_resume2_blocked_report.md"
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
    echo "Resume2/final output already exists; refusing to overwrite: ${path}" >&2
    exit 2
  fi
done

write_blocked() {
  local exit_code="$1"
  local failed_line="$2"
  local failed_command="$3"
  python - \
    "${ROOT}" \
    "${exit_code}" \
    "${failed_line}" \
    "${failed_command}" \
    "${PREFLIGHT}" \
    "${PILOT}" \
    "${PILOT_SHA}" \
    "${PYTHONPATH}" <<'PY'
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
pythonpath = sys.argv[8]
pilot = root / pilot_relative
actual_sha = hashlib.sha256(pilot.read_bytes()).hexdigest() if pilot.is_file() else None

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None

payload = {
    "phase": "Phase3.14b-r2.5.3-Resume2",
    "verdict": "BLOCKED",
    "root_cause": "phase314b_r253_resume2_finalization_failed",
    "exit_code": exit_code,
    "failed_line": failed_line,
    "failed_command": failed_command,
    "finalizer_only": True,
    "gpu_pilot_rerun": False,
    "training_rerun": False,
    "reverse_rerun": False,
    "pythonpath": pythonpath,
    "pythonpath_repository_root_first": (
        pythonpath.split(":", 1)[0] == str(root)
    ),
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
json_path = root / "reports/phase3_14b_r253_resume2_blocked_summary.json"
md_path = root / "reports/phase3_14b_r253_resume2_blocked_report.md"
json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
md_path.write_text(
    "# Phase3.14b-r2.5.3 Resume2 Blocked Report\n\n"
    f"- Exit code: `{exit_code}`\n"
    f"- Failed line: `{failed_line}`\n"
    f"- Failed command: `{failed_command}`\n"
    "- GPU pilot rerun: `False`\n"
    "- Training rerun: `False`\n"
    "- Reverse rerun: `False`\n"
    f"- Repository root first in PYTHONPATH: "
    f"`{payload['pythonpath_repository_root_first']}`\n"
    f"- Pilot SHA unchanged: `{payload['pilot_summary_unchanged']}`\n"
    "- Train-only recommendation: `None`\n"
    "- Selected configuration: `None`\n",
    encoding="utf-8",
)
PY
}

trap 'code=$?; line=${BASH_LINENO[0]:-unknown}; command=${BASH_COMMAND:-unknown}; trap - ERR; write_blocked "$code" "$line" "$command"; exit "$code"' ERR

python -c 'import ccda_phase3; from ccda_phase3.phase314b_r253_resume1_finalizer import corrected_pilot_view; print("repository import contract: PASS")'

python scripts/phase3_14b_r253_resume2_preflight.py \
  --root "${ROOT}" \
  --output "${PREFLIGHT}"

python scripts/phase3_14b_r253_resume2_finalize.py \
  --root "${ROOT}" \
  --preflight-report "${PREFLIGHT}" \
  --pilot-report "${PILOT}"

trap - ERR

echo "Phase3.14b-r2.5.3 Resume2 finalizer completed without GPU, training, or reverse rerun."
