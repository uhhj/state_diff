#!/usr/bin/env bash
set -euo pipefail

CONFIG="configs/experiment3/soft_blockpush_phase0b_r2r1.json"
PROFILES="configs/experiment3/material_profiles_r2r1"
DATA="/data/Experiment3/data/ccda_soft_blockpush_r2r1/family_confirmation"
REPORT="/data/Experiment3/reports/phase0_soft_blockpush_r2r1"
SELECTION="${REPORT}/mechanics_validation/selected_microsteps.json"

python scripts/experiment3/phase0_soft_blockpush_r2r1/run_family_confirmation.py \
  --config "${CONFIG}" \
  --material-profile "${PROFILES}/soft_block_kv_r2r1_a1p50_z025.json"
python scripts/experiment3/phase0_soft_blockpush_r2r1/analyze_family_confirmation.py \
  --confirmation-dir "${DATA}" --report-dir "${REPORT}/mechanics_validation"
python - "${REPORT}/mechanics_validation/summary.json" <<'PY'
import json, sys
if json.load(open(sys.argv[1], encoding="utf-8"))["verdict"] != \
        "PHASE0B_R2R1_FAMILY_CONFIRMATION_COMPLETE":
    raise SystemExit("family confirmation blocked")
PY
python scripts/experiment3/phase0_soft_blockpush_r2r1/run_calibration.py \
  --config "${CONFIG}" --profile-dir "${PROFILES}" \
  --selected-microsteps "${SELECTION}"
