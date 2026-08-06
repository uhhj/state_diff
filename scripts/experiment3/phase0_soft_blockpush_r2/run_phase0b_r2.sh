#!/usr/bin/env bash
set -euo pipefail

CONFIG="configs/experiment3/soft_blockpush_phase0b_r2.json"
PROFILE_DIR="configs/experiment3/material_profiles_r2"
DATA_ROOT="/data/Experiment3/data/ccda_soft_blockpush_r2"
REPORT_ROOT="/data/Experiment3/reports/phase0_soft_blockpush_r2"
SELECTION="${REPORT_ROOT}/mechanics_validation/selected_microsteps.json"

python scripts/experiment3/phase0_soft_blockpush_r2/run_mechanics_validation.py \
  --config "${CONFIG}" --material-profile "${PROFILE_DIR}/soft_block_kv_r2_a8_z025.json"
python scripts/experiment3/phase0_soft_blockpush_r2/analyze_mechanics_validation.py \
  --validation-dir "${DATA_ROOT}/mechanics_validation" \
  --report-dir "${REPORT_ROOT}/mechanics_validation"
python - "${SELECTION}" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
if value["verdict"] != "PHASE0B_R2_MICROSTEP_VALIDATION_COMPLETE":
    raise SystemExit("microstep validation blocked")
PY

run_coupon() {
  local profile="$1" mode="$2"
  python scripts/experiment3/phase0_soft_blockpush_r2/run_material_coupon.py \
    --config "${CONFIG}" --material-profile "${PROFILE_DIR}/soft_block_${profile}.json" \
    --selected-microsteps "${SELECTION}" --mode "${mode}"
  python scripts/experiment3/phase0_soft_blockpush_r2/analyze_material_coupon.py \
    --coupon-dir "${DATA_ROOT}/coupon/${profile}/${mode}" \
    --report-dir "${REPORT_ROOT}/coupon/${profile}/${mode}"
}

PROFILE="kv_r2_a6_z025"
run_coupon "${PROFILE}" axial
AXIAL_VERDICT="$(python - "${REPORT_ROOT}/coupon/${PROFILE}/axial/metrics.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["verdict"])
PY
)"
if [[ "${AXIAL_VERDICT}" == *TOO_SOFT ]]; then PROFILE="kv_r2_a8_z025"
elif [[ "${AXIAL_VERDICT}" == *TOO_STIFF ]]; then PROFILE="kv_r2_a4_z025"
elif [[ "${AXIAL_VERDICT}" != "PHASE0B_R2_AXIAL_COMPLETE" ]]; then exit 2
fi
if [[ "${PROFILE}" != "kv_r2_a6_z025" ]]; then run_coupon "${PROFILE}" axial; fi
python - "${REPORT_ROOT}/coupon/${PROFILE}/axial/metrics.json" <<'PY'
import json, sys
if json.load(open(sys.argv[1], encoding="utf-8"))["verdict"] != "PHASE0B_R2_AXIAL_COMPLETE":
    raise SystemExit("no axial profile passed")
PY

run_coupon "${PROFILE}" shear
SHEAR_VERDICT="$(python - "${REPORT_ROOT}/coupon/${PROFILE}/shear/metrics.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["verdict"])
PY
)"
if [[ "${SHEAR_VERDICT}" != "PHASE0B_R2_SHEAR_COMPLETE" ]]; then
  if [[ "${PROFILE}" != "kv_r2_a6_z025" ]]; then exit 3; fi
  if [[ "${SHEAR_VERDICT}" == *TOO_SOFT ]]; then PROFILE="kv_r2_a8_z025"
  elif [[ "${SHEAR_VERDICT}" == *TOO_STIFF ]]; then PROFILE="kv_r2_a4_z025"
  else exit 3
  fi
  run_coupon "${PROFILE}" axial
  python - "${REPORT_ROOT}/coupon/${PROFILE}/axial/metrics.json" <<'PY'
import json, sys
if json.load(open(sys.argv[1], encoding="utf-8"))["verdict"] != "PHASE0B_R2_AXIAL_COMPLETE":
    raise SystemExit("second profile did not pass axial")
PY
  run_coupon "${PROFILE}" shear
fi
python - "${REPORT_ROOT}/coupon/${PROFILE}/shear/metrics.json" <<'PY'
import json, sys
if json.load(open(sys.argv[1], encoding="utf-8"))["verdict"] != "PHASE0B_R2_SHEAR_COMPLETE":
    raise SystemExit("selected profile did not pass shear; stop without Pair")
PY
python scripts/experiment3/phase0_soft_blockpush_r2/run_table_settle.py \
  --config "${CONFIG}" --material-profile "${PROFILE_DIR}/soft_block_${PROFILE}.json" \
  --selected-microsteps "${SELECTION}"
python scripts/experiment3/phase0_soft_blockpush_r2/analyze_table_settle.py \
  --settle-dir "${DATA_ROOT}/table_settle/${PROFILE}" \
  --report-dir "${REPORT_ROOT}/table_settle/${PROFILE}"
