#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/experiment3/soft_blockpush_phase0b_r3.json}"
PROFILE_DIR="${2:-configs/experiment3/material_profiles_r3}"
START_SHA="${START_SHA:-8b8e2a2a4ec9339389e962cb0b40e26fe7aaabff}"
TESTS_PASSED="${R3_TESTS_PASSED:-0}"

python -m scripts.experiment3.phase0_soft_blockpush_r3.run_angle_validation \
  --config "$CONFIG" --material-profile "$PROFILE_DIR/soft_block_kv_angle_r3_a070_z025.json"
python -m scripts.experiment3.phase0_soft_blockpush_r3.analyze_angle_validation \
  --data-root /data/Experiment3/data/ccda_soft_blockpush_r3/angle_validation \
  --report-root /data/Experiment3/reports/phase0_soft_blockpush_r3/angle_validation
python - <<'PY'
import json
p='/data/Experiment3/reports/phase0_soft_blockpush_r3/angle_validation/metrics.json'
assert json.load(open(p))['verdict']=='PHASE0B_R3_ANGLE_VALIDATION_COMPLETE'
PY
mkdir -p /data/Experiment3/reports/phase0_soft_blockpush_r3/mechanics_validation
cp /data/Experiment3/reports/phase0_soft_blockpush_r3/angle_validation/selected_microsteps.json \
  /data/Experiment3/reports/phase0_soft_blockpush_r3/mechanics_validation/selected_microsteps.json
python -m scripts.experiment3.phase0_soft_blockpush_r3.run_calibration \
  --config "$CONFIG" --profile-dir "$PROFILE_DIR"
python -m scripts.experiment3.phase0_soft_blockpush_r3.build_result \
  --config "$CONFIG" --starting-sha "$START_SHA" \
  --committed-report-dir reports/experiment3/phase0_soft_blockpush_r3 \
  --tests-passed "$TESTS_PASSED" --tests-failed 0
