#!/usr/bin/env bash
set -euo pipefail

CONFIG="configs/experiment3/soft_blockpush_phase0b_r2r2.json"
PROFILES="configs/experiment3/material_profiles_r2r2"

python scripts/experiment3/phase0_soft_blockpush_r2r2/run_calibration.py \
  --config "${CONFIG}" --profile-dir "${PROFILES}"
