#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/experiment3/soft_blockpush_phase0b_r1.json}"
PROFILE="${2:-configs/experiment3/material_profiles/soft_block_kv_c0.json}"
PROFILE_NAME=$(python -c "import json; print(json.load(open('$PROFILE'))['profile_name'])")
DATA=/data/Experiment3/data/ccda_soft_blockpush_r1
REPORT=/data/Experiment3/reports/phase0_soft_blockpush_r1

python scripts/experiment3/phase0_soft_blockpush_r1/run_material_coupon.py --config "$CONFIG" --material-profile "$PROFILE"
python scripts/experiment3/phase0_soft_blockpush_r1/analyze_material_coupon.py --coupon-dir "$DATA/coupon/$PROFILE_NAME" --report-dir "$REPORT/coupon/$PROFILE_NAME"
COUPON="$REPORT/coupon/$PROFILE_NAME/metrics.json"
VERDICT=$(python -c "import json; print(json.load(open('$COUPON'))['verdict'])")
if [[ "$VERDICT" != "PHASE0B_R1_COUPON_COMPLETE" ]]; then
  echo "Coupon gate stopped execution: $VERDICT"; exit 0
fi

python scripts/experiment3/phase0_soft_blockpush_r1/run_single_pair.py --config "$CONFIG" --frozen-material-report "$COUPON" --stop-after-probe
python scripts/experiment3/phase0_soft_blockpush_r1/analyze_pair.py --pair-dir "$DATA/pair_sbp_r1_074101_probe_only" --report-dir "$REPORT/probe_only"
PROBE=$(python -c "import json; print(json.load(open('$REPORT/probe_only/metrics.json'))['verdict'])")
if [[ "$PROBE" != "PHASE0B_R1_PROBE_MECHANISM_COMPLETE" ]]; then
  echo "Probe gate stopped execution: $PROBE"; exit 0
fi

python scripts/experiment3/phase0_soft_blockpush_r1/run_single_pair.py --config "$CONFIG" --frozen-material-report "$COUPON"
python scripts/experiment3/phase0_soft_blockpush_r1/analyze_pair.py --pair-dir "$DATA/pair_sbp_r1_074101" --report-dir "$REPORT/full_pair"
python scripts/experiment3/phase0_soft_blockpush_r1/visualize.py --pair-dir "$DATA/pair_sbp_r1_074101" --report-dir "$REPORT/full_pair"
