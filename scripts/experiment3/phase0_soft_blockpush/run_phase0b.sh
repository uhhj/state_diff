#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/experiment3/soft_blockpush_phase0b.json}"
PAIR_ROOT=/data/Experiment3/data/ccda_soft_blockpush_audit
REPORT_ROOT=/data/Experiment3/reports/phase0_soft_blockpush

python scripts/experiment3/phase0_soft_blockpush/run_single_pair.py --config "$CONFIG" --stop-after-probe
python scripts/experiment3/phase0_soft_blockpush/analyze_pair.py --pair-dir "$PAIR_ROOT/pair_sbp_074001_probe_only" --report-dir "$REPORT_ROOT/probe_only"
python scripts/experiment3/phase0_soft_blockpush/visualize_pair.py --pair-dir "$PAIR_ROOT/pair_sbp_074001_probe_only" --report-dir "$REPORT_ROOT/probe_only"

VERDICT=$(python -c "import json; print(json.load(open('$REPORT_ROOT/probe_only/metrics.json'))['verdict'])")
if [[ "$VERDICT" != "PHASE0B_PROBE_MECHANISM_COMPLETE" ]]; then
  echo "Probe gate stopped execution: $VERDICT"
  exit 0
fi

python scripts/experiment3/phase0_soft_blockpush/run_single_pair.py --config "$CONFIG"
python scripts/experiment3/phase0_soft_blockpush/analyze_pair.py --pair-dir "$PAIR_ROOT/pair_sbp_074001" --report-dir "$REPORT_ROOT/full_pair"
python scripts/experiment3/phase0_soft_blockpush/visualize_pair.py --pair-dir "$PAIR_ROOT/pair_sbp_074001" --report-dir "$REPORT_ROOT/full_pair"
