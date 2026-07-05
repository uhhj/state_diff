#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "[Phase3.1] root=$ROOT"
echo "[Phase3.1] HEAD=$(git rev-parse HEAD)"
echo "[Phase3.1] branch=$(git branch --show-current)"
echo "[Phase3.1] status:"
git status --short

mkdir -p reports

python scripts/phase3_1_code_hazard_audit.py \
  --root "$ROOT" \
  --out_json reports/phase3_1_code_hazard_summary.json \
  --out_md reports/phase3_1_code_hazard_report.md

python scripts/phase3_1_medium_evidence_audit.py \
  --root "$ROOT" \
  --data data/phase3_state_diff_windows/phase3_windows.npz \
  --pred_csv reports/phase3_baseline_eval_predictions.csv \
  --eval_summary reports/phase3_baseline_eval_summary.json \
  --leak_json reports/phase3_input_leakage_summary.json \
  --integration_json reports/phase3_condition_integration_audit_summary.json \
  --canonical_json reports/phase3_canonicalization_summary.json \
  --action_debug_json reports/phase3_action_idm_debug_summary.json \
  --sanity_json reports/phase3_sanity_check_summary.json \
  --out_json reports/phase3_1_medium_evidence_summary.json \
  --out_md reports/phase3_1_medium_evidence_report.md \
  --out_pair_delta_csv reports/phase3_1_primary_pair_deltas.csv \
  --out_action_dim_csv reports/phase3_1_action_dim_review.csv \
  --out_ood_topk_csv reports/phase3_1_idm_ood_topk.csv

echo "[Phase3.1] done"
echo "[Phase3.1] reports:"
echo "  reports/phase3_1_code_hazard_report.md"
echo "  reports/phase3_1_medium_evidence_report.md"
