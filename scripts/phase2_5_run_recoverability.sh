#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFRAVENS_ROOT="$ROOT/external/deformable-ravens"

NUM_SEEDS="${NUM_SEEDS:-20}"
SEED_START="${SEED_START:-300000}"
MAX_STEPS="${MAX_STEPS:-20}"
RANDOM_TRIALS="${RANDOM_TRIALS:-64}"
WORKERS="${WORKERS:-6}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/phase2_5_recoverability}"
export CCDA_SETTLE_SECONDS="${CCDA_SETTLE_SECONDS:-0.02}"
export CCDA_AUDIT_T_LIM="${CCDA_AUDIT_T_LIM:-15.0}"

CONDITIONS="${CONDITIONS:-free hidden_pin hidden_high_friction hidden_partial_pin hidden_soft_pin hidden_breakaway_pin hidden_friction_patch}"
POLICIES="${POLICIES:-nominal oracle_pull oracle_regrasp oracle_wiggle random_search}"

echo "[Phase2.5] root=$ROOT"
echo "[Phase2.5] deformable-ravens=$DEFRAVENS_ROOT"
echo "[Phase2.5] conditions=$CONDITIONS"
echo "[Phase2.5] policies=$POLICIES"
echo "[Phase2.5] primitive_timeout_seconds=$CCDA_AUDIT_T_LIM"
echo "[Phase2.5] workers=$WORKERS"
echo "[Phase2.5] settle_seconds=$CCDA_SETTLE_SECONDS"
echo "[Phase2.5] output_root=$OUTPUT_ROOT"

cd "$DEFRAVENS_ROOT"

python ccda_recoverability_audit.py \
  --task hidden-contact-cable-line \
  --conditions $CONDITIONS \
  --policies $POLICIES \
  --num_seeds "$NUM_SEEDS" \
  --seed_start "$SEED_START" \
  --output_root "$OUTPUT_ROOT" \
  --max_steps "$MAX_STEPS" \
  --random_trials "$RANDOM_TRIALS" \
  --workers "$WORKERS"

cd "$ROOT"

python scripts/phase2_5_recoverability_audit.py \
  --root "$ROOT" \
  --trials_csv "$DEFRAVENS_ROOT/$OUTPUT_ROOT/recoverability_trials.csv" \
  --summary_json "$DEFRAVENS_ROOT/$OUTPUT_ROOT/recoverability_summary.json" \
  --out_json reports/phase2_5_recoverability_summary.json \
  --out_md reports/phase2_5_recoverability_report.md

python scripts/phase2_5_visualize_recoverability.py \
  --root "$ROOT" \
  --trials_csv "$DEFRAVENS_ROOT/$OUTPUT_ROOT/recoverability_trials.csv" \
  --out_dir "$ROOT/reports/phase2_5_recoverability_visuals" || true

echo "[Phase2.5] done. See reports/phase2_5_recoverability_report.md"
