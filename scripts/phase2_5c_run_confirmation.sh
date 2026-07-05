#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFRAVENS_ROOT="$ROOT/external/deformable-ravens"

NUM_SEEDS="${NUM_SEEDS:-30}"
SEED_START="${SEED_START:-330000}"
MAX_STEPS="${MAX_STEPS:-16}"
RANDOM_TRIALS="${RANDOM_TRIALS:-32}"
WORKERS="${WORKERS:-6}"

OUT_ROOT="$DEFRAVENS_ROOT/data/phase2_5c_confirmation"

CONDITIONS="${CONDITIONS:-free hidden_pin hidden_high_friction hidden_breakaway_pin}"
POLICIES="${POLICIES:-nominal oracle_breakaway_then_place oracle_pull oracle_regrasp oracle_wiggle guided_search}"

export CCDA_BREAKAWAY_FORCE="${CCDA_BREAKAWAY_FORCE:-2.6}"
export CCDA_BREAKAWAY_DISP="${CCDA_BREAKAWAY_DISP:-0.045}"
export CCDA_BREAKAWAY_BEAD_RATIO="${CCDA_BREAKAWAY_BEAD_RATIO:-0.45}"
export CCDA_ORACLE_BREAKAWAY_PULL_DIST="${CCDA_ORACLE_BREAKAWAY_PULL_DIST:-0.36}"
export CCDA_SETTLE_SECONDS="${CCDA_SETTLE_SECONDS:-0.02}"
export CCDA_AUDIT_T_LIM="${CCDA_AUDIT_T_LIM:-15.0}"

cd "$ROOT"

echo "[Phase2.5c] root=$ROOT"
echo "[Phase2.5c] deformable-ravens=$DEFRAVENS_ROOT"
echo "[Phase2.5c] NUM_SEEDS=$NUM_SEEDS"
echo "[Phase2.5c] SEED_START=$SEED_START"
echo "[Phase2.5c] MAX_STEPS=$MAX_STEPS"
echo "[Phase2.5c] RANDOM_TRIALS=$RANDOM_TRIALS"
echo "[Phase2.5c] WORKERS=$WORKERS"
echo "[Phase2.5c] CONDITIONS=$CONDITIONS"
echo "[Phase2.5c] POLICIES=$POLICIES"
echo "[Phase2.5c] CCDA_BREAKAWAY_FORCE=$CCDA_BREAKAWAY_FORCE"
echo "[Phase2.5c] CCDA_BREAKAWAY_DISP=$CCDA_BREAKAWAY_DISP"
echo "[Phase2.5c] CCDA_BREAKAWAY_BEAD_RATIO=$CCDA_BREAKAWAY_BEAD_RATIO"
echo "[Phase2.5c] CCDA_ORACLE_BREAKAWAY_PULL_DIST=$CCDA_ORACLE_BREAKAWAY_PULL_DIST"

mkdir -p "$OUT_ROOT" "$ROOT/reports"

cat > "$ROOT/reports/phase2_5c_runtime_config.md" <<EOF
# Phase2.5c Runtime Config

- Timestamp: \`$(date -Is)\`
- NUM_SEEDS: \`$NUM_SEEDS\`
- SEED_START: \`$SEED_START\`
- MAX_STEPS: \`$MAX_STEPS\`
- RANDOM_TRIALS: \`$RANDOM_TRIALS\`
- WORKERS: \`$WORKERS\`
- CONDITIONS: \`$CONDITIONS\`
- POLICIES: \`$POLICIES\`

## Selected Env

- CCDA_BREAKAWAY_FORCE: \`$CCDA_BREAKAWAY_FORCE\`
- CCDA_BREAKAWAY_DISP: \`$CCDA_BREAKAWAY_DISP\`
- CCDA_BREAKAWAY_BEAD_RATIO: \`$CCDA_BREAKAWAY_BEAD_RATIO\`
- CCDA_ORACLE_BREAKAWAY_PULL_DIST: \`$CCDA_ORACLE_BREAKAWAY_PULL_DIST\`
EOF

cd "$DEFRAVENS_ROOT"

python ccda_recoverability_audit.py \
  --task hidden-contact-cable-line \
  --conditions $CONDITIONS \
  --policies $POLICIES \
  --num_seeds "$NUM_SEEDS" \
  --seed_start "$SEED_START" \
  --output_root "$OUT_ROOT" \
  --max_steps "$MAX_STEPS" \
  --random_trials "$RANDOM_TRIALS" \
  --workers "$WORKERS" \
  --fresh

cd "$ROOT"

python scripts/phase2_5c_confirm_recoverable_config.py \
  --root "$ROOT" \
  --trials_csv "$OUT_ROOT/recoverability_trials.csv" \
  --summary_json "$OUT_ROOT/recoverability_summary.json" \
  --out_json "$ROOT/reports/phase2_5c_confirmation_summary.json" \
  --out_md "$ROOT/reports/phase2_5c_confirmation_report.md" \
  --out_plan "$ROOT/reports/phase3_condition_plan_after_phase2_5c.md"

echo "[Phase2.5c] done."
echo "[Phase2.5c] report: reports/phase2_5c_confirmation_report.md"
