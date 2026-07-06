#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p reports

export PYTHONPATH="$ROOT/external/deformable-ravens:${PYTHONPATH:-}"

export PHASE3_CONDITIONS="${PHASE3_CONDITIONS:-free hidden_pin hidden_high_friction hidden_breakaway_pin}"
export PHASE3_PRIMARY_HIDDEN_CONDITION="${PHASE3_PRIMARY_HIDDEN_CONDITION:-hidden_breakaway_pin}"
export PHASE3_DIAGNOSTIC_HIDDEN_CONDITION="${PHASE3_DIAGNOSTIC_HIDDEN_CONDITION:-hidden_pin}"

export CCDA_BREAKAWAY_FORCE="${CCDA_BREAKAWAY_FORCE:-2.6}"
export CCDA_BREAKAWAY_DISP="${CCDA_BREAKAWAY_DISP:-0.045}"
export CCDA_BREAKAWAY_BEAD_RATIO="${CCDA_BREAKAWAY_BEAD_RATIO:-0.45}"
export CCDA_ORACLE_BREAKAWAY_PULL_DIST="${CCDA_ORACLE_BREAKAWAY_PULL_DIST:-0.36}"

echo "[Phase3.5] root=$ROOT"
echo "[Phase3.5] python=$(which python)"
echo "[Phase3.5] conda_env=${CONDA_DEFAULT_ENV:-}"
echo "[Phase3.5] head=$(git rev-parse HEAD)"
echo "[Phase3.5] conditions=$PHASE3_CONDITIONS"

read -r -a CONDITIONS_ARR <<< "$PHASE3_CONDITIONS"
read -r -a BASELINES_ARR <<< "${PHASE3_5_BASELINES:-paper_state state_action}"
read -r -a MOTION_TIMEOUTS_ARR <<< "${PHASE3_5_MOTION_TIMEOUTS:-5.0 15.0}"

python scripts/phase3_5_action_execution_preflight.py \
  --root "$ROOT" \
  --windows data/phase3_state_diff_windows/phase3_windows.npz \
  --action_template data/phase3_state_diff_windows/phase3_action_template.pkl \
  --checkpoint_root checkpoints/phase3 \
  --out_json reports/phase3_5_action_execution_preflight_summary.json \
  --out_md reports/phase3_5_action_execution_preflight_report.md

python scripts/phase3_5_action_codec_inspect.py \
  --root "$ROOT" \
  --windows data/phase3_state_diff_windows/phase3_windows.npz \
  --action_template data/phase3_state_diff_windows/phase3_action_template.pkl \
  --samples_per_condition "${PHASE3_5_CODEC_SAMPLES_PER_CONDITION:-8}" \
  --out_csv reports/phase3_5_action_codec_decode_samples.csv \
  --out_json reports/phase3_5_action_codec_inspect_summary.json \
  --out_md reports/phase3_5_action_codec_inspect_report.md

if [[ "${PHASE3_ALLOW_ACTION_DIAGNOSTIC:-0}" != "1" ]]; then
  echo "[Phase3.5][BLOCKED] Preflight/codec checks completed. Action diagnostic not run because PHASE3_ALLOW_ACTION_DIAGNOSTIC != 1."
  exit 1
fi

if [[ "${PHASE3_ACTION_DIAGNOSTIC_CONFIRMED:-0}" != "1" ]]; then
  echo "[Phase3.5][BLOCKED] Preflight/codec checks completed. Action diagnostic not run because PHASE3_ACTION_DIAGNOSTIC_CONFIRMED != 1."
  exit 1
fi

cat > reports/phase3_5_action_execution_runtime_config.md <<EOF
# Phase3.5 Action Execution Runtime Config

- Timestamp: \`$(date -Is)\`
- Python: \`$(which python)\`
- Conda env: \`${CONDA_DEFAULT_ENV:-}\`
- Main HEAD: \`$(git rev-parse HEAD)\`
- Conditions: \`$PHASE3_CONDITIONS\`
- Primary hidden condition: \`$PHASE3_PRIMARY_HIDDEN_CONDITION\`
- Diagnostic hidden condition: \`$PHASE3_DIAGNOSTIC_HIDDEN_CONDITION\`
- Motion timeouts: \`${PHASE3_5_MOTION_TIMEOUTS:-5.0 15.0}\`
- Seeds per condition: \`${PHASE3_5_SEEDS_PER_CONDITION:-2}\`
- Oracle steps: \`${PHASE3_5_ORACLE_STEPS:-8}\`
- Learned steps: \`${PHASE3_5_LEARNED_STEPS:-4}\`
- Baselines: \`${PHASE3_5_BASELINES:-paper_state state_action}\`

Scope:
- action execution diagnostic only
- no Phase4
- no CPS
- no paper-level claim
EOF

python scripts/phase3_5_action_execution_diag.py \
  --root "$ROOT" \
  --windows data/phase3_state_diff_windows/phase3_windows.npz \
  --action_template data/phase3_state_diff_windows/phase3_action_template.pkl \
  --conditions "${CONDITIONS_ARR[@]}" \
  --baselines "${BASELINES_ARR[@]}" \
  --seed_start "${PHASE3_5_SEED_START:-310000}" \
  --seeds_per_condition "${PHASE3_5_SEEDS_PER_CONDITION:-2}" \
  --motion_timeouts "${MOTION_TIMEOUTS_ARR[@]}" \
  --oracle_steps "${PHASE3_5_ORACLE_STEPS:-8}" \
  --learned_steps "${PHASE3_5_LEARNED_STEPS:-4}" \
  --samples_per_step "${PHASE3_5_SAMPLES_PER_STEP:-16}" \
  --action_clip_std "${PHASE3_5_ACTION_CLIP_STD:-3.0}" \
  --gt_samples_per_condition "${PHASE3_5_GT_SAMPLES_PER_CONDITION:-2}" \
  --out_csv reports/phase3_5_action_execution_trials.csv \
  --out_json reports/phase3_5_action_execution_raw_summary.json

python scripts/phase3_5_analyze_action_diagnostics.py \
  --root "$ROOT" \
  --trials_csv reports/phase3_5_action_execution_trials.csv \
  --out_json reports/phase3_5_action_execution_diagnostic_summary.json \
  --out_md reports/phase3_5_action_execution_diagnostic_report.md

cat > reports/phase3_5_no_phase4_confirmation.md <<EOF
# Phase3.5 No Phase4 / No CPS Confirmation

- Timestamp: \`$(date -Is)\`
- Phase3.5 ran action execution diagnostics only.
- No Phase4 was run.
- No CPS was run.
- No paper-level claim should be made from this diagnostic alone.
EOF

echo "[Phase3.5] done"
