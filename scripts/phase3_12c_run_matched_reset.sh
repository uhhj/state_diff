#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p reports

export PYTHONPATH="$ROOT:$ROOT/scripts:$ROOT/external/deformable-ravens:${PYTHONPATH:-}"

export PHASE3_CONDITIONS="${PHASE3_CONDITIONS:-free hidden_pin hidden_high_friction hidden_breakaway_pin}"
export PHASE3_PRIMARY_HIDDEN_CONDITION="${PHASE3_PRIMARY_HIDDEN_CONDITION:-hidden_breakaway_pin}"
export PHASE3_DIAGNOSTIC_HIDDEN_CONDITION="${PHASE3_DIAGNOSTIC_HIDDEN_CONDITION:-hidden_pin}"

export CCDA_BREAKAWAY_FORCE="${CCDA_BREAKAWAY_FORCE:-2.6}"
export CCDA_BREAKAWAY_DISP="${CCDA_BREAKAWAY_DISP:-0.045}"
export CCDA_BREAKAWAY_BEAD_RATIO="${CCDA_BREAKAWAY_BEAD_RATIO:-0.45}"
export CCDA_ORACLE_BREAKAWAY_PULL_DIST="${CCDA_ORACLE_BREAKAWAY_PULL_DIST:-0.36}"

SELECTORS="${PHASE3_12C_SELECTORS:-ddpm_mean condition_nearest_upper proxy_state_motion_nn proxy_combined_topk_action_geom}"
CONDITIONS="${PHASE3_12C_CONDITIONS:-free hidden_breakaway_pin hidden_high_friction}"
VISIBLE_SEEDS="${PHASE3_12C_VISIBLE_SEEDS:-312000 312001 312002 312003 312500 312501 312502 312503}"

write_no_phase4_confirmation() {
  cat > reports/phase3_12c_no_phase4_confirmation.md <<EOF
# Phase3.12c No Phase4 / No CPS Confirmation

- Timestamp: \`$(date -Is)\`
- Phase3.12c performed matched-reset integrity and paired selector diagnostics only.
- No model training was run.
- No future DDPM was trained.
- No Phase4 was run.
- No CPS was run.
- No medium/full rollout was run.
- condition_nearest_upper used condition labels only as a diagnostic upper bound.
- Main comparisons use paired delta_fraction under matched condition + visible_seed.
EOF
}

echo "[Phase3.12c] root=$ROOT"
echo "[Phase3.12c] python=$(which python)"
echo "[Phase3.12c] conda_env=${CONDA_DEFAULT_ENV:-}"
echo "[Phase3.12c] head=$(git rev-parse HEAD)"
echo "[Phase3.12c] selectors=$SELECTORS"
echo "[Phase3.12c] conditions=$CONDITIONS"
echo "[Phase3.12c] seeds=$VISIBLE_SEEDS"

cat > reports/phase3_12c_runtime_config.md <<EOF
# Phase3.12c Matched-Reset Runtime Config

- Timestamp: \`$(date -Is)\`
- Python: \`$(which python)\`
- Conda env: \`${CONDA_DEFAULT_ENV:-}\`
- Main HEAD: \`$(git rev-parse HEAD)\`
- Submodule: \`$(git submodule status external/deformable-ravens)\`
- Selectors: \`$SELECTORS\`
- Conditions: \`$CONDITIONS\`
- Visible seeds: \`$VISIBLE_SEEDS\`
- Baseline selector: \`ddpm_mean\`
- Primary condition: \`hidden_breakaway_pin\`
- Expected reset rows: \`96\`
- Expected rollout rows: \`96\`
- Maximum step rows: \`1536\`
- Max rollout steps: \`${PHASE3_12C_MAX_STEPS:-16}\`
- DDPM samples per step: \`${PHASE3_12C_SAMPLES_PER_STEP:-32}\`
- Top-K: \`${PHASE3_12C_TOP_K:-64}\`

Deterministic reset:
- minimum settle steps: \`${PHASE3_12C_MIN_SETTLE_STEPS:-540}\`
- maximum settle steps: \`${PHASE3_12C_MAX_SETTLE_STEPS:-2400}\`
- static checks required: \`${PHASE3_12C_STATIC_CHECKS_REQUIRED:-8}\`
- static check interval: \`${PHASE3_12C_STATIC_CHECK_INTERVAL:-10}\`
- rounded state decimals: \`${PHASE3_12C_INITIAL_ROUND_DECIMALS:-7}\`
- wall-clock settling disabled: \`True\`
- explicit Python/NumPy/Torch seeding: \`True\`
- selector-independent pair group: \`True\`
- condition-independent pair group: \`True\`

Matched-reset thresholds:
- initial state max abs: \`${PHASE3_12C_MAX_ABS_THRESHOLD:-1e-6}\`
- initial state MAE: \`${PHASE3_12C_MAE_THRESHOLD:-1e-7}\`
- initial fraction diff: \`${PHASE3_12C_FRACTION_THRESHOLD:-1e-9}\`
- initial curve diff: \`${PHASE3_12C_CURVE_THRESHOLD:-1e-7}\`

Paired improvement criteria:
- mean paired gain threshold: \`${PHASE3_12C_IMPROVE_THRESHOLD:-0.05}\`
- minimum pairs: \`${PHASE3_12C_MIN_PAIRS:-6}\`
- minimum positive pair fraction: \`${PHASE3_12C_MIN_POSITIVE_FRACTION:-0.75}\`
- bootstrap samples: \`${PHASE3_12C_BOOTSTRAP_SAMPLES:-10000}\`
- bootstrap CI lower bound must be above zero: \`True\`

Scope:
- no model training
- no future DDPM training
- no Phase4
- no CPS
- no medium/full rollout
EOF

python scripts/phase3_12c_preflight.py \
  --root "$ROOT" \
  --windows data/phase3_state_diff_windows/phase3_windows.npz \
  --action_template data/phase3_state_diff_windows/phase3_action_template.pkl \
  --old_checkpoint_root checkpoints/phase3 \
  --phase39b_raw reports/phase3_9b_ablation_raw_summary.json \
  --phase312b_summary reports/phase3_12b_proxy_score_summary.json \
  --best_ablation "${PHASE3_12C_BEST_ABLATION:-xy_only_high_weight}" \
  --out_json reports/phase3_12c_preflight_summary.json \
  --out_md reports/phase3_12c_preflight_report.md

for gate in \
  PHASE3_ALLOW_MATCHED_RESET_AUDIT \
  PHASE3_MATCHED_RESET_AUDIT_CONFIRMED
do
  if [[ "${!gate:-0}" != "1" ]]; then
    echo "[Phase3.12c][BLOCKED] $gate must be 1"
    write_no_phase4_confirmation
    exit 1
  fi
done

rm -rf reports/phase3_12c_reset_workers
mkdir -p reports/phase3_12c_reset_workers

echo "[Phase3.12c] Stage A: matched-reset audit"

set +e
python scripts/phase3_12c_matched_reset_experiment.py \
  --root "$ROOT" \
  --stage reset \
  --windows data/phase3_state_diff_windows/phase3_windows.npz \
  --action_template data/phase3_state_diff_windows/phase3_action_template.pkl \
  --old_checkpoint_root checkpoints/phase3 \
  --phase39b_raw reports/phase3_9b_ablation_raw_summary.json \
  --best_ablation "${PHASE3_12C_BEST_ABLATION:-xy_only_high_weight}" \
  --selectors $SELECTORS \
  --conditions $CONDITIONS \
  --visible_seeds $VISIBLE_SEEDS \
  --min_settle_steps "${PHASE3_12C_MIN_SETTLE_STEPS:-540}" \
  --max_settle_steps "${PHASE3_12C_MAX_SETTLE_STEPS:-2400}" \
  --static_checks_required "${PHASE3_12C_STATIC_CHECKS_REQUIRED:-8}" \
  --static_check_interval "${PHASE3_12C_STATIC_CHECK_INTERVAL:-10}" \
  --initial_round_decimals "${PHASE3_12C_INITIAL_ROUND_DECIMALS:-7}" \
  --row_timeout_sec "${PHASE3_12C_RESET_ROW_TIMEOUT_SEC:-300}" \
  --total_timeout_sec "${PHASE3_12C_RESET_TOTAL_TIMEOUT_SEC:-21600}" \
  --max_rows "${PHASE3_12C_MAX_ROWS:-96}" \
  --out_csv reports/phase3_12c_reset_integrity_rows.csv \
  --out_step_csv reports/phase3_12c_reset_unused_steps.csv \
  --out_json reports/phase3_12c_reset_integrity_raw.json \
  --progress_json reports/phase3_12c_reset_integrity_progress.json \
  --worker_dir reports/phase3_12c_reset_workers
RESET_RC=$?
set -e

set +e
python scripts/phase3_12c_analyze.py \
  --root "$ROOT" \
  --stage reset \
  --input_csv reports/phase3_12c_reset_integrity_rows.csv \
  --progress_json reports/phase3_12c_reset_integrity_progress.json \
  --raw_json reports/phase3_12c_reset_integrity_raw.json \
  --selectors $SELECTORS \
  --conditions $CONDITIONS \
  --visible_seeds $VISIBLE_SEEDS \
  --baseline_selector ddpm_mean \
  --max_abs_threshold "${PHASE3_12C_MAX_ABS_THRESHOLD:-1e-6}" \
  --mae_threshold "${PHASE3_12C_MAE_THRESHOLD:-1e-7}" \
  --fraction_threshold "${PHASE3_12C_FRACTION_THRESHOLD:-1e-9}" \
  --curve_threshold "${PHASE3_12C_CURVE_THRESHOLD:-1e-7}" \
  --out_json reports/phase3_12c_reset_integrity_summary.json \
  --out_md reports/phase3_12c_reset_integrity_report.md \
  --pairs_csv reports/phase3_12c_reset_integrity_pairs.csv
RESET_ANALYZE_RC=$?
set -e

if [[ "$RESET_RC" -ne 0 || "$RESET_ANALYZE_RC" -ne 0 ]]; then
  echo "[Phase3.12c][FAIL] matched-reset integrity did not pass"
  write_no_phase4_confirmation
  exit 1
fi

for gate in \
  PHASE3_ALLOW_PAIRED_SELECTOR_REEVAL \
  PHASE3_PAIRED_SELECTOR_REEVAL_CONFIRMED
do
  if [[ "${!gate:-0}" != "1" ]]; then
    echo "[Phase3.12c][BLOCKED] $gate must be 1"
    write_no_phase4_confirmation
    exit 1
  fi
done

rm -rf reports/phase3_12c_rollout_workers
mkdir -p reports/phase3_12c_rollout_workers

echo "[Phase3.12c] Stage B: paired selector re-evaluation"

set +e
python scripts/phase3_12c_matched_reset_experiment.py \
  --root "$ROOT" \
  --stage rollout \
  --windows data/phase3_state_diff_windows/phase3_windows.npz \
  --action_template data/phase3_state_diff_windows/phase3_action_template.pkl \
  --old_checkpoint_root checkpoints/phase3 \
  --phase39b_raw reports/phase3_9b_ablation_raw_summary.json \
  --best_ablation "${PHASE3_12C_BEST_ABLATION:-xy_only_high_weight}" \
  --selectors $SELECTORS \
  --conditions $CONDITIONS \
  --visible_seeds $VISIBLE_SEEDS \
  --max_steps "${PHASE3_12C_MAX_STEPS:-16}" \
  --samples_per_step "${PHASE3_12C_SAMPLES_PER_STEP:-32}" \
  --top_k "${PHASE3_12C_TOP_K:-64}" \
  --motion_timeout "${PHASE3_12C_MOTION_TIMEOUT:-15.0}" \
  --action_clip_std "${PHASE3_12C_ACTION_CLIP_STD:-3.0}" \
  --score_proxy_weight "${PHASE3_12C_SCORE_PROXY_WEIGHT:-1.0}" \
  --score_ood_weight "${PHASE3_12C_SCORE_OOD_WEIGHT:-0.15}" \
  --score_clip_weight "${PHASE3_12C_SCORE_CLIP_WEIGHT:-2.0}" \
  --score_action_mae_weight "${PHASE3_12C_SCORE_ACTION_MAE_WEIGHT:-4.0}" \
  --score_pull_diff_weight "${PHASE3_12C_SCORE_PULL_DIFF_WEIGHT:-2.0}" \
  --score_small_pull_weight "${PHASE3_12C_SCORE_SMALL_PULL_WEIGHT:-2.0}" \
  --min_pull "${PHASE3_12C_MIN_PULL:-0.08}" \
  --min_settle_steps "${PHASE3_12C_MIN_SETTLE_STEPS:-540}" \
  --max_settle_steps "${PHASE3_12C_MAX_SETTLE_STEPS:-2400}" \
  --static_checks_required "${PHASE3_12C_STATIC_CHECKS_REQUIRED:-8}" \
  --static_check_interval "${PHASE3_12C_STATIC_CHECK_INTERVAL:-10}" \
  --initial_round_decimals "${PHASE3_12C_INITIAL_ROUND_DECIMALS:-7}" \
  --row_timeout_sec "${PHASE3_12C_ROLLOUT_ROW_TIMEOUT_SEC:-900}" \
  --total_timeout_sec "${PHASE3_12C_ROLLOUT_TOTAL_TIMEOUT_SEC:-43200}" \
  --max_rows "${PHASE3_12C_MAX_ROWS:-96}" \
  --out_csv reports/phase3_12c_paired_selector_episodes.csv \
  --out_step_csv reports/phase3_12c_paired_selector_steps.csv \
  --out_json reports/phase3_12c_paired_selector_raw.json \
  --progress_json reports/phase3_12c_paired_selector_progress.json \
  --worker_dir reports/phase3_12c_rollout_workers
ROLLOUT_RC=$?
set -e

set +e
python scripts/phase3_12c_analyze.py \
  --root "$ROOT" \
  --stage rollout \
  --input_csv reports/phase3_12c_paired_selector_episodes.csv \
  --step_csv reports/phase3_12c_paired_selector_steps.csv \
  --progress_json reports/phase3_12c_paired_selector_progress.json \
  --raw_json reports/phase3_12c_paired_selector_raw.json \
  --selectors $SELECTORS \
  --conditions $CONDITIONS \
  --visible_seeds $VISIBLE_SEEDS \
  --baseline_selector ddpm_mean \
  --primary_condition hidden_breakaway_pin \
  --max_abs_threshold "${PHASE3_12C_MAX_ABS_THRESHOLD:-1e-6}" \
  --mae_threshold "${PHASE3_12C_MAE_THRESHOLD:-1e-7}" \
  --fraction_threshold "${PHASE3_12C_FRACTION_THRESHOLD:-1e-9}" \
  --curve_threshold "${PHASE3_12C_CURVE_THRESHOLD:-1e-7}" \
  --improve_threshold "${PHASE3_12C_IMPROVE_THRESHOLD:-0.05}" \
  --min_pairs "${PHASE3_12C_MIN_PAIRS:-6}" \
  --min_positive_fraction "${PHASE3_12C_MIN_POSITIVE_FRACTION:-0.75}" \
  --bootstrap_samples "${PHASE3_12C_BOOTSTRAP_SAMPLES:-10000}" \
  --bootstrap_seed "${PHASE3_12C_BOOTSTRAP_SEED:-312012}" \
  --out_json reports/phase3_12c_paired_selector_summary.json \
  --out_md reports/phase3_12c_paired_selector_report.md \
  --pairs_csv reports/phase3_12c_paired_gains.csv \
  --selector_summary_csv reports/phase3_12c_selector_summary.csv \
  --cohort_summary_csv reports/phase3_12c_cohort_summary.csv
ANALYZE_RC=$?
set -e

write_no_phase4_confirmation

if [[ "$ROLLOUT_RC" -ne 0 || "$ANALYZE_RC" -ne 0 ]]; then
  exit 1
fi

echo "[Phase3.12c] completed"
