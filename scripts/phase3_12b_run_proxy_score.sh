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

echo "[Phase3.12b] root=$ROOT"
echo "[Phase3.12b] python=$(which python)"
echo "[Phase3.12b] conda_env=${CONDA_DEFAULT_ENV:-}"
echo "[Phase3.12b] head=$(git rev-parse HEAD)"

python scripts/phase3_12b_proxy_score_preflight.py \
  --root "$ROOT" \
  --windows data/phase3_state_diff_windows/phase3_windows.npz \
  --action_template data/phase3_state_diff_windows/phase3_action_template.pkl \
  --old_checkpoint_root checkpoints/phase3 \
  --phase39b_raw reports/phase3_9b_ablation_raw_summary.json \
  --phase312_summary reports/phase3_12_compat_selection_summary.json \
  --best_ablation "${PHASE3_12B_BEST_ABLATION:-xy_only_high_weight}" \
  --out_json reports/phase3_12b_proxy_score_preflight_summary.json \
  --out_md reports/phase3_12b_proxy_score_preflight_report.md

for gate in \
  PHASE3_ALLOW_PROXY_SCORE_DIAGNOSTIC \
  PHASE3_PROXY_SCORE_DIAGNOSTIC_CONFIRMED
do
  if [[ "${!gate:-0}" != "1" ]]; then
    echo "[Phase3.12b][BLOCKED] $gate must be 1"
    exit 1
  fi
done

cat > reports/phase3_12b_proxy_score_runtime_config.md <<EOF
# Phase3.12b Condition Proxy / Score Ablation Runtime Config

- Timestamp: \`$(date -Is)\`
- Python: \`$(which python)\`
- Conda env: \`${CONDA_DEFAULT_ENV:-}\`
- Main HEAD: \`$(git rev-parse HEAD)\`
- Selectors: \`${PHASE3_12B_SELECTORS:-ddpm_mean condition_nearest_upper proxy_action_nn proxy_state_motion_nn proxy_combined_nn proxy_combined_topk_action_geom proxy_combined_topk_no_action_geom}\`
- Conditions: \`${PHASE3_12B_CONDITIONS:-free hidden_breakaway_pin hidden_high_friction}\`
- Episodes per condition: \`${PHASE3_12B_EPISODES_PER_CONDITION:-4}\`
- Max steps: \`${PHASE3_12B_MAX_STEPS:-16}\`
- Samples per step: \`${PHASE3_12B_SAMPLES_PER_STEP:-32}\`
- Top-K: \`${PHASE3_12B_TOP_K:-64}\`
- Motion timeout: \`${PHASE3_12B_MOTION_TIMEOUT:-15.0}\`
- Action clip std: \`${PHASE3_12B_ACTION_CLIP_STD:-3.0}\`
- Row timeout sec: \`${PHASE3_12B_ROW_TIMEOUT_SEC:-900}\`
- Total timeout sec: \`${PHASE3_12B_TOTAL_TIMEOUT_SEC:-28800}\`
- Max rows: \`${PHASE3_12B_MAX_ROWS:-84}\`
- Best ablation: \`${PHASE3_12B_BEST_ABLATION:-xy_only_high_weight}\`

Score weights:
- proxy: \`${PHASE3_12B_SCORE_PROXY_WEIGHT:-1.0}\`
- action_ood: \`${PHASE3_12B_SCORE_OOD_WEIGHT:-0.15}\`
- clip: \`${PHASE3_12B_SCORE_CLIP_WEIGHT:-2.0}\`
- action_mae: \`${PHASE3_12B_SCORE_ACTION_MAE_WEIGHT:-4.0}\`
- pull_diff: \`${PHASE3_12B_SCORE_PULL_DIFF_WEIGHT:-2.0}\`
- small_pull: \`${PHASE3_12B_SCORE_SMALL_PULL_WEIGHT:-2.0}\`
- min_pull: \`${PHASE3_12B_MIN_PULL:-0.08}\`

Scope:
- observable condition-proxy and score-ablation diagnostic only
- no model training
- no future DDPM training
- no Phase4
- no CPS
- no medium/full rollout
- condition_nearest_upper is diagnostic upper bound only
EOF

rm -rf reports/phase3_12b_workers
mkdir -p reports/phase3_12b_workers

set +e
python scripts/phase3_12b_proxy_score_rollout.py \
  --root "$ROOT" \
  --windows data/phase3_state_diff_windows/phase3_windows.npz \
  --action_template data/phase3_state_diff_windows/phase3_action_template.pkl \
  --old_checkpoint_root checkpoints/phase3 \
  --phase39b_raw reports/phase3_9b_ablation_raw_summary.json \
  --best_ablation "${PHASE3_12B_BEST_ABLATION:-xy_only_high_weight}" \
  --selectors ${PHASE3_12B_SELECTORS:-ddpm_mean condition_nearest_upper proxy_action_nn proxy_state_motion_nn proxy_combined_nn proxy_combined_topk_action_geom proxy_combined_topk_no_action_geom} \
  --conditions ${PHASE3_12B_CONDITIONS:-free hidden_breakaway_pin hidden_high_friction} \
  --episodes_per_condition "${PHASE3_12B_EPISODES_PER_CONDITION:-4}" \
  --seed_start "${PHASE3_12B_SEED_START:-312500}" \
  --max_steps "${PHASE3_12B_MAX_STEPS:-16}" \
  --samples_per_step "${PHASE3_12B_SAMPLES_PER_STEP:-32}" \
  --top_k "${PHASE3_12B_TOP_K:-64}" \
  --motion_timeout "${PHASE3_12B_MOTION_TIMEOUT:-15.0}" \
  --action_clip_std "${PHASE3_12B_ACTION_CLIP_STD:-3.0}" \
  --score_proxy_weight "${PHASE3_12B_SCORE_PROXY_WEIGHT:-1.0}" \
  --score_ood_weight "${PHASE3_12B_SCORE_OOD_WEIGHT:-0.15}" \
  --score_clip_weight "${PHASE3_12B_SCORE_CLIP_WEIGHT:-2.0}" \
  --score_action_mae_weight "${PHASE3_12B_SCORE_ACTION_MAE_WEIGHT:-4.0}" \
  --score_pull_diff_weight "${PHASE3_12B_SCORE_PULL_DIFF_WEIGHT:-2.0}" \
  --score_small_pull_weight "${PHASE3_12B_SCORE_SMALL_PULL_WEIGHT:-2.0}" \
  --min_pull "${PHASE3_12B_MIN_PULL:-0.08}" \
  --row_timeout_sec "${PHASE3_12B_ROW_TIMEOUT_SEC:-900}" \
  --total_timeout_sec "${PHASE3_12B_TOTAL_TIMEOUT_SEC:-28800}" \
  --max_rows "${PHASE3_12B_MAX_ROWS:-84}" \
  --out_episode_csv reports/phase3_12b_proxy_score_episodes.csv \
  --out_step_csv reports/phase3_12b_proxy_score_steps.csv \
  --out_json reports/phase3_12b_proxy_score_raw_summary.json \
  --progress_json reports/phase3_12b_proxy_score_progress.json \
  --worker_dir reports/phase3_12b_workers
ROLLOUT_RC=$?
set -e

set +e
python scripts/phase3_12b_analyze_proxy_score.py \
  --root "$ROOT" \
  --episode_csv reports/phase3_12b_proxy_score_episodes.csv \
  --step_csv reports/phase3_12b_proxy_score_steps.csv \
  --progress_json reports/phase3_12b_proxy_score_progress.json \
  --raw_summary reports/phase3_12b_proxy_score_raw_summary.json \
  --out_json reports/phase3_12b_proxy_score_summary.json \
  --out_md reports/phase3_12b_proxy_score_report.md \
  --primary_condition hidden_breakaway_pin \
  --improve_threshold "${PHASE3_12B_IMPROVE_THRESHOLD:-0.05}" \
  --min_ok_rows_per_selector_condition "${PHASE3_12B_MIN_OK_ROWS_PER_SELECTOR_CONDITION:-2}"
ANALYZE_RC=$?
set -e

cat > reports/phase3_12b_no_phase4_confirmation.md <<EOF
# Phase3.12b No Phase4 / No CPS Confirmation

- Timestamp: \`$(date -Is)\`
- Phase3.12b ran condition-proxy and score-ablation diagnostics only.
- No model training was run.
- No future DDPM was trained.
- No Phase4 was run.
- No CPS was run.
- No medium/full rollout was run.
- condition_name / y_state / y_action were used only for diagnostic target selection and reporting.
- condition_nearest_upper is diagnostic upper bound, not deployable policy input.
- No paper-level claim should be made from this diagnostic alone.
EOF

if [[ "$ROLLOUT_RC" -ne 0 || "$ANALYZE_RC" -ne 0 ]]; then
  exit 1
fi

echo "[Phase3.12b] done"
