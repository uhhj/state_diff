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

echo "[Phase3.11] root=$ROOT"
echo "[Phase3.11] python=$(which python)"
echo "[Phase3.11] conda_env=${CONDA_DEFAULT_ENV:-}"
echo "[Phase3.11] head=$(git rev-parse HEAD)"

python scripts/phase3_11_future_source_swap_preflight.py \
  --root "$ROOT" \
  --windows data/phase3_state_diff_windows/phase3_windows.npz \
  --action_template data/phase3_state_diff_windows/phase3_action_template.pkl \
  --old_checkpoint_root checkpoints/phase3 \
  --phase39_train_summary reports/phase3_9_geometry_idm_train_summary.json \
  --phase39b_summary reports/phase3_9b_ablation_summary.json \
  --phase39b_raw reports/phase3_9b_ablation_raw_summary.json \
  --phase310b_summary reports/phase3_10b_future_rollout_audit_summary.json \
  --best_ablation "${PHASE3_11_BEST_ABLATION:-xy_only_high_weight}" \
  --out_json reports/phase3_11_future_source_swap_preflight_summary.json \
  --out_md reports/phase3_11_future_source_swap_preflight_report.md

for gate in \
  PHASE3_ALLOW_FUTURE_SOURCE_SWAP \
  PHASE3_FUTURE_SOURCE_SWAP_CONFIRMED
do
  if [[ "${!gate:-0}" != "1" ]]; then
    echo "[Phase3.11][BLOCKED] $gate must be 1"
    exit 1
  fi
done

cat > reports/phase3_11_future_source_swap_runtime_config.md <<EOF
# Phase3.11 Future-Source Swap Runtime Config

- Timestamp: \`$(date -Is)\`
- Python: \`$(which python)\`
- Conda env: \`${CONDA_DEFAULT_ENV:-}\`
- Main HEAD: \`$(git rev-parse HEAD)\`
- IDM policies: \`${PHASE3_11_IDM_POLICIES:-phase39b_xy_only_high_weight}\`
- Future sources: \`${PHASE3_11_FUTURE_SOURCES:-ddpm_mean ddpm_best_of_k_by_train_nn global_input_retrieval condition_matched_retrieval}\`
- Conditions: \`${PHASE3_11_CONDITIONS:-free hidden_breakaway_pin hidden_high_friction}\`
- Episodes per condition: \`${PHASE3_11_EPISODES_PER_CONDITION:-3}\`
- Max steps: \`${PHASE3_11_MAX_STEPS:-16}\`
- Samples per step: \`${PHASE3_11_SAMPLES_PER_STEP:-32}\`
- Motion timeout: \`${PHASE3_11_MOTION_TIMEOUT:-15.0}\`
- Action clip std: \`${PHASE3_11_ACTION_CLIP_STD:-3.0}\`
- Row timeout sec: \`${PHASE3_11_ROW_TIMEOUT_SEC:-900}\`
- Total timeout sec: \`${PHASE3_11_TOTAL_TIMEOUT_SEC:-14400}\`
- Max rows: \`${PHASE3_11_MAX_ROWS:-36}\`
- Best ablation: \`${PHASE3_11_BEST_ABLATION:-xy_only_high_weight}\`

Scope:
- future-source swap diagnostic only
- no model training
- no future DDPM training
- no Phase4
- no CPS
- no medium/full rollout
- condition_matched_retrieval is diagnostic upper bound only
EOF

rm -rf reports/phase3_11_workers
mkdir -p reports/phase3_11_workers

set +e
python scripts/phase3_11_future_source_swap_rollout.py \
  --root "$ROOT" \
  --windows data/phase3_state_diff_windows/phase3_windows.npz \
  --action_template data/phase3_state_diff_windows/phase3_action_template.pkl \
  --old_checkpoint_root checkpoints/phase3 \
  --phase39_train_summary reports/phase3_9_geometry_idm_train_summary.json \
  --phase39b_raw reports/phase3_9b_ablation_raw_summary.json \
  --best_ablation "${PHASE3_11_BEST_ABLATION:-xy_only_high_weight}" \
  --idm_policies ${PHASE3_11_IDM_POLICIES:-phase39b_xy_only_high_weight} \
  --future_sources ${PHASE3_11_FUTURE_SOURCES:-ddpm_mean ddpm_best_of_k_by_train_nn global_input_retrieval condition_matched_retrieval} \
  --conditions ${PHASE3_11_CONDITIONS:-free hidden_breakaway_pin hidden_high_friction} \
  --episodes_per_condition "${PHASE3_11_EPISODES_PER_CONDITION:-3}" \
  --seed_start "${PHASE3_11_SEED_START:-311000}" \
  --max_steps "${PHASE3_11_MAX_STEPS:-16}" \
  --samples_per_step "${PHASE3_11_SAMPLES_PER_STEP:-32}" \
  --motion_timeout "${PHASE3_11_MOTION_TIMEOUT:-15.0}" \
  --action_clip_std "${PHASE3_11_ACTION_CLIP_STD:-3.0}" \
  --row_timeout_sec "${PHASE3_11_ROW_TIMEOUT_SEC:-900}" \
  --total_timeout_sec "${PHASE3_11_TOTAL_TIMEOUT_SEC:-14400}" \
  --max_rows "${PHASE3_11_MAX_ROWS:-36}" \
  --out_episode_csv reports/phase3_11_future_source_swap_episodes.csv \
  --out_step_csv reports/phase3_11_future_source_swap_steps.csv \
  --out_json reports/phase3_11_future_source_swap_raw_summary.json \
  --progress_json reports/phase3_11_future_source_swap_progress.json \
  --worker_dir reports/phase3_11_workers
SWAP_RC=$?
set -e

set +e
python scripts/phase3_11_analyze_future_source_swap.py \
  --root "$ROOT" \
  --episode_csv reports/phase3_11_future_source_swap_episodes.csv \
  --step_csv reports/phase3_11_future_source_swap_steps.csv \
  --progress_json reports/phase3_11_future_source_swap_progress.json \
  --raw_summary reports/phase3_11_future_source_swap_raw_summary.json \
  --out_json reports/phase3_11_future_source_swap_summary.json \
  --out_md reports/phase3_11_future_source_swap_report.md \
  --idm_policy "${PHASE3_11_ANALYZE_IDM_POLICY:-phase39b_xy_only_high_weight}" \
  --primary_condition hidden_breakaway_pin \
  --improve_threshold "${PHASE3_11_IMPROVE_THRESHOLD:-0.05}" \
  --min_ok_rows_per_source_condition "${PHASE3_11_MIN_OK_ROWS_PER_SOURCE_CONDITION:-2}"
ANALYZE_RC=$?
set -e

cat > reports/phase3_11_no_phase4_confirmation.md <<EOF
# Phase3.11 No Phase4 / No CPS Confirmation

- Timestamp: \`$(date -Is)\`
- Phase3.11 ran future-source swap diagnostics only.
- No model training was run.
- No future DDPM was trained.
- No Phase4 was run.
- No CPS was run.
- No medium/full rollout was run.
- condition_matched_retrieval used condition labels only as diagnostic upper bound, not deployable policy input.
- No paper-level claim should be made from this diagnostic alone.
EOF

if [[ "$SWAP_RC" -ne 0 || "$ANALYZE_RC" -ne 0 ]]; then
  exit 1
fi

echo "[Phase3.11] done"
