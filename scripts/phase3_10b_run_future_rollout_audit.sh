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

echo "[Phase3.10b] root=$ROOT"
echo "[Phase3.10b] python=$(which python)"
echo "[Phase3.10b] conda_env=${CONDA_DEFAULT_ENV:-}"
echo "[Phase3.10b] head=$(git rev-parse HEAD)"

python scripts/phase3_10b_future_rollout_audit_preflight.py \
  --root "$ROOT" \
  --windows data/phase3_state_diff_windows/phase3_windows.npz \
  --action_template data/phase3_state_diff_windows/phase3_action_template.pkl \
  --old_checkpoint_root checkpoints/phase3 \
  --phase39_train_summary reports/phase3_9_geometry_idm_train_summary.json \
  --phase39b_summary reports/phase3_9b_ablation_summary.json \
  --phase39b_raw reports/phase3_9b_ablation_raw_summary.json \
  --phase310_summary reports/phase3_10_controlled_learned_rollout_summary.json \
  --best_ablation "${PHASE3_10B_BEST_ABLATION:-xy_only_high_weight}" \
  --out_json reports/phase3_10b_future_rollout_audit_preflight_summary.json \
  --out_md reports/phase3_10b_future_rollout_audit_preflight_report.md

for gate in \
  PHASE3_ALLOW_FUTURE_ROLLOUT_AUDIT \
  PHASE3_FUTURE_ROLLOUT_AUDIT_CONFIRMED
do
  if [[ "${!gate:-0}" != "1" ]]; then
    echo "[Phase3.10b][BLOCKED] $gate must be 1"
    exit 1
  fi
done

cat > reports/phase3_10b_future_rollout_audit_runtime_config.md <<EOF
# Phase3.10b Future Rollout Audit Runtime Config

- Timestamp: \`$(date -Is)\`
- Python: \`$(which python)\`
- Conda env: \`${CONDA_DEFAULT_ENV:-}\`
- Main HEAD: \`$(git rev-parse HEAD)\`
- Policies: \`${PHASE3_10B_POLICIES:-old_state_action phase39_default_geometry phase39b_xy_only_high_weight}\`
- Conditions: \`${PHASE3_10B_CONDITIONS:-free hidden_breakaway_pin hidden_high_friction}\`
- Episodes per condition: \`${PHASE3_10B_EPISODES_PER_CONDITION:-4}\`
- Max steps: \`${PHASE3_10B_MAX_STEPS:-16}\`
- Samples per step: \`${PHASE3_10B_SAMPLES_PER_STEP:-16}\`
- Motion timeout: \`${PHASE3_10B_MOTION_TIMEOUT:-15.0}\`
- Action clip std: \`${PHASE3_10B_ACTION_CLIP_STD:-3.0}\`
- Row timeout sec: \`${PHASE3_10B_ROW_TIMEOUT_SEC:-900}\`
- Total timeout sec: \`${PHASE3_10B_TOTAL_TIMEOUT_SEC:-14400}\`
- Max rows: \`${PHASE3_10B_MAX_ROWS:-36}\`
- Best ablation: \`${PHASE3_10B_BEST_ABLATION:-xy_only_high_weight}\`

Scope:
- learned future quality + rollout error audit only
- no model training
- no future DDPM training
- no Phase4
- no CPS
- no medium/full rollout
EOF

rm -rf reports/phase3_10b_workers
mkdir -p reports/phase3_10b_workers

set +e
python scripts/phase3_10b_trace_rollout.py \
  --root "$ROOT" \
  --windows data/phase3_state_diff_windows/phase3_windows.npz \
  --action_template data/phase3_state_diff_windows/phase3_action_template.pkl \
  --old_checkpoint_root checkpoints/phase3 \
  --phase39_train_summary reports/phase3_9_geometry_idm_train_summary.json \
  --phase39b_raw reports/phase3_9b_ablation_raw_summary.json \
  --best_ablation "${PHASE3_10B_BEST_ABLATION:-xy_only_high_weight}" \
  --policies ${PHASE3_10B_POLICIES:-old_state_action phase39_default_geometry phase39b_xy_only_high_weight} \
  --conditions ${PHASE3_10B_CONDITIONS:-free hidden_breakaway_pin hidden_high_friction} \
  --episodes_per_condition "${PHASE3_10B_EPISODES_PER_CONDITION:-4}" \
  --seed_start "${PHASE3_10B_SEED_START:-310000}" \
  --max_steps "${PHASE3_10B_MAX_STEPS:-16}" \
  --samples_per_step "${PHASE3_10B_SAMPLES_PER_STEP:-16}" \
  --motion_timeout "${PHASE3_10B_MOTION_TIMEOUT:-15.0}" \
  --action_clip_std "${PHASE3_10B_ACTION_CLIP_STD:-3.0}" \
  --row_timeout_sec "${PHASE3_10B_ROW_TIMEOUT_SEC:-900}" \
  --total_timeout_sec "${PHASE3_10B_TOTAL_TIMEOUT_SEC:-14400}" \
  --max_rows "${PHASE3_10B_MAX_ROWS:-36}" \
  --out_episode_csv reports/phase3_10b_future_rollout_audit_episodes.csv \
  --out_step_csv reports/phase3_10b_future_rollout_audit_steps.csv \
  --out_json reports/phase3_10b_future_rollout_audit_raw_summary.json \
  --progress_json reports/phase3_10b_future_rollout_audit_progress.json \
  --worker_dir reports/phase3_10b_workers
AUDIT_RC=$?
set -e

set +e
python scripts/phase3_10b_analyze_trace.py \
  --root "$ROOT" \
  --episode_csv reports/phase3_10b_future_rollout_audit_episodes.csv \
  --step_csv reports/phase3_10b_future_rollout_audit_steps.csv \
  --progress_json reports/phase3_10b_future_rollout_audit_progress.json \
  --raw_summary reports/phase3_10b_future_rollout_audit_raw_summary.json \
  --out_json reports/phase3_10b_future_rollout_audit_summary.json \
  --out_md reports/phase3_10b_future_rollout_audit_report.md \
  --min_ok_rows_per_policy_condition "${PHASE3_10B_MIN_OK_ROWS_PER_POLICY_CONDITION:-2}" \
  --future_match_low_threshold "${PHASE3_10B_FUTURE_MATCH_LOW_THRESHOLD:-0.50}" \
  --pull_gap_threshold "${PHASE3_10B_PULL_GAP_THRESHOLD:-0.05}" \
  --rollout_improve_threshold "${PHASE3_10B_ROLLOUT_IMPROVE_THRESHOLD:-0.05}"
ANALYZE_RC=$?
set -e

cat > reports/phase3_10b_no_phase4_confirmation.md <<EOF
# Phase3.10b No Phase4 / No CPS Confirmation

- Timestamp: \`$(date -Is)\`
- Phase3.10b ran learned future quality + rollout error audit only.
- No model training was run.
- No future DDPM was trained.
- No Phase4 was run.
- No CPS was run.
- No medium/full rollout was run.
- Offline y_state / condition_name were used only for diagnostic nearest-neighbor audit, not model input.
- No paper-level claim should be made from this diagnostic alone.
EOF

if [[ "$AUDIT_RC" -ne 0 || "$ANALYZE_RC" -ne 0 ]]; then
  exit 1
fi

echo "[Phase3.10b] done"