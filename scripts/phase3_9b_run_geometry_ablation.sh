#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p reports checkpoints/phase3_9b_geometry_idm

export PYTHONPATH="$ROOT:$ROOT/scripts:$ROOT/external/deformable-ravens:${PYTHONPATH:-}"

export PHASE3_CONDITIONS="${PHASE3_CONDITIONS:-free hidden_pin hidden_high_friction hidden_breakaway_pin}"
export PHASE3_PRIMARY_HIDDEN_CONDITION="${PHASE3_PRIMARY_HIDDEN_CONDITION:-hidden_breakaway_pin}"
export PHASE3_DIAGNOSTIC_HIDDEN_CONDITION="${PHASE3_DIAGNOSTIC_HIDDEN_CONDITION:-hidden_pin}"

export CCDA_BREAKAWAY_FORCE="${CCDA_BREAKAWAY_FORCE:-2.6}"
export CCDA_BREAKAWAY_DISP="${CCDA_BREAKAWAY_DISP:-0.045}"
export CCDA_BREAKAWAY_BEAD_RATIO="${CCDA_BREAKAWAY_BEAD_RATIO:-0.45}"
export CCDA_ORACLE_BREAKAWAY_PULL_DIST="${CCDA_ORACLE_BREAKAWAY_PULL_DIST:-0.36}"

echo "[Phase3.9b] root=$ROOT"
echo "[Phase3.9b] python=$(which python)"
echo "[Phase3.9b] conda_env=${CONDA_DEFAULT_ENV:-}"
echo "[Phase3.9b] head=$(git rev-parse HEAD)"

python scripts/phase3_9b_geometry_ablation_preflight.py \
  --root "$ROOT" \
  --windows data/phase3_state_diff_windows/phase3_windows.npz \
  --action_template data/phase3_state_diff_windows/phase3_action_template.pkl \
  --phase39_summary reports/phase3_9_geometry_idm_controlled_retry_summary.json \
  --phase39_train_summary reports/phase3_9_geometry_idm_train_summary.json \
  --out_json reports/phase3_9b_geometry_ablation_preflight_summary.json \
  --out_md reports/phase3_9b_geometry_ablation_preflight_report.md

for gate in \
  PHASE3_ALLOW_GEOMETRY_IDM_ABLATION \
  PHASE3_GEOMETRY_IDM_ABLATION_CONFIRMED \
  PHASE3_ALLOW_GEOMETRY_IDM_TRAIN \
  PHASE3_GEOMETRY_IDM_TRAIN_CONFIRMED \
  PHASE3_ALLOW_GEOMETRY_IDM_RETRY \
  PHASE3_GEOMETRY_IDM_RETRY_CONFIRMED
do
  if [[ "${!gate:-0}" != "1" ]]; then
    echo "[Phase3.9b][BLOCKED] $gate must be 1"
    exit 1
  fi
done

cat > reports/phase3_9b_geometry_ablation_runtime_config.md <<EOF
# Phase3.9b Geometry IDM Ablation Runtime Config

- Timestamp: \`$(date -Is)\`
- Python: \`$(which python)\`
- Conda env: \`${CONDA_DEFAULT_ENV:-}\`
- Main HEAD: \`$(git rev-parse HEAD)\`
- Ablations: \`${PHASE3_9B_ABLATIONS:-default_geometry no_pull_loss no_coupled_xy_loss xy_only_high_weight no_quat_loss}\`
- Baseline: \`${PHASE3_9B_BASELINE:-state_action}\`
- Epochs: \`${PHASE3_9B_EPOCHS:-250}\`
- Samples per condition: \`${PHASE3_9B_SAMPLES_PER_CONDITION:-4}\`
- Conditions: \`${PHASE3_9B_CONDITIONS:-free hidden_breakaway_pin hidden_high_friction}\`
- Actions: \`${PHASE3_9B_ACTIONS:-gt_reference old_idm_gt_future phase39_idm_gt_future}\`
- Row timeout sec: \`${PHASE3_9B_ROW_TIMEOUT_SEC:-240}\`
- Total timeout sec per ablation: \`${PHASE3_9B_TOTAL_TIMEOUT_SEC:-7200}\`
- Max rows per ablation: \`${PHASE3_9B_MAX_ROWS:-36}\`

Scope:
- geometry-aware inverse dynamics ablation only
- one-step matched-prefix controlled retry only
- no future DDPM training
- no Phase4
- no CPS
- no medium/full rollout
- checkpoints are local-only and must not be committed
EOF

rm -rf reports/phase3_9b_ablation

set +e
python scripts/phase3_9b_run_ablation_matrix.py \
  --root "$ROOT" \
  --ablations "${PHASE3_9B_ABLATIONS:-default_geometry no_pull_loss no_coupled_xy_loss xy_only_high_weight no_quat_loss}" \
  --baseline "${PHASE3_9B_BASELINE:-state_action}" \
  --windows data/phase3_state_diff_windows/phase3_windows.npz \
  --action_template data/phase3_state_diff_windows/phase3_action_template.pkl \
  --old_checkpoint_root checkpoints/phase3 \
  --phase37_raw reports/phase3_7_learned_action_alignment_raw_summary.json \
  --epochs "${PHASE3_9B_EPOCHS:-250}" \
  --batch_size "${PHASE3_9B_BATCH_SIZE:-128}" \
  --hidden_dim "${PHASE3_9B_HIDDEN_DIM:-256}" \
  --lr "${PHASE3_9B_LR:-0.001}" \
  --weight_decay "${PHASE3_9B_WEIGHT_DECAY:-0.001}" \
  --seed_base "${PHASE3_9B_SEED_BASE:-392000}" \
  --samples_per_condition "${PHASE3_9B_SAMPLES_PER_CONDITION:-4}" \
  --conditions "${PHASE3_9B_CONDITIONS:-free hidden_breakaway_pin hidden_high_friction}" \
  --actions "${PHASE3_9B_ACTIONS:-gt_reference old_idm_gt_future phase39_idm_gt_future}" \
  --pred_samples "${PHASE3_9B_PRED_SAMPLES:-16}" \
  --motion_timeout "${PHASE3_9B_MOTION_TIMEOUT:-15.0}" \
  --max_prefix_actions "${PHASE3_9B_MAX_PREFIX_ACTIONS:-20}" \
  --row_timeout_sec "${PHASE3_9B_ROW_TIMEOUT_SEC:-240}" \
  --total_timeout_sec "${PHASE3_9B_TOTAL_TIMEOUT_SEC:-7200}" \
  --max_rows "${PHASE3_9B_MAX_ROWS:-36}" \
  --improve_threshold "${PHASE3_9B_IMPROVE_THRESHOLD:-0.05}" \
  --min_ok_rows_per_condition_action "${PHASE3_9B_MIN_OK_ROWS_PER_CONDITION_ACTION:-2}" \
  --out_dir reports/phase3_9b_ablation \
  --checkpoint_root checkpoints/phase3_9b_geometry_idm \
  --progress_json reports/phase3_9b_ablation_progress.json \
  --summary_json reports/phase3_9b_ablation_raw_summary.json
RUN_RC=$?
set -e

set +e
python scripts/phase3_9b_analyze_ablation_matrix.py \
  --root "$ROOT" \
  --raw_summary reports/phase3_9b_ablation_raw_summary.json \
  --out_dir reports/phase3_9b_ablation \
  --out_json reports/phase3_9b_ablation_summary.json \
  --out_md reports/phase3_9b_ablation_report.md \
  --support_threshold "${PHASE3_9B_SUPPORT_THRESHOLD:-0.05}" \
  --ablation_gap_threshold "${PHASE3_9B_ABLATION_GAP_THRESHOLD:-0.05}"
ANALYZE_RC=$?
set -e

cat > reports/phase3_9b_no_phase4_confirmation.md <<EOF
# Phase3.9b No Phase4 / No CPS Confirmation

- Timestamp: \`$(date -Is)\`
- Phase3.9b trained geometry-aware inverse dynamics ablations only.
- Phase3.9b ran one-step matched-prefix controlled retry only.
- No future DDPM was trained.
- No Phase4 was run.
- No CPS was run.
- No medium/full rollout was run.
- No paper-level claim should be made from this diagnostic alone.
- Checkpoints are local diagnostic artifacts and must not be committed.
EOF

if [[ "$RUN_RC" -ne 0 || "$ANALYZE_RC" -ne 0 ]]; then
  exit 1
fi

echo "[Phase3.9b] done"