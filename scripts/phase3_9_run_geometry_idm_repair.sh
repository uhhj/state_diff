#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p reports checkpoints/phase3_9_geometry_idm

export PYTHONPATH="$ROOT:$ROOT/scripts:$ROOT/external/deformable-ravens:${PYTHONPATH:-}"

export PHASE3_CONDITIONS="${PHASE3_CONDITIONS:-free hidden_pin hidden_high_friction hidden_breakaway_pin}"
export PHASE3_PRIMARY_HIDDEN_CONDITION="${PHASE3_PRIMARY_HIDDEN_CONDITION:-hidden_breakaway_pin}"
export PHASE3_DIAGNOSTIC_HIDDEN_CONDITION="${PHASE3_DIAGNOSTIC_HIDDEN_CONDITION:-hidden_pin}"

export CCDA_BREAKAWAY_FORCE="${CCDA_BREAKAWAY_FORCE:-2.6}"
export CCDA_BREAKAWAY_DISP="${CCDA_BREAKAWAY_DISP:-0.045}"
export CCDA_BREAKAWAY_BEAD_RATIO="${CCDA_BREAKAWAY_BEAD_RATIO:-0.45}"
export CCDA_ORACLE_BREAKAWAY_PULL_DIST="${CCDA_ORACLE_BREAKAWAY_PULL_DIST:-0.36}"

echo "[Phase3.9] root=$ROOT"
echo "[Phase3.9] python=$(which python)"
echo "[Phase3.9] conda_env=${CONDA_DEFAULT_ENV:-}"
echo "[Phase3.9] head=$(git rev-parse HEAD)"

python scripts/phase3_9_geometry_idm_preflight.py \
  --root "$ROOT" \
  --windows data/phase3_state_diff_windows/phase3_windows.npz \
  --action_template data/phase3_state_diff_windows/phase3_action_template.pkl \
  --checkpoint_root checkpoints/phase3 \
  --phase38c_summary reports/phase3_8c_pose0_confirmation_summary.json \
  --out_json reports/phase3_9_geometry_idm_preflight_summary.json \
  --out_md reports/phase3_9_geometry_idm_preflight_report.md

if [[ "${PHASE3_ALLOW_GEOMETRY_IDM_TRAIN:-0}" != "1" ]]; then
  echo "[Phase3.9][BLOCKED] Preflight completed. Training not run because PHASE3_ALLOW_GEOMETRY_IDM_TRAIN != 1."
  exit 1
fi

if [[ "${PHASE3_GEOMETRY_IDM_TRAIN_CONFIRMED:-0}" != "1" ]]; then
  echo "[Phase3.9][BLOCKED] Preflight completed. Training not run because PHASE3_GEOMETRY_IDM_TRAIN_CONFIRMED != 1."
  exit 1
fi

PHASE3_9_OUT_DIR="${PHASE3_9_OUT_DIR:-checkpoints/phase3_9_geometry_idm/state_action/fold_phase3_9_seed_${PHASE3_9_SEED:-390000}}"

cat > reports/phase3_9_geometry_idm_runtime_config.md <<EOF
# Phase3.9 Geometry-Aware IDM Runtime Config

- Timestamp: \`$(date -Is)\`
- Python: \`$(which python)\`
- Conda env: \`${CONDA_DEFAULT_ENV:-}\`
- Main HEAD: \`$(git rev-parse HEAD)\`
- Baseline: \`${PHASE3_9_BASELINE:-state_action}\`
- Out dir: \`$PHASE3_9_OUT_DIR\`
- Epochs: \`${PHASE3_9_EPOCHS:-300}\`
- Batch size: \`${PHASE3_9_BATCH_SIZE:-128}\`
- Hidden dim: \`${PHASE3_9_HIDDEN_DIM:-256}\`
- LR: \`${PHASE3_9_LR:-0.001}\`
- action_z_weight: \`${PHASE3_9_ACTION_Z_WEIGHT:-1.0}\`
- pose0_xy_weight: \`${PHASE3_9_POSE0_XY_WEIGHT:-8.0}\`
- pose1_xy_weight: \`${PHASE3_9_POSE1_XY_WEIGHT:-8.0}\`
- coupled_xy_weight: \`${PHASE3_9_COUPLED_XY_WEIGHT:-8.0}\`
- pull_xy_weight: \`${PHASE3_9_PULL_XY_WEIGHT:-4.0}\`
- z_weight: \`${PHASE3_9_Z_WEIGHT:-0.25}\`
- quat_weight: \`${PHASE3_9_QUAT_WEIGHT:-0.02}\`

Scope:
- train inverse dynamics only
- no future DDPM training
- no Phase4
- no CPS
- no medium/full rollout
- checkpoint is local diagnostic artifact and should not be committed
EOF

python scripts/phase3_9_train_geometry_idm.py \
  --root "$ROOT" \
  --windows data/phase3_state_diff_windows/phase3_windows.npz \
  --action_template data/phase3_state_diff_windows/phase3_action_template.pkl \
  --old_checkpoint_root checkpoints/phase3 \
  --baseline "${PHASE3_9_BASELINE:-state_action}" \
  --out_dir "$PHASE3_9_OUT_DIR" \
  --epochs "${PHASE3_9_EPOCHS:-300}" \
  --batch_size "${PHASE3_9_BATCH_SIZE:-128}" \
  --idm_hidden_dim "${PHASE3_9_HIDDEN_DIM:-256}" \
  --idm_lr "${PHASE3_9_LR:-0.001}" \
  --idm_weight_decay "${PHASE3_9_WEIGHT_DECAY:-0.001}" \
  --action_z_weight "${PHASE3_9_ACTION_Z_WEIGHT:-1.0}" \
  --pose0_xy_weight "${PHASE3_9_POSE0_XY_WEIGHT:-8.0}" \
  --pose1_xy_weight "${PHASE3_9_POSE1_XY_WEIGHT:-8.0}" \
  --coupled_xy_weight "${PHASE3_9_COUPLED_XY_WEIGHT:-8.0}" \
  --pull_xy_weight "${PHASE3_9_PULL_XY_WEIGHT:-4.0}" \
  --z_weight "${PHASE3_9_Z_WEIGHT:-0.25}" \
  --quat_weight "${PHASE3_9_QUAT_WEIGHT:-0.02}" \
  --seed "${PHASE3_9_SEED:-390000}" \
  --summary_json reports/phase3_9_geometry_idm_train_summary.json \
  --summary_md reports/phase3_9_geometry_idm_train_report.md

NEW_IDM_PATH="$ROOT/$PHASE3_9_OUT_DIR/inverse_dynamics.pt"

if [[ ! -f "$NEW_IDM_PATH" ]]; then
  echo "[Phase3.9][FAIL] missing new inverse dynamics checkpoint: $NEW_IDM_PATH"
  exit 1
fi

if [[ "${PHASE3_ALLOW_GEOMETRY_IDM_RETRY:-0}" != "1" ]]; then
  echo "[Phase3.9][BLOCKED] Training completed. Retry not run because PHASE3_ALLOW_GEOMETRY_IDM_RETRY != 1."
  exit 1
fi

if [[ "${PHASE3_GEOMETRY_IDM_RETRY_CONFIRMED:-0}" != "1" ]]; then
  echo "[Phase3.9][BLOCKED] Training completed. Retry not run because PHASE3_GEOMETRY_IDM_RETRY_CONFIRMED != 1."
  exit 1
fi

rm -rf reports/phase3_9_workers
mkdir -p reports/phase3_9_workers

set +e
python scripts/phase3_9_controlled_retry.py \
  --root "$ROOT" \
  --windows data/phase3_state_diff_windows/phase3_windows.npz \
  --action_template data/phase3_state_diff_windows/phase3_action_template.pkl \
  --old_checkpoint_root checkpoints/phase3 \
  --baseline "${PHASE3_9_BASELINE:-state_action}" \
  --new_inverse_path "$NEW_IDM_PATH" \
  --phase37_raw reports/phase3_7_learned_action_alignment_raw_summary.json \
  --conditions ${PHASE3_9_RETRY_CONDITIONS:-free hidden_breakaway_pin hidden_high_friction} \
  --actions ${PHASE3_9_RETRY_ACTIONS:-gt_reference old_idm_gt_future phase39_idm_gt_future} \
  --samples_per_condition "${PHASE3_9_RETRY_SAMPLES_PER_CONDITION:-2}" \
  --pred_samples "${PHASE3_9_RETRY_PRED_SAMPLES:-16}" \
  --seed_base "${PHASE3_9_RETRY_SEED_BASE:-391000}" \
  --motion_timeout "${PHASE3_9_RETRY_MOTION_TIMEOUT:-15.0}" \
  --max_prefix_actions "${PHASE3_9_RETRY_MAX_PREFIX_ACTIONS:-20}" \
  --row_timeout_sec "${PHASE3_9_RETRY_ROW_TIMEOUT_SEC:-240}" \
  --total_timeout_sec "${PHASE3_9_RETRY_TOTAL_TIMEOUT_SEC:-3600}" \
  --max_rows "${PHASE3_9_RETRY_MAX_ROWS:-18}" \
  --out_csv reports/phase3_9_geometry_idm_controlled_retry_trials.csv \
  --out_json reports/phase3_9_geometry_idm_controlled_retry_raw_summary.json \
  --progress_json reports/phase3_9_geometry_idm_controlled_retry_progress.json \
  --worker_dir reports/phase3_9_workers
RETRY_RC=$?
set -e

set +e
python scripts/phase3_9_analyze_controlled_retry.py \
  --root "$ROOT" \
  --trials_csv reports/phase3_9_geometry_idm_controlled_retry_trials.csv \
  --progress_json reports/phase3_9_geometry_idm_controlled_retry_progress.json \
  --raw_summary reports/phase3_9_geometry_idm_controlled_retry_raw_summary.json \
  --train_summary reports/phase3_9_geometry_idm_train_summary.json \
  --out_json reports/phase3_9_geometry_idm_controlled_retry_summary.json \
  --out_md reports/phase3_9_geometry_idm_controlled_retry_report.md \
  --improve_threshold "${PHASE3_9_IMPROVE_THRESHOLD:-0.05}" \
  --min_ok_rows_per_condition_action "${PHASE3_9_MIN_OK_ROWS_PER_CONDITION_ACTION:-1}"
ANALYZE_RC=$?
set -e

cat > reports/phase3_9_no_phase4_confirmation.md <<EOF
# Phase3.9 No Phase4 / No CPS Confirmation

- Timestamp: \`$(date -Is)\`
- Phase3.9 trained geometry-aware inverse dynamics only.
- Phase3.9 ran one-step matched-prefix controlled retry only.
- No future DDPM was trained.
- No Phase4 was run.
- No CPS was run.
- No medium/full rollout was run.
- No paper-level claim should be made from this diagnostic alone.
EOF

if [[ "$RETRY_RC" -ne 0 || "$ANALYZE_RC" -ne 0 ]]; then
  exit 1
fi

echo "[Phase3.9] done"
