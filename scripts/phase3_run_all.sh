#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="${MODE:-smoke}"
PHASE3_STEP="${PHASE3_STEP:-all}"
ALLOW_NUMPY_FALLBACK="${ALLOW_NUMPY_FALLBACK:-0}"
PHASE3_ALLOW_ROLLOUT="${PHASE3_ALLOW_ROLLOUT:-0}"
PHASE3_ROLLOUT_CONFIRMED="${PHASE3_ROLLOUT_CONFIRMED:-0}"

PHASE3_CONDITIONS="${PHASE3_CONDITIONS:-free hidden_pin hidden_high_friction hidden_breakaway_pin}"
PHASE3_PRIMARY_HIDDEN_CONDITION="${PHASE3_PRIMARY_HIDDEN_CONDITION:-hidden_breakaway_pin}"
PHASE3_DIAGNOSTIC_HIDDEN_CONDITION="${PHASE3_DIAGNOSTIC_HIDDEN_CONDITION:-hidden_pin}"
PHASE2_5_SELECTED_CONFIG="${PHASE2_5_SELECTED_CONFIG:-breakaway_force_2p6_disp_0p045_pull_0p36}"
export CCDA_BREAKAWAY_FORCE="${CCDA_BREAKAWAY_FORCE:-2.6}"
export CCDA_BREAKAWAY_DISP="${CCDA_BREAKAWAY_DISP:-0.045}"
export CCDA_BREAKAWAY_BEAD_RATIO="${CCDA_BREAKAWAY_BEAD_RATIO:-0.45}"
export CCDA_ORACLE_BREAKAWAY_PULL_DIST="${CCDA_ORACLE_BREAKAWAY_PULL_DIST:-0.36}"
export PHASE3_CONDITIONS PHASE3_PRIMARY_HIDDEN_CONDITION PHASE3_DIAGNOSTIC_HIDDEN_CONDITION PHASE2_5_SELECTED_CONFIG

if [[ "$ALLOW_NUMPY_FALLBACK" == "1" ]]; then
  echo "[Phase3][ERROR] NumPy fallback is disabled for the DDPM Phase3 pipeline."
  echo "[Phase3][ERROR] Activate coord_bimanual for train/eval and rerun without ALLOW_NUMPY_FALLBACK."
  exit 1
fi

if [[ "$MODE" == "smoke" ]]; then
  TRAIN_SEEDS="${TRAIN_SEEDS:-10}"
  HELDOUT_SEEDS="${HELDOUT_SEEDS:-6}"
  EPOCHS_STATE="${EPOCHS_STATE:-50}"
  EPOCHS_IDM="${EPOCHS_IDM:-50}"
  FOLDS="${FOLDS:-2}"
  SEEDS="${SEEDS:-0}"
  ROLLOUT_SEEDS="${ROLLOUT_SEEDS:-3}"
elif [[ "$MODE" == "medium" ]]; then
  TRAIN_SEEDS="${TRAIN_SEEDS:-100}"
  HELDOUT_SEEDS="${HELDOUT_SEEDS:-30}"
  EPOCHS_STATE="${EPOCHS_STATE:-300}"
  EPOCHS_IDM="${EPOCHS_IDM:-200}"
  FOLDS="${FOLDS:-3}"
  SEEDS="${SEEDS:-0 1}"
  ROLLOUT_SEEDS="${ROLLOUT_SEEDS:-10}"
elif [[ "$MODE" == "full" ]]; then
  TRAIN_SEEDS="${TRAIN_SEEDS:-500}"
  HELDOUT_SEEDS="${HELDOUT_SEEDS:-100}"
  EPOCHS_STATE="${EPOCHS_STATE:-1000}"
  EPOCHS_IDM="${EPOCHS_IDM:-500}"
  FOLDS="${FOLDS:-5}"
  SEEDS="${SEEDS:-0 1 2}"
  ROLLOUT_SEEDS="${ROLLOUT_SEEDS:-30}"
else
  echo "[Phase3][ERROR] unknown MODE=$MODE"
  exit 1
fi

DIFFUSION_STEPS="${DIFFUSION_STEPS:-100}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-$DIFFUSION_STEPS}"
BETA_SCHEDULE="${BETA_SCHEDULE:-squaredcos_cap_v2}"
PREDICTION_TYPE="${PREDICTION_TYPE:-epsilon}"
VARIANCE_TYPE="${VARIANCE_TYPE:-fixed_small}"
SAMPLE_TEMPERATURE="${SAMPLE_TEMPERATURE:-1.0}"
HIDDEN_DIM="${HIDDEN_DIM:-512}"
TIME_DIM="${TIME_DIM:-128}"
LR="${LR:-1e-3}"

TRAIN_DATA_ROOT="$ROOT/external/deformable-ravens/data/phase3_ccda_large/train/hidden-contact-cable-line"
HELDOUT_DATA_ROOT="$ROOT/external/deformable-ravens/data/phase3_ccda_large/heldout/hidden-contact-cable-line"
WINDOWS="$ROOT/data/phase3_state_diff_windows/phase3_windows.npz"
CKPT_ROOT="$ROOT/checkpoints/phase3"

condition_args=(--conditions $PHASE3_CONDITIONS --primary_hidden_condition "$PHASE3_PRIMARY_HIDDEN_CONDITION" --diagnostic_hidden_condition "$PHASE3_DIAGNOSTIC_HIDDEN_CONDITION")

echo "[Phase3] MODE=$MODE PHASE3_STEP=$PHASE3_STEP DDPM_STEPS=$DIFFUSION_STEPS HIDDEN_DIM=$HIDDEN_DIM TIME_DIM=$TIME_DIM"
echo "[Phase3] conditions=$PHASE3_CONDITIONS"
echo "[Phase3] primary_hidden_condition=$PHASE3_PRIMARY_HIDDEN_CONDITION diagnostic_hidden_condition=$PHASE3_DIAGNOSTIC_HIDDEN_CONDITION"
echo "[Phase3] selected_config=$PHASE2_5_SELECTED_CONFIG breakaway_force=$CCDA_BREAKAWAY_FORCE breakaway_disp=$CCDA_BREAKAWAY_DISP"

run_generate() {
  echo "[Phase3] Step generate. Expected env: defravens37"
  python scripts/phase3_check_runtime_env.py \
    --role data_generation \
    --root "$ROOT" \
    --write_json "$ROOT/reports/phase3_runtime_generate_env.json"

  TRAIN_SEEDS="$TRAIN_SEEDS" \
  HELDOUT_SEEDS="$HELDOUT_SEEDS" \
  PHASE3_CONDITIONS="$PHASE3_CONDITIONS" \
  PHASE3_PRIMARY_HIDDEN_CONDITION="$PHASE3_PRIMARY_HIDDEN_CONDITION" \
  PHASE3_DIAGNOSTIC_HIDDEN_CONDITION="$PHASE3_DIAGNOSTIC_HIDDEN_CONDITION" \
  bash scripts/phase3_generate_large_dataset.sh
}

run_prepare() {
  echo "[Phase3] Step prepare windows"
  python scripts/phase3_prepare_windows.py \
    --root "$ROOT" \
    --train_data_root "$TRAIN_DATA_ROOT" \
    --heldout_data_root "$HELDOUT_DATA_ROOT" \
    --out "$WINDOWS" \
    --th 3 \
    --tf 4 \
    "${condition_args[@]}"

  echo "[Phase3] Step leakage check"
  python scripts/phase3_check_input_leakage.py \
    --data "$WINDOWS" \
    --out_json "$ROOT/reports/phase3_input_leakage_summary.json" \
    --out_md "$ROOT/reports/phase3_input_leakage_report.md" \
    "${condition_args[@]}"

  echo "[Phase3] Step condition integration audit"
  python scripts/phase3_condition_integration_audit.py \
    --root "$ROOT" \
    --data "$WINDOWS" \
    --leak_json "$ROOT/reports/phase3_input_leakage_summary.json" \
    --out_json "$ROOT/reports/phase3_condition_integration_audit_summary.json" \
    --out_md "$ROOT/reports/phase3_condition_integration_audit.md" \
    "${condition_args[@]}"
}

run_train_eval() {
  echo "[Phase3] Step train/eval. Expected env: coord_bimanual"
  python scripts/phase3_check_runtime_env.py \
    --role torch_train_eval \
    --root "$ROOT" \
    --write_json "$ROOT/reports/phase3_runtime_train_eval_env.json"

  python scripts/phase3_train_baselines.py \
    --data "$WINDOWS" \
    --out_root "$CKPT_ROOT" \
    --baselines paper_state state_action \
    --folds "$FOLDS" \
    --seeds $SEEDS \
    --epochs_state "$EPOCHS_STATE" \
    --epochs_idm "$EPOCHS_IDM" \
    --batch_size 128 \
    --diffusion_steps "$DIFFUSION_STEPS" \
    --num_inference_steps "$NUM_INFERENCE_STEPS" \
    --beta_schedule "$BETA_SCHEDULE" \
    --prediction_type "$PREDICTION_TYPE" \
    --variance_type "$VARIANCE_TYPE" \
    --sample_temperature "$SAMPLE_TEMPERATURE" \
    --denoiser_arch mlp \
    --hidden_dim "$HIDDEN_DIM" \
    --time_dim "$TIME_DIM" \
    --lr "$LR" \
    "${condition_args[@]}"

  python scripts/phase3_eval_baselines.py \
    --data "$WINDOWS" \
    --ckpt_root "$CKPT_ROOT" \
    --out_csv "$ROOT/reports/phase3_baseline_eval_predictions.csv" \
    --out_json "$ROOT/reports/phase3_baseline_eval_summary.json" \
    --samples_per_prefix 64 \
    --diffusion_steps "$DIFFUSION_STEPS" \
    "${condition_args[@]}"
}

run_rollout() {
  if [[ "$PHASE3_ALLOW_ROLLOUT" != "1" ]]; then
    echo "[Phase3][ERROR] rollout is disabled by default. Set PHASE3_ALLOW_ROLLOUT=1 to run it explicitly."
    exit 1
  fi
  if [[ "$PHASE3_ROLLOUT_CONFIRMED" != "1" ]]; then
    echo "[Phase3][ERROR] rollout is disabled until the user confirms. Set PHASE3_ROLLOUT_CONFIRMED=1."
    exit 1
  fi
  echo "[Phase3] Step rollout. Expected env: defravens37 with torch."
  python scripts/phase3_check_runtime_env.py \
    --role rollout \
    --root "$ROOT" \
    --write_json "$ROOT/reports/phase3_runtime_rollout_env.json"

  python scripts/phase3_policy_rollout.py \
    --root "$ROOT" \
    --checkpoint_root "$CKPT_ROOT" \
    --rollout_models paper_state state_action \
    --max_episodes_per_condition "$ROLLOUT_SEEDS" \
    --seed_start 200000 \
    --max_steps 10 \
    --samples_per_step 16 \
    --motion_timeout "${MOTION_TIMEOUT:-5}" \
    --action_clip_std "${ACTION_CLIP_STD:-3}" \
    --conditions $PHASE3_CONDITIONS \
    --primary_hidden_condition "$PHASE3_PRIMARY_HIDDEN_CONDITION" \
    --diagnostic_hidden_condition "$PHASE3_DIAGNOSTIC_HIDDEN_CONDITION" \
    --selected_recoverable_config "$PHASE2_5_SELECTED_CONFIG" \
    --windows "$WINDOWS" \
    --out_csv "$ROOT/reports/phase3_policy_rollout_trials.csv" \
    --out_json "$ROOT/reports/phase3_policy_rollout_summary.json" \
    --out_md "$ROOT/reports/phase3_policy_rollout_report.md"
}

run_aggregate() {
  echo "[Phase3] Step aggregate"
  python scripts/phase3_check_runtime_env.py \
    --role aggregate \
    --root "$ROOT" \
    --write_json "$ROOT/reports/phase3_runtime_aggregate_env.json"

  python scripts/phase3_aggregate_folds.py \
    --root "$ROOT" \
    "${condition_args[@]}"
}

run_sanity() {
  echo "[Phase3] Step sanity diagnostics. Expected env: coord_bimanual"

  python scripts/phase3_check_runtime_env.py \
    --role torch_train_eval \
    --root "$ROOT" \
    --write_json "$ROOT/reports/phase3_runtime_sanity_env.json"

  python scripts/phase3_sanity_check_metrics.py \
    --root "$ROOT" \
    --pred_csv "$ROOT/reports/phase3_baseline_eval_predictions.csv" \
    --leak_json "$ROOT/reports/phase3_input_leakage_summary.json" \
    --out_json "$ROOT/reports/phase3_sanity_check_summary.json" \
    --out_md "$ROOT/reports/phase3_sanity_check_report.md" \
    "${condition_args[@]}"

  python scripts/phase3_debug_action_codec_idm.py \
    --data "$ROOT/data/phase3_state_diff_windows/phase3_windows.npz" \
    --ckpt_root "$ROOT/checkpoints/phase3" \
    --out_json "$ROOT/reports/phase3_action_idm_debug_summary.json" \
    --out_md "$ROOT/reports/phase3_action_idm_debug_report.md"

  run_aggregate
}

run_pre_medium_audit() {
  echo "[Phase3] Step pre_medium_audit. Expected env: coord_bimanual"
  python scripts/phase3_check_runtime_env.py \
    --role torch_train_eval \
    --root "$ROOT" \
    --write_json "$ROOT/reports/phase3_runtime_pre_medium_audit_env.json"

  python scripts/phase3_pre_medium_audit.py \
    --root "$ROOT" \
    --data "$WINDOWS" \
    --out_json "$ROOT/reports/phase3_pre_medium_audit_summary.json" \
    --out_md "$ROOT/reports/phase3_pre_medium_audit_report.md"
}

case "$PHASE3_STEP" in
  generate) run_generate ;;
  prepare) run_prepare ;;
  train_eval) run_train_eval; run_aggregate ;;
  rollout) run_rollout; run_aggregate ;;
  sanity) run_sanity ;;
  pre_medium_audit) run_pre_medium_audit ;;
  aggregate) run_aggregate ;;
  all) run_generate; run_prepare; run_train_eval; run_sanity ;;
  *) echo "[Phase3][ERROR] unknown PHASE3_STEP=$PHASE3_STEP"; exit 1 ;;
esac

echo "[Phase3] done: MODE=$MODE PHASE3_STEP=$PHASE3_STEP"
