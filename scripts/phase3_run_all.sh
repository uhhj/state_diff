#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="${MODE:-smoke}"
PHASE3_STEP="${PHASE3_STEP:-all}"
ALLOW_NUMPY_FALLBACK="${ALLOW_NUMPY_FALLBACK:-0}"

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

TRAIN_DATA_ROOT="$ROOT/external/deformable-ravens/data/phase3_ccda_large/train/hidden-contact-cable-line"
HELDOUT_DATA_ROOT="$ROOT/external/deformable-ravens/data/phase3_ccda_large/heldout/hidden-contact-cable-line"
WINDOWS="$ROOT/data/phase3_state_diff_windows/phase3_windows.npz"
CKPT_ROOT="$ROOT/checkpoints/phase3"

fallback_arg=()
if [[ "$ALLOW_NUMPY_FALLBACK" == "1" ]]; then
  fallback_arg=(--allow_numpy_fallback)
fi

echo "[Phase3] MODE=$MODE PHASE3_STEP=$PHASE3_STEP ALLOW_NUMPY_FALLBACK=$ALLOW_NUMPY_FALLBACK"

run_generate() {
  echo "[Phase3] Step generate. Expected env: defravens37"
  python scripts/phase3_check_runtime_env.py \
    --role data_generation \
    --root "$ROOT" \
    --write_json "$ROOT/reports/phase3_runtime_generate_env.json"

  TRAIN_SEEDS="$TRAIN_SEEDS" \
  HELDOUT_SEEDS="$HELDOUT_SEEDS" \
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
    --tf 4

  echo "[Phase3] Step leakage check"
  python scripts/phase3_check_input_leakage.py \
    --data "$WINDOWS" \
    --out_json "$ROOT/reports/phase3_input_leakage_summary.json" \
    --out_md "$ROOT/reports/phase3_input_leakage_report.md"
}

run_train_eval() {
  echo "[Phase3] Step train/eval. Expected env: coord_bimanual"
  python scripts/phase3_check_runtime_env.py \
    --role torch_train_eval \
    --root "$ROOT" \
    "${fallback_arg[@]}" \
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
    --diffusion_steps 100 \
    "${fallback_arg[@]}"

  python scripts/phase3_eval_baselines.py \
    --data "$WINDOWS" \
    --ckpt_root "$CKPT_ROOT" \
    --out_csv "$ROOT/reports/phase3_baseline_eval_predictions.csv" \
    --out_json "$ROOT/reports/phase3_baseline_eval_summary.json" \
    --samples_per_prefix 64 \
    --diffusion_steps 100
}

run_rollout() {
  echo "[Phase3] Step rollout. Expected env: defravens37 with torch, or explicit fallback smoke."
  python scripts/phase3_check_runtime_env.py \
    --role rollout \
    --root "$ROOT" \
    "${fallback_arg[@]}" \
    --write_json "$ROOT/reports/phase3_runtime_rollout_env.json"

  python scripts/phase3_policy_rollout.py \
    --root "$ROOT" \
    --ckpt_root "$CKPT_ROOT" \
    --baselines paper_state state_action \
    --num_seeds "$ROLLOUT_SEEDS" \
    --seed_start 200000 \
    --max_steps 10 \
    --samples_per_step 16 \
    --motion_timeout "${MOTION_TIMEOUT:-5}" \
    --action_clip_std "${ACTION_CLIP_STD:-3}" \
    "${fallback_arg[@]}"
}

run_aggregate() {
  echo "[Phase3] Step aggregate"
  python scripts/phase3_check_runtime_env.py \
    --role aggregate \
    --root "$ROOT" \
    --write_json "$ROOT/reports/phase3_runtime_aggregate_env.json"

  agg_extra=()
  if [[ "$ALLOW_NUMPY_FALLBACK" == "1" ]]; then
    agg_extra=(--allow_fallback_report)
  fi

  python scripts/phase3_aggregate_folds.py \
    --root "$ROOT" \
    "${agg_extra[@]}"
}

case "$PHASE3_STEP" in
  generate) run_generate ;;
  prepare) run_prepare ;;
  train_eval) run_train_eval; run_aggregate ;;
  rollout) run_rollout; run_aggregate ;;
  aggregate) run_aggregate ;;
  all) run_generate; run_prepare; run_train_eval; run_rollout; run_aggregate ;;
  *) echo "[Phase3][ERROR] unknown PHASE3_STEP=$PHASE3_STEP"; exit 1 ;;
esac

echo "[Phase3] done: MODE=$MODE PHASE3_STEP=$PHASE3_STEP"
