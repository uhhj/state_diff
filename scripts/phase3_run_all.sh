#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="${MODE:-smoke}"
PHASE3_STEP="${PHASE3_STEP:-all}"

if [[ "$MODE" == "smoke" ]]; then
  TRAIN_SEEDS="${TRAIN_SEEDS:-10}"
  HELDOUT_SEEDS="${HELDOUT_SEEDS:-6}"
  EPOCHS_STATE="${EPOCHS_STATE:-50}"
  EPOCHS_IDM="${EPOCHS_IDM:-50}"
  FOLDS="${FOLDS:-2}"
  SEEDS="${SEEDS:-0}"
  ROLLOUT_SEEDS="${ROLLOUT_SEEDS:-3}"
elif [[ "$MODE" == "full" ]]; then
  TRAIN_SEEDS="${TRAIN_SEEDS:-500}"
  HELDOUT_SEEDS="${HELDOUT_SEEDS:-100}"
  EPOCHS_STATE="${EPOCHS_STATE:-1000}"
  EPOCHS_IDM="${EPOCHS_IDM:-500}"
  FOLDS="${FOLDS:-5}"
  SEEDS="${SEEDS:-0 1 2}"
  ROLLOUT_SEEDS="${ROLLOUT_SEEDS:-30}"
else
  echo "unknown MODE=$MODE"
  exit 1
fi

echo "[Phase3] MODE=$MODE PHASE3_STEP=$PHASE3_STEP"

run_generate() {
  echo "[Phase3] Step 1: generate large dataset. Requires defravens37."
  TRAIN_SEEDS="$TRAIN_SEEDS" HELDOUT_SEEDS="$HELDOUT_SEEDS" bash scripts/phase3_generate_large_dataset.sh
}

run_prepare() {
  echo "[Phase3] Step 2: prepare windows."
  python scripts/phase3_prepare_windows.py \
    --root "$ROOT" \
    --train_data_root "$ROOT/external/deformable-ravens/data/phase3_ccda_large/train/hidden-contact-cable-line" \
    --heldout_data_root "$ROOT/external/deformable-ravens/data/phase3_ccda_large/heldout/hidden-contact-cable-line" \
    --out "$ROOT/data/phase3_state_diff_windows/phase3_windows.npz" \
    --th 3 \
    --tf 4
}

run_leakage() {
  echo "[Phase3] Step 3: leakage check"
  python scripts/phase3_check_input_leakage.py \
    --data "$ROOT/data/phase3_state_diff_windows/phase3_windows.npz" \
    --out_json "$ROOT/reports/phase3_input_leakage_summary.json" \
    --out_md "$ROOT/reports/phase3_input_leakage_report.md"
}

run_train() {
  echo "[Phase3] Step 4: train baselines"
  python scripts/phase3_train_baselines.py \
    --data "$ROOT/data/phase3_state_diff_windows/phase3_windows.npz" \
    --out_root "$ROOT/checkpoints/phase3" \
    --baselines paper_state state_action \
    --folds "$FOLDS" \
    --seeds $SEEDS \
    --epochs_state "$EPOCHS_STATE" \
    --epochs_idm "$EPOCHS_IDM" \
    --batch_size 128 \
    --diffusion_steps 100
}

run_eval() {
  echo "[Phase3] Step 5: offline eval"
  python scripts/phase3_eval_baselines.py \
    --data "$ROOT/data/phase3_state_diff_windows/phase3_windows.npz" \
    --ckpt_root "$ROOT/checkpoints/phase3" \
    --out_csv "$ROOT/reports/phase3_baseline_eval_predictions.csv" \
    --out_json "$ROOT/reports/phase3_baseline_eval_summary.json" \
    --samples_per_prefix 64 \
    --diffusion_steps 100
}

run_policy() {
  echo "[Phase3] Step 6: policy rollout"
  python scripts/phase3_policy_rollout.py \
    --root "$ROOT" \
    --ckpt_root "$ROOT/checkpoints/phase3" \
    --baselines paper_state state_action \
    --num_seeds "$ROLLOUT_SEEDS" \
    --seed_start 200000 \
    --max_steps 10 \
    --samples_per_step 16 \
    --motion_timeout "${MOTION_TIMEOUT:-5}" \
    --action_clip_std "${ACTION_CLIP_STD:-3}"
}

run_aggregate() {
  echo "[Phase3] Step 7: aggregate"
  python scripts/phase3_aggregate_folds.py --root "$ROOT"
  echo "[Phase3] complete. See reports/phase3_final_report.md"
}

case "$PHASE3_STEP" in
  all) run_generate; run_prepare; run_leakage; run_train; run_eval; run_policy; run_aggregate ;;
  generate) run_generate ;;
  prepare) run_prepare ;;
  leakage) run_leakage ;;
  train) run_train ;;
  eval) run_eval ;;
  policy) run_policy ;;
  aggregate) run_aggregate ;;
  train_eval) run_prepare; run_leakage; run_train; run_eval; run_policy; run_aggregate ;;
  *) echo "unknown PHASE3_STEP=$PHASE3_STEP"; exit 1 ;;
esac
