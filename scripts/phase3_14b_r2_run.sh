#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="$ROOT:$ROOT/scripts:$ROOT/external/deformable-ravens:${PYTHONPATH:-}"

for gate in \
  PHASE314B_R2_ALLOW_SMOKE \
  PHASE314B_R2_ALLOW_TRAINING \
  PHASE314B_R2_TRAINING_CONFIRMED \
  PHASE314B_R2_ALLOW_TEST \
  PHASE314B_R2_TEST_CONFIRMED
do
  if [[ "${!gate:-0}" != "1" ]]; then
    echo "[Phase3.14b-r2][BLOCKED] $gate must be 1"
    exit 1
  fi
done

mkdir -p \
  reports \
  checkpoints/phase3_14b_r2 \
  data/phase3_14b_r2_eval

python -m py_compile \
  ccda_phase3/phase314b_r2_contract.py \
  ccda_phase3/phase314b_r2_diffusion.py \
  ccda_phase3/phase314b_r2_metrics.py \
  scripts/phase3_14b_r2_preflight.py \
  scripts/phase3_14b_r2_smoke.py \
  scripts/phase3_14b_r2_train.py \
  scripts/phase3_14b_r2_select.py \
  scripts/phase3_14b_r2_eval.py \
  scripts/phase3_14b_r2_analyze.py

bash -n scripts/phase3_14b_r2_run.sh

pytest -q \
  tests/test_phase3_14b_r2_contract.py \
  tests/test_phase3_14b_r2_diffusion.py \
  tests/test_phase3_14b_r2_metrics.py \
  tests/test_phase3_14b_r11_exact_scheduler.py

git diff --check
git -C external/deformable-ravens status --short

python scripts/phase3_14b_r2_preflight.py \
  --root "$ROOT"

python scripts/phase3_14b_r2_smoke.py \
  --root "$ROOT"

TRAIN_ARGS=()
EVAL_ARGS=()
if [[ "${PHASE314B_R2_ALLOW_CPU_FULL:-0}" == "1" ]]; then
  TRAIN_ARGS+=(--allow-cpu-full)
  EVAL_ARGS+=(--allow-cpu-eval)
fi
if [[ "${PHASE314B_R2_REPLACE_CHECKPOINTS:-0}" == "1" ]]; then
  TRAIN_ARGS+=(--replace-checkpoints)
fi

python scripts/phase3_14b_r2_train.py \
  --root "$ROOT" \
  --configs \
    epsilon_cosine_cap_0p5 \
    sample_cosine \
    v_prediction_cosine \
  --seeds 31431 31432 31433 \
  --max-epochs "${PHASE314B_R2_MAX_EPOCHS:-750}" \
  --batch-size "${PHASE314B_R2_BATCH_SIZE:-128}" \
  --eval-interval 25 \
  --patience-evals 6 \
  "${TRAIN_ARGS[@]}"

set +e
python scripts/phase3_14b_r2_select.py \
  --root "$ROOT"
SELECTION_STATUS=$?
set -e

if [[ "$SELECTION_STATUS" -ne 0 ]]; then
  python scripts/phase3_14b_r2_analyze.py --root "$ROOT"
  echo "[Phase3.14b-r2] no stable validation configuration"
  exit "$SELECTION_STATUS"
fi

set +e
python scripts/phase3_14b_r2_eval.py \
  --root "$ROOT" \
  --sample-seed 385000 \
  --bootstraps "${PHASE314B_R2_BOOTSTRAPS:-10000}" \
  "${EVAL_ARGS[@]}"
EVAL_STATUS=$?
set -e

python scripts/phase3_14b_r2_analyze.py \
  --root "$ROOT"

if [[ "$EVAL_STATUS" -ne 0 ]]; then
  echo "[Phase3.14b-r2] formal test gate failed"
  exit "$EVAL_STATUS"
fi

echo "[Phase3.14b-r2] completed"
