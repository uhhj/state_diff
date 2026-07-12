#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="$ROOT:$ROOT/scripts:$ROOT/external/deformable-ravens:${PYTHONPATH:-}"

for gate in \
  PHASE314B_ALLOW_DDPM_TRAINING \
  PHASE314B_DDPM_TRAINING_CONFIRMED \
  PHASE314B_ALLOW_OFFLINE_SUPPORT \
  PHASE314B_OFFLINE_SUPPORT_CONFIRMED
do
  if [[ "${!gate:-0}" != "1" ]]; then
    echo "[Phase3.14b][BLOCKED] $gate must be 1"
    exit 1
  fi
done

mkdir -p \
  reports \
  checkpoints/phase3_14b/ddpm \
  data/phase3_14b_eval

python -m py_compile \
  ccda_phase3/phase314b_contract.py \
  ccda_phase3/phase314b_models.py \
  ccda_phase3/phase314b_diffusion.py \
  ccda_phase3/phase314b_metrics.py \
  scripts/phase3_14b_preflight.py \
  scripts/phase3_14b_train_ddpm.py \
  scripts/phase3_14b_eval_ddpm.py \
  scripts/phase3_14b_analyze.py

bash -n scripts/phase3_14b_run.sh

pytest -q \
  tests/test_phase3_14b_contract.py \
  tests/test_phase3_14b_models.py \
  tests/test_phase3_14b_diffusion.py \
  tests/test_phase3_14b_metrics.py \
  tests/test_phase3_14a_contract.py \
  tests/test_phase3_14a_metrics.py

git diff --check
git -C external/deformable-ravens status --short

python scripts/phase3_14b_preflight.py \
  --root "$ROOT"

TRAIN_ARGS=()
EVAL_ARGS=()
if [[ "${PHASE314B_ALLOW_CPU_FULL:-0}" == "1" ]]; then
  TRAIN_ARGS+=(--allow-cpu-full)
  EVAL_ARGS+=(--allow-cpu-eval)
fi

python scripts/phase3_14b_train_ddpm.py \
  --root "$ROOT" \
  --families mlp_ddpm temporal_unet_ddpm \
  --input-variants paper_state state_action \
  --training-seeds 31421 31422 31423 \
  --max-epochs "${PHASE314B_MAX_EPOCHS:-750}" \
  --batch-size "${PHASE314B_BATCH_SIZE:-128}" \
  --eval-interval 25 \
  --patience-evals 6 \
  "${TRAIN_ARGS[@]}"

set +e
python scripts/phase3_14b_eval_ddpm.py \
  --root "$ROOT" \
  --max-k 32 \
  --bootstraps "${PHASE314B_BOOTSTRAPS:-10000}" \
  --sample-seed 350000 \
  "${EVAL_ARGS[@]}"
EVAL_STATUS=$?
set -e

python scripts/phase3_14b_analyze.py \
  --root "$ROOT"

if [[ "$EVAL_STATUS" -ne 0 ]]; then
  echo "[Phase3.14b] candidate-support gate failed"
  exit "$EVAL_STATUS"
fi

echo "[Phase3.14b] completed"
