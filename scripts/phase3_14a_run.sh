#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="$ROOT:$ROOT/scripts:$ROOT/external/deformable-ravens:${PYTHONPATH:-}"

for gate in \
  PHASE314A_ALLOW_CACHE_BUILD \
  PHASE314A_CACHE_BUILD_CONFIRMED \
  PHASE314A_ALLOW_DETERMINISTIC_TRAINING \
  PHASE314A_DETERMINISTIC_TRAINING_CONFIRMED
do
  if [[ "${!gate:-0}" != "1" ]]; then
    echo "[Phase3.14a][BLOCKED] $gate must be 1"
    exit 1
  fi
done

mkdir -p reports data/phase3_14_cache checkpoints/phase3_14a/deterministic

python -m py_compile \
  ccda_phase3/phase314a_contract.py \
  ccda_phase3/phase314a_metrics.py \
  scripts/phase3_14a_build_training_cache.py \
  scripts/phase3_14a_preflight.py \
  scripts/phase3_14a_train_deterministic.py \
  scripts/phase3_14a_analyze.py

pytest -q \
  tests/test_phase3_14a_contract.py \
  tests/test_phase3_14a_metrics.py \
  tests/test_phase3_13_r1_provenance.py \
  tests/test_ccda_state_v2.py

git diff --check
git -C external/deformable-ravens status --short

python scripts/phase3_14_provenance_gate.py \
  --root "$ROOT"

CACHE_ARGS=()
if [[ "${PHASE314A_REPLACE_CACHE:-0}" == "1" ]]; then
  CACHE_ARGS+=(--replace)
fi

CACHE_PATH="$ROOT/data/phase3_14_cache/phase3_14a_training_cache.npz"
if [[ -e "$CACHE_PATH" && "${PHASE314A_REPLACE_CACHE:-0}" != "1" ]]; then
  echo "[Phase3.14a] immutable cache exists; preflight will verify it"
else
  python scripts/phase3_14a_build_training_cache.py \
    --root "$ROOT" \
    "${CACHE_ARGS[@]}"
fi

python scripts/phase3_14a_preflight.py \
  --root "$ROOT" \
  --skip-provenance-refresh

python scripts/phase3_14a_train_deterministic.py \
  --root "$ROOT" \
  --training-seeds 31401 31402 31403 \
  --max-epochs "${PHASE314A_MAX_EPOCHS:-500}" \
  --batch-size "${PHASE314A_BATCH_SIZE:-256}" \
  --eval-interval 5 \
  --patience-evals 10 \
  --bootstraps "${PHASE314A_BOOTSTRAPS:-10000}"

python scripts/phase3_14a_analyze.py \
  --root "$ROOT"

echo "[Phase3.14a] completed"
