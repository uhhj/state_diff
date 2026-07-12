#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="$ROOT:$ROOT/scripts:$ROOT/external/deformable-ravens:${PYTHONPATH:-}"

for gate in \
  PHASE314B_R1_ALLOW_DIAGNOSTICS \
  PHASE314B_R1_DIAGNOSTICS_CONFIRMED
do
  if [[ "${!gate:-0}" != "1" ]]; then
    echo "[Phase3.14b-r1][BLOCKED] $gate must be 1"
    exit 1
  fi
done

mkdir -p reports

python -m py_compile \
  ccda_phase3/phase314b_r1_diagnostics.py \
  scripts/phase3_14b_r1_preflight.py \
  scripts/phase3_14b_r1_audit.py \
  scripts/phase3_14b_r1_analyze.py

bash -n scripts/phase3_14b_r1_run.sh

pytest -q \
  tests/test_phase3_14b_r1_diagnostics.py \
  tests/test_phase3_14b_contract.py \
  tests/test_phase3_14b_diffusion.py \
  tests/test_phase3_14b_metrics.py

git diff --check
git -C external/deformable-ravens status --short

python scripts/phase3_14b_r1_preflight.py \
  --root "$ROOT"

python scripts/phase3_14b_r1_audit.py \
  --root "$ROOT" \
  --max-val-pair-keys "${PHASE314B_R1_VAL_PAIR_KEYS:-8}" \
  --rows-per-timestep-bin "${PHASE314B_R1_ROWS_PER_BIN:-64}"

python scripts/phase3_14b_r1_analyze.py \
  --root "$ROOT"

echo "[Phase3.14b-r1] completed"
