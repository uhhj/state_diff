#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="$ROOT:$ROOT/scripts:$ROOT/external/deformable-ravens:${PYTHONPATH:-}"

for gate in \
  PHASE314B_R11_ALLOW_DIAGNOSTICS \
  PHASE314B_R11_DIAGNOSTICS_CONFIRMED
do
  if [[ "${!gate:-0}" != "1" ]]; then
    echo "[Phase3.14b-r1.1][BLOCKED] $gate must be 1"
    exit 1
  fi
done

mkdir -p reports

python -m py_compile \
  ccda_phase3/phase314b_r11_exact_scheduler.py \
  scripts/phase3_14b_r11_preflight.py \
  scripts/phase3_14b_r11_audit.py \
  scripts/phase3_14b_r11_analyze.py

bash -n scripts/phase3_14b_r11_run.sh

pytest -q \
  tests/test_phase3_14b_r11_exact_scheduler.py \
  tests/test_phase3_14b_r1_diagnostics.py \
  tests/test_phase3_14b_diffusion.py

git diff --check
git -C external/deformable-ravens status --short

python scripts/phase3_14b_r11_preflight.py \
  --root "$ROOT"

python scripts/phase3_14b_r11_audit.py \
  --root "$ROOT" \
  --max-val-pair-keys "${PHASE314B_R11_VAL_PAIR_KEYS:-8}" \
  --rows-per-timestep-bin "${PHASE314B_R11_ROWS_PER_BIN:-64}"

python scripts/phase3_14b_r11_analyze.py \
  --root "$ROOT"

echo "[Phase3.14b-r1.1] completed"
