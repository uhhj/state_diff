#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="$ROOT:$ROOT/scripts:$ROOT/external/deformable-ravens:${PYTHONPATH:-}"

for gate in \
  PHASE314B_R21_ALLOW_DIAGNOSTICS \
  PHASE314B_R21_DIAGNOSTICS_CONFIRMED
do
  if [[ "${!gate:-0}" != "1" ]]; then
    echo "[Phase3.14b-r2.1][BLOCKED] $gate must be 1"
    exit 1
  fi
done

mkdir -p reports data/phase3_14b_r21_eval

python -m py_compile \
  ccda_phase3/phase314b_r21_contract.py \
  ccda_phase3/phase314b_r21_geometry.py \
  scripts/phase3_14b_r21_preflight.py \
  scripts/phase3_14b_r21_audit.py \
  scripts/phase3_14b_r21_analyze.py

bash -n scripts/phase3_14b_r21_run.sh

pytest -q \
  tests/test_phase3_14b_r21_contract.py \
  tests/test_phase3_14b_r21_geometry.py \
  tests/test_phase3_14b_r2_metrics.py \
  tests/test_phase3_14b_r2_diffusion.py

git diff --check
git -C external/deformable-ravens status --short

python scripts/phase3_14b_r21_preflight.py \
  --root "$ROOT"

python scripts/phase3_14b_r21_audit.py \
  --root "$ROOT"

python scripts/phase3_14b_r21_analyze.py \
  --root "$ROOT"

echo "[Phase3.14b-r2.1] completed"
