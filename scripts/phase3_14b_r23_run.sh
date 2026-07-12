#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/data/state_diff2}"
cd "$ROOT"

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export PHASE314B_R23_REQUIRE_CUDA=1

test "$(git branch --show-current)" = "Experiment1"
git merge-base --is-ancestor \
  a54cdd20d88f5eb00ebe25e8d4a20c28fe7d6e7f HEAD
test -z "$(git status --porcelain --untracked-files=all)"
test "$(git -C external/deformable-ravens rev-parse HEAD)" = \
  "633a88752445cf5d6776ed374fdbbdb35f93050c"
test -z "$(git -C external/deformable-ravens status --porcelain --untracked-files=all)"

python -m pytest -q \
  tests/test_phase314b_r22_contract.py \
  tests/test_phase314b_r22_geometry.py \
  tests/test_phase314b_r22_loss.py \
  tests/test_phase314b_r23_diagnostics.py

python scripts/phase3_14b_r23_preflight.py --root "$ROOT"

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required")
print(torch.cuda.get_device_name(0))
PY

python scripts/phase3_14b_r23_checkpoint_diagnosis.py \
  --root "$ROOT" \
  --device cuda \
  --validation-rows 64 \
  --trace-rows 16 \
  --gradient-rows 32

python scripts/phase3_14b_r23_tiny_overfit.py \
  --root "$ROOT" \
  --tiny-rows 16 \
  --fixed-steps 3000 \
  --random-steps 5000 \
  --learning-rate 1e-3

python scripts/phase3_14b_r23_finalize.py --root "$ROOT"

test ! -e reports/phase3_14b_r23_test_summary.json
test ! -e checkpoints/phase3_14b_r23
test -z "$(git -C external/deformable-ravens status --porcelain --untracked-files=all)"

echo "Phase3.14b-r2.3 diagnosis completed. Review reports before committing."
