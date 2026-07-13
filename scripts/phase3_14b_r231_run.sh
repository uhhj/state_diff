#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/data/state_diff2}"
cd "${ROOT}"

python scripts/phase3_14b_r231_preflight.py --root "${ROOT}"

python scripts/phase3_14b_r231_run_controls.py \
  --root "${ROOT}" \
  --direct-one-steps 2000 \
  --direct-multi-steps 5000 \
  --fixed-one-steps 3000 \
  --fixed-pair-steps 5000 \
  --random-one-steps 6000 \
  --random-multi-steps 8000

python scripts/phase3_14b_r231_finalize.py --root "${ROOT}"
