#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"; export PYTHONPATH="$ROOT:$ROOT/scripts:$ROOT/external/deformable-ravens:${PYTHONPATH:-}"
[[ "${PHASE314B_R22_ALLOW_FORMAL:-0}" == 1 && "${PHASE314B_R22_FORMAL_CONFIRMED:-0}" == 1 && "${PHASE314B_R22_REQUIRE_CUDA:-0}" == 1 ]] || { echo blocked; exit 1; }
python - <<'PY'
import torch
assert torch.cuda.is_available(), 'CUDA required'
from ccda_phase3.phase314b_r22_contract import load_self_hashed_json
load_self_hashed_json('reports/phase3_14b_r22_frozen_contract.json'); load_self_hashed_json('reports/phase3_14b_r22_pilot_selection.json')
PY
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo tracked-worktree-dirty; exit 1; }
python scripts/phase3_14b_r22_train.py --mode formal --seeds 31441 31442 31443 --max-epochs "${PHASE314B_R22_MAX_EPOCHS:-750}" --batch-size "${PHASE314B_R22_BATCH_SIZE:-128}"
set +e; python scripts/phase3_14b_r22_select_validation.py --root "$ROOT"; STATUS=$?; set -e
python scripts/phase3_14b_r22_analyze.py --root "$ROOT"; exit $STATUS
