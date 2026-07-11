#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/scripts:$ROOT/external/deformable-ravens:${PYTHONPATH:-}"

require_gate() {
  local name="$1"
  if [[ "${!name:-0}" != "1" ]]; then
    echo "[Phase3.12d-r2.2][BLOCKED] $name=1 is required."
    exit 2
  fi
}

write_no_phase4() {
  mkdir -p reports
  cat > reports/phase3_12d_r22_no_phase4_confirmation.md <<EOF
# Phase3.12d-r2.2 No Phase4 Confirmation

- Timestamp: \`$(date -Is)\`
- No future DDPM training was run.
- No inverse-dynamics model training was run.
- No Phase4 was run.
- No CPS was run.
- No candidate matrix was run.
- The linear classifier is diagnostic-only and tests condition decodability.
- Existing windows/checkpoints use legacy environment semantics and remain bridge diagnostics only.
- Formal model conclusions remain blocked.
EOF
}
trap write_no_phase4 EXIT

require_gate PHASE3_ALLOW_R22_DATA_AUDIT
require_gate PHASE3_R22_DATA_AUDIT_CONFIRMED
require_gate PHASE3_ALLOW_R22_LEAKAGE_COLLECTION
require_gate PHASE3_R22_LEAKAGE_COLLECTION_CONFIRMED

mkdir -p reports
cat > reports/phase3_12d_r22_start.md <<EOF
# Phase3.12d-r2.2 Start

- Timestamp: \`$(date -Is)\`
- Main branch: \`$(git branch --show-current)\`
- Main HEAD: \`$(git rev-parse HEAD)\`
- Submodule: \`$(git submodule status external/deformable-ravens)\`
- Scope: code/data integrity and diagnostic observation-space leakage only.
EOF

python -m py_compile \
  ccda_phase3/data_io.py \
  ccda_phase3/rollout.py \
  scripts/phase3_12d_r22_integrity.py \
  scripts/phase3_12d_r22_preflight.py \
  scripts/phase3_12d_r22_repo_data_audit.py \
  scripts/phase3_12d_r22_collect_observation_leakage.py \
  scripts/phase3_12d_r22_analyze_observation_leakage.py
bash -n scripts/phase3_12d_r22_run.sh

pytest -q \
  tests/test_ccda_phase3_state_safety.py \
  tests/test_ccda_phase3_visible_seed.py \
  tests/test_phase3_12d_r22_integrity.py

python scripts/phase3_12d_r22_preflight.py --root "$ROOT"
python scripts/phase3_12d_r22_repo_data_audit.py --root "$ROOT"

python scripts/phase3_12d_r22_collect_observation_leakage.py \
  --root "$ROOT" \
  --num-seeds "${PHASE3_12D_R22_NUM_SEEDS:-128}" \
  --seed-start "${PHASE3_12D_R22_SEED_START:-313000}" \
  --workers "${PHASE3_12D_R22_WORKERS:-4}" \
  --horizons ${PHASE3_12D_R22_HORIZONS:-0 1 5 20}

python scripts/phase3_12d_r22_analyze_observation_leakage.py \
  --root "$ROOT" \
  --folds "${PHASE3_12D_R22_GROUP_FOLDS:-5}" \
  --repeats "${PHASE3_12D_R22_GROUP_REPEATS:-5}" \
  --bootstraps "${PHASE3_12D_R22_BOOTSTRAPS:-10000}" \
  --permutations "${PHASE3_12D_R22_PERMUTATIONS:-1000}"

write_no_phase4
echo "[Phase3.12d-r2.2] complete"
