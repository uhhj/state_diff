#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="$ROOT:$ROOT/scripts:$ROOT/external/deformable-ravens:${PYTHONPATH:-}"

mkdir -p reports

cat > reports/phase3_12d_r23_start.md <<EOF
# Phase3.12d-r2.3 Start

- Timestamp: \`$(date -Is)\`
- Main branch: \`$(git branch --show-current)\`
- Main HEAD: \`$(git rev-parse HEAD)\`
- Submodule: \`$(git submodule status external/deformable-ravens)\`
- Python: \`$(which python)\`
- Conda env: \`${CONDA_DEFAULT_ENV:-}\`

## Objective

- Audit the observation contract after r2.2 found velocity-driven leakage.
- Collect real physics-step trajectories rather than repeating one state.
- Separate direct simulator bead velocity from causal XY history.
- Do not change the physical environment.
- Do not train StateDiff or IDM.
- Do not run candidate matrix, Phase4, or CPS.
EOF

python -m py_compile \
  ccda_phase3/observation_contract.py \
  scripts/phase3_12d_r23_preflight.py \
  scripts/phase3_12d_r23_collect_trajectory_observations.py \
  scripts/phase3_12d_r23_analyze_observation_contract.py

bash -n scripts/phase3_12d_r23_run.sh

pytest -q \
  tests/test_phase3_12d_r23_observation_contract.py \
  tests/test_ccda_phase3_state_safety.py \
  tests/test_ccda_phase3_visible_seed.py \
  tests/test_phase3_12d_r22_integrity.py

git diff --check

python scripts/phase3_12d_r23_preflight.py \
  --root "$ROOT" \
  --out-json reports/phase3_12d_r23_preflight_summary.json \
  --out-md reports/phase3_12d_r23_preflight_report.md

for gate in \
  PHASE3_ALLOW_R23_TRAJECTORY_COLLECTION \
  PHASE3_R23_TRAJECTORY_COLLECTION_CONFIRMED
do
  if [[ "${!gate:-0}" != "1" ]]; then
    echo "[r2.3][BLOCKED] $gate must be 1"
    exit 1
  fi
done

python scripts/phase3_12d_r23_collect_trajectory_observations.py \
  --root "$ROOT" \
  --out-dir data/phase3_12d_r23_observation_contract \
  --num-seeds "${PHASE3_12D_R23_NUM_SEEDS:-128}" \
  --seed-start "${PHASE3_12D_R23_SEED_START:-314000}" \
  --max-step "${PHASE3_12D_R23_MAX_STEP:-20}" \
  --workers "${PHASE3_12D_R23_WORKERS:-4}" \
  --min-settle-steps "${PHASE3_12D_R23_MIN_SETTLE_STEPS:-540}" \
  --max-settle-steps "${PHASE3_12D_R23_MAX_SETTLE_STEPS:-2400}" \
  --static-checks-required "${PHASE3_12D_R23_STATIC_CHECKS_REQUIRED:-8}" \
  --static-check-interval "${PHASE3_12D_R23_STATIC_CHECK_INTERVAL:-10}" \
  --motion-timeout "${PHASE3_12D_R23_MOTION_TIMEOUT:-15.0}"

python scripts/phase3_12d_r23_analyze_observation_contract.py \
  --root "$ROOT" \
  --data data/phase3_12d_r23_observation_contract/trajectory_observations.npz \
  --primary-steps ${PHASE3_12D_R23_PRIMARY_STEPS:-0 1 5 20} \
  --max-step "${PHASE3_12D_R23_MAX_STEP:-20}" \
  --th 3 \
  --action-dim 14 \
  --physics-hz "${PHASE3_12D_R23_PHYSICS_HZ:-480.0}" \
  --folds "${PHASE3_12D_R23_FOLDS:-5}" \
  --repeats "${PHASE3_12D_R23_REPEATS:-5}" \
  --bootstraps "${PHASE3_12D_R23_BOOTSTRAPS:-10000}" \
  --out-prefix reports/phase3_12d_r23

cat > reports/phase3_12d_r23_no_phase4_confirmation.md <<EOF
# Phase3.12d-r2.3 No Phase4 / No CPS Confirmation

- Timestamp: \`$(date -Is)\`
- No physical task parameter was changed.
- No formal state schema was activated.
- No future DDPM or IDM was trained.
- No candidate matrix was run.
- No Phase4 was run.
- No CPS was run.
- Simulator bead velocity was evaluated only as a privileged diagnostic.
- Any future state-v2 dataset must be versioned and regenerated from the
  deferred-arming environment.
EOF

echo "[r2.3] completed"
