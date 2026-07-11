#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/scripts:$ROOT/external/deformable-ravens:${PYTHONPATH:-}"

for gate in \
  PHASE3_ALLOW_R24_SUBMODULE_CHANGE \
  PHASE3_R24_SUBMODULE_CHANGE_CONFIRMED \
  PHASE3_ALLOW_R24_PARAMETER_SWEEP \
  PHASE3_R24_PARAMETER_SWEEP_CONFIRMED \
  PHASE3_ALLOW_R24_ENVIRONMENT_AUDIT \
  PHASE3_R24_ENVIRONMENT_AUDIT_CONFIRMED; do
  if [[ "${!gate:-0}" != "1" ]]; then
    echo "[r2.4][BLOCKED] missing gate: $gate"
    exit 1
  fi
done

test -s GOALS.md

python -m py_compile \
  external/deformable-ravens/ravens/tasks/ccda_slack_breakaway.py \
  external/deformable-ravens/ravens/tasks/ccda_hidden_contact_cable.py \
  external/deformable-ravens/ravens/environment.py \
  scripts/phase3_12d_r24_*.py \
  tests/test_phase3_12d_r24_slack_breakaway.py

pytest -q tests/test_phase3_12d_r24_slack_breakaway.py
python scripts/phase3_12d_r24_static_selftest.py

if grep -Eq 'createConstraint|createCollisionShape|createMultiBody' \
  external/deformable-ravens/ravens/tasks/ccda_slack_breakaway.py; then
  echo "[r2.4][FAIL] forbidden PyBullet geometry API in v2 helper"
  exit 1
fi

python scripts/phase3_12d_r24_preflight.py
python scripts/phase3_12d_r24_parameter_sweep.py
python scripts/phase3_12d_r24_environment_audit.py
python scripts/phase3_12d_r24_collect_no_action.py \
  --num-seeds "${PHASE3_12D_R24_OBSERVATION_SEEDS:-128}" \
  --seed-start "${PHASE3_12D_R24_OBSERVATION_SEED_START:-316000}" \
  --workers "${PHASE3_12D_R24_WORKERS:-4}"
python scripts/phase3_12d_r24_observation_audit.py \
  --bootstraps "${PHASE3_12D_R24_BOOTSTRAPS:-10000}"
python scripts/phase3_12d_r24_snapshot_audit.py
python scripts/phase3_12d_r24_analyze.py

echo "[r2.4] completed"
