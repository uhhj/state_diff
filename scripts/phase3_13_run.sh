#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/scripts:$ROOT/external/deformable-ravens:${PYTHONPATH:-}"

for gate in \
  PHASE313_ALLOW_DATA_GENERATION \
  PHASE313_DATA_GENERATION_CONFIRMED \
  PHASE313_ALLOW_LEGACY_PURGE \
  PHASE313_LEGACY_PURGE_CONFIRMED; do
  if [[ "${!gate:-0}" != "1" ]]; then
    echo "[Phase3.13][BLOCKED] missing gate: $gate"
    exit 1
  fi
done

python -m py_compile ccda_phase3/schema_v2.py ccda_phase3/data_io.py \
  ccda_phase3/observation_contract.py ccda_phase3/rollout.py \
  scripts/phase3_13_*.py \
  external/deformable-ravens/ravens/tasks/ccda_slack_breakaway.py \
  external/deformable-ravens/ravens/tasks/ccda_slack_cable_v2.py \
  external/deformable-ravens/ravens/environment.py \
  external/deformable-ravens/ravens/tasks/__init__.py

pytest -q tests/test_ccda_slack_breakaway.py tests/test_ccda_state_v2.py \
  tests/test_ccda_dataset_windows.py tests/test_phase3_13_legacy_purge.py

python scripts/phase3_13_task_smoke.py

SMOKE_ROOT="data/phase3_state_v2_slack_smoke"
python scripts/phase3_13_generate_raw.py --split train --seed-start 390000 \
  --num-seeds 8 --output-root "$SMOKE_ROOT/raw" --workers "${PHASE313_WORKERS:-4}" --fresh --smoke
python scripts/phase3_13_generate_raw.py --split val --seed-start 390100 \
  --num-seeds 4 --output-root "$SMOKE_ROOT/raw" --workers "${PHASE313_WORKERS:-4}" --fresh --smoke
python scripts/phase3_13_generate_raw.py --split test --seed-start 390200 \
  --num-seeds 4 --output-root "$SMOKE_ROOT/raw" --workers "${PHASE313_WORKERS:-4}" --fresh --smoke
python scripts/phase3_13_build_windows.py --raw-root "$SMOKE_ROOT/raw" \
  --output-dir "$SMOKE_ROOT/windows" --manifest "$SMOKE_ROOT/manifest.json"
python scripts/phase3_13_audit_dataset.py --raw-root "$SMOKE_ROOT/raw" \
  --windows "$SMOKE_ROOT/windows/phase3_13_windows.npz" \
  --template "$SMOKE_ROOT/windows/action_template.pkl" --smoke \
  --out-prefix reports/phase3_13_smoke

python scripts/phase3_13_legacy_purge.py --root "$ROOT"
if [[ "${PHASE313_LEGACY_PURGE_CONFIRM:-}" != "DELETE_PHASE3_LEGACY_ASSETS" ]]; then
  echo "[Phase3.13][BLOCKED] destructive purge confirmation missing"
  exit 1
fi
python scripts/phase3_13_legacy_purge.py --root "$ROOT" --apply

pytest -q tests/test_ccda_slack_breakaway.py tests/test_ccda_state_v2.py \
  tests/test_ccda_dataset_windows.py tests/test_phase3_13_legacy_purge.py

RAW="data/phase3_state_v2_slack/raw"
python scripts/phase3_13_generate_raw.py --split train --seed-start 400000 \
  --num-seeds 256 --output-root "$RAW" --workers "${PHASE313_WORKERS:-4}" --fresh
python scripts/phase3_13_generate_raw.py --split val --seed-start 410000 \
  --num-seeds 64 --output-root "$RAW" --workers "${PHASE313_WORKERS:-4}" --fresh
python scripts/phase3_13_generate_raw.py --split test --seed-start 420000 \
  --num-seeds 128 --output-root "$RAW" --workers "${PHASE313_WORKERS:-4}" --fresh
python scripts/phase3_13_build_windows.py
python scripts/phase3_13_audit_dataset.py

python scripts/phase3_13_fresh_worktree_check.py
python scripts/phase3_13_finalize.py

echo "[Phase3.13] complete"
