#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFRAVENS_ROOT="${DEFRAVENS_ROOT:-$ROOT/external/deformable-ravens}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:-73982748bdfe756e04553e28698948facccecb41}"
TASK="${TASK:-hidden-contact-cable-line}"
NUM_DEMOS="${NUM_DEMOS:-5}"
SEED_START="${SEED_START:-0}"
HZ="${HZ:-240}"
FRESH="${FRESH:-1}"

echo "[Phase1] root: $ROOT"
echo "[Phase1] defravens: $DEFRAVENS_ROOT"
echo "[Phase1] task: $TASK"
echo "[Phase1] num demos per condition: $NUM_DEMOS"
echo "[Phase1] seed start: $SEED_START"
echo "[Phase1] hz: $HZ"

cd "$DEFRAVENS_ROOT"

CURRENT_COMMIT="$(git rev-parse HEAD)"
echo "[Phase1] current submodule commit: $CURRENT_COMMIT"

if [[ "$CURRENT_COMMIT" != "$EXPECTED_COMMIT" ]]; then
  echo "[Phase1][WARN] submodule commit differs from expected base."
  echo "[Phase1][WARN] expected: $EXPECTED_COMMIT"
  echo "[Phase1][WARN] current:  $CURRENT_COMMIT"
  echo "[Phase1][WARN] continuing because Phase1 may already contain edits."
fi

python -m pip install -e .

echo "[Phase1] smoke import task registration..."
python - <<'PY'
from ravens import tasks
name = "hidden-contact-cable-line"
print("registered:", name in tasks.names)
print("task class:", tasks.names.get(name))
assert name in tasks.names
PY

if [[ "$FRESH" == "1" ]]; then
  FRESH_ARG="--fresh"
else
  FRESH_ARG=""
fi

echo "[Phase1] generating paired hidden-contact data..."
python ccda_generate_hidden_contact.py \
  --task "$TASK" \
  --conditions free hidden_pin hidden_high_friction hidden_side_jam \
  --num_demos "$NUM_DEMOS" \
  --seed_start "$SEED_START" \
  --hz "$HZ" \
  $FRESH_ARG

cd "$ROOT"

echo "[Phase1] inspecting data..."
python scripts/phase1_inspect_hidden_contact.py \
  --root "$ROOT" \
  --task "$TASK"

echo "[Phase1] generating visualization..."
python scripts/phase1_visualize_hidden_contact.py \
  --root "$ROOT" \
  --task "$TASK" \
  --max_groups 5

echo "[Phase1] done."
echo "Reports:"
echo "  reports/phase1_hidden_contact_summary.json"
echo "  reports/phase1_hidden_contact_report.md"
echo "  reports/phase1_hidden_contact_visuals/"
