#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFRAVENS_ROOT="${DEFRAVENS_ROOT:-$ROOT/external/deformable-ravens}"
TASK="${TASK:-hidden-contact-cable-line}"
HZ="${HZ:-240}"

TRAIN_SEEDS="${TRAIN_SEEDS:-500}"
HELDOUT_SEEDS="${HELDOUT_SEEDS:-100}"
TRAIN_SEED_START="${TRAIN_SEED_START:-1000}"
HELDOUT_SEED_START="${HELDOUT_SEED_START:-100000}"
FRESH="${FRESH:-1}"

TRAIN_OUT="${TRAIN_OUT:-data/phase3_ccda_large/train}"
HELDOUT_OUT="${HELDOUT_OUT:-data/phase3_ccda_large/heldout}"

cd "$DEFRAVENS_ROOT"

if [[ ! -f ccda_generate_hidden_contact.py ]]; then
  echo "[Phase3][ERROR] missing ccda_generate_hidden_contact.py"
  exit 1
fi

python - <<'PY'
from ravens import tasks
assert "hidden-contact-cable-line" in tasks.names, "hidden-contact-cable-line not registered"
t = tasks.names["hidden-contact-cable-line"]()
assert "hidden_side_jam" not in t.CONDITIONS
print("[Phase3] task registered", t.CONDITIONS)
PY

python -m pip install -e .

if [[ "$FRESH" == "1" ]]; then
  rm -rf "$DEFRAVENS_ROOT/$TRAIN_OUT/$TASK"
  rm -rf "$DEFRAVENS_ROOT/$HELDOUT_OUT/$TASK"
fi

echo "[Phase3] generating training dataset"
python ccda_generate_hidden_contact.py \
  --task "$TASK" \
  --conditions free hidden_pin hidden_high_friction \
  --num_demos "$TRAIN_SEEDS" \
  --seed_start "$TRAIN_SEED_START" \
  --hz "$HZ" \
  --output_root "$TRAIN_OUT" \
  --fresh

echo "[Phase3] generating held-out dataset"
python ccda_generate_hidden_contact.py \
  --task "$TASK" \
  --conditions free hidden_pin hidden_high_friction \
  --num_demos "$HELDOUT_SEEDS" \
  --seed_start "$HELDOUT_SEED_START" \
  --hz "$HZ" \
  --output_root "$HELDOUT_OUT" \
  --fresh

echo "[Phase3] generated:"
echo "  $DEFRAVENS_ROOT/$TRAIN_OUT/$TASK"
echo "  $DEFRAVENS_ROOT/$HELDOUT_OUT/$TASK"
