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
PHASE3_CONDITIONS="${PHASE3_CONDITIONS:-free hidden_pin hidden_high_friction hidden_breakaway_pin}"
PHASE3_PRIMARY_HIDDEN_CONDITION="${PHASE3_PRIMARY_HIDDEN_CONDITION:-hidden_breakaway_pin}"
PHASE3_DIAGNOSTIC_HIDDEN_CONDITION="${PHASE3_DIAGNOSTIC_HIDDEN_CONDITION:-hidden_pin}"
PHASE2_5_SELECTED_CONFIG="${PHASE2_5_SELECTED_CONFIG:-breakaway_force_2p6_disp_0p045_pull_0p36}"
export CCDA_BREAKAWAY_FORCE="${CCDA_BREAKAWAY_FORCE:-2.6}"
export CCDA_BREAKAWAY_DISP="${CCDA_BREAKAWAY_DISP:-0.045}"
export CCDA_BREAKAWAY_BEAD_RATIO="${CCDA_BREAKAWAY_BEAD_RATIO:-0.45}"
export CCDA_ORACLE_BREAKAWAY_PULL_DIST="${CCDA_ORACLE_BREAKAWAY_PULL_DIST:-0.36}"
export PHASE3_CONDITIONS PHASE3_PRIMARY_HIDDEN_CONDITION PHASE3_DIAGNOSTIC_HIDDEN_CONDITION PHASE2_5_SELECTED_CONFIG

TRAIN_OUT="${TRAIN_OUT:-data/phase3_ccda_large/train}"
HELDOUT_OUT="${HELDOUT_OUT:-data/phase3_ccda_large/heldout}"

cd "$DEFRAVENS_ROOT"

if [[ ! -f ccda_generate_hidden_contact.py ]]; then
  echo "[Phase3][ERROR] missing ccda_generate_hidden_contact.py"
  exit 1
fi

python - <<'PY'
import os
from ravens import tasks
assert "hidden-contact-cable-line" in tasks.names, "hidden-contact-cable-line not registered"
t = tasks.names["hidden-contact-cable-line"]()
requested = os.environ.get("PHASE3_CONDITIONS", "").split()
missing = [c for c in requested if c not in t.CONDITIONS]
assert "hidden_side_jam" not in t.CONDITIONS
assert not missing, f"requested Phase3 conditions missing from task: {missing}; task has {t.CONDITIONS}"
print("[Phase3] task registered", t.CONDITIONS)
print("[Phase3] requested conditions", requested)
PY

python -m pip install -e .

mkdir -p "$ROOT/reports"
cat > "$ROOT/reports/phase3_generation_config.md" <<EOF
# Phase3 Generation Configuration

- Conditions: \`$PHASE3_CONDITIONS\`
- Primary hidden condition: \`$PHASE3_PRIMARY_HIDDEN_CONDITION\`
- Diagnostic hidden condition: \`$PHASE3_DIAGNOSTIC_HIDDEN_CONDITION\`
- Selected Phase2.5 config: \`$PHASE2_5_SELECTED_CONFIG\`
- CCDA_BREAKAWAY_FORCE: \`$CCDA_BREAKAWAY_FORCE\`
- CCDA_BREAKAWAY_DISP: \`$CCDA_BREAKAWAY_DISP\`
- CCDA_BREAKAWAY_BEAD_RATIO: \`$CCDA_BREAKAWAY_BEAD_RATIO\`
- CCDA_ORACLE_BREAKAWAY_PULL_DIST: \`$CCDA_ORACLE_BREAKAWAY_PULL_DIST\`
EOF

if [[ "$FRESH" == "1" ]]; then
  rm -rf "$DEFRAVENS_ROOT/$TRAIN_OUT/$TASK"
  rm -rf "$DEFRAVENS_ROOT/$HELDOUT_OUT/$TASK"
fi

echo "[Phase3] generating training dataset"
python ccda_generate_hidden_contact.py \
  --task "$TASK" \
  --conditions $PHASE3_CONDITIONS \
  --num_demos "$TRAIN_SEEDS" \
  --seed_start "$TRAIN_SEED_START" \
  --hz "$HZ" \
  --output_root "$TRAIN_OUT" \
  --fresh

echo "[Phase3] generating held-out dataset"
python ccda_generate_hidden_contact.py \
  --task "$TASK" \
  --conditions $PHASE3_CONDITIONS \
  --num_demos "$HELDOUT_SEEDS" \
  --seed_start "$HELDOUT_SEED_START" \
  --hz "$HZ" \
  --output_root "$HELDOUT_OUT" \
  --fresh

echo "[Phase3] generated:"
echo "  $DEFRAVENS_ROOT/$TRAIN_OUT/$TASK"
echo "  $DEFRAVENS_ROOT/$HELDOUT_OUT/$TASK"
