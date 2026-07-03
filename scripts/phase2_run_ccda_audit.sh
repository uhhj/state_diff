#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFRAVENS_ROOT="${DEFRAVENS_ROOT:-$ROOT/external/deformable-ravens}"

TASK="${TASK:-hidden-contact-cable-line}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/phase2_hidden_contact_cable_line}"
NUM_DEMOS="${NUM_DEMOS:-50}"
SEED_START="${SEED_START:-0}"
HZ="${HZ:-240}"
FRESH="${FRESH:-1}"

TAU_VIS="${TAU_VIS:-0.002}"
TAU_ACTION="${TAU_ACTION:-0.02}"
TAU_FUTURE="${TAU_FUTURE:-0.05}"
TAU_RGB_MEAN="${TAU_RGB_MEAN:-1.0}"
TAU_DEPTH_MEAN="${TAU_DEPTH_MEAN:-1e-4}"

echo "[Phase2] root: $ROOT"
echo "[Phase2] defravens: $DEFRAVENS_ROOT"
echo "[Phase2] task: $TASK"
echo "[Phase2] output_root inside submodule: $OUTPUT_ROOT"
echo "[Phase2] num demos per condition: $NUM_DEMOS"
echo "[Phase2] seed_start: $SEED_START"
echo "[Phase2] hz: $HZ"
echo "[Phase2] fresh: $FRESH"

cd "$DEFRAVENS_ROOT"

if [[ ! -f "ccda_generate_hidden_contact.py" ]]; then
  echo "[Phase2][ERROR] Missing ccda_generate_hidden_contact.py."
  echo "Phase1 hidden-contact generator is required."
  exit 1
fi

python - <<'PY'
from ravens import tasks
name = "hidden-contact-cable-line"
print("[Phase2] task registered:", name in tasks.names)
if name not in tasks.names:
    raise SystemExit("[Phase2][ERROR] hidden-contact-cable-line is not registered.")
print("[Phase2] task class:", tasks.names[name])
PY

python -m pip install -e .

if [[ "$FRESH" == "1" ]]; then
  echo "[Phase2] removing previous dataset: $DEFRAVENS_ROOT/$OUTPUT_ROOT/$TASK"
  rm -rf "$DEFRAVENS_ROOT/$OUTPUT_ROOT/$TASK"
fi

echo "[Phase2] generating paired dataset..."
python ccda_generate_hidden_contact.py \
  --task "$TASK" \
  --conditions free hidden_pin hidden_high_friction \
  --num_demos "$NUM_DEMOS" \
  --seed_start "$SEED_START" \
  --hz "$HZ" \
  --output_root "$OUTPUT_ROOT" \
  --fresh

cd "$ROOT"

DATA_ROOT="$DEFRAVENS_ROOT/$OUTPUT_ROOT/$TASK"

echo "[Phase2] auditing CCDA pairs..."
python scripts/phase2_audit_ccda_pairs.py \
  --root "$ROOT" \
  --data_root "$DATA_ROOT" \
  --task "$TASK" \
  --tau_vis "$TAU_VIS" \
  --tau_action "$TAU_ACTION" \
  --tau_future "$TAU_FUTURE" \
  --tau_rgb_mean "$TAU_RGB_MEAN" \
  --tau_depth_mean "$TAU_DEPTH_MEAN"

echo "[Phase2] visualizing representative CCDA examples..."
python scripts/phase2_visualize_ccda_examples.py \
  --root "$ROOT" \
  --data_root "$DATA_ROOT" \
  --task "$TASK" \
  --max_examples 8

echo "[Phase2] done."
echo "Outputs:"
echo "  reports/phase2_ccda_audit_summary.json"
echo "  reports/phase2_ccda_audit_report.md"
echo "  reports/phase2_ccda_records.csv"
echo "  reports/phase2_condition_summary.csv"
echo "  reports/phase2_rgbd_leakage_summary.csv"
echo "  reports/phase2_ccda_visuals/"
