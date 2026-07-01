#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFRAVENS_ROOT="${DEFRAVENS_ROOT:-$ROOT/external/deformable-ravens}"
TASK="${TASK:-cable-line-notarget}"
HZ="${HZ:-240}"
GOALS="${GOALS:-20}"
FRESH="${FRESH:-1}"

echo "[Phase0] root: $ROOT"
echo "[Phase0] defravens: $DEFRAVENS_ROOT"
echo "[Phase0] task: $TASK"
echo "[Phase0] hz: $HZ"
echo "[Phase0] goals: $GOALS"
echo "[Phase0] fresh: $FRESH"

if [[ ! -d "$DEFRAVENS_ROOT" ]]; then
  echo "[Phase0][ERROR] Missing $DEFRAVENS_ROOT"
  echo "Run: bash scripts/phase0_defravens_setup.sh"
  exit 1
fi

cd "$ROOT"

echo "[Phase0] running import smoke test..."
python scripts/phase0_defravens_smoke.py

echo "[Phase0] creating main_phase0_smoke.py with MAX_ORDER=1..."
python scripts/phase0_make_defravens_smoke_main.py

cd "$DEFRAVENS_ROOT"

if [[ "$FRESH" == "1" ]]; then
  echo "[Phase0] deleting old smoke data/goals for task=$TASK"
  rm -rf "data/$TASK" "goals/$TASK"
fi

echo "[Phase0] generating 10 smoke demos..."
python main_phase0_smoke.py \
  --gpu=0 \
  --agent=dummy \
  --hz="$HZ" \
  --task="$TASK"

echo "[Phase0] generating goals..."
python generate_goals.py \
  --hz="$HZ" \
  --task="$TASK" \
  --num_goals="$GOALS"

cd "$ROOT"

echo "[Phase0] inspecting generated data..."
TASK="$TASK" DEFRAVENS_ROOT="$DEFRAVENS_ROOT" python scripts/phase0_inspect_defravens_data.py

echo "[Phase0] visualizing generated demos..."
TASK="$TASK" DEFRAVENS_ROOT="$DEFRAVENS_ROOT" python scripts/phase0_visualize_defravens_data.py

echo "[Phase0] done."
echo "Report:"
echo "  reports/phase0_defravens_setup.md"
echo "  reports/phase0_defravens_summary.json"
echo "  reports/phase0_visualizations/$TASK/"
