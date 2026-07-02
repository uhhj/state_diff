#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFRAVENS_ROOT="${DEFRAVENS_ROOT:-$ROOT/external/deformable-ravens}"
TASK="${TASK:-hidden-contact-cable-line}"
VISIBLE_SEED="${VISIBLE_SEED:-0}"
HZ="${HZ:-240}"
MAX_STEPS="${MAX_STEPS:-5}"
FPS="${FPS:-20}"
STRIDE="${STRIDE:-4}"
FRESH="${FRESH:-1}"
OUTDIR="$ROOT/reports/phase1_continuous_videos"

CONDITIONS=(free hidden_pin hidden_high_friction)

echo "[Phase1.1] root: $ROOT"
echo "[Phase1.1] defravens: $DEFRAVENS_ROOT"
echo "[Phase1.1] task: $TASK"
echo "[Phase1.1] visible seed: $VISIBLE_SEED"
echo "[Phase1.1] max steps: $MAX_STEPS"
echo "[Phase1.1] fps: $FPS"
echo "[Phase1.1] stride: $STRIDE"

if [[ ! -d "$DEFRAVENS_ROOT" ]]; then
  echo "[Phase1.1][ERROR] Missing DeformableRavens root: $DEFRAVENS_ROOT"
  exit 1
fi

mkdir -p "$ROOT/reports"
if [[ "$FRESH" == "1" ]]; then
  rm -rf "$OUTDIR"
fi
mkdir -p "$OUTDIR"

cd "$DEFRAVENS_ROOT"
python ccda_record_continuous_rollout.py \
  --task "$TASK" \
  --visible_seed "$VISIBLE_SEED" \
  --conditions "${CONDITIONS[@]}" \
  --hz "$HZ" \
  --max_steps "$MAX_STEPS" \
  --fps "$FPS" \
  --stride "$STRIDE" \
  --outdir "$OUTDIR"

echo "[Phase1.1] done."
echo "Outputs:"
echo "  reports/phase1_continuous_videos/"
echo "  reports/phase1_continuous_video_summary.json"
echo "  reports/phase1_continuous_video_report.md"
