#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASK="${TASK:-hidden-contact-cable-line}"
CAMERA_INDEX="${CAMERA_INDEX:-0}"
MAX_GROUPS="${MAX_GROUPS:-20}"
FPS="${FPS:-2}"

echo "[Phase1] root: $ROOT"
echo "[Phase1] task: $TASK"
echo "[Phase1] camera index: $CAMERA_INDEX"
echo "[Phase1] max groups: $MAX_GROUPS"

cd "$ROOT"

DATA_ROOT="$ROOT/external/deformable-ravens/data/$TASK"

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "[Phase1][ERROR] Missing data root: $DATA_ROOT"
  echo "Run the hidden-contact data generation first:"
  echo "  bash scripts/phase1_defravens_run_hidden_contact.sh"
  exit 1
fi

echo "[Phase1] generating RGB-D observation check..."
python scripts/phase1_visualize_rgbd_check.py \
  --root "$ROOT" \
  --task "$TASK" \
  --camera_index "$CAMERA_INDEX" \
  --max_groups "$MAX_GROUPS"

echo "[Phase1] generating action-step execution videos..."
python scripts/phase1_make_action_videos.py \
  --root "$ROOT" \
  --task "$TASK" \
  --camera_index "$CAMERA_INDEX" \
  --max_groups "$MAX_GROUPS" \
  --fps "$FPS"

echo "[Phase1] done."
echo "Outputs:"
echo "  reports/phase1_rgbd_check/"
echo "  reports/phase1_rgbd_check_summary.json"
echo "  reports/phase1_rgbd_check_report.md"
echo "  reports/phase1_action_videos/"
echo "  reports/phase1_action_video_summary.json"
echo "  reports/phase1_action_video_report.md"
