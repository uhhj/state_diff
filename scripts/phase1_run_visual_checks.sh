#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASK="${TASK:-hidden-contact-cable-line}"
CAMERA_INDEX="${CAMERA_INDEX:-0}"
MAX_GROUPS="${MAX_GROUPS:-20}"

cd "$ROOT"

python scripts/phase1_visualize_rgbd_check.py \
  --root "$ROOT" \
  --task "$TASK" \
  --camera_index "$CAMERA_INDEX" \
  --max_groups "$MAX_GROUPS"

python scripts/phase1_visualize_hidden_contact.py \
  --root "$ROOT" \
  --task "$TASK" \
  --max_groups "$MAX_GROUPS"
