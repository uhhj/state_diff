#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/experiment3/published_benchmark/dlolab_wiring_post_pb0t.json}"

python \
  scripts/experiment3/dlolab_wiring_post/native_contact_targeted_replay_pb0t.py \
  --config "$CONFIG"
