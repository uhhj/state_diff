#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/experiment3/published_benchmark/dlolab_wiring_post_pb0s_rev4.json}"

python \
  scripts/experiment3/dlolab_wiring_post/hidden_descriptor_robustness_rev4.py \
  --config "$CONFIG"
