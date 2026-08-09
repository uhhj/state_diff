#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/experiment3/published_benchmark/dlolab_wiring_post_pb0s_rev3.json}"

python \
  scripts/experiment3/dlolab_wiring_post/screening_audit_rev3.py \
  --config "$CONFIG"
