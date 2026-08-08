#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/experiment3/phase0f/dhr_control_smoke.json}"

python \
  scripts/experiment3/phase0f_dhr_control_smoke/run_control_smoke.py \
  --config "$CONFIG"

python \
  scripts/experiment3/phase0f_dhr_control_smoke/analyze_control_smoke.py \
  --config "$CONFIG"
