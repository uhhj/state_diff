#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/experiment3/phase0e/ohj_r13_scientific_audit.json}"

python scripts/experiment3/phase0e_ohj_audit/run_control_matrix_audit.py \
  --config "$CONFIG"

python scripts/experiment3/phase0e_ohj_audit/analyze_scientific_audit.py \
  --config "$CONFIG"
