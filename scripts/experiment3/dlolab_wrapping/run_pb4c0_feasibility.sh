#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
COMMAND="${1:-derive-family}"
CONFIG="${2:-configs/experiment3/published_benchmark/dlolab_wrapping_pb4c0_feasibility.json}"

cd "${ROOT}"

python -m \
  scripts.experiment3.dlolab_wrapping.feasibility_normalized_qpos_family_pb4c0 \
  --config "${CONFIG}" \
  "${COMMAND}"
