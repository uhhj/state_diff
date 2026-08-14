#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
COMMAND="${1:-run}"
CONFIG="${2:-configs/experiment3/published_benchmark/dlolab_wrapping_pb4c1.json}"

cd "${ROOT}"

python -m \
  scripts.experiment3.dlolab_wrapping.normalized_qpos_control_exploration_pb4c1 \
  --config "${CONFIG}" \
  "${COMMAND}"
