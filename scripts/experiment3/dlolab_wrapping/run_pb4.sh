#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
COMMAND="${1:-run}"
CONFIG="${2:-configs/experiment3/published_benchmark/dlolab_wrapping_pb4.json}"

cd "${ROOT}"

python -m \
  scripts.experiment3.dlolab_wrapping.control_relevance_pb4 \
  --config "${CONFIG}" \
  "${COMMAND}"
