#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
COMMAND="${1:-screen}"
CONFIG="${2:-configs/experiment3/published_benchmark/dlolab_wrapping_pb3b1.json}"

cd "${ROOT}"

python -m \
  scripts.experiment3.dlolab_wrapping.prospective_live_cohort_pb3b1 \
  --config "${CONFIG}" \
  "${COMMAND}"
