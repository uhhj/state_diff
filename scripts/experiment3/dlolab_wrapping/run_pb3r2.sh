#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CONFIG="${1:-configs/experiment3/published_benchmark/dlolab_wrapping_pb3r2.json}"

cd "${ROOT}"

python -m \
  scripts.experiment3.dlolab_wrapping.replay_floor_calibration_pb3r2 \
  run \
  --config "${CONFIG}"
