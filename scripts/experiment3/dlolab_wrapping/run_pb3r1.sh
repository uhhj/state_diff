#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CONFIG="${1:-configs/experiment3/published_benchmark/dlolab_wrapping_pb3r1.json}"

cd "${ROOT}"

python -m \
  scripts.experiment3.dlolab_wrapping.replay_alignment_root_cause_pb3r1 \
  --config "${CONFIG}"
