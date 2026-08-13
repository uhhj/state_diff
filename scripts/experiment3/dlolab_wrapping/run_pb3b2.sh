#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CONFIG="${1:-configs/experiment3/published_benchmark/dlolab_wrapping_pb3b2.json}"

cd "${ROOT}"
python -m scripts.experiment3.dlolab_wrapping.prospective_causal_audit_pb3b2 \
  --config "${CONFIG}"
