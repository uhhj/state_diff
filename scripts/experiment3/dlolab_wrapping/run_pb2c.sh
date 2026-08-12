#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CONFIG="${1:-configs/experiment3/published_benchmark/dlolab_wrapping_pb2c.json}"
RAW="/data/Experiment3/data/pb2c_dlolab_wrapping"

INITIAL="${RAW}/rollouts_initial.npz"
EXTRA="${RAW}/rollouts_scaleup_extra.npz"

cd "${ROOT}"

mkdir -p "${RAW}"

if [[ ! -f "${INITIAL}" ]]; then
  echo "[PB2-C] collect initial 4 batches / 128 rollouts"
  python -m scripts.experiment3.dlolab_wrapping.collect_natural_rollouts_pb2c \
    --config "${CONFIG}" \
    --start-batch 0 \
    --batches 4 \
    --output "${INITIAL}"
else
  echo "[PB2-C] reuse complete initial rollout file"
fi

python -m scripts.experiment3.dlolab_wrapping.pair_discovery_pb2c \
  --config "${CONFIG}" \
  --datasets "${INITIAL}" \
  --label initial

COUNT="$(
python - <<'PY'
import json
from pathlib import Path

p = Path(
    "/data/Experiment3/data/pb2c_dlolab_wrapping/MINING_INITIAL.json"
)
row = json.loads(p.read_text(encoding="utf-8"))
print(int(row["candidate_count"]))
PY
)"

if (( COUNT >= 5 )); then
  echo "[PB2-C] initial collection found ${COUNT} candidates; no scale-up"
  exit 0
fi

if [[ ! -f "${EXTRA}" ]]; then
  echo "[PB2-C] initial count ${COUNT} < 5; collect only remaining 12 batches"
  python -m scripts.experiment3.dlolab_wrapping.collect_natural_rollouts_pb2c \
    --config "${CONFIG}" \
    --start-batch 4 \
    --batches 12 \
    --output "${EXTRA}"
else
  echo "[PB2-C] reuse complete scale-up rollout file"
fi

python -m scripts.experiment3.dlolab_wrapping.pair_discovery_pb2c \
  --config "${CONFIG}" \
  --datasets "${INITIAL}" "${EXTRA}" \
  --label final
