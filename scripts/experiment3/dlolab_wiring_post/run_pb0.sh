#!/usr/bin/env bash
set -euo pipefail

if [ -n "${CONDA_PREFIX:-}" ]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

CONFIG="${1:-configs/experiment3/published_benchmark/dlolab_wiring_post_pb0.json}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

DLO="$ROOT/external/dlo-lab"
EXP="$DLO/experiments"

BEST_QPOS="$EXP/logs/wiring_post/cmaes-01/best_qpos.npy"
BEST_TRAJ="$EXP/logs/wiring_post/cmaes-01/best_traj.npy"

RAW="/data/Experiment3/data/pb0_dlolab_wiring_post"

mkdir -p "$RAW"

if [ ! -f "$BEST_QPOS" ] || [ ! -f "$BEST_TRAJ" ]; then
  (
    cd "$EXP"
    bash scripts/cmaes/wiring_post.sh
  )
fi

test -f "$BEST_QPOS"
test -f "$BEST_TRAJ"

python \
  scripts/experiment3/dlolab_wiring_post/collect_rollouts.py \
  --config "$CONFIG"

python \
  scripts/experiment3/dlolab_wiring_post/mine_pairs.py \
  --config "$CONFIG" \
  --dataset "$RAW/rollouts.npz"

COUNT="$(
python - <<'PY'
import json
path = "/data/Experiment3/data/pb0_dlolab_wiring_post/PAIR_CANDIDATES.json"
print(json.load(open(path))["candidate_count"])
PY
)"

MINIMUM="$(
python - <<'PY'
import json
path = "configs/experiment3/published_benchmark/dlolab_wiring_post_pb0.json"
print(json.load(open(path))["pair_mining"]["min_candidates_to_continue"])
PY
)"

if [ "$COUNT" -ge "$MINIMUM" ]; then
  python \
    scripts/experiment3/dlolab_wiring_post/build_result.py \
    --config "$CONFIG" \
    --pairs "$RAW/PAIR_CANDIDATES.json"
else
  python \
    scripts/experiment3/dlolab_wiring_post/collect_rollouts.py \
    --config "$CONFIG" \
    --batches 16

  python \
    scripts/experiment3/dlolab_wiring_post/mine_pairs.py \
    --config "$CONFIG" \
    --dataset "$RAW/rollouts_scaleup.npz"

  python \
    scripts/experiment3/dlolab_wiring_post/build_result.py \
    --config "$CONFIG" \
    --pairs "$RAW/PAIR_CANDIDATES_SCALEUP.json"
fi
