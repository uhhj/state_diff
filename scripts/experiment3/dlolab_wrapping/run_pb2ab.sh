#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CONFIG="${1:-configs/experiment3/published_benchmark/dlolab_wrapping_pb2ab.json}"
EXP="${ROOT}/external/dlo-lab/experiments"
LOG="${EXP}/logs/wrapping/cmaes-01"

cd "${ROOT}"

status="$(
python - "${LOG}" <<'PY'
import json
from pathlib import Path
import sys
import numpy as np

log = Path(sys.argv[1])
cfg_path = log / "run_config.json"
meta_path = log / "resume_meta.json"

# Important distinction:
# - no log directory: clean first official run is allowed;
# - existing log directory but no run_config: provenance/config is unknown -> STOP.
if not log.exists():
    print("LOG_ABSENT")
    raise SystemExit

if not cfg_path.is_file():
    print("EXISTING_LOG_MISSING_RUN_CONFIG")
    raise SystemExit

run = json.loads(cfg_path.read_text(encoding="utf-8"))
expected = {
    "n_envs": 100,
    "n_steps": 10,
    "n_steps_sub": 10,
    "act_dim": 12,
    "popsize": 400,
    "sigma0": 0.05,
    "per_comp_bound": 0.1,
    "l2_bound": 0.1,
    "angle_bound": 1.0,
    "angle_scale": [1.0, 1.0, 1.0],
    "max_iters": 51,
    "seed": 123,
    "use_last_state_reward": False,
    "randomized_args": None,
}

def same(a, b):
    if isinstance(b, float):
        try:
            return bool(np.isclose(float(a), b, rtol=0.0, atol=1e-12))
        except Exception:
            return False
    if isinstance(b, list):
        try:
            return bool(np.allclose(
                np.asarray(a, dtype=float),
                np.asarray(b, dtype=float),
                rtol=0.0,
                atol=1e-12,
            ))
        except Exception:
            return False
    return a == b

missing = [key for key in expected if key not in run]
if missing:
    print("MISMATCH:" + json.dumps({
        "missing_fields": missing,
    }, sort_keys=True))
    raise SystemExit

bad = {
    key: [run[key], value]
    for key, value in expected.items()
    if not same(run[key], value)
}
if bad:
    print("MISMATCH:" + json.dumps(bad, sort_keys=True))
    raise SystemExit

required_outputs = [
    log / "best_traj.npy",
    log / "best_qpos.npy",
    meta_path,
]
if not all(path.is_file() for path in required_outputs):
    print("MATCHING_INCOMPLETE")
    raise SystemExit

meta = json.loads(meta_path.read_text(encoding="utf-8"))
print(
    "MATCHING_COMPLETE"
    if int(meta.get("iter", -1)) == 50
    else "MATCHING_INCOMPLETE"
)
PY
)"

case "${status}" in
  MATCHING_COMPLETE)
    echo "[PB2-A] reuse exact completed official Wrapping CMA-ES"
    ;;
  LOG_ABSENT)
    echo "[PB2-A] no existing Wrapping log; run exact pinned official CMA-ES"
    (cd "${EXP}" && bash scripts/cmaes/wrapping.sh)
    ;;

  MATCHING_INCOMPLETE)
    echo "[PB2-A] matching official Wrapping log is incomplete; resume exact pinned run"
    (cd "${EXP}" && bash scripts/cmaes/wrapping.sh)
    ;;

  EXISTING_LOG_MISSING_RUN_CONFIG)
    echo "[PB2-A] STOP: wrapping/cmaes-01 exists but run_config.json is missing"
    echo "Do not resume an optimizer state whose effective parameter contract cannot be verified."
    exit 2
    ;;
  MISMATCH:*)
    echo "[PB2-A] existing cmaes-01 has a different effective parameter set"
    echo "${status}"
    exit 2
    ;;
  *)
    echo "Unexpected optimizer status: ${status}"
    exit 3
    ;;
esac

python -m scripts.experiment3.dlolab_wrapping.replay_winding_audit \
  --config "${CONFIG}"
