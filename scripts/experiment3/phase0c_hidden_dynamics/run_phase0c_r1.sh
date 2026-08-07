#!/usr/bin/env bash
set -euo pipefail

CONFIG="configs/experiment3/hlf_sbp_phase0c_r1_highrate.json"

python -m scripts.experiment3.phase0c_hidden_dynamics.run_pair \
  --config "$CONFIG"

python -m scripts.experiment3.phase0c_hidden_dynamics.analyze_highrate_pair \
  --config "$CONFIG"

VERDICT="$(python - "$CONFIG" <<'PY'
import json, sys
from pathlib import Path
cfg = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
path = Path(cfg["report_root"]) / cfg["pair_id"] / "highrate_pair_metrics.json"
print(json.loads(path.read_text(encoding="utf-8"))["verdict"])
PY
)"

if [ "$VERDICT" = "PHASE0C_R1_SENSOR_OBSERVABILITY_COMPLETE" ]; then
  python -m scripts.experiment3.phase0c_hidden_dynamics.run_control_relevance \
    --config "$CONFIG"
  python -m scripts.experiment3.phase0c_hidden_dynamics.analyze_control_relevance \
    --config "$CONFIG"
fi

python -m scripts.experiment3.phase0c_hidden_dynamics.build_result \
  --config "$CONFIG"
