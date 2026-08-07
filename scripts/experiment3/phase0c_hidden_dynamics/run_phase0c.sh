#!/usr/bin/env bash
set -euo pipefail

CONFIG="configs/experiment3/hlf_sbp_phase0c_dynamics.json"

python -m scripts.experiment3.phase0c_hidden_dynamics.run_pair --config "$CONFIG"
python -m scripts.experiment3.phase0c_hidden_dynamics.analyze_pair --config "$CONFIG"

python - "$CONFIG" <<'PY'
import json
import sys
from pathlib import Path
cfg = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
path = Path(cfg["report_root"]) / cfg["pair_id"] / "pair_metrics.json"
result = json.loads(path.read_text(encoding="utf-8"))
if result["verdict"] != "PHASE0C_PAIR_COMPLETE":
    raise SystemExit("pair gate not complete")
PY

python -m scripts.experiment3.phase0c_hidden_dynamics.run_control_relevance --config "$CONFIG"
python -m scripts.experiment3.phase0c_hidden_dynamics.analyze_control_relevance --config "$CONFIG"
python -m scripts.experiment3.phase0c_hidden_dynamics.build_result --config "$CONFIG"
