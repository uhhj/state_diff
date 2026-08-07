#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/experiment3/phase0/ohj_cable_phase0d.json}"

python -m scripts.experiment3.phase0_ohj_cable.run_pair --config "$CONFIG"
python -m scripts.experiment3.phase0_ohj_cable.analyze_pair --config "$CONFIG"

VERDICT="$(python - "$CONFIG" <<'PY'
import json, sys
from pathlib import Path
cfg = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
path = Path(cfg["report_root"]) / cfg["pair_id"] / "pair_metrics.json"
print(json.loads(path.read_text(encoding="utf-8"))["verdict"])
PY
)"

if [ "$VERDICT" = "PHASE0D_PAIR_COMPLETE" ]; then
  python -m scripts.experiment3.phase0_ohj_cable.run_control_relevance \
    --config "$CONFIG"
  python -m scripts.experiment3.phase0_ohj_cable.analyze_control_relevance \
    --config "$CONFIG"
fi

python -m scripts.experiment3.phase0_ohj_cable.build_result --config "$CONFIG"
