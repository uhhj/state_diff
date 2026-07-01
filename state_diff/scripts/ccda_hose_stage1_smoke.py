from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from state_diff.env.ccda_hose.config import FREE_INSERT, RIGHT_HIDDEN_JAM, HoseEnvConfig
from state_diff.env.ccda_hose.env import scripted_rollout, summarize_rollouts


def main() -> None:
    cfg = HoseEnvConfig(push_steps=25, approach_steps=15)
    rollouts = [
        scripted_rollout(FREE_INSERT, seed=0, config=cfg, record_frames=False),
        scripted_rollout(RIGHT_HIDDEN_JAM, seed=1, config=cfg, record_frames=False),
    ]
    summary = summarize_rollouts(rollouts)
    print(json.dumps(summary["by_condition"], indent=2))
    for r in rollouts:
        tr = r["trace"]
        print(
            r["condition"],
            "branch=", r["final_branch"],
            "success=", r["final_success"],
            "final_depth=", float(tr["insertion_depth"][-1, 0]),
            "final_curvature=", float(tr["max_curvature"][-1, 0]),
        )


if __name__ == "__main__":
    main()
