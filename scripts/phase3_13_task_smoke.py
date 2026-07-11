#!/usr/bin/env python3
import json
import os
from pathlib import Path

import numpy as np

from phase3_13_runtime import close_env, deterministic_reset, install_minimal_ravens, seed_everything


def main():
    root = Path("/data/state_diff2")
    tasks, Environment = install_minimal_ravens(root)
    checks = {"task_registered": "ccda-slack-cable-v2" in tasks.names}
    captures = {}
    for condition in ("free", "hidden_slack_breakaway_pin_v2"):
        seed_everything(390000)
        os.environ.update({"CCDA_HIDDEN_CONDITION": condition, "CCDA_VISIBLE_SEED": "390000", "CCDA_PAIR_GROUP": "phase313_smoke_seed_390000"})
        env = Environment(disp=False, hz=480)
        task = tasks.names["ccda-slack-cable-v2"](); task.mode = "train"
        try:
            reset = deterministic_reset(env, task)
            extras = reset["info"]["extras"]
            captures[condition] = np.asarray(extras["bead_positions"], dtype=np.float64)
            checks[f"{condition}_no_velocity_field"] = "bead_velocities" not in extras
            checks[f"{condition}_24_beads"] = len(extras["bead_positions"]) == 24
            checks[f"{condition}_no_hidden_geometry"] = not task.hidden_body_ids and not task.hidden_constraint_ids
        finally:
            close_env(env)
    checks["paired_initial_geometry_exact"] = bool(np.array_equal(captures["free"], captures["hidden_slack_breakaway_pin_v2"]))
    payload = {"verdict": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "task": "ccda-slack-cable-v2", "conditions": ["free", "hidden_slack_breakaway_pin_v2"]}
    path = root / "reports/phase3_13_task_smoke_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    if payload["verdict"] != "PASS": raise SystemExit("formal task smoke failed")


if __name__ == "__main__": main()
