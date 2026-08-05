#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pybullet as p

REPO_ROOT = Path(__file__).resolve().parents[3]
SUBMODULE_ROOT = REPO_ROOT / "external" / "deformable-ravens"
for path in (REPO_ROOT, SUBMODULE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ravens import Environment, tasks


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def run_repeat(repeat):
    env = Environment(
        disp=False,
        hz=480,
        deterministic=True,
        control_substeps=1,
        post_action_settle_steps=0,
    )
    try:
        env.reset(tasks.names["cable-line-notarget"]())
        env.reset_ccda_motion_events()
        state = p.getLinkState(env.ur5, env.ee_tip_link, computeForwardKinematics=True)
        home = np.asarray(state[0] + state[1], dtype=np.float64)
        rows = []
        for label, distance in (("half_mm", 0.0005), ("one_mm", 0.001)):
            target = home.copy()
            target[0] += distance
            success = env.movep_precise(
                target,
                speed=0.001,
                joint_tolerance=1e-4,
                cartesian_tolerance=2e-4,
                label=label,
            )
            event = dict(env.ccda_motion_events()[-1])
            event["repeat"] = repeat
            rows.append(event)
            assert success
            assert event["achieved_fraction"] >= 0.8
            assert env.movep_precise(
                home,
                speed=0.001,
                joint_tolerance=1e-4,
                cartesian_tolerance=2e-4,
                label=label + "_return",
            )
        return rows
    finally:
        env.stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="reports/experiment2/phase0_hidden_hook/phase0g/motion_precision.json",
    )
    args = parser.parse_args()
    rows = []
    for repeat in range(2):
        rows.extend(run_repeat(repeat))
    result = {
        "execution": "fixed_step",
        "hz": 480,
        "joint_tolerance": 1e-4,
        "cartesian_tolerance": 2e-4,
        "rows": rows,
        "passed": all(row["success"] and row["achieved_fraction"] >= 0.8 for row in rows),
    }
    output = (REPO_ROOT / args.output).resolve()
    write_json(output, result)
    print(output)


if __name__ == "__main__":
    main()
