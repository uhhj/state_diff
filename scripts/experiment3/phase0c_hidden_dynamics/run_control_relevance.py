"""Execute the fixed Phase 0C action library from post-probe snapshots."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiment3.phase0c_hidden_dynamics.commands import (  # noqa: E402
    build_candidate_plan, build_probe_test_plan)
from scripts.experiment3.phase0c_hidden_dynamics.common import (  # noqa: E402
    CONDITIONS, load_config, report_pair_dir)
from scripts.experiment3.phase0c_hidden_dynamics.io_utils import write_json  # noqa: E402
from scripts.experiment3.phase0c_hidden_dynamics.run_pair import (  # noqa: E402
    execute_exact_plan)
from state_diff.env.block_pushing.soft_block_pushing import SoftBlockPushEnv  # noqa: E402


def run_control_relevance(config_path: str) -> dict:
    config = load_config(config_path)
    pair_metrics = json.loads((report_pair_dir(config) / "pair_metrics.json").read_text(
        encoding="utf-8"))
    if pair_metrics["verdict"] != "PHASE0C_PAIR_COMPLETE":
        raise RuntimeError("control relevance requires PHASE0C_PAIR_COMPLETE")
    env = SoftBlockPushEnv(config)
    try:
        probe_plan = build_probe_test_plan(env, config)
        prefix = sum(int(config["execution"][key]) for key in (
            "no_action_outer_steps", "probe_outer_steps",
            "post_probe_outer_steps"))
        start_target = probe_plan.ee_target_xyz[prefix - 1].copy()
        candidate_plans = {
            candidate["id"]: build_candidate_plan(env, start_target, candidate)
            for candidate in config["control_relevance"]["candidates"]}
        snapshots = {}
        for condition in CONDITIONS:
            env.restore_saved_state(env.saved_state_id)
            env.clear_trace(); env.arm_condition(condition)
            execute_exact_plan(env, probe_plan, outer_limit=prefix)
            snapshots[condition] = int(env.pybullet_client.saveState())
        matrix = {}
        for condition in CONDITIONS:
            matrix[condition] = {}
            for candidate in config["control_relevance"]["candidates"]:
                env.pybullet_client.restoreState(stateId=snapshots[condition])
                env.set_ee_target_position(start_target)
                env.clear_trace(); env.arm_condition(condition)
                rollout = execute_exact_plan(env, candidate_plans[candidate["id"]])
                initial = float(rollout["goal_distance"][0])
                final = float(rollout["goal_distance"][-1])
                matrix[condition][candidate["id"]] = {
                    "initial_goal_distance_m": initial,
                    "final_goal_distance_m": final,
                    "progress_m": initial - final,
                    "success": final <= float(config["goal"]["tolerance_m"]),
                    "final_state": rollout["state"][-1],
                    "joint_target": candidate_plans[candidate["id"]].joint_target,
                    "ee_target": candidate_plans[candidate["id"]].ee_target_xyz}
        output = (Path(config["output_root"]) / config["pair_id"] /
                  "control_relevance.json")
        write_json(output, {"matrix": matrix, "online_selection": False})
        return {"output": str(output), "matrix": matrix}
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print("output={}".format(run_control_relevance(args.config)["output"]))


if __name__ == "__main__":
    main()
