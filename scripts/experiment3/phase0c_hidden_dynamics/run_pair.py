"""Execute the strict same-command hidden-dynamics pair and repeat."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiment3.phase0c_hidden_dynamics.commands import (  # noqa: E402
    CommandPlan, build_probe_test_plan)
from scripts.experiment3.phase0c_hidden_dynamics.common import (  # noqa: E402
    external_pair_dir, load_config)
from scripts.experiment3.phase0c_hidden_dynamics.io_utils import write_json  # noqa: E402
from scripts.experiment3.phase0c_hidden_dynamics.highrate_sampling import (  # noqa: E402
    build_policy_sensor_windows, sample_post_step, stack_rows)
from state_diff.env.block_pushing.soft_block_pushing import SoftBlockPushEnv  # noqa: E402


def execute_exact_plan(env, plan: CommandPlan, outer_limit=None) -> dict:
    """Execute exact commands and sample only after each real outer step."""
    limit = len(plan.phase) if outer_limit is None else int(outer_limit)
    stride = int(env.config["execution"]["policy_sample_stride_outer_steps"])
    if limit % stride or limit > len(plan.phase):
        raise ValueError("outer_limit must end on a policy boundary")
    policy_states = [env.get_statediff_state()]
    policy_goals = [float(np.linalg.norm(
        env.soft_block.center_of_mass()[:2]
        - np.asarray(env.config["goal"]["center_xy"], dtype=np.float64)))]
    policy_steps, outer_steps = [0], [0]
    sample_phases, sample_actions = ["initial"], [np.zeros(2)]
    highrate_rows = []
    for index in range(limit):
        env.set_phase(str(plan.phase[index]))
        env.set_ee_target_position(plan.ee_target_xyz[index])
        env.set_fixed_joint_target(plan.joint_target[index])
        env.step_one_physics()
        fresh = sample_post_step(env)
        stats = env._last_mechanics_stats
        oracle = env._patch_oracle()
        highrate_rows.append({
            "outer_step": index + 1,
            "policy_index": index // stride,
            "phase": str(plan.phase[index]),
            "state": fresh["state"],
            "sensor": fresh["sensor"],
            "proprio": fresh["proprio"],
            "goal_distance": fresh["goal_distance"],
            "spring_cap_count": int(stats.capped_force_count),
            "spring_evaluation_count": int(stats.force_evaluation_count),
            "oracle_patch_contact_count": len(
                oracle["oracle_patch_contact_node_indices"]),
            "oracle_patch_tangential_force": float(
                oracle["oracle_patch_tangential_force"]),
            "oracle_patch_mean_slip_speed": float(
                oracle["oracle_patch_mean_slip_speed"]),
        })
        if (index + 1) % stride == 0:
            policy_index = (index + 1) // stride
            policy_states.append(fresh["state"])
            policy_goals.append(fresh["goal_distance"])
            policy_steps.append(policy_index)
            outer_steps.append(index + 1)
            sample_phases.append(str(plan.phase[index]))
            sample_actions.append(plan.action_xy[policy_index - 1])
    outer = stack_rows(highrate_rows)
    sensor_windows, sensor_window_phase = build_policy_sensor_windows(
        outer["sensor"], outer["phase"], stride)
    result = {
        "state": np.asarray(policy_states, dtype=np.float64),
        "goal_distance": np.asarray(policy_goals, dtype=np.float64),
        "policy_step": np.asarray(policy_steps, dtype=np.int64),
        "outer_step": np.asarray(outer_steps, dtype=np.int64),
        "phase": np.asarray(sample_phases),
        "action": np.asarray(sample_actions, dtype=np.float64),
        "sensor_window": sensor_windows,
        "sensor_window_phase": sensor_window_phase,
        "outer_step_hr": outer["outer_step"].astype(np.int64),
        "policy_index_hr": outer["policy_index"].astype(np.int64),
        "phase_hr": outer["phase"],
        "state_hr": outer["state"],
        "contact_sensor_hr": outer["sensor"],
        "robot_proprio_extended_hr": outer["proprio"],
        "goal_distance_hr": outer["goal_distance"],
        "spring_cap_count_hr": outer["spring_cap_count"].astype(np.int64),
        "spring_evaluation_count_hr": outer[
            "spring_evaluation_count"].astype(np.int64),
        "oracle_patch_contact_count_hr": outer[
            "oracle_patch_contact_count"].astype(np.int64),
        "oracle_patch_tangential_force_hr": outer[
            "oracle_patch_tangential_force"],
        "oracle_patch_mean_slip_speed_hr": outer[
            "oracle_patch_mean_slip_speed"],
        "command_phase": plan.phase[:limit].copy(),
        "command_action": plan.action_xy[:limit // stride].copy(),
        "ee_target": plan.ee_target_xyz[:limit].copy(),
        "joint_target": plan.joint_target[:limit].copy(),
    }
    return result


def save_rollout(config, condition, rollout, metadata) -> Path:
    output = external_pair_dir(config, condition)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "trajectory.npz", **rollout)
    write_json(output / "metadata.json", metadata)
    return output


def run_pair(config_path: str) -> dict:
    config = load_config(config_path)
    env = SoftBlockPushEnv(config)
    try:
        common_state = env.capture_explicit_state()
        plan = build_probe_test_plan(env, config)
        outputs = {}
        for condition in ("uniform_low", "right_local_high", "uniform_low_repeat"):
            env.restore_saved_state(env.saved_state_id)
            env.clear_trace()
            actual = "uniform_low" if condition.endswith("_repeat") else condition
            arm = env.arm_condition(actual)
            rollout = execute_exact_plan(env, plan)
            metadata = {
                "config": config, "condition": condition,
                "actual_condition": actual, "arm": arm,
                "common_state_keys": sorted(common_state),
                "pairing_mode": "save_restore_exact_precomputed_joint_targets",
                "branch_local_ik": False,
                "material_profile": env.material_profile,
                "microsteps_per_outer": env._microstep_config.microsteps_per_outer,
                "bullet_num_substeps": 1,
                "internal_evaluations_per_microstep": sum(
                    len(value) for value in env.soft_block.edges.values())}
            outputs[condition] = str(save_rollout(
                config, condition, rollout, metadata))
        return {"outputs": outputs, "policy_targets": len(plan.action_xy),
                "outer_steps": len(plan.phase)}
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(run_pair(args.config))


if __name__ == "__main__":
    main()
