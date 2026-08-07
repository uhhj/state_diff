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
from state_diff.env.block_pushing.soft_block_pushing import SoftBlockPushEnv  # noqa: E402


def _sample(env) -> dict:
    oracle = env._patch_oracle()
    return {
        "state": env.get_statediff_state(),
        "contact_sensor": env.get_contact_sensor(),
        "robot_proprio_extended": env.get_extended_proprio(),
        "goal_distance": float(np.linalg.norm(
            env.soft_block.center_of_mass()[:2]
            - np.asarray(env.config["goal"]["center_xy"]))),
        "oracle_patch_contact_count": len(
            oracle["oracle_patch_contact_node_indices"]),
        "oracle_patch_tangential_force":
            oracle["oracle_patch_tangential_force"],
        "oracle_patch_mean_slip_speed":
            oracle["oracle_patch_mean_slip_speed"]}


def execute_exact_plan(env, plan: CommandPlan, outer_limit=None) -> dict:
    """Execute precomputed arrays without branch-local inverse kinematics."""
    limit = len(plan.phase) if outer_limit is None else int(outer_limit)
    stride = int(env.config["execution"]["policy_sample_stride_outer_steps"])
    if limit % stride or limit > len(plan.phase):
        raise ValueError("outer_limit must end on a policy boundary")
    initial = _sample(env)
    rows = {key: [value] for key, value in initial.items()}
    policy_steps, outer_steps = [0], [0]
    sample_phases, sample_actions = ["initial"], [np.zeros(2)]
    cap_counts, evaluation_counts = [0], [0]
    policy_caps = policy_evaluations = 0
    for index in range(limit):
        env.set_phase(str(plan.phase[index]))
        env.set_ee_target_position(plan.ee_target_xyz[index])
        env.set_fixed_joint_target(plan.joint_target[index])
        env.step_one_physics()
        stats = env._last_mechanics_stats
        policy_caps += int(stats.capped_force_count)
        policy_evaluations += int(stats.force_evaluation_count)
        if (index + 1) % stride == 0:
            policy_index = (index + 1) // stride
            sample = _sample(env)
            for key, value in sample.items():
                rows[key].append(value)
            policy_steps.append(policy_index)
            outer_steps.append(index + 1)
            sample_phases.append(str(plan.phase[index]))
            sample_actions.append(plan.action_xy[policy_index - 1])
            cap_counts.append(policy_caps)
            evaluation_counts.append(policy_evaluations)
            policy_caps = policy_evaluations = 0
    result = {key: np.asarray(value) for key, value in rows.items()}
    result.update({
        "policy_step": np.asarray(policy_steps, dtype=np.int64),
        "outer_step": np.asarray(outer_steps, dtype=np.int64),
        "phase": np.asarray(sample_phases),
        "action": np.asarray(sample_actions, dtype=np.float64),
        "spring_cap_count": np.asarray(cap_counts, dtype=np.int64),
        "spring_evaluation_count": np.asarray(
            evaluation_counts, dtype=np.int64),
        "command_phase": plan.phase[:limit].copy(),
        "command_action": plan.action_xy[:limit // stride].copy(),
        "ee_target": plan.ee_target_xyz[:limit].copy(),
        "joint_target": plan.joint_target[:limit].copy()})
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
