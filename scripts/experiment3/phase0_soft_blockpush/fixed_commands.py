"""Build and execute one branch-independent fixed joint-target script."""
from __future__ import annotations

import copy
from typing import Any, Dict

import numpy as np

from state_diff.env.block_pushing import block_pushing
from state_diff.env.block_pushing.utils.pose3d import Pose3d


def _phase_targets(env, start_xy: np.ndarray, end_xy: np.ndarray,
                   count: int, base_rest: np.ndarray, height: float):
    ee_targets = np.linspace(start_xy, end_xy, count + 1)[1:]
    joint_targets = []
    previous = np.asarray(base_rest, dtype=np.float64)
    for xy in ee_targets:
        position = np.array([xy[0], xy[1], height], dtype=np.float64)
        pose = Pose3d(block_pushing.EFFECTOR_DOWN_ROTATION, position)
        target = env.robot.inverse_kinematics_fixed_rest(
            pose, rest_joint_positions=base_rest)
        delta = target - previous
        target = target - 2 * np.pi * np.round(delta / (2 * np.pi))
        joint_targets.append(target.tolist())
        previous = target
    positions = np.column_stack([
        ee_targets, np.full(len(ee_targets), height, dtype=np.float64)])
    return positions.tolist(), joint_targets


def build_fixed_command_script(env, config: Dict[str, Any],
                               base_state: Dict[str, np.ndarray]) -> dict:
    """Create probe/test EE and fixed-rest joint targets exactly once."""
    base_joints = np.asarray(base_state["joint_positions"], dtype=np.float64)
    start = np.asarray(base_state["ee_target_position"], dtype=np.float64)[:2]
    probe_end = start + np.asarray(config["motion"]["probe_delta_xy_m"])
    test_end = start + np.asarray(config["motion"]["test_delta_xy_m"])
    height = float(config["robot"]["effector_height"])
    probe_ee, probe_joint = _phase_targets(
        env, start, probe_end,
        int(config["execution"]["probe_command_steps"]), base_joints, height)
    test_ee, test_joint = _phase_targets(
        env, probe_end, test_end,
        int(config["execution"]["test_command_steps"]), base_joints, height)
    return {
        "base_rest_joint_positions": base_joints.tolist(),
        "hold_joint_target": base_joints.tolist(),
        "orientation": block_pushing.EFFECTOR_DOWN_ROTATION.as_quat().tolist(),
        "phases": [
            {"name": "probe", "phase": "probe",
             "ee_targets": probe_ee, "joint_targets": probe_joint},
            {"name": "test", "phase": "test",
             "ee_targets": test_ee, "joint_targets": test_joint}],
    }


def copy_fixed_command_script(script: dict) -> dict:
    """Deep-copy one already generated command script."""
    return copy.deepcopy(script)


def command_arrays(script: dict) -> Dict[str, np.ndarray]:
    """Return canonical NumPy arrays used for strict branch equality."""
    probe, test = script["phases"]
    return {
        "probe_ee_targets": np.asarray(probe["ee_targets"], dtype=np.float64),
        "probe_joint_targets": np.asarray(
            probe["joint_targets"], dtype=np.float64),
        "test_ee_targets": np.asarray(test["ee_targets"], dtype=np.float64),
        "test_joint_targets": np.asarray(test["joint_targets"], dtype=np.float64),
        "hold_joint_target": np.asarray(
            script["hold_joint_target"], dtype=np.float64),
    }


def execute_fixed_phase(env, phase_payload: dict) -> dict:
    """Execute every fixed target for exactly one physics step."""
    start_step = int(env._physics_step)
    actual = []
    for ee_target, joint_target in zip(
            phase_payload["ee_targets"], phase_payload["joint_targets"]):
        env.set_phase(phase_payload["phase"])
        env.set_ee_target_position(np.asarray(ee_target, dtype=np.float64))
        env.set_fixed_joint_target(np.asarray(joint_target, dtype=np.float64))
        env.step_one_physics()
        actual.append(env._observation()["ee_position"].tolist())
    target = np.asarray(phase_payload["ee_targets"][-1], dtype=np.float64)
    endpoint = np.asarray(actual[-1], dtype=np.float64)
    return {
        "phase": phase_payload["phase"],
        "command_count": len(phase_payload["joint_targets"]),
        "physics_step_start": start_step,
        "physics_step_end": int(env._physics_step),
        "expected_physics_steps": len(phase_payload["joint_targets"]),
        "actual_physics_steps": int(env._physics_step) - start_step,
        "final_target_position": target.tolist(),
        "final_actual_position": endpoint.tolist(),
        "endpoint_error_m": float(np.linalg.norm(endpoint - target)),
        "actual_ee_trajectory": actual,
    }
