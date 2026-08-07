"""Branch-independent low-level command plans for Phase 0C."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from state_diff.env.block_pushing import block_pushing
from state_diff.env.block_pushing.utils.pose3d import Pose3d


@dataclass(frozen=True)
class CommandPlan:
    phase: np.ndarray
    action_xy: np.ndarray
    ee_target_xyz: np.ndarray
    joint_target: np.ndarray


def _append_policy_target(env, stride, phase, target, action, phases, actions,
                          ee_targets, joint_targets):
    pose = Pose3d(block_pushing.EFFECTOR_DOWN_ROTATION, target)
    joints = env.robot.inverse_kinematics_fixed_rest(
        pose, env._start_joint_positions)
    phases.extend([phase] * stride)
    ee_targets.extend([target.copy()] * stride)
    joint_targets.extend([joints.copy()] * stride)
    actions.append(np.asarray(action, dtype=np.float64))


def build_probe_test_plan(env, config) -> CommandPlan:
    """Build one common no-action/probe/test plan before intervention."""
    stride = int(config["execution"]["policy_sample_stride_outer_steps"])
    phases, actions, ee_targets, joint_targets = [], [], [], []
    current = env._ee_target_position.copy()
    sections = [
        ("no_action", config["execution"]["no_action_outer_steps"], None),
        ("probe", config["execution"]["probe_outer_steps"],
         config["motion"]["probe_delta_xy_m"]),
        ("post_probe", config["execution"]["post_probe_outer_steps"], None),
        ("test", config["execution"]["test_outer_steps"],
         config["motion"]["test_delta_xy_m"]),
        ("post_test", config["execution"]["post_test_outer_steps"], None)]
    for phase, outer_steps, delta in sections:
        if int(outer_steps) % stride:
            raise ValueError("phase is not divisible by policy stride")
        count = int(outer_steps) // stride
        start = current.copy()
        delta3 = np.zeros(3, dtype=np.float64)
        if delta is not None:
            delta3[:2] = np.asarray(delta, dtype=np.float64)
        for index in range(1, count + 1):
            target = start + delta3 * (index / count) if delta is not None else start
            action = delta3[:2] / count if delta is not None else np.zeros(2)
            _append_policy_target(
                env, stride, phase, target, action, phases, actions,
                ee_targets, joint_targets)
        current = start + delta3
    return CommandPlan(
        np.asarray(phases), np.asarray(actions, dtype=np.float64),
        np.asarray(ee_targets, dtype=np.float64),
        np.asarray(joint_targets, dtype=np.float64))


def build_candidate_plan(env, start_target_xyz, candidate) -> CommandPlan:
    """Build one fixed candidate path with branch-independent IK rest."""
    stride = int(env.config["execution"]["policy_sample_stride_outer_steps"])
    count = int(candidate["policy_steps"])
    start = np.asarray(start_target_xyz, dtype=np.float64)
    delta = np.zeros(3, dtype=np.float64)
    delta[:2] = np.asarray(candidate["delta_xy_m"], dtype=np.float64)
    phases, actions, ee_targets, joint_targets = [], [], [], []
    for index in range(1, count + 1):
        _append_policy_target(
            env, stride, "test", start + delta * (index / count),
            delta[:2] / count, phases, actions, ee_targets, joint_targets)
    return CommandPlan(
        np.asarray(phases), np.asarray(actions, dtype=np.float64),
        np.asarray(ee_targets, dtype=np.float64),
        np.asarray(joint_targets, dtype=np.float64))
