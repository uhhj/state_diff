"""Small PyBullet/Python snapshot helpers for paired OHJ rollouts."""
from __future__ import annotations

import numpy as np
import pybullet as p


def capture_world_state(env, task):
    bead_pose = [p.getBasePositionAndOrientation(body)
                 for body in task.cable_bead_IDs]
    bead_velocity = [p.getBaseVelocity(body) for body in task.cable_bead_IDs]
    joint = [p.getJointState(env.ur5, int(index)) for index in env.joints]
    ee = p.getLinkState(
        env.ur5, env.ee_tip_link,
        computeLinkVelocity=1, computeForwardKinematics=True)
    return {
        "bead_positions": np.asarray([value[0] for value in bead_pose]),
        "bead_orientations": np.asarray([value[1] for value in bead_pose]),
        "bead_linear_velocities": np.asarray(
            [value[0] for value in bead_velocity]),
        "bead_angular_velocities": np.asarray(
            [value[1] for value in bead_velocity]),
        "joint_positions": np.asarray([value[0] for value in joint]),
        "joint_velocities": np.asarray([value[1] for value in joint]),
        "ee_position": np.asarray(ee[0]),
        "ee_orientation": np.asarray(ee[1]),
        "grasp_active": np.asarray([
            int(bool(getattr(env.ee, "activated", False)))], dtype=np.int64),
    }


def capture_python_runtime_state(env, task):
    constraint = getattr(env.ee, "contact_constraint", None)
    return {
        "ee_activated": bool(getattr(env.ee, "activated", False)),
        "ee_contact_constraint": None if constraint is None else int(constraint),
        "task_phase": str(task._phase),
        "task_physics_step": int(task.physics_step_count()),
    }


def restore_python_runtime_state(env, task, state):
    env.ee.activated = bool(state["ee_activated"])
    env.ee.contact_constraint = state["ee_contact_constraint"]
    task._phase = str(state["task_phase"])
    task._physics_step_count = int(state["task_physics_step"])


def max_state_difference(first, second):
    maximum = 0.0
    for name in first:
        left, right = np.asarray(first[name]), np.asarray(second[name])
        if left.dtype.kind in "OUS" or right.dtype.kind in "OUS":
            if not np.array_equal(left, right):
                return float("inf")
        elif left.size:
            maximum = max(maximum, float(np.max(np.abs(left - right))))
    return maximum
