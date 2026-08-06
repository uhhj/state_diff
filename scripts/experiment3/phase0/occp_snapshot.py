"""Exact PyBullet and Python-side snapshot helpers for OCCP pairing."""
from __future__ import annotations

import numpy as np
import pybullet as p


def capture_world_state(env, task):
    if getattr(task, 'active_endpoint_stabilizer_id', None) is not None:
        raise RuntimeError('temporary active stabilizer must be released')
    bead_pose = [
        p.getBasePositionAndOrientation(body_id)
        for body_id in task.cable_bead_IDs]
    bead_velocity = [p.getBaseVelocity(body_id) for body_id in task.cable_bead_IDs]
    joint_states = [p.getJointState(env.ur5, int(joint)) for joint in env.joints]
    ee_state = p.getLinkState(
        env.ur5, env.ee_tip_link,
        computeLinkVelocity=1, computeForwardKinematics=True)
    constraint = getattr(env.ee, 'contact_constraint', None)
    return {
        'bead_positions': np.asarray([value[0] for value in bead_pose]),
        'bead_orientations': np.asarray([value[1] for value in bead_pose]),
        'bead_linear_velocities': np.asarray(
            [value[0] for value in bead_velocity]),
        'bead_angular_velocities': np.asarray(
            [value[1] for value in bead_velocity]),
        'joint_positions': np.asarray([value[0] for value in joint_states]),
        'joint_velocities': np.asarray([value[1] for value in joint_states]),
        'ee_position': np.asarray(ee_state[0]),
        'ee_orientation': np.asarray(ee_state[1]),
        'ee_linear_velocity': np.asarray(ee_state[6]),
        'ee_angular_velocity': np.asarray(ee_state[7]),
        'grasp_activated': np.asarray(
            [int(bool(getattr(env.ee, 'activated', False)))], dtype=np.int64),
        'grasp_constraint_id': np.asarray(
            [-1 if constraint is None else int(constraint)], dtype=np.int64),
        'task_physics_step': np.asarray(
            [task.physics_step_count()], dtype=np.int64),
        'task_phase': np.asarray([task._phase]),
    }


def max_state_difference(first, second):
    if set(first) != set(second):
        raise ValueError('world-state keys differ')
    maximum = 0.0
    for name in sorted(first):
        left = np.asarray(first[name])
        right = np.asarray(second[name])
        if left.shape != right.shape:
            raise ValueError('state shape differs for {}'.format(name))
        if left.dtype.kind in 'OUS' or right.dtype.kind in 'OUS':
            if not np.array_equal(left, right):
                return float('inf')
        elif left.size:
            maximum = max(
                maximum,
                float(np.max(np.abs(
                    left.astype(np.float64) - right.astype(np.float64)))))
    return maximum


def capture_python_runtime_state(env, task):
    constraint = getattr(env.ee, 'contact_constraint', None)
    return {
        'ee_activated': bool(getattr(env.ee, 'activated', False)),
        'ee_contact_constraint': (
            None if constraint is None else int(constraint)),
        'task_phase': str(task._phase),
        'task_physics_step': int(task.physics_step_count()),
    }


def restore_python_runtime_state(env, task, payload):
    env.ee.activated = bool(payload['ee_activated'])
    env.ee.contact_constraint = payload['ee_contact_constraint']
    task._phase = str(payload['task_phase'])
    task._physics_step_count = int(payload['task_physics_step'])
