"""Build and execute one immutable low-level command sequence for OCCP R1."""
from __future__ import annotations

import copy

import numpy as np
import pybullet as p


def _targets(env, start, end, orientation, command_steps, previous):
    ee_targets, joint_targets = [], []
    for alpha in np.linspace(1.0 / command_steps, 1.0, command_steps):
        ee_target = start + alpha * (end - start)
        joint_target = np.asarray(env.solve_IK(
            ee_target.tolist() + orientation.tolist()), dtype=np.float64)
        delta = joint_target - previous
        joint_target -= 2 * np.pi * np.round(delta / (2 * np.pi))
        ee_targets.append(ee_target.astype(float).tolist())
        joint_targets.append(joint_target.astype(float).tolist())
        previous = joint_target
    return ee_targets, joint_targets, previous


def build_fixed_command_script(
        env, *, start_ee_position, orientation, layout,
        probe_command_steps, test_command_steps, control_substeps):
    start = np.asarray(start_ee_position, dtype=np.float64)
    orientation = np.asarray(orientation, dtype=np.float64)
    previous = np.asarray([
        p.getJointState(env.ur5, int(joint))[0] for joint in env.joints],
        dtype=np.float64)
    probe_end = start + np.asarray(layout['probe_delta'], dtype=np.float64)
    probe_ee, probe_joint, previous = _targets(
        env, start, probe_end, orientation, int(probe_command_steps), previous)
    test_end = probe_end + np.asarray(layout['test_delta'], dtype=np.float64)
    test_ee, test_joint, _ = _targets(
        env, probe_end, test_end, orientation, int(test_command_steps), previous)
    return {
        'orientation': orientation.astype(float).tolist(),
        'phases': [
            {'name': 'probe', 'phase': 'probe', 'ee_targets': probe_ee,
             'joint_targets': probe_joint,
             'control_substeps': int(control_substeps)},
            {'name': 'test_pull', 'phase': 'test_pull', 'ee_targets': test_ee,
             'joint_targets': test_joint,
             'control_substeps': int(control_substeps)},
        ],
    }


def copy_fixed_command_script(command_script):
    return copy.deepcopy(command_script)


def command_arrays(command_script):
    probe, test = command_script['phases']
    return {
        'probe_ee_targets': np.asarray(probe['ee_targets'], dtype=np.float64),
        'probe_joint_targets': np.asarray(
            probe['joint_targets'], dtype=np.float64),
        'test_ee_targets': np.asarray(test['ee_targets'], dtype=np.float64),
        'test_joint_targets': np.asarray(test['joint_targets'], dtype=np.float64),
    }


def execute_fixed_command_phase(
        env, task, phase_payload, *, position_gains):
    start_step = task.physics_step_count()
    actual_trajectory = []
    for ee_target, joint_target in zip(
            phase_payload['ee_targets'], phase_payload['joint_targets']):
        task.set_ccda_phase(phase_payload['phase'])
        task.set_ee_target_position(ee_target)
        p.setJointMotorControlArray(
            bodyIndex=env.ur5,
            jointIndices=env.joints,
            controlMode=p.POSITION_CONTROL,
            targetPositions=joint_target,
            targetVelocities=np.zeros(len(env.joints)),
            positionGains=np.full(len(env.joints), float(position_gains)))
        env.step_physics(int(phase_payload['control_substeps']))
        actual_trajectory.append(list(p.getLinkState(
            env.ur5, env.ee_tip_link, computeForwardKinematics=True)[0]))
    end_step = task.physics_step_count()
    final_target = np.asarray(phase_payload['ee_targets'][-1], dtype=np.float64)
    final_actual = np.asarray(actual_trajectory[-1], dtype=np.float64)
    expected = (len(phase_payload['joint_targets'])
                * int(phase_payload['control_substeps']))
    return {
        'phase': phase_payload['phase'],
        'command_count': len(phase_payload['joint_targets']),
        'physics_step_start': int(start_step),
        'physics_step_end': int(end_step),
        'expected_physics_steps': int(expected),
        'actual_physics_steps': int(end_step - start_step),
        'final_target_position': final_target.astype(float).tolist(),
        'final_actual_position': final_actual.astype(float).tolist(),
        'endpoint_error': float(np.linalg.norm(final_actual - final_target)),
        'actual_ee_trajectory': actual_trajectory,
        'grasp_retained': bool(
            getattr(env.ee, 'activated', False)
            and getattr(env.ee, 'contact_constraint', None) is not None),
    }
