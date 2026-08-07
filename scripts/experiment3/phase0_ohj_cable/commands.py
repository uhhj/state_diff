"""Precompute and execute branch-independent OHJ command arrays."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pybullet as p


@dataclass(frozen=True)
class CommandPhase:
    name: str
    ee_targets: np.ndarray
    joint_targets: np.ndarray


def interpolate_targets(env, start, stop, orientation, steps, previous):
    start = np.asarray(start, dtype=np.float64)
    stop = np.asarray(stop, dtype=np.float64)
    previous = np.asarray(previous, dtype=np.float64)
    ee_rows, joint_rows = [], []
    for alpha in np.linspace(1.0 / int(steps), 1.0, int(steps)):
        ee = start + alpha * (stop - start)
        joint = np.asarray(env.solve_IK(
            ee.tolist() + np.asarray(orientation).tolist()), dtype=np.float64)
        joint -= 2 * np.pi * np.round((joint - previous) / (2 * np.pi))
        ee_rows.append(ee)
        joint_rows.append(joint)
        previous = joint
    return np.asarray(ee_rows), np.asarray(joint_rows), previous


def _phase(env, name, start, stop, orientation, steps, previous):
    ee, joint, previous = interpolate_targets(
        env, start, stop, orientation, steps, previous)
    return CommandPhase(name, ee, joint), previous


def build_probe_script(env, start, orientation, motion, previous):
    start = np.asarray(start, dtype=np.float64)
    delta = np.asarray(motion["probe_delta_xyz_m"], dtype=np.float64)
    peak = start + delta
    forward, previous = _phase(
        env, "probe_forward", start, peak, orientation,
        motion["probe_forward_steps"], previous)
    hold = CommandPhase(
        "probe_hold",
        np.repeat(peak[None], int(motion["probe_hold_steps"]), axis=0),
        np.repeat(previous[None], int(motion["probe_hold_steps"]), axis=0))
    returned, previous = _phase(
        env, "probe_return", peak, start, orientation,
        motion["probe_return_steps"], previous)
    np.testing.assert_allclose(returned.ee_targets[-1], start)
    return (forward, hold, returned), previous


def build_straight_script(env, start, orientation, motion, previous):
    phase, _ = _phase(
        env, "test_pull", start,
        np.asarray(start) + np.asarray(motion["straight_pull_delta_xyz_m"]),
        orientation, motion["straight_pull_steps"], previous)
    return (phase,)


def _release_script(env, start, orientation, motion, previous, sign):
    start = np.asarray(start, dtype=np.float64)
    backoff_stop = start + np.asarray(motion["release_backoff_xyz_m"])
    backoff, previous = _phase(
        env, "release_backoff", start, backoff_stop, orientation,
        motion["release_backoff_steps"], previous)
    lateral_stop = backoff_stop + np.array(
        [0.0, sign * float(motion["release_lateral_m"]), 0.0])
    lateral, previous = _phase(
        env, "release_lateral", backoff_stop, lateral_stop, orientation,
        motion["release_lateral_steps"], previous)
    pull_stop = lateral_stop + np.asarray(motion["straight_pull_delta_xyz_m"])
    pull, _ = _phase(
        env, "test_pull", lateral_stop, pull_stop, orientation,
        motion["release_pull_steps"], previous)
    return backoff, lateral, pull


def build_candidate_scripts(env, start, orientation, motion, previous):
    return {
        "straight": build_straight_script(
            env, start, orientation, motion, previous),
        "left_release": _release_script(
            env, start, orientation, motion, previous, +1.0),
        "right_release": _release_script(
            env, start, orientation, motion, previous, -1.0),
    }


def execute_fixed_phase(env, task, phase, position_gain):
    for ee, joint in zip(phase.ee_targets, phase.joint_targets):
        task.set_ccda_phase(phase.name)
        task.set_ee_target_position(ee)
        p.setJointMotorControlArray(
            bodyIndex=env.ur5, jointIndices=env.joints,
            controlMode=p.POSITION_CONTROL, targetPositions=joint,
            targetVelocities=np.zeros(len(env.joints)),
            positionGains=np.full(len(env.joints), float(position_gain)))
        env.step_physics(1)


def command_arrays(phases):
    return {
        "phase": np.concatenate([
            np.repeat(phase.name, len(phase.ee_targets)) for phase in phases]),
        "ee_target": np.concatenate([phase.ee_targets for phase in phases]),
        "joint_target": np.concatenate([phase.joint_targets for phase in phases]),
    }
