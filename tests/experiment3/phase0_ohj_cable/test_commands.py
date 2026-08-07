import inspect
import types

import numpy as np

from scripts.experiment3.phase0_ohj_cable.commands import (
    build_candidate_scripts, build_probe_script, command_arrays)
from scripts.experiment3.phase0_ohj_cable.run_pair import (
    _run_branch, set_canonical_robot_hold)


class FakeEnv:
    def __init__(self):
        self.calls = 0

    def solve_IK(self, pose):
        self.calls += 1
        return np.full(6, pose[0] + 2 * np.pi * self.calls)


class HoldTask:
    def __init__(self):
        self.ee_target = None

    def set_ee_target_position(self, value):
        self.ee_target = np.asarray(value, dtype=np.float64)


class HoldEnv:
    def __init__(self):
        self.ur5 = 7
        self.joints = [2, 3, 4, 5, 6, 7]


MOTION = {
    "probe_delta_xyz_m": [.002, 0., 0.],
    "probe_forward_steps": 2, "probe_hold_steps": 2,
    "probe_return_steps": 2,
    "straight_pull_delta_xyz_m": [.06, 0., 0.],
    "straight_pull_steps": 4,
    "release_backoff_xyz_m": [-.003, 0., 0.],
    "release_lateral_m": .018,
    "release_backoff_steps": 2, "release_lateral_steps": 2,
    "release_pull_steps": 4,
}


def test_probe_is_zero_net_and_candidate_commands_are_fixed():
    env = FakeEnv()
    start = np.array([.5, 0., .1])
    orientation = np.array([0., 0., 0., 1.])
    previous = np.zeros(6)
    probe, _ = build_probe_script(
        env, start, orientation, MOTION, previous)
    np.testing.assert_allclose(probe[-1].ee_targets[-1], start)
    candidates = build_candidate_scripts(
        env, start, orientation, MOTION, previous)
    assert set(candidates) == {"straight", "left_release", "right_release"}
    assert command_arrays(candidates["straight"])["ee_target"].shape == (4, 3)
    assert np.allclose(
        command_arrays(candidates["left_release"])["ee_target"][-1, 1], .018)
    assert np.allclose(
        command_arrays(candidates["right_release"])["ee_target"][-1, 1], -.018)


def test_branch_execution_contains_no_inverse_kinematics():
    source = inspect.getsource(_run_branch)
    assert "solve_IK" not in source
    assert ".movep(" not in source
    assert ".movej(" not in source


def test_canonical_hold_reasserts_position_control(monkeypatch):
    calls = []

    def fake_set_joint_motor_control_array(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(
        "scripts.experiment3.phase0_ohj_cable.run_pair."
        "p.setJointMotorControlArray",
        fake_set_joint_motor_control_array,
    )
    env = HoldEnv()
    task = HoldTask()
    joint = np.linspace(-1.0, 1.0, 6)
    ee = np.array([0.4, -0.1, 0.1])
    set_canonical_robot_hold(env, task, joint, ee, 0.01)
    np.testing.assert_allclose(task.ee_target, ee)
    assert len(calls) == 1
    call = calls[0]
    np.testing.assert_allclose(call["targetPositions"], joint)
    np.testing.assert_allclose(call["targetVelocities"], np.zeros(6))


def test_branch_reasserts_hold_before_no_action_step():
    source = inspect.getsource(_run_branch)
    hold_at = source.index("set_canonical_robot_hold")
    no_action_at = source.index('task.set_ccda_phase("no_action")')
    assert hold_at < no_action_at
