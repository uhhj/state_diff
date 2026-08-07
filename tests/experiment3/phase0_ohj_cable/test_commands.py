import inspect
import types

import numpy as np

from scripts.experiment3.phase0_ohj_cable.commands import (
    build_candidate_scripts, build_probe_script, command_arrays)
from scripts.experiment3.phase0_ohj_cable.run_pair import _run_branch


class FakeEnv:
    def __init__(self):
        self.calls = 0

    def solve_IK(self, pose):
        self.calls += 1
        return np.full(6, pose[0] + 2 * np.pi * self.calls)


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
