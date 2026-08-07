import numpy as np

from scripts.experiment3.phase0c_hidden_dynamics.commands import (
    build_candidate_plan, build_probe_test_plan)
from scripts.experiment3.phase0c_hidden_dynamics.common import load_config


class Robot:
    def inverse_kinematics_fixed_rest(self, pose, rest):
        return np.concatenate([pose.translation, np.asarray(rest)[3:]])


class Env:
    def __init__(self, config):
        self.config = config
        self.robot = Robot()
        self._start_joint_positions = np.zeros(6)
        self._ee_target_position = np.asarray([.382, -.1775, .06])


def test_probe_and_candidate_commands_are_branch_independent():
    config = load_config("configs/experiment3/hlf_sbp_phase0c_dynamics.json")
    env = Env(config)
    common = build_probe_test_plan(env, config)
    low = common
    high = type(common)(*(value.copy() for value in (
        common.phase, common.action_xy, common.ee_target_xyz, common.joint_target)))
    assert np.array_equal(low.joint_target, high.joint_target)
    assert np.array_equal(low.ee_target_xyz, high.ee_target_xyz)
    assert np.array_equal(low.action_xy, high.action_xy)
    assert np.array_equal(low.phase, high.phase)
    assert len(low.phase) == 912 and len(low.action_xy) == 38
    candidate = build_candidate_plan(
        env, low.ee_target_xyz[311], config["control_relevance"]["candidates"][0])
    assert candidate.joint_target.shape == (360, 6)
    assert candidate.action_xy.shape == (15, 2)
    assert candidate.joint_target.dtype != object
    assert candidate.ee_target_xyz.dtype != object
