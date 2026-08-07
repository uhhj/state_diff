from types import SimpleNamespace

import numpy as np

from scripts.experiment3.phase0c_hidden_dynamics.commands import CommandPlan
from scripts.experiment3.phase0c_hidden_dynamics.run_pair import execute_exact_plan


class FakeBlock:
    def center_of_mass(self):
        return np.array([0.0, 0.0, 0.0])


class FakeEnv:
    def __init__(self):
        self.config = {
            "execution": {"policy_sample_stride_outer_steps": 24},
            "goal": {"center_xy": [0.0, 1.0]},
        }
        self.soft_block = FakeBlock()
        self.step = 0
        self.sensor_reads = []
        self._last_mechanics_stats = SimpleNamespace(
            capped_force_count=0, force_evaluation_count=10)

    def get_statediff_state(self):
        return np.full(74, self.step, dtype=np.float64)

    def get_contact_sensor(self):
        self.sensor_reads.append(self.step)
        return np.full(45, self.step, dtype=np.float64)

    def get_extended_proprio(self):
        return np.full(24, self.step, dtype=np.float64)

    def set_phase(self, phase):
        self.phase = phase

    def set_ee_target_position(self, target):
        pass

    def set_fixed_joint_target(self, target):
        pass

    def step_one_physics(self):
        self.step += 1

    def _patch_oracle(self):
        return {"oracle_patch_contact_node_indices": [],
                "oracle_patch_tangential_force": 0.0,
                "oracle_patch_mean_slip_speed": 0.0}


def test_every_outer_step_is_sampled_causally_without_prestep_sensor():
    env = FakeEnv()
    plan = CommandPlan(
        phase=np.asarray(["no_action"] * 24),
        action_xy=np.zeros((1, 2)),
        ee_target_xyz=np.zeros((24, 3)),
        joint_target=np.zeros((24, 6)))
    rollout = execute_exact_plan(env, plan)
    assert rollout["state_hr"].shape == (24, 74)
    assert rollout["contact_sensor_hr"].shape == (24, 45)
    assert rollout["robot_proprio_extended_hr"].shape == (24, 24)
    assert rollout["outer_step_hr"][0] == 1
    assert env.sensor_reads == list(range(1, 25))
    assert "contact_sensor" not in rollout
    assert "robot_proprio_extended" not in rollout
