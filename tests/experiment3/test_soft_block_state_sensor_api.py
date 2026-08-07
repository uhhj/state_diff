import numpy as np

from scripts.experiment3.phase0c_hidden_dynamics.common import load_config
from state_diff.env.block_pushing.soft_block_pushing import SoftBlockPushEnv

CONFIG = "configs/experiment3/hlf_sbp_phase0c_dynamics.json"


def test_state_sensor_and_oracle_are_separate(monkeypatch):
    config = load_config(CONFIG)
    config["execution"]["pre_snapshot_settle_outer_steps"] = 1
    env = SoftBlockPushEnv(config)
    try:
        obs = env._observation()
        state = env.get_statediff_state()
        assert state.shape == (74,)
        assert np.array_equal(state[:72], obs["deformable_keypoints"].reshape(-1))
        assert np.array_equal(state[72:], obs["ee_position"][:2])
        joints, velocities, torque = env.robot.get_joints_measured()
        reaction = env.robot.get_joint_reaction_wrenches().reshape(-1)
        sensor = env.get_contact_sensor()
        assert sensor.shape == (45,)
        assert np.array_equal(sensor[:6], torque)
        assert np.array_equal(sensor[6:42], reaction)
        assert np.array_equal(sensor[42:], obs["ee_tracking_error"])
        monkeypatch.setattr(env, "_oracle_ee_contact_wrench",
                            lambda: (_ for _ in ()).throw(AssertionError("oracle leak")))
        assert env.get_contact_sensor().shape == (45,)
        assert env.get_extended_proprio().shape == (24,)
        assert not np.array_equal(state[-2:], joints[:2])
        assert velocities.shape == (6,)
    finally:
        env.close()
