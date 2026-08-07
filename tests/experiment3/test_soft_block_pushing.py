import numpy as np

from scripts.experiment3.phase0c_hidden_dynamics.common import load_config
from state_diff.env.block_pushing.soft_block_pushing import SoftBlockPushEnv

CONFIG = "configs/experiment3/hlf_sbp_phase0c_dynamics.json"


def test_soft_env_active_state_sensors_render_and_restore():
    config = load_config(CONFIG)
    config["execution"]["pre_snapshot_settle_outer_steps"] = 2
    env = SoftBlockPushEnv(config)
    try:
        first = env.reset()
        order = env.soft_block.positions().copy()
        sample = env.trace_sample()
        assert first["deformable_keypoints"].shape == (24, 3)
        assert env.get_statediff_state().shape == (74,)
        assert env.get_contact_sensor().shape == (45,)
        assert env.get_extended_proprio().shape == (24,)
        assert np.asarray(sample["oracle_ee_contact_wrench"]).shape == (6,)
        assert env.render().shape == (240, 320, 3)
        env.step_one_physics()
        assert np.all(np.isfinite(env.soft_block.positions()))
        env.restore_saved_state(env.saved_state_id)
        assert np.array_equal(order, env.soft_block.positions())
    finally:
        env.close()
