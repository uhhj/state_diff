from pathlib import Path

import numpy as np

from scripts.experiment3.phase0_soft_blockpush.common import load_config
from state_diff.env.block_pushing.soft_block_pushing import SoftBlockPushEnv


CONFIG = Path(__file__).resolve().parents[2] / "configs/experiment3/soft_blockpush_phase0b.json"


def test_soft_env_observation_sensors_render_and_restore():
    config = load_config(str(CONFIG))
    config["execution"]["pre_snapshot_settle_steps"] = 2
    env = SoftBlockPushEnv(config)
    try:
        first = env.reset()
        state = env.get_lowdim_state()
        order = env.soft_block.positions().copy()
        sample = env.trace_sample()
        assert first["deformable_keypoints"].shape == (24, 3)
        assert state.shape == (98,)
        assert sample["joint_motor_torque"] and np.asarray(sample["joint_reaction_wrench"]).shape == (6, 6)
        assert np.asarray(sample["ee_tracking_error_xyz"]).shape == (3,)
        assert np.asarray(sample["ee_contact_wrench"]).shape == (6,)
        image = env.render()
        assert image.shape == (240, 320, 3)
        visuals = env.pybullet_client.getVisualShapeData(env.floor.patch_body_id)
        assert not visuals or all(np.isclose(shape[7][3], 0.0) for shape in visuals)
        for _ in range(2):
            env.step_one_physics()
        assert np.all(np.isfinite(env.soft_block.positions()))
        env.restore_saved_state(env.saved_state_id)
        assert np.array_equal(order, env.soft_block.positions())
    finally:
        env.close()
