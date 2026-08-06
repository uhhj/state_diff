import numpy as np

from state_diff.env.block_pushing.block_pushing_multimodal import BlockPushMultimodal


def test_original_blockpush_reset_step_render():
    env = BlockPushMultimodal(seed=0)
    try:
        obs = env.reset()
        assert obs is not None
        action = np.zeros(2, dtype=np.float32)
        next_obs, reward, done, info = env.step(action)
        assert next_obs is not None
        image = env.render()
        assert image.ndim == 3
        assert np.isfinite(reward)
    finally:
        env.close()
