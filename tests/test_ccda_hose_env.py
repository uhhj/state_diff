import os
import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MUJOCO_GL", "egl")
pytest.importorskip("mujoco")

from state_diff.env.ccda_hose.config import FREE_INSERT, RIGHT_HIDDEN_JAM, HoseEnvConfig
from state_diff.env.ccda_hose.env import HiddenJamHoseInsertionEnv, scripted_rollout


BRANCHES = {
    "success_insert",
    "lateral_jam",
    "s_buckle",
    "half_insert_wrong_angle",
    "failed_insert",
}


def _frame_diff(a, b):
    import numpy as np

    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))


def test_ccda_hose_env_reset_and_step():
    cfg = HoseEnvConfig(approach_steps=3, push_steps=3, settle_steps=2)
    env = HiddenJamHoseInsertionEnv(config=cfg, condition=FREE_INSERT, seed=0)
    obs = env.reset(condition=FREE_INSERT, seed=0)
    assert "visible_state" in obs
    assert "proprio" in obs
    assert "hose_keypoints" in obs
    assert obs["hose_keypoints"].shape[0] == cfg.n_segments + 1
    obs2 = env.step()
    assert obs2["visible_state"].ndim == 1
    env.close()


def test_ccda_hose_scripted_rollout_shapes():
    cfg = HoseEnvConfig(approach_steps=4, push_steps=5, settle_steps=2)
    rollout = scripted_rollout(RIGHT_HIDDEN_JAM, seed=3, config=cfg, record_frames=False)
    trace = rollout["trace"]
    expected_len = 1 + cfg.approach_steps + cfg.push_steps
    assert trace["visible_state"].shape[0] == expected_len
    assert trace["proprio"].shape[0] == expected_len
    assert trace["action"].shape[0] == expected_len
    assert trace["privileged_contact"].shape[1] == 8
    assert rollout["final_branch"] in BRANCHES


def test_ccda_hose_side_top_render_changes():
    cfg = HoseEnvConfig(approach_steps=3, push_steps=8, settle_steps=2)
    rollout = scripted_rollout(FREE_INSERT, seed=0, config=cfg, record_frames=True, camera_name="side_top")
    frames = rollout["frames"]
    assert len(frames) > 2
    assert _frame_diff(frames[0], frames[-1]) >= 0.0


def test_ccda_hose_transparent_rollout():
    cfg = HoseEnvConfig(approach_steps=3, push_steps=8, settle_steps=2)
    rollout = scripted_rollout(
        RIGHT_HIDDEN_JAM,
        seed=4,
        config=cfg,
        transparent_socket=True,
        show_occluder=False,
        record_frames=False,
        camera_name="side_top",
    )
    assert "trace" in rollout
    assert "visible_state" in rollout["trace"]
    assert rollout["final_branch"] in BRANCHES
