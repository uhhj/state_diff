import os
import numpy as np
import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MUJOCO_GL", "egl")
pytest.importorskip("mujoco")

from state_diff.env.ccda_hose.config import FREE_INSERT, RIGHT_HIDDEN_JAM, HoseEnvConfig
from state_diff.env.ccda_hose.audit import AuditThresholds, compute_pair_metrics
from state_diff.env.ccda_hose.env import HiddenJamHoseInsertionEnv, scripted_rollout
from state_diff.env.ccda_hose.model import make_hose_insert_xml


BRANCHES = {
    "success_insert",
    "lateral_jam",
    "s_buckle",
    "half_insert_wrong_angle",
    "failed_insert",
}


def _frame_diff(a, b):
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
        camera_name="debug_close",
    )
    assert "trace" in rollout
    assert "visible_state" in rollout["trace"]
    assert rollout["final_branch"] in BRANCHES


def test_ccda_hose_transparent_xml_has_visual_socket_and_debug_camera():
    cfg = HoseEnvConfig()
    xml = make_hose_insert_xml(
        cfg,
        condition=RIGHT_HIDDEN_JAM,
        transparent_socket=True,
        show_occluder=False,
    )
    assert 'name="socket_wall_right"' in xml
    assert 'name="socket_wall_left"' in xml
    assert 'name="socket_wall_top"' in xml
    assert 'name="socket_wall_bottom"' in xml
    assert 'name="socket_visual_right"' in xml
    assert 'name="socket_visual_left"' in xml
    assert 'name="socket_visual_top"' in xml
    assert 'name="socket_visual_bottom"' in xml
    assert 'name="socket_visual_right" type="box"' in xml
    assert 'contype="0" conaffinity="0"' in xml
    assert 'name="debug_close"' in xml
    assert 'name="front_occluder_right"' not in xml

    formal_xml = make_hose_insert_xml(cfg, condition=RIGHT_HIDDEN_JAM)
    assert 'name="socket_visual_right"' not in formal_xml
    assert 'rgba="0.05 0.05 0.05 1" contype="1" conaffinity="1"' in formal_xml


def test_ccda_hose_pair_audit_metrics_and_geometry_success():
    cfg = HoseEnvConfig(approach_steps=4, push_steps=8, settle_steps=2)
    rollouts = [
        scripted_rollout(FREE_INSERT, seed=10, config=cfg, record_frames=False),
        scripted_rollout(RIGHT_HIDDEN_JAM, seed=11, config=cfg, record_frames=False),
    ]
    traces = [r["trace"] for r in rollouts]
    data = {k: np.stack([t[k] for t in traces], axis=0) for k in traces[0].keys()}
    data["condition"] = np.array([r["condition"] for r in rollouts])
    data["seed"] = np.array([int(r["seed"]) for r in rollouts])
    data["audit_index"] = np.array([int(r["audit_index"]) for r in rollouts])
    data["final_success"] = np.array([float(r["final_success"]) for r in rollouts])
    data["final_branch"] = np.array([str(r["final_branch"]) for r in rollouts])

    rows, summary = compute_pair_metrics(
        data,
        history_steps=3,
        future_steps=4,
        thresholds=AuditThresholds(),
        max_pairs_per_type=10,
    )
    assert rows
    assert "A_vs_B" in summary["by_pair_type"]

    jam_rollout = rollouts[1]
    final = jam_rollout["trace"]
    geometry_success = (
        final["insertion_depth"][-1, 0] >= cfg.success_insert_depth
        and abs(final["lateral_offset"][-1, 0]) <= cfg.lateral_offset_threshold
        and final["max_curvature"][-1, 0] <= cfg.max_curvature_success_threshold
    )
    assert bool(jam_rollout["final_success"]) == bool(geometry_success)
