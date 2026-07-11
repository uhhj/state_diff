import pickle
from pathlib import Path

import numpy as np

from ccda_phase3.data_io import build_windows_from_dataset


def action(value):
    quat = (0.0, 0.0, 0.0, 1.0)
    return {
        "primitive": "pick_place",
        "params": {
            "pose0": ((value, 0.0, 0.01), quat),
            "pose1": ((value + 0.1, 0.0, 0.01), quat),
        },
        "camera_config": [{"image_size": (640, 480)}],
    }


def info(seed, group, value, condition):
    return {
        "extras": {
            "hidden_condition": condition,
            "ccda_visible_seed": seed,
            "ccda_pair_group": group,
            "bead_positions": [[value + index * 0.001, 0.0, 0.01] for index in range(24)],
            "robot_pose_proxy": {
                "source": "test",
                "joint_positions": [],
                "joint_velocities": [],
                "ee_position": [0.0, 0.0, 0.0],
                "ee_orientation": [0.0, 0.0, 0.0, 1.0],
            },
            "hidden_contact_meta": {},
            "task.done": False,
            "total_rewards": 0.0,
        }
    }


def write_episode(root: Path, condition: str, value: float):
    group = "phase313_train_seed_1"
    directory = root / condition
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "infos": [info(1, group, value + step * 0.01, condition) for step in range(3)],
        "last_info": info(1, group, value + 0.03, condition),
        "actions": [action(0.1), action(0.2), action(0.3)],
        "manifest": {"visible_seed": 1, "pair_group": group, "condition": condition},
    }
    with (directory / "seed_1.pkl").open("wb") as handle:
        pickle.dump(payload, handle)


def test_real_consecutive_windows_and_no_cross_condition_copy(tmp_path):
    write_episode(tmp_path, "free", 0.0)
    write_episode(tmp_path, "hidden_slack_breakaway_pin_v2", 0.5)
    windows, codec, _ = build_windows_from_dataset("train", tmp_path, 3, 4)
    assert codec.dim() == 14
    assert len(windows) == 6
    free = [row for row in windows if row["condition_name"] == "free"]
    hidden = [row for row in windows if row["condition_name"] != "free"]
    assert not np.array_equal(free[0]["paper_x"], hidden[0]["paper_x"])
    assert free[2]["window_t"] == 2
    action_history = free[2]["state_action_x"][261:].reshape(3, 14)
    np.testing.assert_array_equal(action_history[0], np.zeros(14, dtype=np.float32))
    assert not np.array_equal(action_history[-1], free[2]["y_action"])
    for row in windows:
        for key in ("paper_x", "state_action_x", "y_state", "y_final_state", "y_action"):
            assert row[key].dtype == np.float32


def test_npz_is_pickle_free(tmp_path):
    write_episode(tmp_path, "free", 0.0)
    write_episode(tmp_path, "hidden_slack_breakaway_pin_v2", 0.5)
    windows, _, _ = build_windows_from_dataset("train", tmp_path, 3, 4)
    output = tmp_path / "windows.npz"
    np.savez_compressed(
        output,
        paper_x=np.stack([row["paper_x"] for row in windows]),
        condition_name=np.asarray([row["condition_name"] for row in windows], dtype="<U64"),
    )
    with np.load(output, allow_pickle=False) as loaded:
        assert loaded["paper_x"].dtype == np.float32
        assert loaded["condition_name"].dtype.kind == "U"
        assert all(value.dtype.kind != "O" for value in loaded.values())
