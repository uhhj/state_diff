import numpy as np
import pytest

from ccda_phase3.schema_v2 import (
    ACTION_DIM,
    FORMAL_CONDITIONS,
    PAPER_X_DIM,
    STATE_ACTION_X_DIM,
    STATE_DIM,
    build_window,
    future_states,
    left_padded_history,
    state_v2_from_info,
)


def info(*, beads=24, value=0.0):
    return {
        "extras": {
            "bead_positions": [[value + index * 0.001, 0.0, 0.01] for index in range(beads)],
            "robot_pose_proxy": {
                "source": "test",
                "joint_positions": [],
                "joint_velocities": [],
                "ee_position": [0.0, 0.0, 0.0],
                "ee_orientation": [0.0, 0.0, 0.0, 1.0],
            },
            "hidden_contact_meta": {"label": 999},
            "bead_velocities": [[999.0, 999.0, 0.0] for _ in range(beads)],
        }
    }


def test_formal_dimensions_and_conditions():
    assert STATE_DIM == 87
    assert PAPER_X_DIM == 261
    assert STATE_ACTION_X_DIM == 303
    assert ACTION_DIM == 14
    assert FORMAL_CONDITIONS == ("free", "hidden_slack_breakaway_pin_v2")


def test_state_uses_position_and_robot_only():
    state, source, count = state_v2_from_info(info(value=0.25))
    assert state.shape == (87,)
    assert source == "test"
    assert count == 24
    assert 999.0 not in state


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_rejected(bad):
    value = info()
    value["extras"]["bead_positions"][0][0] = bad
    with pytest.raises(ValueError):
        state_v2_from_info(value)


def test_bead_count_rejected():
    with pytest.raises(ValueError):
        state_v2_from_info(info(beads=23))


def test_history_and_future_padding_boundaries():
    states = np.arange(5 * STATE_DIM, dtype=np.float32).reshape(5, STATE_DIM)
    history = left_padded_history(states, end_index=1, th=3)
    np.testing.assert_array_equal(history, states[[0, 0, 1]])
    middle = left_padded_history(states, end_index=3, th=3)
    np.testing.assert_array_equal(middle, states[[1, 2, 3]])
    future = future_states(states, current_index=3, tf=4)
    np.testing.assert_array_equal(future, np.repeat(states[4:5], 4, axis=0))


def test_window_shapes_and_action_history_excludes_target():
    states = np.arange(6 * STATE_DIM, dtype=np.float32).reshape(6, STATE_DIM)
    actions = np.arange(5 * ACTION_DIM, dtype=np.float32).reshape(5, ACTION_DIM)
    window = build_window(states=states, action_vectors=actions, current_index=2)
    assert window["paper_x"].shape == (261,)
    assert window["state_action_x"].shape == (303,)
    assert window["y_state"].shape == (4, 87)
    assert window["y_action"].shape == (14,)
    action_history = window["state_action_x"][261:].reshape(3, 14)
    np.testing.assert_array_equal(action_history[0], np.zeros(14, dtype=np.float32))
    np.testing.assert_array_equal(action_history[1:], actions[:2])
    assert not np.array_equal(action_history[-1], actions[2])
