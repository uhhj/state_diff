from __future__ import annotations

import numpy as np
import pytest

from ccda_phase3.data_io import ROBOT_PROXY_DIM
from ccda_phase3.observation_contract import (
    add_deterministic_xy_noise,
    build_causal_fd_history,
    build_model_x_v2,
    build_position_proprio_state,
    deterministic_noise,
    history_indices,
    schema_dimensions,
    split_privileged_state,
)


def test_history_indices_are_causal_and_left_padded():
    assert history_indices(0, th=3, stride=5) == [0, 0, 0]
    assert history_indices(5, th=3, stride=5) == [0, 0, 5]
    assert history_indices(20, th=3, stride=5) == [10, 15, 20]


def test_state_v2_excludes_simulator_velocity():
    xy = np.arange(48, dtype=np.float32).reshape(24, 2)
    robot = np.arange(ROBOT_PROXY_DIM, dtype=np.float32)
    state = build_position_proprio_state(xy, robot)
    assert state.shape == (87,)
    np.testing.assert_array_equal(state[:48], xy.reshape(-1))
    np.testing.assert_array_equal(state[48:], robot)


def test_split_privileged_state_round_trip():
    xy = np.arange(48, dtype=np.float32).reshape(24, 2)
    velocity = np.arange(48, dtype=np.float32).reshape(24, 2) * 0.01
    robot = np.arange(ROBOT_PROXY_DIM, dtype=np.float32)
    state = np.concatenate([xy.reshape(-1), velocity.reshape(-1), robot])
    got_xy, got_velocity, got_robot = split_privileged_state(
        state,
        n_beads=24,
    )
    np.testing.assert_array_equal(got_xy, xy)
    np.testing.assert_array_equal(got_velocity, velocity)
    np.testing.assert_array_equal(got_robot, robot)


def test_model_x_v2_dimension_is_303():
    xy = np.zeros((21, 24, 2), dtype=np.float32)
    robot = np.zeros((21, ROBOT_PROXY_DIM), dtype=np.float32)
    model_x = build_model_x_v2(
        xy,
        robot,
        step=20,
        th=3,
        stride=5,
        action_dim=14,
    )
    assert model_x.shape == (303,)
    assert np.all(model_x[-42:] == 0)


def test_causal_fd_uses_positions_not_sim_velocity():
    xy = np.zeros((3, 24, 2), dtype=np.float32)
    xy[1] = 0.01
    xy[2] = 0.03
    robot = np.zeros((3, ROBOT_PROXY_DIM), dtype=np.float32)
    value = build_causal_fd_history(
        xy,
        robot,
        step=2,
        th=3,
        stride=1,
        physics_dt=0.5,
    ).reshape(3, -1)
    velocity_start = 48
    velocity_end = 96
    np.testing.assert_allclose(
        value[1, velocity_start:velocity_end],
        0.02,
    )
    np.testing.assert_allclose(
        value[2, velocity_start:velocity_end],
        0.04,
    )


def test_deterministic_noise_is_stable_and_keyed():
    first = deterministic_noise((4, 3), std_m=0.001, key="a")
    second = deterministic_noise((4, 3), std_m=0.001, key="a")
    third = deterministic_noise((4, 3), std_m=0.001, key="b")
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, third)


def test_noisy_xy_remains_finite():
    xy = np.zeros((21, 24, 2), dtype=np.float32)
    noisy = add_deterministic_xy_noise(
        xy,
        std_m=0.001,
        key="seed=1|track=free",
    )
    assert noisy.shape == xy.shape
    assert np.all(np.isfinite(noisy))


def test_schema_dimensions():
    dims = schema_dimensions(n_beads=24, th=3, action_dim=14)
    assert dims["privileged_state_dim"] == 135
    assert dims["position_proprio_state_dim"] == 87
    assert dims["privileged_state_action_x_dim"] == 447
    assert dims["position_proprio_state_action_x_dim"] == 303


@pytest.mark.parametrize("stride", [0, -1])
def test_invalid_stride_rejected(stride):
    with pytest.raises(ValueError):
        history_indices(20, th=3, stride=stride)
