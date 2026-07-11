import numpy as np
import pytest

from ccda_phase3.data_io import finite_velocity_or_difference, state_from_info
from ccda_phase3.rollout import state_from_live_info


def make_info(xy, velocity):
    return {
        "extras": {
            "bead_positions": [[float(x), float(y), 0.01] for x, y in xy],
            "bead_velocities": [[float(vx), float(vy), 0.0] for vx, vy in velocity],
            "robot_pose_proxy": {
                "source": "test",
                "joint_positions": [0.0] * 6,
                "joint_velocities": [0.0] * 6,
                "ee_position": [0.4, 0.0, 0.2],
                "ee_orientation": [0.0, 0.0, 0.0, 1.0],
            },
        }
    }


def test_valid_zero_velocity_is_preserved():
    xy = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    velocity = np.zeros_like(xy)
    result = finite_velocity_or_difference(xy, velocity, prev_xy=xy - 1.0)
    np.testing.assert_array_equal(result, velocity)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_partial_nonfinite_velocity_falls_back(bad_value):
    previous = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    current = previous + 0.02
    velocity = np.zeros_like(current)
    velocity[0, 0] = bad_value
    result = finite_velocity_or_difference(current, velocity, prev_xy=previous, dt=2.0)
    np.testing.assert_allclose(result, np.full_like(current, 0.01), atol=1e-7, rtol=0.0)
    assert np.all(np.isfinite(result))


def test_nonfinite_without_previous_returns_zero():
    xy = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    velocity = np.array([[np.nan, 0.0], [0.0, 0.0]], dtype=np.float32)
    result = finite_velocity_or_difference(xy, velocity, prev_xy=None)
    np.testing.assert_array_equal(result, np.zeros_like(xy))


def test_offline_live_state_parity():
    previous = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    current = previous + 0.01
    velocity = np.array([[np.nan, 0.0], [0.0, 0.0]], dtype=np.float32)
    info = make_info(current, velocity)
    offline, _, _ = state_from_info(info, prev_xy=previous)
    live = state_from_live_info(info, prev_xy=previous)
    np.testing.assert_allclose(offline, live, atol=0.0, rtol=0.0)
    assert np.all(np.isfinite(offline))


def test_nonfinite_xy_is_rejected():
    xy = np.array([[np.nan, 0.2], [0.3, 0.4]], dtype=np.float32)
    with pytest.raises(ValueError):
        finite_velocity_or_difference(xy, np.zeros_like(xy))
