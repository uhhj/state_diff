import numpy as np

from scripts.experiment3.dlolab_wrapping.pair_discovery_pb2c import (
    batch_symmetric_chamfer,
    compute_visibility_mask,
    occlusion_radius,
    quaternion_geodesic_rad,
    robot_history_metrics,
    signed_winding_index,
    winding_integer_residual,
)


def _config():
    return {
        "partial_state_observation": {
            "rope_component": {
                "post_radius_m": 0.015,
                "rope_radius_m": 0.01,
                "local_occlusion_radius_multiplier": 2.0,
            }
        }
    }


def test_occlusion_radius_is_geometry_derived():
    assert np.isclose(occlusion_radius(_config()), 0.05)


def test_visibility_hides_local_post_neighborhood():
    rope = np.array(
        [[[[0.00, 0.00, 0.0], [0.06, 0.00, 0.0]]]],
        dtype=np.float32,
    )
    posts = np.array(
        [[[[0.00, 0.00, 0.0], [1.00, 1.00, 0.0], [2.00, 2.00, 0.0]]]],
        dtype=np.float32,
    )
    visible = compute_visibility_mask(rope, posts, radius_m=0.05)
    assert visible[0, 0].tolist() == [False, True]


def test_signed_winding_index_and_residual_are_both_available():
    turns = np.array([[0.0001, 0.998, -1.004]], dtype=np.float64)
    assert signed_winding_index(turns).tolist() == [[0, 1, -1]]
    assert np.allclose(
        winding_integer_residual(turns),
        [[0.0001, 0.002, 0.004]],
    )


def test_quaternion_geodesic_is_sign_invariant():
    q = np.array([[1.0, 0.0, 0.0, 0.0]])
    assert np.isclose(quaternion_geodesic_rad(q, -q)[0], 0.0)


def test_robot_history_metrics_include_pose_orientation_and_joint_state():
    data = {
        "ee1_pos": np.zeros((2, 3, 3), dtype=np.float32),
        "ee2_pos": np.zeros((2, 3, 3), dtype=np.float32),
        "ee1_quat": np.tile(
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            (2, 3, 1),
        ),
        "ee2_quat": np.tile(
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            (2, 3, 1),
        ),
        "motor_qpos_1": np.zeros((2, 3, 7), dtype=np.float32),
        "motor_qpos_2": np.zeros((2, 3, 7), dtype=np.float32),
    }
    data["ee1_pos"][1, :, 0] = 0.01
    data["ee2_pos"][1, :, 0] = 0.01
    data["motor_qpos_1"][1] = 0.02
    data["motor_qpos_2"][1] = 0.02

    result = robot_history_metrics(
        data,
        np.array([0]),
        np.array([1]),
        time_index=2,
        history=3,
    )
    assert np.isclose(result["dual_ee_position_history_mean_m"][0], 0.01)
    assert np.isclose(
        result["dual_ee_quaternion_geodesic_history_mean_rad"][0], 0.0
    )
    assert np.isclose(result["dual_motor_qpos_history_rms_rad"][0], 0.02)


def test_chamfer_zero_for_identical_visible_sets():
    points = np.array(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]],
        dtype=np.float32,
    )
    mask = np.array([[True, True]])
    distance = batch_symmetric_chamfer(points, mask, points.copy(), mask.copy())
    assert np.isclose(distance[0], 0.0)
