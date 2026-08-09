import numpy as np

from scripts.experiment3.dlolab_wiring_post.screening_audit_rev3 import (
    global_open_rope_angle,
    horizon_profile_for_pairs,
    ordered_visibility_diagnostic,
    symmetric_visible_curve_distance,
)


def test_global_open_rope_angle_semicircle():
    theta = np.linspace(
        -0.5 * np.pi,
        0.5 * np.pi,
        21,
    )
    rope = np.stack(
        [
            0.05 * np.cos(theta),
            0.05 * np.sin(theta),
            np.zeros_like(theta),
        ],
        axis=1,
    )
    assert np.isclose(
        global_open_rope_angle(
            rope,
            np.array([0.0, 0.0]),
        ),
        np.pi,
        atol=1e-6,
    )


def test_ordered_diagnostic_reports_mask_mismatch():
    first = np.array([[
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [0.2, 0.0, 0.0],
    ]])
    second = np.array([[
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [0.2, 0.2, 0.0],
    ]])
    result = ordered_visibility_diagnostic(
        first,
        second,
        np.array([
            [0.2, 0.2],
            [2.0, 2.0],
        ]),
        0.055,
    )
    assert result[
        "visible_mask_mismatch_fraction"
    ] > 0.0


def test_curve_distance_uses_polyline_not_only_vertices():
    first = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
    ])
    second = np.array([
        [0.5, 0.0, 0.0],
        [1.5, 0.0, 0.0],
        [2.5, 0.0, 0.0],
    ])
    distance, _, _ = symmetric_visible_curve_distance(
        first,
        second,
        np.array([
            [10.0, 10.0],
            [20.0, 20.0],
        ]),
        0.055,
    )
    assert distance < 0.5


def test_horizons_use_their_own_eligible_time_range():
    # Pair at t=7 in a length-10 sequence:
    # t+1 and t+2 exist; t+5 does not.
    rope = np.zeros((2, 10, 3, 3), dtype=np.float64)
    rope[1, :, :, 0] = np.arange(10)[:, None] * 0.001
    post_xyz = np.zeros((2, 10, 2, 3), dtype=np.float64)

    pairs = [{
        "rollout_a": 0,
        "rollout_b": 1,
        "time_index": 7,
    }]

    pb0_cfg = {
        "observation_protocol": {
            "post_occlusion_radius_m": 0.0,
        }
    }

    result = horizon_profile_for_pairs(
        rope,
        post_xyz,
        pairs,
        pb0_cfg,
        [1, 2, 5],
    )

    assert result[
        "horizon_eligible_pair_count"
    ]["1"] == 1
    assert result[
        "horizon_eligible_pair_count"
    ]["2"] == 1
    assert result[
        "horizon_eligible_pair_count"
    ]["5"] == 0
