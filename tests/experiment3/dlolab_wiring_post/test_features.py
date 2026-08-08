import numpy as np

from scripts.experiment3.dlolab_wiring_post.features import (
    minimum_surface_clearance,
    symmetric_chamfer,
    wrap_angle,
)


def test_symmetric_chamfer_zero_for_identical_points():
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ])

    assert symmetric_chamfer(
        points,
        points,
    ) == 0.0


def test_clearance_uses_rope_and_post_radii():
    rope = np.array([
        [0.04, 0.0, 0.0],
        [0.10, 0.0, 0.0],
    ])

    clearance = minimum_surface_clearance(
        rope,
        np.array(
            [0.0, 0.0]
        ),
        rope_radius_m=0.01,
        post_radius_m=0.02,
    )

    assert np.isclose(
        clearance,
        0.01,
    )


def test_wrap_angle_preserves_rope_order():
    theta = np.linspace(
        -1.0,
        1.0,
        8,
    )

    rope = np.stack(
        [
            0.05
            * np.cos(
                theta
            ),
            0.05
            * np.sin(
                theta
            ),
            np.zeros_like(
                theta
            ),
        ],
        axis=1,
    )

    result = wrap_angle(
        rope,
        np.array(
            [0.0, 0.0]
        ),
        roi_radius_m=0.08,
    )

    assert np.isclose(
        result,
        2.0,
    )
