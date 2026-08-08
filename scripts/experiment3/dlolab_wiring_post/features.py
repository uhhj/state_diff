from __future__ import annotations

import numpy as np


def post_visible_mask(
        rope_xyz,
        post_xy,
        occlusion_radius_m):
    rope_xyz = np.asarray(
        rope_xyz,
        dtype=np.float64,
    )

    post_xy = np.asarray(
        post_xy,
        dtype=np.float64,
    )

    delta = (
        rope_xyz[
            :,
            None,
            :2
        ]
        - post_xy[
            None,
            :,
            :
        ]
    )

    distance = np.linalg.norm(
        delta,
        axis=-1,
    )

    return np.all(
        distance
        > float(
            occlusion_radius_m
        ),
        axis=1,
    )


def visible_points(
        rope_xyz,
        post_xy,
        occlusion_radius_m):
    mask = post_visible_mask(
        rope_xyz,
        post_xy,
        occlusion_radius_m,
    )

    return np.asarray(
        rope_xyz,
        dtype=np.float64,
    )[mask]


def symmetric_chamfer(
        points_a,
        points_b):
    a = np.asarray(
        points_a,
        dtype=np.float64,
    )

    b = np.asarray(
        points_b,
        dtype=np.float64,
    )

    if len(a) == 0 or len(b) == 0:
        return float("inf")

    delta = (
        a[
            :,
            None,
            :
        ]
        - b[
            None,
            :,
            :
        ]
    )

    distance = np.linalg.norm(
        delta,
        axis=-1,
    )

    return float(
        0.5
        * (
            distance.min(
                axis=1
            ).mean()
            + distance.min(
                axis=0
            ).mean()
        )
    )


def minimum_surface_clearance(
        rope_xyz,
        post_xy,
        rope_radius_m,
        post_radius_m):
    rope_xy = np.asarray(
        rope_xyz,
        dtype=np.float64,
    )[
        :,
        :2
    ]

    post_xy = np.asarray(
        post_xy,
        dtype=np.float64,
    )

    radial = np.linalg.norm(
        rope_xy
        - post_xy[
            None,
            :
        ],
        axis=1,
    )

    return float(
        radial.min()
        - float(
            rope_radius_m
        )
        - float(
            post_radius_m
        )
    )


def nearest_vertex_index(
        rope_xyz,
        post_xy):
    rope_xy = np.asarray(
        rope_xyz,
        dtype=np.float64,
    )[
        :,
        :2
    ]

    post_xy = np.asarray(
        post_xy,
        dtype=np.float64,
    )

    radial = np.linalg.norm(
        rope_xy
        - post_xy[
            None,
            :
        ],
        axis=1,
    )

    return int(
        np.argmin(
            radial
        )
    )


def wrap_angle(
        rope_xyz,
        post_xy,
        roi_radius_m):
    rope_xy = np.asarray(
        rope_xyz,
        dtype=np.float64,
    )[
        :,
        :2
    ]

    post_xy = np.asarray(
        post_xy,
        dtype=np.float64,
    )

    relative = (
        rope_xy
        - post_xy[
            None,
            :
        ]
    )

    radial = np.linalg.norm(
        relative,
        axis=1,
    )

    ids = np.flatnonzero(
        radial
        <= float(
            roi_radius_m
        )
    )

    if ids.size < 2:
        return 0.0

    # Preserve cable order.
    theta = np.arctan2(
        relative[
            ids,
            1
        ],
        relative[
            ids,
            0
        ],
    )

    unwrapped = np.unwrap(
        theta
    )

    return float(
        unwrapped[
            -1
        ]
        - unwrapped[
            0
        ]
    )


def oracle_descriptor(
        rope_xyz,
        post_xy,
        *,
        rope_radius_m,
        post_radius_m,
        roi_radius_m,
        contact_proxy_margin_m):
    descriptors = []

    for center in np.asarray(
            post_xy,
            dtype=np.float64):
        clearance = minimum_surface_clearance(
            rope_xyz,
            center,
            rope_radius_m,
            post_radius_m,
        )

        descriptors.append(
            {
                "wrap_angle_rad":
                    wrap_angle(
                        rope_xyz,
                        center,
                        roi_radius_m,
                    ),

                "surface_clearance_m":
                    clearance,

                "contact_like":
                    bool(
                        clearance
                        <= float(
                            contact_proxy_margin_m
                        )
                    ),

                "nearest_vertex":
                    nearest_vertex_index(
                        rope_xyz,
                        center,
                    ),
            }
        )

    return descriptors


def history_observable_distance(
        rope_history_a,
        rope_history_b,
        ee_history_a,
        ee_history_b,
        post_xy,
        occlusion_radius_m):
    rope_a = np.asarray(
        rope_history_a,
        dtype=np.float64,
    )

    rope_b = np.asarray(
        rope_history_b,
        dtype=np.float64,
    )

    ee_a = np.asarray(
        ee_history_a,
        dtype=np.float64,
    )

    ee_b = np.asarray(
        ee_history_b,
        dtype=np.float64,
    )

    if rope_a.shape != rope_b.shape:
        raise ValueError(
            "rope history shape mismatch"
        )

    rope_distance = []

    for index in range(
            rope_a.shape[
                0
            ]):
        a = visible_points(
            rope_a[
                index
            ],
            post_xy,
            occlusion_radius_m,
        )

        b = visible_points(
            rope_b[
                index
            ],
            post_xy,
            occlusion_radius_m,
        )

        rope_distance.append(
            symmetric_chamfer(
                a,
                b,
            )
        )

    ee_position_distance = float(
        np.linalg.norm(
            ee_a[
                :,
                :3
            ]
            - ee_b[
                :,
                :3
            ],
            axis=1,
        ).mean()
    )

    return (
        float(
            np.mean(
                rope_distance
            )
        ),
        ee_position_distance,
    )


def hidden_difference(
        descriptor_a,
        descriptor_b,
        wrap_difference_min_rad):
    wrap_diff = [
        abs(
            float(
                a[
                    "wrap_angle_rad"
                ]
            )
            - float(
                b[
                    "wrap_angle_rad"
                ]
            )
        )
        for a, b
        in zip(
            descriptor_a,
            descriptor_b,
        )
    ]

    contact_mismatch = any(
        bool(
            a[
                "contact_like"
            ]
        )
        != bool(
            b[
                "contact_like"
            ]
        )
        for a, b
        in zip(
            descriptor_a,
            descriptor_b,
        )
    )

    return {
        "different":
            bool(
                contact_mismatch
                or max(
                    wrap_diff,
                    default=0.0,
                )
                >= float(
                    wrap_difference_min_rad
                )
            ),

        "contact_like_mismatch":
            bool(
                contact_mismatch
            ),

        "wrap_difference_rad":
            wrap_diff,
    }
