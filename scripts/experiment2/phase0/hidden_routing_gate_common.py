"""Configuration and fixed actions for hidden routing-gate cable."""
from __future__ import annotations

import os
from typing import Any, Dict, List

import numpy as np

from ravens.tasks.ccda_hidden_routing_gate_geometry import (
    HiddenRoutingGateGeometryConfig,
    compute_hidden_routing_gate_layout,
    public_routing_layout,
)


def _pose(position):
    return {
        "position": (
            np.asarray(
                position,
                dtype=np.float64,
            )
            .astype(float)
            .tolist()
        ),
        "quaternion": [
            0.0,
            0.0,
            0.0,
            1.0,
        ],
    }


def routing_geometry_config(
    config: Dict[str, Any],
) -> HiddenRoutingGateGeometryConfig:
    gate = config["routing_gate"]
    action = config["action"]
    bounds = config["workspace_bounds"]
    return HiddenRoutingGateGeometryConfig(
        center_ratio=gate["center_ratio"],
        probe_roof_clearance=gate[
            "probe_roof_clearance"
        ],
        probe_roof_depth=gate[
            "probe_roof_depth"
        ],
        probe_roof_width=gate[
            "probe_roof_width"
        ],
        probe_roof_thickness=gate[
            "probe_roof_thickness"
        ],
        barrier_offset=gate[
            "barrier_offset"
        ],
        barrier_thickness=gate[
            "barrier_thickness"
        ],
        barrier_width=gate[
            "barrier_width"
        ],
        barrier_height=gate[
            "barrier_height"
        ],
        stage1_pull_distance=action[
            "stage1_pull_distance"
        ],
        final_pull_distance=action[
            "final_pull_distance"
        ],
        target_plane_offset=gate[
            "target_plane_offset"
        ],
        target_zone_depth=gate[
            "target_zone_depth"
        ],
        target_corridor_half_width=gate[
            "target_corridor_half_width"
        ],
        leading_segment_size=int(
            gate["leading_segment_size"]
        ),
        workspace_x=tuple(bounds["x"]),
        workspace_y=tuple(bounds["y"]),
        topology_id=str(
            config["topology_id"]
        ),
    )


def configure_hidden_routing_gate_environment(
    config,
    condition,
    seed,
    group_id,
):
    if condition not in {
        "free",
        "hidden_hook",
    }:
        raise ValueError(
            "unsupported routing condition "
            f"{condition!r}"
        )

    gate = config["routing_gate"]
    action = config["action"]
    bounds = config["workspace_bounds"]

    values = {
        "CCDA_HIDDEN_CONDITION": condition,
        "CCDA_VISIBLE_SEED": seed,
        "CCDA_PAIR_GROUP": group_id,
        "CCDA_TRACE_STRIDE": (
            config["trace_stride"]
        ),
        "CCDA_FRICTION_MECHANISM": (
            "external_patch"
        ),
        "CCDA_ROUTING_TOPOLOGY_ID": (
            config["topology_id"]
        ),
        "CCDA_ROUTING_CENTER_RATIO": (
            gate["center_ratio"]
        ),
        "CCDA_ROUTING_PROBE_ROOF_CLEARANCE": (
            gate["probe_roof_clearance"]
        ),
        "CCDA_ROUTING_PROBE_ROOF_DEPTH": (
            gate["probe_roof_depth"]
        ),
        "CCDA_ROUTING_PROBE_ROOF_WIDTH": (
            gate["probe_roof_width"]
        ),
        "CCDA_ROUTING_PROBE_ROOF_THICKNESS": (
            gate["probe_roof_thickness"]
        ),
        "CCDA_ROUTING_BARRIER_OFFSET": (
            gate["barrier_offset"]
        ),
        "CCDA_ROUTING_BARRIER_THICKNESS": (
            gate["barrier_thickness"]
        ),
        "CCDA_ROUTING_BARRIER_WIDTH": (
            gate["barrier_width"]
        ),
        "CCDA_ROUTING_BARRIER_HEIGHT": (
            gate["barrier_height"]
        ),
        "CCDA_ROUTING_STAGE1_DISTANCE": (
            action["stage1_pull_distance"]
        ),
        "CCDA_ROUTING_FINAL_DISTANCE": (
            action["final_pull_distance"]
        ),
        "CCDA_ROUTING_TARGET_PLANE_OFFSET": (
            gate["target_plane_offset"]
        ),
        "CCDA_ROUTING_TARGET_ZONE_DEPTH": (
            gate["target_zone_depth"]
        ),
        "CCDA_ROUTING_CORRIDOR_HALF_WIDTH": (
            gate[
                "target_corridor_half_width"
            ]
        ),
        "CCDA_ROUTING_LEADING_SEGMENT_SIZE": (
            gate["leading_segment_size"]
        ),
        "CCDA_ROUTING_MIN_INITIAL_CLEARANCE": (
            gate["min_initial_clearance"]
        ),
        "CCDA_ROUTING_WORKSPACE_X_MIN": (
            bounds["x"][0]
        ),
        "CCDA_ROUTING_WORKSPACE_X_MAX": (
            bounds["x"][1]
        ),
        "CCDA_ROUTING_WORKSPACE_Y_MIN": (
            bounds["y"][0]
        ),
        "CCDA_ROUTING_WORKSPACE_Y_MAX": (
            bounds["y"][1]
        ),
    }
    for name, value in values.items():
        os.environ[name] = str(value)


def generate_hidden_routing_gate_action_script(
    config: Dict[str, Any],
    beads: np.ndarray,
) -> List[Dict[str, Any]]:
    positions = np.asarray(
        beads,
        dtype=np.float64,
    )
    layout = (
        compute_hidden_routing_gate_layout(
            positions,
            routing_geometry_config(config),
        )
    )
    public = public_routing_layout(layout)
    action = config["action"]

    probe_index = int(
        public["probe_index"]
    )
    endpoint_index = int(
        public["endpoint_index"]
    )
    pick_z = float(action["pick_z"])

    probe_start = positions[
        probe_index
    ].copy()
    probe_start[2] = pick_z

    main_start = positions[
        endpoint_index
    ].copy()
    main_start[2] = pick_z

    stage1_target = main_start.copy()
    stage1_target[:2] = np.asarray(
        public["stage1_target_xy"],
        dtype=np.float64,
    )
    final_target = main_start.copy()
    final_target[:2] = np.asarray(
        public["final_target_xy"],
        dtype=np.float64,
    )

    # Public action metadata must not contain any
    # hidden fixture geometry.
    serialized_public = str(public).lower()
    for forbidden in (
        "barrier",
        "roof",
        "boxes",
    ):
        if forbidden in serialized_public:
            raise RuntimeError(
                "public routing action leaks "
                f"{forbidden!r}"
            )

    return [
        {
            "name": "routing_contact_probe",
            "phase": "preload",
            "primitive": (
                "pick_precise_latch_probe"
            ),
            "pose0": _pose(probe_start),
            "lift_height": float(
                action["probe_lift_height"]
            ),
            "hold_steps": int(
                action["probe_hold_steps"]
            ),
            "return_hold_steps": int(
                action[
                    "probe_return_hold_steps"
                ]
            ),
            "post_release_steps": int(
                action[
                    "probe_post_release_steps"
                ]
            ),
            "approach_height": float(
                action[
                    "probe_approach_height"
                ]
            ),
            "retreat_z": float(
                action["probe_retreat_z"]
            ),
            "joint_tolerance": float(
                action["joint_tolerance"]
            ),
            "cartesian_tolerance": float(
                action[
                    "cartesian_tolerance"
                ]
            ),
            "min_achieved_fraction": float(
                action[
                    "min_achieved_fraction"
                ]
            ),
            "public_task_layout": public,
        },
        {
            "name": "routing_main_pull",
            "phase": "main_pull",
            "primitive": (
                "pick_precise_tension_extension"
            ),
            "pose0": _pose(main_start),
            "pose_stage1": _pose(
                stage1_target
            ),
            "pose1": _pose(final_target),
            "lift_height": float(
                action["routing_lift_height"]
            ),
            "approach_height": float(
                action[
                    "routing_approach_height"
                ]
            ),
            "retreat_z": float(
                action["routing_retreat_z"]
            ),
            "joint_tolerance": float(
                action["joint_tolerance"]
            ),
            "cartesian_tolerance": float(
                action[
                    "cartesian_tolerance"
                ]
            ),
            "min_achieved_fraction": float(
                action[
                    "min_achieved_fraction"
                ]
            ),
            "public_task_layout": public,
        },
    ]
