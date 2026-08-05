"""Hidden-Hook environment configuration and paired actions."""
from __future__ import annotations

import os
from typing import Any, Dict, List

import numpy as np

from ravens.tasks.ccda_hidden_hook_geometry import (
    HiddenHookGeometryConfig,
    compute_hidden_hook_layout,
)


def _pose(position):
    return {
        "position": np.asarray(position, dtype=np.float64).astype(float).tolist(),
        "quaternion": [0.0, 0.0, 0.0, 1.0],
    }


def hook_geometry_config(config):
    hook = config["hook"]
    action = config["action"]
    bounds = config["workspace_bounds"]
    return HiddenHookGeometryConfig(
        center_ratio=hook["center_ratio"],
        mouth_offset=hook["mouth_offset"],
        width=hook["width"],
        depth=hook["depth"],
        thickness=hook["thickness"],
        height=hook["height"],
        probe_distance=action["probe_distance"],
        main_pull_distance=action["main_pull_distance"],
        workspace_x=tuple(bounds["x"]),
        workspace_y=tuple(bounds["y"]),
    )


def configure_hidden_hook_environment(config, condition, seed, group_id):
    if condition not in {"free", "hidden_hook"}:
        raise ValueError(f"unsupported hidden-hook condition {condition}")
    hook = config["hook"]
    action = config["action"]
    bounds = config["workspace_bounds"]
    values = {
        "CCDA_HIDDEN_CONDITION": condition,
        "CCDA_VISIBLE_SEED": seed,
        "CCDA_PAIR_GROUP": group_id,
        "CCDA_TRACE_STRIDE": config["trace_stride"],
        "CCDA_FRICTION_MECHANISM": "external_patch",
        "CCDA_HOOK_CENTER_RATIO": hook["center_ratio"],
        "CCDA_HOOK_MOUTH_OFFSET": hook["mouth_offset"],
        "CCDA_HOOK_WIDTH": hook["width"],
        "CCDA_HOOK_DEPTH": hook["depth"],
        "CCDA_HOOK_THICKNESS": hook["thickness"],
        "CCDA_HOOK_HEIGHT": hook["height"],
        "CCDA_HOOK_MIN_INITIAL_CLEARANCE": hook["min_initial_clearance"],
        "CCDA_HOOK_PROBE_DISTANCE": action["probe_distance"],
        "CCDA_HOOK_MAIN_PULL_DISTANCE": action["main_pull_distance"],
        "CCDA_HOOK_WORKSPACE_X_MIN": bounds["x"][0],
        "CCDA_HOOK_WORKSPACE_X_MAX": bounds["x"][1],
        "CCDA_HOOK_WORKSPACE_Y_MIN": bounds["y"][0],
        "CCDA_HOOK_WORKSPACE_Y_MAX": bounds["y"][1],
    }
    for name, value in values.items():
        os.environ[name] = str(value)


def generate_hidden_hook_action_script(config, beads) -> List[Dict[str, Any]]:
    positions = np.asarray(beads, dtype=np.float64)
    layout = compute_hidden_hook_layout(positions, hook_geometry_config(config))
    action = config["action"]
    pick_z = float(action["pick_z"])

    probe_start = positions[int(layout["probe_index"])].copy()
    probe_start[2] = pick_z
    probe_target = probe_start.copy()
    probe_target[:2] = np.asarray(layout["probe_target_xy"])

    main_start = positions[int(layout["endpoint_index"])].copy()
    main_start[2] = pick_z
    main_target = main_start.copy()
    main_target[:2] = np.asarray(layout["main_target_xy"])

    return [
        {
            "name": "hook_probe",
            "phase": "preload",
            "primitive": "pick_precise_probe_return",
            "pose0": _pose(probe_start),
            "pose_probe": _pose(probe_target),
            "pose_return": _pose(probe_start),
            "lift_height": float(action["probe_lift_height"]),
            "hold_steps": int(action["probe_hold_steps"]),
            "return_hold_steps": int(action["probe_return_hold_steps"]),
            "post_release_steps": int(action["probe_post_release_steps"]),
            "approach_height": float(action["probe_approach_height"]),
            "retreat_z": float(action["probe_retreat_z"]),
            "joint_tolerance": float(action["joint_tolerance"]),
            "cartesian_tolerance": float(action["cartesian_tolerance"]),
            "min_achieved_fraction": float(action["min_achieved_fraction"]),
            "layout": layout,
        },
        {
            "name": "hook_main_pull",
            "phase": "main_pull",
            "primitive": "pick_place",
            "pose0": _pose(main_start),
            "pose1": _pose(main_target),
        },
    ]
