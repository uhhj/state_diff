"""Hidden-Hook environment configuration and paired actions."""
from __future__ import annotations

import os
from typing import Any, Dict, List

import numpy as np

from ravens.tasks.ccda_hidden_hook_geometry import (
    HiddenHookGeometryConfig,
    compute_hidden_hook_layout,
)
from ravens.tasks.ccda_hidden_latch_geometry import (
    HiddenLatchGeometryConfig,
    compute_hidden_latch_layout,
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


def latch_geometry_config(config):
    latch = config["latch"]
    bounds = config["workspace_bounds"]
    return HiddenLatchGeometryConfig(
        topology_id=str(
            config.get("topology_id", "delayed_z_latch_v1")
        ),
        center_ratio=latch["center_ratio"],
        stop_clearance=latch["stop_clearance"],
        roof_clearance=latch["roof_clearance"],
        wall_thickness=latch["wall_thickness"],
        wall_width=latch["wall_width"],
        wall_height=latch["wall_height"],
        roof_depth=latch["roof_depth"],
        roof_width=latch["roof_width"],
        roof_thickness=latch["roof_thickness"],
        main_pull_distance=config["action"]["main_pull_distance"],
        workspace_x=tuple(bounds["x"]),
        workspace_y=tuple(bounds["y"]),
    )


def configure_hidden_latch_environment(config, condition, seed, group_id):
    if condition not in {"free", "hidden_hook"}:
        raise ValueError(f"unsupported hidden-latch condition {condition}")
    latch = config["latch"]
    bounds = config["workspace_bounds"]
    values = {
        "CCDA_HIDDEN_CONDITION": condition,
        "CCDA_VISIBLE_SEED": seed,
        "CCDA_PAIR_GROUP": group_id,
        "CCDA_TRACE_STRIDE": config["trace_stride"],
        "CCDA_FRICTION_MECHANISM": "external_patch",
        "CCDA_LATCH_TOPOLOGY_ID": str(
            config.get("topology_id", "delayed_z_latch_v1")
        ),
        "CCDA_LATCH_CENTER_RATIO": latch["center_ratio"],
        "CCDA_LATCH_STOP_CLEARANCE": latch["stop_clearance"],
        "CCDA_LATCH_ROOF_CLEARANCE": latch["roof_clearance"],
        "CCDA_LATCH_WALL_THICKNESS": latch["wall_thickness"],
        "CCDA_LATCH_WALL_WIDTH": latch["wall_width"],
        "CCDA_LATCH_WALL_HEIGHT": latch["wall_height"],
        "CCDA_LATCH_ROOF_DEPTH": latch["roof_depth"],
        "CCDA_LATCH_ROOF_WIDTH": latch["roof_width"],
        "CCDA_LATCH_ROOF_THICKNESS": latch["roof_thickness"],
        "CCDA_LATCH_MIN_INITIAL_CLEARANCE": latch["min_initial_clearance"],
        "CCDA_LATCH_MAIN_PULL_DISTANCE": config["action"]["main_pull_distance"],
        "CCDA_LATCH_WORKSPACE_X_MIN": bounds["x"][0],
        "CCDA_LATCH_WORKSPACE_X_MAX": bounds["x"][1],
        "CCDA_LATCH_WORKSPACE_Y_MIN": bounds["y"][0],
        "CCDA_LATCH_WORKSPACE_Y_MAX": bounds["y"][1],
    }
    for name, value in values.items():
        os.environ[name] = str(value)


def generate_hidden_latch_action_script(config, beads) -> List[Dict[str, Any]]:
    positions = np.asarray(beads, dtype=np.float64)
    layout = compute_hidden_latch_layout(
        positions, latch_geometry_config(config)
    )
    action = config["action"]
    pick_z = float(action["pick_z"])
    probe_start = positions[int(layout["probe_index"])].copy()
    probe_start[2] = pick_z
    main_start = positions[int(layout["endpoint_index"])].copy()
    main_start[2] = pick_z
    main_target = main_start.copy()
    main_target[:2] = np.asarray(layout["main_target_xy"], dtype=np.float64)
    return [
        {
            "name": "latch_probe",
            "phase": "preload",
            "primitive": "pick_precise_latch_probe",
            "pose0": _pose(probe_start),
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
            "name": "latch_main_pull",
            "phase": "main_pull",
            "primitive": "pick_place",
            "pose0": _pose(main_start),
            "pose1": _pose(main_target),
        },
    ]


def generate_hidden_latch_tension_action_script(
    config,
    beads,
) -> List[Dict[str, Any]]:
    """Phase 0K fixed same-grasp, same-direction tension extension."""
    if str(config.get("topology_id")) != "delayed_z_latch_v1":
        raise ValueError("Phase 0K must restore delayed_z_latch_v1")
    if str(config.get("intervention_id")) != (
        "same_end_tension_extension_v1"
    ):
        raise ValueError("unsupported Phase 0K intervention_id")

    base_actions = generate_hidden_latch_action_script(config, beads)
    preload_actions = [
        action for action in base_actions if action["phase"] == "preload"
    ]
    main_actions = [
        action for action in base_actions if action["phase"] == "main_pull"
    ]
    if len(preload_actions) != 1 or len(main_actions) != 1:
        raise ValueError("expected one preload and one base main action")

    probe_action = preload_actions[0]
    base_main = main_actions[0]
    action = config["action"]
    pose0 = np.asarray(base_main["pose0"]["position"], dtype=np.float64)
    pose_stage1 = np.asarray(base_main["pose1"]["position"], dtype=np.float64)
    if pose0.shape != (3,) or pose_stage1.shape != (3,):
        raise ValueError("invalid base main poses")

    stage1_vector = pose_stage1[:2] - pose0[:2]
    stage1_distance = float(np.linalg.norm(stage1_vector))
    configured_stage1 = float(action["main_pull_distance"])
    if abs(stage1_distance - configured_stage1) > 1e-9:
        raise ValueError(
            "geometry stage-1 target does not match configured main_pull_distance"
        )
    if stage1_distance <= 1e-12:
        raise ValueError("zero stage-1 pull distance")

    direction = stage1_vector / stage1_distance
    extension = float(action["tension_extension_distance"])
    if not np.isfinite(extension) or extension <= 0:
        raise ValueError("tension_extension_distance must be positive")
    pose_final = pose_stage1.copy()
    pose_final[:2] += direction * extension

    bounds = config["workspace_bounds"]
    x_bounds = np.asarray(bounds["x"], dtype=np.float64)
    y_bounds = np.asarray(bounds["y"], dtype=np.float64)
    legal = bool(
        x_bounds[0] <= pose_final[0] <= x_bounds[1]
        and y_bounds[0] <= pose_final[1] <= y_bounds[1]
    )
    if not legal:
        raise RuntimeError(
            "fixed Phase 0K final target leaves workspace; "
            "do not search another extension"
        )

    final_distance = float(np.linalg.norm(pose_final[:2] - pose0[:2]))
    expected_final = configured_stage1 + extension
    if abs(final_distance - expected_final) > 1e-9:
        raise RuntimeError("Phase 0K final displacement is not collinear")

    main_action = {
        "name": "latch_same_end_tension_extension",
        "phase": "main_pull",
        "primitive": "pick_precise_tension_extension",
        "pose0": _pose(pose0),
        "pose_stage1": _pose(pose_stage1),
        "pose1": _pose(pose_final),
        "stage1_distance": configured_stage1,
        "extension_distance": extension,
        "final_distance": expected_final,
        "lift_height": float(action["tension_lift_height"]),
        "approach_height": float(action["tension_approach_height"]),
        "retreat_z": float(action["tension_retreat_z"]),
        "joint_tolerance": float(action["joint_tolerance"]),
        "cartesian_tolerance": float(action["cartesian_tolerance"]),
        "min_achieved_fraction": float(action["min_achieved_fraction"]),
    }
    return [probe_action, main_action]
