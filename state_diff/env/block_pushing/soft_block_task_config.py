"""Typed config helpers for the active HLF-SBP environment."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from state_diff.env.block_pushing.friction_tiles import FrictionFloorConfig
from state_diff.env.block_pushing.soft_block_lattice import SoftBlockConfig


def load_hlf_sbp_config(path: str) -> Dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_hlf_sbp_config(config)
    return config


def validate_hlf_sbp_config(config: Dict[str, Any]) -> None:
    if config.get("task_name") != "hidden-local-friction-soft-blockpush":
        raise ValueError("unexpected task_name")
    if config["soft_block"].get("material_model") != "kelvin_voigt":
        raise ValueError("active HLF-SBP uses Kelvin-Voigt only")
    if tuple(config["soft_block"]["grid_shape"]) != (6, 4, 3):
        raise ValueError("HLF-SBP requires 6x4x3")
    physics = config["physics"]
    if abs(float(physics["outer_timestep_s"]) - 1.0 / 240.0) > 1e-15:
        raise ValueError("outer dt must be 1/240 s")
    if int(physics["microsteps_per_outer"]) != 8:
        raise ValueError("active mechanics requires M8")
    outer_hz, policy_hz = int(physics["outer_hz"]), int(physics["policy_hz"])
    if outer_hz % policy_hz:
        raise ValueError("outer_hz must divide policy_hz exactly")
    stride = int(config["execution"]["policy_sample_stride_outer_steps"])
    if stride != outer_hz // policy_hz:
        raise ValueError("policy sample stride mismatch")
    if int(config["state"]["state_dim"]) != 74:
        raise ValueError("V4.1 active State Diff state is 74D")
    if int(config["sensor"]["sensor_dim"]) != 45:
        raise ValueError("V4.1 formal contact sensor is 45D")
    soft_block_config(config).validate()
    floor_config(config).validate()


def soft_block_config(config: Dict[str, Any]) -> SoftBlockConfig:
    payload = config["soft_block"]
    nx, ny, nz = payload["grid_shape"]
    return SoftBlockConfig(
        nx=int(nx), ny=int(ny), nz=int(nz),
        node_radius_m=float(payload["node_radius_m"]),
        spacing_xyz_m=tuple(float(v) for v in payload["spacing_m"]),
        total_mass_kg=float(payload["total_mass_kg"]),
        node_lateral_friction=float(payload["node_lateral_friction"]),
        node_rolling_friction=float(payload["node_rolling_friction"]),
        linear_damping=float(payload["linear_damping"]),
        angular_damping=float(payload["angular_damping"]),
        structural_max_force_n=float(payload.get("structural_max_force_n", .8)),
        shear_max_force_n=float(payload.get("shear_max_force_n", .35)),
        bending_max_force_n=float(payload.get("bending_max_force_n", .15)))


def floor_config(config: Dict[str, Any]) -> FrictionFloorConfig:
    payload = config["floor"]
    return FrictionFloorConfig(
        x_bounds=tuple(float(v) for v in payload["x_bounds"]),
        y_bounds=tuple(float(v) for v in payload["y_bounds"]),
        top_z_m=float(payload["top_z_m"]),
        thickness_m=float(payload["thickness_m"]),
        outer_lateral_friction=float(payload["outer_lateral_friction"]),
        patch_free_lateral_friction=float(payload["patch_free_lateral_friction"]),
        patch_high_lateral_friction=float(payload["patch_high_lateral_friction"]),
        spinning_friction=float(payload["spinning_friction"]),
        rolling_friction=float(payload["rolling_friction"]),
        patch_center_xy=tuple(float(v) for v in payload["patch_center_xy"]),
        patch_size_xy=tuple(float(v) for v in payload["patch_size_xy"]))
