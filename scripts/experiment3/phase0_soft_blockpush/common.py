"""Configuration and serialization helpers for Soft BlockPush Phase 0B."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from state_diff.env.block_pushing.friction_tiles import FrictionFloorConfig
from state_diff.env.block_pushing.soft_block_lattice import SoftBlockConfig


CONDITIONS = ("uniform_low", "right_local_high")


def load_config(path: str) -> Dict[str, Any]:
    """Load and validate one Phase 0B JSON config."""
    with Path(path).open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    validate_config(config)
    return config


def validate_config(config: Dict[str, Any]) -> None:
    """Validate the frozen Phase 0B layout and positive execution counts."""
    if config.get("task_name") != "hidden-local-friction-soft-blockpush":
        raise ValueError("unexpected task_name")
    if config.get("dataset_role") != "ccda_audit":
        raise ValueError("dataset_role must be ccda_audit")
    if int(config.get("seed", -1)) < 0 or not config.get("pair_id"):
        raise ValueError("seed and pair_id are required")
    if tuple(config["soft_block"]["grid_shape"]) != (6, 4, 3):
        raise ValueError("Phase 0B requires the fixed 6x4x3 grid")
    for value in config["execution"].values():
        if int(value) <= 0:
            raise ValueError("execution values must be positive")
    if float(config["analysis"]["sigma_multiplier"]) != 5.0:
        raise ValueError("Phase 0B requires a 5-sigma threshold")
    if int(config["analysis"]["consecutive_samples"]) != 3:
        raise ValueError("Phase 0B requires three consecutive samples")
    soft_block_config(config).validate()
    floor_config(config).validate()


def soft_block_config(config: Dict[str, Any]) -> SoftBlockConfig:
    """Construct the typed lattice config from JSON."""
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
        structural_max_force_n=float(payload["structural_max_force_n"]),
        shear_max_force_n=float(payload["shear_max_force_n"]),
        bending_max_force_n=float(payload["bending_max_force_n"]))


def floor_config(config: Dict[str, Any]) -> FrictionFloorConfig:
    """Construct the typed friction-floor config from JSON."""
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


def write_json(path: str, payload: Any) -> None:
    """Write deterministic JSON while mapping non-finite floats to null."""
    def clean(value):
        if isinstance(value, dict):
            return {str(k): clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        if isinstance(value, np.ndarray):
            return clean(value.tolist())
        if isinstance(value, (np.floating, float)):
            return None if not np.isfinite(value) else float(value)
        if isinstance(value, (np.integer,)):
            return int(value)
        return value
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(clean(payload), indent=2, sort_keys=True)
                      + "\n", encoding="utf-8")
