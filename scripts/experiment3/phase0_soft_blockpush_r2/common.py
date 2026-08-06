"""Shared R2 configuration, profile, and selection helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from scripts.experiment3.phase0_soft_blockpush.common import write_json
from state_diff.env.block_pushing.kelvin_voigt_soft_block import (
    load_material_profile)
from state_diff.env.block_pushing.soft_block_lattice import SoftBlockConfig


def load_config(path: str) -> Dict[str, Any]:
    """Load and minimally validate the frozen R2 calibration config."""
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("phase_name") != "phase0b-r2-material-calibration":
        raise ValueError("not a Phase 0B-R2 config")
    if config["physics"]["microstep_candidates"] != [4, 8, 16, 32]:
        raise ValueError("R2 microstep candidates are frozen")
    if tuple(config["soft_block"]["grid_shape"]) != (6, 4, 3):
        raise ValueError("R2 requires 6x4x3")
    return config


def load_profile(path: str) -> tuple[Any, dict]:
    """Load an R2 profile with coefficient recomputation checks."""
    material, payload = load_material_profile(path)
    if payload.get("profile_version") != 2:
        raise ValueError("R2 requires profile_version=2")
    return material, payload


def soft_block_config(config: dict, grid_shape: Optional[tuple] = None) -> SoftBlockConfig:
    """Construct a typed block config while preserving node mass for small cubes."""
    payload = config["soft_block"]
    nx, ny, nz = grid_shape or tuple(payload["grid_shape"])
    node_mass = float(payload["total_mass_kg"]) / np.prod(payload["grid_shape"])
    return SoftBlockConfig(
        nx=int(nx), ny=int(ny), nz=int(nz),
        node_radius_m=float(payload["node_radius_m"]),
        spacing_xyz_m=tuple(payload["spacing_m"]),
        total_mass_kg=float(node_mass * nx * ny * nz),
        node_lateral_friction=float(payload["node_lateral_friction"]),
        node_rolling_friction=float(payload["node_rolling_friction"]),
        linear_damping=float(payload["linear_damping"]),
        angular_damping=float(payload["angular_damping"]),
        structural_max_force_n=.8, shear_max_force_n=.35,
        bending_max_force_n=.15)


def ramp_scale(one_based_step: int, total_steps: int) -> float:
    """Return a half-cosine load scale for a one-based outer step."""
    if not 1 <= one_based_step <= total_steps:
        raise ValueError("ramp step outside range")
    return float(.5 - .5 * np.cos(np.pi * one_based_step / total_steps))


def load_selected_microsteps(path: str) -> dict:
    """Load the sole mechanics-validation microstep selection artifact."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("verdict") != "PHASE0B_R2_MICROSTEP_VALIDATION_COMPLETE":
        raise ValueError("microstep validation is not complete")
    if payload["microsteps_per_outer"] not in (4, 8, 16):
        raise ValueError("selected microsteps must have a 2M reference")
    return payload


def select_next_profile(profile_name: str, verdict: str,
                        profiles_run: int) -> Optional[str]:
    """Centralize the at-most-two-profile A6→A4/A8 selection rule."""
    if profiles_run >= 2:
        return None
    if profile_name != "kv_r2_a6_z025":
        return None
    if verdict.endswith("TOO_SOFT"):
        return "kv_r2_a8_z025"
    if verdict.endswith("TOO_STIFF"):
        return "kv_r2_a4_z025"
    return None


__all__ = ["load_config", "load_profile", "load_selected_microsteps",
           "ramp_scale", "select_next_profile", "soft_block_config", "write_json"]
