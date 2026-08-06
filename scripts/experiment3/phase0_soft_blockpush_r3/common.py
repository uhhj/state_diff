"""Frozen R3 configuration, profile, and selection helpers."""
import json
from pathlib import Path

from scripts.experiment3.phase0_soft_blockpush.common import write_json
from scripts.experiment3.phase0_soft_blockpush_r2.common import (
    load_selected_microsteps, ramp_scale, soft_block_config)
from state_diff.env.block_pushing.angle_elastic_soft_block import (
    load_angle_material_profile)

START_PROFILE = "kv_angle_r3_a055_z025"
SOFTER_PROFILE = "kv_angle_r3_a040_z025"
STIFFER_PROFILE = "kv_angle_r3_a070_z025"


def load_config(path: str) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("phase_name") != "phase0b-r3-angle-elastic-calibration":
        raise ValueError("not a Phase 0B-R3 config")
    if tuple(config["soft_block"]["grid_shape"]) != (6, 4, 3):
        raise ValueError("R3 requires 6x4x3")
    if config["physics"]["frozen_microsteps_per_outer"] != 8:
        raise ValueError("R3 freezes M8")
    if config["physics"]["angle_confirmation_microsteps"] != [8, 16]:
        raise ValueError("R3 angle confirmation must be M8/M16")
    return config


def load_profile(path: str):
    return load_angle_material_profile(path)


def choose_next_profile(shear_verdict: str):
    if "_SHEAR_" not in shear_verdict:
        return None
    if shear_verdict.endswith("TOO_SOFT"):
        return STIFFER_PROFILE
    if shear_verdict.endswith("TOO_STIFF"):
        return SOFTER_PROFILE
    return None


__all__ = ["START_PROFILE", "SOFTER_PROFILE", "STIFFER_PROFILE",
           "choose_next_profile", "load_config", "load_profile",
           "load_selected_microsteps", "ramp_scale", "soft_block_config",
           "write_json"]
