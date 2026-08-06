"""Shared decoupled-shear calibration helpers."""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Optional

from scripts.experiment3.phase0_soft_blockpush_r2.common import (
    load_config as load_r2_config, load_profile as load_r2_profile, write_json)


START_PROFILE = "kv_r2r2_s12p5_z025"
SOFTER_PROFILE = "kv_r2r2_s10p5_z025"
STIFFER_PROFILE = "kv_r2r2_s15p0_z025"
PROFILE_SHEAR = {START_PROFILE: 12.5, SOFTER_PROFILE: 10.5,
                 STIFFER_PROFILE: 15.0}


@dataclass(frozen=True)
class ProfileDecision:
    """One bounded adjacent-profile decision."""

    next_profile_name: Optional[str]
    stop: bool
    reason: str


def load_config(path: str) -> dict:
    """Load the R2-R2 calibration config and enforce frozen fields."""
    config = load_r2_config(path)
    if config.get("revision_name") != "phase0b-r2-r2-decoupled-shear-family":
        raise ValueError("not an R2-R2 config")
    if config["physics"]["frozen_microsteps_per_outer"] != 8:
        raise ValueError("R2-R2 requires M=8")
    expected = {"structural_stiffness_n_per_m": 10.0,
                "bending_stiffness_n_per_m": 1.0, "damping_ratio": .25,
                "start_shear_stiffness_n_per_m": 12.5,
                "softer_shear_stiffness_n_per_m": 10.5,
                "stiffer_shear_stiffness_n_per_m": 15.0,
                "max_profiles_run": 2}
    for key, value in expected.items():
        if config["material_family"][key] != value:
            raise ValueError("unexpected material field: " + key)
    return config


def load_profile(path: str, config: dict) -> tuple[object, dict]:
    """Load and independently verify one decoupled profile."""
    material, payload = load_r2_profile(path)
    name = payload["profile_name"]
    if name not in PROFILE_SHEAR:
        raise ValueError("profile outside R2-R2 family")
    if Path(path).stem != "soft_block_" + name:
        raise ValueError("profile path/name mismatch")
    if payload.get("profile_version") != 2 or float(
            payload.get("stiffness_multiplier", 0)) != 1.0:
        raise ValueError("profile compatibility fields changed")
    expected_k = {"structural": 10.0, "shear": PROFILE_SHEAR[name],
                  "bending": 1.0}
    if payload["base_stiffness_n_per_m"] != expected_k:
        raise ValueError("base stiffness changed")
    node_mass = float(config["soft_block"]["total_mass_kg"]) / 72.0
    reduced_mass, zeta = node_mass / 2.0, .25
    if abs(float(payload["node_mass_kg"]) - node_mass) > 1e-12:
        raise ValueError("node mass changed")
    if float(payload["damping_ratio"]) != zeta:
        raise ValueError("damping ratio changed")
    for kind, stiffness in expected_k.items():
        actual_k = float(payload[kind + "_stiffness_n_per_m"])
        actual_c = float(payload[kind + "_damping_ns_per_m"])
        expected_c = 2.0 * zeta * math.sqrt(stiffness * reduced_mass)
        if abs(actual_k - stiffness) > 1e-12:
            raise ValueError("stiffness mismatch: " + kind)
        if abs(actual_c - expected_c) > 1e-12:
            raise ValueError("damping mismatch: " + kind)
    return material, payload


def choose_next_profile(current_profile_name: str, verdict: str,
                        profiles_run: int) -> ProfileDecision:
    """Choose at most one adjacent shear profile."""
    if profiles_run >= 2:
        return ProfileDecision(None, True, "two_profile_limit_reached")
    if current_profile_name != START_PROFILE:
        return ProfileDecision(None, True, "second_profile_already_used")
    if verdict.endswith("TOO_SOFT"):
        return ProfileDecision(STIFFER_PROFILE, False, "increase_shear_to_15")
    if verdict.endswith("TOO_STIFF"):
        return ProfileDecision(SOFTER_PROFILE, False, "decrease_shear_to_10p5")
    return ProfileDecision(None, True, "engineering_or_stability_failure")


__all__ = ["PROFILE_SHEAR", "START_PROFILE", "SOFTER_PROFILE",
           "STIFFER_PROFILE", "ProfileDecision", "choose_next_profile",
           "load_config", "load_profile", "write_json"]
