"""Strict configuration, profile, and selection rules for R2-R1."""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Optional

from scripts.experiment3.phase0_soft_blockpush_r2.common import (
    load_config as load_r2_config, load_profile as load_r2_profile, write_json)


PROFILE_NAMES = {
    1.0: "kv_r2r1_a1p00_z025",
    1.25: "kv_r2r1_a1p25_z025",
    1.5: "kv_r2r1_a1p50_z025",
}
START_PROFILE = PROFILE_NAMES[1.25]
SOFTER_PROFILE = PROFILE_NAMES[1.0]
STIFFER_PROFILE = PROFILE_NAMES[1.5]


@dataclass(frozen=True)
class ProfileDecision:
    """One bounded profile transition or an explicit stop decision."""

    next_profile_name: Optional[str]
    stop: bool
    reason: str


def load_config(path: str) -> dict:
    """Load R2-compatible config and enforce the R2-R1 frozen fields."""
    config = load_r2_config(path)
    if config.get("revision_name") != "phase0b-r2-r1-low-stiffness-family":
        raise ValueError("not an R2-R1 config")
    if config["physics"]["frozen_microsteps_per_outer"] != 8:
        raise ValueError("R2-R1 must retain M=8")
    if config["physics"]["family_confirmation_microsteps"] != [8, 16]:
        raise ValueError("family confirmation must be [8,16]")
    expected = {"start_multiplier": 1.25, "softer_multiplier": 1.0,
                "stiffer_multiplier": 1.5, "max_profiles_run": 2,
                "confirmation_multiplier": 1.5}
    for key, value in expected.items():
        if config["material_family"][key] != value:
            raise ValueError("unexpected material family field: {}".format(key))
    return config


def load_profile(path: str, config: dict) -> tuple[object, dict]:
    """Load a profile and independently verify every R2-R1 coefficient."""
    material, payload = load_r2_profile(path)
    if payload.get("profile_version") != 2:
        raise ValueError("R2-R1 requires profile_version=2")
    multiplier = float(payload["stiffness_multiplier"])
    if multiplier not in PROFILE_NAMES:
        raise ValueError("profile outside R2-R1 family")
    expected_name = PROFILE_NAMES[multiplier]
    if payload["profile_name"] != expected_name:
        raise ValueError("profile name/multiplier mismatch")
    if Path(path).stem != "soft_block_" + expected_name:
        raise ValueError("profile name/path mismatch")
    expected_base = {"structural": 8.0, "shear": 3.0, "bending": 0.8}
    if payload["base_stiffness_n_per_m"] != expected_base:
        raise ValueError("base stiffness changed")
    if float(payload["damping_ratio"]) != .25:
        raise ValueError("damping ratio changed")
    node_mass = float(config["soft_block"]["total_mass_kg"]) / 72.0
    if abs(float(payload["node_mass_kg"]) - node_mass) > 1e-12:
        raise ValueError("node mass changed")
    reduced_mass = node_mass / 2.0
    for kind, base_k in expected_base.items():
        expected_k = base_k * multiplier
        expected_c = 2.0 * .25 * math.sqrt(expected_k * reduced_mass)
        actual_k = float(payload[kind + "_stiffness_n_per_m"])
        actual_c = float(payload[kind + "_damping_ns_per_m"])
        if abs(actual_k - expected_k) > 1e-12:
            raise ValueError("stiffness mismatch: " + kind)
        if abs(actual_c - expected_c) > 1e-12:
            raise ValueError("damping mismatch: " + kind)
    return material, payload


def choose_next_profile(current_profile_name: str, verdict: str,
                        profiles_run: int) -> ProfileDecision:
    """Apply the axial/shear two-profile closed-world transition rule."""
    if profiles_run >= 2:
        return ProfileDecision(None, True, "two_profile_limit_reached")
    if current_profile_name != START_PROFILE:
        return ProfileDecision(None, True, "second_profile_already_used")
    if verdict.endswith("TOO_STIFF"):
        return ProfileDecision(SOFTER_PROFILE, False, "move_to_a1p00")
    if verdict.endswith("TOO_SOFT"):
        return ProfileDecision(STIFFER_PROFILE, False, "move_to_a1p50")
    return ProfileDecision(None, True, "engineering_or_stability_failure")


__all__ = ["PROFILE_NAMES", "START_PROFILE", "SOFTER_PROFILE",
           "STIFFER_PROFILE", "ProfileDecision", "choose_next_profile",
           "load_config", "load_profile", "write_json"]
