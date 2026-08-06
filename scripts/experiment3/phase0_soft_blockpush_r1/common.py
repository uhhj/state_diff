"""Shared configuration and array helpers for Phase 0B-R1."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from scripts.experiment3.phase0_soft_blockpush.common import (
    validate_config as validate_base_config, write_json)
from state_diff.env.block_pushing.kelvin_voigt_soft_block import (
    load_material_profile)


SPRING_FIELDS = (
    "spring_energy_structural_j", "spring_energy_shear_j",
    "spring_energy_bending_j", "spring_energy_total_j",
    "spring_max_abs_force_n", "spring_capped_force_count",
    "spring_force_evaluation_count")


def load_config(path: str) -> Dict[str, Any]:
    """Load and validate the frozen R1 configuration."""
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_base_config(config)
    if config.get("phase_name") != "phase0b-r1-compliant-soft-block":
        raise ValueError("not a Phase 0B-R1 config")
    if config["soft_block"].get("material_model") != "kelvin_voigt":
        raise ValueError("R1 requires kelvin_voigt")
    return config


def with_material_profile(config: Dict[str, Any], path: str) -> Dict[str, Any]:
    """Return a config copy referring to one validated material profile."""
    result = copy.deepcopy(config)
    _, payload = load_material_profile(path)
    result["soft_block"]["material_profile_path"] = str(path)
    result["material_profile"] = payload
    return result


def trace_arrays(trace: list, fields: tuple) -> Dict[str, np.ndarray]:
    """Convert fixed-shape trace fields to non-object NumPy arrays."""
    return {field: np.asarray([row[field] for row in trace]) for field in fields}


def sustained_max(values: np.ndarray, eligible: np.ndarray,
                  consecutive: int) -> float:
    """Return the largest minimum over eligible consecutive windows."""
    values, eligible = np.asarray(values), np.asarray(eligible, dtype=bool)
    candidates = [float(np.min(values[i:i + consecutive]))
                  for i in range(len(values) - consecutive + 1)
                  if np.all(eligible[i:i + consecutive])]
    return max(candidates) if candidates else float("-inf")


__all__ = ["SPRING_FIELDS", "load_config", "sustained_max", "trace_arrays",
           "with_material_profile", "write_json"]
