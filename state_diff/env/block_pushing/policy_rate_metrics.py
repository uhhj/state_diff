"""Phase-aligned policy-rate resampling and physically scaled gap metrics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class PolicyRateSeries:
    """Values and labels for complete phase-local policy bins."""

    physics_step_end: np.ndarray
    phase: np.ndarray
    values: np.ndarray


def _resample(values: np.ndarray, phases: np.ndarray, physics_steps: np.ndarray,
              steps_per_policy: int, use_last: bool) -> PolicyRateSeries:
    values = np.asarray(values)
    phases = np.asarray(phases).astype(str)
    physics_steps = np.asarray(physics_steps)
    if not (len(values) == len(phases) == len(physics_steps)):
        raise ValueError("series lengths differ")
    if steps_per_policy <= 0:
        raise ValueError("steps_per_policy must be positive")
    output_values, output_phases, output_steps = [], [], []
    start = 0
    while start < len(phases):
        end = start + 1
        while end < len(phases) and phases[end] == phases[start]:
            end += 1
        if (end - start) % steps_per_policy:
            raise ValueError("phase length is not divisible by policy bin")
        for index in range(start, end, steps_per_policy):
            block = values[index:index + steps_per_policy]
            output_values.append(block[-1] if use_last else np.mean(block, axis=0))
            output_phases.append(phases[start])
            output_steps.append(physics_steps[index + steps_per_policy - 1])
        start = end
    return PolicyRateSeries(np.asarray(output_steps), np.asarray(output_phases),
                            np.asarray(output_values))


def resample_phase_aligned_mean(
    values: np.ndarray, phases: np.ndarray, physics_steps: np.ndarray,
    steps_per_policy: int,
) -> PolicyRateSeries:
    """Mean complete bins independently within each phase."""
    return _resample(values, phases, physics_steps, steps_per_policy, False)


def resample_phase_aligned_last(
    values: np.ndarray, phases: np.ndarray, physics_steps: np.ndarray,
    steps_per_policy: int,
) -> PolicyRateSeries:
    """Take each complete phase-local bin's final sample."""
    return _resample(values, phases, physics_steps, steps_per_policy, True)


def formal_feature_scale_floors(config: dict) -> np.ndarray:
    """Return physical scale floors matching the flattened formal layout."""
    floors = config["analysis"]["sensor_scale_floors"]
    values = []
    values.extend([floors["joint_motor_torque_nm"]] * 6)
    for _ in range(6):
        values.extend([floors["joint_reaction_force_n"]] * 3)
        values.extend([floors["joint_reaction_torque_nm"]] * 3)
    values.extend([floors["ee_tracking_error_m"]] * 3)
    values.extend([floors["ee_contact_force_n"]] * 3)
    values.extend([floors["ee_contact_torque_nm"]] * 3)
    return np.asarray(values, dtype=np.float64)


def scaled_fused_gap(free: np.ndarray, high: np.ndarray,
                     baseline: np.ndarray, scale_floors: np.ndarray) -> np.ndarray:
    """Return RMS branch gap scaled by pooled baseline or physical floors."""
    free = np.asarray(free, dtype=np.float64).reshape(len(free), -1)
    high = np.asarray(high, dtype=np.float64).reshape(len(high), -1)
    baseline = np.asarray(baseline, dtype=bool)
    floors = np.asarray(scale_floors, dtype=np.float64)
    if free.shape != high.shape or free.shape[1] != len(floors):
        raise ValueError("formal feature layout does not match scale floors")
    pooled = np.concatenate([free[baseline], high[baseline]], axis=0)
    scale = np.maximum(np.std(pooled, axis=0), floors)
    return np.sqrt(np.mean(np.square((free - high) / scale), axis=1))


def onset_with_floor(gap: np.ndarray, baseline: np.ndarray,
                     sigma_multiplier: float, effect_floor: float,
                     consecutive: int = 1) -> tuple[float, Optional[int]]:
    """Return nonzero threshold and first sustained non-baseline onset."""
    gap = np.asarray(gap, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=bool)
    threshold = max(float(np.mean(gap[baseline]) + sigma_multiplier
                          * np.std(gap[baseline])), float(effect_floor))
    for index in range(len(gap) - consecutive + 1):
        if (np.all(~baseline[index:index + consecutive])
                and np.all(gap[index:index + consecutive] > threshold)):
            return threshold, int(index)
    return threshold, None
