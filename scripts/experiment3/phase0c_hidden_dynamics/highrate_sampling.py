"""High-rate causal sampling utilities for Phase 0C-R1."""
from __future__ import annotations

import numpy as np


def sample_post_step(env) -> dict:
    return {
        "state": env.get_statediff_state(),
        "sensor": env.get_contact_sensor(),
        "proprio": env.get_extended_proprio(),
        "goal_distance": float(
            np.linalg.norm(
                env.soft_block.center_of_mass()[:2]
                - np.asarray(
                    env.config["goal"]["center_xy"],
                    dtype=np.float64,
                )
            )
        ),
    }


def stack_rows(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("no high-rate rows")
    return {
        key: np.asarray([row[key] for row in rows])
        for key in rows[0]
    }


def build_policy_sensor_windows(
    sensor_outer: np.ndarray,
    phase_outer: np.ndarray,
    steps_per_policy: int,
):
    sensor_outer = np.asarray(sensor_outer, dtype=np.float64)
    phase_outer = np.asarray(phase_outer).astype(str)

    if len(sensor_outer) != len(phase_outer):
        raise ValueError("sensor/phase length mismatch")
    if len(sensor_outer) % steps_per_policy:
        raise ValueError("outer samples do not align to policy intervals")

    windows = []
    labels = []
    for start in range(0, len(sensor_outer), steps_per_policy):
        stop = start + steps_per_policy
        block_phase = phase_outer[start:stop]
        if np.any(block_phase != block_phase[0]):
            raise ValueError("policy window crosses phase boundary")
        windows.append(sensor_outer[start:stop])
        labels.append(block_phase[0])

    return np.asarray(windows), np.asarray(labels)
