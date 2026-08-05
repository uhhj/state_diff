"""Frozen event-aligned robot-observable sensor features for Phase 0I."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np


LOAD_STAGES = {
    "hook_probe_lift",
    "hook_probe_out",
    "latch_probe_lift",
    "latch_probe_hold",
}

UNLOAD_STAGES = {
    "hook_probe_return",
    "hook_probe_lower_release",
    "latch_probe_lower_release",
    "latch_probe_post_release",
}


def motion_stage_mask(
    physics_steps: np.ndarray,
    motion_events: Iterable[Dict[str, Any]],
    stages: Sequence[str],
) -> np.ndarray:
    """Return a closed-interval mask using only robot motion events."""
    steps = np.asarray(physics_steps)
    if steps.ndim != 1:
        raise ValueError("physics_steps must be one-dimensional")
    selected = {str(value) for value in stages}
    mask = np.zeros(steps.shape, dtype=bool)
    for event in motion_events:
        stage = str(event.get("stage", event.get("label", "")))
        if stage not in selected:
            continue
        start = event.get("physics_step_start")
        end = event.get("physics_step_end")
        if start is None or end is None:
            raise ValueError(f"motion event {stage} has no physics-step interval")
        start, end = int(start), int(end)
        if end < start:
            raise ValueError(f"motion event {stage} has a reversed interval")
        mask |= (steps >= start) & (steps <= end)
    return mask


def _continuous_stats(
    name: str,
    values: np.ndarray,
    mask: np.ndarray,
    dt: float,
) -> Tuple[List[float], List[str]]:
    selected = np.asarray(values, dtype=np.float64)[mask]
    if selected.size == 0:
        raise ValueError(f"empty formal sensor window for {name}")
    if not np.all(np.isfinite(selected)):
        raise ValueError(f"non-finite formal sensor values for {name}")
    stats = [
        float(np.mean(selected)),
        float(np.std(selected)),
        float(np.percentile(selected, 95.0)),
        float(np.sum(selected) * dt),
    ]
    schema = [
        f"{name}.mean",
        f"{name}.std",
        f"{name}.p95",
        f"{name}.integral",
    ]
    return stats, schema


def _window_feature(
    prefix: str,
    mask: np.ndarray,
    continuous: Dict[str, np.ndarray],
    grasp: np.ndarray,
    available: np.ndarray,
    dt: float,
) -> Tuple[List[float], List[str], Dict[str, float]]:
    if not np.any(mask):
        raise ValueError(f"empty event-aligned window {prefix}")
    values: List[float] = []
    schema: List[str] = []
    means: Dict[str, float] = {}
    for name in sorted(continuous):
        selected = np.asarray(continuous[name], dtype=np.float64)[mask]
        means[name] = float(np.mean(selected))
        block, names = _continuous_stats(
            f"{prefix}.{name}", continuous[name], mask, dt
        )
        values.extend(block)
        schema.extend(names)

    grasp_fraction = float(np.mean(np.asarray(grasp)[mask] > 0))
    available_fraction = float(np.mean(np.asarray(available)[mask] > 0))
    values.extend([grasp_fraction, available_fraction])
    schema.extend([
        f"{prefix}.grasp_fraction",
        f"{prefix}.constraint_available_fraction",
    ])
    return values, schema, means


def _validated_signals(trace: Dict[str, np.ndarray]):
    required = (
        "physics_step",
        "sensor_joint_motor_torque",
        "sensor_joint_reaction_force_torque",
        "sensor_suction_force_xyz",
        "sensor_suction_torque_xyz",
        "sensor_grasp_active",
        "sensor_constraint_available",
    )
    missing = [key for key in required if key not in trace]
    if missing:
        raise ValueError(f"formal sensor trace missing {missing}")
    steps = np.asarray(trace["physics_step"])
    motor = np.asarray(trace["sensor_joint_motor_torque"], dtype=np.float64)
    reaction = np.asarray(
        trace["sensor_joint_reaction_force_torque"], dtype=np.float64
    )
    suction_force = np.asarray(
        trace["sensor_suction_force_xyz"], dtype=np.float64
    )
    suction_torque = np.asarray(
        trace["sensor_suction_torque_xyz"], dtype=np.float64
    )
    if steps.ndim != 1:
        raise ValueError("physics_step must be one-dimensional")
    if motor.shape[0] != steps.size or reaction.shape[0] != steps.size:
        raise ValueError("joint sensor length does not match physics_step")
    if reaction.ndim < 3 or reaction.shape[-1] != 6:
        raise ValueError("joint reaction wrench must have final dimension 6")
    for name, value in (
        ("suction force", suction_force),
        ("suction torque", suction_torque),
    ):
        if value.shape != (steps.size, 3):
            raise ValueError(f"{name} must have shape [N,3]")
    grasp = np.asarray(trace["sensor_grasp_active"])
    available = np.asarray(trace["sensor_constraint_available"])
    if grasp.shape != steps.shape or available.shape != steps.shape:
        raise ValueError("formal sensor state length does not match physics_step")

    reaction_force_per_joint = np.linalg.norm(reaction[..., :3], axis=-1)
    reaction_torque_per_joint = np.linalg.norm(reaction[..., 3:], axis=-1)
    continuous = {
        "constraint_force_norm": np.linalg.norm(suction_force, axis=-1),
        "constraint_torque_norm": np.linalg.norm(suction_torque, axis=-1),
        "motor_norm": np.linalg.norm(motor, axis=-1),
        "reaction_force_norm": np.linalg.norm(
            reaction_force_per_joint, axis=-1
        ),
        "reaction_torque_norm": np.linalg.norm(
            reaction_torque_per_joint, axis=-1
        ),
    }
    if not all(np.all(np.isfinite(value)) for value in continuous.values()):
        raise ValueError("formal sensor trace contains non-finite values")
    return steps, continuous, grasp, available


def stage_aligned_formal_sensor_feature(
    trace: Dict[str, np.ndarray],
    motion_events: Iterable[Dict[str, Any]],
    hz: float,
    trace_stride: int,
) -> Tuple[np.ndarray, List[str]]:
    """Build frozen 49-D formal feature v1 without privileged inputs."""
    hz = float(hz)
    trace_stride = int(trace_stride)
    if hz <= 0 or trace_stride <= 0:
        raise ValueError("hz and trace_stride must be positive")
    steps, continuous, grasp, available = _validated_signals(trace)
    events = list(motion_events)
    load_mask = motion_stage_mask(steps, events, LOAD_STAGES)
    unload_mask = motion_stage_mask(steps, events, UNLOAD_STAGES)
    dt = trace_stride / hz
    load_values, load_schema, load_means = _window_feature(
        "load", load_mask, continuous, grasp, available, dt
    )
    unload_values, unload_schema, unload_means = _window_feature(
        "unload", unload_mask, continuous, grasp, available, dt
    )
    delta_names = sorted(continuous)
    delta_values = [
        unload_means[name] - load_means[name] for name in delta_names
    ]
    delta_schema = [f"unload_minus_load.{name}.mean" for name in delta_names]
    feature = np.asarray(
        load_values + unload_values + delta_values, dtype=np.float64
    )
    schema = load_schema + unload_schema + delta_schema
    if feature.shape != (49,) or len(schema) != 49:
        raise RuntimeError("formal sensor v1 schema must contain 49 values")
    if not np.all(np.isfinite(feature)):
        raise ValueError("formal sensor v1 contains non-finite values")
    return feature, schema


def reaction_only_feature(
    trace: Dict[str, np.ndarray],
    motion_events: Iterable[Dict[str, Any]],
    hz: float,
    trace_stride: int,
) -> Tuple[np.ndarray, List[str]]:
    """Build a frozen 18-D reaction-wrench-only diagnostic ablation."""
    steps, continuous, _, _ = _validated_signals(trace)
    reactions = {
        key: value for key, value in continuous.items() if key.startswith("reaction_")
    }
    events = list(motion_events)
    load_mask = motion_stage_mask(steps, events, LOAD_STAGES)
    unload_mask = motion_stage_mask(steps, events, UNLOAD_STAGES)
    dt = int(trace_stride) / float(hz)
    dummy = np.ones(steps.shape, dtype=np.int64)
    load, load_schema, load_means = _window_feature(
        "load", load_mask, reactions, dummy, dummy, dt
    )
    unload, unload_schema, unload_means = _window_feature(
        "unload", unload_mask, reactions, dummy, dummy, dt
    )
    # Remove the four state fractions injected by _window_feature.
    load, load_schema = load[:-2], load_schema[:-2]
    unload, unload_schema = unload[:-2], unload_schema[:-2]
    names = sorted(reactions)
    delta = [unload_means[name] - load_means[name] for name in names]
    schema = load_schema + unload_schema + [
        f"unload_minus_load.{name}.mean" for name in names
    ]
    feature = np.asarray(load + unload + delta, dtype=np.float64)
    if feature.shape != (18,) or len(schema) != 18:
        raise RuntimeError("reaction-only schema must contain 18 values")
    if not np.all(np.isfinite(feature)):
        raise ValueError("reaction-only feature contains non-finite values")
    return feature, schema
