"""Metrics and ranking helpers for Phase 0F."""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np

from scripts.experiment2.phase0.calibration_common import summarize_candidate
from scripts.experiment2.phase0.common import phase_slice


def _integral(values: np.ndarray, dt: float) -> float:
    return float(np.sum(np.asarray(values, dtype=np.float64)) * dt)


def oracle_contact_feature(trace, hz, trace_stride):
    preload = phase_slice(trace, "preload")
    force = np.asarray(preload["contact_force_norm"], dtype=np.float64)
    maximum = np.asarray(
        preload["contact_max_force_norm"], dtype=np.float64
    )
    active = np.asarray(preload["contact_active_beads"], dtype=np.float64)
    speed = np.asarray(preload["contact_mean_speed"], dtype=np.float64)
    if force.size == 0:
        raise ValueError("preload Oracle contact is empty")
    dt = float(trace_stride) / float(hz)
    return np.asarray([
        _integral(force, dt),
        float(np.max(force)),
        float(np.max(maximum)),
        float(np.mean(active > 0)),
        float(np.max(active)),
        float(np.mean(speed)),
    ], dtype=np.float64)


def sensor_contact_feature(trace, hz, trace_stride):
    preload = phase_slice(trace, "preload")
    joint = np.asarray(
        preload["sensor_joint_motor_torque"], dtype=np.float64
    )
    joint_norm = np.asarray(
        preload["sensor_joint_motor_torque_norm"], dtype=np.float64
    )
    suction_force = np.asarray(
        preload["sensor_suction_force_norm"], dtype=np.float64
    )
    suction_torque = np.asarray(
        preload["sensor_suction_torque_norm"], dtype=np.float64
    )
    grasp = np.asarray(preload["sensor_grasp_active"], dtype=np.float64)
    available = np.asarray(
        preload["sensor_constraint_available"], dtype=np.float64
    )
    if joint_norm.size == 0:
        raise ValueError("preload sensor trace is empty")
    dt = float(trace_stride) / float(hz)
    joint_abs = np.abs(joint)
    return np.concatenate([
        np.asarray([
            _integral(joint_norm, dt),
            float(np.max(joint_norm)),
            float(np.mean(joint_norm)),
            _integral(suction_force, dt),
            float(np.max(suction_force)),
            _integral(suction_torque, dt),
            float(np.max(suction_torque)),
            float(np.mean(grasp > 0)),
            float(np.mean(available > 0)),
        ], dtype=np.float64),
        np.mean(joint_abs, axis=0),
        np.max(joint_abs, axis=0),
    ])


def summarize_phase0f_candidate(
    candidate_id: str,
    candidate: Dict[str, Any],
    pair_rows: List[Dict[str, Any]],
    targets: Dict[str, float],
    vision_raw: Dict[str, Any],
    vision_delta: Dict[str, Any],
    sensor_classifier: Dict[str, Any],
    oracle_classifier: Dict[str, Any],
) -> Dict[str, Any]:
    base_keys = (
        "max_initial_abs_xy", "max_arm_jump",
        "max_median_no_action_drift",
        "max_median_preload_visible_difference",
        "min_median_contact_impulse_gap",
        "min_median_main_branch_ade",
        "min_median_main_branch_fde",
        "min_median_branch_amplification",
    )
    summary = summarize_candidate(
        candidate_id, candidate, pair_rows,
        {key: targets[key] for key in base_keys},
    )
    vision_accuracy = max(
        float(vision_raw["accuracy"]),
        float(vision_delta["accuracy"]),
    )
    sensor_accuracy = float(sensor_classifier["accuracy"])
    oracle_accuracy = float(oracle_classifier["accuracy"])
    fde_fraction = float(np.mean([
        float(row["main_branch_fde"])
        >= targets["min_median_main_branch_fde"]
        for row in pair_rows
    ]))
    amplification_fraction = float(np.mean([
        float(row["branch_amplification"])
        >= targets["min_median_branch_amplification"]
        for row in pair_rows
    ]))
    sensor_margin = sensor_accuracy - vision_accuracy
    summary["checks"].update({
        "vision_screen": (
            vision_accuracy <= targets["max_grouped_vision_accuracy"]
        ),
        "formal_sensor_classifier": (
            sensor_accuracy >= targets["min_grouped_sensor_accuracy"]
        ),
        "oracle_classifier": (
            oracle_accuracy >= targets["min_grouped_oracle_accuracy"]
        ),
        "sensor_over_vision_margin": (
            sensor_margin >= targets["min_sensor_over_vision_margin"]
        ),
        "fde_seed_fraction": (
            fde_fraction >= targets["min_seed_pass_fraction"]
        ),
        "amplification_seed_fraction": (
            amplification_fraction >= targets["min_seed_pass_fraction"]
        ),
    })
    summary["eligible"] = bool(all(summary["checks"].values()))
    summary["classifiers"] = {
        "vision_raw": vision_raw,
        "vision_delta": vision_delta,
        "formal_sensor": sensor_classifier,
        "oracle_contact": oracle_classifier,
        "maximum_vision_accuracy": vision_accuracy,
        "sensor_over_vision_margin": sensor_margin,
    }
    summary["seed_pass_fraction"] = {
        "main_fde": fde_fraction,
        "branch_amplification": amplification_fraction,
    }
    summary["score"] = float(
        summary["score"] + 2.0 * sensor_accuracy + oracle_accuracy
        + max(0.0, sensor_margin) + (1.0 - vision_accuracy)
        + fde_fraction + amplification_fraction
    )
    return summary


def rank_phase0f(rows: Sequence[Dict[str, Any]]):
    return sorted(list(rows), key=lambda row: (
        not bool(row["eligible"]),
        -float(row["score"]),
        float(row["classifiers"]["maximum_vision_accuracy"]),
        -float(row["classifiers"]["formal_sensor"]["accuracy"]),
        -float(row["median"]["main_branch_fde"]),
        str(row["candidate_id"]),
    ))
