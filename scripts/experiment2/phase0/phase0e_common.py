"""Metrics and ranking helpers for Phase 0E recalibration."""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np

from scripts.experiment2.phase0.calibration_common import summarize_candidate
from scripts.experiment2.phase0.common import phase_slice


def _mean_frame_distance(first: np.ndarray, second: np.ndarray) -> float:
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 3:
        raise ValueError(f"expected matching [N,3] frames, got {a.shape}, {b.shape}")
    return float(np.mean(np.linalg.norm(a[:, :2] - b[:, :2], axis=-1)))


def contact_feature(
    trace: Dict[str, np.ndarray],
    hz: float,
    trace_stride: int,
) -> np.ndarray:
    preload = phase_slice(trace, "preload")
    force = np.asarray(preload["contact_force_norm"], dtype=np.float64)
    maximum = np.asarray(
        preload["contact_max_force_norm"], dtype=np.float64
    )
    active = np.asarray(
        preload["contact_active_beads"], dtype=np.float64
    )
    speed = np.asarray(preload["contact_mean_speed"], dtype=np.float64)
    if force.size == 0:
        raise ValueError("preload contact trace is empty")
    dt = float(trace_stride) / float(hz)
    return np.asarray(
        [
            np.sum(force) * dt,
            np.max(force),
            np.max(maximum),
            np.mean(active > 0),
            np.max(active),
            np.mean(speed),
        ],
        dtype=np.float64,
    )


def preload_return_residual(trace: Dict[str, np.ndarray]) -> float:
    no_action = phase_slice(trace, "no_action")
    preload = phase_slice(trace, "preload")
    if no_action["bead_positions"].shape[0] == 0:
        raise ValueError("no-action phase is empty")
    if preload["bead_positions"].shape[0] == 0:
        raise ValueError("preload phase is empty")
    return _mean_frame_distance(
        no_action["bead_positions"][-1],
        preload["bead_positions"][-1],
    )


def summarize_phase0e_candidate(
    candidate_id: str,
    candidate: Dict[str, Any],
    pair_rows: List[Dict[str, Any]],
    targets: Dict[str, float],
    vision_raw: Dict[str, Any],
    vision_delta: Dict[str, Any],
    contact_classifier: Dict[str, Any],
) -> Dict[str, Any]:
    base_targets = {
        key: targets[key]
        for key in (
            "max_initial_abs_xy",
            "max_arm_jump",
            "max_median_no_action_drift",
            "max_median_preload_visible_difference",
            "min_median_contact_impulse_gap",
            "min_median_main_branch_ade",
            "min_median_main_branch_fde",
            "min_median_branch_amplification",
        )
    }
    summary = summarize_candidate(
        candidate_id,
        candidate,
        pair_rows,
        base_targets,
    )

    vision_accuracy = max(
        float(vision_raw["accuracy"]),
        float(vision_delta["accuracy"]),
    )
    contact_accuracy = float(contact_classifier["accuracy"])
    fde_fraction = float(
        np.mean(
            [
                float(row["main_branch_fde"])
                >= float(targets["min_median_main_branch_fde"])
                for row in pair_rows
            ]
        )
    )
    amplification_fraction = float(
        np.mean(
            [
                float(row["branch_amplification"])
                >= float(targets["min_median_branch_amplification"])
                for row in pair_rows
            ]
        )
    )

    extra_checks = {
        "vision_screen": vision_accuracy
        <= float(targets["max_grouped_vision_accuracy"]),
        "contact_classifier": contact_accuracy
        >= float(targets["min_grouped_contact_accuracy"]),
        "fde_seed_fraction": fde_fraction
        >= float(targets["min_seed_pass_fraction"]),
        "amplification_seed_fraction": amplification_fraction
        >= float(targets["min_seed_pass_fraction"]),
    }
    summary["checks"].update(extra_checks)
    summary["eligible"] = bool(all(summary["checks"].values()))
    summary["classifiers"] = {
        "vision_raw": vision_raw,
        "vision_delta": vision_delta,
        "contact": contact_classifier,
        "maximum_vision_accuracy": vision_accuracy,
    }
    summary["seed_pass_fraction"] = {
        "main_fde": fde_fraction,
        "branch_amplification": amplification_fraction,
    }
    summary["median"]["free_preload_return_residual"] = float(
        np.median([row["free_preload_return_residual"] for row in pair_rows])
    )
    summary["median"]["hidden_preload_return_residual"] = float(
        np.median([row["hidden_preload_return_residual"] for row in pair_rows])
    )

    summary["score"] = float(
        summary["score"]
        + 2.0 * contact_accuracy
        + (1.0 - vision_accuracy)
        + fde_fraction
        + amplification_fraction
    )
    return summary


def rank_phase0e(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        list(rows),
        key=lambda row: (
            not bool(row["eligible"]),
            -float(row["score"]),
            float(row["classifiers"]["maximum_vision_accuracy"]),
            -float(row["median"]["main_branch_fde"]),
            str(row["candidate_id"]),
        ),
    )
