"""Official outcome and summary for hidden routing-gate smoke."""
from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np

from scripts.experiment2.phase0.calibration_common import (
    summarize_candidate,
)
from scripts.experiment2.phase0.common import (
    phase_slice,
)


def _single_main_action(
    action_script: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    actions = [
        action
        for action in action_script
        if str(action.get("phase"))
        == "main_pull"
    ]
    if len(actions) != 1:
        raise ValueError(
            "expected exactly one main_pull"
        )
    action = actions[0]
    if action.get("primitive") != (
        "pick_precise_tension_extension"
    ):
        raise ValueError(
            "unexpected routing main primitive"
        )
    return action


def _last_positions(
    trace: Dict[str, np.ndarray],
    phase: str,
) -> np.ndarray:
    block = phase_slice(trace, phase)
    values = np.asarray(
        block["bead_positions"],
        dtype=np.float64,
    )
    if (
        values.ndim != 3
        or values.shape[-1] != 3
        or values.shape[0] == 0
    ):
        raise ValueError(
            f"invalid {phase} bead positions"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(
            f"non-finite {phase} positions"
        )
    return values[-1]


def _normalize(value, name):
    vector = np.asarray(
        value,
        dtype=np.float64,
    ).reshape(2)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError(
            f"{name} is near zero"
        )
    return vector / norm


def _endpoint_result(
    positions: np.ndarray,
    public: Dict[str, Any],
) -> Dict[str, Any]:
    endpoint_index = int(
        public["endpoint_index"]
    )
    endpoint = np.asarray(
        positions[endpoint_index, :2],
        dtype=np.float64,
    )
    plane_point = np.asarray(
        public["target_plane_point_xy"],
        dtype=np.float64,
    )
    normal = _normalize(
        public["normal_xy"],
        "normal_xy",
    )
    tangent = _normalize(
        public["tangent_xy"],
        "tangent_xy",
    )
    half_width = float(
        public[
            "target_corridor_half_width"
        ]
    )
    margin = float(
        np.dot(
            endpoint - plane_point,
            normal,
        )
    )
    lateral = float(
        np.dot(
            endpoint - plane_point,
            tangent,
        )
    )
    success = bool(
        margin >= 0.0
        and abs(lateral) <= half_width
    )
    return {
        "endpoint_index": endpoint_index,
        "endpoint_xy": (
            endpoint.astype(float).tolist()
        ),
        "target_normal_margin": margin,
        "target_tangent_offset": lateral,
        "target_corridor_half_width": (
            half_width
        ),
        "endpoint_target_success": (
            success
        ),
    }


def _leading_crossing_fraction(
    positions: np.ndarray,
    public: Dict[str, Any],
) -> float:
    indices = np.asarray(
        public["leading_segment_indices"],
        dtype=np.int64,
    )
    if indices.size == 0:
        raise ValueError(
            "leading segment is empty"
        )
    plane_point = np.asarray(
        public["target_plane_point_xy"],
        dtype=np.float64,
    )
    normal = _normalize(
        public["normal_xy"],
        "normal_xy",
    )
    tangent = _normalize(
        public["tangent_xy"],
        "tangent_xy",
    )
    half_width = float(
        public[
            "target_corridor_half_width"
        ]
    )
    delta = (
        np.asarray(
            positions[indices, :2],
            dtype=np.float64,
        )
        - plane_point.reshape(1, 2)
    )
    normal_margin = delta @ normal
    tangent_offset = delta @ tangent
    crossed = (
        (normal_margin >= 0.0)
        & (
            np.abs(tangent_offset)
            <= half_width
        )
    )
    return float(np.mean(crossed))


def hidden_routing_gate_outcome_metrics(
    free_trace: Dict[str, np.ndarray],
    hidden_trace: Dict[str, np.ndarray],
    action_script: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    action = _single_main_action(
        action_script
    )
    public = dict(
        action["public_task_layout"]
    )

    serialized = str(public).lower()
    for forbidden in (
        "barrier",
        "roof",
        "boxes",
    ):
        if forbidden in serialized:
            raise ValueError(
                "official metric received hidden "
                f"layout field {forbidden!r}"
            )

    free_final = _last_positions(
        free_trace,
        "post_main",
    )
    hidden_final = _last_positions(
        hidden_trace,
        "post_main",
    )
    if free_final.shape != hidden_final.shape:
        raise ValueError(
            "free/hidden bead shape mismatch"
        )

    free = _endpoint_result(
        free_final,
        public,
    )
    hidden = _endpoint_result(
        hidden_final,
        public,
    )
    free_leading = (
        _leading_crossing_fraction(
            free_final,
            public,
        )
    )
    hidden_leading = (
        _leading_crossing_fraction(
            hidden_final,
            public,
        )
    )
    return {
        "routing_metric_version": (
            "pulled_endpoint_target_success_v1"
        ),
        "official_outcome": (
            "pulled_endpoint_target_success_gap"
        ),
        "public_task_layout": public,
        "free_endpoint_target_success": (
            free["endpoint_target_success"]
        ),
        "hidden_endpoint_target_success": (
            hidden["endpoint_target_success"]
        ),
        "routing_success_gap": int(
            free["endpoint_target_success"]
        ) - int(
            hidden["endpoint_target_success"]
        ),
        "free_endpoint_target_margin": (
            free["target_normal_margin"]
        ),
        "hidden_endpoint_target_margin": (
            hidden["target_normal_margin"]
        ),
        "endpoint_target_margin_gap": float(
            free["target_normal_margin"]
            - hidden["target_normal_margin"]
        ),
        "free_endpoint_tangent_offset": (
            free["target_tangent_offset"]
        ),
        "hidden_endpoint_tangent_offset": (
            hidden["target_tangent_offset"]
        ),
        "free_leading_crossing_fraction": (
            free_leading
        ),
        "hidden_leading_crossing_fraction": (
            hidden_leading
        ),
        "leading_crossing_fraction_gap": float(
            free_leading - hidden_leading
        ),
    }


def summarize_hidden_routing_gate(
    candidate_id,
    candidate,
    pair_rows,
    targets,
    vision_raw,
    vision_delta,
    formal_sensor,
    oracle_contact,
):
    base_keys = (
        "max_initial_abs_xy",
        "max_arm_jump",
        "max_median_no_action_drift",
        "max_median_preload_visible_difference",
        "min_median_contact_impulse_gap",
        "min_median_main_branch_ade",
        "min_median_main_branch_fde",
        "min_median_branch_amplification",
    )
    summary = summarize_candidate(
        candidate_id,
        candidate,
        pair_rows,
        {
            key: targets[key]
            for key in base_keys
        },
    )

    vision_accuracy = max(
        float(vision_raw["accuracy"]),
        float(vision_delta["accuracy"]),
    )
    sensor_accuracy = float(
        formal_sensor["accuracy"]
    )
    oracle_accuracy = float(
        oracle_contact["accuracy"]
    )
    sensor_margin = (
        sensor_accuracy - vision_accuracy
    )

    routing_gap = float(np.median([
        float(row["routing_success_gap"])
        for row in pair_rows
    ]))
    margin_gap = float(np.median([
        float(
            row["endpoint_target_margin_gap"]
        )
        for row in pair_rows
    ]))
    leading_gap = float(np.median([
        float(
            row[
                "leading_crossing_fraction_gap"
            ]
        )
        for row in pair_rows
    ]))
    free_success_fraction = float(np.mean([
        bool(
            row[
                "free_endpoint_target_success"
            ]
        )
        for row in pair_rows
    ]))
    hidden_success_fraction = float(np.mean([
        bool(
            row[
                "hidden_endpoint_target_success"
            ]
        )
        for row in pair_rows
    ]))
    engagement = float(np.median([
        float(
            row[
                "hidden_gate_engagement_fraction"
            ]
        )
        for row in pair_rows
    ]))
    fde_fraction = float(np.mean([
        row["main_branch_fde"]
        >= targets[
            "min_median_main_branch_fde"
        ]
        for row in pair_rows
    ]))
    amplification_fraction = float(np.mean([
        row["branch_amplification"]
        >= targets[
            "min_median_branch_amplification"
        ]
        for row in pair_rows
    ]))

    summary["checks"].update({
        "vision_screen": (
            vision_accuracy
            <= targets[
                "max_grouped_vision_accuracy"
            ]
        ),
        "formal_sensor_classifier": (
            sensor_accuracy
            >= targets[
                "min_grouped_sensor_accuracy"
            ]
        ),
        "oracle_classifier": (
            oracle_accuracy
            >= targets[
                "min_grouped_oracle_accuracy"
            ]
        ),
        "sensor_over_vision_margin": (
            sensor_margin
            >= targets[
                "min_sensor_over_vision_margin"
            ]
        ),
        "routing_outcome": (
            routing_gap
            >= targets[
                "min_median_routing_success_gap"
            ]
        ),
        "gate_engagement": (
            engagement
            >= targets[
                "min_median_engagement_fraction"
            ]
        ),
        "precise_probe_execution": all(
            row["probe_motion_valid"]
            for row in pair_rows
        ),
        "routing_motion_execution": all(
            row["routing_motion_valid"]
            for row in pair_rows
        ),
        "fde_seed_fraction": (
            fde_fraction
            >= targets[
                "min_seed_pass_fraction"
            ]
        ),
        "amplification_seed_fraction": (
            amplification_fraction
            >= targets[
                "min_seed_pass_fraction"
            ]
        ),
    })
    summary["eligible"] = bool(
        all(summary["checks"].values())
    )
    summary["classifiers"] = {
        "vision_raw": vision_raw,
        "vision_delta": vision_delta,
        "formal_sensor": formal_sensor,
        "oracle_contact": oracle_contact,
        "maximum_vision_accuracy": (
            vision_accuracy
        ),
        "sensor_over_vision_margin": (
            sensor_margin
        ),
    }
    summary["median"].update({
        "routing_success_gap": routing_gap,
        "endpoint_target_margin_gap": (
            margin_gap
        ),
        "leading_crossing_fraction_gap": (
            leading_gap
        ),
        "hidden_gate_engagement_fraction": (
            engagement
        ),
    })
    summary["success_fraction"] = {
        "free_endpoint_target": (
            free_success_fraction
        ),
        "hidden_endpoint_target": (
            hidden_success_fraction
        ),
    }
    summary["seed_pass_fraction"] = {
        "main_fde": fde_fraction,
        "branch_amplification": (
            amplification_fraction
        ),
    }
    return summary
