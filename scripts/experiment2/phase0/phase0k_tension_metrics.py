"""Diagnostic stage metrics for Phase 0K tension extension."""
from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np

from scripts.experiment2.phase0.common import phase_slice


TENSION_STAGES = (
    "tension_pull_lift",
    "tension_pull_stage1",
    "tension_pull_stage2",
    "tension_pull_lower_release",
)


def _single_main_action(
    action_script: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    actions = [
        action for action in action_script
        if str(action.get("phase")) == "main_pull"
    ]
    if len(actions) != 1:
        raise ValueError("expected exactly one main_pull action")
    action = actions[0]
    if action.get("primitive") != "pick_precise_tension_extension":
        raise ValueError("unexpected Phase 0K main primitive")
    return action


def _event(metadata: Dict[str, Any], stage: str) -> Dict[str, Any]:
    rows = [
        row for row in metadata.get("motion_events", [])
        if str(row.get("stage", row.get("label", ""))) == stage
    ]
    if len(rows) != 1:
        raise ValueError(f"expected one event for {stage}, got {len(rows)}")
    return dict(rows[0])


def _last_phase_positions(
    trace: Dict[str, np.ndarray], phase: str
) -> np.ndarray:
    block = phase_slice(trace, phase)
    positions = np.asarray(block["bead_positions"], dtype=np.float64)
    if positions.ndim != 3:
        raise ValueError(f"invalid {phase} bead positions")
    if positions.shape[0] == 0:
        raise ValueError(f"empty phase {phase}")
    return positions[-1]


def _positions_at_or_before(
    trace: Dict[str, np.ndarray], physics_step: int
) -> np.ndarray:
    steps = np.asarray(trace["physics_step"], dtype=np.int64)
    positions = np.asarray(trace["bead_positions"], dtype=np.float64)
    if steps.ndim != 1:
        raise ValueError("physics_step must be one-dimensional")
    if positions.shape[0] != steps.size:
        raise ValueError("trace position/step length mismatch")
    indices = np.flatnonzero(steps <= int(physics_step))
    if indices.size == 0:
        raise ValueError("no trace sample before event")
    return positions[int(indices[-1])]


def _direction(action: Dict[str, Any]) -> np.ndarray:
    start = np.asarray(action["pose0"]["position"][:2], dtype=np.float64)
    final = np.asarray(action["pose1"]["position"][:2], dtype=np.float64)
    vector = final - start
    distance = float(np.linalg.norm(vector))
    if distance <= 1e-12:
        raise ValueError("zero final pull distance")
    return vector / distance


def _progress(
    before: np.ndarray, after: np.ndarray, direction: np.ndarray
) -> np.ndarray:
    before = np.asarray(before, dtype=np.float64)
    after = np.asarray(after, dtype=np.float64)
    if before.shape != after.shape:
        raise ValueError("before/after position shape mismatch")
    return (after[:, :2] - before[:, :2]) @ direction


def _stage_row(
    free_before: np.ndarray,
    hidden_before: np.ndarray,
    free_after: np.ndarray,
    hidden_after: np.ndarray,
    direction: np.ndarray,
) -> Dict[str, float]:
    free_progress = _progress(free_before, free_after, direction)
    hidden_progress = _progress(hidden_before, hidden_after, direction)
    difference = np.asarray(free_after) - np.asarray(hidden_after)
    distance = np.linalg.norm(difference, axis=1)
    return {
        "free_mean_progress": float(np.mean(free_progress)),
        "hidden_mean_progress": float(np.mean(hidden_progress)),
        "mean_progress_gap": float(
            np.mean(free_progress) - np.mean(hidden_progress)
        ),
        "mean_bead_branch_distance": float(np.mean(distance)),
        "maximum_bead_branch_distance": float(np.max(distance)),
    }


def tension_motion_valid(
    metadata: Dict[str, Any], minimum_fraction: float
) -> bool:
    minimum_fraction = float(minimum_fraction)
    events = {stage: _event(metadata, stage) for stage in TENSION_STAGES}
    for stage, event in events.items():
        if not bool(event.get("success")):
            return False
        if float(event.get("achieved_fraction", 0.0)) < minimum_fraction:
            return False
        if stage in {
            "tension_pull_lift",
            "tension_pull_stage1",
            "tension_pull_stage2",
        }:
            if not bool(event.get("grasp_active_after", False)):
                return False
    return bool(
        events["tension_pull_lower_release"].get(
            "grasp_active_before_release", False
        )
    )


def tension_extension_metrics(
    free_trace: Dict[str, np.ndarray],
    hidden_trace: Dict[str, np.ndarray],
    free_metadata: Dict[str, Any],
    hidden_metadata: Dict[str, Any],
    action_script: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    action = _single_main_action(action_script)
    direction = _direction(action)
    free_before = _last_phase_positions(free_trace, "preload")
    hidden_before = _last_phase_positions(hidden_trace, "preload")

    free_stage1_event = _event(free_metadata, "tension_pull_stage1")
    hidden_stage1_event = _event(hidden_metadata, "tension_pull_stage1")
    free_stage2_event = _event(free_metadata, "tension_pull_stage2")
    hidden_stage2_event = _event(hidden_metadata, "tension_pull_stage2")
    free_stage1 = _positions_at_or_before(
        free_trace, free_stage1_event["physics_step_end"]
    )
    hidden_stage1 = _positions_at_or_before(
        hidden_trace, hidden_stage1_event["physics_step_end"]
    )
    free_stage2 = _positions_at_or_before(
        free_trace, free_stage2_event["physics_step_end"]
    )
    hidden_stage2 = _positions_at_or_before(
        hidden_trace, hidden_stage2_event["physics_step_end"]
    )
    free_final = _last_phase_positions(free_trace, "post_main")
    hidden_final = _last_phase_positions(hidden_trace, "post_main")

    stage1 = _stage_row(
        free_before, hidden_before, free_stage1, hidden_stage1, direction
    )
    stage2 = _stage_row(
        free_before, hidden_before, free_stage2, hidden_stage2, direction
    )
    final = _stage_row(
        free_before, hidden_before, free_final, hidden_final, direction
    )
    return {
        "metric_version": "same_end_tension_extension_metrics_v1",
        "diagnostic_only": True,
        "changes_official_progress_gate": False,
        "main_direction_xy": direction.astype(float).tolist(),
        "stage1": stage1,
        "stage2": stage2,
        "final": final,
        "stage2_minus_stage1_progress_gap": float(
            stage2["mean_progress_gap"] - stage1["mean_progress_gap"]
        ),
        "final_minus_stage1_progress_gap": float(
            final["mean_progress_gap"] - stage1["mean_progress_gap"]
        ),
    }
