"""Outcome localization diagnostics for fixed hidden-latch cable audits."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence
import numpy as np

from scripts.experiment2.phase0.common import phase_slice


def _last_positions(trace: Dict[str, np.ndarray], phase: str) -> np.ndarray:
    block = phase_slice(trace, phase)
    positions = np.asarray(block["bead_positions"], dtype=np.float64)
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError(
            f"expected {phase} bead positions [T,N,3], got {positions.shape}"
        )
    if positions.shape[0] == 0:
        raise ValueError(f"empty phase {phase}")
    if not np.all(np.isfinite(positions)):
        raise ValueError(f"non-finite bead positions in phase {phase}")
    return positions[-1]


def _single_main_action(
    action_script: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    actions = [
        action for action in action_script
        if str(action.get("phase")) == "main_pull"
    ]
    if len(actions) != 1:
        raise ValueError(f"expected one main_pull action, got {len(actions)}")
    return actions[0]


def _single_layout(
    action_script: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    layouts = [
        action["layout"]
        for action in action_script
        if isinstance(action.get("layout"), dict)
    ]
    if len(layouts) != 1:
        raise ValueError(f"expected one action layout, got {len(layouts)}")
    return dict(layouts[0])


def _normalized_xy(value: Iterable[float], name: str) -> np.ndarray:
    vector = np.asarray(list(value), dtype=np.float64).reshape(-1)
    if vector.size != 2 or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite XY vector")
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError(f"{name} is near zero")
    return vector / norm


def _stop_wall(layout: Dict[str, Any]) -> Dict[str, Any]:
    boxes = [
        box for box in layout.get("boxes", [])
        if str(box.get("name")) == "stop_wall"
    ]
    if len(boxes) != 1:
        raise ValueError(f"expected one stop_wall, got {len(boxes)}")
    return dict(boxes[0])


def blocked_indices_from_layout(
    pre_main_positions: np.ndarray,
    layout: Dict[str, Any],
) -> np.ndarray:
    """Beads covered by the stop wall along cable tangent."""
    positions = np.asarray(pre_main_positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"expected positions [N,3], got {positions.shape}")
    wall = _stop_wall(layout)
    tangent = _normalized_xy(layout["tangent_xy"], "tangent_xy")
    center = np.asarray(wall["center_xy"], dtype=np.float64).reshape(2)
    half = np.asarray(wall["half_extents"], dtype=np.float64).reshape(-1)
    if half.size != 3 or np.any(half <= 0):
        raise ValueError("invalid stop-wall half extents")
    bead_half_extent = float(
        layout.get("bead_collision_half_extent", 0.005)
    )
    if not np.isfinite(bead_half_extent) or bead_half_extent <= 0:
        raise ValueError("invalid bead half extent")
    along = (positions[:, :2] - center.reshape(1, 2)) @ tangent
    indices = np.flatnonzero(
        np.abs(along) <= float(half[0]) + bead_half_extent + 1e-12
    )
    if indices.size == 0:
        raise ValueError("stop wall covers no cable beads")
    # A folded cable can place remote ordered segments at the same tangent
    # coordinate as the wall. The physical stop covers the local ordered
    # segment around the latch anchor, not those projection-only outliers.
    probe_index = int(layout["probe_index"])
    if probe_index not in indices:
        raise ValueError("stop wall does not cover latch probe bead")
    selected = {probe_index}
    cursor = probe_index - 1
    candidate_set = set(int(index) for index in indices)
    while cursor in candidate_set:
        selected.add(cursor)
        cursor -= 1
    cursor = probe_index + 1
    while cursor in candidate_set:
        selected.add(cursor)
        cursor += 1
    return np.asarray(sorted(selected), dtype=np.int64)


def ordered_segment_indices(
    bead_count: int,
    blocked_indices: np.ndarray,
    endpoint_index: int,
) -> Dict[str, np.ndarray]:
    blocked = np.unique(
        np.asarray(blocked_indices, dtype=np.int64).reshape(-1)
    )
    if bead_count < 3 or blocked.size == 0:
        raise ValueError("invalid cable partition")
    if np.any(blocked < 0) or np.any(blocked >= bead_count):
        raise ValueError("blocked index outside cable")
    if endpoint_index not in (0, bead_count - 1):
        raise ValueError("endpoint_index is not an ordered endpoint")

    lower = int(np.min(blocked))
    upper = int(np.max(blocked))
    if endpoint_index == 0:
        pulled = np.arange(0, lower, dtype=np.int64)
        trailing = np.arange(upper + 1, bead_count, dtype=np.int64)
    else:
        pulled = np.arange(upper + 1, bead_count, dtype=np.int64)
        trailing = np.arange(0, lower, dtype=np.int64)
    if pulled.size == 0 or trailing.size == 0:
        raise ValueError("empty pulled or trailing segment")
    return {
        "all": np.arange(bead_count, dtype=np.int64),
        "blocked": blocked,
        "pulled_side": pulled,
        "trailing_side": trailing,
    }


def _segment_progress(
    free_progress: np.ndarray,
    hidden_progress: np.ndarray,
    indices: np.ndarray,
) -> Dict[str, float]:
    selected = np.asarray(indices, dtype=np.int64)
    free = np.asarray(free_progress, dtype=np.float64)[selected]
    hidden = np.asarray(hidden_progress, dtype=np.float64)[selected]
    if free.size == 0 or hidden.size == 0:
        raise ValueError("empty segment")
    return {
        "bead_count": int(selected.size),
        "free_mean_progress": float(np.mean(free)),
        "hidden_mean_progress": float(np.mean(hidden)),
        "progress_gap": float(np.mean(free) - np.mean(hidden)),
        "free_min_progress": float(np.min(free)),
        "hidden_min_progress": float(np.min(hidden)),
        "free_max_progress": float(np.max(free)),
        "hidden_max_progress": float(np.max(hidden)),
    }


def _rms(values: np.ndarray) -> float:
    value = np.asarray(values, dtype=np.float64)
    if value.size == 0:
        raise ValueError("empty RMS input")
    return float(np.sqrt(np.mean(np.square(value))))


def hidden_latch_outcome_decomposition(
    free_trace: Dict[str, np.ndarray],
    hidden_trace: Dict[str, np.ndarray],
    action_script: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Diagnostic decomposition; does not replace official progress gate."""
    main = _single_main_action(action_script)
    layout = _single_layout(action_script)

    pose0 = np.asarray(main["pose0"]["position"][:2], dtype=np.float64)
    pose1 = np.asarray(main["pose1"]["position"][:2], dtype=np.float64)
    vector = pose1 - pose0
    distance = float(np.linalg.norm(vector))
    if distance <= 1e-12:
        raise ValueError("zero main-pull distance")
    normal = vector / distance
    tangent = _normalized_xy(layout["tangent_xy"], "tangent_xy")

    free_before = _last_positions(free_trace, "preload")
    hidden_before = _last_positions(hidden_trace, "preload")
    free_after = _last_positions(free_trace, "post_main")
    hidden_after = _last_positions(hidden_trace, "post_main")
    shapes = {
        free_before.shape,
        hidden_before.shape,
        free_after.shape,
        hidden_after.shape,
    }
    if len(shapes) != 1:
        raise ValueError(f"bead shape mismatch: {shapes}")

    endpoint_index = int(layout["endpoint_index"])
    blocked = blocked_indices_from_layout(free_before, layout)
    ordered_partition_valid = True
    ordered_partition_error = None
    try:
        segments = ordered_segment_indices(
            free_before.shape[0], blocked, endpoint_index
        )
    except ValueError as error:
        # A sufficiently wide stop can span an ordered endpoint or every bead
        # in tangent projection. Preserve every non-empty segment diagnostic
        # without inventing statistics for an empty side. This state is not a
        # gate; direct ordered_segment_indices calls remain strict.
        ordered_partition_valid = False
        ordered_partition_error = str(error)
        all_indices = np.arange(free_before.shape[0], dtype=np.int64)
        segments = {"all": all_indices, "blocked": blocked}
        lower, upper = int(np.min(blocked)), int(np.max(blocked))
        if endpoint_index == 0:
            pulled = np.arange(0, lower, dtype=np.int64)
            trailing = np.arange(upper + 1, free_before.shape[0], dtype=np.int64)
        else:
            pulled = np.arange(upper + 1, free_before.shape[0], dtype=np.int64)
            trailing = np.arange(0, lower, dtype=np.int64)
        if pulled.size:
            segments["pulled_side"] = pulled
        if trailing.size:
            segments["trailing_side"] = trailing

    free_disp = free_after - free_before
    hidden_disp = hidden_after - hidden_before
    free_progress = free_disp[:, :2] @ normal
    hidden_progress = hidden_disp[:, :2] @ normal
    free_tangent = free_disp[:, :2] @ tangent
    hidden_tangent = hidden_disp[:, :2] @ tangent
    motion_difference = free_disp - hidden_disp
    final_difference = free_after - hidden_after

    segment_rows = {
        name: _segment_progress(free_progress, hidden_progress, indices)
        for name, indices in segments.items()
    }

    target_offset = 0.5 * distance
    free_crossed = free_progress >= target_offset
    hidden_crossed = hidden_progress >= target_offset

    membership: Dict[int, List[str]] = {}
    for name, indices in segments.items():
        for index in indices:
            membership.setdefault(int(index), []).append(name)

    per_bead = []
    for index in range(free_before.shape[0]):
        per_bead.append({
            "bead_index": int(index),
            "segments": sorted(membership.get(index, [])),
            "free_progress": float(free_progress[index]),
            "hidden_progress": float(hidden_progress[index]),
            "progress_gap": float(
                free_progress[index] - hidden_progress[index]
            ),
            "free_tangent_motion": float(free_tangent[index]),
            "hidden_tangent_motion": float(hidden_tangent[index]),
            "free_vertical_motion": float(free_disp[index, 2]),
            "hidden_vertical_motion": float(hidden_disp[index, 2]),
            "motion_difference_xyz": (
                motion_difference[index].astype(float).tolist()
            ),
            "final_shape_difference_xyz": (
                final_difference[index].astype(float).tolist()
            ),
            "free_crossed_halfway_plane": bool(free_crossed[index]),
            "hidden_crossed_halfway_plane": bool(hidden_crossed[index]),
        })

    return {
        "metric_version": "hidden_latch_outcome_decomposition_v1",
        "diagnostic_only": True,
        "changes_official_progress_gate": False,
        "ordered_partition_valid": ordered_partition_valid,
        "ordered_partition_error": ordered_partition_error,
        "endpoint_index": endpoint_index,
        "probe_index": int(layout["probe_index"]),
        "blocked_indices": blocked.astype(int).tolist(),
        "segment_indices": {
            name: indices.astype(int).tolist()
            for name, indices in segments.items()
        },
        "main_pull_distance": distance,
        "main_direction_xy": normal.astype(float).tolist(),
        "tangent_direction_xy": tangent.astype(float).tolist(),
        "halfway_target_offset": target_offset,
        "segments": segment_rows,
        "halfway_crossing": {
            "free_fraction": float(np.mean(free_crossed)),
            "hidden_fraction": float(np.mean(hidden_crossed)),
            "fraction_gap": float(
                np.mean(free_crossed) - np.mean(hidden_crossed)
            ),
        },
        "motion_difference_components": {
            "normal_rms": _rms(motion_difference[:, :2] @ normal),
            "tangent_rms": _rms(motion_difference[:, :2] @ tangent),
            "vertical_rms": _rms(motion_difference[:, 2]),
        },
        "final_shape_difference_components": {
            "normal_rms": _rms(final_difference[:, :2] @ normal),
            "tangent_rms": _rms(final_difference[:, :2] @ tangent),
            "vertical_rms": _rms(final_difference[:, 2]),
        },
        "per_bead": per_bead,
    }
