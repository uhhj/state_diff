from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .data_io import (
    extract_bead_vel_xy,
    extract_bead_xy,
    extract_robot_pose_proxy,
    finite_velocity_or_difference,
    get_extras,
)


def state_from_live_info(
    info: Dict[str, Any],
    prev_xy: Optional[np.ndarray] = None,
    dt: float = 1.0,
) -> np.ndarray:
    """Build the live state using the same finite-value policy as offline data."""
    xy = extract_bead_xy(info)
    if xy.size == 0:
        raise ValueError("missing live bead_positions")

    vel = finite_velocity_or_difference(
        xy,
        extract_bead_vel_xy(info),
        prev_xy=prev_xy,
        dt=dt,
    )
    robot, _ = extract_robot_pose_proxy(info)
    if not np.all(np.isfinite(robot)):
        raise ValueError("non-finite live robot_pose_proxy")

    state = np.concatenate(
        [xy.reshape(-1), vel.reshape(-1), robot], axis=0
    ).astype(np.float32)
    if not np.all(np.isfinite(state)):
        raise ValueError("state_from_live_info produced non-finite state")
    return state


def pad_history(history: List[np.ndarray], th: int) -> np.ndarray:
    if th <= 0:
        raise ValueError(f"th must be positive, got {th}")
    if not history:
        raise ValueError("history must not be empty")

    arrays = [np.asarray(item, dtype=np.float32).reshape(-1) for item in history]
    dim = arrays[-1].size
    if any(item.size != dim for item in arrays):
        raise ValueError("history entries have inconsistent dimensions")

    selected = arrays[-th:]
    if len(selected) < th:
        selected = [selected[0].copy() for _ in range(th - len(selected))] + selected
    return np.stack(selected, axis=0).astype(np.float32)

def final_fraction_from_info(info: Dict[str, Any]) -> float:
    extras = get_extras(info)
    for key in ("task.final_fraction", "final_fraction", "task.fraction", "fraction"):
        if key in extras:
            try:
                return float(extras[key])
            except Exception:
                pass
    return float("nan")
