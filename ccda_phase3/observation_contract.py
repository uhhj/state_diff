"""General causal observation-history helpers for the formal schema."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .data_io import ROBOT_PROXY_DIM
from .schema_v2 import ACTION_DIM, DEFAULT_TH, N_BEADS, STATE_DIM


def _finite_array(value: np.ndarray, *, name: str, ndim: Optional[int] = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or Inf")
    return array


def history_indices(step: int, *, th: int = DEFAULT_TH, stride: int = 1) -> List[int]:
    if step < 0 or th <= 0 or stride <= 0:
        raise ValueError("step/th/stride are invalid")
    first = int(step) - int(stride) * (int(th) - 1)
    return [max(0, first + int(stride) * offset) for offset in range(int(th))]


def gather_history(series: np.ndarray, *, step: int, th: int = DEFAULT_TH, stride: int = 1, name: str = "series") -> np.ndarray:
    value = _finite_array(series, name=name)
    if value.ndim < 2 or int(step) >= value.shape[0]:
        raise ValueError(f"invalid {name} history request")
    return value[np.asarray(history_indices(step, th=th, stride=stride), dtype=np.int64)].copy()


def build_xy_history(bead_xy_series: np.ndarray, *, step: int, th: int = DEFAULT_TH, stride: int = 1) -> np.ndarray:
    history = gather_history(bead_xy_series, step=step, th=th, stride=stride, name="bead_xy_series")
    if history.ndim != 3 or history.shape[-1] != 2:
        raise ValueError(f"expected [th,N,2], got {history.shape}")
    return history.reshape(-1).astype(np.float32)


def build_position_proprio_state(bead_xy: np.ndarray, robot_proxy: np.ndarray) -> np.ndarray:
    xy = _finite_array(bead_xy, name="bead_xy", ndim=2)
    robot = _finite_array(robot_proxy, name="robot_proxy").reshape(-1)
    if xy.shape != (N_BEADS, 2) or robot.size != ROBOT_PROXY_DIM:
        raise ValueError("formal position/proprio dimensions are invalid")
    state = np.concatenate([xy.reshape(-1), robot]).astype(np.float32)
    if state.size != STATE_DIM:
        raise ValueError("formal state dimension mismatch")
    return state


def build_position_proprio_history(bead_xy_series: np.ndarray, robot_proxy_series: np.ndarray, *, step: int, th: int = DEFAULT_TH, stride: int = 1) -> np.ndarray:
    xy = gather_history(bead_xy_series, step=step, th=th, stride=stride, name="bead_xy_series")
    robot = gather_history(robot_proxy_series, step=step, th=th, stride=stride, name="robot_proxy_series")
    return np.stack([build_position_proprio_state(xy[index], robot[index]) for index in range(xy.shape[0])]).reshape(-1).astype(np.float32)


def schema_dimensions(*, n_beads: int = N_BEADS, th: int = DEFAULT_TH, action_dim: int = ACTION_DIM, robot_dim: int = ROBOT_PROXY_DIM) -> Dict[str, int]:
    if int(n_beads) != N_BEADS or int(robot_dim) != ROBOT_PROXY_DIM:
        raise ValueError("formal schema dimensions are fixed")
    state_dim = int(n_beads) * 2 + int(robot_dim)
    return {
        "state_dim": state_dim,
        "paper_x_dim": int(th) * state_dim,
        "state_action_x_dim": int(th) * state_dim + int(th) * int(action_dim),
    }
