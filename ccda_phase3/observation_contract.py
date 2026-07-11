from __future__ import annotations

import hashlib
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .data_io import ROBOT_PROXY_DIM


SCHEMA_PRIVILEGED_V1 = "ccda_state_v1_xy_sim_velocity_robot"
SCHEMA_POSITION_PROPRIO_V2 = "ccda_state_v2_position_proprio"

DEFAULT_TH = 3
DEFAULT_ACTION_DIM = 14


def _finite_array(
    value: np.ndarray,
    *,
    name: str,
    ndim: Optional[int] = None,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if ndim is not None and array.ndim != int(ndim):
        raise ValueError(
            f"{name} must have ndim={ndim}, got shape={array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or Inf")
    return array


def history_indices(
    step: int,
    *,
    th: int = DEFAULT_TH,
    stride: int = 1,
) -> List[int]:
    """Return causal left-padded history indices ending at ``step``."""
    step = int(step)
    th = int(th)
    stride = int(stride)
    if step < 0:
        raise ValueError("step must be non-negative")
    if th <= 0:
        raise ValueError("th must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")
    first = step - stride * (th - 1)
    return [max(0, first + stride * offset) for offset in range(th)]


def split_privileged_state(
    state: np.ndarray,
    *,
    n_beads: int,
    robot_dim: int = ROBOT_PROXY_DIM,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split state-v1 into bead XY, simulator velocity, and robot proxy."""
    vector = _finite_array(state, name="state").reshape(-1)
    n_beads = int(n_beads)
    robot_dim = int(robot_dim)
    xy_dim = n_beads * 2
    expected = xy_dim * 2 + robot_dim
    if vector.size != expected:
        raise ValueError(
            f"state-v1 dimension mismatch: {vector.size} != {expected}"
        )
    xy = vector[:xy_dim].reshape(n_beads, 2)
    velocity = vector[xy_dim : xy_dim * 2].reshape(n_beads, 2)
    robot = vector[xy_dim * 2 :]
    return xy.copy(), velocity.copy(), robot.copy()


def build_position_proprio_state(
    bead_xy: np.ndarray,
    robot_proxy: np.ndarray,
) -> np.ndarray:
    """Build state-v2 without simulator bead velocity."""
    xy = _finite_array(bead_xy, name="bead_xy", ndim=2)
    if xy.shape[1] != 2:
        raise ValueError(f"bead_xy must be [N,2], got {xy.shape}")
    robot = _finite_array(robot_proxy, name="robot_proxy").reshape(-1)
    if robot.size != ROBOT_PROXY_DIM:
        raise ValueError(
            f"robot proxy dimension mismatch: {robot.size} != {ROBOT_PROXY_DIM}"
        )
    return np.concatenate([xy.reshape(-1), robot], axis=0).astype(np.float32)


def gather_history(
    series: np.ndarray,
    *,
    step: int,
    th: int = DEFAULT_TH,
    stride: int = 1,
    name: str = "series",
) -> np.ndarray:
    """Gather a causal, left-padded history from a [T,...] array."""
    value = _finite_array(series, name=name)
    if value.ndim < 2:
        raise ValueError(f"{name} must be [T,...], got {value.shape}")
    if int(step) >= value.shape[0]:
        raise ValueError(
            f"step {step} is outside {name} length {value.shape[0]}"
        )
    indices = history_indices(step, th=th, stride=stride)
    return value[np.asarray(indices, dtype=np.int64)].copy()


def build_xy_history(
    bead_xy_series: np.ndarray,
    *,
    step: int,
    th: int = DEFAULT_TH,
    stride: int = 1,
) -> np.ndarray:
    history = gather_history(
        bead_xy_series,
        step=step,
        th=th,
        stride=stride,
        name="bead_xy_series",
    )
    if history.ndim != 3 or history.shape[-1] != 2:
        raise ValueError(f"expected [th,N,2], got {history.shape}")
    return history.reshape(-1).astype(np.float32)


def build_position_proprio_history(
    bead_xy_series: np.ndarray,
    robot_proxy_series: np.ndarray,
    *,
    step: int,
    th: int = DEFAULT_TH,
    stride: int = 1,
) -> np.ndarray:
    xy_history = gather_history(
        bead_xy_series,
        step=step,
        th=th,
        stride=stride,
        name="bead_xy_series",
    )
    robot_history = gather_history(
        robot_proxy_series,
        step=step,
        th=th,
        stride=stride,
        name="robot_proxy_series",
    )
    if xy_history.shape[0] != robot_history.shape[0]:
        raise ValueError("history length mismatch")
    states = [
        build_position_proprio_state(xy_history[index], robot_history[index])
        for index in range(xy_history.shape[0])
    ]
    return np.stack(states, axis=0).reshape(-1).astype(np.float32)


def build_causal_fd_history(
    bead_xy_series: np.ndarray,
    robot_proxy_series: np.ndarray,
    *,
    step: int,
    th: int = DEFAULT_TH,
    stride: int = 1,
    physics_dt: float = 1.0,
) -> np.ndarray:
    """Build observable history with finite-difference motion, never sim velocity."""
    xy = _finite_array(
        bead_xy_series,
        name="bead_xy_series",
        ndim=3,
    )
    robot = _finite_array(
        robot_proxy_series,
        name="robot_proxy_series",
        ndim=2,
    )
    indices = history_indices(step, th=th, stride=stride)
    safe_dt = max(float(physics_dt) * int(stride), 1e-9)
    rows: List[np.ndarray] = []
    for index in indices:
        previous = max(0, index - int(stride))
        finite_difference = (xy[index] - xy[previous]) / safe_dt
        row = np.concatenate(
            [
                xy[index].reshape(-1),
                finite_difference.reshape(-1),
                robot[index].reshape(-1),
            ],
            axis=0,
        )
        rows.append(row.astype(np.float32))
    result = np.stack(rows, axis=0).reshape(-1).astype(np.float32)
    if not np.all(np.isfinite(result)):
        raise ValueError("causal finite-difference history is non-finite")
    return result


def build_model_x_v2(
    bead_xy_series: np.ndarray,
    robot_proxy_series: np.ndarray,
    *,
    step: int,
    th: int = DEFAULT_TH,
    stride: int = 1,
    action_dim: int = DEFAULT_ACTION_DIM,
) -> np.ndarray:
    state_history = build_position_proprio_history(
        bead_xy_series,
        robot_proxy_series,
        step=step,
        th=th,
        stride=stride,
    )
    action_history = np.zeros(
        int(th) * int(action_dim),
        dtype=np.float32,
    )
    return np.concatenate([state_history, action_history], axis=0)


def build_privileged_model_x(
    privileged_state_series: np.ndarray,
    *,
    step: int,
    th: int = DEFAULT_TH,
    stride: int = 1,
    action_dim: int = DEFAULT_ACTION_DIM,
) -> np.ndarray:
    history = gather_history(
        privileged_state_series,
        step=step,
        th=th,
        stride=stride,
        name="privileged_state_series",
    ).reshape(-1)
    return np.concatenate(
        [
            history.astype(np.float32),
            np.zeros(int(th) * int(action_dim), dtype=np.float32),
        ],
        axis=0,
    )


def deterministic_noise(
    shape: Sequence[int],
    *,
    std_m: float,
    key: str,
) -> np.ndarray:
    """Generate stable independent Gaussian noise without Python hash()."""
    std_m = float(std_m)
    if std_m < 0:
        raise ValueError("std_m must be non-negative")
    digest = hashlib.sha256(str(key).encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, std_m, size=tuple(shape)).astype(np.float32)


def add_deterministic_xy_noise(
    bead_xy_series: np.ndarray,
    *,
    std_m: float,
    key: str,
) -> np.ndarray:
    xy = _finite_array(
        bead_xy_series,
        name="bead_xy_series",
        ndim=3,
    )
    noisy = xy + deterministic_noise(xy.shape, std_m=std_m, key=key)
    if not np.all(np.isfinite(noisy)):
        raise ValueError("noisy XY is non-finite")
    return noisy.astype(np.float32)


def schema_dimensions(
    *,
    n_beads: int,
    th: int = DEFAULT_TH,
    action_dim: int = DEFAULT_ACTION_DIM,
    robot_dim: int = ROBOT_PROXY_DIM,
) -> Dict[str, int]:
    n_beads = int(n_beads)
    th = int(th)
    action_dim = int(action_dim)
    robot_dim = int(robot_dim)
    privileged_state_dim = n_beads * 4 + robot_dim
    position_proprio_state_dim = n_beads * 2 + robot_dim
    return {
        "privileged_state_dim": privileged_state_dim,
        "position_proprio_state_dim": position_proprio_state_dim,
        "privileged_paper_x_dim": th * privileged_state_dim,
        "position_proprio_paper_x_dim": th * position_proprio_state_dim,
        "privileged_state_action_x_dim": (
            th * privileged_state_dim + th * action_dim
        ),
        "position_proprio_state_action_x_dim": (
            th * position_proprio_state_dim + th * action_dim
        ),
    }
