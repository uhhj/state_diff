"""Formal Phase3 state-v2 observation schema.

Only deployable position/proprioception signals are included. Simulator bead
velocity and hidden-condition metadata are intentionally excluded.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

SCHEMA_VERSION = "ccda_state_v2_position_proprio"
ENVIRONMENT_VERSION = "ccda_hidden_slack_breakaway_v2"
FORMAL_TASK_NAME = "ccda-slack-cable-v2"
FORMAL_CONDITIONS = (
    "free",
    "hidden_slack_breakaway_pin_v2",
)
N_BEADS = 24
ROBOT_PROXY_DIM = 39
STATE_DIM = N_BEADS * 2 + ROBOT_PROXY_DIM
ACTION_DIM = 14
DEFAULT_TH = 3
DEFAULT_TF = 4
PAPER_X_DIM = DEFAULT_TH * STATE_DIM
STATE_ACTION_X_DIM = PAPER_X_DIM + DEFAULT_TH * ACTION_DIM


@dataclass(frozen=True)
class SchemaManifest:
    schema_version: str = SCHEMA_VERSION
    environment_version: str = ENVIRONMENT_VERSION
    task_name: str = FORMAL_TASK_NAME
    n_beads: int = N_BEADS
    robot_proxy_dim: int = ROBOT_PROXY_DIM
    state_dim: int = STATE_DIM
    action_dim: int = ACTION_DIM
    th: int = DEFAULT_TH
    tf: int = DEFAULT_TF
    paper_x_dim: int = PAPER_X_DIM
    state_action_x_dim: int = STATE_ACTION_X_DIM
    contains_simulator_bead_velocity: bool = False
    copies_free_input_to_hidden_target: bool = False


def _finite(
    value: np.ndarray,
    *,
    name: str,
    shape: Optional[Tuple[int, ...]] = None,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if shape is not None and array.shape != shape:
        raise ValueError(
            f"{name} shape mismatch: {array.shape} != {shape}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or Inf")
    return array


def state_v2_from_info(info: Any) -> Tuple[np.ndarray, str, int]:
    # Delayed import avoids a module cycle while data_io imports the formal
    # schema constants and canonical window builder from this module.
    from .data_io import extract_bead_xy, extract_robot_pose_proxy

    xy = extract_bead_xy(info)
    if xy.shape != (N_BEADS, 2):
        raise ValueError(
            f"expected {N_BEADS} ordered beads, got {xy.shape}"
        )
    robot, source = extract_robot_pose_proxy(info)
    robot = _finite(
        robot,
        name="robot_proxy",
        shape=(ROBOT_PROXY_DIM,),
    )
    state = np.concatenate(
        [xy.reshape(-1), robot],
        axis=0,
    ).astype(np.float32)
    state = _finite(
        state,
        name="state_v2",
        shape=(STATE_DIM,),
    )
    return state, source, N_BEADS


def left_padded_history(
    states: np.ndarray,
    *,
    end_index: int,
    th: int = DEFAULT_TH,
) -> np.ndarray:
    array = _finite(states, name="states")
    if array.ndim != 2 or array.shape[1] != STATE_DIM:
        raise ValueError(
            f"states must be [T,{STATE_DIM}], got {array.shape}"
        )
    if not 0 <= int(end_index) < array.shape[0]:
        raise IndexError("end_index outside state sequence")
    indices = [
        max(0, int(end_index) - (int(th) - 1) + offset)
        for offset in range(int(th))
    ]
    return array[np.asarray(indices, dtype=np.int64)].copy()


def future_states(
    states: np.ndarray,
    *,
    current_index: int,
    tf: int = DEFAULT_TF,
) -> np.ndarray:
    array = _finite(states, name="states")
    start = int(current_index) + 1
    if start >= array.shape[0]:
        raise ValueError("no future state available")
    selected = array[start:start + int(tf)]
    if selected.shape[0] < int(tf):
        selected = np.concatenate(
            [
                selected,
                np.repeat(
                    selected[-1:],
                    int(tf) - selected.shape[0],
                    axis=0,
                ),
            ],
            axis=0,
        )
    return _finite(
        selected,
        name="future_states",
        shape=(int(tf), STATE_DIM),
    )


def past_action_history(
    action_vectors: np.ndarray,
    *,
    current_index: int,
    th: int = DEFAULT_TH,
) -> np.ndarray:
    actions = _finite(action_vectors, name="action_vectors")
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
        raise ValueError(
            f"actions must be [T,{ACTION_DIM}], got {actions.shape}"
        )
    rows: List[np.ndarray] = []
    for index in range(
        int(current_index) - int(th),
        int(current_index),
    ):
        if index < 0:
            rows.append(np.zeros(ACTION_DIM, dtype=np.float32))
        else:
            rows.append(actions[index].copy())
    return np.stack(rows, axis=0).astype(np.float32)


def build_window(
    *,
    states: np.ndarray,
    action_vectors: np.ndarray,
    current_index: int,
    th: int = DEFAULT_TH,
    tf: int = DEFAULT_TF,
) -> Dict[str, np.ndarray]:
    history = left_padded_history(
        states,
        end_index=current_index,
        th=th,
    )
    action_history = past_action_history(
        action_vectors,
        current_index=current_index,
        th=th,
    )
    future = future_states(
        states,
        current_index=current_index,
        tf=tf,
    )
    target_action = _finite(
        action_vectors[int(current_index)],
        name="target_action",
        shape=(ACTION_DIM,),
    )
    paper_x = history.reshape(-1).astype(np.float32)
    state_action_x = np.concatenate(
        [paper_x, action_history.reshape(-1)],
        axis=0,
    ).astype(np.float32)
    return {
        "paper_x": _finite(
            paper_x,
            name="paper_x",
            shape=(int(th) * STATE_DIM,),
        ),
        "state_action_x": _finite(
            state_action_x,
            name="state_action_x",
            shape=(int(th) * STATE_DIM + int(th) * ACTION_DIM,),
        ),
        "y_state": future,
        "y_final_state": future[-1].copy(),
        "y_action": target_action.copy(),
    }
