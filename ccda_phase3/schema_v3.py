"""Formal CCDA state-v3 schema for controlled UR5 proprioception.

State-v3 replaces the ambiguous 39-dimensional legacy robot proxy with the
six controlled revolute joints and the official suction tool-tip pose.  This
module is additive: the historical state-v2 schema remains immutable.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

SCHEMA_VERSION = "ccda_state_v3_controlled_revolute_ee_tip"
ROBOT_PROXY_SCHEMA_VERSION = "ccda_robot_proxy_v3_controlled_revolute_ee_tip"
ENVIRONMENT_VERSION = "ccda_hidden_slack_breakaway_v2"
FORMAL_TASK_NAME = "ccda-slack-cable-v2"
FORMAL_CONDITIONS = (
    "free",
    "hidden_slack_breakaway_pin_v2",
)

N_BEADS = 24
CABLE_DIM = N_BEADS * 2
CONTROLLED_JOINT_COUNT = 6
JOINT_POSITION_DIM = CONTROLLED_JOINT_COUNT
JOINT_VELOCITY_DIM = CONTROLLED_JOINT_COUNT
EE_POSITION_DIM = 3
EE_QUATERNION_DIM = 4
ROBOT_PROXY_DIM = (
    JOINT_POSITION_DIM
    + JOINT_VELOCITY_DIM
    + EE_POSITION_DIM
    + EE_QUATERNION_DIM
)
STATE_DIM = CABLE_DIM + ROBOT_PROXY_DIM
ACTION_DIM = 14
DEFAULT_TH = 3
DEFAULT_TF = 4
PAPER_X_DIM = DEFAULT_TH * STATE_DIM
STATE_ACTION_X_DIM = PAPER_X_DIM + DEFAULT_TH * ACTION_DIM

ROBOT_JOINT_POSITION_SLICE = slice(0, JOINT_POSITION_DIM)
ROBOT_JOINT_VELOCITY_SLICE = slice(
    ROBOT_JOINT_POSITION_SLICE.stop,
    ROBOT_JOINT_POSITION_SLICE.stop + JOINT_VELOCITY_DIM,
)
ROBOT_EE_POSITION_SLICE = slice(
    ROBOT_JOINT_VELOCITY_SLICE.stop,
    ROBOT_JOINT_VELOCITY_SLICE.stop + EE_POSITION_DIM,
)
ROBOT_EE_QUATERNION_SLICE = slice(
    ROBOT_EE_POSITION_SLICE.stop,
    ROBOT_EE_POSITION_SLICE.stop + EE_QUATERNION_DIM,
)

CABLE_SLICE = slice(0, CABLE_DIM)
JOINT_POSITION_SLICE = slice(CABLE_DIM, CABLE_DIM + JOINT_POSITION_DIM)
JOINT_VELOCITY_SLICE = slice(
    JOINT_POSITION_SLICE.stop,
    JOINT_POSITION_SLICE.stop + JOINT_VELOCITY_DIM,
)
EE_POSITION_SLICE = slice(
    JOINT_VELOCITY_SLICE.stop,
    JOINT_VELOCITY_SLICE.stop + EE_POSITION_DIM,
)
EE_QUATERNION_SLICE = slice(
    EE_POSITION_SLICE.stop,
    EE_POSITION_SLICE.stop + EE_QUATERNION_DIM,
)
QUATERNION_NORM_TOLERANCE = 1.0e-5
QUATERNION_CANONICAL_TOLERANCE = 1.0e-12


@dataclass(frozen=True)
class SchemaV3Manifest:
    schema_version: str = SCHEMA_VERSION
    robot_proxy_schema_version: str = ROBOT_PROXY_SCHEMA_VERSION
    environment_version: str = ENVIRONMENT_VERSION
    task_name: str = FORMAL_TASK_NAME
    n_beads: int = N_BEADS
    cable_dim: int = CABLE_DIM
    controlled_joint_count: int = CONTROLLED_JOINT_COUNT
    robot_proxy_dim: int = ROBOT_PROXY_DIM
    state_dim: int = STATE_DIM
    action_dim: int = ACTION_DIM
    th: int = DEFAULT_TH
    tf: int = DEFAULT_TF
    paper_x_dim: int = PAPER_X_DIM
    state_action_x_dim: int = STATE_ACTION_X_DIM
    missing_value_policy: str = "reject_record"
    quaternion_policy: str = "unit_norm_and_deterministic_hemisphere"
    model_presence_mask: bool = False
    contains_simulator_bead_velocity: bool = False
    copies_free_input_to_hidden_target: bool = False

    def validate(self) -> None:
        if self.n_beads != 24 or self.cable_dim != 48:
            raise ValueError("formal cable layout changed")
        if self.controlled_joint_count != 6:
            raise ValueError("formal UR5 contract requires six controlled joints")
        if self.robot_proxy_dim != 19 or self.state_dim != 67:
            raise ValueError("state-v3 dimensional contract changed")
        if self.action_dim != 14 or self.th != 3 or self.tf != 4:
            raise ValueError("formal action/history/future contract changed")
        if self.paper_x_dim != 201 or self.state_action_x_dim != 243:
            raise ValueError("state-v3 flattened input contract changed")
        if ROBOT_EE_QUATERNION_SLICE.stop != ROBOT_PROXY_DIM:
            raise ValueError("robot-local slice layout changed")
        if EE_QUATERNION_SLICE.stop != STATE_DIM:
            raise ValueError("state-v3 slice layout changed")
        if self.missing_value_policy != "reject_record":
            raise ValueError("state-v3 cannot encode missing robot state")
        if self.model_presence_mask:
            raise ValueError("presence masks are not model inputs")

    def to_dict(self) -> Dict[str, object]:
        self.validate()
        return asdict(self)


def _finite(
    value: np.ndarray,
    *,
    name: str,
    shape: Optional[Tuple[int, ...]] = None,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} shape mismatch: {array.shape} != {shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or Inf")
    return array


def validate_robot_proxy_v3(robot_proxy: np.ndarray) -> np.ndarray:
    robot = _finite(
        robot_proxy,
        name="robot_proxy_v3",
        shape=(ROBOT_PROXY_DIM,),
    )
    quaternion = robot[ROBOT_EE_QUATERNION_SLICE].astype(np.float64)
    norm = float(np.linalg.norm(quaternion))
    if abs(norm - 1.0) > QUATERNION_NORM_TOLERANCE:
        raise ValueError(
            "robot-proxy quaternion norm is outside the state-v3 contract: "
            f"{norm}"
        )
    # Stage A fixes the q == -q ambiguity by choosing the non-negative-w
    # representative, with the first non-zero xyz component as the tie-break.
    if quaternion[3] < -QUATERNION_CANONICAL_TOLERANCE:
        raise ValueError("robot-proxy quaternion is not in the canonical hemisphere")
    if abs(float(quaternion[3])) <= QUATERNION_CANONICAL_TOLERANCE:
        for component in quaternion[:3]:
            if abs(float(component)) > QUATERNION_CANONICAL_TOLERANCE:
                if component < 0.0:
                    raise ValueError(
                        "robot-proxy quaternion violates the canonical tie-break"
                    )
                break
    return robot.copy()


def state_v3_from_components(
    cable_xy: np.ndarray,
    robot_proxy: np.ndarray,
) -> np.ndarray:
    cable = _finite(
        cable_xy,
        name="cable_xy",
        shape=(N_BEADS, 2),
    )
    robot = validate_robot_proxy_v3(robot_proxy)
    state = np.concatenate([cable.reshape(-1), robot], axis=0).astype(np.float32)
    return _finite(state, name="state_v3", shape=(STATE_DIM,)).copy()


def left_padded_history(
    states: np.ndarray,
    *,
    end_index: int,
    th: int = DEFAULT_TH,
) -> np.ndarray:
    array = _finite(states, name="states")
    if array.ndim != 2 or array.shape[1] != STATE_DIM:
        raise ValueError(f"states must be [T,{STATE_DIM}], got {array.shape}")
    if not 0 <= int(end_index) < array.shape[0]:
        raise IndexError("end_index outside state sequence")
    indices = [
        max(0, int(end_index) - (int(th) - 1) + offset)
        for offset in range(int(th))
    ]
    return array[np.asarray(indices, dtype=np.int64)].copy()


def state_history_valid_mask(current_index: int, th: int = DEFAULT_TH) -> np.ndarray:
    values = [
        int(current_index) - (int(th) - 1) + offset >= 0
        for offset in range(int(th))
    ]
    return np.asarray(values, dtype=np.bool_)


def future_states(
    states: np.ndarray,
    *,
    current_index: int,
    tf: int = DEFAULT_TF,
) -> np.ndarray:
    array = _finite(states, name="states")
    if array.ndim != 2 or array.shape[1] != STATE_DIM:
        raise ValueError(f"states must be [T,{STATE_DIM}], got {array.shape}")
    start = int(current_index) + 1
    if start >= array.shape[0]:
        raise ValueError("no future state available")
    selected = array[start : start + int(tf)]
    if selected.shape[0] < int(tf):
        selected = np.concatenate(
            [
                selected,
                np.repeat(selected[-1:], int(tf) - selected.shape[0], axis=0),
            ],
            axis=0,
        )
    return _finite(
        selected,
        name="future_states",
        shape=(int(tf), STATE_DIM),
    ).copy()


def future_valid_mask(
    current_index: int,
    state_count: int,
    tf: int = DEFAULT_TF,
) -> np.ndarray:
    return np.asarray(
        [int(current_index) + 1 + offset < int(state_count) for offset in range(int(tf))],
        dtype=np.bool_,
    )


def past_action_history(
    action_vectors: np.ndarray,
    *,
    current_index: int,
    th: int = DEFAULT_TH,
) -> np.ndarray:
    actions = _finite(action_vectors, name="action_vectors")
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
        raise ValueError(f"actions must be [T,{ACTION_DIM}], got {actions.shape}")
    rows: List[np.ndarray] = []
    for index in range(int(current_index) - int(th), int(current_index)):
        rows.append(
            np.zeros(ACTION_DIM, dtype=np.float32)
            if index < 0
            else actions[index].copy()
        )
    return np.stack(rows, axis=0).astype(np.float32)


def action_history_valid_mask(current_index: int, th: int = DEFAULT_TH) -> np.ndarray:
    return np.asarray(
        [index >= 0 for index in range(int(current_index) - int(th), int(current_index))],
        dtype=np.bool_,
    )


def build_window(
    *,
    states: np.ndarray,
    action_vectors: np.ndarray,
    current_index: int,
    th: int = DEFAULT_TH,
    tf: int = DEFAULT_TF,
) -> Dict[str, np.ndarray]:
    if not 0 <= int(current_index) < int(action_vectors.shape[0]):
        raise IndexError("current_index outside action sequence")
    history = left_padded_history(states, end_index=current_index, th=th)
    action_history = past_action_history(
        action_vectors,
        current_index=current_index,
        th=th,
    )
    future = future_states(states, current_index=current_index, tf=tf)
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
        ).copy(),
        "state_action_x": _finite(
            state_action_x,
            name="state_action_x",
            shape=(int(th) * STATE_DIM + int(th) * ACTION_DIM,),
        ).copy(),
        "y_state": future,
        "y_final_state": future[-1].copy(),
        "y_action": target_action.copy(),
        "state_history_valid_mask": state_history_valid_mask(current_index, th),
        "future_valid_mask": future_valid_mask(current_index, states.shape[0], tf),
        "action_history_valid_mask": action_history_valid_mask(current_index, th),
    }
