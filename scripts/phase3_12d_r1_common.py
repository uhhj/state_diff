#!/usr/bin/env python3
from __future__ import annotations

import copy
import contextlib
import hashlib
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np


SELECTORS = (
    "ddpm_mean",
    "ddpm_raw",
    "condition_nearest_upper",
    "proxy_state_motion_nn",
    "proxy_combined_topk_action_geom",
    "random_train_future",
    "retrieved_source_action",
    "goal_geometry_oracle",
)

DEFAULT_CONDITIONS = (
    "free",
    "hidden_breakaway_pin",
    "hidden_high_friction",
)

DEFAULT_VISIBLE_SEEDS = (
    312000,
    312001,
    312002,
    312003,
    312500,
    312501,
    312502,
    312503,
)

TASK_STATE_FIELDS = (
    "total_rewards",
    "t",
    "task_stage",
    "exit_gracefully",
    "_ccda_step_count",
    "_ccda_physics_step_count",
    "_breakaway_released",
    "_breakaway_release_step",
    "_breakaway_release_physics_step",
    "_breakaway_anchor_pos",
    "_breakaway_bead_id",
    "_breakaway_constraint_id",
    "_breakaway_max_disp_seen",
    "hidden_constraint_ids",
    "hidden_body_ids",
    "hidden_contact_meta",
)

EE_STATE_FIELDS = (
    "activated",
    "contact_constraint",
    "def_grip_item",
    "def_grip_anchors",
    "def_min_vetex",
    "def_min_distance",
    "init_grip_distance",
    "init_grip_item",
)

FORBIDDEN_MODULE_PREFIXES = (
    "tensorflow",
    "ravens.agents",
    "ravens.models",
    "ravens.datasets",
)


def add_repo_paths(root: Path) -> None:
    for path in (
        root,
        root / "scripts",
        root / "external" / "deformable-ravens",
    ):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def assert_no_forbidden_modules(stage: str) -> None:
    bad: List[str] = []
    for name in sys.modules:
        for prefix in FORBIDDEN_MODULE_PREFIXES:
            if name == prefix or name.startswith(prefix + "."):
                bad.append(name)
    if bad:
        raise RuntimeError(
            "[Phase3.12d-r1] forbidden modules loaded at {}: {}".format(
                stage, sorted(set(bad))[:40]
            )
        )


def scalar_from_npz(data: Any, key: str) -> Any:
    value = data[key]
    if getattr(value, "shape", ()) == ():
        return value.item()
    flat = value.reshape(-1)
    if len(flat) != 1:
        raise ValueError("{} is not scalar-like: {}".format(key, value.shape))
    item = flat[0]
    return item.item() if hasattr(item, "item") else item


def seed_everything(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    except Exception:
        pass


def set_ccda_environment(condition: str, visible_seed: int) -> str:
    pair_group = "phase3_12d_r1_seed_{}".format(int(visible_seed))
    os.environ["CCDA_HIDDEN_CONDITION"] = str(condition)
    os.environ["CCDA_VISIBLE_SEED"] = str(int(visible_seed))
    os.environ["CCDA_PAIR_GROUP"] = pair_group
    os.environ.setdefault("CCDA_SETTLE_SECONDS", "0")
    return pair_group


def finite_mean(values: Iterable[float]) -> float:
    items = [float(x) for x in values if math.isfinite(float(x))]
    return float(np.mean(items)) if items else float("nan")


def finite_median(values: Iterable[float]) -> float:
    items = [float(x) for x in values if math.isfinite(float(x))]
    return float(np.median(items)) if items else float("nan")


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(array.tobytes()).hexdigest()


def json_safe_copy(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_safe_copy(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe_copy(v) for v in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return copy.deepcopy(value)


def capture_rng_state() -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }
    try:
        import torch

        payload["torch_cpu"] = torch.get_rng_state().clone()
        if torch.cuda.is_available():
            payload["torch_cuda"] = [
                state.clone() for state in torch.cuda.get_rng_state_all()
            ]
    except Exception:
        pass
    return payload


def restore_rng_state(payload: Mapping[str, Any]) -> None:
    random.setstate(payload["python"])
    np.random.set_state(payload["numpy"])
    try:
        import torch

        if "torch_cpu" in payload:
            torch.set_rng_state(payload["torch_cpu"])
        if "torch_cuda" in payload and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(payload["torch_cuda"])
    except Exception:
        pass


def capture_object_fields(obj: Any, fields: Sequence[str]) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for name in fields:
        if hasattr(obj, name):
            output[name] = copy.deepcopy(getattr(obj, name))
    return output


def restore_object_fields(obj: Any, payload: Mapping[str, Any]) -> None:
    for name, value in payload.items():
        setattr(obj, name, copy.deepcopy(value))


def pause_and_drain(env: Any, seconds: float = 0.01) -> None:
    env.pause()
    if seconds > 0:
        time.sleep(float(seconds))


def simulation_lock(env: Any):
    lock = getattr(env, "_ccda_step_lock", None)
    return lock if lock is not None else contextlib.nullcontext()


def _constraint_exists(constraint_id: Optional[int]) -> bool:
    if constraint_id is None:
        return False
    try:
        import pybullet as p

        p.getConstraintInfo(int(constraint_id))
        return True
    except Exception:
        return False


def ensure_breakaway_constraint(task: Any) -> Dict[str, Any]:
    """Ensure the unreleased breakaway constraint exists after restoreState.

    Some PyBullet builds restore user constraints, while others can leave a
    removed constraint absent. This function is a no-op when the saved
    constraint exists and a deterministic reconstruction fallback otherwise.
    """
    result = {
        "required": False,
        "existing": False,
        "recreated": False,
        "old_id": None,
        "new_id": None,
    }
    if getattr(task, "hidden_condition", "") != "hidden_breakaway_pin":
        return result
    if bool(getattr(task, "_breakaway_released", False)):
        return result

    bead_id = getattr(task, "_breakaway_bead_id", None)
    anchor = getattr(task, "_breakaway_anchor_pos", None)
    old_id = getattr(task, "_breakaway_constraint_id", None)
    result["required"] = True
    result["old_id"] = old_id

    if _constraint_exists(old_id):
        result["existing"] = True
        result["new_id"] = int(old_id)
        return result

    if bead_id is None or anchor is None:
        raise RuntimeError(
            "Unreleased breakaway state has no bead or anchor after restore."
        )

    import pybullet as p

    cid = p.createConstraint(
        parentBodyUniqueId=int(bead_id),
        parentLinkIndex=-1,
        childBodyUniqueId=-1,
        childLinkIndex=-1,
        jointType=p.JOINT_POINT2POINT,
        jointAxis=(0, 0, 0),
        parentFramePosition=(0, 0, 0),
        childFramePosition=tuple(float(x) for x in anchor),
    )
    max_force = float(os.environ.get("CCDA_BREAKAWAY_FORCE", "2.6"))
    p.changeConstraint(cid, maxForce=max_force)

    task._breakaway_constraint_id = int(cid)
    ids = [
        int(x)
        for x in list(getattr(task, "hidden_constraint_ids", []))
        if _constraint_exists(int(x))
    ]
    if int(cid) not in ids:
        ids.append(int(cid))
    task.hidden_constraint_ids = ids

    meta = copy.deepcopy(getattr(task, "hidden_contact_meta", {}))
    meta["constraint_id"] = int(cid)
    meta["hidden_constraint_ids"] = list(ids)
    task.hidden_contact_meta = meta

    result["recreated"] = True
    result["new_id"] = int(cid)
    return result


def capture_robot_state(env: Any) -> Dict[str, Any]:
    import pybullet as p

    joints = list(getattr(env, "joints", []))
    values = [p.getJointState(env.ur5, int(j)) for j in joints]
    q = np.asarray([v[0] for v in values], dtype=np.float64)
    dq = np.asarray([v[1] for v in values], dtype=np.float64)
    ee_link = int(getattr(env, "ee_tip_link", 12))
    link = p.getLinkState(env.ur5, ee_link, computeLinkVelocity=1)
    return {
        "joints": [int(x) for x in joints],
        "q": q,
        "dq": dq,
        "ee_position": np.asarray(link[0], dtype=np.float64),
        "ee_orientation": np.asarray(link[1], dtype=np.float64),
    }


def restore_robot_state(env: Any, payload: Mapping[str, Any]) -> None:
    import pybullet as p

    joints = [int(x) for x in payload["joints"]]
    q = np.asarray(payload["q"], dtype=np.float64)
    dq = np.asarray(payload["dq"], dtype=np.float64)
    for index, joint_id in enumerate(joints):
        p.resetJointState(
            env.ur5,
            joint_id,
            targetValue=float(q[index]),
            targetVelocity=float(dq[index]),
        )
    p.setJointMotorControlArray(
        bodyIndex=env.ur5,
        jointIndices=joints,
        controlMode=p.POSITION_CONTROL,
        targetPositions=[float(x) for x in q],
        positionGains=np.ones(len(joints)),
    )


def ordered_bead_xy(task: Any) -> np.ndarray:
    import pybullet as p

    points = []
    for bead_id in task.cable_bead_IDs:
        position = p.getBasePositionAndOrientation(int(bead_id))[0]
        points.append([position[0], position[1]])
    return np.asarray(points, dtype=np.float64)


def ordered_bead_velocity(task: Any) -> np.ndarray:
    import pybullet as p

    values = []
    for bead_id in task.cable_bead_IDs:
        linear, _ = p.getBaseVelocity(int(bead_id))
        values.append([linear[0], linear[1], linear[2]])
    return np.asarray(values, dtype=np.float64)


def ordered_target_xy(task: Any) -> np.ndarray:
    targets = []
    for bead_id in task.cable_bead_IDs:
        target = task.goal["places"][bead_id][0]
        targets.append([target[0], target[1]])
    return np.asarray(targets, dtype=np.float64)


def cable_curve(points: np.ndarray) -> float:
    xy = np.asarray(points, dtype=np.float64)
    if len(xy) < 3:
        return 0.0
    first = np.diff(xy, axis=0)
    norms = np.linalg.norm(first, axis=1, keepdims=True)
    unit = first / np.maximum(norms, 1e-12)
    turns = np.diff(unit, axis=0)
    return float(np.mean(np.linalg.norm(turns, axis=1)))


def pairwise_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    return np.linalg.norm(aa[:, None, :] - bb[None, :, :], axis=2)


def geometry_metrics(task: Any) -> Dict[str, float]:
    beads = ordered_bead_xy(task)
    targets = ordered_target_xy(task)
    direct = float(np.mean(np.linalg.norm(beads - targets, axis=1)))
    reverse = float(np.mean(np.linalg.norm(beads - targets[::-1], axis=1)))
    ordered = min(direct, reverse)
    distance = pairwise_distances(beads, targets)
    bead_to_target = np.min(distance, axis=1)
    target_to_bead = np.min(distance, axis=0)
    chamfer = float(0.5 * (np.mean(bead_to_target) + np.mean(target_to_bead)))
    sigma = max(float(getattr(task, "radius", 0.005)) * 3.5, 1e-6)
    soft_coverage = float(np.mean(np.exp(-bead_to_target / sigma)))
    endpoint_direct = float(
        0.5
        * (
            np.linalg.norm(beads[0] - targets[0])
            + np.linalg.norm(beads[-1] - targets[-1])
        )
    )
    endpoint_reverse = float(
        0.5
        * (
            np.linalg.norm(beads[0] - targets[-1])
            + np.linalg.norm(beads[-1] - targets[0])
        )
    )
    endpoint = min(endpoint_direct, endpoint_reverse)
    threshold = float(getattr(task, "radius", 0.005)) * 3.5
    fraction = float(np.mean(bead_to_target < threshold))
    return {
        "ordered_distance": ordered,
        "chamfer_distance": chamfer,
        "soft_coverage": soft_coverage,
        "endpoint_distance": endpoint,
        "curve": cable_curve(beads),
        "fraction": fraction,
    }


def dense_effect(before: Mapping[str, float], after: Mapping[str, float]) -> Dict[str, float]:
    ordered_gain = float(before["ordered_distance"] - after["ordered_distance"])
    chamfer_gain = float(before["chamfer_distance"] - after["chamfer_distance"])
    endpoint_gain = float(before["endpoint_distance"] - after["endpoint_distance"])
    coverage_gain = float(after["soft_coverage"] - before["soft_coverage"])
    curve_gain = float(before["curve"] - after["curve"])
    dense = (
        ordered_gain
        + 0.5 * chamfer_gain
        + 0.25 * endpoint_gain
        + 0.01 * coverage_gain
    )
    return {
        "dense_gain": float(dense),
        "ordered_gain": ordered_gain,
        "chamfer_gain": chamfer_gain,
        "endpoint_gain": endpoint_gain,
        "soft_coverage_gain": coverage_gain,
        "curve_gain": curve_gain,
        "fraction_gain": float(after["fraction"] - before["fraction"]),
    }


def capture_query_state(task: Any, env: Any) -> Dict[str, Any]:
    beads = ordered_bead_xy(task)
    velocities = ordered_bead_velocity(task)
    robot = capture_robot_state(env)
    return {
        "bead_xy": beads,
        "bead_velocity": velocities,
        "robot": robot,
        "metrics": geometry_metrics(task),
        "breakaway_released": bool(
            getattr(task, "_breakaway_released", False)
        ),
        "breakaway_constraint_id": getattr(
            task, "_breakaway_constraint_id", None
        ),
        "breakaway_anchor_pos": copy.deepcopy(
            getattr(task, "_breakaway_anchor_pos", None)
        ),
        "breakaway_bead_id": getattr(task, "_breakaway_bead_id", None),
        "breakaway_max_disp_seen": float(
            getattr(task, "_breakaway_max_disp_seen", 0.0)
        ),
        "task_stage": copy.deepcopy(getattr(task, "task_stage", None)),
        "task_step": int(getattr(task, "t", 0)),
        "action_step_count": int(getattr(task, "_ccda_step_count", 0)),
        "physics_step_count": int(
            getattr(task, "_ccda_physics_step_count", 0)
        ),
        "bead_xy_sha256": sha256_array(beads),
        "robot_q_sha256": sha256_array(robot["q"]),
    }


def compare_query_state(
    reference: Mapping[str, Any],
    restored: Mapping[str, Any],
    *,
    bead_xy_max_abs_threshold: float,
    bead_xy_mae_threshold: float,
    robot_q_max_abs_threshold: float,
    fraction_threshold: float,
    curve_threshold: float,
) -> Dict[str, Any]:
    bead_diff = np.asarray(restored["bead_xy"]) - np.asarray(
        reference["bead_xy"]
    )
    velocity_diff = np.asarray(restored["bead_velocity"]) - np.asarray(
        reference["bead_velocity"]
    )
    robot_diff = np.asarray(restored["robot"]["q"]) - np.asarray(
        reference["robot"]["q"]
    )
    bead_max = float(np.max(np.abs(bead_diff)))
    bead_mae = float(np.mean(np.abs(bead_diff)))
    velocity_max = float(np.max(np.abs(velocity_diff)))
    velocity_mae = float(np.mean(np.abs(velocity_diff)))
    robot_max = float(np.max(np.abs(robot_diff)))
    fraction_diff = abs(
        float(restored["metrics"]["fraction"])
        - float(reference["metrics"]["fraction"])
    )
    curve_diff = abs(
        float(restored["metrics"]["curve"])
        - float(reference["metrics"]["curve"])
    )
    breakaway_equal = (
        bool(restored["breakaway_released"])
        == bool(reference["breakaway_released"])
    )
    task_stage_equal = restored["task_stage"] == reference["task_stage"]
    passed = (
        bead_max <= float(bead_xy_max_abs_threshold)
        and bead_mae <= float(bead_xy_mae_threshold)
        and robot_max <= float(robot_q_max_abs_threshold)
        and fraction_diff <= float(fraction_threshold)
        and curve_diff <= float(curve_threshold)
        and breakaway_equal
        and task_stage_equal
    )
    return {
        "pass": bool(passed),
        "bead_xy_max_abs": bead_max,
        "bead_xy_mae": bead_mae,
        "bead_velocity_max_abs": velocity_max,
        "bead_velocity_mae": velocity_mae,
        "robot_q_max_abs": robot_max,
        "fraction_diff": float(fraction_diff),
        "curve_diff": float(curve_diff),
        "breakaway_equal": bool(breakaway_equal),
        "task_stage_equal": bool(task_stage_equal),
    }


@dataclass
class QuerySnapshot:
    bullet_state_id: int
    task_state: Dict[str, Any]
    ee_state: Dict[str, Any]
    robot_state: Dict[str, Any]
    rng_state: Dict[str, Any]
    reference: Dict[str, Any]

    @classmethod
    def create(cls, task: Any, env: Any) -> "QuerySnapshot":
        import pybullet as p

        pause_and_drain(env)
        if bool(getattr(env.ee, "activated", False)):
            raise RuntimeError("Cannot snapshot while end effector is activated.")
        if getattr(env.ee, "contact_constraint", None) is not None:
            raise RuntimeError("Cannot snapshot with active suction constraint.")
        with simulation_lock(env):
            state_id = int(p.saveState())
        return cls(
            bullet_state_id=state_id,
            task_state=capture_object_fields(task, TASK_STATE_FIELDS),
            ee_state=capture_object_fields(env.ee, EE_STATE_FIELDS),
            robot_state=capture_robot_state(env),
            rng_state=capture_rng_state(),
            reference=capture_query_state(task, env),
        )

    def restore(
        self,
        task: Any,
        env: Any,
        *,
        bead_xy_max_abs_threshold: float,
        bead_xy_mae_threshold: float,
        robot_q_max_abs_threshold: float,
        fraction_threshold: float,
        curve_threshold: float,
    ) -> Dict[str, Any]:
        import pybullet as p

        pause_and_drain(env)
        setattr(env, "_ccda_physics_hook_error", None)
        with simulation_lock(env):
            p.restoreState(self.bullet_state_id)
        restore_object_fields(task, self.task_state)
        restore_object_fields(env.ee, self.ee_state)
        restore_robot_state(env, self.robot_state)
        restore_rng_state(self.rng_state)
        constraint = ensure_breakaway_constraint(task)
        restored = capture_query_state(task, env)
        integrity = compare_query_state(
            self.reference,
            restored,
            bead_xy_max_abs_threshold=bead_xy_max_abs_threshold,
            bead_xy_mae_threshold=bead_xy_mae_threshold,
            robot_q_max_abs_threshold=robot_q_max_abs_threshold,
            fraction_threshold=fraction_threshold,
            curve_threshold=curve_threshold,
        )
        integrity["constraint_restore"] = constraint
        return integrity

    def remove(self) -> None:
        import pybullet as p

        try:
            p.removeState(self.bullet_state_id)
        except Exception:
            pass


def goal_geometry_action(task: Any) -> Dict[str, Any]:
    beads = ordered_bead_xy(task)
    targets = ordered_target_xy(task)
    direct_cost = float(np.mean(np.linalg.norm(beads - targets, axis=1)))
    reverse_targets = targets[::-1]
    reverse_cost = float(
        np.mean(np.linalg.norm(beads - reverse_targets, axis=1))
    )
    assignment = targets if direct_cost <= reverse_cost else reverse_targets
    distances = np.linalg.norm(beads - assignment, axis=1)
    index = int(np.argmax(distances))
    pick_xy = beads[index]
    place_xy = assignment[index]

    p0 = (float(pick_xy[0]), float(pick_xy[1]), 0.001)
    p1 = (float(place_xy[0]), float(place_xy[1]), 0.001)
    quaternion = (0.0, 0.0, 0.0, 1.0)
    return {
        "primitive": "pick_place",
        "params": {
            "pose0": (p0, quaternion),
            "pose1": (p1, quaternion),
        },
    }


def action_pose_metrics(action: Mapping[str, Any]) -> Dict[str, float]:
    try:
        p0 = np.asarray(action["params"]["pose0"][0], dtype=np.float64)
        p1 = np.asarray(action["params"]["pose1"][0], dtype=np.float64)
        return {
            "pose0_x": float(p0[0]),
            "pose0_y": float(p0[1]),
            "pose1_x": float(p1[0]),
            "pose1_y": float(p1[1]),
            "pull_len": float(np.linalg.norm(p1[:2] - p0[:2])),
        }
    except Exception:
        return {
            "pose0_x": float("nan"),
            "pose0_y": float("nan"),
            "pose1_x": float("nan"),
            "pose1_y": float("nan"),
            "pull_len": float("nan"),
        }


def serialize_action_vector(codec: Any, action: Mapping[str, Any]) -> List[float]:
    try:
        value = codec.encode(action)
        return [float(x) for x in np.asarray(value).reshape(-1)]
    except Exception:
        return []


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    samples: int = 10000,
    seed: int = 312013,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    array = np.asarray(
        [float(x) for x in values if math.isfinite(float(x))],
        dtype=np.float64,
    )
    if array.size == 0:
        return float("nan"), float("nan")
    if array.size == 1:
        return float(array[0]), float(array[0])
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(
        0, array.size, size=(int(samples), array.size)
    )
    means = np.mean(array[indices], axis=1)
    return (
        float(np.quantile(means, alpha / 2.0)),
        float(np.quantile(means, 1.0 - alpha / 2.0)),
    )


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe_copy(dict(payload)), indent=2, sort_keys=True)
    )
    temporary.replace(path)
