from __future__ import annotations

import contextlib
import os
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pybullet as p


@contextlib.contextmanager
def temporary_environment(updates: Mapping[str, str]):
    previous = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            os.environ[key] = str(value)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def lock_context(env: Any):
    lock = getattr(env, "_ccda_step_lock", None)
    return lock if lock is not None else contextlib.nullcontext()


def ordered_bead_xy(task: Any) -> np.ndarray:
    return np.asarray(
        [
            p.getBasePositionAndOrientation(int(body_id))[0][:2]
            for body_id in task.cable_bead_IDs
        ],
        dtype=np.float64,
    )


def ordered_bead_velocity_xy(task: Any) -> np.ndarray:
    return np.asarray(
        [
            p.getBaseVelocity(int(body_id))[0][:2]
            for body_id in task.cable_bead_IDs
        ],
        dtype=np.float64,
    )


def difference_stats(left: np.ndarray, right: np.ndarray) -> Dict[str, float]:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError(f"shape mismatch: {left.shape} != {right.shape}")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("non-finite values")
    difference = left - right
    return {
        "max_abs": float(np.max(np.abs(difference)))
        if difference.size
        else 0.0,
        "mae": float(np.mean(np.abs(difference)))
        if difference.size
        else 0.0,
        "rmse": float(np.sqrt(np.mean(difference * difference)))
        if difference.size
        else 0.0,
    }


def direct_physics_step(
    env: Any,
    task: Any,
    *,
    external_force: Optional[
        Tuple[int, Sequence[float], Sequence[float]]
    ] = None,
    dispatch_hooks: bool = True,
) -> None:
    """Advance one deterministic step with production hook ordering.

    Ordering:
    1. task.physics_pre_step_hook()
    2. optional diagnostic external force
    3. p.stepSimulation()
    4. end-effector step
    5. task.physics_step_hook()
    """
    with lock_context(env):
        if dispatch_hooks:
            pre_hook = getattr(task, "physics_pre_step_hook", None)
            if callable(pre_hook):
                pre_hook()

        if external_force is not None:
            body_id, force, position = external_force
            p.applyExternalForce(
                int(body_id),
                -1,
                forceObj=[float(value) for value in force],
                posObj=[float(value) for value in position],
                flags=p.WORLD_FRAME,
            )

        p.stepSimulation()

        ee = getattr(env, "ee", None)
        if ee is not None:
            ee.step()

        if dispatch_hooks:
            post_hook = getattr(task, "physics_step_hook", None)
            if callable(post_hook):
                post_hook()


def slack_snapshot(task: Any) -> Dict[str, Any]:
    capture = getattr(task, "ccda_snapshot_state", None)
    if not callable(capture):
        raise RuntimeError("task has no ccda_snapshot_state()")
    snapshot = capture()
    if not isinstance(snapshot, dict):
        raise RuntimeError("task snapshot is not a dict")
    return snapshot


def restore_slack_snapshot(task: Any, snapshot: Dict[str, Any]) -> None:
    restore = getattr(task, "ccda_restore_state", None)
    if not callable(restore):
        raise RuntimeError("task has no ccda_restore_state()")
    restore(snapshot)


def assert_slack_v2_structure(task: Any) -> None:
    if getattr(task, "hidden_condition", "") != (
        "hidden_slack_breakaway_pin_v2"
    ):
        raise RuntimeError("not a slack-breakaway-v2 task")
    if getattr(task, "_slack_model", None) is None:
        raise RuntimeError("slack model missing")
    if getattr(task, "_slack_bead_id", None) is None:
        raise RuntimeError("slack bead missing")
    if list(getattr(task, "hidden_body_ids", [])):
        raise RuntimeError("v2 created hidden bodies")
    if list(getattr(task, "hidden_constraint_ids", [])):
        raise RuntimeError("v2 created PyBullet constraints")
