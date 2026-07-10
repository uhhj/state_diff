#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pybullet as p

# Re-export the r1 candidate/snapshot helpers. Phase3.12d-r2 changes only
# environment initialization and gating; candidate generation, PyBullet
# saveState/restoreState and realized-effect analysis remain the validated r1
# implementation.
import phase3_12d_r1_common as r1
from phase3_12d_r1_common import *  # noqa: F401,F403
import phase3_12c_matched_reset_common as p12c

# Explicitly expose the deterministic seeding helper used by the audit and
# query-local workers. Do not rely on transitive imports from the r1 module.
seed_everything = p12c.seed_everything


R2_SCOPE = "phase3_12d_r2_paired_visible_arming_no_phase4_no_cps"


@contextlib.contextmanager
def temporary_environment(updates: Mapping[str, str]) -> Iterator[None]:
    old: Dict[str, Optional[str]] = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            os.environ[key] = str(value)
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _lock_context(env: Any):
    lock = getattr(env, "_ccda_step_lock", None)
    if lock is None:
        return contextlib.nullcontext()
    return lock


def ordered_bead_xy(task: Any) -> np.ndarray:
    return np.asarray(
        [
            p.getBasePositionAndOrientation(int(bead_id))[0][:2]
            for bead_id in task.cable_bead_IDs
        ],
        dtype=np.float32,
    )


def ordered_bead_xyz(task: Any) -> np.ndarray:
    return np.asarray(
        [
            p.getBasePositionAndOrientation(int(bead_id))[0]
            for bead_id in task.cable_bead_IDs
        ],
        dtype=np.float32,
    )


def ordered_bead_velocity(task: Any) -> np.ndarray:
    return np.asarray(
        [
            p.getBaseVelocity(int(bead_id))[0]
            for bead_id in task.cable_bead_IDs
        ],
        dtype=np.float32,
    )


def difference_stats(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} != {b.shape}")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("non-finite values in difference_stats")
    diff = a - b
    return {
        "max_abs": float(np.max(np.abs(diff))) if diff.size else 0.0,
        "mae": float(np.mean(np.abs(diff))) if diff.size else 0.0,
        "rmse": float(np.sqrt(np.mean(diff * diff))) if diff.size else 0.0,
    }


def canonicalize_visible_dynamics(env: Any, task: Any) -> Dict[str, Any]:
    """Remove condition-independent residual velocity before latent arming.

    This is applied identically to every condition after the common settling
    trajectory. It does not move any body and therefore preserves the visible
    geometry while preventing a zero-offset constraint from reacting to
    leftover cable velocity on the first post-arm step.
    """
    bead_ids = [int(x) for x in task.cable_bead_IDs]
    robot_joints = [int(x) for x in getattr(env, "joints", [])]

    with _lock_context(env):
        for bead_id in bead_ids:
            p.resetBaseVelocity(
                bead_id,
                linearVelocity=(0.0, 0.0, 0.0),
                angularVelocity=(0.0, 0.0, 0.0),
            )

        ur5 = getattr(env, "ur5", None)
        if isinstance(ur5, (int, np.integer)):
            for joint_id in robot_joints:
                state = p.getJointState(int(ur5), int(joint_id))
                p.resetJointState(
                    int(ur5),
                    int(joint_id),
                    targetValue=float(state[0]),
                    targetVelocity=0.0,
                )

    return {
        "canonicalized_beads": len(bead_ids),
        "canonicalized_robot_joints": len(robot_joints),
        "max_bead_speed_after": float(
            np.max(np.linalg.norm(ordered_bead_velocity(task), axis=1))
        )
        if bead_ids
        else 0.0,
    }


def direct_physics_step(
    env: Any,
    task: Any,
    *,
    dispatch_hook: bool,
    external_force: Optional[Tuple[int, Sequence[float], Sequence[float]]] = None,
) -> None:
    """Advance exactly one synchronous simulation step while env is paused."""
    with _lock_context(env):
        if external_force is not None:
            body_id, force, position = external_force
            p.applyExternalForce(
                int(body_id),
                -1,
                forceObj=[float(v) for v in force],
                posObj=[float(v) for v in position],
                flags=p.WORLD_FRAME,
            )
        p.stepSimulation()
        ee = getattr(env, "ee", None)
        if ee is not None:
            ee.step()
        if dispatch_hook:
            hook = getattr(task, "physics_step_hook", None)
            if callable(hook):
                hook()


def _reward_info(env: Any, task: Any) -> Dict[str, Any]:
    _, extras = task.reward()
    extras["task.done"] = bool(task.done())
    info = env.info
    info["extras"] = extras
    return info


def _arm_anchor_error(task: Any) -> float:
    if getattr(task, "hidden_condition", "") != "hidden_breakaway_pin":
        return 0.0
    bead_id = getattr(task, "_breakaway_bead_id", None)
    anchor = getattr(task, "_breakaway_anchor_pos", None)
    if bead_id is None or anchor is None:
        return float("nan")
    bead = np.asarray(
        p.getBasePositionAndOrientation(int(bead_id))[0], dtype=np.float64
    )
    anchor_arr = np.asarray(anchor, dtype=np.float64)
    return float(np.linalg.norm(bead - anchor_arr))


def deterministic_reset_deferred_arm(
    env: Any,
    task: Any,
    *,
    min_settle_steps: int,
    max_settle_steps: int,
    static_checks_required: int,
    static_check_interval: int,
    arm_after_settle: bool = True,
    post_arm_steps: int = 0,
    canonicalize_velocity: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Condition-independent settle followed by zero-offset latent arming.

    The method deliberately mirrors Phase3.12c deterministic_reset, but the
    hidden condition remains pending during every common settling step. It is
    installed only after the visible cable has settled and residual velocities
    have been canonicalized identically across conditions.
    """
    min_settle_steps = int(min_settle_steps)
    max_settle_steps = int(max_settle_steps)
    static_checks_required = int(static_checks_required)
    static_check_interval = int(static_check_interval)
    post_arm_steps = int(post_arm_steps)

    if min_settle_steps < 0:
        raise ValueError("min_settle_steps must be >= 0")
    if max_settle_steps < min_settle_steps:
        raise ValueError("max_settle_steps must be >= min_settle_steps")
    if static_checks_required < 1:
        raise ValueError("static_checks_required must be >= 1")
    if static_check_interval < 1:
        raise ValueError("static_check_interval must be >= 1")
    if post_arm_steps < 0:
        raise ValueError("post_arm_steps must be >= 0")

    if hasattr(task, "_settle_secs"):
        task._settle_secs = 0.0

    original_start = env.start
    original_step = env.step

    def no_background_start() -> None:
        env.running = False

    def no_action_step(act=None):
        return {}, 0.0, False, {}

    env.pause()
    env.start = no_background_start
    env.step = no_action_step

    with temporary_environment(
        {
            "CCDA_SETTLE_SECONDS": "0",
            "CCDA_POST_ARM_SETTLE_SECONDS": "0",
            "CCDA_DEFER_HIDDEN_CONTACT_ARMING": "1",
            "CCDA_BREAKAWAY_DAMPING": os.environ.get(
                "CCDA_BREAKAWAY_DAMPING", "0.0"
            ),
        }
    ):
        try:
            env.reset(task)
        finally:
            env.start = original_start
            env.step = original_step
            env.pause()

        if getattr(task, "hidden_condition", "free") != "free":
            if not bool(getattr(task, "_hidden_contact_pending", False)):
                raise RuntimeError(
                    "Non-free task was not left pending during common settle"
                )
            if bool(getattr(task, "_hidden_contact_armed", False)):
                raise RuntimeError("Hidden contact armed before common settle")

        try:
            p.setPhysicsEngineParameter(deterministicOverlappingPairs=1)
        except Exception:
            pass
        p.setTimeStep(1.0 / float(env.hz))

        steps_used = 0
        stable_count = 0
        settled_static = False
        while steps_used < max_settle_steps:
            direct_physics_step(env, task, dispatch_hook=False)
            steps_used += 1
            if steps_used < min_settle_steps:
                continue
            if steps_used % static_check_interval != 0:
                continue
            try:
                currently_static = bool(env.is_static())
            except Exception:
                currently_static = False
            stable_count = stable_count + 1 if currently_static else 0
            if stable_count >= static_checks_required:
                settled_static = True
                break

        canonical = (
            canonicalize_visible_dynamics(env, task)
            if canonicalize_velocity
            else {"canonicalized_beads": 0, "canonicalized_robot_joints": 0}
        )
        # Reset physics-contact timing so release steps are measured from latent
        # arming / policy execution, not from the common free settle.
        if hasattr(task, "_ccda_physics_step_count"):
            task._ccda_physics_step_count = 0

        pre_arm_xy = ordered_bead_xy(task)
        pre_arm_vel = ordered_bead_velocity(task)

        arm_result: Dict[str, Any]
        if arm_after_settle:
            arm = getattr(task, "arm_hidden_contact_after_settle", None)
            if not callable(arm):
                raise RuntimeError("Task lacks arm_hidden_contact_after_settle()")
            arm_result = dict(arm(env))
        else:
            arm_result = {
                "already_armed": False,
                "condition": getattr(task, "hidden_condition", ""),
                "arm_mode": "audit_left_unarmed",
                "max_abs_jump": 0.0,
                "mae_jump": 0.0,
            }

        immediate_xy = ordered_bead_xy(task)
        immediate_vel = ordered_bead_velocity(task)
        immediate_stats = difference_stats(immediate_xy, pre_arm_xy)
        anchor_error = _arm_anchor_error(task) if arm_after_settle else 0.0

        for _ in range(post_arm_steps):
            direct_physics_step(env, task, dispatch_hook=True)

        post_xy = ordered_bead_xy(task)
        post_vel = ordered_bead_velocity(task)
        post_stats = difference_stats(post_xy, pre_arm_xy)
        velocity_stats = difference_stats(post_vel, pre_arm_vel)

        hook_error = getattr(env, "_ccda_physics_hook_error", None)
        if hook_error not in (None, "", "None"):
            raise RuntimeError(f"physics hook failed: {hook_error}")

        info = _reward_info(env, task)

    env.pause()
    metadata = {
        "scope": R2_SCOPE,
        "settle_steps_used": int(steps_used),
        "settle_static": bool(settled_static),
        "static_checks_observed": int(stable_count),
        "min_settle_steps": int(min_settle_steps),
        "max_settle_steps": int(max_settle_steps),
        "static_checks_required": int(static_checks_required),
        "static_check_interval": int(static_check_interval),
        "arm_after_settle": bool(arm_after_settle),
        "post_arm_steps": int(post_arm_steps),
        "canonicalization": canonical,
        "arm_result": arm_result,
        "arm_anchor_error": float(anchor_error),
        "pre_arm_xy": pre_arm_xy.astype(float).tolist(),
        "immediate_post_arm_xy": immediate_xy.astype(float).tolist(),
        "post_arm_xy": post_xy.astype(float).tolist(),
        "pre_arm_velocity": pre_arm_vel.astype(float).tolist(),
        "immediate_post_arm_velocity": immediate_vel.astype(float).tolist(),
        "post_arm_velocity": post_vel.astype(float).tolist(),
        "immediate_arm_jump": immediate_stats,
        "post_arm_drift": post_stats,
        "post_arm_velocity_change": velocity_stats,
        "hidden_contact_pending": bool(
            getattr(task, "_hidden_contact_pending", False)
        ),
        "hidden_contact_armed": bool(
            getattr(task, "_hidden_contact_armed", False)
        ),
        "breakaway_released": bool(
            getattr(task, "_breakaway_released", False)
        ),
        "breakaway_release_physics_step": getattr(
            task, "_breakaway_release_physics_step", None
        ),
        "hook_error": str(hook_error),
    }
    return info, metadata


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=True))


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False
