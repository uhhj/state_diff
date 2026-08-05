"""Exact same-world counterfactual replay for Hidden-Friction Cable."""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
SUBMODULE_ROOT = REPO_ROOT / "external" / "deformable-ravens"
for path in (REPO_ROOT, SUBMODULE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import pybullet as p  # noqa: E402
from ravens import Environment, tasks  # noqa: E402
from scripts.experiment2.phase0.common import canonical_json_sha256
from scripts.experiment2.phase0.observation_common import (
    render_fixed_rgb,
    save_rgb,
)
from scripts.experiment2.phase0.run_hidden_friction_pairs import (
    _bead_positions,
    _configure_task_environment,
    _environment_action,
    _trace_arrays,
    generate_action_script,
)


def array_payload_sha256(payload: Dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(payload):
        value = np.ascontiguousarray(np.asarray(payload[name]))
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def capture_world_state(env: Environment, task: Any) -> Dict[str, np.ndarray]:
    bead_positions = []
    bead_orientations = []
    bead_linear_velocities = []
    bead_angular_velocities = []
    for bead in task.cable_bead_IDs:
        position, orientation = p.getBasePositionAndOrientation(int(bead))
        linear, angular = p.getBaseVelocity(int(bead))
        bead_positions.append(position)
        bead_orientations.append(orientation)
        bead_linear_velocities.append(linear)
        bead_angular_velocities.append(angular)

    joint_states = [p.getJointState(env.ur5, int(j)) for j in env.joints]
    # Force the same forward-kinematics path before and after restoreState;
    # otherwise PyBullet can return a pre-restore cached link pose once.
    ee_state = p.getLinkState(
        env.ur5,
        env.ee_tip_link,
        computeForwardKinematics=True,
    )
    return {
        "bead_positions": np.asarray(bead_positions, dtype=np.float64),
        "bead_orientations": np.asarray(bead_orientations, dtype=np.float64),
        "bead_linear_velocities": np.asarray(
            bead_linear_velocities, dtype=np.float64
        ),
        "bead_angular_velocities": np.asarray(
            bead_angular_velocities, dtype=np.float64
        ),
        "joint_positions": np.asarray(
            [state[0] for state in joint_states], dtype=np.float64
        ),
        "joint_velocities": np.asarray(
            [state[1] for state in joint_states], dtype=np.float64
        ),
        "ee_position": np.asarray(ee_state[0], dtype=np.float64),
        "ee_orientation": np.asarray(ee_state[1], dtype=np.float64),
    }


def max_state_difference(
    first: Dict[str, np.ndarray],
    second: Dict[str, np.ndarray],
) -> float:
    if set(first) != set(second):
        raise ValueError("world-state payload keys do not match")
    maximum = 0.0
    for name in sorted(first):
        a = np.asarray(first[name], dtype=np.float64)
        b = np.asarray(second[name], dtype=np.float64)
        if a.shape != b.shape:
            raise ValueError(
                f"state shape mismatch for {name}: {a.shape} != {b.shape}"
            )
        if a.size:
            maximum = max(maximum, float(np.max(np.abs(a - b))))
    return maximum


def _arm_ccda_hidden_factor(task: Any):
    method = getattr(task, "arm_ccda_hidden_factor_after_settle", None)
    if callable(method):
        return method()
    legacy = getattr(task, "arm_hidden_friction_after_settle", None)
    if callable(legacy):
        return legacy()
    raise AttributeError("task has no CCDA hidden-factor arm method")


def _mark_event(task: Any, events: List[Dict[str, Any]], name: str) -> None:
    events.append(
        {
            "name": str(name),
            "physics_step": int(task.physics_step_count()),
        }
    )


def _capture_event_observation(
    env: Environment,
    task: Any,
    output_dir: Optional[Path],
    observation_config: Optional[Dict[str, Any]],
    condition: str,
    event: str,
) -> Optional[Dict[str, Any]]:
    if output_dir is None or observation_config is None:
        return None
    before = capture_world_state(env, task)
    rgb = render_fixed_rgb(observation_config)
    after = capture_world_state(env, task)
    state_difference = max_state_difference(before, after)
    if state_difference != 0.0:
        raise RuntimeError(
            f"rendering changed physics state by {state_difference}"
        )
    result = save_rgb(
        Path(output_dir) / f"{condition}_{event}.png",
        rgb,
    )
    result["state_difference_after_render"] = state_difference
    result["event"] = event
    return result


def _run_restored_branch(
    env: Environment,
    task: Any,
    state_id: int,
    base_state: Dict[str, np.ndarray],
    config: Dict[str, Any],
    action_generator,
    condition: str,
    action_script: Optional[List[Dict[str, Any]]],
    observation_output_dir: Optional[Path],
    observation_config: Optional[Dict[str, Any]],
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any], List[Dict[str, Any]]]:
    env.pause()
    with env._ccda_step_lock:
        p.restoreState(stateId=state_id)
        env.reset_ccda_runtime_after_restore()
        task.reset_ccda_branch(condition)
        _arm_ccda_hidden_factor(task)
        env.reset_ccda_motion_events()
        branch_initial = capture_world_state(env, task)

    base_hash = array_payload_sha256(base_state)
    initial_hash = array_payload_sha256(branch_initial)
    initial_difference = max_state_difference(base_state, branch_initial)
    observations: Dict[str, Any] = {}
    events: List[Dict[str, Any]] = []
    _mark_event(task, events, "branch_start")

    task.set_ccda_phase("no_action")
    env.step_physics(int(config["no_action_steps"]))
    _mark_event(task, events, "no_action_end")
    no_action_observation = _capture_event_observation(
        env,
        task,
        observation_output_dir,
        observation_config,
        condition,
        "no_action_end",
    )
    if no_action_observation is not None:
        observations["no_action_end"] = no_action_observation

    stable_beads = _bead_positions(task)

    if condition == "free" and action_script is None:
        action_script = action_generator(config, stable_beads)
    elif action_script is None:
        raise ValueError("hidden branch requires the free action script")

    action_script = json.loads(json.dumps(action_script, sort_keys=True))
    action_hash = canonical_json_sha256(action_script)
    main_actions = [a for a in action_script if a["phase"] == "main_pull"]
    if len(main_actions) != 1:
        raise ValueError("action script must contain exactly one main_pull")

    preload_actions = [a for a in action_script if a["phase"] == "preload"]
    if not preload_actions:
        raise ValueError("action script contains no preload action")

    for action in preload_actions:
        task.set_ccda_phase("preload")
        _mark_event(task, events, action["name"] + "_start")
        env.step(_environment_action(action))
        _mark_event(task, events, action["name"] + "_end")

    pre_main_observation = _capture_event_observation(
        env,
        task,
        observation_output_dir,
        observation_config,
        condition,
        "pre_main",
    )
    if pre_main_observation is not None:
        observations["pre_main"] = pre_main_observation

    main_action = main_actions[0]
    task.set_ccda_phase("main_pull")
    _mark_event(task, events, "main_pull_start")
    env.step(_environment_action(main_action))
    _mark_event(task, events, "main_pull_end")

    task.set_ccda_phase("post_main")
    env.step_physics(int(config["post_main_steps"]))
    _mark_event(task, events, "post_main_end")

    if env._ccda_physics_hook_error:
        raise RuntimeError(env._ccda_physics_hook_error)

    trace = _trace_arrays(task.ccda_trace())
    metadata = {
        "condition": condition,
        "action_script": action_script,
        "action_hash": action_hash,
        "base_state_hash": base_hash,
        "branch_initial_state_hash": initial_hash,
        "base_state_hash_match": base_hash == initial_hash,
        "initial_state_max_abs_difference": initial_difference,
        "pairing_mode": "pybullet_save_restore_fixed_step",
        "execution": {
            "deterministic": bool(env.deterministic),
            "control_substeps": int(env.control_substeps),
            "post_action_settle_steps": int(
                env.post_action_settle_steps
            ),
            "hz": int(env.hz),
        },
        "events": events,
        "motion_events": env.ccda_motion_events(),
        "observations": observations,
        "trace_hash": array_payload_sha256(trace),
        "trace_length": int(trace["phase"].shape[0]),
        "privileged_state": task.ccda_privileged_state(),
        "physics_steps_observed": int(task.physics_step_count()),
    }
    return trace, metadata, action_script


def run_exact_counterfactual_pair(
    config: Dict[str, Any],
    seed: int,
    group_id: str,
    disp: bool = False,
    execution: Optional[Dict[str, Any]] = None,
    observation_output_dir: Optional[Path] = None,
    observation_config: Optional[Dict[str, Any]] = None,
    hidden_condition: str = "hidden_high_friction",
    configure_environment_fn=None,
    action_generator_fn=None,
):
    configure_environment_fn = configure_environment_fn or _configure_task_environment
    action_generator_fn = action_generator_fn or generate_action_script
    execution = dict(execution or {})
    deterministic = bool(execution.get("deterministic", True))
    control_substeps = int(execution.get("control_substeps", 1))
    post_action_settle_steps = int(
        execution.get("post_action_settle_steps", 240)
    )

    random.seed(seed)
    np.random.seed(seed)
    configure_environment_fn(config, "free", seed, group_id)
    os.environ["CCDA_DEFER_HIDDEN_FACTOR_ARMING"] = "1"
    os.environ["CCDA_DEFER_HIDDEN_FRICTION_ARMING"] = "1"

    env = None
    state_id = None
    try:
        env = Environment(
            disp=disp,
            hz=int(config["hz"]),
            deterministic=deterministic,
            control_substeps=control_substeps,
            post_action_settle_steps=post_action_settle_steps,
        )
        task = tasks.names[config["task_name"]]()
        env.reset(task)
        if task.ccda_is_armed():
            raise RuntimeError(
                "hidden friction armed before base-state snapshot"
            )

        env.pause()
        with env._ccda_step_lock:
            base_state = capture_world_state(env, task)
            base_state_hash = array_payload_sha256(base_state)
            state_id = int(p.saveState())

        free_observation_dir = (
            Path(observation_output_dir) / "free"
            if observation_output_dir is not None
            else None
        )
        hidden_observation_dir = (
            Path(observation_output_dir) / str(hidden_condition)
            if observation_output_dir is not None
            else None
        )
        free_trace, free_meta, action_script = _run_restored_branch(
            env,
            task,
            state_id,
            base_state,
            config,
            action_generator_fn,
            "free",
            None,
            free_observation_dir,
            observation_config,
        )
        hidden_trace, hidden_meta, hidden_action = _run_restored_branch(
            env,
            task,
            state_id,
            base_state,
            config,
            action_generator_fn,
            str(hidden_condition),
            action_script,
            hidden_observation_dir,
            observation_config,
        )

        if canonical_json_sha256(action_script) != canonical_json_sha256(
            hidden_action
        ):
            raise RuntimeError("hidden branch changed the paired action script")

        pair_metadata = {
            "seed": int(seed),
            "group_id": str(group_id),
            "hidden_condition": str(hidden_condition),
            "base_state_hash": base_state_hash,
            "free_initial_hash": free_meta["branch_initial_state_hash"],
            "hidden_initial_hash": hidden_meta[
                "branch_initial_state_hash"
            ],
            "free_base_state_hash_match": free_meta[
                "base_state_hash_match"
            ],
            "hidden_base_state_hash_match": hidden_meta[
                "base_state_hash_match"
            ],
            "free_hidden_initial_hash_match": (
                free_meta["branch_initial_state_hash"]
                == hidden_meta["branch_initial_state_hash"]
            ),
            "max_initial_state_difference": max(
                float(free_meta["initial_state_max_abs_difference"]),
                float(hidden_meta["initial_state_max_abs_difference"]),
            ),
            "action_hash": free_meta["action_hash"],
            "action_hash_match": (
                free_meta["action_hash"] == hidden_meta["action_hash"]
            ),
            "pairing_mode": "pybullet_save_restore_fixed_step",
            "execution": free_meta["execution"],
            "free_trace_hash": free_meta["trace_hash"],
            "hidden_trace_hash": hidden_meta["trace_hash"],
            "free_trace_length": free_meta["trace_length"],
            "hidden_trace_length": hidden_meta["trace_length"],
            "free_events": free_meta["events"],
            "hidden_events": hidden_meta["events"],
        }
        return (
            free_trace,
            hidden_trace,
            free_meta,
            hidden_meta,
            action_script,
            pair_metadata,
        )
    finally:
        os.environ.pop("CCDA_DEFER_HIDDEN_FACTOR_ARMING", None)
        os.environ.pop("CCDA_DEFER_HIDDEN_FRICTION_ARMING", None)
        if env is not None:
            env.pause()
            with env._ccda_step_lock:
                if state_id is not None:
                    try:
                        p.removeState(stateUniqueId=state_id)
                    except Exception:
                        pass
            env.stop()
