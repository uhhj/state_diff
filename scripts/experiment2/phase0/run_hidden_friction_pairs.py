#!/usr/bin/env python3
"""Run small paired Hidden-Friction Cable pilot rollouts."""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
SUBMODULE_ROOT = REPO_ROOT / "external" / "deformable-ravens"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SUBMODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(SUBMODULE_ROOT))

import pybullet as p  # noqa: E402
from ravens import Environment, tasks  # noqa: E402
from scripts.experiment2.phase0.common import (  # noqa: E402
    canonical_json_sha256,
    compute_pair_metrics,
)


TRACE_KEYS = (
    "physics_step",
    "phase",
    "bead_positions",
    "bead_velocities",
    "joint_positions",
    "joint_velocities",
    "ee_position",
    "ee_orientation",
    "contact_force_xyz",
    "contact_force_norm",
    "contact_max_force_norm",
    "contact_active_beads",
    "contact_mean_speed",
)


def _git_sha(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def wait_physics_steps(task, delta_steps: int, timeout_s: float = 30.0) -> None:
    if delta_steps < 0:
        raise ValueError("delta_steps must be non-negative")
    start = task.physics_step_count()
    wall_start = time.time()
    while task.physics_step_count() - start < delta_steps:
        if time.time() - wall_start > timeout_s:
            raise TimeoutError(
                f"timed out waiting for {delta_steps} physics steps; "
                f"observed {task.physics_step_count() - start}"
            )
        time.sleep(0.002)


def _bead_positions(task) -> np.ndarray:
    return np.asarray(
        [p.getBasePositionAndOrientation(int(bead))[0] for bead in task.cable_bead_IDs],
        dtype=np.float64,
    )


def _pose(position: np.ndarray) -> Dict[str, Any]:
    return {
        "position": [float(value) for value in position],
        "quaternion": [0.0, 0.0, 0.0, 1.0],
    }


def generate_action_script(config: Dict[str, Any], beads: np.ndarray) -> List[Dict[str, Any]]:
    values = np.asarray(beads, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] != 3:
        raise ValueError(f"expected ordered beads [N,3], got {values.shape}")
    center = np.mean(values[:, :2], axis=0)
    preload_distance = float(config["action"]["preload_distance"])
    main_distance = float(config["action"]["main_pull_distance"])
    pick_z = float(config["action"]["pick_z"])
    x_bounds = np.asarray(config["workspace_bounds"]["x"], dtype=np.float64)
    y_bounds = np.asarray(config["workspace_bounds"]["y"], dtype=np.float64)

    candidates = []
    for index in (0, values.shape[0] - 1):
        endpoint_xy = values[index, :2]
        outward = endpoint_xy - center
        norm = float(np.linalg.norm(outward))
        if norm <= 1e-12:
            continue
        direction = outward / norm
        final_xy = endpoint_xy + direction * (preload_distance + main_distance)
        legal = bool(
            x_bounds[0] <= final_xy[0] <= x_bounds[1]
            and y_bounds[0] <= final_xy[1] <= y_bounds[1]
        )
        if not legal:
            continue
        margin = min(
            final_xy[0] - x_bounds[0],
            x_bounds[1] - final_xy[0],
            final_xy[1] - y_bounds[0],
            y_bounds[1] - final_xy[1],
        )
        candidates.append((float(margin), index, direction))
    if not candidates:
        raise RuntimeError("neither ordered cable endpoint has a legal fixed pull action")

    _, endpoint_index, direction = max(candidates, key=lambda item: (item[0], -item[1]))
    start = np.array([values[endpoint_index, 0], values[endpoint_index, 1], pick_z])
    preload_end = start.copy()
    preload_end[:2] += direction * preload_distance
    main_end = preload_end.copy()
    main_end[:2] += direction * main_distance
    return [
        {
            "name": "preload",
            "primitive": "pick_place",
            "pose0": _pose(start),
            "pose1": _pose(preload_end),
        },
        {
            "name": "main_pull",
            "primitive": "pick_place",
            "pose0": _pose(preload_end),
            "pose1": _pose(main_end),
        },
    ]


def _environment_action(action: Dict[str, Any]) -> Dict[str, Any]:
    def tuple_pose(value: Dict[str, Any]):
        return (tuple(value["position"]), tuple(value["quaternion"]))

    return {
        "primitive": action["primitive"],
        "params": {
            "pose0": tuple_pose(action["pose0"]),
            "pose1": tuple_pose(action["pose1"]),
        },
    }


def _trace_arrays(frames: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    if not frames:
        raise RuntimeError("rollout produced an empty CCDA trace")
    arrays = {}
    for key in TRACE_KEYS:
        values = [frame[key] for frame in frames]
        if key == "phase":
            arrays[key] = np.asarray(values, dtype="U16")
        elif key in {"physics_step", "contact_active_beads"}:
            arrays[key] = np.asarray(values, dtype=np.int64)
        else:
            arrays[key] = np.asarray(values, dtype=np.float64)
    return arrays


def _configure_task_environment(config: Dict[str, Any], condition: str, seed: int, group_id: str) -> None:
    friction = config["friction"]
    values = {
        "CCDA_HIDDEN_CONDITION": condition,
        "CCDA_VISIBLE_SEED": str(seed),
        "CCDA_PAIR_GROUP": group_id,
        "CCDA_FRICTION_CENTER_RATIO": friction["center_ratio"],
        "CCDA_FRICTION_SELECTED_COUNT": friction["selected_count"],
        "CCDA_FRICTION_PATCH_RADIUS": friction["patch_radius"],
        "CCDA_FRICTION_VISCOUS_GAIN": friction["viscous_gain"],
        "CCDA_FRICTION_COULOMB_FORCE": friction["coulomb_force"],
        "CCDA_FRICTION_MAX_FORCE": friction["max_force"],
        "CCDA_FRICTION_SPEED_EPSILON": friction["speed_epsilon"],
        "CCDA_FRICTION_CONTACT_HEIGHT": friction["contact_height"],
        "CCDA_TRACE_STRIDE": config["trace_stride"],
    }
    for name, value in values.items():
        os.environ[name] = str(value)


def run_rollout(
    config: Dict[str, Any],
    condition: str,
    seed: int,
    group_id: str,
    action_script: Optional[List[Dict[str, Any]]],
    disp: bool,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any], List[Dict[str, Any]]]:
    random.seed(seed)
    np.random.seed(seed)
    _configure_task_environment(config, condition, seed, group_id)
    env = None
    try:
        env = Environment(disp=disp, hz=int(config["hz"]))
        task = tasks.names[config["task_name"]]()
        env.reset(task)
        if not task.ccda_privileged_state():
            raise RuntimeError("hidden friction was not armed after visible settle")

        task.set_ccda_phase("no_action")
        wait_physics_steps(task, int(config["no_action_steps"]))
        stable_beads = _bead_positions(task)
        if condition == "free" and action_script is None:
            action_script = generate_action_script(config, stable_beads)
        elif action_script is None:
            raise ValueError("hidden rollout requires the free-condition action script")
        action_script = json.loads(json.dumps(action_script, sort_keys=True))
        action_hash = canonical_json_sha256(action_script)

        task.set_ccda_phase("preload")
        env.step(_environment_action(action_script[0]))
        task.set_ccda_phase("main_pull")
        env.step(_environment_action(action_script[1]))
        task.set_ccda_phase("post_main")
        wait_physics_steps(task, int(config["post_main_steps"]))

        if env._ccda_physics_hook_error:
            raise RuntimeError(env._ccda_physics_hook_error)
        trace = _trace_arrays(task.ccda_trace())
        metadata = {
            "seed": int(seed),
            "group_id": group_id,
            "condition": condition,
            "action_script": action_script,
            "action_hash": action_hash,
            "config": config,
            "config_hash": canonical_json_sha256(config),
            "privileged_state": task.ccda_privileged_state(),
            "main_repository_sha": _git_sha(REPO_ROOT),
            "submodule_sha": _git_sha(SUBMODULE_ROOT),
            "physics_steps_observed": int(task.physics_step_count()),
        }
        return trace, metadata, action_script
    finally:
        if env is not None:
            env.stop()


def _aggregate(groups: List[Dict[str, Any]]) -> Dict[str, Any]:
    numeric_keys = sorted(
        key
        for key, value in groups[0]["metrics"].items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    )
    return {
        key: {
            "mean": float(np.mean([group["metrics"][key] for group in groups])),
            "std": float(np.std([group["metrics"][key] for group in groups])),
        }
        for key in numeric_keys
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--groups", type=int)
    parser.add_argument("--seed-start", type=int)
    parser.add_argument("--disp", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    output = Path(args.output).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    groups_count = int(args.groups if args.groups is not None else config["groups"])
    seed_start = int(args.seed_start if args.seed_start is not None else config["seed_start"])
    if groups_count <= 0:
        raise ValueError("groups must be positive")
    raw_dir = output / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    group_results = []
    for offset in range(groups_count):
        seed = seed_start + offset
        group_id = f"hf_{seed:06d}"
        free_trace, free_meta, action_script = run_rollout(
            config, "free", seed, group_id, None, args.disp
        )
        hidden_trace, hidden_meta, hidden_action = run_rollout(
            config, "hidden_high_friction", seed, group_id, action_script, args.disp
        )
        if canonical_json_sha256(action_script) != canonical_json_sha256(hidden_action):
            raise RuntimeError("hidden rollout changed the paired action script")

        for condition, trace, metadata in (
            ("free", free_trace, free_meta),
            ("hidden_high_friction", hidden_trace, hidden_meta),
        ):
            np.savez_compressed(raw_dir / f"{group_id}_{condition}.npz", **trace)
            _write_json(raw_dir / f"{group_id}_{condition}.json", metadata)
        metrics = compute_pair_metrics(
            free_trace,
            hidden_trace,
            free_meta,
            hidden_meta,
            hz=float(config["hz"]),
            trace_stride=int(config["trace_stride"]),
        )
        pair = {
            "group_id": group_id,
            "seed": seed,
            "action_hash": free_meta["action_hash"],
            "metrics": metrics,
        }
        _write_json(raw_dir / f"{group_id}_pair.json", pair)
        group_results.append(pair)
        print(f"completed {group_id}: {json.dumps(metrics, sort_keys=True)}", flush=True)

    top_level = {
        "task_name": config["task_name"],
        "pilot_only": True,
        "groups": group_results,
        "aggregate": _aggregate(group_results),
        "config_path": str(config_path.relative_to(REPO_ROOT)),
        "config_hash": canonical_json_sha256(config),
        "main_repository_sha": _git_sha(REPO_ROOT),
        "submodule_sha": _git_sha(SUBMODULE_ROOT),
    }
    _write_json(output / "metrics.json", top_level)
    summary = [
        "# Phase 0A Hidden-Friction Cable Pilot",
        "",
        f"- Groups: {groups_count}",
        f"- Seeds: {seed_start}..{seed_start + groups_count - 1}",
        f"- Config hash: `{top_level['config_hash']}`",
        f"- Main repository: `{top_level['main_repository_sha']}`",
        f"- Submodule: `{top_level['submodule_sha']}`",
        "- Command: `python scripts/experiment2/phase0/run_hidden_friction_pairs.py --config configs/experiment2/phase0/hidden_friction_cable.json --output reports/experiment2/phase0_hidden_friction --groups 3`",
        "- Status: pilot evidence only; no statistical or formal CCDA claim.",
        "",
        "Observed problems are recorded in `metrics.json` and the final Phase 0A report.",
    ]
    (output / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
