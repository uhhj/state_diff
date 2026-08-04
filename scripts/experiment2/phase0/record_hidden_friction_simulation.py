#!/usr/bin/env python3
"""Record real PyBullet videos for paired Hidden-Friction Cable rollouts."""
from __future__ import annotations

import argparse
import json
import random
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

from ravens import Environment, tasks  # noqa: E402
from ravens.ccda_sim_video import CCDASimulationVideoRecorder  # noqa: E402
from scripts.experiment2.phase0.common import canonical_json_sha256  # noqa: E402
from scripts.experiment2.phase0.run_hidden_friction_pairs import (  # noqa: E402
    _bead_positions,
    _configure_task_environment,
    _environment_action,
    generate_action_script,
)
from scripts.experiment2.phase0.simulation_video_utils import compose_side_by_side_video  # noqa: E402


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def wait_and_record(
    env: Environment,
    task: Any,
    delta_steps: int,
    label: str,
    capture_every_steps: int,
    timeout_s: float = 45.0,
) -> None:
    if delta_steps < 0:
        raise ValueError("delta_steps must be non-negative")
    if capture_every_steps <= 0:
        raise ValueError("capture_every_steps must be positive")
    start = int(task.physics_step_count())
    next_capture = start
    wall_start = time.time()
    while int(task.physics_step_count()) - start < delta_steps:
        current = int(task.physics_step_count())
        if current >= next_capture:
            env.record_ccda_frame(label)
            next_capture = current + capture_every_steps
        if time.time() - wall_start > timeout_s:
            raise TimeoutError(f"timed out waiting for {delta_steps} physics steps in {label}")
        time.sleep(0.002)
    env.record_ccda_frame(label)


def run_recorded_condition(
    config: Dict[str, Any],
    condition: str,
    seed: int,
    group_id: str,
    action_script: Optional[List[Dict[str, Any]]],
    output_path: Path,
    disp: bool,
    renderer_override: Optional[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    random.seed(seed)
    np.random.seed(seed)
    _configure_task_environment(config, condition, seed, group_id)
    env = None
    recorder = None
    try:
        env = Environment(disp=disp, hz=int(config["hz"]))
        task = tasks.names[config["task_name"]]()
        env.reset(task)
        video_config = dict(config["simulation_video"])
        if renderer_override is not None:
            video_config["renderer"] = renderer_override
        recorder = CCDASimulationVideoRecorder(
            output_path=output_path,
            task=task,
            condition=condition,
            group_id=group_id,
            seed=seed,
            config=video_config,
        )
        env.set_ccda_video_recorder(recorder)
        env.record_ccda_frame("initial")

        task.set_ccda_phase("no_action")
        wait_and_record(env, task, int(config["no_action_steps"]), "no_action", int(video_config["idle_capture_steps"]))
        stable_beads = _bead_positions(task)
        if condition == "free" and action_script is None:
            action_script = generate_action_script(config, stable_beads)
        elif action_script is None:
            raise ValueError("hidden condition requires the free action script")
        action_script = json.loads(json.dumps(action_script, sort_keys=True))
        action_hash = canonical_json_sha256(action_script)

        task.set_ccda_phase("preload")
        env.record_ccda_frame("preload_start")
        env.step(_environment_action(action_script[0]))
        env.record_ccda_frame("preload_end")

        task.set_ccda_phase("main_pull")
        env.record_ccda_frame("main_pull_start")
        env.step(_environment_action(action_script[1]))
        env.record_ccda_frame("main_pull_end")

        task.set_ccda_phase("post_main")
        wait_and_record(env, task, int(config["post_main_steps"]), "post_main", int(video_config["idle_capture_steps"]))
        if env._ccda_physics_hook_error:
            raise RuntimeError(env._ccda_physics_hook_error)

        video_summary = recorder.close()
        recorder = None
        env.set_ccda_video_recorder(None)
        return action_script, {
            "condition": condition,
            "seed": seed,
            "group_id": group_id,
            "action_hash": action_hash,
            "action_script": action_script,
            "physics_steps_observed": int(task.physics_step_count()),
            "video": video_summary,
        }
    finally:
        if recorder is not None:
            try:
                recorder.close()
            except Exception:
                pass
        if env is not None:
            env.set_ccda_video_recorder(None)
            env.stop()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment2/phase0/hidden_friction_cable.json")
    parser.add_argument("--output", default="reports/experiment2/phase0_hidden_friction/simulation")
    parser.add_argument("--groups", type=int, default=3)
    parser.add_argument("--seed-start", type=int)
    parser.add_argument("--group-id")
    parser.add_argument("--disp", action="store_true")
    parser.add_argument("--renderer", choices=("tiny", "hardware", "auto"))
    args = parser.parse_args()

    config_path = (REPO_ROOT / args.config).resolve()
    output_root = (REPO_ROOT / args.output).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if "simulation_video" not in config:
        raise KeyError("config has no simulation_video section")
    seed_start = int(args.seed_start if args.seed_start is not None else config["seed_start"])
    if args.group_id:
        if not args.group_id.startswith("hf_"):
            raise ValueError("group-id must use hf_XXXXXX form")
        seeds = [int(args.group_id.split("_", 1)[1])]
    else:
        if args.groups <= 0:
            raise ValueError("groups must be positive")
        seeds = [seed_start + offset for offset in range(args.groups)]

    output_root.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in seeds:
        group_id = f"hf_{seed:06d}"
        group_dir = output_root / group_id
        group_dir.mkdir(parents=True, exist_ok=True)
        free_path = group_dir / f"{group_id}_free.mp4"
        hidden_path = group_dir / f"{group_id}_hidden_high_friction.mp4"
        pair_path = group_dir / f"{group_id}_side_by_side.mp4"

        action_script, free_result = run_recorded_condition(config, "free", seed, group_id, None, free_path, args.disp, args.renderer)
        hidden_action, hidden_result = run_recorded_condition(config, "hidden_high_friction", seed, group_id, action_script, hidden_path, args.disp, args.renderer)
        free_hash = canonical_json_sha256(action_script)
        hidden_hash = canonical_json_sha256(hidden_action)
        if free_hash != hidden_hash:
            raise RuntimeError("recorded conditions used different action scripts")

        pair_result = compose_side_by_side_video(
            free_path=free_path,
            hidden_path=hidden_path,
            output_path=pair_path,
            group_id=group_id,
            action_hash=free_hash,
            fps=float(config["simulation_video"]["fps"]),
        )
        group_result = {
            "group_id": group_id,
            "seed": seed,
            "action_hash": free_hash,
            "action_hash_match": True,
            "free": free_result,
            "hidden_high_friction": hidden_result,
            "pair_video": pair_result,
        }
        _write_json(group_dir / f"{group_id}_simulation.json", group_result)
        results.append(group_result)
        print(pair_result["output_path"], flush=True)
        print(pair_result["contact_sheet_path"], flush=True)

    _write_json(
        output_root / "simulation_summary.json",
        {
            "task_name": config["task_name"],
            "config_path": str(config_path.relative_to(REPO_ROOT)),
            "config_hash": canonical_json_sha256(config),
            "groups": results,
            "interpretation": "Qualitative real-simulation visualization only; Phase 0A metrics remain the source for physical and branch auditing.",
        },
    )


if __name__ == "__main__":
    main()
