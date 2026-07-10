#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

import phase3_12d_r1_common as common


def check_source(root: Path) -> Dict[str, Any]:
    env_path = root / "external/deformable-ravens/ravens/environment.py"
    task_path = (
        root
        / "external/deformable-ravens/ravens/tasks/ccda_hidden_contact_cable.py"
    )
    env_text = env_path.read_text(errors="replace")
    task_text = task_path.read_text(errors="replace")

    return {
        "environment_has_physics_hook_dispatch": (
            "physics_step_hook" in env_text
            and "self.task" in env_text
            and "p.stepSimulation()" in env_text
        ),
        "task_has_physics_step_hook": "def physics_step_hook" in task_text,
        "task_tracks_physics_steps": "_ccda_physics_step_count" in task_text,
        "task_logs_release_physics_step": (
            "breakaway_release_physics_step" in task_text
        ),
        "task_still_checks_breakaway_from_reward": (
            "_maybe_update_breakaway" in task_text
            and "def reward" in task_text
        ),
        "hidden_geometry_is_collision_only": (
            "baseVisualShapeIndex=-1" in task_text
            or "baseVisualShapeIndex = -1" in task_text
        ),
        "task_uses_world_point_constraint": (
            "childBodyUniqueId=-1" in task_text
            or "childBodyUniqueId = -1" in task_text
        ),
        "base_cable_is_bead_chain": (
            "JOINT_POINT2POINT"
            in (
                root
                / "external/deformable-ravens/ravens/tasks/defs_cables.py"
            ).read_text(errors="replace")
        ),
    }


def require_runtime(root: Path):
    common.add_repo_paths(root)
    import phase3_policy_rollout as p34

    p34.set_selected_recoverable_env_defaults()
    tasks, Environment = p34.require_runtime(root)
    return p34, tasks, Environment


def reset_task(
    root: Path,
    p34: Any,
    tasks: Any,
    Environment: Any,
    *,
    condition: str,
    seed: int,
):
    import phase3_12c_matched_reset_common as p12c

    common.seed_everything(seed)
    common.set_ccda_environment(condition, seed)
    task = tasks.names["hidden-contact-cable-line"]()
    task.mode = "train"
    env = Environment(disp=False, hz=240)
    env.t_lim = 15.0
    info, reset_meta = p12c.deterministic_reset(
        env,
        task,
        min_settle_steps=540,
        max_settle_steps=2400,
        static_checks_required=8,
        static_check_interval=10,
    )
    common.pause_and_drain(env)
    return task, env, info, reset_meta


def close_env(p34: Any, env: Any) -> None:
    try:
        common.pause_and_drain(env)
    except Exception:
        pass
    try:
        p34.close_env_safely(env)
    except Exception:
        try:
            env.stop()
        except Exception:
            pass


def visible_pair_probe(
    root: Path,
    p34: Any,
    tasks: Any,
    Environment: Any,
    seeds: List[int],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        captures: Dict[str, np.ndarray] = {}
        metrics: Dict[str, Dict[str, float]] = {}
        for condition in ("free", "hidden_breakaway_pin"):
            task = env = None
            try:
                task, env, _, _ = reset_task(
                    root,
                    p34,
                    tasks,
                    Environment,
                    condition=condition,
                    seed=seed,
                )
                captures[condition] = common.ordered_bead_xy(task)
                metrics[condition] = common.geometry_metrics(task)
            finally:
                if env is not None:
                    close_env(p34, env)

        difference = (
            captures["hidden_breakaway_pin"] - captures["free"]
        )
        rows.append(
            {
                "seed": int(seed),
                "bead_xy_max_abs": float(np.max(np.abs(difference))),
                "bead_xy_mae": float(np.mean(np.abs(difference))),
                "fraction_diff": abs(
                    metrics["hidden_breakaway_pin"]["fraction"]
                    - metrics["free"]["fraction"]
                ),
                "curve_diff": abs(
                    metrics["hidden_breakaway_pin"]["curve"]
                    - metrics["free"]["curve"]
                ),
            }
        )
    return rows


def breakaway_physics_probe(
    root: Path,
    p34: Any,
    tasks: Any,
    Environment: Any,
    seed: int,
    force: float,
    max_steps: int,
) -> Dict[str, Any]:
    import pybullet as p

    task = env = None
    try:
        task, env, _, _ = reset_task(
            root,
            p34,
            tasks,
            Environment,
            condition="hidden_breakaway_pin",
            seed=seed,
        )
        bead_id = int(task._breakaway_bead_id)
        anchor = np.asarray(task._breakaway_anchor_pos, dtype=np.float64)
        release_before_reward = False
        release_step = None
        max_disp = 0.0

        # Do not call task.reward() in this loop. The release must be driven by
        # the per-physics-step hook rather than by action-boundary reward code.
        for index in range(int(max_steps)):
            p.applyExternalForce(
                bead_id,
                -1,
                forceObj=[float(force), 0.0, 0.0],
                posObj=[0.0, 0.0, 0.0],
                flags=p.WORLD_FRAME,
            )
            p.stepSimulation()
            hook = getattr(task, "physics_step_hook", None)
            if not callable(hook):
                raise RuntimeError("Task has no physics_step_hook.")
            hook()
            position = np.asarray(
                p.getBasePositionAndOrientation(bead_id)[0],
                dtype=np.float64,
            )
            displacement = float(
                np.linalg.norm((position - anchor)[:2])
            )
            max_disp = max(max_disp, displacement)
            if bool(task._breakaway_released):
                release_before_reward = True
                release_step = index + 1
                break

        return {
            "seed": int(seed),
            "force": float(force),
            "max_steps": int(max_steps),
            "released_before_reward": bool(release_before_reward),
            "release_loop_step": release_step,
            "task_release_physics_step": getattr(
                task, "_breakaway_release_physics_step", None
            ),
            "max_displacement": float(max_disp),
            "threshold": float(
                os.environ.get("CCDA_BREAKAWAY_DISP", "0.045")
            ),
            "hook_error": repr(
                getattr(env, "_ccda_physics_hook_error", None)
            ),
        }
    finally:
        if env is not None:
            close_env(p34, env)



def breakaway_thread_dispatch_probe(
    root: Path,
    p34: Any,
    tasks: Any,
    Environment: Any,
    seed: int,
    force: float,
    max_iterations: int,
) -> Dict[str, Any]:
    """Verify Environment.step_simulation actually invokes the task hook."""
    import pybullet as p

    task = env = None
    try:
        task, env, _, _ = reset_task(
            root, p34, tasks, Environment,
            condition="hidden_breakaway_pin", seed=seed,
        )
        bead_id = int(task._breakaway_bead_id)
        env.start()
        released = False
        iteration = None
        for index in range(int(max_iterations)):
            p.applyExternalForce(
                bead_id, -1,
                forceObj=[float(force), 0.0, 0.0],
                posObj=[0.0, 0.0, 0.0],
                flags=p.WORLD_FRAME,
            )
            time.sleep(0.002)
            if bool(task._breakaway_released):
                released = True
                iteration = index + 1
                break
        common.pause_and_drain(env)
        return {
            "seed": int(seed),
            "force": float(force),
            "released_before_reward": bool(released),
            "release_iteration": iteration,
            "task_release_physics_step": getattr(
                task, "_breakaway_release_physics_step", None
            ),
            "hook_error": repr(
                getattr(env, "_ccda_physics_hook_error", None)
            ),
        }
    finally:
        if env is not None:
            close_env(p34, env)


def snapshot_constraint_probe(
    root: Path,
    p34: Any,
    tasks: Any,
    Environment: Any,
    seed: int,
) -> Dict[str, Any]:
    import pybullet as p

    task = env = None
    state_id = None
    try:
        task, env, _, _ = reset_task(
            root,
            p34,
            tasks,
            Environment,
            condition="hidden_breakaway_pin",
            seed=seed,
        )
        original_id = int(task._breakaway_constraint_id)
        state_id = int(p.saveState())
        p.removeConstraint(original_id)
        p.restoreState(state_id)

        restored_by_pybullet = True
        try:
            p.getConstraintInfo(original_id)
        except Exception:
            restored_by_pybullet = False

        # Validate the fallback used by the query-local snapshot manager.
        task._breakaway_constraint_id = original_id
        fallback = common.ensure_breakaway_constraint(task)
        active_id = int(task._breakaway_constraint_id)
        p.getConstraintInfo(active_id)
        return {
            "seed": int(seed),
            "original_constraint_id": original_id,
            "restored_by_pybullet": bool(restored_by_pybullet),
            "fallback": fallback,
            "active_constraint_id": active_id,
            "pass": True,
        }
    finally:
        if state_id is not None:
            try:
                p.removeState(state_id)
            except Exception:
                pass
        if env is not None:
            close_env(p34, env)


def goal_action_probe(
    root: Path,
    p34: Any,
    tasks: Any,
    Environment: Any,
    seed: int,
) -> Dict[str, Any]:
    task = env = None
    try:
        task, env, info, _ = reset_task(
            root,
            p34,
            tasks,
            Environment,
            condition="free",
            seed=seed,
        )
        before = common.geometry_metrics(task)
        action = common.goal_geometry_action(task)
        env.start()
        try:
            _, reward, done, after_info = env.step(action)
        finally:
            common.pause_and_drain(env)
        after = common.geometry_metrics(task)
        effect = common.dense_effect(before, after)
        return {
            "seed": int(seed),
            "dense_gain": effect["dense_gain"],
            "ordered_gain": effect["ordered_gain"],
            "fraction_gain": effect["fraction_gain"],
            "reward": float(reward),
            "done": bool(done),
            "hook_error": repr(
                getattr(env, "_ccda_physics_hook_error", None)
            ),
        }
    finally:
        if env is not None:
            close_env(p34, env)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--visible-seeds", nargs="+", type=int, default=[312000, 312500])
    parser.add_argument("--breakaway-seed", type=int, default=312000)
    parser.add_argument("--breakaway-force", type=float, default=15.0)
    parser.add_argument("--breakaway-max-steps", type=int, default=2400)
    parser.add_argument("--visible-max-abs-threshold", type=float, default=0.002)
    parser.add_argument("--goal-actionable-threshold", type=float, default=0.003)
    parser.add_argument(
        "--out-json",
        default="reports/phase3_12d_r1_environment_audit_summary.json",
    )
    parser.add_argument(
        "--out-md",
        default="reports/phase3_12d_r1_environment_audit_report.md",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    common.add_repo_paths(root)
    source = check_source(root)
    p34, tasks, Environment = require_runtime(root)
    common.assert_no_forbidden_modules("environment_audit")

    issues: List[Dict[str, Any]] = []

    def issue(level: str, name: str, detail: Any) -> None:
        issues.append({"level": level, "name": name, "detail": detail})

    for key in (
        "environment_has_physics_hook_dispatch",
        "task_has_physics_step_hook",
        "task_tracks_physics_steps",
        "task_logs_release_physics_step",
    ):
        if not source[key]:
            issue("FAIL", key, source)

    if source["task_uses_world_point_constraint"]:
        issue(
            "WARN",
            "breakaway_is_world_anchor_approximation",
            "This is a controlled latent constraint, not a geometrically modeled pin contact.",
        )

    visible = visible_pair_probe(
        root, p34, tasks, Environment, list(args.visible_seeds)
    )
    if any(
        row["bead_xy_max_abs"] > args.visible_max_abs_threshold
        for row in visible
    ):
        issue(
            "FAIL",
            "hidden_condition_changes_initial_visible_geometry_too_much",
            visible,
        )

    breakaway = breakaway_physics_probe(
        root,
        p34,
        tasks,
        Environment,
        args.breakaway_seed,
        args.breakaway_force,
        args.breakaway_max_steps,
    )
    if not breakaway["released_before_reward"]:
        issue(
            "FAIL",
            "breakaway_does_not_release_during_physics",
            breakaway,
        )

    breakaway_thread = breakaway_thread_dispatch_probe(
        root, p34, tasks, Environment, args.breakaway_seed,
        args.breakaway_force, min(args.breakaway_max_steps, 2400),
    )
    if not breakaway_thread["released_before_reward"]:
        issue(
            "FAIL",
            "environment_thread_does_not_dispatch_breakaway_hook",
            breakaway_thread,
        )

    snapshot = snapshot_constraint_probe(
        root, p34, tasks, Environment, args.breakaway_seed
    )

    goal_rows = [
        goal_action_probe(root, p34, tasks, Environment, seed)
        for seed in args.visible_seeds
    ]
    if not any(
        row["dense_gain"] > args.goal_actionable_threshold
        for row in goal_rows
    ):
        issue(
            "WARN",
            "one_step_goal_geometry_action_not_actionable_in_free_control",
            goal_rows,
        )

    has_fail = any(item["level"] == "FAIL" for item in issues)
    has_warn = any(item["level"] == "WARN" for item in issues)
    verdict = "FAIL" if has_fail else ("WARN" if has_warn else "PASS")
    root_cause = (
        "phase312d_r1_environment_semantics_or_task_actionability_failed"
        if has_fail
        else "phase312d_r1_environment_semantics_acceptable_with_approximations"
    )

    payload = {
        "verdict": verdict,
        "root_cause": root_cause,
        "source_checks": source,
        "visible_pair_probe": visible,
        "breakaway_physics_probe": breakaway,
        "breakaway_thread_dispatch_probe": breakaway_thread,
        "snapshot_constraint_probe": snapshot,
        "goal_action_probe": goal_rows,
        "issues": issues,
        "query_local_snapshot_allowed": not has_fail,
        "scope": "environment semantics audit before Phase3.12d-r1",
    }
    common.write_json_atomic(root / args.out_json, payload)

    lines = [
        "# Phase3.12d-r1 Physical Environment / Task Semantics Audit",
        "",
        "## Verdict",
        "",
        "- Verdict: `{}`".format(verdict),
        "- Root cause: `{}`".format(root_cause),
        "- Query-local snapshot allowed: `{}`".format(not has_fail),
        "",
        "## Source checks",
        "",
        "| Check | Result |",
        "|---|---:|",
    ]
    for key, value in source.items():
        lines.append("| `{}` | `{}` |".format(key, value))
    lines += [
        "",
        "## Visible-pair probe",
        "",
        "| Seed | Bead XY max abs | Bead XY MAE | Fraction diff | Curve diff |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in visible:
        lines.append(
            "| {seed} | {bead_xy_max_abs:.8f} | {bead_xy_mae:.8f} | "
            "{fraction_diff:.8f} | {curve_diff:.8f} |".format(**row)
        )
    lines += [
        "",
        "## Breakaway physics probe",
        "",
        "```json",
        json.dumps(breakaway, indent=2, sort_keys=True),
        "```",
        "",
        "## Breakaway thread-dispatch probe",
        "",
        "```json",
        json.dumps(breakaway_thread, indent=2, sort_keys=True),
        "```",
        "",
        "## Snapshot constraint probe",
        "",
        "```json",
        json.dumps(snapshot, indent=2, sort_keys=True),
        "```",
        "",
        "## Goal action probe",
        "",
        "| Seed | Dense gain | Ordered gain | Fraction gain |",
        "|---:|---:|---:|---:|",
    ]
    for row in goal_rows:
        lines.append(
            "| {seed} | {dense_gain:.6f} | {ordered_gain:.6f} | "
            "{fraction_gain:.6f} |".format(**row)
        )
    lines += [
        "",
        "## Issues",
        "",
        "| Level | Name | Detail |",
        "|---|---|---|",
    ]
    if issues:
        for item in issues:
            lines.append(
                "| `{}` | `{}` | {} |".format(
                    item["level"],
                    item["name"],
                    str(item["detail"]).replace("|", "/"),
                )
            )
    else:
        lines.append("| `PASS` | `none` | No blocking issue. |")
    lines += [
        "",
        "## Decision",
        "",
        "- A FAIL blocks Phase3.12d-r1 candidate execution.",
        "- A WARN records modeling approximations but does not invalidate the oracle audit.",
        "- The beaded cable and world-anchor pin are acceptable only as a controlled qualitative benchmark, not as a high-fidelity silicone-cable model.",
    ]
    (root / args.out_md).write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if has_fail:
        raise SystemExit("[Phase3.12d-r1] environment audit failed")


if __name__ == "__main__":
    main()
