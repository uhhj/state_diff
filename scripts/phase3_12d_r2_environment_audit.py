#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pybullet as p

import phase3_12c_matched_reset_common as p12c
import phase3_12d_r2_common as common


DEFAULT_SEEDS = [312000, 312001, 312002, 312003, 312500, 312501, 312502, 312503]


def curve_metric(xy: np.ndarray) -> float:
    xy = np.asarray(xy, dtype=np.float64)
    if xy.shape[0] < 3:
        return 0.0
    second = xy[2:] - 2.0 * xy[1:-1] + xy[:-2]
    return float(np.mean(np.linalg.norm(second, axis=1)))


def targets_xy(task: Any) -> np.ndarray:
    values = []
    for bead_id in task.cable_bead_IDs:
        place = task.goal["places"][int(bead_id)][0]
        values.append(place[:2])
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (len(task.cable_bead_IDs), 2):
        raise RuntimeError(f"bad target shape: {result.shape}")
    return result


def ordered_distance(xy: np.ndarray, goal_xy: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(xy - goal_xy, axis=1)))


def chamfer_distance(xy: np.ndarray, goal_xy: np.ndarray) -> float:
    diff = xy[:, None, :] - goal_xy[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    return float(0.5 * (np.mean(np.min(dist, axis=1)) + np.mean(np.min(dist, axis=0))))


def fraction_from_geometry(task: Any, xy: np.ndarray) -> float:
    goal = targets_xy(task)
    # Match the cable-target semantics: each bead is counted when it lies in
    # the thresholded union of target zones. Ordered distance is reported
    # separately and must not replace the task's nearest-target fraction.
    threshold = 3.5 * float(task.radius)
    distances = np.linalg.norm(
        np.asarray(xy, dtype=np.float64)[:, None, :]
        - np.asarray(goal, dtype=np.float64)[None, :, :],
        axis=2,
    )
    return float(np.mean(np.min(distances, axis=1) <= threshold))


def fraction_from_goal(
    xy: np.ndarray, goal_xy: np.ndarray, radius: float
) -> float:
    threshold = 3.5 * float(radius)
    distances = np.linalg.norm(
        np.asarray(xy, dtype=np.float64)[:, None, :]
        - np.asarray(goal_xy, dtype=np.float64)[None, :, :],
        axis=2,
    )
    return float(np.mean(np.min(distances, axis=1) <= threshold))


def geometry_record(task: Any) -> Dict[str, Any]:
    xy = common.ordered_bead_xy(task)
    goal = targets_xy(task)
    return {
        "xy": xy.astype(float).tolist(),
        "ordered_distance": ordered_distance(xy, goal),
        "chamfer_distance": chamfer_distance(xy, goal),
        "fraction": fraction_from_geometry(task, xy),
        "curve": curve_metric(xy),
    }


def goal_geometry_action(task: Any) -> Dict[str, Any]:
    xy = common.ordered_bead_xy(task)
    goal = targets_xy(task)
    distances = np.linalg.norm(xy - goal, axis=1)
    idx = int(np.argmax(distances))
    p0 = (float(xy[idx, 0]), float(xy[idx, 1]), 0.001)
    p1 = (float(goal[idx, 0]), float(goal[idx, 1]), 0.001)
    quat = (0.0, 0.0, 0.0, 1.0)
    return {
        "primitive": "pick_place",
        "params": {"pose0": (p0, quat), "pose1": (p1, quat)},
        "diagnostic_bead_local_index": idx,
    }


def setup_runtime(root: Path):
    for path in [root, root / "scripts", root / "external/deformable-ravens"]:
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import phase3_policy_rollout as p34

    p34.set_selected_recoverable_env_defaults()
    tasks, Environment = p34.require_runtime(root)
    return p34, tasks, Environment


def configure(condition: str, seed: int, pair_suffix: str) -> None:
    common.seed_everything(int(seed))
    os.environ["CCDA_HIDDEN_CONDITION"] = str(condition)
    os.environ["CCDA_VISIBLE_SEED"] = str(int(seed))
    os.environ["CCDA_PAIR_GROUP"] = f"phase3_12d_r2_{pair_suffix}_{int(seed)}"
    os.environ["CCDA_BREAKAWAY_FORCE"] = os.environ.get(
        "CCDA_BREAKAWAY_FORCE", "2.6"
    )
    os.environ["CCDA_BREAKAWAY_DISP"] = os.environ.get(
        "CCDA_BREAKAWAY_DISP", "0.045"
    )
    os.environ["CCDA_BREAKAWAY_BEAD_RATIO"] = os.environ.get(
        "CCDA_BREAKAWAY_BEAD_RATIO", "0.45"
    )
    os.environ["CCDA_BREAKAWAY_DAMPING"] = "0.0"
    os.environ["CCDA_SETTLE_SECONDS"] = "0"
    os.environ["CCDA_POST_ARM_SETTLE_SECONDS"] = "0"


def reset_repaired(
    root: Path,
    p34: Any,
    tasks: Any,
    Environment: Any,
    condition: str,
    seed: int,
    *,
    arm_after_settle: bool,
    post_arm_steps: int,
    args: argparse.Namespace,
):
    configure(condition, seed, "repaired")
    env = Environment(disp=False, hz=240)
    env.t_lim = float(args.motion_timeout)
    task = tasks.names["hidden-contact-cable-line"]()
    task.mode = "train"
    info, meta = common.deterministic_reset_deferred_arm(
        env,
        task,
        min_settle_steps=args.min_settle_steps,
        max_settle_steps=args.max_settle_steps,
        static_checks_required=args.static_checks_required,
        static_check_interval=args.static_check_interval,
        arm_after_settle=arm_after_settle,
        post_arm_steps=post_arm_steps,
        canonicalize_velocity=True,
    )
    return env, task, info, meta


def reset_legacy(
    root: Path,
    tasks: Any,
    Environment: Any,
    condition: str,
    seed: int,
    args: argparse.Namespace,
):
    configure(condition, seed, "legacy")
    os.environ["CCDA_DEFER_HIDDEN_CONTACT_ARMING"] = "0"
    env = Environment(disp=False, hz=240)
    env.t_lim = float(args.motion_timeout)
    task = tasks.names["hidden-contact-cable-line"]()
    task.mode = "train"
    info, meta = p12c.deterministic_reset(
        env,
        task,
        min_settle_steps=args.min_settle_steps,
        max_settle_steps=args.max_settle_steps,
        static_checks_required=args.static_checks_required,
        static_check_interval=args.static_check_interval,
    )
    return env, task, info, meta


def close(p34: Any, env: Any) -> None:
    try:
        env.pause()
    except Exception:
        pass
    try:
        p34.close_env_safely(env)
    except Exception:
        try:
            env.stop()
        except Exception:
            pass


def _capture_repaired_reset(
    root: Path,
    p34: Any,
    tasks: Any,
    Environment: Any,
    condition: str,
    seed: int,
    *,
    arm_after_settle: bool,
    post_arm_steps: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    env = None
    try:
        env, task, info, meta = reset_repaired(
            root,
            p34,
            tasks,
            Environment,
            condition,
            seed,
            arm_after_settle=arm_after_settle,
            post_arm_steps=post_arm_steps,
            args=args,
        )
        xy = common.ordered_bead_xy(task).astype(np.float64)
        goal = targets_xy(task).astype(np.float64)
        return {
            "condition": condition,
            "seed": int(seed),
            "xy": xy,
            "goal_xy": goal,
            "radius": float(task.radius),
            "fraction": fraction_from_goal(xy, goal, float(task.radius)),
            "curve": curve_metric(xy),
            "meta": meta,
            "hidden_body_count": len(getattr(task, "hidden_body_ids", [])),
            "hidden_constraint_count": len(
                getattr(task, "hidden_constraint_ids", [])
            ),
            "hidden_contact_applied": bool(
                getattr(task, "hidden_contact_applied", False)
            ),
        }
    finally:
        if env is not None:
            close(p34, env)


def _capture_legacy_reset(
    root: Path,
    p34: Any,
    tasks: Any,
    Environment: Any,
    condition: str,
    seed: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    env = None
    try:
        env, task, info, meta = reset_legacy(
            root, tasks, Environment, condition, seed, args
        )
        xy = common.ordered_bead_xy(task).astype(np.float64)
        goal = targets_xy(task).astype(np.float64)
        return {
            "condition": condition,
            "seed": int(seed),
            "xy": xy,
            "goal_xy": goal,
            "radius": float(task.radius),
            "fraction": fraction_from_goal(xy, goal, float(task.radius)),
            "curve": curve_metric(xy),
            "meta": meta,
        }
    finally:
        if env is not None:
            close(p34, env)


def visible_pair_probe(
    root: Path,
    p34: Any,
    tasks: Any,
    Environment: Any,
    seed: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    # DeformableRavens uses a global PyBullet connection. Never keep multiple
    # Environment instances alive while creating another one; each capture is
    # fully materialized and closed before the next reset.
    free = _capture_repaired_reset(
        root, p34, tasks, Environment, "free", seed,
        arm_after_settle=True, post_arm_steps=0, args=args,
    )
    unarmed = _capture_repaired_reset(
        root, p34, tasks, Environment, "hidden_breakaway_pin", seed,
        arm_after_settle=False, post_arm_steps=0, args=args,
    )
    repaired0 = _capture_repaired_reset(
        root, p34, tasks, Environment, "hidden_breakaway_pin", seed,
        arm_after_settle=True, post_arm_steps=0, args=args,
    )
    legacy = _capture_legacy_reset(
        root, p34, tasks, Environment, "hidden_breakaway_pin", seed, args
    )

    free_pre = np.asarray(free["meta"]["pre_arm_xy"], dtype=np.float64)
    hidden_pre = np.asarray(unarmed["meta"]["pre_arm_xy"], dtype=np.float64)
    repaired_post0 = np.asarray(
        repaired0["meta"]["post_arm_xy"], dtype=np.float64
    )
    legacy_xy = np.asarray(legacy["xy"], dtype=np.float64)

    goal_parity = common.difference_stats(
        np.asarray(unarmed["goal_xy"]), np.asarray(free["goal_xy"])
    )
    pre_parity = common.difference_stats(hidden_pre, free_pre)
    immediate_jump = common.difference_stats(repaired_post0, free_pre)
    legacy_leak = common.difference_stats(legacy_xy, free_pre)

    drift: Dict[str, Any] = {}
    for step_count in args.drift_steps:
        capture = _capture_repaired_reset(
            root, p34, tasks, Environment, "hidden_breakaway_pin", seed,
            arm_after_settle=True, post_arm_steps=int(step_count), args=args,
        )
        xy = np.asarray(capture["meta"]["post_arm_xy"], dtype=np.float64)
        drift[str(step_count)] = {
            **common.difference_stats(xy, free_pre),
            "fraction_diff": abs(
                fraction_from_goal(xy, free["goal_xy"], free["radius"])
                - free["fraction"]
            ),
            "curve_diff": abs(curve_metric(xy) - free["curve"]),
            "breakaway_released": bool(
                capture["meta"]["breakaway_released"]
            ),
            "anchor_error": float(capture["meta"]["arm_anchor_error"]),
            "hook_error": capture["meta"]["hook_error"],
        }

    return {
        "seed": int(seed),
        "goal_parity": goal_parity,
        "pre_arm_parity": {
            **pre_parity,
            "fraction_diff": abs(unarmed["fraction"] - free["fraction"]),
            "curve_diff": abs(unarmed["curve"] - free["curve"]),
        },
        "immediate_post_arm": {
            **immediate_jump,
            "fraction_diff": abs(
                fraction_from_goal(
                    repaired_post0, free["goal_xy"], free["radius"]
                )
                - free["fraction"]
            ),
            "curve_diff": abs(curve_metric(repaired_post0) - free["curve"]),
            "anchor_error": float(repaired0["meta"]["arm_anchor_error"]),
            "arm_result": repaired0["meta"]["arm_result"],
            "breakaway_released": bool(
                repaired0["meta"]["breakaway_released"]
            ),
            "hidden_body_count": repaired0["hidden_body_count"],
            "hidden_constraint_count": repaired0["hidden_constraint_count"],
        },
        "post_arm_drift": drift,
        "legacy_pre_settle_leak": {
            **legacy_leak,
            "fraction_diff": abs(legacy["fraction"] - free["fraction"]),
            "curve_diff": abs(legacy["curve"] - free["curve"]),
        },
        "free_settle_steps": int(free["meta"]["settle_steps_used"]),
        "hidden_settle_steps": int(unarmed["meta"]["settle_steps_used"]),
    }


def force_trajectory(
    root: Path,
    p34: Any,
    tasks: Any,
    Environment: Any,
    condition: str,
    seed: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    env = None
    try:
        env, task, _, meta = reset_repaired(
            root, p34, tasks, Environment, condition, seed,
            arm_after_settle=True, post_arm_steps=0, args=args,
        )
        n = len(task.cable_bead_IDs)
        ratio = float(os.environ.get("CCDA_BREAKAWAY_BEAD_RATIO", "0.45"))
        idx = int(np.clip(round(ratio * (n - 1)), 0, n - 1))
        bead_id = int(task.cable_bead_IDs[idx])
        start = common.ordered_bead_xy(task)
        trajectory = [start[idx].astype(float).tolist()]
        released_at = None
        for step_idx in range(int(args.force_steps)):
            pos = p.getBasePositionAndOrientation(bead_id)[0]
            common.direct_physics_step(
                env,
                task,
                dispatch_hook=True,
                external_force=(
                    bead_id,
                    (float(args.force_x), float(args.force_y), 0.0),
                    pos,
                ),
            )
            trajectory.append(common.ordered_bead_xy(task)[idx].astype(float).tolist())
            if bool(getattr(task, "_breakaway_released", False)) and released_at is None:
                released_at = step_idx + 1
        final = common.ordered_bead_xy(task)
        return {
            "condition": condition,
            "seed": int(seed),
            "bead_local_index": idx,
            "trajectory": trajectory,
            "selected_bead_displacement": float(np.linalg.norm(final[idx] - start[idx])),
            "full_cable_change": common.difference_stats(final, start),
            "breakaway_released": bool(getattr(task, "_breakaway_released", False)),
            "release_loop_step": released_at,
            "release_physics_step": getattr(task, "_breakaway_release_physics_step", None),
            "max_disp_seen": float(getattr(task, "_breakaway_max_disp_seen", 0.0)),
            "hook_error": str(getattr(env, "_ccda_physics_hook_error", None)),
            "initial_meta": meta,
        }
    finally:
        if env is not None:
            close(p34, env)


def causal_force_probe(
    root: Path,
    p34: Any,
    tasks: Any,
    Environment: Any,
    seed: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    free = force_trajectory(root, p34, tasks, Environment, "free", seed, args)
    hidden = force_trajectory(
        root, p34, tasks, Environment, "hidden_breakaway_pin", seed, args
    )
    free_traj = np.asarray(free["trajectory"], dtype=np.float64)
    hidden_traj = np.asarray(hidden["trajectory"], dtype=np.float64)
    length = min(len(free_traj), len(hidden_traj))
    divergence = np.linalg.norm(free_traj[:length] - hidden_traj[:length], axis=1)
    return {
        "seed": int(seed),
        "free": {k: v for k, v in free.items() if k not in {"trajectory", "initial_meta"}},
        "hidden": {k: v for k, v in hidden.items() if k not in {"trajectory", "initial_meta"}},
        "max_trajectory_divergence": float(np.max(divergence)),
        "mean_trajectory_divergence": float(np.mean(divergence)),
        "hidden_released": bool(hidden["breakaway_released"]),
        "hook_error_free": free["hook_error"],
        "hook_error_hidden": hidden["hook_error"],
    }


def goal_action_probe(
    root: Path,
    p34: Any,
    tasks: Any,
    Environment: Any,
    seed: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    env = None
    try:
        env, task, info, _ = reset_repaired(
            root, p34, tasks, Environment, "free", seed,
            arm_after_settle=True, post_arm_steps=0, args=args,
        )
        before = geometry_record(task)
        action = goal_geometry_action(task)
        diagnostic_idx = int(action.pop("diagnostic_bead_local_index"))
        try:
            env.start()
            _, reward, done, info = env.step(action)
        finally:
            env.pause()
        after = geometry_record(task)
        return {
            "seed": int(seed),
            "diagnostic_bead_local_index": diagnostic_idx,
            "dense_gain": float(before["chamfer_distance"] - after["chamfer_distance"]),
            "ordered_gain": float(before["ordered_distance"] - after["ordered_distance"]),
            "fraction_gain": float(after["fraction"] - before["fraction"]),
            "reward": float(reward),
            "done": bool(done),
            "hook_error": str(getattr(env, "_ccda_physics_hook_error", None)),
        }
    finally:
        if env is not None:
            close(p34, env)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--goal-seeds", nargs="+", type=int, default=[312000, 312500])
    parser.add_argument("--drift-steps", nargs="+", type=int, default=[1, 5, 20])
    parser.add_argument("--min-settle-steps", type=int, default=540)
    parser.add_argument("--max-settle-steps", type=int, default=2400)
    parser.add_argument("--static-checks-required", type=int, default=8)
    parser.add_argument("--static-check-interval", type=int, default=10)
    parser.add_argument("--motion-timeout", type=float, default=15.0)
    parser.add_argument("--force-x", type=float, default=15.0)
    parser.add_argument("--force-y", type=float, default=0.0)
    parser.add_argument("--force-steps", type=int, default=240)
    parser.add_argument("--pre-arm-max", type=float, default=1e-7)
    parser.add_argument("--pre-arm-mae", type=float, default=1e-8)
    parser.add_argument("--immediate-max", type=float, default=1e-6)
    parser.add_argument("--immediate-mae", type=float, default=1e-7)
    parser.add_argument("--drift-max", type=float, default=1e-4)
    parser.add_argument("--drift-mae", type=float, default=1e-5)
    parser.add_argument("--fraction-threshold", type=float, default=1e-9)
    parser.add_argument("--curve-threshold", type=float, default=1e-5)
    parser.add_argument("--anchor-threshold", type=float, default=1e-7)
    parser.add_argument("--causal-divergence-threshold", type=float, default=0.003)
    parser.add_argument("--goal-actionable-threshold", type=float, default=0.003)
    parser.add_argument(
        "--out-json", default="reports/phase3_12d_r2_environment_audit_summary.json"
    )
    parser.add_argument(
        "--out-md", default="reports/phase3_12d_r2_environment_audit_report.md"
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    p34, tasks, Environment = setup_runtime(root)

    visible = [
        visible_pair_probe(root, p34, tasks, Environment, seed, args)
        for seed in args.seeds
    ]
    force = [
        causal_force_probe(root, p34, tasks, Environment, seed, args)
        for seed in args.goal_seeds
    ]
    goal = [
        goal_action_probe(root, p34, tasks, Environment, seed, args)
        for seed in args.goal_seeds
    ]

    issues: List[Dict[str, Any]] = []

    def fail(name: str, detail: Any) -> None:
        issues.append({"level": "FAIL", "name": name, "detail": detail})

    def warn(name: str, detail: Any) -> None:
        issues.append({"level": "WARN", "name": name, "detail": detail})

    for row in visible:
        goal_parity = row["goal_parity"]
        if goal_parity["max_abs"] > 1e-12 or goal_parity["mae"] > 1e-13:
            fail("paired_goal_geometry_mismatch", {"seed": row["seed"], **goal_parity})
        pre = row["pre_arm_parity"]
        if pre["max_abs"] > args.pre_arm_max or pre["mae"] > args.pre_arm_mae:
            fail("pre_arm_visible_parity_failed", {"seed": row["seed"], **pre})
        immediate = row["immediate_post_arm"]
        if immediate["hidden_body_count"] != 0:
            warn(
                "breakaway_condition_unexpectedly_created_hidden_body",
                {"seed": row["seed"], "count": immediate["hidden_body_count"]},
            )
        if immediate["hidden_constraint_count"] != 1:
            fail(
                "breakaway_constraint_count_invalid",
                {"seed": row["seed"], "count": immediate["hidden_constraint_count"]},
            )
        if (
            immediate["max_abs"] > args.immediate_max
            or immediate["mae"] > args.immediate_mae
            or immediate["fraction_diff"] > args.fraction_threshold
            or immediate["curve_diff"] > args.curve_threshold
            or immediate["anchor_error"] > args.anchor_threshold
            or immediate["breakaway_released"]
        ):
            fail("immediate_zero_offset_arm_failed", {"seed": row["seed"], **immediate})
        last_key = str(max(args.drift_steps))
        drift = row["post_arm_drift"][last_key]
        if (
            drift["max_abs"] > args.drift_max
            or drift["mae"] > args.drift_mae
            or drift["fraction_diff"] > args.fraction_threshold
            or drift["curve_diff"] > args.curve_threshold
            or drift["breakaway_released"]
            or drift["hook_error"] not in ("None", "", None)
        ):
            fail("short_horizon_post_arm_drift_failed", {"seed": row["seed"], **drift})
        legacy = row["legacy_pre_settle_leak"]
        if legacy["max_abs"] <= args.drift_max:
            warn("legacy_pre_settle_leak_not_reproduced", {"seed": row["seed"], **legacy})

    for row in force:
        if not row["hidden_released"]:
            fail("controlled_force_did_not_release_breakaway", row)
        if row["max_trajectory_divergence"] <= args.causal_divergence_threshold:
            fail("latent_constraint_not_causally_effective", row)
        if row["hook_error_free"] not in ("None", "", None):
            fail("free_force_probe_hook_error", row)
        if row["hook_error_hidden"] not in ("None", "", None):
            fail("hidden_force_probe_hook_error", row)

    for row in goal:
        if row["dense_gain"] <= args.goal_actionable_threshold:
            fail("free_goal_action_not_actionable", row)
        if row["hook_error"] not in ("None", "", None):
            fail("goal_action_hook_error", row)

    has_fail = any(item["level"] == "FAIL" for item in issues)
    verdict = "FAIL" if has_fail else ("WARN" if issues else "PASS")
    if has_fail:
        root_cause = "phase312d_r2_paired_visible_arming_or_causal_semantics_failed"
    else:
        root_cause = "phase312d_r2_paired_visible_arming_passed"

    payload = {
        "verdict": verdict,
        "root_cause": root_cause,
        "query_local_snapshot_allowed": not has_fail,
        "visible_pair_probe": visible,
        "causal_force_probe": force,
        "goal_action_probe": goal,
        "thresholds": {
            "pre_arm_max": args.pre_arm_max,
            "pre_arm_mae": args.pre_arm_mae,
            "immediate_max": args.immediate_max,
            "immediate_mae": args.immediate_mae,
            "drift_max": args.drift_max,
            "drift_mae": args.drift_mae,
            "fraction_threshold": args.fraction_threshold,
            "curve_threshold": args.curve_threshold,
            "anchor_threshold": args.anchor_threshold,
            "causal_divergence_threshold": args.causal_divergence_threshold,
            "goal_actionable_threshold": args.goal_actionable_threshold,
        },
        "issues": issues,
        "decision": (
            "Run query-local candidate smoke only when this report has no FAIL."
        ),
        "scope": common.R2_SCOPE,
    }

    out_json = root / args.out_json
    out_md = root / args.out_md
    common.save_json(out_json, payload)

    lines = [
        "# Phase3.12d-r2 Paired-Visible Latent Arming Audit",
        "",
        "## Verdict",
        "",
        f"- Verdict: `{verdict}`",
        f"- Root cause: `{root_cause}`",
        f"- Query-local snapshot allowed: `{not has_fail}`",
        "",
        "## Visible parity / arming",
        "",
        "| Seed | Pre max | Pre MAE | Immediate max | 20-step max | Anchor error | Legacy max |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    last_key = str(max(args.drift_steps))
    for row in visible:
        lines.append(
            f"| {row['seed']} | {row['pre_arm_parity']['max_abs']:.8g} | "
            f"{row['pre_arm_parity']['mae']:.8g} | "
            f"{row['immediate_post_arm']['max_abs']:.8g} | "
            f"{row['post_arm_drift'][last_key]['max_abs']:.8g} | "
            f"{row['immediate_post_arm']['anchor_error']:.8g} | "
            f"{row['legacy_pre_settle_leak']['max_abs']:.8g} |"
        )
    lines += [
        "",
        "## Controlled causal probe",
        "",
        "| Seed | Max trajectory divergence | Hidden released | Release step | Hook errors |",
        "|---:|---:|---:|---:|---|",
    ]
    for row in force:
        lines.append(
            f"| {row['seed']} | {row['max_trajectory_divergence']:.6f} | "
            f"{row['hidden_released']} | {row['hidden']['release_physics_step']} | "
            f"{row['hook_error_free']} / {row['hook_error_hidden']} |"
        )
    lines += [
        "",
        "## Free goal-actionability probe",
        "",
        "| Seed | Dense gain | Ordered gain | Fraction gain |",
        "|---:|---:|---:|---:|",
    ]
    for row in goal:
        lines.append(
            f"| {row['seed']} | {row['dense_gain']:.6f} | "
            f"{row['ordered_gain']:.6f} | {row['fraction_gain']:.6f} |"
        )
    lines += ["", "## Issues", "", "| Level | Name | Detail |", "|---|---|---|"]
    if issues:
        for item in issues:
            lines.append(
                f"| `{item['level']}` | `{item['name']}` | "
                f"{str(item['detail']).replace('|', '/')} |"
            )
    else:
        lines.append("| `PASS` | `none` | All hard gates passed. |")
    lines += [
        "",
        "## Decision",
        "",
        "- Any FAIL blocks all candidate smoke and full matrices.",
        "- PASS means initial visible parity, zero-offset arming, short drift, causal divergence, breakaway release, and free actionability all hold.",
        "- No training, Phase4, or CPS was run.",
    ]
    out_md.write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))

    if has_fail:
        raise SystemExit("[Phase3.12d-r2][FAIL] environment audit failed")


if __name__ == "__main__":
    main()
