#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

import phase3_12d_r2_common as common
import phase3_12d_r2_environment_audit as r2audit


DEFAULT_SEEDS = [312000, 312001, 312002, 312003, 312500, 312501, 312502, 312503]
R21_SCOPE = "phase3_12d_r21_paired_horizon_audit_no_phase4_no_cps"


def as_array(record: Mapping[str, Any], key: str) -> np.ndarray:
    value = record["meta"][key]
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"non-finite values in {key}")
    return array


def finite_float(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite scalar: {value!r}")
    return result


def geometry_scalars(record: Mapping[str, Any], xy: np.ndarray) -> Dict[str, float]:
    goal_xy = np.asarray(record["goal_xy"], dtype=np.float64)
    radius = finite_float(record["radius"])
    return {
        "fraction": float(r2audit.fraction_from_goal(xy, goal_xy, radius)),
        "curve": float(r2audit.curve_metric(xy)),
    }


def scalar_differences(
    record_a: Mapping[str, Any],
    xy_a: np.ndarray,
    record_b: Mapping[str, Any],
    xy_b: np.ndarray,
) -> Dict[str, float]:
    a = geometry_scalars(record_a, xy_a)
    b = geometry_scalars(record_b, xy_b)
    return {
        "fraction_diff": abs(a["fraction"] - b["fraction"]),
        "curve_diff": abs(a["curve"] - b["curve"]),
    }


def difference_with_scalars(
    record_a: Mapping[str, Any],
    xy_a: np.ndarray,
    record_b: Mapping[str, Any],
    xy_b: np.ndarray,
) -> Dict[str, float]:
    return {
        **common.difference_stats(xy_a, xy_b),
        **scalar_differences(record_a, xy_a, record_b, xy_b),
    }


def compare_motion(
    record_a: Mapping[str, Any],
    record_b: Mapping[str, Any],
) -> Dict[str, float]:
    a_pre = as_array(record_a, "pre_arm_xy")
    a_post = as_array(record_a, "post_arm_xy")
    b_pre = as_array(record_b, "pre_arm_xy")
    b_post = as_array(record_b, "post_arm_xy")
    a_motion = a_post - a_pre
    b_motion = b_post - b_pre
    return common.difference_stats(a_motion, b_motion)


def compare_velocity(
    record_a: Mapping[str, Any],
    record_b: Mapping[str, Any],
) -> Dict[str, float]:
    return common.difference_stats(
        as_array(record_a, "post_arm_velocity"),
        as_array(record_b, "post_arm_velocity"),
    )


def validate_capture_shapes(records: Iterable[Mapping[str, Any]]) -> None:
    expected_xy_shape: Optional[Tuple[int, ...]] = None
    expected_vel_shape: Optional[Tuple[int, ...]] = None
    for record in records:
        for key in ["pre_arm_xy", "immediate_post_arm_xy", "post_arm_xy"]:
            xy = as_array(record, key)
            if xy.ndim != 2 or xy.shape[1] != 2:
                raise ValueError(f"bad {key} shape: {xy.shape}")
            if expected_xy_shape is None:
                expected_xy_shape = xy.shape
            elif xy.shape != expected_xy_shape:
                raise ValueError(f"inconsistent XY shape: {xy.shape} != {expected_xy_shape}")
        for key in [
            "pre_arm_velocity",
            "immediate_post_arm_velocity",
            "post_arm_velocity",
        ]:
            vel = as_array(record, key)
            if vel.ndim != 2 or vel.shape[1] != 3:
                raise ValueError(f"bad {key} shape: {vel.shape}")
            if expected_vel_shape is None:
                expected_vel_shape = vel.shape
            elif vel.shape != expected_vel_shape:
                raise ValueError(
                    f"inconsistent velocity shape: {vel.shape} != {expected_vel_shape}"
                )


def build_horizon_comparison(
    *,
    horizon: int,
    free_a: Mapping[str, Any],
    free_b: Mapping[str, Any],
    hidden_unarmed: Mapping[str, Any],
    hidden_armed: Mapping[str, Any],
) -> Dict[str, Any]:
    """Compare all conditions after the same number of no-action physics steps.

    This pure comparison function is deliberately separated from PyBullet capture so
    the comparator can be unit-tested with synthetic records.
    """
    validate_capture_shapes([free_a, free_b, hidden_unarmed, hidden_armed])

    free_a_pre = as_array(free_a, "pre_arm_xy")
    free_a_post = as_array(free_a, "post_arm_xy")
    free_b_pre = as_array(free_b, "pre_arm_xy")
    free_b_post = as_array(free_b, "post_arm_xy")
    unarmed_pre = as_array(hidden_unarmed, "pre_arm_xy")
    unarmed_post = as_array(hidden_unarmed, "post_arm_xy")
    armed_pre = as_array(hidden_armed, "pre_arm_xy")
    armed_immediate = as_array(hidden_armed, "immediate_post_arm_xy")
    armed_post = as_array(hidden_armed, "post_arm_xy")

    goal_free_a = np.asarray(free_a["goal_xy"], dtype=np.float64)
    goal_free_b = np.asarray(free_b["goal_xy"], dtype=np.float64)
    goal_unarmed = np.asarray(hidden_unarmed["goal_xy"], dtype=np.float64)
    goal_armed = np.asarray(hidden_armed["goal_xy"], dtype=np.float64)

    result: Dict[str, Any] = {
        "evaluation_steps": int(horizon),
        "goal_parity": {
            "free_replicate": common.difference_stats(goal_free_b, goal_free_a),
            "hidden_unarmed": common.difference_stats(goal_unarmed, goal_free_a),
            "hidden_armed": common.difference_stats(goal_armed, goal_free_a),
        },
        "settle_steps": {
            "free_a": int(free_a["meta"]["settle_steps_used"]),
            "free_b": int(free_b["meta"]["settle_steps_used"]),
            "hidden_unarmed": int(hidden_unarmed["meta"]["settle_steps_used"]),
            "hidden_armed": int(hidden_armed["meta"]["settle_steps_used"]),
        },
        "pre_arm": {
            "free_reproducibility": difference_with_scalars(
                free_b, free_b_pre, free_a, free_a_pre
            ),
            "hidden_unarmed_vs_free": difference_with_scalars(
                hidden_unarmed, unarmed_pre, free_a, free_a_pre
            ),
            "hidden_armed_vs_free": difference_with_scalars(
                hidden_armed, armed_pre, free_a, free_a_pre
            ),
        },
        "immediate_arm_jump": {
            **common.difference_stats(armed_immediate, armed_pre),
            **scalar_differences(hidden_armed, armed_immediate, hidden_armed, armed_pre),
            "anchor_error": finite_float(hidden_armed["meta"]["arm_anchor_error"]),
        },
        "absolute_no_action_drift": {
            "free": difference_with_scalars(free_a, free_a_post, free_a, free_a_pre),
            "free_replicate": difference_with_scalars(
                free_b, free_b_post, free_b, free_b_pre
            ),
            "hidden_unarmed": difference_with_scalars(
                hidden_unarmed, unarmed_post, hidden_unarmed, unarmed_pre
            ),
            "hidden_armed": difference_with_scalars(
                hidden_armed, armed_post, hidden_armed, armed_pre
            ),
        },
        "same_horizon": {
            "free_reproducibility": difference_with_scalars(
                free_b, free_b_post, free_a, free_a_post
            ),
            "common_evolution_parity": difference_with_scalars(
                hidden_unarmed, unarmed_post, free_a, free_a_post
            ),
            "paired_visible_difference": difference_with_scalars(
                hidden_armed, armed_post, free_a, free_a_post
            ),
            "excess_motion_difference": compare_motion(hidden_armed, free_a),
            "unarmed_excess_motion": compare_motion(hidden_unarmed, free_a),
        },
        "velocity": {
            "free_reproducibility": compare_velocity(free_b, free_a),
            "common_evolution_parity": compare_velocity(hidden_unarmed, free_a),
            "paired_visible_difference": compare_velocity(hidden_armed, free_a),
        },
        "runtime_state": {
            "free_hook_error": str(free_a["meta"].get("hook_error")),
            "free_replicate_hook_error": str(free_b["meta"].get("hook_error")),
            "hidden_unarmed_hook_error": str(
                hidden_unarmed["meta"].get("hook_error")
            ),
            "hidden_armed_hook_error": str(hidden_armed["meta"].get("hook_error")),
            "hidden_unarmed_released": bool(
                hidden_unarmed["meta"].get("breakaway_released", False)
            ),
            "hidden_armed_released": bool(
                hidden_armed["meta"].get("breakaway_released", False)
            ),
            "free_constraint_count": int(free_a.get("hidden_constraint_count", 0)),
            "free_body_count": int(free_a.get("hidden_body_count", 0)),
            "hidden_unarmed_constraint_count": int(
                hidden_unarmed.get("hidden_constraint_count", 0)
            ),
            "hidden_unarmed_contact_applied": bool(
                hidden_unarmed.get("hidden_contact_applied", False)
            ),
            "hidden_armed_constraint_count": int(
                hidden_armed.get("hidden_constraint_count", 0)
            ),
            "hidden_armed_body_count": int(hidden_armed.get("hidden_body_count", 0)),
            "hidden_contact_applied": bool(
                hidden_armed.get("hidden_contact_applied", False)
            ),
        },
    }
    return result


def capture_one(
    root: Path,
    p34: Any,
    tasks: Any,
    Environment: Any,
    *,
    condition: str,
    seed: int,
    arm_after_settle: bool,
    evaluation_steps: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    return r2audit._capture_repaired_reset(
        root,
        p34,
        tasks,
        Environment,
        condition,
        int(seed),
        arm_after_settle=bool(arm_after_settle),
        post_arm_steps=int(evaluation_steps),
        args=args,
    )


def capture_horizon(
    root: Path,
    p34: Any,
    tasks: Any,
    Environment: Any,
    *,
    seed: int,
    horizon: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    # DeformableRavens has a global PyBullet connection. Each capture creates,
    # materializes and closes its Environment before the next capture begins.
    free_a = capture_one(
        root,
        p34,
        tasks,
        Environment,
        condition="free",
        seed=seed,
        arm_after_settle=True,
        evaluation_steps=horizon,
        args=args,
    )
    free_b = capture_one(
        root,
        p34,
        tasks,
        Environment,
        condition="free",
        seed=seed,
        arm_after_settle=True,
        evaluation_steps=horizon,
        args=args,
    )
    hidden_unarmed = capture_one(
        root,
        p34,
        tasks,
        Environment,
        condition="hidden_breakaway_pin",
        seed=seed,
        arm_after_settle=False,
        evaluation_steps=horizon,
        args=args,
    )
    hidden_armed = capture_one(
        root,
        p34,
        tasks,
        Environment,
        condition="hidden_breakaway_pin",
        seed=seed,
        arm_after_settle=True,
        evaluation_steps=horizon,
        args=args,
    )

    comparison = build_horizon_comparison(
        horizon=horizon,
        free_a=free_a,
        free_b=free_b,
        hidden_unarmed=hidden_unarmed,
        hidden_armed=hidden_armed,
    )
    comparison["seed"] = int(seed)
    return comparison


def metric_exceeds(metric: Mapping[str, Any], max_threshold: float, mae_threshold: float) -> bool:
    return (
        finite_float(metric["max_abs"]) > float(max_threshold)
        or finite_float(metric["mae"]) > float(mae_threshold)
    )


def scalar_exceeds(
    metric: Mapping[str, Any], fraction_threshold: float, curve_threshold: float
) -> bool:
    return (
        finite_float(metric.get("fraction_diff", 0.0)) > float(fraction_threshold)
        or finite_float(metric.get("curve_diff", 0.0)) > float(curve_threshold)
    )


def hook_bad(value: Any) -> bool:
    return value not in (None, "", "None")


def summarize_max(rows: Sequence[Mapping[str, Any]], path: Sequence[str], field: str) -> float:
    values: List[float] = []
    for row in rows:
        obj: Any = row
        for key in path:
            obj = obj[key]
        values.append(finite_float(obj[field]))
    return max(values) if values else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--goal-seeds", nargs="+", type=int, default=[312000, 312500])
    parser.add_argument("--evaluation-steps", nargs="+", type=int, default=[0, 1, 5, 20])
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
    parser.add_argument("--repro-max", type=float, default=1e-7)
    parser.add_argument("--repro-mae", type=float, default=1e-8)
    parser.add_argument("--common-max", type=float, default=1e-6)
    parser.add_argument("--common-mae", type=float, default=1e-7)
    parser.add_argument("--paired-max", type=float, default=1e-4)
    parser.add_argument("--paired-mae", type=float, default=1e-5)
    parser.add_argument("--fraction-threshold", type=float, default=1e-9)
    parser.add_argument("--curve-threshold", type=float, default=1e-5)
    parser.add_argument("--anchor-threshold", type=float, default=1e-7)
    parser.add_argument("--velocity-warn-max", type=float, default=1e-3)
    parser.add_argument("--causal-divergence-threshold", type=float, default=0.003)
    parser.add_argument("--goal-actionable-threshold", type=float, default=0.003)

    parser.add_argument(
        "--out-json",
        default="reports/phase3_12d_r21_environment_audit_summary.json",
    )
    parser.add_argument(
        "--out-md",
        default="reports/phase3_12d_r21_environment_audit_report.md",
    )
    args = parser.parse_args()

    horizons = sorted(set(int(value) for value in args.evaluation_steps))
    if not horizons or horizons[0] != 0:
        raise SystemExit("--evaluation-steps must include 0")
    if any(value < 0 for value in horizons):
        raise SystemExit("evaluation steps must be non-negative")

    root = Path(args.root).resolve()
    for path in [root, root / "scripts", root / "external/deformable-ravens"]:
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    p34, tasks, Environment = r2audit.setup_runtime(root)

    horizon_rows: List[Dict[str, Any]] = []
    for seed in args.seeds:
        for horizon in horizons:
            horizon_rows.append(
                capture_horizon(
                    root,
                    p34,
                    tasks,
                    Environment,
                    seed=int(seed),
                    horizon=int(horizon),
                    args=args,
                )
            )

    force_rows = [
        r2audit.causal_force_probe(root, p34, tasks, Environment, int(seed), args)
        for seed in args.goal_seeds
    ]
    goal_rows = [
        r2audit.goal_action_probe(root, p34, tasks, Environment, int(seed), args)
        for seed in args.goal_seeds
    ]

    issues: List[Dict[str, Any]] = []
    failure_categories = {
        "reproducibility": False,
        "common_evolution": False,
        "paired_visible": False,
        "causal_semantics": False,
        "code_or_runtime": False,
    }

    def add_issue(level: str, name: str, detail: Any, category: Optional[str] = None) -> None:
        issues.append({"level": level, "name": name, "detail": detail})
        if level == "FAIL" and category is not None:
            failure_categories[category] = True

    for row in horizon_rows:
        seed = int(row["seed"])
        horizon = int(row["evaluation_steps"])

        for label, metric in row["goal_parity"].items():
            if metric_exceeds(metric, 1e-12, 1e-13):
                add_issue(
                    "FAIL",
                    "paired_goal_geometry_mismatch",
                    {"seed": seed, "steps": horizon, "track": label, **metric},
                    "code_or_runtime",
                )

        settle_values = list(row["settle_steps"].values())
        if len(set(settle_values)) != 1:
            add_issue(
                "FAIL",
                "deterministic_settle_step_count_mismatch",
                {"seed": seed, "steps": horizon, **row["settle_steps"]},
                "reproducibility",
            )

        for label, metric in row["pre_arm"].items():
            if label == "free_reproducibility":
                max_thr, mae_thr, category = args.repro_max, args.repro_mae, "reproducibility"
            else:
                max_thr, mae_thr, category = args.pre_arm_max, args.pre_arm_mae, "common_evolution"
            if metric_exceeds(metric, max_thr, mae_thr) or scalar_exceeds(
                metric, args.fraction_threshold, args.curve_threshold
            ):
                add_issue(
                    "FAIL",
                    "pre_arm_pair_mismatch",
                    {"seed": seed, "steps": horizon, "track": label, **metric},
                    category,
                )

        immediate = row["immediate_arm_jump"]
        if (
            metric_exceeds(immediate, args.immediate_max, args.immediate_mae)
            or scalar_exceeds(immediate, args.fraction_threshold, args.curve_threshold)
            or finite_float(immediate["anchor_error"]) > args.anchor_threshold
        ):
            add_issue(
                "FAIL",
                "immediate_zero_offset_arm_failed",
                {"seed": seed, "steps": horizon, **immediate},
                "paired_visible",
            )

        same = row["same_horizon"]
        repro = same["free_reproducibility"]
        if metric_exceeds(repro, args.repro_max, args.repro_mae) or scalar_exceeds(
            repro, args.fraction_threshold, args.curve_threshold
        ):
            add_issue(
                "FAIL",
                "simulator_reproducibility_floor_failed",
                {"seed": seed, "steps": horizon, **repro},
                "reproducibility",
            )

        common_evolution = same["common_evolution_parity"]
        unarmed_excess = same["unarmed_excess_motion"]
        if (
            metric_exceeds(common_evolution, args.common_max, args.common_mae)
            or scalar_exceeds(
                common_evolution, args.fraction_threshold, args.curve_threshold
            )
            or metric_exceeds(unarmed_excess, args.common_max, args.common_mae)
        ):
            add_issue(
                "FAIL",
                "condition_independent_evolution_mismatch",
                {
                    "seed": seed,
                    "steps": horizon,
                    "same_horizon": common_evolution,
                    "motion": unarmed_excess,
                },
                "common_evolution",
            )

        paired = same["paired_visible_difference"]
        excess = same["excess_motion_difference"]
        if horizon > 0 and (
            metric_exceeds(paired, args.paired_max, args.paired_mae)
            or scalar_exceeds(paired, args.fraction_threshold, args.curve_threshold)
            or metric_exceeds(excess, args.paired_max, args.paired_mae)
        ):
            add_issue(
                "FAIL",
                "latent_constraint_no_action_visible_leak",
                {
                    "seed": seed,
                    "steps": horizon,
                    "same_horizon": paired,
                    "excess_motion": excess,
                },
                "paired_visible",
            )

        runtime = row["runtime_state"]
        for key in [
            "free_hook_error",
            "free_replicate_hook_error",
            "hidden_unarmed_hook_error",
            "hidden_armed_hook_error",
        ]:
            if hook_bad(runtime[key]):
                add_issue(
                    "FAIL",
                    "physics_hook_error",
                    {"seed": seed, "steps": horizon, "track": key, "value": runtime[key]},
                    "code_or_runtime",
                )
        if runtime["hidden_unarmed_released"] or runtime["hidden_armed_released"]:
            add_issue(
                "FAIL",
                "breakaway_released_during_no_action_horizon",
                {"seed": seed, "steps": horizon, **runtime},
                "paired_visible",
            )
        if runtime["free_constraint_count"] != 0 or runtime["free_body_count"] != 0:
            add_issue(
                "FAIL",
                "free_probe_contains_hidden_objects",
                {"seed": seed, "steps": horizon, **runtime},
                "code_or_runtime",
            )
        if (
            runtime["hidden_unarmed_constraint_count"] != 0
            or runtime["hidden_unarmed_contact_applied"]
        ):
            add_issue(
                "FAIL",
                "unarmed_probe_contains_hidden_contact",
                {"seed": seed, "steps": horizon, **runtime},
                "code_or_runtime",
            )
        if runtime["hidden_armed_constraint_count"] != 1:
            add_issue(
                "FAIL",
                "armed_probe_constraint_count_invalid",
                {"seed": seed, "steps": horizon, **runtime},
                "code_or_runtime",
            )
        if not runtime["hidden_contact_applied"]:
            add_issue(
                "FAIL",
                "armed_probe_hidden_contact_not_applied",
                {"seed": seed, "steps": horizon, **runtime},
                "code_or_runtime",
            )
        if runtime["hidden_armed_body_count"] != 0:
            add_issue(
                "WARN",
                "armed_probe_has_hidden_body",
                {"seed": seed, "steps": horizon, **runtime},
            )

        paired_velocity = row["velocity"]["paired_visible_difference"]
        if finite_float(paired_velocity["max_abs"]) > args.velocity_warn_max:
            add_issue(
                "WARN",
                "paired_velocity_difference_large",
                {"seed": seed, "steps": horizon, **paired_velocity},
            )

    for row in force_rows:
        if not bool(row["hidden_released"]):
            add_issue(
                "FAIL",
                "controlled_force_did_not_release_breakaway",
                row,
                "causal_semantics",
            )
        if finite_float(row["max_trajectory_divergence"]) <= args.causal_divergence_threshold:
            add_issue(
                "FAIL",
                "latent_constraint_not_causally_effective",
                row,
                "causal_semantics",
            )
        if hook_bad(row["hook_error_free"]) or hook_bad(row["hook_error_hidden"]):
            add_issue("FAIL", "controlled_force_hook_error", row, "code_or_runtime")

    for row in goal_rows:
        if finite_float(row["dense_gain"]) <= args.goal_actionable_threshold:
            add_issue("FAIL", "free_goal_action_not_actionable", row, "causal_semantics")
        if hook_bad(row["hook_error"]):
            add_issue("FAIL", "goal_action_hook_error", row, "code_or_runtime")

    has_fail = any(item["level"] == "FAIL" for item in issues)
    has_warn = any(item["level"] == "WARN" for item in issues)
    verdict = "FAIL" if has_fail else ("WARN" if has_warn else "PASS")

    if failure_categories["code_or_runtime"]:
        root_cause = "phase312d_r21_code_or_runtime_integrity_failed"
    elif failure_categories["reproducibility"]:
        root_cause = "phase312d_r21_simulator_reproducibility_floor_failed"
    elif failure_categories["common_evolution"]:
        root_cause = "phase312d_r21_condition_independent_reset_evolution_mismatch"
    elif failure_categories["paired_visible"]:
        root_cause = "phase312d_r21_latent_constraint_no_action_visible_leak_supported"
    elif failure_categories["causal_semantics"]:
        root_cause = "phase312d_r21_causal_semantics_or_actionability_failed"
    else:
        root_cause = "phase312d_r21_paired_horizon_visible_parity_passed"

    payload = {
        "verdict": verdict,
        "root_cause": root_cause,
        "query_local_snapshot_allowed": not has_fail,
        "horizon_rows": horizon_rows,
        "causal_force_probe": force_rows,
        "goal_action_probe": goal_rows,
        "maxima": {
            "free_reproducibility_max_abs": summarize_max(
                horizon_rows, ["same_horizon", "free_reproducibility"], "max_abs"
            ),
            "common_evolution_max_abs": summarize_max(
                horizon_rows, ["same_horizon", "common_evolution_parity"], "max_abs"
            ),
            "paired_visible_max_abs": summarize_max(
                horizon_rows, ["same_horizon", "paired_visible_difference"], "max_abs"
            ),
            "excess_motion_max_abs": summarize_max(
                horizon_rows, ["same_horizon", "excess_motion_difference"], "max_abs"
            ),
            "paired_velocity_max_abs": summarize_max(
                horizon_rows, ["velocity", "paired_visible_difference"], "max_abs"
            ),
        },
        "thresholds": {
            "pre_arm_max": args.pre_arm_max,
            "pre_arm_mae": args.pre_arm_mae,
            "immediate_max": args.immediate_max,
            "immediate_mae": args.immediate_mae,
            "repro_max": args.repro_max,
            "repro_mae": args.repro_mae,
            "common_max": args.common_max,
            "common_mae": args.common_mae,
            "paired_max": args.paired_max,
            "paired_mae": args.paired_mae,
            "fraction_threshold": args.fraction_threshold,
            "curve_threshold": args.curve_threshold,
            "anchor_threshold": args.anchor_threshold,
            "velocity_warn_max": args.velocity_warn_max,
            "causal_divergence_threshold": args.causal_divergence_threshold,
            "goal_actionable_threshold": args.goal_actionable_threshold,
        },
        "failure_categories": failure_categories,
        "issues": issues,
        "correction_note": (
            "Unlike r2, all hard visible comparisons pair free and hidden at the "
            "same no-action physics horizon. Absolute motion from t=0 is diagnostic only."
        ),
        "decision": "Run query-local candidate smoke only if this report has no FAIL.",
        "scope": R21_SCOPE,
    }

    out_json = root / args.out_json
    out_md = root / args.out_md
    common.save_json(out_json, payload)

    lines = [
        "# Phase3.12d-r2.1 Paired-Horizon No-Action Drift Audit",
        "",
        "## Verdict",
        "",
        f"- Verdict: `{verdict}`",
        f"- Root cause: `{root_cause}`",
        f"- Query-local snapshot allowed: `{not has_fail}`",
        f"- Horizon rows: `{len(horizon_rows)}`",
        "",
        "## Correction",
        "",
        "- r2 compared `hidden(t=N)` against `free(t=0)`.",
        "- r2.1 compares `hidden(t=N)` against `free(t=N)`.",
        "- Absolute no-action drift remains diagnostic and is not a condition-specific hard gate.",
        "",
        "## Paired-horizon geometry",
        "",
        "| Seed | Steps | Free drift max | Repro max | Common max | Paired max | Paired MAE | Excess max | Fraction diff | Curve diff |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in horizon_rows:
        absolute = row["absolute_no_action_drift"]["free"]
        same = row["same_horizon"]
        paired = same["paired_visible_difference"]
        lines.append(
            f"| {row['seed']} | {row['evaluation_steps']} | "
            f"{absolute['max_abs']:.8g} | "
            f"{same['free_reproducibility']['max_abs']:.8g} | "
            f"{same['common_evolution_parity']['max_abs']:.8g} | "
            f"{paired['max_abs']:.8g} | {paired['mae']:.8g} | "
            f"{same['excess_motion_difference']['max_abs']:.8g} | "
            f"{paired['fraction_diff']:.8g} | {paired['curve_diff']:.8g} |"
        )

    lines += [
        "",
        "## Maximum diagnostics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in payload["maxima"].items():
        lines.append(f"| `{key}` | {value:.8g} |")

    lines += [
        "",
        "## Controlled causal probe",
        "",
        "| Seed | Max divergence | Hidden released | Release physics step | Hook errors |",
        "|---:|---:|---:|---:|---|",
    ]
    for row in force_rows:
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
    for row in goal_rows:
        lines.append(
            f"| {row['seed']} | {row['dense_gain']:.6f} | "
            f"{row['ordered_gain']:.6f} | {row['fraction_gain']:.6f} |"
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
                f"| `{item['level']}` | `{item['name']}` | "
                f"{str(item['detail']).replace('|', '/')} |"
            )
    else:
        lines.append("| `PASS` | `none` | All hard gates passed. |")

    lines += [
        "",
        "## Decision",
        "",
        "- A reproducibility FAIL means the threshold is below simulator repeatability.",
        "- A common-evolution FAIL means condition-independent reset paths diverge.",
        "- A paired-visible FAIL means the armed latent constraint leaks into no-action visible geometry.",
        "- Only a report with no FAIL permits query-local candidate execution.",
        "- No model training, Phase4, or CPS was run.",
    ]
    out_md.write_text("\n".join(lines) + "\n")

    print(json.dumps(payload, indent=2, sort_keys=True))
    if has_fail:
        raise SystemExit("[Phase3.12d-r2.1][FAIL] paired-horizon environment audit failed")


if __name__ == "__main__":
    main()
