#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

import phase3_12d_r2_environment_audit as r2audit
from phase3_12d_r22_integrity import strict_json_dump
from phase3_12d_r24_common import direct_physics_step, ordered_bead_velocity_xy, ordered_bead_xy, temporary_environment
from phase3_12d_r24_parameter_sweep import env_config, reset_args, run_probe

SEEDS = [315100, 315101, 315102, 315103, 315500, 315501, 315502, 315503]
HORIZONS = [0, 1, 5, 20, 60, 120, 240]


def robot_vector(task: Any) -> np.ndarray:
    value = task._robot_pose_proxy()
    return np.asarray(value.get("joint_positions", []) + value.get("joint_velocities", []) + value.get("ee_position", []) + value.get("ee_orientation", []), dtype=np.float64)


def capture_no_action(root: Path, runtime: Any, tasks: Any, Environment: Any, condition: str, seed: int) -> Dict[str, Any]:
    env = None
    try:
        env, task, _, _ = r2audit.reset_repaired(root, runtime, tasks, Environment, condition, seed, arm_after_settle=True, post_arm_steps=0, args=reset_args())
        rows = []
        current = 0
        for horizon in HORIZONS:
            for _ in range(horizon - current):
                direct_physics_step(env, task)
            current = horizon
            model = getattr(task, "_slack_model", None)
            snap = model.snapshot() if model is not None else None
            rows.append({
                "step": horizon, "xy": ordered_bead_xy(task), "velocity": ordered_bead_velocity_xy(task),
                "robot": robot_vector(task), "state": None if snap is None else snap["state"],
                "tension": 0.0 if snap is None else float(snap["last_tension"]),
                "force": np.asarray(getattr(task, "_slack_last_applied_force", [0, 0, 0]), dtype=np.float64),
                "engagement_step": None if snap is None else snap["engagement_physics_step"],
                "release_step": None if snap is None else snap["release_physics_step"],
            })
        return {"rows": rows, "bodies": len(task.hidden_body_ids), "constraints": len(task.hidden_constraint_ids), "hook_error": getattr(env, "_ccda_physics_hook_error", None)}
    finally:
        if env is not None:
            r2audit.close(runtime, env)


def max_diff(left: Dict[str, Any], right: Dict[str, Any], key: str) -> float:
    return max(float(np.max(np.abs(a[key] - b[key]))) for a, b in zip(left["rows"], right["rows"]))


def direction_probe(root: Path, runtime: Any, tasks: Any, Environment: Any, seed: int, vector: tuple) -> Dict[str, Any]:
    env = None
    try:
        env, task, _, _ = r2audit.reset_repaired(root, runtime, tasks, Environment, "hidden_slack_breakaway_pin_v2", seed, arm_after_settle=True, post_arm_steps=0, args=reset_args())
        bead = int(task._slack_bead_id)
        dots, z_forces, tensions = [], [], []
        for _ in range(120):
            pos = ppos = __import__("pybullet").getBasePositionAndOrientation(bead)[0]
            direct_physics_step(env, task, external_force=(bead, vector, pos))
            snap = task._slack_model.snapshot()
            radial = np.asarray(pos[:2]) - np.asarray(snap["anchor_xy"])
            force = np.asarray(task._slack_last_applied_force)
            dots.append(float(np.dot(force[:2], radial)))
            z_forces.append(float(force[2])); tensions.append(float(snap["last_tension"]))
        return {"engaged": task._slack_model.engagement_physics_step is not None, "max_force_dot_radial": max(dots), "max_abs_force_z": max(abs(x) for x in z_forces), "min_tension": min(tensions), "max_tension": max(tensions), "hook_error": getattr(env, "_ccda_physics_hook_error", None)}
    finally:
        if env is not None:
            r2audit.close(runtime, env)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/state_diff2")
    ap.add_argument("--out-json", default="reports/phase3_12d_r24_environment_audit_summary.json")
    ap.add_argument("--out-md", default="reports/phase3_12d_r24_environment_audit_report.md")
    ap.add_argument("--out-csv", default="reports/phase3_12d_r24_probe_trajectories.csv")
    args = ap.parse_args()
    if os.environ.get("PHASE3_ALLOW_R24_ENVIRONMENT_AUDIT") != "1" or os.environ.get("PHASE3_R24_ENVIRONMENT_AUDIT_CONFIRMED") != "1":
        raise SystemExit("[r2.4] environment audit gates missing")
    root = Path(args.root).resolve()
    selected = json.loads((root / "reports/phase3_12d_r24_selected_config.json").read_text())
    if not selected.get("frozen_for_confirmation"):
        raise SystemExit("selected parameters are not frozen")
    config = env_config(selected["slack_distance"], selected["spring_stiffness"], selected["radial_damping"])
    runtime, tasks, Environment = r2audit.setup_runtime(root)
    rows: List[Dict[str, Any]] = []
    no_action = []
    with temporary_environment(config):
        for seed in SEEDS:
            captures = {name: capture_no_action(root, runtime, tasks, Environment, cond, seed) for name, cond in [("free_a", "free"), ("free_b", "free"), ("hidden_v2_a", "hidden_slack_breakaway_pin_v2"), ("hidden_v2_b", "hidden_slack_breakaway_pin_v2")]}
            metric = {
                "seed": seed,
                "free_xy": max_diff(captures["free_a"], captures["free_b"], "xy"),
                "free_velocity": max_diff(captures["free_a"], captures["free_b"], "velocity"),
                "free_robot": max_diff(captures["free_a"], captures["free_b"], "robot"),
                "free_hidden_xy": max_diff(captures["free_a"], captures["hidden_v2_a"], "xy"),
                "free_hidden_velocity": max_diff(captures["free_a"], captures["hidden_v2_a"], "velocity"),
                "free_hidden_robot": max_diff(captures["free_a"], captures["hidden_v2_a"], "robot"),
                "hidden_replicate_xy": max_diff(captures["hidden_v2_a"], captures["hidden_v2_b"], "xy"),
                "hidden_state_dormant": all(r["state"] == "dormant" and r["tension"] == 0 and np.array_equal(r["force"], np.zeros(3)) and r["engagement_step"] is None and r["release_step"] is None for r in captures["hidden_v2_a"]["rows"]),
                "no_hidden_geometry": captures["hidden_v2_a"]["bodies"] == 0 and captures["hidden_v2_a"]["constraints"] == 0,
                "hook_clean": all(c["hook_error"] in (None, "", "None") for c in captures.values()),
            }
            metric["pass"] = max(metric[k] for k in ["free_xy", "free_velocity", "free_robot", "free_hidden_xy", "free_hidden_velocity", "free_hidden_robot", "hidden_replicate_xy"]) <= 1e-7 and metric["hidden_state_dormant"] and metric["no_hidden_geometry"] and metric["hook_clean"]
            no_action.append(metric)

        probes = selected["probes"]
        sub_results, engage_results, release_results = [], [], []
        for seed in SEEDS:
            sf = run_probe(root, runtime, tasks, Environment, seed, "free", probes["sub_deadband"]["force_x"], probes["sub_deadband"]["force_steps"], probes["sub_deadband"]["coast_steps"])
            sh = run_probe(root, runtime, tasks, Environment, seed, "hidden_slack_breakaway_pin_v2", probes["sub_deadband"]["force_x"], probes["sub_deadband"]["force_steps"], probes["sub_deadband"]["coast_steps"])
            n = min(len(sf["xy"]), len(sh["xy"])); sub_div = float(np.max(np.linalg.norm(sf["xy"][:n] - sh["xy"][:n], axis=1)))
            sub = {"seed": seed, "engaged": sh["engage_loop"] is not None, "max_tension": sh["max_tension"], "divergence": sub_div, "pass": sh["engage_loop"] is None and sh["max_tension"] == 0 and sub_div <= 1e-4}
            sub_results.append(sub); rows.append({"probe": "sub_deadband", **sub})
            mf = run_probe(root, runtime, tasks, Environment, seed, "free", probes["engagement"]["force_x"], probes["engagement"]["force_steps"], probes["engagement"]["coast_steps"])
            mh = run_probe(root, runtime, tasks, Environment, seed, "hidden_slack_breakaway_pin_v2", probes["engagement"]["force_x"], probes["engagement"]["force_steps"], probes["engagement"]["coast_steps"])
            n = min(len(mf["xy"]), len(mh["xy"])); med_div = float(np.max(np.linalg.norm(mf["xy"][:n] - mh["xy"][:n], axis=1)))
            engage = {"seed": seed, "engaged": mh["engage_loop"] is not None, "engaged_steps": mh["engaged_steps_before_release"], "divergence": med_div, "pass": mh["engage_loop"] is not None and mh["engaged_steps_before_release"] >= 10 and med_div > 0.003}
            engage_results.append(engage); rows.append({"probe": "engagement", **engage})
            strong = run_probe(root, runtime, tasks, Environment, seed, "hidden_slack_breakaway_pin_v2", probes["breakaway"]["force_x"], probes["breakaway"]["force_steps"], probes["breakaway"]["coast_steps"])
            release = {"seed": seed, "engaged": strong["engage_loop"] is not None, "released": strong["release_loop"] is not None, "engagement_step": strong["engage_loop"], "release_step": strong["release_loop"], "post_release_zero": strong["post_release_zero"], "pass": strong["engage_loop"] is not None and strong["release_loop"] is not None and strong["release_loop"] > strong["engage_loop"] and strong["post_release_zero"] and strong["hook_error"] in (None, "", "None")}
            release_results.append(release); rows.append({"probe": "breakaway", **release})

        directions = {}
        for name, vec in {"+x": (1.0, 0.0, 0.0), "-x": (-1.0, 0.0, 0.0), "+y": (0.0, 1.0, 0.0), "-y": (0.0, -1.0, 0.0), "toward_anchor": (-0.05, 0.0, 0.0)}.items():
            directions[name] = [direction_probe(root, runtime, tasks, Environment, seed, vec) for seed in SEEDS]
        direction_pass = all(x["max_force_dot_radial"] <= 1e-12 and x["max_abs_force_z"] == 0 and x["min_tension"] >= 0 and x["max_tension"] <= selected["max_tension"] + 1e-12 and x["hook_error"] in (None, "", "None") for values in directions.values() for x in values)
        goal = [r2audit.goal_action_probe(root, runtime, tasks, Environment, seed, reset_args()) for seed in SEEDS[:2]]

    checks = {
        "no_action_parity": all(x["pass"] for x in no_action),
        "direction_physics": direction_pass,
        "sub_deadband_8_of_8": sum(x["pass"] for x in sub_results) == 8,
        "engagement_at_least_6_of_8": sum(x["pass"] for x in engage_results) >= 6,
        "breakaway_8_of_8": sum(x["pass"] for x in release_results) == 8,
        "free_goal_actionable": max(x["dense_gain"] for x in goal) > 0.003,
    }
    verdict = "PASS" if all(checks.values()) else "FAIL"
    payload = {
        "verdict": verdict, "checks": checks, "seeds": SEEDS, "horizons": HORIZONS,
        "selected_config_sha256": selected["config_sha256"], "no_action": no_action,
        "sub_deadband": sub_results, "engagement": engage_results, "breakaway": release_results,
        "direction_probes": directions, "free_goal_probes": goal,
        "maxima": {k: max(x[k] for x in no_action) for k in ["free_xy", "free_velocity", "free_robot", "free_hidden_xy", "free_hidden_velocity", "free_hidden_robot", "hidden_replicate_xy"]},
        "observation_audit_allowed": verdict == "PASS", "snapshot_audit_allowed": verdict == "PASS",
    }
    strict_json_dump(root / args.out_json, payload)
    out_csv = root / args.out_csv; out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row})
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    lines = ["# Phase3.12d-r2.4 Environment Audit", "", f"- Verdict: `{verdict}`", "", "| Check | Result |", "|---|---:|"] + [f"| `{k}` | `{v}` |" for k, v in checks.items()]
    lines += ["", "## No-Action Maxima", "", "| Metric | Maximum |", "|---|---:|"] + [f"| `{k}` | `{v:.12g}` |" for k, v in payload["maxima"].items()]
    (root / args.out_md).write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    if verdict != "PASS":
        raise SystemExit("[r2.4] environment audit failed")


if __name__ == "__main__":
    main()
