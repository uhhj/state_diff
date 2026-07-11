#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pybullet as p

import phase3_12d_r2_environment_audit as r2audit
from phase3_12d_r22_integrity import strict_json_dump
from phase3_12d_r24_common import direct_physics_step, ordered_bead_velocity_xy, ordered_bead_xy, restore_slack_snapshot, slack_snapshot, temporary_environment
from phase3_12d_r24_parameter_sweep import env_config, reset_args

SEEDS = [315100, 315101, 315102, 315103, 315500, 315501, 315502, 315503]


def replay(env: Any, task: Any, bead: int, force_x: float, steps: int) -> Dict[str, Any]:
    xy, tension, force = [], [], []
    for _ in range(steps):
        pos = p.getBasePositionAndOrientation(bead)[0]
        direct_physics_step(env, task, external_force=(bead, (force_x, 0.0, 0.0), pos))
        xy.append(ordered_bead_xy(task).copy())
        snap = task._slack_model.snapshot()
        tension.append(float(snap["last_tension"]))
        force.append(np.asarray(task._slack_last_applied_force, dtype=np.float64).copy())
    return {"xy": np.asarray(xy), "tension": np.asarray(tension), "force": np.asarray(force), "snapshot": slack_snapshot(task)}


def audit_seed(root: Path, runtime: Any, tasks: Any, Environment: Any, seed: int, selected: Dict[str, Any]) -> Dict[str, Any]:
    env = None
    bullet_id = None
    try:
        env, task, _, _ = r2audit.reset_repaired(root, runtime, tasks, Environment, "hidden_slack_breakaway_pin_v2", seed, arm_after_settle=True, post_arm_steps=0, args=reset_args())
        bead = int(task._slack_bead_id)
        medium = float(selected["probes"]["engagement"]["force_x"])
        for _ in range(int(selected["probes"]["engagement"]["force_steps"])):
            pos = p.getBasePositionAndOrientation(bead)[0]
            direct_physics_step(env, task, external_force=(bead, (medium, 0.0, 0.0), pos))
            if task._slack_model.state == "engaged":
                break
        if task._slack_model.state != "engaged":
            raise RuntimeError(f"seed {seed} did not reach engaged snapshot state")
        start_xy = ordered_bead_xy(task).copy(); start_vel = ordered_bead_velocity_xy(task).copy()
        python_snapshot = slack_snapshot(task)
        np_state = np.random.get_state(); py_state = random.getstate()
        bullet_id = p.saveState()
        strong = float(selected["probes"]["breakaway"]["force_x"])
        reference = replay(env, task, bead, strong, 80)
        p.restoreState(bullet_id)
        restore_slack_snapshot(task, python_snapshot)
        np.random.set_state(np_state); random.setstate(py_state)
        restored_xy = ordered_bead_xy(task).copy(); restored_vel = ordered_bead_velocity_xy(task).copy()
        restored_snapshot = slack_snapshot(task)
        replayed = replay(env, task, bead, strong, 80)
        start_xy_max = float(np.max(np.abs(restored_xy - start_xy)))
        start_vel_max = float(np.max(np.abs(restored_vel - start_vel)))
        trajectory_max = float(np.max(np.abs(reference["xy"] - replayed["xy"])))
        tension_max = float(np.max(np.abs(reference["tension"] - replayed["tension"])))
        force_max = float(np.max(np.abs(reference["force"] - replayed["force"])))
        ref_snap, rep_snap = reference["snapshot"]["slack_model"], replayed["snapshot"]["slack_model"]
        state_machine_exact = python_snapshot["slack_model"] == restored_snapshot["slack_model"]
        release_exact = ref_snap["release_physics_step"] == rep_snap["release_physics_step"] and ref_snap["release_reason"] == rep_snap["release_reason"]
        post_release_zero = bool(ref_snap["state"] == "released" and rep_snap["state"] == "released" and np.all(reference["force"][-20:] == 0) and np.all(replayed["force"][-20:] == 0) and np.all(reference["tension"][-20:] == 0) and np.all(replayed["tension"][-20:] == 0))
        passed = bool(start_xy_max <= 1e-9 and start_vel_max <= 1e-9 and state_machine_exact and release_exact and trajectory_max <= 1e-7 and tension_max <= 1e-9 and force_max <= 1e-9 and post_release_zero)
        return {"seed": seed, "start_xy_max": start_xy_max, "start_velocity_max": start_vel_max, "state_machine_exact": state_machine_exact, "engagement_step_exact": ref_snap["engagement_physics_step"] == rep_snap["engagement_physics_step"], "release_step_exact": ref_snap["release_physics_step"] == rep_snap["release_physics_step"], "release_reason_exact": ref_snap["release_reason"] == rep_snap["release_reason"], "trajectory_xy_max": trajectory_max, "tension_trace_max": tension_max, "force_trace_max": force_max, "post_release_zero_force": post_release_zero, "pass": passed}
    finally:
        if bullet_id is not None:
            try: p.removeState(bullet_id)
            except Exception: pass
        if env is not None:
            r2audit.close(runtime, env)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/state_diff2")
    ap.add_argument("--out-json", default="reports/phase3_12d_r24_snapshot_audit_summary.json")
    ap.add_argument("--out-md", default="reports/phase3_12d_r24_snapshot_audit_report.md")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    env_audit = json.loads((root / "reports/phase3_12d_r24_environment_audit_summary.json").read_text())
    if env_audit.get("verdict") != "PASS":
        raise SystemExit("environment audit did not pass")
    selected = json.loads((root / "reports/phase3_12d_r24_selected_config.json").read_text())
    runtime, tasks, Environment = r2audit.setup_runtime(root)
    with temporary_environment(env_config(selected["slack_distance"], selected["spring_stiffness"], selected["radial_damping"])):
        results = [audit_seed(root, runtime, tasks, Environment, seed, selected) for seed in SEEDS]
    verdict = "PASS" if all(row["pass"] for row in results) else "FAIL"
    maxima = {key: max(row[key] for row in results) for key in ["start_xy_max", "start_velocity_max", "trajectory_xy_max", "tension_trace_max", "force_trace_max"]}
    payload = {"verdict": verdict, "seeds": SEEDS, "results": results, "maxima": maxima, "all_release_steps_exact": all(r["release_step_exact"] for r in results), "all_release_reasons_exact": all(r["release_reason_exact"] for r in results)}
    strict_json_dump(root / args.out_json, payload)
    lines = ["# Phase3.12d-r2.4 Snapshot Restore Audit", "", f"- Verdict: `{verdict}`", "", "| Metric | Maximum |", "|---|---:|"] + [f"| `{k}` | `{v:.12g}` |" for k, v in maxima.items()]
    lines += ["", f"- Release step exact for all seeds: `{payload['all_release_steps_exact']}`", f"- Release reason exact for all seeds: `{payload['all_release_reasons_exact']}`"]
    (root / args.out_md).write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    if verdict != "PASS": raise SystemExit("[r2.4] snapshot audit failed")


if __name__ == "__main__":
    main()
