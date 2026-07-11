#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pybullet as p

import phase3_12d_r2_environment_audit as r2audit
from phase3_12d_r22_integrity import sha256_file, strict_json_dump
from phase3_12d_r24_common import direct_physics_step, ordered_bead_xy, temporary_environment

PILOT_SEEDS = [315000, 315001]
GRID = list(itertools.product([0.005, 0.010, 0.015], [50.0, 100.0, 150.0], [0.10, 0.20]))


def reset_args() -> argparse.Namespace:
    return argparse.Namespace(min_settle_steps=540, max_settle_steps=2400, static_checks_required=8, static_check_interval=10, motion_timeout=15.0)


def env_config(slack: float, stiffness: float, damping: float) -> Dict[str, str]:
    return {
        "CCDA_SLACK_V2_DISTANCE": str(slack), "CCDA_SLACK_V2_STIFFNESS": str(stiffness),
        "CCDA_SLACK_V2_DAMPING": str(damping), "CCDA_SLACK_V2_MAX_TENSION": "4.0",
        "CCDA_SLACK_V2_BREAKAWAY_EXTENSION": "0.030", "CCDA_SLACK_V2_BREAKAWAY_FORCE": "3.0",
        "CCDA_SLACK_V2_BEAD_RATIO": "0.45",
    }


def run_probe(root: Path, runtime: Any, tasks: Any, Environment: Any, seed: int, condition: str, force: float, force_steps: int, coast_steps: int = 0) -> Dict[str, Any]:
    env = None
    try:
        env, task, _, _ = r2audit.reset_repaired(root, runtime, tasks, Environment, condition, seed, arm_after_settle=True, post_arm_steps=0, args=reset_args())
        idx = int(round(0.45 * (len(task.cable_bead_IDs) - 1)))
        bead = int(task.cable_bead_IDs[idx])
        anchor = ordered_bead_xy(task)[idx].copy()
        xy_trace = [anchor.copy()]
        state_trace: List[str] = []
        tension_trace: List[float] = []
        force_trace: List[float] = []
        engage_loop = None
        release_loop = None
        total = int(force_steps + coast_steps)
        for i in range(total):
            pos = p.getBasePositionAndOrientation(bead)[0]
            diagnostic = (bead, (float(force if i < force_steps else 0.0), 0.0, 0.0), pos)
            direct_physics_step(env, task, external_force=diagnostic)
            xy_trace.append(ordered_bead_xy(task)[idx].copy())
            model = getattr(task, "_slack_model", None)
            snap = model.snapshot() if model is not None else {"state": "free", "last_tension": 0.0, "engagement_physics_step": None, "release_physics_step": None}
            state_trace.append(str(snap["state"]))
            tension_trace.append(float(snap["last_tension"]))
            force_trace.append(float(np.linalg.norm(getattr(task, "_slack_last_applied_force", [0, 0, 0]))))
            if snap.get("engagement_physics_step") is not None and engage_loop is None:
                engage_loop = i + 1
            if snap.get("release_physics_step") is not None and release_loop is None:
                release_loop = i + 1
        trace = np.asarray(xy_trace, dtype=np.float64)
        radial = np.linalg.norm(trace - anchor[None, :], axis=1)
        engaged_count_before_release = 0
        if engage_loop is not None:
            stop = (release_loop - 1) if release_loop is not None else len(state_trace)
            engaged_count_before_release = max(0, stop - (engage_loop - 1))
        post_zero = True
        if release_loop is not None:
            start = release_loop - 1
            post_zero = all(v == 0.0 for v in tension_trace[start:]) and all(v == 0.0 for v in force_trace[start:])
        return {
            "xy": trace, "max_radial": float(np.max(radial)), "states": state_trace,
            "max_tension": max(tension_trace, default=0.0), "max_force": max(force_trace, default=0.0),
            "engage_loop": engage_loop, "release_loop": release_loop,
            "engaged_steps_before_release": engaged_count_before_release,
            "post_release_zero": post_zero, "hook_error": getattr(env, "_ccda_physics_hook_error", None),
            "finite": bool(np.all(np.isfinite(trace)) and np.all(np.isfinite(tension_trace)) and np.all(np.isfinite(force_trace))),
        }
    finally:
        if env is not None:
            r2audit.close(runtime, env)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/state_diff2")
    ap.add_argument("--out-csv", default="reports/phase3_12d_r24_parameter_sweep.csv")
    ap.add_argument("--out-json", default="reports/phase3_12d_r24_selected_config.json")
    ap.add_argument("--out-md", default="reports/phase3_12d_r24_parameter_sweep_report.md")
    args = ap.parse_args()
    if os.environ.get("PHASE3_ALLOW_R24_PARAMETER_SWEEP") != "1" or os.environ.get("PHASE3_R24_PARAMETER_SWEEP_CONFIRMED") != "1":
        raise SystemExit("[r2.4] parameter sweep gates missing")
    root = Path(args.root).resolve()
    pre = json.loads((root / "reports/phase3_12d_r24_preflight_summary.json").read_text())
    if pre.get("verdict") != "PASS":
        raise SystemExit("preflight did not pass")
    runtime, tasks, Environment = r2audit.setup_runtime(root)
    rows: List[Dict[str, Any]] = []
    candidates = []
    for slack, stiffness, damping in GRID:
        per_seed = []
        with temporary_environment(env_config(slack, stiffness, damping)):
            for seed in PILOT_SEEDS:
                no_action = run_probe(root, runtime, tasks, Environment, seed, "hidden_slack_breakaway_pin_v2", 0.0, 0, 240)
                low_free = run_probe(root, runtime, tasks, Environment, seed, "free", 0.05, 20, 40)
                low_hidden = run_probe(root, runtime, tasks, Environment, seed, "hidden_slack_breakaway_pin_v2", 0.05, 20, 40)
                medium_free = run_probe(root, runtime, tasks, Environment, seed, "free", 7.0, 120, 40)
                medium_hidden = run_probe(root, runtime, tasks, Environment, seed, "hidden_slack_breakaway_pin_v2", 7.0, 120, 40)
                strong = run_probe(root, runtime, tasks, Environment, seed, "hidden_slack_breakaway_pin_v2", 8.0, 120, 20)
                low_len = min(len(low_free["xy"]), len(low_hidden["xy"]))
                med_len = min(len(medium_free["xy"]), len(medium_hidden["xy"]))
                low_div = float(np.max(np.linalg.norm(low_free["xy"][:low_len] - low_hidden["xy"][:low_len], axis=1)))
                med_div = float(np.max(np.linalg.norm(medium_free["xy"][:med_len] - medium_hidden["xy"][:med_len], axis=1)))
                result = {
                    "seed": seed,
                    "no_action_pass": set(no_action["states"]) <= {"dormant"} and no_action["max_tension"] == 0 and no_action["max_force"] == 0,
                    "sub_deadband_pass": low_hidden["max_radial"] < 0.9 * slack and set(low_hidden["states"]) <= {"dormant"} and low_hidden["max_tension"] == 0 and low_div <= 1e-4,
                    "engagement_pass": medium_hidden["engage_loop"] is not None and medium_hidden["engaged_steps_before_release"] >= 10 and med_div > 0.003,
                    "release_pass": strong["engage_loop"] is not None and strong["release_loop"] is not None and strong["release_loop"] > strong["engage_loop"] and strong["post_release_zero"],
                    "finite": all(x["finite"] and x["hook_error"] in (None, "", "None") for x in [no_action, low_free, low_hidden, medium_free, medium_hidden, strong]),
                    "sub_max_radial": low_hidden["max_radial"], "sub_divergence": low_div,
                    "engagement_divergence": med_div, "engaged_steps": medium_hidden["engaged_steps_before_release"],
                    "release_step": strong["release_loop"],
                }
                per_seed.append(result)
        passed = all(all(r[k] for k in ["no_action_pass", "sub_deadband_pass", "engagement_pass", "release_pass", "finite"]) for r in per_seed)
        row = {
            "slack_distance": slack, "spring_stiffness": stiffness, "radial_damping": damping,
            "max_tension": 4.0, "breakaway_extension": 0.030, "breakaway_force": 3.0, "bead_ratio": 0.45,
            "passed": passed, "no_action_pass": all(r["no_action_pass"] for r in per_seed),
            "sub_deadband_pass": all(r["sub_deadband_pass"] for r in per_seed), "engagement_pass": all(r["engagement_pass"] for r in per_seed),
            "release_pass": all(r["release_pass"] for r in per_seed), "finite": all(r["finite"] for r in per_seed),
            "max_sub_radial": max(r["sub_max_radial"] for r in per_seed), "max_sub_divergence": max(r["sub_divergence"] for r in per_seed),
            "min_engagement_divergence": min(r["engagement_divergence"] for r in per_seed), "min_engaged_steps": min(r["engaged_steps"] for r in per_seed),
            "release_steps": json.dumps([r["release_step"] for r in per_seed]),
        }
        rows.append(row)
        if passed:
            candidates.append(row)
        print(f"[r2.4] grid slack={slack} k={stiffness} c={damping} pass={passed}", flush=True)
    out_csv = root / args.out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    candidates.sort(key=lambda x: (-x["slack_distance"], x["spring_stiffness"], x["radial_damping"], x["max_tension"]))
    if not candidates:
        raise SystemExit("[r2.4] no pilot configuration passed all probes")
    selected = {k: candidates[0][k] for k in ["slack_distance", "spring_stiffness", "radial_damping", "max_tension", "breakaway_extension", "breakaway_force", "bead_ratio"]}
    payload = {
        **selected, "pilot_seeds": PILOT_SEEDS,
        "probes": {"no_action_steps": 240, "sub_deadband": {"force_x": 0.05, "force_steps": 20, "coast_steps": 40}, "engagement": {"force_x": 7.0, "force_steps": 120, "coast_steps": 40}, "breakaway": {"force_x": 8.0, "force_steps": 120, "coast_steps": 20}},
        "selection_rule": ["slack_distance descending", "spring_stiffness ascending", "radial_damping ascending", "max_tension ascending"],
        "main_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "submodule_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root / "external/deformable-ravens", text=True).strip(),
        "source_hashes": pre["source_sha256_after"], "parameter_sweep_csv_sha256": sha256_file(out_csv),
        "frozen_for_confirmation": True,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    payload["config_sha256"] = hashlib.sha256(canonical).hexdigest()
    strict_json_dump(root / args.out_json, payload)
    lines = ["# Phase3.12d-r2.4 Frozen Parameter Sweep", "", f"- Passing configurations: `{len(candidates)}` / `{len(rows)}`", f"- Selected config SHA256: `{payload['config_sha256']}`", "", "| Parameter | Value |", "|---|---:|"]
    lines += [f"| `{k}` | `{v}` |" for k, v in selected.items()]
    (root / args.out_md).write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
