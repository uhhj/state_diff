#!/usr/bin/env python3
"""Collect paired no-action observations for the r2.2 leakage audit only."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing
import os
import pickle
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

import numpy as np

from phase3_12d_r22_integrity import sha256_file, strict_json_dump


TRACKS = (
    ("free_a", "free", True),
    ("free_b", "free", True),
    ("hidden_unarmed", "hidden_breakaway_pin", False),
    ("hidden_armed", "hidden_breakaway_pin", True),
)

_WORKER_RUNTIME = None


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=str(root), text=True).strip()


def strings(values: Iterable[str]) -> np.ndarray:
    items = [str(value) for value in values]
    width = max(1, *(len(item) for item in items))
    return np.asarray(items, dtype=f"<U{width}")


def seeds_from_tree(value: Any, key: str = "") -> Set[int]:
    found: Set[int] = set()
    if isinstance(value, dict):
        for child_key, child in value.items():
            found.update(seeds_from_tree(child, str(child_key)))
    elif isinstance(value, (list, tuple)):
        for child in value:
            found.update(seeds_from_tree(child, key))
    elif "seed" in key.lower() and isinstance(value, (int, np.integer)):
        if int(value) >= 0:
            found.add(int(value))
    elif isinstance(value, str):
        found.update(int(match) for match in re.findall(r"(?:^|_)seed_([0-9]+)(?:$|_)", value))
    return found


def discover_used_seeds(root: Path, windows_path: Path) -> Set[int]:
    used: Set[int] = set()
    with np.load(windows_path, allow_pickle=False) as data:
        used.update(int(value) for value in data["visible_seed"].reshape(-1) if int(value) >= 0)
    for path in sorted((root / "reports").glob("phase3*.json")):
        try:
            used.update(seeds_from_tree(json.loads(path.read_text())))
        except Exception:
            continue
    # Raw info pickles are trusted local experiment artifacts. Scanning them
    # prevents a fresh diagnostic seed from silently reusing an old episode.
    for path in sorted((root / "external/deformable-ravens/data").glob("**/info/*.pkl")):
        try:
            with path.open("rb") as handle:
                used.update(seeds_from_tree(pickle.load(handle)))
        except Exception:
            continue
    return used


def materialize_capture(task: Any, info: Dict[str, Any], reset_meta: Dict[str, Any], args: argparse.Namespace, track: str, condition: str, seed: int, horizon: int) -> Dict[str, Any]:
    from ccda_phase3.data_io import extract_bead_vel_xy, extract_bead_xy, extract_robot_pose_proxy
    from ccda_phase3.rollout import pad_history, state_from_live_info
    import phase3_12d_r2_environment_audit as r2audit

    state = state_from_live_info(info, prev_xy=None).astype(np.float32)
    bead_xy = extract_bead_xy(info).astype(np.float32)
    bead_velocity = extract_bead_vel_xy(info).astype(np.float32)
    goal_xy = r2audit.targets_xy(task).astype(np.float32)
    if bead_velocity.shape != bead_xy.shape or not np.all(np.isfinite(bead_velocity)):
        raise RuntimeError("captured bead velocity is unavailable or non-finite")
    robot, robot_source = extract_robot_pose_proxy(info)
    robot = robot.astype(np.float32)
    model_x = np.concatenate(
        [
            pad_history([state], args.th).reshape(-1),
            np.zeros(args.th * args.action_dim, dtype=np.float32),
        ]
    ).astype(np.float32)
    for name, value in (("state", state), ("bead_xy", bead_xy), ("bead_velocity_xy", bead_velocity), ("robot_proxy", robot), ("goal_xy", goal_xy), ("model_x", model_x)):
        if not np.all(np.isfinite(value)):
            raise RuntimeError(f"non-finite {name}")
    return {
        "state": state,
        "bead_xy": bead_xy,
        "bead_velocity_xy": bead_velocity,
        "robot_proxy": robot,
        "goal_xy": goal_xy,
        "model_x": model_x,
        "track_name": track,
        "condition_name": condition,
        "visible_seed": int(seed),
        "evaluation_steps": int(horizon),
        "pair_group": f"phase3_12d_r22_seed_{int(seed)}",
        "state_sha256": hashlib.sha256(np.ascontiguousarray(state).tobytes()).hexdigest(),
        "robot_proxy_source": robot_source,
        "hidden_constraint_count": len(getattr(task, "hidden_constraint_ids", [])),
        "hidden_body_count": len(getattr(task, "hidden_body_ids", [])),
        "hidden_contact_armed": bool(getattr(task, "_hidden_contact_armed", reset_meta.get("hidden_contact_armed", False))),
        "breakaway_released": bool(getattr(task, "_breakaway_released", reset_meta.get("breakaway_released", False))),
        "settle_steps": int(reset_meta.get("settle_steps_used", -1)),
    }


def capture_track(root: Path, runtime: Any, tasks: Any, Environment: Any, args: argparse.Namespace, track: str, condition: str, armed: bool, seed: int, horizons: List[int]) -> List[Dict[str, Any]]:
    """Capture several no-action horizons along one deterministic track reset."""
    import phase3_12d_r2_common as common
    import phase3_12d_r2_environment_audit as r2audit

    env = None
    try:
        env, task, info, reset_meta = r2audit.reset_repaired(
            root,
            runtime,
            tasks,
            Environment,
            condition,
            int(seed),
            arm_after_settle=bool(armed),
            post_arm_steps=0,
            args=args,
        )
        rows = []
        current_step = 0
        for horizon in sorted(set(int(value) for value in horizons)):
            for _ in range(horizon - current_step):
                common.direct_physics_step(env, task, dispatch_hook=True)
            current_step = horizon
            hook_error = getattr(env, "_ccda_physics_hook_error", None)
            if hook_error not in (None, "", "None"):
                raise RuntimeError(f"physics hook failed: {hook_error}")
            _, extras = task.reward()
            extras["task.done"] = bool(task.done())
            info = env.info
            info["extras"] = extras
            rows.append(materialize_capture(task, info, reset_meta, args, track, condition, seed, horizon))
        return rows
    finally:
        if env is not None:
            r2audit.close(runtime, env)


def collect_seed_job(payload: Dict[str, Any]) -> int:
    """Collect one seed in a process-local PyBullet runtime."""
    global _WORKER_RUNTIME
    root = Path(payload["root"])
    for path in (root, root / "scripts", root / "external/deformable-ravens"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import phase3_12d_r2_environment_audit as r2audit

    if _WORKER_RUNTIME is None:
        _WORKER_RUNTIME = r2audit.setup_runtime(root)
    runtime, tasks, Environment = _WORKER_RUNTIME
    args = argparse.Namespace(**payload["args"])
    seed = int(payload["seed"])
    rows = []
    for track, condition, armed in TRACKS:
        rows.extend(capture_track(root, runtime, tasks, Environment, args, track, condition, armed, seed, payload["horizons"]))
    shard = Path(payload["shard"])
    temporary = shard.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **rows_to_arrays(rows, payload["main_head"], payload["submodule_head"]))
    os.replace(temporary, shard)
    return seed


def rows_to_arrays(rows: List[Dict[str, Any]], main_head: str, submodule_head: str) -> Dict[str, np.ndarray]:
    numeric_blocks = ("state", "bead_xy", "bead_velocity_xy", "robot_proxy", "goal_xy", "model_x")
    arrays: Dict[str, np.ndarray] = {name: np.stack([row[name] for row in rows]).astype(np.float32) for name in numeric_blocks}
    for name in ("visible_seed", "evaluation_steps", "hidden_constraint_count", "hidden_body_count", "settle_steps"):
        arrays[name] = np.asarray([row[name] for row in rows], dtype=np.int64)
    for name in ("hidden_contact_armed", "breakaway_released"):
        arrays[name] = np.asarray([row[name] for row in rows], dtype=np.bool_)
    for name in ("track_name", "condition_name", "pair_group", "state_sha256", "robot_proxy_source"):
        arrays[name] = strings(row[name] for row in rows)
    arrays["main_head"] = strings([main_head] * len(rows))
    arrays["submodule_head"] = strings([submodule_head] * len(rows))
    return arrays


def max_pair_difference(arrays: Dict[str, np.ndarray], left: str, right: str, horizon: int) -> float:
    tracks = arrays["track_name"].astype(str)
    steps = arrays["evaluation_steps"]
    seeds = arrays["visible_seed"]
    values = []
    for seed in np.unique(seeds):
        li = np.flatnonzero((tracks == left) & (steps == horizon) & (seeds == seed))
        ri = np.flatnonzero((tracks == right) & (steps == horizon) & (seeds == seed))
        if li.size != 1 or ri.size != 1:
            raise RuntimeError(f"incomplete pair seed={seed} horizon={horizon} {left}/{right}")
        values.append(float(np.max(np.abs(arrays["state"][li[0]] - arrays["state"][ri[0]]))))
    return max(values, default=0.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--out-dir", default="data/phase3_12d_r22_observation_leakage")
    parser.add_argument("--num-seeds", type=int, default=128)
    parser.add_argument("--seed-start", type=int, default=313000)
    parser.add_argument("--horizons", nargs="+", type=int, default=[0, 1, 5, 20])
    parser.add_argument("--min-settle-steps", type=int, default=540)
    parser.add_argument("--max-settle-steps", type=int, default=2400)
    parser.add_argument("--static-checks-required", type=int, default=8)
    parser.add_argument("--static-check-interval", type=int, default=10)
    parser.add_argument("--motion-timeout", type=float, default=15.0)
    parser.add_argument("--windows", default="data/phase3_state_diff_windows/phase3_windows.npz")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    for path in (root, root / "scripts", root / "external/deformable-ravens"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from phase3_12d_r22_integrity import sha256_array
    import phase3_12d_r2_environment_audit as r2audit

    audit_path = root / "reports/phase3_12d_r22_repo_data_audit_summary.json"
    audit = json.loads(audit_path.read_text())
    if not audit.get("collection_allowed"):
        raise SystemExit("repository/data audit did not allow collection")
    windows_path = root / args.windows
    with np.load(windows_path, allow_pickle=False) as windows:
        args.th = int(np.asarray(windows["th"]).reshape(-1)[0])
        args.action_dim = int(np.asarray(windows["action_dim"]).reshape(-1)[0])
        expected_n_beads = int(np.asarray(windows["n_beads"]).reshape(-1)[0])
    if args.action_dim != 14:
        raise SystemExit(f"expected action_dim=14, got {args.action_dim}")

    used = discover_used_seeds(root, windows_path)
    selected: List[int] = []
    candidate = int(args.seed_start)
    while len(selected) < int(args.num_seeds):
        if candidate not in used:
            selected.append(candidate)
        candidate += 1

    out_dir = root / args.out_dir
    shard_dir = out_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    main_head = git(root, "rev-parse", "HEAD")
    submodule_head = git(root, "-C", "external/deformable-ravens", "rev-parse", "HEAD")
    expected_per_seed = len(args.horizons) * len(TRACKS)
    pending = []
    for position, seed in enumerate(selected, 1):
        shard = shard_dir / f"seed_{seed}.npz"
        if shard.exists():
            with np.load(shard, allow_pickle=False) as old:
                if "goal_xy" in old.files and old["state"].shape[0] == expected_per_seed:
                    print(f"[r2.2] resume seed {seed} ({position}/{len(selected)})")
                    continue
        pending.append(
            {
                "root": str(root),
                "seed": seed,
                "horizons": sorted(set(args.horizons)),
                "shard": str(shard),
                "main_head": main_head,
                "submodule_head": submodule_head,
                "args": {
                    "th": args.th,
                    "action_dim": args.action_dim,
                    "min_settle_steps": args.min_settle_steps,
                    "max_settle_steps": args.max_settle_steps,
                    "static_checks_required": args.static_checks_required,
                    "static_check_interval": args.static_check_interval,
                    "motion_timeout": args.motion_timeout,
                },
            }
        )
    if pending:
        context = multiprocessing.get_context("spawn")
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=max(1, args.workers), mp_context=context)
        processes = []
        try:
            futures = {executor.submit(collect_seed_job, payload): int(payload["seed"]) for payload in pending}
            completed = len(selected) - len(pending)
            for future in concurrent.futures.as_completed(futures):
                seed = future.result()
                completed += 1
                print(f"[r2.2] collected seed {seed} ({completed}/{len(selected)})", flush=True)
            # DeformableRavens can leave process-local background resources
            # alive after the last seed. All futures are complete here, so
            # explicitly reap idle workers instead of hanging before merge.
            processes = list(getattr(executor, "_processes", {}).values())
        finally:
            if not processes:
                processes = list(getattr(executor, "_processes", {}).values())
            executor.shutdown(wait=False, cancel_futures=True)
            for process in processes:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=5.0)

    parts = []
    for seed in selected:
        with np.load(shard_dir / f"seed_{seed}.npz", allow_pickle=False) as shard:
            parts.append({key: shard[key] for key in shard.files})
    keys = parts[0].keys()
    arrays = {key: np.concatenate([part[key] for part in parts], axis=0) for key in keys}
    expected_rows = len(selected) * expected_per_seed
    if arrays["state"].shape[0] != expected_rows:
        raise RuntimeError(f"wrong row count {arrays['state'].shape[0]} != {expected_rows}")
    if arrays["bead_xy"].shape[1] != expected_n_beads:
        raise RuntimeError("bead count mismatch")
    keys_seen = list(zip(arrays["visible_seed"].tolist(), arrays["evaluation_steps"].tolist(), arrays["track_name"].astype(str).tolist()))
    if len(set(keys_seen)) != len(keys_seen):
        raise RuntimeError("duplicate seed/horizon/track key")
    for block in ("state", "bead_xy", "bead_velocity_xy", "robot_proxy", "goal_xy", "model_x"):
        if not np.all(np.isfinite(arrays[block])):
            raise RuntimeError(f"non-finite final {block}")

    controls = {
        "free_replicate_max_abs": max(max_pair_difference(arrays, "free_a", "free_b", h) for h in args.horizons),
        "hidden_unarmed_max_abs": max(max_pair_difference(arrays, "free_a", "hidden_unarmed", h) for h in args.horizons),
        "horizon0_hidden_armed_max_abs": max_pair_difference(arrays, "free_a", "hidden_armed", 0),
    }
    thresholds = {"free_replicate": 1e-7, "hidden_unarmed": 1e-6, "horizon0_armed": 1e-6}
    controls_pass = controls["free_replicate_max_abs"] <= thresholds["free_replicate"] and controls["hidden_unarmed_max_abs"] <= thresholds["hidden_unarmed"] and controls["horizon0_hidden_armed_max_abs"] <= thresholds["horizon0_armed"]
    if not controls_pass:
        raise RuntimeError(f"collection control failed: {controls}")

    out_npz = out_dir / "paired_observations.npz"
    np.savez_compressed(out_npz, **arrays)
    with np.load(out_npz, allow_pickle=False) as checked:
        if checked["state"].shape[0] != expected_rows:
            raise RuntimeError("saved NPZ verification failed")
    manifest = {
        "diagnostic_only": True,
        "policy_training_allowed": False,
        "main_head": main_head,
        "submodule_head": submodule_head,
        "file": str(out_npz),
        "file_sha256": sha256_file(out_npz),
        "array_sha256": {key: sha256_array(value) for key, value in arrays.items()},
        "num_seeds": len(selected),
        "seeds": selected,
        "used_seed_count_before_selection": len(used),
        "rows": expected_rows,
        "horizons": sorted(set(args.horizons)),
        "tracks": [item[0] for item in TRACKS],
        "horizon_capture_semantics": "one deterministic reset per seed/track; cumulative no-action physics snapshots at each requested horizon; physics steps are never action-history steps",
        "state_dim": int(arrays["state"].shape[1]),
        "model_x_dim": int(arrays["model_x"].shape[1]),
        "th": args.th,
        "action_dim": args.action_dim,
        "n_beads": expected_n_beads,
        "settle_config": {key: getattr(args, key) for key in ("min_settle_steps", "max_settle_steps", "static_checks_required", "static_check_interval")},
        "environment": {key: os.environ.get(key) for key in ("CCDA_BREAKAWAY_FORCE", "CCDA_BREAKAWAY_DISP", "CCDA_BREAKAWAY_BEAD_RATIO", "CCDA_BREAKAWAY_DAMPING")},
        "feature_provenance": {
            "state": "ccda_phase3.rollout.state_from_live_info(info, prev_xy=None)",
            "model_x": "pad_history([state], th) plus zero past executable actions",
            "forbidden_metadata_excluded": ["track", "condition", "visible_seed", "horizon", "constraint_count", "breakaway_state", "goal", "reward", "success"],
        },
        "controls": controls,
        "control_thresholds": thresholds,
        "controls_pass": controls_pass,
        "code_hashes": {str(path.relative_to(root)): sha256_file(path) for path in (root / "ccda_phase3/data_io.py", root / "ccda_phase3/rollout.py", root / "scripts/phase3_12d_r2_common.py", Path(__file__).resolve())},
        "dependencies": {name: getattr(__import__(name), "__version__", "unknown") for name in ("numpy", "torch", "pybullet")},
    }
    strict_json_dump(out_dir / "manifest.json", manifest)
    print(json.dumps({"rows": expected_rows, "controls": controls, "out": str(out_npz)}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
