#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from phase3_12d_r22_integrity import sha256_file, strict_json_dump


TRACKS = (
    ("free_a", "free", False),
    ("free_b", "free", False),
    ("hidden_unarmed", "hidden_breakaway_pin", False),
    ("hidden_armed", "hidden_breakaway_pin", True),
)

_WORKER_RUNTIME = None


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=str(root),
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()


def used_seeds_from_r22(path: Path) -> set[int]:
    if not path.exists():
        return set()
    with np.load(path, allow_pickle=False) as data:
        if "visible_seed" not in data.files:
            return set()
        return {int(value) for value in data["visible_seed"].reshape(-1)}


def collect_seed_job(payload: Dict[str, Any]) -> int:
    global _WORKER_RUNTIME

    root = Path(payload["root"]).resolve()
    for path in (root, root / "scripts", root / "external/deformable-ravens"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    import phase3_12d_r22_collect_observation_leakage as r22
    import phase3_12d_r2_environment_audit as r2audit

    if _WORKER_RUNTIME is None:
        _WORKER_RUNTIME = r2audit.setup_runtime(root)
    runtime, tasks, Environment = _WORKER_RUNTIME

    args = argparse.Namespace(**payload["args"])
    seed = int(payload["seed"])
    rows: List[Dict[str, Any]] = []
    horizons = list(range(0, int(payload["max_step"]) + 1))

    for track, condition, armed in TRACKS:
        rows.extend(
            r22.capture_track(
                root,
                runtime,
                tasks,
                Environment,
                args,
                track,
                condition,
                armed,
                seed,
                horizons,
            )
        )

    arrays = r22.rows_to_arrays(
        rows,
        payload["main_head"],
        payload["submodule_head"],
    )
    arrays["pair_group"] = np.asarray(
        [f"phase3_12d_r23_seed_{seed}"] * arrays["state"].shape[0],
        dtype="<U64",
    )

    expected = len(TRACKS) * len(horizons)
    if arrays["state"].shape[0] != expected:
        raise RuntimeError(
            f"seed {seed} row count mismatch: "
            f"{arrays['state'].shape[0]} != {expected}"
        )

    shard = Path(payload["shard"])
    shard.parent.mkdir(parents=True, exist_ok=True)
    temporary = shard.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, shard)
    return seed


def pair_max_abs(
    arrays: Dict[str, np.ndarray],
    *,
    left: str,
    right: str,
    block: str,
) -> float:
    tracks = arrays["track_name"].astype(str)
    seeds = arrays["visible_seed"].astype(np.int64)
    steps = arrays["evaluation_steps"].astype(np.int64)
    values: List[float] = []
    for seed in np.unique(seeds):
        for step in np.unique(steps):
            li = np.flatnonzero(
                (tracks == left) & (seeds == seed) & (steps == step)
            )
            ri = np.flatnonzero(
                (tracks == right) & (seeds == seed) & (steps == step)
            )
            if li.size != 1 or ri.size != 1:
                raise RuntimeError(
                    f"incomplete pair seed={seed} step={step} "
                    f"{left}/{right}"
                )
            values.append(
                float(
                    np.max(
                        np.abs(
                            arrays[block][li[0]]
                            - arrays[block][ri[0]]
                        )
                    )
                )
            )
    return max(values, default=0.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument(
        "--out-dir",
        default="data/phase3_12d_r23_observation_contract",
    )
    parser.add_argument("--num-seeds", type=int, default=128)
    parser.add_argument("--seed-start", type=int, default=314000)
    parser.add_argument("--max-step", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--windows",
        default="data/phase3_state_diff_windows/phase3_windows.npz",
    )
    parser.add_argument(
        "--r22-data",
        default="data/phase3_12d_r22_observation_leakage/paired_observations.npz",
    )
    parser.add_argument(
        "--preflight",
        default="reports/phase3_12d_r23_preflight_summary.json",
    )
    parser.add_argument("--min-settle-steps", type=int, default=540)
    parser.add_argument("--max-settle-steps", type=int, default=2400)
    parser.add_argument("--static-checks-required", type=int, default=8)
    parser.add_argument("--static-check-interval", type=int, default=10)
    parser.add_argument("--motion-timeout", type=float, default=15.0)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    for path in (root, root / "scripts", root / "external/deformable-ravens"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    if (
        os.environ.get("PHASE3_ALLOW_R23_TRAJECTORY_COLLECTION", "0") != "1"
        or os.environ.get("PHASE3_R23_TRAJECTORY_COLLECTION_CONFIRMED", "0")
        != "1"
    ):
        raise SystemExit(
            "[r2.3][BLOCKED] trajectory collection gates are not enabled"
        )

    preflight = json.loads((root / args.preflight).read_text())
    if not preflight.get("collection_allowed"):
        raise SystemExit("[r2.3] preflight did not allow collection")

    windows_path = root / args.windows
    with np.load(windows_path, allow_pickle=False) as windows:
        th = int(np.asarray(windows["th"]).reshape(-1)[0])
        action_dim = int(
            np.asarray(windows["action_dim"]).reshape(-1)[0]
        )
        n_beads = int(np.asarray(windows["n_beads"]).reshape(-1)[0])
        used = {
            int(value)
            for value in windows["visible_seed"].reshape(-1)
            if int(value) >= 0
        }
    if th != 3:
        raise SystemExit(f"expected th=3, got {th}")
    if action_dim != 14:
        raise SystemExit(f"expected action_dim=14, got {action_dim}")

    used.update(used_seeds_from_r22(root / args.r22_data))
    selected: List[int] = []
    candidate = int(args.seed_start)
    while len(selected) < int(args.num_seeds):
        if candidate not in used:
            selected.append(candidate)
        candidate += 1

    main_head = git(root, "rev-parse", "HEAD")
    submodule_head = git(
        root,
        "-C",
        "external/deformable-ravens",
        "rev-parse",
        "HEAD",
    )

    out_dir = root / args.out_dir
    shard_dir = out_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    expected_per_seed = len(TRACKS) * (int(args.max_step) + 1)

    pending: List[Dict[str, Any]] = []
    for seed in selected:
        shard = shard_dir / f"seed_{seed}.npz"
        valid = False
        if shard.exists():
            try:
                with np.load(shard, allow_pickle=False) as old:
                    valid = (
                        old["state"].shape[0] == expected_per_seed
                        and np.array_equal(
                            np.unique(old["evaluation_steps"]),
                            np.arange(args.max_step + 1),
                        )
                    )
            except Exception:
                valid = False
        if valid:
            print(f"[r2.3] resume seed {seed}", flush=True)
            continue
        pending.append(
            {
                "root": str(root),
                "seed": seed,
                "max_step": int(args.max_step),
                "shard": str(shard),
                "main_head": main_head,
                "submodule_head": submodule_head,
                "args": {
                    "th": th,
                    "action_dim": action_dim,
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
        executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=max(1, int(args.workers)),
            mp_context=context,
        )
        processes = []
        try:
            futures = {
                executor.submit(collect_seed_job, payload): payload["seed"]
                for payload in pending
            }
            completed = len(selected) - len(pending)
            for future in concurrent.futures.as_completed(futures):
                seed = future.result()
                completed += 1
                print(
                    f"[r2.3] collected seed {seed} "
                    f"({completed}/{len(selected)})",
                    flush=True,
                )
            processes = list(
                getattr(executor, "_processes", {}).values()
            )
        finally:
            if not processes:
                processes = list(
                    getattr(executor, "_processes", {}).values()
                )
            executor.shutdown(wait=False, cancel_futures=True)
            for process in processes:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=5.0)

    parts: List[Dict[str, np.ndarray]] = []
    for seed in selected:
        with np.load(
            shard_dir / f"seed_{seed}.npz",
            allow_pickle=False,
        ) as shard:
            parts.append({key: shard[key] for key in shard.files})

    keys = parts[0].keys()
    arrays = {
        key: np.concatenate([part[key] for part in parts], axis=0)
        for key in keys
    }

    expected_rows = len(selected) * expected_per_seed
    if arrays["state"].shape[0] != expected_rows:
        raise RuntimeError(
            f"row count mismatch: "
            f"{arrays['state'].shape[0]} != {expected_rows}"
        )
    if arrays["bead_xy"].shape[1] != n_beads:
        raise RuntimeError("bead count mismatch")

    unique_keys = list(
        zip(
            arrays["visible_seed"].tolist(),
            arrays["evaluation_steps"].tolist(),
            arrays["track_name"].astype(str).tolist(),
        )
    )
    if len(unique_keys) != len(set(unique_keys)):
        raise RuntimeError("duplicate seed/step/track row")

    for block in (
        "state",
        "bead_xy",
        "bead_velocity_xy",
        "robot_proxy",
        "model_x",
    ):
        if not np.all(np.isfinite(arrays[block])):
            raise RuntimeError(f"non-finite {block}")

    controls = {
        "free_replicate_state_max_abs": pair_max_abs(
            arrays,
            left="free_a",
            right="free_b",
            block="state",
        ),
        "hidden_unarmed_state_max_abs": pair_max_abs(
            arrays,
            left="free_a",
            right="hidden_unarmed",
            block="state",
        ),
    }
    thresholds = {
        "free_replicate_state_max_abs": 1e-7,
        "hidden_unarmed_state_max_abs": 1e-6,
    }
    controls_pass = all(
        controls[key] <= thresholds[key] for key in thresholds
    )
    if not controls_pass:
        raise RuntimeError(f"trajectory controls failed: {controls}")

    out_npz = out_dir / "trajectory_observations.npz"
    np.savez_compressed(out_npz, **arrays)
    with np.load(out_npz, allow_pickle=False) as checked:
        if checked["state"].shape[0] != expected_rows:
            raise RuntimeError("saved trajectory NPZ verification failed")

    manifest = {
        "diagnostic_only": True,
        "policy_training_allowed": False,
        "main_head": main_head,
        "submodule_head": submodule_head,
        "file": str(out_npz),
        "file_sha256": sha256_file(out_npz),
        "selected_seeds": selected,
        "num_seeds": len(selected),
        "tracks": [item[0] for item in TRACKS],
        "physics_steps": list(range(args.max_step + 1)),
        "rows": expected_rows,
        "th": th,
        "action_dim": action_dim,
        "n_beads": n_beads,
        "state_dim": int(arrays["state"].shape[1]),
        "robot_proxy_dim": int(arrays["robot_proxy"].shape[1]),
        "collector_reuse": (
            "phase3_12d_r22_collect_observation_leakage.capture_track"
        ),
        "feature_policy": {
            "simulator_velocity_saved_for_privileged_ablation_only": True,
            "formal_observation_contract_not_changed": True,
            "metadata_excluded_from_features": True,
        },
        "controls": controls,
        "control_thresholds": thresholds,
        "controls_pass": controls_pass,
        "settle": {
            "min_steps": args.min_settle_steps,
            "max_steps": args.max_settle_steps,
            "static_checks_required": args.static_checks_required,
            "static_check_interval": args.static_check_interval,
        },
        "scope": {
            "candidate_matrix": False,
            "model_training": False,
            "phase4": False,
            "cps": False,
        },
    }
    strict_json_dump(out_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
