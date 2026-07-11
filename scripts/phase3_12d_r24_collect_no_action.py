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
    ("hidden_v2_a", "hidden_slack_breakaway_pin_v2", True),
    ("hidden_v2_b", "hidden_slack_breakaway_pin_v2", True),
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
    import phase3_12d_r2_common as legacy_common
    import phase3_12d_r2_environment_audit as r2audit
    import phase3_12d_r24_common as r24common

    # Keep no-action stepping identical to the production pre/step/EE/post
    # ordering even though the v2 pre-hook is expected to remain dormant.
    legacy_common.direct_physics_step = lambda env, task, dispatch_hook=True: (
        r24common.direct_physics_step(env, task, dispatch_hooks=dispatch_hook)
    )
    for key, value in payload["environment"].items():
        os.environ[key] = str(value)

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
        [f"phase3_12d_r24_seed_{seed}"] * arrays["state"].shape[0],
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
        default="data/phase3_12d_r24_no_action",
    )
    parser.add_argument("--num-seeds", type=int, default=128)
    parser.add_argument("--seed-start", type=int, default=316000)
    parser.add_argument("--max-step", type=int, default=240)
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
        default="reports/phase3_12d_r24_environment_audit_summary.json",
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
        os.environ.get("PHASE3_ALLOW_R24_ENVIRONMENT_AUDIT", "0") != "1"
        or os.environ.get("PHASE3_R24_ENVIRONMENT_AUDIT_CONFIRMED", "0")
        != "1"
    ):
        raise SystemExit(
            "[r2.4][BLOCKED] observation collection gates are not enabled"
        )

    preflight = json.loads((root / args.preflight).read_text())
    if preflight.get("verdict") != "PASS":
        raise SystemExit("[r2.4] environment audit did not pass")

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

    import phase3_12d_r22_collect_observation_leakage as r22
    used.update(r22.discover_used_seeds(root, windows_path))
    used.update(used_seeds_from_r22(root / args.r22_data))
    selected_config = json.loads(
        (root / "reports/phase3_12d_r24_selected_config.json").read_text()
    )
    selected_environment = {
        "CCDA_SLACK_V2_DISTANCE": selected_config["slack_distance"],
        "CCDA_SLACK_V2_STIFFNESS": selected_config["spring_stiffness"],
        "CCDA_SLACK_V2_DAMPING": selected_config["radial_damping"],
        "CCDA_SLACK_V2_MAX_TENSION": selected_config["max_tension"],
        "CCDA_SLACK_V2_BREAKAWAY_EXTENSION": selected_config["breakaway_extension"],
        "CCDA_SLACK_V2_BREAKAWAY_FORCE": selected_config["breakaway_force"],
        "CCDA_SLACK_V2_BEAD_RATIO": selected_config["bead_ratio"],
    }
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
            print(f"[r2.4] resume seed {seed}", flush=True)
            continue
        pending.append(
            {
                "root": str(root),
                "seed": seed,
                "max_step": int(args.max_step),
                "shard": str(shard),
                "main_head": main_head,
                "submodule_head": submodule_head,
                "environment": selected_environment,
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
                    f"[r2.4] collected seed {seed} "
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
        "free_hidden_v2_state_max_abs": pair_max_abs(
            arrays,
            left="free_a",
            right="hidden_v2_a",
            block="state",
        ),
        "hidden_v2_replicate_state_max_abs": pair_max_abs(
            arrays,
            left="hidden_v2_a",
            right="hidden_v2_b",
            block="state",
        ),
    }
    thresholds = {
        "free_replicate_state_max_abs": 1e-7,
        "free_hidden_v2_state_max_abs": 1e-7,
        "hidden_v2_replicate_state_max_abs": 1e-7,
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
