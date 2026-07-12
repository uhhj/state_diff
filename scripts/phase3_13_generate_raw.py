#!/usr/bin/env python3
"""Generate real paired state-v2 raw episodes from the formal task."""
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import json
import multiprocessing
import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from ccda_phase3.action_codec import ExecutableActionCodec
from ccda_phase3.schema_v2 import ENVIRONMENT_VERSION, FORMAL_CONDITIONS, FORMAL_TASK_NAME, SCHEMA_VERSION
from phase3_13_runtime import close_env, deterministic_reset, install_minimal_ravens, seed_everything


def git(cwd: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def strict_dump(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def action_sha(actions: List[Dict[str, Any]]) -> tuple[str, List[np.ndarray]]:
    codec = ExecutableActionCodec(actions[0])
    vectors = [codec.encode(action).astype(np.float32) for action in actions]
    digest = hashlib.sha256(np.ascontiguousarray(np.stack(vectors)).tobytes()).hexdigest()
    return digest, vectors


def execute_free(root: Path, seed: int, pair_group: str, hz: int):
    tasks, Environment = install_minimal_ravens(root)
    seed_everything(seed)
    os.environ.update({"CCDA_HIDDEN_CONDITION": "free", "CCDA_VISIBLE_SEED": str(seed), "CCDA_PAIR_GROUP": pair_group})
    env = Environment(disp=False, hz=hz)
    task = tasks.names[FORMAL_TASK_NAME](); task.mode = "train"
    try:
        reset = deterministic_reset(env, task)
        info = reset["info"]
        oracle = task.oracle(env)
        env.start()
        obs = {}
        infos, actions = [], []
        done = False
        for _ in range(int(task.max_steps)):
            action = oracle.act(obs, info)
            if not action.get("primitive"):
                break
            infos.append(copy.deepcopy(info))
            actions.append(copy.deepcopy(action))
            obs, _, done, info = env.step(action)
            if done:
                break
        if not actions:
            raise RuntimeError(f"free oracle produced no actions for seed {seed}")
        return infos, actions, copy.deepcopy(info), bool(done), reset
    finally:
        close_env(env)


def execute_replay(root: Path, seed: int, pair_group: str, hz: int, actions: List[Dict[str, Any]]):
    tasks, Environment = install_minimal_ravens(root)
    seed_everything(seed)
    condition = "hidden_slack_breakaway_pin_v2"
    os.environ.update({"CCDA_HIDDEN_CONDITION": condition, "CCDA_VISIBLE_SEED": str(seed), "CCDA_PAIR_GROUP": pair_group})
    env = Environment(disp=False, hz=hz)
    task = tasks.names[FORMAL_TASK_NAME](); task.mode = "train"
    try:
        reset = deterministic_reset(env, task)
        info = reset["info"]
        env.start()
        infos = []
        done = False
        for action in actions:
            infos.append(copy.deepcopy(info))
            _, _, done, info = env.step(copy.deepcopy(action))
        return infos, list(actions), copy.deepcopy(info), bool(done), reset
    finally:
        close_env(env)


def generate_seed(payload: Dict[str, Any]) -> int:
    root = Path(payload["root"])
    split = payload["split"]
    seed = int(payload["seed"])
    output = Path(payload["output"])
    pair_group = f"phase313_{split}_seed_{seed}"
    free_infos, free_actions, free_last, free_success, free_reset = execute_free(root, seed, pair_group, payload["hz"])
    hidden_infos, hidden_actions, hidden_last, hidden_success, hidden_reset = execute_replay(root, seed, pair_group, payload["hz"], free_actions)
    if len(hidden_actions) != len(free_actions):
        raise RuntimeError("paired replay action count mismatch")
    digest, vectors = action_sha(free_actions)
    hidden_digest, _ = action_sha(hidden_actions)
    if digest != hidden_digest:
        raise RuntimeError("paired executable action sequences differ")
    initial_free = np.asarray(free_infos[0]["extras"]["bead_positions"], dtype=np.float64)
    initial_hidden = np.asarray(hidden_infos[0]["extras"]["bead_positions"], dtype=np.float64)
    if not np.array_equal(initial_free, initial_hidden):
        raise RuntimeError(f"paired initial geometry mismatch for seed {seed}")
    main_head = git(root, "rev-parse", "HEAD")
    sub_head = git(root / "external/deformable-ravens", "rev-parse", "HEAD")
    for condition, infos, actions, last_info, success, reset in [
        ("free", free_infos, free_actions, free_last, free_success, free_reset),
        ("hidden_slack_breakaway_pin_v2", hidden_infos, hidden_actions, hidden_last, hidden_success, hidden_reset),
    ]:
        manifest = {
            "environment_semantics_version": ENVIRONMENT_VERSION,
            "observation_schema_version": SCHEMA_VERSION,
            "task": FORMAL_TASK_NAME,
            "condition": condition,
            "visible_seed": seed,
            "pair_group": pair_group,
            "split": split,
            "action_source_condition": "free",
            "action_sequence_sha256": digest,
            "main_commit": main_head,
            "submodule_commit": sub_head,
            "contains_simulator_bead_velocity": False,
            "input_canonicalization": False,
            "num_actions": len(actions),
            "settle_steps": reset["settle_steps"],
        }
        episode = {
            "infos": infos,
            "last_info": last_info,
            "actions": actions,
            "action_vectors": np.stack(vectors[: len(actions)]).astype(np.float32),
            "success": success,
            "manifest": manifest,
        }
        directory = output / split / condition
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"seed_{seed}.pkl"
        temporary_episode = path.with_suffix(path.suffix + ".tmp")
        with temporary_episode.open("wb") as handle:
            pickle.dump(episode, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_episode, path)
        strict_dump(directory / f"seed_{seed}.manifest.json", manifest)
    return seed


def existing_seeds(output_root: Path) -> set[int]:
    found = set()
    for path in output_root.glob("**/*.manifest.json"):
        try:
            found.add(int(json.loads(path.read_text())["visible_seed"]))
        except Exception:
            continue
    return found


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/state_diff2")
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--seed-start", type=int, required=True)
    ap.add_argument("--num-seeds", type=int, required=True)
    ap.add_argument("--output-root", default="data/phase3_state_v2_slack/raw")
    ap.add_argument("--task", default=FORMAL_TASK_NAME)
    ap.add_argument("--conditions", nargs="+", default=list(FORMAL_CONDITIONS))
    ap.add_argument("--hz", type=int, default=480)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if os.environ.get("PHASE313_ALLOW_DATA_GENERATION") != "1" or os.environ.get("PHASE313_DATA_GENERATION_CONFIRMED") != "1":
        raise SystemExit("Phase3.13 data-generation gates are missing")
    if args.task != FORMAL_TASK_NAME or tuple(args.conditions) != FORMAL_CONDITIONS:
        raise SystemExit("formal task/condition surface mismatch")
    root = Path(args.root).resolve()
    output = root / args.output_root
    split_dir = output / args.split
    if args.fresh and split_dir.exists():
        shutil.rmtree(split_dir)
    orphan_temporary = sorted(output.glob("**/*.tmp"))
    if orphan_temporary:
        raise SystemExit(
            "orphan temporary files exist: "
            + ", ".join(str(path) for path in orphan_temporary[:10])
        )
    selected = list(range(args.seed_start, args.seed_start + args.num_seeds))
    overlap = set(selected).intersection(existing_seeds(output))
    if overlap:
        raise SystemExit(f"visible seeds already exist in formal raw root: {sorted(overlap)[:10]}")
    jobs = [{"root": str(root), "split": args.split, "seed": seed, "output": str(output), "hz": args.hz} for seed in selected]
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, args.workers), mp_context=context) as executor:
        futures = [executor.submit(generate_seed, job) for job in jobs]
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            seed = future.result()
            print(f"[Phase3.13] {args.split} seed {seed} ({index}/{len(jobs)})", flush=True)
    summary = {
        "split": args.split,
        "visible_seeds": selected,
        "num_visible_seeds": len(selected),
        "num_episodes": 2 * len(selected),
        "smoke": bool(args.smoke),
        "task": FORMAL_TASK_NAME,
        "conditions": list(FORMAL_CONDITIONS),
    }
    strict_dump(output / f"{args.split}_generation_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
