"""PB2-C REV1 natural rollout collection for DLO-Lab Wrapping.

Only repository-provided Wrapping position randomization is activated.
The common official best_qpos sequence is replayed for every rollout.

Crucially, the collector mirrors the published Wrapping eval_traj rollout-
validity semantics. Failed rollouts remain in raw storage but are excluded from
PB2-C pair mining.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch

from scripts.experiment3.dlolab_wrapping.replay_winding_audit import (
    _dual_arm_command,
    build_env,
    load_json,
    normalize_qpos,
    sample_env,
)
from scripts.experiment3.dlolab_wrapping.paths import official_log_dir
from utils.domain_randomization import wrapping_args  # noqa: E402


def _to_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _jsonable(value):
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    return value


def validate_source(config):
    evidence_path = Path(config["source"]["pb2ab_evidence"])
    if not evidence_path.is_file():
        raise FileNotFoundError(str(evidence_path))

    evidence = load_json(evidence_path)
    actual = evidence.get("verdict")
    expected = config["source"]["expected_pb2ab_verdict"]
    if actual != expected:
        raise RuntimeError(f"PB2-A/B verdict mismatch: {actual} != {expected}")

    expected_args = config["collection"]["expected_wrapping_args"]
    actual_args = _jsonable(dict(wrapping_args))
    if actual_args != expected_args:
        raise RuntimeError(
            f"Pinned wrapping_args mismatch: {actual_args} != {expected_args}"
        )


def _stack_time(rows, key):
    return np.stack([row[key] for row in rows], axis=1)


def _initial_validity(env, total_micro_steps):
    """Mirror the pre-step NaN portion of published Wrapping eval_traj."""
    rope = _to_numpy(env.rope.get_all_verts())
    initial_nan = np.isnan(rope).any(axis=(1, 2))

    alive = ~initial_nan
    first_fail_step = np.full(
        env.n_envs,
        int(total_micro_steps),
        dtype=np.int32,
    )
    # Published eval_traj clamps pre-step failure to at least step 1.
    first_fail_step[initial_nan] = 1

    return {
        "alive": alive,
        "first_fail_step": first_fail_step,
        "failed_rope_nan": initial_nan.copy(),
        "failed_stretch": np.zeros(env.n_envs, dtype=bool),
        "control_dist_init": _to_numpy(env.control_dist_init).reshape(env.n_envs),
        "max_stretch_ratio": np.zeros(env.n_envs, dtype=np.float32),
    }


def _update_official_validity(
        env,
        validity,
        *,
        global_step,
        stretch_ratio_limit):
    """Mirror published post-microstep stretch and rope-NaN failure checks."""
    alive = validity["alive"]

    control_dist_now = _to_numpy(
        env.rope.get_geodesic_distance(
            env.control_idx[0],
            env.control_idx[1],
        )
    ).reshape(env.n_envs)

    ratio = control_dist_now / validity["control_dist_init"]
    validity["max_stretch_ratio"] = np.maximum(
        validity["max_stretch_ratio"],
        ratio.astype(np.float32),
    )

    stretched = ratio > float(stretch_ratio_limit)
    newly_stretched = stretched & alive
    if newly_stretched.any():
        validity["first_fail_step"][newly_stretched] = np.minimum(
            validity["first_fail_step"][newly_stretched],
            int(global_step),
        )
        validity["failed_stretch"][newly_stretched] = True
        alive[newly_stretched] = False

    rope = _to_numpy(env.rope.get_all_verts())
    nan_after = np.isnan(rope).any(axis=(1, 2))
    newly_nan = nan_after & alive
    if newly_nan.any():
        validity["first_fail_step"][newly_nan] = np.minimum(
            validity["first_fail_step"][newly_nan],
            int(global_step),
        )
        validity["failed_rope_nan"][newly_nan] = True
        alive[newly_nan] = False


def replay_batch(env, qpos, seed, stretch_ratio_limit):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))

    env.use_qpos = True
    env.reset()

    total_micro_steps = int(qpos.shape[0] - 1)
    validity = _initial_validity(env, total_micro_steps)
    rows = [sample_env(env)]

    n_intervals = env.steps_interval // env._cmaes_n_steps_sub
    if n_intervals <= 0:
        raise RuntimeError("Invalid Wrapping replay interval count")

    # best_qpos has one command per CMA-ES microstep after the initial state.
    for global_step, command in enumerate(qpos[1:], start=1):
        _dual_arm_command(env, command)
        for _ in range(n_intervals):
            env.scene.step()

        _update_official_validity(
            env,
            validity,
            global_step=global_step,
            stretch_ratio_limit=stretch_ratio_limit,
        )
        rows.append(sample_env(env))

    # Published eval_traj clamps a surviving final NaN reward to -100. Treat
    # that rollout as invalid for pair discovery as well.
    final_reward = np.asarray(rows[-1]["reward"], dtype=np.float64)
    final_reward_nan = np.isnan(final_reward)
    official_rollout_valid = validity["alive"] & ~final_reward_nan

    merged = {key: _stack_time(rows, key) for key in rows[0]}
    merged.update(
        {
            "official_rollout_valid": official_rollout_valid.astype(bool),
            "official_first_fail_step": validity["first_fail_step"].astype(np.int32),
            "official_failed_stretch": validity["failed_stretch"].astype(bool),
            "official_failed_rope_nan": validity["failed_rope_nan"].astype(bool),
            "official_final_reward_nan": final_reward_nan.astype(bool),
            "official_max_stretch_ratio": validity["max_stretch_ratio"].astype(np.float32),
        }
    )

    # Failed rollouts may legitimately contain NaNs. Official-valid rollouts
    # must be finite on all fields used by the PB2-C selector.
    valid = official_rollout_valid
    for key in (
        "rope_xyz",
        "post_xyz",
        "ee1_pos",
        "ee1_quat",
        "ee2_pos",
        "ee2_quat",
        "motor_qpos_1",
        "motor_qpos_2",
        "signed_winding_turns",
    ):
        values = merged[key][valid]
        if values.size and not np.isfinite(values).all():
            raise RuntimeError(
                f"Official-valid rollout has non-finite selection data: {key}"
            )

    return merged


def run(config_path, start_batch, batches, output):
    config = load_json(config_path)
    validate_source(config)

    log_dir = official_log_dir()
    qpos_path = log_dir / "best_qpos.npy"
    if not qpos_path.is_file():
        raise FileNotFoundError(str(qpos_path))
    qpos = normalize_qpos(np.load(qpos_path))

    n_envs = int(config["collection"]["n_envs"])
    env = build_env(
        n_envs=n_envs,
        n_steps_sub=config["collection"]["n_steps_sub"],
        log_dir=log_dir,
    )

    if config["collection"]["activate_repo_wrapping_position_randomization"]:
        env.init_domain_randomization(**wrapping_args)

    batch_rows = []
    rollout_ids = []
    batch_indices = []
    env_indices = []
    seeds = []

    try:
        for batch_index in range(int(start_batch), int(start_batch) + int(batches)):
            seed = int(config["collection"]["base_seed"]) + batch_index
            row = replay_batch(
                env,
                qpos,
                seed,
                config["collection"]["official_stretch_ratio_limit"],
            )
            batch_rows.append(row)

            for env_index in range(n_envs):
                rollout_ids.append(batch_index * n_envs + env_index)
                batch_indices.append(batch_index)
                env_indices.append(env_index)
                seeds.append(seed)
    finally:
        env.stop()

    merged = {
        key: np.concatenate([batch[key] for batch in batch_rows], axis=0)
        for key in batch_rows[0]
    }
    merged["rollout_id"] = np.asarray(rollout_ids, dtype=np.int32)
    merged["batch_index"] = np.asarray(batch_indices, dtype=np.int16)
    merged["env_index"] = np.asarray(env_indices, dtype=np.int16)
    merged["batch_seed"] = np.asarray(seeds, dtype=np.int32)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(output.stem + ".tmp.npz")
    np.savez_compressed(temp, **merged, common_qpos_replay=qpos)
    temp.replace(output)

    valid = merged["official_rollout_valid"].astype(bool)
    metadata = {
        "output": str(output),
        "start_batch": int(start_batch),
        "batches": int(batches),
        "n_envs": n_envs,
        "rollout_count": int(len(rollout_ids)),
        "official_valid_rollout_count": int(valid.sum()),
        "official_invalid_rollout_count": int((~valid).sum()),
        "official_failed_stretch_count": int(merged["official_failed_stretch"].sum()),
        "official_failed_rope_nan_count": int(merged["official_failed_rope_nan"].sum()),
        "official_final_reward_nan_count": int(merged["official_final_reward_nan"].sum()),
        "time_samples": int(merged["rope_xyz"].shape[1]),
        "base_seed": int(config["collection"]["base_seed"]),
        "batch_seeds": sorted(set(int(v) for v in seeds)),
        "wrapping_args": _jsonable(dict(wrapping_args)),
        "common_qpos_shape": list(qpos.shape),
        "official_rollout_validity": {
            "stretch_test": (
                "geodesic(control_idx[0], control_idx[1]) / "
                "control_dist_init > 1.2"
            ),
            "rope_nan_failure": True,
            "final_reward_nan_invalid": True,
        },
        "benchmark_physics_modified": False,
    }

    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"rollouts={output}")
    print(f"count={len(rollout_ids)}")
    print(f"official_valid={metadata['official_valid_rollout_count']}")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--start-batch", type=int, required=True)
    parser.add_argument("--batches", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.config, args.start_batch, args.batches, args.output)


if __name__ == "__main__":
    main()
