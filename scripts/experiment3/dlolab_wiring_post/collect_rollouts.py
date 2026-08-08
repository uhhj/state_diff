from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.experiment3.dlolab_wiring_post.paths import (
    add_dlolab_to_path,
    official_log_dir,
    required_asset_path,
)

add_dlolab_to_path()

import genesis as gs  # noqa: E402
from omegaconf import DictConfig  # noqa: E402
from envs.env_wiring_post import Train_Env_Wiring_post  # noqa: E402
from utils.domain_randomization import wiring_post_args  # noqa: E402


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def to_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def normalize_qpos(qpos):
    qpos = np.asarray(qpos, dtype=np.float32)
    if qpos.ndim == 3:
        qpos = qpos[0]
    if qpos.ndim != 2:
        raise ValueError(
            "best_qpos.npy must be [T,D] or [1,T,D]")
    return qpos


def build_env(n_envs, n_steps_sub, log_dir):
    cfg = DictConfig({
        "task": "wiring_post",
        "log_dir": str(log_dir),
        "n_envs": int(n_envs),
        "GUI": False,
        "camera": False,
        "raytracer": False,
        "requires_grad": False,
    })
    env = Train_Env_Wiring_post(config=cfg)
    env.init_cmaes_env(n_steps_sub=int(n_steps_sub))
    return env


def sample_env(env):
    rope_xyz = to_numpy(env.rope.get_all_verts()).astype(np.float32)
    rope_vel = to_numpy(env.rope.get_all_vels_tc()).astype(np.float32)
    ee_pos = to_numpy(env.c1.ef.get_pos()).astype(np.float32)
    ee_quat = to_numpy(env.c1.ef.get_quat()).astype(np.float32)
    motor_qpos = to_numpy(
        env.c1.robot.get_dofs_position(env.c1.motors_dof)
    ).astype(np.float32)
    motor_force = to_numpy(
        env.c1.robot.get_dofs_force(env.c1.motors_dof)
    ).astype(np.float32)
    motor_control_force = to_numpy(
        env.c1.robot.get_dofs_control_force(env.c1.motors_dof)
    ).astype(np.float32)
    reward = np.asarray(env.reward(), dtype=np.float32)
    stick1 = to_numpy(env.stick1.get_pos()).astype(np.float32)
    stick2 = to_numpy(env.stick2.get_pos()).astype(np.float32)
    post_xyz = np.stack([stick1, stick2], axis=1)
    return {
        "rope_xyz": rope_xyz,
        "rope_vel": rope_vel,
        "ee_pos": ee_pos,
        "ee_quat": ee_quat,
        "motor_qpos": motor_qpos,
        "motor_force": motor_force,
        "motor_control_force": motor_control_force,
        "reward": reward,
        "post_xyz": post_xyz,
    }


def stack_time(rows, key):
    return np.stack([row[key] for row in rows], axis=1)


def replay_batch(env, qpos, *, seed):
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    env.reset()
    rows = [sample_env(env)]
    n_intervals = env.steps_interval // env._cmaes_n_steps_sub

    for command in qpos[1:]:
        command_batch = np.repeat(
            command[None, :], env.n_envs, axis=0)
        command_tc = torch.tensor(command_batch, dtype=gs.tc_float)
        env.c1.robot.control_dofs_position(
            command_tc[..., :-2], env.c1.motors_dof)
        env.c1.robot.control_dofs_position(
            command_tc[..., -2:], env.c1.fingers_dof)
        for _ in range(n_intervals):
            env.scene.step()
        rows.append(sample_env(env))

    return {
        key: stack_time(rows, key)
        for key in rows[0]
    }


def run(config_path, *, batches_override=None):
    config = load_json(config_path)
    asset = required_asset_path()
    if not asset.is_file():
        raise FileNotFoundError(str(asset))

    log_dir = official_log_dir()
    qpos_path = log_dir / "best_qpos.npy"
    traj_path = log_dir / "best_traj.npy"
    if not qpos_path.is_file():
        raise FileNotFoundError(str(qpos_path))
    if not traj_path.is_file():
        raise FileNotFoundError(str(traj_path))

    qpos = normalize_qpos(np.load(qpos_path))
    n_envs = int(config["collection"]["n_envs"])
    batches = int(
        config["collection"]["batches"]
        if batches_override is None
        else batches_override)
    env = build_env(
        n_envs,
        config["collection"]["n_steps_sub"],
        log_dir,
    )
    if config["collection"][
            "activate_repo_wiring_post_position_randomization"]:
        env.init_domain_randomization(**wiring_post_args)

    batch_rows = []
    try:
        for batch in range(batches):
            seed = int(config["collection"]["base_seed"]) + batch
            batch_rows.append(replay_batch(env, qpos, seed=seed))
    finally:
        env.stop()

    merged = {
        key: np.concatenate(
            [batch[key] for batch in batch_rows], axis=0)
        for key in batch_rows[0]
    }
    if not all(np.isfinite(value).all() for value in merged.values()):
        raise ValueError("PB0 replay contains non-finite values")

    raw_root = Path(config["outputs"]["raw_root"])
    raw_root.mkdir(parents=True, exist_ok=True)
    output = raw_root / (
        "rollouts.npz"
        if batches_override is None
        else "rollouts_scaleup.npz")
    np.savez_compressed(output, **merged, common_qpos_replay=qpos)

    metadata = {
        "n_envs_per_batch": n_envs,
        "batches": batches,
        "num_rollouts": int(n_envs * batches),
        "time_samples": int(merged["rope_xyz"].shape[1]),
        "finite_replay_reward": bool(
            np.isfinite(merged["reward"]).all()),
        "official_best_qpos": str(qpos_path),
        "official_best_traj": str(traj_path),
        "position_randomization": dict(wiring_post_args),
    }
    metadata_path = raw_root / (
        "collection_metadata.json"
        if batches_override is None
        else "collection_metadata_scaleup.json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--batches", type=int, default=None)
    args = parser.parse_args()
    output = run(args.config, batches_override=args.batches)
    print("rollouts={}".format(output))


if __name__ == "__main__":
    main()
