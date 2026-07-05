#!/usr/bin/env python3
import sys
from pathlib import Path as _Phase3Path
ROOT = _Phase3Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import json
import os
from pathlib import Path

import numpy as np

from ccda_phase3.data_io import normalize_conditions
from ccda_phase3.train_utils import (
    grouped_folds,
    save_json,
    set_seed,
    train_torch_ddpm_future_model,
    train_torch_inverse_model,
)


def infer_conda_env() -> str:
    env = os.environ.get("CONDA_DEFAULT_ENV", "")
    if env:
        return env
    parts = Path(sys.executable).parts
    if "envs" in parts:
        i = parts.index("envs")
        if i + 1 < len(parts):
            return parts[i + 1]
    return ""


def require_torch_or_fail() -> None:
    try:
        import torch  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "PyTorch is required for DDPM-aligned Phase3 training. "
            "Activate coord_bimanual; numpy_fallback is not allowed for DDPM Phase3."
        ) from exc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--baselines", nargs="+", default=["paper_state", "state_action"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs_state", type=int, default=1000)
    ap.add_argument("--epochs_idm", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--diffusion_steps", type=int, default=100)
    ap.add_argument("--hidden_dim", type=int, default=512)
    ap.add_argument("--time_dim", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--beta_schedule", default="squaredcos_cap_v2")
    ap.add_argument("--prediction_type", default="epsilon")
    ap.add_argument("--variance_type", default="fixed_small")
    ap.add_argument("--clip_sample", action="store_true", default=True)
    ap.add_argument("--no_clip_sample", dest="clip_sample", action="store_false")
    ap.add_argument("--num_inference_steps", type=int, default=None)
    ap.add_argument("--sample_temperature", type=float, default=1.0)
    ap.add_argument("--denoiser_arch", default="mlp", choices=["mlp"])
    ap.add_argument("--conditions", nargs="+", default=None)
    ap.add_argument("--primary_hidden_condition", default=None)
    ap.add_argument("--diagnostic_hidden_condition", default=None)
    args = ap.parse_args()

    if args.variance_type == "learned_range":
        raise SystemExit(
            "learned_range requires 2*y_dim output and is not enabled in pre-medium MLP denoiser. "
            "Use fixed_small for scheduler-aligned medium, or implement learned_range in a separate commit."
        )

    require_torch_or_fail()
    training_backend = "torch"
    python_executable = sys.executable
    conda_env = infer_conda_env()

    data = np.load(args.data, allow_pickle=True)
    meta = json.loads(str(data["meta_json"]))
    conditions = normalize_conditions(args.conditions or meta.get("conditions"))
    primary_hidden = args.primary_hidden_condition or meta.get("primary_hidden_condition", "hidden_breakaway_pin")
    diagnostic_hidden = args.diagnostic_hidden_condition or meta.get("diagnostic_hidden_condition", "hidden_pin")
    if primary_hidden not in conditions:
        raise SystemExit(f"[Phase3][FAIL] primary_hidden_condition={primary_hidden} not in conditions={conditions}")
    if diagnostic_hidden not in conditions:
        raise SystemExit(f"[Phase3][FAIL] diagnostic_hidden_condition={diagnostic_hidden} not in conditions={conditions}")
    split = data["split_name"].astype(str)
    train_mask = split == "train"
    train_seeds = sorted(set(data["visible_seed"][train_mask].astype(int).tolist()))
    folds = grouped_folds(train_seeds, args.folds)
    y_state = data["y_state"].astype(np.float32)
    y_flat = y_state.reshape(len(y_state), -1)
    paper_x = data["paper_x"].astype(np.float32)
    state_action_x = data["state_action_x"].astype(np.float32)
    y_action = data["y_action"].astype(np.float32)
    idm_x_all = np.concatenate([paper_x, y_flat], axis=1).astype(np.float32)
    visible_seed = data["visible_seed"].astype(int)
    heldout_seeds = sorted(set(data["visible_seed"][split == "heldout"].astype(int).tolist()))
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for baseline in args.baselines:
        x_all = paper_x if baseline == "paper_state" else state_action_x
        for fold_idx, (fold_train_seeds, fold_val_seeds) in enumerate(folds):
            fold_train = train_mask & np.isin(visible_seed, fold_train_seeds)
            fold_val = train_mask & np.isin(visible_seed, fold_val_seeds)
            for seed in args.seeds:
                set_seed(seed)
                ckpt = out_root / baseline / f"fold_{fold_idx}_seed_{seed}"
                ckpt.mkdir(parents=True, exist_ok=True)
                state_cfg = {
                    "training_backend": training_backend,
                    "backend": training_backend,
                    "baseline": baseline,
                    "seed": seed,
                    "fold": fold_idx,
                    "future_model_type": "torch_conditional_ddpm_future_state",
                    "ddpm_used": True,
                    "scheduler_type": "diffusers.DDPMScheduler",
                    "beta_schedule": args.beta_schedule,
                    "prediction_type": args.prediction_type,
                    "variance_type": args.variance_type,
                    "clip_sample": bool(args.clip_sample),
                    "diffusion_steps": args.diffusion_steps,
                    "num_train_timesteps": int(args.diffusion_steps),
                    "num_inference_steps": int(args.num_inference_steps or args.diffusion_steps),
                    "sample_temperature": float(args.sample_temperature),
                    "denoiser_arch": args.denoiser_arch,
                    "conditional_unet1d_used": False,
                    "paper_alignment_level": "ddpm_scheduler_aligned_mlp_denoiser",
                    "epochs_state_requested": args.epochs_state,
                    "hidden_dim": args.hidden_dim,
                    "time_dim": args.time_dim,
                    "lr": args.lr,
                    "python_executable": python_executable,
                    "conda_env": conda_env,
                    "conditions": conditions,
                    "primary_hidden_condition": primary_hidden,
                    "diagnostic_hidden_condition": diagnostic_hidden,
                    "primary_branch_pair": f"free_vs_{primary_hidden}",
                    "diagnostic_branch_pair": f"free_vs_{diagnostic_hidden}",
                }
                idm_cfg = {
                    "training_backend": training_backend,
                    "backend": training_backend,
                    "baseline": baseline,
                    "seed": seed,
                    "fold": fold_idx,
                    "epochs_idm_requested": args.epochs_idm,
                    "idm_feature_mode": "paper_full_state_history_future",
                    "idm_x_dim": int(idm_x_all.shape[1]),
                    "idm_x_std_floor": 0.1,
                    "idm_hidden_dim": 64,
                    "python_executable": python_executable,
                    "conda_env": conda_env,
                    "conditions": conditions,
                    "primary_hidden_condition": primary_hidden,
                    "diagnostic_hidden_condition": diagnostic_hidden,
                    "primary_branch_pair": f"free_vs_{primary_hidden}",
                    "diagnostic_branch_pair": f"free_vs_{diagnostic_hidden}",
                }
                state_model = train_torch_ddpm_future_model(
                    x_all[fold_train],
                    y_state[fold_train],
                    state_cfg,
                    args.epochs_state,
                    args.batch_size,
                    seed,
                )
                idm = train_torch_inverse_model(idm_x_all[fold_train], y_action[fold_train], idm_cfg, args.epochs_idm, args.batch_size, seed)
                state_model.save(ckpt / "state_model.pt")
                idm.save(ckpt / "inverse_dynamics.pt")

                branch_means = {}
                for cond in conditions:
                    m = fold_train & (data["condition_name"].astype(str) == cond)
                    if np.any(m):
                        branch_means[cond] = np.mean(data["y_final_state"][m], axis=0).tolist()
                cfg = {
                    "baseline": baseline,
                    "fold": fold_idx,
                    "seed": seed,
                    "backend": training_backend,
                    "training_backend": training_backend,
                    "future_model_type": "torch_conditional_ddpm_future_state",
                    "ddpm_used": True,
                    "scheduler_type": "diffusers.DDPMScheduler",
                    "beta_schedule": args.beta_schedule,
                    "prediction_type": args.prediction_type,
                    "variance_type": args.variance_type,
                    "clip_sample": bool(args.clip_sample),
                    "diffusion_steps": args.diffusion_steps,
                    "num_train_timesteps": int(args.diffusion_steps),
                    "num_inference_steps": int(args.num_inference_steps or args.diffusion_steps),
                    "sample_temperature": float(args.sample_temperature),
                    "denoiser_arch": args.denoiser_arch,
                    "conditional_unet1d_used": False,
                    "paper_alignment_level": "ddpm_scheduler_aligned_mlp_denoiser",
                    "python_executable": python_executable,
                    "conda_env": conda_env,
                    "fold_train_seeds": [int(x) for x in fold_train_seeds],
                    "fold_val_seeds": [int(x) for x in fold_val_seeds],
                    "heldout_seeds": [int(x) for x in heldout_seeds],
                    "state_dim": int(data["state_dim"]),
                    "robot_pose_dim": int(data["robot_pose_dim"]),
                    "action_dim": int(data["action_dim"]),
                    "idm_feature_mode": "paper_full_state_history_future",
                    "idm_x_dim": int(idm_x_all.shape[1]),
                    "n_beads": int(data["n_beads"]),
                    "th": int(data["th"]),
                    "tf": int(data["tf"]),
                    "paper_x_dim": int(paper_x.shape[1]),
                    "state_action_x_dim": int(state_action_x.shape[1]),
                    "y_dim": int(y_flat.shape[1]),
                    "action_template_json_or_pickle_path": str(data["action_template_json_or_pickle_path"]),
                    "branch_mean_final_state": branch_means,
                    "conditions": conditions,
                    "primary_hidden_condition": primary_hidden,
                    "diagnostic_hidden_condition": diagnostic_hidden,
                    "primary_branch_pair": f"free_vs_{primary_hidden}",
                    "diagnostic_branch_pair": f"free_vs_{diagnostic_hidden}",
                }
                save_json(ckpt / "config.json", cfg)
                val_pred = state_model.predict_mean(x_all[fold_val], n_samples=8, seed=seed) if np.any(fold_val) else np.zeros((0, y_flat.shape[1]))
                val_mse = float(np.mean((val_pred - y_flat[fold_val]) ** 2)) if np.any(fold_val) else None
                train_log = {
                    "training_backend": training_backend,
                    "future_model_type": "torch_conditional_ddpm_future_state",
                    "ddpm_used": True,
                    "python_executable": python_executable,
                    "conda_env": conda_env,
                    "state_val_mse": val_mse,
                    "num_train_windows": int(np.sum(fold_train)),
                    "num_val_windows": int(np.sum(fold_val)),
                    "idm_feature_mode": "paper_full_state_history_future",
                    "idm_x_dim": int(idm_x_all.shape[1]),
                    "conditions": conditions,
                    "primary_hidden_condition": primary_hidden,
                    "diagnostic_hidden_condition": diagnostic_hidden,
                    "primary_branch_pair": f"free_vs_{primary_hidden}",
                    "diagnostic_branch_pair": f"free_vs_{diagnostic_hidden}",
                }
                save_json(ckpt / "train_log.json", train_log)
                print("[Phase3] trained", ckpt, train_log)


if __name__ == "__main__":
    main()
