#!/usr/bin/env python3
import sys
from pathlib import Path as _Phase3Path
ROOT = _Phase3Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import json
from pathlib import Path

import numpy as np

from ccda_phase3.train_utils import grouped_folds, save_json, set_seed, train_numpy_future_model, train_numpy_inverse_model


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
    args = ap.parse_args()

    data = np.load(args.data, allow_pickle=True)
    split = data["split_name"].astype(str)
    train_mask = split == "train"
    train_seeds = sorted(set(data["visible_seed"][train_mask].astype(int).tolist()))
    folds = grouped_folds(train_seeds, args.folds)
    y_state = data["y_state"].astype(np.float32)
    y_flat = y_state.reshape(len(y_state), -1)
    paper_x = data["paper_x"].astype(np.float32)
    state_action_x = data["state_action_x"].astype(np.float32)
    y_action = data["y_action"].astype(np.float32)
    visible_seed = data["visible_seed"].astype(int)
    heldout_seeds = sorted(set(data["visible_seed"][split == "heldout"].astype(int).tolist()))
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    backend = "numpy_ridge_diffusion_surrogate"
    try:
        import torch  # noqa
        backend = "numpy_ridge_diffusion_surrogate_torch_available_not_used_for_smoke_speed"
    except Exception:
        pass

    for baseline in args.baselines:
        x_all = paper_x if baseline == "paper_state" else state_action_x
        for fold_idx, (fold_train_seeds, fold_val_seeds) in enumerate(folds):
            fold_train = train_mask & np.isin(visible_seed, fold_train_seeds)
            fold_val = train_mask & np.isin(visible_seed, fold_val_seeds)
            for seed in args.seeds:
                set_seed(seed)
                ckpt = out_root / baseline / f"fold_{fold_idx}_seed_{seed}"
                ckpt.mkdir(parents=True, exist_ok=True)
                state_cfg = {"backend": backend, "baseline": baseline, "seed": seed, "fold": fold_idx, "diffusion_steps": args.diffusion_steps, "epochs_state_requested": args.epochs_state}
                state_model = train_numpy_future_model(x_all[fold_train], y_state[fold_train], state_cfg)
                state_model.save(ckpt / "state_model.pt")
                idm_x = np.concatenate([paper_x[fold_train], y_flat[fold_train]], axis=1)
                idm_cfg = {"backend": "numpy_ridge_inverse_dynamics", "baseline": baseline, "seed": seed, "fold": fold_idx, "epochs_idm_requested": args.epochs_idm}
                idm = train_numpy_inverse_model(idm_x, y_action[fold_train], idm_cfg)
                idm.save(ckpt / "inverse_dynamics.pt")
                branch_means = {}
                for cond in ["free", "hidden_pin", "hidden_high_friction"]:
                    m = fold_train & (data["condition_name"].astype(str) == cond)
                    if np.any(m):
                        branch_means[cond] = np.mean(data["y_final_state"][m], axis=0).tolist()
                cfg = {
                    "baseline": baseline,
                    "fold": fold_idx,
                    "seed": seed,
                    "backend": backend,
                    "fold_train_seeds": [int(x) for x in fold_train_seeds],
                    "fold_val_seeds": [int(x) for x in fold_val_seeds],
                    "heldout_seeds": [int(x) for x in heldout_seeds],
                    "state_dim": int(data["state_dim"]),
                    "robot_pose_dim": int(data["robot_pose_dim"]),
                    "action_dim": int(data["action_dim"]),
                    "n_beads": int(data["n_beads"]),
                    "th": int(data["th"]),
                    "tf": int(data["tf"]),
                    "paper_x_dim": int(paper_x.shape[1]),
                    "state_action_x_dim": int(state_action_x.shape[1]),
                    "y_dim": int(y_flat.shape[1]),
                    "action_template_json_or_pickle_path": str(data["action_template_json_or_pickle_path"]),
                    "branch_mean_final_state": branch_means,
                }
                save_json(ckpt / "config.json", cfg)
                val_pred = state_model.predict_mean(x_all[fold_val]) if np.any(fold_val) else np.zeros((0, y_flat.shape[1]))
                val_mse = float(np.mean((val_pred - y_flat[fold_val]) ** 2)) if np.any(fold_val) else None
                train_log = {"state_val_mse": val_mse, "num_train_windows": int(np.sum(fold_train)), "num_val_windows": int(np.sum(fold_val))}
                save_json(ckpt / "train_log.json", train_log)
                print("[Phase3] trained", ckpt, train_log)


if __name__ == "__main__":
    main()
