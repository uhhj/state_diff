#!/usr/bin/env python3
import sys
from pathlib import Path as _Phase3Path
ROOT = _Phase3Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from ccda_phase3.metrics import branch_stats, finite_mean, finite_std, state_chamfer
from ccda_phase3.train_utils import load_future_model, load_inverse_model


def row_summary(rows, keys):
    out = {}
    for key in keys:
        vals = [r.get(key) for r in rows]
        out[f"mean_{key}"] = finite_mean(vals)
        out[f"std_{key}"] = finite_std(vals)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt_root", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--samples_per_prefix", type=int, default=64)
    ap.add_argument("--diffusion_steps", type=int, default=100)
    args = ap.parse_args()

    data = np.load(args.data, allow_pickle=True)
    split = data["split_name"].astype(str)
    held = np.where(split == "heldout")[0]
    cond_name = data["condition_name"].astype(str)
    visible_seed = data["visible_seed"].astype(int)
    paper_x = data["paper_x"].astype(np.float32)
    state_action_x = data["state_action_x"].astype(np.float32)
    y_state = data["y_state"].astype(np.float32)
    y_flat = y_state.reshape(len(y_state), -1)
    y_final = data["y_final_state"].astype(np.float32)
    y_action = data["y_action"].astype(np.float32)
    n_beads = int(data["n_beads"])
    tf = int(data["tf"])
    state_dim = int(data["state_dim"])

    refs = defaultdict(dict)
    for i in held:
        refs[int(visible_seed[i])][str(cond_name[i])] = y_final[i]

    rows = []
    ckpt_root = Path(args.ckpt_root)
    for baseline_dir in sorted(ckpt_root.glob("*")):
        if not baseline_dir.is_dir():
            continue
        baseline = baseline_dir.name
        for ckpt in sorted(baseline_dir.glob("fold_*_seed_*")):
            cfg_path = ckpt / "config.json"
            if not cfg_path.exists():
                continue
            cfg = json.loads(cfg_path.read_text())
            fold = int(cfg["fold"])
            seed = int(cfg["seed"])
            state_model = load_future_model(ckpt / "state_model.pt")
            idm = load_inverse_model(ckpt / "inverse_dynamics.pt")
            x_all = paper_x if baseline == "paper_state" else state_action_x
            for i in held:
                x = x_all[i : i + 1]
                samples = state_model.sample(x, n_samples=args.samples_per_prefix, seed=seed + int(visible_seed[i]))[:, 0, :]
                samples_traj = samples.reshape(args.samples_per_prefix, tf, state_dim)
                samples_final = samples_traj[:, -1, :]
                mean_future = np.mean(samples, axis=0, keepdims=True)
                idm_x = np.concatenate([paper_x[i : i + 1], mean_future], axis=1)
                pred_action = idm.predict(idm_x)
                action_mse = float(np.mean((pred_action[0] - y_action[i]) ** 2))
                action_ood = float(idm.ood_score(pred_action)[0])
                ref = refs[int(visible_seed[i])]
                free_final = ref.get("free", y_final[i])
                pin_final = ref.get("hidden_pin", y_final[i])
                bs = branch_stats(samples_final, y_final[i], free_final, pin_final, str(cond_name[i]), n_beads)
                rows.append({
                    "baseline": baseline,
                    "fold": fold,
                    "seed": seed,
                    "condition": str(cond_name[i]),
                    "visible_seed": int(visible_seed[i]),
                    "source_file": str(data["source_file"][i]),
                    "window_t": int(data["window_t"][i]),
                    "primary_pair": "free_vs_hidden_pin",
                    "heldout": True,
                    "future_chamfer_to_true": bs["future_chamfer_to_true"],
                    "final_chamfer_to_true": state_chamfer(np.mean(samples_final, axis=0), y_final[i], n_beads),
                    "min_sample_chamfer_to_true": bs["min_sample_chamfer_to_true"],
                    "mean_prediction_chamfer_to_true": bs["mean_prediction_chamfer_to_true"],
                    "physical_violation_length": bs["physical_violation_length"],
                    "curve_error": bs["curve_error"],
                    "p_free_branch": bs["p_free_branch"],
                    "p_pin_branch": bs["p_pin_branch"],
                    "sample_wrong_branch_rate": bs["sample_wrong_branch_rate"],
                    "majority_wrong_branch": bs["majority_wrong_branch"],
                    "branch_entropy": bs["branch_entropy"],
                    "branch_accuracy": bs["branch_accuracy"],
                    "averaging_score": bs["averaging_score"],
                    "action_mse": action_mse,
                    "action_ood_score": action_ood,
                })

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with Path(args.out_csv).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    groups = defaultdict(list)
    for r in rows:
        groups[(r["baseline"], r["fold"], r["seed"], r["condition"])].append(r)
    metrics = ["sample_wrong_branch_rate", "branch_accuracy", "future_chamfer_to_true", "mean_prediction_chamfer_to_true", "averaging_score", "action_mse", "action_ood_score"]
    summary_rows = []
    for key, rs in sorted(groups.items()):
        item = {"baseline": key[0], "fold": key[1], "seed": key[2], "condition": key[3], "count": len(rs)}
        item.update(row_summary(rs, metrics))
        summary_rows.append(item)
    summary = {"num_prediction_rows": len(rows), "summary_rows": summary_rows}
    Path(args.out_json).write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps({"num_prediction_rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
