#!/usr/bin/env python3
import sys
from pathlib import Path as _Phase3Path
ROOT = _Phase3Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ccda_phase3.metrics import branch_stats, finite_mean, finite_std, state_chamfer
from ccda_phase3.train_utils import load_future_model, load_inverse_model


BRANCH_REFERENCE_MODE = "split_visible_seed_window_t"
DDPM_MODEL_TYPE = "torch_conditional_ddpm_future_state"
SCHEDULER_TYPE = "diffusers.DDPMScheduler"
BETA_SCHEDULE = "squaredcos_cap_v2"
PREDICTION_TYPE = "epsilon"
VARIANCE_TYPE = "fixed_small"
DENOISER_ARCH = "mlp"
PAPER_ALIGNMENT_LEVEL = "ddpm_scheduler_aligned_mlp_denoiser"


def row_summary(rows, keys):
    out = {}
    for key in keys:
        vals = [r.get(key) for r in rows]
        out[f"mean_{key}"] = finite_mean(vals)
        out[f"std_{key}"] = finite_std(vals)
    return out


def missing_branch_stats(samples_final, true_final, n_beads):
    mean_final = np.mean(np.asarray(samples_final, dtype=np.float32), axis=0)
    return {
        "future_chamfer_to_true": float(np.mean([state_chamfer(s, true_final, n_beads) for s in samples_final])),
        "final_chamfer_to_true": state_chamfer(mean_final, true_final, n_beads),
        "min_sample_chamfer_to_true": float(np.min([state_chamfer(s, true_final, n_beads) for s in samples_final])),
        "mean_prediction_chamfer_to_true": state_chamfer(mean_final, true_final, n_beads),
        "p_free_branch": float("nan"),
        "p_pin_branch": float("nan"),
        "sample_wrong_branch_rate": float("nan"),
        "majority_wrong_branch": float("nan"),
        "branch_entropy": float("nan"),
        "branch_accuracy": float("nan"),
        "averaging_score": float("nan"),
        "physical_violation_length": float("nan"),
        "curve_error": float("nan"),
    }


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
    window_t = data["window_t"].astype(int)
    paper_x = data["paper_x"].astype(np.float32)
    state_action_x = data["state_action_x"].astype(np.float32)
    y_state = data["y_state"].astype(np.float32)
    y_final = data["y_final_state"].astype(np.float32)
    y_action = data["y_action"].astype(np.float32)
    n_beads = int(data["n_beads"])
    tf = int(data["tf"])
    state_dim = int(data["state_dim"])

    refs = defaultdict(dict)
    for i in held:
        key = (str(split[i]), int(visible_seed[i]), int(window_t[i]))
        refs[key][str(cond_name[i])] = y_final[i]

    rows = []
    num_missing_primary_branch_refs = 0
    num_valid_primary_branch_refs = 0
    model_types = Counter()
    ddpm_flags = Counter()
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
            future_model_type = str(cfg.get("future_model_type", ""))
            ddpm_used = bool(cfg.get("ddpm_used", False))
            model_types[future_model_type] += 1
            ddpm_flags[str(ddpm_used).lower()] += 1
            if future_model_type != DDPM_MODEL_TYPE or not ddpm_used:
                raise SystemExit("[Phase3][FAIL] Non-DDPM future model found in DDPM-aligned Phase3 eval.")
            fold = int(cfg["fold"])
            seed = int(cfg["seed"])
            training_backend = str(cfg.get("training_backend", cfg.get("backend", "unknown")))
            if training_backend != "torch":
                raise SystemExit("[Phase3][FAIL] Non-torch backend found in DDPM-aligned Phase3 eval.")
            if str(cfg.get("scheduler_type", "")) != SCHEDULER_TYPE:
                raise SystemExit("[Phase3][FAIL] Phase3 eval requires scheduler_type=diffusers.DDPMScheduler.")
            if str(cfg.get("beta_schedule", "")) != BETA_SCHEDULE:
                raise SystemExit("[Phase3][FAIL] Phase3 eval requires beta_schedule=squaredcos_cap_v2.")
            if str(cfg.get("prediction_type", "")) != PREDICTION_TYPE:
                raise SystemExit("[Phase3][FAIL] Phase3 eval requires prediction_type=epsilon.")
            python_executable = str(cfg.get("python_executable", ""))
            conda_env = str(cfg.get("conda_env", ""))
            state_model = load_future_model(ckpt / "state_model.pt")
            idm = load_inverse_model(ckpt / "inverse_dynamics.pt")
            x_all = paper_x if baseline == "paper_state" else state_action_x
            for i in held:
                x = x_all[i : i + 1]
                eval_sample_seed = int(seed + int(visible_seed[i]))
                samples = state_model.sample(x, n_samples=args.samples_per_prefix, seed=eval_sample_seed)[:, 0, :]
                samples_traj = samples.reshape(args.samples_per_prefix, tf, state_dim)
                samples_final = samples_traj[:, -1, :]
                mean_future = np.mean(samples, axis=0, keepdims=True)
                idm_x = np.concatenate([paper_x[i : i + 1], mean_future], axis=1)
                pred_action = idm.predict(idm_x)
                action_mse = float(np.mean((pred_action[0] - y_action[i]) ** 2))
                action_ood = float(idm.ood_score(pred_action)[0])

                ref_key = (str(split[i]), int(visible_seed[i]), int(window_t[i]))
                ref = refs.get(ref_key, {})
                has_free_ref = "free" in ref
                has_pin_ref = "hidden_pin" in ref
                if has_free_ref and has_pin_ref:
                    num_valid_primary_branch_refs += 1
                    bs = branch_stats(samples_final, y_final[i], ref["free"], ref["hidden_pin"], str(cond_name[i]), n_beads)
                else:
                    num_missing_primary_branch_refs += 1
                    bs = missing_branch_stats(samples_final, y_final[i], n_beads)

                rows.append({
                    "baseline": baseline,
                    "fold": fold,
                    "seed": seed,
                    "training_backend": training_backend,
                    "python_executable": python_executable,
                    "conda_env": conda_env,
                    "future_model_type": future_model_type,
                    "ddpm_used": ddpm_used,
                    "scheduler_type": str(cfg.get("scheduler_type", "")),
                    "beta_schedule": str(cfg.get("beta_schedule", "")),
                    "prediction_type": str(cfg.get("prediction_type", "")),
                    "variance_type": str(cfg.get("variance_type", "")),
                    "clip_sample": cfg.get("clip_sample", ""),
                    "num_train_timesteps": cfg.get("num_train_timesteps", ""),
                    "num_inference_steps": cfg.get("num_inference_steps", ""),
                    "sample_temperature": cfg.get("sample_temperature", ""),
                    "denoiser_arch": str(cfg.get("denoiser_arch", "")),
                    "conditional_unet1d_used": cfg.get("conditional_unet1d_used", False),
                    "paper_alignment_level": str(cfg.get("paper_alignment_level", "")),
                    "eval_sample_seed_mode": "paired_shared_visible_seed",
                    "eval_sample_seed": eval_sample_seed,
                    "idm_feature_mode": str(cfg.get("idm_feature_mode", "")),
                    "branch_reference_mode": BRANCH_REFERENCE_MODE,
                    "ref_key": f"{split[i]}_{visible_seed[i]}_{window_t[i]}",
                    "has_free_ref": bool(has_free_ref),
                    "has_pin_ref": bool(has_pin_ref),
                    "condition": str(cond_name[i]),
                    "visible_seed": int(visible_seed[i]),
                    "source_file": str(data["source_file"][i]),
                    "window_t": int(window_t[i]),
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
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    groups = defaultdict(list)
    for r in rows:
        groups[(r["baseline"], r["fold"], r["seed"], r["condition"], r["training_backend"])].append(r)
    metrics = ["sample_wrong_branch_rate", "branch_accuracy", "future_chamfer_to_true", "mean_prediction_chamfer_to_true", "averaging_score", "action_mse", "action_ood_score"]
    summary_rows = []
    for key, rs in sorted(groups.items()):
        item = {"baseline": key[0], "fold": key[1], "seed": key[2], "condition": key[3], "training_backend": key[4], "count": len(rs)}
        item.update(row_summary(rs, metrics))
        summary_rows.append(item)
    backends = Counter([r["training_backend"] for r in rows])
    future_model_types_seen = Counter([r["future_model_type"] for r in rows])
    ddpm_seen = Counter([str(r["ddpm_used"]).lower() for r in rows])
    scheduler_type_counts = Counter([str(r.get("scheduler_type", "")) for r in rows])
    beta_schedule_counts = Counter([str(r.get("beta_schedule", "")) for r in rows])
    prediction_type_counts = Counter([str(r.get("prediction_type", "")) for r in rows])
    variance_type_counts = Counter([str(r.get("variance_type", "")) for r in rows])
    denoiser_arch_counts = Counter([str(r.get("denoiser_arch", "")) for r in rows])
    conditional_unet_counts = Counter([str(r.get("conditional_unet1d_used", "")).lower() for r in rows])
    paper_alignment_counts = Counter([str(r.get("paper_alignment_level", "")) for r in rows])
    eval_seed_mode_counts = Counter([str(r.get("eval_sample_seed_mode", "")) for r in rows])
    metadata_issues = []
    if set(scheduler_type_counts.keys()) != {SCHEDULER_TYPE}:
        metadata_issues.append({"level": "FAIL", "name": "scheduler_type_mismatch", "counts": dict(scheduler_type_counts)})
    if set(beta_schedule_counts.keys()) != {BETA_SCHEDULE}:
        metadata_issues.append({"level": "FAIL", "name": "beta_schedule_mismatch", "counts": dict(beta_schedule_counts)})
    if set(prediction_type_counts.keys()) != {PREDICTION_TYPE}:
        metadata_issues.append({"level": "FAIL", "name": "prediction_type_mismatch", "counts": dict(prediction_type_counts)})
    if set(variance_type_counts.keys()) != {VARIANCE_TYPE}:
        metadata_issues.append({"level": "WARN", "name": "variance_type_not_fixed_small", "counts": dict(variance_type_counts)})
    else:
        metadata_issues.append({"level": "WARN", "name": "variance_fixed_small_not_learned_range", "detail": "fixed_small is accepted before medium; learned_range requires a separate architecture-level change."})
    if set(denoiser_arch_counts.keys()) != {DENOISER_ARCH}:
        metadata_issues.append({"level": "WARN", "name": "denoiser_arch_unexpected", "counts": dict(denoiser_arch_counts)})
    else:
        metadata_issues.append({"level": "WARN", "name": "mlp_denoiser_not_conditional_unet1d", "detail": "Scheduler is aligned, but denoiser remains MLP over low-dimensional future states."})
    summary = {
        "num_prediction_rows": len(rows),
        "summary_rows": summary_rows,
        "backends_seen": dict(backends),
        "all_torch_backend": bool(rows) and set(backends.keys()) == {"torch"},
        "branch_reference_mode": BRANCH_REFERENCE_MODE,
        "num_missing_primary_branch_refs": int(num_missing_primary_branch_refs),
        "num_valid_primary_branch_refs": int(num_valid_primary_branch_refs),
        "future_model_types_seen": dict(future_model_types_seen),
        "all_ddpm_used": bool(rows) and set(ddpm_seen.keys()) == {"true"},
        "ddpm_used_counts": dict(ddpm_seen),
        "scheduler_type_counts": dict(scheduler_type_counts),
        "beta_schedule_counts": dict(beta_schedule_counts),
        "prediction_type_counts": dict(prediction_type_counts),
        "variance_type_counts": dict(variance_type_counts),
        "denoiser_arch_counts": dict(denoiser_arch_counts),
        "conditional_unet1d_used_counts": dict(conditional_unet_counts),
        "paper_alignment_level_counts": dict(paper_alignment_counts),
        "eval_sample_seed_mode_counts": dict(eval_seed_mode_counts),
        "scheduler_metadata_issues": metadata_issues,
        "fallback_note": "This is fallback smoke, not a PyTorch DDPM result." if "numpy_fallback" in backends else "",
    }
    Path(args.out_json).write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps({
        "num_prediction_rows": len(rows),
        "backends_seen": dict(backends),
        "future_model_types_seen": dict(future_model_types_seen),
        "all_ddpm_used": summary["all_ddpm_used"],
        "num_missing_primary_branch_refs": num_missing_primary_branch_refs,
        "scheduler_type_counts": dict(scheduler_type_counts),
        "beta_schedule_counts": dict(beta_schedule_counts),
        "prediction_type_counts": dict(prediction_type_counts),
        "variance_type_counts": dict(variance_type_counts),
        "denoiser_arch_counts": dict(denoiser_arch_counts),
    }, indent=2))


if __name__ == "__main__":
    main()
