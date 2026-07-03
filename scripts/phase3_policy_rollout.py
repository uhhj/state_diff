#!/usr/bin/env python3
import sys
from pathlib import Path as _Phase3Path
ROOT = _Phase3Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import csv
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from ccda_phase3.data_io import load_action_codec_from_template
from ccda_phase3.metrics import curve_metric_from_state
from ccda_phase3.rollout import final_fraction_from_info, pad_history, state_from_live_info
from ccda_phase3.train_utils import load_future_model, load_inverse_model


def first_checkpoint(root: Path, baseline: str) -> Path:
    cands = sorted((root / baseline).glob("fold_*_seed_*"))
    if not cands:
        raise FileNotFoundError(f"no checkpoint for baseline={baseline} under {root}")
    return cands[0]


def write_outputs(trials, out_trials, out_summary):
    Path(out_trials).parent.mkdir(parents=True, exist_ok=True)
    fields = list(trials[0].keys()) if trials else []
    with Path(out_trials).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in trials:
            w.writerow(r)
    groups = defaultdict(list)
    for r in trials:
        groups[(r["baseline"], r["condition"])].append(r)
    rows = []
    by_baseline = defaultdict(dict)
    for (baseline, cond), rs in sorted(groups.items()):
        item = {
            "baseline": baseline,
            "condition": cond,
            "num_trials": len(rs),
            "success_rate": float(np.mean([float(r["success"]) for r in rs])) if rs else None,
            "mean_final_fraction": float(np.nanmean([float(r["final_fraction"]) for r in rs])) if rs else None,
            "mean_final_curve": float(np.nanmean([float(r["final_curve"]) for r in rs])) if rs else None,
            "mean_action_ood_score": float(np.nanmean([float(r["mean_action_ood_score"]) for r in rs])) if rs else None,
        }
        rows.append(item)
        by_baseline[baseline][cond] = item
    for item in rows:
        b = item["baseline"]
        free = by_baseline[b].get("free", {}).get("success_rate")
        pin = by_baseline[b].get("hidden_pin", {}).get("success_rate")
        item["free_vs_hidden_pin_success_gap"] = None if free is None or pin is None else float(free - pin)
    summary = {"num_trials": len(trials), "summary_rows": rows}
    Path(out_summary).write_text(json.dumps(summary, indent=2, sort_keys=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/state_diff2")
    ap.add_argument("--data", default="data/phase3_state_diff_windows/phase3_windows.npz")
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--checkpoint_dir", default=None)
    ap.add_argument("--ckpt_root", default="checkpoints/phase3")
    ap.add_argument("--baselines", nargs="+", default=["paper_state", "state_action"])
    ap.add_argument("--conditions", nargs="+", default=["free", "hidden_pin", "hidden_high_friction"])
    ap.add_argument("--seed_start", type=int, default=200000)
    ap.add_argument("--num_seeds", type=int, default=30)
    ap.add_argument("--max_steps", type=int, default=10)
    ap.add_argument("--samples_per_step", type=int, default=16)
    ap.add_argument("--motion_timeout", type=float, default=5.0, help="Per movej timeout for learned policy execution; lower values make OOD action failures explicit.")
    ap.add_argument("--action_clip_std", type=float, default=3.0, help="Clip decoded action vector to train action mean +/- N std for numerical rollout stability.")
    ap.add_argument("--out_trials", default="reports/phase3_policy_rollout_trials.csv")
    ap.add_argument("--out_summary", default="reports/phase3_policy_rollout_summary.json")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    sys.path.insert(0, str(root / "external" / "deformable-ravens"))
    try:
        from ravens import Environment, tasks
    except Exception as exc:
        raise SystemExit(f"[Phase3][FAIL] DeformableRavens import failed. Run in defravens37. Error: {exc}")

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = root / data_path
    data = np.load(data_path, allow_pickle=True)
    template_path = Path(str(data["action_template_json_or_pickle_path"]))
    codec = load_action_codec_from_template(template_path)
    th = int(data["th"])
    tf = int(data["tf"])
    state_dim = int(data["state_dim"])
    action_dim = int(data["action_dim"])
    n_beads = int(data["n_beads"])

    ckpts = []
    if args.checkpoint_dir and args.baseline:
        ckpts.append((args.baseline, Path(args.checkpoint_dir)))
    else:
        for b in args.baselines:
            ckpts.append((b, first_checkpoint(root / args.ckpt_root, b)))

    trials = []
    for baseline, ckpt in ckpts:
        cfg = json.loads((ckpt / "config.json").read_text())
        state_model = load_future_model(ckpt / "state_model.pt")
        idm = load_inverse_model(ckpt / "inverse_dynamics.pt")
        for condition in args.conditions:
            for local_i in range(args.num_seeds):
                seed = args.seed_start + local_i
                random.seed(seed)
                np.random.seed(seed)
                os.environ["CCDA_HIDDEN_CONDITION"] = condition
                os.environ["CCDA_VISIBLE_SEED"] = str(seed)
                os.environ["CCDA_PAIR_GROUP"] = f"phase3_policy_seed_{seed}"
                task = tasks.names["hidden-contact-cable-line"]()
                task.mode = "train"
                env = Environment(disp=False, hz=240)
                env.t_lim = float(args.motion_timeout)
                info = {}
                state_hist = []
                action_hist = []
                ood_scores = []
                failure = ""
                success = False
                final_fraction = float("nan")
                final_curve = float("nan")
                try:
                    env.reset(task)
                    reward_extras = task.reward()[1]
                    info = env.info
                    reward_extras["task.done"] = task.done()
                    info["extras"] = reward_extras
                    prev_xy = None
                    done = False
                    for step in range(args.max_steps):
                        state = state_from_live_info(info, prev_xy=prev_xy)
                        prev_xy = state[: n_beads * 2].reshape(n_beads, 2)
                        state_hist.append(state)
                        hist_states = pad_history(state_hist, th).reshape(-1)
                        if action_hist:
                            hist_actions = pad_history(action_hist, th).reshape(-1)
                        else:
                            hist_actions = np.zeros((th * action_dim,), dtype=np.float32)
                        model_x = hist_states.reshape(1, -1) if baseline == "paper_state" else np.concatenate([hist_states, hist_actions]).reshape(1, -1)
                        samples = state_model.sample(model_x, n_samples=args.samples_per_step, seed=seed + step)[:, 0, :]
                        mean_future = np.mean(samples, axis=0, keepdims=True)
                        idm_x = np.concatenate([hist_states.reshape(1, -1), mean_future], axis=1)
                        raw_action_vec = idm.predict(idm_x)[0]
                        ood_scores.append(float(idm.ood_score(raw_action_vec.reshape(1, -1))[0]))
                        lo = idm.train_action_mean - float(args.action_clip_std) * idm.train_action_std
                        hi = idm.train_action_mean + float(args.action_clip_std) * idm.train_action_std
                        action_vec = np.clip(raw_action_vec, lo, hi).astype(np.float32)
                        action = codec.decode(action_vec)
                        obs, reward, done, info = env.step(action)
                        action_hist.append(action_vec.astype(np.float32))
                        if done:
                            break
                    success = bool(info.get("extras", {}).get("task.done", done))
                    final_fraction = final_fraction_from_info(info)
                    try:
                        final_state = state_from_live_info(info)
                        final_curve = curve_metric_from_state(final_state, n_beads)
                    except Exception:
                        pass
                except Exception as exc:
                    failure = repr(exc)
                finally:
                    try:
                        env.stop()
                    except Exception:
                        pass
                trials.append({
                    "baseline": baseline,
                    "checkpoint_dir": str(ckpt),
                    "condition": condition,
                    "visible_seed": seed,
                    "success": int(success),
                    "final_fraction": final_fraction,
                    "final_curve": final_curve,
                    "num_steps": len(action_hist),
                    "mean_action_ood_score": float(np.mean(ood_scores)) if ood_scores else float("nan"),
                    "predicted_branch": "nearest_branch_proxy_unavailable",
                    "failure_reason": failure,
                })
                print("[Phase3] rollout", trials[-1], flush=True)
    out_trials = Path(args.out_trials); out_summary = Path(args.out_summary)
    if not out_trials.is_absolute(): out_trials = root / out_trials
    if not out_summary.is_absolute(): out_summary = root / out_summary
    write_outputs(trials, out_trials, out_summary)
    print("[Phase3] wrote", out_trials)
    print("[Phase3] wrote", out_summary)


if __name__ == "__main__":
    main()
