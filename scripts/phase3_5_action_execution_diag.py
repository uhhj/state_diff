#!/usr/bin/env python3
import argparse
import csv
import importlib
import json
import math
import os
import pickle
import random
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT_DEFAULT = "/data/state_diff2"
REQUIRED_CONDITIONS = ["free", "hidden_pin", "hidden_high_friction", "hidden_breakaway_pin"]
PRIMARY = "hidden_breakaway_pin"
DIAGNOSTIC = "hidden_pin"

FORBIDDEN_PREFIXES = ["tensorflow", "ravens.agents", "ravens.models", "ravens.datasets"]

def assert_gate():
    if os.environ.get("PHASE3_ALLOW_ACTION_DIAGNOSTIC", "0") != "1":
        raise SystemExit("[Phase3.5][BLOCKED] Set PHASE3_ALLOW_ACTION_DIAGNOSTIC=1")
    if os.environ.get("PHASE3_ACTION_DIAGNOSTIC_CONFIRMED", "0") != "1":
        raise SystemExit("[Phase3.5][BLOCKED] Set PHASE3_ACTION_DIAGNOSTIC_CONFIRMED=1")

def assert_no_forbidden(stage):
    bad = []
    for m in sys.modules:
        for p in FORBIDDEN_PREFIXES:
            if m == p or m.startswith(p + "."):
                bad.append(m)
    if bad:
        raise RuntimeError(f"forbidden modules loaded at {stage}: {bad[:30]}")

def import_tf_free_ravens(root):
    defravens = Path(root) / "external/deformable-ravens"
    if str(defravens) not in sys.path:
        sys.path.insert(0, str(defravens))
    assert_no_forbidden("before_ravens_import")

    # Prefer rollout helper if available, because Phase3.3b fixed the lightweight stub there.
    scripts_dir = Path(root) / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    try:
        pr = importlib.import_module("phase3_policy_rollout")
        if hasattr(pr, "patch_pybullet_pkg_resources_metadata"):
            pr.patch_pybullet_pkg_resources_metadata()
        if hasattr(pr, "import_ravens_runtime"):
            tasks, Environment = pr.import_ravens_runtime(root)
            assert_no_forbidden("after_policy_rollout_import_ravens_runtime")
            return tasks, Environment
    except Exception:
        pass

    # Fallback for this fork: ravens.environment.Environment.
    tasks = importlib.import_module("ravens.tasks")
    env_mod = importlib.import_module("ravens.environment")
    Environment = getattr(env_mod, "Environment")
    assert_no_forbidden("after_fallback_tf_free_import")
    return tasks, Environment

def load_windows(root, path):
    data = np.load(Path(root) / path, allow_pickle=True)
    meta = {}
    if "meta_json" in data:
        raw = data["meta_json"]
        try:
            meta = json.loads(str(raw.item() if getattr(raw, "shape", ()) == () else raw.reshape(-1)[0]))
        except Exception:
            meta = {}
    return data, meta

def array_str(data, *names):
    for n in names:
        if n in data.files:
            return np.asarray([str(x) for x in data[n]])
    return None

def first_checkpoint(root, baseline):
    base = Path(root) / "checkpoints/phase3" / baseline
    cands = sorted([p for p in base.glob("fold_*_seed_*") if (p / "state_model.pt").exists() and (p / "inverse_dynamics.pt").exists()])
    if not cands:
        raise FileNotFoundError(f"missing checkpoint for {baseline} under {base}")
    return cands[0]

def load_models(root, baseline):
    from ccda_phase3.train_utils import load_future_model, load_inverse_model
    ckpt = first_checkpoint(root, baseline)
    return ckpt, load_future_model(ckpt / "state_model.pt"), load_inverse_model(ckpt / "inverse_dynamics.pt")

def action_info(action):
    info = {
        "primitive": "",
        "has_params": False,
        "pose0_x": float("nan"),
        "pose0_y": float("nan"),
        "pose0_z": float("nan"),
        "pose1_x": float("nan"),
        "pose1_y": float("nan"),
        "pose1_z": float("nan"),
        "pose0_quat_norm": float("nan"),
        "pose1_quat_norm": float("nan"),
        "action_valid": False,
        "action_repr": "",
    }
    try:
        info["action_repr"] = json.dumps(action, default=str)[:1200]
        if not isinstance(action, dict):
            return info
        info["primitive"] = str(action.get("primitive", ""))
        params = action.get("params", {})
        info["has_params"] = isinstance(params, dict)
        for key in ["pose0", "pose1"]:
            val = params.get(key) if isinstance(params, dict) else None
            if val is None:
                continue
            pos = np.asarray(val[0], dtype=float).reshape(-1)
            rot = np.asarray(val[1], dtype=float).reshape(-1)
            info[f"{key}_x"] = float(pos[0]) if len(pos) > 0 else float("nan")
            info[f"{key}_y"] = float(pos[1]) if len(pos) > 1 else float("nan")
            info[f"{key}_z"] = float(pos[2]) if len(pos) > 2 else float("nan")
            info[f"{key}_quat_norm"] = float(np.linalg.norm(rot)) if len(rot) else float("nan")
        info["action_valid"] = info["primitive"] == "pick_place" and info["has_params"]
        return info
    except Exception as e:
        info["action_repr"] = f"action_info_error: {repr(e)}"
        return info

def final_fraction_from_info_safe(info):
    try:
        from ccda_phase3.rollout import final_fraction_from_info
        return float(final_fraction_from_info(info))
    except Exception:
        try:
            extras = info.get("extras", {})
            for k in ["final_fraction", "progress_fraction", "insert_fraction"]:
                if k in extras:
                    return float(extras[k])
        except Exception:
            pass
    return float("nan")

def state_from_info_safe(info, prev_xy, n_beads):
    from ccda_phase3.rollout import state_from_live_info
    return state_from_live_info(info, prev_xy=prev_xy)

def reset_env(tasks, Environment, condition, seed, motion_timeout, root):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["CCDA_HIDDEN_CONDITION"] = condition
    os.environ["CCDA_VISIBLE_SEED"] = str(seed)
    os.environ["CCDA_PAIR_GROUP"] = f"phase3_5_diag_seed_{seed}"
    os.environ.setdefault("CCDA_BREAKAWAY_FORCE", "2.6")
    os.environ.setdefault("CCDA_BREAKAWAY_DISP", "0.045")
    os.environ.setdefault("CCDA_BREAKAWAY_BEAD_RATIO", "0.45")
    os.environ.setdefault("CCDA_ORACLE_BREAKAWAY_PULL_DIST", "0.36")
    task = tasks.names["hidden-contact-cable-line"]()
    task.mode = "train"
    env = Environment(disp=False, hz=240)
    env.t_lim = float(motion_timeout)
    obs = env.reset(task)
    reward_extras = task.reward()[1]
    info = env.info
    reward_extras["task.done"] = task.done()
    info["extras"] = reward_extras
    return env, task, obs, info

def env_stop(env):
    try:
        env.pause()
    except Exception:
        pass
    try:
        env.running = False
    except Exception:
        pass
    try:
        env.ee = None
    except Exception:
        pass
    time.sleep(0.05)
    try:
        env.stop()
    except Exception:
        pass

def pick_indices_by_condition(data, condition, n):
    y = np.asarray(data["y_action"])
    conds = array_str(data, "condition_name", "condition", "conditions")
    if conds is None:
        return list(range(min(n, len(y))))
    idx = np.where(conds == condition)[0]
    return [int(i) for i in idx[:n]]

def maybe_oracle_action(task, env, obs, info):
    if not hasattr(task, "oracle"):
        return None, "task_has_no_oracle"
    try:
        oracle = task.oracle(env)
        action = oracle.act(obs, info)
        return action, ""
    except Exception as e:
        return None, repr(e)

def execute_one(env, task, action):
    before_info = env.info
    before_extras = task.reward()[1]
    before_info["extras"] = before_extras
    before_frac = final_fraction_from_info_safe(before_info)
    obs, reward, done, info = env.step(action)
    after_frac = final_fraction_from_info_safe(info)
    success = bool(info.get("extras", {}).get("task.done", done))
    failure_reason = ""
    return obs, reward, done, info, success, before_frac, after_frac, failure_reason

def run_oracle_trial(root, tasks, Environment, condition, seed, motion_timeout, max_steps):
    rows = []
    env = None
    try:
        env, task, obs, info = reset_env(tasks, Environment, condition, seed, motion_timeout, root)
        for step in range(max_steps):
            action, err = maybe_oracle_action(task, env, obs, info)
            if action is None:
                rows.append(base_row("oracle_action", "", condition, seed, motion_timeout, step, err))
                break
            ai = action_info(action)
            try:
                obs, reward, done, info, success, before, after, failure = execute_one(env, task, action)
                row = base_row("oracle_action", "", condition, seed, motion_timeout, step, failure)
                row.update(ai)
                row.update({
                    "env_done": int(bool(done)),
                    "success": int(bool(success)),
                    "reward": float(reward),
                    "final_fraction_before": before,
                    "final_fraction_after": after,
                    "delta_final_fraction": after - before if np.isfinite(after) and np.isfinite(before) else float("nan"),
                    "num_env_steps_executed": step + 1,
                })
                rows.append(row)
                if done or success:
                    break
            except Exception as e:
                row = base_row("oracle_action", "", condition, seed, motion_timeout, step, repr(e))
                row.update(ai)
                rows.append(row)
                break
    except Exception as e:
        rows.append(base_row("oracle_action", "", condition, seed, motion_timeout, 0, repr(e)))
    finally:
        if env is not None:
            env_stop(env)
    return rows

def base_row(control_source, baseline, condition, seed, motion_timeout, step, failure_reason):
    return {
        "control_source": control_source,
        "baseline": baseline,
        "condition": condition,
        "visible_seed": seed,
        "motion_timeout": motion_timeout,
        "step": step,
        "success": 0,
        "env_done": 0,
        "reward": float("nan"),
        "final_fraction_before": float("nan"),
        "final_fraction_after": float("nan"),
        "delta_final_fraction": float("nan"),
        "failure_reason": failure_reason or "",
        "primitive": "",
        "has_params": False,
        "pose0_x": float("nan"),
        "pose0_y": float("nan"),
        "pose0_z": float("nan"),
        "pose1_x": float("nan"),
        "pose1_y": float("nan"),
        "pose1_z": float("nan"),
        "pose0_quat_norm": float("nan"),
        "pose1_quat_norm": float("nan"),
        "action_valid": False,
        "action_repr": "",
        "action_ood_score": float("nan"),
        "clip_fraction": float("nan"),
        "raw_action_norm": float("nan"),
        "clipped_action_norm": float("nan"),
        "gt_action_idx": "",
        "selected_recoverable_config": "breakaway_force_2p6_disp_0p045_pull_0p36",
        "primary_hidden_condition": PRIMARY,
        "diagnostic_hidden_condition": DIAGNOSTIC,
        "primary_pair": f"free_vs_{PRIMARY}",
        "diagnostic_pair": f"free_vs_{DIAGNOSTIC}",
        "rollout_runtime": "tf_free_action_execution_diagnostic",
    }

def run_gt_replay_trial(root, tasks, Environment, data, codec, condition, seed, motion_timeout, action_idx):
    rows = []
    env = None
    try:
        env, task, obs, info = reset_env(tasks, Environment, condition, seed, motion_timeout, root)
        vec = np.asarray(data["y_action"][action_idx], dtype=np.float32)
        action = codec.decode(vec)
        ai = action_info(action)
        obs, reward, done, info, success, before, after, failure = execute_one(env, task, action)
        row = base_row("gt_y_action_replay", "", condition, seed, motion_timeout, 0, failure)
        row.update(ai)
        row.update({
            "env_done": int(bool(done)),
            "success": int(bool(success)),
            "reward": float(reward),
            "final_fraction_before": before,
            "final_fraction_after": after,
            "delta_final_fraction": after - before if np.isfinite(after) and np.isfinite(before) else float("nan"),
            "gt_action_idx": str(action_idx),
            "raw_action_norm": float(np.linalg.norm(vec)),
            "clipped_action_norm": float(np.linalg.norm(vec)),
        })
        rows.append(row)
    except Exception as e:
        row = base_row("gt_y_action_replay", "", condition, seed, motion_timeout, 0, repr(e))
        row["gt_action_idx"] = str(action_idx)
        rows.append(row)
    finally:
        if env is not None:
            env_stop(env)
    return rows

def run_learned_trial(root, tasks, Environment, data, codec, baseline, condition, seed, motion_timeout, max_steps, samples_per_step, action_clip_std):
    rows = []
    env = None
    try:
        from ccda_phase3.rollout import pad_history
        ckpt, state_model, idm = load_models(root, baseline)
        th = int(data["th"])
        action_dim = int(data["action_dim"])
        n_beads = int(data["n_beads"])
        env, task, obs, info = reset_env(tasks, Environment, condition, seed, motion_timeout, root)
        state_hist, action_hist = [], []
        prev_xy = None
        for step in range(max_steps):
            state = state_from_info_safe(info, prev_xy, n_beads)
            prev_xy = state[: n_beads * 2].reshape(n_beads, 2)
            state_hist.append(state)
            hist_states = pad_history(state_hist, th).reshape(-1)
            if action_hist:
                hist_actions = pad_history(action_hist, th).reshape(-1)
            else:
                hist_actions = np.zeros((th * action_dim,), dtype=np.float32)

            # IMPORTANT: model_x uses observable history only.
            # Do not concatenate hidden condition, hidden contact metadata, success labels, final_fraction, or future labels.
            if baseline == "paper_state":
                model_x = hist_states.reshape(1, -1)
            else:
                model_x = np.concatenate([hist_states, hist_actions]).reshape(1, -1)

            samples = state_model.sample(model_x, n_samples=samples_per_step, seed=seed + step)[:, 0, :]
            mean_future = np.mean(samples, axis=0, keepdims=True)
            idm_x = np.concatenate([hist_states.reshape(1, -1), mean_future], axis=1)
            raw_action_vec = idm.predict(idm_x)[0]
            ood = float(idm.ood_score(raw_action_vec.reshape(1, -1))[0])
            lo = idm.train_action_mean - float(action_clip_std) * idm.train_action_std
            hi = idm.train_action_mean + float(action_clip_std) * idm.train_action_std
            clipped = np.clip(raw_action_vec, lo, hi).astype(np.float32)
            clip_fraction = float(np.mean(np.abs(clipped - raw_action_vec) > 1e-8))
            action = codec.decode(clipped)
            ai = action_info(action)
            try:
                obs, reward, done, info, success, before, after, failure = execute_one(env, task, action)
                row = base_row("learned_action", baseline, condition, seed, motion_timeout, step, failure)
                row.update(ai)
                row.update({
                    "env_done": int(bool(done)),
                    "success": int(bool(success)),
                    "reward": float(reward),
                    "final_fraction_before": before,
                    "final_fraction_after": after,
                    "delta_final_fraction": after - before if np.isfinite(after) and np.isfinite(before) else float("nan"),
                    "action_ood_score": ood,
                    "clip_fraction": clip_fraction,
                    "raw_action_norm": float(np.linalg.norm(raw_action_vec)),
                    "clipped_action_norm": float(np.linalg.norm(clipped)),
                })
                rows.append(row)
                action_hist.append(clipped)
                if done or success:
                    break
            except Exception as e:
                row = base_row("learned_action", baseline, condition, seed, motion_timeout, step, repr(e))
                row.update(ai)
                row.update({
                    "action_ood_score": ood,
                    "clip_fraction": clip_fraction,
                    "raw_action_norm": float(np.linalg.norm(raw_action_vec)),
                    "clipped_action_norm": float(np.linalg.norm(clipped)),
                })
                rows.append(row)
                break
    except Exception as e:
        rows.append(base_row("learned_action", baseline, condition, seed, motion_timeout, 0, repr(e)))
    finally:
        if env is not None:
            env_stop(env)
    return rows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=ROOT_DEFAULT)
    parser.add_argument("--windows", default="data/phase3_state_diff_windows/phase3_windows.npz")
    parser.add_argument("--action_template", default="data/phase3_state_diff_windows/phase3_action_template.pkl")
    parser.add_argument("--conditions", nargs="+", default=REQUIRED_CONDITIONS)
    parser.add_argument("--baselines", nargs="+", default=["paper_state", "state_action"])
    parser.add_argument("--seed_start", type=int, default=310000)
    parser.add_argument("--seeds_per_condition", type=int, default=2)
    parser.add_argument("--motion_timeouts", nargs="+", type=float, default=[5.0, 15.0])
    parser.add_argument("--oracle_steps", type=int, default=8)
    parser.add_argument("--learned_steps", type=int, default=4)
    parser.add_argument("--samples_per_step", type=int, default=16)
    parser.add_argument("--action_clip_std", type=float, default=3.0)
    parser.add_argument("--gt_samples_per_condition", type=int, default=2)
    parser.add_argument("--out_csv", default="reports/phase3_5_action_execution_trials.csv")
    parser.add_argument("--out_json", default="reports/phase3_5_action_execution_raw_summary.json")
    args = parser.parse_args()

    assert_gate()
    root = Path(args.root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    tasks, Environment = import_tf_free_ravens(root)
    assert_no_forbidden("after_tf_free_import")

    if args.conditions != REQUIRED_CONDITIONS:
        raise SystemExit(f"[Phase3.5][FAIL] conditions must be {REQUIRED_CONDITIONS}, got {args.conditions}")

    data, meta = load_windows(root, args.windows)
    if meta.get("conditions") != REQUIRED_CONDITIONS:
        raise SystemExit(f"[Phase3.5][FAIL] windows conditions mismatch: {meta.get('conditions')}")
    if meta.get("primary_hidden_condition") != PRIMARY:
        raise SystemExit(f"[Phase3.5][FAIL] windows primary mismatch: {meta.get('primary_hidden_condition')}")

    with open(root / args.action_template, "rb") as f:
        codec = pickle.load(f)

    all_rows = []
    for condition in args.conditions:
        gt_idxs = pick_indices_by_condition(data, condition, args.gt_samples_per_condition)
        for local_i in range(args.seeds_per_condition):
            seed = args.seed_start + local_i
            for timeout in args.motion_timeouts:
                all_rows.extend(run_oracle_trial(root, tasks, Environment, condition, seed, timeout, args.oracle_steps))
                if gt_idxs:
                    action_idx = gt_idxs[min(local_i, len(gt_idxs) - 1)]
                    all_rows.extend(run_gt_replay_trial(root, tasks, Environment, data, codec, condition, seed, timeout, action_idx))
                for baseline in args.baselines:
                    all_rows.extend(run_learned_trial(
                        root, tasks, Environment, data, codec,
                        baseline, condition, seed, timeout,
                        args.learned_steps, args.samples_per_step, args.action_clip_std,
                    ))
                assert_no_forbidden(f"after_condition_{condition}_seed_{seed}_timeout_{timeout}")

    out_csv = root / args.out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(all_rows[0].keys()) if all_rows else []
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    payload = {
        "num_rows": len(all_rows),
        "conditions": args.conditions,
        "baselines": args.baselines,
        "motion_timeouts": args.motion_timeouts,
        "oracle_steps": args.oracle_steps,
        "learned_steps": args.learned_steps,
        "seeds_per_condition": args.seeds_per_condition,
        "out_csv": args.out_csv,
        "scope": "phase3_5_action_execution_diagnostic_no_phase4_no_cps",
    }
    (root / args.out_json).write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
