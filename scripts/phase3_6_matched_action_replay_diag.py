#!/usr/bin/env python3
import argparse
import csv
import glob
import importlib
import json
import math
import os
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT_DEFAULT = "/data/state_diff2"
REQUIRED_CONDITIONS = ["free", "hidden_pin", "hidden_high_friction", "hidden_breakaway_pin"]
PRIMARY = "hidden_breakaway_pin"
DIAGNOSTIC = "hidden_pin"
FORBIDDEN_PREFIXES = ["tensorflow", "ravens.agents", "ravens.models", "ravens.datasets"]

def assert_gate():
    if os.environ.get("PHASE3_ALLOW_MATCHED_REPLAY", "0") != "1":
        raise SystemExit("[Phase3.6][BLOCKED] Set PHASE3_ALLOW_MATCHED_REPLAY=1")
    if os.environ.get("PHASE3_MATCHED_REPLAY_CONFIRMED", "0") != "1":
        raise SystemExit("[Phase3.6][BLOCKED] Set PHASE3_MATCHED_REPLAY_CONFIRMED=1")

def assert_no_forbidden(stage):
    bad = []
    for m in sys.modules:
        for p in FORBIDDEN_PREFIXES:
            if m == p or m.startswith(p + "."):
                bad.append(m)
    if bad:
        raise RuntimeError(f"forbidden modules loaded at {stage}: {bad[:30]}")

def import_tf_free_ravens(root):
    root = Path(root)
    defravens = root / "external/deformable-ravens"
    if str(defravens) not in sys.path:
        sys.path.insert(0, str(defravens))
    scripts_dir = root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    assert_no_forbidden("before_import_ravens_runtime")
    try:
        pr = importlib.import_module("phase3_policy_rollout")
        if hasattr(pr, "patch_pybullet_pkg_resources_metadata"):
            pr.patch_pybullet_pkg_resources_metadata()
        if hasattr(pr, "import_ravens_runtime"):
            tasks, Environment = pr.import_ravens_runtime(root)
            assert_no_forbidden("after_phase3_policy_import_ravens_runtime")
            return tasks, Environment, getattr(pr, "close_env_safely", None)
    except Exception:
        pass
    tasks = importlib.import_module("ravens.tasks")
    env_mod = importlib.import_module("ravens.environment")
    Environment = getattr(env_mod, "Environment")
    assert_no_forbidden("after_fallback_import")
    return tasks, Environment, None

def close_env(env, close_helper=None):
    if close_helper is not None:
        try:
            close_helper(env)
            return
        except Exception:
            pass
    try:
        env.pause()
    except Exception:
        pass
    try:
        env.running = False
    except Exception:
        pass
    time.sleep(0.03)
    try:
        env.stop()
    except Exception:
        pass

def load_windows(root, path):
    data = np.load(Path(root) / path, allow_pickle=True)
    meta = {}
    if "meta_json" in data.files:
        raw = data["meta_json"]
        try:
            meta = json.loads(str(raw.item() if getattr(raw, "shape", ()) == () else raw.reshape(-1)[0]))
        except Exception:
            meta = {}
    return data, meta

def get_str_array(data, key):
    if key is None or key not in data.files:
        return None
    return np.asarray([str(x) for x in data[key]])

def get_int_array(data, key):
    if key is None or key not in data.files:
        return None
    return np.asarray(data[key]).astype(int)

def load_schema(root, schema_json):
    p = Path(root) / schema_json
    return json.loads(p.read_text())

def resolve_action_file(root, source_value):
    root = Path(root)
    if not source_value:
        return None
    s = str(source_value)

    candidates = []
    p = Path(s)
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append(root / s)
        candidates.append(root / "external/deformable-ravens" / s)

    # Common DeformableRavens structure: info/<basename>.pkl -> action/<basename>.pkl
    for base in list(candidates):
        st = str(base)
        candidates.append(Path(st.replace("/info/", "/action/")))
        candidates.append(Path(st.replace("\\info\\", "\\action\\")))

    basename = Path(s).name
    if basename:
        candidates.extend(Path(x) for x in glob.glob(str(root / "external/deformable-ravens/data/**/action" / basename), recursive=True))
        candidates.extend(Path(x) for x in glob.glob(str(root / "data/**/action" / basename), recursive=True))

    seen = set()
    for c in candidates:
        if str(c) in seen:
            continue
        seen.add(str(c))
        if c.exists() and c.is_file():
            return c
    return None

def load_action_sequence(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for key in ["actions", "action", "acts"]:
            if key in obj and isinstance(obj[key], list):
                return obj[key]
    return None

def decode_or_passthrough(codec, item):
    if isinstance(item, dict):
        return item
    arr = np.asarray(item, dtype=np.float32).reshape(-1)
    return codec.decode(arr)

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
        info["action_valid"] = bool(info["primitive"] == "pick_place" and info["has_params"])
    except Exception as e:
        info["action_repr"] = f"action_info_error: {repr(e)}"
    return info

def final_fraction_from_info(info):
    try:
        from ccda_phase3.rollout import final_fraction_from_info as ffi
        return float(ffi(info))
    except Exception:
        try:
            extras = info.get("extras", {})
            for k in ["final_fraction", "progress_fraction", "insert_fraction"]:
                if k in extras:
                    return float(extras[k])
        except Exception:
            pass
    return float("nan")

def live_state_from_info(info, prev_xy):
    try:
        from ccda_phase3.rollout import state_from_live_info
        return state_from_live_info(info, prev_xy=prev_xy)
    except Exception:
        return None

def reset_env(tasks, Environment, condition, visible_seed, pair_group, motion_timeout):
    seed_int = int(visible_seed) if str(visible_seed).lstrip("-").isdigit() else abs(hash(str(visible_seed))) % (2**31)
    random.seed(seed_int)
    np.random.seed(seed_int)
    os.environ["CCDA_HIDDEN_CONDITION"] = str(condition)
    os.environ["CCDA_VISIBLE_SEED"] = str(visible_seed)
    os.environ["CCDA_PAIR_GROUP"] = str(pair_group)
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

def execute_one(env, task, action):
    before_info = env.info
    before_extras = task.reward()[1]
    before_info["extras"] = before_extras
    before = final_fraction_from_info(before_info)
    obs, reward, done, info = env.step(action)
    after = final_fraction_from_info(info)
    success = bool(info.get("extras", {}).get("task.done", done))
    return obs, reward, done, info, int(success), before, after

def get_prefix_actions_from_raw(root, codec, source_value, window_t):
    af = resolve_action_file(root, source_value)
    if af is None:
        return None, "raw_action_file_not_found", ""
    seq = load_action_sequence(af)
    if seq is None:
        return None, "raw_action_file_unrecognized", str(af)
    prefix_n = max(0, min(int(window_t), len(seq)))
    actions = []
    for item in seq[:prefix_n]:
        actions.append(decode_or_passthrough(codec, item))
    return actions, "raw_action_file_high_confidence", str(af)

def get_prefix_actions_from_state_action_tail(data, codec, idx):
    if "state_action_x" not in data.files or "paper_x" not in data.files:
        return None, "state_action_tail_unavailable", ""
    if "action_dim" in data.files:
        action_dim = int(data["action_dim"])
    else:
        action_dim = 14
    extra = int(data["state_action_x"].shape[1] - data["paper_x"].shape[1])
    if extra <= 0 or extra % action_dim != 0:
        return None, "state_action_tail_bad_extra_dim", ""
    hist = np.asarray(data["state_action_x"][idx, -extra:], dtype=np.float32).reshape(-1, action_dim)
    actions = []
    # Drop all-zero padded actions.
    for vec in hist:
        if float(np.linalg.norm(vec)) < 1e-8:
            continue
        actions.append(codec.decode(vec))
    return actions, "state_action_x_tail_low_confidence", f"extra_dim={extra}, num_actions={len(actions)}"

def current_ref_state_from_paper_x(data, idx, live_len):
    if "paper_x" not in data.files or live_len is None or live_len <= 0:
        return None
    x = np.asarray(data["paper_x"][idx], dtype=np.float32).reshape(-1)
    if len(x) < live_len:
        return None
    return x[-live_len:]

def replay_prefix(env, task, obs, info, prefix_actions, max_prefix_actions):
    rows = []
    cur_obs, cur_info = obs, info
    count = 0
    for j, action in enumerate(prefix_actions[:max_prefix_actions]):
        ai = action_info(action)
        try:
            cur_obs, reward, done, cur_info, success, before, after = execute_one(env, task, action)
            rows.append({
                "prefix_step": j,
                "prefix_action_valid": ai["action_valid"],
                "prefix_success": success,
                "prefix_done": int(bool(done)),
                "prefix_reward": float(reward),
                "prefix_delta_final_fraction": after - before if np.isfinite(after) and np.isfinite(before) else float("nan"),
                "prefix_failure_reason": "",
            })
            count += 1
            if done or success:
                break
        except Exception as e:
            rows.append({
                "prefix_step": j,
                "prefix_action_valid": ai["action_valid"],
                "prefix_success": 0,
                "prefix_done": 1,
                "prefix_reward": float("nan"),
                "prefix_delta_final_fraction": float("nan"),
                "prefix_failure_reason": repr(e),
            })
            break
    return cur_obs, cur_info, rows, count

def make_base_row(window_idx, condition, visible_seed, window_t, source_value, prefix_source, prefix_detail, motion_timeout, test_type):
    return {
        "window_idx": int(window_idx),
        "condition": str(condition),
        "visible_seed": str(visible_seed),
        "window_t": int(window_t),
        "source_value": str(source_value),
        "prefix_source": str(prefix_source),
        "prefix_detail": str(prefix_detail),
        "motion_timeout": float(motion_timeout),
        "test_type": test_type,
        "prefix_actions_requested": 0,
        "prefix_actions_executed": 0,
        "prefix_had_failure": 0,
        "prefix_reached_done_or_success": 0,
        "prefix_state_l2_error": float("nan"),
        "prefix_state_mean_abs_error": float("nan"),
        "primitive": "",
        "has_params": False,
        "action_valid": False,
        "pose0_x": float("nan"),
        "pose0_y": float("nan"),
        "pose0_z": float("nan"),
        "pose1_x": float("nan"),
        "pose1_y": float("nan"),
        "pose1_z": float("nan"),
        "pose0_quat_norm": float("nan"),
        "pose1_quat_norm": float("nan"),
        "success": 0,
        "env_done": 0,
        "reward": float("nan"),
        "final_fraction_before": float("nan"),
        "final_fraction_after": float("nan"),
        "delta_final_fraction": float("nan"),
        "failure_reason": "",
        "action_repr": "",
        "primary_hidden_condition": PRIMARY,
        "diagnostic_hidden_condition": DIAGNOSTIC,
        "primary_pair": f"free_vs_{PRIMARY}",
        "diagnostic_pair": f"free_vs_{DIAGNOSTIC}",
        "scope": "phase3_6_matched_state_replay_no_phase4_no_cps",
    }

def run_one_test(root, tasks, Environment, close_helper, data, codec, idx, condition, visible_seed, window_t,
                 source_value, motion_timeout, test_type, max_prefix_actions, allow_low_conf_prefix):
    env = None
    row = make_base_row(idx, condition, visible_seed, window_t, source_value, "", "", motion_timeout, test_type)
    try:
        prefix_actions, prefix_source, prefix_detail = get_prefix_actions_from_raw(root, codec, source_value, window_t)
        if prefix_actions is None and allow_low_conf_prefix:
            prefix_actions, prefix_source, prefix_detail = get_prefix_actions_from_state_action_tail(data, codec, idx)
        if prefix_actions is None:
            row["prefix_source"] = prefix_source
            row["prefix_detail"] = prefix_detail
            row["failure_reason"] = "no_prefix_actions_available"
            return row

        row["prefix_source"] = prefix_source
        row["prefix_detail"] = prefix_detail
        row["prefix_actions_requested"] = int(len(prefix_actions))

        env, task, obs, info = reset_env(
            tasks, Environment, condition, visible_seed,
            f"phase3_6_window_{idx}", motion_timeout
        )
        obs, info, prefix_rows, executed = replay_prefix(env, task, obs, info, prefix_actions, max_prefix_actions)
        row["prefix_actions_executed"] = int(executed)
        row["prefix_had_failure"] = int(any(x.get("prefix_failure_reason") for x in prefix_rows))
        row["prefix_reached_done_or_success"] = int(any(x.get("prefix_done") or x.get("prefix_success") for x in prefix_rows))

        prev_xy = None
        live_state = live_state_from_info(info, prev_xy)
        if live_state is not None:
            ref = current_ref_state_from_paper_x(data, idx, len(live_state))
            if ref is not None:
                diff = np.asarray(live_state, dtype=np.float32) - np.asarray(ref, dtype=np.float32)
                row["prefix_state_l2_error"] = float(np.linalg.norm(diff))
                row["prefix_state_mean_abs_error"] = float(np.mean(np.abs(diff)))

        if test_type == "matched_gt_y_action":
            vec = np.asarray(data["y_action"][idx], dtype=np.float32)
            action = codec.decode(vec)
        elif test_type == "same_state_oracle":
            oracle = task.oracle(env)
            action = oracle.act(obs, info)
        else:
            row["failure_reason"] = f"unknown_test_type:{test_type}"
            return row

        ai = action_info(action)
        row.update({k: v for k, v in ai.items() if k in row or k == "action_repr"})

        obs2, reward, done, info2, success, before, after = execute_one(env, task, action)
        row["success"] = int(success)
        row["env_done"] = int(bool(done))
        row["reward"] = float(reward)
        row["final_fraction_before"] = before
        row["final_fraction_after"] = after
        row["delta_final_fraction"] = after - before if np.isfinite(after) and np.isfinite(before) else float("nan")
        return row
    except Exception as e:
        row["failure_reason"] = repr(e)
        return row
    finally:
        if env is not None:
            close_env(env, close_helper)

def select_indices(data, schema, samples_per_condition):
    cond_key = schema.get("condition_key")
    visible_seed_key = schema.get("visible_seed_key")
    window_t_key = schema.get("window_t_key")
    source_key = schema.get("source_key")

    conds = get_str_array(data, cond_key)
    visible = get_str_array(data, visible_seed_key)
    window_t = get_int_array(data, window_t_key)
    source = get_str_array(data, source_key)

    if conds is None or visible is None or window_t is None:
        raise RuntimeError("Missing condition/visible_seed/window_t arrays for matched replay.")

    selected = []
    for c in REQUIRED_CONDITIONS:
        idxs = np.where(conds == c)[0]
        for i in idxs[:samples_per_condition]:
            selected.append({
                "idx": int(i),
                "condition": str(conds[i]),
                "visible_seed": str(visible[i]),
                "window_t": int(window_t[i]),
                "source_value": str(source[i]) if source is not None else "",
            })
    return selected

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT_DEFAULT)
    ap.add_argument("--windows", default="data/phase3_state_diff_windows/phase3_windows.npz")
    ap.add_argument("--action_template", default="data/phase3_state_diff_windows/phase3_action_template.pkl")
    ap.add_argument("--schema_json", default="reports/phase3_6_window_schema_audit_summary.json")
    ap.add_argument("--samples_per_condition", type=int, default=3)
    ap.add_argument("--motion_timeouts", nargs="+", type=float, default=[15.0])
    ap.add_argument("--max_prefix_actions", type=int, default=20)
    ap.add_argument("--allow_low_conf_prefix", action="store_true")
    ap.add_argument("--out_csv", default="reports/phase3_6_matched_action_replay_trials.csv")
    ap.add_argument("--out_json", default="reports/phase3_6_matched_action_replay_raw_summary.json")
    args = ap.parse_args()

    assert_gate()
    root = Path(args.root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    tasks, Environment, close_helper = import_tf_free_ravens(root)
    assert_no_forbidden("after_import_runtime")

    data, meta = load_windows(root, args.windows)
    if meta.get("conditions") != REQUIRED_CONDITIONS:
        raise SystemExit(f"[Phase3.6][FAIL] windows conditions mismatch: {meta.get('conditions')}")
    if meta.get("primary_hidden_condition") != PRIMARY:
        raise SystemExit(f"[Phase3.6][FAIL] windows primary mismatch: {meta.get('primary_hidden_condition')}")

    with open(root / args.action_template, "rb") as f:
        codec = pickle.load(f)

    schema = load_schema(root, args.schema_json)
    selected = select_indices(data, schema, args.samples_per_condition)

    rows = []
    for item in selected:
        for timeout in args.motion_timeouts:
            for test_type in ["matched_gt_y_action", "same_state_oracle"]:
                row = run_one_test(
                    root, tasks, Environment, close_helper, data, codec,
                    idx=item["idx"],
                    condition=item["condition"],
                    visible_seed=item["visible_seed"],
                    window_t=item["window_t"],
                    source_value=item["source_value"],
                    motion_timeout=timeout,
                    test_type=test_type,
                    max_prefix_actions=args.max_prefix_actions,
                    allow_low_conf_prefix=args.allow_low_conf_prefix,
                )
                rows.append(row)
                assert_no_forbidden(f"after_{test_type}_{item['condition']}_{item['idx']}")

    out_csv = root / args.out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    payload = {
        "num_rows": len(rows),
        "selected_windows": selected,
        "samples_per_condition": args.samples_per_condition,
        "motion_timeouts": args.motion_timeouts,
        "max_prefix_actions": args.max_prefix_actions,
        "allow_low_conf_prefix": bool(args.allow_low_conf_prefix),
        "out_csv": args.out_csv,
        "scope": "phase3_6_matched_state_replay_no_phase4_no_cps",
    }
    (root / args.out_json).write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
