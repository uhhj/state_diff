#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

CONDITIONS = ["free", "hidden_breakaway_pin", "hidden_high_friction"]
VARIANTS = [
    "source_gt_action",
    "source_old_idm_future",
    "source_repaired_idm_future",
    "live_retrieved_gt_action",
    "live_repaired_idm_retrieved_future",
    "live_ddpm_mean_repaired_idm",
    "live_condition_retrieval_repaired_idm",
]
FORBIDDEN_PREFIXES = ["tensorflow", "ravens.agents", "ravens.models", "ravens.datasets"]

FIELDNAMES = [
    "status",
    "started_at",
    "finished_at",
    "duration_sec",
    "worker_returncode",
    "worker_timeout",
    "worker_stderr_tail",
    "condition",
    "visible_seed",
    "episode_idx",
    "target_step_idx",
    "variant",
    "context",
    "future_source",
    "uses_condition_label",
    "retrieval_train_index",
    "retrieval_window_idx",
    "retrieval_condition",
    "retrieval_visible_seed",
    "retrieval_window_t",
    "retrieval_source_file",
    "current_model_x_to_retrieved_x_l2",
    "current_model_x_to_retrieved_x_mae",
    "future_nn_condition",
    "future_nn_l2",
    "action_source",
    "primitive",
    "has_params",
    "pose0_x",
    "pose0_y",
    "pose0_z",
    "pose1_x",
    "pose1_y",
    "pose1_z",
    "pull_xy_len",
    "action_ood",
    "clip_frac",
    "action_l2_to_retrieved_gt",
    "action_mae_to_retrieved_gt",
    "pose0_xy_dist_to_retrieved_gt",
    "pose1_xy_dist_to_retrieved_gt",
    "pull_angle_deg_to_retrieved_gt",
    "exec_success",
    "exec_done",
    "exec_reward",
    "exec_final_fraction_before",
    "exec_final_fraction_after",
    "exec_delta_final_fraction",
    "exec_prefix_state_mae",
    "exec_prefix_state_l2",
    "exec_prefix_source",
    "exec_failure_reason",
    "scope",
]


def assert_gate() -> None:
    if os.environ.get("PHASE3_ALLOW_RETRIEVAL_FEASIBILITY", "0") != "1":
        raise SystemExit("[Phase3.11b][BLOCKED] Set PHASE3_ALLOW_RETRIEVAL_FEASIBILITY=1")
    if os.environ.get("PHASE3_RETRIEVAL_FEASIBILITY_CONFIRMED", "0") != "1":
        raise SystemExit("[Phase3.11b][BLOCKED] Set PHASE3_RETRIEVAL_FEASIBILITY_CONFIRMED=1")


def assert_no_forbidden(stage: str) -> None:
    bad: List[str] = []
    for name in sys.modules:
        for prefix in FORBIDDEN_PREFIXES:
            if name == prefix or name.startswith(prefix + "."):
                bad.append(name)
    if bad:
        raise RuntimeError(f"[Phase3.11b][FAIL] forbidden modules loaded at {stage}: {bad[:30]}")


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(path)


def write_row(csv_path: Path, row: Dict[str, Any], write_header: bool) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in FIELDNAMES})
        f.flush()
        os.fsync(f.fileno())


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def first_old_checkpoint(root: Path, checkpoint_root: str) -> Path:
    base = root / checkpoint_root / "state_action"
    for cand in sorted(p for p in base.glob("fold_*_seed_*") if p.is_dir()):
        if (cand / "state_model.pt").exists() and (cand / "inverse_dynamics.pt").exists():
            return cand
    raise FileNotFoundError(f"No complete old checkpoint under {base}")


def checkpoint_from_phase39b_raw(root: Path, raw_path: str, ablation: str) -> Path:
    raw = load_json(root / raw_path)
    p = raw.get("results", {}).get(ablation, {}).get("checkpoint")
    if not p:
        raise FileNotFoundError(f"No checkpoint for {ablation} in {raw_path}")
    pp = Path(p)
    if not pp.is_absolute():
        pp = root / pp
    if not pp.exists():
        raise FileNotFoundError(str(pp))
    return pp


def train_index_to_window_idx(data: Any, retrieval_train_index: int) -> int:
    cond_all = np.asarray([str(x) for x in data["condition_name"]])
    split = np.asarray([str(x) for x in data["split_name"]]) if "split_name" in data.files else np.asarray(["train"] * len(cond_all))
    train_mask = split == "train"
    if not np.any(train_mask):
        train_mask = np.ones(len(cond_all), dtype=bool)
    train_indices = np.where(train_mask)[0]
    if retrieval_train_index < 0 or retrieval_train_index >= len(train_indices):
        raise IndexError(f"retrieval_train_index={retrieval_train_index} out of range for train_indices={len(train_indices)}")
    return int(train_indices[retrieval_train_index])


def sf(x: Any, default: float = float("nan")) -> float:
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default


def si(x: Any, default: int = -1) -> int:
    try:
        if x is None or x == "":
            return default
        return int(float(x))
    except Exception:
        return default


def choose_candidates(root: Path, args: argparse.Namespace) -> List[Dict[str, Any]]:
    step_rows = read_csv_rows(root / args.phase311_step_csv)
    if not step_rows:
        raise FileNotFoundError(f"No Phase3.11 step rows: {root / args.phase311_step_csv}")

    out: List[Dict[str, Any]] = []
    seen: Dict[str, int] = {c: 0 for c in args.conditions}
    seen_keys = set()

    for r in step_rows:
        if r.get("future_source") != "condition_matched_retrieval":
            continue
        cond = r.get("condition", "")
        if cond not in seen:
            continue
        if seen[cond] >= args.candidates_per_condition:
            continue

        retrieval_index = si(r.get("retrieval_index"), -1)
        if retrieval_index < 0:
            continue

        key = (r.get("condition"), r.get("visible_seed"), r.get("episode_idx"), r.get("step_idx"), retrieval_index)
        if key in seen_keys:
            continue
        seen_keys.add(key)

        item = {
            "condition": cond,
            "visible_seed": si(r.get("visible_seed"), 0),
            "episode_idx": si(r.get("episode_idx"), 0),
            "target_step_idx": si(r.get("step_idx"), 0),
            "retrieval_train_index": retrieval_index,
        }
        out.append(item)
        seen[cond] += 1

    if not out:
        raise RuntimeError("No eligible condition_matched_retrieval step candidates found.")

    specs: List[Dict[str, Any]] = []
    for item in out:
        for variant in args.variants:
            specs.append({
                "root": str(root),
                "windows": args.windows,
                "action_template": args.action_template,
                "old_checkpoint_root": args.old_checkpoint_root,
                "phase39b_raw": args.phase39b_raw,
                "best_ablation": args.best_ablation,
                "phase311_step_csv": args.phase311_step_csv,
                "item": item,
                "variant": variant,
                "samples_per_step": args.samples_per_step,
                "motion_timeout": args.motion_timeout,
                "action_clip_std": args.action_clip_std,
                "max_steps_replay": args.max_steps_replay,
                "max_prefix_actions": args.max_prefix_actions,
                "seed_base": args.seed_base,
            })

    if args.max_rows > 0:
        specs = specs[: args.max_rows]
    return specs


def make_error_row(spec: Dict[str, Any], started: float, finished: float, status: str, stderr: str, returncode: Any = "") -> Dict[str, Any]:
    item = spec.get("item", {})
    row = {k: "" for k in FIELDNAMES}
    row.update({
        "status": status,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(started)),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(finished)),
        "duration_sec": f"{finished - started:.3f}",
        "worker_returncode": returncode,
        "worker_timeout": 1 if "timeout" in status else 0,
        "worker_stderr_tail": stderr[-2000:],
        "condition": item.get("condition", ""),
        "visible_seed": item.get("visible_seed", ""),
        "episode_idx": item.get("episode_idx", ""),
        "target_step_idx": item.get("target_step_idx", ""),
        "variant": spec.get("variant", ""),
        "exec_failure_reason": status,
        "scope": "phase3_11b_retrieval_feasibility_no_phase4_no_cps",
    })
    return row


def run_parent(args: argparse.Namespace) -> None:
    assert_gate()
    root = Path(args.root).resolve()
    out_csv = root / args.out_csv
    progress_json = root / args.progress_json
    raw_json = root / args.out_json
    worker_dir = root / args.worker_dir

    if worker_dir.exists() and not args.resume:
        import shutil
        shutil.rmtree(worker_dir)
    worker_dir.mkdir(parents=True, exist_ok=True)

    if out_csv.exists() and not args.resume:
        out_csv.unlink()

    write_header = not out_csv.exists()
    specs = choose_candidates(root, args)

    existing_rows = 0
    if args.resume and out_csv.exists():
        existing_rows = max(0, sum(1 for _ in out_csv.open()) - 1)
        specs = specs[existing_rows:]
        write_header = False

    progress = {
        "status": "running",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "total_specs": existing_rows + len(specs),
        "completed": existing_rows,
        "ok": 0,
        "timeout": 0,
        "failed": 0,
        "last_row": None,
        "scope": "phase3_11b_retrieval_feasibility_no_phase4_no_cps",
    }
    write_json_atomic(progress_json, progress)

    start_all = time.time()
    script = Path(__file__).resolve()

    for i, spec in enumerate(specs, start=existing_rows):
        if args.total_timeout_sec > 0 and time.time() - start_all > args.total_timeout_sec:
            progress["status"] = "stopped_total_timeout"
            write_json_atomic(progress_json, progress)
            break

        spec_path = worker_dir / f"worker_{i:04d}.json"
        out_path = worker_dir / f"worker_{i:04d}_out.json"
        err_path = worker_dir / f"worker_{i:04d}.stderr.txt"
        write_json_atomic(spec_path, spec)

        started = time.time()
        cmd = [sys.executable, str(script), "--worker_json", str(spec_path), "--worker_out_json", str(out_path)]
        stderr_text = ""

        try:
            proc = subprocess.run(
                cmd,
                cwd=str(root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=float(args.row_timeout_sec),
                env=os.environ.copy(),
            )
            finished = time.time()
            stderr_text = (proc.stderr or "") + "\n" + (proc.stdout or "")
            err_path.write_text(stderr_text[-12000:])

            if out_path.exists():
                row = json.loads(out_path.read_text())
                row["worker_returncode"] = proc.returncode
                row["worker_timeout"] = 0
                row["worker_stderr_tail"] = stderr_text[-2000:]
                row["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(started))
                row["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(finished))
                row["duration_sec"] = f"{finished - started:.3f}"
                if proc.returncode != 0 and row.get("status") == "ok":
                    row["status"] = "worker_returned_nonzero"
            else:
                row = make_error_row(spec, started, finished, "worker_failed_no_output", stderr_text, proc.returncode)

        except subprocess.TimeoutExpired as exc:
            finished = time.time()
            stderr_text = ((exc.stderr or "") if isinstance(exc.stderr, str) else str(exc.stderr))[-12000:]
            err_path.write_text(stderr_text)
            row = make_error_row(spec, started, finished, "timeout", stderr_text, "timeout")

        write_row(out_csv, row, write_header)
        write_header = False

        status = str(row.get("status", ""))
        progress["completed"] += 1
        if status == "ok":
            progress["ok"] += 1
        elif "timeout" in status:
            progress["timeout"] += 1
        else:
            progress["failed"] += 1

        progress["last_row"] = {
            "status": row.get("status", ""),
            "condition": row.get("condition", ""),
            "variant": row.get("variant", ""),
            "delta": row.get("exec_delta_final_fraction", ""),
            "failure": row.get("exec_failure_reason", ""),
        }
        write_json_atomic(progress_json, progress)

    if progress["status"] == "running":
        progress["status"] = "completed"
        progress["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        write_json_atomic(progress_json, progress)

    raw = {
        "status": progress["status"],
        "num_rows_written": progress["completed"],
        "ok": progress["ok"],
        "timeout": progress["timeout"],
        "failed": progress["failed"],
        "matrix": {
            "conditions": args.conditions,
            "variants": args.variants,
            "candidates_per_condition": args.candidates_per_condition,
            "max_rows": args.max_rows,
        },
        "out_csv": args.out_csv,
        "progress_json": args.progress_json,
        "worker_dir": args.worker_dir,
        "scope": "phase3_11b_retrieval_feasibility_no_phase4_no_cps",
    }
    write_json_atomic(raw_json, raw)
    print(json.dumps(raw, indent=2, sort_keys=True))


def action_basic_info(action: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        "primitive": action.get("primitive", ""),
        "has_params": int("params" in action),
        "pose0_x": float("nan"),
        "pose0_y": float("nan"),
        "pose0_z": float("nan"),
        "pose1_x": float("nan"),
        "pose1_y": float("nan"),
        "pose1_z": float("nan"),
        "pull_xy_len": float("nan"),
    }
    try:
        p0 = np.asarray(action["params"]["pose0"][0], dtype=np.float32)
        p1 = np.asarray(action["params"]["pose1"][0], dtype=np.float32)
        out.update({
            "pose0_x": float(p0[0]),
            "pose0_y": float(p0[1]),
            "pose0_z": float(p0[2]),
            "pose1_x": float(p1[0]),
            "pose1_y": float(p1[1]),
            "pose1_z": float(p1[2]),
            "pull_xy_len": float(np.linalg.norm(p1[:2] - p0[:2])),
        })
    except Exception:
        pass
    return out


def vector_compare(vec: np.ndarray, gt: np.ndarray) -> Dict[str, Any]:
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    g = np.asarray(gt, dtype=np.float32).reshape(-1)
    diff = v - g
    return {
        "action_l2_to_retrieved_gt": float(np.sqrt(np.mean(diff * diff))),
        "action_mae_to_retrieved_gt": float(np.mean(np.abs(diff))),
    }


def geom_compare(action: Dict[str, Any], gt_action: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        "pose0_xy_dist_to_retrieved_gt": float("nan"),
        "pose1_xy_dist_to_retrieved_gt": float("nan"),
        "pull_angle_deg_to_retrieved_gt": float("nan"),
    }
    try:
        p0 = np.asarray(action["params"]["pose0"][0], dtype=np.float32)[:2]
        p1 = np.asarray(action["params"]["pose1"][0], dtype=np.float32)[:2]
        q0 = np.asarray(gt_action["params"]["pose0"][0], dtype=np.float32)[:2]
        q1 = np.asarray(gt_action["params"]["pose1"][0], dtype=np.float32)[:2]
        a = p1 - p0
        b = q1 - q0
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        angle = float("nan")
        if denom > 1e-8:
            c = float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))
            angle = float(np.degrees(np.arccos(c)))
        out.update({
            "pose0_xy_dist_to_retrieved_gt": float(np.linalg.norm(p0 - q0)),
            "pose1_xy_dist_to_retrieved_gt": float(np.linalg.norm(p1 - q1)),
            "pull_angle_deg_to_retrieved_gt": angle,
        })
    except Exception:
        pass
    return out


def clip_predict_action(codec: Any, idm: Any, hist_states: np.ndarray, future: np.ndarray, clip_std: float) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:
    idm_x = np.concatenate([hist_states.reshape(1, -1), future.reshape(1, -1)], axis=1)
    raw = idm.predict(idm_x)[0].astype(np.float32)
    lo = idm.train_action_mean - clip_std * idm.train_action_std
    hi = idm.train_action_mean + clip_std * idm.train_action_std
    vec = np.clip(raw, lo, hi).astype(np.float32)
    action = codec.decode(vec)
    info = action_basic_info(action)
    info["action_ood"] = float(idm.ood_score(vec.reshape(1, -1))[0])
    info["clip_frac"] = float(np.mean(np.abs(raw - vec) > 1e-6))
    return vec, action, info


def select_condition_retrieval(model_x: np.ndarray, data: Any, condition: str) -> Tuple[int, int]:
    cond_all = np.asarray([str(x) for x in data["condition_name"]])
    split = np.asarray([str(x) for x in data["split_name"]]) if "split_name" in data.files else np.asarray(["train"] * len(cond_all))
    train_mask = split == "train"
    if not np.any(train_mask):
        train_mask = np.ones(len(cond_all), dtype=bool)

    train_indices = np.where(train_mask)[0]
    x_bank = np.asarray(data["state_action_x"], dtype=np.float32)[train_mask]
    labels = cond_all[train_mask]
    cond_mask = labels == condition
    if not np.any(cond_mask):
        raise RuntimeError(f"No train samples for condition={condition}")

    cond_x = x_bank[cond_mask]
    diff = cond_x - model_x.reshape(1, -1)
    d = np.sqrt(np.mean(diff * diff, axis=1))
    local = int(np.argmin(d))
    train_local_indices = np.where(cond_mask)[0]
    retrieval_train_index = int(train_local_indices[local])
    retrieval_window_idx = int(train_indices[retrieval_train_index])
    return retrieval_train_index, retrieval_window_idx


def prepare_live_prefix(spec: Dict[str, Any], data: Any, codec: Any, state_model: Any, idm: Any, tasks: Any, Environment: Any, p34: Any) -> Dict[str, Any]:
    from ccda_phase3.rollout import final_fraction_from_info, pad_history, state_from_live_info

    condition = str(spec["item"]["condition"])
    visible_seed = int(spec["item"]["visible_seed"])
    target_step_idx = int(spec["item"]["target_step_idx"])
    future_source_for_replay = "condition_matched_retrieval"

    th = int(data["th"])
    action_dim = int(data["action_dim"])
    n_beads = int(data["n_beads"])

    os.environ["CCDA_HIDDEN_CONDITION"] = condition
    os.environ["CCDA_VISIBLE_SEED"] = str(visible_seed)
    os.environ["CCDA_PAIR_GROUP"] = f"phase3_11b_live_prefix_{visible_seed}_{target_step_idx}"

    task = tasks.names["hidden-contact-cable-line"]()
    task.mode = "train"
    env = Environment(disp=False, hz=240)
    env.t_lim = float(spec["motion_timeout"])

    state_hist: List[np.ndarray] = []
    action_hist: List[np.ndarray] = []
    prev_xy = None

    try:
        env.reset(task)
        reward_extras = task.reward()[1]
        info = env.info
        reward_extras["task.done"] = task.done()
        info["extras"] = reward_extras

        for step_idx in range(target_step_idx):
            state = state_from_live_info(info, prev_xy=prev_xy)
            prev_xy = state[: n_beads * 2].reshape(n_beads, 2)
            state_hist.append(state)

            hist_states = pad_history(state_hist, th).reshape(-1)
            hist_actions = pad_history(action_hist, th).reshape(-1) if action_hist else np.zeros((th * action_dim,), dtype=np.float32)
            model_x = np.concatenate([hist_states, hist_actions], axis=0).reshape(1, -1)

            if future_source_for_replay == "condition_matched_retrieval":
                _, ridx = select_condition_retrieval(model_x, data, condition)
                future = np.asarray(data["y_state"][ridx], dtype=np.float32).reshape(1, -1)
            else:
                samples = state_model.sample(model_x, n_samples=int(spec["samples_per_step"]), seed=visible_seed + step_idx)[:, 0, :]
                future = np.mean(samples, axis=0, keepdims=True)

            vec, action, _ = clip_predict_action(codec, idm, hist_states, future, float(spec["action_clip_std"]))
            obs, reward, done, info = env.step(action)
            action_hist.append(vec.astype(np.float32))
            if done:
                break

        state = state_from_live_info(info, prev_xy=prev_xy)
        prev_xy = state[: n_beads * 2].reshape(n_beads, 2)
        state_hist.append(state)
        hist_states = pad_history(state_hist, th).reshape(-1)
        hist_actions = pad_history(action_hist, th).reshape(-1) if action_hist else np.zeros((th * action_dim,), dtype=np.float32)
        model_x = np.concatenate([hist_states, hist_actions], axis=0).reshape(1, -1)
        before = final_fraction_from_info(info)

        return {
            "env": env,
            "info": info,
            "hist_states": hist_states,
            "model_x": model_x,
            "fraction_before": before,
        }
    except Exception:
        try:
            p34.close_env_safely(env)
        except Exception:
            pass
        raise


def exec_live_action(ctx: Dict[str, Any], action: Dict[str, Any], p34: Any) -> Dict[str, Any]:
    from ccda_phase3.rollout import final_fraction_from_info

    env = ctx["env"]
    info_before = ctx["info"]
    before = float(ctx["fraction_before"])

    try:
        obs, reward, done, info = env.step(action)
        after = final_fraction_from_info(info)
        success = bool(info.get("extras", {}).get("task.done", done))
        failure = ""
    except Exception as exc:
        reward = float("nan")
        done = False
        after = before
        success = False
        failure = repr(exc)
    finally:
        try:
            p34.close_env_safely(env)
        except Exception:
            pass

    return {
        "exec_success": int(success),
        "exec_done": int(bool(done)),
        "exec_reward": reward,
        "exec_final_fraction_before": before,
        "exec_final_fraction_after": after,
        "exec_delta_final_fraction": after - before if math.isfinite(after) and math.isfinite(before) else float("nan"),
        "exec_prefix_state_mae": float("nan"),
        "exec_prefix_state_l2": float("nan"),
        "exec_prefix_source": "live_replayed_condition_matched_prefix",
        "exec_failure_reason": failure,
    }


def run_worker(args: argparse.Namespace) -> None:
    assert_gate()
    spec = json.loads(Path(args.worker_json).read_text())
    root = Path(spec["root"]).resolve()

    for p in [root, root / "scripts", root / "external/deformable-ravens"]:
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

    import phase3_8_idm_geometry_repair_probe as p38
    import phase3_policy_rollout as p34
    from ccda_phase3.train_utils import load_inverse_model

    p34.set_selected_recoverable_env_defaults()
    tasks, Environment = p34.require_runtime(root)
    assert_no_forbidden("after_runtime_import")

    data, meta = p38.load_windows(root, spec["windows"])
    codec = p38.load_codec(root, data, spec["action_template"])

    old_ckpt = first_old_checkpoint(root, spec["old_checkpoint_root"])
    state_model = p38.load_future_model(old_ckpt / "state_model.pt") if hasattr(p38, "load_future_model") else None
    if state_model is None:
        from ccda_phase3.train_utils import load_future_model
        state_model = load_future_model(old_ckpt / "state_model.pt")

    old_idm = load_inverse_model(old_ckpt / "inverse_dynamics.pt")
    repaired_idm = load_inverse_model(checkpoint_from_phase39b_raw(root, spec["phase39b_raw"], spec["best_ablation"]))

    item = spec["item"]
    variant = str(spec["variant"])
    retrieval_train_index = int(item["retrieval_train_index"])
    retrieval_window_idx = train_index_to_window_idx(data, retrieval_train_index)

    cond_all = np.asarray([str(x) for x in data["condition_name"]])
    retrieval_condition = str(cond_all[retrieval_window_idx])
    retrieval_visible_seed = str(np.asarray([str(x) for x in data["visible_seed"]])[retrieval_window_idx])
    retrieval_window_t = int(np.asarray(data["window_t"]).astype(int)[retrieval_window_idx])
    retrieval_source_file = str(np.asarray([str(x) for x in data["source_file"]])[retrieval_window_idx])

    retrieved_future = np.asarray(data["y_state"][retrieval_window_idx], dtype=np.float32).reshape(1, -1)
    retrieved_gt_vec = np.asarray(data["y_action"][retrieval_window_idx], dtype=np.float32).reshape(-1)
    retrieved_gt_action = codec.decode(retrieved_gt_vec)
    retrieved_gt_info = action_basic_info(retrieved_gt_action)

    base_row: Dict[str, Any] = {
        "status": "ok",
        "condition": item["condition"],
        "visible_seed": item["visible_seed"],
        "episode_idx": item["episode_idx"],
        "target_step_idx": item["target_step_idx"],
        "variant": variant,
        "future_source": "retrieved_y_state",
        "uses_condition_label": int("condition" in variant or "retrieved" in variant),
        "retrieval_train_index": retrieval_train_index,
        "retrieval_window_idx": retrieval_window_idx,
        "retrieval_condition": retrieval_condition,
        "retrieval_visible_seed": retrieval_visible_seed,
        "retrieval_window_t": retrieval_window_t,
        "retrieval_source_file": retrieval_source_file,
        "scope": "phase3_11b_retrieval_feasibility_no_phase4_no_cps",
    }

    if variant.startswith("source_"):
        p36 = p38.import_phase36(root)
        tasks2, Environment2, close_helper = p36.import_tf_free_ravens(root)
        assert_no_forbidden("after_source_runtime_import")

        source_item = {
            "idx": retrieval_window_idx,
            "condition": retrieval_condition,
            "visible_seed": retrieval_visible_seed,
            "window_t": retrieval_window_t,
            "source_file": retrieval_source_file,
        }

        if variant == "source_gt_action":
            vec = retrieved_gt_vec
            action = retrieved_gt_action
            action_source = "retrieved_y_action"
            info = retrieved_gt_info
            info["action_ood"] = float("nan")
            info["clip_frac"] = 0.0

        elif variant == "source_old_idm_future":
            _, old_vec, _, _, _ = p38.build_idm_vectors(
                data=data,
                idx=retrieval_window_idx,
                baseline="state_action",
                state_model=state_model,
                idm=old_idm,
                pred_samples=int(spec["samples_per_step"]),
                seed=int(spec["seed_base"]) + retrieval_window_idx,
                clip_std=float(spec["action_clip_std"]),
            )
            vec = old_vec
            action = codec.decode(vec)
            info = action_basic_info(action)
            info["action_ood"] = float(old_idm.ood_score(np.asarray(vec).reshape(1, -1))[0])
            info["clip_frac"] = float("nan")
            action_source = "old_idm_retrieved_future"

        elif variant == "source_repaired_idm_future":
            _, new_vec, _, _, _ = p38.build_idm_vectors(
                data=data,
                idx=retrieval_window_idx,
                baseline="state_action",
                state_model=state_model,
                idm=repaired_idm,
                pred_samples=int(spec["samples_per_step"]),
                seed=int(spec["seed_base"]) + retrieval_window_idx,
                clip_std=float(spec["action_clip_std"]),
            )
            vec = new_vec
            action = codec.decode(vec)
            info = action_basic_info(action)
            info["action_ood"] = float(repaired_idm.ood_score(np.asarray(vec).reshape(1, -1))[0])
            info["clip_frac"] = float("nan")
            action_source = "repaired_idm_retrieved_future"
        else:
            raise RuntimeError(f"Unknown source variant: {variant}")

        exec_info = p38.execute_after_matched_prefix(
            root=root,
            p36=p36,
            tasks=tasks2,
            Environment=Environment2,
            close_helper=close_helper,
            data=data,
            codec=codec,
            item=source_item,
            action=action,
            timeout=float(spec["motion_timeout"]),
            max_prefix_actions=int(spec["max_prefix_actions"]),
        )

        row = dict(base_row)
        row["context"] = "source_prefix"
        row["action_source"] = action_source
        row.update(info)
        row.update(vector_compare(np.asarray(vec), retrieved_gt_vec))
        row.update(geom_compare(action, retrieved_gt_action))
        row.update(exec_info)

    else:
        ctx = prepare_live_prefix(spec, data, codec, state_model, repaired_idm, tasks, Environment, p34)
        model_x = ctx["model_x"]
        hist_states = ctx["hist_states"]

        retrieved_x = np.asarray(data["state_action_x"][retrieval_window_idx], dtype=np.float32).reshape(1, -1)
        diff = model_x - retrieved_x
        current_l2 = float(np.sqrt(np.mean(diff * diff)))
        current_mae = float(np.mean(np.abs(diff)))

        if variant == "live_retrieved_gt_action":
            vec = retrieved_gt_vec
            action = retrieved_gt_action
            info = retrieved_gt_info
            info["action_ood"] = float("nan")
            info["clip_frac"] = 0.0
            action_source = "retrieved_y_action"

        elif variant == "live_repaired_idm_retrieved_future":
            vec, action, info = clip_predict_action(codec, repaired_idm, hist_states, retrieved_future, float(spec["action_clip_std"]))
            action_source = "repaired_idm_retrieved_future"

        elif variant == "live_ddpm_mean_repaired_idm":
            samples = state_model.sample(
                model_x,
                n_samples=int(spec["samples_per_step"]),
                seed=int(item["visible_seed"]) + int(item["target_step_idx"]),
            )[:, 0, :]
            future = np.mean(samples, axis=0, keepdims=True)
            vec, action, info = clip_predict_action(codec, repaired_idm, hist_states, future, float(spec["action_clip_std"]))
            action_source = "repaired_idm_ddpm_mean"

        elif variant == "live_condition_retrieval_repaired_idm":
            _, ridx = select_condition_retrieval(model_x, data, str(item["condition"]))
            future = np.asarray(data["y_state"][ridx], dtype=np.float32).reshape(1, -1)
            vec, action, info = clip_predict_action(codec, repaired_idm, hist_states, future, float(spec["action_clip_std"]))
            action_source = "repaired_idm_condition_retrieval"
        else:
            raise RuntimeError(f"Unknown live variant: {variant}")

        exec_info = exec_live_action(ctx, action, p34)

        row = dict(base_row)
        row["context"] = "live_prefix"
        row["action_source"] = action_source
        row["current_model_x_to_retrieved_x_l2"] = current_l2
        row["current_model_x_to_retrieved_x_mae"] = current_mae
        row.update(info)
        row.update(vector_compare(np.asarray(vec), retrieved_gt_vec))
        row.update(geom_compare(action, retrieved_gt_action))
        row.update(exec_info)

    Path(args.worker_out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.worker_out_json).write_text(json.dumps(row, indent=2, sort_keys=True))
    print(json.dumps({
        "status": row.get("status"),
        "condition": row.get("condition"),
        "variant": row.get("variant"),
        "delta": row.get("exec_delta_final_fraction"),
        "failure": row.get("exec_failure_reason"),
    }, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--windows", default="data/phase3_state_diff_windows/phase3_windows.npz")
    parser.add_argument("--action_template", default="data/phase3_state_diff_windows/phase3_action_template.pkl")
    parser.add_argument("--old_checkpoint_root", default="checkpoints/phase3")
    parser.add_argument("--phase39b_raw", default="reports/phase3_9b_ablation_raw_summary.json")
    parser.add_argument("--phase311_step_csv", default="reports/phase3_11_future_source_swap_steps.csv")
    parser.add_argument("--best_ablation", default="xy_only_high_weight")
    parser.add_argument("--conditions", nargs="+", default=CONDITIONS)
    parser.add_argument("--variants", nargs="+", default=VARIANTS)
    parser.add_argument("--candidates_per_condition", type=int, default=3)
    parser.add_argument("--samples_per_step", type=int, default=32)
    parser.add_argument("--motion_timeout", type=float, default=15.0)
    parser.add_argument("--action_clip_std", type=float, default=3.0)
    parser.add_argument("--max_steps_replay", type=int, default=16)
    parser.add_argument("--max_prefix_actions", type=int, default=20)
    parser.add_argument("--seed_base", type=int, default=311500)
    parser.add_argument("--row_timeout_sec", type=float, default=360.0)
    parser.add_argument("--total_timeout_sec", type=float, default=14400.0)
    parser.add_argument("--max_rows", type=int, default=63)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--out_csv", default="reports/phase3_11b_retrieval_feasibility_trials.csv")
    parser.add_argument("--out_json", default="reports/phase3_11b_retrieval_feasibility_raw_summary.json")
    parser.add_argument("--progress_json", default="reports/phase3_11b_retrieval_feasibility_progress.json")
    parser.add_argument("--worker_dir", default="reports/phase3_11b_workers")
    parser.add_argument("--worker_json", default="")
    parser.add_argument("--worker_out_json", default="")
    args = parser.parse_args()

    if args.worker_json:
        run_worker(args)
    else:
        run_parent(args)


if __name__ == "__main__":
    main()
