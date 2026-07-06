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
from typing import Any, Dict, List, Tuple

import numpy as np

AUDIT_CONDITIONS = ["free", "hidden_breakaway_pin", "hidden_high_friction"]
POLICIES = ["old_state_action", "phase39_default_geometry", "phase39b_xy_only_high_weight"]
FORBIDDEN_PREFIXES = ["tensorflow", "ravens.agents", "ravens.models", "ravens.datasets"]

EPISODE_FIELDS = [
    "status", "started_at", "finished_at", "duration_sec", "worker_returncode",
    "worker_timeout", "worker_stderr_tail",
    "policy", "condition", "visible_seed", "episode_idx",
    "success", "final_fraction", "initial_fraction", "delta_fraction",
    "final_curve", "num_steps", "failure_reason",
    "mean_step_fraction_delta", "num_positive_fraction_steps",
    "mean_model_x_ood_l2", "mean_input_nn_l2",
    "mean_future_std", "mean_future_norm", "mean_future_nn_l2",
    "mean_future_nn_mae", "future_condition_match_rate",
    "dominant_nearest_future_condition",
    "mean_chosen_action_ood", "max_chosen_action_ood",
    "mean_chosen_clip_frac", "mean_chosen_pull_len",
    "mean_old_pull_len", "mean_default_pull_len", "mean_best_pull_len",
    "mean_old_action_ood", "mean_default_action_ood", "mean_best_action_ood",
    "checkpoint_state_model", "checkpoint_inverse_dynamics",
    "primary_hidden_condition", "diagnostic_hidden_condition", "scope",
]

STEP_FIELDS = [
    "policy", "condition", "visible_seed", "episode_idx", "step_idx",
    "fraction_before", "fraction_after", "fraction_delta",
    "done_after", "reward",
    "model_x_ood_l2", "model_x_ood_mean_abs", "input_nn_l2",
    "future_std_mean", "future_norm", "future_nn_l2", "future_nn_mae",
    "future_nearest_condition", "future_condition_match",
    "chosen_action_ood", "chosen_clip_frac",
    "chosen_pose0_x", "chosen_pose0_y", "chosen_pose1_x", "chosen_pose1_y",
    "chosen_pull_len", "chosen_pose0_step_delta", "chosen_pose1_step_delta",
    "old_action_ood", "old_clip_frac", "old_pose0_x", "old_pose0_y", "old_pose1_x", "old_pose1_y", "old_pull_len",
    "default_action_ood", "default_clip_frac", "default_pose0_x", "default_pose0_y", "default_pose1_x", "default_pose1_y", "default_pull_len",
    "best_action_ood", "best_clip_frac", "best_pose0_x", "best_pose0_y", "best_pose1_x", "best_pose1_y", "best_pull_len",
    "scope",
]


def assert_gate() -> None:
    if os.environ.get("PHASE3_ALLOW_FUTURE_ROLLOUT_AUDIT", "0") != "1":
        raise SystemExit("[Phase3.10b][BLOCKED] Set PHASE3_ALLOW_FUTURE_ROLLOUT_AUDIT=1")
    if os.environ.get("PHASE3_FUTURE_ROLLOUT_AUDIT_CONFIRMED", "0") != "1":
        raise SystemExit("[Phase3.10b][BLOCKED] Set PHASE3_FUTURE_ROLLOUT_AUDIT_CONFIRMED=1")


def assert_no_forbidden(stage: str) -> None:
    bad: List[str] = []
    for name in sys.modules:
        for prefix in FORBIDDEN_PREFIXES:
            if name == prefix or name.startswith(prefix + "."):
                bad.append(name)
    if bad:
        raise RuntimeError(f"[Phase3.10b][FAIL] forbidden modules loaded at {stage}: {bad[:30]}")


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(path)


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def write_csv_row(path: Path, row: Dict[str, Any], fields: List[str], write_header: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fields})
        f.flush()
        os.fsync(f.fileno())


def first_old_checkpoint(root: Path, checkpoint_root: str) -> Path:
    base = root / checkpoint_root / "state_action"
    for cand in sorted(p for p in base.glob("fold_*_seed_*") if p.is_dir()):
        if (cand / "state_model.pt").exists() and (cand / "inverse_dynamics.pt").exists():
            return cand
    raise FileNotFoundError(f"No complete checkpoint under {base}")


def checkpoint_from_phase39_train(root: Path, path: str) -> Path:
    summary = load_json(root / path)
    p = summary.get("inverse_dynamics_path")
    if not p:
        raise FileNotFoundError(f"No inverse_dynamics_path in {path}")
    pp = Path(p)
    if not pp.is_absolute():
        pp = root / pp
    if not pp.exists():
        raise FileNotFoundError(str(pp))
    return pp


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


def build_policy_table(args: argparse.Namespace) -> Dict[str, Dict[str, str]]:
    root = Path(args.root).resolve()
    old_ckpt = first_old_checkpoint(root, args.old_checkpoint_root)
    default_idm = checkpoint_from_phase39_train(root, args.phase39_train_summary)
    best_idm = checkpoint_from_phase39b_raw(root, args.phase39b_raw, args.best_ablation)
    state_model = old_ckpt / "state_model.pt"
    return {
        "old_state_action": {
            "state_model_path": str(state_model),
            "idm_path": str(old_ckpt / "inverse_dynamics.pt"),
        },
        "phase39_default_geometry": {
            "state_model_path": str(state_model),
            "idm_path": str(default_idm),
        },
        f"phase39b_{args.best_ablation}": {
            "state_model_path": str(state_model),
            "idm_path": str(best_idm),
        },
    }


def build_specs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    policies = build_policy_table(args)
    missing = [p for p in args.policies if p not in policies]
    if missing:
        raise RuntimeError(f"unknown policies={missing}; available={sorted(policies)}")

    specs: List[Dict[str, Any]] = []
    for policy in args.policies:
        for condition in args.conditions:
            for ep in range(int(args.episodes_per_condition)):
                seed = int(args.seed_start) + ep
                specs.append({
                    "root": str(Path(args.root).resolve()),
                    "policy": policy,
                    "condition": condition,
                    "visible_seed": seed,
                    "episode_idx": ep,
                    "state_model_path": policies[policy]["state_model_path"],
                    "idm_paths": {k: v["idm_path"] for k, v in policies.items()},
                    "chosen_idm_path": policies[policy]["idm_path"],
                    "windows": args.windows,
                    "action_template": args.action_template,
                    "max_steps": args.max_steps,
                    "samples_per_step": args.samples_per_step,
                    "motion_timeout": args.motion_timeout,
                    "action_clip_std": args.action_clip_std,
                })
    if args.max_rows > 0:
        specs = specs[: args.max_rows]
    return specs


def error_episode(spec: Dict[str, Any], started: float, finished: float, status: str, stderr: str, returncode: Any = "") -> Dict[str, Any]:
    row = {k: "" for k in EPISODE_FIELDS}
    row.update({
        "status": status,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(started)),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(finished)),
        "duration_sec": f"{finished - started:.3f}",
        "worker_returncode": returncode,
        "worker_timeout": 1 if "timeout" in status else 0,
        "worker_stderr_tail": stderr[-2000:],
        "policy": spec.get("policy", ""),
        "condition": spec.get("condition", ""),
        "visible_seed": spec.get("visible_seed", ""),
        "episode_idx": spec.get("episode_idx", ""),
        "failure_reason": status,
        "checkpoint_state_model": spec.get("state_model_path", ""),
        "checkpoint_inverse_dynamics": spec.get("chosen_idm_path", ""),
        "primary_hidden_condition": "hidden_breakaway_pin",
        "diagnostic_hidden_condition": "hidden_pin",
        "scope": "phase3_10b_future_rollout_audit_no_phase4_no_cps",
    })
    return row


def run_parent(args: argparse.Namespace) -> None:
    assert_gate()
    root = Path(args.root).resolve()
    episode_csv = root / args.out_episode_csv
    step_csv = root / args.out_step_csv
    progress_path = root / args.progress_json
    raw_path = root / args.out_json
    worker_dir = root / args.worker_dir

    if worker_dir.exists() and not args.resume:
        import shutil
        shutil.rmtree(worker_dir)
    worker_dir.mkdir(parents=True, exist_ok=True)

    if not args.resume:
        for p in [episode_csv, step_csv]:
            if p.exists():
                p.unlink()

    episode_header = not episode_csv.exists()
    step_header = not step_csv.exists()
    specs = build_specs(args)

    existing = 0
    if args.resume and episode_csv.exists():
        existing = max(0, sum(1 for _ in episode_csv.open()) - 1)
        specs = specs[existing:]
        episode_header = False
        step_header = step_csv.stat().st_size == 0 if step_csv.exists() else True

    progress = {
        "status": "running",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "total_specs": existing + len(specs),
        "completed": existing,
        "ok": 0,
        "timeout": 0,
        "failed": 0,
        "last_row": None,
        "out_episode_csv": args.out_episode_csv,
        "out_step_csv": args.out_step_csv,
        "scope": "phase3_10b_future_rollout_audit_no_phase4_no_cps",
    }
    write_json_atomic(progress_path, progress)

    start_all = time.time()
    script = Path(__file__).resolve()

    for i, spec in enumerate(specs, start=existing):
        if args.total_timeout_sec > 0 and time.time() - start_all > args.total_timeout_sec:
            progress["status"] = "stopped_total_timeout"
            write_json_atomic(progress_path, progress)
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
                payload = json.loads(out_path.read_text())
                ep = payload.get("episode", {})
                steps = payload.get("steps", [])
                ep["worker_returncode"] = proc.returncode
                ep["worker_timeout"] = 0
                ep["worker_stderr_tail"] = stderr_text[-2000:]
                ep["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(started))
                ep["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(finished))
                ep["duration_sec"] = f"{finished - started:.3f}"
                if proc.returncode != 0 and ep.get("status") == "ok":
                    ep["status"] = "worker_returned_nonzero"
            else:
                ep = error_episode(spec, started, finished, "worker_failed_no_output", stderr_text, proc.returncode)
                steps = []
        except subprocess.TimeoutExpired as exc:
            finished = time.time()
            stderr_text = ((exc.stderr or "") if isinstance(exc.stderr, str) else str(exc.stderr))[-12000:]
            err_path.write_text(stderr_text)
            ep = error_episode(spec, started, finished, "timeout", stderr_text, "timeout")
            steps = []

        write_csv_row(episode_csv, ep, EPISODE_FIELDS, episode_header)
        episode_header = False
        for step in steps:
            write_csv_row(step_csv, step, STEP_FIELDS, step_header)
            step_header = False

        status = str(ep.get("status", ""))
        progress["completed"] += 1
        if status == "ok" and not ep.get("failure_reason"):
            progress["ok"] += 1
        elif "timeout" in status:
            progress["timeout"] += 1
        else:
            progress["failed"] += 1
        progress["last_row"] = {
            "status": ep.get("status", ""),
            "policy": ep.get("policy", ""),
            "condition": ep.get("condition", ""),
            "final_fraction": ep.get("final_fraction", ""),
            "failure_reason": ep.get("failure_reason", ""),
        }
        write_json_atomic(progress_path, progress)

    if progress["status"] == "running":
        progress["status"] = "completed"
        progress["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        write_json_atomic(progress_path, progress)

    raw = {
        "status": progress["status"],
        "num_episodes_written": progress["completed"],
        "ok": progress["ok"],
        "timeout": progress["timeout"],
        "failed": progress["failed"],
        "matrix": {
            "policies": args.policies,
            "conditions": args.conditions,
            "episodes_per_condition": args.episodes_per_condition,
            "max_steps": args.max_steps,
            "samples_per_step": args.samples_per_step,
        },
        "out_episode_csv": args.out_episode_csv,
        "out_step_csv": args.out_step_csv,
        "progress_json": args.progress_json,
        "worker_dir": args.worker_dir,
        "scope": "phase3_10b_future_rollout_audit_no_phase4_no_cps",
    }
    write_json_atomic(raw_path, raw)
    print(json.dumps(raw, indent=2, sort_keys=True))


def nearest_stats(vec: np.ndarray, mat: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    if mat.size == 0:
        return {
            "future_nn_l2": float("nan"),
            "future_nn_mae": float("nan"),
            "future_nearest_condition": "",
        }
    v = np.asarray(vec, dtype=np.float32).reshape(1, -1)
    m = np.asarray(mat, dtype=np.float32)
    diff = m - v
    l2 = np.sqrt(np.mean(diff * diff, axis=1))
    idx = int(np.argmin(l2))
    mae = float(np.mean(np.abs(diff[idx])))
    return {
        "future_nn_l2": float(l2[idx]),
        "future_nn_mae": mae,
        "future_nearest_condition": str(labels[idx]) if len(labels) > idx else "",
    }


def input_nn_l2(x: np.ndarray, mat: np.ndarray) -> float:
    if mat.size == 0:
        return float("nan")
    diff = np.asarray(mat, dtype=np.float32) - np.asarray(x, dtype=np.float32).reshape(1, -1)
    d = np.sqrt(np.mean(diff * diff, axis=1))
    return float(np.min(d))


def model_ood(model: Any, x: np.ndarray) -> Tuple[float, float]:
    try:
        z = model.x_std.transform(x)
        return float(np.sqrt(np.mean(z * z))), float(np.mean(np.abs(z)))
    except Exception:
        return float("nan"), float("nan")


def action_info(codec: Any, idm: Any, idm_x: np.ndarray, clip_std: float) -> Dict[str, Any]:
    raw = idm.predict(idm_x)[0].astype(np.float32)
    lo = idm.train_action_mean - clip_std * idm.train_action_std
    hi = idm.train_action_mean + clip_std * idm.train_action_std
    vec = np.clip(raw, lo, hi).astype(np.float32)
    clip_frac = float(np.mean(np.abs(raw - vec) > 1e-6))
    act = codec.decode(vec)
    out: Dict[str, Any] = {
        "vec": vec,
        "action": act,
        "action_ood": float(idm.ood_score(vec.reshape(1, -1))[0]),
        "clip_frac": clip_frac,
        "pose0_x": float("nan"),
        "pose0_y": float("nan"),
        "pose1_x": float("nan"),
        "pose1_y": float("nan"),
        "pull_len": float("nan"),
    }
    try:
        p0 = np.asarray(act["params"]["pose0"][0], dtype=np.float32)[:2]
        p1 = np.asarray(act["params"]["pose1"][0], dtype=np.float32)[:2]
        out.update({
            "pose0_x": float(p0[0]),
            "pose0_y": float(p0[1]),
            "pose1_x": float(p1[0]),
            "pose1_y": float(p1[1]),
            "pull_len": float(np.linalg.norm(p1 - p0)),
        })
    except Exception:
        pass
    return out


def mode_string(xs: List[str]) -> str:
    counts: Dict[str, int] = {}
    for x in xs:
        counts[x] = counts.get(x, 0) + 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0] if counts else ""


def run_worker(args: argparse.Namespace) -> None:
    assert_gate()
    spec = json.loads(Path(args.worker_json).read_text())
    root = Path(spec["root"]).resolve()
    for p in [root, root / "scripts", root / "external/deformable-ravens"]:
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

    from ccda_phase3.data_io import load_action_codec_from_template
    from ccda_phase3.metrics import curve_metric_from_state
    from ccda_phase3.rollout import final_fraction_from_info, pad_history, state_from_live_info
    from ccda_phase3.train_utils import load_future_model, load_inverse_model
    import phase3_policy_rollout as p34

    p34.set_selected_recoverable_env_defaults()
    tasks, Environment = p34.require_runtime(root)
    assert_no_forbidden("after_runtime_import")

    data = np.load(root / spec["windows"], allow_pickle=True)
    template_raw = Path(str(data["action_template_json_or_pickle_path"]))
    template_path = template_raw if template_raw.is_absolute() else root / template_raw
    codec = load_action_codec_from_template(template_path)
    if codec.dim() != 14 or codec.summary().get("num_camera_config_paths") != 0:
        raise RuntimeError(f"bad action codec: {codec.summary()}")

    th = int(data["th"])
    action_dim = int(data["action_dim"])
    n_beads = int(data["n_beads"])
    if action_dim != 14:
        raise RuntimeError(f"action_dim must be 14, got {action_dim}")

    state_model = load_future_model(Path(spec["state_model_path"]))
    idms = {name: load_inverse_model(Path(path)) for name, path in spec["idm_paths"].items()}
    assert_no_forbidden("after_model_load")

    train_future = np.asarray(data["y_state"], dtype=np.float32).reshape(len(data["y_state"]), -1)
    cond_labels = np.asarray([str(x) for x in data["condition_name"]])
    split = np.asarray([str(x) for x in data["split_name"]]) if "split_name" in data.files else np.asarray(["train"] * len(cond_labels))
    train_mask = split == "train"
    future_bank = train_future[train_mask] if np.any(train_mask) else train_future
    future_labels = cond_labels[train_mask] if np.any(train_mask) else cond_labels

    x_bank = np.asarray(data["state_action_x"], dtype=np.float32)
    x_bank = x_bank[train_mask] if np.any(train_mask) and len(x_bank) == len(train_mask) else x_bank

    policy = str(spec["policy"])
    condition = str(spec["condition"])
    visible_seed = int(spec["visible_seed"])
    episode_idx = int(spec["episode_idx"])

    os.environ["CCDA_HIDDEN_CONDITION"] = condition
    os.environ["CCDA_VISIBLE_SEED"] = str(visible_seed)
    os.environ["CCDA_PAIR_GROUP"] = f"phase3_10b_{policy}_{visible_seed}"

    task = tasks.names["hidden-contact-cable-line"]()
    task.mode = "train"

    env = None
    steps: List[Dict[str, Any]] = []
    episode: Dict[str, Any] = {
        "status": "ok",
        "policy": policy,
        "condition": condition,
        "visible_seed": visible_seed,
        "episode_idx": episode_idx,
        "success": 0,
        "initial_fraction": float("nan"),
        "final_fraction": float("nan"),
        "delta_fraction": float("nan"),
        "final_curve": float("nan"),
        "num_steps": 0,
        "failure_reason": "",
        "checkpoint_state_model": spec["state_model_path"],
        "checkpoint_inverse_dynamics": spec["chosen_idm_path"],
        "primary_hidden_condition": "hidden_breakaway_pin",
        "diagnostic_hidden_condition": "hidden_pin",
        "scope": "phase3_10b_future_rollout_audit_no_phase4_no_cps",
    }

    try:
        env = Environment(disp=False, hz=240)
        env.t_lim = float(spec["motion_timeout"])
        env.reset(task)

        reward_extras = task.reward()[1]
        info = env.info
        reward_extras["task.done"] = task.done()
        info["extras"] = reward_extras

        state_hist: List[np.ndarray] = []
        action_hist: List[np.ndarray] = []
        prev_xy = None
        prev_pose0 = None
        prev_pose1 = None
        previous_fraction = final_fraction_from_info(info)
        episode["initial_fraction"] = previous_fraction

        chosen_oods: List[float] = []
        chosen_clip: List[float] = []
        chosen_pull: List[float] = []
        old_pull: List[float] = []
        default_pull: List[float] = []
        best_pull: List[float] = []
        old_ood: List[float] = []
        default_ood: List[float] = []
        best_ood: List[float] = []
        model_ood_l2s: List[float] = []
        input_nns: List[float] = []
        future_stds: List[float] = []
        future_norms: List[float] = []
        future_nns: List[float] = []
        future_maes: List[float] = []
        future_match: List[float] = []
        future_condition_match: List[float] = []
        future_nearest_conditions: List[str] = []
        step_deltas: List[float] = []

        done = False

        for step_idx in range(int(spec["max_steps"])):
            # Observable model input only:
            # live state history + past action history.
            # Do not concatenate condition labels, hidden metadata, reward/success,
            # final_fraction, branch labels, or future labels.
            state = state_from_live_info(info, prev_xy=prev_xy)
            prev_xy = state[: n_beads * 2].reshape(n_beads, 2)
            state_hist.append(state)

            hist_states = pad_history(state_hist, th).reshape(-1)
            if action_hist:
                hist_actions = pad_history(action_hist, th).reshape(-1)
            else:
                hist_actions = np.zeros((th * action_dim,), dtype=np.float32)

            model_x = np.concatenate([hist_states, hist_actions], axis=0).reshape(1, -1)
            mx_l2, mx_abs = model_ood(state_model, model_x)
            model_ood_l2s.append(mx_l2)
            input_nn = input_nn_l2(model_x.reshape(-1), x_bank)
            input_nns.append(input_nn)

            samples = state_model.sample(
                model_x,
                n_samples=int(spec["samples_per_step"]),
                seed=visible_seed + step_idx,
            )[:, 0, :]
            mean_future = np.mean(samples, axis=0, keepdims=True)
            future_std = float(np.mean(np.std(samples, axis=0)))
            future_norm = float(np.sqrt(np.mean(mean_future * mean_future)))
            future_stds.append(future_std)
            future_norms.append(future_norm)

            nn = nearest_stats(mean_future.reshape(-1), future_bank, future_labels)
            future_nns.append(float(nn["future_nn_l2"]))
            future_maes.append(float(nn["future_nn_mae"]))
            nearest_cond = str(nn["future_nearest_condition"])
            future_nearest_conditions.append(nearest_cond)
            future_condition_match.append(1.0 if nearest_cond == condition else 0.0)

            idm_x = np.concatenate([hist_states.reshape(1, -1), mean_future], axis=1)

            infos: Dict[str, Dict[str, Any]] = {}
            for name, idm in idms.items():
                infos[name] = action_info(codec, idm, idm_x, float(spec["action_clip_std"]))

            chosen = infos[policy]
            chosen_action = chosen["action"]
            action_hist.append(chosen["vec"].astype(np.float32))

            fraction_before = previous_fraction
            try:
                obs, reward, done, info = env.step(chosen_action)
            except Exception as exc:
                reward = float("nan")
                done = False
                episode["failure_reason"] = repr(exc)
                break

            fraction_after = final_fraction_from_info(info)
            delta = fraction_after - fraction_before if math.isfinite(fraction_after) and math.isfinite(fraction_before) else float("nan")
            previous_fraction = fraction_after
            step_deltas.append(delta)

            p0_step_delta = float("nan")
            p1_step_delta = float("nan")
            try:
                cp0 = np.asarray([chosen["pose0_x"], chosen["pose0_y"]], dtype=np.float32)
                cp1 = np.asarray([chosen["pose1_x"], chosen["pose1_y"]], dtype=np.float32)
                if prev_pose0 is not None:
                    p0_step_delta = float(np.linalg.norm(cp0 - prev_pose0))
                if prev_pose1 is not None:
                    p1_step_delta = float(np.linalg.norm(cp1 - prev_pose1))
                prev_pose0 = cp0
                prev_pose1 = cp1
            except Exception:
                pass

            chosen_oods.append(float(chosen["action_ood"]))
            chosen_clip.append(float(chosen["clip_frac"]))
            chosen_pull.append(float(chosen["pull_len"]))
            old_pull.append(float(infos["old_state_action"]["pull_len"]))
            default_pull.append(float(infos["phase39_default_geometry"]["pull_len"]))
            best_pull.append(float(infos["phase39b_xy_only_high_weight"]["pull_len"]))
            old_ood.append(float(infos["old_state_action"]["action_ood"]))
            default_ood.append(float(infos["phase39_default_geometry"]["action_ood"]))
            best_ood.append(float(infos["phase39b_xy_only_high_weight"]["action_ood"]))

            step_row: Dict[str, Any] = {
                "policy": policy,
                "condition": condition,
                "visible_seed": visible_seed,
                "episode_idx": episode_idx,
                "step_idx": step_idx,
                "fraction_before": fraction_before,
                "fraction_after": fraction_after,
                "fraction_delta": delta,
                "done_after": int(bool(done)),
                "reward": reward,
                "model_x_ood_l2": mx_l2,
                "model_x_ood_mean_abs": mx_abs,
                "input_nn_l2": input_nn,
                "future_std_mean": future_std,
                "future_norm": future_norm,
                "future_nn_l2": nn["future_nn_l2"],
                "future_nn_mae": nn["future_nn_mae"],
                "future_nearest_condition": nearest_cond,
                "future_condition_match": 1 if nearest_cond == condition else 0,
                "chosen_action_ood": chosen["action_ood"],
                "chosen_clip_frac": chosen["clip_frac"],
                "chosen_pose0_x": chosen["pose0_x"],
                "chosen_pose0_y": chosen["pose0_y"],
                "chosen_pose1_x": chosen["pose1_x"],
                "chosen_pose1_y": chosen["pose1_y"],
                "chosen_pull_len": chosen["pull_len"],
                "chosen_pose0_step_delta": p0_step_delta,
                "chosen_pose1_step_delta": p1_step_delta,
                "old_action_ood": infos["old_state_action"]["action_ood"],
                "old_clip_frac": infos["old_state_action"]["clip_frac"],
                "old_pose0_x": infos["old_state_action"]["pose0_x"],
                "old_pose0_y": infos["old_state_action"]["pose0_y"],
                "old_pose1_x": infos["old_state_action"]["pose1_x"],
                "old_pose1_y": infos["old_state_action"]["pose1_y"],
                "old_pull_len": infos["old_state_action"]["pull_len"],
                "default_action_ood": infos["phase39_default_geometry"]["action_ood"],
                "default_clip_frac": infos["phase39_default_geometry"]["clip_frac"],
                "default_pose0_x": infos["phase39_default_geometry"]["pose0_x"],
                "default_pose0_y": infos["phase39_default_geometry"]["pose0_y"],
                "default_pose1_x": infos["phase39_default_geometry"]["pose1_x"],
                "default_pose1_y": infos["phase39_default_geometry"]["pose1_y"],
                "default_pull_len": infos["phase39_default_geometry"]["pull_len"],
                "best_action_ood": infos["phase39b_xy_only_high_weight"]["action_ood"],
                "best_clip_frac": infos["phase39b_xy_only_high_weight"]["clip_frac"],
                "best_pose0_x": infos["phase39b_xy_only_high_weight"]["pose0_x"],
                "best_pose0_y": infos["phase39b_xy_only_high_weight"]["pose0_y"],
                "best_pose1_x": infos["phase39b_xy_only_high_weight"]["pose1_x"],
                "best_pose1_y": infos["phase39b_xy_only_high_weight"]["pose1_y"],
                "best_pull_len": infos["phase39b_xy_only_high_weight"]["pull_len"],
                "scope": "phase3_10b_future_rollout_audit_no_phase4_no_cps",
            }
            steps.append(step_row)

            if done:
                break

        success = bool(info.get("extras", {}).get("task.done", done))
        final_fraction = final_fraction_from_info(info)
        final_curve = float("nan")
        try:
            final_state = state_from_live_info(info)
            final_curve = curve_metric_from_state(final_state, n_beads)
        except Exception:
            pass

        episode.update({
            "status": "ok" if not episode.get("failure_reason") else "worker_exception",
            "success": int(success),
            "final_fraction": final_fraction,
            "delta_fraction": final_fraction - float(episode["initial_fraction"]) if math.isfinite(final_fraction) and math.isfinite(float(episode["initial_fraction"])) else float("nan"),
            "final_curve": final_curve,
            "num_steps": len(action_hist),
            "mean_step_fraction_delta": float(np.nanmean(step_deltas)) if step_deltas else float("nan"),
            "num_positive_fraction_steps": int(sum(1 for x in step_deltas if math.isfinite(x) and x > 1e-9)),
            "mean_model_x_ood_l2": float(np.nanmean(model_ood_l2s)) if model_ood_l2s else float("nan"),
            "mean_input_nn_l2": float(np.nanmean(input_nns)) if input_nns else float("nan"),
            "mean_future_std": float(np.nanmean(future_stds)) if future_stds else float("nan"),
            "mean_future_norm": float(np.nanmean(future_norms)) if future_norms else float("nan"),
            "mean_future_nn_l2": float(np.nanmean(future_nns)) if future_nns else float("nan"),
            "mean_future_nn_mae": float(np.nanmean(future_maes)) if future_maes else float("nan"),
            "future_condition_match_rate": float(np.nanmean(future_condition_match)) if future_condition_match else float("nan"),
            "dominant_nearest_future_condition": mode_string(future_nearest_conditions),
            "mean_chosen_action_ood": float(np.nanmean(chosen_oods)) if chosen_oods else float("nan"),
            "max_chosen_action_ood": float(np.nanmax(chosen_oods)) if chosen_oods else float("nan"),
            "mean_chosen_clip_frac": float(np.nanmean(chosen_clip)) if chosen_clip else float("nan"),
            "mean_chosen_pull_len": float(np.nanmean(chosen_pull)) if chosen_pull else float("nan"),
            "mean_old_pull_len": float(np.nanmean(old_pull)) if old_pull else float("nan"),
            "mean_default_pull_len": float(np.nanmean(default_pull)) if default_pull else float("nan"),
            "mean_best_pull_len": float(np.nanmean(best_pull)) if best_pull else float("nan"),
            "mean_old_action_ood": float(np.nanmean(old_ood)) if old_ood else float("nan"),
            "mean_default_action_ood": float(np.nanmean(default_ood)) if default_ood else float("nan"),
            "mean_best_action_ood": float(np.nanmean(best_ood)) if best_ood else float("nan"),
        })

    except Exception as exc:
        episode["status"] = "worker_exception"
        episode["failure_reason"] = repr(exc)
    finally:
        if env is not None:
            try:
                p34.close_env_safely(env)
            except Exception:
                try:
                    env.stop()
                except Exception:
                    pass

    out = {"episode": episode, "steps": steps}
    Path(args.worker_out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.worker_out_json).write_text(json.dumps(out, indent=2, sort_keys=True))
    print(json.dumps({
        "status": episode.get("status"),
        "policy": policy,
        "condition": condition,
        "final_fraction": episode.get("final_fraction"),
        "future_match": episode.get("future_condition_match_rate"),
        "failure": episode.get("failure_reason"),
    }, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--windows", default="data/phase3_state_diff_windows/phase3_windows.npz")
    parser.add_argument("--action_template", default="data/phase3_state_diff_windows/phase3_action_template.pkl")
    parser.add_argument("--old_checkpoint_root", default="checkpoints/phase3")
    parser.add_argument("--phase39_train_summary", default="reports/phase3_9_geometry_idm_train_summary.json")
    parser.add_argument("--phase39b_raw", default="reports/phase3_9b_ablation_raw_summary.json")
    parser.add_argument("--best_ablation", default="xy_only_high_weight")
    parser.add_argument("--policies", nargs="+", default=POLICIES)
    parser.add_argument("--conditions", nargs="+", default=AUDIT_CONDITIONS)
    parser.add_argument("--episodes_per_condition", type=int, default=4)
    parser.add_argument("--seed_start", type=int, default=310000)
    parser.add_argument("--max_steps", type=int, default=16)
    parser.add_argument("--samples_per_step", type=int, default=16)
    parser.add_argument("--motion_timeout", type=float, default=15.0)
    parser.add_argument("--action_clip_std", type=float, default=3.0)
    parser.add_argument("--row_timeout_sec", type=float, default=900.0)
    parser.add_argument("--total_timeout_sec", type=float, default=14400.0)
    parser.add_argument("--max_rows", type=int, default=36)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--out_episode_csv", default="reports/phase3_10b_future_rollout_audit_episodes.csv")
    parser.add_argument("--out_step_csv", default="reports/phase3_10b_future_rollout_audit_steps.csv")
    parser.add_argument("--out_json", default="reports/phase3_10b_future_rollout_audit_raw_summary.json")
    parser.add_argument("--progress_json", default="reports/phase3_10b_future_rollout_audit_progress.json")
    parser.add_argument("--worker_dir", default="reports/phase3_10b_workers")
    parser.add_argument("--worker_json", default="")
    parser.add_argument("--worker_out_json", default="")
    args = parser.parse_args()

    if args.worker_json:
        run_worker(args)
    else:
        run_parent(args)


if __name__ == "__main__":
    main()