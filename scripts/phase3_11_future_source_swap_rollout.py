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

CONDITIONS = ["free", "hidden_breakaway_pin", "hidden_high_friction"]
FUTURE_SOURCES = [
    "ddpm_mean",
    "ddpm_best_of_k_by_train_nn",
    "global_input_retrieval",
    "condition_matched_retrieval",
]
IDM_POLICIES = ["phase39b_xy_only_high_weight"]
FORBIDDEN_PREFIXES = ["tensorflow", "ravens.agents", "ravens.models", "ravens.datasets"]

EPISODE_FIELDS = [
    "status", "started_at", "finished_at", "duration_sec",
    "worker_returncode", "worker_timeout", "worker_stderr_tail",
    "idm_policy", "future_source", "future_source_uses_condition_label",
    "condition", "visible_seed", "episode_idx",
    "success", "initial_fraction", "final_fraction", "delta_fraction",
    "final_curve", "num_steps", "failure_reason",
    "mean_step_fraction_delta", "num_positive_fraction_steps",
    "mean_future_match", "mean_future_nn_l2", "mean_future_nn_mae",
    "dominant_future_nearest_condition",
    "mean_input_nn_l2", "mean_model_x_ood_l2",
    "mean_action_ood", "max_action_ood", "mean_clip_frac",
    "mean_pull_len", "mean_future_std", "mean_future_norm",
    "checkpoint_state_model", "checkpoint_inverse_dynamics",
    "primary_hidden_condition", "diagnostic_hidden_condition", "scope",
]

STEP_FIELDS = [
    "idm_policy", "future_source", "future_source_uses_condition_label",
    "condition", "visible_seed", "episode_idx", "step_idx",
    "fraction_before", "fraction_after", "fraction_delta",
    "done_after", "reward",
    "model_x_ood_l2", "model_x_ood_mean_abs", "input_nn_l2",
    "future_std_mean", "future_norm", "future_nn_l2", "future_nn_mae",
    "future_nearest_condition", "future_condition_match",
    "chosen_sample_idx", "retrieval_index",
    "action_ood", "clip_frac",
    "pose0_x", "pose0_y", "pose1_x", "pose1_y",
    "pull_len", "pose0_step_delta", "pose1_step_delta",
    "scope",
]


def assert_gate() -> None:
    if os.environ.get("PHASE3_ALLOW_FUTURE_SOURCE_SWAP", "0") != "1":
        raise SystemExit("[Phase3.11][BLOCKED] Set PHASE3_ALLOW_FUTURE_SOURCE_SWAP=1")
    if os.environ.get("PHASE3_FUTURE_SOURCE_SWAP_CONFIRMED", "0") != "1":
        raise SystemExit("[Phase3.11][BLOCKED] Set PHASE3_FUTURE_SOURCE_SWAP_CONFIRMED=1")


def assert_no_forbidden(stage: str) -> None:
    bad: List[str] = []
    for name in sys.modules:
        for prefix in FORBIDDEN_PREFIXES:
            if name == prefix or name.startswith(prefix + "."):
                bad.append(name)
    if bad:
        raise RuntimeError(f"[Phase3.11][FAIL] forbidden modules loaded at {stage}: {bad[:30]}")


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


def build_idm_table(args: argparse.Namespace) -> Dict[str, str]:
    root = Path(args.root).resolve()
    best = checkpoint_from_phase39b_raw(root, args.phase39b_raw, args.best_ablation)
    default = checkpoint_from_phase39_train(root, args.phase39_train_summary)
    old = first_old_checkpoint(root, args.old_checkpoint_root) / "inverse_dynamics.pt"
    return {
        "old_state_action": str(old),
        "phase39_default_geometry": str(default),
        f"phase39b_{args.best_ablation}": str(best),
    }


def build_specs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    root = Path(args.root).resolve()
    old_ckpt = first_old_checkpoint(root, args.old_checkpoint_root)
    state_model_path = old_ckpt / "state_model.pt"
    idm_table = build_idm_table(args)

    missing_idm = [p for p in args.idm_policies if p not in idm_table]
    if missing_idm:
        raise RuntimeError(f"unknown idm_policies={missing_idm}; available={sorted(idm_table)}")

    missing_sources = [s for s in args.future_sources if s not in FUTURE_SOURCES]
    if missing_sources:
        raise RuntimeError(f"unknown future_sources={missing_sources}; available={FUTURE_SOURCES}")

    specs: List[Dict[str, Any]] = []
    for idm_policy in args.idm_policies:
        for future_source in args.future_sources:
            for condition in args.conditions:
                for ep in range(int(args.episodes_per_condition)):
                    seed = int(args.seed_start) + ep
                    specs.append({
                        "root": str(root),
                        "idm_policy": idm_policy,
                        "future_source": future_source,
                        "future_source_uses_condition_label": int(future_source == "condition_matched_retrieval"),
                        "condition": condition,
                        "visible_seed": seed,
                        "episode_idx": ep,
                        "state_model_path": str(state_model_path),
                        "idm_path": idm_table[idm_policy],
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
        "idm_policy": spec.get("idm_policy", ""),
        "future_source": spec.get("future_source", ""),
        "future_source_uses_condition_label": spec.get("future_source_uses_condition_label", ""),
        "condition": spec.get("condition", ""),
        "visible_seed": spec.get("visible_seed", ""),
        "episode_idx": spec.get("episode_idx", ""),
        "failure_reason": status,
        "checkpoint_state_model": spec.get("state_model_path", ""),
        "checkpoint_inverse_dynamics": spec.get("idm_path", ""),
        "primary_hidden_condition": "hidden_breakaway_pin",
        "diagnostic_hidden_condition": "hidden_pin",
        "scope": "phase3_11_future_source_swap_no_phase4_no_cps",
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
        "scope": "phase3_11_future_source_swap_no_phase4_no_cps",
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
            "idm_policy": ep.get("idm_policy", ""),
            "future_source": ep.get("future_source", ""),
            "condition": ep.get("condition", ""),
            "final_fraction": ep.get("final_fraction", ""),
            "future_match": ep.get("mean_future_match", ""),
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
            "idm_policies": args.idm_policies,
            "future_sources": args.future_sources,
            "conditions": args.conditions,
            "episodes_per_condition": args.episodes_per_condition,
            "max_steps": args.max_steps,
            "samples_per_step": args.samples_per_step,
        },
        "out_episode_csv": args.out_episode_csv,
        "out_step_csv": args.out_step_csv,
        "progress_json": args.progress_json,
        "worker_dir": args.worker_dir,
        "scope": "phase3_11_future_source_swap_no_phase4_no_cps",
    }
    write_json_atomic(raw_path, raw)
    print(json.dumps(raw, indent=2, sort_keys=True))


def nearest_future_stats(vec: np.ndarray, mat: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    if mat.size == 0:
        return {
            "future_nn_l2": float("nan"),
            "future_nn_mae": float("nan"),
            "future_nearest_condition": "",
            "future_nn_index": -1,
        }
    v = np.asarray(vec, dtype=np.float32).reshape(1, -1)
    m = np.asarray(mat, dtype=np.float32)
    diff = m - v
    l2 = np.sqrt(np.mean(diff * diff, axis=1))
    idx = int(np.argmin(l2))
    return {
        "future_nn_l2": float(l2[idx]),
        "future_nn_mae": float(np.mean(np.abs(diff[idx]))),
        "future_nearest_condition": str(labels[idx]) if len(labels) > idx else "",
        "future_nn_index": idx,
    }


def input_nn_index(x: np.ndarray, mat: np.ndarray) -> Tuple[int, float]:
    if mat.size == 0:
        return -1, float("nan")
    diff = np.asarray(mat, dtype=np.float32) - np.asarray(x, dtype=np.float32).reshape(1, -1)
    d = np.sqrt(np.mean(diff * diff, axis=1))
    idx = int(np.argmin(d))
    return idx, float(d[idx])


def model_ood(model: Any, x: np.ndarray) -> Tuple[float, float]:
    try:
        z = model.x_std.transform(x)
        return float(np.sqrt(np.mean(z * z))), float(np.mean(np.abs(z)))
    except Exception:
        return float("nan"), float("nan")


def select_future(
    future_source: str,
    condition: str,
    model_x: np.ndarray,
    samples: np.ndarray,
    future_bank: np.ndarray,
    future_labels: np.ndarray,
    x_bank: np.ndarray,
    y_bank: np.ndarray,
    x_labels: np.ndarray,
) -> Dict[str, Any]:
    retrieval_index = -1
    chosen_sample_idx = -1

    if future_source == "ddpm_mean":
        future = np.mean(samples, axis=0)
        chosen_sample_idx = -1

    elif future_source == "ddpm_best_of_k_by_train_nn":
        best_i = 0
        best_d = float("inf")
        for i, sample in enumerate(samples):
            st = nearest_future_stats(sample, future_bank, future_labels)
            d = float(st["future_nn_l2"])
            if d < best_d:
                best_d = d
                best_i = i
        future = samples[best_i]
        chosen_sample_idx = int(best_i)

    elif future_source == "global_input_retrieval":
        retrieval_index, _ = input_nn_index(model_x.reshape(-1), x_bank)
        if retrieval_index < 0:
            future = np.mean(samples, axis=0)
        else:
            future = y_bank[retrieval_index]

    elif future_source == "condition_matched_retrieval":
        mask = x_labels == condition
        if np.any(mask):
            cond_x = x_bank[mask]
            cond_y = y_bank[mask]
            idx_local, _ = input_nn_index(model_x.reshape(-1), cond_x)
            if idx_local >= 0:
                global_idxs = np.where(mask)[0]
                retrieval_index = int(global_idxs[idx_local])
                future = cond_y[idx_local]
            else:
                future = np.mean(samples, axis=0)
        else:
            future = np.mean(samples, axis=0)

    else:
        raise RuntimeError(f"unknown future_source={future_source}")

    st = nearest_future_stats(future, future_bank, future_labels)
    st.update({
        "future": np.asarray(future, dtype=np.float32).reshape(1, -1),
        "chosen_sample_idx": chosen_sample_idx,
        "retrieval_index": retrieval_index,
    })
    return st


def action_info(codec: Any, idm: Any, idm_x: np.ndarray, clip_std: float) -> Dict[str, Any]:
    raw = idm.predict(idm_x)[0].astype(np.float32)
    lo = idm.train_action_mean - clip_std * idm.train_action_std
    hi = idm.train_action_mean + clip_std * idm.train_action_std
    vec = np.clip(raw, lo, hi).astype(np.float32)
    clip_frac = float(np.mean(np.abs(raw - vec) > 1e-6))
    action = codec.decode(vec)

    out: Dict[str, Any] = {
        "vec": vec,
        "action": action,
        "action_ood": float(idm.ood_score(vec.reshape(1, -1))[0]),
        "clip_frac": clip_frac,
        "pose0_x": float("nan"),
        "pose0_y": float("nan"),
        "pose1_x": float("nan"),
        "pose1_y": float("nan"),
        "pull_len": float("nan"),
    }

    try:
        p0 = np.asarray(action["params"]["pose0"][0], dtype=np.float32)[:2]
        p1 = np.asarray(action["params"]["pose1"][0], dtype=np.float32)[:2]
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
        counts[str(x)] = counts.get(str(x), 0) + 1
    if not counts:
        return ""
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def nanmean(xs: List[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return float(np.mean(vals)) if vals else float("nan")


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
    idm = load_inverse_model(Path(spec["idm_path"]))
    assert_no_forbidden("after_model_load")

    y_all = np.asarray(data["y_state"], dtype=np.float32).reshape(len(data["y_state"]), -1)
    x_all = np.asarray(data["state_action_x"], dtype=np.float32)
    cond_all = np.asarray([str(x) for x in data["condition_name"]])
    split = np.asarray([str(x) for x in data["split_name"]]) if "split_name" in data.files else np.asarray(["train"] * len(cond_all))
    train_mask = split == "train"
    if not np.any(train_mask):
        train_mask = np.ones(len(cond_all), dtype=bool)

    future_bank = y_all[train_mask]
    future_labels = cond_all[train_mask]
    x_bank = x_all[train_mask]
    y_bank = y_all[train_mask]
    x_labels = cond_all[train_mask]

    idm_policy = str(spec["idm_policy"])
    future_source = str(spec["future_source"])
    uses_cond_label = int(spec.get("future_source_uses_condition_label", 0))
    condition = str(spec["condition"])
    visible_seed = int(spec["visible_seed"])
    episode_idx = int(spec["episode_idx"])

    os.environ["CCDA_HIDDEN_CONDITION"] = condition
    os.environ["CCDA_VISIBLE_SEED"] = str(visible_seed)
    os.environ["CCDA_PAIR_GROUP"] = f"phase3_11_{idm_policy}_{future_source}_{visible_seed}"

    task = tasks.names["hidden-contact-cable-line"]()
    task.mode = "train"

    env = None
    steps: List[Dict[str, Any]] = []
    episode: Dict[str, Any] = {
        "status": "ok",
        "idm_policy": idm_policy,
        "future_source": future_source,
        "future_source_uses_condition_label": uses_cond_label,
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
        "checkpoint_inverse_dynamics": spec["idm_path"],
        "primary_hidden_condition": "hidden_breakaway_pin",
        "diagnostic_hidden_condition": "hidden_pin",
        "scope": "phase3_11_future_source_swap_no_phase4_no_cps",
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

        step_deltas: List[float] = []
        future_matches: List[float] = []
        future_nns: List[float] = []
        future_maes: List[float] = []
        future_nearest_conditions: List[str] = []
        input_nns: List[float] = []
        model_oods: List[float] = []
        future_stds: List[float] = []
        future_norms: List[float] = []
        action_oods: List[float] = []
        clips: List[float] = []
        pulls: List[float] = []

        done = False

        for step_idx in range(int(spec["max_steps"])):
            # Observable model input only:
            # live state history + past action history.
            # Do not concatenate condition labels, hidden metadata, reward/success,
            # final_fraction, branch labels, or future labels into model_x/idm_x.
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
            model_oods.append(mx_l2)

            input_idx, input_nn = input_nn_index(model_x.reshape(-1), x_bank)
            input_nns.append(input_nn)

            samples = state_model.sample(
                model_x,
                n_samples=int(spec["samples_per_step"]),
                seed=visible_seed + step_idx,
            )[:, 0, :].astype(np.float32)

            future_std = float(np.mean(np.std(samples, axis=0)))
            future_stds.append(future_std)

            chosen = select_future(
                future_source=future_source,
                condition=condition,
                model_x=model_x,
                samples=samples,
                future_bank=future_bank,
                future_labels=future_labels,
                x_bank=x_bank,
                y_bank=y_bank,
                x_labels=x_labels,
            )

            future_vec = np.asarray(chosen["future"], dtype=np.float32).reshape(1, -1)
            future_norm = float(np.sqrt(np.mean(future_vec * future_vec)))
            future_norms.append(future_norm)

            nn_l2 = float(chosen["future_nn_l2"])
            nn_mae = float(chosen["future_nn_mae"])
            nn_cond = str(chosen["future_nearest_condition"])
            future_nns.append(nn_l2)
            future_maes.append(nn_mae)
            future_nearest_conditions.append(nn_cond)
            match = 1.0 if nn_cond == condition else 0.0
            future_matches.append(match)

            idm_x = np.concatenate([hist_states.reshape(1, -1), future_vec], axis=1)
            ai = action_info(codec, idm, idm_x, float(spec["action_clip_std"]))
            action = ai["action"]
            action_hist.append(ai["vec"].astype(np.float32))

            fraction_before = previous_fraction
            try:
                obs, reward, done, info = env.step(action)
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
                p0 = np.asarray([ai["pose0_x"], ai["pose0_y"]], dtype=np.float32)
                p1 = np.asarray([ai["pose1_x"], ai["pose1_y"]], dtype=np.float32)
                if prev_pose0 is not None:
                    p0_step_delta = float(np.linalg.norm(p0 - prev_pose0))
                if prev_pose1 is not None:
                    p1_step_delta = float(np.linalg.norm(p1 - prev_pose1))
                prev_pose0 = p0
                prev_pose1 = p1
            except Exception:
                pass

            action_oods.append(float(ai["action_ood"]))
            clips.append(float(ai["clip_frac"]))
            pulls.append(float(ai["pull_len"]))

            steps.append({
                "idm_policy": idm_policy,
                "future_source": future_source,
                "future_source_uses_condition_label": uses_cond_label,
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
                "future_nn_l2": nn_l2,
                "future_nn_mae": nn_mae,
                "future_nearest_condition": nn_cond,
                "future_condition_match": int(match),
                "chosen_sample_idx": int(chosen.get("chosen_sample_idx", -1)),
                "retrieval_index": int(chosen.get("retrieval_index", -1)),
                "action_ood": ai["action_ood"],
                "clip_frac": ai["clip_frac"],
                "pose0_x": ai["pose0_x"],
                "pose0_y": ai["pose0_y"],
                "pose1_x": ai["pose1_x"],
                "pose1_y": ai["pose1_y"],
                "pull_len": ai["pull_len"],
                "pose0_step_delta": p0_step_delta,
                "pose1_step_delta": p1_step_delta,
                "scope": "phase3_11_future_source_swap_no_phase4_no_cps",
            })

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
            "mean_step_fraction_delta": nanmean(step_deltas),
            "num_positive_fraction_steps": int(sum(1 for x in step_deltas if math.isfinite(x) and x > 1e-9)),
            "mean_future_match": nanmean(future_matches),
            "mean_future_nn_l2": nanmean(future_nns),
            "mean_future_nn_mae": nanmean(future_maes),
            "dominant_future_nearest_condition": mode_string(future_nearest_conditions),
            "mean_input_nn_l2": nanmean(input_nns),
            "mean_model_x_ood_l2": nanmean(model_oods),
            "mean_action_ood": nanmean(action_oods),
            "max_action_ood": float(np.nanmax(action_oods)) if action_oods else float("nan"),
            "mean_clip_frac": nanmean(clips),
            "mean_pull_len": nanmean(pulls),
            "mean_future_std": nanmean(future_stds),
            "mean_future_norm": nanmean(future_norms),
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
        "idm_policy": idm_policy,
        "future_source": future_source,
        "condition": condition,
        "final_fraction": episode.get("final_fraction"),
        "future_match": episode.get("mean_future_match"),
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
    parser.add_argument("--idm_policies", nargs="+", default=IDM_POLICIES)
    parser.add_argument("--future_sources", nargs="+", default=FUTURE_SOURCES)
    parser.add_argument("--conditions", nargs="+", default=CONDITIONS)
    parser.add_argument("--episodes_per_condition", type=int, default=3)
    parser.add_argument("--seed_start", type=int, default=311000)
    parser.add_argument("--max_steps", type=int, default=16)
    parser.add_argument("--samples_per_step", type=int, default=32)
    parser.add_argument("--motion_timeout", type=float, default=15.0)
    parser.add_argument("--action_clip_std", type=float, default=3.0)
    parser.add_argument("--row_timeout_sec", type=float, default=900.0)
    parser.add_argument("--total_timeout_sec", type=float, default=14400.0)
    parser.add_argument("--max_rows", type=int, default=36)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--out_episode_csv", default="reports/phase3_11_future_source_swap_episodes.csv")
    parser.add_argument("--out_step_csv", default="reports/phase3_11_future_source_swap_steps.csv")
    parser.add_argument("--out_json", default="reports/phase3_11_future_source_swap_raw_summary.json")
    parser.add_argument("--progress_json", default="reports/phase3_11_future_source_swap_progress.json")
    parser.add_argument("--worker_dir", default="reports/phase3_11_workers")
    parser.add_argument("--worker_json", default="")
    parser.add_argument("--worker_out_json", default="")
    args = parser.parse_args()

    if args.worker_json:
        run_worker(args)
    else:
        run_parent(args)


if __name__ == "__main__":
    main()
