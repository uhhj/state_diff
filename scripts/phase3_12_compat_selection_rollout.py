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

SELECTORS = [
    "ddpm_mean",
    "input_nearest",
    "condition_nearest",
    "compat_global_topk",
    "compat_condition_topk",
    "compat_condition_action_geom",
]
CONDITIONS = ["free", "hidden_breakaway_pin", "hidden_high_friction"]
FORBIDDEN_PREFIXES = ["tensorflow", "ravens.agents", "ravens.models", "ravens.datasets"]

EPISODE_FIELDS = [
    "status", "started_at", "finished_at", "duration_sec", "worker_returncode", "worker_timeout",
    "worker_stderr_tail", "selector", "uses_condition_label", "condition", "visible_seed", "episode_idx",
    "success", "initial_fraction", "final_fraction", "delta_fraction", "num_steps", "failure_reason",
    "mean_step_delta", "num_positive_steps", "mean_future_match", "mean_future_nn_l2", "mean_input_nn_l2",
    "mean_selected_context_l2", "mean_selected_score", "mean_action_ood", "max_action_ood",
    "mean_clip_frac", "mean_pull_len", "mean_source_pull_len", "mean_action_mae_to_source",
    "mean_pull_diff_to_source", "mean_future_std", "scope",
]

STEP_FIELDS = [
    "selector", "uses_condition_label", "condition", "visible_seed", "episode_idx", "step_idx",
    "fraction_before", "fraction_after", "fraction_delta", "done_after", "reward",
    "model_x_ood_l2", "model_x_ood_abs", "future_std_mean", "future_nn_l2", "future_nn_mae",
    "future_nearest_condition", "future_condition_match", "input_nn_l2", "selected_train_index",
    "selected_window_idx", "selected_condition", "selected_context_l2", "selected_score",
    "score_context", "score_action_ood", "score_clip", "score_action_mae", "score_pull_diff",
    "action_ood", "clip_frac", "pose0_x", "pose0_y", "pose1_x", "pose1_y", "pull_len",
    "source_pull_len", "action_mae_to_source", "action_l2_to_source", "scope",
]


def assert_gate() -> None:
    if os.environ.get("PHASE3_ALLOW_COMPAT_FUTURE_SELECTION", "0") != "1":
        raise SystemExit("[Phase3.12][BLOCKED] Set PHASE3_ALLOW_COMPAT_FUTURE_SELECTION=1")
    if os.environ.get("PHASE3_COMPAT_FUTURE_SELECTION_CONFIRMED", "0") != "1":
        raise SystemExit("[Phase3.12][BLOCKED] Set PHASE3_COMPAT_FUTURE_SELECTION_CONFIRMED=1")


def assert_no_forbidden(stage: str) -> None:
    bad = []
    for name in sys.modules:
        for prefix in FORBIDDEN_PREFIXES:
            if name == prefix or name.startswith(prefix + "."):
                bad.append(name)
    if bad:
        raise RuntimeError(f"[Phase3.12][FAIL] forbidden modules loaded at {stage}: {bad[:30]}")


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(path)


def write_csv_row(path: Path, row: Dict[str, Any], fields: List[str], write_header: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fields})
        f.flush()
        os.fsync(f.fileno())


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


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


def build_specs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    root = Path(args.root).resolve()
    old_ckpt = first_old_checkpoint(root, args.old_checkpoint_root)
    state_model_path = old_ckpt / "state_model.pt"
    idm_path = checkpoint_from_phase39b_raw(root, args.phase39b_raw, args.best_ablation)

    specs: List[Dict[str, Any]] = []
    for selector in args.selectors:
        if selector not in SELECTORS:
            raise RuntimeError(f"unknown selector={selector}")
        for cond in args.conditions:
            for ep in range(int(args.episodes_per_condition)):
                specs.append({
                    "root": str(root),
                    "selector": selector,
                    "condition": cond,
                    "visible_seed": int(args.seed_start) + ep,
                    "episode_idx": ep,
                    "state_model_path": str(state_model_path),
                    "idm_path": str(idm_path),
                    "windows": args.windows,
                    "action_template": args.action_template,
                    "max_steps": args.max_steps,
                    "samples_per_step": args.samples_per_step,
                    "top_k": args.top_k,
                    "motion_timeout": args.motion_timeout,
                    "action_clip_std": args.action_clip_std,
                    "score_context_weight": args.score_context_weight,
                    "score_ood_weight": args.score_ood_weight,
                    "score_clip_weight": args.score_clip_weight,
                    "score_action_mae_weight": args.score_action_mae_weight,
                    "score_pull_diff_weight": args.score_pull_diff_weight,
                    "score_small_pull_weight": args.score_small_pull_weight,
                    "min_pull": args.min_pull,
                })
    if args.max_rows > 0:
        specs = specs[: args.max_rows]
    return specs


def error_episode(spec: Dict[str, Any], started: float, finished: float, status: str, stderr: str, rc: Any = "") -> Dict[str, Any]:
    row = {k: "" for k in EPISODE_FIELDS}
    row.update({
        "status": status,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(started)),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(finished)),
        "duration_sec": f"{finished - started:.3f}",
        "worker_returncode": rc,
        "worker_timeout": 1 if "timeout" in status else 0,
        "worker_stderr_tail": stderr[-2000:],
        "selector": spec.get("selector", ""),
        "uses_condition_label": int(str(spec.get("selector", "")).startswith("condition") or "condition" in str(spec.get("selector", ""))),
        "condition": spec.get("condition", ""),
        "visible_seed": spec.get("visible_seed", ""),
        "episode_idx": spec.get("episode_idx", ""),
        "failure_reason": status,
        "scope": "phase3_12_compat_selection_no_phase4_no_cps",
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
        "scope": "phase3_12_compat_selection_no_phase4_no_cps",
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
            "selector": ep.get("selector", ""),
            "condition": ep.get("condition", ""),
            "final_fraction": ep.get("final_fraction", ""),
            "delta_fraction": ep.get("delta_fraction", ""),
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
            "selectors": args.selectors,
            "conditions": args.conditions,
            "episodes_per_condition": args.episodes_per_condition,
            "max_steps": args.max_steps,
            "samples_per_step": args.samples_per_step,
            "top_k": args.top_k,
        },
        "out_episode_csv": args.out_episode_csv,
        "out_step_csv": args.out_step_csv,
        "scope": "phase3_12_compat_selection_no_phase4_no_cps",
    }
    write_json_atomic(raw_path, raw)
    print(json.dumps(raw, indent=2, sort_keys=True))


def nanmean(xs: List[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return float(np.mean(vals)) if vals else float("nan")


def action_pose_info(action: Dict[str, Any]) -> Dict[str, float]:
    out = {
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


def action_predict(codec: Any, idm: Any, hist_states: np.ndarray, future: np.ndarray, clip_std: float) -> Dict[str, Any]:
    idm_x = np.concatenate([hist_states.reshape(1, -1), future.reshape(1, -1)], axis=1)
    raw = idm.predict(idm_x)[0].astype(np.float32)
    lo = idm.train_action_mean - clip_std * idm.train_action_std
    hi = idm.train_action_mean + clip_std * idm.train_action_std
    vec = np.clip(raw, lo, hi).astype(np.float32)
    action = codec.decode(vec)
    info = action_pose_info(action)
    info.update({
        "vec": vec,
        "raw_vec": raw,
        "action": action,
        "action_ood": float(idm.ood_score(vec.reshape(1, -1))[0]),
        "clip_frac": float(np.mean(np.abs(raw - vec) > 1e-6)),
    })
    return info


def future_nn_stats(future: np.ndarray, y_bank: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    f = future.reshape(1, -1).astype(np.float32)
    diff = y_bank - f
    d = np.sqrt(np.mean(diff * diff, axis=1))
    idx = int(np.argmin(d))
    return {
        "future_nn_l2": float(d[idx]),
        "future_nn_mae": float(np.mean(np.abs(diff[idx]))),
        "future_nearest_condition": str(labels[idx]),
    }


def source_action_pull(codec: Any, y_action: np.ndarray) -> float:
    try:
        act = codec.decode(y_action.astype(np.float32))
        info = action_pose_info(act)
        return float(info["pull_len"])
    except Exception:
        return float("nan")


def topk_indices_by_l2(model_x: np.ndarray, x_bank: np.ndarray, mask: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    idxs = np.where(mask)[0]
    if idxs.size == 0:
        idxs = np.arange(len(x_bank))
    diff = x_bank[idxs] - model_x.reshape(1, -1)
    d = np.sqrt(np.mean(diff * diff, axis=1))
    order = np.argsort(d)[: max(1, min(int(k), len(d)))]
    return idxs[order], d[order]


def score_candidate(
    codec: Any,
    idm: Any,
    hist_states: np.ndarray,
    candidate_future: np.ndarray,
    candidate_y_action: np.ndarray,
    context_l2: float,
    clip_std: float,
    weights: Dict[str, float],
) -> Dict[str, Any]:
    pred = action_predict(codec, idm, hist_states, candidate_future.reshape(1, -1), clip_std)
    vec = pred["vec"]
    source_vec = candidate_y_action.reshape(-1).astype(np.float32)
    diff = vec.reshape(-1) - source_vec
    action_mae = float(np.mean(np.abs(diff)))
    action_l2 = float(np.sqrt(np.mean(diff * diff)))
    source_pull = source_action_pull(codec, source_vec)
    pull_len = float(pred["pull_len"])
    pull_diff = abs(pull_len - source_pull) if math.isfinite(pull_len) and math.isfinite(source_pull) else 1.0
    small_pull = max(0.0, float(weights["min_pull"]) - pull_len) if math.isfinite(pull_len) else 1.0

    score = (
        float(weights["context"]) * float(context_l2)
        + float(weights["ood"]) * float(pred["action_ood"])
        + float(weights["clip"]) * float(pred["clip_frac"])
        + float(weights["action_mae"]) * action_mae
        + float(weights["pull_diff"]) * pull_diff
        + float(weights["small_pull"]) * small_pull
    )

    return {
        "score": float(score),
        "score_context": float(context_l2),
        "score_action_ood": float(pred["action_ood"]),
        "score_clip": float(pred["clip_frac"]),
        "score_action_mae": action_mae,
        "score_pull_diff": pull_diff,
        "source_pull_len": source_pull,
        "action_l2_to_source": action_l2,
        "action_mae_to_source": action_mae,
        "pred": pred,
    }


def select_future(
    selector: str,
    condition: str,
    model_x: np.ndarray,
    hist_states: np.ndarray,
    samples: np.ndarray,
    bank: Dict[str, Any],
    codec: Any,
    idm: Any,
    spec: Dict[str, Any],
) -> Dict[str, Any]:
    x_bank = bank["x"]
    y_bank = bank["y"]
    ya_bank = bank["ya"]
    labels = bank["labels"]
    orig_idx = bank["orig_idx"]

    uses_condition = int(selector in {"condition_nearest", "compat_condition_topk", "compat_condition_action_geom"})
    selected_train_index = -1
    selected_window_idx = -1
    selected_condition = ""
    selected_context_l2 = float("nan")
    selected_score = float("nan")
    score_parts = {
        "score_context": float("nan"),
        "score_action_ood": float("nan"),
        "score_clip": float("nan"),
        "score_action_mae": float("nan"),
        "score_pull_diff": float("nan"),
    }
    source_pull_len = float("nan")
    action_mae_to_source = float("nan")
    action_l2_to_source = float("nan")

    if selector == "ddpm_mean":
        future = np.mean(samples, axis=0).astype(np.float32)
        source_info = None

    elif selector in {"input_nearest", "condition_nearest"}:
        mask = np.ones(len(labels), dtype=bool)
        if selector == "condition_nearest":
            mask = labels == condition
        idxs, dists = topk_indices_by_l2(model_x, x_bank, mask, 1)
        selected_train_index = int(idxs[0])
        selected_window_idx = int(orig_idx[selected_train_index])
        selected_condition = str(labels[selected_train_index])
        selected_context_l2 = float(dists[0])
        future = y_bank[selected_train_index].astype(np.float32)
        source_info = score_candidate(
            codec,
            idm,
            hist_states,
            future,
            ya_bank[selected_train_index],
            selected_context_l2,
            float(spec["action_clip_std"]),
            {
                "context": 0.0,
                "ood": 0.0,
                "clip": 0.0,
                "action_mae": 0.0,
                "pull_diff": 0.0,
                "small_pull": 0.0,
                "min_pull": float(spec["min_pull"]),
            },
        )

    elif selector in {"compat_global_topk", "compat_condition_topk", "compat_condition_action_geom"}:
        mask = np.ones(len(labels), dtype=bool)
        if selector in {"compat_condition_topk", "compat_condition_action_geom"}:
            mask = labels == condition

        idxs, dists = topk_indices_by_l2(model_x, x_bank, mask, int(spec["top_k"]))

        if selector == "compat_condition_action_geom":
            weights = {
                "context": float(spec["score_context_weight"]),
                "ood": float(spec["score_ood_weight"]) * 0.5,
                "clip": float(spec["score_clip_weight"]),
                "action_mae": float(spec["score_action_mae_weight"]) * 2.0,
                "pull_diff": float(spec["score_pull_diff_weight"]) * 2.0,
                "small_pull": float(spec["score_small_pull_weight"]),
                "min_pull": float(spec["min_pull"]),
            }
        else:
            weights = {
                "context": float(spec["score_context_weight"]),
                "ood": float(spec["score_ood_weight"]),
                "clip": float(spec["score_clip_weight"]),
                "action_mae": float(spec["score_action_mae_weight"]),
                "pull_diff": float(spec["score_pull_diff_weight"]),
                "small_pull": float(spec["score_small_pull_weight"]),
                "min_pull": float(spec["min_pull"]),
            }

        best = None
        best_i = None
        for ii, d in zip(idxs, dists):
            cand = score_candidate(
                codec,
                idm,
                hist_states,
                y_bank[ii],
                ya_bank[ii],
                float(d),
                float(spec["action_clip_std"]),
                weights,
            )
            if best is None or cand["score"] < best["score"]:
                best = cand
                best_i = int(ii)

        assert best is not None and best_i is not None
        selected_train_index = best_i
        selected_window_idx = int(orig_idx[selected_train_index])
        selected_condition = str(labels[selected_train_index])
        selected_context_l2 = float(best["score_context"])
        selected_score = float(best["score"])
        source_pull_len = float(best["source_pull_len"])
        action_mae_to_source = float(best["action_mae_to_source"])
        action_l2_to_source = float(best["action_l2_to_source"])
        score_parts = {k: float(best[k]) for k in score_parts.keys()}
        future = y_bank[selected_train_index].astype(np.float32)
        source_info = best

    else:
        raise RuntimeError(f"unknown selector={selector}")

    return {
        "future": future.reshape(1, -1),
        "uses_condition_label": uses_condition,
        "selected_train_index": selected_train_index,
        "selected_window_idx": selected_window_idx,
        "selected_condition": selected_condition,
        "selected_context_l2": selected_context_l2,
        "selected_score": selected_score,
        "source_pull_len": source_pull_len,
        "action_mae_to_source": action_mae_to_source,
        "action_l2_to_source": action_l2_to_source,
        **score_parts,
    }


def build_bank(data: Any) -> Dict[str, Any]:
    cond_all = np.asarray([str(x) for x in data["condition_name"]])
    split = np.asarray([str(x) for x in data["split_name"]]) if "split_name" in data.files else np.asarray(["train"] * len(cond_all))
    mask = split == "train"
    if not np.any(mask):
        mask = np.ones(len(cond_all), dtype=bool)
    return {
        "x": np.asarray(data["state_action_x"], dtype=np.float32)[mask],
        "y": np.asarray(data["y_state"], dtype=np.float32).reshape(len(cond_all), -1)[mask],
        "ya": np.asarray(data["y_action"], dtype=np.float32)[mask],
        "labels": cond_all[mask],
        "orig_idx": np.where(mask)[0],
    }


def run_worker(args: argparse.Namespace) -> None:
    assert_gate()
    spec = json.loads(Path(args.worker_json).read_text())
    root = Path(spec["root"]).resolve()

    for p in [root, root / "scripts", root / "external/deformable-ravens"]:
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

    from ccda_phase3.data_io import load_action_codec_from_template
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
        raise RuntimeError(f"bad action codec summary={codec.summary()}")

    th = int(data["th"])
    action_dim = int(data["action_dim"])
    n_beads = int(data["n_beads"])
    if action_dim != 14:
        raise RuntimeError(f"action_dim must be 14, got {action_dim}")

    bank = build_bank(data)
    state_model = load_future_model(Path(spec["state_model_path"]))
    idm = load_inverse_model(Path(spec["idm_path"]))
    assert_no_forbidden("after_model_load")

    selector = str(spec["selector"])
    condition = str(spec["condition"])
    visible_seed = int(spec["visible_seed"])
    episode_idx = int(spec["episode_idx"])
    uses_condition = int(selector in {"condition_nearest", "compat_condition_topk", "compat_condition_action_geom"})

    os.environ["CCDA_HIDDEN_CONDITION"] = condition
    os.environ["CCDA_VISIBLE_SEED"] = str(visible_seed)
    os.environ["CCDA_PAIR_GROUP"] = f"phase3_12_{selector}_{condition}_{visible_seed}"

    task = tasks.names["hidden-contact-cable-line"]()
    task.mode = "train"

    episode: Dict[str, Any] = {
        "status": "ok",
        "selector": selector,
        "uses_condition_label": uses_condition,
        "condition": condition,
        "visible_seed": visible_seed,
        "episode_idx": episode_idx,
        "success": 0,
        "initial_fraction": float("nan"),
        "final_fraction": float("nan"),
        "delta_fraction": float("nan"),
        "num_steps": 0,
        "failure_reason": "",
        "scope": "phase3_12_compat_selection_no_phase4_no_cps",
    }
    steps: List[Dict[str, Any]] = []
    env = None

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
        before0 = final_fraction_from_info(info)
        episode["initial_fraction"] = before0

        step_deltas: List[float] = []
        future_matches: List[float] = []
        future_nns: List[float] = []
        input_nns: List[float] = []
        selected_contexts: List[float] = []
        selected_scores: List[float] = []
        action_oods: List[float] = []
        clip_fracs: List[float] = []
        pulls: List[float] = []
        source_pulls: List[float] = []
        action_maes: List[float] = []
        pull_diffs: List[float] = []
        future_stds: List[float] = []

        done = False
        previous_fraction = before0

        for step_idx in range(int(spec["max_steps"])):
            # Observable model input only:
            # live state history + past action history.
            # condition/y_state/y_action are used only inside diagnostic selectors.
            state = state_from_live_info(info, prev_xy=prev_xy)
            prev_xy = state[: n_beads * 2].reshape(n_beads, 2)
            state_hist.append(state)

            hist_states = pad_history(state_hist, th).reshape(-1)
            hist_actions = pad_history(action_hist, th).reshape(-1) if action_hist else np.zeros((th * action_dim,), dtype=np.float32)
            model_x = np.concatenate([hist_states, hist_actions], axis=0).reshape(1, -1)

            try:
                z = state_model.x_std.transform(model_x)
                model_ood_l2 = float(np.sqrt(np.mean(z * z)))
                model_ood_abs = float(np.mean(np.abs(z)))
            except Exception:
                model_ood_l2 = float("nan")
                model_ood_abs = float("nan")

            samples = state_model.sample(
                model_x,
                n_samples=int(spec["samples_per_step"]),
                seed=visible_seed + step_idx,
            )[:, 0, :].astype(np.float32)
            future_std = float(np.mean(np.std(samples, axis=0)))
            future_stds.append(future_std)

            selected = select_future(selector, condition, model_x, hist_states, samples, bank, codec, idm, spec)
            future = selected["future"]
            fut_nn = future_nn_stats(future, bank["y"], bank["labels"])
            future_match = 1 if str(fut_nn["future_nearest_condition"]) == condition else 0

            pred = action_predict(codec, idm, hist_states, future, float(spec["action_clip_std"]))
            action = pred["action"]
            action_hist.append(pred["vec"].astype(np.float32))

            before = previous_fraction
            try:
                obs, reward, done, info = env.step(action)
                after = final_fraction_from_info(info)
                success = bool(info.get("extras", {}).get("task.done", done))
            except Exception as exc:
                reward = float("nan")
                done = False
                after = before
                success = False
                episode["failure_reason"] = repr(exc)
                break

            delta = after - before if math.isfinite(after) and math.isfinite(before) else float("nan")
            previous_fraction = after

            step_deltas.append(delta)
            future_matches.append(float(future_match))
            future_nns.append(float(fut_nn["future_nn_l2"]))
            input_nns.append(float(selected["selected_context_l2"]) if math.isfinite(float(selected["selected_context_l2"])) else float("nan"))
            selected_contexts.append(float(selected["selected_context_l2"]) if math.isfinite(float(selected["selected_context_l2"])) else float("nan"))
            selected_scores.append(float(selected["selected_score"]) if math.isfinite(float(selected["selected_score"])) else float("nan"))
            action_oods.append(float(pred["action_ood"]))
            clip_fracs.append(float(pred["clip_frac"]))
            pulls.append(float(pred["pull_len"]))
            source_pulls.append(float(selected["source_pull_len"]) if math.isfinite(float(selected["source_pull_len"])) else float("nan"))
            action_maes.append(float(selected["action_mae_to_source"]) if math.isfinite(float(selected["action_mae_to_source"])) else float("nan"))
            pull_diff = abs(float(pred["pull_len"]) - float(selected["source_pull_len"])) if math.isfinite(float(selected["source_pull_len"])) and math.isfinite(float(pred["pull_len"])) else float("nan")
            pull_diffs.append(pull_diff)

            steps.append({
                "selector": selector,
                "uses_condition_label": uses_condition,
                "condition": condition,
                "visible_seed": visible_seed,
                "episode_idx": episode_idx,
                "step_idx": step_idx,
                "fraction_before": before,
                "fraction_after": after,
                "fraction_delta": delta,
                "done_after": int(bool(done)),
                "reward": reward,
                "model_x_ood_l2": model_ood_l2,
                "model_x_ood_abs": model_ood_abs,
                "future_std_mean": future_std,
                "future_nn_l2": fut_nn["future_nn_l2"],
                "future_nn_mae": fut_nn["future_nn_mae"],
                "future_nearest_condition": fut_nn["future_nearest_condition"],
                "future_condition_match": future_match,
                "input_nn_l2": selected["selected_context_l2"],
                "selected_train_index": selected["selected_train_index"],
                "selected_window_idx": selected["selected_window_idx"],
                "selected_condition": selected["selected_condition"],
                "selected_context_l2": selected["selected_context_l2"],
                "selected_score": selected["selected_score"],
                "score_context": selected["score_context"],
                "score_action_ood": selected["score_action_ood"],
                "score_clip": selected["score_clip"],
                "score_action_mae": selected["score_action_mae"],
                "score_pull_diff": selected["score_pull_diff"],
                "action_ood": pred["action_ood"],
                "clip_frac": pred["clip_frac"],
                "pose0_x": pred["pose0_x"],
                "pose0_y": pred["pose0_y"],
                "pose1_x": pred["pose1_x"],
                "pose1_y": pred["pose1_y"],
                "pull_len": pred["pull_len"],
                "source_pull_len": selected["source_pull_len"],
                "action_mae_to_source": selected["action_mae_to_source"],
                "action_l2_to_source": selected["action_l2_to_source"],
                "scope": "phase3_12_compat_selection_no_phase4_no_cps",
            })

            if done:
                break

        final_fraction = final_fraction_from_info(info)
        episode.update({
            "status": "ok" if not episode.get("failure_reason") else "worker_exception",
            "success": int(bool(success)),
            "final_fraction": final_fraction,
            "delta_fraction": final_fraction - before0 if math.isfinite(final_fraction) and math.isfinite(before0) else float("nan"),
            "num_steps": len(action_hist),
            "mean_step_delta": nanmean(step_deltas),
            "num_positive_steps": int(sum(1 for x in step_deltas if math.isfinite(x) and x > 1e-9)),
            "mean_future_match": nanmean(future_matches),
            "mean_future_nn_l2": nanmean(future_nns),
            "mean_input_nn_l2": nanmean(input_nns),
            "mean_selected_context_l2": nanmean(selected_contexts),
            "mean_selected_score": nanmean(selected_scores),
            "mean_action_ood": nanmean(action_oods),
            "max_action_ood": float(np.nanmax(action_oods)) if action_oods else float("nan"),
            "mean_clip_frac": nanmean(clip_fracs),
            "mean_pull_len": nanmean(pulls),
            "mean_source_pull_len": nanmean(source_pulls),
            "mean_action_mae_to_source": nanmean(action_maes),
            "mean_pull_diff_to_source": nanmean(pull_diffs),
            "mean_future_std": nanmean(future_stds),
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

    Path(args.worker_out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.worker_out_json).write_text(json.dumps({"episode": episode, "steps": steps}, indent=2, sort_keys=True))
    print(json.dumps({
        "status": episode.get("status"),
        "selector": selector,
        "condition": condition,
        "final_fraction": episode.get("final_fraction"),
        "delta_fraction": episode.get("delta_fraction"),
        "failure": episode.get("failure_reason"),
    }, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--windows", default="data/phase3_state_diff_windows/phase3_windows.npz")
    parser.add_argument("--action_template", default="data/phase3_state_diff_windows/phase3_action_template.pkl")
    parser.add_argument("--old_checkpoint_root", default="checkpoints/phase3")
    parser.add_argument("--phase39b_raw", default="reports/phase3_9b_ablation_raw_summary.json")
    parser.add_argument("--best_ablation", default="xy_only_high_weight")
    parser.add_argument("--selectors", nargs="+", default=SELECTORS)
    parser.add_argument("--conditions", nargs="+", default=CONDITIONS)
    parser.add_argument("--episodes_per_condition", type=int, default=3)
    parser.add_argument("--seed_start", type=int, default=312000)
    parser.add_argument("--max_steps", type=int, default=16)
    parser.add_argument("--samples_per_step", type=int, default=32)
    parser.add_argument("--top_k", type=int, default=32)
    parser.add_argument("--motion_timeout", type=float, default=15.0)
    parser.add_argument("--action_clip_std", type=float, default=3.0)
    parser.add_argument("--score_context_weight", type=float, default=1.0)
    parser.add_argument("--score_ood_weight", type=float, default=0.15)
    parser.add_argument("--score_clip_weight", type=float, default=2.0)
    parser.add_argument("--score_action_mae_weight", type=float, default=4.0)
    parser.add_argument("--score_pull_diff_weight", type=float, default=2.0)
    parser.add_argument("--score_small_pull_weight", type=float, default=2.0)
    parser.add_argument("--min_pull", type=float, default=0.08)
    parser.add_argument("--row_timeout_sec", type=float, default=900.0)
    parser.add_argument("--total_timeout_sec", type=float, default=21600.0)
    parser.add_argument("--max_rows", type=int, default=54)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--out_episode_csv", default="reports/phase3_12_compat_selection_episodes.csv")
    parser.add_argument("--out_step_csv", default="reports/phase3_12_compat_selection_steps.csv")
    parser.add_argument("--out_json", default="reports/phase3_12_compat_selection_raw_summary.json")
    parser.add_argument("--progress_json", default="reports/phase3_12_compat_selection_progress.json")
    parser.add_argument("--worker_dir", default="reports/phase3_12_workers")
    parser.add_argument("--worker_json", default="")
    parser.add_argument("--worker_out_json", default="")
    args = parser.parse_args()

    if args.worker_json:
        run_worker(args)
    else:
        run_parent(args)


if __name__ == "__main__":
    main()
