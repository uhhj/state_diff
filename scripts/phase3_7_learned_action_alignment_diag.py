#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

ROOT_DEFAULT = "/data/state_diff2"
REQUIRED_CONDITIONS = ["free", "hidden_pin", "hidden_high_friction", "hidden_breakaway_pin"]
PRIMARY = "hidden_breakaway_pin"
DIAGNOSTIC = "hidden_pin"
FORBIDDEN_PREFIXES = ["tensorflow", "ravens.agents", "ravens.models", "ravens.datasets"]


def assert_gate() -> None:
    if os.environ.get("PHASE3_ALLOW_ACTION_ALIGNMENT", "0") != "1":
        raise SystemExit("[Phase3.7][BLOCKED] Set PHASE3_ALLOW_ACTION_ALIGNMENT=1")
    if os.environ.get("PHASE3_ACTION_ALIGNMENT_CONFIRMED", "0") != "1":
        raise SystemExit("[Phase3.7][BLOCKED] Set PHASE3_ACTION_ALIGNMENT_CONFIRMED=1")


def assert_no_forbidden(stage: str) -> None:
    bad: List[str] = []
    for name in sys.modules:
        for prefix in FORBIDDEN_PREFIXES:
            if name == prefix or name.startswith(prefix + "."):
                bad.append(name)
    if bad:
        raise RuntimeError(f"[Phase3.7][FAIL] forbidden modules loaded at {stage}: {bad[:30]}")


def as_np(x: Any) -> np.ndarray:
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if denom < 1e-12:
        return float("nan")
    return float(np.dot(aa, bb) / denom)


def angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    uu = np.asarray(u, dtype=np.float64).reshape(-1)
    vv = np.asarray(v, dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(uu) * np.linalg.norm(vv))
    if denom < 1e-12:
        return float("nan")
    c = float(np.clip(np.dot(uu, vv) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def load_windows(root: Path, windows: str) -> Tuple[Any, Dict[str, Any]]:
    path = Path(windows)
    if not path.is_absolute():
        path = root / path
    data = np.load(path, allow_pickle=True)
    meta: Dict[str, Any] = {}
    if "meta_json" in data.files:
        raw = data["meta_json"]
        try:
            text = str(raw.item() if getattr(raw, "shape", ()) == () else raw.reshape(-1)[0])
            meta = json.loads(text)
        except Exception:
            meta = {}
    return data, meta


def load_codec(root: Path, data: Any, fallback: str) -> Any:
    candidates: List[Path] = []
    if "action_template_json_or_pickle_path" in data.files:
        raw = data["action_template_json_or_pickle_path"]
        val = str(raw.item() if getattr(raw, "shape", ()) == () else raw.reshape(-1)[0])
        candidates.append(Path(val))
    candidates.append(Path(fallback))
    for cand in candidates:
        path = cand if cand.is_absolute() else root / cand
        if not path.exists():
            continue
        try:
            from ccda_phase3.data_io import load_action_codec_from_template

            return load_action_codec_from_template(path)
        except Exception:
            with path.open("rb") as f:
                return pickle.load(f)
    raise FileNotFoundError(f"action template not found from candidates={candidates}")


def codec_decode(codec: Any, vec: np.ndarray) -> Dict[str, Any]:
    return codec.decode(np.asarray(vec, dtype=np.float32).reshape(-1))


def first_checkpoint(root: Path, checkpoint_root: str, baseline: str) -> Path:
    ckpt_root = Path(checkpoint_root)
    if not ckpt_root.is_absolute():
        ckpt_root = root / ckpt_root
    base = ckpt_root / baseline
    cands = sorted([p for p in base.glob("fold_*_seed_*") if p.is_dir()])
    for cand in cands:
        if (cand / "state_model.pt").exists() and (cand / "inverse_dynamics.pt").exists():
            return cand
    raise FileNotFoundError(f"No complete checkpoint for {baseline} under {base}")


def load_models(root: Path, checkpoint_root: str, baseline: str) -> Tuple[Path, Any, Any]:
    from ccda_phase3.train_utils import load_future_model, load_inverse_model

    ckpt = first_checkpoint(root, checkpoint_root, baseline)
    state_model = load_future_model(ckpt / "state_model.pt")
    idm = load_inverse_model(ckpt / "inverse_dynamics.pt")
    return ckpt, state_model, idm


def call_future_sample(model: Any, model_x: np.ndarray, n_samples: int, seed: int) -> np.ndarray:
    x = np.asarray(model_x, dtype=np.float32)
    try:
        out = model.sample(x, n_samples=int(n_samples), seed=int(seed))
    except TypeError:
        try:
            out = model.sample(x, n_samples=int(n_samples))
        except TypeError:
            out = model.sample(x)
    arr = as_np(out).astype(np.float32)
    if arr.ndim == 3:
        return arr[0]
    if arr.ndim == 2:
        return arr
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    raise RuntimeError(f"Unexpected future sample shape: {arr.shape}")


def idm_predict(idm: Any, idm_x: np.ndarray) -> np.ndarray:
    pred = idm.predict(np.asarray(idm_x, dtype=np.float32))
    return as_np(pred).astype(np.float32).reshape(-1)


def idm_ood(idm: Any, action_vec: np.ndarray) -> float:
    try:
        out = idm.ood_score(np.asarray(action_vec, dtype=np.float32).reshape(1, -1))
        return float(as_np(out).reshape(-1)[0])
    except Exception:
        return float("nan")


def idm_clip_stats(idm: Any, action_vec: np.ndarray, clip_std: float) -> Tuple[np.ndarray, float]:
    vec = np.asarray(action_vec, dtype=np.float32).reshape(-1)
    mean = np.asarray(getattr(idm, "train_action_mean"), dtype=np.float32).reshape(-1)
    std = np.asarray(getattr(idm, "train_action_std"), dtype=np.float32).reshape(-1)
    lo = mean - float(clip_std) * std
    hi = mean + float(clip_std) * std
    clipped = np.clip(vec, lo, hi).astype(np.float32)
    clip_fraction = float(np.mean(np.abs(clipped - vec) > 1e-8))
    return clipped, clip_fraction


def flatten_future(data: Any, key: str, idx: int) -> np.ndarray:
    return np.asarray(data[key][idx], dtype=np.float32).reshape(-1)


def infer_future_key(data: Any, pred_dim: int) -> str:
    candidates = ["y_state", "y_final_state", "y_future_state", "future_state", "y"]
    matches: List[str] = []
    for key in candidates:
        if key in data.files:
            arr = np.asarray(data[key])
            if arr.shape[0] > 0 and int(np.prod(arr.shape[1:])) == int(pred_dim):
                matches.append(key)
    if matches:
        return matches[0]
    available = {k: list(np.asarray(data[k]).shape) for k in data.files if k.startswith("y")}
    raise RuntimeError(f"No GT future key matches pred_dim={pred_dim}. Available y* keys: {available}")


def str_array(data: Any, key: str) -> np.ndarray:
    return np.asarray([str(x) for x in data[key]])


def int_array(data: Any, key: str) -> np.ndarray:
    return np.asarray(data[key]).astype(int)


def select_indices(data: Any, phase36_raw: Dict[str, Any], samples_per_condition: int) -> List[Dict[str, Any]]:
    selected_from_p36 = phase36_raw.get("selected_windows") or []
    out: List[Dict[str, Any]] = []
    if selected_from_p36:
        seen = set()
        for item in selected_from_p36:
            idx = int(item["idx"])
            if idx in seen:
                continue
            seen.add(idx)
            out.append({
                "idx": idx,
                "condition": str(item.get("condition", str_array(data, "condition_name")[idx])),
                "visible_seed": str(item.get("visible_seed", str_array(data, "visible_seed")[idx])),
                "window_t": int(item.get("window_t", int_array(data, "window_t")[idx])),
                "source_file": str(item.get("source_value", str_array(data, "source_file")[idx])),
            })
    if out:
        grouped: Dict[str, List[Dict[str, Any]]] = {c: [] for c in REQUIRED_CONDITIONS}
        for item in out:
            grouped.setdefault(item["condition"], []).append(item)
        limited: List[Dict[str, Any]] = []
        for cond in REQUIRED_CONDITIONS:
            limited.extend(grouped.get(cond, [])[:samples_per_condition])
        return limited

    conds = str_array(data, "condition_name")
    seeds = str_array(data, "visible_seed")
    ts = int_array(data, "window_t")
    src = str_array(data, "source_file")
    for cond in REQUIRED_CONDITIONS:
        idxs = np.where(conds == cond)[0]
        for idx in idxs[:samples_per_condition]:
            out.append({
                "idx": int(idx),
                "condition": str(conds[idx]),
                "visible_seed": str(seeds[idx]),
                "window_t": int(ts[idx]),
                "source_file": str(src[idx]),
            })
    return out


def action_info(action: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
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
        "pull_xy_len": float("nan"),
        "action_valid": False,
        "action_repr": "",
    }
    try:
        out["action_repr"] = json.dumps(action, default=str)[:1200]
        if not isinstance(action, dict):
            return out
        out["primitive"] = str(action.get("primitive", ""))
        params = action.get("params", {})
        out["has_params"] = isinstance(params, dict)
        for key in ["pose0", "pose1"]:
            val = params.get(key) if isinstance(params, dict) else None
            if val is None:
                continue
            pos = np.asarray(val[0], dtype=float).reshape(-1)
            rot = np.asarray(val[1], dtype=float).reshape(-1)
            out[f"{key}_x"] = float(pos[0]) if len(pos) > 0 else float("nan")
            out[f"{key}_y"] = float(pos[1]) if len(pos) > 1 else float("nan")
            out[f"{key}_z"] = float(pos[2]) if len(pos) > 2 else float("nan")
            out[f"{key}_quat_norm"] = float(np.linalg.norm(rot)) if len(rot) else float("nan")
        if math.isfinite(out["pose0_x"]) and math.isfinite(out["pose1_x"]):
            pull = np.array([out["pose1_x"] - out["pose0_x"], out["pose1_y"] - out["pose0_y"]], dtype=float)
            out["pull_xy_len"] = float(np.linalg.norm(pull))
        out["action_valid"] = bool(out["primitive"] == "pick_place" and out["has_params"])
    except Exception as exc:
        out["action_repr"] = f"action_info_error:{repr(exc)}"
    return out


def geometry_compare(pred: Dict[str, Any], gt: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for pose in ["pose0", "pose1"]:
        p = np.array([safe_float(pred.get(f"{pose}_x")), safe_float(pred.get(f"{pose}_y"))], dtype=float)
        g = np.array([safe_float(gt.get(f"{pose}_x")), safe_float(gt.get(f"{pose}_y"))], dtype=float)
        out[f"{pose}_xy_dist_to_gt"] = float(np.linalg.norm(p - g)) if np.all(np.isfinite(p)) and np.all(np.isfinite(g)) else float("nan")
        out[f"{pose}_z_abs_diff_to_gt"] = abs(safe_float(pred.get(f"{pose}_z")) - safe_float(gt.get(f"{pose}_z")))
    pred_pull = np.array([safe_float(pred.get("pose1_x")) - safe_float(pred.get("pose0_x")), safe_float(pred.get("pose1_y")) - safe_float(pred.get("pose0_y"))])
    gt_pull = np.array([safe_float(gt.get("pose1_x")) - safe_float(gt.get("pose0_x")), safe_float(gt.get("pose1_y")) - safe_float(gt.get("pose0_y"))])
    out["pull_angle_deg_to_gt"] = angle_deg(pred_pull, gt_pull)
    pred_len = float(np.linalg.norm(pred_pull)) if np.all(np.isfinite(pred_pull)) else float("nan")
    gt_len = float(np.linalg.norm(gt_pull)) if np.all(np.isfinite(gt_pull)) else float("nan")
    out["pull_len_ratio_to_gt"] = pred_len / gt_len if math.isfinite(pred_len) and math.isfinite(gt_len) and gt_len > 1e-12 else float("nan")
    return out


def vector_compare(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    p = np.asarray(pred, dtype=np.float32).reshape(-1)
    g = np.asarray(gt, dtype=np.float32).reshape(-1)
    diff = p - g
    return {
        "action_l2_to_gt": float(np.linalg.norm(diff)),
        "action_mae_to_gt": float(np.mean(np.abs(diff))),
        "action_max_abs_to_gt": float(np.max(np.abs(diff))),
        "action_cosine_to_gt": cosine(p, g),
        "action_norm": float(np.linalg.norm(p)),
        "gt_action_norm": float(np.linalg.norm(g)),
    }


def try_import_phase36(root: Path) -> Any:
    scripts = root / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    return importlib.import_module("phase3_6_matched_action_replay_diag")


def execute_action_after_matched_prefix(
    root: Path,
    p36: Any,
    tasks: Any,
    Environment: Any,
    close_helper: Any,
    data: Any,
    codec: Any,
    item: Dict[str, Any],
    action: Dict[str, Any],
    motion_timeout: float,
    max_prefix_actions: int,
) -> Dict[str, Any]:
    env = None
    out: Dict[str, Any] = {
        "exec_attempted": True,
        "exec_success": 0,
        "exec_done": 0,
        "exec_reward": float("nan"),
        "exec_final_fraction_before": float("nan"),
        "exec_final_fraction_after": float("nan"),
        "exec_delta_final_fraction": float("nan"),
        "exec_prefix_state_mae": float("nan"),
        "exec_prefix_state_l2": float("nan"),
        "exec_prefix_source": "",
        "exec_prefix_detail": "",
        "exec_failure_reason": "",
    }
    try:
        prefix_actions, prefix_source, prefix_detail = p36.get_prefix_actions_from_raw(
            root, codec, item["source_file"], int(item["window_t"])
        )
        out["exec_prefix_source"] = str(prefix_source)
        out["exec_prefix_detail"] = str(prefix_detail)
        if prefix_actions is None:
            out["exec_failure_reason"] = "no_raw_prefix_actions"
            return out

        env, task, obs, info = p36.reset_env(
            tasks,
            Environment,
            item["condition"],
            item["visible_seed"],
            f"phase3_7_window_{item['idx']}",
            motion_timeout,
        )
        obs, info, prefix_rows, executed = p36.replay_prefix(
            env, task, obs, info, prefix_actions, max_prefix_actions
        )

        live_state = p36.live_state_from_info(info, None)
        if live_state is not None and "paper_x" in data.files:
            ref = p36.current_ref_state_from_paper_x(data, int(item["idx"]), len(live_state))
            if ref is not None:
                diff = np.asarray(live_state, dtype=np.float32) - np.asarray(ref, dtype=np.float32)
                out["exec_prefix_state_l2"] = float(np.linalg.norm(diff))
                out["exec_prefix_state_mae"] = float(np.mean(np.abs(diff)))

        obs2, reward, done, info2, success, before, after = p36.execute_one(env, task, action)
        out["exec_success"] = int(success)
        out["exec_done"] = int(bool(done))
        out["exec_reward"] = float(reward)
        out["exec_final_fraction_before"] = float(before)
        out["exec_final_fraction_after"] = float(after)
        out["exec_delta_final_fraction"] = float(after - before) if np.isfinite(after) and np.isfinite(before) else float("nan")
        return out
    except Exception as exc:
        out["exec_failure_reason"] = repr(exc)
        return out
    finally:
        if env is not None:
            try:
                p36.close_env(env, close_helper)
            except Exception:
                try:
                    env.stop()
                except Exception:
                    pass


def maybe_oracle_after_matched_prefix(
    root: Path,
    p36: Any,
    tasks: Any,
    Environment: Any,
    close_helper: Any,
    data: Any,
    codec: Any,
    item: Dict[str, Any],
    motion_timeout: float,
    max_prefix_actions: int,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    env = None
    exec_info: Dict[str, Any] = {
        "exec_attempted": True,
        "exec_success": 0,
        "exec_done": 0,
        "exec_reward": float("nan"),
        "exec_final_fraction_before": float("nan"),
        "exec_final_fraction_after": float("nan"),
        "exec_delta_final_fraction": float("nan"),
        "exec_prefix_state_mae": float("nan"),
        "exec_prefix_state_l2": float("nan"),
        "exec_prefix_source": "",
        "exec_prefix_detail": "",
        "exec_failure_reason": "",
    }
    try:
        prefix_actions, prefix_source, prefix_detail = p36.get_prefix_actions_from_raw(
            root, codec, item["source_file"], int(item["window_t"])
        )
        exec_info["exec_prefix_source"] = str(prefix_source)
        exec_info["exec_prefix_detail"] = str(prefix_detail)
        if prefix_actions is None:
            exec_info["exec_failure_reason"] = "no_raw_prefix_actions"
            return None, exec_info

        env, task, obs, info = p36.reset_env(
            tasks,
            Environment,
            item["condition"],
            item["visible_seed"],
            f"phase3_7_oracle_window_{item['idx']}",
            motion_timeout,
        )
        obs, info, prefix_rows, executed = p36.replay_prefix(
            env, task, obs, info, prefix_actions, max_prefix_actions
        )

        live_state = p36.live_state_from_info(info, None)
        if live_state is not None and "paper_x" in data.files:
            ref = p36.current_ref_state_from_paper_x(data, int(item["idx"]), len(live_state))
            if ref is not None:
                diff = np.asarray(live_state, dtype=np.float32) - np.asarray(ref, dtype=np.float32)
                exec_info["exec_prefix_state_l2"] = float(np.linalg.norm(diff))
                exec_info["exec_prefix_state_mae"] = float(np.mean(np.abs(diff)))

        oracle = task.oracle(env)
        action = oracle.act(obs, info)
        obs2, reward, done, info2, success, before, after = p36.execute_one(env, task, action)
        exec_info["exec_success"] = int(success)
        exec_info["exec_done"] = int(bool(done))
        exec_info["exec_reward"] = float(reward)
        exec_info["exec_final_fraction_before"] = float(before)
        exec_info["exec_final_fraction_after"] = float(after)
        exec_info["exec_delta_final_fraction"] = float(after - before) if np.isfinite(after) and np.isfinite(before) else float("nan")
        return action, exec_info
    except Exception as exc:
        exec_info["exec_failure_reason"] = repr(exc)
        return None, exec_info
    finally:
        if env is not None:
            try:
                p36.close_env(env, close_helper)
            except Exception:
                try:
                    env.stop()
                except Exception:
                    pass


def base_row(item: Dict[str, Any], baseline: str, action_source: str, future_key: str) -> Dict[str, Any]:
    return {
        "window_idx": int(item["idx"]),
        "condition": str(item["condition"]),
        "visible_seed": str(item["visible_seed"]),
        "window_t": int(item["window_t"]),
        "source_file": str(item["source_file"]),
        "baseline": baseline,
        "action_source": action_source,
        "future_key": future_key,
        "primary_hidden_condition": PRIMARY,
        "diagnostic_hidden_condition": DIAGNOSTIC,
        "primary_pair": f"free_vs_{PRIMARY}",
        "diagnostic_pair": f"free_vs_{DIAGNOSTIC}",
        "scope": "phase3_7_learned_action_alignment_no_phase4_no_cps",
    }


def build_rows_for_window(
    root: Path,
    data: Any,
    codec: Any,
    item: Dict[str, Any],
    baseline: str,
    state_model: Any,
    idm: Any,
    pred_samples: int,
    seed_base: int,
    action_clip_std: float,
    execute_actions: bool,
    p36: Any,
    tasks: Any,
    Environment: Any,
    close_helper: Any,
    motion_timeout: float,
    max_prefix_actions: int,
) -> List[Dict[str, Any]]:
    idx = int(item["idx"])
    gt_action_vec = np.asarray(data["y_action"][idx], dtype=np.float32).reshape(-1)
    gt_action = codec_decode(codec, gt_action_vec)
    gt_info = action_info(gt_action)

    model_x = np.asarray(data["paper_x"][idx] if baseline == "paper_state" else data["state_action_x"][idx], dtype=np.float32).reshape(1, -1)
    pred_samples_arr = call_future_sample(state_model, model_x, pred_samples, seed_base + idx)
    pred_future = np.mean(pred_samples_arr, axis=0).reshape(-1)
    future_key = infer_future_key(data, pred_future.shape[0])
    gt_future = flatten_future(data, future_key, idx)

    hist_states = np.asarray(data["paper_x"][idx], dtype=np.float32).reshape(1, -1)

    idm_x_gt = np.concatenate([hist_states, gt_future.reshape(1, -1)], axis=1)
    idm_gt_raw = idm_predict(idm, idm_x_gt)
    idm_gt_vec, idm_gt_clip_fraction = idm_clip_stats(idm, idm_gt_raw, action_clip_std)

    idm_x_pred = np.concatenate([hist_states, pred_future.reshape(1, -1)], axis=1)
    idm_pred_raw = idm_predict(idm, idm_x_pred)
    idm_pred_vec, idm_pred_clip_fraction = idm_clip_stats(idm, idm_pred_raw, action_clip_std)

    future_error_l2 = float(np.linalg.norm(pred_future - gt_future))
    future_error_mae = float(np.mean(np.abs(pred_future - gt_future)))
    future_cos = cosine(pred_future, gt_future)

    rows: List[Dict[str, Any]] = []

    action_defs: List[Tuple[str, np.ndarray, Dict[str, Any], float, float]] = [
        ("gt_y_action", gt_action_vec, gt_action, float("nan"), 0.0),
        ("idm_gt_future", idm_gt_vec, codec_decode(codec, idm_gt_vec), idm_ood(idm, idm_gt_vec), idm_gt_clip_fraction),
        ("idm_pred_future", idm_pred_vec, codec_decode(codec, idm_pred_vec), idm_ood(idm, idm_pred_vec), idm_pred_clip_fraction),
    ]

    for source, vec, decoded, ood, clip_fraction in action_defs:
        row = base_row(item, baseline, source, future_key)
        ai = action_info(decoded)
        row.update(ai)
        row.update(vector_compare(vec, gt_action_vec))
        row.update(geometry_compare(ai, gt_info))
        row.update({
            "future_error_l2": future_error_l2,
            "future_error_mae": future_error_mae,
            "future_cosine_to_gt": future_cos,
            "action_ood_score": float(ood),
            "clip_fraction": float(clip_fraction),
            "raw_or_clipped_action_norm": float(np.linalg.norm(vec)),
            "gt_progress_reference_available": True,
        })
        if execute_actions:
            row.update(execute_action_after_matched_prefix(
                root, p36, tasks, Environment, close_helper, data, codec, item, decoded,
                motion_timeout, max_prefix_actions
            ))
        else:
            row.update({
                "exec_attempted": False,
                "exec_success": 0,
                "exec_done": 0,
                "exec_reward": float("nan"),
                "exec_final_fraction_before": float("nan"),
                "exec_final_fraction_after": float("nan"),
                "exec_delta_final_fraction": float("nan"),
                "exec_prefix_state_mae": float("nan"),
                "exec_prefix_state_l2": float("nan"),
                "exec_prefix_source": "",
                "exec_prefix_detail": "",
                "exec_failure_reason": "",
            })
        rows.append(row)

    if execute_actions:
        oracle_action, oracle_exec = maybe_oracle_after_matched_prefix(
            root, p36, tasks, Environment, close_helper, data, codec, item,
            motion_timeout, max_prefix_actions
        )
        if oracle_action is not None:
            oracle_info = action_info(oracle_action)
        else:
            oracle_info = {"action_valid": False}
        row = base_row(item, baseline, "same_state_oracle", future_key)
        row.update(oracle_info)
        row.update({
            "action_l2_to_gt": float("nan"),
            "action_mae_to_gt": float("nan"),
            "action_max_abs_to_gt": float("nan"),
            "action_cosine_to_gt": float("nan"),
            "action_norm": float("nan"),
            "gt_action_norm": float(np.linalg.norm(gt_action_vec)),
            "pose0_xy_dist_to_gt": geometry_compare(oracle_info, gt_info).get("pose0_xy_dist_to_gt", float("nan")),
            "pose1_xy_dist_to_gt": geometry_compare(oracle_info, gt_info).get("pose1_xy_dist_to_gt", float("nan")),
            "pose0_z_abs_diff_to_gt": geometry_compare(oracle_info, gt_info).get("pose0_z_abs_diff_to_gt", float("nan")),
            "pose1_z_abs_diff_to_gt": geometry_compare(oracle_info, gt_info).get("pose1_z_abs_diff_to_gt", float("nan")),
            "pull_angle_deg_to_gt": geometry_compare(oracle_info, gt_info).get("pull_angle_deg_to_gt", float("nan")),
            "pull_len_ratio_to_gt": geometry_compare(oracle_info, gt_info).get("pull_len_ratio_to_gt", float("nan")),
            "future_error_l2": future_error_l2,
            "future_error_mae": future_error_mae,
            "future_cosine_to_gt": future_cos,
            "action_ood_score": float("nan"),
            "clip_fraction": float("nan"),
            "raw_or_clipped_action_norm": float("nan"),
            "gt_progress_reference_available": True,
        })
        row.update(oracle_exec)
        rows.append(row)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=ROOT_DEFAULT)
    parser.add_argument("--windows", default="data/phase3_state_diff_windows/phase3_windows.npz")
    parser.add_argument("--action_template", default="data/phase3_state_diff_windows/phase3_action_template.pkl")
    parser.add_argument("--checkpoint_root", default="checkpoints/phase3")
    parser.add_argument("--phase36_raw", default="reports/phase3_6_matched_action_replay_raw_summary.json")
    parser.add_argument("--baselines", nargs="+", default=["paper_state", "state_action"])
    parser.add_argument("--samples_per_condition", type=int, default=3)
    parser.add_argument("--pred_samples", type=int, default=16)
    parser.add_argument("--seed_base", type=int, default=370000)
    parser.add_argument("--action_clip_std", type=float, default=3.0)
    parser.add_argument("--execute_actions", action="store_true")
    parser.add_argument("--motion_timeout", type=float, default=15.0)
    parser.add_argument("--max_prefix_actions", type=int, default=20)
    parser.add_argument("--out_csv", default="reports/phase3_7_learned_action_alignment_trials.csv")
    parser.add_argument("--out_json", default="reports/phase3_7_learned_action_alignment_raw_summary.json")
    args = parser.parse_args()

    assert_gate()
    root = Path(args.root).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    data, meta = load_windows(root, args.windows)
    if meta.get("conditions") != REQUIRED_CONDITIONS:
        raise SystemExit(f"[Phase3.7][FAIL] windows conditions mismatch: {meta.get('conditions')}")
    if meta.get("primary_hidden_condition") != PRIMARY:
        raise SystemExit(f"[Phase3.7][FAIL] primary mismatch: {meta.get('primary_hidden_condition')}")
    if "y_action" not in data.files or int(data["y_action"].shape[1]) != 14:
        raise SystemExit(f"[Phase3.7][FAIL] y_action dim must be 14, got {data['y_action'].shape if 'y_action' in data.files else None}")

    codec = load_codec(root, data, args.action_template)
    phase36_raw = load_json(root / args.phase36_raw)
    selected = select_indices(data, phase36_raw, args.samples_per_condition)

    p36 = None
    tasks = None
    Environment = None
    close_helper = None
    if args.execute_actions:
        p36 = try_import_phase36(root)
        tasks, Environment, close_helper = p36.import_tf_free_ravens(root)
        assert_no_forbidden("after_phase36_runtime_import")

    rows: List[Dict[str, Any]] = []
    checkpoint_info: Dict[str, Any] = {}
    for baseline in args.baselines:
        ckpt, state_model, idm = load_models(root, args.checkpoint_root, baseline)
        checkpoint_info[baseline] = str(ckpt)
        assert_no_forbidden(f"after_load_models_{baseline}")
        for item in selected:
            rows.extend(build_rows_for_window(
                root=root,
                data=data,
                codec=codec,
                item=item,
                baseline=baseline,
                state_model=state_model,
                idm=idm,
                pred_samples=args.pred_samples,
                seed_base=args.seed_base,
                action_clip_std=args.action_clip_std,
                execute_actions=bool(args.execute_actions),
                p36=p36,
                tasks=tasks,
                Environment=Environment,
                close_helper=close_helper,
                motion_timeout=args.motion_timeout,
                max_prefix_actions=args.max_prefix_actions,
            ))
            assert_no_forbidden(f"after_window_{baseline}_{item['idx']}")

    out_csv = root / args.out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    payload = {
        "num_rows": len(rows),
        "selected_windows": selected,
        "baselines": args.baselines,
        "checkpoint_info": checkpoint_info,
        "pred_samples": args.pred_samples,
        "execute_actions": bool(args.execute_actions),
        "motion_timeout": args.motion_timeout,
        "max_prefix_actions": args.max_prefix_actions,
        "out_csv": args.out_csv,
        "scope": "phase3_7_learned_action_alignment_no_phase4_no_cps",
    }
    (root / args.out_json).write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
