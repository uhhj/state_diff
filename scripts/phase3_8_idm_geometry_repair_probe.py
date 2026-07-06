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
from typing import Any, Dict, List, Tuple

import numpy as np

REQUIRED_CONDITIONS = ["free", "hidden_pin", "hidden_high_friction", "hidden_breakaway_pin"]
PRIMARY = "hidden_breakaway_pin"
DIAGNOSTIC = "hidden_pin"
FORBIDDEN_PREFIXES = ["tensorflow", "ravens.agents", "ravens.models", "ravens.datasets"]


def assert_gate() -> None:
    if os.environ.get("PHASE3_ALLOW_IDM_GEOMETRY_REPAIR", "0") != "1":
        raise SystemExit("[Phase3.8][BLOCKED] Set PHASE3_ALLOW_IDM_GEOMETRY_REPAIR=1")
    if os.environ.get("PHASE3_IDM_GEOMETRY_REPAIR_CONFIRMED", "0") != "1":
        raise SystemExit("[Phase3.8][BLOCKED] Set PHASE3_IDM_GEOMETRY_REPAIR_CONFIRMED=1")


def assert_no_forbidden(stage: str) -> None:
    bad: List[str] = []
    for name in sys.modules:
        for prefix in FORBIDDEN_PREFIXES:
            if name == prefix or name.startswith(prefix + "."):
                bad.append(name)
    if bad:
        raise RuntimeError(f"[Phase3.8][FAIL] forbidden modules loaded at {stage}: {bad[:30]}")


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
    return json.loads(path.read_text()) if path.exists() else {}


def load_windows(root: Path, path: str) -> Tuple[Any, Dict[str, Any]]:
    p = Path(path)
    if not p.is_absolute():
        p = root / p
    data = np.load(p, allow_pickle=True)
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
    for c in candidates:
        p = c if c.is_absolute() else root / c
        if not p.exists():
            continue
        try:
            from ccda_phase3.data_io import load_action_codec_from_template
            return load_action_codec_from_template(p)
        except Exception:
            with p.open("rb") as f:
                return pickle.load(f)
    raise FileNotFoundError("action template not found")


def codec_paths(codec: Any) -> List[str]:
    summary = getattr(codec, "summary", None)
    if callable(summary):
        s = summary()
        paths = s.get("paths", [])
        return [str(x) for x in paths]
    paths = getattr(codec, "path_strings", getattr(codec, "paths", []))
    return [str(x) for x in paths]


def decode(codec: Any, vec: np.ndarray) -> Dict[str, Any]:
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
    raise FileNotFoundError(f"No checkpoint for {baseline}")


def load_models(root: Path, checkpoint_root: str, baseline: str) -> Tuple[Path, Any, Any]:
    from ccda_phase3.train_utils import load_future_model, load_inverse_model
    ckpt = first_checkpoint(root, checkpoint_root, baseline)
    return ckpt, load_future_model(ckpt / "state_model.pt"), load_inverse_model(ckpt / "inverse_dynamics.pt")


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
    raise RuntimeError(f"Unexpected future sample shape {arr.shape}")


def idm_predict(idm: Any, x: np.ndarray) -> np.ndarray:
    return as_np(idm.predict(np.asarray(x, dtype=np.float32))).astype(np.float32).reshape(-1)


def infer_future_key(data: Any, pred_dim: int) -> str:
    for key in ["y_state", "y_final_state", "y_future_state", "future_state"]:
        if key in data.files:
            arr = np.asarray(data[key])
            if arr.shape[0] > 0 and int(np.prod(arr.shape[1:])) == int(pred_dim):
                return key
    available = {k: list(np.asarray(data[k]).shape) for k in data.files if k.startswith("y")}
    raise RuntimeError(f"No future key matches pred_dim={pred_dim}; available={available}")


def idm_ood(idm: Any, action_vec: np.ndarray) -> float:
    try:
        out = idm.ood_score(np.asarray(action_vec, dtype=np.float32).reshape(1, -1))
        return float(as_np(out).reshape(-1)[0])
    except Exception:
        return float("nan")


def idm_clip(idm: Any, vec: np.ndarray, clip_std: float) -> Tuple[np.ndarray, float]:
    action = np.asarray(vec, dtype=np.float32).reshape(-1)
    mean = np.asarray(getattr(idm, "train_action_mean"), dtype=np.float32).reshape(-1)
    std = np.asarray(getattr(idm, "train_action_std"), dtype=np.float32).reshape(-1)
    lo = mean - clip_std * std
    hi = mean + clip_std * std
    clipped = np.clip(action, lo, hi).astype(np.float32)
    return clipped, float(np.mean(np.abs(clipped - action) > 1e-8))


def str_array(data: Any, key: str) -> np.ndarray:
    return np.asarray([str(x) for x in data[key]])


def int_array(data: Any, key: str) -> np.ndarray:
    return np.asarray(data[key]).astype(int)


def select_indices(data: Any, phase37_raw: Dict[str, Any], samples_per_condition: int) -> List[Dict[str, Any]]:
    selected = phase37_raw.get("selected_windows") or []
    out: List[Dict[str, Any]] = []
    if selected:
        grouped: Dict[str, List[Dict[str, Any]]] = {c: [] for c in REQUIRED_CONDITIONS}
        for item in selected:
            cond = str(item.get("condition", ""))
            grouped.setdefault(cond, []).append({
                "idx": int(item["idx"]),
                "condition": cond,
                "visible_seed": str(item["visible_seed"]),
                "window_t": int(item["window_t"]),
                "source_file": str(item["source_file"]),
            })
        for cond in REQUIRED_CONDITIONS:
            out.extend(grouped.get(cond, [])[:samples_per_condition])
        return out

    conds = str_array(data, "condition_name")
    seeds = str_array(data, "visible_seed")
    ts = int_array(data, "window_t")
    src = str_array(data, "source_file")
    for cond in REQUIRED_CONDITIONS:
        idxs = np.where(conds == cond)[0]
        for idx in idxs[:samples_per_condition]:
            out.append({
                "idx": int(idx),
                "condition": cond,
                "visible_seed": str(seeds[idx]),
                "window_t": int(ts[idx]),
                "source_file": str(src[idx]),
            })
    return out


def idx_for(paths: List[str], suffix: str) -> int:
    for i, p in enumerate(paths):
        if p.endswith(suffix) or p == suffix:
            return i
    raise KeyError(f"missing codec path suffix {suffix}; paths={paths}")


def make_path_indices(paths: List[str]) -> Dict[str, Any]:
    return {
        "p0x": idx_for(paths, "params/pose0/0/0"),
        "p0y": idx_for(paths, "params/pose0/0/1"),
        "p0z": idx_for(paths, "params/pose0/0/2"),
        "q0": [idx_for(paths, f"params/pose0/1/{i}") for i in range(4)],
        "p1x": idx_for(paths, "params/pose1/0/0"),
        "p1y": idx_for(paths, "params/pose1/0/1"),
        "p1z": idx_for(paths, "params/pose1/0/2"),
        "q1": [idx_for(paths, f"params/pose1/1/{i}") for i in range(4)],
    }


def normalize_quat_in_vec(vec: np.ndarray, idxs: Dict[str, Any]) -> np.ndarray:
    out = np.asarray(vec, dtype=np.float32).copy()
    for key in ["q0", "q1"]:
        qidx = idxs[key]
        q = out[qidx].astype(np.float64)
        n = float(np.linalg.norm(q))
        if n > 1e-9:
            out[qidx] = (q / n).astype(np.float32)
    return out


def blend_xy(idm: np.ndarray, gt: np.ndarray, idxs: Dict[str, Any], alpha: float) -> np.ndarray:
    out = idm.copy()
    for key in ["p0x", "p0y", "p1x", "p1y"]:
        out[idxs[key]] = (1.0 - alpha) * idm[idxs[key]] + alpha * gt[idxs[key]]
    return out


def repair_variants(idm_vec: np.ndarray, gt_vec: np.ndarray, idxs: Dict[str, Any]) -> Dict[str, np.ndarray]:
    idm = np.asarray(idm_vec, dtype=np.float32).reshape(-1)
    gt = np.asarray(gt_vec, dtype=np.float32).reshape(-1)

    variants: Dict[str, np.ndarray] = {}
    variants["gt_reference"] = gt.copy()
    variants["idm_original"] = idm.copy()

    q = idm.copy()
    q[idxs["q0"]] = gt[idxs["q0"]]
    q[idxs["q1"]] = gt[idxs["q1"]]
    variants["idm_gt_quat"] = q

    z = idm.copy()
    z[idxs["p0z"]] = gt[idxs["p0z"]]
    z[idxs["p1z"]] = gt[idxs["p1z"]]
    variants["idm_gt_z"] = z

    qz = q.copy()
    qz[idxs["p0z"]] = gt[idxs["p0z"]]
    qz[idxs["p1z"]] = gt[idxs["p1z"]]
    variants["idm_gt_quat_z"] = qz

    p0 = idm.copy()
    p0[idxs["p0x"]] = gt[idxs["p0x"]]
    p0[idxs["p0y"]] = gt[idxs["p0y"]]
    variants["idm_gt_pose0_xy"] = p0

    p1 = idm.copy()
    p1[idxs["p1x"]] = gt[idxs["p1x"]]
    p1[idxs["p1y"]] = gt[idxs["p1y"]]
    variants["idm_gt_pose1_xy"] = p1

    both = idm.copy()
    for key in ["p0x", "p0y", "p1x", "p1y"]:
        both[idxs[key]] = gt[idxs[key]]
    variants["idm_gt_pose0_pose1_xy"] = both

    for alpha in [0.25, 0.50, 0.75]:
        variants[f"idm_xy_blend_{alpha:.2f}"] = blend_xy(idm, gt, idxs, alpha)

    full = both.copy()
    full[idxs["p0z"]] = gt[idxs["p0z"]]
    full[idxs["p1z"]] = gt[idxs["p1z"]]
    full[idxs["q0"]] = gt[idxs["q0"]]
    full[idxs["q1"]] = gt[idxs["q1"]]
    variants["idm_gt_xy_z_quat"] = full

    # Pull direction variants.
    idm_p0 = np.array([idm[idxs["p0x"]], idm[idxs["p0y"]]], dtype=np.float64)
    idm_p1 = np.array([idm[idxs["p1x"]], idm[idxs["p1y"]]], dtype=np.float64)
    gt_p0 = np.array([gt[idxs["p0x"]], gt[idxs["p0y"]]], dtype=np.float64)
    gt_p1 = np.array([gt[idxs["p1x"]], gt[idxs["p1y"]]], dtype=np.float64)
    idm_pull = idm_p1 - idm_p0
    gt_pull = gt_p1 - gt_p0
    idm_len = float(np.linalg.norm(idm_pull))
    gt_len = float(np.linalg.norm(gt_pull))
    if gt_len > 1e-9:
        gt_dir = gt_pull / gt_len

        d1 = idm.copy()
        new_p1 = idm_p0 + gt_dir * idm_len
        d1[idxs["p1x"]] = float(new_p1[0])
        d1[idxs["p1y"]] = float(new_p1[1])
        variants["idm_gt_pull_dir_keep_idm_len"] = d1

        d2 = idm.copy()
        new_p1 = idm_p0 + gt_dir * gt_len
        d2[idxs["p1x"]] = float(new_p1[0])
        d2[idxs["p1y"]] = float(new_p1[1])
        variants["idm_gt_pull_dir_gt_len"] = d2

        d3 = idm.copy()
        d3[idxs["p0x"]] = float(gt_p0[0])
        d3[idxs["p0y"]] = float(gt_p0[1])
        new_p1 = gt_p0 + gt_dir * idm_len
        d3[idxs["p1x"]] = float(new_p1[0])
        d3[idxs["p1y"]] = float(new_p1[1])
        variants["idm_gt_pick_gt_dir_keep_idm_len"] = d3

    return {k: normalize_quat_in_vec(v, idxs) for k, v in variants.items()}


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
    pred_pull = np.array([safe_float(pred.get("pose1_x")) - safe_float(pred.get("pose0_x")),
                          safe_float(pred.get("pose1_y")) - safe_float(pred.get("pose0_y"))])
    gt_pull = np.array([safe_float(gt.get("pose1_x")) - safe_float(gt.get("pose0_x")),
                        safe_float(gt.get("pose1_y")) - safe_float(gt.get("pose0_y"))])
    denom = float(np.linalg.norm(pred_pull) * np.linalg.norm(gt_pull))
    if denom > 1e-12:
        c = float(np.clip(np.dot(pred_pull, gt_pull) / denom, -1.0, 1.0))
        out["pull_angle_deg_to_gt"] = float(np.degrees(np.arccos(c)))
    else:
        out["pull_angle_deg_to_gt"] = float("nan")
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


def import_phase36(root: Path) -> Any:
    scripts = root / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    return importlib.import_module("phase3_6_matched_action_replay_diag")


def execute_after_matched_prefix(
    root: Path,
    p36: Any,
    tasks: Any,
    Environment: Any,
    close_helper: Any,
    data: Any,
    codec: Any,
    item: Dict[str, Any],
    action: Dict[str, Any],
    timeout: float,
    max_prefix_actions: int,
) -> Dict[str, Any]:
    env = None
    out: Dict[str, Any] = {
        "exec_success": 0,
        "exec_done": 0,
        "exec_reward": float("nan"),
        "exec_final_fraction_before": float("nan"),
        "exec_final_fraction_after": float("nan"),
        "exec_delta_final_fraction": float("nan"),
        "exec_prefix_state_mae": float("nan"),
        "exec_prefix_state_l2": float("nan"),
        "exec_prefix_source": "",
        "exec_failure_reason": "",
    }
    try:
        prefix_actions, prefix_source, prefix_detail = p36.get_prefix_actions_from_raw(
            root, codec, item["source_file"], int(item["window_t"])
        )
        out["exec_prefix_source"] = str(prefix_source)
        if prefix_actions is None:
            out["exec_failure_reason"] = "no_raw_prefix_actions"
            return out

        env, task, obs, info = p36.reset_env(
            tasks,
            Environment,
            item["condition"],
            item["visible_seed"],
            f"phase3_8_window_{item['idx']}",
            timeout,
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


def build_idm_vectors(
    data: Any,
    idx: int,
    baseline: str,
    state_model: Any,
    idm: Any,
    pred_samples: int,
    seed: int,
    clip_std: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str, float]:
    model_x = np.asarray(data["paper_x"][idx] if baseline == "paper_state" else data["state_action_x"][idx], dtype=np.float32).reshape(1, -1)
    samples = call_future_sample(state_model, model_x, pred_samples, seed)
    pred_future = np.mean(samples, axis=0).reshape(-1)
    future_key = infer_future_key(data, pred_future.shape[0])
    gt_future = np.asarray(data[future_key][idx], dtype=np.float32).reshape(-1)
    hist_states = np.asarray(data["paper_x"][idx], dtype=np.float32).reshape(1, -1)

    idm_x_gt = np.concatenate([hist_states, gt_future.reshape(1, -1)], axis=1)
    idm_x_pred = np.concatenate([hist_states, pred_future.reshape(1, -1)], axis=1)

    idm_gt_raw = idm_predict(idm, idm_x_gt)
    idm_pred_raw = idm_predict(idm, idm_x_pred)

    idm_gt, _ = idm_clip(idm, idm_gt_raw, clip_std)
    idm_pred, _ = idm_clip(idm, idm_pred_raw, clip_std)
    gt = np.asarray(data["y_action"][idx], dtype=np.float32).reshape(-1)
    future_mae = float(np.mean(np.abs(pred_future - gt_future)))
    return gt, idm_gt, idm_pred, future_key, future_mae


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--windows", default="data/phase3_state_diff_windows/phase3_windows.npz")
    parser.add_argument("--action_template", default="data/phase3_state_diff_windows/phase3_action_template.pkl")
    parser.add_argument("--checkpoint_root", default="checkpoints/phase3")
    parser.add_argument("--phase37_raw", default="reports/phase3_7_learned_action_alignment_raw_summary.json")
    parser.add_argument("--baselines", nargs="+", default=["paper_state", "state_action"])
    parser.add_argument("--sources", nargs="+", default=["idm_gt_future", "idm_pred_future"])
    parser.add_argument("--samples_per_condition", type=int, default=3)
    parser.add_argument("--pred_samples", type=int, default=16)
    parser.add_argument("--seed_base", type=int, default=380000)
    parser.add_argument("--action_clip_std", type=float, default=3.0)
    parser.add_argument("--motion_timeout", type=float, default=15.0)
    parser.add_argument("--max_prefix_actions", type=int, default=20)
    parser.add_argument("--out_csv", default="reports/phase3_8_idm_geometry_repair_trials.csv")
    parser.add_argument("--out_json", default="reports/phase3_8_idm_geometry_repair_raw_summary.json")
    args = parser.parse_args()

    assert_gate()
    root = Path(args.root).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    data, meta = load_windows(root, args.windows)
    if meta.get("conditions") != REQUIRED_CONDITIONS:
        raise SystemExit(f"[Phase3.8][FAIL] windows conditions mismatch: {meta.get('conditions')}")
    if meta.get("primary_hidden_condition") != PRIMARY:
        raise SystemExit(f"[Phase3.8][FAIL] primary mismatch: {meta.get('primary_hidden_condition')}")

    codec = load_codec(root, data, args.action_template)
    paths = codec_paths(codec)
    idxs = make_path_indices(paths)
    phase37_raw = load_json(root / args.phase37_raw)
    selected = select_indices(data, phase37_raw, args.samples_per_condition)

    p36 = import_phase36(root)
    tasks, Environment, close_helper = p36.import_tf_free_ravens(root)
    assert_no_forbidden("after_phase36_runtime_import")

    rows: List[Dict[str, Any]] = []
    checkpoint_info: Dict[str, str] = {}

    for baseline in args.baselines:
        ckpt, state_model, idm = load_models(root, args.checkpoint_root, baseline)
        checkpoint_info[baseline] = str(ckpt)
        for item in selected:
            idx = int(item["idx"])
            gt_vec, idm_gt_vec, idm_pred_vec, future_key, future_mae = build_idm_vectors(
                data, idx, baseline, state_model, idm, args.pred_samples, args.seed_base + idx, args.action_clip_std
            )
            gt_action = decode(codec, gt_vec)
            gt_info = action_info(gt_action)

            source_vectors = {
                "idm_gt_future": idm_gt_vec,
                "idm_pred_future": idm_pred_vec,
            }

            for source_name in args.sources:
                base_vec = source_vectors[source_name]
                variants = repair_variants(base_vec, gt_vec, idxs)

                for variant_name, vec in variants.items():
                    action = decode(codec, vec)
                    ai = action_info(action)
                    row: Dict[str, Any] = {
                        "window_idx": idx,
                        "condition": item["condition"],
                        "visible_seed": item["visible_seed"],
                        "window_t": int(item["window_t"]),
                        "source_file": item["source_file"],
                        "baseline": baseline,
                        "source_action": source_name,
                        "repair_variant": variant_name,
                        "future_key": future_key,
                        "future_mae": future_mae,
                        "primary_hidden_condition": PRIMARY,
                        "diagnostic_hidden_condition": DIAGNOSTIC,
                        "primary_pair": f"free_vs_{PRIMARY}",
                        "diagnostic_pair": f"free_vs_{DIAGNOSTIC}",
                        "scope": "phase3_8_idm_geometry_repair_no_phase4_no_cps",
                    }
                    row.update(ai)
                    row.update(vector_compare(vec, gt_vec))
                    row.update(geometry_compare(ai, gt_info))
                    row.update(execute_after_matched_prefix(
                        root, p36, tasks, Environment, close_helper,
                        data, codec, item, action,
                        args.motion_timeout, args.max_prefix_actions,
                    ))
                    row["is_gt_blended_diagnostic"] = int(variant_name != "idm_original")
                    row["is_deployable_policy_action"] = int(variant_name == "idm_original")
                    rows.append(row)
                    assert_no_forbidden(f"after_{baseline}_{source_name}_{variant_name}_{idx}")

    out_csv = root / args.out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "num_rows": len(rows),
        "selected_windows": selected,
        "baselines": args.baselines,
        "sources": args.sources,
        "checkpoint_info": checkpoint_info,
        "pred_samples": args.pred_samples,
        "motion_timeout": args.motion_timeout,
        "max_prefix_actions": args.max_prefix_actions,
        "out_csv": args.out_csv,
        "scope": "phase3_8_idm_geometry_repair_no_phase4_no_cps",
        "note": "GT-blended variants are diagnostic upper bounds, not deployable policy actions.",
    }
    (root / args.out_json).write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
