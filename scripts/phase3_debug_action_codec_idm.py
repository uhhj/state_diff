#!/usr/bin/env python3
"""Debug Phase3 executable action codec and inverse dynamics targets."""

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ccda_phase3.action_codec import ExecutableActionCodec


def arr_stats(x: np.ndarray) -> Dict[str, Any]:
    x = np.asarray(x, dtype=np.float64)
    return {
        "shape": list(x.shape),
        "mean": float(np.mean(x)) if x.size else None,
        "std": float(np.std(x)) if x.size else None,
        "min": float(np.min(x)) if x.size else None,
        "max": float(np.max(x)) if x.size else None,
        "mean_abs": float(np.mean(np.abs(x))) if x.size else None,
        "p95_abs": float(np.percentile(np.abs(x), 95)) if x.size else None,
        "p99_abs": float(np.percentile(np.abs(x), 99)) if x.size else None,
        "num_nan": int(np.isnan(x).sum()) if x.size else 0,
        "num_inf": int(np.isinf(x).sum()) if x.size else 0,
    }


def add_issue(issues, level, name, detail):
    issues.append({"level": level, "name": name, "detail": detail})


def safe_float(x: Any):
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def scan_checkpoint_backends(ckpt_root: Path):
    out = []
    for cfg_path in sorted(Path(ckpt_root).glob("*/*/config.json")):
        cfg = load_json(cfg_path)
        out.append({
            "checkpoint": str(cfg_path.parent),
            "training_backend": cfg.get("training_backend", cfg.get("backend", "unknown")),
            "future_model_type": cfg.get("future_model_type", ""),
            "ddpm_used": bool(cfg.get("ddpm_used", False)),
            "idm_feature_mode": cfg.get("idm_feature_mode", ""),
        })
    return out


def load_action_template_summary(path: Path, data) -> Dict[str, Any]:
    summary = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return summary
    try:
        codec = ExecutableActionCodec.load(path)
        summary.update(codec.summary())
        paths = summary.get("paths", [])
        summary["camera_config_paths_head"] = [p for p in paths if "camera_config" in p][:20]
        summary["forbidden_token_paths"] = [p for p in paths if any(tok in p for tok in ["seed", "reward", "success", "hidden", "object", "info"])]
        summary["invalid_executable_paths"] = [
            p for p in paths
            if not (p.startswith("params/pose0") or p.startswith("params/pose1") or p.startswith("pose0") or p.startswith("pose1"))
        ]
        target = data["y_action"][0].astype(np.float32)
        summary["roundtrip_error"] = codec.roundtrip_error(codec.decode(target))
    except Exception as exc:
        summary["error"] = repr(exc)
    return summary


def action_history_diagnostics(paper_x, state_action_x, y_action, th, action_dim):
    extra = state_action_x[:, paper_x.shape[1]:]
    out = arr_stats(extra)
    expected_extra_dim = int(th) * int(action_dim)
    out["state_action_extra_dim"] = int(extra.shape[1]) if extra.ndim == 2 else None
    out["expected_extra_dim"] = expected_extra_dim
    out["action_history_target_exact_match_rate"] = None
    out["action_history_target_corr_max"] = None
    if extra.ndim == 2 and extra.shape[1] == expected_extra_dim:
        hist = extra.reshape(len(extra), int(th), int(action_dim))
        target = y_action.reshape(len(y_action), 1, int(action_dim))
        exact = np.all(np.isclose(hist, target, atol=1e-7), axis=2)
        out["action_history_target_exact_match_rate"] = float(np.mean(exact))
        corr_max = 0.0
        for d in range(int(action_dim)):
            a = hist[:, -1, d]
            b = y_action[:, d]
            if np.std(a) > 1e-8 and np.std(b) > 1e-8:
                corr = abs(float(np.corrcoef(a, b)[0, 1]))
                corr_max = max(corr_max, corr)
        out["action_history_target_corr_max"] = corr_max
    return out


def compute_idm_diagnostics(data, ckpt_root: Path):
    try:
        from ccda_phase3.train_utils import load_inverse_model
    except Exception as exc:
        return [{"error": f"could not import load_inverse_model: {exc!r}"}]
    split = data["split_name"].astype(str)
    held = np.where(split == "heldout")[0]
    if len(held) == 0:
        return [{"error": "no heldout rows"}]
    paper_x = data["paper_x"].astype(np.float32)
    y_flat = data["y_state"].astype(np.float32).reshape(len(split), -1)
    y_action = data["y_action"].astype(np.float32)
    x_all = np.concatenate([paper_x, y_flat], axis=1).astype(np.float32)
    out = []
    for cfg_path in sorted(Path(ckpt_root).glob("*/*/config.json")):
        try:
            meta = load_json(cfg_path)
            idm = load_inverse_model(cfg_path.parent / "inverse_dynamics.pt")
            pred = idm.predict(x_all[held]).astype(np.float32)
            target = y_action[held].astype(np.float32)
            raw_mse = float(np.mean((pred - target) ** 2))
            std = np.asarray(getattr(idm, "train_action_std", np.std(y_action, axis=0)), dtype=np.float32)
            std = np.where(std < 1e-6, 1.0, std)
            norm_mse = float(np.mean(((pred - target) / std) ** 2))
            ood = idm.ood_score(pred) if hasattr(idm, "ood_score") else np.sqrt(np.mean((pred / std) ** 2, axis=-1))
            out.append({
                "checkpoint": str(cfg_path.parent),
                "baseline": meta.get("baseline", cfg_path.parent.parent.name),
                "training_backend": meta.get("training_backend", meta.get("backend", "unknown")),
                "future_model_type": meta.get("future_model_type", ""),
                "ddpm_used": bool(meta.get("ddpm_used", False)),
                "idm_feature_mode": meta.get("idm_feature_mode", ""),
                "idm_x_dim": int(x_all.shape[1]),
                "raw_action_mse": raw_mse,
                "normalized_action_mse": norm_mse,
                "pred_action_ood_mean": float(np.mean(ood)),
                "pred_action_ood_max": float(np.max(ood)),
                "uses_standardized_target": bool(hasattr(idm, "y_std")),
                "heldout_rows": int(len(held)),
            })
        except Exception as exc:
            out.append({"checkpoint": str(cfg_path.parent), "error": repr(exc)})
    return out


def write_report(path: Path, payload: Dict[str, Any]) -> None:
    lines = []
    lines.append("# Phase3 Action Codec / Inverse Dynamics Debug Report")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append(f"- Verdict: `{payload['verdict']}`")
    lines.append("")
    lines.append("## Input Dimensions")
    lines.append("")
    for k, v in payload["dimension_summary"].items():
        lines.append(f"- `{k}`: `{v}`")
    lines.append("")
    lines.append("## Action Target Stats")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    for k, v in payload["action_stats"].items():
        lines.append(f"| `{k}` | `{v}` |")
    lines.append("")
    lines.append("## Action-History Leakage")
    lines.append("")
    for k, v in payload["state_action_extra_block"].items():
        lines.append(f"- `{k}`: `{v}`")
    lines.append("")
    lines.append("## Action Template")
    lines.append("")
    ats = payload.get("action_template_summary", {})
    for k in ["path", "class", "dim", "num_paths", "num_camera_config_paths", "num_param_paths", "roundtrip_error"]:
        lines.append(f"- `{k}`: `{ats.get(k)}`")
    if ats.get("invalid_executable_paths"):
        lines.append(f"- `invalid_executable_paths`: `{ats.get('invalid_executable_paths')}`")
    lines.append("")
    lines.append("## IDM Heldout Reconstruction")
    lines.append("")
    lines.append("| Checkpoint | Backend | Future model | DDPM | IDM features | Raw MSE | Normalized MSE | Pred OOD mean | Pred OOD max |")
    lines.append("|---|---|---|---|---|---:|---:|---:|---:|")
    for item in payload.get("idm_debug", []):
        ck = Path(item.get("checkpoint", "")).name if item.get("checkpoint") else "NA"
        label = f"{item.get('baseline', '')}/{ck}"
        lines.append(f"| `{label}` | `{item.get('training_backend', 'NA')}` | `{item.get('future_model_type', 'NA')}` | `{item.get('ddpm_used', 'NA')}` | `{item.get('idm_feature_mode', 'NA')}` | `{item.get('raw_action_mse')}` | `{item.get('normalized_action_mse')}` | `{item.get('pred_action_ood_mean')}` | `{item.get('pred_action_ood_max')}` |")
    lines.append("")
    lines.append("## Issues")
    lines.append("")
    lines.append("| Level | Name | Detail |")
    lines.append("|---|---|---|")
    if payload["issues"]:
        for issue in payload["issues"]:
            lines.append(f"| `{issue['level']}` | `{issue['name']}` | {issue['detail']} |")
    else:
        lines.append("| `PASS` | `none` | No action/IDM diagnostic issue detected. |")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- IDM diagnostics use `paper_x + future_state` full-state inputs, matching the formal Phase3 action pipeline.")
    lines.append("- If this report is FAIL, do not trust policy rollout or move to Phase4.")
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/data/state_diff2/data/phase3_state_diff_windows/phase3_windows.npz")
    parser.add_argument("--ckpt_root", default="/data/state_diff2/checkpoints/phase3")
    parser.add_argument("--out_json", default="/data/state_diff2/reports/phase3_action_idm_debug_summary.json")
    parser.add_argument("--out_md", default="/data/state_diff2/reports/phase3_action_idm_debug_report.md")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    data = np.load(args.data, allow_pickle=True)
    issues = []
    paper_x = data["paper_x"].astype(np.float32)
    state_action_x = data["state_action_x"].astype(np.float32)
    y_action = data["y_action"].astype(np.float32)
    th = int(data["th"])
    action_dim = int(data["action_dim"])
    episode_action_len = data["episode_action_len"].astype(int) if "episode_action_len" in data.files else np.asarray([], dtype=int)

    dimension_summary = {
        "paper_x_shape": list(paper_x.shape),
        "state_action_x_shape": list(state_action_x.shape),
        "y_action_shape": list(y_action.shape),
        "episode_action_len_max": int(np.max(episode_action_len)) if episode_action_len.size else None,
    }
    if y_action.ndim != 2 or y_action.shape[1] != 14:
        add_issue(issues, "FAIL", "y_action_dim_not_14", f"Expected executable y_action dim 14, got shape {list(y_action.shape)}.")

    action_stats = arr_stats(y_action)
    ya = y_action.astype(np.float64)
    zero_var_dims = np.where(np.std(ya, axis=0) < 1e-12)[0].astype(int).tolist() if ya.ndim == 2 else []
    near_zero_dims = np.where(np.std(ya, axis=0) < 1e-8)[0].astype(int).tolist() if ya.ndim == 2 else []
    large_scale_dims = np.where(np.percentile(np.abs(ya), 99, axis=0) > 10.0)[0].astype(int).tolist() if ya.ndim == 2 else []
    if action_stats.get("num_nan") or action_stats.get("num_inf"):
        add_issue(issues, "FAIL", "action_nan_inf", "y_action contains NaN or Inf.")
    if action_stats.get("p99_abs") is not None and action_stats["p99_abs"] > 10.0:
        add_issue(issues, "FAIL", "large_raw_action_scale", f"y_action p99 abs={action_stats['p99_abs']:.6f}.")
    if y_action.shape[1] > 0 and len(near_zero_dims) > 0.5 * y_action.shape[1]:
        add_issue(issues, "WARN", "many_near_zero_action_dims", f"{len(near_zero_dims)}/{y_action.shape[1]} action dims have std < 1e-8.")

    extra_stats = action_history_diagnostics(paper_x, state_action_x, y_action, th, action_dim)
    if extra_stats.get("state_action_extra_dim") != extra_stats.get("expected_extra_dim"):
        add_issue(issues, "FAIL", "state_action_extra_dim_mismatch", f"extra dim={extra_stats.get('state_action_extra_dim')} expected={extra_stats.get('expected_extra_dim')}.")
    exact_rate = safe_float(extra_stats.get("action_history_target_exact_match_rate"))
    if exact_rate is not None:
        if exact_rate > 0.20:
            add_issue(issues, "FAIL", "action_history_target_leakage", f"exact target-action match rate={exact_rate:.6f} > 0.20.")
        elif exact_rate > 0.05:
            add_issue(issues, "WARN", "action_history_target_possible_leakage", f"exact target-action match rate={exact_rate:.6f} > 0.05.")
    corr = safe_float(extra_stats.get("action_history_target_corr_max"))
    if corr is not None and corr > 0.999:
        add_issue(issues, "WARN", "action_history_target_corr_high", f"max corr between last history action and target action={corr:.6f}.")

    template_path = Path(str(data["action_template_json_or_pickle_path"])) if "action_template_json_or_pickle_path" in data.files else None
    action_template_summary = load_action_template_summary(template_path, data) if template_path else {"exists": False}
    if action_template_summary.get("dim") != 14:
        add_issue(issues, "FAIL", "action_codec_dim_not_14", f"Expected action codec dim 14, got {action_template_summary.get('dim')}.")
    if action_template_summary.get("num_camera_config_paths", 0) > 0:
        add_issue(issues, "FAIL", "action_codec_encodes_camera_config", f"Action codec target includes {action_template_summary.get('num_camera_config_paths')} camera_config paths.")
    if action_template_summary.get("invalid_executable_paths"):
        add_issue(issues, "FAIL", "action_codec_invalid_paths", f"Non-executable action paths: {action_template_summary.get('invalid_executable_paths')[:10]}")
    rt_err = action_template_summary.get("roundtrip_error")
    if isinstance(rt_err, (int, float)) and rt_err > 1e-6:
        add_issue(issues, "FAIL", "action_codec_roundtrip_error", f"roundtrip error={rt_err} > 1e-6.")

    checkpoint_backends = scan_checkpoint_backends(Path(args.ckpt_root))
    for item in checkpoint_backends:
        if item.get("training_backend") != "torch":
            add_issue(issues, "FAIL", "non_torch_checkpoint_backend", f"{item['checkpoint']} backend={item.get('training_backend')}")
        if item.get("future_model_type") != "torch_conditional_ddpm_future_state" or not item.get("ddpm_used"):
            add_issue(issues, "FAIL", "non_ddpm_future_checkpoint", f"{item['checkpoint']} future={item.get('future_model_type')} ddpm={item.get('ddpm_used')}")
        if item.get("idm_feature_mode") != "paper_full_state_history_future":
            add_issue(issues, "FAIL", "idm_not_full_state", f"{item['checkpoint']} idm_feature_mode={item.get('idm_feature_mode')}")
    if not checkpoint_backends:
        add_issue(issues, "WARN", "no_checkpoint_config_found", "No config/train_log backend metadata found under ckpt_root.")

    idm_debug = compute_idm_diagnostics(data, Path(args.ckpt_root))
    raw_vals = [safe_float(x.get("raw_action_mse")) for x in idm_debug]
    norm_vals = [safe_float(x.get("normalized_action_mse")) for x in idm_debug]
    ood_vals = [safe_float(x.get("pred_action_ood_mean")) for x in idm_debug]
    raw_vals = [x for x in raw_vals if x is not None]
    norm_vals = [x for x in norm_vals if x is not None]
    ood_vals = [x for x in ood_vals if x is not None]
    if raw_vals and max(raw_vals) > 10.0:
        add_issue(issues, "FAIL", "idm_raw_mse_too_large", f"heldout IDM raw-space MSE max={max(raw_vals):.6f} > 10.0.")
    if norm_vals and max(norm_vals) > 5.0:
        add_issue(issues, "FAIL", "idm_normalized_mse_too_large", f"heldout IDM normalized MSE max={max(norm_vals):.6f} > 5.0.")
    if ood_vals and max(ood_vals) > 5.0:
        add_issue(issues, "FAIL", "idm_pred_ood_too_large", f"heldout IDM predicted action OOD mean max={max(ood_vals):.6f} > 5.0.")

    split = data["split_name"].astype(str)
    cond = data["condition_name"].astype(str)
    payload = {
        "verdict": "FAIL" if any(i["level"] == "FAIL" for i in issues) else ("WARN" if any(i["level"] == "WARN" for i in issues) else "PASS"),
        "dimension_summary": dimension_summary,
        "condition_counts": {str(k): int(v) for k, v in zip(*np.unique(cond, return_counts=True))},
        "split_counts": {str(k): int(v) for k, v in zip(*np.unique(split, return_counts=True))},
        "action_stats": action_stats,
        "zero_variance_action_dims_count": len(zero_var_dims),
        "near_zero_variance_action_dims_count": len(near_zero_dims),
        "large_scale_action_dims_count": len(large_scale_dims),
        "zero_variance_action_dims_head": zero_var_dims[:20],
        "near_zero_variance_action_dims_head": near_zero_dims[:20],
        "large_scale_action_dims_head": large_scale_dims[:20],
        "state_action_extra_block": extra_stats,
        "action_template_summary": action_template_summary,
        "checkpoint_backends": checkpoint_backends,
        "checkpoint_backend_counts": dict(Counter(item.get("training_backend", "unknown") for item in checkpoint_backends)),
        "idm_debug": idm_debug,
        "issues": issues,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(payload, indent=2, sort_keys=True))
    write_report(Path(args.out_md), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    print("[Phase3] wrote", args.out_json)
    print("[Phase3] wrote", args.out_md)

    if args.strict and payload["verdict"] == "FAIL":
        raise SystemExit("[Phase3][FAIL] action/IDM diagnostics found fatal issues")


if __name__ == "__main__":
    main()
