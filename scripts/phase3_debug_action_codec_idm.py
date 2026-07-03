#!/usr/bin/env python3
"""Debug Phase3 action codec and inverse dynamics targets.

Focuses on abnormal Phase3 action metrics:
- raw y_action scale;
- zero-variance action dimensions;
- paper_x vs state_action_x dimensionality;
- action-history extra block variance;
- checkpoint backend metadata;
- optional action template diagnostics.

This script intentionally does not require DeformableRavens runtime.
"""

import argparse
import json
import math
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


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


def load_json_maybe(path: Path) -> Dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}
    return {}


def get_key(data, names):
    keys = set(data.files)
    for name in names:
        if name in keys:
            return name
    return None


def add_issue(issues, level, name, detail):
    issues.append({"level": level, "name": name, "detail": detail})


def scan_checkpoint_backends(ckpt_root: Path):
    out = []
    seen = set()

    for cfg in sorted(ckpt_root.glob("*/*/config.json")):
        j = load_json_maybe(cfg)
        item = {
            "checkpoint": str(cfg.parent),
            "training_backend": j.get("training_backend", j.get("backend", "unknown")),
        }
        key = (item["checkpoint"], item["training_backend"])
        if key not in seen:
            seen.add(key)
            out.append(item)

    for cfg in sorted(ckpt_root.glob("*/*/train_log.json")):
        j = load_json_maybe(cfg)
        if isinstance(j, dict) and "training_backend" in j:
            item = {
                "checkpoint": str(cfg.parent),
                "training_backend": j.get("training_backend", "unknown"),
            }
            key = (item["checkpoint"], item["training_backend"])
            if key not in seen:
                seen.add(key)
                out.append(item)

    return out


def load_action_template_summary(path: Path) -> Dict[str, Any]:
    if not path or not Path(path).exists():
        return {"exists": False, "path": str(path)}
    with Path(path).open("rb") as f:
        obj = pickle.load(f)
    paths = obj.get("paths", []) if isinstance(obj, dict) else []
    path_strings = ["/".join(str(x) for x in p) for p in paths]
    forbidden_tokens = ["hidden", "condition", "seed", "object", "success", "reward", "file", "id"]
    forbidden = [p for p in path_strings if any(tok in p.lower() for tok in forbidden_tokens)]
    camera = [p for p in path_strings if "camera_config" in p]
    params = [p for p in path_strings if p.startswith("params/")]
    return {
        "exists": True,
        "path": str(path),
        "num_paths": len(path_strings),
        "num_camera_config_paths": len(camera),
        "num_param_paths": len(params),
        "camera_config_paths_head": camera[:20],
        "param_paths_head": params[:20],
        "forbidden_token_paths": forbidden[:50],
    }


def safe_float(x: Any):
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


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
    out = []
    for cfg in sorted(Path(ckpt_root).glob("*/*/config.json")):
        try:
            meta = load_json_maybe(cfg)
            idm = load_inverse_model(cfg.parent / "inverse_dynamics.pt")
            x = np.concatenate([paper_x[held], y_flat[held]], axis=1)
            pred = idm.predict(x).astype(np.float32)
            target = y_action[held].astype(np.float32)
            raw_mse = float(np.mean((pred - target) ** 2))
            std = np.asarray(getattr(idm, "train_action_std", np.std(y_action, axis=0)), dtype=np.float32)
            std = np.where(std < 1e-6, 1.0, std)
            norm_mse = float(np.mean(((pred - target) / std) ** 2))
            ood = idm.ood_score(pred) if hasattr(idm, "ood_score") else np.sqrt(np.mean((pred / std) ** 2, axis=-1))
            out.append({
                "checkpoint": str(cfg.parent),
                "baseline": meta.get("baseline", cfg.parent.parent.name),
                "training_backend": meta.get("training_backend", meta.get("backend", "unknown")),
                "raw_action_mse": raw_mse,
                "normalized_action_mse": norm_mse,
                "pred_action_ood_mean": float(np.mean(ood)),
                "pred_action_ood_max": float(np.max(ood)),
                "uses_standardized_target": bool(hasattr(idm, "y_std")),
                "heldout_rows": int(len(held)),
            })
        except Exception as exc:
            out.append({"checkpoint": str(cfg.parent), "error": repr(exc)})
    return out


def write_report(path: Path, payload: Dict):
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
    lines.append("## Dataset Counts")
    lines.append("")
    lines.append(f"- Condition counts: `{payload['condition_counts']}`")
    lines.append(f"- Split counts: `{payload['split_counts']}`")
    lines.append("")
    lines.append("## Action Target Stats")
    lines.append("")
    s = payload["action_stats"]
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    for k in ["shape", "mean", "std", "min", "max", "mean_abs", "p95_abs", "p99_abs", "num_nan", "num_inf"]:
        lines.append(f"| `{k}` | `{s.get(k)}` |")
    lines.append("")
    lines.append("## Action Dimension Diagnostics")
    lines.append("")
    lines.append(f"- Zero-variance dims: `{payload['zero_variance_action_dims_count']}`")
    lines.append(f"- Near-zero-variance dims: `{payload['near_zero_variance_action_dims_count']}`")
    lines.append(f"- Large-scale dims count: `{payload['large_scale_action_dims_count']}`")
    lines.append(f"- Zero-variance dims head: `{payload['zero_variance_action_dims_head']}`")
    lines.append(f"- Near-zero dims head: `{payload['near_zero_variance_action_dims_head']}`")
    lines.append(f"- Large-scale dims head: `{payload['large_scale_action_dims_head']}`")
    lines.append("")
    lines.append("## State-Action Extra Block")
    lines.append("")
    for k, v in payload["state_action_extra_block"].items():
        lines.append(f"- `{k}`: `{v}`")
    lines.append("")
    lines.append("## Action Template")
    lines.append("")
    ats = payload.get("action_template_summary", {})
    for k in ["path", "num_paths", "num_camera_config_paths", "num_param_paths"]:
        lines.append(f"- `{k}`: `{ats.get(k)}`")
    if ats.get("camera_config_paths_head"):
        lines.append(f"- `camera_config_paths_head`: `{ats.get('camera_config_paths_head')}`")
    if ats.get("forbidden_token_paths"):
        lines.append(f"- `forbidden_token_paths`: `{ats.get('forbidden_token_paths')}`")
    lines.append("")
    lines.append("## IDM Heldout Reconstruction")
    lines.append("")
    lines.append("| Checkpoint | Backend | Raw MSE | Normalized MSE | Pred OOD mean | Pred OOD max | Standardized target |")
    lines.append("|---|---|---:|---:|---:|---:|---|")
    for item in payload.get("idm_debug", []):
        ck = Path(item.get("checkpoint", "")).name if item.get("checkpoint") else "NA"
        label = f"{item.get('baseline', '')}/{ck}"
        lines.append(f"| `{label}` | `{item.get('training_backend', 'NA')}` | `{item.get('raw_action_mse')}` | `{item.get('normalized_action_mse')}` | `{item.get('pred_action_ood_mean')}` | `{item.get('pred_action_ood_max')}` | `{item.get('uses_standardized_target', 'NA')}` |")
    lines.append("")
    lines.append("## Checkpoint Backends")
    lines.append("")
    lines.append("| Checkpoint | Backend |")
    lines.append("|---|---|")
    for item in payload["checkpoint_backends"]:
        lines.append(f"| `{item['checkpoint']}` | `{item['training_backend']}` |")
    lines.append("")
    lines.append("## Issues")
    lines.append("")
    lines.append("| Level | Name | Detail |")
    lines.append("|---|---|---|")
    for issue in payload["issues"]:
        lines.append(f"| `{issue['level']}` | `{issue['name']}` | {issue['detail']} |")
    if not payload["issues"]:
        lines.append("| `PASS` | `none` | No action/IDM diagnostic issue detected. |")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- If this report is FAIL, do not trust policy rollout.")
    lines.append("- High raw action scale may make raw MSE misleading, but high OOD still requires action normalization/codec diagnosis.")
    lines.append("- If `state_action_x` is not larger than `paper_x`, the state_action baseline is not implemented correctly.")
    path.write_text("\n".join(lines))


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

    paper_key = get_key(data, ["paper_x"])
    state_action_key = get_key(data, ["state_action_x"])
    y_action_key = get_key(data, ["y_action", "action_y", "target_action"])
    cond_key = get_key(data, ["condition_name"])
    split_key = get_key(data, ["split_name"])

    if paper_key is None:
        add_issue(issues, "FAIL", "paper_x_missing", "NPZ has no paper_x.")
    if state_action_key is None:
        add_issue(issues, "FAIL", "state_action_x_missing", "NPZ has no state_action_x.")
    if y_action_key is None:
        add_issue(issues, "FAIL", "y_action_missing", "NPZ has no y_action/action target.")

    paper_x = data[paper_key] if paper_key else None
    state_action_x = data[state_action_key] if state_action_key else None
    y_action = data[y_action_key] if y_action_key else None

    dimension_summary = {}
    if paper_x is not None:
        dimension_summary["paper_x_shape"] = list(paper_x.shape)
    if state_action_x is not None:
        dimension_summary["state_action_x_shape"] = list(state_action_x.shape)
    if y_action is not None:
        dimension_summary["y_action_shape"] = list(y_action.shape)

    if paper_x is not None and state_action_x is not None:
        if state_action_x.shape[1] <= paper_x.shape[1]:
            add_issue(
                issues,
                "FAIL",
                "state_action_not_larger_than_paper",
                f"state_action_x dim {state_action_x.shape[1]} <= paper_x dim {paper_x.shape[1]}. "
                "The state_action baseline may not include action history.",
            )
            extra = np.zeros((state_action_x.shape[0], 0), dtype=np.float32)
        else:
            extra = state_action_x[:, paper_x.shape[1]:]
        extra_stats = arr_stats(extra) if extra.size else {"shape": list(extra.shape), "std": 0.0}
        if extra.size and np.std(extra) < 1e-8:
            add_issue(
                issues,
                "FAIL",
                "state_action_extra_block_constant",
                "state_action baseline has limited additional information because the current primitive dataset contains very short action histories.",
            )
    else:
        extra_stats = {}

    action_stats = arr_stats(y_action) if y_action is not None else {}

    if y_action is not None:
        ya = y_action.astype(np.float64)
        per_dim_std = np.std(ya, axis=0)
        per_dim_abs_p99 = np.percentile(np.abs(ya), 99, axis=0)

        zero_var_dims = np.where(per_dim_std == 0)[0].tolist()
        near_zero_dims = np.where(per_dim_std < 1e-8)[0].tolist()
        large_scale_dims = np.where(per_dim_abs_p99 > 10.0)[0].tolist()

        if action_stats.get("num_nan", 0) > 0 or action_stats.get("num_inf", 0) > 0:
            add_issue(issues, "FAIL", "action_nan_inf", "y_action contains NaN or Inf.")

        if action_stats.get("p99_abs") is not None and action_stats["p99_abs"] > 10.0:
            add_issue(
                issues,
                "WARN",
                "large_raw_action_scale",
                f"y_action p99 abs={action_stats['p99_abs']:.6f}. "
                "Raw action MSE may be dominated by unnormalized coordinates or wrong fields.",
            )

        if y_action.shape[1] > 0 and len(near_zero_dims) > 0.5 * y_action.shape[1]:
            add_issue(
                issues,
                "WARN",
                "many_near_zero_action_dims",
                f"{len(near_zero_dims)}/{y_action.shape[1]} action dims have std < 1e-8.",
            )
    else:
        zero_var_dims = []
        near_zero_dims = []
        large_scale_dims = []

    if cond_key:
        condition_counts = {str(k): int(v) for k, v in zip(*np.unique(data[cond_key].astype(str), return_counts=True))}
    else:
        condition_counts = {}

    if split_key:
        split_counts = {str(k): int(v) for k, v in zip(*np.unique(data[split_key].astype(str), return_counts=True))}
    else:
        split_counts = {}

    checkpoint_backends = scan_checkpoint_backends(Path(args.ckpt_root))
    if checkpoint_backends:
        for item in checkpoint_backends:
            if item["training_backend"] not in {"torch", "unknown"}:
                add_issue(
                    issues,
                    "FAIL",
                    "non_torch_checkpoint_backend",
                    f"{item['checkpoint']} backend={item['training_backend']}",
                )
    else:
        add_issue(issues, "WARN", "no_checkpoint_config_found", "No config/train_log backend metadata found under ckpt_root.")

    checkpoint_backend_counts = dict(Counter(item.get("training_backend", "unknown") for item in checkpoint_backends))

    template_path = Path(str(data["action_template_json_or_pickle_path"])) if "action_template_json_or_pickle_path" in data.files else None
    action_template_summary = load_action_template_summary(template_path) if template_path else {"exists": False}
    if action_template_summary.get("num_camera_config_paths", 0) > 0:
        add_issue(
            issues,
            "FAIL",
            "action_codec_encodes_camera_config",
            f"Action codec target includes {action_template_summary.get('num_camera_config_paths')} camera_config numeric paths. These are observation metadata, not executable pick-place action parameters.",
        )
    if action_template_summary.get("forbidden_token_paths"):
        add_issue(
            issues,
            "FAIL",
            "action_codec_forbidden_metadata_paths",
            f"Action codec target includes forbidden metadata-like paths: {action_template_summary.get('forbidden_token_paths')[:10]}",
        )

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

    has_fail = any(i["level"] == "FAIL" for i in issues)
    has_warn = any(i["level"] == "WARN" for i in issues)
    verdict = "FAIL" if has_fail else ("WARN" if has_warn else "PASS")

    payload = {
        "verdict": verdict,
        "dimension_summary": dimension_summary,
        "condition_counts": condition_counts,
        "split_counts": split_counts,
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
        "checkpoint_backend_counts": checkpoint_backend_counts,
        "idm_debug": idm_debug,
        "issues": issues,
    }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(payload, indent=2, sort_keys=True))
    write_report(Path(args.out_md), payload)

    print(json.dumps(payload, indent=2, sort_keys=True))
    print("[Phase3] wrote", args.out_json)
    print("[Phase3] wrote", args.out_md)

    if args.strict and has_fail:
        raise SystemExit("[Phase3][FAIL] action/IDM diagnostics found fatal issues")


if __name__ == "__main__":
    main()