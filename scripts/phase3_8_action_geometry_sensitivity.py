#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

REQUIRED_CONDITIONS = ["free", "hidden_pin", "hidden_high_friction", "hidden_breakaway_pin"]


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


def str_array(data: Any, key: str) -> np.ndarray:
    return np.asarray([str(x) for x in data[key]])


def select_indices(data: Any, phase37_raw: Dict[str, Any], samples_per_condition: int) -> List[Dict[str, Any]]:
    selected = phase37_raw.get("selected_windows") or []
    if selected:
        grouped: Dict[str, List[Dict[str, Any]]] = {c: [] for c in REQUIRED_CONDITIONS}
        for item in selected:
            cond = str(item.get("condition", ""))
            grouped.setdefault(cond, []).append({
                "idx": int(item["idx"]),
                "condition": cond,
            })
        out: List[Dict[str, Any]] = []
        for cond in REQUIRED_CONDITIONS:
            out.extend(grouped.get(cond, [])[:samples_per_condition])
        return out

    conds = str_array(data, "condition_name")
    out = []
    for cond in REQUIRED_CONDITIONS:
        idxs = np.where(conds == cond)[0]
        for idx in idxs[:samples_per_condition]:
            out.append({"idx": int(idx), "condition": cond})
    return out


def component_from_path(path: str) -> str:
    if "pose0/0/" in path:
        return "pose0_xyz"
    if "pose0/1/" in path:
        return "pose0_quat"
    if "pose1/0/" in path:
        return "pose1_xyz"
    if "pose1/1/" in path:
        return "pose1_quat"
    return "other"


def dim_name(path: str, i: int) -> str:
    return f"{i:02d}:{path}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--windows", default="data/phase3_state_diff_windows/phase3_windows.npz")
    parser.add_argument("--action_template", default="data/phase3_state_diff_windows/phase3_action_template.pkl")
    parser.add_argument("--checkpoint_root", default="checkpoints/phase3")
    parser.add_argument("--phase37_raw", default="reports/phase3_7_learned_action_alignment_raw_summary.json")
    parser.add_argument("--baselines", nargs="+", default=["paper_state", "state_action"])
    parser.add_argument("--samples_per_condition", type=int, default=6)
    parser.add_argument("--pred_samples", type=int, default=16)
    parser.add_argument("--seed_base", type=int, default=380000)
    parser.add_argument("--out_csv", default="reports/phase3_8_action_geometry_sensitivity_dims.csv")
    parser.add_argument("--out_json", default="reports/phase3_8_action_geometry_sensitivity_summary.json")
    parser.add_argument("--out_md", default="reports/phase3_8_action_geometry_sensitivity_report.md")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    data, meta = load_windows(root, args.windows)
    codec = load_codec(root, data, args.action_template)
    paths = codec_paths(codec)
    if len(paths) != 14:
        raise SystemExit(f"[Phase3.8][FAIL] codec paths must be 14, got {len(paths)}")

    phase37_raw = load_json(root / args.phase37_raw)
    selected = select_indices(data, phase37_raw, args.samples_per_condition)

    all_dim_rows: List[Dict[str, Any]] = []
    aggregate_records: List[Dict[str, Any]] = []

    for baseline in args.baselines:
        ckpt, state_model, idm = load_models(root, args.checkpoint_root, baseline)

        for item in selected:
            idx = int(item["idx"])
            cond = str(item["condition"])
            model_x = np.asarray(data["paper_x"][idx] if baseline == "paper_state" else data["state_action_x"][idx], dtype=np.float32).reshape(1, -1)
            pred_samples = call_future_sample(state_model, model_x, args.pred_samples, args.seed_base + idx)
            pred_future = np.mean(pred_samples, axis=0).reshape(-1)

            future_key = infer_future_key(data, pred_future.shape[0])
            gt_future = np.asarray(data[future_key][idx], dtype=np.float32).reshape(-1)
            hist_states = np.asarray(data["paper_x"][idx], dtype=np.float32).reshape(1, -1)

            idm_x_gt = np.concatenate([hist_states, gt_future.reshape(1, -1)], axis=1)
            idm_x_pred = np.concatenate([hist_states, pred_future.reshape(1, -1)], axis=1)
            idm_gt = idm_predict(idm, idm_x_gt)
            idm_pred = idm_predict(idm, idm_x_pred)
            gt = np.asarray(data["y_action"][idx], dtype=np.float32).reshape(-1)

            for source, vec in [("idm_gt_future", idm_gt), ("idm_pred_future", idm_pred)]:
                err = vec - gt
                for i, path in enumerate(paths):
                    all_dim_rows.append({
                        "baseline": baseline,
                        "condition": cond,
                        "window_idx": idx,
                        "action_source": source,
                        "dim": i,
                        "path": path,
                        "dim_name": dim_name(path, i),
                        "component": component_from_path(path),
                        "gt_value": float(gt[i]),
                        "pred_value": float(vec[i]),
                        "signed_error": float(err[i]),
                        "abs_error": float(abs(err[i])),
                    })
                aggregate_records.append({
                    "baseline": baseline,
                    "condition": cond,
                    "window_idx": idx,
                    "action_source": source,
                    "future_key": future_key,
                    "future_mae": float(np.mean(np.abs(pred_future - gt_future))),
                    "action_mae": float(np.mean(np.abs(err))),
                    "action_l2": float(np.linalg.norm(err)),
                    "action_cosine": cosine(vec, gt),
                })

    out_csv = root / args.out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_dim_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_dim_rows)

    # Aggregate dimension stats.
    by_key: Dict[tuple, List[Dict[str, Any]]] = {}
    for row in all_dim_rows:
        key = (row["baseline"], row["action_source"], row["component"], row["dim"], row["path"])
        by_key.setdefault(key, []).append(row)

    dim_stats = []
    for (baseline, source, component, dim, path), rows in sorted(by_key.items()):
        abs_vals = [float(r["abs_error"]) for r in rows]
        signed_vals = [float(r["signed_error"]) for r in rows]
        dim_stats.append({
            "baseline": baseline,
            "action_source": source,
            "component": component,
            "dim": dim,
            "path": path,
            "mean_abs_error": float(np.mean(abs_vals)),
            "max_abs_error": float(np.max(abs_vals)),
            "mean_signed_error": float(np.mean(signed_vals)),
            "num_rows": len(rows),
        })

    component_stats: Dict[str, Dict[str, Any]] = {}
    for row in all_dim_rows:
        key = f"{row['baseline']}::{row['action_source']}::{row['component']}"
        component_stats.setdefault(key, {"abs": [], "signed": []})
        component_stats[key]["abs"].append(float(row["abs_error"]))
        component_stats[key]["signed"].append(float(row["signed_error"]))
    component_table = []
    for key, vals in sorted(component_stats.items()):
        baseline, source, component = key.split("::")
        component_table.append({
            "baseline": baseline,
            "action_source": source,
            "component": component,
            "mean_abs_error": float(np.mean(vals["abs"])),
            "max_abs_error": float(np.max(vals["abs"])),
            "mean_signed_error": float(np.mean(vals["signed"])),
        })

    top_dims = sorted(dim_stats, key=lambda x: x["mean_abs_error"], reverse=True)[:20]

    payload = {
        "verdict": "PASS",
        "num_dim_rows": len(all_dim_rows),
        "num_windows": len(selected),
        "selected_windows": selected,
        "dim_stats": dim_stats,
        "component_table": component_table,
        "top_dims": top_dims,
        "aggregate_records": aggregate_records,
        "interpretation": "Sensitivity only; no action execution or Phase4/CPS.",
    }

    (root / args.out_json).write_text(json.dumps(payload, indent=2, sort_keys=True))

    lines = [
        "# Phase3.8 Action Geometry Sensitivity",
        "",
        "## Verdict",
        "",
        "- Verdict: `PASS`",
        f"- Dimension rows: `{len(all_dim_rows)}`",
        f"- Selected windows: `{len(selected)}`",
        "",
        "## Component Error Table",
        "",
        "| Baseline | Source | Component | Mean abs error | Max abs error | Mean signed error |",
        "|---|---|---|---:|---:|---:|",
    ]
    for item in component_table:
        lines.append(
            f"| `{item['baseline']}` | `{item['action_source']}` | `{item['component']}` | "
            f"{item['mean_abs_error']:.6f} | {item['max_abs_error']:.6f} | {item['mean_signed_error']:.6f} |"
        )

    lines += [
        "",
        "## Top Error Dimensions",
        "",
        "| Baseline | Source | Dim | Path | Component | Mean abs | Max abs | Mean signed |",
        "|---|---|---:|---|---|---:|---:|---:|",
    ]
    for item in top_dims:
        lines.append(
            f"| `{item['baseline']}` | `{item['action_source']}` | {item['dim']} | `{item['path']}` | "
            f"`{item['component']}` | {item['mean_abs_error']:.6f} | {item['max_abs_error']:.6f} | {item['mean_signed_error']:.6f} |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "- This report identifies which encoded pick/place dimensions dominate IDM error.",
        "- It does not prove execution repair by itself.",
        "- Use the Phase3.8 repair probe for matched-prefix execution evidence.",
    ]
    (root / args.out_md).write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
