#!/usr/bin/env python3
"""Phase1 RGB-D observation check for hidden-contact cable.

This script verifies whether hidden-contact conditions leak into visual
observations. It compares the first saved RGB-D observation of each hidden
condition against the corresponding free condition under the same
ccda_visible_seed.

Outputs:
  - reports/phase1_rgbd_check/seed_*.png
  - reports/phase1_rgbd_check_summary.json
  - reports/phase1_rgbd_check_report.md
"""

import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CONDITIONS = ["free", "hidden_pin", "hidden_high_friction", "hidden_side_jam"]


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def get_extras(info: Dict) -> Dict:
    if isinstance(info, dict):
        return info.get("extras", {})
    return {}


def load_first_info(cond_dir: Path, fname: str) -> Dict:
    info_path = cond_dir / "info" / fname
    info_list = load_pickle(info_path)
    if isinstance(info_list, list) and len(info_list) > 0:
        return info_list[0]
    return {}


def episode_files(cond_dir: Path) -> List[Path]:
    color_dir = cond_dir / "color"
    if not color_dir.exists():
        return []
    return sorted(color_dir.glob("*.pkl"))


def parse_episode_len(path: Path) -> int:
    try:
        return int(path.stem.split("-")[-1])
    except Exception:
        return -1


def get_seed_for_episode(cond_dir: Path, color_file: Path) -> str:
    info = load_first_info(cond_dir, color_file.name)
    extras = get_extras(info)
    seed = extras.get("ccda_visible_seed", None)
    if seed is None:
        return color_file.stem
    return str(seed)


def group_by_seed(data_root: Path) -> Dict[str, Dict[str, Path]]:
    out: Dict[str, Dict[str, Path]] = {}
    for cond in CONDITIONS:
        cond_dir = data_root / cond
        for color_file in episode_files(cond_dir):
            seed = get_seed_for_episode(cond_dir, color_file)
            out.setdefault(seed, {})[cond] = color_file
    return out


def select_color_frame(color_obj: Any, timestep: int, camera_index: int) -> np.ndarray:
    arr = np.asarray(color_obj)

    if arr.ndim == 5:
        # (T, C, H, W, 3)
        t = min(max(timestep, 0), arr.shape[0] - 1)
        c = min(max(camera_index, 0), arr.shape[1] - 1)
        frame = arr[t, c]
    elif arr.ndim == 4:
        if arr.shape[-1] == 3 and arr.shape[0] <= 8:
            # (C, H, W, 3), e.g. last_color
            c = min(max(camera_index, 0), arr.shape[0] - 1)
            frame = arr[c]
        elif arr.shape[-1] == 3:
            # (T, H, W, 3)
            t = min(max(timestep, 0), arr.shape[0] - 1)
            frame = arr[t]
        else:
            raise ValueError(f"Unsupported color shape: {arr.shape}")
    elif arr.ndim == 3 and arr.shape[-1] == 3:
        frame = arr
    else:
        raise ValueError(f"Unsupported color shape: {arr.shape}")

    frame = np.asarray(frame)
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def select_depth_frame(depth_obj: Any, timestep: int, camera_index: int) -> np.ndarray:
    arr = np.asarray(depth_obj)

    if arr.ndim == 4:
        # (T, C, H, W)
        t = min(max(timestep, 0), arr.shape[0] - 1)
        c = min(max(camera_index, 0), arr.shape[1] - 1)
        frame = arr[t, c]
    elif arr.ndim == 3:
        if arr.shape[0] <= 8:
            # (C, H, W), e.g. last_depth
            c = min(max(camera_index, 0), arr.shape[0] - 1)
            frame = arr[c]
        else:
            # (T, H, W)
            t = min(max(timestep, 0), arr.shape[0] - 1)
            frame = arr[t]
    elif arr.ndim == 2:
        frame = arr
    else:
        raise ValueError(f"Unsupported depth shape: {arr.shape}")

    return np.asarray(frame, dtype=np.float32)


def load_first_rgbd(cond_dir: Path, fname: str, camera_index: int) -> Tuple[np.ndarray, np.ndarray]:
    color = load_pickle(cond_dir / "color" / fname)
    depth = load_pickle(cond_dir / "depth" / fname)
    rgb = select_color_frame(color, timestep=0, camera_index=camera_index)
    dep = select_depth_frame(depth, timestep=0, camera_index=camera_index)
    return rgb, dep


def normalize_rgb(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float32)
    if rgb.max() > 1.5:
        rgb = rgb / 255.0
    return np.clip(rgb, 0.0, 1.0)


def normalize_depth_for_display(depth: np.ndarray) -> np.ndarray:
    d = np.asarray(depth, dtype=np.float32)
    mask = np.isfinite(d) & (d > 0)
    if not mask.any():
        return np.zeros_like(d, dtype=np.float32)
    lo = float(np.percentile(d[mask], 2))
    hi = float(np.percentile(d[mask], 98))
    if hi <= lo:
        hi = lo + 1e-6
    out = (d - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def diff_metrics(rgb_ref: np.ndarray, depth_ref: np.ndarray, rgb: np.ndarray, depth: np.ndarray) -> Dict[str, float]:
    a = normalize_rgb(rgb_ref)
    b = normalize_rgb(rgb)
    rgb_diff = np.abs(a - b)

    da = np.asarray(depth_ref, dtype=np.float32)
    db = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(da) & np.isfinite(db) & (da > 0) & (db > 0)

    if valid.any():
        depth_abs = np.abs(da[valid] - db[valid])
        depth_mean = float(np.mean(depth_abs))
        depth_max = float(np.max(depth_abs))
    else:
        depth_mean = float("nan")
        depth_max = float("nan")

    return {
        "rgb_abs_mean": float(np.mean(rgb_diff)),
        "rgb_abs_max": float(np.max(rgb_diff)),
        "depth_abs_mean": depth_mean,
        "depth_abs_max": depth_max,
        "valid_depth_pixels": int(np.sum(valid)),
    }


def fmt(x: Any) -> str:
    if x is None:
        return "NA"
    try:
        xf = float(x)
        if math.isnan(xf):
            return "nan"
        return f"{xf:.6f}"
    except Exception:
        return str(x)


def plot_seed(
    seed: str,
    group: Dict[str, Path],
    data_root: Path,
    camera_index: int,
    outdir: Path,
) -> List[Dict[str, Any]]:
    if "free" not in group:
        return []

    free_file = group["free"]
    free_dir = data_root / "free"
    rgb_free, depth_free = load_first_rgbd(free_dir, free_file.name, camera_index)

    records: List[Dict[str, Any]] = []

    n_rows = len(CONDITIONS)
    n_cols = 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))

    if n_rows == 1:
        axes = np.expand_dims(axes, 0)

    for r, cond in enumerate(CONDITIONS):
        for c in range(n_cols):
            axes[r, c].axis("off")

        axes[r, 0].set_ylabel(cond, fontsize=12)

        if cond not in group:
            axes[r, 0].text(0.5, 0.5, f"{cond}\nmissing", ha="center", va="center")
            continue

        cond_file = group[cond]
        cond_dir = data_root / cond
        rgb, depth = load_first_rgbd(cond_dir, cond_file.name, camera_index)
        metrics = diff_metrics(rgb_free, depth_free, rgb, depth)

        records.append(
            {
                "visible_seed": seed,
                "condition": cond,
                "file": cond_file.name,
                "camera_index": camera_index,
                **metrics,
            }
        )

        rgb_disp = normalize_rgb(rgb)
        depth_disp = normalize_depth_for_display(depth)

        rgb_diff = np.mean(np.abs(normalize_rgb(rgb_free) - normalize_rgb(rgb)), axis=2)
        depth_diff_raw = np.abs(depth_free.astype(np.float32) - depth.astype(np.float32))
        depth_diff = normalize_depth_for_display(depth_diff_raw)

        axes[r, 0].imshow(rgb_disp)
        axes[r, 0].set_title(f"{cond} RGB")

        axes[r, 1].imshow(depth_disp, cmap="gray")
        axes[r, 1].set_title(f"{cond} depth")

        axes[r, 2].imshow(rgb_diff, cmap="magma")
        axes[r, 2].set_title(f"|RGB - free|\nmean={fmt(metrics['rgb_abs_mean'])}")

        axes[r, 3].imshow(depth_diff, cmap="magma")
        axes[r, 3].set_title(f"|Depth - free|\nmean={fmt(metrics['depth_abs_mean'])}")

    fig.suptitle(f"Phase1 RGB-D observation check | visible_seed={seed} | camera={camera_index}")
    fig.tight_layout()
    out_path = outdir / f"seed_{seed}_rgbd_check.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return records


def summarize_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_cond: Dict[str, Dict[str, List[float]]] = {}

    for rec in records:
        cond = rec["condition"]
        by_cond.setdefault(
            cond,
            {
                "rgb_abs_mean": [],
                "rgb_abs_max": [],
                "depth_abs_mean": [],
                "depth_abs_max": [],
            },
        )
        for k in by_cond[cond]:
            v = rec.get(k)
            if v is not None:
                try:
                    vf = float(v)
                    if not math.isnan(vf):
                        by_cond[cond][k].append(vf)
                except Exception:
                    pass

    out = {}
    for cond, vals in by_cond.items():
        out[cond] = {}
        for k, xs in vals.items():
            out[cond][k] = float(np.mean(xs)) if xs else None
        out[cond]["num_records"] = len([r for r in records if r["condition"] == cond])
    return out


def write_report(path: Path, summary: Dict[str, Any]) -> None:
    lines = []
    lines.append("# Phase1 RGB-D Observation Check Report")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append("This report checks whether hidden-contact conditions leak into saved RGB-D observations.")
    lines.append("For each visible seed, the first observation under each hidden condition is compared against `free`.")
    lines.append("")
    lines.append("## Aggregate Metrics")
    lines.append("")
    lines.append("| Condition | N | Mean RGB Abs Diff | Max RGB Abs Diff | Mean Depth Abs Diff | Max Depth Abs Diff |")
    lines.append("|---|---:|---:|---:|---:|---:|")

    for cond in CONDITIONS:
        agg = summary["aggregate"].get(cond, {})
        lines.append(
            "| {} | {} | {} | {} | {} | {} |".format(
                cond,
                agg.get("num_records", 0),
                fmt(agg.get("rgb_abs_mean")),
                fmt(agg.get("rgb_abs_max")),
                fmt(agg.get("depth_abs_mean")),
                fmt(agg.get("depth_abs_max")),
            )
        )

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- Low RGB/depth difference between `free` and hidden conditions supports the claim that hidden contact is not visually leaked.")
    lines.append("- Nonzero difference can still occur because the cable may settle slightly differently after hidden contact is applied.")
    lines.append("- This report is a Phase1 visual sanity check, not the final CCDA threshold audit.")
    lines.append("")

    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--task", default="hidden-contact-cable-line")
    parser.add_argument("--camera_index", type=int, default=0)
    parser.add_argument("--max_groups", type=int, default=20)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    data_root = root / "external" / "deformable-ravens" / "data" / args.task
    outdir = root / "reports" / "phase1_rgbd_check"
    outdir.mkdir(parents=True, exist_ok=True)

    groups = group_by_seed(data_root)
    if not groups:
        raise SystemExit(f"No groups found under {data_root}")

    records: List[Dict[str, Any]] = []
    for i, (seed, group) in enumerate(sorted(groups.items(), key=lambda kv: str(kv[0]))):
        if i >= args.max_groups:
            break
        recs = plot_seed(seed, group, data_root, args.camera_index, outdir)
        records.extend(recs)

    summary = {
        "task": args.task,
        "data_root": str(data_root),
        "camera_index": args.camera_index,
        "num_groups_visualized": min(len(groups), args.max_groups),
        "conditions": CONDITIONS,
        "records": records,
        "aggregate": summarize_records(records),
        "output_dir": str(outdir),
    }

    json_path = root / "reports" / "phase1_rgbd_check_summary.json"
    md_path = root / "reports" / "phase1_rgbd_check_report.md"

    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    write_report(md_path, summary)

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"[Phase1] wrote {json_path}")
    print(f"[Phase1] wrote {md_path}")
    print(f"[Phase1] wrote RGB-D check figures to {outdir}")

    if not records:
        raise SystemExit("[Phase1][FAIL] no RGB-D records generated")


if __name__ == "__main__":
    main()
