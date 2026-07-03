#!/usr/bin/env python3
"""Visualize representative Phase2 CCDA paired examples."""

import argparse
import csv
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def get_extras(info: Dict) -> Dict:
    if isinstance(info, dict):
        ex = info.get("extras", {})
        if isinstance(ex, dict):
            return ex
    return {}


def get_bead_positions(info: Dict) -> Optional[np.ndarray]:
    ex = get_extras(info)

    if "bead_positions" in ex:
        arr = np.asarray(ex["bead_positions"], dtype=np.float32)
        if arr.ndim == 2 and arr.shape[1] >= 3:
            return arr[:, :3]

    if "bead_ids" in ex:
        pos = []
        for bid in ex["bead_ids"]:
            if bid in info:
                v = info[bid]
                if isinstance(v, tuple) and len(v) >= 1:
                    pos.append(v[0])
        if pos:
            return np.asarray(pos, dtype=np.float32)[:, :3]

    int_keys = sorted(k for k in info.keys() if isinstance(k, int))
    pos = []
    for k in int_keys:
        v = info[k]
        if isinstance(v, tuple) and len(v) >= 1 and len(v[0]) == 3:
            pos.append(v[0])
    if pos:
        return np.asarray(pos, dtype=np.float32)[:, :3]

    return None


def read_records(path: Path) -> List[Dict]:
    with path.open("r", newline="") as f:
        rows = list(csv.DictReader(f))

    def to_bool(x):
        return str(x).lower() in {"true", "1", "yes"}

    for r in rows:
        r["ccda_success_bool"] = to_bool(r.get("ccda_success"))
        try:
            r["final_chamfer_float"] = float(r.get("final_chamfer", "nan"))
        except Exception:
            r["final_chamfer_float"] = float("nan")

    rows = [r for r in rows if r["ccda_success_bool"]]
    rows = sorted(rows, key=lambda r: r["final_chamfer_float"], reverse=True)
    return rows


def load_traj(data_root: Path, cond: str, fname: str) -> List[np.ndarray]:
    cond_dir = data_root / cond
    info_list = load_pickle(cond_dir / "info" / fname)
    last_info = load_pickle(cond_dir / "last_info" / fname)

    traj = []
    for info in info_list:
        b = get_bead_positions(info)
        if b is not None:
            traj.append(b)

    b = get_bead_positions(last_info)
    if b is not None:
        traj.append(b)

    return traj


def load_first_rgbd(data_root: Path, cond: str, fname: str):
    color = load_pickle(data_root / cond / "color" / fname)
    depth = load_pickle(data_root / cond / "depth" / fname)
    return color, depth


def first_image(arr):
    arr = np.asarray(arr)

    # Common DeformableRavens RGB layouts:
    # T x Cams x H x W x 3, Cams x H x W x 3, T x H x W x 3, or H x W x 3.
    # Common depth layouts:
    # T x Cams x H x W, Cams x H x W, T x H x W, or H x W.
    if arr.ndim == 5:
        return arr[0, 0]
    if arr.ndim == 4:
        if arr.shape[-1] == 3:
            return arr[0]
        return arr[0, 0]
    if arr.ndim == 3:
        if arr.shape[-1] == 3:
            return arr
        return arr[0]
    if arr.ndim == 2:
        return arr

    return np.squeeze(arr)


def plot_record(data_root: Path, row: Dict, outdir: Path, idx: int) -> None:
    seed = row["visible_seed"]
    ca = row["condition_a"]
    cb = row["condition_b"]
    fa = row["file_a"]
    fb = row["file_b"]

    traj_a = load_traj(data_root, ca, fa)
    traj_b = load_traj(data_root, cb, fb)

    color_a, depth_a = load_first_rgbd(data_root, ca, fa)
    color_b, depth_b = load_first_rgbd(data_root, cb, fb)

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 3)

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[1, 2])

    # Trajectory overlay.
    ax0.set_title(f"Bead trajectory: {ca}")
    plot_traj(ax0, traj_a)

    ax1.set_title(f"Bead trajectory: {cb}")
    plot_traj(ax1, traj_b)

    ax2.set_title("Final overlay")
    if traj_a:
        ax2.plot(traj_a[-1][:, 0], traj_a[-1][:, 1], marker="o", linewidth=1, label=ca)
    if traj_b:
        ax2.plot(traj_b[-1][:, 0], traj_b[-1][:, 1], marker="x", linewidth=1, label=cb)
    ax2.set_aspect("equal", adjustable="box")
    ax2.grid(True, linewidth=0.3)
    ax2.legend(fontsize=8)

    # RGB / depth leakage check views.
    img_a = first_image(color_a)
    img_b = first_image(color_b)

    if img_a.ndim == 3:
        ax3.imshow(img_a.astype(np.uint8) if img_a.max() > 1.5 else img_a)
    else:
        ax3.imshow(img_a)
    ax3.set_title(f"Initial RGB: {ca}")
    ax3.axis("off")

    if img_b.ndim == 3:
        ax4.imshow(img_b.astype(np.uint8) if img_b.max() > 1.5 else img_b)
    else:
        ax4.imshow(img_b)
    ax4.set_title(f"Initial RGB: {cb}")
    ax4.axis("off")

    da = np.asarray(first_image(depth_a), dtype=np.float32)
    db = np.asarray(first_image(depth_b), dtype=np.float32)
    if da.shape == db.shape:
        diff = np.abs(da - db)
        ax5.imshow(diff)
        ax5.set_title("Initial depth abs diff")
    else:
        ax5.text(0.5, 0.5, "Depth shape mismatch", ha="center")
    ax5.axis("off")

    fig.suptitle(
        "Phase2 CCDA example seed={} pair={} vs {} final_chamfer={} success=({}, {})".format(
            seed,
            ca,
            cb,
            row.get("final_chamfer"),
            row.get("success_a"),
            row.get("success_b"),
        )
    )
    fig.tight_layout()

    safe_seed = str(seed).replace("/", "_")
    out = outdir / f"phase2_ccda_{idx:03d}_seed_{safe_seed}_{ca}_vs_{cb}.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_traj(ax, traj: List[np.ndarray]) -> None:
    if not traj:
        ax.text(0.5, 0.5, "missing traj", transform=ax.transAxes, ha="center")
        return

    first = traj[0]
    last = traj[-1]

    ax.plot(first[:, 0], first[:, 1], marker="o", linewidth=1, label="initial")
    for mid in traj[1:-1]:
        ax.plot(mid[:, 0], mid[:, 1], alpha=0.18, linewidth=0.7)
    ax.plot(last[:, 0], last[:, 1], marker="x", linewidth=1, label="final")

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.3)
    ax.legend(fontsize=8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--task", default="hidden-contact-cable-line")
    parser.add_argument("--max_examples", type=int, default=8)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    data_root = Path(args.data_root).resolve()
    records_csv = root / "reports" / "phase2_ccda_records.csv"
    outdir = root / "reports" / "phase2_ccda_visuals"
    outdir.mkdir(parents=True, exist_ok=True)

    if not records_csv.exists():
        raise FileNotFoundError(records_csv)

    rows = read_records(records_csv)
    if not rows:
        raise SystemExit("[Phase2][WARN] no ccda_success records to visualize")

    for i, row in enumerate(rows[: args.max_examples]):
        plot_record(data_root, row, outdir, i)

    print(f"[Phase2] wrote {min(len(rows), args.max_examples)} examples to {outdir}")


if __name__ == "__main__":
    main()
