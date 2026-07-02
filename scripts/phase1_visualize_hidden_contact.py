#!/usr/bin/env python3
"""Visualize Phase1 hidden-contact cable paired rollouts."""

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CONDITIONS = ["free", "hidden_pin", "hidden_high_friction"]


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def get_extras(info: Dict) -> Dict:
    if isinstance(info, dict):
        return info.get("extras", {})
    return {}


def get_bead_positions(info: Dict) -> Optional[np.ndarray]:
    extras = get_extras(info)
    if "bead_positions" in extras:
        arr = np.asarray(extras["bead_positions"], dtype=np.float32)
        if arr.ndim == 2 and arr.shape[1] >= 3:
            return arr[:, :3]

    if "bead_ids" in extras:
        bead_ids = extras["bead_ids"]
        pos = []
        for bid in bead_ids:
            if bid in info:
                pos.append(info[bid][0])
        if pos:
            return np.asarray(pos, dtype=np.float32)

    int_keys = sorted([k for k in info.keys() if isinstance(k, int)])
    pos = []
    for k in int_keys:
        v = info[k]
        if isinstance(v, tuple) and len(v) >= 1 and len(v[0]) == 3:
            pos.append(v[0])
    if pos:
        return np.asarray(pos, dtype=np.float32)
    return None


def parse_len(path: Path) -> int:
    return int(path.stem.split("-")[-1])


def find_episodes(data_root: Path) -> Dict[str, Dict[str, Path]]:
    by_seed: Dict[str, Dict[str, Path]] = {}
    for cond in CONDITIONS:
        color_dir = data_root / cond / "color"
        if not color_dir.exists():
            continue

        for color_file in sorted(color_dir.glob("*.pkl")):
            fname = color_file.name
            info_path = data_root / cond / "info" / fname
            if not info_path.exists():
                continue
            info_list = load_pickle(info_path)
            if not info_list:
                continue
            seed = str(get_extras(info_list[0]).get("ccda_visible_seed", color_file.stem))
            by_seed.setdefault(seed, {})[cond] = color_file
    return by_seed


def load_traj(base: Path, fname: str) -> List[np.ndarray]:
    info_list = load_pickle(base / "info" / fname)
    last_info = load_pickle(base / "last_info" / fname)

    traj = []
    for info in info_list:
        b = get_bead_positions(info)
        if b is not None:
            traj.append(b)
    b_last = get_bead_positions(last_info)
    if b_last is not None:
        traj.append(b_last)
    return traj


def plot_group(seed: str, group: Dict[str, Path], data_root: Path, outdir: Path) -> None:
    fig, axes = plt.subplots(1, len(CONDITIONS), figsize=(4 * len(CONDITIONS), 4))
    if len(CONDITIONS) == 1:
        axes = [axes]

    for ax, cond in zip(axes, CONDITIONS):
        ax.set_title(cond)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linewidth=0.3)

        if cond not in group:
            ax.text(0.5, 0.5, "missing", transform=ax.transAxes, ha="center")
            continue

        fname = group[cond].name
        base = data_root / cond
        traj = load_traj(base, fname)

        if not traj:
            ax.text(0.5, 0.5, "no bead states", transform=ax.transAxes, ha="center")
            continue

        first = traj[0]
        last = traj[-1]

        ax.plot(first[:, 0], first[:, 1], marker="o", linewidth=1, label="initial")
        ax.plot(last[:, 0], last[:, 1], marker="x", linewidth=1, label="final")

        for step in traj[1:-1]:
            ax.plot(step[:, 0], step[:, 1], alpha=0.20, linewidth=0.8)

        ax.legend(loc="best", fontsize=8)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig.suptitle("Phase1 hidden-contact cable visible_seed={}".format(seed))
    fig.tight_layout()
    out = outdir / "phase1_visible_seed_{}.png".format(seed)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--task", default="hidden-contact-cable-line")
    parser.add_argument("--max_groups", type=int, default=5)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    data_root = root / "external" / "deformable-ravens" / "data" / args.task
    outdir = root / "reports" / "phase1_hidden_contact_visuals"
    outdir.mkdir(parents=True, exist_ok=True)

    groups = find_episodes(data_root)

    n = 0
    for seed, group in sorted(groups.items()):
        if n >= args.max_groups:
            break
        plot_group(seed, group, data_root, outdir)
        n += 1

    print("[Phase1] wrote {} visualization groups to {}".format(n, outdir))
    if n == 0:
        raise SystemExit("[Phase1][FAIL] no groups visualized")


if __name__ == "__main__":
    main()
