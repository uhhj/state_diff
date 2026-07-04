#!/usr/bin/env python3
"""Create lightweight bead trajectory/final-state PNGs for Phase2.5."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def as_bool(v):
    return str(v).lower() in {"true", "1", "yes"}


def read_rows(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def row_score(row):
    return (1 if as_bool(row.get("success")) else 0, float(row.get("final_fraction") or 0))


def choose(rows, condition, policy=None, want_success=None, best=True):
    cands = [r for r in rows if r.get("condition") == condition]
    if policy:
        cands = [r for r in cands if r.get("policy") == policy]
    if want_success is not None:
        cands = [r for r in cands if as_bool(r.get("success")) == want_success]
    if not cands:
        return None
    return sorted(cands, key=row_score, reverse=best)[0]


def plot_row(row, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xy = json.loads(row.get("final_bead_xy") or "[]")
    if not xy:
        return False
    xs = [p[0] for p in xy]
    ys = [p[1] for p in xy]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(xs, ys, "o-", lw=1.5, ms=3)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("{} / {} / seed {} / success {}".format(row.get("condition"), row.get("policy"), row.get("visible_seed"), row.get("success")))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.25)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/state_diff2")
    ap.add_argument("--trials_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    rows = read_rows(args.trials_csv)
    root = Path(args.root)
    summary_path = root / "reports/phase2_5_recoverability_summary.json"
    selected = None
    if summary_path.exists():
        selected = json.loads(summary_path.read_text()).get("selected_recoverable_condition")
    choices = [
        ("free_nominal_success", choose(rows, "free", "nominal", True, True)),
        ("hidden_pin_oracle_failure", choose(rows, "hidden_pin", "oracle_pull", False, False)),
        ("hidden_high_friction_nominal", choose(rows, "hidden_high_friction", "nominal", None, True)),
    ]
    if selected:
        choices.append(("selected_recoverable_oracle_success", choose(rows, selected, "oracle_pull", True, True) or choose(rows, selected, "oracle_regrasp", True, True) or choose(rows, selected, "oracle_wiggle", True, True)))
        choices.append(("selected_recoverable_nominal_failure", choose(rows, selected, "nominal", False, False)))
    written = []
    out_dir = Path(args.out_dir)
    for name, row in choices:
        if row is None:
            continue
        out = out_dir / (name + ".png")
        if plot_row(row, out):
            written.append(str(out))
    print(json.dumps({"written": written}, indent=2))


if __name__ == "__main__":
    main()
