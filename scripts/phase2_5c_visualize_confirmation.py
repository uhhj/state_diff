#!/usr/bin/env python3
"""Create lightweight Phase2.5c confirmation bead-state PNGs."""

import argparse
import csv
import json
from pathlib import Path


def parse_bool(x):
    return str(x).strip().lower() in {"true", "1", "yes"}


def read_rows(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def score(row):
    try:
        frac = float(row.get("final_fraction") or 0.0)
    except Exception:
        frac = 0.0
    return (1 if parse_bool(row.get("success")) else 0, frac)


def choose(rows, condition, policy=None, want_success=None, best=True):
    cands = [r for r in rows if r.get("condition") == condition]
    if policy:
        cands = [r for r in cands if r.get("policy") == policy]
    if want_success is not None:
        cands = [r for r in cands if parse_bool(r.get("success")) == want_success]
    if not cands:
        return None
    return sorted(cands, key=score, reverse=best)[0]


def plot_row(row, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xy = json.loads(row.get("final_bead_xy") or "[]")
    if not xy:
        return False
    xs = [p[0] for p in xy]
    ys = [p[1] for p in xy]
    title = (
        f"{row.get('condition')} / {row.get('policy')} / seed {row.get('visible_seed')} / "
        f"success {row.get('success')}\n"
        f"fraction={row.get('final_fraction')} released={row.get('breakaway_released')} "
        f"release_step={row.get('breakaway_release_step')}"
    )
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(xs, ys, "o-", lw=1.5, ms=3)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=9)
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
    ap.add_argument("--trials_csv", required=True)
    ap.add_argument("--out_dir", default="/data/state_diff2/reports/phase2_5c_confirmation_visuals")
    args = ap.parse_args()
    rows = read_rows(args.trials_csv)
    choices = [
        ("free_nominal_success", choose(rows, "free", "nominal", True, True)),
        ("free_guided_search_success", choose(rows, "free", "guided_search", True, True)),
        ("hidden_pin_oracle_best_failure", choose([r for r in rows if r.get("policy", "").startswith("oracle")], "hidden_pin", None, False, False)),
        ("hidden_breakaway_nominal_failure", choose(rows, "hidden_breakaway_pin", "nominal", False, False)),
        ("hidden_breakaway_oracle_breakaway_success", choose(rows, "hidden_breakaway_pin", "oracle_breakaway_then_place", True, True)),
        ("hidden_breakaway_oracle_breakaway_failure", choose(rows, "hidden_breakaway_pin", "oracle_breakaway_then_place", False, False)),
    ]
    written = []
    out_dir = Path(args.out_dir)
    for name, row in choices:
        if row is None:
            continue
        out = out_dir / f"{name}.png"
        if plot_row(row, out):
            written.append(str(out))
    print(json.dumps({"written": written}, indent=2))


if __name__ == "__main__":
    main()
