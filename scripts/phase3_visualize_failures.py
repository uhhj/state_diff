#!/usr/bin/env python3
import sys
from pathlib import Path as _Phase3Path
ROOT = _Phase3Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import csv
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/state_diff2")
    ap.add_argument("--predictions", default="reports/phase3_baseline_eval_predictions.csv")
    ap.add_argument("--outdir", default="reports/phase3_failure_visuals")
    args = ap.parse_args()
    root = Path(args.root)
    pred = root / args.predictions
    out = root / args.outdir
    out.mkdir(parents=True, exist_ok=True)
    if not pred.exists():
        print("[Phase3] no predictions yet")
        return
    rows = list(csv.DictReader(pred.open()))
    top = sorted(rows, key=lambda r: float(r.get("sample_wrong_branch_rate") or 0), reverse=True)[:20]
    (out / "top_wrong_branch_examples.csv").write_text("".join([]))
    with (out / "top_wrong_branch_examples.csv").open("w", newline="") as f:
        if top:
            w = csv.DictWriter(f, fieldnames=list(top[0].keys()))
            w.writeheader(); w.writerows(top)
    print("[Phase3] wrote", out / "top_wrong_branch_examples.csv")


if __name__ == "__main__":
    main()
