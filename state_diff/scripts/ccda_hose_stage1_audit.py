from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from state_diff.env.ccda_hose.audit import (
    AuditThresholds,
    compute_pair_metrics,
    load_npz_dataset,
    plot_future_xy_overlay,
    plot_pair_metric_bars,
    write_audit_report,
    write_pair_metrics_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--history-steps", type=int, default=8)
    parser.add_argument("--future-steps", type=int, default=32)
    parser.add_argument("--max-pairs-per-type", type=int, default=5000)

    parser.add_argument("--tau-vis", type=float, default=0.020)
    parser.add_argument("--tau-prop", type=float, default=0.010)
    parser.add_argument("--tau-act", type=float, default=0.003)
    parser.add_argument("--tau-contact", type=float, default=0.200)
    parser.add_argument("--tau-future", type=float, default=0.025)

    args = parser.parse_args()

    data_path = Path(args.data)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_npz_dataset(data_path)
    thresholds = AuditThresholds(
        tau_vis=args.tau_vis,
        tau_prop=args.tau_prop,
        tau_act=args.tau_act,
        tau_contact=args.tau_contact,
        tau_future=args.tau_future,
    )
    rows, summary = compute_pair_metrics(
        data=data,
        history_steps=args.history_steps,
        future_steps=args.future_steps,
        thresholds=thresholds,
        max_pairs_per_type=args.max_pairs_per_type,
    )

    write_pair_metrics_csv(rows, out_dir / "pair_metrics.csv")
    write_audit_report(summary, rows, out_dir, data_path)
    plot_pair_metric_bars(summary, out_dir)
    plot_future_xy_overlay(data, out_dir, future_steps=args.future_steps)

    print(json.dumps(summary, indent=2))
    print(f"Saved pair metrics to {out_dir / 'pair_metrics.csv'}")
    print(f"Saved audit report to {out_dir / 'ccda_audit_report.md'}")


if __name__ == "__main__":
    main()
