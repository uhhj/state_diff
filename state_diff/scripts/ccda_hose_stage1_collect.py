from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from state_diff.env.ccda_hose.config import FREE_INSERT, RIGHT_HIDDEN_JAM, HoseEnvConfig
from state_diff.env.ccda_hose.audit import (
    AuditThresholds,
    compute_pair_metrics,
    plot_future_xy_overlay,
    plot_pair_metric_bars,
    write_audit_report,
    write_pair_metrics_csv,
)
from state_diff.env.ccda_hose.env import scripted_rollout, summarize_rollouts


def _save_npz(rollouts: List[Dict[str, object]], out_path: Path) -> None:
    traces = [r["trace"] for r in rollouts]
    keys = list(traces[0].keys())
    arrays = {}
    for k in keys:
        arrays[k] = np.stack([t[k] for t in traces], axis=0)
    arrays["condition"] = np.array([r["condition"] for r in rollouts])
    arrays["seed"] = np.array([int(r["seed"]) for r in rollouts])
    arrays["audit_index"] = np.array([int(r["audit_index"]) for r in rollouts])
    arrays["final_success"] = np.array([float(r["final_success"]) for r in rollouts])
    arrays["final_branch"] = np.array([str(r["final_branch"]) for r in rollouts])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **arrays)


def _rollouts_to_npz_like_dict(rollouts: List[Dict[str, object]]) -> Dict[str, np.ndarray]:
    traces = [r["trace"] for r in rollouts]
    keys = list(traces[0].keys())
    arrays = {}
    for k in keys:
        arrays[k] = np.stack([t[k] for t in traces], axis=0)
    arrays["condition"] = np.array([r["condition"] for r in rollouts])
    arrays["seed"] = np.array([int(r["seed"]) for r in rollouts])
    arrays["audit_index"] = np.array([int(r["audit_index"]) for r in rollouts])
    arrays["final_success"] = np.array([float(r["final_success"]) for r in rollouts])
    arrays["final_branch"] = np.array([str(r["final_branch"]) for r in rollouts])
    return arrays


def _plot_metric(rollouts: List[Dict[str, object]], key: str, ylabel: str, out_path: Path) -> None:
    plt.figure(figsize=(8, 4.5))
    used_labels = set()
    for r in rollouts:
        tr = r["trace"]
        y = tr[key].reshape(len(tr[key]), -1)[:, 0]
        label = r["condition"] if r["condition"] not in used_labels else None
        used_labels.add(r["condition"])
        plt.plot(y, alpha=0.35, label=label)
        audit_idx = int(r["audit_index"])
        plt.axvline(audit_idx, linestyle="--", linewidth=0.7, alpha=0.18)
    plt.xlabel("Environment step")
    plt.ylabel(ylabel)
    plt.title(ylabel)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def _write_markdown_report(summary: Dict[str, object], report_dir: Path, dataset_path: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    by = summary["by_condition"]

    def cond_line(cond: str) -> str:
        s = by.get(cond, {})
        return (
            f"| `{cond}` | {s.get('n', 0)} | {s.get('success_rate', 0):.3f} | "
            f"{s.get('mean_final_insert_depth', 0):.4f} | "
            f"{s.get('mean_final_lateral_offset', 0):.4f} | "
            f"{s.get('mean_final_max_curvature', 0):.2f} | "
            f"{s.get('mean_max_jam_contact_force', 0):.3f} | "
            f"{s.get('mean_max_lateral_contact_force', 0):.3f} |"
        )

    md = f"""# Stage 1 Report: Hidden Lateral-Jam Hose Insertion

## Goal

This report summarizes the MuJoCo stage-1 environment for CCDA auditing. The environment creates two visually similar conditions at audit time:

- `free_insert`: no hidden side contact.
- `right_hidden_jam`: an occluded lateral friction block inside the socket creates hidden contact.

The stage-1 goal is not policy training. It is to verify that the environment can generate visible/proprio/action similarity at audit time, but different hidden contact, different future hose states, and different success outcomes.

## Dataset

Saved debug dataset:

`{dataset_path}`

## Summary

| Condition | N | Success rate | Mean final insertion depth | Mean final lateral offset | Mean final max curvature | Mean max jam force | Mean max lateral force |
|---|---:|---:|---:|---:|---:|---:|---:|
{cond_line(FREE_INSERT)}
{cond_line(RIGHT_HIDDEN_JAM)}

## Branch counts

```json
{json.dumps({k: v.get("branch_counts", {}) for k, v in by.items()}, indent=2)}
```

## Figures

* `insertion_depth.png`
* `lateral_offset.png`
* `max_curvature.png`
* `jam_contact_force.png`
* `lateral_contact_force.png`

## Go / No-Go criteria for Stage 2

Proceed to data collection if:

1. `free_insert` success rate is high.
2. `right_hidden_jam` produces lateral jam, S-buckle, or half-insertion failures.
3. The hidden-jam condition has higher lateral/jam contact force.
4. Future state divergence is visible after the audit time.
5. Audit-time visual state remains similar enough for CCDA pair mining.

If `right_hidden_jam` always completely blocks insertion, reduce `jam_friction` or shrink `jam_block_size`.
If `free_insert` fails often, enlarge `socket_half_width`, reduce wall friction, or increase hose stiffness.
"""
    (report_dir / "stage1_report.md").write_text(md, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes-per-condition", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="data/ccda_hose_stage1/debug_v1.npz")
    parser.add_argument("--report-dir", type=str, default="reports/ccda_hose_stage1/debug_v1")
    args = parser.parse_args()

    cfg = HoseEnvConfig()
    rollouts = []
    for i in range(args.episodes_per_condition):
        rollouts.append(scripted_rollout(FREE_INSERT, seed=args.seed + i, config=cfg, record_frames=False))
        rollouts.append(scripted_rollout(RIGHT_HIDDEN_JAM, seed=args.seed + 10000 + i, config=cfg, record_frames=False))

    out_path = Path(args.out)
    report_dir = Path(args.report_dir)
    _save_npz(rollouts, out_path)

    summary = summarize_rollouts(rollouts)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    _plot_metric(rollouts, "insertion_depth", "Insertion depth", report_dir / "insertion_depth.png")
    _plot_metric(rollouts, "lateral_offset", "Lateral offset", report_dir / "lateral_offset.png")
    _plot_metric(rollouts, "max_curvature", "Max curvature", report_dir / "max_curvature.png")

    for col, name, ylabel in [
        (2, "jam_contact_force.png", "Jam contact force"),
        (3, "lateral_contact_force.png", "Lateral contact force"),
    ]:
        plt.figure(figsize=(8, 4.5))
        used_labels = set()
        for r in rollouts:
            tr = r["trace"]
            y = tr["privileged_contact"][:, col]
            label = r["condition"] if r["condition"] not in used_labels else None
            used_labels.add(r["condition"])
            plt.plot(y, alpha=0.35, label=label)
            plt.axvline(int(r["audit_index"]), linestyle="--", linewidth=0.7, alpha=0.18)
        plt.xlabel("Environment step")
        plt.ylabel(ylabel)
        plt.title(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(report_dir / name, dpi=160)
        plt.close()

    _write_markdown_report(summary, report_dir, out_path)

    data_like = _rollouts_to_npz_like_dict(rollouts)
    thresholds = AuditThresholds()
    pair_rows, pair_summary = compute_pair_metrics(
        data_like,
        history_steps=8,
        future_steps=32,
        thresholds=thresholds,
        max_pairs_per_type=5000,
    )
    audit_dir = report_dir / "ccda_pair_audit"
    write_pair_metrics_csv(pair_rows, audit_dir / "pair_metrics.csv")
    write_audit_report(pair_summary, pair_rows, audit_dir, out_path)
    plot_pair_metric_bars(pair_summary, audit_dir)
    plot_future_xy_overlay(data_like, audit_dir, future_steps=32)

    print(json.dumps(summary["by_condition"], indent=2))
    print(f"Saved dataset to {out_path}")
    print(f"Saved report to {report_dir / 'stage1_report.md'}")
    print(f"Saved CCDA pair audit to {audit_dir / 'ccda_audit_report.md'}")


if __name__ == "__main__":
    main()
