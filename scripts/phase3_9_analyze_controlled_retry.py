#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

CONDITIONS = ["free", "hidden_breakaway_pin", "hidden_high_friction"]
ACTIONS = ["gt_reference", "old_idm_gt_future", "phase39_idm_gt_future"]


def sf(x: Any) -> float:
    try:
        if x is None or x == "":
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def pb(x: Any) -> bool:
    return str(x).strip().lower() in {"1", "true", "yes", "1.0", "pass", "success"}


def mean(xs: List[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def read_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def rows_for(rows: List[Dict[str, str]], condition: str, action: str) -> List[Dict[str, str]]:
    return [r for r in rows if r.get("condition") == condition and r.get("action_source") == action and r.get("baseline") == "state_action"]


def delta(rows: List[Dict[str, str]], condition: str, action: str) -> float:
    return mean([sf(r.get("exec_delta_final_fraction")) for r in rows_for(rows, condition, action) if r.get("status") == "ok"])


def success(rows: List[Dict[str, str]], condition: str, action: str) -> float:
    return mean([1.0 if pb(r.get("exec_success")) else 0.0 for r in rows_for(rows, condition, action) if r.get("status") == "ok"])


def ok_count(rows: List[Dict[str, str]], condition: str, action: str) -> int:
    return sum(1 for r in rows_for(rows, condition, action) if r.get("status") == "ok")


def group_table(rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    by: Dict[Tuple[str, str, str], List[Dict[str, str]]] = defaultdict(list)
    for r in rows:
        by[(r.get("baseline", ""), r.get("action_source", ""), r.get("condition", ""))].append(r)

    table = []
    for (baseline, action, condition), group in sorted(by.items()):
        ok = [r for r in group if r.get("status") == "ok"]
        table.append({
            "baseline": baseline,
            "action_source": action,
            "condition": condition,
            "rows": len(group),
            "ok_rows": len(ok),
            "timeout_rows": sum(1 for r in group if "timeout" in str(r.get("status", ""))),
            "failed_rows": sum(1 for r in group if r.get("status") not in {"ok", ""} and "timeout" not in str(r.get("status", ""))),
            "success_rate": mean([1.0 if pb(r.get("exec_success")) else 0.0 for r in ok]),
            "mean_delta_final_fraction": mean([sf(r.get("exec_delta_final_fraction")) for r in ok]),
            "mean_final_fraction_after": mean([sf(r.get("exec_final_fraction_after")) for r in ok]),
            "mean_prefix_mae": mean([sf(r.get("exec_prefix_state_mae")) for r in ok]),
            "mean_action_mae_to_gt": mean([sf(r.get("action_mae_to_gt")) for r in ok]),
            "mean_pose0_xy_dist_to_gt": mean([sf(r.get("pose0_xy_dist_to_gt")) for r in ok]),
            "mean_pose1_xy_dist_to_gt": mean([sf(r.get("pose1_xy_dist_to_gt")) for r in ok]),
            "mean_pull_angle_deg_to_gt": mean([sf(r.get("pull_angle_deg_to_gt")) for r in ok]),
            "exec_failure_count": sum(1 for r in group if r.get("exec_failure_reason")),
        })
    return table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--trials_csv", default="reports/phase3_9_geometry_idm_controlled_retry_trials.csv")
    parser.add_argument("--progress_json", default="reports/phase3_9_geometry_idm_controlled_retry_progress.json")
    parser.add_argument("--raw_summary", default="reports/phase3_9_geometry_idm_controlled_retry_raw_summary.json")
    parser.add_argument("--train_summary", default="reports/phase3_9_geometry_idm_train_summary.json")
    parser.add_argument("--out_json", default="reports/phase3_9_geometry_idm_controlled_retry_summary.json")
    parser.add_argument("--out_md", default="reports/phase3_9_geometry_idm_controlled_retry_report.md")
    parser.add_argument("--improve_threshold", type=float, default=0.05)
    parser.add_argument("--min_ok_rows_per_condition_action", type=int, default=1)
    args = parser.parse_args()

    root = Path(args.root)
    rows = read_rows(root / args.trials_csv)
    progress = json.loads((root / args.progress_json).read_text()) if (root / args.progress_json).exists() else {}
    raw = json.loads((root / args.raw_summary).read_text()) if (root / args.raw_summary).exists() else {}
    train = json.loads((root / args.train_summary).read_text()) if (root / args.train_summary).exists() else {}

    issues: List[Dict[str, Any]] = []

    def issue(level: str, name: str, detail: Any) -> None:
        issues.append({"level": level, "name": name, "detail": str(detail)})

    if not rows:
        issue("FAIL", "no_rows", "No controlled retry rows found.")

    ok_rows = [r for r in rows if r.get("status") == "ok"]
    timeout_rows = [r for r in rows if "timeout" in str(r.get("status", ""))]
    failed_rows = [r for r in rows if r.get("status") not in {"ok", ""} and "timeout" not in str(r.get("status", ""))]

    if timeout_rows:
        issue("WARN", "row_timeouts_present", len(timeout_rows))
    if failed_rows:
        issue("WARN", "row_failures_present", len(failed_rows))

    missing = []
    for cond in CONDITIONS:
        for action in ACTIONS:
            n_ok = ok_count(rows, cond, action)
            if n_ok < args.min_ok_rows_per_condition_action:
                missing.append(f"{cond}/{action}: ok={n_ok}")
    if missing:
        issue("FAIL", "missing_required_condition_action_rows", missing)

    condition_table: List[Dict[str, Any]] = []
    improved_conditions: List[str] = []
    primary_improved = False

    for cond in CONDITIONS:
        gt = delta(rows, cond, "gt_reference")
        old = delta(rows, cond, "old_idm_gt_future")
        new = delta(rows, cond, "phase39_idm_gt_future")
        improvement = new - old if math.isfinite(new) and math.isfinite(old) else float("nan")
        ratio_to_gt = new / gt if math.isfinite(new) and math.isfinite(gt) and gt > 1e-9 else float("nan")
        improved = math.isfinite(improvement) and improvement > args.improve_threshold and math.isfinite(new) and new > args.improve_threshold
        if improved:
            improved_conditions.append(cond)
        if cond == "hidden_breakaway_pin" and improved:
            primary_improved = True

        condition_table.append({
            "condition": cond,
            "gt_reference_delta": gt,
            "old_idm_delta": old,
            "phase39_idm_delta": new,
            "improvement": improvement,
            "ratio_to_gt": ratio_to_gt,
            "improved": improved,
            "gt_success": success(rows, cond, "gt_reference"),
            "old_success": success(rows, cond, "old_idm_gt_future"),
            "phase39_success": success(rows, cond, "phase39_idm_gt_future"),
        })

    table = group_table(rows)

    if primary_improved and len(improved_conditions) >= 2:
        issue("WARN", "phase39_geometry_idm_repair_supported", improved_conditions)
        root_cause = "phase39_geometry_idm_repair_supported"
    elif primary_improved:
        issue("WARN", "phase39_primary_branch_partial_improvement", improved_conditions)
        root_cause = "phase39_primary_branch_partial_improvement"
    elif len(improved_conditions) >= 2:
        issue("WARN", "phase39_nonprimary_improvement_only", improved_conditions)
        root_cause = "phase39_nonprimary_improvement_only"
    else:
        issue("FAIL", "phase39_repair_not_supported", improved_conditions)
        root_cause = "phase39_repair_not_supported"

    if progress and progress.get("status") != "completed":
        issue("WARN", "controlled_retry_not_fully_completed", progress.get("status"))

    try:
        completed_fraction = float(progress.get("completed", 0)) / float(progress.get("total_specs", 1))
    except Exception:
        completed_fraction = None

    has_fail = any(i["level"] == "FAIL" for i in issues)
    has_warn = any(i["level"] == "WARN" for i in issues)
    verdict = "FAIL" if has_fail else ("WARN" if has_warn else "PASS")

    payload = {
        "verdict": verdict,
        "root_cause": root_cause,
        "num_rows": len(rows),
        "num_ok_rows": len(ok_rows),
        "num_timeout_rows": len(timeout_rows),
        "num_failed_rows": len(failed_rows),
        "completed_fraction": completed_fraction,
        "improved_conditions": improved_conditions,
        "primary_improved": primary_improved,
        "condition_table": condition_table,
        "table": table,
        "issues": issues,
        "progress": progress,
        "raw_summary": raw,
        "train_summary": {
            "inverse_dynamics_path": train.get("inverse_dynamics_path"),
            "metrics": train.get("metrics"),
        },
        "important_note": "This is one-step matched-prefix diagnostic only, not Phase4/CPS and not paper-level rollout evidence.",
        "recommendation": (
            "Do not enter Phase4/CPS. Next run a slightly larger Phase3.9b controlled retry or repair loss ablation before rollout."
            if root_cause in {"phase39_geometry_idm_repair_supported", "phase39_primary_branch_partial_improvement"}
            else
            "Do not enter Phase4/CPS. Geometry-aware IDM repair did not clear the one-step blocker; inspect loss weights and primitive pick/place geometry."
        ),
    }

    out_json = root / args.out_json
    out_md = root / args.out_md
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True))

    lines = [
        "# Phase3.9 Geometry-Aware IDM Controlled Retry Report",
        "",
        "## Verdict",
        "",
        f"- Verdict: `{verdict}`",
        f"- Root cause: `{root_cause}`",
        f"- Rows: `{len(rows)}`",
        f"- OK rows: `{len(ok_rows)}`",
        f"- Timeout rows: `{len(timeout_rows)}`",
        f"- Failed rows: `{len(failed_rows)}`",
        f"- Progress status: `{progress.get('status', 'missing')}`",
        f"- Improved conditions: `{improved_conditions}`",
        f"- Primary improved: `{primary_improved}`",
        "",
        "## Per-Condition Retry",
        "",
        "| Condition | GT Δ | Old IDM Δ | Phase3.9 IDM Δ | Improvement | Ratio to GT | Improved |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in condition_table:
        lines.append(
            f"| `{item['condition']}` | {item['gt_reference_delta']:.4f} | {item['old_idm_delta']:.4f} | "
            f"{item['phase39_idm_delta']:.4f} | {item['improvement']:.4f} | {item['ratio_to_gt']:.4f} | `{item['improved']}` |"
        )

    lines += [
        "",
        "## Summary Table",
        "",
        "| Baseline | Action | Condition | Rows | OK | Timeout | Success | Δ final_fraction | Final fraction | Prefix MAE | Action MAE | Pose0 dist | Pose1 dist | Pull angle | Failures |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in table:
        lines.append(
            f"| `{item['baseline']}` | `{item['action_source']}` | `{item['condition']}` | "
            f"{item['rows']} | {item['ok_rows']} | {item['timeout_rows']} | {item['success_rate']:.3f} | "
            f"{item['mean_delta_final_fraction']:.4f} | {item['mean_final_fraction_after']:.4f} | "
            f"{item['mean_prefix_mae']:.4f} | {item['mean_action_mae_to_gt']:.4f} | "
            f"{item['mean_pose0_xy_dist_to_gt']:.4f} | {item['mean_pose1_xy_dist_to_gt']:.4f} | "
            f"{item['mean_pull_angle_deg_to_gt']:.2f} | {item['exec_failure_count']} |"
        )

    lines += [
        "",
        "## Issues",
        "",
        "| Level | Name | Detail |",
        "|---|---|---|",
    ]
    if issues:
        for item in issues:
            lines.append(f"| `{item['level']}` | `{item['name']}` | {str(item['detail']).replace('|', '/')} |")
    else:
        lines.append("| `PASS` | `none` | No issues found. |")

    lines += [
        "",
        "## Interpretation",
        "",
        "- This is one-step matched-prefix diagnostic only.",
        "- It does not run Phase4 or CPS.",
        "- It does not prove deployable closed-loop policy success.",
        "- If Phase3.9 IDM improves hidden_breakaway_pin, the next step is still a controlled retry/ablation, not CPS.",
    ]

    out_md.write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))

    if verdict == "FAIL":
        raise SystemExit("[Phase3.9][FAIL] geometry-aware IDM retry did not support repair")


if __name__ == "__main__":
    main()
