#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

CONFIRM_CONDITIONS = ["free", "hidden_breakaway_pin", "hidden_high_friction"]
REQUIRED_VARIANTS = [
    "gt_reference",
    "idm_original",
    "idm_gt_pose0_xy",
    "idm_gt_pose1_xy",
    "idm_gt_pose0_pose1_xy",
]


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


def rows_for(rows: List[Dict[str, str]], condition: str, variant: str) -> List[Dict[str, str]]:
    return [
        r for r in rows
        if r.get("condition") == condition
        and r.get("repair_variant") == variant
        and r.get("source_action") == "idm_gt_future"
        and r.get("baseline") == "state_action"
    ]


def delta(rows: List[Dict[str, str]], condition: str, variant: str) -> float:
    vals = [
        sf(r.get("exec_delta_final_fraction"))
        for r in rows_for(rows, condition, variant)
        if r.get("status") == "ok"
    ]
    return mean(vals)


def success_rate(rows: List[Dict[str, str]], condition: str, variant: str) -> float:
    vals = [
        1.0 if pb(r.get("exec_success")) else 0.0
        for r in rows_for(rows, condition, variant)
        if r.get("status") == "ok"
    ]
    return mean(vals)


def metric_mean(rows: List[Dict[str, str]], condition: str, variant: str, key: str) -> float:
    vals = [sf(r.get(key)) for r in rows_for(rows, condition, variant) if r.get("status") == "ok"]
    return mean(vals)


def all_ok_count(rows: List[Dict[str, str]], condition: str, variant: str) -> int:
    return sum(1 for r in rows_for(rows, condition, variant) if r.get("status") == "ok")


def group_table(rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    by: Dict[Tuple[str, str, str, str], List[Dict[str, str]]] = defaultdict(list)
    for r in rows:
        by[(r.get("baseline", ""), r.get("source_action", ""), r.get("repair_variant", ""), r.get("condition", ""))].append(r)

    out = []
    for (baseline, source, variant, condition), group in sorted(by.items()):
        ok_group = [r for r in group if r.get("status") == "ok"]
        out.append({
            "baseline": baseline,
            "source_action": source,
            "repair_variant": variant,
            "condition": condition,
            "rows": len(group),
            "ok_rows": len(ok_group),
            "timeout_rows": sum(1 for r in group if "timeout" in str(r.get("status", ""))),
            "failed_rows": sum(1 for r in group if r.get("status") not in {"ok", ""} and "timeout" not in str(r.get("status", ""))),
            "success_rate": mean([1.0 if pb(r.get("exec_success")) else 0.0 for r in ok_group]),
            "mean_delta_final_fraction": mean([sf(r.get("exec_delta_final_fraction")) for r in ok_group]),
            "mean_final_fraction_after": mean([sf(r.get("exec_final_fraction_after")) for r in ok_group]),
            "mean_prefix_mae": mean([sf(r.get("exec_prefix_state_mae")) for r in ok_group]),
            "mean_action_mae_to_gt": mean([sf(r.get("action_mae_to_gt")) for r in ok_group]),
            "mean_pose0_xy_dist_to_gt": mean([sf(r.get("pose0_xy_dist_to_gt")) for r in ok_group]),
            "mean_pose1_xy_dist_to_gt": mean([sf(r.get("pose1_xy_dist_to_gt")) for r in ok_group]),
            "mean_pull_angle_deg_to_gt": mean([sf(r.get("pull_angle_deg_to_gt")) for r in ok_group]),
            "exec_failure_count": sum(1 for r in group if r.get("exec_failure_reason")),
        })
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--trials_csv", default="reports/phase3_8c_pose0_confirmation_trials.csv")
    parser.add_argument("--progress_json", default="reports/phase3_8c_pose0_confirmation_progress.json")
    parser.add_argument("--raw_summary", default="reports/phase3_8c_pose0_confirmation_raw_summary.json")
    parser.add_argument("--out_json", default="reports/phase3_8c_pose0_confirmation_summary.json")
    parser.add_argument("--out_md", default="reports/phase3_8c_pose0_confirmation_report.md")
    parser.add_argument("--effective_delta_threshold", type=float, default=0.05)
    parser.add_argument("--min_ok_rows_per_condition_variant", type=int, default=1)
    args = parser.parse_args()

    root = Path(args.root)
    rows = read_rows(root / args.trials_csv)
    progress = json.loads((root / args.progress_json).read_text()) if (root / args.progress_json).exists() else {}
    raw = json.loads((root / args.raw_summary).read_text()) if (root / args.raw_summary).exists() else {}

    issues: List[Dict[str, Any]] = []

    def issue(level: str, name: str, detail: Any) -> None:
        issues.append({"level": level, "name": name, "detail": str(detail)})

    if not rows:
        issue("FAIL", "no_rows", "No Phase3.8c confirmation rows found.")

    table = group_table(rows)

    ok_rows = [r for r in rows if r.get("status") == "ok"]
    timeout_rows = [r for r in rows if "timeout" in str(r.get("status", ""))]
    failed_rows = [r for r in rows if r.get("status") not in {"ok", ""} and "timeout" not in str(r.get("status", ""))]

    if timeout_rows:
        issue("WARN", "row_timeouts_present", len(timeout_rows))
    if failed_rows:
        issue("WARN", "row_failures_present", len(failed_rows))

    missing: List[str] = []
    for cond in CONFIRM_CONDITIONS:
        for variant in REQUIRED_VARIANTS:
            n_ok = all_ok_count(rows, cond, variant)
            if n_ok < args.min_ok_rows_per_condition_variant:
                missing.append(f"{cond}/{variant}: ok={n_ok}")
    if missing:
        issue("FAIL", "missing_required_condition_variant_rows", missing)

    condition_table: List[Dict[str, Any]] = []
    support: Dict[str, bool] = {}
    pose1_support: Dict[str, bool] = {}
    both_support: Dict[str, bool] = {}

    for cond in CONFIRM_CONDITIONS:
        gt_delta = delta(rows, cond, "gt_reference")
        orig_delta = delta(rows, cond, "idm_original")
        pose0_delta = delta(rows, cond, "idm_gt_pose0_xy")
        pose1_delta = delta(rows, cond, "idm_gt_pose1_xy")
        both_delta = delta(rows, cond, "idm_gt_pose0_pose1_xy")

        gt_ok = math.isfinite(gt_delta) and gt_delta > args.effective_delta_threshold
        pose0_eff = (
            gt_ok
            and math.isfinite(orig_delta)
            and math.isfinite(pose0_delta)
            and pose0_delta > max(orig_delta + args.effective_delta_threshold, args.effective_delta_threshold)
        )
        pose1_eff = (
            gt_ok
            and math.isfinite(orig_delta)
            and math.isfinite(pose1_delta)
            and pose1_delta > max(orig_delta + args.effective_delta_threshold, args.effective_delta_threshold)
        )
        both_eff = (
            gt_ok
            and math.isfinite(orig_delta)
            and math.isfinite(both_delta)
            and both_delta > max(orig_delta + args.effective_delta_threshold, args.effective_delta_threshold)
        )

        support[cond] = pose0_eff
        pose1_support[cond] = pose1_eff
        both_support[cond] = both_eff

        if not gt_ok:
            issue("WARN", "gt_reference_weak_for_condition", f"{cond}: gt_delta={gt_delta:.6f}")

        condition_table.append({
            "condition": cond,
            "gt_reference_delta": gt_delta,
            "idm_original_delta": orig_delta,
            "pose0_xy_delta": pose0_delta,
            "pose1_xy_delta": pose1_delta,
            "pose0_pose1_xy_delta": both_delta,
            "pose0_effective": pose0_eff,
            "pose1_effective": pose1_eff,
            "both_xy_effective": both_eff,
            "gt_reference_success": success_rate(rows, cond, "gt_reference"),
            "idm_original_success": success_rate(rows, cond, "idm_original"),
            "pose0_xy_success": success_rate(rows, cond, "idm_gt_pose0_xy"),
            "pose1_xy_success": success_rate(rows, cond, "idm_gt_pose1_xy"),
            "pose0_pose1_xy_success": success_rate(rows, cond, "idm_gt_pose0_pose1_xy"),
            "pose0_xy_pose0_dist": metric_mean(rows, cond, "idm_gt_pose0_xy", "pose0_xy_dist_to_gt"),
            "pose0_xy_pose1_dist": metric_mean(rows, cond, "idm_gt_pose0_xy", "pose1_xy_dist_to_gt"),
            "pose1_xy_pose0_dist": metric_mean(rows, cond, "idm_gt_pose1_xy", "pose0_xy_dist_to_gt"),
            "pose1_xy_pose1_dist": metric_mean(rows, cond, "idm_gt_pose1_xy", "pose1_xy_dist_to_gt"),
        })

    pose0_supported_conditions = [cond for cond, ok in support.items() if ok]
    pose1_supported_conditions = [cond for cond, ok in pose1_support.items() if ok]
    both_supported_conditions = [cond for cond, ok in both_support.items() if ok]

    hidden_breakaway_pose0 = support.get("hidden_breakaway_pin", False)
    free_pose0 = support.get("free", False)
    hidden_high_friction_pose0 = support.get("hidden_high_friction", False)

    if hidden_breakaway_pose0 and len(pose0_supported_conditions) >= 2 and len(pose1_supported_conditions) < len(pose0_supported_conditions):
        issue("WARN", "pose0_xy_repair_confirmed_diagnostic_upper_bound", pose0_supported_conditions)
        root_cause = "pose0_pick_xy_geometry_blocker_confirmed"
    elif hidden_breakaway_pose0 and len(pose0_supported_conditions) >= 1:
        issue("WARN", "pose0_xy_supported_on_primary_branch_only_partial", pose0_supported_conditions)
        root_cause = "pose0_pick_xy_geometry_blocker_supported_primary_partial"
    elif len(pose0_supported_conditions) >= 2 and not hidden_breakaway_pose0:
        issue("WARN", "pose0_xy_supported_but_not_primary_branch", pose0_supported_conditions)
        root_cause = "pose0_pick_xy_geometry_blocker_supported_nonprimary"
    elif len(both_supported_conditions) >= 2:
        issue("WARN", "coupled_xy_repair_supported", both_supported_conditions)
        root_cause = "coupled_pick_place_xy_geometry_blocker_supported"
    elif len(pose1_supported_conditions) >= 2 and len(pose0_supported_conditions) == 0:
        issue("WARN", "pose1_xy_repair_supported_in_confirmation", pose1_supported_conditions)
        root_cause = "pose1_place_xy_geometry_blocker_supported_after_confirmation"
    else:
        issue("FAIL", "pose0_confirmation_not_supported", {
            "pose0_supported_conditions": pose0_supported_conditions,
            "pose1_supported_conditions": pose1_supported_conditions,
            "both_supported_conditions": both_supported_conditions,
        })
        root_cause = "pose0_confirmation_not_supported"

    if progress:
        status = progress.get("status")
        if status != "completed":
            issue("WARN", "confirmation_not_fully_completed", status)

    try:
        completed_fraction = float(progress.get("completed", 0)) / float(progress.get("total_specs", 1))
    except Exception:
        completed_fraction = None

    has_fail = any(item["level"] == "FAIL" for item in issues)
    has_warn = any(item["level"] == "WARN" for item in issues)
    verdict = "FAIL" if has_fail else ("WARN" if has_warn else "PASS")

    payload = {
        "verdict": verdict,
        "root_cause": root_cause,
        "num_rows": len(rows),
        "num_ok_rows": len(ok_rows),
        "num_timeout_rows": len(timeout_rows),
        "num_failed_rows": len(failed_rows),
        "completed_fraction": completed_fraction,
        "pose0_supported_conditions": pose0_supported_conditions,
        "pose1_supported_conditions": pose1_supported_conditions,
        "both_xy_supported_conditions": both_supported_conditions,
        "condition_table": condition_table,
        "table": table,
        "issues": issues,
        "progress": progress,
        "raw_summary": raw,
        "important_note": "This is a confirmation diagnostic. GT-blended variants are diagnostic upper bounds only, not deployable policy actions.",
        "recommendation": (
            "Do not enter Phase4/CPS. Proceed to Phase3.9 IDM geometry-aware loss or pose0/pick weighted action repair."
            if root_cause in {
                "pose0_pick_xy_geometry_blocker_confirmed",
                "pose0_pick_xy_geometry_blocker_supported_primary_partial",
                "pose0_pick_xy_geometry_blocker_supported_nonprimary",
                "coupled_pick_place_xy_geometry_blocker_supported",
            }
            else
            "Do not enter Phase4/CPS. Pose0 confirmation failed; inspect primitive-level pick/grasp contact and row stderr."
        ),
    }

    out_json = root / args.out_json
    out_md = root / args.out_md
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True))

    lines = [
        "# Phase3.8c Pose0-XY Repair Confirmation Report",
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
        f"- Pose0 supported conditions: `{pose0_supported_conditions}`",
        f"- Pose1 supported conditions: `{pose1_supported_conditions}`",
        f"- Coupled xy supported conditions: `{both_supported_conditions}`",
        "",
        "## Per-Condition Confirmation",
        "",
        "| Condition | GT Δ | IDM original Δ | Pose0 XY Δ | Pose1 XY Δ | Pose0+Pose1 XY Δ | Pose0 effective | Pose1 effective | Both XY effective |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in condition_table:
        lines.append(
            f"| `{item['condition']}` | {item['gt_reference_delta']:.4f} | {item['idm_original_delta']:.4f} | "
            f"{item['pose0_xy_delta']:.4f} | {item['pose1_xy_delta']:.4f} | {item['pose0_pose1_xy_delta']:.4f} | "
            f"`{item['pose0_effective']}` | `{item['pose1_effective']}` | `{item['both_xy_effective']}` |"
        )

    lines += [
        "",
        "## Summary Table",
        "",
        "| Baseline | Source | Variant | Condition | Rows | OK | Timeout | Success | Δ final_fraction | Final fraction | Prefix MAE | Action MAE | Pose0 dist | Pose1 dist | Pull angle | Failures |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in table:
        lines.append(
            f"| `{item['baseline']}` | `{item['source_action']}` | `{item['repair_variant']}` | `{item['condition']}` | "
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
        lines.append("| `PASS` | `none` | No confirmation issues found. |")

    lines += [
        "",
        "## Interpretation",
        "",
        "- This confirms or rejects the Phase3.8b early-stop pose0_xy finding across multiple conditions.",
        "- GT-blended pose0/pose1 repairs are diagnostic upper bounds only.",
        "- If pose0_xy is effective on hidden_breakaway_pin, the next repair target is pose0/pick-point geometry.",
        "- If only pose0+pose1_xy is effective, use coupled decoded-pose-space loss.",
        "- This is not Phase4 and not CPS evidence.",
    ]

    out_md.write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))

    if verdict == "FAIL":
        raise SystemExit("[Phase3.8c][FAIL] pose0 confirmation did not support an effective repair")


if __name__ == "__main__":
    main()
