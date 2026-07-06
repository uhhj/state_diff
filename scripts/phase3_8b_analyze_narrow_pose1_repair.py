#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


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


def group_table(rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    by: Dict[tuple, List[Dict[str, str]]] = defaultdict(list)
    for r in rows:
        by[(r.get("baseline", ""), r.get("source_action", ""), r.get("repair_variant", ""), r.get("condition", ""))].append(r)

    table = []
    for (baseline, source_action, variant, condition), group in sorted(by.items()):
        table.append({
            "baseline": baseline,
            "source_action": source_action,
            "repair_variant": variant,
            "condition": condition,
            "rows": len(group),
            "ok_rows": sum(1 for r in group if r.get("status") == "ok"),
            "timeout_rows": sum(1 for r in group if "timeout" in str(r.get("status", ""))),
            "failure_rows": sum(1 for r in group if r.get("status") not in {"ok", ""} and "timeout" not in str(r.get("status", ""))),
            "success_rate": mean([1.0 if pb(r.get("exec_success")) else 0.0 for r in group]),
            "mean_delta_final_fraction": mean([sf(r.get("exec_delta_final_fraction")) for r in group]),
            "mean_final_fraction_after": mean([sf(r.get("exec_final_fraction_after")) for r in group]),
            "mean_prefix_mae": mean([sf(r.get("exec_prefix_state_mae")) for r in group]),
            "mean_action_mae_to_gt": mean([sf(r.get("action_mae_to_gt")) for r in group]),
            "mean_pose0_xy_dist_to_gt": mean([sf(r.get("pose0_xy_dist_to_gt")) for r in group]),
            "mean_pose1_xy_dist_to_gt": mean([sf(r.get("pose1_xy_dist_to_gt")) for r in group]),
            "mean_pull_angle_deg_to_gt": mean([sf(r.get("pull_angle_deg_to_gt")) for r in group]),
            "mean_pull_len_ratio_to_gt": mean([sf(r.get("pull_len_ratio_to_gt")) for r in group]),
            "exec_failure_count": sum(1 for r in group if r.get("exec_failure_reason")),
        })
    return table


def delta(rows: List[Dict[str, str]], variant: str, source: str = "idm_gt_future") -> float:
    vals = [
        sf(r.get("exec_delta_final_fraction"))
        for r in rows
        if r.get("repair_variant") == variant and r.get("source_action") == source and r.get("status") == "ok"
    ]
    return mean(vals)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--trials_csv", default="reports/phase3_8b_narrow_pose1_repair_trials.csv")
    parser.add_argument("--progress_json", default="reports/phase3_8b_narrow_pose1_repair_progress.json")
    parser.add_argument("--raw_summary", default="reports/phase3_8b_narrow_pose1_repair_raw_summary.json")
    parser.add_argument("--out_json", default="reports/phase3_8b_narrow_pose1_repair_summary.json")
    parser.add_argument("--out_md", default="reports/phase3_8b_narrow_pose1_repair_report.md")
    parser.add_argument("--effective_delta_threshold", type=float, default=0.05)
    args = parser.parse_args()

    root = Path(args.root)
    rows = read_rows(root / args.trials_csv)
    progress = json.loads((root / args.progress_json).read_text()) if (root / args.progress_json).exists() else {}
    raw = json.loads((root / args.raw_summary).read_text()) if (root / args.raw_summary).exists() else {}

    issues: List[Dict[str, Any]] = []

    def issue(level: str, name: str, detail: Any) -> None:
        issues.append({"level": level, "name": name, "detail": str(detail)})

    if not rows:
        issue("FAIL", "no_rows", "No narrowed repair rows found.")

    table = group_table(rows)
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    timeout_rows = [r for r in rows if "timeout" in str(r.get("status", ""))]
    failed_rows = [r for r in rows if r.get("status") not in {"ok", ""} and "timeout" not in str(r.get("status", ""))]

    if timeout_rows:
        issue("WARN", "row_timeouts_present", len(timeout_rows))
    if failed_rows:
        issue("WARN", "row_failures_present", len(failed_rows))

    gt_delta = delta(rows, "gt_reference")
    orig_delta = delta(rows, "idm_original")
    pose1_delta = delta(rows, "idm_gt_pose1_xy")
    pose0_delta = delta(rows, "idm_gt_pose0_xy")
    both_xy_delta = delta(rows, "idm_gt_pose0_pose1_xy")
    pull_dir_delta = delta(rows, "idm_gt_pull_dir_gt_len")

    if math.isfinite(gt_delta) and gt_delta <= args.effective_delta_threshold:
        issue("FAIL", "gt_reference_no_progress", f"gt_reference delta={gt_delta:.6f}")

    if not math.isfinite(orig_delta):
        issue("FAIL", "idm_original_missing", "No ok idm_original rows.")
    elif math.isfinite(gt_delta) and gt_delta > args.effective_delta_threshold and orig_delta > gt_delta * 0.75:
        issue("WARN", "idm_original_unexpectedly_progresses", f"orig_delta={orig_delta:.6f}, gt_delta={gt_delta:.6f}")

    effective = {}
    for name, val in {
        "pose1_xy": pose1_delta,
        "pose0_xy": pose0_delta,
        "pose0_pose1_xy": both_xy_delta,
        "pull_direction_gt_len": pull_dir_delta,
    }.items():
        if math.isfinite(val) and math.isfinite(orig_delta) and val > max(orig_delta + args.effective_delta_threshold, args.effective_delta_threshold):
            effective[name] = val

    if effective:
        issue("WARN", "narrowed_gt_blended_repair_restores_progress", effective)
    else:
        issue("FAIL", "narrowed_repair_no_effective_variant", {
            "gt_reference": gt_delta,
            "idm_original": orig_delta,
            "pose1_xy": pose1_delta,
            "pose0_xy": pose0_delta,
            "pose0_pose1_xy": both_xy_delta,
            "pull_direction_gt_len": pull_dir_delta,
        })

    if "pose1_xy" in effective and "pose0_xy" not in effective:
        root_cause = "pose1_place_xy_geometry_blocker_supported"
    elif "pose0_xy" in effective and "pose1_xy" not in effective:
        root_cause = "pose0_pick_xy_geometry_blocker_supported"
    elif "pose0_pose1_xy" in effective:
        root_cause = "coupled_pick_place_xy_geometry_blocker_supported"
    elif "pull_direction_gt_len" in effective:
        root_cause = "pull_direction_or_length_geometry_blocker_supported"
    elif any(i["name"] == "narrowed_repair_no_effective_variant" for i in issues):
        root_cause = "narrowed_probe_no_effective_repair"
    else:
        root_cause = "undetermined_narrowed_pose1_repair"

    completed_fraction = None
    try:
        completed_fraction = float(progress.get("completed", 0)) / float(progress.get("total_specs", 1))
    except Exception:
        completed_fraction = None

    if progress and progress.get("status") not in {"completed", "stopped_after_first_effective"}:
        issue("WARN", "probe_not_fully_completed", progress.get("status"))

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
        "gt_reference_delta": gt_delta,
        "idm_original_delta": orig_delta,
        "pose1_xy_delta": pose1_delta,
        "pose0_xy_delta": pose0_delta,
        "pose0_pose1_xy_delta": both_xy_delta,
        "pull_direction_gt_len_delta": pull_dir_delta,
        "effective_gt_blended_repairs": effective,
        "table": table,
        "issues": issues,
        "progress": progress,
        "raw_summary": raw,
        "important_note": "This is a narrowed diagnostic. GT-blended variants are diagnostic upper bounds only, not deployable policy actions.",
        "recommendation": (
            "Do not enter Phase4/CPS. Use the supported geometry blocker to design an IDM geometry loss or action weighting repair."
            if effective else
            "Do not enter Phase4/CPS. Narrowed repair did not restore progress; inspect primitive-level grasp/contact and row stderr."
        ),
    }

    out_json = root / args.out_json
    out_md = root / args.out_md
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True))

    lines = [
        "# Phase3.8b Narrowed Pose1-XY Repair Report",
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
        f"- GT reference Δ final_fraction: `{gt_delta:.6f}`",
        f"- IDM original Δ final_fraction: `{orig_delta:.6f}`",
        f"- pose1_xy repair Δ final_fraction: `{pose1_delta:.6f}`",
        f"- pose0_xy repair Δ final_fraction: `{pose0_delta:.6f}`",
        f"- pose0+pose1_xy repair Δ final_fraction: `{both_xy_delta:.6f}`",
        f"- pull_direction_gt_len repair Δ final_fraction: `{pull_dir_delta:.6f}`",
        "",
        "## Effective GT-Blended Repairs",
        "",
        "| Repair | Mean Δ final_fraction |",
        "|---|---:|",
    ]
    if effective:
        for k, v in effective.items():
            lines.append(f"| `{k}` | {v:.6f} |")
    else:
        lines.append("| `none` | nan |")

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
        lines.append("| `PASS` | `none` | No narrowed repair issues found. |")

    lines += [
        "",
        "## Interpretation",
        "",
        "- This is a narrowed diagnostic, not a full Phase3.8 repair matrix.",
        "- GT-blended variants are diagnostic upper bounds only.",
        "- If pose1_xy restores progress, pose1/place endpoint geometry is supported as the first repair target.",
        "- If pose0_pose1_xy is required, the blocker is coupled pick/place geometry.",
        "- This is not Phase4 and not CPS evidence.",
    ]

    out_md.write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))

    if verdict == "FAIL":
        raise SystemExit("[Phase3.8b][FAIL] narrowed repair did not identify an effective repair")


if __name__ == "__main__":
    main()
