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
    return str(x).strip().lower() in {"1", "true", "yes", "pass", "success"}


def mean(xs: List[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--trials_csv", default="reports/phase3_8_idm_geometry_repair_trials.csv")
    parser.add_argument("--sensitivity_json", default="reports/phase3_8_action_geometry_sensitivity_summary.json")
    parser.add_argument("--out_json", default="reports/phase3_8_idm_geometry_repair_summary.json")
    parser.add_argument("--out_md", default="reports/phase3_8_idm_geometry_repair_report.md")
    args = parser.parse_args()

    root = Path(args.root)
    path = root / args.trials_csv
    issues: List[Dict[str, Any]] = []

    def issue(level: str, name: str, detail: Any) -> None:
        issues.append({"level": level, "name": name, "detail": str(detail)})

    if not path.exists():
        issue("FAIL", "missing_trials_csv", str(path))
        rows: List[Dict[str, str]] = []
    else:
        rows = read_rows(path)

    if not rows:
        issue("FAIL", "no_rows", "No geometry repair rows.")

    by_key: Dict[tuple, List[Dict[str, str]]] = defaultdict(list)
    for r in rows:
        key = (
            r.get("baseline", ""),
            r.get("source_action", ""),
            r.get("repair_variant", ""),
            r.get("condition", ""),
        )
        by_key[key].append(r)

    table: List[Dict[str, Any]] = []
    for (baseline, source_action, variant, condition), group in sorted(by_key.items()):
        item = {
            "baseline": baseline,
            "source_action": source_action,
            "repair_variant": variant,
            "condition": condition,
            "rows": len(group),
            "success_rate": mean([1.0 if pb(r.get("exec_success")) else 0.0 for r in group]),
            "mean_delta_final_fraction": mean([sf(r.get("exec_delta_final_fraction")) for r in group]),
            "mean_final_fraction_after": mean([sf(r.get("exec_final_fraction_after")) for r in group]),
            "mean_prefix_mae": mean([sf(r.get("exec_prefix_state_mae")) for r in group]),
            "mean_action_mae_to_gt": mean([sf(r.get("action_mae_to_gt")) for r in group]),
            "mean_pose0_xy_dist_to_gt": mean([sf(r.get("pose0_xy_dist_to_gt")) for r in group]),
            "mean_pose1_xy_dist_to_gt": mean([sf(r.get("pose1_xy_dist_to_gt")) for r in group]),
            "mean_pull_angle_deg_to_gt": mean([sf(r.get("pull_angle_deg_to_gt")) for r in group]),
            "mean_pull_len_ratio_to_gt": mean([sf(r.get("pull_len_ratio_to_gt")) for r in group]),
            "failure_count": sum(1 for r in group if r.get("exec_failure_reason")),
            "deployable_rate": mean([1.0 if pb(r.get("is_deployable_policy_action")) else 0.0 for r in group]),
        }
        table.append(item)

    def subset(source_action: str, variant: str, condition: str | None = None) -> List[Dict[str, str]]:
        out = [r for r in rows if r.get("source_action") == source_action and r.get("repair_variant") == variant]
        if condition is not None:
            out = [r for r in out if r.get("condition") == condition]
        return out

    def delta_for(source_action: str, variant: str) -> float:
        return mean([sf(r.get("exec_delta_final_fraction")) for r in subset(source_action, variant)])

    def succ_for(source_action: str, variant: str) -> float:
        return mean([1.0 if pb(r.get("exec_success")) else 0.0 for r in subset(source_action, variant)])

    gt_delta = delta_for("idm_gt_future", "gt_reference")
    idm_gt_orig_delta = delta_for("idm_gt_future", "idm_original")
    idm_pred_orig_delta = delta_for("idm_pred_future", "idm_original")

    variant_delta = {
        item["repair_variant"]: mean([t["mean_delta_final_fraction"] for t in table if t["repair_variant"] == item["repair_variant"]])
        for item in table
    }

    pose0_delta = delta_for("idm_gt_future", "idm_gt_pose0_xy")
    pose1_delta = delta_for("idm_gt_future", "idm_gt_pose1_xy")
    both_xy_delta = delta_for("idm_gt_future", "idm_gt_pose0_pose1_xy")
    quat_delta = delta_for("idm_gt_future", "idm_gt_quat")
    z_delta = delta_for("idm_gt_future", "idm_gt_z")
    qz_delta = delta_for("idm_gt_future", "idm_gt_quat_z")
    full_delta = delta_for("idm_gt_future", "idm_gt_xy_z_quat")
    dir_delta = max(
        delta_for("idm_gt_future", "idm_gt_pull_dir_keep_idm_len"),
        delta_for("idm_gt_future", "idm_gt_pull_dir_gt_len"),
        delta_for("idm_gt_future", "idm_gt_pick_gt_dir_keep_idm_len"),
    )

    exec_failures = sum(1 for r in rows if r.get("exec_failure_reason"))
    if exec_failures > 0:
        issue("FAIL", "execution_failures_present", f"{exec_failures} rows have exec_failure_reason")

    if math.isfinite(gt_delta) and gt_delta <= 0.05:
        issue("FAIL", "gt_reference_no_progress", f"gt_reference delta={gt_delta:.6f}; matched execution reference broken")

    if math.isfinite(idm_gt_orig_delta) and idm_gt_orig_delta <= 0.01 and math.isfinite(gt_delta) and gt_delta > 0.05:
        issue("WARN", "idm_original_reproduces_phase37_failure", f"idm_gt_original_delta={idm_gt_orig_delta:.6f}, gt_delta={gt_delta:.6f}")

    # Identify repair effects. GT-blended variants are diagnostic only.
    repair_candidates = {
        "pose0_xy": pose0_delta,
        "pose1_xy": pose1_delta,
        "pose0_pose1_xy": both_xy_delta,
        "quat": quat_delta,
        "z": z_delta,
        "quat_z": qz_delta,
        "xy_z_quat": full_delta,
        "pull_direction": dir_delta,
    }
    effective = {k: v for k, v in repair_candidates.items() if math.isfinite(v) and v > max(idm_gt_orig_delta + 0.05, 0.05)}

    if effective:
        issue("WARN", "gt_blended_geometry_variant_restores_progress", effective)
    else:
        issue("FAIL", "no_single_geometry_variant_restores_progress", repair_candidates)

    if "pose1_xy" in effective and "pose0_xy" not in effective:
        root_cause = "pose1_place_xy_geometry_blocker"
    elif "pose0_xy" in effective and "pose1_xy" not in effective:
        root_cause = "pose0_pick_xy_geometry_blocker"
    elif "pose0_pose1_xy" in effective or "xy_z_quat" in effective:
        root_cause = "coupled_pick_place_xy_geometry_blocker"
    elif "pull_direction" in effective:
        root_cause = "pull_direction_or_length_geometry_blocker"
    elif "quat" in effective or "quat_z" in effective:
        root_cause = "quat_or_z_execution_geometry_blocker"
    elif any(i["name"] == "no_single_geometry_variant_restores_progress" for i in issues):
        root_cause = "no_single_geometry_repair_identified"
    else:
        root_cause = "undetermined_idm_geometry_repair"

    # Future prediction amplification.
    if math.isfinite(idm_pred_orig_delta) and math.isfinite(idm_gt_orig_delta) and idm_pred_orig_delta <= idm_gt_orig_delta + 0.01:
        future_note = "future_prediction_not_primary_in_execution_delta"
    else:
        future_note = "future_prediction_may_amplify_geometry_error"

    has_fail = any(i["level"] == "FAIL" for i in issues)
    has_warn = any(i["level"] == "WARN" for i in issues)
    verdict = "FAIL" if has_fail else ("WARN" if has_warn else "PASS")

    payload = {
        "verdict": verdict,
        "root_cause": root_cause,
        "num_rows": len(rows),
        "gt_reference_delta": gt_delta,
        "idm_gt_original_delta": idm_gt_orig_delta,
        "idm_pred_original_delta": idm_pred_orig_delta,
        "repair_candidates": repair_candidates,
        "effective_gt_blended_repairs": effective,
        "future_note": future_note,
        "table": table,
        "issues": issues,
        "recommendation": (
            "Do not enter Phase4/CPS. Use the identified geometry blocker to design an IDM/action-loss repair."
            if verdict in {"WARN", "PASS"}
            else "Do not enter Phase4/CPS. Repair probe did not identify a clean geometry fix."
        ),
        "important_note": "GT-blended repair variants are diagnostic upper bounds only, not deployable policy actions.",
    }

    out_json = root / args.out_json
    out_md = root / args.out_md
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True))

    lines = [
        "# Phase3.8 IDM Action Geometry Repair Report",
        "",
        "## Verdict",
        "",
        f"- Verdict: `{verdict}`",
        f"- Root cause: `{root_cause}`",
        f"- Rows: `{len(rows)}`",
        f"- GT reference Δ final_fraction: `{gt_delta:.6f}`",
        f"- IDM(GT future) original Δ final_fraction: `{idm_gt_orig_delta:.6f}`",
        f"- IDM(pred future) original Δ final_fraction: `{idm_pred_orig_delta:.6f}`",
        f"- Future note: `{future_note}`",
        "",
        "## Repair Candidates",
        "",
        "| Candidate | Mean Δ final_fraction | Effective |",
        "|---|---:|---:|",
    ]
    for k, v in repair_candidates.items():
        lines.append(f"| `{k}` | {v:.6f} | `{k in effective}` |")

    lines += [
        "",
        "## Summary Table",
        "",
        "| Baseline | Source | Variant | Condition | Rows | Success | Δ final_fraction | Final fraction | Prefix MAE | Action MAE | Pose0 dist | Pose1 dist | Pull angle | Failures |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in table:
        lines.append(
            f"| `{item['baseline']}` | `{item['source_action']}` | `{item['repair_variant']}` | `{item['condition']}` | "
            f"{item['rows']} | {item['success_rate']:.3f} | {item['mean_delta_final_fraction']:.4f} | "
            f"{item['mean_final_fraction_after']:.4f} | {item['mean_prefix_mae']:.4f} | "
            f"{item['mean_action_mae_to_gt']:.4f} | {item['mean_pose0_xy_dist_to_gt']:.4f} | "
            f"{item['mean_pose1_xy_dist_to_gt']:.4f} | {item['mean_pull_angle_deg_to_gt']:.2f} | {item['failure_count']} |"
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
        lines.append("| `PASS` | `none` | No geometry repair issues found. |")

    lines += [
        "",
        "## Interpretation",
        "",
        "- GT-blended variants are diagnostic upper bounds, not deployable policy actions.",
        "- If pose1_xy repair restores progress, place-point geometry is the main blocker.",
        "- If pose0_xy repair restores progress, pick-point geometry is the main blocker.",
        "- If only combined xy/z/quat repair works, the blocker is coupled pick/place geometry.",
        "- This is not Phase4 and not CPS evidence.",
    ]
    out_md.write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))

    if verdict == "FAIL":
        raise SystemExit("[Phase3.8][FAIL] geometry repair probe did not identify a clean repair")


if __name__ == "__main__":
    main()
