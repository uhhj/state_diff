#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

PRIMARY = "hidden_breakaway_pin"


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
    parser.add_argument("--trials_csv", default="reports/phase3_7_learned_action_alignment_trials.csv")
    parser.add_argument("--out_json", default="reports/phase3_7_learned_action_alignment_summary.json")
    parser.add_argument("--out_md", default="reports/phase3_7_learned_action_alignment_report.md")
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
        issue("FAIL", "no_rows", "No alignment rows found.")

    by_key: Dict[tuple, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_key[(row.get("baseline", ""), row.get("action_source", ""), row.get("condition", ""))].append(row)

    table: List[Dict[str, Any]] = []
    for (baseline, source, condition), group in sorted(by_key.items()):
        item = {
            "baseline": baseline,
            "action_source": source,
            "condition": condition,
            "rows": len(group),
            "mean_action_mae_to_gt": mean([sf(r.get("action_mae_to_gt")) for r in group]),
            "mean_action_l2_to_gt": mean([sf(r.get("action_l2_to_gt")) for r in group]),
            "mean_action_cosine_to_gt": mean([sf(r.get("action_cosine_to_gt")) for r in group]),
            "mean_pose0_xy_dist_to_gt": mean([sf(r.get("pose0_xy_dist_to_gt")) for r in group]),
            "mean_pose1_xy_dist_to_gt": mean([sf(r.get("pose1_xy_dist_to_gt")) for r in group]),
            "mean_pull_angle_deg_to_gt": mean([sf(r.get("pull_angle_deg_to_gt")) for r in group]),
            "mean_pull_len_ratio_to_gt": mean([sf(r.get("pull_len_ratio_to_gt")) for r in group]),
            "mean_future_error_mae": mean([sf(r.get("future_error_mae")) for r in group]),
            "mean_future_error_l2": mean([sf(r.get("future_error_l2")) for r in group]),
            "mean_action_ood": mean([sf(r.get("action_ood_score")) for r in group]),
            "mean_clip_fraction": mean([sf(r.get("clip_fraction")) for r in group]),
            "exec_attempted_rate": mean([1.0 if pb(r.get("exec_attempted")) else 0.0 for r in group]),
            "exec_success_rate": mean([1.0 if pb(r.get("exec_success")) else 0.0 for r in group]),
            "mean_exec_delta_final_fraction": mean([sf(r.get("exec_delta_final_fraction")) for r in group]),
            "mean_exec_prefix_state_mae": mean([sf(r.get("exec_prefix_state_mae")) for r in group]),
            "exec_failure_count": sum(1 for r in group if r.get("exec_failure_reason")),
        }
        table.append(item)

    def rows_for(source: str, baseline: str | None = None, condition: str | None = None) -> List[Dict[str, str]]:
        out = [r for r in rows if r.get("action_source") == source]
        if baseline is not None:
            out = [r for r in out if r.get("baseline") == baseline]
        if condition is not None:
            out = [r for r in out if r.get("condition") == condition]
        return out

    idm_gt_rows = rows_for("idm_gt_future")
    idm_pred_rows = rows_for("idm_pred_future")
    gt_rows = rows_for("gt_y_action")
    oracle_rows = rows_for("same_state_oracle")

    idm_gt_mae = mean([sf(r.get("action_mae_to_gt")) for r in idm_gt_rows])
    idm_pred_mae = mean([sf(r.get("action_mae_to_gt")) for r in idm_pred_rows])
    idm_gt_pose1 = mean([sf(r.get("pose1_xy_dist_to_gt")) for r in idm_gt_rows])
    idm_pred_pose1 = mean([sf(r.get("pose1_xy_dist_to_gt")) for r in idm_pred_rows])
    future_mae = mean([sf(r.get("future_error_mae")) for r in idm_pred_rows])
    idm_gt_exec_delta = mean([sf(r.get("exec_delta_final_fraction")) for r in idm_gt_rows])
    idm_pred_exec_delta = mean([sf(r.get("exec_delta_final_fraction")) for r in idm_pred_rows])
    gt_exec_delta = mean([sf(r.get("exec_delta_final_fraction")) for r in gt_rows])
    oracle_exec_delta = mean([sf(r.get("exec_delta_final_fraction")) for r in oracle_rows])

    # Conservative thresholds for smoke diagnostics.
    teacher_action_mae_bad = math.isfinite(idm_gt_mae) and idm_gt_mae > 0.12
    teacher_pose_bad = math.isfinite(idm_gt_pose1) and idm_gt_pose1 > 0.08
    pred_much_worse_than_teacher = (
        math.isfinite(idm_pred_mae)
        and math.isfinite(idm_gt_mae)
        and idm_pred_mae > max(idm_gt_mae * 1.5, idm_gt_mae + 0.05)
    )
    pred_pose_much_worse = (
        math.isfinite(idm_pred_pose1)
        and math.isfinite(idm_gt_pose1)
        and idm_pred_pose1 > max(idm_gt_pose1 * 1.5, idm_gt_pose1 + 0.05)
    )

    if teacher_action_mae_bad or teacher_pose_bad:
        issue(
            "FAIL",
            "idm_gt_future_action_misaligned",
            f"IDM(GT future) does not match y_action well: mean_action_mae={idm_gt_mae:.6f}, mean_pose1_xy_dist={idm_gt_pose1:.6f}",
        )

    if (not teacher_action_mae_bad) and (pred_much_worse_than_teacher or pred_pose_much_worse):
        issue(
            "WARN",
            "future_prediction_degrades_idm_action",
            f"IDM(pred future) worse than IDM(GT future): gt_mae={idm_gt_mae:.6f}, pred_mae={idm_pred_mae:.6f}, gt_pose1={idm_gt_pose1:.6f}, pred_pose1={idm_pred_pose1:.6f}",
        )

    if gt_rows and math.isfinite(gt_exec_delta) and gt_exec_delta > 0.05 and idm_gt_rows and math.isfinite(idm_gt_exec_delta) and idm_gt_exec_delta <= 0.01:
        issue(
            "FAIL",
            "idm_gt_future_executes_poorly_despite_gt_progress",
            f"GT action progresses but IDM(GT future) does not: gt_delta={gt_exec_delta:.6f}, idm_gt_delta={idm_gt_exec_delta:.6f}",
        )

    if idm_gt_rows and math.isfinite(idm_gt_exec_delta) and idm_gt_exec_delta > 0.05 and idm_pred_rows and math.isfinite(idm_pred_exec_delta) and idm_pred_exec_delta <= 0.01:
        issue(
            "WARN",
            "pred_future_action_executes_poorly_despite_idm_gt_progress",
            f"IDM(GT future) progresses but IDM(pred future) does not: idm_gt_delta={idm_gt_exec_delta:.6f}, idm_pred_delta={idm_pred_exec_delta:.6f}",
        )

    exec_failures = sum(1 for r in rows if r.get("exec_failure_reason"))
    if exec_failures > 0:
        issue("FAIL", "execution_failures_present", f"{exec_failures} rows have exec_failure_reason")

    if oracle_rows and math.isfinite(oracle_exec_delta) and oracle_exec_delta <= 0.0 and gt_rows and math.isfinite(gt_exec_delta) and gt_exec_delta <= 0.0:
        issue(
            "FAIL",
            "same_state_oracle_and_gt_no_progress",
            "Neither oracle nor GT progresses in Phase3.7 matched execution; matched environment may be broken.",
        )

    # Branch-bias diagnostic for primary pair.
    free_pred = rows_for("idm_pred_future", condition="free")
    hidden_pred = rows_for("idm_pred_future", condition=PRIMARY)
    if free_pred and hidden_pred:
        free_mae = mean([sf(r.get("action_mae_to_gt")) for r in free_pred])
        hidden_mae = mean([sf(r.get("action_mae_to_gt")) for r in hidden_pred])
        if math.isfinite(free_mae) and math.isfinite(hidden_mae) and hidden_mae > free_mae + 0.05:
            issue(
                "WARN",
                "primary_hidden_action_alignment_worse_than_free",
                f"hidden_breakaway action alignment worse than free: free_mae={free_mae:.6f}, hidden_mae={hidden_mae:.6f}",
            )

    has_fail = any(i["level"] == "FAIL" for i in issues)
    has_warn = any(i["level"] == "WARN" for i in issues)
    verdict = "FAIL" if has_fail else ("WARN" if has_warn else "PASS")

    if any(i["name"] == "idm_gt_future_action_misaligned" for i in issues):
        root_cause = "idm_teacher_forced_action_alignment_blocker"
    elif any(i["name"] == "idm_gt_future_executes_poorly_despite_gt_progress" for i in issues):
        root_cause = "idm_action_execution_geometry_blocker"
    elif any(i["name"] == "future_prediction_degrades_idm_action" for i in issues) or any(i["name"] == "pred_future_action_executes_poorly_despite_idm_gt_progress" for i in issues):
        root_cause = "future_prediction_to_action_alignment_blocker"
    elif any(i["name"] == "primary_hidden_action_alignment_worse_than_free" for i in issues):
        root_cause = "primary_branch_action_alignment_weakness"
    elif verdict == "PASS":
        root_cause = "offline_action_alignment_passed_closed_loop_history_suspect"
    else:
        root_cause = "undetermined_learned_action_alignment"

    payload = {
        "verdict": verdict,
        "root_cause": root_cause,
        "num_rows": len(rows),
        "idm_gt_mean_action_mae_to_gt": idm_gt_mae,
        "idm_pred_mean_action_mae_to_gt": idm_pred_mae,
        "idm_gt_mean_pose1_xy_dist_to_gt": idm_gt_pose1,
        "idm_pred_mean_pose1_xy_dist_to_gt": idm_pred_pose1,
        "mean_future_error_mae": future_mae,
        "gt_exec_mean_delta_final_fraction": gt_exec_delta,
        "oracle_exec_mean_delta_final_fraction": oracle_exec_delta,
        "idm_gt_exec_mean_delta_final_fraction": idm_gt_exec_delta,
        "idm_pred_exec_mean_delta_final_fraction": idm_pred_exec_delta,
        "table": table,
        "issues": issues,
        "recommendation": (
            "Do not enter Phase4/CPS. Fix the blocking alignment issue first."
            if verdict == "FAIL"
            else "Review WARNs before any larger rollout. This is not CPS evidence."
        ),
    }

    out_json = root / args.out_json
    out_md = root / args.out_md
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True))

    lines = [
        "# Phase3.7 Learned Action Alignment Report",
        "",
        "## Verdict",
        "",
        f"- Verdict: `{verdict}`",
        f"- Root cause: `{root_cause}`",
        f"- Rows: `{len(rows)}`",
        f"- IDM(GT future) mean action MAE to GT: `{idm_gt_mae:.6f}`",
        f"- IDM(pred future) mean action MAE to GT: `{idm_pred_mae:.6f}`",
        f"- Mean future error MAE: `{future_mae:.6f}`",
        f"- GT exec mean Δ final_fraction: `{gt_exec_delta:.6f}`",
        f"- Oracle exec mean Δ final_fraction: `{oracle_exec_delta:.6f}`",
        f"- IDM(GT future) exec mean Δ final_fraction: `{idm_gt_exec_delta:.6f}`",
        f"- IDM(pred future) exec mean Δ final_fraction: `{idm_pred_exec_delta:.6f}`",
        "",
        "## Summary Table",
        "",
        "| Baseline | Source | Condition | Rows | Action MAE | Action L2 | Cosine | Pose0 dist | Pose1 dist | Pull angle | Future MAE | OOD | Clip | Exec success | Exec Δ final_fraction | Prefix MAE | Failures |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in table:
        lines.append(
            f"| `{item['baseline']}` | `{item['action_source']}` | `{item['condition']}` | {item['rows']} | "
            f"{item['mean_action_mae_to_gt']:.4f} | {item['mean_action_l2_to_gt']:.4f} | {item['mean_action_cosine_to_gt']:.4f} | "
            f"{item['mean_pose0_xy_dist_to_gt']:.4f} | {item['mean_pose1_xy_dist_to_gt']:.4f} | {item['mean_pull_angle_deg_to_gt']:.2f} | "
            f"{item['mean_future_error_mae']:.4f} | {item['mean_action_ood']:.4f} | {item['mean_clip_fraction']:.4f} | "
            f"{item['exec_success_rate']:.3f} | {item['mean_exec_delta_final_fraction']:.4f} | {item['mean_exec_prefix_state_mae']:.4f} | {item['exec_failure_count']} |"
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
        lines.append("| `PASS` | `none` | No learned action alignment issues found at this smoke scale. |")

    lines += [
        "",
        "## Interpretation",
        "",
        "- If IDM(GT future) is far from GT y_action, the first blocker is IDM teacher-forced alignment.",
        "- If IDM(GT future) is close but IDM(pred future) is far, the first blocker is future DDPM to action alignment.",
        "- If both are close offline but closed-loop rollout failed, the likely blocker is live closed-loop history update.",
        "- This is not Phase4 and not CPS evidence.",
    ]

    out_md.write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))

    if verdict == "FAIL":
        raise SystemExit("[Phase3.7][FAIL] learned action alignment found a blocking issue")


if __name__ == "__main__":
    main()
