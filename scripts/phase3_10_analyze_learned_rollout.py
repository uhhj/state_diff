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
POLICIES = ["old_state_action", "phase39_default_geometry", "phase39b_xy_only_high_weight"]


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


def rows_for(rows: List[Dict[str, str]], policy: str, condition: str) -> List[Dict[str, str]]:
    return [r for r in rows if r.get("policy") == policy and r.get("condition") == condition]


def metric(rows: List[Dict[str, str]], policy: str, condition: str, key: str) -> float:
    return mean([sf(r.get(key)) for r in rows_for(rows, policy, condition) if r.get("status") == "ok" and not r.get("failure_reason")])


def success_rate(rows: List[Dict[str, str]], policy: str, condition: str) -> float:
    return mean([1.0 if pb(r.get("success")) else 0.0 for r in rows_for(rows, policy, condition) if r.get("status") == "ok" and not r.get("failure_reason")])


def ok_count(rows: List[Dict[str, str]], policy: str, condition: str) -> int:
    return sum(1 for r in rows_for(rows, policy, condition) if r.get("status") == "ok" and not r.get("failure_reason"))


def group_table(rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    by: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    for r in rows:
        by[(r.get("policy", ""), r.get("condition", ""))].append(r)

    table = []
    for (policy, condition), group in sorted(by.items()):
        ok = [r for r in group if r.get("status") == "ok" and not r.get("failure_reason")]
        table.append({
            "policy": policy,
            "condition": condition,
            "rows": len(group),
            "ok_rows": len(ok),
            "timeout_rows": sum(1 for r in group if "timeout" in str(r.get("status", ""))),
            "failed_rows": sum(1 for r in group if r.get("status") != "ok" or r.get("failure_reason")),
            "success_rate": mean([1.0 if pb(r.get("success")) else 0.0 for r in ok]),
            "mean_final_fraction": mean([sf(r.get("final_fraction")) for r in ok]),
            "mean_final_curve": mean([sf(r.get("final_curve")) for r in ok]),
            "mean_steps": mean([sf(r.get("num_steps")) for r in ok]),
            "mean_action_ood": mean([sf(r.get("mean_action_ood_score")) for r in ok]),
            "max_action_ood": mean([sf(r.get("max_action_ood_score")) for r in ok]),
            "mean_future_std": mean([sf(r.get("mean_future_sample_std")) for r in ok]),
            "mean_pull_xy_len": mean([sf(r.get("mean_pull_xy_len")) for r in ok]),
            "failure_count": sum(1 for r in group if r.get("failure_reason")),
        })
    return table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--trials_csv", default="reports/phase3_10_controlled_learned_rollout_trials.csv")
    parser.add_argument("--progress_json", default="reports/phase3_10_controlled_learned_rollout_progress.json")
    parser.add_argument("--raw_summary", default="reports/phase3_10_controlled_learned_rollout_raw_summary.json")
    parser.add_argument("--out_json", default="reports/phase3_10_controlled_learned_rollout_summary.json")
    parser.add_argument("--out_md", default="reports/phase3_10_controlled_learned_rollout_report.md")
    parser.add_argument("--improve_threshold", type=float, default=0.05)
    parser.add_argument("--min_ok_rows_per_policy_condition", type=int, default=2)
    args = parser.parse_args()

    root = Path(args.root)
    rows = read_rows(root / args.trials_csv)
    progress = json.loads((root / args.progress_json).read_text()) if (root / args.progress_json).exists() else {}
    raw = json.loads((root / args.raw_summary).read_text()) if (root / args.raw_summary).exists() else {}

    issues: List[Dict[str, Any]] = []

    def issue(level: str, name: str, detail: Any) -> None:
        issues.append({"level": level, "name": name, "detail": str(detail)})

    if not rows:
        issue("FAIL", "no_rows", "No Phase3.10 rollout rows found.")

    missing = []
    for policy in POLICIES:
        for cond in CONDITIONS:
            n = ok_count(rows, policy, cond)
            if n < args.min_ok_rows_per_policy_condition:
                missing.append(f"{policy}/{cond}: ok={n}")
    if missing:
        issue("FAIL", "missing_required_policy_condition_rows", missing)

    timeout_rows = [r for r in rows if "timeout" in str(r.get("status", ""))]
    failed_rows = [r for r in rows if r.get("status") != "ok" or r.get("failure_reason")]
    if timeout_rows:
        issue("WARN", "timeouts_present", len(timeout_rows))
    if failed_rows:
        issue("WARN", "failures_present", len(failed_rows))

    table = group_table(rows)
    condition_table: List[Dict[str, Any]] = []
    improved_default: List[str] = []
    improved_best: List[str] = []

    for cond in CONDITIONS:
        old_ff = metric(rows, "old_state_action", cond, "final_fraction")
        default_ff = metric(rows, "phase39_default_geometry", cond, "final_fraction")
        best_ff = metric(rows, "phase39b_xy_only_high_weight", cond, "final_fraction")
        default_imp = default_ff - old_ff if math.isfinite(default_ff) and math.isfinite(old_ff) else float("nan")
        best_imp = best_ff - old_ff if math.isfinite(best_ff) and math.isfinite(old_ff) else float("nan")

        default_improved = math.isfinite(default_imp) and default_imp > args.improve_threshold and default_ff > args.improve_threshold
        best_improved = math.isfinite(best_imp) and best_imp > args.improve_threshold and best_ff > args.improve_threshold

        if default_improved:
            improved_default.append(cond)
        if best_improved:
            improved_best.append(cond)

        condition_table.append({
            "condition": cond,
            "old_final_fraction": old_ff,
            "default_final_fraction": default_ff,
            "best_final_fraction": best_ff,
            "default_improvement": default_imp,
            "best_improvement": best_imp,
            "default_improved": default_improved,
            "best_improved": best_improved,
            "old_success": success_rate(rows, "old_state_action", cond),
            "default_success": success_rate(rows, "phase39_default_geometry", cond),
            "best_success": success_rate(rows, "phase39b_xy_only_high_weight", cond),
        })

    primary_best_improved = "hidden_breakaway_pin" in improved_best
    primary_default_improved = "hidden_breakaway_pin" in improved_default

    if primary_best_improved and len(improved_best) >= 2:
        issue("WARN", "phase310_best_repaired_rollout_supported", improved_best)
        root_cause = "phase310_best_repaired_rollout_supported"
    elif primary_default_improved and len(improved_default) >= 2:
        issue("WARN", "phase310_default_repaired_rollout_supported", improved_default)
        root_cause = "phase310_default_repaired_rollout_supported"
    elif primary_best_improved or primary_default_improved:
        issue("WARN", "phase310_primary_partial_rollout_improvement", {
            "best": improved_best,
            "default": improved_default,
        })
        root_cause = "phase310_primary_partial_rollout_improvement"
    elif len(improved_best) >= 2 or len(improved_default) >= 2:
        issue("WARN", "phase310_nonprimary_rollout_improvement", {
            "best": improved_best,
            "default": improved_default,
        })
        root_cause = "phase310_nonprimary_rollout_improvement"
    else:
        issue("FAIL", "phase310_repaired_rollout_not_supported", {
            "best": improved_best,
            "default": improved_default,
        })
        root_cause = "phase310_repaired_rollout_not_supported"

    if progress and progress.get("status") != "completed":
        issue("WARN", "rollout_not_fully_completed", progress.get("status"))

    try:
        completed_fraction = float(progress.get("completed", 0)) / float(progress.get("total_specs", 1))
    except Exception:
        completed_fraction = None

    has_fail = any(i["level"] == "FAIL" for i in issues)
    has_warn = any(i["level"] == "WARN" for i in issues)
    verdict = "FAIL" if has_fail and "supported" not in root_cause else ("WARN" if has_warn or has_fail else "PASS")

    payload = {
        "verdict": verdict,
        "root_cause": root_cause,
        "num_rows": len(rows),
        "completed_fraction": completed_fraction,
        "improved_default_conditions": improved_default,
        "improved_best_conditions": improved_best,
        "primary_best_improved": primary_best_improved,
        "primary_default_improved": primary_default_improved,
        "condition_table": condition_table,
        "table": table,
        "issues": issues,
        "progress": progress,
        "raw_summary": raw,
        "important_note": "This is controlled learned rollout retry only. No Phase4/CPS and no paper-level evidence.",
        "recommendation": (
            "Do not enter Phase4/CPS. Next run Phase3.10b repeat/diagnostic rollout or predicted-future error audit."
            if "supported" in root_cause else
            "Do not enter Phase4/CPS. Repaired IDM did not improve learned rollout; inspect DDPM predicted future quality and closed-loop history update."
        ),
    }

    out_json = root / args.out_json
    out_md = root / args.out_md
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True))

    lines = [
        "# Phase3.10 Controlled Learned Rollout Retry Report",
        "",
        "## Verdict",
        "",
        f"- Verdict: `{verdict}`",
        f"- Root cause: `{root_cause}`",
        f"- Rows: `{len(rows)}`",
        f"- Progress status: `{progress.get('status', 'missing')}`",
        f"- Improved default conditions: `{improved_default}`",
        f"- Improved best conditions: `{improved_best}`",
        f"- Primary default improved: `{primary_default_improved}`",
        f"- Primary best improved: `{primary_best_improved}`",
        "",
        "## Per-Condition Learned Rollout",
        "",
        "| Condition | Old final fraction | Default final fraction | Best final fraction | Default improvement | Best improvement | Default improved | Best improved |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for item in condition_table:
        lines.append(
            f"| `{item['condition']}` | {item['old_final_fraction']:.4f} | {item['default_final_fraction']:.4f} | "
            f"{item['best_final_fraction']:.4f} | {item['default_improvement']:.4f} | {item['best_improvement']:.4f} | "
            f"`{item['default_improved']}` | `{item['best_improved']}` |"
        )

    lines += [
        "",
        "## Summary Table",
        "",
        "| Policy | Condition | Rows | OK | Timeout | Success | Final fraction | Steps | Action OOD | Future std | Pull len | Failures |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in table:
        lines.append(
            f"| `{item['policy']}` | `{item['condition']}` | {item['rows']} | {item['ok_rows']} | {item['timeout_rows']} | "
            f"{item['success_rate']:.3f} | {item['mean_final_fraction']:.4f} | {item['mean_steps']:.2f} | "
            f"{item['mean_action_ood']:.3f} | {item['mean_future_std']:.4f} | {item['mean_pull_xy_len']:.4f} | {item['failure_count']} |"
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
        "- This tests DDPM predicted future + repaired IDM in a small closed-loop runtime.",
        "- It does not run Phase4 or CPS.",
        "- It does not prove paper-level policy success.",
        "- If repaired rollout improves primary branch, next step is still Phase3.10b repeat/error audit, not CPS.",
    ]

    out_md.write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))

    if verdict == "FAIL":
        raise SystemExit("[Phase3.10][FAIL] learned rollout retry did not support repaired pipeline")


if __name__ == "__main__":
    main()