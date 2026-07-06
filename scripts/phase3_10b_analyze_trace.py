#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

POLICIES = ["old_state_action", "phase39_default_geometry", "phase39b_xy_only_high_weight"]
CONDITIONS = ["free", "hidden_breakaway_pin", "hidden_high_friction"]


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


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def rows_for(rows: List[Dict[str, str]], policy: str, condition: str) -> List[Dict[str, str]]:
    return [r for r in rows if r.get("policy") == policy and r.get("condition") == condition]


def ok_rows_for(rows: List[Dict[str, str]], policy: str, condition: str) -> List[Dict[str, str]]:
    return [r for r in rows_for(rows, policy, condition) if r.get("status") == "ok" and not r.get("failure_reason")]


def metric(rows: List[Dict[str, str]], policy: str, condition: str, key: str) -> float:
    return mean([sf(r.get(key)) for r in ok_rows_for(rows, policy, condition)])


def step_metric(step_rows: List[Dict[str, str]], policy: str, condition: str, key: str) -> float:
    rs = [r for r in step_rows if r.get("policy") == policy and r.get("condition") == condition]
    return mean([sf(r.get(key)) for r in rs])


def count_ok(rows: List[Dict[str, str]], policy: str, condition: str) -> int:
    return len(ok_rows_for(rows, policy, condition))


def aggregate_table(episodes: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    out = []
    for policy in POLICIES:
        for cond in CONDITIONS:
            rs = rows_for(episodes, policy, cond)
            ok = [r for r in rs if r.get("status") == "ok" and not r.get("failure_reason")]
            out.append({
                "policy": policy,
                "condition": cond,
                "rows": len(rs),
                "ok_rows": len(ok),
                "timeout_rows": sum(1 for r in rs if "timeout" in str(r.get("status", ""))),
                "failed_rows": sum(1 for r in rs if r.get("status") != "ok" or r.get("failure_reason")),
                "success_rate": mean([1.0 if pb(r.get("success")) else 0.0 for r in ok]),
                "mean_final_fraction": mean([sf(r.get("final_fraction")) for r in ok]),
                "mean_delta_fraction": mean([sf(r.get("delta_fraction")) for r in ok]),
                "mean_future_match": mean([sf(r.get("future_condition_match_rate")) for r in ok]),
                "mean_future_nn_l2": mean([sf(r.get("mean_future_nn_l2")) for r in ok]),
                "mean_input_nn_l2": mean([sf(r.get("mean_input_nn_l2")) for r in ok]),
                "mean_model_x_ood_l2": mean([sf(r.get("mean_model_x_ood_l2")) for r in ok]),
                "mean_chosen_pull_len": mean([sf(r.get("mean_chosen_pull_len")) for r in ok]),
                "mean_old_pull_len": mean([sf(r.get("mean_old_pull_len")) for r in ok]),
                "mean_default_pull_len": mean([sf(r.get("mean_default_pull_len")) for r in ok]),
                "mean_best_pull_len": mean([sf(r.get("mean_best_pull_len")) for r in ok]),
                "mean_chosen_action_ood": mean([sf(r.get("mean_chosen_action_ood")) for r in ok]),
                "dominant_nearest_future_condition": "",
            })
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--episode_csv", default="reports/phase3_10b_future_rollout_audit_episodes.csv")
    parser.add_argument("--step_csv", default="reports/phase3_10b_future_rollout_audit_steps.csv")
    parser.add_argument("--progress_json", default="reports/phase3_10b_future_rollout_audit_progress.json")
    parser.add_argument("--raw_summary", default="reports/phase3_10b_future_rollout_audit_raw_summary.json")
    parser.add_argument("--out_json", default="reports/phase3_10b_future_rollout_audit_summary.json")
    parser.add_argument("--out_md", default="reports/phase3_10b_future_rollout_audit_report.md")
    parser.add_argument("--min_ok_rows_per_policy_condition", type=int, default=2)
    parser.add_argument("--future_match_low_threshold", type=float, default=0.50)
    parser.add_argument("--pull_gap_threshold", type=float, default=0.05)
    parser.add_argument("--rollout_improve_threshold", type=float, default=0.05)
    args = parser.parse_args()

    root = Path(args.root)
    episodes = read_csv(root / args.episode_csv)
    steps = read_csv(root / args.step_csv)
    progress = json.loads((root / args.progress_json).read_text()) if (root / args.progress_json).exists() else {}
    raw = json.loads((root / args.raw_summary).read_text()) if (root / args.raw_summary).exists() else {}

    issues: List[Dict[str, Any]] = []

    def issue(level: str, name: str, detail: Any) -> None:
        issues.append({"level": level, "name": name, "detail": str(detail)})

    if not episodes:
        issue("FAIL", "no_episode_rows", "No episode rows found.")
    if not steps:
        issue("WARN", "no_step_rows", "No step rows found.")

    missing = []
    for policy in POLICIES:
        for cond in CONDITIONS:
            n = count_ok(episodes, policy, cond)
            if n < args.min_ok_rows_per_policy_condition:
                missing.append(f"{policy}/{cond}: ok={n}")
    if missing:
        issue("FAIL", "missing_required_policy_condition_rows", missing)

    timeout_rows = [r for r in episodes if "timeout" in str(r.get("status", ""))]
    failed_rows = [r for r in episodes if r.get("status") != "ok" or r.get("failure_reason")]
    if timeout_rows:
        issue("WARN", "timeouts_present", len(timeout_rows))
    if failed_rows:
        issue("WARN", "failures_present", len(failed_rows))

    table = aggregate_table(episodes)

    condition_table: List[Dict[str, Any]] = []
    for cond in CONDITIONS:
        old_ff = metric(episodes, "old_state_action", cond, "final_fraction")
        default_ff = metric(episodes, "phase39_default_geometry", cond, "final_fraction")
        best_ff = metric(episodes, "phase39b_xy_only_high_weight", cond, "final_fraction")

        old_match = metric(episodes, "old_state_action", cond, "future_condition_match_rate")
        default_match = metric(episodes, "phase39_default_geometry", cond, "future_condition_match_rate")
        best_match = metric(episodes, "phase39b_xy_only_high_weight", cond, "future_condition_match_rate")

        old_pull = metric(episodes, "old_state_action", cond, "mean_chosen_pull_len")
        default_pull = metric(episodes, "phase39_default_geometry", cond, "mean_chosen_pull_len")
        best_pull = metric(episodes, "phase39b_xy_only_high_weight", cond, "mean_chosen_pull_len")

        old_input_nn = metric(episodes, "old_state_action", cond, "mean_input_nn_l2")
        default_input_nn = metric(episodes, "phase39_default_geometry", cond, "mean_input_nn_l2")
        best_input_nn = metric(episodes, "phase39b_xy_only_high_weight", cond, "mean_input_nn_l2")

        best_imp = best_ff - old_ff if math.isfinite(best_ff) and math.isfinite(old_ff) else float("nan")
        default_imp = default_ff - old_ff if math.isfinite(default_ff) and math.isfinite(old_ff) else float("nan")

        condition_table.append({
            "condition": cond,
            "old_final_fraction": old_ff,
            "default_final_fraction": default_ff,
            "best_final_fraction": best_ff,
            "default_improvement": default_imp,
            "best_improvement": best_imp,
            "old_future_match": old_match,
            "default_future_match": default_match,
            "best_future_match": best_match,
            "old_pull_len": old_pull,
            "default_pull_len": default_pull,
            "best_pull_len": best_pull,
            "old_input_nn_l2": old_input_nn,
            "default_input_nn_l2": default_input_nn,
            "best_input_nn_l2": best_input_nn,
        })

    hidden = next((r for r in condition_table if r["condition"] == "hidden_breakaway_pin"), {})
    hidden_best_match = sf(hidden.get("best_future_match"))
    hidden_default_match = sf(hidden.get("default_future_match"))
    hidden_old_match = sf(hidden.get("old_future_match"))
    hidden_best_pull = sf(hidden.get("best_pull_len"))
    hidden_old_pull = sf(hidden.get("old_pull_len"))
    hidden_default_pull = sf(hidden.get("default_pull_len"))
    hidden_best_imp = sf(hidden.get("best_improvement"))
    hidden_default_imp = sf(hidden.get("default_improvement"))

    future_branch_mismatch = (
        math.isfinite(hidden_best_match)
        and hidden_best_match < args.future_match_low_threshold
    )
    default_future_branch_mismatch = (
        math.isfinite(hidden_default_match)
        and hidden_default_match < args.future_match_low_threshold
    )
    weak_best_pull = (
        math.isfinite(hidden_best_pull)
        and math.isfinite(hidden_old_pull)
        and hidden_best_pull < hidden_old_pull - args.pull_gap_threshold
    )
    weak_default_pull = (
        math.isfinite(hidden_default_pull)
        and math.isfinite(hidden_old_pull)
        and hidden_default_pull < hidden_old_pull - args.pull_gap_threshold
    )
    primary_best_improved = (
        math.isfinite(hidden_best_imp)
        and hidden_best_imp > args.rollout_improve_threshold
    )
    primary_default_improved = (
        math.isfinite(hidden_default_imp)
        and hidden_default_imp > args.rollout_improve_threshold
    )

    if future_branch_mismatch or default_future_branch_mismatch:
        issue("WARN", "predicted_future_branch_mismatch_supported", {
            "hidden_best_future_match": hidden_best_match,
            "hidden_default_future_match": hidden_default_match,
            "hidden_old_future_match": hidden_old_match,
        })

    if weak_best_pull or weak_default_pull:
        issue("WARN", "predicted_future_action_pull_collapse_supported", {
            "hidden_old_pull": hidden_old_pull,
            "hidden_default_pull": hidden_default_pull,
            "hidden_best_pull": hidden_best_pull,
        })

    if not primary_best_improved and not primary_default_improved:
        issue("WARN", "phase310_primary_rollout_still_not_improved", {
            "best_improvement": hidden_best_imp,
            "default_improvement": hidden_default_imp,
        })

    if progress and progress.get("status") != "completed":
        issue("WARN", "audit_not_fully_completed", progress.get("status"))

    if future_branch_mismatch:
        root_cause = "phase310b_predicted_future_branch_mismatch_supported"
    elif weak_best_pull or weak_default_pull:
        root_cause = "phase310b_predicted_future_action_geometry_collapse_supported"
    elif not primary_best_improved and not primary_default_improved:
        root_cause = "phase310b_learned_future_rollout_blocker_supported"
    else:
        root_cause = "phase310b_audit_inconclusive_or_partial_improvement"

    has_fail = any(i["level"] == "FAIL" for i in issues)
    has_warn = any(i["level"] == "WARN" for i in issues)
    verdict = "FAIL" if has_fail else ("WARN" if has_warn else "PASS")

    payload = {
        "verdict": verdict,
        "root_cause": root_cause,
        "num_episode_rows": len(episodes),
        "num_step_rows": len(steps),
        "condition_table": condition_table,
        "table": table,
        "issues": issues,
        "progress": progress,
        "raw_summary": raw,
        "diagnostic_flags": {
            "future_branch_mismatch": future_branch_mismatch,
            "default_future_branch_mismatch": default_future_branch_mismatch,
            "weak_best_pull": weak_best_pull,
            "weak_default_pull": weak_default_pull,
            "primary_best_improved": primary_best_improved,
            "primary_default_improved": primary_default_improved,
        },
        "important_note": "This is a trace/error audit only. No Phase4/CPS and no paper-level evidence.",
        "recommendation": (
            "Do not enter Phase4/CPS. Next repair or audit future DDPM conditioning / predicted future branch quality."
            if future_branch_mismatch else
            "Do not enter Phase4/CPS. Next inspect predicted-future-driven action geometry, pull length, and closed-loop history drift."
        ),
    }

    out_json = root / args.out_json
    out_md = root / args.out_md
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True))

    lines = [
        "# Phase3.10b Learned Future Quality + Rollout Error Audit Report",
        "",
        "## Verdict",
        "",
        f"- Verdict: `{verdict}`",
        f"- Root cause: `{root_cause}`",
        f"- Episode rows: `{len(episodes)}`",
        f"- Step rows: `{len(steps)}`",
        f"- Progress status: `{progress.get('status', 'missing')}`",
        "",
        "## Per-Condition Diagnosis",
        "",
        "| Condition | Old final | Default final | Best final | Default Δ | Best Δ | Old future match | Default future match | Best future match | Old pull | Default pull | Best pull |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for r in condition_table:
        lines.append(
            f"| `{r['condition']}` | {sf(r['old_final_fraction']):.4f} | {sf(r['default_final_fraction']):.4f} | "
            f"{sf(r['best_final_fraction']):.4f} | {sf(r['default_improvement']):.4f} | {sf(r['best_improvement']):.4f} | "
            f"{sf(r['old_future_match']):.3f} | {sf(r['default_future_match']):.3f} | {sf(r['best_future_match']):.3f} | "
            f"{sf(r['old_pull_len']):.4f} | {sf(r['default_pull_len']):.4f} | {sf(r['best_pull_len']):.4f} |"
        )

    lines += [
        "",
        "## Episode Summary",
        "",
        "| Policy | Condition | Rows | OK | Timeout | Success | Final fraction | Future match | Future NN L2 | Input NN L2 | Chosen pull | Action OOD | Failures |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for r in table:
        lines.append(
            f"| `{r['policy']}` | `{r['condition']}` | {r['rows']} | {r['ok_rows']} | {r['timeout_rows']} | "
            f"{sf(r['success_rate']):.3f} | {sf(r['mean_final_fraction']):.4f} | "
            f"{sf(r['mean_future_match']):.3f} | {sf(r['mean_future_nn_l2']):.4f} | "
            f"{sf(r['mean_input_nn_l2']):.4f} | {sf(r['mean_chosen_pull_len']):.4f} | "
            f"{sf(r['mean_chosen_action_ood']):.3f} | {r['failed_rows']} |"
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
        "- This is a learned-future rollout error audit only.",
        "- It uses offline y_state / condition_name only for diagnostic nearest-neighbor analysis, not model input.",
        "- No model training was run.",
        "- No future DDPM was trained.",
        "- No Phase4 or CPS was run.",
        "- If predicted future branch mismatch is supported, next step is future DDPM / conditioning diagnosis, not IDM repair or CPS.",
    ]

    out_md.write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))

    if verdict == "FAIL":
        raise SystemExit("[Phase3.10b][FAIL] audit failed or missing required rows")


if __name__ == "__main__":
    main()