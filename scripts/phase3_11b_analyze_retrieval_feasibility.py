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
VARIANTS = [
    "source_gt_action",
    "source_old_idm_future",
    "source_repaired_idm_future",
    "live_retrieved_gt_action",
    "live_repaired_idm_retrieved_future",
    "live_ddpm_mean_repaired_idm",
    "live_condition_retrieval_repaired_idm",
]


def sf(x: Any) -> float:
    try:
        if x is None or x == "":
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def mean(xs: List[float]) -> float:
    vals = [x for x in xs if math.isfinite(float(x))]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def rows_for(rows: List[Dict[str, str]], condition: str, variant: str) -> List[Dict[str, str]]:
    return [r for r in rows if r.get("condition") == condition and r.get("variant") == variant]


def ok_rows_for(rows: List[Dict[str, str]], condition: str, variant: str) -> List[Dict[str, str]]:
    return [r for r in rows_for(rows, condition, variant) if r.get("status") == "ok" and not r.get("exec_failure_reason")]


def delta(rows: List[Dict[str, str]], condition: str, variant: str) -> float:
    return mean([sf(r.get("exec_delta_final_fraction")) for r in ok_rows_for(rows, condition, variant)])


def final_after(rows: List[Dict[str, str]], condition: str, variant: str) -> float:
    return mean([sf(r.get("exec_final_fraction_after")) for r in ok_rows_for(rows, condition, variant)])


def pull(rows: List[Dict[str, str]], condition: str, variant: str) -> float:
    return mean([sf(r.get("pull_xy_len")) for r in ok_rows_for(rows, condition, variant)])


def action_ood(rows: List[Dict[str, str]], condition: str, variant: str) -> float:
    return mean([sf(r.get("action_ood")) for r in ok_rows_for(rows, condition, variant)])


def context_l2(rows: List[Dict[str, str]], condition: str, variant: str) -> float:
    return mean([sf(r.get("current_model_x_to_retrieved_x_l2")) for r in ok_rows_for(rows, condition, variant)])


def ok_count(rows: List[Dict[str, str]], condition: str, variant: str) -> int:
    return len(ok_rows_for(rows, condition, variant))


def aggregate_table(rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    out = []
    for cond in CONDITIONS:
        for variant in VARIANTS:
            group = rows_for(rows, cond, variant)
            ok = [r for r in group if r.get("status") == "ok" and not r.get("exec_failure_reason")]
            out.append({
                "condition": cond,
                "variant": variant,
                "rows": len(group),
                "ok_rows": len(ok),
                "timeout_rows": sum(1 for r in group if "timeout" in str(r.get("status", ""))),
                "failed_rows": sum(1 for r in group if r.get("status") != "ok" or r.get("exec_failure_reason")),
                "mean_delta": mean([sf(r.get("exec_delta_final_fraction")) for r in ok]),
                "mean_final_after": mean([sf(r.get("exec_final_fraction_after")) for r in ok]),
                "mean_pull": mean([sf(r.get("pull_xy_len")) for r in ok]),
                "mean_action_ood": mean([sf(r.get("action_ood")) for r in ok]),
                "mean_context_l2": mean([sf(r.get("current_model_x_to_retrieved_x_l2")) for r in ok]),
            })
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--trials_csv", default="reports/phase3_11b_retrieval_feasibility_trials.csv")
    parser.add_argument("--progress_json", default="reports/phase3_11b_retrieval_feasibility_progress.json")
    parser.add_argument("--raw_summary", default="reports/phase3_11b_retrieval_feasibility_raw_summary.json")
    parser.add_argument("--out_json", default="reports/phase3_11b_retrieval_feasibility_summary.json")
    parser.add_argument("--out_md", default="reports/phase3_11b_retrieval_feasibility_report.md")
    parser.add_argument("--primary_condition", default="hidden_breakaway_pin")
    parser.add_argument("--progress_threshold", type=float, default=0.05)
    parser.add_argument("--min_ok_rows_per_variant_condition", type=int, default=2)
    args = parser.parse_args()

    root = Path(args.root)
    rows = read_csv_rows(root / args.trials_csv)
    progress = json.loads((root / args.progress_json).read_text()) if (root / args.progress_json).exists() else {}
    raw = json.loads((root / args.raw_summary).read_text()) if (root / args.raw_summary).exists() else {}

    issues: List[Dict[str, Any]] = []

    def issue(level: str, name: str, detail: Any) -> None:
        issues.append({"level": level, "name": name, "detail": str(detail)})

    if not rows:
        issue("FAIL", "no_rows", "No rows found.")

    missing = []
    for cond in CONDITIONS:
        for variant in VARIANTS:
            n = ok_count(rows, cond, variant)
            if n < args.min_ok_rows_per_variant_condition:
                missing.append(f"{cond}/{variant}: ok={n}")
    if missing:
        issue("FAIL", "missing_required_rows", missing)

    table = aggregate_table(rows)

    condition_table: List[Dict[str, Any]] = []
    for cond in CONDITIONS:
        item = {
            "condition": cond,
            "source_gt_delta": delta(rows, cond, "source_gt_action"),
            "source_old_idm_delta": delta(rows, cond, "source_old_idm_future"),
            "source_repaired_idm_delta": delta(rows, cond, "source_repaired_idm_future"),
            "live_retrieved_gt_delta": delta(rows, cond, "live_retrieved_gt_action"),
            "live_repaired_retrieved_delta": delta(rows, cond, "live_repaired_idm_retrieved_future"),
            "live_ddpm_mean_delta": delta(rows, cond, "live_ddpm_mean_repaired_idm"),
            "live_condition_retrieval_delta": delta(rows, cond, "live_condition_retrieval_repaired_idm"),
            "source_gt_pull": pull(rows, cond, "source_gt_action"),
            "source_repaired_pull": pull(rows, cond, "source_repaired_idm_future"),
            "live_retrieved_gt_pull": pull(rows, cond, "live_retrieved_gt_action"),
            "live_repaired_retrieved_pull": pull(rows, cond, "live_repaired_idm_retrieved_future"),
            "live_ddpm_mean_pull": pull(rows, cond, "live_ddpm_mean_repaired_idm"),
            "live_context_l2": context_l2(rows, cond, "live_repaired_idm_retrieved_future"),
        }
        item["source_gt_effective"] = sf(item["source_gt_delta"]) > args.progress_threshold
        item["source_repaired_effective"] = sf(item["source_repaired_idm_delta"]) > args.progress_threshold
        item["live_gt_effective"] = sf(item["live_retrieved_gt_delta"]) > args.progress_threshold
        item["live_repaired_effective"] = sf(item["live_repaired_retrieved_delta"]) > args.progress_threshold
        condition_table.append(item)

    primary = next((x for x in condition_table if x["condition"] == args.primary_condition), {})
    source_gt_effective = bool(primary.get("source_gt_effective", False))
    source_repaired_effective = bool(primary.get("source_repaired_effective", False))
    live_gt_effective = bool(primary.get("live_gt_effective", False))
    live_repaired_effective = bool(primary.get("live_repaired_effective", False))

    if not source_gt_effective:
        issue("WARN", "retrieved_source_gt_action_not_effective_on_primary", primary)
    if source_gt_effective and not live_gt_effective:
        issue("WARN", "retrieved_action_transfer_to_live_context_fails", primary)
    if source_repaired_effective and not live_repaired_effective:
        issue("WARN", "repaired_idm_retrieved_future_transfer_to_live_context_fails", primary)
    if source_gt_effective and not source_repaired_effective:
        issue("WARN", "repaired_idm_not_matching_retrieved_action_in_source_context", primary)
    if not source_gt_effective and not source_repaired_effective:
        issue("WARN", "retrieved_future_action_source_context_not_executable", primary)

    if source_gt_effective and source_repaired_effective and (not live_gt_effective or not live_repaired_effective):
        root_cause = "phase311b_live_source_context_mismatch_supported"
    elif not source_gt_effective and not source_repaired_effective:
        root_cause = "phase311b_retrieved_source_not_executable_or_bad_candidate"
    elif source_gt_effective and not source_repaired_effective:
        root_cause = "phase311b_repaired_idm_retrieved_future_incompatibility_supported"
    elif live_gt_effective or live_repaired_effective:
        root_cause = "phase311b_retrieval_transfer_partially_supported"
    else:
        root_cause = "phase311b_retrieval_feasibility_inconclusive"

    if progress and progress.get("status") != "completed":
        issue("WARN", "probe_not_completed", progress.get("status"))

    has_fail = any(i["level"] == "FAIL" for i in issues)
    has_warn = any(i["level"] == "WARN" for i in issues)
    verdict = "FAIL" if has_fail else ("WARN" if has_warn else "PASS")

    payload = {
        "verdict": verdict,
        "root_cause": root_cause,
        "num_rows": len(rows),
        "condition_table": condition_table,
        "table": table,
        "issues": issues,
        "progress": progress,
        "raw_summary": raw,
        "important_note": "Phase3.11b is a diagnostic feasibility audit only. No Phase4/CPS and no model training.",
        "recommendation": (
            "Do not enter Phase4/CPS. If live-source context mismatch is supported, next diagnose closed-loop state/history compatibility and future target locality."
            if root_cause == "phase311b_live_source_context_mismatch_supported"
            else "Do not enter Phase4/CPS. Inspect retrieved candidates, source executability, and IDM/future compatibility before CPS."
        ),
    }

    out_json = root / args.out_json
    out_md = root / args.out_md
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True))

    lines = [
        "# Phase3.11b Future Target Compatibility + Retrieved Action Feasibility Report",
        "",
        "## Verdict",
        "",
        f"- Verdict: `{verdict}`",
        f"- Root cause: `{root_cause}`",
        f"- Rows: `{len(rows)}`",
        f"- Progress status: `{progress.get('status', 'missing')}`",
        "",
        "## Per-Condition Compatibility Diagnosis",
        "",
        "| Condition | Source GT Δ | Source repaired Δ | Live GT Δ | Live repaired Δ | Live DDPM Δ | Live condition retrieval Δ | Source GT effective | Source repaired effective | Live GT effective | Live repaired effective | Live context L2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for r in condition_table:
        lines.append(
            f"| `{r['condition']}` | {sf(r['source_gt_delta']):.4f} | {sf(r['source_repaired_idm_delta']):.4f} | "
            f"{sf(r['live_retrieved_gt_delta']):.4f} | {sf(r['live_repaired_retrieved_delta']):.4f} | "
            f"{sf(r['live_ddpm_mean_delta']):.4f} | {sf(r['live_condition_retrieval_delta']):.4f} | "
            f"`{r['source_gt_effective']}` | `{r['source_repaired_effective']}` | "
            f"`{r['live_gt_effective']}` | `{r['live_repaired_effective']}` | {sf(r['live_context_l2']):.4f} |"
        )

    lines += [
        "",
        "## Variant Summary",
        "",
        "| Condition | Variant | Rows | OK | Timeout | Delta | Final after | Pull | Action OOD | Context L2 | Failures |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for r in table:
        lines.append(
            f"| `{r['condition']}` | `{r['variant']}` | {r['rows']} | {r['ok_rows']} | {r['timeout_rows']} | "
            f"{sf(r['mean_delta']):.4f} | {sf(r['mean_final_after']):.4f} | {sf(r['mean_pull']):.4f} | "
            f"{sf(r['mean_action_ood']):.3f} | {sf(r['mean_context_l2']):.4f} | {r['failed_rows']} |"
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
        "- This audit tests whether retrieved future/action is executable in its source prefix and transferable to current live prefix.",
        "- Retrieved y_state / y_action / condition labels are diagnostic only and not deployable policy inputs.",
        "- No model training was run.",
        "- No future DDPM was trained.",
        "- No Phase4 or CPS was run.",
    ]

    out_md.write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))

    if verdict == "FAIL":
        raise SystemExit("[Phase3.11b][FAIL] missing rows or probe failed")


if __name__ == "__main__":
    main()
