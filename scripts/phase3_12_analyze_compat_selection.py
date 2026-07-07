#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List

SELECTORS = [
    "ddpm_mean",
    "input_nearest",
    "condition_nearest",
    "compat_global_topk",
    "compat_condition_topk",
    "compat_condition_action_geom",
]
CONDITIONS = ["free", "hidden_breakaway_pin", "hidden_high_friction"]


def sf(x: Any) -> float:
    try:
        if x is None or x == "":
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def mean(xs: List[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def rows_for(rows: List[Dict[str, str]], selector: str, condition: str) -> List[Dict[str, str]]:
    return [r for r in rows if r.get("selector") == selector and r.get("condition") == condition]


def ok_rows_for(rows: List[Dict[str, str]], selector: str, condition: str) -> List[Dict[str, str]]:
    return [r for r in rows_for(rows, selector, condition) if r.get("status") == "ok" and not r.get("failure_reason")]


def metric(rows: List[Dict[str, str]], selector: str, condition: str, key: str) -> float:
    return mean([sf(r.get(key)) for r in ok_rows_for(rows, selector, condition)])


def ok_count(rows: List[Dict[str, str]], selector: str, condition: str) -> int:
    return len(ok_rows_for(rows, selector, condition))


def aggregate_table(rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for selector in SELECTORS:
        for condition in CONDITIONS:
            rs = rows_for(rows, selector, condition)
            ok = ok_rows_for(rows, selector, condition)
            out.append({
                "selector": selector,
                "condition": condition,
                "rows": len(rs),
                "ok_rows": len(ok),
                "timeout_rows": sum(1 for r in rs if "timeout" in str(r.get("status", ""))),
                "failed_rows": sum(1 for r in rs if r.get("status") != "ok" or r.get("failure_reason")),
                "uses_condition_label": int(max([sf(r.get("uses_condition_label")) for r in ok], default=0)),
                "mean_final_fraction": mean([sf(r.get("final_fraction")) for r in ok]),
                "mean_delta_fraction": mean([sf(r.get("delta_fraction")) for r in ok]),
                "mean_future_match": mean([sf(r.get("mean_future_match")) for r in ok]),
                "mean_selected_context_l2": mean([sf(r.get("mean_selected_context_l2")) for r in ok]),
                "mean_selected_score": mean([sf(r.get("mean_selected_score")) for r in ok]),
                "mean_action_ood": mean([sf(r.get("mean_action_ood")) for r in ok]),
                "mean_clip_frac": mean([sf(r.get("mean_clip_frac")) for r in ok]),
                "mean_pull_len": mean([sf(r.get("mean_pull_len")) for r in ok]),
                "mean_source_pull_len": mean([sf(r.get("mean_source_pull_len")) for r in ok]),
                "mean_action_mae_to_source": mean([sf(r.get("mean_action_mae_to_source")) for r in ok]),
                "mean_pull_diff_to_source": mean([sf(r.get("mean_pull_diff_to_source")) for r in ok]),
            })
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--episode_csv", default="reports/phase3_12_compat_selection_episodes.csv")
    parser.add_argument("--step_csv", default="reports/phase3_12_compat_selection_steps.csv")
    parser.add_argument("--progress_json", default="reports/phase3_12_compat_selection_progress.json")
    parser.add_argument("--raw_summary", default="reports/phase3_12_compat_selection_raw_summary.json")
    parser.add_argument("--out_json", default="reports/phase3_12_compat_selection_summary.json")
    parser.add_argument("--out_md", default="reports/phase3_12_compat_selection_report.md")
    parser.add_argument("--primary_condition", default="hidden_breakaway_pin")
    parser.add_argument("--improve_threshold", type=float, default=0.05)
    parser.add_argument("--min_ok_rows_per_selector_condition", type=int, default=2)
    args = parser.parse_args()

    root = Path(args.root)
    episodes = read_csv_rows(root / args.episode_csv)
    steps = read_csv_rows(root / args.step_csv)
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
    for selector in SELECTORS:
        for condition in CONDITIONS:
            n = ok_count(episodes, selector, condition)
            if n < args.min_ok_rows_per_selector_condition:
                missing.append(f"{selector}/{condition}: ok={n}")
    if missing:
        issue("FAIL", "missing_required_selector_condition_rows", missing)

    table = aggregate_table(episodes)

    condition_table: List[Dict[str, Any]] = []
    for condition in CONDITIONS:
        base = metric(episodes, "ddpm_mean", condition, "final_fraction")
        row = {
            "condition": condition,
            "ddpm_mean_final": base,
            "input_nearest_final": metric(episodes, "input_nearest", condition, "final_fraction"),
            "condition_nearest_final": metric(episodes, "condition_nearest", condition, "final_fraction"),
            "compat_global_topk_final": metric(episodes, "compat_global_topk", condition, "final_fraction"),
            "compat_condition_topk_final": metric(episodes, "compat_condition_topk", condition, "final_fraction"),
            "compat_condition_action_geom_final": metric(episodes, "compat_condition_action_geom", condition, "final_fraction"),
            "ddpm_mean_pull": metric(episodes, "ddpm_mean", condition, "mean_pull_len"),
            "compat_global_pull": metric(episodes, "compat_global_topk", condition, "mean_pull_len"),
            "compat_condition_pull": metric(episodes, "compat_condition_topk", condition, "mean_pull_len"),
            "compat_action_geom_pull": metric(episodes, "compat_condition_action_geom", condition, "mean_pull_len"),
            "ddpm_mean_future_match": metric(episodes, "ddpm_mean", condition, "mean_future_match"),
            "condition_nearest_future_match": metric(episodes, "condition_nearest", condition, "mean_future_match"),
            "compat_condition_future_match": metric(episodes, "compat_condition_topk", condition, "mean_future_match"),
            "compat_action_geom_future_match": metric(episodes, "compat_condition_action_geom", condition, "mean_future_match"),
            "compat_global_context_l2": metric(episodes, "compat_global_topk", condition, "mean_selected_context_l2"),
            "compat_condition_context_l2": metric(episodes, "compat_condition_topk", condition, "mean_selected_context_l2"),
            "compat_action_geom_context_l2": metric(episodes, "compat_condition_action_geom", condition, "mean_selected_context_l2"),
        }
        for key in [
            "input_nearest_final",
            "condition_nearest_final",
            "compat_global_topk_final",
            "compat_condition_topk_final",
            "compat_condition_action_geom_final",
        ]:
            row[key.replace("_final", "_improvement")] = row[key] - base if math.isfinite(row[key]) and math.isfinite(base) else float("nan")
        condition_table.append(row)

    primary = next((r for r in condition_table if r["condition"] == args.primary_condition), {})
    g_gain = sf(primary.get("compat_global_topk_improvement"))
    c_gain = sf(primary.get("compat_condition_topk_improvement"))
    a_gain = sf(primary.get("compat_condition_action_geom_improvement"))
    in_gain = sf(primary.get("input_nearest_improvement"))
    cn_gain = sf(primary.get("condition_nearest_improvement"))

    compat_global_supported = math.isfinite(g_gain) and g_gain > args.improve_threshold
    compat_condition_supported = math.isfinite(c_gain) and c_gain > args.improve_threshold
    compat_action_geom_supported = math.isfinite(a_gain) and a_gain > args.improve_threshold
    simple_input_supported = math.isfinite(in_gain) and in_gain > args.improve_threshold
    simple_condition_supported = math.isfinite(cn_gain) and cn_gain > args.improve_threshold

    if compat_global_supported:
        issue("WARN", "compat_global_topk_improves_primary", {"gain": g_gain})
    if compat_condition_supported:
        issue("WARN", "compat_condition_topk_improves_primary_upper_bound", {"gain": c_gain})
    if compat_action_geom_supported:
        issue("WARN", "compat_condition_action_geom_improves_primary_upper_bound", {"gain": a_gain})
    if simple_input_supported:
        issue("WARN", "input_nearest_improves_primary", {"gain": in_gain})
    if simple_condition_supported:
        issue("WARN", "condition_nearest_improves_primary_upper_bound", {"gain": cn_gain})

    if progress and progress.get("status") != "completed":
        issue("WARN", "rollout_not_completed", progress.get("status"))

    if compat_global_supported:
        root_cause = "phase312_deployable_locality_action_compat_supported"
    elif compat_condition_supported or compat_action_geom_supported:
        root_cause = "phase312_conditioned_compat_selection_upper_bound_supported"
    elif simple_input_supported or simple_condition_supported:
        root_cause = "phase312_simple_locality_selection_supported"
    else:
        root_cause = "phase312_compat_selection_not_supported_or_inconclusive"
        issue("WARN", "compat_selection_not_clearly_supported", primary)

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
            "compat_global_supported": compat_global_supported,
            "compat_condition_supported": compat_condition_supported,
            "compat_action_geom_supported": compat_action_geom_supported,
            "simple_input_supported": simple_input_supported,
            "simple_condition_supported": simple_condition_supported,
        },
        "important_note": "Phase3.12 is diagnostic future selection only. Condition-aware selectors use labels as upper-bound diagnostics. No Phase4/CPS.",
        "recommendation": (
            "Do not enter Phase4/CPS. If deployable locality/action compatibility is supported, next repeat and then prototype future branch selector."
            if compat_global_supported else
            "Do not enter Phase4/CPS. If only condition-aware upper bound works, next design contact/locality proxy before CPS."
            if (compat_condition_supported or compat_action_geom_supported) else
            "Do not enter Phase4/CPS. Inspect step traces and compatibility score components."
        ),
    }

    out_json = root / args.out_json
    out_md = root / args.out_md
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True))

    lines = [
        "# Phase3.12 Future Target Locality / Compatibility-Aware Selection Report",
        "",
        "## Verdict",
        "",
        f"- Verdict: `{verdict}`",
        f"- Root cause: `{root_cause}`",
        f"- Episode rows: `{len(episodes)}`",
        f"- Step rows: `{len(steps)}`",
        f"- Progress status: `{progress.get('status', 'missing')}`",
        "",
        "## Per-Condition Selector Comparison",
        "",
        "| Condition | DDPM mean | Input NN | Condition NN | Compat global | Compat condition | Compat action-geom | Input Δ | Condition Δ | Compat global Δ | Compat condition Δ | Compat action-geom Δ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for r in condition_table:
        lines.append(
            f"| `{r['condition']}` | {sf(r['ddpm_mean_final']):.4f} | {sf(r['input_nearest_final']):.4f} | "
            f"{sf(r['condition_nearest_final']):.4f} | {sf(r['compat_global_topk_final']):.4f} | "
            f"{sf(r['compat_condition_topk_final']):.4f} | {sf(r['compat_condition_action_geom_final']):.4f} | "
            f"{sf(r['input_nearest_improvement']):.4f} | {sf(r['condition_nearest_improvement']):.4f} | "
            f"{sf(r['compat_global_topk_improvement']):.4f} | {sf(r['compat_condition_topk_improvement']):.4f} | "
            f"{sf(r['compat_condition_action_geom_improvement']):.4f} |"
        )

    lines += [
        "",
        "## Pull / Match / Context Diagnostics",
        "",
        "| Condition | DDPM pull | Compat global pull | Compat condition pull | Action-geom pull | DDPM match | Condition match | Compat condition match | Compat global context L2 | Compat condition context L2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for r in condition_table:
        lines.append(
            f"| `{r['condition']}` | {sf(r['ddpm_mean_pull']):.4f} | {sf(r['compat_global_pull']):.4f} | "
            f"{sf(r['compat_condition_pull']):.4f} | {sf(r['compat_action_geom_pull']):.4f} | "
            f"{sf(r['ddpm_mean_future_match']):.3f} | {sf(r['condition_nearest_future_match']):.3f} | "
            f"{sf(r['compat_condition_future_match']):.3f} | {sf(r['compat_global_context_l2']):.4f} | "
            f"{sf(r['compat_condition_context_l2']):.4f} |"
        )

    lines += [
        "",
        "## Episode Summary",
        "",
        "| Selector | Condition | Rows | OK | Timeout | Uses condition | Final | Δ | Future match | Context L2 | Score | Pull | OOD | Clip | Action MAE | Pull diff | Failures |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for r in table:
        lines.append(
            f"| `{r['selector']}` | `{r['condition']}` | {r['rows']} | {r['ok_rows']} | {r['timeout_rows']} | "
            f"{r['uses_condition_label']} | {sf(r['mean_final_fraction']):.4f} | {sf(r['mean_delta_fraction']):.4f} | "
            f"{sf(r['mean_future_match']):.3f} | {sf(r['mean_selected_context_l2']):.4f} | {sf(r['mean_selected_score']):.4f} | "
            f"{sf(r['mean_pull_len']):.4f} | {sf(r['mean_action_ood']):.3f} | {sf(r['mean_clip_frac']):.3f} | "
            f"{sf(r['mean_action_mae_to_source']):.4f} | {sf(r['mean_pull_diff_to_source']):.4f} | {r['failed_rows']} |"
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
        "- This is a compatibility-aware future target selection diagnostic only.",
        "- No model training was run.",
        "- No future DDPM was trained.",
        "- No Phase4 or CPS was run.",
        "- Condition-aware selectors are diagnostic upper bounds and not deployable policy evidence.",
        "- If only condition-aware compatibility works, the next step is a contact/locality proxy diagnostic, not full CPS.",
    ]

    out_md.write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))

    if verdict == "FAIL":
        raise SystemExit("[Phase3.12][FAIL] compatibility selection failed or missing rows")


if __name__ == "__main__":
    main()
