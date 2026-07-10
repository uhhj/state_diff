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
    "condition_nearest_upper",
    "proxy_action_nn",
    "proxy_state_motion_nn",
    "proxy_combined_nn",
    "proxy_combined_topk_action_geom",
    "proxy_combined_topk_no_action_geom",
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
                "mean_proxy_distance": mean([sf(r.get("mean_proxy_distance")) for r in ok]),
                "mean_action_ood": mean([sf(r.get("mean_action_ood")) for r in ok]),
                "mean_pull_len": mean([sf(r.get("mean_pull_len")) for r in ok]),
                "mean_source_pull_len": mean([sf(r.get("mean_source_pull_len")) for r in ok]),
                "mean_action_mae_to_source": mean([sf(r.get("mean_action_mae_to_source")) for r in ok]),
                "mean_pull_diff_to_source": mean([sf(r.get("mean_pull_diff_to_source")) for r in ok]),
            })
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--episode_csv", default="reports/phase3_12b_proxy_score_episodes.csv")
    parser.add_argument("--step_csv", default="reports/phase3_12b_proxy_score_steps.csv")
    parser.add_argument("--progress_json", default="reports/phase3_12b_proxy_score_progress.json")
    parser.add_argument("--raw_summary", default="reports/phase3_12b_proxy_score_raw_summary.json")
    parser.add_argument("--out_json", default="reports/phase3_12b_proxy_score_summary.json")
    parser.add_argument("--out_md", default="reports/phase3_12b_proxy_score_report.md")
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
        ddpm = metric(episodes, "ddpm_mean", condition, "final_fraction")
        row = {
            "condition": condition,
            "ddpm_mean_final": ddpm,
            "condition_nearest_upper_final": metric(episodes, "condition_nearest_upper", condition, "final_fraction"),
            "proxy_action_nn_final": metric(episodes, "proxy_action_nn", condition, "final_fraction"),
            "proxy_state_motion_nn_final": metric(episodes, "proxy_state_motion_nn", condition, "final_fraction"),
            "proxy_combined_nn_final": metric(episodes, "proxy_combined_nn", condition, "final_fraction"),
            "proxy_combined_topk_action_geom_final": metric(episodes, "proxy_combined_topk_action_geom", condition, "final_fraction"),
            "proxy_combined_topk_no_action_geom_final": metric(episodes, "proxy_combined_topk_no_action_geom", condition, "final_fraction"),
            "ddpm_mean_pull": metric(episodes, "ddpm_mean", condition, "mean_pull_len"),
            "condition_nearest_upper_pull": metric(episodes, "condition_nearest_upper", condition, "mean_pull_len"),
            "proxy_action_nn_pull": metric(episodes, "proxy_action_nn", condition, "mean_pull_len"),
            "proxy_state_motion_nn_pull": metric(episodes, "proxy_state_motion_nn", condition, "mean_pull_len"),
            "proxy_combined_nn_pull": metric(episodes, "proxy_combined_nn", condition, "mean_pull_len"),
            "proxy_topk_action_geom_pull": metric(episodes, "proxy_combined_topk_action_geom", condition, "mean_pull_len"),
            "proxy_topk_no_action_geom_pull": metric(episodes, "proxy_combined_topk_no_action_geom", condition, "mean_pull_len"),
            "ddpm_match": metric(episodes, "ddpm_mean", condition, "mean_future_match"),
            "condition_upper_match": metric(episodes, "condition_nearest_upper", condition, "mean_future_match"),
            "proxy_action_match": metric(episodes, "proxy_action_nn", condition, "mean_future_match"),
            "proxy_state_match": metric(episodes, "proxy_state_motion_nn", condition, "mean_future_match"),
            "proxy_combined_match": metric(episodes, "proxy_combined_nn", condition, "mean_future_match"),
            "proxy_topk_action_geom_match": metric(episodes, "proxy_combined_topk_action_geom", condition, "mean_future_match"),
            "proxy_topk_no_action_geom_match": metric(episodes, "proxy_combined_topk_no_action_geom", condition, "mean_future_match"),
        }
        for key in [
            "condition_nearest_upper_final",
            "proxy_action_nn_final",
            "proxy_state_motion_nn_final",
            "proxy_combined_nn_final",
            "proxy_combined_topk_action_geom_final",
            "proxy_combined_topk_no_action_geom_final",
        ]:
            row[key.replace("_final", "_improvement")] = row[key] - ddpm if math.isfinite(row[key]) and math.isfinite(ddpm) else float("nan")
        condition_table.append(row)

    primary = next((x for x in condition_table if x["condition"] == args.primary_condition), {})
    cond_gain = sf(primary.get("condition_nearest_upper_improvement"))
    action_gain = sf(primary.get("proxy_action_nn_improvement"))
    state_gain = sf(primary.get("proxy_state_motion_nn_improvement"))
    combined_gain = sf(primary.get("proxy_combined_nn_improvement"))
    topk_geom_gain = sf(primary.get("proxy_combined_topk_action_geom_improvement"))
    topk_no_geom_gain = sf(primary.get("proxy_combined_topk_no_action_geom_improvement"))

    condition_upper_supported = math.isfinite(cond_gain) and cond_gain > args.improve_threshold
    proxy_action_supported = math.isfinite(action_gain) and action_gain > args.improve_threshold
    proxy_state_supported = math.isfinite(state_gain) and state_gain > args.improve_threshold
    proxy_combined_supported = math.isfinite(combined_gain) and combined_gain > args.improve_threshold
    proxy_topk_geom_supported = math.isfinite(topk_geom_gain) and topk_geom_gain > args.improve_threshold
    proxy_topk_no_geom_supported = math.isfinite(topk_no_geom_gain) and topk_no_geom_gain > args.improve_threshold

    geom_score_helpful = (
        math.isfinite(topk_geom_gain)
        and math.isfinite(topk_no_geom_gain)
        and topk_geom_gain > topk_no_geom_gain + args.improve_threshold
    )

    if condition_upper_supported:
        issue("WARN", "condition_nearest_upper_bound_replicated_on_primary", {"gain": cond_gain})
    if proxy_action_supported:
        issue("WARN", "proxy_action_nn_improves_primary", {"gain": action_gain})
    if proxy_state_supported:
        issue("WARN", "proxy_state_motion_nn_improves_primary", {"gain": state_gain})
    if proxy_combined_supported:
        issue("WARN", "proxy_combined_nn_improves_primary", {"gain": combined_gain})
    if proxy_topk_geom_supported:
        issue("WARN", "proxy_combined_topk_action_geom_improves_primary", {"gain": topk_geom_gain})
    if proxy_topk_no_geom_supported:
        issue("WARN", "proxy_combined_topk_no_action_geom_improves_primary", {"gain": topk_no_geom_gain})
    if geom_score_helpful:
        issue("WARN", "action_geometry_score_helpful_on_primary", {
            "topk_action_geom_gain": topk_geom_gain,
            "topk_no_action_geom_gain": topk_no_geom_gain,
        })

    if progress and progress.get("status") != "completed":
        issue("WARN", "rollout_not_completed", progress.get("status"))

    any_proxy = any([
        proxy_action_supported,
        proxy_state_supported,
        proxy_combined_supported,
        proxy_topk_geom_supported,
        proxy_topk_no_geom_supported,
    ])

    if any_proxy and geom_score_helpful:
        root_cause = "phase312b_deployable_proxy_plus_action_geom_supported"
    elif any_proxy:
        root_cause = "phase312b_deployable_observable_proxy_supported"
    elif condition_upper_supported:
        root_cause = "phase312b_condition_upper_bound_replicated_but_proxy_not_supported"
    else:
        root_cause = "phase312b_proxy_score_not_supported_or_inconclusive"
        issue("WARN", "proxy_score_not_clearly_supported", primary)

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
            "condition_upper_supported": condition_upper_supported,
            "proxy_action_supported": proxy_action_supported,
            "proxy_state_supported": proxy_state_supported,
            "proxy_combined_supported": proxy_combined_supported,
            "proxy_topk_geom_supported": proxy_topk_geom_supported,
            "proxy_topk_no_geom_supported": proxy_topk_no_geom_supported,
            "geom_score_helpful": geom_score_helpful,
        },
        "important_note": "Phase3.12b is diagnostic proxy/score ablation only. condition_nearest_upper uses labels as upper-bound diagnostics. No Phase4/CPS.",
        "recommendation": (
            "Do not enter Phase4/CPS. If deployable proxy is supported, next repeat Phase3.12c and then consider a branch-selector prototype."
            if any_proxy else
            "Do not enter Phase4/CPS. If only condition upper bound works, next design a learned/online contact proxy before CPS."
        ),
    }

    out_json = root / args.out_json
    out_md = root / args.out_md
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True))

    lines = [
        "# Phase3.12b Condition Proxy / Score Ablation Diagnostic Report",
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
        "| Condition | DDPM mean | Condition upper | Proxy action | Proxy state | Proxy combined | Proxy topK geom | Proxy topK no-geom | Cond Δ | Action Δ | State Δ | Combined Δ | TopK geom Δ | TopK no-geom Δ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for r in condition_table:
        lines.append(
            f"| `{r['condition']}` | {sf(r['ddpm_mean_final']):.4f} | {sf(r['condition_nearest_upper_final']):.4f} | "
            f"{sf(r['proxy_action_nn_final']):.4f} | {sf(r['proxy_state_motion_nn_final']):.4f} | "
            f"{sf(r['proxy_combined_nn_final']):.4f} | {sf(r['proxy_combined_topk_action_geom_final']):.4f} | "
            f"{sf(r['proxy_combined_topk_no_action_geom_final']):.4f} | "
            f"{sf(r['condition_nearest_upper_improvement']):.4f} | {sf(r['proxy_action_nn_improvement']):.4f} | "
            f"{sf(r['proxy_state_motion_nn_improvement']):.4f} | {sf(r['proxy_combined_nn_improvement']):.4f} | "
            f"{sf(r['proxy_combined_topk_action_geom_improvement']):.4f} | {sf(r['proxy_combined_topk_no_action_geom_improvement']):.4f} |"
        )

    lines += [
        "",
        "## Pull / Future-Match Diagnostics",
        "",
        "| Condition | DDPM pull | Condition pull | Proxy action pull | Proxy state pull | Proxy combined pull | TopK geom pull | TopK no-geom pull | DDPM match | Condition match | Proxy combined match | TopK geom match |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for r in condition_table:
        lines.append(
            f"| `{r['condition']}` | {sf(r['ddpm_mean_pull']):.4f} | {sf(r['condition_nearest_upper_pull']):.4f} | "
            f"{sf(r['proxy_action_nn_pull']):.4f} | {sf(r['proxy_state_motion_nn_pull']):.4f} | "
            f"{sf(r['proxy_combined_nn_pull']):.4f} | {sf(r['proxy_topk_action_geom_pull']):.4f} | "
            f"{sf(r['proxy_topk_no_action_geom_pull']):.4f} | {sf(r['ddpm_match']):.3f} | "
            f"{sf(r['condition_upper_match']):.3f} | {sf(r['proxy_combined_match']):.3f} | "
            f"{sf(r['proxy_topk_action_geom_match']):.3f} |"
        )

    lines += [
        "",
        "## Episode Summary",
        "",
        "| Selector | Condition | Rows | OK | Timeout | Uses condition | Final | Δ | Future match | Proxy dist | Pull | OOD | Action MAE | Pull diff | Failures |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for r in table:
        lines.append(
            f"| `{r['selector']}` | `{r['condition']}` | {r['rows']} | {r['ok_rows']} | {r['timeout_rows']} | "
            f"{r['uses_condition_label']} | {sf(r['mean_final_fraction']):.4f} | {sf(r['mean_delta_fraction']):.4f} | "
            f"{sf(r['mean_future_match']):.3f} | {sf(r['mean_proxy_distance']):.4f} | {sf(r['mean_pull_len']):.4f} | "
            f"{sf(r['mean_action_ood']):.3f} | {sf(r['mean_action_mae_to_source']):.4f} | "
            f"{sf(r['mean_pull_diff_to_source']):.4f} | {r['failed_rows']} |"
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
        "- This is condition proxy / score ablation diagnostic only.",
        "- No model training was run.",
        "- No future DDPM was trained.",
        "- No Phase4 or CPS was run.",
        "- `condition_nearest_upper` uses condition labels and is an upper bound, not deployable policy evidence.",
        "- If observable proxy selectors do not improve primary, the next step is contact/proprio proxy design, not full CPS.",
    ]

    out_md.write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))

    if verdict == "FAIL":
        raise SystemExit("[Phase3.12b][FAIL] proxy/score diagnostic failed or missing rows")


if __name__ == "__main__":
    main()
