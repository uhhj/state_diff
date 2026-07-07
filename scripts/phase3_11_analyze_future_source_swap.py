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
FUTURE_SOURCES = [
    "ddpm_mean",
    "ddpm_best_of_k_by_train_nn",
    "global_input_retrieval",
    "condition_matched_retrieval",
]
IDM_POLICIES = ["phase39b_xy_only_high_weight"]


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


def rows_for(rows: List[Dict[str, str]], idm_policy: str, future_source: str, condition: str) -> List[Dict[str, str]]:
    return [
        r for r in rows
        if r.get("idm_policy") == idm_policy
        and r.get("future_source") == future_source
        and r.get("condition") == condition
    ]


def ok_rows_for(rows: List[Dict[str, str]], idm_policy: str, future_source: str, condition: str) -> List[Dict[str, str]]:
    return [
        r for r in rows_for(rows, idm_policy, future_source, condition)
        if r.get("status") == "ok" and not r.get("failure_reason")
    ]


def metric(rows: List[Dict[str, str]], idm_policy: str, future_source: str, condition: str, key: str) -> float:
    return mean([sf(r.get(key)) for r in ok_rows_for(rows, idm_policy, future_source, condition)])


def ok_count(rows: List[Dict[str, str]], idm_policy: str, future_source: str, condition: str) -> int:
    return len(ok_rows_for(rows, idm_policy, future_source, condition))


def aggregate_table(episodes: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    policies = sorted({r.get("idm_policy", "") for r in episodes if r.get("idm_policy")}) or IDM_POLICIES
    sources = sorted({r.get("future_source", "") for r in episodes if r.get("future_source")}) or FUTURE_SOURCES
    out: List[Dict[str, Any]] = []

    for policy in policies:
        for source in sources:
            for cond in CONDITIONS:
                rs = rows_for(episodes, policy, source, cond)
                ok = [r for r in rs if r.get("status") == "ok" and not r.get("failure_reason")]
                out.append({
                    "idm_policy": policy,
                    "future_source": source,
                    "condition": cond,
                    "rows": len(rs),
                    "ok_rows": len(ok),
                    "timeout_rows": sum(1 for r in rs if "timeout" in str(r.get("status", ""))),
                    "failed_rows": sum(1 for r in rs if r.get("status") != "ok" or r.get("failure_reason")),
                    "success_rate": mean([1.0 if pb(r.get("success")) else 0.0 for r in ok]),
                    "mean_final_fraction": mean([sf(r.get("final_fraction")) for r in ok]),
                    "mean_delta_fraction": mean([sf(r.get("delta_fraction")) for r in ok]),
                    "mean_future_match": mean([sf(r.get("mean_future_match")) for r in ok]),
                    "mean_future_nn_l2": mean([sf(r.get("mean_future_nn_l2")) for r in ok]),
                    "mean_future_nn_mae": mean([sf(r.get("mean_future_nn_mae")) for r in ok]),
                    "mean_input_nn_l2": mean([sf(r.get("mean_input_nn_l2")) for r in ok]),
                    "mean_action_ood": mean([sf(r.get("mean_action_ood")) for r in ok]),
                    "mean_clip_frac": mean([sf(r.get("mean_clip_frac")) for r in ok]),
                    "mean_pull_len": mean([sf(r.get("mean_pull_len")) for r in ok]),
                    "uses_condition_label": int(max([sf(r.get("future_source_uses_condition_label")) for r in ok], default=0)),
                })
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--episode_csv", default="reports/phase3_11_future_source_swap_episodes.csv")
    parser.add_argument("--step_csv", default="reports/phase3_11_future_source_swap_steps.csv")
    parser.add_argument("--progress_json", default="reports/phase3_11_future_source_swap_progress.json")
    parser.add_argument("--raw_summary", default="reports/phase3_11_future_source_swap_raw_summary.json")
    parser.add_argument("--out_json", default="reports/phase3_11_future_source_swap_summary.json")
    parser.add_argument("--out_md", default="reports/phase3_11_future_source_swap_report.md")
    parser.add_argument("--idm_policy", default="phase39b_xy_only_high_weight")
    parser.add_argument("--primary_condition", default="hidden_breakaway_pin")
    parser.add_argument("--improve_threshold", type=float, default=0.05)
    parser.add_argument("--min_ok_rows_per_source_condition", type=int, default=2)
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

    missing: List[str] = []
    for source in FUTURE_SOURCES:
        for cond in CONDITIONS:
            n = ok_count(episodes, args.idm_policy, source, cond)
            if n < args.min_ok_rows_per_source_condition:
                missing.append(f"{args.idm_policy}/{source}/{cond}: ok={n}")
    if missing:
        issue("FAIL", "missing_required_future_source_condition_rows", missing)

    timeout_rows = [r for r in episodes if "timeout" in str(r.get("status", ""))]
    failed_rows = [r for r in episodes if r.get("status") != "ok" or r.get("failure_reason")]
    if timeout_rows:
        issue("WARN", "timeouts_present", len(timeout_rows))
    if failed_rows:
        issue("WARN", "failures_present", len(failed_rows))

    table = aggregate_table(episodes)

    condition_table: List[Dict[str, Any]] = []
    for cond in CONDITIONS:
        ddpm = metric(episodes, args.idm_policy, "ddpm_mean", cond, "final_fraction")
        bestk = metric(episodes, args.idm_policy, "ddpm_best_of_k_by_train_nn", cond, "final_fraction")
        global_ret = metric(episodes, args.idm_policy, "global_input_retrieval", cond, "final_fraction")
        cond_ret = metric(episodes, args.idm_policy, "condition_matched_retrieval", cond, "final_fraction")

        ddpm_match = metric(episodes, args.idm_policy, "ddpm_mean", cond, "mean_future_match")
        bestk_match = metric(episodes, args.idm_policy, "ddpm_best_of_k_by_train_nn", cond, "mean_future_match")
        global_match = metric(episodes, args.idm_policy, "global_input_retrieval", cond, "mean_future_match")
        cond_match = metric(episodes, args.idm_policy, "condition_matched_retrieval", cond, "mean_future_match")

        ddpm_pull = metric(episodes, args.idm_policy, "ddpm_mean", cond, "mean_pull_len")
        bestk_pull = metric(episodes, args.idm_policy, "ddpm_best_of_k_by_train_nn", cond, "mean_pull_len")
        global_pull = metric(episodes, args.idm_policy, "global_input_retrieval", cond, "mean_pull_len")
        cond_pull = metric(episodes, args.idm_policy, "condition_matched_retrieval", cond, "mean_pull_len")

        condition_table.append({
            "condition": cond,
            "ddpm_mean_final": ddpm,
            "ddpm_bestk_final": bestk,
            "global_retrieval_final": global_ret,
            "condition_retrieval_final": cond_ret,
            "bestk_improvement": bestk - ddpm if math.isfinite(bestk) and math.isfinite(ddpm) else float("nan"),
            "global_retrieval_improvement": global_ret - ddpm if math.isfinite(global_ret) and math.isfinite(ddpm) else float("nan"),
            "condition_retrieval_improvement": cond_ret - ddpm if math.isfinite(cond_ret) and math.isfinite(ddpm) else float("nan"),
            "ddpm_mean_match": ddpm_match,
            "ddpm_bestk_match": bestk_match,
            "global_retrieval_match": global_match,
            "condition_retrieval_match": cond_match,
            "ddpm_mean_pull": ddpm_pull,
            "ddpm_bestk_pull": bestk_pull,
            "global_retrieval_pull": global_pull,
            "condition_retrieval_pull": cond_pull,
        })

    primary = next((r for r in condition_table if r["condition"] == args.primary_condition), {})
    bestk_primary_gain = sf(primary.get("bestk_improvement"))
    global_primary_gain = sf(primary.get("global_retrieval_improvement"))
    condition_primary_gain = sf(primary.get("condition_retrieval_improvement"))
    ddpm_primary_match = sf(primary.get("ddpm_mean_match"))
    bestk_primary_match = sf(primary.get("ddpm_bestk_match"))
    condition_primary_match = sf(primary.get("condition_retrieval_match"))
    ddpm_primary_pull = sf(primary.get("ddpm_mean_pull"))
    condition_primary_pull = sf(primary.get("condition_retrieval_pull"))

    bestk_supported = math.isfinite(bestk_primary_gain) and bestk_primary_gain > args.improve_threshold
    global_retrieval_supported = math.isfinite(global_primary_gain) and global_primary_gain > args.improve_threshold
    condition_retrieval_supported = math.isfinite(condition_primary_gain) and condition_primary_gain > args.improve_threshold
    condition_match_improved = math.isfinite(condition_primary_match) and math.isfinite(ddpm_primary_match) and condition_primary_match > ddpm_primary_match + 0.2
    condition_pull_restored = math.isfinite(condition_primary_pull) and math.isfinite(ddpm_primary_pull) and condition_primary_pull > ddpm_primary_pull + 0.05

    if bestk_supported:
        issue("WARN", "ddpm_best_of_k_beats_mean_on_primary", {
            "gain": bestk_primary_gain,
            "ddpm_match": ddpm_primary_match,
            "bestk_match": bestk_primary_match,
        })

    if global_retrieval_supported:
        issue("WARN", "global_input_retrieval_beats_ddpm_mean_on_primary", {
            "gain": global_primary_gain,
        })

    if condition_retrieval_supported:
        issue("WARN", "condition_matched_retrieval_upper_bound_beats_ddpm_mean_on_primary", {
            "gain": condition_primary_gain,
            "ddpm_match": ddpm_primary_match,
            "condition_match": condition_primary_match,
            "ddpm_pull": ddpm_primary_pull,
            "condition_pull": condition_primary_pull,
        })

    if condition_match_improved:
        issue("WARN", "condition_matched_future_restores_branch_match", {
            "ddpm_match": ddpm_primary_match,
            "condition_match": condition_primary_match,
        })

    if condition_pull_restored:
        issue("WARN", "condition_matched_future_restores_pull_length", {
            "ddpm_pull": ddpm_primary_pull,
            "condition_pull": condition_primary_pull,
        })

    if progress and progress.get("status") != "completed":
        issue("WARN", "future_source_swap_not_fully_completed", progress.get("status"))

    if condition_retrieval_supported and (condition_match_improved or condition_pull_restored):
        root_cause = "phase311_future_branch_conditioning_blocker_supported"
    elif bestk_supported:
        root_cause = "phase311_ddpm_sample_selection_beats_mean_supported"
    elif global_retrieval_supported:
        root_cause = "phase311_global_retrieval_future_supported"
    elif condition_retrieval_supported:
        root_cause = "phase311_condition_matched_retrieval_upper_bound_supported"
    else:
        root_cause = "phase311_future_source_swap_inconclusive_or_not_supported"
        issue("WARN", "future_source_swap_not_clearly_supported", primary)

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
            "bestk_supported": bestk_supported,
            "global_retrieval_supported": global_retrieval_supported,
            "condition_retrieval_supported": condition_retrieval_supported,
            "condition_match_improved": condition_match_improved,
            "condition_pull_restored": condition_pull_restored,
        },
        "important_note": "This is a future-source swap diagnostic. condition_matched_retrieval uses condition label and is an upper bound only. No Phase4/CPS.",
        "recommendation": (
            "Do not enter Phase4/CPS. Next step should diagnose or repair future DDPM branch conditioning."
            if root_cause in {
                "phase311_future_branch_conditioning_blocker_supported",
                "phase311_ddpm_sample_selection_beats_mean_supported",
                "phase311_condition_matched_retrieval_upper_bound_supported",
            }
            else
            "Do not enter Phase4/CPS. Future-source swap is inconclusive; inspect step CSV and closed-loop input OOD."
        ),
    }

    out_json = root / args.out_json
    out_md = root / args.out_md
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True))

    lines = [
        "# Phase3.11 Future-Source Swap / Branch Diagnosis Report",
        "",
        "## Verdict",
        "",
        f"- Verdict: `{verdict}`",
        f"- Root cause: `{root_cause}`",
        f"- Episode rows: `{len(episodes)}`",
        f"- Step rows: `{len(steps)}`",
        f"- Progress status: `{progress.get('status', 'missing')}`",
        "",
        "## Per-Condition Future Source Comparison",
        "",
        "| Condition | DDPM mean | DDPM best-k | Global retrieval | Condition retrieval | Best-k Δ | Global Δ | Condition Δ | DDPM match | Best-k match | Condition match | DDPM pull | Condition pull |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for r in condition_table:
        lines.append(
            f"| `{r['condition']}` | {sf(r['ddpm_mean_final']):.4f} | {sf(r['ddpm_bestk_final']):.4f} | "
            f"{sf(r['global_retrieval_final']):.4f} | {sf(r['condition_retrieval_final']):.4f} | "
            f"{sf(r['bestk_improvement']):.4f} | {sf(r['global_retrieval_improvement']):.4f} | "
            f"{sf(r['condition_retrieval_improvement']):.4f} | {sf(r['ddpm_mean_match']):.3f} | "
            f"{sf(r['ddpm_bestk_match']):.3f} | {sf(r['condition_retrieval_match']):.3f} | "
            f"{sf(r['ddpm_mean_pull']):.4f} | {sf(r['condition_retrieval_pull']):.4f} |"
        )

    lines += [
        "",
        "## Episode Summary",
        "",
        "| IDM | Future source | Condition | Rows | OK | Timeout | Success | Final fraction | Future match | Future NN L2 | Pull len | Action OOD | Uses condition label | Failures |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for r in table:
        lines.append(
            f"| `{r['idm_policy']}` | `{r['future_source']}` | `{r['condition']}` | "
            f"{r['rows']} | {r['ok_rows']} | {r['timeout_rows']} | {sf(r['success_rate']):.3f} | "
            f"{sf(r['mean_final_fraction']):.4f} | {sf(r['mean_future_match']):.3f} | "
            f"{sf(r['mean_future_nn_l2']):.4f} | {sf(r['mean_pull_len']):.4f} | "
            f"{sf(r['mean_action_ood']):.3f} | {r['uses_condition_label']} | {r['failed_rows']} |"
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
        "- This is a future-source swap diagnostic only.",
        "- `condition_matched_retrieval` uses condition labels and is a diagnostic upper bound, not deployable policy evidence.",
        "- No model training was run.",
        "- No future DDPM was trained.",
        "- No Phase4 or CPS was run.",
        "- If condition-matched retrieval beats DDPM mean, future branch conditioning is the likely blocker.",
        "- If best-of-k beats mean, DDPM sampling contains useful branch information but mean aggregation is harmful.",
    ]

    out_md.write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))

    if verdict == "FAIL":
        raise SystemExit("[Phase3.11][FAIL] future-source swap failed or missing rows")


if __name__ == "__main__":
    main()
