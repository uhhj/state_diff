#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

import phase3_12d_r1_common as common


def sf(value: Any) -> float:
    try:
        if value in (None, ""):
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def si(value: Any, default: int = -1) -> int:
    try:
        if value in (None, ""):
            return default
        return int(float(value))
    except Exception:
        return default


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def finite(values: Iterable[float]) -> List[float]:
    return [float(x) for x in values if math.isfinite(float(x))]


def mean(values: Iterable[float]) -> float:
    xs = finite(values)
    return float(np.mean(xs)) if xs else float("nan")


def rate(flags: Iterable[bool]) -> float:
    values = [bool(x) for x in flags]
    return float(np.mean(values)) if values else float("nan")


def query_summary(
    query_rows: Sequence[Mapping[str, str]],
    candidate_rows: Sequence[Mapping[str, str]],
    *,
    raw_samples: int,
    actionable_threshold: float,
    headroom_threshold: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    issues: List[Dict[str, Any]] = []
    by_query: Dict[str, List[Mapping[str, str]]] = defaultdict(list)
    for row in candidate_rows:
        by_query[str(row.get("query_id", ""))].append(row)

    summaries: List[Dict[str, Any]] = []
    for query in query_rows:
        qid = str(query.get("query_id", ""))
        rows = by_query.get(qid, [])
        valid = [r for r in rows if si(r.get("valid"), 0) == 1]
        counts = Counter(str(r.get("candidate_family", "")) for r in valid)
        ids = [str(r.get("candidate_id", "")) for r in rows]

        expected_total = int(raw_samples) + 7
        complete = (
            query.get("status") == "complete"
            and len(rows) == expected_total
            and len(valid) == expected_total
            and len(set(ids)) == expected_total
            and counts.get("ddpm_mean", 0) == 1
            and counts.get("ddpm_raw", 0) == int(raw_samples)
            and counts.get("goal_geometry_oracle", 0) == 1
        )
        if not complete:
            issues.append(
                {
                    "level": "FAIL",
                    "name": "query_candidate_set_incomplete",
                    "detail": {
                        "query_id": qid,
                        "query_status": query.get("status"),
                        "rows": len(rows),
                        "valid": len(valid),
                        "unique_ids": len(set(ids)),
                        "family_counts": dict(counts),
                    },
                }
            )
            continue

        baseline = next(r for r in valid if r.get("candidate_family") == "ddpm_mean")
        raw = [r for r in valid if r.get("candidate_family") == "ddpm_raw"]
        goal = next(r for r in valid if r.get("candidate_family") == "goal_geometry_oracle")
        best_raw = max(raw, key=lambda r: sf(r.get("dense_gain")))

        baseline_dense = sf(baseline.get("dense_gain"))
        best_raw_dense = sf(best_raw.get("dense_gain"))
        raw_headroom = best_raw_dense - baseline_dense
        raw_actionable = any(sf(r.get("dense_gain")) > actionable_threshold for r in raw)
        goal_actionable = sf(goal.get("dense_gain")) > actionable_threshold
        any_dense_actionable = any(sf(r.get("dense_gain")) > actionable_threshold for r in valid)
        any_sparse_change = any(sf(r.get("fraction_gain")) > 0 for r in valid)
        sparse_masked = any_dense_actionable and not any_sparse_change

        summaries.append(
            {
                "query_id": qid,
                "condition": query.get("condition"),
                "visible_seed": si(query.get("visible_seed")),
                "query_step": si(query.get("query_step")),
                "baseline_dense": baseline_dense,
                "best_raw_dense": best_raw_dense,
                "best_raw_id": best_raw.get("candidate_id"),
                "raw_headroom": raw_headroom,
                "raw_headroom_above_threshold": int(raw_headroom > headroom_threshold),
                "raw_actionable": int(raw_actionable),
                "raw_actionable_count": sum(
                    sf(r.get("dense_gain")) > actionable_threshold for r in raw
                ),
                "raw_candidate_count": len(raw),
                "goal_dense": sf(goal.get("dense_gain")),
                "goal_actionable": int(goal_actionable),
                "sparse_masked": int(sparse_masked),
                "baseline_fraction_gain": sf(baseline.get("fraction_gain")),
                "best_raw_fraction_gain": sf(best_raw.get("fraction_gain")),
                "goal_fraction_gain": sf(goal.get("fraction_gain")),
                "baseline_breakaway_release": si(baseline.get("breakaway_released_after"), 0),
                "best_raw_breakaway_release": si(best_raw.get("breakaway_released_after"), 0),
                "goal_breakaway_release": si(goal.get("breakaway_released_after"), 0),
            }
        )
    return summaries, issues


def family_summary(rows: Sequence[Mapping[str, str]], actionable_threshold: float) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, int, str], List[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        if si(row.get("valid"), 0) != 1:
            continue
        key = (
            str(row.get("condition", "")),
            si(row.get("query_step")),
            str(row.get("candidate_family", "")),
        )
        groups[key].append(row)

    output: List[Dict[str, Any]] = []
    for (condition, step, family), group in sorted(groups.items()):
        output.append(
            {
                "condition": condition,
                "query_step": step,
                "candidate_family": family,
                "rows": len(group),
                "mean_dense_gain": mean(sf(r.get("dense_gain")) for r in group),
                "median_dense_gain": float(np.median(finite(sf(r.get("dense_gain")) for r in group))) if finite(sf(r.get("dense_gain")) for r in group) else float("nan"),
                "actionable_rate": rate(sf(r.get("dense_gain")) > actionable_threshold for r in group),
                "mean_fraction_gain": mean(sf(r.get("fraction_gain")) for r in group),
                "breakaway_release_rate": rate(si(r.get("breakaway_released_after"), 0) == 1 for r in group),
                "mean_action_ood": mean(sf(r.get("action_ood")) for r in group),
                "mean_pull_len": mean(sf(r.get("pull_len")) for r in group),
            }
        )
    return output


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(fields),
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--query-csv", default="reports/phase3_12d_r1_queries.csv")
    parser.add_argument("--candidate-csv", default="reports/phase3_12d_r1_candidate_effects.csv")
    parser.add_argument("--progress-json", default="reports/phase3_12d_r1_progress.json")
    parser.add_argument("--raw-json", default="reports/phase3_12d_r1_raw_summary.json")
    parser.add_argument("--environment-audit", default="reports/phase3_12d_r1_environment_audit_summary.json")
    parser.add_argument("--conditions", nargs="+", default=list(common.DEFAULT_CONDITIONS))
    parser.add_argument("--visible-seeds", nargs="+", type=int, default=list(common.DEFAULT_VISIBLE_SEEDS))
    parser.add_argument("--query-steps", nargs="+", type=int, default=[0, 4])
    parser.add_argument("--raw-samples", type=int, default=8)
    parser.add_argument("--primary-condition", default="hidden_breakaway_pin")
    parser.add_argument("--actionable-threshold", type=float, default=0.003)
    parser.add_argument("--headroom-threshold", type=float, default=0.003)
    parser.add_argument("--min-positive-query-fraction", type=float, default=0.5)
    parser.add_argument("--goal-actionable-rate-threshold", type=float, default=0.5)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=312014)
    parser.add_argument("--out-json", default="reports/phase3_12d_r1_summary.json")
    parser.add_argument("--out-md", default="reports/phase3_12d_r1_report.md")
    parser.add_argument("--out-query-summary-csv", default="reports/phase3_12d_r1_query_headroom.csv")
    parser.add_argument("--out-family-summary-csv", default="reports/phase3_12d_r1_family_summary.csv")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    query_rows = read_csv(root / args.query_csv)
    candidate_rows = read_csv(root / args.candidate_csv)
    progress = read_json(root / args.progress_json)
    raw = read_json(root / args.raw_json)
    environment = read_json(root / args.environment_audit)
    issues: List[Dict[str, Any]] = []

    def issue(level: str, name: str, detail: Any) -> None:
        issues.append({"level": level, "name": name, "detail": detail})

    expected_queries = len(args.conditions) * len(args.visible_seeds) * len(args.query_steps)
    expected_candidates = expected_queries * (int(args.raw_samples) + 7)

    if environment.get("verdict") == "FAIL" or not environment.get("query_local_snapshot_allowed", False):
        issue("FAIL", "environment_or_task_semantics_audit_failed", environment)
    if progress.get("status") != "completed":
        issue("FAIL", "candidate_execution_not_completed", progress.get("status"))
    if len(query_rows) != expected_queries:
        issue("FAIL", "query_row_count_mismatch", {"expected": expected_queries, "actual": len(query_rows)})
    if len(candidate_rows) != expected_candidates:
        issue("FAIL", "candidate_row_count_mismatch", {"expected": expected_candidates, "actual": len(candidate_rows)})

    invalid_rows = [r for r in candidate_rows if si(r.get("valid"), 0) != 1]
    restore_failures = [r for r in candidate_rows if si(r.get("snapshot_restore_pass"), 0) != 1]
    if invalid_rows:
        issue("FAIL", "invalid_candidate_rows", len(invalid_rows))
    if restore_failures:
        issue("FAIL", "snapshot_restore_integrity_failure", len(restore_failures))

    summaries, summary_issues = query_summary(
        query_rows,
        candidate_rows,
        raw_samples=args.raw_samples,
        actionable_threshold=args.actionable_threshold,
        headroom_threshold=args.headroom_threshold,
    )
    issues.extend(summary_issues)
    families = family_summary(candidate_rows, args.actionable_threshold)

    primary = [x for x in summaries if x["condition"] == args.primary_condition]
    headrooms = finite(x["raw_headroom"] for x in primary)
    ci_low, ci_high = common.bootstrap_mean_ci(
        headrooms,
        samples=args.bootstrap_samples,
        seed=args.bootstrap_seed,
    )
    positive_queries = sum(x["raw_headroom"] > args.headroom_threshold for x in primary)
    required_positive = int(math.ceil(args.min_positive_query_fraction * max(1, len(primary))))
    headroom_supported = (
        len(primary) == len(args.visible_seeds) * len(args.query_steps)
        and mean(headrooms) > args.headroom_threshold
        and math.isfinite(ci_low)
        and ci_low > 0
        and positive_queries >= required_positive
    )
    raw_actionable_rate = rate(bool(x["raw_actionable"]) for x in primary)
    goal_actionable_rate = rate(bool(x["goal_actionable"]) for x in primary)
    sparse_mask_rate = rate(bool(x["sparse_masked"]) for x in primary)

    has_fail = any(x["level"] == "FAIL" for x in issues)
    if has_fail:
        verdict = "FAIL"
        root_cause = "phase312d_r1_snapshot_or_environment_integrity_failed"
    else:
        verdict = "WARN"
        if headroom_supported:
            root_cause = "phase312d_r1_candidate_oracle_headroom_supported"
            issue("WARN", "raw_ddpm_realized_headroom_supported", {"mean": mean(headrooms), "ci": [ci_low, ci_high], "positive": positive_queries})
        elif math.isfinite(goal_actionable_rate) and goal_actionable_rate >= args.goal_actionable_rate_threshold:
            root_cause = "phase312d_r1_candidate_pool_has_no_actionable_headroom"
            issue("WARN", "goal_oracle_actionable_but_raw_pool_not_supported", {"raw_actionable_rate": raw_actionable_rate, "goal_actionable_rate": goal_actionable_rate})
        else:
            root_cause = "phase312d_r1_task_or_primitive_actionability_not_supported"
            issue("WARN", "goal_geometry_oracle_not_consistently_actionable", {"goal_actionable_rate": goal_actionable_rate})
        if math.isfinite(sparse_mask_rate) and sparse_mask_rate >= 0.25:
            issue("WARN", "sparse_progress_metric_masks_dense_action_effect", {"rate": sparse_mask_rate})

    payload = {
        "verdict": verdict,
        "root_cause": root_cause,
        "candidate_effect_rows": len(candidate_rows),
        "valid_rows": len(candidate_rows) - len(invalid_rows),
        "queries": len(summaries),
        "expected_queries": expected_queries,
        "expected_candidate_rows": expected_candidates,
        "primary_condition": args.primary_condition,
        "primary_headroom": {
            "raw_ddpm_headroom_mean": mean(headrooms),
            "raw_ddpm_headroom_ci_low": ci_low,
            "raw_ddpm_headroom_ci_high": ci_high,
            "raw_headroom_positive_queries": positive_queries,
            "raw_headroom_total_queries": len(primary),
            "required_positive_queries": required_positive,
            "raw_candidate_actionable_rate": raw_actionable_rate,
            "goal_geometry_oracle_actionable_rate": goal_actionable_rate,
            "sparse_metric_mask_rate": sparse_mask_rate,
            "raw_ddpm_headroom_supported": bool(headroom_supported),
        },
        "query_summary": summaries,
        "family_summary": families,
        "issues": issues,
        "environment_audit": environment,
        "progress": progress,
        "raw": raw,
        "scientific_scope": (
            "Realized best-of-K and goal-geometry actions are diagnostic oracles, not deployable policies. "
            "No Phase4/CPS claim is allowed."
        ),
        "recommendation": (
            "Do not enter Phase4/CPS. If raw headroom is supported, next validate an observable contact score on independent seeds. "
            "If the goal oracle is actionable but raw candidates are not, repair candidate generation/IDM support first."
        ),
    }
    common.write_json_atomic(root / args.out_json, payload)

    query_fields = [
        "query_id", "condition", "visible_seed", "query_step", "baseline_dense", "best_raw_dense", "best_raw_id", "raw_headroom", "raw_actionable", "raw_actionable_count", "goal_dense", "goal_actionable", "sparse_masked", "baseline_fraction_gain", "best_raw_fraction_gain", "goal_fraction_gain", "baseline_breakaway_release", "best_raw_breakaway_release", "goal_breakaway_release",
    ]
    family_fields = [
        "condition", "query_step", "candidate_family", "rows", "mean_dense_gain", "median_dense_gain", "actionable_rate", "mean_fraction_gain", "breakaway_release_rate", "mean_action_ood", "mean_pull_len",
    ]
    write_csv(root / args.out_query_summary_csv, summaries, query_fields)
    write_csv(root / args.out_family_summary_csv, families, family_fields)

    lines = [
        "# Phase3.12d-r1 Query-Local Candidate Oracle Headroom Report",
        "",
        "## Verdict",
        "",
        f"- Verdict: `{verdict}`",
        f"- Root cause: `{root_cause}`",
        f"- Candidate effect rows: `{len(candidate_rows)}`",
        f"- Valid rows: `{len(candidate_rows) - len(invalid_rows)}`",
        f"- Complete queries: `{len(summaries)}`",
        "",
        "## Primary-condition headroom",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Raw DDPM headroom mean | {mean(headrooms):.6f} |",
        f"| Raw DDPM headroom 95% CI | [{ci_low:.6f}, {ci_high:.6f}] |",
        f"| Raw headroom positive queries | {positive_queries} / {len(primary)} |",
        f"| Raw candidate actionable rate | {raw_actionable_rate:.4f} |",
        f"| Goal-geometry oracle actionable rate | {goal_actionable_rate:.4f} |",
        f"| Sparse metric mask rate | {sparse_mask_rate:.4f} |",
        f"| Raw DDPM headroom supported | `{headroom_supported}` |",
        "",
        "## Per-query summary",
        "",
        "| Query | Condition | Step | Baseline dense | Best raw dense | Raw headroom | Raw actionable | Goal dense | Goal actionable | Sparse masked |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            "| `{query_id}` | `{condition}` | {query_step} | {baseline_dense:.5f} | {best_raw_dense:.5f} | {raw_headroom:.5f} | {raw_actionable} | {goal_dense:.5f} | {goal_actionable} | {sparse_masked} |".format(**row)
        )
    lines += ["", "## Candidate-family summary", "", "| Condition | Step | Family | Rows | Dense gain | Actionable | Fraction gain | Breakaway release | OOD | Pull |", "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in families:
        lines.append(
            "| `{condition}` | {query_step} | `{candidate_family}` | {rows} | {mean_dense_gain:.5f} | {actionable_rate:.3f} | {mean_fraction_gain:.5f} | {breakaway_release_rate:.3f} | {mean_action_ood:.3f} | {mean_pull_len:.4f} |".format(**row)
        )
    lines += ["", "## Issues", "", "| Level | Name | Detail |", "|---|---|---|"]
    if issues:
        for item in issues:
            lines.append("| `{}` | `{}` | {} |".format(item["level"], item["name"], str(item["detail"]).replace("|", "/")))
    else:
        lines.append("| `PASS` | `none` | No issues. |")
    lines += [
        "",
        "## Interpretation",
        "",
        "- Every candidate is executed from a query-local PyBullet snapshot after one prefix execution.",
        "- Raw DDPM best-of-K is a realized-effect oracle and is not deployable.",
        "- Dense geometric effect is primary; quantized bead fraction is secondary.",
        "- `condition_nearest_upper`, retrieved source action, and goal geometry are diagnostic upper bounds.",
        "- No model training, Phase4, or CPS was run.",
    ]
    (root / args.out_md).write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if verdict == "FAIL":
        raise SystemExit("[Phase3.12d-r1] analysis failed integrity gate")


if __name__ == "__main__":
    main()
