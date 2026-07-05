#!/usr/bin/env python3
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def parse_bool(x):
    return str(x).lower() in {"1", "true", "yes"}


def rate(k, n):
    return float(k) / float(n) if n else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--trials_csv", default="reports/phase3_2_rollout_smoke_trials.csv")
    parser.add_argument("--out_json", default="reports/phase3_2_rollout_smoke_summary_checked.json")
    parser.add_argument("--out_md", default="reports/phase3_2_rollout_smoke_summary_checked.md")
    args = parser.parse_args()

    root = Path(args.root)
    path = root / args.trials_csv
    if not path.exists():
        raise SystemExit(f"[Phase3.2][FAIL] missing rollout trials csv: {path}")
    with path.open() as f:
        rows = list(csv.DictReader(f))

    required_conditions = ["free", "hidden_pin", "hidden_high_friction", "hidden_breakaway_pin"]
    by_baseline_condition = defaultdict(list)
    for row in rows:
        by_baseline_condition[(row.get("baseline", ""), row.get("condition", ""))].append(row)

    table = []
    success_by = {}
    for (baseline, condition), rs in sorted(by_baseline_condition.items()):
        k = sum(1 for r in rs if parse_bool(r.get("success")))
        n = len(rs)
        final_vals = [float(r.get("final_fraction", 0.0) or 0.0) for r in rs]
        item = {
            "baseline": baseline,
            "condition": condition,
            "success_count": k,
            "total": n,
            "success_rate": rate(k, n),
            "final_fraction_mean": sum(final_vals) / n if n else float("nan"),
        }
        table.append(item)
        success_by[(baseline, condition)] = item["success_rate"]

    baselines = sorted({r["baseline"] for r in table})
    gaps = {}
    for baseline in baselines:
        free = success_by.get((baseline, "free"), float("nan"))
        primary = success_by.get((baseline, "hidden_breakaway_pin"), float("nan"))
        diagnostic = success_by.get((baseline, "hidden_pin"), float("nan"))
        gaps[baseline] = {
            "primary_pair": "free_vs_hidden_breakaway_pin",
            "diagnostic_pair": "free_vs_hidden_pin",
            "primary_success_gap_free_minus_hidden_breakaway": free - primary,
            "diagnostic_success_gap_free_minus_hidden_pin": free - diagnostic,
        }

    conditions = sorted({r.get("condition", "") for r in rows})
    primary_pairs = sorted({r.get("primary_pair", "") for r in rows if "primary_pair" in r})
    diagnostic_pairs = sorted({r.get("diagnostic_pair", "") for r in rows if "diagnostic_pair" in r})
    checks = {
        "conditions_present": all(c in conditions for c in required_conditions),
        "primary_pair_correct": primary_pairs in (["free_vs_hidden_breakaway_pin"], []),
        "diagnostic_pair_correct": diagnostic_pairs in (["free_vs_hidden_pin"], []),
        "has_rows": len(rows) > 0,
    }
    verdict = "PASS" if all(checks.values()) else "FAIL"
    payload = {
        "verdict": verdict,
        "checks": checks,
        "num_rows": len(rows),
        "conditions": conditions,
        "primary_pairs": primary_pairs,
        "diagnostic_pairs": diagnostic_pairs,
        "table": table,
        "gaps": gaps,
        "recommendation": "Rollout smoke complete. Do not enter Phase4 until reviewed." if verdict == "PASS" else "Rollout smoke invalid; do not use.",
    }

    (root / args.out_json).write_text(json.dumps(payload, indent=2, sort_keys=True))
    lines = [
        "# Phase3.2 Rollout Smoke Summary Check",
        "",
        "## Verdict",
        "",
        f"- Verdict: `{verdict}`",
        f"- Rows: `{len(rows)}`",
        "",
        "## Checks",
        "",
        "| Check | Result |",
        "|---|---:|",
    ]
    for key, val in checks.items():
        lines.append(f"| `{key}` | `{val}` |")
    lines += [
        "",
        "## Success by Baseline / Condition",
        "",
        "| Baseline | Condition | Success | Total | Rate | Final fraction mean |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in table:
        lines.append(f"| `{row['baseline']}` | `{row['condition']}` | {row['success_count']} | {row['total']} | {row['success_rate']:.3f} | {row['final_fraction_mean']:.3f} |")
    lines += [
        "",
        "## Success Gaps",
        "",
        "| Baseline | Primary gap free-hidden_breakaway | Diagnostic gap free-hidden_pin |",
        "|---|---:|---:|",
    ]
    for baseline, gap in gaps.items():
        lines.append(f"| `{baseline}` | {gap['primary_success_gap_free_minus_hidden_breakaway']:.3f} | {gap['diagnostic_success_gap_free_minus_hidden_pin']:.3f} |")
    lines += [
        "",
        "## Interpretation",
        "",
        "- This is rollout smoke only.",
        "- It is not Phase4 and not CPS evidence.",
        "- Primary rollout metric is `free_vs_hidden_breakaway_pin`.",
        "- `free_vs_hidden_pin` is diagnostic only.",
    ]
    (root / args.out_md).write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if verdict != "PASS":
        raise SystemExit("[Phase3.2][FAIL] rollout summary check failed")


if __name__ == "__main__":
    main()
