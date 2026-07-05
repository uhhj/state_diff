#!/usr/bin/env python3
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

REQUIRED_CONDITIONS = ["free", "hidden_pin", "hidden_high_friction", "hidden_breakaway_pin"]
PRIMARY_PAIR = "free_vs_hidden_breakaway_pin"
DIAGNOSTIC_PAIR = "free_vs_hidden_pin"
PRIMARY = "hidden_breakaway_pin"
DIAGNOSTIC = "hidden_pin"


def parse_bool(x):
    return str(x).strip().lower() in {"1", "true", "yes", "pass", "success"}


def safe_float(x):
    try:
        if x is None or x == "":
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def mean(xs):
    xs = [x for x in xs if math.isfinite(x)]
    return float(sum(xs) / len(xs)) if xs else float("nan")


def read_json(path):
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--trials_csv", default="reports/phase3_4_rollout_smoke_trials.csv")
    parser.add_argument("--summary_json", default="reports/phase3_4_rollout_smoke_summary.json")
    parser.add_argument("--out_json", default="reports/phase3_4_rollout_result_audit_summary.json")
    parser.add_argument("--out_md", default="reports/phase3_4_rollout_result_audit_report.md")
    args = parser.parse_args()

    root = Path(args.root)
    trials_path = root / args.trials_csv
    summary = read_json(root / args.summary_json)
    issues = []

    def issue(level, name, detail):
        issues.append({"level": level, "name": name, "detail": str(detail)})

    if not trials_path.exists():
        issue("FAIL", "trials_csv_missing", trials_path)
        rows = []
    else:
        with trials_path.open(newline="") as f:
            rows = list(csv.DictReader(f))
    if not rows:
        issue("FAIL", "no_trial_rows", len(rows))

    conditions = sorted({r.get("condition", "") for r in rows})
    baselines = sorted({r.get("baseline", "") for r in rows})
    primary_pairs = sorted({r.get("primary_pair", "") for r in rows if "primary_pair" in r})
    diagnostic_pairs = sorted({r.get("diagnostic_pair", "") for r in rows if "diagnostic_pair" in r})
    runtimes = sorted({r.get("rollout_runtime", "") for r in rows if "rollout_runtime" in r})

    if not all(c in conditions for c in REQUIRED_CONDITIONS):
        issue("FAIL", "missing_required_conditions", {"got": conditions, "required": REQUIRED_CONDITIONS})
    if not all(b in baselines for b in ["paper_state", "state_action"]):
        issue("FAIL", "missing_required_baselines", baselines)
    if primary_pairs != [PRIMARY_PAIR]:
        issue("FAIL", "primary_pair_mismatch", primary_pairs)
    if diagnostic_pairs != [DIAGNOSTIC_PAIR]:
        issue("FAIL", "diagnostic_pair_mismatch", diagnostic_pairs)
    if any(r.get("primary_hidden_condition") != PRIMARY for r in rows):
        issue("FAIL", "primary_hidden_condition_mismatch", PRIMARY)
    if any(r.get("diagnostic_hidden_condition") != DIAGNOSTIC for r in rows):
        issue("FAIL", "diagnostic_hidden_condition_mismatch", DIAGNOSTIC)
    if any(r.get("failure_reason") for r in rows):
        issue("FAIL", "runtime_errors_in_trials", [r.get("failure_reason") for r in rows if r.get("failure_reason")][:5])
    if runtimes != ["tf_free_learned_policy"]:
        issue("FAIL", "rollout_runtime_mismatch", runtimes)

    if summary:
        if summary.get("primary_pair") != PRIMARY_PAIR:
            issue("FAIL", "summary_primary_pair_mismatch", summary.get("primary_pair"))
        if summary.get("diagnostic_pair") != DIAGNOSTIC_PAIR:
            issue("FAIL", "summary_diagnostic_pair_mismatch", summary.get("diagnostic_pair"))
        if "primary_success_gap" not in json.dumps(summary):
            issue("FAIL", "summary_missing_primary_success_gap", "")
        if "diagnostic_success_gap" not in json.dumps(summary):
            issue("FAIL", "summary_missing_diagnostic_success_gap", "")
    else:
        issue("FAIL", "summary_json_missing", args.summary_json)

    by_group = defaultdict(list)
    for row in rows:
        by_group[(row.get("baseline", ""), row.get("condition", ""))].append(row)

    table = []
    success_by = {}
    for (baseline, condition), group in sorted(by_group.items()):
        successes = [parse_bool(r.get("success")) for r in group]
        final_fracs = [safe_float(r.get("final_fraction")) for r in group]
        runtime_errors = sum(1 for r in group if r.get("failure_reason"))
        item = {
            "baseline": baseline,
            "condition": condition,
            "success_count": int(sum(successes)),
            "total": len(group),
            "success_rate": float(sum(successes) / len(group)) if group else float("nan"),
            "final_fraction_mean": mean(final_fracs),
            "runtime_errors": runtime_errors,
        }
        table.append(item)
        success_by[(baseline, condition)] = item["success_rate"]

    total_successes = sum(item["success_count"] for item in table)
    if rows and total_successes == 0:
        issue(
            "WARN",
            "no_policy_success_observed",
            "All rollout smoke trials executed without runtime errors, but none reached task success. "
            "Treat this as runtime smoke only, not valid policy execution evidence.",
        )

    gaps = {}
    for baseline in sorted({r["baseline"] for r in table}):
        free = success_by.get((baseline, "free"), float("nan"))
        primary = success_by.get((baseline, PRIMARY), float("nan"))
        diagnostic = success_by.get((baseline, DIAGNOSTIC), float("nan"))
        gaps[baseline] = {
            "primary_pair": PRIMARY_PAIR,
            "diagnostic_pair": DIAGNOSTIC_PAIR,
            "primary_success_gap_free_minus_hidden_breakaway": free - primary,
            "diagnostic_success_gap_free_minus_hidden_pin": free - diagnostic,
        }

    has_fail = any(i["level"] == "FAIL" for i in issues)
    has_warn = any(i["level"] == "WARN" for i in issues)
    verdict = "FAIL" if has_fail else ("WARN" if has_warn else "PASS")
    payload = {
        "verdict": verdict,
        "issues": issues,
        "num_rows": len(rows),
        "conditions": conditions,
        "baselines": baselines,
        "primary_pairs": primary_pairs,
        "diagnostic_pairs": diagnostic_pairs,
        "rollout_runtimes": runtimes,
        "table": table,
        "gaps": gaps,
        "recommendation": "Review smoke only; do not treat as Phase4/CPS evidence. All-zero success requires policy/runtime diagnosis before Phase4.",
    }

    out_json = root / args.out_json
    out_md = root / args.out_md
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True))

    lines = [
        "# Phase3.4 Rollout Result Audit",
        "",
        "## Verdict",
        "",
        f"- Verdict: `{verdict}`",
        f"- Rows: `{len(rows)}`",
        f"- Primary pair: `{PRIMARY_PAIR}`",
        f"- Diagnostic pair: `{DIAGNOSTIC_PAIR}`",
        "",
        "## Success Table",
        "",
        "| Baseline | Condition | Success | Total | Success rate | Final fraction mean | Runtime errors |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for item in table:
        lines.append(
            f"| `{item['baseline']}` | `{item['condition']}` | {item['success_count']} | {item['total']} | "
            f"{item['success_rate']:.3f} | {item['final_fraction_mean']:.3f} | {item['runtime_errors']} |"
        )
    lines += [
        "",
        "## Success Gaps",
        "",
        "| Baseline | Primary gap free-hidden_breakaway | Diagnostic gap free-hidden_pin |",
        "|---|---:|---:|",
    ]
    for baseline, gap in gaps.items():
        lines.append(
            f"| `{baseline}` | {gap['primary_success_gap_free_minus_hidden_breakaway']:.3f} | "
            f"{gap['diagnostic_success_gap_free_minus_hidden_pin']:.3f} |"
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
        lines.append("| `PASS` | `none` | No result-audit issues found. |")
    lines += [
        "",
        "## Scope",
        "",
        "- This is rollout smoke only.",
        "- No Phase4 was run.",
        "- No CPS was run.",
        "- Do not make paper-level claims from this smoke.",
    ]
    out_md.write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if verdict == "FAIL":
        raise SystemExit("[Phase3.4][FAIL] rollout result audit failed")


if __name__ == "__main__":
    main()
