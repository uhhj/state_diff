#!/usr/bin/env python3
"""Main-repo wrapper for Phase2.5 recoverability audit outputs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ORACLE_POLICIES = {"oracle_pull", "oracle_regrasp", "oracle_wiggle"}
SEARCH_POLICIES = {"random_search", "cem_search"}


def as_bool(v: Any) -> bool:
    return str(v).strip().lower() in {"true", "1", "yes"}


def f(v: Any, default: float = float("nan")) -> float:
    try:
        x = float(v)
        return x if np.isfinite(x) else default
    except Exception:
        return default


def read_csv(path: Path) -> List[Dict[str, str]]:
    with Path(path).open(newline="") as fh:
        return list(csv.DictReader(fh))


def read_json(path: Path) -> Dict[str, Any]:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else {}


def row_success(row: Dict[str, str]) -> bool:
    if "final_fraction" in row and str(row.get("final_fraction", "")).strip() != "":
        return f(row.get("final_fraction"), default=-1.0) >= 0.95
    return as_bool(row.get("success"))


def success_rate(rows: List[Dict[str, str]]) -> float:
    if not rows:
        return 0.0
    return float(np.mean([1.0 if row_success(r) else 0.0 for r in rows]))


def mean_value(rows: List[Dict[str, str]], key: str) -> float:
    vals = [f(r.get(key)) for r in rows]
    vals = [x for x in vals if np.isfinite(x)]
    return float(np.mean(vals)) if vals else float("nan")


def classify(condition: str, nominal_success: float, oracle_success: float) -> str:
    gap = oracle_success - nominal_success
    if oracle_success <= 0.05:
        return "impossible_diagnostic"
    if nominal_success >= 0.80 and oracle_success >= 0.80:
        return "weak_easy_control"
    if nominal_success <= 0.40 and oracle_success >= 0.50 and gap >= 0.30:
        return "recoverable_cps_candidate"
    if condition == "hidden_high_friction" and nominal_success >= 0.50:
        return "weak_candidate"
    return "ambiguous_needs_tuning"


def summarize_condition_rows(condition: str, rows: List[Dict[str, str]], label: str = "") -> Dict[str, Any]:
    by_policy: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for r in rows:
        by_policy[r.get("policy", "")].append(r)
    nominal = success_rate(by_policy.get("nominal", []))
    oracle_rates = [success_rate(by_policy.get(p, [])) for p in ORACLE_POLICIES if by_policy.get(p)]
    oracle = max(oracle_rates) if oracle_rates else 0.0
    search_rows = []
    for p in SEARCH_POLICIES:
        search_rows.extend(by_policy.get(p, []))
    search_best = success_rate(search_rows)
    gap = oracle - nominal
    cls = classify(condition, nominal, oracle)
    return {
        "label": label or condition,
        "condition": condition,
        "nominal_success": nominal,
        "oracle_success": oracle,
        "search_best_success": search_best,
        "gap": gap,
        "future_divergence_vs_free": mean_value(rows, "final_chamfer_to_free"),
        "class": cls,
        "num_trials": len(rows),
    }


def read_sweep_tables(root: Path, sweep_root: str | None) -> List[Dict[str, Any]]:
    if not sweep_root:
        return []
    base = Path(sweep_root)
    if not base.is_absolute():
        base = root / sweep_root
    if not base.exists():
        return []
    out = []
    for csv_path in sorted(base.glob("*/recoverability_trials.csv")):
        label = csv_path.parent.name
        rows = read_csv(csv_path)
        conditions = sorted({r.get("condition", "") for r in rows if r.get("condition")})
        for condition in conditions:
            crows = [r for r in rows if r.get("condition") == condition]
            out.append(summarize_condition_rows(condition, crows, label=label))
    return out


def sanitize_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [sanitize_json(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    return obj


def write_plan(root: Path, selected: str) -> None:
    if not selected:
        return
    path = root / "reports/phase3_condition_plan_after_recoverability.md"
    path.write_text(
        "# Phase3 Condition Plan After Recoverability Audit\n\n"
        "- hard diagnostic condition: `hidden_pin`\n"
        "- weak control condition: `hidden_high_friction`\n"
        f"- selected recoverable CPS branch: `{selected}`\n\n"
        "## Phase3 medium should use\n\n"
        "conditions:\n"
        "- `free`\n"
        "- `hidden_pin`\n"
        "- `hidden_high_friction`\n"
        f"- `{selected}`\n\n"
        "primary_branch_pair:\n"
        f"- `free` vs `{selected}`\n\n"
        "diagnostic_branch_pair:\n"
        "- `free` vs `hidden_pin`\n\n"
        "## Required Phase3 Script Follow-up\n\n"
        "Before running Phase3 medium, parameterize branch metrics with `PHASE3_PRIMARY_HIDDEN_CONDITION` "
        "or `--primary_hidden_condition`, so the primary wrong-branch pair uses the selected recoverable condition. "
        "Keep `hidden_pin` as a hard diagnostic branch, not the success-improvement target.\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/state_diff2")
    ap.add_argument("--trials_csv", required=True)
    ap.add_argument("--summary_json", required=True)
    ap.add_argument("--out_json", default="reports/phase2_5_recoverability_summary.json")
    ap.add_argument("--out_md", default="reports/phase2_5_recoverability_report.md")
    ap.add_argument("--sweep_root", default=None)
    args = ap.parse_args()

    root = Path(args.root)
    rows = read_csv(Path(args.trials_csv))
    sub_summary = read_json(Path(args.summary_json))
    by_cond_policy: Dict[tuple, List[Dict[str, str]]] = defaultdict(list)
    for r in rows:
        by_cond_policy[(r.get("condition", ""), r.get("policy", ""))].append(r)

    conditions = sorted({r.get("condition", "") for r in rows if r.get("condition")})
    table = []
    for condition in conditions:
        crows = [r for r in rows if r.get("condition") == condition]
        table.append(summarize_condition_rows(condition, crows))

    sweep_table = read_sweep_tables(root, args.sweep_root)
    all_for_selection = table + sweep_table
    candidates = [r for r in all_for_selection if r["class"] == "recoverable_cps_candidate"]
    selected = None
    selected_label = None
    if candidates:
        chosen = sorted(candidates, key=lambda r: (r["gap"], r["oracle_success"]), reverse=True)[0]
        selected = chosen["condition"]
        selected_label = chosen.get("label", selected)
    impossible = [r["condition"] for r in table if r["class"] == "impossible_diagnostic"]
    weak = [r["condition"] for r in table if r["class"] in {"weak_easy_control", "weak_candidate"}]
    ambiguous = [r["condition"] for r in table if r["class"] == "ambiguous_needs_tuning"]

    if selected:
        verdict = "PASS"
        recommendation = "Use `{}` as the primary recoverable hidden branch for CPS success-improvement studies; keep hidden_pin as hard diagnostic.".format(selected)
    else:
        verdict = "FAIL"
        recommendation = "No recoverable candidate found. Do not run Phase3 medium; tune soft/breakaway/friction parameters and rerun Phase2.5."

    submodule_runtime = {
        "audit_runtime": sub_summary.get("audit_runtime", {}),
        "num_rows": sub_summary.get("num_rows"),
        "conditions": sub_summary.get("conditions", []),
        "policies": sub_summary.get("policies", []),
        "source_summary_json": str(Path(args.summary_json)),
    }

    payload = {
        "verdict": verdict,
        "selected_recoverable_condition": selected,
        "impossible_conditions": impossible,
        "weak_conditions": weak,
        "ambiguous_conditions": ambiguous,
        "condition_table": table,
        "parameter_sweep_table": sweep_table,
        "selected_recoverable_label": selected_label,
        "recommendation": recommendation,
        "submodule_runtime": submodule_runtime,
    }

    out_json = root / args.out_json
    out_md = root / args.out_md
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = sanitize_json(payload)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))

    lines = [
        "# Phase2.5 Hidden-Contact Recoverability Audit",
        "",
        "## Verdict",
        "",
        f"- Verdict: `{verdict}`",
        f"- Selected recoverable condition: `{selected}`",
        f"- Recommendation: {recommendation}",
        "",
        "## Condition Summary",
        "",
        "| Condition | Nominal Success | Oracle Success | Search Best Success | Gap | Class |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for r in table:
        lines.append("| {condition} | {nominal_success:.3f} | {oracle_success:.3f} | {search_best_success:.3f} | {gap:.3f} | {class} |".format(**r))
    if sweep_table:
        lines += [
            "",
            "## Parameter Sweep",
            "",
            "| Sweep | Condition | Nominal Success | Oracle Success | Search Best Success | Gap | Class |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
        for r in sweep_table:
            lines.append("| {label} | {condition} | {nominal_success:.3f} | {oracle_success:.3f} | {search_best_success:.3f} | {gap:.3f} | {class} |".format(**r))
    lines += [
        "",
        "## Branch Roles",
        "",
        f"- Hard diagnostic branch: `hidden_pin`",
        f"- Weak control branch: `hidden_high_friction`",
        f"- Selected recoverable CPS branch: `{selected}`",
        "",
        "## Interpretation",
        "",
        "- Success rates are computed from `final_fraction >= 0.95`, not from internal episode bookkeeping flags.",
        "- `hidden_pin` is retained even if impossible; it should not be the success-improvement target.",
        "- A recoverable branch requires low nominal success and high hidden-condition-aware oracle/search success.",
        "- Do not run Phase3 medium unless this report selects a recoverable condition.",
    ]
    out_md.write_text("\n".join(lines) + "\n")
    write_plan(root, selected)
    print(json.dumps(payload, indent=2, sort_keys=True))

    if verdict == "FAIL":
        raise SystemExit("[Phase2.5][FAIL] no recoverable hidden-contact condition selected")


if __name__ == "__main__":
    main()
