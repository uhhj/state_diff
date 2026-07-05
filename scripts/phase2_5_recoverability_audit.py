#!/usr/bin/env python3
"""Main-repo wrapper for Phase2.5/2.5b recoverability audit outputs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ORACLE_POLICIES = {
    "oracle_pull",
    "oracle_regrasp",
    "oracle_wiggle",
    "oracle_breakaway_then_place",
    "oracle_partial_release_then_place",
}
SEARCH_POLICIES = {"random_search", "guided_search", "cem_search"}


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


def mean_value(rows: List[Dict[str, str]], key: str) -> Optional[float]:
    vals = [f(r.get(key)) for r in rows]
    vals = [x for x in vals if np.isfinite(x)]
    return float(np.mean(vals)) if vals else None


def classify(nominal_success: float, oracle_success: float, search_success: float = 0.0, free_search_sanity: bool = True) -> str:
    gap = oracle_success - nominal_success
    if oracle_success <= 0.05:
        return "impossible_diagnostic"
    if nominal_success >= 0.80 and oracle_success >= 0.80:
        return "weak_easy_control"
    if nominal_success <= 0.40 and oracle_success >= 0.50 and gap >= 0.30:
        return "recoverable_cps_candidate"
    if nominal_success <= 0.50 and oracle_success >= 0.50 and gap >= 0.20:
        return "near_recoverable_candidate"
    return "ambiguous_needs_tuning"


def summarize_condition_rows(condition: str, rows: List[Dict[str, str]], label: str = "") -> Dict[str, Any]:
    by_policy: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for r in rows:
        by_policy[r.get("policy", "")].append(r)
    nominal = success_rate(by_policy.get("nominal", []))
    oracle_rates = [success_rate(by_policy.get(p, [])) for p in ORACLE_POLICIES if by_policy.get(p)]
    oracle = max(oracle_rates) if oracle_rates else 0.0
    search_rates = [success_rate(by_policy.get(p, [])) for p in SEARCH_POLICIES if by_policy.get(p)]
    search_best = max(search_rates) if search_rates else 0.0
    gap = oracle - nominal
    cls = classify(nominal, oracle, search_best)
    release_rows = [r for r in rows if as_bool(r.get("breakaway_released"))]
    release_steps = [f(r.get("breakaway_release_step")) for r in release_rows]
    release_steps = [x for x in release_steps if np.isfinite(x)]
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
        "breakaway_released_rate": success_rate([{"final_fraction": "1" if as_bool(r.get("breakaway_released")) else "0"} for r in rows]),
        "breakaway_release_step_mean": float(np.mean(release_steps)) if release_steps else None,
        "breakaway_max_disp_seen_mean": mean_value(rows, "breakaway_max_disp_seen"),
    }


def compute_search_sanity(rows: List[Dict[str, str]]) -> Dict[str, Any]:
    free_nom = [r for r in rows if r.get("condition") == "free" and r.get("policy") == "nominal"]
    free_guided = [r for r in rows if r.get("condition") == "free" and r.get("policy") == "guided_search"]
    free_random = [r for r in rows if r.get("condition") == "free" and r.get("policy") in SEARCH_POLICIES]
    nominal = success_rate(free_nom)
    guided = success_rate(free_guided)
    search_best = success_rate(free_random)
    return {
        "free_nominal_success": nominal,
        "free_guided_search_success": guided,
        "free_search_best_success": search_best,
        "search_sanity_pass": bool(nominal >= 0.95 and guided >= 0.50),
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


def write_plan(root: Path, selected: Optional[Dict[str, Any]]) -> None:
    if not selected:
        return
    selected_condition = selected["condition"]
    selected_label = selected.get("label", selected_condition)
    path = root / "reports/phase3_condition_plan_after_recoverability.md"
    path.write_text(
        "# Phase3 Condition Plan After Recoverability Audit\n\n"
        "- hard diagnostic condition: `hidden_pin`\n"
        "- weak control condition: `hidden_high_friction`\n"
        f"- selected recoverable CPS branch: `{selected_condition}`\n"
        f"- selected recoverable config: `{selected_label}`\n\n"
        "## Phase3 medium should use\n\n"
        "conditions:\n"
        "- `free`\n"
        "- `hidden_pin`\n"
        "- `hidden_high_friction`\n"
        f"- `{selected_condition}`\n\n"
        "primary_branch_pair:\n"
        f"- `free` vs `{selected_condition}`\n\n"
        "diagnostic_branch_pair:\n"
        "- `free` vs `hidden_pin`\n"
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

    conditions = sorted({r.get("condition", "") for r in rows if r.get("condition")})
    table = [summarize_condition_rows(c, [r for r in rows if r.get("condition") == c]) for c in conditions]
    sweep_table = read_sweep_tables(root, args.sweep_root)
    search_sanity = compute_search_sanity(rows)

    all_for_selection = table + sweep_table
    candidates = [r for r in all_for_selection if r["class"] == "recoverable_cps_candidate"]
    near = [r for r in all_for_selection if r["class"] == "near_recoverable_candidate"]
    selected = None
    verdict = "FAIL"
    if candidates and search_sanity["search_sanity_pass"]:
        selected = sorted(candidates, key=lambda r: (r["gap"], r["oracle_success"], -r["nominal_success"]), reverse=True)[0]
        verdict = "PASS"
    elif candidates or near:
        selected = sorted(candidates or near, key=lambda r: (r["gap"], r["oracle_success"], -r["nominal_success"]), reverse=True)[0]
        verdict = "WARN"

    impossible = [r["condition"] for r in table if r["class"] == "impossible_diagnostic"]
    weak = [r["condition"] for r in table if r["class"] == "weak_easy_control"]
    ambiguous = [r["condition"] for r in table if r["class"] == "ambiguous_needs_tuning"]

    recommendation = (
        "Use selected config for Phase3 medium only if verdict=PASS."
        if verdict == "PASS"
        else "No fully qualified recoverable branch yet. Continue tuning; do not run Phase3 medium."
    )
    if not search_sanity["search_sanity_pass"]:
        recommendation = "Search sanity failed; search evidence is invalid. Fix guided_search before interpreting search_best_success."

    payload = {
        "verdict": verdict,
        "condition_table": table,
        "parameter_sweep_table": sweep_table,
        "search_sanity": search_sanity,
        "selected_recoverable_condition": selected["condition"] if selected and verdict == "PASS" else None,
        "selected_recoverable_config": selected["label"] if selected and verdict == "PASS" else None,
        "selected_recoverable_label": selected["label"] if selected and verdict == "PASS" else None,
        "selected_row": selected,
        "candidate_rows": candidates,
        "near_candidate_rows": near,
        "impossible_conditions": impossible,
        "weak_conditions": weak,
        "ambiguous_conditions": ambiguous,
        "branch_roles": {
            "hard_diagnostic_branch": "hidden_pin",
            "weak_control_branch": "hidden_high_friction",
            "selected_recoverable_cps_branch": selected["condition"] if selected and verdict == "PASS" else None,
        },
        "recommendation": recommendation,
        "submodule_runtime": {
            "audit_runtime": sub_summary.get("audit_runtime", {}),
            "num_rows": sub_summary.get("num_rows"),
            "conditions": sub_summary.get("conditions", []),
            "policies": sub_summary.get("policies", []),
            "source_summary_json": str(Path(args.summary_json)),
        },
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
        f"- Selected recoverable condition: `{payload['selected_recoverable_condition']}`",
        f"- Selected recoverable config: `{payload['selected_recoverable_config']}`",
        f"- Recommendation: {recommendation}",
        "",
        "## Search Sanity",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| free nominal success | {search_sanity['free_nominal_success']:.3f} |",
        f"| free guided_search success | {search_sanity['free_guided_search_success']:.3f} |",
        f"| search sanity pass | {search_sanity['search_sanity_pass']} |",
        "",
        "If search sanity fails, search_best_success is diagnostic only and cannot be used for selection.",
        "",
        "## Condition Summary",
        "",
        "| Condition | Nominal Success | Oracle Success | Search Best Success | Gap | Class |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for r in table:
        lines.append("| {condition} | {nominal_success:.3f} | {oracle_success:.3f} | {search_best_success:.3f} | {gap:.3f} | {class} |".format(**r))
    if sweep_table:
        lines += ["", "## Parameter Sweep", "", "| Sweep | Condition | Nominal Success | Oracle Success | Search Best Success | Gap | Class |", "|---|---|---:|---:|---:|---:|---|"]
        for r in sweep_table:
            lines.append("| {label} | {condition} | {nominal_success:.3f} | {oracle_success:.3f} | {search_best_success:.3f} | {gap:.3f} | {class} |".format(**r))
    lines += [
        "",
        "## Selected Config",
        "",
        f"- selected_recoverable_condition: `{payload['selected_recoverable_condition']}`",
        f"- selected_recoverable_config: `{payload['selected_recoverable_config']}`",
        "",
        "## Interpretation",
        "",
        "- Success rates are computed from `final_fraction >= 0.95`.",
        "- PASS requires a recoverable candidate and search sanity pass.",
        "- WARN means a near candidate exists or search sanity is incomplete, but Phase3 medium is still blocked.",
        "- `hidden_pin` remains hard diagnostic, not the CPS success-improvement target.",
    ]
    out_md.write_text("\n".join(lines) + "\n")
    write_plan(root, selected if verdict == "PASS" else None)
    print(json.dumps(payload, indent=2, sort_keys=True))

    if verdict == "FAIL":
        raise SystemExit("[Phase2.5][FAIL] no recoverable hidden-contact condition selected")


if __name__ == "__main__":
    main()