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


def safe_int(v: Any, default: int = -1) -> int:
    try:
        return int(float(v))
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


def success_count_total(rows: List[Dict[str, str]]) -> tuple[int, int, float]:
    n = len(rows)
    k = sum(1 for r in rows if row_success(r))
    return k, n, float(k) / float(n) if n else float("nan")


def mean_value(rows: List[Dict[str, str]], key: str) -> Optional[float]:
    vals = [f(r.get(key)) for r in rows]
    vals = [x for x in vals if np.isfinite(x)]
    return float(np.mean(vals)) if vals else None


def fmt_count(k: int, n: int, rate: float) -> str:
    return f"{k}/{n} = {rate:.3f}" if n else "NA"


def policy_stats(rows: List[Dict[str, str]]) -> Dict[str, Dict[str, Any]]:
    by_policy: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for r in rows:
        by_policy[r.get("policy", "")].append(r)
    out: Dict[str, Dict[str, Any]] = {}
    for policy, rs in sorted(by_policy.items()):
        k, n, rate = success_count_total(rs)
        release_rows = [r for r in rs if "breakaway_released" in r]
        release_k = sum(1 for r in release_rows if as_bool(r.get("breakaway_released")))
        release_rate = float(release_k) / float(len(release_rows)) if release_rows else float("nan")
        release_steps = [f(r.get("breakaway_release_step")) for r in release_rows if as_bool(r.get("breakaway_released"))]
        release_steps = [x for x in release_steps if np.isfinite(x)]
        out[policy] = {
            "success_count": int(k),
            "total": int(n),
            "success_rate": float(rate),
            "breakaway_released_count": int(release_k),
            "breakaway_release_total": int(len(release_rows)),
            "breakaway_released_rate": float(release_rate),
            "breakaway_release_step_mean": float(np.mean(release_steps)) if release_steps else None,
            "breakaway_max_disp_seen_mean": mean_value(rs, "breakaway_max_disp_seen"),
            "final_fraction_mean": mean_value(rs, "final_fraction"),
            "final_curve_mean": mean_value(rs, "final_curve"),
            "future_divergence_vs_free_mean": mean_value(rs, "final_chamfer_to_free"),
        }
    return out


def oracle_best_stats(rows: List[Dict[str, str]]) -> tuple[int, int, float]:
    oracle_rows = [r for r in rows if r.get("policy") in ORACLE_POLICIES]
    by_seed: Dict[int, List[Dict[str, str]]] = defaultdict(list)
    for r in oracle_rows:
        by_seed[safe_int(r.get("visible_seed"), -1)].append(r)
    total = len(by_seed)
    count = sum(1 for rs in by_seed.values() if any(row_success(r) for r in rs))
    return count, total, float(count) / float(total) if total else float("nan")


def classify(nominal_success: float, oracle_for_classification: float) -> str:
    gap = oracle_for_classification - nominal_success
    if oracle_for_classification <= 0.05:
        return "impossible_diagnostic"
    if nominal_success >= 0.80 and oracle_for_classification >= 0.80:
        return "weak_easy_control"
    if nominal_success <= 0.40 and oracle_for_classification >= 0.60 and gap >= 0.30:
        return "recoverable_cps_candidate"
    if nominal_success <= 0.50 and oracle_for_classification >= 0.50 and gap >= 0.20:
        return "near_recoverable_candidate"
    return "ambiguous_needs_tuning"


def summarize_condition_rows(condition: str, rows: List[Dict[str, str]], label: str = "") -> Dict[str, Any]:
    stats = policy_stats(rows)
    nominal = stats.get("nominal", {"success_count": 0, "total": 0, "success_rate": float("nan")})
    guided = stats.get("guided_search", {"success_count": 0, "total": 0, "success_rate": float("nan")})
    breakaway = stats.get("oracle_breakaway_then_place", {"success_count": 0, "total": 0, "success_rate": float("nan")})
    oracle_rows = [r for r in rows if r.get("policy") in ORACLE_POLICIES]
    oracle_mean_k, oracle_mean_n, oracle_mean = success_count_total(oracle_rows)
    oracle_best_k, oracle_best_n, oracle_best = oracle_best_stats(rows)
    oracle_break = float(breakaway["success_rate"])
    oracle_for = max(
        oracle_best if np.isfinite(oracle_best) else -1.0,
        oracle_break if np.isfinite(oracle_break) else -1.0,
    )
    if oracle_for < 0:
        oracle_for = 0.0
    nominal_rate = float(nominal["success_rate"])
    if not np.isfinite(nominal_rate):
        nominal_rate = 0.0
    release_rows = [r for r in rows if "breakaway_released" in r]
    release_k = sum(1 for r in release_rows if as_bool(r.get("breakaway_released")))
    release_rate = float(release_k) / float(len(release_rows)) if release_rows else float("nan")
    release_steps = [f(r.get("breakaway_release_step")) for r in release_rows if as_bool(r.get("breakaway_released"))]
    release_steps = [x for x in release_steps if np.isfinite(x)]
    return {
        "label": label or condition,
        "condition": condition,
        "policy_stats": stats,
        "nominal_success_count": int(nominal["success_count"]),
        "nominal_total": int(nominal["total"]),
        "nominal_success": nominal_rate,
        "guided_search_success_count": int(guided["success_count"]),
        "guided_search_total": int(guided["total"]),
        "guided_search_success": float(guided["success_rate"]),
        "oracle_mean_success_count": int(oracle_mean_k),
        "oracle_mean_total": int(oracle_mean_n),
        "oracle_mean_success": float(oracle_mean),
        "oracle_best_success_count": int(oracle_best_k),
        "oracle_best_total": int(oracle_best_n),
        "oracle_best_success": float(oracle_best),
        "oracle_breakaway_then_place_success_count": int(breakaway["success_count"]),
        "oracle_breakaway_then_place_total": int(breakaway["total"]),
        "oracle_breakaway_then_place_success": oracle_break,
        "oracle_for_classification": float(oracle_for),
        "gap": float(oracle_for - nominal_rate),
        "future_divergence_vs_free": mean_value(rows, "final_chamfer_to_free"),
        "class": classify(nominal_rate, oracle_for),
        "num_trials": len(rows),
        "breakaway_released_count": int(release_k),
        "breakaway_release_total": int(len(release_rows)),
        "breakaway_released_rate": float(release_rate),
        "breakaway_release_step_mean": float(np.mean(release_steps)) if release_steps else None,
        "breakaway_max_disp_seen_mean": mean_value(rows, "breakaway_max_disp_seen"),
    }


def compute_search_sanity(rows: List[Dict[str, str]]) -> Dict[str, Any]:
    free_rows = [r for r in rows if r.get("condition") == "free"]
    s = summarize_condition_rows("free", free_rows)
    return {
        "free_nominal_success_count": s["nominal_success_count"],
        "free_nominal_total": s["nominal_total"],
        "free_nominal_success": s["nominal_success"],
        "free_guided_search_success_count": s["guided_search_success_count"],
        "free_guided_search_total": s["guided_search_total"],
        "free_guided_search_success": s["guided_search_success"],
        "search_sanity_pass": bool(s["nominal_success"] >= 0.95 and s["guided_search_success"] >= 0.50),
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
    selected: Optional[Dict[str, Any]] = None
    verdict = "FAIL"
    if candidates and search_sanity["search_sanity_pass"]:
        selected = sorted(candidates, key=lambda r: (r["gap"], r["oracle_for_classification"], -r["nominal_success"]), reverse=True)[0]
        verdict = "PASS"
    elif candidates or near:
        selected = sorted(candidates or near, key=lambda r: (r["gap"], r["oracle_for_classification"], -r["nominal_success"]), reverse=True)[0]
        verdict = "WARN"

    recommendation = (
        "Candidate proposed only. Phase2.5c confirmation is required before Phase3 medium."
        if verdict == "PASS"
        else "No fully qualified recoverable branch yet. Continue tuning; do not run Phase3 medium."
    )
    if not search_sanity["search_sanity_pass"]:
        recommendation = "Search sanity failed; search evidence is invalid. Fix guided_search before interpreting search_best_success."

    payload = sanitize_json({
        "verdict": verdict,
        "requires_phase2_5c_confirmation": True,
        "condition_table": table,
        "parameter_sweep_table": sweep_table,
        "search_sanity": search_sanity,
        "selected_recoverable_condition": selected["condition"] if selected and verdict == "PASS" else None,
        "selected_recoverable_config": selected["label"] if selected and verdict == "PASS" else None,
        "selected_row": selected,
        "candidate_rows": candidates,
        "near_candidate_rows": near,
        "recommendation": recommendation,
        "submodule_runtime": {
            "audit_runtime": sub_summary.get("audit_runtime", {}),
            "num_rows": sub_summary.get("num_rows"),
            "conditions": sub_summary.get("conditions", []),
            "policies": sub_summary.get("policies", []),
            "source_summary_json": str(Path(args.summary_json)),
        },
    })

    out_json = root / args.out_json
    out_md = root / args.out_md
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))

    lines = [
        "# Phase2.5 Hidden-Contact Recoverability Audit",
        "",
        "## Verdict",
        "",
        f"- Verdict: `{verdict}`",
        f"- Requires Phase2.5c confirmation: `True`",
        f"- Selected recoverable condition: `{payload['selected_recoverable_condition']}`",
        f"- Selected recoverable config: `{payload['selected_recoverable_config']}`",
        f"- Recommendation: {recommendation}",
        "",
        "## Warning",
        "",
        "This grid selector proposes a candidate only. A candidate with oracle success below 0.60 must not unlock Phase3 medium. Phase2.5c confirmation is required.",
        "",
        "## Search Sanity",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| free nominal success | {fmt_count(search_sanity['free_nominal_success_count'], search_sanity['free_nominal_total'], search_sanity['free_nominal_success'])} |",
        f"| free guided_search success | {fmt_count(search_sanity['free_guided_search_success_count'], search_sanity['free_guided_search_total'], search_sanity['free_guided_search_success'])} |",
        f"| search sanity pass | {search_sanity['search_sanity_pass']} |",
        "",
        "## Condition Summary",
        "",
        "| Condition | Nominal | Guided Search | Oracle Mean | Oracle Best | Oracle Breakaway | Gap(best-nominal) | Class |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in table:
        lines.append(
            "| `{condition}` | {nom} | {guided} | {omean} | {obest} | {obreak} | {gap:.3f} | {cls} |".format(
                condition=r["condition"],
                nom=fmt_count(r["nominal_success_count"], r["nominal_total"], r["nominal_success"]),
                guided=fmt_count(r["guided_search_success_count"], r["guided_search_total"], r["guided_search_success"]),
                omean=fmt_count(r["oracle_mean_success_count"], r["oracle_mean_total"], r["oracle_mean_success"]),
                obest=fmt_count(r["oracle_best_success_count"], r["oracle_best_total"], r["oracle_best_success"]),
                obreak=fmt_count(r["oracle_breakaway_then_place_success_count"], r["oracle_breakaway_then_place_total"], r["oracle_breakaway_then_place_success"]),
                gap=r["gap"],
                cls=r["class"],
            )
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- `oracle_mean_success` is reported for transparency but is not used for classification.",
        "- Classification uses `max(oracle_best_success, oracle_breakaway_then_place_success)`.",
        "- Phase2.5b cannot by itself unlock Phase3 medium; Phase2.5c confirmation is required.",
        "- `hidden_pin` remains hard diagnostic, not the CPS success-improvement target.",
    ]
    out_md.write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))

    if verdict == "FAIL":
        raise SystemExit("[Phase2.5][FAIL] no recoverable hidden-contact condition selected")


if __name__ == "__main__":
    main()
