#!/usr/bin/env python3
"""Strict Phase2.5c confirmation for the selected recoverable CCDA branch."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

ORACLE_POLICIES = {
    "oracle_pull",
    "oracle_regrasp",
    "oracle_wiggle",
    "oracle_breakaway_then_place",
    "oracle_partial_release_then_place",
}

SELECTED_CONDITION = "hidden_breakaway_pin"
SELECTED_CONFIG = "breakaway_force_2p6_disp_0p045_pull_0p36"
SELECTED_ENV = {
    "CCDA_BREAKAWAY_FORCE": "2.6",
    "CCDA_BREAKAWAY_DISP": "0.045",
    "CCDA_BREAKAWAY_BEAD_RATIO": "0.45",
    "CCDA_ORACLE_BREAKAWAY_PULL_DIST": "0.36",
}

THRESHOLDS = {
    "free_nominal_min": 0.95,
    "free_guided_min": 0.50,
    "hidden_pin_oracle_best_max": 0.05,
    "selected_nominal_max": 0.40,
    "selected_oracle_breakaway_min": 0.60,
    "selected_oracle_best_min": 0.60,
    "selected_gap_min": 0.30,
    "selected_breakaway_release_min": 0.50,
}


def safe_float(x, default=float("nan")):
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default


def safe_int(x, default=0):
    try:
        return int(float(x))
    except Exception:
        return default


def parse_bool(x):
    return str(x).strip().lower() in {"true", "1", "yes"}


def read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def success_count(rows):
    return sum(1 for r in rows if parse_bool(r.get("success"))), len(rows)


def rate(k, n):
    return float(k) / float(n) if n else float("nan")


def by_key(rows, key):
    out = defaultdict(list)
    for r in rows:
        out[r.get(key, "")].append(r)
    return out


def policy_stats(rows):
    out = {}
    for pol, rs in by_key(rows, "policy").items():
        k, n = success_count(rs)
        release_rows = [r for r in rs if "breakaway_released" in r]
        rel_k = sum(1 for r in release_rows if parse_bool(r.get("breakaway_released")))
        out[pol] = {
            "success_count": k,
            "total": n,
            "success_rate": rate(k, n),
            "breakaway_released_count": rel_k,
            "breakaway_release_total": len(release_rows),
            "breakaway_released_rate": rate(rel_k, len(release_rows)),
        }
    return out


def oracle_best_stats(rows):
    oracle_rows = [r for r in rows if r.get("policy") in ORACLE_POLICIES]
    by_seed = defaultdict(list)
    for r in oracle_rows:
        by_seed[safe_int(r.get("visible_seed"), -1)].append(r)
    total = len(by_seed)
    count = sum(1 for rs in by_seed.values() if any(parse_bool(r.get("success")) for r in rs))
    return count, total, rate(count, total)


def condition_summary(rows, condition):
    rs = [r for r in rows if r.get("condition") == condition]
    stats = policy_stats(rs)

    nominal = stats.get("nominal", {"success_count": 0, "total": 0, "success_rate": float("nan")})
    guided = stats.get("guided_search", {"success_count": 0, "total": 0, "success_rate": float("nan")})
    breakaway = stats.get("oracle_breakaway_then_place", {"success_count": 0, "total": 0, "success_rate": float("nan")})

    oracle_rows = [r for r in rs if r.get("policy") in ORACLE_POLICIES]
    oracle_mean_k, oracle_mean_n = success_count(oracle_rows)
    oracle_best_k, oracle_best_n, oracle_best_rate = oracle_best_stats(rs)

    release_rows = [r for r in rs if "breakaway_released" in r]
    release_k = sum(1 for r in release_rows if parse_bool(r.get("breakaway_released")))

    return {
        "condition": condition,
        "policy_stats": stats,
        "nominal_success_count": nominal["success_count"],
        "nominal_total": nominal["total"],
        "nominal_success": nominal["success_rate"],
        "guided_search_success_count": guided["success_count"],
        "guided_search_total": guided["total"],
        "guided_search_success": guided["success_rate"],
        "oracle_mean_success_count": oracle_mean_k,
        "oracle_mean_total": oracle_mean_n,
        "oracle_mean_success": rate(oracle_mean_k, oracle_mean_n),
        "oracle_best_success_count": oracle_best_k,
        "oracle_best_total": oracle_best_n,
        "oracle_best_success": oracle_best_rate,
        "oracle_breakaway_success_count": breakaway["success_count"],
        "oracle_breakaway_total": breakaway["total"],
        "oracle_breakaway_success": breakaway["success_rate"],
        "breakaway_released_count": release_k,
        "breakaway_release_total": len(release_rows),
        "breakaway_released_rate": rate(release_k, len(release_rows)),
    }


def finite_or_neg(x):
    return x if x == x else -1.0


def frac(count, total, value):
    if total:
        return f"{count}/{total} = {value:.3f}"
    return "NA"


def classify_selected(sel):
    nominal = sel["nominal_success"]
    oracle_best = sel["oracle_best_success"]
    oracle_break = sel["oracle_breakaway_success"]
    oracle_for_pass = max(finite_or_neg(oracle_best), finite_or_neg(oracle_break))
    gap = oracle_for_pass - nominal
    release_rate = sel["breakaway_released_rate"]
    checks = {
        "selected_nominal_low": nominal <= THRESHOLDS["selected_nominal_max"],
        "selected_oracle_breakaway_or_best_high": oracle_for_pass >= THRESHOLDS["selected_oracle_best_min"],
        "selected_gap_high": gap >= THRESHOLDS["selected_gap_min"],
        "selected_breakaway_released_high": release_rate >= THRESHOLDS["selected_breakaway_release_min"],
    }
    return checks, oracle_for_pass, gap


def write_next_tuning_plan(path: Path, checks: Dict[str, bool], selected: Dict[str, Any], free: Dict[str, Any], hidden_pin: Dict[str, Any]) -> None:
    lines = [
        "# Phase2.5c Next Tuning Plan",
        "",
        "Phase2.5c confirmation failed. Do not run Phase3 medium.",
        "",
        "## Failed Checks",
        "",
    ]
    for k, v in checks.items():
        if not v:
            lines.append(f"- `{k}`")
    lines += ["", "## Suggested Actions", ""]
    if not checks.get("selected_oracle_breakaway_or_best_high", True):
        lines += [
            "- Increase `CCDA_ORACLE_BREAKAWAY_PULL_DIST` to `0.40` or `0.44`.",
            "- Or lower `CCDA_BREAKAWAY_DISP` to `0.040`.",
            "- Or lower `CCDA_BREAKAWAY_FORCE` to `2.2` / `2.4`.",
            "- Check whether successful release samples have `breakaway_release_step <= 3`.",
        ]
    if not checks.get("selected_nominal_low", True):
        lines += [
            "- Increase `CCDA_BREAKAWAY_FORCE` to `2.8` / `3.0`.",
            "- Increase `CCDA_BREAKAWAY_DISP` to `0.050`.",
            "- Try `CCDA_BREAKAWAY_BEAD_RATIO=0.35` or `0.55`.",
        ]
    if not checks.get("selected_breakaway_released_high", True):
        lines += [
            "- Lower the release threshold.",
            "- Verify `_maybe_update_breakaway()` is called every step.",
            "- Verify breakaway metadata is updated in real time.",
        ]
    if not checks.get("free_guided_ok", True):
        lines += [
            "- Search sanity failed; do not interpret `search_best_success`.",
            "- Fix `guided_search` before selecting recoverable branches.",
        ]
    path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/state_diff2")
    ap.add_argument("--trials_csv", required=True)
    ap.add_argument("--summary_json", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_md", required=True)
    ap.add_argument("--out_plan", required=True)
    args = ap.parse_args()

    root = Path(args.root)
    rows = read_csv(Path(args.trials_csv))
    condition_summaries = {c: condition_summary(rows, c) for c in sorted({r.get("condition") for r in rows})}

    free = condition_summaries.get("free", condition_summary(rows, "free"))
    hidden_pin = condition_summaries.get("hidden_pin", condition_summary(rows, "hidden_pin"))
    selected = condition_summaries.get(SELECTED_CONDITION, condition_summary(rows, SELECTED_CONDITION))

    selected_checks, selected_oracle_for_pass, selected_gap = classify_selected(selected)
    confirmation_n = int(free.get("nominal_total", 0) or 0)
    preferred_confirmation_n = 30
    confirmation_scale_note = (
        f"confirmation_n = {confirmation_n}, weaker than preferred n=30"
        if confirmation_n and confirmation_n < preferred_confirmation_n
        else f"confirmation_n = {confirmation_n}"
    )
    checks = {
        "free_nominal_ok": free["nominal_success"] >= THRESHOLDS["free_nominal_min"],
        "free_guided_ok": free["guided_search_success"] >= THRESHOLDS["free_guided_min"],
        "hidden_pin_oracle_best_low": hidden_pin["oracle_best_success"] <= THRESHOLDS["hidden_pin_oracle_best_max"],
        **selected_checks,
    }
    verdict = "PASS" if all(checks.values()) else "FAIL"

    payload = {
        "verdict": verdict,
        "selected_recoverable_condition": SELECTED_CONDITION if verdict == "PASS" else None,
        "selected_recoverable_config": SELECTED_CONFIG if verdict == "PASS" else None,
        "selected_recoverable_env": SELECTED_ENV if verdict == "PASS" else {},
        "thresholds": THRESHOLDS,
        "checks": checks,
        "selected_gap": selected_gap,
        "selected_oracle_for_pass": selected_oracle_for_pass,
        "condition_summaries": condition_summaries,
        "raw_summary_path": str(args.summary_json),
        "trials_csv": str(args.trials_csv),
        "num_rows": len(rows),
        "confirmation_n": confirmation_n,
        "preferred_confirmation_n": preferred_confirmation_n,
        "confirmation_scale_note": confirmation_scale_note,
        "recommendation": (
            "Phase2.5c confirmed selected recoverable branch. Phase3 medium may be configured with free/hidden_pin/hidden_high_friction/hidden_breakaway_pin and primary pair free vs hidden_breakaway_pin, but do not start it without user confirmation."
            if verdict == "PASS"
            else "Phase2.5c did not confirm selected recoverable branch. Do not run Phase3 medium; inspect per-policy counts and tune breakaway/oracle."
        ),
    }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(payload, indent=2, sort_keys=True))

    lines = [
        "# Phase2.5c Recoverability Confirmation Report",
        "",
        "## Verdict",
        "",
        f"- Verdict: `{verdict}`",
        f"- Selected condition if PASS: `{payload['selected_recoverable_condition']}`",
        f"- Selected config if PASS: `{payload['selected_recoverable_config']}`",
        f"- Confirmation scale: `{confirmation_scale_note}`",
        "",
        "## Strict Checks",
        "",
        "| Check | Pass |",
        "|---|---:|",
    ]
    for k, v in checks.items():
        lines.append(f"| `{k}` | `{v}` |")

    lines += [
        "",
        "## Condition Summary with Counts",
        "",
        "| Condition | Nominal | Guided Search | Oracle Mean | Oracle Best | Oracle Breakaway | Breakaway Released |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for c in ["free", "hidden_pin", "hidden_high_friction", SELECTED_CONDITION]:
        if c not in condition_summaries:
            continue
        s = condition_summaries[c]
        lines.append(
            f"| `{c}` | "
            f"{frac(s['nominal_success_count'], s['nominal_total'], s['nominal_success'])} | "
            f"{frac(s['guided_search_success_count'], s['guided_search_total'], s['guided_search_success'])} | "
            f"{frac(s['oracle_mean_success_count'], s['oracle_mean_total'], s['oracle_mean_success'])} | "
            f"{frac(s['oracle_best_success_count'], s['oracle_best_total'], s['oracle_best_success'])} | "
            f"{frac(s['oracle_breakaway_success_count'], s['oracle_breakaway_total'], s['oracle_breakaway_success'])} | "
            f"{frac(s['breakaway_released_count'], s['breakaway_release_total'], s['breakaway_released_rate'])} |"
        )

    lines += ["", "## Per-Policy Counts", ""]
    for c in ["free", "hidden_pin", "hidden_high_friction", SELECTED_CONDITION]:
        if c not in condition_summaries:
            continue
        lines += [
            f"### `{c}`",
            "",
            "| Policy | Success | Total | Rate | Breakaway Released |",
            "|---|---:|---:|---:|---:|",
        ]
        for pol, st in sorted(condition_summaries[c]["policy_stats"].items()):
            lines.append(
                f"| `{pol}` | {st['success_count']} | {st['total']} | {st['success_rate']:.3f} | "
                f"{st['breakaway_released_count']}/{st['breakaway_release_total']} = {st['breakaway_released_rate']:.3f} |"
            )
        lines.append("")

    lines += [
        "## Interpretation",
        "",
        "- This report uses count/total statistics to avoid over-interpreting small-sample fractions.",
        "- `oracle_mean_success` is not used as the main recoverability criterion.",
        "- Recoverability is confirmed only if `oracle_best_success` or `oracle_breakaway_then_place_success` is at least 0.60.",
        "- `hidden_pin` remains hard diagnostic and is not the CPS success-improvement target.",
    ]
    Path(args.out_md).write_text("\n".join(lines) + "\n")

    plan = [
        "# Phase3 Condition Plan After Phase2.5c",
        "",
        f"- Phase2.5c verdict: `{verdict}`",
        "- hard diagnostic condition: `hidden_pin`",
        "- weak/auxiliary condition: `hidden_high_friction`",
        f"- selected recoverable CPS branch: `{payload['selected_recoverable_condition']}`",
        f"- selected recoverable config: `{payload['selected_recoverable_config']}`",
        "",
        "## Selected Env Vars",
        "",
    ]
    if verdict == "PASS":
        for k, v in SELECTED_ENV.items():
            plan.append(f"- `{k}={v}`")
        plan += [
            "",
            "## Phase3 medium should use",
            "",
            "conditions:",
            "- `free`",
            "- `hidden_pin`",
            "- `hidden_high_friction`",
            "- `hidden_breakaway_pin`",
            "",
            "primary_branch_pair:",
            "- `free` vs `hidden_breakaway_pin`",
            "",
            "diagnostic_branch_pair:",
            "- `free` vs `hidden_pin`",
        ]
    else:
        plan += [
            "- None, because confirmation failed.",
            "",
            "## Phase3 medium status",
            "",
            "- Do not run Phase3 medium.",
            "- Inspect per-policy counts and continue tuning breakaway/oracle.",
        ]
        write_next_tuning_plan(root / "reports/phase2_5c_next_tuning_plan.md", checks, selected, free, hidden_pin)

    Path(args.out_plan).write_text("\n".join(plan) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if verdict != "PASS":
        raise SystemExit("[Phase2.5c][FAIL] confirmation failed; do not run Phase3 medium")


if __name__ == "__main__":
    main()
