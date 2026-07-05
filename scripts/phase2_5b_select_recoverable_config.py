#!/usr/bin/env python3
"""Select a recoverable hidden-contact config from Phase2.5b grid summaries."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


CONFIG_ENVS = {
    "breakaway_default": {
        "CCDA_BREAKAWAY_FORCE": "1.2",
        "CCDA_BREAKAWAY_DISP": "0.035",
        "CCDA_BREAKAWAY_BEAD_RATIO": "0.45",
        "CCDA_ORACLE_BREAKAWAY_PULL_DIST": "0.16",
    },
    "breakaway_force_1p0_disp_0p025_pull_0p20": {
        "CCDA_BREAKAWAY_FORCE": "1.0",
        "CCDA_BREAKAWAY_DISP": "0.025",
        "CCDA_BREAKAWAY_BEAD_RATIO": "0.45",
        "CCDA_ORACLE_BREAKAWAY_PULL_DIST": "0.20",
    },
    "breakaway_force_1p2_disp_0p025_pull_0p22": {
        "CCDA_BREAKAWAY_FORCE": "1.2",
        "CCDA_BREAKAWAY_DISP": "0.025",
        "CCDA_BREAKAWAY_BEAD_RATIO": "0.45",
        "CCDA_ORACLE_BREAKAWAY_PULL_DIST": "0.22",
    },
    "breakaway_force_1p5_disp_0p030_pull_0p22": {
        "CCDA_BREAKAWAY_FORCE": "1.5",
        "CCDA_BREAKAWAY_DISP": "0.030",
        "CCDA_BREAKAWAY_BEAD_RATIO": "0.45",
        "CCDA_ORACLE_BREAKAWAY_PULL_DIST": "0.22",
    },
    "breakaway_force_1p8_disp_0p035_pull_0p25": {
        "CCDA_BREAKAWAY_FORCE": "1.8",
        "CCDA_BREAKAWAY_DISP": "0.035",
        "CCDA_BREAKAWAY_BEAD_RATIO": "0.45",
        "CCDA_ORACLE_BREAKAWAY_PULL_DIST": "0.25",
    },
    "breakaway_force_2p2_disp_0p040_pull_0p32": {
        "CCDA_BREAKAWAY_FORCE": "2.2",
        "CCDA_BREAKAWAY_DISP": "0.040",
        "CCDA_BREAKAWAY_BEAD_RATIO": "0.45",
        "CCDA_ORACLE_BREAKAWAY_PULL_DIST": "0.32",
    },
    "breakaway_force_2p6_disp_0p045_pull_0p36": {
        "CCDA_BREAKAWAY_FORCE": "2.6",
        "CCDA_BREAKAWAY_DISP": "0.045",
        "CCDA_BREAKAWAY_BEAD_RATIO": "0.45",
        "CCDA_ORACLE_BREAKAWAY_PULL_DIST": "0.36",
    },
    "breakaway_force_3p0_disp_0p050_pull_0p40": {
        "CCDA_BREAKAWAY_FORCE": "3.0",
        "CCDA_BREAKAWAY_DISP": "0.050",
        "CCDA_BREAKAWAY_BEAD_RATIO": "0.45",
        "CCDA_ORACLE_BREAKAWAY_PULL_DIST": "0.40",
    },
}


def read_json(p):
    return json.loads(Path(p).read_text())


def classify(nominal, oracle):
    gap = oracle - nominal
    if oracle <= 0.05:
        return "impossible_diagnostic"
    if nominal >= 0.80 and oracle >= 0.80:
        return "weak_easy_control"
    if nominal <= 0.40 and oracle >= 0.60 and gap >= 0.30:
        return "recoverable_cps_candidate"
    if nominal <= 0.50 and oracle >= 0.50 and gap >= 0.20:
        return "near_recoverable_candidate"
    return "ambiguous_needs_tuning"


def get_table(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    table = payload.get("condition_table") or payload.get("table") or []
    if table:
        return table
    if "condition_summaries" in payload:
        for cond, item in payload["condition_summaries"].items():
            agg = item.get("aggregate", {})
            table.append({
                "condition": cond,
                "nominal_success": agg.get("nominal_success", 0.0),
                "oracle_success": agg.get("oracle_success", 0.0),
                "search_best_success": agg.get("search_best_success", 0.0),
                "gap": agg.get("oracle_gap", 0.0),
                "class": "",
            })
    return table


def row_float(row: Dict[str, Any], key: str) -> float:
    try:
        return float(row.get(key, 0.0) or 0.0)
    except Exception:
        return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid_dir", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_md", required=True)
    args = ap.parse_args()

    grid_dir = Path(args.grid_dir)
    rows = []
    condition_rows = []
    free_guided_values = []
    free_nominal_values = []

    for p in sorted(grid_dir.glob("*_summary.json")):
        label = p.name.replace("_summary.json", "")
        payload = read_json(p)
        search_sanity = payload.get("search_sanity", {})
        if "free_guided_search_success" in search_sanity:
            free_guided_values.append(float(search_sanity.get("free_guided_search_success", 0.0)))
        if "free_nominal_success" in search_sanity:
            free_nominal_values.append(float(search_sanity.get("free_nominal_success", 0.0)))

        for r in get_table(payload):
            cond = r.get("condition")
            labelled_row = dict(r)
            labelled_row["label"] = label
            condition_rows.append(labelled_row)
            if cond == "free":
                free_nominal_values.append(row_float(r, "nominal_success"))
                free_guided_values.append(row_float(r, "search_best_success"))
            if cond not in {"hidden_breakaway_pin", "hidden_partial_pin", "hidden_soft_pin"}:
                continue
            nominal = row_float(r, "nominal_success")
            oracle = row_float(r, "oracle_success")
            search = row_float(r, "search_best_success")
            cls = classify(nominal, oracle)
            rows.append({
                "label": label,
                "condition": cond,
                "nominal_success": nominal,
                "oracle_success": oracle,
                "search_best_success": search,
                "gap": oracle - nominal,
                "class": cls,
                "search_sanity_pass": bool(search_sanity.get("search_sanity_pass", False)),
                "breakaway_released_rate": r.get("breakaway_released_rate"),
                "breakaway_release_step_mean": r.get("breakaway_release_step_mean"),
                "breakaway_max_disp_seen_mean": r.get("breakaway_max_disp_seen_mean"),
            })

    free_nominal = max(free_nominal_values) if free_nominal_values else 0.0
    free_guided = max(free_guided_values) if free_guided_values else 0.0
    search_sanity_pass = bool(free_nominal >= 0.95 and free_guided >= 0.50)
    candidates = [r for r in rows if r["class"] == "recoverable_cps_candidate"]
    near = [r for r in rows if r["class"] == "near_recoverable_candidate"]

    selected: Optional[Dict[str, Any]] = None
    verdict = "FAIL"
    if candidates and search_sanity_pass:
        selected = sorted(candidates, key=lambda r: (r["gap"], r["oracle_success"], -r["nominal_success"]), reverse=True)[0]
        verdict = "PASS"
    elif candidates or near:
        selected = sorted(candidates or near, key=lambda r: (r["gap"], r["oracle_success"], -r["nominal_success"]), reverse=True)[0]
        verdict = "WARN"

    selected_env = CONFIG_ENVS.get(selected["label"], {}) if selected else {}

    payload = {
        "verdict": verdict,
        "selected_recoverable_condition": selected["condition"] if selected and verdict == "PASS" else None,
        "selected_recoverable_config": selected["label"] if selected and verdict == "PASS" else None,
        "selected_recoverable_env": selected_env if selected and verdict == "PASS" else {},
        "selected_row": selected,
        "condition_table": condition_rows,
        "candidate_rows": candidates,
        "near_candidate_rows": near,
        "all_rows": rows,
        "search_sanity": {
            "free_nominal_success": free_nominal,
            "free_guided_search_success": free_guided,
            "search_sanity_pass": search_sanity_pass,
        },
        "pass_thresholds": {
            "nominal_success_max": 0.40,
            "oracle_success_min": 0.60,
            "oracle_gap_min": 0.30,
        },
        "requires_phase2_5c_confirmation": True,
        "recommendation": (
            "Phase2.5b grid proposes a candidate only; Phase2.5c confirmation is required before Phase3 medium."
            if verdict == "PASS"
            else "No fully qualified recoverable branch yet. Continue tuning; do not run Phase3 medium."
        ),
    }

    Path(args.out_json).write_text(json.dumps(payload, indent=2, sort_keys=True))

    lines = [
        "# Phase2.5b Recoverability Grid Selection",
        "",
        "## Verdict",
        "",
        f"- Verdict: `{verdict}`",
        f"- Selected recoverable condition: `{payload['selected_recoverable_condition']}`",
        f"- Selected recoverable config: `{payload['selected_recoverable_config']}`",
        f"- Selected env vars: `{json.dumps(payload.get('selected_recoverable_env', {}), sort_keys=True)}`",
        "",
        "## Warning",
        "",
        "This grid selector proposes a candidate only. A candidate with oracle success below 0.60 must not unlock Phase3 medium. Phase2.5c confirmation is required.",
        "",
        "## Search Sanity",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| free nominal success | {free_nominal:.3f} |",
        f"| free guided_search success | {free_guided:.3f} |",
        f"| search sanity pass | {search_sanity_pass} |",
        "",
        "## Audited Conditions",
        "",
        "| Config | Condition | Nominal | Oracle | Search | Gap | Class |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for r in sorted(condition_rows, key=lambda x: (str(x.get("label")), str(x.get("condition")))):
        lines.append(
            "| {label} | {condition} | {nominal_success:.3f} | {oracle_success:.3f} | "
            "{search_best_success:.3f} | {gap:.3f} | {class} |".format(**r)
        )
    lines += [
        "",
        "## Candidate Table",
        "",
        "| Config | Condition | Nominal | Oracle | Search | Gap | Class |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for r in sorted(rows, key=lambda x: (x["condition"], x["label"])):
        lines.append(
            "| {label} | {condition} | {nominal_success:.3f} | {oracle_success:.3f} | "
            "{search_best_success:.3f} | {gap:.3f} | {class} |".format(**r)
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- PASS means this grid proposes a candidate; Phase2.5c confirmation is still required before Phase3 medium.",
        "- WARN means a near candidate exists but is not strong enough for paper-level CPS success improvement.",
        "- FAIL means do not run Phase3 medium.",
        "- `hidden_pin` remains hard diagnostic, not the CPS success-improvement target.",
    ]
    if payload.get("selected_recoverable_env"):
        lines += ["", "## Selected Environment Variables", ""]
        for k, v in sorted(payload["selected_recoverable_env"].items()):
            lines.append(f"- `{k}={v}`")
    Path(args.out_md).write_text("\n".join(lines) + "\n")

    if verdict == "PASS":
        plan = Path(args.out_json).resolve().parent / "phase3_condition_plan_after_recoverability.md"
        plan.write_text(
            "# Phase3 Condition Plan After Recoverability Audit\n\n"
            "- hard diagnostic condition: `hidden_pin`\n"
            "- weak/control condition: `hidden_high_friction`\n"
            f"- selected recoverable CPS branch: `{payload['selected_recoverable_condition']}`\n"
            f"- selected recoverable config: `{payload['selected_recoverable_config']}`\n\n"
            "## Selected Env Vars\n\n"
            + "".join(f"- `{k}={v}`\n" for k, v in sorted(payload["selected_recoverable_env"].items()))
            + "\n## Phase3 medium should use\n\n"
            "conditions:\n"
            "- `free`\n"
            "- `hidden_pin`\n"
            "- `hidden_high_friction`\n"
            f"- `{payload['selected_recoverable_condition']}`\n\n"
            "primary_branch_pair:\n"
            f"- `free` vs `{payload['selected_recoverable_condition']}`\n\n"
            "diagnostic_branch_pair:\n"
            "- `free` vs `hidden_pin`\n"
        )

    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()