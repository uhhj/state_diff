#!/usr/bin/env python3
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

REQUIRED_CONDITIONS = ["free", "hidden_pin", "hidden_high_friction", "hidden_breakaway_pin"]

def parse_bool(x):
    return str(x).strip().lower() in {"1", "true", "yes", "pass", "success"}

def sf(x):
    try:
        if x is None or x == "":
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")

def mean(xs):
    xs = [x for x in xs if math.isfinite(x)]
    return float(sum(xs) / len(xs)) if xs else float("nan")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/state_diff2")
    ap.add_argument("--trials_csv", default="reports/phase3_6_matched_action_replay_trials.csv")
    ap.add_argument("--out_json", default="reports/phase3_6_matched_replay_summary.json")
    ap.add_argument("--out_md", default="reports/phase3_6_matched_replay_report.md")
    args = ap.parse_args()

    root = Path(args.root)
    path = root / args.trials_csv
    issues = []

    def issue(level, name, detail):
        issues.append({"level": level, "name": name, "detail": str(detail)})

    if not path.exists():
        issue("FAIL", "missing_trials_csv", str(path))
        rows = []
    else:
        with path.open(newline="") as f:
            rows = list(csv.DictReader(f))

    if not rows:
        issue("FAIL", "no_rows", "No matched replay rows.")

    by = defaultdict(list)
    for r in rows:
        key = (
            r.get("test_type", ""),
            r.get("condition", ""),
            r.get("prefix_source", ""),
            r.get("motion_timeout", ""),
        )
        by[key].append(r)

    table = []
    for (test_type, condition, prefix_source, timeout), group in sorted(by.items()):
        successes = [parse_bool(r.get("success")) for r in group]
        failures = [r for r in group if r.get("failure_reason")]
        valid = [parse_bool(r.get("action_valid")) for r in group]
        deltas = [sf(r.get("delta_final_fraction")) for r in group]
        afters = [sf(r.get("final_fraction_after")) for r in group]
        state_l2 = [sf(r.get("prefix_state_l2_error")) for r in group]
        state_mae = [sf(r.get("prefix_state_mean_abs_error")) for r in group]
        prefix_fail = [parse_bool(r.get("prefix_had_failure")) for r in group]
        prefix_done = [parse_bool(r.get("prefix_reached_done_or_success")) for r in group]

        table.append({
            "test_type": test_type,
            "condition": condition,
            "prefix_source": prefix_source,
            "timeout": timeout,
            "rows": len(group),
            "success_count": int(sum(successes)),
            "success_rate": float(sum(successes) / len(group)) if group else float("nan"),
            "failure_count": len(failures),
            "valid_action_rate": float(sum(valid) / len(group)) if group else float("nan"),
            "mean_delta_final_fraction": mean(deltas),
            "mean_final_fraction_after": mean(afters),
            "mean_prefix_state_l2_error": mean(state_l2),
            "mean_prefix_state_mae": mean(state_mae),
            "prefix_failure_rate": float(sum(prefix_fail) / len(group)) if group else float("nan"),
            "prefix_done_or_success_rate": float(sum(prefix_done) / len(group)) if group else float("nan"),
        })

    conditions = sorted({r.get("condition", "") for r in rows})
    if not all(c in conditions for c in REQUIRED_CONDITIONS):
        issue("FAIL", "missing_conditions", {"got": conditions, "required": REQUIRED_CONDITIONS})

    prefix_sources = sorted({r.get("prefix_source", "") for r in rows})
    high_conf = any(x == "raw_action_file_high_confidence" for x in prefix_sources)
    low_conf = any(x == "state_action_x_tail_low_confidence" for x in prefix_sources)
    no_prefix_rows = [r for r in rows if "no_prefix" in r.get("failure_reason", "") or not r.get("prefix_source")]

    if not high_conf and low_conf:
        issue(
            "WARN",
            "low_confidence_prefix_only",
            "Matched replay used state_action_x tail fallback only. Canonicalization may make this low confidence.",
        )
    if not high_conf and not low_conf:
        issue("FAIL", "no_prefix_source_used", prefix_sources)
    if no_prefix_rows:
        issue("FAIL", "some_rows_missing_prefix_actions", len(no_prefix_rows))

    # Prefix-state match threshold is deliberately loose for smoke.
    state_maes = [sf(r.get("prefix_state_mean_abs_error")) for r in rows if r.get("prefix_state_mean_abs_error")]
    state_match = bool(state_maes) and mean(state_maes) < 0.15
    if state_maes and not state_match:
        issue(
            "FAIL",
            "prefix_state_mismatch",
            f"mean prefix state MAE={mean(state_maes):.6f}; replay may not match original window state.",
        )
    if not state_maes:
        issue("WARN", "prefix_state_error_unavailable", "Could not compare live prefix state to window paper_x current slice.")

    gt_rows = [r for r in rows if r.get("test_type") == "matched_gt_y_action"]
    oracle_rows = [r for r in rows if r.get("test_type") == "same_state_oracle"]

    gt_success = sum(parse_bool(r.get("success")) for r in gt_rows)
    oracle_success = sum(parse_bool(r.get("success")) for r in oracle_rows)
    gt_delta = mean([sf(r.get("delta_final_fraction")) for r in gt_rows])
    oracle_delta = mean([sf(r.get("delta_final_fraction")) for r in oracle_rows])
    gt_failures = sum(1 for r in gt_rows if r.get("failure_reason"))
    oracle_failures = sum(1 for r in oracle_rows if r.get("failure_reason"))

    if oracle_rows and oracle_success == 0 and (not math.isfinite(oracle_delta) or oracle_delta <= 0):
        issue(
            "FAIL",
            "same_state_oracle_no_progress",
            "Oracle does not progress after matched prefix; environment/prefix reconstruction may be wrong.",
        )

    if state_match and oracle_delta > 0 and gt_delta <= 0 and gt_success == 0:
        issue(
            "FAIL",
            "matched_gt_no_progress_but_oracle_progresses",
            "Matched state appears close and oracle progresses, but matched GT y_action does not. Suspect action codec or action adapter.",
        )

    if not state_match and gt_success == 0:
        issue(
            "WARN",
            "gt_failure_not_decisive_due_state_mismatch",
            "GT replay failure is not decisive because prefix state did not match the window state.",
        )

    if gt_success > 0 or gt_delta > 0:
        issue(
            "WARN",
            "phase35_gt_failure_may_be_unmatched_reset_artifact",
            "Matched GT y_action shows progress; Phase3.5 unmatched replay failure should not be interpreted as codec failure.",
        )

    has_fail = any(i["level"] == "FAIL" for i in issues)
    has_warn = any(i["level"] == "WARN" for i in issues)
    verdict = "FAIL" if has_fail else ("WARN" if has_warn else "PASS")

    if any(i["name"] == "prefix_state_mismatch" for i in issues):
        root_cause = "matched_state_reconstruction_blocker"
    elif any(i["name"] == "matched_gt_no_progress_but_oracle_progresses" for i in issues):
        root_cause = "action_codec_or_adapter_blocker_confirmed"
    elif any(i["name"] == "phase35_gt_failure_may_be_unmatched_reset_artifact" for i in issues):
        root_cause = "phase35_unmatched_replay_artifact_possible"
    elif any(i["name"] == "same_state_oracle_no_progress" for i in issues):
        root_cause = "environment_or_prefix_replay_blocker"
    else:
        root_cause = "undetermined_matched_replay"

    payload = {
        "verdict": verdict,
        "root_cause": root_cause,
        "num_rows": len(rows),
        "prefix_sources": prefix_sources,
        "high_conf_raw_prefix_available": high_conf,
        "low_conf_prefix_used": low_conf,
        "state_match": state_match,
        "mean_prefix_state_mae": mean(state_maes),
        "gt_success_count": int(gt_success),
        "oracle_success_count": int(oracle_success),
        "gt_mean_delta_final_fraction": gt_delta,
        "oracle_mean_delta_final_fraction": oracle_delta,
        "gt_failure_count": int(gt_failures),
        "oracle_failure_count": int(oracle_failures),
        "table": table,
        "issues": issues,
        "recommendation": (
            "Do not enter Phase4/CPS. Resolve matched replay/root-cause blocker first."
            if has_fail else
            "Review WARNs; Phase3.6 may have downgraded Phase3.5 codec conclusion."
        ),
    }

    out_json = root / args.out_json
    out_md = root / args.out_md
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True))

    lines = [
        "# Phase3.6 Matched-State Action Replay Report",
        "",
        "## Verdict",
        "",
        f"- Verdict: `{verdict}`",
        f"- Root cause: `{root_cause}`",
        f"- Rows: `{len(rows)}`",
        f"- Prefix sources: `{prefix_sources}`",
        f"- state_match: `{state_match}`",
        f"- mean_prefix_state_mae: `{mean(state_maes):.6f}`",
        f"- GT success count: `{gt_success}`",
        f"- Oracle success count: `{oracle_success}`",
        f"- GT mean Δ final fraction: `{gt_delta:.6f}`",
        f"- Oracle mean Δ final fraction: `{oracle_delta:.6f}`",
        "",
        "## Summary Table",
        "",
        "| Test | Condition | Prefix source | Timeout | Rows | Success | Rate | Failures | Valid action | Δ final fraction | Final fraction | Prefix MAE | Prefix failure |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for t in table:
        lines.append(
            f"| `{t['test_type']}` | `{t['condition']}` | `{t['prefix_source']}` | {t['timeout']} | "
            f"{t['rows']} | {t['success_count']} | {t['success_rate']:.3f} | {t['failure_count']} | "
            f"{t['valid_action_rate']:.3f} | {t['mean_delta_final_fraction']:.3f} | "
            f"{t['mean_final_fraction_after']:.3f} | {t['mean_prefix_state_mae']:.3f} | "
            f"{t['prefix_failure_rate']:.3f} |"
        )

    lines += [
        "",
        "## Issues",
        "",
        "| Level | Name | Detail |",
        "|---|---|---|",
    ]
    if issues:
        for i in issues:
            lines.append(f"| `{i['level']}` | `{i['name']}` | {str(i['detail']).replace('|','/')} |")
    else:
        lines.append("| `PASS` | `none` | No matched replay issues found. |")

    lines += [
        "",
        "## Interpretation",
        "",
        "- If prefix state mismatch is high, Phase3.6 cannot confirm codec/adapter failure.",
        "- If prefix state matches and oracle progresses but matched GT does not, codec/adapter blocker is confirmed.",
        "- If matched GT progresses, Phase3.5 GT failure was likely an unmatched-reset artifact.",
        "- This is not Phase4 and not CPS evidence.",
    ]

    out_md.write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))

    if verdict == "FAIL":
        raise SystemExit("[Phase3.6][FAIL] matched replay diagnostic found blocking issue")

if __name__ == "__main__":
    main()
