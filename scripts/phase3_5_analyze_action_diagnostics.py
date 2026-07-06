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
    ap.add_argument("--trials_csv", default="reports/phase3_5_action_execution_trials.csv")
    ap.add_argument("--out_json", default="reports/phase3_5_action_execution_diagnostic_summary.json")
    ap.add_argument("--out_md", default="reports/phase3_5_action_execution_diagnostic_report.md")
    args = ap.parse_args()

    root = Path(args.root)
    path = root / args.trials_csv
    issues = []

    def issue(level, name, detail):
        issues.append({"level": level, "name": name, "detail": str(detail)})

    if not path.exists():
        issue("FAIL", "missing_trials_csv", path)
        rows = []
    else:
        with path.open(newline="") as f:
            rows = list(csv.DictReader(f))

    if not rows:
        issue("FAIL", "no_rows", "no diagnostic rows")

    by = defaultdict(list)
    for r in rows:
        key = (
            r.get("control_source", ""),
            r.get("baseline", ""),
            r.get("condition", ""),
            r.get("motion_timeout", ""),
        )
        by[key].append(r)

    table = []
    for (src, baseline, condition, timeout), group in sorted(by.items()):
        successes = [parse_bool(r.get("success")) for r in group]
        failures = [r for r in group if r.get("failure_reason")]
        valid = [parse_bool(r.get("action_valid")) for r in group]
        deltas = [sf(r.get("delta_final_fraction")) for r in group]
        afters = [sf(r.get("final_fraction_after")) for r in group]
        oods = [sf(r.get("action_ood_score")) for r in group]
        clips = [sf(r.get("clip_fraction")) for r in group]
        item = {
            "control_source": src,
            "baseline": baseline,
            "condition": condition,
            "motion_timeout": timeout,
            "rows": len(group),
            "success_count": int(sum(successes)),
            "success_rate": float(sum(successes) / len(group)) if group else float("nan"),
            "failure_count": len(failures),
            "action_valid_rate": float(sum(valid) / len(group)) if group else float("nan"),
            "mean_delta_final_fraction": mean(deltas),
            "mean_final_fraction_after": mean(afters),
            "mean_action_ood": mean(oods),
            "mean_clip_fraction": mean(clips),
        }
        table.append(item)

    # Root-cause checks.
    def rows_for(src=None, baseline=None, condition=None, timeout=None):
        out = rows
        if src is not None:
            out = [r for r in out if r.get("control_source") == src]
        if baseline is not None:
            out = [r for r in out if r.get("baseline") == baseline]
        if condition is not None:
            out = [r for r in out if r.get("condition") == condition]
        if timeout is not None:
            out = [r for r in out if str(r.get("motion_timeout")) == str(timeout)]
        return out

    oracle_rows = rows_for(src="oracle_action")
    gt_rows = rows_for(src="gt_y_action_replay")
    learned_rows = rows_for(src="learned_action")

    oracle_success = sum(parse_bool(r.get("success")) for r in oracle_rows)
    gt_success = sum(parse_bool(r.get("success")) for r in gt_rows)
    learned_success = sum(parse_bool(r.get("success")) for r in learned_rows)

    oracle_delta = mean([sf(r.get("delta_final_fraction")) for r in oracle_rows])
    gt_delta = mean([sf(r.get("delta_final_fraction")) for r in gt_rows])
    learned_delta = mean([sf(r.get("delta_final_fraction")) for r in learned_rows])

    gt_failures = sum(1 for r in gt_rows if r.get("failure_reason"))
    learned_failures = sum(1 for r in learned_rows if r.get("failure_reason"))

    if oracle_rows and oracle_success == 0 and (not math.isfinite(oracle_delta) or oracle_delta <= 0):
        issue(
            "FAIL",
            "oracle_does_not_progress",
            "Oracle action does not produce success/progress in TensorFlow-free runtime. Environment path or success metric may be inconsistent.",
        )

    if gt_rows and gt_success == 0 and gt_failures > 0:
        issue(
            "FAIL",
            "gt_y_action_replay_has_runtime_failures",
            "Ground-truth y_action replay causes runtime failures. Suspect action codec or rollout adapter.",
        )

    if gt_rows and gt_success == 0 and gt_delta <= 0 and oracle_success > 0:
        issue(
            "FAIL",
            "gt_y_action_replay_no_progress_but_oracle_progresses",
            "Oracle progresses but decoded y_action replay does not. Suspect action codec / coordinate scale / action-template mismatch.",
        )

    if learned_rows and learned_failures > 0:
        issue(
            "FAIL",
            "learned_action_runtime_failures",
            f"Learned action rows have {learned_failures} runtime failures.",
        )

    if learned_rows and learned_success == 0 and oracle_success > 0 and (gt_success > 0 or gt_delta > 0):
        issue(
            "WARN",
            "learned_policy_quality_or_closed_loop_failure",
            "Oracle/GT path appears executable but learned actions still fail. Suspect IDM/future prediction/closed-loop alignment.",
        )

    # Motion timeout signal.
    by_timeout = defaultdict(list)
    for r in rows:
        by_timeout[str(r.get("motion_timeout"))].append(r)
    timeout_summary = {}
    for t, rs in by_timeout.items():
        timeout_summary[t] = {
            "rows": len(rs),
            "success_rate": float(sum(parse_bool(r.get("success")) for r in rs) / len(rs)) if rs else float("nan"),
            "failure_count": sum(1 for r in rs if r.get("failure_reason")),
            "mean_delta_final_fraction": mean([sf(r.get("delta_final_fraction")) for r in rs]),
        }

    if len(timeout_summary) >= 2:
        sorted_ts = sorted(timeout_summary, key=lambda x: float(x))
        lo, hi = sorted_ts[0], sorted_ts[-1]
        if timeout_summary[hi]["success_rate"] > timeout_summary[lo]["success_rate"]:
            issue(
                "WARN",
                "motion_timeout_affects_success",
                f"success_rate improves from timeout {lo} to {hi}; Phase3.4 may have been timeout-limited.",
            )

    conditions = sorted({r.get("condition", "") for r in rows})
    if not all(c in conditions for c in REQUIRED_CONDITIONS):
        issue("FAIL", "missing_required_conditions", {"got": conditions, "required": REQUIRED_CONDITIONS})

    has_fail = any(i["level"] == "FAIL" for i in issues)
    has_warn = any(i["level"] == "WARN" for i in issues)
    verdict = "FAIL" if has_fail else ("WARN" if has_warn else "PASS")

    root_cause = "undetermined"
    if any(i["name"] == "oracle_does_not_progress" for i in issues):
        root_cause = "environment_or_success_metric_blocker"
    elif any("gt_y_action" in i["name"] for i in issues):
        root_cause = "action_codec_or_rollout_adapter_blocker"
    elif any(i["name"] == "learned_policy_quality_or_closed_loop_failure" for i in issues):
        root_cause = "learned_policy_or_closed_loop_alignment_blocker"
    elif learned_success > 0:
        root_cause = "learned_action_can_succeed_at_smoke_scale"
    else:
        root_cause = "runtime_smoke_ok_but_no_success_signal"

    payload = {
        "verdict": verdict,
        "root_cause": root_cause,
        "num_rows": len(rows),
        "oracle_success_count": int(oracle_success),
        "gt_success_count": int(gt_success),
        "learned_success_count": int(learned_success),
        "oracle_mean_delta_final_fraction": oracle_delta,
        "gt_mean_delta_final_fraction": gt_delta,
        "learned_mean_delta_final_fraction": learned_delta,
        "timeout_summary": timeout_summary,
        "table": table,
        "issues": issues,
        "recommendation": (
            "Do not enter Phase4/CPS until FAIL items are resolved."
            if has_fail
            else "If only WARN, review learned policy/closed-loop quality before larger rollout or CPS."
        ),
    }

    (root / args.out_json).write_text(json.dumps(payload, indent=2, sort_keys=True))

    lines = [
        "# Phase3.5 Action Execution Diagnostic Report",
        "",
        "## Verdict",
        "",
        f"- Verdict: `{verdict}`",
        f"- Root cause: `{root_cause}`",
        f"- Rows: `{len(rows)}`",
        f"- Oracle success count: `{oracle_success}`",
        f"- GT y_action replay success count: `{gt_success}`",
        f"- Learned action success count: `{learned_success}`",
        "",
        "## Summary Table",
        "",
        "| Source | Baseline | Condition | Timeout | Rows | Success | Rate | Failures | Valid action rate | Δ final fraction | Final fraction | OOD | Clip fraction |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for t in table:
        lines.append(
            f"| `{t['control_source']}` | `{t['baseline']}` | `{t['condition']}` | {t['motion_timeout']} | "
            f"{t['rows']} | {t['success_count']} | {t['success_rate']:.3f} | {t['failure_count']} | "
            f"{t['action_valid_rate']:.3f} | {t['mean_delta_final_fraction']:.3f} | "
            f"{t['mean_final_fraction_after']:.3f} | {t['mean_action_ood']:.3f} | {t['mean_clip_fraction']:.3f} |"
        )

    lines += [
        "",
        "## Timeout Summary",
        "",
        "| Timeout | Rows | Success rate | Failure count | Mean Δ final fraction |",
        "|---:|---:|---:|---:|---:|",
    ]
    for t, s in timeout_summary.items():
        lines.append(f"| {t} | {s['rows']} | {s['success_rate']:.3f} | {s['failure_count']} | {s['mean_delta_final_fraction']:.3f} |")

    lines += [
        "",
        "## Issues",
        "",
        "| Level | Name | Detail |",
        "|---|---|---|",
    ]
    if issues:
        for i in issues:
            lines.append(f"| `{i['level']}` | `{i['name']}` | {str(i['detail']).replace('|', '/')} |")
    else:
        lines.append("| `PASS` | `none` | No diagnostic issues found. |")

    lines += [
        "",
        "## Interpretation",
        "",
        "- This is action execution diagnostic only.",
        "- It is not Phase4 and not CPS evidence.",
        "- Oracle/GT failures point to environment/adapter/action-codec issues.",
        "- Oracle/GT progress with learned failure points to learned policy or closed-loop alignment.",
    ]

    (root / args.out_md).write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))

    if verdict == "FAIL":
        raise SystemExit("[Phase3.5][FAIL] action diagnostic found blocking issue")

if __name__ == "__main__":
    main()
