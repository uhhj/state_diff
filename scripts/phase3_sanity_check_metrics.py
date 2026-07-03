#!/usr/bin/env python3
"""Sanity-check Phase3 PyTorch smoke/medium metrics.

Detects suspicious regularities before running medium/full:
- fallback backend leakage;
- tiny held-out size;
- identical paper_state/state_action behavior;
- condition-invariant future error;
- constant averaging score;
- abnormal inverse dynamics MSE/OOD;
- input leakage check failure.

This is a diagnostic gate, not a replacement for formal evaluation.
"""

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def to_float(x: Any) -> Optional[float]:
    try:
        if x is None or x == "" or str(x).lower() == "nan":
            return None
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def mean(xs):
    vals = [to_float(x) for x in xs]
    vals = [x for x in vals if x is not None]
    if not vals:
        return None
    return float(np.mean(vals))


def std(xs):
    vals = [to_float(x) for x in xs]
    vals = [x for x in vals if x is not None]
    if len(vals) == 0:
        return None
    return float(np.std(vals))


def fmt(x, digits=6):
    v = to_float(x)
    if v is None:
        return "NA"
    return f"{v:.{digits}f}"


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def group_rows(rows: List[Dict], *keys: str) -> Dict[tuple, List[Dict]]:
    out = defaultdict(list)
    for r in rows:
        out[tuple(r.get(k, "") for k in keys)].append(r)
    return dict(out)


def add_issue(issues: List[Dict], level: str, name: str, detail: str):
    issues.append({"level": level, "name": name, "detail": detail})


def metric_name_candidates(rows: List[Dict]) -> Dict[str, str]:
    if not rows:
        return {}

    keys = set(rows[0].keys())
    candidates = {
        "wrong_branch": [
            "sample_wrong_branch_rate",
            "wrong_branch_rate",
            "wrong_branch",
            "Wrong-Branch mean",
        ],
        "branch_accuracy": [
            "sample_branch_accuracy",
            "branch_accuracy",
            "Branch-Accuracy mean",
        ],
        "future_error": [
            "future_chamfer_to_true",
            "mean_chamfer_to_true",
            "future_error",
            "Future Error mean",
            "mean_future_error",
        ],
        "averaging_score": [
            "physical_averaging_score",
            "averaging_score",
            "Averaging Score mean",
        ],
        "action_mse": [
            "action_mse",
            "Action MSE mean",
            "inverse_action_mse",
        ],
        "action_ood": [
            "action_ood_score",
            "action_ood",
            "Action OOD mean",
            "inverse_action_ood",
        ],
        "backend": [
            "training_backend",
            "backend",
        ],
        "baseline": [
            "baseline",
            "Baseline",
        ],
        "condition": [
            "condition",
            "Condition",
        ],
        "fold": [
            "fold",
            "Fold",
        ],
        "seed": [
            "seed",
            "random_seed",
            "Seed",
        ],
    }

    resolved = {}
    for canonical, opts in candidates.items():
        for k in opts:
            if k in keys:
                resolved[canonical] = k
                break
    return resolved


def summarize_by_baseline_condition(rows: List[Dict], colmap: Dict[str, str]) -> List[Dict]:
    bkey = colmap.get("baseline", "baseline")
    ckey = colmap.get("condition", "condition")
    out = []
    groups = group_rows(rows, bkey, ckey)

    for (baseline, condition), rs in sorted(groups.items()):
        out.append({
            "baseline": baseline,
            "condition": condition,
            "count": len(rs),
            "wrong_branch_mean": mean([r.get(colmap.get("wrong_branch", "")) for r in rs]),
            "wrong_branch_std": std([r.get(colmap.get("wrong_branch", "")) for r in rs]),
            "branch_acc_mean": mean([r.get(colmap.get("branch_accuracy", "")) for r in rs]),
            "future_error_mean": mean([r.get(colmap.get("future_error", "")) for r in rs]),
            "future_error_std": std([r.get(colmap.get("future_error", "")) for r in rs]),
            "averaging_score_mean": mean([r.get(colmap.get("averaging_score", "")) for r in rs]),
            "averaging_score_std": std([r.get(colmap.get("averaging_score", "")) for r in rs]),
            "action_mse_mean": mean([r.get(colmap.get("action_mse", "")) for r in rs]),
            "action_ood_mean": mean([r.get(colmap.get("action_ood", "")) for r in rs]),
        })
    return out


def compare_baselines(summary_rows: List[Dict], issues: List[Dict]):
    by_cond = defaultdict(dict)
    for r in summary_rows:
        by_cond[r["condition"]][r["baseline"]] = r

    for cond, d in sorted(by_cond.items()):
        if "paper_state" not in d or "state_action" not in d:
            continue
        p = d["paper_state"]
        s = d["state_action"]

        for key in ["wrong_branch_mean", "future_error_mean", "averaging_score_mean", "action_mse_mean"]:
            pv = to_float(p.get(key))
            sv = to_float(s.get(key))
            if pv is None or sv is None:
                continue
            if abs(pv - sv) < 1e-8:
                add_issue(
                    issues,
                    "WARN",
                    f"baseline_identical_{key}_{cond}",
                    f"paper_state and state_action have identical {key} on {cond}: {pv}",
                )
            elif abs(pv - sv) < 1e-3:
                add_issue(
                    issues,
                    "WARN",
                    f"baseline_nearly_identical_{key}_{cond}",
                    f"paper_state and state_action differ by <1e-3 for {key} on {cond}: {pv} vs {sv}",
                )


def check_future_error_contrast(summary_rows: List[Dict], issues: List[Dict]):
    by_base = defaultdict(dict)
    for r in summary_rows:
        by_base[r["baseline"]][r["condition"]] = r

    for base, d in by_base.items():
        if "free" in d and "hidden_pin" in d:
            free_err = to_float(d["free"].get("future_error_mean"))
            pin_err = to_float(d["hidden_pin"].get("future_error_mean"))
            if free_err is not None and pin_err is not None:
                delta = pin_err - free_err
                if abs(delta) < 0.02:
                    add_issue(
                        issues,
                        "WARN",
                        f"low_future_error_contrast_{base}",
                        f"hidden_pin and free future errors are close: pin-free={delta:.6f}. "
                        "This may indicate mean prediction collapse or an overly coarse metric.",
                    )


def check_averaging_constant(rows: List[Dict], colmap: Dict[str, str], issues: List[Dict]):
    bkey = colmap.get("baseline", "baseline")
    akey = colmap.get("averaging_score")
    if not akey:
        return

    groups = group_rows(rows, bkey)
    for (baseline,), rs in groups.items():
        vals = [to_float(r.get(akey)) for r in rs]
        vals = [v for v in vals if v is not None]
        if len(vals) >= 10 and np.std(vals) < 1e-10:
            add_issue(
                issues,
                "WARN",
                f"constant_averaging_score_{baseline}",
                f"averaging_score is constant over {len(vals)} rows for {baseline}. "
                "Check whether aggregation reused a global value.",
            )


def check_action_metrics(summary_rows: List[Dict], issues: List[Dict], mse_warn: float, ood_warn: float):
    vals_mse = [to_float(r.get("action_mse_mean")) for r in summary_rows]
    vals_ood = [to_float(r.get("action_ood_mean")) for r in summary_rows]
    vals_mse = [v for v in vals_mse if v is not None]
    vals_ood = [v for v in vals_ood if v is not None]

    if vals_mse and max(vals_mse) > mse_warn:
        add_issue(
            issues,
            "FAIL",
            "action_mse_too_large",
            f"max action MSE={max(vals_mse):.6f} > {mse_warn}. "
            "Do not trust rollout until action codec / normalization / IDM target are diagnosed.",
        )

    if vals_ood and max(vals_ood) > ood_warn:
        add_issue(
            issues,
            "FAIL",
            "action_ood_too_large",
            f"max action OOD={max(vals_ood):.6f} > {ood_warn}. "
            "Predicted actions are far from expert action distribution.",
        )


def check_backend(rows: List[Dict], colmap: Dict[str, str], issues: List[Dict]):
    bkey = colmap.get("backend")
    if not bkey:
        add_issue(issues, "WARN", "backend_column_missing", "No training_backend column found.")
        return

    counts = Counter(r.get(bkey, "") for r in rows)
    if counts.get("numpy_fallback", 0) > 0:
        add_issue(
            issues,
            "FAIL",
            "numpy_fallback_present",
            f"Found numpy_fallback rows: {counts.get('numpy_fallback')}. Current report is not PyTorch evidence.",
        )
    if counts.get("torch", 0) == 0:
        add_issue(
            issues,
            "FAIL",
            "torch_backend_missing",
            "No torch backend rows found.",
        )


def check_leakage(leak: Dict, issues: List[Dict]):
    if not leak:
        add_issue(issues, "WARN", "leakage_report_missing", "No leakage JSON found.")
        return

    flat = {}

    def collect(prefix, obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                collect(prefix + [k], v)
        else:
            flat[".".join(prefix)] = obj

    collect([], leak)

    def find_metric(name):
        for k, v in flat.items():
            if k.endswith(name):
                return to_float(v)
        return None

    paper = find_metric("mean_pair_paper_x_max_abs_diff")
    sa = find_metric("mean_pair_state_action_x_max_abs_diff")
    probe = find_metric("free_vs_pin_probe_acc")

    if paper is not None and paper > 1e-3:
        add_issue(issues, "FAIL", "paper_x_pair_inconsistent", f"mean pair paper_x max diff={paper}")
    if sa is not None and sa > 1e-3:
        add_issue(issues, "FAIL", "state_action_pair_inconsistent", f"mean pair state_action_x max diff={sa}")
    if probe is not None and probe > 0.60:
        add_issue(issues, "FAIL", "free_vs_pin_probe_too_high", f"free-vs-pin probe acc={probe}")


def write_report(path: Path, payload: Dict):
    lines = []
    lines.append("# Phase3 Sanity Check Report")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append(f"- Verdict: `{payload['verdict']}`")
    lines.append(f"- Prediction rows: `{payload['num_prediction_rows']}`")
    lines.append("")
    lines.append("## Backend Counts")
    lines.append("")
    lines.append("| Backend | Count |")
    lines.append("|---|---:|")
    for k, v in payload["backend_counts"].items():
        lines.append(f"| `{k}` | {v} |")
    lines.append("")
    lines.append("## Metric Summary")
    lines.append("")
    lines.append("| Baseline | Condition | Count | Wrong-Branch | Future Error | Averaging Score | Action MSE | Action OOD |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for r in payload["summary_by_baseline_condition"]:
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                r["baseline"],
                r["condition"],
                r["count"],
                fmt(r["wrong_branch_mean"]),
                fmt(r["future_error_mean"]),
                fmt(r["averaging_score_mean"]),
                fmt(r["action_mse_mean"]),
                fmt(r["action_ood_mean"]),
            )
        )
    lines.append("")
    lines.append("## Issues")
    lines.append("")
    lines.append("| Level | Name | Detail |")
    lines.append("|---|---|---|")
    for issue in payload["issues"]:
        lines.append(f"| `{issue['level']}` | `{issue['name']}` | {issue['detail']} |")
    if not payload["issues"]:
        lines.append("| `PASS` | `none` | No suspicious metric pattern detected. |")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- `FAIL` means do not run policy rollout or Phase4 until fixed.")
    lines.append("- `WARN` means acceptable for smoke, but inspect before medium/full.")
    lines.append("- High wrong-branch on `hidden_pin` is expected; high action MSE/OOD is not expected.")
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--pred_csv", default="/data/state_diff2/reports/phase3_baseline_eval_predictions.csv")
    parser.add_argument("--leak_json", default="/data/state_diff2/reports/phase3_input_leakage_summary.json")
    parser.add_argument("--out_json", default="/data/state_diff2/reports/phase3_sanity_check_summary.json")
    parser.add_argument("--out_md", default="/data/state_diff2/reports/phase3_sanity_check_report.md")
    parser.add_argument("--action_mse_warn", type=float, default=10.0)
    parser.add_argument("--action_ood_warn", type=float, default=5.0)
    parser.add_argument("--min_rows_warn", type=int, default=300)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    rows = read_csv(Path(args.pred_csv))
    colmap = metric_name_candidates(rows)
    issues = []

    check_backend(rows, colmap, issues)
    check_leakage(read_json(Path(args.leak_json)), issues)

    if len(rows) < args.min_rows_warn:
        add_issue(
            issues,
            "WARN",
            "small_prediction_table",
            f"Only {len(rows)} prediction rows. This is smoke-scale, not paper-scale.",
        )

    summary_rows = summarize_by_baseline_condition(rows, colmap)
    compare_baselines(summary_rows, issues)
    check_future_error_contrast(summary_rows, issues)
    check_averaging_constant(rows, colmap, issues)
    check_action_metrics(summary_rows, issues, args.action_mse_warn, args.action_ood_warn)

    backend_counts = Counter()
    bkey = colmap.get("backend")
    if bkey:
        backend_counts.update(r.get(bkey, "") for r in rows)

    has_fail = any(i["level"] == "FAIL" for i in issues)
    has_warn = any(i["level"] == "WARN" for i in issues)
    verdict = "FAIL" if has_fail else ("WARN" if has_warn else "PASS")

    payload = {
        "verdict": verdict,
        "num_prediction_rows": len(rows),
        "column_map": colmap,
        "backend_counts": dict(backend_counts),
        "summary_by_baseline_condition": summary_rows,
        "issues": issues,
    }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(payload, indent=2, sort_keys=True))
    write_report(Path(args.out_md), payload)

    print(json.dumps(payload, indent=2, sort_keys=True))
    print("[Phase3] wrote", args.out_json)
    print("[Phase3] wrote", args.out_md)

    if args.strict and has_fail:
        raise SystemExit("[Phase3][FAIL] sanity check found fatal issues")


if __name__ == "__main__":
    main()