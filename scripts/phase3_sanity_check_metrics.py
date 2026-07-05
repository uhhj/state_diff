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


DDPM_MODEL_TYPE = "torch_conditional_ddpm_future_state"
BRANCH_REFERENCE_MODE = "split_visible_seed_window_t"
SCHEDULER_TYPE = "diffusers.DDPMScheduler"
BETA_SCHEDULE = "squaredcos_cap_v2"
PREDICTION_TYPE = "epsilon"
VARIANCE_TYPE = "fixed_small"
DENOISER_ARCH = "mlp"
PAPER_ALIGNMENT_LEVEL = "ddpm_scheduler_aligned_mlp_denoiser"


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


def check_future_error_contrast(summary_rows: List[Dict], issues: List[Dict], primary_hidden_condition: str):
    by_base = defaultdict(dict)
    for r in summary_rows:
        by_base[r["baseline"]][r["condition"]] = r

    for base, d in by_base.items():
        if "free" in d and primary_hidden_condition in d:
            free_err = to_float(d["free"].get("future_error_mean"))
            hidden_err = to_float(d[primary_hidden_condition].get("future_error_mean"))
            if free_err is not None and hidden_err is not None:
                delta = hidden_err - free_err
                if abs(delta) < 0.02:
                    add_issue(
                        issues,
                        "WARN",
                        f"low_future_error_contrast_{base}",
                        f"{primary_hidden_condition} and free future errors are close: hidden-free={delta:.6f}. "
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
            "WARN",
            "cascade_action_mse_large",
            f"max cascaded action MSE={max(vals_mse):.6f} > {mse_warn}. "
            "This uses predicted future states; pure IDM health is checked in phase3_action_idm_debug.",
        )

    if vals_ood and max(vals_ood) > ood_warn:
        add_issue(
            issues,
            "WARN",
            "cascade_action_ood_large",
            f"max cascaded action OOD={max(vals_ood):.6f} > {ood_warn}. "
            "This uses predicted future states; pure IDM health is checked in phase3_action_idm_debug.",
        )


def check_backend(rows: List[Dict], colmap: Dict[str, str], issues: List[Dict]):
    bkey = colmap.get("backend")
    if not bkey:
        add_issue(issues, "FAIL", "backend_column_missing", "No training_backend column found.")
        return

    counts = Counter(r.get(bkey, "") for r in rows)
    if counts.get("numpy_fallback", 0) > 0:
        add_issue(
            issues,
            "FAIL",
            "numpy_fallback_present",
            f"Found numpy_fallback rows: {counts.get('numpy_fallback')}. Current report is not PyTorch evidence.",
        )
    non_torch = {k: v for k, v in counts.items() if k != "torch"}
    if non_torch:
        add_issue(
            issues,
            "FAIL",
            "non_torch_backend_present",
            f"All Phase3 DDPM rows must use torch backend; found {dict(non_torch)}.",
        )
    if counts.get("torch", 0) == 0:
        add_issue(
            issues,
            "FAIL",
            "torch_backend_missing",
            "No torch backend rows found.",
        )


def check_ddpm_metadata(rows: List[Dict], issues: List[Dict]) -> Dict[str, Dict[str, int]]:
    if not rows:
        add_issue(issues, "FAIL", "prediction_rows_missing", "No prediction rows found.")
        return {}

    expected_columns = [
        "future_model_type",
        "ddpm_used",
        "branch_reference_mode",
        "scheduler_type",
        "beta_schedule",
        "prediction_type",
        "variance_type",
        "denoiser_arch",
        "paper_alignment_level",
    ]
    for col in expected_columns:
        if col not in rows[0]:
            add_issue(issues, "FAIL", f"{col}_missing", f"Prediction CSV lacks required `{col}` metadata column.")

    counts = {
        "future_model_type_counts": Counter(r.get("future_model_type", "") for r in rows),
        "ddpm_used_counts": Counter(str(r.get("ddpm_used", "")).lower() for r in rows),
        "branch_reference_mode_counts": Counter(r.get("branch_reference_mode", "") for r in rows),
        "scheduler_type_counts": Counter(r.get("scheduler_type", "") for r in rows),
        "beta_schedule_counts": Counter(r.get("beta_schedule", "") for r in rows),
        "prediction_type_counts": Counter(r.get("prediction_type", "") for r in rows),
        "variance_type_counts": Counter(r.get("variance_type", "") for r in rows),
        "denoiser_arch_counts": Counter(r.get("denoiser_arch", "") for r in rows),
        "conditional_unet1d_used_counts": Counter(str(r.get("conditional_unet1d_used", "")).lower() for r in rows),
        "paper_alignment_level_counts": Counter(r.get("paper_alignment_level", "") for r in rows),
        "eval_sample_seed_mode_counts": Counter(r.get("eval_sample_seed_mode", "") for r in rows),
    }

    future_counts = counts["future_model_type_counts"]
    bad_future = {k: v for k, v in future_counts.items() if k != DDPM_MODEL_TYPE}
    if bad_future or future_counts.get(DDPM_MODEL_TYPE, 0) != len(rows):
        add_issue(
            issues,
            "FAIL",
            "non_ddpm_future_model_present",
            f"Expected only {DDPM_MODEL_TYPE}; counts={dict(future_counts)}.",
        )

    ddpm_counts = counts["ddpm_used_counts"]
    bad_ddpm = {k: v for k, v in ddpm_counts.items() if k not in {"true", "1", "yes"}}
    if bad_ddpm or sum(v for k, v in ddpm_counts.items() if k in {"true", "1", "yes"}) != len(rows):
        add_issue(
            issues,
            "FAIL",
            "ddpm_used_not_true",
            f"All rows must report ddpm_used=true; counts={dict(ddpm_counts)}.",
        )

    ref_counts = counts["branch_reference_mode_counts"]
    bad_ref = {k: v for k, v in ref_counts.items() if k != BRANCH_REFERENCE_MODE}
    if counts["scheduler_type_counts"].get(SCHEDULER_TYPE, 0) != len(rows):
        add_issue(issues, "FAIL", "scheduler_type_wrong", f"Expected only {SCHEDULER_TYPE}; counts={dict(counts['scheduler_type_counts'])}.")
    if counts["beta_schedule_counts"].get(BETA_SCHEDULE, 0) != len(rows):
        add_issue(issues, "FAIL", "beta_schedule_wrong", f"Expected only {BETA_SCHEDULE}; counts={dict(counts['beta_schedule_counts'])}.")
    if counts["prediction_type_counts"].get(PREDICTION_TYPE, 0) != len(rows):
        add_issue(issues, "FAIL", "prediction_type_wrong", f"Expected only {PREDICTION_TYPE}; counts={dict(counts['prediction_type_counts'])}.")
    if counts["variance_type_counts"].get(VARIANCE_TYPE, 0) != len(rows):
        add_issue(issues, "WARN", "variance_type_not_fixed_small", f"Expected pre-medium variance_type={VARIANCE_TYPE}; counts={dict(counts['variance_type_counts'])}.")
    if counts["denoiser_arch_counts"].get(DENOISER_ARCH, 0) != len(rows):
        add_issue(issues, "WARN", "denoiser_arch_not_mlp", f"Expected denoiser_arch={DENOISER_ARCH}; counts={dict(counts['denoiser_arch_counts'])}.")
    if counts["paper_alignment_level_counts"].get(PAPER_ALIGNMENT_LEVEL, 0) != len(rows):
        add_issue(issues, "WARN", "paper_alignment_level_unexpected", f"Expected {PAPER_ALIGNMENT_LEVEL}; counts={dict(counts['paper_alignment_level_counts'])}.")

    return {k: dict(v) for k, v in counts.items()}


def check_eval_summary(eval_summary: Dict, issues: List[Dict]):
    if not eval_summary:
        add_issue(issues, "WARN", "eval_summary_missing", "No evaluation summary JSON found.")
        return

    missing_refs = int(eval_summary.get("num_missing_primary_branch_refs", 0) or 0)
    if missing_refs:
        add_issue(
            issues,
            "FAIL",
            "missing_primary_branch_refs",
            f"Evaluation had {missing_refs} rows without the required split+visible_seed+window_t branch reference.",
        )

    if eval_summary.get("all_ddpm_used") is False:
        add_issue(issues, "FAIL", "eval_summary_ddpm_false", "Evaluation summary reports all_ddpm_used=false.")

    seen = eval_summary.get("future_model_types_seen") or []
    if seen and set(seen) != {DDPM_MODEL_TYPE}:
        add_issue(
            issues,
            "FAIL",
            "eval_summary_non_ddpm_future_model",
            f"Evaluation summary future_model_types_seen={seen}; expected only {DDPM_MODEL_TYPE}.",
        )

    mode = eval_summary.get("branch_reference_mode")
    if mode and mode != BRANCH_REFERENCE_MODE:
        add_issue(
            issues,
            "FAIL",
            "eval_summary_branch_reference_mode_wrong",
            f"Evaluation summary branch_reference_mode={mode}; expected {BRANCH_REFERENCE_MODE}.",
        )

def check_leakage(leak: Dict, issues: List[Dict], primary_hidden_condition: str, diagnostic_hidden_condition: str):
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
    primary_probe = leak.get("free_vs_primary_probe_acc")
    diagnostic_probe = leak.get("free_vs_diagnostic_probe_acc", leak.get("free_vs_pin_probe_acc"))

    if paper is not None and paper > 1e-8:
        add_issue(issues, "FAIL", "paper_x_pair_inconsistent", f"mean pair paper_x max diff={paper}")
    if sa is not None and sa > 1e-8:
        add_issue(issues, "FAIL", "state_action_pair_inconsistent", f"mean pair state_action_x max diff={sa}")
    if primary_probe is not None and to_float(primary_probe) is not None and to_float(primary_probe) > 0.60:
        add_issue(issues, "FAIL", "free_vs_primary_probe_too_high", f"free-vs-{primary_hidden_condition} probe acc={primary_probe}")
    if diagnostic_probe is not None and to_float(diagnostic_probe) is not None and to_float(diagnostic_probe) > 0.60:
        add_issue(issues, "FAIL", "free_vs_diagnostic_probe_too_high", f"free-vs-{diagnostic_hidden_condition} probe acc={diagnostic_probe}")


def write_report(path: Path, payload: Dict):
    lines = []
    lines.append("# Phase3 Sanity Check Report")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append(f"- Verdict: `{payload['verdict']}`")
    lines.append(f"- Prediction rows: `{payload['num_prediction_rows']}`")
    lines.append(f"- Primary hidden condition: `{payload.get('primary_hidden_condition')}`")
    lines.append(f"- Diagnostic hidden condition: `{payload.get('diagnostic_hidden_condition')}`")
    lines.append("")
    lines.append("## Backend Counts")
    lines.append("")
    lines.append("| Backend | Count |")
    lines.append("|---|---:|")
    for k, v in payload["backend_counts"].items():
        lines.append(f"| `{k}` | {v} |")
    lines.append("")
    lines.append("## Scheduler Metadata Counts")
    lines.append("")
    for name, counts in payload.get("ddpm_metadata_counts", {}).items():
        lines.append(f"### {name}")
        lines.append("")
        lines.append("| Value | Count |")
        lines.append("|---|---:|")
        for k, v in counts.items():
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
    lines.append(f"- High wrong-branch on `{payload.get('primary_hidden_condition')}` is the primary CCDA signal. `hidden_pin` remains a hard diagnostic branch.")
    lines.append("- Cascaded action MSE/OOD is a warning; pure IDM health is checked separately.")
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--pred_csv", default="/data/state_diff2/reports/phase3_baseline_eval_predictions.csv")
    parser.add_argument("--leak_json", default="/data/state_diff2/reports/phase3_input_leakage_summary.json")
    parser.add_argument("--eval_json", default="/data/state_diff2/reports/phase3_baseline_eval_summary.json")
    parser.add_argument("--out_json", default="/data/state_diff2/reports/phase3_sanity_check_summary.json")
    parser.add_argument("--out_md", default="/data/state_diff2/reports/phase3_sanity_check_report.md")
    parser.add_argument("--action_mse_warn", type=float, default=10.0)
    parser.add_argument("--action_ood_warn", type=float, default=5.0)
    parser.add_argument("--min_rows_warn", type=int, default=300)
    parser.add_argument("--conditions", nargs="+", default=None)
    parser.add_argument("--primary_hidden_condition", default="hidden_breakaway_pin")
    parser.add_argument("--diagnostic_hidden_condition", default="hidden_pin")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    rows = read_csv(Path(args.pred_csv))
    colmap = metric_name_candidates(rows)
    issues = []

    check_backend(rows, colmap, issues)
    ddpm_metadata_counts = check_ddpm_metadata(rows, issues)
    check_eval_summary(read_json(Path(args.eval_json)), issues)
    check_leakage(read_json(Path(args.leak_json)), issues, args.primary_hidden_condition, args.diagnostic_hidden_condition)

    if len(rows) < args.min_rows_warn:
        add_issue(
            issues,
            "WARN",
            "small_prediction_table",
            f"Only {len(rows)} prediction rows. This is smoke-scale, not paper-scale.",
        )

    summary_rows = summarize_by_baseline_condition(rows, colmap)
    compare_baselines(summary_rows, issues)
    check_future_error_contrast(summary_rows, issues, args.primary_hidden_condition)
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
        "conditions": args.conditions or [],
        "primary_hidden_condition": args.primary_hidden_condition,
        "diagnostic_hidden_condition": args.diagnostic_hidden_condition,
        "ddpm_metadata_counts": ddpm_metadata_counts,
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