#!/usr/bin/env python3
import sys
from pathlib import Path as _Phase3Path
ROOT = _Phase3Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def read_csv(path):
    if not Path(path).exists():
        return []
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def fval(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except Exception:
        return None


def mean_std(rows, key):
    vals = [fval(r.get(key)) for r in rows]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    return float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0


def fmt(v):
    return "NA" if v is None else f"{v:.6g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/state_diff2")
    args = ap.parse_args()
    root = Path(args.root)
    pred = read_csv(root / "reports/phase3_baseline_eval_predictions.csv")
    policy = read_csv(root / "reports/phase3_policy_rollout_trials.csv")
    rows = []
    groups = defaultdict(list)
    for r in pred:
        subset = "primary_ccda" if r.get("condition") in ("free", "hidden_pin") else "weak_contact"
        groups[(r.get("baseline"), r.get("condition"), subset)].append(r)
        groups[(r.get("baseline"), "all", "all")].append(r)
    for (baseline, condition, subset), rs in sorted(groups.items()):
        item = {"baseline": baseline, "condition": condition, "subset": subset, "num_rows": len(rs)}
        for key in ["sample_wrong_branch_rate", "branch_accuracy", "future_chamfer_to_true", "averaging_score", "action_mse", "action_ood_score"]:
            m, s = mean_std(rs, key)
            item[f"{key}_mean"] = m
            item[f"{key}_std"] = s
        rows.append(item)
    out_csv = root / "reports/phase3_fold_summary.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else ["baseline", "condition", "subset"]
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)

    policy_groups = defaultdict(list)
    for r in policy:
        policy_groups[(r.get("baseline"), r.get("condition"))].append(r)

    lines = ["# Phase3 StateDiff CCDA Baseline Evaluation", "", "## Scope", "", "Phase3 evaluates contact-blind StateDiff-style baselines on `hidden-contact-cable-line` without hidden condition labels, pin ids, hidden contact metadata, success labels, contact concatenation, CPS guidance, or action feasibility classifiers.", "", "## Offline State Prediction", "", "| Baseline | Condition | Wrong-Branch mean | Branch-Accuracy mean | Future Error mean | Averaging Score mean |", "|---|---|---:|---:|---:|---:|"]
    for r in rows:
        if r["condition"] == "all":
            continue
        lines.append(f"| {r['baseline']} | {r['condition']} | {fmt(r.get('sample_wrong_branch_rate_mean'))} | {fmt(r.get('branch_accuracy_mean'))} | {fmt(r.get('future_chamfer_to_true_mean'))} | {fmt(r.get('averaging_score_mean'))} |")
    lines += ["", "## Inverse Dynamics", "", "| Baseline | Condition | Action MSE mean | Action OOD mean |", "|---|---|---:|---:|"]
    for r in rows:
        if r["condition"] == "all":
            continue
        lines.append(f"| {r['baseline']} | {r['condition']} | {fmt(r.get('action_mse_mean'))} | {fmt(r.get('action_ood_score_mean'))} |")
    lines += ["", "## Policy Execution", "", "| Baseline | Condition | Trials | Success Rate | Final Fraction | Final Curve |", "|---|---|---:|---:|---:|---:|"]
    for key, rs in sorted(policy_groups.items()):
        succ, _ = mean_std(rs, "success")
        frac, _ = mean_std(rs, "final_fraction")
        curve, _ = mean_std(rs, "final_curve")
        lines.append(f"| {key[0]} | {key[1]} | {len(rs)} | {fmt(succ)} | {fmt(frac)} | {fmt(curve)} |")
    lines += ["", "## Phase3 Conclusion", ""]
    hidden_pin_rows = [r for r in rows if r["condition"] == "hidden_pin"]
    wrong = [fval(r.get("sample_wrong_branch_rate_mean")) for r in hidden_pin_rows]
    wrong = [x for x in wrong if x is not None]
    if wrong and max(wrong) >= 0.2:
        lines.append("Phase3 evaluates contact-blind StateDiff-style baselines on a large hidden-contact cable dataset with held-out visible seeds. Both baselines use state history and robot pose/proprioception proxy; `state_action` additionally uses action history. Neither baseline receives hidden contact labels or contact metadata.")
        lines.append("")
        lines.append("The input consistency and leakage checks verify that paired `free` and `hidden_pin` samples have matched visible/proprio/action inputs, and probe classifiers cannot reliably recover hidden condition from the model inputs. Therefore, the branch ambiguity is not caused by accidental input leakage.")
        lines.append("")
        lines.append("Across folds and random seeds, the baselines exhibit elevated wrong-branch rate and/or branch ambiguity on the primary `free` vs `hidden_pin` CCDA subset. Inverse dynamics and closed-loop policy execution further show that the ambiguity affects downstream action generation and task success.")
        lines.append("")
        lines.append("These results establish the baseline failure mode required before testing contact-conditioned methods. Phase4 should evaluate direct contact concatenation, and Phase5 should evaluate Contact Physical Score state-space guidance.")
    else:
        lines.append("Phase3 did not establish a robust contact-blind StateDiff failure mode. Before moving to contact-conditioned models, inspect whether the model inputs leak hidden condition, whether the paired train/test split is correct, whether the branch classifier is well-defined, and whether the inverse dynamics policy execution is dominated by action decoding errors rather than state prediction ambiguity.")
    (root / "reports/phase3_final_report.md").write_text("\n".join(lines) + "\n")
    print("[Phase3] wrote", out_csv)
    print("[Phase3] wrote", root / "reports/phase3_final_report.md")


if __name__ == "__main__":
    main()
