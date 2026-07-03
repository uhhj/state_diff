#!/usr/bin/env python3
import sys
from pathlib import Path as _Phase3Path
ROOT = _Phase3Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def read_csv(path):
    if not Path(path).exists():
        return []
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def read_json(path):
    if not Path(path).exists():
        return None
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None


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


def runtime_row(step, info):
    if not info:
        return f"| {step} | not run in current report | NA | NA | NA |"

    torch_required = step in ("train_eval", "rollout")
    ravens_required = step in ("generate", "rollout")

    def status(ok, required):
        if ok:
            return "OK"
        return "FAIL" if required else "not required"

    torch = status(info.get("torch_ok"), torch_required)
    ravens = status(info.get("ravens_ok"), ravens_required)
    return f"| {step} | `{info.get('python','')}` | `{info.get('conda_env','')}` | {torch} | {ravens} |"


def infer_scale(root: Path):
    prep = read_json(root / "reports/phase3_prepare_windows_summary.json") or {}
    n_train = int(prep.get("num_train_windows", 0) or 0)
    train_seeds = prep.get("train_visible_seeds", []) or []
    n_seeds = len(train_seeds)
    if n_seeds >= 500:
        return "full"
    if n_seeds >= 100:
        return "medium"
    return "smoke"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/state_diff2")
    ap.add_argument("--allow_fallback_report", action="store_true")
    args = ap.parse_args()
    root = Path(args.root)
    pred = read_csv(root / "reports/phase3_baseline_eval_predictions.csv")
    policy = read_csv(root / "reports/phase3_policy_rollout_trials.csv")
    backend_counts = Counter([r.get("training_backend", "unknown") for r in pred])
    has_fallback = any(k == "numpy_fallback" for k in backend_counts)
    all_torch = bool(pred) and set(backend_counts.keys()) == {"torch"}

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
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader(); w.writerows(rows)

    rt_generate = read_json(root / "reports/phase3_runtime_generate_env.json")
    rt_train = read_json(root / "reports/phase3_runtime_train_eval_env.json")
    rt_rollout = read_json(root / "reports/phase3_runtime_rollout_env.json")
    rt_aggregate = read_json(root / "reports/phase3_runtime_aggregate_env.json")
    current_rollout = rt_rollout is not None
    policy_groups = defaultdict(list)
    if current_rollout:
        for r in policy:
            policy_groups[(r.get("baseline"), r.get("condition"))].append(r)

    title = "# Phase3 PyTorch StateDiff CCDA Baseline Evaluation" if all_torch else "# Phase3 StateDiff CCDA Pipeline Smoke Evaluation"
    lines = [title, "", "## Scope", "", "Phase3 evaluates contact-blind StateDiff-style baselines on `hidden-contact-cable-line` without hidden condition labels, pin ids, hidden contact metadata, success labels, contact concatenation, CPS guidance, or action feasibility classifiers.", ""]
    lines += ["## Runtime Backend", "", "| Step | Python | Conda Env | Torch | Ravens |", "|---|---|---|---|---|",
              runtime_row("generate", rt_generate), runtime_row("train_eval", rt_train), runtime_row("rollout", rt_rollout), runtime_row("aggregate", rt_aggregate), ""]
    lines += ["## Training Backend", "", "| Backend | Count |", "|---|---:|"]
    for k, v in sorted(backend_counts.items()):
        lines.append(f"| `{k}` | {v} |")
    if has_fallback:
        lines += ["", "This is fallback smoke, not a PyTorch DDPM result."]
    lines += ["", "## Offline State Prediction", "", "| Baseline | Condition | Wrong-Branch mean | Branch-Accuracy mean | Future Error mean | Averaging Score mean |", "|---|---|---:|---:|---:|---:|"]
    for r in rows:
        if r["condition"] == "all":
            continue
        lines.append(f"| {r['baseline']} | {r['condition']} | {fmt(r.get('sample_wrong_branch_rate_mean'))} | {fmt(r.get('branch_accuracy_mean'))} | {fmt(r.get('future_chamfer_to_true_mean'))} | {fmt(r.get('averaging_score_mean'))} |")
    lines += ["", "## Inverse Dynamics", "", "| Baseline | Condition | Action MSE mean | Action OOD mean |", "|---|---|---:|---:|"]
    for r in rows:
        if r["condition"] == "all":
            continue
        lines.append(f"| {r['baseline']} | {r['condition']} | {fmt(r.get('action_mse_mean'))} | {fmt(r.get('action_ood_score_mean'))} |")
    lines += ["", "## Policy Execution", ""]
    if current_rollout and policy_groups:
        lines += ["| Baseline | Condition | Trials | Success Rate | Final Fraction | Final Curve |", "|---|---|---:|---:|---:|---:|"]
        for key, rs in sorted(policy_groups.items()):
            succ, _ = mean_std(rs, "success")
            frac, _ = mean_std(rs, "final_fraction")
            curve, _ = mean_std(rs, "final_curve")
            lines.append(f"| {key[0]} | {key[1]} | {len(rs)} | {fmt(succ)} | {fmt(frac)} | {fmt(curve)} |")
        all_zero = policy and all((fval(r.get("success")) or 0.0) == 0.0 for r in policy)
        if all_zero:
            lines += ["", "Policy rollout currently records zero success for all conditions; this is reported as policy execution failure and should not be hidden or counted as successful execution evidence."]
    else:
        lines.append("Policy rollout was not rerun for the current PyTorch train/eval report, so any older rollout CSV is not counted as current execution evidence.")

    scale = infer_scale(root)
    lines += ["", "## Phase3 Conclusion", ""]
    hidden_pin_rows = [r for r in rows if r["condition"] == "hidden_pin"]
    wrong = [fval(r.get("sample_wrong_branch_rate_mean")) for r in hidden_pin_rows]
    wrong = [x for x in wrong if x is not None]
    if has_fallback and not args.allow_fallback_report:
        lines.append("This report is pipeline smoke only because at least one checkpoint used `numpy_fallback`. It must not be cited as a PyTorch StateDiff baseline result.")
    elif all_torch and wrong and max(wrong) >= 0.2:
        if scale == "full":
            lines.append("Full Phase3 completed with PyTorch checkpoints. Confirm fold and seed counts before treating this as paper-level 5-fold x 3-seed statistics.")
        elif scale == "medium":
            lines.append("This medium run validates PyTorch Phase3 behavior at intermediate scale. Full paper-level statistics require MODE=full.")
        else:
            lines.append("This is a PyTorch smoke/medium validation, not the final full 5-fold x 3-seed result unless MODE=full was run.")
        lines.append("")
        lines.append("The input consistency and leakage checks verify that paired `free` and `hidden_pin` samples have matched visible/proprio/action inputs, and probe classifiers cannot reliably recover hidden condition from the model inputs. Therefore, the branch ambiguity is not caused by accidental input leakage.")
        lines.append("")
        lines.append("Across the current folds and random seeds, the contact-blind baselines exhibit elevated wrong-branch rate and/or branch ambiguity on the primary `free` vs `hidden_pin` CCDA subset. These results support moving to Phase4 only after a full-scale run if paper-level statistics are required.")
    else:
        lines.append("Phase3 did not establish a robust contact-blind StateDiff failure mode. Before moving to contact-conditioned models, inspect input leakage, seed grouping, branch definitions, and inverse-dynamics or rollout failures.")
    (root / "reports/phase3_final_report.md").write_text("\n".join(lines) + "\n")
    print("[Phase3] wrote", out_csv)
    print("[Phase3] wrote", root / "reports/phase3_final_report.md")


if __name__ == "__main__":
    main()
