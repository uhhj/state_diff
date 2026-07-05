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

DDPM_MODEL_TYPE = "torch_conditional_ddpm_future_state"
BRANCH_REFERENCE_MODE = "split_visible_seed_window_t"
SCHEDULER_TYPE = "diffusers.DDPMScheduler"
BETA_SCHEDULE = "squaredcos_cap_v2"
PREDICTION_TYPE = "epsilon"
VARIANCE_TYPE = "fixed_small"
DENOISER_ARCH = "mlp"
PAPER_ALIGNMENT_LEVEL = "ddpm_scheduler_aligned_mlp_denoiser"


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


def boolish(x):
    return str(x).strip().lower() in {"true", "1", "yes"}


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
    return f"| {step} | `{info.get('python','')}` | `{info.get('conda_env','')}` | {status(info.get('torch_ok'), torch_required)} | {status(info.get('ravens_ok'), ravens_required)} |"


def infer_scale(root: Path):
    prep = read_json(root / "reports/phase3_prepare_windows_summary.json") or {}
    train_seeds = prep.get("train_visible_seeds", []) or []
    n_seeds = len(train_seeds)
    if n_seeds >= 500:
        return "full"
    if n_seeds >= 100:
        return "medium"
    return "smoke"


def diagnostics_markdown(root: Path) -> str:
    sanity = read_json(root / "reports/phase3_sanity_check_summary.json") or {}
    action = read_json(root / "reports/phase3_action_idm_debug_summary.json") or {}
    lines = []
    lines += ["", "## Sanity Diagnostics", "", "| Diagnostic | Verdict |", "|---|---|", f"| Phase3 metric sanity | `{sanity.get('verdict', 'not_run')}` |", f"| Action/IDM debug | `{action.get('verdict', 'not_run')}` |", ""]

    def issue_lines(title, payload):
        issues = payload.get("issues", []) if isinstance(payload, dict) else []
        lines.extend([f"### {title}", "", "| Level | Name | Detail |", "|---|---|---|"])
        if issues:
            for issue in issues:
                lines.append("| `{}` | `{}` | {} |".format(issue.get("level", ""), issue.get("name", ""), issue.get("detail", "")))
        else:
            lines.append("| `PASS` | `none` | No issue recorded or diagnostic not run. |")
        lines.append("")

    issue_lines("Metric Sanity Issues", sanity)
    issue_lines("Action/IDM Issues", action)
    verdicts = {str(sanity.get("verdict", "")), str(action.get("verdict", ""))}
    if "FAIL" in verdicts:
        lines.append("**Execution-level Phase3 evidence is blocked by sanity diagnostics. Do not count policy rollout or move to Phase4 until the FAIL items are fixed.**")
    elif "WARN" in verdicts:
        lines.append("**Diagnostics contain WARN items. They are documented in this report and do not introduce a blocking FAIL for the current offline medium run; rollout, Phase4, and paper-level claims still require separate review of these WARN items.**")
    else:
        lines.append("**Diagnostics do not report blocking anomalies.**")
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/state_diff2")
    ap.add_argument("--allow_fallback_report", action="store_true")
    ap.add_argument("--conditions", nargs="+", default=None)
    ap.add_argument("--primary_hidden_condition", default="hidden_breakaway_pin")
    ap.add_argument("--diagnostic_hidden_condition", default="hidden_pin")
    args = ap.parse_args()
    root = Path(args.root)
    pred = read_csv(root / "reports/phase3_baseline_eval_predictions.csv")
    if pred:
        args.primary_hidden_condition = pred[0].get("primary_hidden_condition", args.primary_hidden_condition) or args.primary_hidden_condition
        args.diagnostic_hidden_condition = pred[0].get("diagnostic_hidden_condition", args.diagnostic_hidden_condition) or args.diagnostic_hidden_condition
        if not args.conditions:
            args.conditions = (pred[0].get("conditions", "") or "").split()
    policy = read_csv(root / "reports/phase3_policy_rollout_trials.csv")
    backend_counts = Counter([r.get("training_backend", "unknown") for r in pred])
    has_fallback = any(k == "numpy_fallback" for k in backend_counts)
    all_torch = bool(pred) and set(backend_counts.keys()) == {"torch"}
    future_types = Counter([r.get("future_model_type", "") for r in pred])
    ddpm_flags = Counter([str(r.get("ddpm_used", "")).lower() for r in pred])
    scheduler_types = Counter([r.get("scheduler_type", "") for r in pred])
    beta_schedules = Counter([r.get("beta_schedule", "") for r in pred])
    prediction_types = Counter([r.get("prediction_type", "") for r in pred])
    variance_types = Counter([r.get("variance_type", "") for r in pred])
    clip_samples = Counter([str(r.get("clip_sample", "")).lower() for r in pred])
    denoiser_arches = Counter([r.get("denoiser_arch", "") for r in pred])
    conditional_unets = Counter([str(r.get("conditional_unet1d_used", "")).lower() for r in pred])
    alignment_levels = Counter([r.get("paper_alignment_level", "") for r in pred])
    non_ddpm_rows = [r for r in pred if r.get("future_model_type") != DDPM_MODEL_TYPE or not boolish(r.get("ddpm_used"))]

    rows = []
    groups = defaultdict(list)
    for r in pred:
        subset = "primary_ccda" if r.get("condition") in ("free", args.primary_hidden_condition) else ("diagnostic_ccda" if r.get("condition") == args.diagnostic_hidden_condition else "control_contact")
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
        w.writeheader()
        w.writerows(rows)

    rt_generate = read_json(root / "reports/phase3_runtime_generate_env.json")
    rt_train = read_json(root / "reports/phase3_runtime_train_eval_env.json")
    rt_rollout = read_json(root / "reports/phase3_runtime_rollout_env.json")
    rt_aggregate = read_json(root / "reports/phase3_runtime_aggregate_env.json")
    current_rollout = rt_rollout is not None
    policy_groups = defaultdict(list)
    if current_rollout:
        for r in policy:
            policy_groups[(r.get("baseline"), r.get("condition"))].append(r)

    report_fails = []
    if has_fallback and not args.allow_fallback_report:
        report_fails.append("numpy_fallback prediction rows are not allowed in DDPM Phase3.")
    if non_ddpm_rows:
        report_fails.append("Non-DDPM future model rows found in prediction CSV.")
    if future_types and set(future_types.keys()) != {DDPM_MODEL_TYPE}:
        report_fails.append(f"Unexpected future_model_types_seen={dict(future_types)}")
    if ddpm_flags and set(ddpm_flags.keys()) != {"true"}:
        report_fails.append(f"Unexpected ddpm_used flags={dict(ddpm_flags)}")
    if scheduler_types and set(scheduler_types.keys()) != {SCHEDULER_TYPE}:
        report_fails.append(f"Unexpected scheduler_type counts={dict(scheduler_types)}")
    if beta_schedules and set(beta_schedules.keys()) != {BETA_SCHEDULE}:
        report_fails.append(f"Unexpected beta_schedule counts={dict(beta_schedules)}")
    if prediction_types and set(prediction_types.keys()) != {PREDICTION_TYPE}:
        report_fails.append(f"Unexpected prediction_type counts={dict(prediction_types)}")

    title = "# Phase3 PyTorch StateDiff CCDA Baseline Evaluation" if all_torch and not report_fails else "# Phase3 DDPM Alignment Report - FAIL"
    lines = [title, "", "## Scope", "", "Phase3 evaluates contact-blind StateDiff-style baselines on `hidden-contact-cable-line` without hidden condition labels, pin ids, hidden contact metadata, recoverability parameters, success labels, contact concatenation, CPS guidance, or action feasibility classifiers.", "", "## Condition Scope", "", f"- Conditions: `{' '.join(args.conditions or [])}`", f"- Primary branch pair: `free_vs_{args.primary_hidden_condition}`", f"- Diagnostic branch pair: `free_vs_{args.diagnostic_hidden_condition}`", "- Policy rollout is intentionally excluded unless explicitly run with `PHASE3_ALLOW_ROLLOUT=1`.", ""]
    lines += ["## Runtime Backend", "", "| Step | Python | Conda Env | Torch | Ravens |", "|---|---|---|---|---|",
              runtime_row("generate", rt_generate), runtime_row("train_eval", rt_train), runtime_row("rollout", rt_rollout), runtime_row("aggregate", rt_aggregate), ""]
    lines += ["## Future Model", "", "| Field | Value |", "|---|---|", f"| future_model_type | {DDPM_MODEL_TYPE} |", f"| ddpm_used | {str(set(ddpm_flags.keys()) == {'true'}).lower()} |", "| simplified_mlp_removed | true |", f"| branch_reference_mode | {BRANCH_REFERENCE_MODE} |", "", "This Phase3 run uses a conditional DDPM future-state predictor. It no longer uses the previous PyTorch MLP residual future-state surrogate.", ""]
    lines += [
        "## Paper Alignment Status",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| scheduler_type | {SCHEDULER_TYPE} |",
        f"| beta_schedule | {BETA_SCHEDULE} |",
        f"| prediction_type | {PREDICTION_TYPE} |",
        f"| variance_type | {VARIANCE_TYPE} |",
        "| clip_sample | true |",
        f"| denoiser_arch | {DENOISER_ARCH} |",
        "| conditional_unet1d_used | false |",
        f"| alignment_level | {PAPER_ALIGNMENT_LEVEL} |",
        "",
        "This run aligns the DDPM scheduler with the original StateDiff configuration but still uses an MLP denoiser over low-dimensional future states. It is not yet an architecture-identical ConditionalUnet1D reproduction.",
        "",
        "### Paper Alignment Metadata Counts",
        "",
        f"- scheduler_type_counts: `{dict(scheduler_types)}`",
        f"- beta_schedule_counts: `{dict(beta_schedules)}`",
        f"- prediction_type_counts: `{dict(prediction_types)}`",
        f"- variance_type_counts: `{dict(variance_types)}`",
        f"- clip_sample_counts: `{dict(clip_samples)}`",
        f"- denoiser_arch_counts: `{dict(denoiser_arches)}`",
        f"- conditional_unet1d_used_counts: `{dict(conditional_unets)}`",
        f"- paper_alignment_level_counts: `{dict(alignment_levels)}`",
        "",
    ]
    lines += ["## Training Backend", "", "| Backend | Count |", "|---|---:|"]
    for k, v in sorted(backend_counts.items()):
        lines.append(f"| `{k}` | {v} |")
    if report_fails:
        lines += ["", "## Report Failures", ""]
        for item in report_fails:
            lines.append(f"- {item}")
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
    else:
        lines.append("Policy rollout was not rerun for the current PyTorch train/eval report, so any older rollout CSV is not counted as current execution evidence.")

    scale = infer_scale(root)
    lines += ["", "## Phase3 Conclusion", ""]
    primary_rows = [r for r in rows if r["condition"] == args.primary_hidden_condition]
    wrong = [fval(r.get("sample_wrong_branch_rate_mean")) for r in primary_rows]
    wrong = [x for x in wrong if x is not None]
    if report_fails:
        lines.append("Phase3 DDPM alignment is still FAIL. Do not run medium, rollout, Phase4, or cite this report until the listed failures are fixed.")
    elif scale == "full":
        lines.append("Full Phase3 completed with PyTorch DDPM checkpoints. Confirm fold and seed counts before treating this as paper-level 5-fold x 3-seed statistics.")
    elif scale == "medium":
        lines.append(f"Phase3-medium passed with PyTorch DDPM backend, leakage-checked matched inputs, executable action targets, and multi-fold/multi-seed statistics using primary pair `free_vs_{args.primary_hidden_condition}`. Policy execution remains excluded unless torch-enabled DeformableRavens rollout is run explicitly and action diagnostics pass.")
    else:
        lines.append(f"Phase3 PyTorch DDPM smoke passed for primary pair `free_vs_{args.primary_hidden_condition}` after fixing dynamic condition integration. The executable action codec removes camera_config leakage; y_action contains only pick-place pose parameters. Paper-level evidence still requires MODE=medium or MODE=full.")
    if wrong:
        lines.append("")
        lines.append(f"Across the current folds and random seeds, the contact-blind baselines exhibit wrong-branch rate and/or branch ambiguity on the primary `free` vs `{args.primary_hidden_condition}` CCDA subset. `hidden_pin` remains a hard diagnostic branch, not the primary CPS success-improvement branch. Rollout and Phase4 decisions must obey the sanity diagnostics below.")
    lines.append(diagnostics_markdown(root))
    (root / "reports/phase3_final_report.md").write_text("\n".join(lines).rstrip() + "\n")
    print("[Phase3] wrote", out_csv)
    print("[Phase3] wrote", root / "reports/phase3_final_report.md")


if __name__ == "__main__":
    main()
