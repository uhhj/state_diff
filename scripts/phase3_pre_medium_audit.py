#!/usr/bin/env python3
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

CONDITIONS = {"free", "hidden_pin", "hidden_high_friction"}
DDPM_MODEL_TYPE = "torch_conditional_ddpm_future_state"
BRANCH_REFERENCE_MODE = "split_visible_seed_window_t"
SCHEDULER_TYPE = "diffusers.DDPMScheduler"
BETA_SCHEDULE = "squaredcos_cap_v2"
PREDICTION_TYPE = "epsilon"
VARIANCE_TYPE = "fixed_small"
DENOISER_ARCH = "mlp"
PAPER_ALIGNMENT_LEVEL = "ddpm_scheduler_aligned_mlp_denoiser"
REQUIRED_FORBIDDEN = {
    "hidden_condition",
    "hidden_contact_meta",
    "success",
    "final_fraction",
    "condition_id",
    "condition_name",
    "ccda_pair_group",
    "source_file",
}


def read_json(path):
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def read_csv(path):
    p = Path(path)
    if not p.exists():
        return []
    with p.open(newline="") as f:
        return list(csv.DictReader(f))


def add(issues, level, name, detail):
    issues.append({"level": level, "name": name, "detail": str(detail)})


def as_bool_str(v):
    return str(v).strip().lower() in {"true", "1", "yes"}


def scalar_json(raw):
    arr = np.asarray(raw)
    if arr.shape == ():
        return json.loads(str(arr.item()))
    return json.loads(str(arr.reshape(-1)[0]))


def counter_dict(counter):
    return {str(k): int(v) for k, v in counter.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/state_diff2")
    ap.add_argument("--data", default="/data/state_diff2/data/phase3_state_diff_windows/phase3_windows.npz")
    ap.add_argument("--out_json", default="/data/state_diff2/reports/phase3_pre_medium_audit_summary.json")
    ap.add_argument("--out_md", default="/data/state_diff2/reports/phase3_pre_medium_audit_report.md")
    args = ap.parse_args()

    root = Path(args.root)
    data = np.load(args.data, allow_pickle=True)
    issues = []

    split = data["split_name"].astype(str)
    seed = data["visible_seed"].astype(int)
    wt = data["window_t"].astype(int)
    cond = data["condition_name"].astype(str)

    train_seeds = set(seed[split == "train"].tolist())
    held_seeds = set(seed[split == "heldout"].tolist())
    overlap = sorted(train_seeds & held_seeds)
    if overlap:
        add(issues, "FAIL", "train_heldout_seed_overlap", overlap)

    bad_splits = sorted(set(split.tolist()) - {"train", "heldout"})
    if bad_splits:
        add(issues, "FAIL", "unexpected_split_names", bad_splits)

    groups = defaultdict(set)
    for s, sd, t, c in zip(split, seed, wt, cond):
        groups[(str(s), int(sd), int(t))].add(str(c))
    incomplete = {str(k): sorted(CONDITIONS - v) for k, v in groups.items() if not CONDITIONS.issubset(v)}
    if incomplete:
        add(issues, "FAIL", "incomplete_condition_groups", dict(list(incomplete.items())[:10]))
    complete_condition_groups = not bool(incomplete)

    th = int(np.asarray(data["th"]).reshape(-1)[0])
    action_dim = int(np.asarray(data["action_dim"]).reshape(-1)[0])
    y_action = data["y_action"]
    paper_x = data["paper_x"]
    state_action_x = data["state_action_x"]

    if y_action.shape[1] != 14:
        add(issues, "FAIL", "bad_y_action_dim", list(y_action.shape))

    extra_dim = int(state_action_x.shape[1] - paper_x.shape[1])
    expected_extra_dim = int(th * action_dim)
    if extra_dim != expected_extra_dim:
        add(issues, "FAIL", "bad_state_action_extra_dim", f"{extra_dim} != {expected_extra_dim}")

    meta = scalar_json(data["meta_json"])
    action_codec = meta.get("action_codec_summary", meta.get("action_codec", {}))
    if action_codec.get("num_camera_config_paths", None) != 0:
        add(issues, "FAIL", "camera_config_in_action_codec", action_codec)
    if int(action_codec.get("dim", -1)) != 14:
        add(issues, "FAIL", "action_codec_dim_not_14", action_codec)

    feature_schema = meta.get("feature_schema", {})
    forbidden = set(feature_schema.get("forbidden_not_in_x", [])) if isinstance(feature_schema, dict) else set()
    if not feature_schema:
        add(issues, "FAIL", "feature_schema_missing", "meta_json has no feature_schema")
    missing_forbidden = sorted(REQUIRED_FORBIDDEN - forbidden)
    if missing_forbidden:
        add(issues, "FAIL", "forbidden_schema_incomplete", missing_forbidden)

    meta_overlap = meta.get("train_heldout_seed_overlap", [])
    if meta_overlap:
        add(issues, "FAIL", "meta_train_heldout_seed_overlap", meta_overlap)

    pred_rows = read_csv(root / "reports/phase3_baseline_eval_predictions.csv")
    pred_counts = {
        "training_backend": Counter(r.get("training_backend", "") for r in pred_rows),
        "future_model_type": Counter(r.get("future_model_type", "") for r in pred_rows),
        "ddpm_used": Counter(str(r.get("ddpm_used", "")).lower() for r in pred_rows),
        "branch_reference_mode": Counter(r.get("branch_reference_mode", "") for r in pred_rows),
        "scheduler_type": Counter(r.get("scheduler_type", "") for r in pred_rows),
        "beta_schedule": Counter(r.get("beta_schedule", "") for r in pred_rows),
        "prediction_type": Counter(r.get("prediction_type", "") for r in pred_rows),
        "variance_type": Counter(r.get("variance_type", "") for r in pred_rows),
        "denoiser_arch": Counter(r.get("denoiser_arch", "") for r in pred_rows),
        "conditional_unet1d_used": Counter(str(r.get("conditional_unet1d_used", "")).lower() for r in pred_rows),
        "paper_alignment_level": Counter(r.get("paper_alignment_level", "") for r in pred_rows),
        "eval_sample_seed_mode": Counter(r.get("eval_sample_seed_mode", "") for r in pred_rows),
    }

    def all_eq(col, val):
        return bool(pred_rows) and all(str(r.get(col, "")) == val for r in pred_rows)

    if not pred_rows:
        add(issues, "FAIL", "prediction_rows_missing", "phase3_baseline_eval_predictions.csv is empty or missing")
    if not all_eq("training_backend", "torch"):
        add(issues, "FAIL", "non_torch_backend", counter_dict(pred_counts["training_backend"]))
    if not all_eq("future_model_type", DDPM_MODEL_TYPE):
        add(issues, "FAIL", "bad_future_model_type", counter_dict(pred_counts["future_model_type"]))
    if not pred_rows or not all(as_bool_str(r.get("ddpm_used", "")) for r in pred_rows):
        add(issues, "FAIL", "ddpm_used_not_true", counter_dict(pred_counts["ddpm_used"]))
    if not all_eq("branch_reference_mode", BRANCH_REFERENCE_MODE):
        add(issues, "FAIL", "bad_branch_reference_mode", counter_dict(pred_counts["branch_reference_mode"]))
    if not all_eq("scheduler_type", SCHEDULER_TYPE):
        add(issues, "FAIL", "bad_scheduler_type", counter_dict(pred_counts["scheduler_type"]))
    if not all_eq("beta_schedule", BETA_SCHEDULE):
        add(issues, "FAIL", "bad_beta_schedule", counter_dict(pred_counts["beta_schedule"]))
    if not all_eq("prediction_type", PREDICTION_TYPE):
        add(issues, "FAIL", "bad_prediction_type", counter_dict(pred_counts["prediction_type"]))
    if not all_eq("variance_type", VARIANCE_TYPE):
        add(issues, "WARN", "variance_not_fixed_small", counter_dict(pred_counts["variance_type"]))
    else:
        add(issues, "WARN", "variance_fixed_small_not_learned_range", "fixed_small is acceptable before medium, but not exact learned_range alignment.")
    if not all_eq("denoiser_arch", DENOISER_ARCH):
        add(issues, "WARN", "unexpected_denoiser_arch", counter_dict(pred_counts["denoiser_arch"]))
    else:
        add(issues, "WARN", "mlp_denoiser_not_conditional_unet1d", "Scheduler is aligned, but denoiser is still MLP, not original ConditionalUnet1D.")
    if not all_eq("paper_alignment_level", PAPER_ALIGNMENT_LEVEL):
        add(issues, "WARN", "paper_alignment_level_unexpected", counter_dict(pred_counts["paper_alignment_level"]))

    missing_refs = 0
    for r in pred_rows:
        if str(r.get("has_free_ref", "")).lower() in {"false", "0"} or str(r.get("has_pin_ref", "")).lower() in {"false", "0"}:
            missing_refs += 1
    if missing_refs:
        add(issues, "FAIL", "missing_branch_refs", missing_refs)

    leak = read_json(root / "reports/phase3_input_leakage_summary.json")
    if leak.get("pass") is not True:
        add(issues, "FAIL", "input_leakage_not_pass", leak)

    action_dbg = read_json(root / "reports/phase3_action_idm_debug_summary.json")
    if action_dbg.get("verdict") == "FAIL":
        add(issues, "FAIL", "action_idm_debug_fail", action_dbg.get("issues", []))
    extra_block = action_dbg.get("state_action_extra_block", {}) if isinstance(action_dbg, dict) else {}
    leak_exact = extra_block.get("action_history_target_exact_match_rate", action_dbg.get("action_history_target_exact_match_rate"))
    if leak_exact is not None and float(leak_exact) > 0.20:
        add(issues, "FAIL", "action_history_target_match_too_high", leak_exact)

    canon = read_json(root / "reports/phase3_canonicalization_summary.json")
    if not canon:
        add(issues, "FAIL", "canonicalization_summary_missing", "")
    else:
        raw_max = canon.get("raw_max_pair_paper_x_max_abs_diff", None)
        post_max = canon.get("post_max_pair_paper_x_max_abs_diff", None)
        raw_sa = canon.get("raw_max_pair_state_action_x_max_abs_diff", None)
        post_sa = canon.get("post_max_pair_state_action_x_max_abs_diff", None)
        if None in (raw_max, post_max, raw_sa, post_sa):
            add(issues, "FAIL", "canonicalization_diff_missing", canon)
        elif float(raw_max) > 0.1:
            add(issues, "WARN", "canonicalization_strong_intervention", f"raw_max_pair_paper_x_max_abs_diff={raw_max}; acceptable as matched-input audit control.")

    eval_summary = read_json(root / "reports/phase3_baseline_eval_summary.json")
    missing_branch_refs_summary = int(eval_summary.get("num_missing_primary_branch_refs", 0) or 0)
    if missing_branch_refs_summary:
        add(issues, "FAIL", "eval_summary_missing_branch_refs", missing_branch_refs_summary)

    has_fail = any(i["level"] == "FAIL" for i in issues)
    has_warn = any(i["level"] == "WARN" for i in issues)
    verdict = "FAIL" if has_fail else ("WARN" if has_warn else "PASS")

    payload = {
        "verdict": verdict,
        "train_visible_seed_count": len(train_seeds),
        "heldout_visible_seed_count": len(held_seeds),
        "train_heldout_seed_overlap": overlap,
        "complete_condition_groups": complete_condition_groups,
        "num_condition_groups": len(groups),
        "num_incomplete_condition_groups": len(incomplete),
        "num_prediction_rows": len(pred_rows),
        "prediction_metadata_counts": {k: counter_dict(v) for k, v in pred_counts.items()},
        "feature_schema": feature_schema,
        "forbidden_not_in_x": sorted(forbidden),
        "branch_reference_mode": BRANCH_REFERENCE_MODE,
        "missing_branch_refs": missing_refs,
        "eval_summary_missing_branch_refs": missing_branch_refs_summary,
        "canonicalization_summary": canon,
        "input_leakage_summary": leak,
        "input_leakage_pass": leak.get("pass"),
        "action_idm_debug_verdict": action_dbg.get("verdict"),
        "action_history_target_exact_match_rate": leak_exact,
        "issues": issues,
    }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(payload, indent=2, sort_keys=True))

    lines = [
        "# Phase3 Pre-Medium Audit Report",
        "",
        "## Verdict",
        "",
        f"- Verdict: `{verdict}`",
        f"- Train visible seeds: `{len(train_seeds)}`",
        f"- Heldout visible seeds: `{len(held_seeds)}`",
        f"- Train/Heldout seed overlap: `{overlap}`",
        f"- Complete condition groups: `{complete_condition_groups}`",
        f"- Missing branch refs: `{missing_refs}`",
        f"- Input leakage pass: `{leak.get('pass')}`",
        f"- Action-history target exact match rate: `{leak_exact}`",
        f"- Prediction rows: `{len(pred_rows)}`",
        "",
        "## Feature Schema",
        "",
        f"- Feature schema present: `{bool(feature_schema)}`",
        f"- forbidden_not_in_x: `{sorted(forbidden)}`",
        "",
        "## Prediction Metadata Counts",
        "",
    ]
    for name, counts in payload["prediction_metadata_counts"].items():
        lines += [f"### {name}", "", "| Value | Count |", "|---|---:|"]
        for k, v in sorted(counts.items()):
            lines.append(f"| `{k}` | {v} |")
        lines.append("")
    lines += [
        "## Canonicalization",
        "",
        f"- raw_max_pair_paper_x_max_abs_diff: `{canon.get('raw_max_pair_paper_x_max_abs_diff')}`",
        f"- post_max_pair_paper_x_max_abs_diff: `{canon.get('post_max_pair_paper_x_max_abs_diff')}`",
        f"- raw_max_pair_state_action_x_max_abs_diff: `{canon.get('raw_max_pair_state_action_x_max_abs_diff')}`",
        f"- post_max_pair_state_action_x_max_abs_diff: `{canon.get('post_max_pair_state_action_x_max_abs_diff')}`",
        "",
        "## Issues",
        "",
        "| Level | Name | Detail |",
        "|---|---|---|",
    ]
    if issues:
        for i in issues:
            lines.append(f"| `{i['level']}` | `{i['name']}` | {i['detail']} |")
    else:
        lines.append("| `PASS` | `none` | No issues. |")
    lines += [
        "",
        "## Interpretation",
        "",
        "- FAIL blocks MODE=medium.",
        "- WARN is allowed before medium only if explicitly discussed.",
        "- `fixed_small` variance and `mlp` denoiser are acceptable for scheduler-aligned medium, but not exact ConditionalUnet1D / learned_range reproduction.",
    ]
    Path(args.out_md).write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))

    if has_fail:
        raise SystemExit("[Phase3][FAIL] pre-medium audit failed")


if __name__ == "__main__":
    main()
