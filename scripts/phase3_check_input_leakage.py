#!/usr/bin/env python3
import sys
from pathlib import Path as _Phase3Path
ROOT = _Phase3Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from ccda_phase3.data_io import FORBIDDEN_METADATA_NOT_IN_X, normalize_conditions


def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


def fit_probe(x_train, y_train, x_test, y_test, classes, steps=250, lr=0.2, l2=1e-3):
    classes = list(classes)
    if len(classes) < 2 or len(x_train) == 0 or len(x_test) == 0:
        return float("nan")
    x_train = np.asarray(x_train, dtype=np.float32)
    x_test = np.asarray(x_test, dtype=np.float32)
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    xt = (x_train - mean) / std
    xv = (x_test - mean) / std
    label_to_i = {int(c): i for i, c in enumerate(classes)}
    yi = np.asarray([label_to_i[int(y)] for y in y_train], dtype=np.int64)
    yv = np.asarray([label_to_i[int(y)] for y in y_test], dtype=np.int64)
    k = len(classes)
    w = np.zeros((xt.shape[1], k), dtype=np.float32)
    b = np.zeros((1, k), dtype=np.float32)
    y_one = np.eye(k, dtype=np.float32)[yi]
    for _ in range(steps):
        p = softmax(xt @ w + b)
        grad = (p - y_one) / max(1, len(xt))
        w -= lr * (xt.T @ grad + l2 * w)
        b -= lr * np.sum(grad, axis=0, keepdims=True)
    pred = np.argmax(softmax(xv @ w + b), axis=1)
    return float(np.mean(pred == yv)) if len(yv) else float("nan")


def hidden_conditions(conditions: Iterable[str]) -> List[str]:
    return [c for c in conditions if c != "free"]


def stat(xs):
    if not xs:
        return {"mean": None, "max": None}
    return {"mean": float(np.mean(xs)), "max": float(np.max(xs))}


def pair_consistency(data, conditions) -> Dict[str, object]:
    split = data["split_name"].astype(str)
    cond = data["condition_name"].astype(str)
    seed = data["visible_seed"].astype(int)
    wt = data["window_t"].astype(int)
    held = np.where(split == "heldout")[0]
    free = {(int(seed[i]), int(wt[i])): i for i in held if cond[i] == "free"}
    by_hidden = {}
    all_paper = []
    all_state_action = []
    for h in hidden_conditions(conditions):
        hidx = {(int(seed[i]), int(wt[i])): i for i in held if cond[i] == h}
        keys = sorted(set(free) & set(hidx))
        paper_vals = []
        sa_vals = []
        for key in keys:
            a = free[key]
            b = hidx[key]
            paper = float(np.max(np.abs(data["paper_x"][a] - data["paper_x"][b])))
            state_action = float(np.max(np.abs(data["state_action_x"][a] - data["state_action_x"][b])))
            paper_vals.append(paper)
            sa_vals.append(state_action)
            all_paper.append(paper)
            all_state_action.append(state_action)
        by_hidden[h] = {
            "num_pairs": int(len(keys)),
            "mean_pair_paper_x_max_abs_diff": stat(paper_vals)["mean"],
            "max_pair_paper_x_max_abs_diff": stat(paper_vals)["max"],
            "mean_pair_state_action_x_max_abs_diff": stat(sa_vals)["mean"],
            "max_pair_state_action_x_max_abs_diff": stat(sa_vals)["max"],
        }
    return {
        "num_pairs": int(sum(v["num_pairs"] for v in by_hidden.values())),
        "mean_pair_paper_x_max_abs_diff": stat(all_paper)["mean"],
        "max_pair_paper_x_max_abs_diff": stat(all_paper)["max"],
        "mean_pair_state_action_x_max_abs_diff": stat(all_state_action)["mean"],
        "max_pair_state_action_x_max_abs_diff": stat(all_state_action)["max"],
        "by_hidden_condition": by_hidden,
    }


def binary_probe(data, condition, x_key="paper_x"):
    split = data["split_name"].astype(str)
    cond_name = data["condition_name"].astype(str)
    train = split == "train"
    held = split == "heldout"
    mask = np.isin(cond_name, ["free", condition])
    y = (cond_name == condition).astype(int)
    return fit_probe(data[x_key][train & mask], y[train & mask], data[x_key][held & mask], y[held & mask], [0, 1])


def forbidden_schema_check(meta: Dict) -> Dict[str, object]:
    schema = meta.get("feature_schema", {}) if isinstance(meta, dict) else {}
    forbidden = set(meta.get("forbidden_metadata_not_in_x") or schema.get("forbidden_not_in_x") or [])
    required = set(FORBIDDEN_METADATA_NOT_IN_X)
    x_fields = []
    for k in ["paper_x", "state_action_x"]:
        x_fields.extend(str(v) for v in schema.get(k, []))
    present_in_x = sorted([name for name in forbidden if any(name in field for field in x_fields)])
    missing_required = sorted(required - forbidden)
    return {
        "forbidden_count": int(len(forbidden)),
        "missing_required_forbidden_entries": missing_required,
        "forbidden_names_present_in_x_schema": present_in_x,
        "pass": not missing_required and not present_in_x,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_md", required=True)
    ap.add_argument("--conditions", nargs="+", default=None)
    ap.add_argument("--primary_hidden_condition", default=None)
    ap.add_argument("--diagnostic_hidden_condition", default=None)
    ap.add_argument("--pair_threshold", type=float, default=1e-8)
    args = ap.parse_args()
    data = np.load(args.data, allow_pickle=True)
    meta = json.loads(str(data["meta_json"]))
    conditions = normalize_conditions(args.conditions or meta.get("conditions"))
    primary = args.primary_hidden_condition or meta.get("primary_hidden_condition", "hidden_breakaway_pin")
    diagnostic = args.diagnostic_hidden_condition or meta.get("diagnostic_hidden_condition", "hidden_pin")
    if primary not in conditions:
        raise SystemExit(f"[Phase3][FAIL] primary_hidden_condition={primary} not in conditions={conditions}")
    if diagnostic not in conditions:
        raise SystemExit(f"[Phase3][FAIL] diagnostic_hidden_condition={diagnostic} not in conditions={conditions}")

    split = data["split_name"].astype(str)
    cond_id = data["condition_id"].astype(int)
    cond_name = data["condition_name"].astype(str)
    success = data["success"].astype(int)
    train = split == "train"
    held = split == "heldout"

    pair = pair_consistency(data, conditions)
    paper_acc = fit_probe(data["paper_x"][train], cond_id[train], data["paper_x"][held], cond_id[held], sorted(set(cond_id.tolist())))
    sa_acc = fit_probe(data["state_action_x"][train], cond_id[train], data["state_action_x"][held], cond_id[held], sorted(set(cond_id.tolist())))
    primary_acc = binary_probe(data, primary, "paper_x")
    diagnostic_acc = binary_probe(data, diagnostic, "paper_x")
    success_acc = fit_probe(data["paper_x"][train], success[train], data["paper_x"][held], success[held], sorted(set(success.tolist())))

    source_counts = {str(k): int(v) for k, v in meta.get("robot_pose_proxy_source_counts", {}).items()}
    max_pair = pair.get("max_pair_paper_x_max_abs_diff")
    max_pair_sa = pair.get("max_pair_state_action_x_max_abs_diff")
    pair_ok = (
        max_pair is not None
        and max_pair_sa is not None
        and float(max_pair) <= args.pair_threshold
        and float(max_pair_sa) <= args.pair_threshold
        and all(v.get("num_pairs", 0) > 0 for v in pair.get("by_hidden_condition", {}).values())
    )
    forbidden_check = forbidden_schema_check(meta)
    leakage_fail = bool(
        not pair_ok
        or not forbidden_check["pass"]
        or (np.isfinite(primary_acc) and primary_acc > 0.60)
        or (np.isfinite(diagnostic_acc) and diagnostic_acc > 0.60)
    )
    warning = bool(
        (np.isfinite(paper_acc) and paper_acc > 0.45)
        or (np.isfinite(sa_acc) and sa_acc > 0.45)
        or (np.isfinite(success_acc) and success_acc > 0.75)
    )
    summary = {
        "conditions": conditions,
        "primary_hidden_condition": primary,
        "diagnostic_hidden_condition": diagnostic,
        "primary_branch_pair": f"free_vs_{primary}",
        "diagnostic_branch_pair": f"free_vs_{diagnostic}",
        "pair_consistency": pair,
        "pair_consistency_by_hidden_condition": pair.get("by_hidden_condition", {}),
        "primary_pair_consistency": pair.get("by_hidden_condition", {}).get(primary, {}),
        "diagnostic_pair_consistency": pair.get("by_hidden_condition", {}).get(diagnostic, {}),
        "robot_pose_proxy_source_counts": source_counts,
        "paper_x_condition_probe_acc": paper_acc,
        "state_action_x_condition_probe_acc": sa_acc,
        "free_vs_primary_probe_acc": primary_acc,
        "free_vs_diagnostic_probe_acc": diagnostic_acc,
        "free_vs_pin_probe_acc": diagnostic_acc if diagnostic == "hidden_pin" else None,
        "paper_x_success_probe_acc": success_acc,
        "forbidden_metadata_schema_check": forbidden_check,
        "pair_consistency_pass": pair_ok,
        "condition_leakage_fail": leakage_fail,
        "warning_high_probe_accuracy": warning,
        "pass": bool(pair_ok and not leakage_fail),
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(summary, indent=2, sort_keys=True))

    lines = [
        "# Phase3 Input Consistency and Leakage Report",
        "",
        "## Condition Scope",
        "",
        f"- Conditions: `{', '.join(conditions)}`",
        f"- Primary pair: `free_vs_{primary}`",
        f"- Diagnostic pair: `free_vs_{diagnostic}`",
        "",
        "## Pair Consistency By Hidden Condition",
        "",
        "| Hidden condition | Pairs | Max paper_x diff | Max state_action_x diff |",
        "|---|---:|---:|---:|",
    ]
    for h, item in pair.get("by_hidden_condition", {}).items():
        lines.append(f"| `{h}` | {item.get('num_pairs')} | `{item.get('max_pair_paper_x_max_abs_diff')}` | `{item.get('max_pair_state_action_x_max_abs_diff')}` |")
    lines += ["", "## Robot Pose Proxy Sources", "", "| Source | Count |", "|---|---:|"]
    for k, v in sorted(source_counts.items()):
        lines.append(f"| `{k}` | {v} |")
    lines += [
        "",
        "## Probe Accuracies",
        "",
        "| Probe | Accuracy |",
        "|---|---:|",
        f"| `paper_x -> condition_id` | `{paper_acc:.6f}` |",
        f"| `state_action_x -> condition_id` | `{sa_acc:.6f}` |",
        f"| `paper_x -> free_vs_{primary}` | `{primary_acc:.6f}` |",
        f"| `paper_x -> free_vs_{diagnostic}` | `{diagnostic_acc:.6f}` |",
        f"| `paper_x -> success` | `{success_acc:.6f}` |",
        "",
        "## Forbidden Metadata Schema Check",
        "",
        f"- Pass: `{forbidden_check['pass']}`",
        f"- Missing required forbidden entries: `{forbidden_check['missing_required_forbidden_entries']}`",
        f"- Forbidden names present in x schema: `{forbidden_check['forbidden_names_present_in_x_schema']}`",
        "",
        "## Conclusion",
        "",
    ]
    if summary["pass"]:
        lines.append("PASS: paired `free` and hidden-branch inputs are identical after canonicalization, and hidden/contact metadata remains outside model inputs.")
    else:
        lines.append("FAIL: Phase3 must stop because paired inputs differ, a primary/diagnostic probe detects hidden-condition leakage, or forbidden metadata appears in the input schema.")
    Path(args.out_md).write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["pass"]:
        raise SystemExit("[Phase3][FAIL] input leakage checks failed")


if __name__ == "__main__":
    main()
