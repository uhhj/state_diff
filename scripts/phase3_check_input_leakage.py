#!/usr/bin/env python3
import sys
from pathlib import Path as _Phase3Path
ROOT = _Phase3Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


def fit_probe(x_train, y_train, x_test, y_test, classes, steps=250, lr=0.2, l2=1e-3):
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


def pair_consistency(data) -> Dict[str, float]:
    split = data["split_name"].astype(str)
    cond = data["condition_name"].astype(str)
    seed = data["visible_seed"].astype(int)
    wt = data["window_t"].astype(int)
    held = np.where(split == "heldout")[0]
    free = {(int(seed[i]), int(wt[i])): i for i in held if cond[i] == "free"}
    pin = {(int(seed[i]), int(wt[i])): i for i in held if cond[i] == "hidden_pin"}
    keys = sorted(set(free) & set(pin))
    rows = []
    for key in keys:
        a = free[key]
        b = pin[key]
        paper = float(np.max(np.abs(data["paper_x"][a] - data["paper_x"][b])))
        state_action = float(np.max(np.abs(data["state_action_x"][a] - data["state_action_x"][b])))
        rows.append((paper, state_action))
    arr = np.asarray(rows, dtype=np.float32) if rows else np.zeros((0, 2), dtype=np.float32)
    return {
        "num_pairs": int(len(rows)),
        "mean_pair_paper_x_max_abs_diff": float(np.mean(arr[:, 0])) if len(arr) else None,
        "max_pair_paper_x_max_abs_diff": float(np.max(arr[:, 0])) if len(arr) else None,
        "mean_pair_state_action_x_max_abs_diff": float(np.mean(arr[:, 1])) if len(arr) else None,
        "max_pair_state_action_x_max_abs_diff": float(np.max(arr[:, 1])) if len(arr) else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_md", required=True)
    ap.add_argument("--pair_threshold", type=float, default=1e-3)
    args = ap.parse_args()
    data = np.load(args.data, allow_pickle=True)
    split = data["split_name"].astype(str)
    cond_id = data["condition_id"].astype(int)
    cond_name = data["condition_name"].astype(str)
    success = data["success"].astype(int)
    train = split == "train"
    held = split == "heldout"

    pair = pair_consistency(data)
    paper_acc = fit_probe(data["paper_x"][train], cond_id[train], data["paper_x"][held], cond_id[held], sorted(set(cond_id.tolist())))
    sa_acc = fit_probe(data["state_action_x"][train], cond_id[train], data["state_action_x"][held], cond_id[held], sorted(set(cond_id.tolist())))
    fp_train = train & np.isin(cond_name, ["free", "hidden_pin"])
    fp_held = held & np.isin(cond_name, ["free", "hidden_pin"])
    fp_y = (cond_name == "hidden_pin").astype(int)
    fp_acc = fit_probe(data["paper_x"][fp_train], fp_y[fp_train], data["paper_x"][fp_held], fp_y[fp_held], [0, 1])
    success_acc = fit_probe(data["paper_x"][train], success[train], data["paper_x"][held], success[held], sorted(set(success.tolist())))

    meta = json.loads(str(data["meta_json"]))
    source_counts = {str(k): int(v) for k, v in meta.get("robot_pose_proxy_source_counts", {}).items()}
    pair_ok = (pair["mean_pair_paper_x_max_abs_diff"] is not None and pair["mean_pair_paper_x_max_abs_diff"] <= args.pair_threshold and pair["mean_pair_state_action_x_max_abs_diff"] <= args.pair_threshold)
    leakage_fail = bool(fp_acc > 0.60 or not pair_ok)
    warning = bool(paper_acc > 0.45 or sa_acc > 0.45 or success_acc > 0.75)
    summary = {
        "pair_consistency": pair,
        "robot_pose_proxy_source_counts": source_counts,
        "paper_x_condition_probe_acc": paper_acc,
        "state_action_x_condition_probe_acc": sa_acc,
        "free_vs_pin_probe_acc": fp_acc,
        "paper_x_success_probe_acc": success_acc,
        "pair_consistency_pass": pair_ok,
        "condition_leakage_fail": leakage_fail,
        "warning_high_probe_accuracy": warning,
        "pass": bool(pair_ok and not leakage_fail),
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(summary, indent=2, sort_keys=True))
    lines = ["# Phase3 Input Consistency and Leakage Report", "", "## Pair Consistency", "", "| Metric | Value |", "|---|---:|"]
    for k, v in pair.items():
        lines.append(f"| `{k}` | `{v}` |")
    lines += ["", "## Robot Pose Proxy Sources", "", "| Source | Count |", "|---|---:|"]
    for k, v in sorted(source_counts.items()):
        lines.append(f"| `{k}` | {v} |")
    lines += ["", "## Probe Accuracies", "", "| Probe | Accuracy |", "|---|---:|",
              f"| `paper_x -> condition_id` | `{paper_acc:.6f}` |",
              f"| `state_action_x -> condition_id` | `{sa_acc:.6f}` |",
              f"| `paper_x -> free_vs_pin` | `{fp_acc:.6f}` |",
              f"| `paper_x -> success` | `{success_acc:.6f}` |",
              "", "## Conclusion", ""]
    if summary["pass"]:
        lines.append("PASS: paired `free` and `hidden_pin` inputs are consistent and probes do not recover hidden condition above the failure threshold.")
    else:
        lines.append("FAIL: Phase3 must stop because paired inputs differ or a probe detects hidden-condition leakage.")
    Path(args.out_md).write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["pass"]:
        raise SystemExit("[Phase3][FAIL] input leakage checks failed")


if __name__ == "__main__":
    main()
