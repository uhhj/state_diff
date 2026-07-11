#!/usr/bin/env python3
"""Grouped observation-space identifiability audit for Phase3.12d-r2.2."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

from phase3_12d_r22_integrity import assert_disjoint_groups, grouped_fold_indices, strict_json_dump


FEATURES = ("xy", "xy_velocity", "full_state", "model_x")
COMPARISONS = (
    ("free_a_vs_hidden_armed", "free_a", "hidden_armed"),
    ("free_a_vs_free_b", "free_a", "free_b"),
    ("free_a_vs_hidden_unarmed", "free_a", "hidden_unarmed"),
)


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    order = np.argsort(-score, kind="mergesort")
    labels = y[order]
    positives = int(labels.sum())
    if positives == 0:
        return 0.0
    precision = np.cumsum(labels) / np.arange(1, labels.size + 1)
    return float(np.sum(precision * labels) / positives)


def roc_auc_fast(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if not n_pos or not n_neg:
        return 0.5
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(score.size, dtype=np.float64)
    start = 0
    while start < order.size:
        end = start + 1
        while end < order.size and score[order[end]] == score[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    return float((np.sum(ranks[y == 1]) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def metrics(y: np.ndarray, score: np.ndarray) -> Dict[str, Any]:
    y = np.asarray(y, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    pred = score >= 0.5
    tp = int(np.sum((y == 1) & pred))
    tn = int(np.sum((y == 0) & ~pred))
    fp = int(np.sum((y == 0) & pred))
    fn = int(np.sum((y == 1) & ~pred))
    tpr = tp / max(1, tp + fn)
    tnr = tn / max(1, tn + fp)
    return {
        "accuracy": float((tp + tn) / max(1, y.size)),
        "balanced_accuracy": float(0.5 * (tpr + tnr)),
        "roc_auc": roc_auc_fast(y, score),
        "average_precision": average_precision(y, score),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def fit_torch_linear(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, seed: int) -> np.ndarray:
    mean = x_train.mean(axis=0, dtype=np.float64)
    scale = x_train.std(axis=0, dtype=np.float64)
    scale[scale < 1e-8] = 1.0
    train = torch.from_numpy(((x_train - mean) / scale).astype(np.float32))
    test = torch.from_numpy(((x_test - mean) / scale).astype(np.float32))
    target = torch.from_numpy(y_train.astype(np.float32)).reshape(-1, 1)
    torch.manual_seed(int(seed))
    model = torch.nn.Linear(train.shape[1], 1)
    torch.nn.init.zeros_(model.weight)
    torch.nn.init.zeros_(model.bias)
    counts = np.bincount(y_train, minlength=2).astype(np.float32)
    sample_weight = torch.from_numpy(np.asarray([len(y_train) / (2.0 * max(1.0, counts[int(label)])) for label in y_train], dtype=np.float32)).reshape(-1, 1)
    optimizer = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=80, tolerance_grad=1e-7, tolerance_change=1e-9, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        logits = model(train)
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none") * sample_weight).mean()
        loss = loss + 5e-4 * torch.sum(model.weight * model.weight)
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        return torch.sigmoid(model(test)).reshape(-1).cpu().numpy().astype(np.float64)


def feature_matrix(data: Dict[str, np.ndarray], feature: str) -> np.ndarray:
    if feature == "xy":
        return data["bead_xy"].reshape(data["bead_xy"].shape[0], -1)
    if feature == "xy_velocity":
        return np.concatenate([data["bead_xy"].reshape(data["bead_xy"].shape[0], -1), data["bead_velocity_xy"].reshape(data["bead_velocity_xy"].shape[0], -1)], axis=1)
    if feature == "full_state":
        return data["state"]
    if feature == "model_x":
        return data["model_x"]
    raise KeyError(feature)


def select_comparison(data: Dict[str, np.ndarray], feature: str, horizon: int, left: str, right: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tracks = data["track_name"].astype(str)
    mask = (data["evaluation_steps"] == int(horizon)) & np.isin(tracks, [left, right])
    indices = np.flatnonzero(mask)
    x = feature_matrix(data, feature)[indices].astype(np.float32)
    y = (tracks[indices] == right).astype(np.int64)
    groups = data["visible_seed"][indices].astype(np.int64)
    if np.unique(groups).size * 2 != indices.size:
        raise RuntimeError(f"comparison is not paired: {left}/{right} horizon={horizon}")
    return x, y, groups, indices


def group_bootstrap(y: np.ndarray, score: np.ndarray, groups: np.ndarray, iterations: int, seed: int) -> Dict[str, Tuple[float, float]]:
    unique = np.unique(groups)
    inverse = np.searchsorted(unique, groups)
    for group in unique:
        labels = y[groups == group]
        if labels.size != 2 or set(labels.tolist()) != {0, 1}:
            raise RuntimeError("fast group bootstrap requires one paired negative/positive per seed")
    rng = np.random.default_rng(int(seed))
    counts = rng.multinomial(
        unique.size,
        np.full(unique.size, 1.0 / unique.size),
        size=int(iterations),
    ).astype(np.float64)
    row_weights = counts[:, inverse]
    predicted = score >= 0.5
    correct = (predicted == y).astype(np.float64)
    accuracy = np.sum(row_weights * correct[None, :], axis=1) / (2.0 * unique.size)

    negative_score = np.asarray([score[(groups == group) & (y == 0)][0] for group in unique])
    positive_score = np.asarray([score[(groups == group) & (y == 1)][0] for group in unique])
    pair_order = positive_score[:, None] - negative_score[None, :]
    pair_credit = (pair_order > 0).astype(np.float64) + 0.5 * (pair_order == 0)
    auc = np.einsum("bi,ij,bj->b", counts, pair_credit, counts, optimize=True) / float(unique.size * unique.size)

    order = np.argsort(-score, kind="mergesort")
    ordered_weights = row_weights[:, order]
    ordered_positive_weights = ordered_weights * y[order][None, :]
    cumulative_positive = np.cumsum(ordered_positive_weights, axis=1)
    cumulative_total = np.cumsum(ordered_weights, axis=1)
    precision = np.divide(cumulative_positive, cumulative_total, out=np.zeros_like(cumulative_positive), where=cumulative_total > 0)
    ap = np.sum(precision * ordered_positive_weights, axis=1) / float(unique.size)
    values = {
        "accuracy": accuracy,
        "balanced_accuracy": accuracy.copy(),
        "roc_auc": auc,
        "average_precision": ap,
    }
    return {key: (float(np.quantile(value, 0.025)), float(np.quantile(value, 0.975))) for key, value in values.items()}


def paired_permutation_p(y: np.ndarray, score: np.ndarray, groups: np.ndarray, iterations: int, seed: int) -> float:
    unique = np.unique(groups)
    observed = roc_auc_fast(y, score)
    rng = np.random.default_rng(int(seed))
    exceed = 0
    for _ in range(iterations):
        permuted = y.copy()
        for group in unique:
            indices = np.flatnonzero(groups == group)
            if rng.random() < 0.5:
                permuted[indices] = 1 - permuted[indices]
        if roc_auc_fast(permuted, score) >= observed - 1e-12:
            exceed += 1
    return float((exceed + 1) / (iterations + 1))


def paired_distance(data: Dict[str, np.ndarray], feature: str, horizon: int, left: str, right: str) -> Dict[str, Any]:
    matrix = feature_matrix(data, feature).astype(np.float64)
    tracks = data["track_name"].astype(str)
    steps = data["evaluation_steps"]
    seeds = data["visible_seed"]
    diffs = []
    left_values = []
    for seed in np.unique(seeds):
        li = np.flatnonzero((tracks == left) & (steps == horizon) & (seeds == seed))
        ri = np.flatnonzero((tracks == right) & (steps == horizon) & (seeds == seed))
        if li.size != 1 or ri.size != 1:
            raise RuntimeError("missing paired distance row")
        left_values.append(matrix[li[0]])
        diffs.append(matrix[ri[0]] - matrix[li[0]])
    diff = np.stack(diffs)
    base = np.stack(left_values)
    std = np.std(base, axis=0)
    safe_std = np.where(std < 1e-8, 1.0, std)
    between_seed = float(np.mean(np.linalg.norm(base - base.mean(axis=0), axis=1)))
    l2 = np.linalg.norm(diff, axis=1)
    return {
        "feature": feature,
        "comparison": f"{left}_vs_{right}",
        "horizon": int(horizon),
        "paired_l2_mean": float(np.mean(l2)),
        "paired_mae": float(np.mean(np.abs(diff))),
        "paired_max_abs": float(np.max(np.abs(diff))),
        "normalized_paired_l2_mean": float(np.mean(np.linalg.norm(diff / safe_std, axis=1))),
        "leakage_over_train_std": float(np.mean(np.abs(diff) / safe_std)),
        "leakage_over_between_seed": float(np.mean(l2) / max(between_seed, 1e-12)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--data", default="data/phase3_12d_r22_observation_leakage/paired_observations.npz")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--bootstraps", type=int, default=10000)
    parser.add_argument("--permutations", type=int, default=1000)
    args = parser.parse_args()
    torch.set_num_threads(1)
    root = Path(args.root).resolve()
    with np.load(root / args.data, allow_pickle=False) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    if data["state"].shape[0] != 2048:
        raise SystemExit(f"expected 2048 rows, got {data['state'].shape[0]}")
    if any(not np.all(np.isfinite(data[key])) for key in ("state", "bead_xy", "bead_velocity_xy", "robot_proxy", "model_x")):
        raise SystemExit("non-finite classifier feature")

    fold_rows: List[Dict[str, Any]] = []
    prediction_rows: List[Dict[str, Any]] = []
    aggregate_rows: List[Dict[str, Any]] = []
    bootstrap_rows: List[Dict[str, Any]] = []
    distance_rows: List[Dict[str, Any]] = []
    horizons = sorted(int(value) for value in np.unique(data["evaluation_steps"]))
    for horizon in horizons:
        for comparison, left, right in COMPARISONS:
            for feature in FEATURES:
                x, y, groups, source_indices = select_comparison(data, feature, horizon, left, right)
                repeat_scores = []
                for repeat in range(args.repeats):
                    scores = np.empty(y.size, dtype=np.float64)
                    for fold, (train, test) in enumerate(grouped_fold_indices(groups, args.folds, repeat)):
                        assert_disjoint_groups(groups[train], groups[test])
                        scores[test] = fit_torch_linear(x[train], y[train], x[test], seed=312220 + repeat * 100 + fold)
                        result = metrics(y[test], scores[test])
                        fold_rows.append({"comparison": comparison, "horizon": horizon, "feature": feature, "repeat": repeat, "fold": fold, "train_groups": np.unique(groups[train]).size, "test_groups": np.unique(groups[test]).size, **result})
                        for local in test:
                            prediction_rows.append({"comparison": comparison, "horizon": horizon, "feature": feature, "repeat": repeat, "fold": fold, "source_index": int(source_indices[local]), "visible_seed": int(groups[local]), "label": int(y[local]), "score": float(scores[local])})
                    repeat_scores.append(scores)
                score = np.mean(np.stack(repeat_scores), axis=0)
                result = metrics(y, score)
                ci = group_bootstrap(y, score, groups, args.bootstraps, seed=312220 + horizon * 100 + len(aggregate_rows))
                pvalue = paired_permutation_p(y, score, groups, args.permutations, seed=322220 + horizon * 100 + len(aggregate_rows))
                row = {"comparison": comparison, "horizon": horizon, "feature": feature, **result, "auc_ci_low": ci["roc_auc"][0], "auc_ci_high": ci["roc_auc"][1], "balanced_accuracy_ci_low": ci["balanced_accuracy"][0], "balanced_accuracy_ci_high": ci["balanced_accuracy"][1], "permutation_p": pvalue}
                aggregate_rows.append(row)
                bootstrap_rows.append({"comparison": comparison, "horizon": horizon, "feature": feature, **{f"{key}_ci_low": value[0] for key, value in ci.items()}, **{f"{key}_ci_high": value[1] for key, value in ci.items()}, "iterations": args.bootstraps})
                distance_rows.append(paired_distance(data, feature, horizon, left, right))
                print(f"[r2.2] analyzed {comparison} h={horizon} feature={feature} auc={result['roc_auc']:.4f}", flush=True)

    controls = [row for row in aggregate_rows if row["comparison"] != "free_a_vs_hidden_armed" or row["horizon"] == 0]
    control_failures = [row for row in controls if row["auc_ci_low"] > 0.60]
    primary = [row for row in aggregate_rows if row["comparison"] == "free_a_vs_hidden_armed" and row["horizon"] == 20]
    observed = [row for row in primary if row["roc_auc"] >= 0.70 and row["auc_ci_low"] > 0.65 and row["permutation_p"] < 0.01 and row["balanced_accuracy"] >= 0.65]
    below = bool(primary) and all(row["roc_auc"] <= 0.60 and row["auc_ci_high"] < 0.65 and row["balanced_accuracy"] <= 0.60 for row in primary)
    if control_failures:
        verdict, root_cause = "FAIL", "phase312d_r22_control_leakage_detected"
    elif observed:
        verdict, root_cause = "FAIL", "phase312d_r22_latent_condition_observably_leaked"
    elif below:
        verdict, root_cause = "PASS", "phase312d_r22_leakage_below_observation_identifiability"
    else:
        verdict, root_cause = "WARN", "phase312d_r22_identifiability_inconclusive"

    control_rows = []
    for name in ("free_a_vs_free_b", "free_a_vs_hidden_unarmed"):
        rows = [row for row in aggregate_rows if row["comparison"] == name]
        control_rows.append({"comparison": name, "max_auc": max(row["roc_auc"] for row in rows), "max_auc_ci_low": max(row["auc_ci_low"] for row in rows), "result": "FAIL" if any(row["auc_ci_low"] > 0.60 for row in rows) else "PASS"})
    rows = [row for row in aggregate_rows if row["comparison"] == "free_a_vs_hidden_armed" and row["horizon"] == 0]
    control_rows.append({"comparison": "horizon0_free_a_vs_hidden_armed", "max_auc": max(row["roc_auc"] for row in rows), "max_auc_ci_low": max(row["auc_ci_low"] for row in rows), "result": "FAIL" if any(row["auc_ci_low"] > 0.60 for row in rows) else "PASS"})

    payload = {
        "verdict": verdict,
        "root_cause": root_cause,
        "classifier_backend": "torch_linear_bce_lbfgs",
        "grouping": "visible_seed",
        "folds": args.folds,
        "repeats": args.repeats,
        "bootstraps": args.bootstraps,
        "permutations": args.permutations,
        "rows": int(data["state"].shape[0]),
        "visible_seeds": int(np.unique(data["visible_seed"]).size),
        "controls": control_rows,
        "control_failures": control_failures,
        "horizon20_primary": primary,
        "observably_leaked_features": [row["feature"] for row in observed],
        "scientific_boundary": "This diagnostic tests decodability for the specified seeds, horizons, and structured observation features. It does not prove permanent unobservability, CPS effectiveness, or model correctness.",
    }
    strict_json_dump(root / "reports/phase3_12d_r22_observation_leakage_summary.json", payload)
    write_csv(root / "reports/phase3_12d_r22_fold_metrics.csv", fold_rows, list(fold_rows[0]))
    write_csv(root / "reports/phase3_12d_r22_bootstrap_metrics.csv", bootstrap_rows, list(bootstrap_rows[0]))
    write_csv(root / "reports/phase3_12d_r22_distance_summary.csv", distance_rows, list(distance_rows[0]))
    write_csv(root / "reports/phase3_12d_r22_control_summary.csv", control_rows, list(control_rows[0]))
    write_csv(root / "reports/phase3_12d_r22_predictions.csv", prediction_rows, list(prediction_rows[0]))
    lines = ["# Phase3.12d-r2.2 Observation-Space Leakage Audit", "", f"- Verdict: `{verdict}`", f"- Root cause: `{root_cause}`", f"- Backend: `torch_linear_bce_lbfgs`", f"- Groups: `{payload['visible_seeds']} visible seeds`", "", "## Horizon-20 Primary Comparison", "", "| Feature | ROC-AUC | 95% CI | Balanced accuracy | Permutation p |", "|---|---:|---|---:|---:|"]
    lines.extend(f"| `{row['feature']}` | `{row['roc_auc']:.6f}` | `[{row['auc_ci_low']:.6f}, {row['auc_ci_high']:.6f}]` | `{row['balanced_accuracy']:.6f}` | `{row['permutation_p']:.6f}` |" for row in primary)
    lines += ["", "## Controls", "", "| Comparison | Max AUC | Max lower CI | Result |", "|---|---:|---:|---|"]
    lines.extend(f"| `{row['comparison']}` | `{row['max_auc']:.6f}` | `{row['max_auc_ci_low']:.6f}` | `{row['result']}` |" for row in control_rows)
    lines += ["", "## Interpretation", "", payload["scientific_boundary"], ""]
    (root / "reports/phase3_12d_r22_observation_leakage_report.md").write_text("\n".join(lines))
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
