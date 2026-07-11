#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np
import torch

from phase3_12d_r22_integrity import (
    assert_disjoint_groups,
    grouped_fold_indices,
    strict_json_dump,
)


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    legal_observation: bool
    family: str
    builder: Callable[..., np.ndarray]


def write_csv(
    path: Path,
    rows: List[Dict[str, Any]],
    fields: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fields),
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    positive = score[y == 1]
    negative = score[y == 0]
    if not positive.size or not negative.size:
        return 0.5
    difference = positive[:, None] - negative[None, :]
    return float(
        np.mean(difference > 0)
        + 0.5 * np.mean(difference == 0)
    )


def rank01(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    order = np.argsort(value, kind="mergesort")
    ranks = np.empty(value.size, dtype=np.float64)
    start = 0
    while start < order.size:
        end = start + 1
        while (
            end < order.size
            and value[order[end]] == value[order[start]]
        ):
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    denominator = max(1, value.size - 1)
    return ranks / float(denominator)


def classification_metrics(
    y: np.ndarray,
    score: np.ndarray,
) -> Dict[str, Any]:
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
        "roc_auc": roc_auc(y, score),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def fit_linear_logits(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    mean = x_train.mean(axis=0, dtype=np.float64)
    scale = x_train.std(axis=0, dtype=np.float64)
    scale[scale < 1e-8] = 1.0
    train = torch.from_numpy(
        ((x_train - mean) / scale).astype(np.float32)
    )
    test = torch.from_numpy(
        ((x_test - mean) / scale).astype(np.float32)
    )
    target = torch.from_numpy(
        y_train.astype(np.float32)
    ).reshape(-1, 1)

    torch.manual_seed(int(seed))
    model = torch.nn.Linear(train.shape[1], 1)
    torch.nn.init.zeros_(model.weight)
    torch.nn.init.zeros_(model.bias)

    counts = np.bincount(y_train, minlength=2).astype(np.float32)
    sample_weight = torch.from_numpy(
        np.asarray(
            [
                len(y_train)
                / (2.0 * max(1.0, counts[int(label)]))
                for label in y_train
            ],
            dtype=np.float32,
        )
    ).reshape(-1, 1)

    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=80,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        logits = model(train)
        loss = (
            torch.nn.functional.binary_cross_entropy_with_logits(
                logits,
                target,
                reduction="none",
            )
            * sample_weight
        ).mean()
        loss = loss + 5e-4 * torch.sum(model.weight * model.weight)
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        return model(test).reshape(-1).cpu().numpy().astype(np.float64)


def exact_two_sided_sign_p(
    positive: int,
    negative: int,
) -> float:
    n = int(positive) + int(negative)
    if n <= 0:
        return 1.0
    k = min(int(positive), int(negative))
    tail = sum(math.comb(n, index) for index in range(k + 1))
    return float(min(1.0, 2.0 * tail / (2.0**n)))


def paired_order_metrics(
    y: np.ndarray,
    score: np.ndarray,
    groups: np.ndarray,
) -> Dict[str, Any]:
    y = np.asarray(y, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    groups = np.asarray(groups)
    positive = 0
    negative = 0
    ties = 0
    margins: List[float] = []
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        if indices.size != 2:
            raise RuntimeError("paired metric requires two rows per group")
        neg = indices[y[indices] == 0]
        pos = indices[y[indices] == 1]
        if neg.size != 1 or pos.size != 1:
            raise RuntimeError("paired metric requires one row per class")
        margin = float(score[pos[0]] - score[neg[0]])
        margins.append(margin)
        if margin > 0:
            positive += 1
        elif margin < 0:
            negative += 1
        else:
            ties += 1
    total = positive + negative + ties
    paired_accuracy = (
        positive + 0.5 * ties
    ) / max(1, total)
    return {
        "paired_accuracy": float(paired_accuracy),
        "paired_positive": positive,
        "paired_negative": negative,
        "paired_ties": ties,
        "paired_margin_mean": float(np.mean(margins)),
        "paired_margin_median": float(np.median(margins)),
        "paired_sign_p": exact_two_sided_sign_p(positive, negative),
    }


def bootstrap_metrics(
    y: np.ndarray,
    score: np.ndarray,
    groups: np.ndarray,
    *,
    iterations: int,
    seed: int,
) -> Dict[str, Tuple[float, float]]:
    """Seed-group bootstrap for paired binary observations."""
    unique = np.unique(groups)
    inverse = np.searchsorted(unique, groups)
    for group in unique:
        labels = y[groups == group]
        if labels.size != 2 or set(labels.tolist()) != {0, 1}:
            raise RuntimeError(
                "bootstrap requires one paired negative/positive per seed"
            )
    rng = np.random.default_rng(int(seed))
    counts = rng.multinomial(
        unique.size,
        np.full(unique.size, 1.0 / unique.size),
        size=int(iterations),
    ).astype(np.float64)
    row_weights = counts[:, inverse]
    pred = score >= 0.5
    correct = (pred == y).astype(np.float64)
    balanced = (
        np.sum(row_weights * correct[None, :], axis=1)
        / (2.0 * unique.size)
    )

    negative_score = np.asarray(
        [score[(groups == group) & (y == 0)][0] for group in unique]
    )
    positive_score = np.asarray(
        [score[(groups == group) & (y == 1)][0] for group in unique]
    )
    difference = positive_score[:, None] - negative_score[None, :]
    credit = (difference > 0).astype(np.float64) + 0.5 * (difference == 0)
    auc = (
        np.einsum("bi,ij,bj->b", counts, credit, counts, optimize=True)
        / float(unique.size * unique.size)
    )
    margin_credit = (
        (positive_score > negative_score).astype(np.float64)
        + 0.5 * (positive_score == negative_score)
    )
    paired = counts @ margin_credit / float(unique.size)

    return {
        "roc_auc": (
            float(np.quantile(auc, 0.025)),
            float(np.quantile(auc, 0.975)),
        ),
        "balanced_accuracy": (
            float(np.quantile(balanced, 0.025)),
            float(np.quantile(balanced, 0.975)),
        ),
        "paired_accuracy": (
            float(np.quantile(paired, 0.025)),
            float(np.quantile(paired, 0.975)),
        ),
    }


def make_index(data: Dict[str, np.ndarray]) -> Dict[Tuple[int, str, int], int]:
    mapping: Dict[Tuple[int, str, int], int] = {}
    for index in range(data["state"].shape[0]):
        key = (
            int(data["visible_seed"][index]),
            str(data["track_name"][index]),
            int(data["evaluation_steps"][index]),
        )
        if key in mapping:
            raise RuntimeError(f"duplicate trajectory key: {key}")
        mapping[key] = index
    return mapping


def trajectory(
    data: Dict[str, np.ndarray],
    mapping: Dict[Tuple[int, str, int], int],
    *,
    seed: int,
    track: str,
    max_step: int,
) -> Dict[str, np.ndarray]:
    indices = [
        mapping[(int(seed), str(track), step)]
        for step in range(int(max_step) + 1)
    ]
    return {
        name: data[name][indices]
        for name in (
            "state",
            "bead_xy",
            "bead_velocity_xy",
            "robot_proxy",
        )
    }


def build_feature_specs(
    *,
    th: int,
    action_dim: int,
    physics_dt: float,
) -> List[FeatureSpec]:
    from ccda_phase3.observation_contract import (
        build_causal_fd_history,
        build_model_x_v2,
        build_position_proprio_history,
        build_position_proprio_state,
        build_privileged_model_x,
        build_xy_history,
    )

    specs: List[FeatureSpec] = [
        FeatureSpec(
            "xy_single",
            True,
            "position",
            lambda tr, step, seed, track: tr["bead_xy"][step].reshape(-1),
        ),
        FeatureSpec(
            "sim_velocity_only",
            False,
            "privileged_velocity",
            lambda tr, step, seed, track: tr[
                "bead_velocity_xy"
            ][step].reshape(-1),
        ),
        FeatureSpec(
            "xy_sim_velocity",
            False,
            "privileged_velocity",
            lambda tr, step, seed, track: np.concatenate(
                [
                    tr["bead_xy"][step].reshape(-1),
                    tr["bead_velocity_xy"][step].reshape(-1),
                ]
            ),
        ),
        FeatureSpec(
            "robot_only",
            True,
            "robot_proprio",
            lambda tr, step, seed, track: tr["robot_proxy"][step].reshape(-1),
        ),
        FeatureSpec(
            "position_proprio_single",
            True,
            "position_proprio",
            lambda tr, step, seed, track: build_position_proprio_state(
                tr["bead_xy"][step],
                tr["robot_proxy"][step],
            ),
        ),
    ]

    for stride in (1, 5, 10):
        specs.extend(
            [
                FeatureSpec(
                    f"xy_history_s{stride}",
                    True,
                    "observable_motion",
                    lambda tr, step, seed, track, stride=stride: build_xy_history(
                        tr["bead_xy"], step=step, th=th, stride=stride
                    ),
                ),
                FeatureSpec(
                    f"position_proprio_history_s{stride}",
                    True,
                    "observable_motion",
                    lambda tr, step, seed, track, stride=stride: build_position_proprio_history(
                        tr["bead_xy"],
                        tr["robot_proxy"],
                        step=step,
                        th=th,
                        stride=stride,
                    ),
                ),
                FeatureSpec(
                    f"causal_fd_history_s{stride}",
                    True,
                    "observable_motion",
                    lambda tr, step, seed, track, stride=stride: build_causal_fd_history(
                        tr["bead_xy"],
                        tr["robot_proxy"],
                        step=step,
                        th=th,
                        stride=stride,
                        physics_dt=physics_dt,
                    ),
                ),
                FeatureSpec(
                    f"model_x_v2_s{stride}",
                    True,
                    "proposed_model_input",
                    lambda tr, step, seed, track, stride=stride: build_model_x_v2(
                        tr["bead_xy"],
                        tr["robot_proxy"],
                        step=step,
                        th=th,
                        stride=stride,
                        action_dim=action_dim,
                    ),
                ),
                FeatureSpec(
                    f"privileged_model_x_v1_s{stride}",
                    False,
                    "privileged_velocity",
                    lambda tr, step, seed, track, stride=stride: build_privileged_model_x(
                        tr["state"],
                        step=step,
                        th=th,
                        stride=stride,
                        action_dim=action_dim,
                    ),
                ),
            ]
        )

    return specs


def comparison_matrix(
    data: Dict[str, np.ndarray],
    mapping: Dict[Tuple[int, str, int], int],
    spec: FeatureSpec,
    *,
    left: str,
    right: str,
    step: int,
    max_step: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    seeds = np.unique(data["visible_seed"].astype(np.int64))
    rows: List[np.ndarray] = []
    labels: List[int] = []
    groups: List[int] = []
    for seed in seeds:
        for label, track in ((0, left), (1, right)):
            tr = trajectory(
                data,
                mapping,
                seed=int(seed),
                track=track,
                max_step=max_step,
            )
            value = np.asarray(
                spec.builder(tr, int(step), int(seed), track),
                dtype=np.float32,
            ).reshape(-1)
            if not np.all(np.isfinite(value)):
                raise RuntimeError(
                    f"non-finite feature {spec.name} seed={seed} track={track}"
                )
            rows.append(value)
            labels.append(label)
            groups.append(int(seed))
    return (
        np.stack(rows).astype(np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(groups, dtype=np.int64),
    )


def cross_validated_scores(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    folds: int,
    repeats: int,
    seed_base: int,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    repeat_scores: List[np.ndarray] = []
    fold_rows: List[Dict[str, Any]] = []
    for repeat in range(int(repeats)):
        scores = np.empty(y.size, dtype=np.float64)
        for fold, (train, test) in enumerate(
            grouped_fold_indices(groups, folds, repeat)
        ):
            assert_disjoint_groups(groups[train], groups[test])
            logits = fit_linear_logits(
                x[train],
                y[train],
                x[test],
                seed=seed_base + repeat * 100 + fold,
            )
            scores[test] = rank01(logits)
            result = classification_metrics(y[test], scores[test])
            paired = paired_order_metrics(
                y[test],
                scores[test],
                groups[test],
            )
            fold_rows.append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "train_groups": int(np.unique(groups[train]).size),
                    "test_groups": int(np.unique(groups[test]).size),
                    **result,
                    **paired,
                }
            )
        repeat_scores.append(scores)
    return np.mean(np.stack(repeat_scores), axis=0), fold_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument(
        "--data",
        default="data/phase3_12d_r24_no_action/trajectory_observations.npz",
    )
    parser.add_argument(
        "--primary-steps",
        nargs="+",
        type=int,
        default=[20, 60, 120, 240],
    )
    parser.add_argument("--max-step", type=int, default=240)
    parser.add_argument("--th", type=int, default=3)
    parser.add_argument("--action-dim", type=int, default=14)
    parser.add_argument("--physics-hz", type=float, default=480.0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--bootstraps", type=int, default=10000)
    parser.add_argument(
        "--out-prefix",
        default="reports/phase3_12d_r24",
    )
    args = parser.parse_args()

    torch.set_num_threads(1)
    root = Path(args.root).resolve()
    for path in (root, root / "scripts"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    with np.load(root / args.data, allow_pickle=False) as loaded:
        data = {key: loaded[key] for key in loaded.files}

    required = {
        "state",
        "bead_xy",
        "bead_velocity_xy",
        "robot_proxy",
        "visible_seed",
        "evaluation_steps",
        "track_name",
    }
    missing = required.difference(data)
    if missing:
        raise SystemExit(f"missing trajectory arrays: {sorted(missing)}")
    for block in ("state", "bead_xy", "bead_velocity_xy", "robot_proxy"):
        if not np.all(np.isfinite(data[block])):
            raise SystemExit(f"non-finite {block}")

    seeds = np.unique(data["visible_seed"].astype(np.int64))
    tracks = sorted(set(data["track_name"].astype(str).tolist()))
    steps = np.unique(data["evaluation_steps"].astype(np.int64))
    expected_tracks = {
        "free_a",
        "free_b",
        "hidden_v2_a",
        "hidden_v2_b",
    }
    if set(tracks) != expected_tracks:
        raise SystemExit(f"unexpected tracks: {tracks}")
    if not np.array_equal(steps, np.arange(args.max_step + 1)):
        raise SystemExit(f"unexpected physics steps: {steps.tolist()}")
    if seeds.size < 100:
        raise SystemExit(f"too few independent seeds: {seeds.size}")

    mapping = make_index(data)
    expected_rows = seeds.size * len(tracks) * (args.max_step + 1)
    if len(mapping) != expected_rows:
        raise SystemExit(
            f"trajectory key count mismatch: {len(mapping)} != {expected_rows}"
        )

    specs = build_feature_specs(
        th=args.th,
        action_dim=args.action_dim,
        physics_dt=1.0 / float(args.physics_hz),
    )
    primary_steps = sorted(set(int(value) for value in args.primary_steps))
    if not primary_steps or primary_steps[-1] > int(args.max_step):
        raise SystemExit(
            f"invalid primary steps {primary_steps} for max_step={args.max_step}"
        )
    decision_step = max(primary_steps)
    comparisons = [
        (f"primary_step_{step}", "free_a", "hidden_v2_a", step)
        for step in primary_steps
    ]
    comparisons += [
        ("control_free_replicate", "free_a", "free_b", decision_step),
        (
            "control_hidden_v2_replicate",
            "hidden_v2_a",
            "hidden_v2_b",
            decision_step,
        ),
        ("control_horizon0", "free_a", "hidden_v2_a", 0),
    ]

    summary_rows: List[Dict[str, Any]] = []
    fold_rows_all: List[Dict[str, Any]] = []
    prediction_rows: List[Dict[str, Any]] = []

    for comparison, left, right, step in comparisons:
        for spec_index, spec in enumerate(specs):
            x, y, groups = comparison_matrix(
                data,
                mapping,
                spec,
                left=left,
                right=right,
                step=step,
                max_step=args.max_step,
            )
            score, fold_rows = cross_validated_scores(
                x,
                y,
                groups,
                folds=args.folds,
                repeats=args.repeats,
                seed_base=323000 + spec_index * 1000 + step * 10,
            )
            result = classification_metrics(y, score)
            paired = paired_order_metrics(y, score, groups)
            ci = bootstrap_metrics(
                y,
                score,
                groups,
                iterations=args.bootstraps,
                seed=333000 + spec_index * 100 + step,
            )
            row = {
                "comparison": comparison,
                "left_track": left,
                "right_track": right,
                "step": step,
                "feature": spec.name,
                "feature_family": spec.family,
                "legal_observation": bool(spec.legal_observation),
                "feature_dim": int(x.shape[1]),
                **result,
                **paired,
                "roc_auc_ci_low": ci["roc_auc"][0],
                "roc_auc_ci_high": ci["roc_auc"][1],
                "balanced_accuracy_ci_low": ci[
                    "balanced_accuracy"
                ][0],
                "balanced_accuracy_ci_high": ci[
                    "balanced_accuracy"
                ][1],
                "paired_accuracy_ci_low": ci["paired_accuracy"][0],
                "paired_accuracy_ci_high": ci["paired_accuracy"][1],
            }
            summary_rows.append(row)
            for fold_row in fold_rows:
                fold_rows_all.append(
                    {
                        "comparison": comparison,
                        "step": step,
                        "feature": spec.name,
                        **fold_row,
                    }
                )
            for index in range(y.size):
                prediction_rows.append(
                    {
                        "comparison": comparison,
                        "step": step,
                        "feature": spec.name,
                        "visible_seed": int(groups[index]),
                        "label": int(y[index]),
                        "score": float(score[index]),
                    }
                )
            print(
                f"[r2.4] {comparison} step={step} {spec.name} "
                f"auc={result['roc_auc']:.4f} "
                f"paired={paired['paired_accuracy']:.4f}",
                flush=True,
            )

    controls = [
        row for row in summary_rows if row["comparison"].startswith("control_")
    ]
    control_failures = [
        row
        for row in controls
        if (
            row["roc_auc_ci_low"] > 0.60
            or row["paired_accuracy_ci_low"] > 0.60
        )
    ]
    primary = [
        row
        for row in summary_rows
        if row["comparison"] == f"primary_step_{decision_step}"
    ]
    all_primary = [
        row for row in summary_rows
        if row["comparison"].startswith("primary_step_")
    ]

    def supported(row: Dict[str, Any]) -> bool:
        return bool(
            row["roc_auc"] >= 0.70
            and row["roc_auc_ci_low"] > 0.65
            and row["balanced_accuracy"] >= 0.65
            and row["paired_accuracy"] >= 0.65
            and row["paired_sign_p"] < 0.01
        )

    supported_leaks = [row for row in all_primary if supported(row)]
    hiddenness_pass = bool(all_primary) and all(
        row["roc_auc"] <= 0.60
        and row["roc_auc_ci_high"] < 0.65
        and row["balanced_accuracy"] <= 0.60
        and row["paired_accuracy"] <= 0.60
        for row in all_primary
    )

    if control_failures:
        verdict = "FAIL"
        root_cause = "phase312d_r24_collection_or_statistics_invalid"
    elif supported_leaks or not hiddenness_pass:
        verdict = "FAIL"
        root_cause = "phase312d_r24_observation_leakage_detected"
    else:
        verdict = "PASS"
        root_cause = "phase312d_r24_no_action_observation_hiddenness_passed"

    from ccda_phase3.observation_contract import schema_dimensions

    dimensions = schema_dimensions(
        n_beads=int(data["bead_xy"].shape[1]),
        th=args.th,
        action_dim=args.action_dim,
    )

    payload = {
        "verdict": verdict,
        "root_cause": root_cause,
        "backend": "torch_linear_bce_lbfgs",
        "num_visible_seeds": int(seeds.size),
        "trajectory_rows": int(data["state"].shape[0]),
        "physics_steps": steps.tolist(),
        "primary_steps": primary_steps,
        "decision_step": decision_step,
        "summary": summary_rows,
        "controls_pass": not bool(control_failures),
        "control_failures": control_failures,
        "supported_leakage_features": [
            {"step": row["step"], "feature": row["feature"]}
            for row in supported_leaks
        ],
        "all_features_below_hiddenness_thresholds": hiddenness_pass,
        "proposed_schema": {
            "name": "ccda_state_v2_position_proprio",
            "contains_simulator_bead_velocity": False,
            "observation_contract": (
                "bead_position_history + robot_proprio + "
                "past_executable_action_history"
            ),
            "dimensions": dimensions,
            "not_activated_by_this_phase": True,
        },
        "scientific_boundary": (
            "This is a slack-breakaway-v2 no-action hiddenness audit. It does not "
            "train StateDiff/IDM, execute the candidate matrix, prove "
            "CPS effectiveness, or authorize Phase4."
        ),
        "next_step": (
            "If PASS, continue only to the r2.4 snapshot and final environment "
            "gate. State-v2 dataset regeneration remains a later phase."
        ),
    }

    prefix = root / args.out_prefix
    strict_json_dump(
        Path(str(prefix) + "_observation_summary.json"),
        payload,
    )
    write_csv(
        Path(str(prefix) + "_observation_features.csv"),
        summary_rows,
        list(summary_rows[0]),
    )
    write_csv(
        Path(str(prefix) + "_observation_folds.csv"),
        fold_rows_all,
        list(fold_rows_all[0]),
    )
    # Predictions are intentionally not persisted; grouped feature and fold
    # summaries are sufficient for the audit and keep the committed surface small.

    lines = [
        "# Phase3.12d-r2.4 No-Action Observation Audit",
        "",
        f"- Verdict: `{verdict}`",
        f"- Root cause: `{root_cause}`",
        f"- Backend: `torch_linear_bce_lbfgs`",
        f"- Visible seeds: `{seeds.size}`",
        f"- Trajectory rows: `{data['state'].shape[0]}`",
        "",
        f"## Decision comparison at step {decision_step}",
        "",
        (
            "| Feature | Legal | Family | Dim | ROC-AUC | 95% CI | "
            "Balanced acc | Paired acc | Sign p |"
        ),
        "|---|---:|---|---:|---:|---|---:|---:|---:|",
    ]
    for row in primary:
        lines.append(
            f"| `{row['feature']}` | `{row['legal_observation']}` | "
            f"`{row['feature_family']}` | {row['feature_dim']} | "
            f"{row['roc_auc']:.6f} | "
            f"[{row['roc_auc_ci_low']:.6f}, "
            f"{row['roc_auc_ci_high']:.6f}] | "
            f"{row['balanced_accuracy']:.6f} | "
            f"{row['paired_accuracy']:.6f} | "
            f"{row['paired_sign_p']:.6g} |"
        )
    lines += [
        "",
        "## Controls",
        "",
        "| Comparison | Feature | AUC | AUC lower CI | Paired lower CI |",
        "|---|---|---:|---:|---:|",
    ]
    for row in controls:
        lines.append(
            f"| `{row['comparison']}` | `{row['feature']}` | "
            f"{row['roc_auc']:.6f} | "
            f"{row['roc_auc_ci_low']:.6f} | "
            f"{row['paired_accuracy_ci_low']:.6f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- `sim_velocity_only` and privileged state features use direct "
        "PyBullet bead velocity and are not a visual observation contract.",
        "- Causal histories use only previously captured bead XY and robot "
        "proprio; no simulator bead velocity is included.",
        "- The proposed 87-D state-v2 schema is specified but not activated.",
        "- No model training, candidate matrix, Phase4, or CPS was run.",
    ]
    Path(str(prefix) + "_observation_report.md").write_text(
        "\n".join(lines) + "\n"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
