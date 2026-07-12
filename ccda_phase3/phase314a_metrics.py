"""Deterministic state-v2 metrics for Phase3.14a."""
from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from .phase314a_contract import BEAD_XY_DIM, N_BEADS
from .schema_v2 import DEFAULT_TF, STATE_DIM


def _finite(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or Inf")
    return array


def chamfer_xy(left: np.ndarray, right: np.ndarray) -> float:
    a = _finite(left, "left").reshape(-1, 2)
    b = _finite(right, "right").reshape(-1, 2)
    if a.shape[0] == 0 or b.shape[0] == 0:
        raise ValueError("Chamfer requires non-empty point sets")
    distance = np.linalg.norm(
        a[:, None, :] - b[None, :, :],
        axis=-1,
    )
    return float(
        0.5
        * (
            np.min(distance, axis=1).mean()
            + np.min(distance, axis=0).mean()
        )
    )


def final_valid_indices(valid_mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(valid_mask, dtype=np.bool_)
    if mask.ndim != 2 or mask.shape[1] != DEFAULT_TF:
        raise ValueError("future mask must be [N,4]")
    counts = np.sum(mask, axis=1)
    if np.any(counts <= 0):
        raise ValueError("every row must contain a valid future state")
    expected = np.arange(DEFAULT_TF)[None, :] < counts[:, None]
    if not np.array_equal(mask, expected):
        raise ValueError("future masks must be contiguous from horizon zero")
    return counts.astype(np.int64) - 1


def final_valid_states(
    future: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    value = _finite(future, "future")
    if value.ndim != 3 or value.shape[1:] != (DEFAULT_TF, STATE_DIM):
        raise ValueError("future must be [N,4,87]")
    final = final_valid_indices(valid_mask)
    return value[np.arange(value.shape[0]), final]


def masked_mse(
    prediction: np.ndarray,
    target: np.ndarray,
    valid_mask: np.ndarray,
    *,
    state_slice: slice = slice(None),
) -> float:
    pred = _finite(prediction, "prediction")[..., state_slice]
    truth = _finite(target, "target")[..., state_slice]
    if pred.shape != truth.shape or pred.ndim != 3:
        raise ValueError("prediction and target must share [N,T,D]")
    mask = np.asarray(valid_mask, dtype=np.bool_)
    if mask.shape != pred.shape[:2]:
        raise ValueError("mask shape mismatch")
    expanded = np.broadcast_to(mask[:, :, None], pred.shape)
    if not np.any(expanded):
        raise ValueError("mask contains no valid elements")
    error = pred - truth
    return float(np.mean(error[expanded] ** 2))


def final_chamfer_rows(
    prediction: np.ndarray,
    target: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    pred_final = final_valid_states(prediction, valid_mask)
    true_final = final_valid_states(target, valid_mask)
    values = np.asarray(
        [
            chamfer_xy(
                left[:BEAD_XY_DIM].reshape(N_BEADS, 2),
                right[:BEAD_XY_DIM].reshape(N_BEADS, 2),
            )
            for left, right in zip(pred_final, true_final)
        ],
        dtype=np.float64,
    )
    return values


def trajectory_chamfer_rows(
    prediction: np.ndarray,
    target: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    pred = _finite(prediction, "prediction")
    truth = _finite(target, "target")
    mask = np.asarray(valid_mask, dtype=np.bool_)
    if pred.shape != truth.shape or pred.shape[1:] != (
        DEFAULT_TF,
        STATE_DIM,
    ):
        raise ValueError("future shape mismatch")
    values = []
    for row in range(pred.shape[0]):
        row_values = []
        for horizon in range(DEFAULT_TF):
            if not mask[row, horizon]:
                continue
            row_values.append(
                chamfer_xy(
                    pred[row, horizon, :BEAD_XY_DIM].reshape(
                        N_BEADS,
                        2,
                    ),
                    truth[row, horizon, :BEAD_XY_DIM].reshape(
                        N_BEADS,
                        2,
                    ),
                )
            )
        values.append(float(np.mean(row_values)))
    return np.asarray(values, dtype=np.float64)


def group_bootstrap_mean_ci(
    values: Sequence[float],
    groups: Sequence[int],
    *,
    iterations: int = 10000,
    seed: int = 314000,
) -> Dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    group_array = np.asarray(groups)
    if data.ndim != 1 or data.shape[0] != group_array.shape[0]:
        raise ValueError("values/groups length mismatch")
    if not np.all(np.isfinite(data)):
        raise ValueError("bootstrap values are non-finite")
    unique = np.unique(group_array)
    if unique.size < 2:
        raise ValueError("at least two independent groups required")
    group_sums = np.asarray(
        [np.sum(data[group_array == group]) for group in unique],
        dtype=np.float64,
    )
    group_counts = np.asarray(
        [np.sum(group_array == group) for group in unique],
        dtype=np.float64,
    )
    rng = np.random.default_rng(int(seed))
    draws = rng.integers(
        0,
        unique.size,
        size=(int(iterations), unique.size),
        endpoint=False,
    )
    estimates = np.sum(group_sums[draws], axis=1) / np.sum(
        group_counts[draws],
        axis=1,
    )
    return {
        "mean": float(np.mean(data)),
        "ci_low": float(np.quantile(estimates, 0.025)),
        "ci_high": float(np.quantile(estimates, 0.975)),
        "groups": int(unique.size),
    }


def evaluate_future_prediction(
    prediction: np.ndarray,
    target: np.ndarray,
    valid_mask: np.ndarray,
    *,
    visible_seed: np.ndarray,
    condition_name: np.ndarray,
    pre_engagement: np.ndarray,
    bootstrap_iterations: int = 10000,
    bootstrap_seed: int = 314000,
) -> Dict[str, Any]:
    pred = _finite(prediction, "prediction")
    truth = _finite(target, "target")
    mask = np.asarray(valid_mask, dtype=np.bool_)
    seed_values = np.asarray(visible_seed, dtype=np.int64)
    conditions = np.asarray(condition_name).astype(str)
    pre = np.asarray(pre_engagement, dtype=np.bool_)
    rows = pred.shape[0]
    if any(
        value.shape[0] != rows
        for value in (truth, mask, seed_values, conditions, pre)
    ):
        raise ValueError("evaluation row count mismatch")

    final_rows = final_chamfer_rows(pred, truth, mask)
    trajectory_rows = trajectory_chamfer_rows(pred, truth, mask)

    subsets = {
        "all": np.ones(rows, dtype=np.bool_),
        "pre_engagement": pre,
        "free": conditions == "free",
        "hidden_v2": (
            conditions == "hidden_slack_breakaway_pin_v2"
        ),
        "pre_engagement_free": pre & (conditions == "free"),
        "pre_engagement_hidden_v2": pre
        & (conditions == "hidden_slack_breakaway_pin_v2"),
    }

    subset_metrics: Dict[str, Any] = {}
    for offset, (name, selected) in enumerate(subsets.items()):
        if np.sum(selected) == 0:
            raise ValueError(f"evaluation subset {name} is empty")
        subset_metrics[name] = {
            "rows": int(np.sum(selected)),
            "visible_seeds": int(
                np.unique(seed_values[selected]).size
            ),
            "full_state_mse": masked_mse(
                pred[selected],
                truth[selected],
                mask[selected],
            ),
            "bead_xy_mse": masked_mse(
                pred[selected],
                truth[selected],
                mask[selected],
                state_slice=slice(0, BEAD_XY_DIM),
            ),
            "robot_proxy_mse": masked_mse(
                pred[selected],
                truth[selected],
                mask[selected],
                state_slice=slice(BEAD_XY_DIM, STATE_DIM),
            ),
            "final_xy_chamfer": group_bootstrap_mean_ci(
                final_rows[selected],
                seed_values[selected],
                iterations=bootstrap_iterations,
                seed=bootstrap_seed + offset,
            ),
            "trajectory_xy_chamfer": group_bootstrap_mean_ci(
                trajectory_rows[selected],
                seed_values[selected],
                iterations=bootstrap_iterations,
                seed=bootstrap_seed + 100 + offset,
            ),
        }
    return {
        "subsets": subset_metrics,
        "final_chamfer_rows": final_rows,
        "trajectory_chamfer_rows": trajectory_rows,
    }
