"""Offline DDPM and paired-branch support metrics for Phase3.14b."""
from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from ccda_phase3.phase314a_contract import BEAD_XY_DIM, N_BEADS
from ccda_phase3.phase314a_metrics import (
    chamfer_xy,
    group_bootstrap_mean_ci,
)
from ccda_phase3.phase314b_contract import (
    BRANCH_SEPARATION_MIN_METERS,
    BRANCH_SUPPORT_RADIUS_FRACTION,
    K_VALUES,
)
from ccda_phase3.schema_v2 import DEFAULT_TF, STATE_DIM


def _finite(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or Inf")
    return array


def final_xy(states: np.ndarray) -> np.ndarray:
    value = _finite(states, "states")
    if value.shape[-1] != STATE_DIM:
        raise ValueError("state dimension mismatch")
    return value[..., :BEAD_XY_DIM].reshape(
        value.shape[:-1] + (N_BEADS, 2)
    )


def candidate_final_chamfer(
    sample_pool: np.ndarray,
    target_future: np.ndarray,
) -> np.ndarray:
    """Return [K,N] final-state Chamfer distances."""
    samples = _finite(sample_pool, "sample_pool")
    target = _finite(target_future, "target_future")
    if samples.ndim != 4 or samples.shape[2:] != (
        DEFAULT_TF,
        STATE_DIM,
    ):
        raise ValueError("sample_pool must be [K,N,4,87]")
    if target.shape != samples.shape[1:]:
        raise ValueError("target future shape mismatch")
    sample_final = final_xy(samples[:, :, -1, :])
    target_final = final_xy(target[:, -1, :])
    output = np.empty(
        (samples.shape[0], samples.shape[1]),
        dtype=np.float64,
    )
    for sample_index in range(samples.shape[0]):
        for row in range(samples.shape[1]):
            output[sample_index, row] = chamfer_xy(
                sample_final[sample_index, row],
                target_final[row],
            )
    return output


def nested_best_of_k(
    distances: np.ndarray,
    k_values: Sequence[int] = K_VALUES,
) -> Dict[int, np.ndarray]:
    value = np.asarray(distances, dtype=np.float64)
    if value.ndim != 2:
        raise ValueError("distances must be [K,N]")
    result: Dict[int, np.ndarray] = {}
    for raw_k in k_values:
        k = int(raw_k)
        if k <= 0 or k > value.shape[0]:
            raise ValueError(f"invalid nested K={k}")
        result[k] = np.min(value[:k], axis=0)
    return result


def sample_mean_future(sample_pool: np.ndarray) -> np.ndarray:
    value = _finite(sample_pool, "sample_pool")
    if value.ndim != 4:
        raise ValueError("sample pool must be [K,N,T,D]")
    return np.mean(value, axis=0, dtype=np.float64).astype(np.float32)


def pairwise_pool_diversity(sample_pool: np.ndarray) -> np.ndarray:
    """Mean pairwise final Chamfer per query, shape [N]."""
    value = _finite(sample_pool, "sample_pool")
    if value.ndim != 4:
        raise ValueError("sample pool must be [K,N,T,D]")
    final = final_xy(value[:, :, -1, :])
    query_values = np.zeros(value.shape[1], dtype=np.float64)
    for row in range(value.shape[1]):
        distances = []
        for left in range(value.shape[0]):
            for right in range(left + 1, value.shape[0]):
                distances.append(
                    chamfer_xy(final[left, row], final[right, row])
                )
        query_values[row] = (
            float(np.mean(distances)) if distances else 0.0
        )
    return query_values


def branch_support_for_pair(
    samples: np.ndarray,
    *,
    free_final: np.ndarray,
    hidden_final: np.ndarray,
) -> Dict[str, Any]:
    pool = _finite(samples, "samples")
    if pool.ndim != 2 or pool.shape[1] != STATE_DIM:
        raise ValueError("samples must be [K,87]")
    free_xy = final_xy(free_final)
    hidden_xy = final_xy(hidden_final)
    separation = chamfer_xy(free_xy, hidden_xy)
    if separation < BRANCH_SEPARATION_MIN_METERS:
        return {
            "eligible": False,
            "branch_separation": float(separation),
        }

    sample_xy = final_xy(pool)
    to_free = np.asarray(
        [chamfer_xy(item, free_xy) for item in sample_xy],
        dtype=np.float64,
    )
    to_hidden = np.asarray(
        [chamfer_xy(item, hidden_xy) for item in sample_xy],
        dtype=np.float64,
    )
    radius = BRANCH_SUPPORT_RADIUS_FRACTION * separation
    return {
        "eligible": True,
        "branch_separation": float(separation),
        "support_radius": float(radius),
        "min_to_free": float(np.min(to_free)),
        "min_to_hidden": float(np.min(to_hidden)),
        "free_supported": bool(np.min(to_free) <= radius),
        "hidden_supported": bool(np.min(to_hidden) <= radius),
        "both_supported": bool(
            np.min(to_free) <= radius
            and np.min(to_hidden) <= radius
        ),
        "free_vote_rate": float(np.mean(to_free < to_hidden)),
        "hidden_vote_rate": float(np.mean(to_hidden < to_free)),
        "tie_rate": float(np.mean(to_free == to_hidden)),
    }


def fit_segment_bounds(
    train_future: np.ndarray,
    *,
    lower_percentile: float = 0.5,
    upper_percentile: float = 99.5,
    margin_fraction: float = 0.10,
) -> Tuple[np.ndarray, np.ndarray]:
    future = _finite(train_future, "train_future")
    if future.ndim != 3 or future.shape[1:] != (
        DEFAULT_TF,
        STATE_DIM,
    ):
        raise ValueError("train_future must be [N,4,87]")
    xy = final_xy(future[:, -1, :])
    segments = np.linalg.norm(
        xy[..., 1:, :] - xy[..., :-1, :],
        axis=-1,
    )
    lower = np.percentile(
        segments,
        float(lower_percentile),
        axis=0,
    )
    upper = np.percentile(
        segments,
        float(upper_percentile),
        axis=0,
    )
    width = np.maximum(upper - lower, 1e-6)
    return (
        (lower - float(margin_fraction) * width).astype(np.float32),
        (upper + float(margin_fraction) * width).astype(np.float32),
    )


def candidate_physical_validity(
    sample_pool: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> Dict[str, Any]:
    samples = _finite(sample_pool, "sample_pool")
    final = final_xy(samples[:, :, -1, :])
    segments = np.linalg.norm(
        final[..., 1:, :] - final[..., :-1, :],
        axis=-1,
    )
    lower_value = np.asarray(lower, dtype=np.float32)
    upper_value = np.asarray(upper, dtype=np.float32)
    if lower_value.shape != (N_BEADS - 1,):
        raise ValueError("segment lower bound shape mismatch")
    valid = np.all(
        (segments >= lower_value[None, None, :])
        & (segments <= upper_value[None, None, :]),
        axis=-1,
    )
    return {
        "sample_validity_rate": float(np.mean(valid)),
        "query_has_valid_candidate_rate": float(
            np.mean(np.any(valid, axis=0))
        ),
        "valid_mask": valid,
        "max_segment_length": float(np.max(segments)),
        "min_segment_length": float(np.min(segments)),
    }


def improvement_bootstrap(
    reference_error: np.ndarray,
    candidate_error: np.ndarray,
    visible_seed: np.ndarray,
    *,
    iterations: int,
    seed: int,
) -> Dict[str, float]:
    reference = np.asarray(reference_error, dtype=np.float64)
    candidate = np.asarray(candidate_error, dtype=np.float64)
    if reference.shape != candidate.shape:
        raise ValueError("reference/candidate error shape mismatch")
    improvement = reference - candidate
    return group_bootstrap_mean_ci(
        improvement,
        np.asarray(visible_seed, dtype=np.int64),
        iterations=int(iterations),
        seed=int(seed),
    )
