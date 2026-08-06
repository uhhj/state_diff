"""Geometry and branch metrics for Phase 0B Soft BlockPush."""
from typing import Optional, Tuple

import numpy as np


def rigid_aligned_rmse(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Return SE(3)-aligned RMSE for two [N, 3] point sets."""
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if reference.shape != candidate.shape or reference.ndim != 2 or reference.shape[1] != 3:
        raise ValueError("point sets must have equal shape [N, 3]")
    ref_center = np.mean(reference, axis=0)
    can_center = np.mean(candidate, axis=0)
    ref_zero = reference - ref_center
    can_zero = candidate - can_center
    u, _, vt = np.linalg.svd(can_zero.T.dot(ref_zero))
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(u.dot(vt)))
    rotation = u.dot(correction).dot(vt)
    aligned = can_zero.dot(rotation) + ref_center
    return float(np.sqrt(np.mean(np.square(reference - aligned))))


def paired_rmse(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return time-wise coordinate RMSE for equal [T, N, 3] arrays."""
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.shape != second.shape or first.ndim != 3:
        raise ValueError("paired arrays must have equal shape [T, N, 3]")
    return np.sqrt(np.mean(np.square(first - second), axis=(1, 2)))


def rigid_aligned_series(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return time-wise rigid-aligned RMSE."""
    if np.asarray(first).shape != np.asarray(second).shape:
        raise ValueError("paired arrays must have equal shape")
    return np.asarray([rigid_aligned_rmse(a, b)
                       for a, b in zip(first, second)], dtype=np.float64)


def edge_strain(positions: np.ndarray, edges: np.ndarray,
                rest_lengths: np.ndarray) -> np.ndarray:
    """Return edge strain for one [N,3] point set."""
    positions = np.asarray(positions, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.int64)
    rest = np.asarray(rest_lengths, dtype=np.float64)
    lengths = np.linalg.norm(
        positions[edges[:, 0]] - positions[edges[:, 1]], axis=1)
    return (lengths - rest) / rest


def first_sustained_onset(values: np.ndarray, threshold: float,
                          eligible: np.ndarray,
                          consecutive: int = 3) -> Optional[int]:
    """Return the first eligible index with K consecutive threshold crossings."""
    values = np.asarray(values, dtype=np.float64)
    eligible = np.asarray(eligible, dtype=bool)
    for index in range(len(values) - consecutive + 1):
        if (np.all(eligible[index:index + consecutive])
                and np.all(values[index:index + consecutive] > threshold)):
            return int(index)
    return None


def standardized_branch_gap(first: np.ndarray, second: np.ndarray,
                            baseline: np.ndarray, std_floor: float = 1e-6
                            ) -> np.ndarray:
    """Return pooled-baseline standardized RMS gap for one formal channel."""
    first = np.asarray(first, dtype=np.float64).reshape(len(first), -1)
    second = np.asarray(second, dtype=np.float64).reshape(len(second), -1)
    baseline = np.asarray(baseline, dtype=bool)
    pooled = np.concatenate([first[baseline], second[baseline]], axis=0)
    mean = np.mean(pooled, axis=0)
    std = np.maximum(np.std(pooled, axis=0), float(std_floor))
    return np.sqrt(np.mean(np.square(
        (first - mean) / std - (second - mean) / std), axis=1))


def onset_from_baseline(gap: np.ndarray, baseline: np.ndarray,
                        sigma_multiplier: float = 5.0,
                        consecutive: int = 3) -> Tuple[float, Optional[int]]:
    """Compute baseline mean+sigma threshold and sustained post-baseline onset."""
    gap = np.asarray(gap, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=bool)
    threshold = float(np.mean(gap[baseline])
                      + sigma_multiplier * np.std(gap[baseline]))
    return threshold, first_sustained_onset(
        gap, threshold, ~baseline, consecutive)
