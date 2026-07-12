"""Phase3.14b-r2 stability, physical-validity, and candidate metrics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from ccda_phase3.phase314a_contract import BEAD_XY_DIM, N_BEADS
from ccda_phase3.phase314a_metrics import (
    chamfer_xy,
    group_bootstrap_mean_ci,
)
from ccda_phase3.phase314b_metrics import (
    branch_support_for_pair,
    candidate_final_chamfer,
    nested_best_of_k,
    pairwise_pool_diversity,
    sample_mean_future,
)
from ccda_phase3.phase314b_r2_contract import (
    K_VALUES,
    MAX_RUN_Z_ABS_MAX,
    MAX_RUN_Z_ABS_P99,
    MIN_RUN_PHYSICAL_VALIDITY,
    MIN_RUN_QUERY_VALIDITY,
    PARTIAL_T10_LIMIT,
    PARTIAL_T99_LIMIT,
    VAL_BEST8_LIMIT,
    VAL_K1_LIMIT,
)
from ccda_phase3.schema_v2 import DEFAULT_TF, STATE_DIM


@dataclass(frozen=True)
class ValidityContract:
    coordinate_lower: np.ndarray
    coordinate_upper: np.ndarray
    segment_lower: np.ndarray
    segment_upper: np.ndarray
    quaternion_norm_lower: float
    quaternion_norm_upper: float

    def to_json(self) -> Dict[str, Any]:
        return {
            "coordinate_lower": self.coordinate_lower.tolist(),
            "coordinate_upper": self.coordinate_upper.tolist(),
            "segment_lower": self.segment_lower.tolist(),
            "segment_upper": self.segment_upper.tolist(),
            "quaternion_norm_lower": self.quaternion_norm_lower,
            "quaternion_norm_upper": self.quaternion_norm_upper,
        }


def _finite(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or Inf")
    return array


def fit_validity_contract(
    train_future: np.ndarray,
) -> ValidityContract:
    future = _finite(train_future, "train_future")
    if future.ndim != 3 or future.shape[1:] != (
        DEFAULT_TF,
        STATE_DIM,
    ):
        raise ValueError("train_future must be [N,4,87]")
    xy = future[..., :BEAD_XY_DIM].reshape(
        future.shape[0],
        DEFAULT_TF,
        N_BEADS,
        2,
    )
    flattened_xy = xy.reshape(-1, 2)
    coordinate_low = np.percentile(
        flattened_xy,
        0.1,
        axis=0,
    )
    coordinate_high = np.percentile(
        flattened_xy,
        99.9,
        axis=0,
    )
    coordinate_width = np.maximum(
        coordinate_high - coordinate_low,
        1e-6,
    )
    coordinate_margin = np.maximum(
        0.02,
        0.20 * coordinate_width,
    )

    segments = np.linalg.norm(
        xy[..., 1:, :] - xy[..., :-1, :],
        axis=-1,
    )
    segment_low = np.percentile(
        segments,
        0.5,
        axis=(0, 1),
    )
    segment_high = np.percentile(
        segments,
        99.5,
        axis=(0, 1),
    )
    segment_width = np.maximum(
        segment_high - segment_low,
        1e-6,
    )
    segment_margin = np.maximum(
        0.001,
        0.20 * segment_width,
    )
    return ValidityContract(
        coordinate_lower=(
            coordinate_low - coordinate_margin
        ).astype(np.float32),
        coordinate_upper=(
            coordinate_high + coordinate_margin
        ).astype(np.float32),
        segment_lower=np.maximum(
            0.0,
            segment_low - segment_margin,
        ).astype(np.float32),
        segment_upper=(
            segment_high + segment_margin
        ).astype(np.float32),
        quaternion_norm_lower=0.90,
        quaternion_norm_upper=1.10,
    )


def sample_validity(
    sample_pool: np.ndarray,
    contract: ValidityContract,
) -> Dict[str, Any]:
    samples = _finite(sample_pool, "sample_pool")
    if samples.ndim != 4 or samples.shape[2:] != (
        DEFAULT_TF,
        STATE_DIM,
    ):
        raise ValueError("sample_pool must be [K,N,4,87]")
    xy = samples[..., :BEAD_XY_DIM].reshape(
        samples.shape[0],
        samples.shape[1],
        DEFAULT_TF,
        N_BEADS,
        2,
    )
    coordinate_valid = np.all(
        (xy >= contract.coordinate_lower[None, None, None, None, :])
        & (xy <= contract.coordinate_upper[None, None, None, None, :]),
        axis=(2, 3, 4),
    )
    segments = np.linalg.norm(
        xy[..., 1:, :] - xy[..., :-1, :],
        axis=-1,
    )
    segment_valid = np.all(
        (
            segments
            >= contract.segment_lower[None, None, None, :]
        )
        & (
            segments
            <= contract.segment_upper[None, None, None, :]
        ),
        axis=(2, 3),
    )
    quaternion = samples[..., 83:87]
    quaternion_norm = np.linalg.norm(quaternion, axis=-1)
    quaternion_valid = np.all(
        (
            quaternion_norm
            >= float(contract.quaternion_norm_lower)
        )
        & (
            quaternion_norm
            <= float(contract.quaternion_norm_upper)
        ),
        axis=2,
    )
    valid = coordinate_valid & segment_valid & quaternion_valid
    return {
        "sample_valid_mask": valid,
        "sample_validity_rate": float(np.mean(valid)),
        "query_has_valid_candidate_rate": float(
            np.mean(np.any(valid, axis=0))
        ),
        "coordinate_validity_rate": float(
            np.mean(coordinate_valid)
        ),
        "segment_validity_rate": float(np.mean(segment_valid)),
        "quaternion_validity_rate": float(
            np.mean(quaternion_valid)
        ),
        "raw_xy_abs_max": float(np.max(np.abs(xy))),
        "segment_min": float(np.min(segments)),
        "segment_max": float(np.max(segments)),
        "quaternion_norm_min": float(
            np.min(quaternion_norm)
        ),
        "quaternion_norm_max": float(
            np.max(quaternion_norm)
        ),
    }


def z_pool_stats(
    sample_pool_z: np.ndarray,
    active_mask: np.ndarray,
) -> Dict[str, float]:
    samples = _finite(sample_pool_z, "sample_pool_z")
    active = np.asarray(active_mask, dtype=np.bool_)
    if active.shape != (DEFAULT_TF, STATE_DIM):
        raise ValueError("active mask must be [4,87]")
    expanded = np.broadcast_to(
        active,
        samples.shape,
    )
    values = np.abs(samples[expanded]).astype(np.float64)
    return {
        "abs_p99": float(np.percentile(values, 99.0)),
        "abs_p99_9": float(np.percentile(values, 99.9)),
        "abs_max": float(np.max(values)),
        "rms": float(np.sqrt(np.mean(values * values))),
    }


def evaluate_pool(
    *,
    sample_pool_z: np.ndarray,
    sample_pool_raw: np.ndarray,
    target_raw: np.ndarray,
    active_mask: np.ndarray,
    validity_contract: ValidityContract,
    k_values: Sequence[int] = (1, 4, 8),
) -> Dict[str, Any]:
    distances = candidate_final_chamfer(
        sample_pool_raw,
        target_raw,
    )
    nested = nested_best_of_k(distances, k_values)
    validity = sample_validity(
        sample_pool_raw,
        validity_contract,
    )
    z_stats = z_pool_stats(sample_pool_z, active_mask)
    diversity = pairwise_pool_diversity(sample_pool_raw)
    mean_future = sample_mean_future(sample_pool_raw)
    mean_error = candidate_final_chamfer(
        mean_future[None, ...],
        target_raw,
    )[0]
    return {
        "k_metrics": {
            str(k): {
                "mean": float(np.mean(nested[int(k)])),
                "median": float(np.median(nested[int(k)])),
            }
            for k in k_values
        },
        "distances": distances,
        "nested": nested,
        "physical_validity": {
            key: value
            for key, value in validity.items()
            if key != "sample_valid_mask"
        },
        "sample_valid_mask": validity["sample_valid_mask"],
        "z_stats": z_stats,
        "pool_diversity_mean": float(np.mean(diversity)),
        "sample_mean_error": float(np.mean(mean_error)),
    }


def run_stability_gate(
    *,
    pool_metrics: Mapping[str, Any],
    partial_t10_chamfer: float,
    partial_t99_chamfer: float,
) -> Dict[str, Any]:
    k1 = float(pool_metrics["k_metrics"]["1"]["mean"])
    best8 = float(pool_metrics["k_metrics"]["8"]["mean"])
    physical = float(
        pool_metrics["physical_validity"][
            "sample_validity_rate"
        ]
    )
    query_valid = float(
        pool_metrics["physical_validity"][
            "query_has_valid_candidate_rate"
        ]
    )
    z_p99 = float(pool_metrics["z_stats"]["abs_p99"])
    z_max = float(pool_metrics["z_stats"]["abs_max"])
    checks = {
        "finite": True,
        "physical_validity": physical
        >= MIN_RUN_PHYSICAL_VALIDITY,
        "query_validity": query_valid
        >= MIN_RUN_QUERY_VALIDITY,
        "z_abs_p99": z_p99 <= MAX_RUN_Z_ABS_P99,
        "z_abs_max": z_max <= MAX_RUN_Z_ABS_MAX,
        "k1_scale": k1 <= VAL_K1_LIMIT,
        "best8_scale": best8 <= VAL_BEST8_LIMIT,
        "partial_t10": float(partial_t10_chamfer)
        <= PARTIAL_T10_LIMIT,
        "partial_t99": float(partial_t99_chamfer)
        <= PARTIAL_T99_LIMIT,
    }
    return {
        "stable": bool(all(checks.values())),
        "checks": checks,
        "values": {
            "k1": k1,
            "best8": best8,
            "physical_validity": physical,
            "query_validity": query_valid,
            "z_abs_p99": z_p99,
            "z_abs_max": z_max,
            "partial_t10_chamfer": float(
                partial_t10_chamfer
            ),
            "partial_t99_chamfer": float(
                partial_t99_chamfer
            ),
        },
        "limits": {
            "k1": VAL_K1_LIMIT,
            "best8": VAL_BEST8_LIMIT,
            "physical_validity": MIN_RUN_PHYSICAL_VALIDITY,
            "query_validity": MIN_RUN_QUERY_VALIDITY,
            "z_abs_p99": MAX_RUN_Z_ABS_P99,
            "z_abs_max": MAX_RUN_Z_ABS_MAX,
            "partial_t10": PARTIAL_T10_LIMIT,
            "partial_t99": PARTIAL_T99_LIMIT,
        },
    }


def improvement_bootstrap(
    reference: np.ndarray,
    candidate: np.ndarray,
    visible_seed: np.ndarray,
    *,
    iterations: int,
    seed: int,
) -> Dict[str, float]:
    difference = (
        np.asarray(reference, dtype=np.float64)
        - np.asarray(candidate, dtype=np.float64)
    )
    return group_bootstrap_mean_ci(
        difference,
        np.asarray(visible_seed, dtype=np.int64),
        iterations=int(iterations),
        seed=int(seed),
    )


def branch_support_summary(
    *,
    sample_pool: np.ndarray,
    row_indices: np.ndarray,
    pair_map: Mapping[str, Tuple[int, int]],
    arrays: Mapping[str, np.ndarray],
    k: int,
) -> Dict[str, Any]:
    lookup = {
        int(original): local
        for local, original in enumerate(row_indices.tolist())
    }
    rows = []
    for pair_key in sorted(pair_map):
        free_original, hidden_original = pair_map[pair_key]
        free_final = arrays["y_state"][free_original, -1]
        hidden_final = arrays["y_state"][hidden_original, -1]
        for original in (free_original, hidden_original):
            local = lookup[original]
            result = branch_support_for_pair(
                sample_pool[: int(k), local, -1, :],
                free_final=free_final,
                hidden_final=hidden_final,
            )
            rows.append(
                {
                    "pair_key": pair_key,
                    "row_index": int(original),
                    **result,
                }
            )
    eligible = [row for row in rows if row["eligible"]]
    if not eligible:
        raise RuntimeError("no branch-eligible test queries")
    return {
        "rows": rows,
        "eligible_queries": len(eligible),
        "both_supported_rate": float(
            np.mean(
                [bool(row["both_supported"]) for row in eligible]
            )
        ),
        "free_supported_rate": float(
            np.mean(
                [bool(row["free_supported"]) for row in eligible]
            )
        ),
        "hidden_supported_rate": float(
            np.mean(
                [bool(row["hidden_supported"]) for row in eligible]
            )
        ),
        "mean_branch_separation": float(
            np.mean(
                [float(row["branch_separation"]) for row in eligible]
            )
        ),
    }
