"""Ordered-cable validity calibration and failure decomposition."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from ccda_phase3.phase314a_contract import (
    BEAD_XY_DIM,
    N_BEADS,
)
from ccda_phase3.phase314a_metrics import chamfer_xy
from ccda_phase3.phase314b_r2_metrics import (
    ValidityContract,
    fit_validity_contract,
    sample_validity,
)
from ccda_phase3.phase314b_r21_contract import (
    CALIBRATION_ALPHA,
    COORDINATE_SCALE_FLOOR_METERS,
    ROBUST_SCALE_FLOOR_METERS,
    SEGMENT_GROSS_COMPRESSION_RATIO,
    SEGMENT_GROSS_STRETCH_RATIO,
    conformal_quantile,
)
from ccda_phase3.schema_v2 import DEFAULT_TF, STATE_DIM


@dataclass(frozen=True)
class CalibratedGeometryContract:
    coordinate_lower: np.ndarray
    coordinate_upper: np.ndarray
    segment_center: np.ndarray
    segment_scale: np.ndarray
    segment_score_threshold: float
    chain_center: np.ndarray
    chain_scale: np.ndarray
    chain_score_threshold: float
    quaternion_norm_lower: float = 0.90
    quaternion_norm_upper: float = 1.10

    def to_json(self) -> Dict[str, Any]:
        return {
            "coordinate_lower": self.coordinate_lower.tolist(),
            "coordinate_upper": self.coordinate_upper.tolist(),
            "segment_center": self.segment_center.tolist(),
            "segment_scale": self.segment_scale.tolist(),
            "segment_score_threshold": self.segment_score_threshold,
            "chain_center": self.chain_center.tolist(),
            "chain_scale": self.chain_scale.tolist(),
            "chain_score_threshold": self.chain_score_threshold,
            "quaternion_norm_lower": self.quaternion_norm_lower,
            "quaternion_norm_upper": self.quaternion_norm_upper,
        }


def _finite(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or Inf")
    return array


def ordered_xy(states: np.ndarray) -> np.ndarray:
    value = _finite(states, "states")
    if value.shape[-1] != STATE_DIM:
        raise ValueError("state dimension mismatch")
    return value[..., :BEAD_XY_DIM].reshape(
        value.shape[:-1] + (N_BEADS, 2)
    )


def segment_lengths(states: np.ndarray) -> np.ndarray:
    xy = ordered_xy(states)
    return np.linalg.norm(
        xy[..., 1:, :] - xy[..., :-1, :],
        axis=-1,
    )


def chain_lengths(states: np.ndarray) -> np.ndarray:
    return np.sum(segment_lengths(states), axis=-1)


def robust_center_scale(
    value: np.ndarray,
    *,
    axis: int | Tuple[int, ...],
    floor: float,
) -> Tuple[np.ndarray, np.ndarray]:
    array = _finite(value, "robust_values")
    center = np.median(array, axis=axis)
    expanded = center
    if isinstance(axis, tuple):
        for item in sorted(axis):
            expanded = np.expand_dims(expanded, axis=item)
    else:
        expanded = np.expand_dims(expanded, axis=axis)
    mad = np.median(np.abs(array - expanded), axis=axis)
    scale = np.maximum(1.4826 * mad, float(floor))
    return (
        np.asarray(center, dtype=np.float32),
        np.asarray(scale, dtype=np.float32),
    )


def _coordinate_bounds(fit_future: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    xy = ordered_xy(fit_future).reshape(-1, 2)
    lower = np.percentile(xy, 0.1, axis=0)
    upper = np.percentile(xy, 99.9, axis=0)
    width = np.maximum(
        upper - lower,
        COORDINATE_SCALE_FLOOR_METERS,
    )
    margin = np.maximum(0.02, 0.20 * width)
    return (
        (lower - margin).astype(np.float32),
        (upper + margin).astype(np.float32),
    )


def segment_anomaly_scores(
    states: np.ndarray,
    *,
    center: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    lengths = segment_lengths(states)
    expected_shape = (DEFAULT_TF, N_BEADS - 1)
    if center.shape != expected_shape or scale.shape != expected_shape:
        raise ValueError("segment center/scale shape mismatch")
    normalized = np.abs(
        (lengths - center) / scale
    )
    return np.max(normalized, axis=(-2, -1))


def chain_anomaly_scores(
    states: np.ndarray,
    *,
    center: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    lengths = chain_lengths(states)
    if center.shape != (DEFAULT_TF,) or scale.shape != (DEFAULT_TF,):
        raise ValueError("chain center/scale shape mismatch")
    normalized = np.abs((lengths - center) / scale)
    return np.max(normalized, axis=-1)


def fit_calibrated_geometry_contract(
    *,
    fit_future: np.ndarray,
    calibration_future: np.ndarray,
    alpha: float = CALIBRATION_ALPHA,
) -> CalibratedGeometryContract:
    fit = _finite(fit_future, "fit_future")
    calibration = _finite(
        calibration_future,
        "calibration_future",
    )
    if fit.ndim != 3 or fit.shape[1:] != (DEFAULT_TF, STATE_DIM):
        raise ValueError("fit_future must be [N,4,87]")
    if (
        calibration.ndim != 3
        or calibration.shape[1:] != (DEFAULT_TF, STATE_DIM)
    ):
        raise ValueError("calibration_future must be [N,4,87]")

    fit_segments = segment_lengths(fit)
    segment_center, segment_scale = robust_center_scale(
        fit_segments,
        axis=0,
        floor=ROBUST_SCALE_FLOOR_METERS,
    )
    fit_chain = chain_lengths(fit)
    chain_center, chain_scale = robust_center_scale(
        fit_chain,
        axis=0,
        floor=ROBUST_SCALE_FLOOR_METERS,
    )
    calibration_segment_score = segment_anomaly_scores(
        calibration,
        center=segment_center,
        scale=segment_scale,
    )
    calibration_chain_score = chain_anomaly_scores(
        calibration,
        center=chain_center,
        scale=chain_scale,
    )
    coordinate_lower, coordinate_upper = _coordinate_bounds(fit)

    return CalibratedGeometryContract(
        coordinate_lower=coordinate_lower,
        coordinate_upper=coordinate_upper,
        segment_center=segment_center,
        segment_scale=segment_scale,
        segment_score_threshold=conformal_quantile(
            calibration_segment_score,
            alpha=alpha,
        ),
        chain_center=chain_center,
        chain_scale=chain_scale,
        chain_score_threshold=conformal_quantile(
            calibration_chain_score,
            alpha=alpha,
        ),
    )


def calibrated_validity(
    sample_pool: np.ndarray,
    contract: CalibratedGeometryContract,
) -> Dict[str, Any]:
    samples = _finite(sample_pool, "sample_pool")
    if samples.ndim != 4 or samples.shape[2:] != (
        DEFAULT_TF,
        STATE_DIM,
    ):
        raise ValueError("sample_pool must be [K,N,4,87]")
    xy = ordered_xy(samples)
    coordinate_valid = np.all(
        (
            xy
            >= contract.coordinate_lower[
                None, None, None, None, :
            ]
        )
        & (
            xy
            <= contract.coordinate_upper[
                None, None, None, None, :
            ]
        ),
        axis=(2, 3, 4),
    )
    segment_score = segment_anomaly_scores(
        samples,
        center=contract.segment_center,
        scale=contract.segment_scale,
    )
    chain_score = chain_anomaly_scores(
        samples,
        center=contract.chain_center,
        scale=contract.chain_scale,
    )
    segment_valid = (
        segment_score <= float(contract.segment_score_threshold)
    )
    chain_valid = chain_score <= float(contract.chain_score_threshold)

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
    valid = (
        coordinate_valid
        & segment_valid
        & chain_valid
        & quaternion_valid
    )
    return {
        "sample_valid_mask": valid,
        "sample_validity_rate": float(np.mean(valid)),
        "query_has_valid_candidate_rate": float(
            np.mean(np.any(valid, axis=0))
        ),
        "coordinate_validity_rate": float(
            np.mean(coordinate_valid)
        ),
        "segment_score_validity_rate": float(
            np.mean(segment_valid)
        ),
        "chain_score_validity_rate": float(
            np.mean(chain_valid)
        ),
        "quaternion_validity_rate": float(
            np.mean(quaternion_valid)
        ),
        "segment_score_p50": float(
            np.percentile(segment_score, 50)
        ),
        "segment_score_p95": float(
            np.percentile(segment_score, 95)
        ),
        "segment_score_p99": float(
            np.percentile(segment_score, 99)
        ),
        "segment_score_max": float(np.max(segment_score)),
        "chain_score_p95": float(
            np.percentile(chain_score, 95)
        ),
        "chain_score_max": float(np.max(chain_score)),
    }


def evaluate_original_and_calibrated(
    states: np.ndarray,
    *,
    original_contract: ValidityContract,
    calibrated_contract: CalibratedGeometryContract,
) -> Dict[str, Any]:
    value = _finite(states, "states")
    if value.ndim == 3:
        pool = value[None, ...]
    elif value.ndim == 4:
        pool = value
    else:
        raise ValueError("states must be [N,4,87] or [K,N,4,87]")
    original = sample_validity(pool, original_contract)
    calibrated = calibrated_validity(pool, calibrated_contract)
    return {
        "original": {
            key: val
            for key, val in original.items()
            if key != "sample_valid_mask"
        },
        "calibrated": {
            key: val
            for key, val in calibrated.items()
            if key != "sample_valid_mask"
        },
    }


def per_edge_violation_summary(
    sample_pool: np.ndarray,
    original_contract: ValidityContract,
) -> Dict[str, Any]:
    lengths = segment_lengths(sample_pool)
    lower = original_contract.segment_lower[
        None, None, None, :
    ]
    upper = original_contract.segment_upper[
        None, None, None, :
    ]
    under = lengths < lower
    over = lengths > upper
    any_violation = under | over
    return {
        "under_rate_by_horizon_edge": np.mean(
            under, axis=(0, 1)
        ).tolist(),
        "over_rate_by_horizon_edge": np.mean(
            over, axis=(0, 1)
        ).tolist(),
        "violation_rate_by_horizon_edge": np.mean(
            any_violation,
            axis=(0, 1),
        ).tolist(),
        "worst_under_horizon_edge": list(
            np.unravel_index(
                int(np.argmax(np.mean(under, axis=(0, 1)))),
                (DEFAULT_TF, N_BEADS - 1),
            )
        ),
        "worst_over_horizon_edge": list(
            np.unravel_index(
                int(np.argmax(np.mean(over, axis=(0, 1)))),
                (DEFAULT_TF, N_BEADS - 1),
            )
        ),
        "under_fraction": float(np.mean(under)),
        "over_fraction": float(np.mean(over)),
        "any_violation_fraction": float(np.mean(any_violation)),
    }


def _ordered_errors(
    prediction_xy: np.ndarray,
    target_xy: np.ndarray,
) -> Tuple[float, float, float]:
    direct = float(
        np.sqrt(np.mean((prediction_xy - target_xy) ** 2))
    )
    reversed_error = float(
        np.sqrt(
            np.mean(
                (
                    prediction_xy
                    - target_xy[::-1]
                )
                ** 2
            )
        )
    )
    chamfer = float(chamfer_xy(prediction_xy, target_xy))
    return direct, reversed_error, chamfer


def _nearest_target_order_metrics(
    prediction_xy: np.ndarray,
    target_xy: np.ndarray,
) -> Dict[str, float]:
    distances = np.linalg.norm(
        prediction_xy[:, None, :] - target_xy[None, :, :],
        axis=-1,
    )
    nearest = np.argmin(distances, axis=1).astype(np.int64)
    inversions = 0
    for left in range(nearest.size):
        inversions += int(
            np.sum(nearest[left + 1 :] < nearest[left])
        )
    max_inversions = nearest.size * (nearest.size - 1) / 2
    adjacent_backward = float(np.mean(np.diff(nearest) < 0))
    adjacent_jump = float(np.mean(np.abs(np.diff(nearest))))
    unique_fraction = float(
        np.unique(nearest).size / nearest.size
    )
    return {
        "nearest_index_inversion_rate": float(
            inversions / max(max_inversions, 1.0)
        ),
        "nearest_index_backward_rate": adjacent_backward,
        "nearest_index_mean_abs_jump": adjacent_jump,
        "nearest_index_unique_fraction": unique_fraction,
    }


def candidate_geometry_rows(
    *,
    sample_pool: np.ndarray,
    target_future: np.ndarray,
    row_indices: np.ndarray,
    pair_keys: np.ndarray,
    conditions: np.ndarray,
    visible_seeds: np.ndarray,
    segment_center: np.ndarray,
) -> list[Dict[str, Any]]:
    samples = _finite(sample_pool, "sample_pool")
    target = _finite(target_future, "target_future")
    if samples.shape[1:] != (
        target.shape[0],
        DEFAULT_TF,
        STATE_DIM,
    ):
        raise ValueError("sample/target shape mismatch")
    sample_xy = ordered_xy(samples)
    target_xy = ordered_xy(target)
    sample_segments = segment_lengths(samples)
    target_segments = segment_lengths(target)
    center = np.asarray(segment_center, dtype=np.float32)
    rows: list[Dict[str, Any]] = []

    for sample_index in range(samples.shape[0]):
        for local in range(samples.shape[1]):
            final_prediction = sample_xy[
                sample_index, local, -1
            ]
            final_target = target_xy[local, -1]
            direct, reversed_error, chamfer = _ordered_errors(
                final_prediction,
                final_target,
            )
            order = _nearest_target_order_metrics(
                final_prediction,
                final_target,
            )
            predicted_lengths = sample_segments[
                sample_index, local
            ]
            target_lengths = target_segments[local]
            reference = np.maximum(
                center,
                ROBUST_SCALE_FLOOR_METERS,
            )
            ratio = predicted_lengths / reference
            target_ratio = predicted_lengths / np.maximum(
                target_lengths,
                ROBUST_SCALE_FLOOR_METERS,
            )
            rows.append(
                {
                    "sample_index": sample_index,
                    "row_index": int(row_indices[local]),
                    "pair_key": str(pair_keys[local]),
                    "condition": str(conditions[local]),
                    "visible_seed": int(visible_seeds[local]),
                    "final_ordered_rmse": direct,
                    "final_reverse_ordered_rmse": reversed_error,
                    "final_chamfer": chamfer,
                    "permutation_gap": float(
                        min(direct, reversed_error) - chamfer
                    ),
                    "chain_length_ratio_to_target": float(
                        np.sum(predicted_lengths[-1])
                        / max(
                            float(np.sum(target_lengths[-1])),
                            ROBUST_SCALE_FLOOR_METERS,
                        )
                    ),
                    "max_segment_ratio_to_train_center": float(
                        np.max(ratio)
                    ),
                    "min_segment_ratio_to_train_center": float(
                        np.min(ratio)
                    ),
                    "max_segment_ratio_to_target": float(
                        np.max(target_ratio)
                    ),
                    "min_segment_ratio_to_target": float(
                        np.min(target_ratio)
                    ),
                    "gross_stretch_fraction": float(
                        np.mean(
                            ratio
                            >= SEGMENT_GROSS_STRETCH_RATIO
                        )
                    ),
                    "gross_compression_fraction": float(
                        np.mean(
                            ratio
                            <= SEGMENT_GROSS_COMPRESSION_RATIO
                        )
                    ),
                    **order,
                }
            )
    return rows


def aggregate_geometry_rows(
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, float]:
    if not rows:
        raise ValueError("cannot aggregate empty geometry rows")
    numeric_keys = (
        "final_ordered_rmse",
        "final_reverse_ordered_rmse",
        "final_chamfer",
        "permutation_gap",
        "chain_length_ratio_to_target",
        "max_segment_ratio_to_train_center",
        "min_segment_ratio_to_train_center",
        "max_segment_ratio_to_target",
        "min_segment_ratio_to_target",
        "gross_stretch_fraction",
        "gross_compression_fraction",
        "nearest_index_inversion_rate",
        "nearest_index_backward_rate",
        "nearest_index_mean_abs_jump",
        "nearest_index_unique_fraction",
    )
    output: Dict[str, float] = {}
    for key in numeric_keys:
        values = np.asarray(
            [float(row[key]) for row in rows],
            dtype=np.float64,
        )
        output[f"{key}_mean"] = float(np.mean(values))
        output[f"{key}_p95"] = float(np.percentile(values, 95))
        output[f"{key}_max"] = float(np.max(values))
    return output
