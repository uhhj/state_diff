"""Phase3.14b-r2.5.6 Stage C physical-gate and branch attribution.

This stage freezes and replays the exact r2.5.6 Stage-B cable-only diagnostic.
It does not change the model, scheduler, seed, split, normalization, candidate
count, training steps, branch threshold, or physical threshold.

The purpose is to answer two questions that Stage B did not separate:

1. Is the historical segment-length physical gate itself calibrated on the
   diagnostic-training and probe ground truth?
2. If the gate is calibrated, at which reverse stage do ordered cable segment
   lengths become invalid?

Only the train/full-horizon view is opened.  Model weights, predictions,
candidates and snapshots are never persisted.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from ccda_phase3 import phase314b_r256_stageb_cable_diffusion as stageb

PHASE = "Phase3.14b-r2.5.6 Stage C"
PHASE_ID = "phase314b_r256_stagec"
BASE_EVIDENCE_COMMIT = "85cacdf8875847445dae8a364116ca662fc3c1b5"
EXPECTED_SUBMODULE_COMMIT = "633a88752445cf5d6776ed374fdbbdb35f93050c"

EXPECTED_STAGEB_SOURCE_SHA256 = (
    "13613c38ab6e3d3d1cc114cd0f708f2b3845be20f8568f0b95ab00082954bc90"
)
EXPECTED_STAGEB_TEST_GATE_SHA256 = (
    "8172a70d0794b6bf94f6a0db1583b2a230276b21c2efc33127b3e325a0bd5d81"
)
EXPECTED_STAGEB_WORKER_EVIDENCE_SHA256 = (
    "d871a208fea651227cef43f0fde645ca58ff066b261120ff90aa76b3b1fad299"
)
EXPECTED_STAGEB_SUMMARY_SHA256 = (
    "330bbc07c5cd31dde0c151af61ece7cfa37f97b8add6bec5fca6cfb0a01ff21f"
)
EXPECTED_STAGEB_REPORT_SHA256 = (
    "eb62b5ec49db575289bc065f2f6856cacbba2ca89b3173bdac21a4e000c6534c"
)

STAGEB_SOURCE = "ccda_phase3/phase314b_r256_stageb_cable_diffusion.py"
STAGEB_TEST_GATE = "reports/phase3_14b_r256_stageb_test_gate_summary.json"
STAGEB_WORKER_EVIDENCE = (
    "reports/phase3_14b_r256_stageb_worker_evidence.json"
)
STAGEB_SUMMARY = "reports/phase3_14b_r256_stageb_summary.json"
STAGEB_REPORT = "reports/phase3_14b_r256_stageb_report.md"

EXPECTED_HISTORICAL_PHYSICAL = {
    "finite_rate": 1.0,
    "coordinate_rate": 0.9920634920634921,
    "segment_rate": 0.0,
    "topology_rate": 0.9246031746031746,
    "candidate_rate": 0.0,
    "row_any_rate": 0.0,
}
EXPECTED_HISTORICAL_BRANCH_SUPPORT = 0.7380952380952381
EXPECTED_HISTORICAL_MODEL_SHA256 = (
    "4a9ebcdb9d1995602a501c8820e07c0efcb757f1e3efff4ac0f2c5a4a23421d2"
)
EXPECTED_HISTORICAL_OPTIMIZER_SHA256 = (
    "ff8feee0e677c2720a7807ad19984764f2d69e1e0c296a9c4056f426d2f22601"
)
EXPECTED_HISTORICAL_REVERSE_SHA256 = (
    "1ab9f8d0026f215bf9e3b8aafeb4f8799bc1f8d329d8773e415413699a6c6d2d"
)
EXPECTED_HISTORICAL_ONE_STEP_SHA256 = (
    "d647ac6b547a42212bbc82582fabed565f3f8340cd78a09d6c61ebb8ab6742ef"
)
EXPECTED_HISTORICAL_TRAIN_CONTROL_SHA256 = (
    "bb9b91da6eb324b24af10322ee20509ad426e54d851c194d62f71e63c68eb076"
)

SNAPSHOT_TIMESTEPS = (99, 75, 50, 25, 10, 0)


class StageCAttributionError(RuntimeError):
    """Raised when the frozen Stage-B replay or attribution contract fails."""


@dataclass(frozen=True)
class AttributionSpec:
    training_ground_truth_candidate_rate_min: float = 0.95
    probe_ground_truth_candidate_rate_min: float = 0.95
    one_step_segment_candidate_rate_min: float = 0.50
    segment_element_rate_min: float = 0.99
    bootstrap_resamples: int = 4096
    bootstrap_seed: int = 256300
    snapshot_timesteps: Tuple[int, ...] = SNAPSHOT_TIMESTEPS
    branch_prefix_k: Tuple[int, ...] = (1, 2, 4, 8)

    def validate(self) -> None:
        for value in (
            self.training_ground_truth_candidate_rate_min,
            self.probe_ground_truth_candidate_rate_min,
            self.one_step_segment_candidate_rate_min,
            self.segment_element_rate_min,
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError("rate threshold is outside [0,1]")
        if self.bootstrap_resamples <= 0:
            raise ValueError("bootstrap resamples must be positive")
        if tuple(sorted(set(self.snapshot_timesteps), reverse=True)) != (
            self.snapshot_timesteps
        ):
            raise ValueError("snapshot timesteps must be unique and descending")
        if self.snapshot_timesteps[0] != 99 or self.snapshot_timesteps[-1] != 0:
            raise ValueError("snapshot contract must cover t=99 through x0")
        if self.branch_prefix_k[-1] != stageb.DiagnosticSpec().reverse_candidates:
            raise ValueError("branch K curve must end at historical K")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(json.dumps(list(array.shape)).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise StageCAttributionError(f"JSON root is not an object: {path}")
    return value


def validate_immutable_inputs(root: Path) -> Dict[str, Any]:
    expected = {
        STAGEB_SOURCE: EXPECTED_STAGEB_SOURCE_SHA256,
        STAGEB_TEST_GATE: EXPECTED_STAGEB_TEST_GATE_SHA256,
        STAGEB_WORKER_EVIDENCE: EXPECTED_STAGEB_WORKER_EVIDENCE_SHA256,
        STAGEB_SUMMARY: EXPECTED_STAGEB_SUMMARY_SHA256,
        STAGEB_REPORT: EXPECTED_STAGEB_REPORT_SHA256,
    }
    for relative, expected_sha in expected.items():
        path = Path(root) / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = sha256_file(path)
        if observed != expected_sha:
            raise StageCAttributionError(
                f"immutable Stage-B input changed: {relative}: {observed}"
            )
    summary = load_json(Path(root) / STAGEB_SUMMARY)
    evidence = load_json(Path(root) / STAGEB_WORKER_EVIDENCE)
    if summary.get("verdict") != "PASS":
        raise StageCAttributionError("Stage-B audit verdict is not PASS")
    if summary.get("scientific_status") != "BLOCKED":
        raise StageCAttributionError("Stage-B scientific boundary changed")
    if summary.get("root_cause") != (
        "phase314b_r256_stageb_cable_only_branch_support_failed"
    ):
        raise StageCAttributionError("Stage-B root cause changed")
    worker = evidence.get("worker_result", {})
    if worker.get("identity", {}).get("final_model_sha256") != (
        EXPECTED_HISTORICAL_MODEL_SHA256
    ):
        raise StageCAttributionError("Stage-B model identity changed")
    physical = worker.get("evaluation", {}).get("physical", {})
    for key, expected_value in EXPECTED_HISTORICAL_PHYSICAL.items():
        if not math.isclose(
            float(physical.get(key, float("nan"))),
            float(expected_value),
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ):
            raise StageCAttributionError(
                f"Stage-B physical metric changed: {key}"
            )
    return {
        "file_sha256": expected,
        "summary": summary,
        "worker_evidence": evidence,
    }


def source_logic_audit(root: Path) -> Dict[str, Any]:
    text = (Path(root) / STAGEB_SOURCE).read_text(encoding="utf-8")
    branch_position = text.find("elif not branch_pass:")
    physical_position = text.find("elif not physical_pass:")
    checks = {
        "per_horizon_per_segment_quantile_contract": (
            "np.percentile(lengths, 0.5, axis=0)" in text
            and "np.percentile(lengths, 99.5, axis=0)" in text
        ),
        "segment_candidate_requires_all_horizons_and_segments": (
            "axis=(2, 3)" in text
            and '"segment": segment' in text
        ),
        "branch_failure_checked_before_physical_failure": (
            branch_position >= 0
            and physical_position >= 0
            and branch_position < physical_position
        ),
        "historical_reverse_is_deterministic_x0_ddim": (
            "predicted_x0 = model(value, timestep, condition_tensor)" in text
            and "math.sqrt(previous_alpha) * predicted_x0" in text
        ),
    }
    if not all(checks.values()):
        raise StageCAttributionError(
            f"Stage-B source assumptions changed: {checks}"
        )
    return {
        "checks": checks,
        "source_sha256": sha256_file(Path(root) / STAGEB_SOURCE),
        "historical_classifier_masks_simultaneous_physical_failure": True,
    }


def as_candidates(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 3 and array.shape[1:] == (
        stageb.FUTURE_STEPS,
        stageb.CABLE_DIM,
    ):
        return array[:, None]
    if array.ndim == 4 and array.shape[2:] == (
        stageb.FUTURE_STEPS,
        stageb.CABLE_DIM,
    ):
        return array
    raise ValueError(f"unsupported cable candidate shape: {array.shape}")


def _safe_stats(value: np.ndarray) -> Dict[str, float]:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "p05": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "minimum": 0.0,
            "maximum": 0.0,
        }
    if not np.all(np.isfinite(array)):
        raise StageCAttributionError("statistics input is non-finite")
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "p05": float(np.percentile(array, 5.0)),
        "p50": float(np.percentile(array, 50.0)),
        "p95": float(np.percentile(array, 95.0)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def physical_decomposition(
    value: np.ndarray,
    contract: stageb.GeometryContract,
) -> Dict[str, Any]:
    candidates = as_candidates(value)
    historical = stageb.physical_validity(candidates, contract)
    lengths = stageb.segment_lengths(candidates).astype(np.float64)
    lower = contract.segment_lower[None, None].astype(np.float64)
    upper = contract.segment_upper[None, None].astype(np.float64)

    lower_pass = lengths >= lower
    upper_pass = lengths <= upper
    element_pass = lower_pass & upper_pass
    horizon_all = np.all(element_pass, axis=-1)
    candidate_all = np.all(horizon_all, axis=-1)
    candidate_element_fraction = np.mean(element_pass, axis=(2, 3))
    lower_deficit = np.maximum(lower - lengths, 0.0)
    upper_excess = np.maximum(lengths - upper, 0.0)
    lower_relative = lower_deficit / np.maximum(lower, 1.0e-12)
    upper_relative = upper_excess / np.maximum(upper, 1.0e-12)

    if not np.array_equal(candidate_all, historical["segment"]):
        raise StageCAttributionError(
            "segment decomposition differs from historical gate"
        )
    finite = np.asarray(historical["finite"], dtype=np.bool_)
    coordinate = np.asarray(historical["coordinate"], dtype=np.bool_)
    topology = np.asarray(historical["topology"], dtype=np.bool_)
    combined = finite & coordinate & candidate_all & topology
    if not np.array_equal(combined, historical["valid"]):
        raise StageCAttributionError(
            "combined decomposition differs from historical gate"
        )

    segment_position_rate = np.mean(element_pass, axis=(0, 1))
    lower_position_rate = np.mean(lower_pass, axis=(0, 1))
    upper_position_rate = np.mean(upper_pass, axis=(0, 1))
    worst_positions: List[Dict[str, Any]] = []
    flat_order = np.argsort(segment_position_rate.reshape(-1))
    for flat_index in flat_order[: min(12, flat_order.size)]:
        horizon, segment_index = np.unravel_index(
            int(flat_index),
            segment_position_rate.shape,
        )
        worst_positions.append(
            {
                "horizon": int(horizon),
                "segment_index": int(segment_index),
                "combined_pass_rate":
                    float(segment_position_rate[horizon, segment_index]),
                "lower_pass_rate":
                    float(lower_position_rate[horizon, segment_index]),
                "upper_pass_rate":
                    float(upper_position_rate[horizon, segment_index]),
                "lower_bound":
                    float(contract.segment_lower[horizon, segment_index]),
                "upper_bound":
                    float(contract.segment_upper[horizon, segment_index]),
                "observed_length": _safe_stats(
                    lengths[..., horizon, segment_index]
                ),
            }
        )

    return {
        "shape": list(candidates.shape),
        "sha256": sha256_array(candidates),
        "finite_candidate_rate": float(np.mean(finite)),
        "coordinate_candidate_rate": float(np.mean(coordinate)),
        "topology_candidate_rate": float(np.mean(topology)),
        "combined_candidate_rate": float(np.mean(combined)),
        "combined_row_any_rate": float(np.mean(np.any(combined, axis=1))),
        "maximum_intersections": int(
            np.max(historical["intersection_max"])
        ),
        "segment": {
            "element_pass_rate": float(np.mean(element_pass)),
            "lower_element_pass_rate": float(np.mean(lower_pass)),
            "upper_element_pass_rate": float(np.mean(upper_pass)),
            "candidate_all_pass_rate": float(np.mean(candidate_all)),
            "row_any_candidate_pass_rate":
                float(np.mean(np.any(candidate_all, axis=1))),
            "horizon_all_pass_rate": [
                float(value) for value in np.mean(horizon_all, axis=(0, 1))
            ],
            "candidate_element_fraction":
                _safe_stats(candidate_element_fraction),
            "lower_violation_count":
                int(np.sum(~lower_pass)),
            "upper_violation_count":
                int(np.sum(~upper_pass)),
            "lower_deficit": _safe_stats(lower_deficit[~lower_pass]),
            "upper_excess": _safe_stats(upper_excess[~upper_pass]),
            "lower_relative_deficit":
                _safe_stats(lower_relative[~lower_pass]),
            "upper_relative_excess":
                _safe_stats(upper_relative[~upper_pass]),
            "worst_horizon_segment_positions": worst_positions,
        },
    }


def _pair_rows(pair_key: Sequence[Any]) -> List[Tuple[str, int, int]]:
    keys = np.asarray(pair_key).astype(str)
    groups: MutableMapping[str, List[int]] = {}
    for index, key in enumerate(keys):
        groups.setdefault(key, []).append(index)
    result = []
    for key in sorted(groups):
        indices = groups[key]
        if len(indices) != 2:
            raise StageCAttributionError(
                f"pair key does not contain two rows: {key}"
            )
        result.append((key, int(indices[0]), int(indices[1])))
    return result


def branch_prefix_curve(
    *,
    pair_key: Sequence[Any],
    condition_name: Sequence[Any],
    target: np.ndarray,
    candidates: np.ndarray,
    prefix_k: Sequence[int],
) -> Dict[str, Any]:
    pred = as_candidates(candidates)
    maximum_k = pred.shape[1]
    curve: Dict[str, Any] = {}
    for k in prefix_k:
        if not 1 <= int(k) <= maximum_k:
            raise ValueError(f"invalid branch prefix K={k}")
        metrics = stageb.branch_metrics(
            pair_key=pair_key,
            condition_name=condition_name,
            target=target,
            candidates=pred[:, : int(k)],
        )
        curve[str(k)] = {
            "row_own_branch_support_rate":
                metrics["row_own_branch_support_rate"],
            "row_mean_prediction_correct_rate":
                metrics["row_mean_prediction_correct_rate"],
        }
    return curve


def branch_margin_records(
    *,
    pair_key: Sequence[Any],
    condition_name: Sequence[Any],
    target: np.ndarray,
    candidates: np.ndarray,
) -> Dict[str, Any]:
    names = np.asarray(condition_name).astype(str)
    truth = np.asarray(target, dtype=np.float64)
    pred = as_candidates(candidates).astype(np.float64)
    rows: List[Dict[str, Any]] = []
    support_values = []
    margins = []
    condition_support: MutableMapping[str, List[bool]] = {}
    for key, left, right in _pair_rows(pair_key):
        if names[left] == names[right]:
            raise StageCAttributionError(f"duplicate pair condition: {key}")
        for own, other in ((left, right), (right, left)):
            own_error = np.mean(
                (pred[own] - truth[own][None]) ** 2,
                axis=(1, 2),
            )
            other_error = np.mean(
                (pred[own] - truth[other][None]) ** 2,
                axis=(1, 2),
            )
            candidate_margin = other_error - own_error
            supported = bool(np.any(candidate_margin > 0.0))
            support_values.append(supported)
            margins.append(float(np.max(candidate_margin)))
            condition_support.setdefault(names[own], []).append(supported)
            rows.append(
                {
                    "pair_key": key,
                    "condition": names[own],
                    "supported": supported,
                    "best_own_margin": float(np.max(candidate_margin)),
                    "mean_own_margin": float(np.mean(candidate_margin)),
                    "own_error_min": float(np.min(own_error)),
                    "counterpart_error_at_own_best": float(
                        other_error[int(np.argmin(own_error))]
                    ),
                }
            )
    return {
        "row_count": len(rows),
        "support_rate": float(np.mean(support_values)),
        "best_margin": _safe_stats(np.asarray(margins)),
        "condition_support_rate": {
            key: float(np.mean(value))
            for key, value in sorted(condition_support.items())
        },
        "rows": rows,
    }


def physical_valid_branch_metrics(
    *,
    pair_key: Sequence[Any],
    condition_name: Sequence[Any],
    target: np.ndarray,
    candidates: np.ndarray,
    valid_mask: np.ndarray,
) -> Dict[str, Any]:
    names = np.asarray(condition_name).astype(str)
    truth = np.asarray(target, dtype=np.float64)
    pred = as_candidates(candidates).astype(np.float64)
    valid = np.asarray(valid_mask, dtype=np.bool_)
    if valid.shape != pred.shape[:2]:
        raise ValueError("physical-valid mask shape mismatch")
    eligible = 0
    supported = 0
    condition_records: MutableMapping[str, List[bool]] = {}
    rows = []
    for key, left, right in _pair_rows(pair_key):
        if names[left] == names[right]:
            raise StageCAttributionError(f"duplicate pair condition: {key}")
        for own, other in ((left, right), (right, left)):
            selected = valid[own]
            if not np.any(selected):
                rows.append(
                    {
                        "pair_key": key,
                        "condition": names[own],
                        "eligible": False,
                        "supported": None,
                        "valid_candidate_count": 0,
                    }
                )
                continue
            own_error = np.mean(
                (pred[own, selected] - truth[own][None]) ** 2,
                axis=(1, 2),
            )
            other_error = np.mean(
                (pred[own, selected] - truth[other][None]) ** 2,
                axis=(1, 2),
            )
            row_supported = bool(np.any(own_error < other_error))
            eligible += 1
            supported += int(row_supported)
            condition_records.setdefault(names[own], []).append(row_supported)
            rows.append(
                {
                    "pair_key": key,
                    "condition": names[own],
                    "eligible": True,
                    "supported": row_supported,
                    "valid_candidate_count": int(np.sum(selected)),
                }
            )
    total_rows = len(rows)
    return {
        "available": bool(eligible > 0),
        "total_rows": total_rows,
        "eligible_rows": eligible,
        "eligible_row_rate": float(eligible / max(total_rows, 1)),
        "support_rate_among_eligible": (
            float(supported / eligible) if eligible else None
        ),
        "support_rate_all_rows": float(supported / max(total_rows, 1)),
        "physical_valid_candidate_count": int(np.sum(valid)),
        "condition_support_rate_among_eligible": {
            key: float(np.mean(value))
            for key, value in sorted(condition_records.items())
        },
        "rows": rows,
        "reason": (
            None
            if eligible
            else "no physical-valid reverse candidate under historical gate"
        ),
    }


def deterministic_pair_bootstrap_ci(
    *,
    pair_key: Sequence[Any],
    condition_name: Sequence[Any],
    target: np.ndarray,
    candidates: np.ndarray,
    resamples: int,
    seed: int,
) -> Dict[str, Any]:
    names = np.asarray(condition_name).astype(str)
    truth = np.asarray(target, dtype=np.float64)
    pred = as_candidates(candidates).astype(np.float64)
    pair_support: List[float] = []
    for _, left, right in _pair_rows(pair_key):
        row_support = []
        for own, other in ((left, right), (right, left)):
            own_error = np.mean(
                (pred[own] - truth[own][None]) ** 2,
                axis=(1, 2),
            )
            other_error = np.mean(
                (pred[own] - truth[other][None]) ** 2,
                axis=(1, 2),
            )
            row_support.append(float(np.any(own_error < other_error)))
        pair_support.append(float(np.mean(row_support)))
    values = np.asarray(pair_support, dtype=np.float64)
    rng = np.random.RandomState(int(seed))
    sampled = np.empty(int(resamples), dtype=np.float64)
    for index in range(int(resamples)):
        selection = rng.randint(0, values.shape[0], size=values.shape[0])
        sampled[index] = np.mean(values[selection])
    return {
        "pair_count": int(values.shape[0]),
        "point_estimate": float(np.mean(values)),
        "bootstrap_resamples": int(resamples),
        "bootstrap_seed": int(seed),
        "ci95": [
            float(np.percentile(sampled, 2.5)),
            float(np.percentile(sampled, 97.5)),
        ],
        "bootstrap_sha256": sha256_array(sampled),
    }


def one_step_predictions(
    *,
    model: Any,
    condition: np.ndarray,
    target: np.ndarray,
    spec: stageb.DiagnosticSpec,
    condition_standardizer: stageb.ArrayStandardizer,
    target_standardizer: stageb.ArrayStandardizer,
    noise_seed: int,
) -> Tuple[Dict[int, np.ndarray], str]:
    scheduler = stageb.scheduler_arrays(spec)
    target_z = target_standardizer.normalize(target)
    noise = stageb._fixed_noise(target.shape, seed=noise_seed)
    predictions: Dict[int, np.ndarray] = {}
    digest = hashlib.sha256()
    for timestep_value in spec.one_step_timesteps:
        timestep = np.full(
            target.shape[0],
            timestep_value,
            dtype=np.int64,
        )
        noisy_z = stageb.q_sample_numpy(
            target_z,
            noise,
            timestep,
            scheduler["alpha_bar"],
        )
        prediction = stageb.predict_x0(
            model=model,
            noisy=target_standardizer.denormalize(noisy_z),
            timestep=timestep,
            condition=condition,
            condition_standardizer=condition_standardizer,
            target_standardizer=target_standardizer,
        )
        prediction = np.asarray(prediction, dtype=np.float32)
        predictions[int(timestep_value)] = prediction
        digest.update(np.ascontiguousarray(prediction).tobytes())
    return predictions, digest.hexdigest()


def reproduce_train_control_prediction(
    *,
    model: Any,
    train_condition: np.ndarray,
    train_target: np.ndarray,
    spec: stageb.DiagnosticSpec,
    condition_standardizer: stageb.ArrayStandardizer,
    target_standardizer: stageb.ArrayStandardizer,
) -> np.ndarray:
    scheduler = stageb.scheduler_arrays(spec)
    rows = min(spec.train_control_rows, train_condition.shape[0])
    condition = train_condition[:rows]
    target = train_target[:rows]
    target_z = target_standardizer.normalize(target)
    timestep = np.full(rows, 25, dtype=np.int64)
    noise = stageb._fixed_noise(
        target.shape,
        seed=spec.seed + 1001,
    )
    noisy_z = stageb.q_sample_numpy(
        target_z,
        noise,
        timestep,
        scheduler["alpha_bar"],
    )
    return stageb.predict_x0(
        model=model,
        noisy=target_standardizer.denormalize(noisy_z),
        timestep=timestep,
        condition=condition,
        condition_standardizer=condition_standardizer,
        target_standardizer=target_standardizer,
    )


def reverse_sample_with_trace(
    *,
    model: Any,
    condition: np.ndarray,
    spec: stageb.DiagnosticSpec,
    condition_standardizer: stageb.ArrayStandardizer,
    target_standardizer: stageb.ArrayStandardizer,
    geometry: stageb.GeometryContract,
    noise_seed: int,
    snapshot_timesteps: Sequence[int],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    torch, _ = stageb._torch_imports()
    device = next(model.parameters()).device
    scheduler = stageb.scheduler_arrays(spec)
    alpha_bar = scheduler["alpha_bar"].astype(np.float64)
    rows = condition.shape[0]
    candidates = spec.reverse_candidates
    initial = stageb._fixed_noise(
        (rows, candidates, stageb.FUTURE_STEPS, stageb.CABLE_DIM),
        seed=noise_seed,
    )
    value = torch.as_tensor(
        initial.reshape(
            rows * candidates,
            stageb.FUTURE_STEPS,
            stageb.CABLE_DIM,
        ),
        dtype=torch.float32,
        device=device,
    )
    condition_z = condition_standardizer.normalize(condition)
    condition_repeated = np.repeat(
        condition_z[:, None, :],
        candidates,
        axis=1,
    ).reshape(rows * candidates, stageb.CONDITION_DIM)
    condition_tensor = torch.as_tensor(
        condition_repeated,
        dtype=torch.float32,
        device=device,
    )
    wanted = set(int(value) for value in snapshot_timesteps)
    snapshots: Dict[str, Any] = {}
    model.eval()
    with torch.no_grad():
        for timestep_value in reversed(range(spec.reverse_steps)):
            timestep = torch.full(
                (rows * candidates,),
                timestep_value,
                dtype=torch.long,
                device=device,
            )
            predicted_x0 = model(value, timestep, condition_tensor)
            if timestep_value in wanted and timestep_value != 0:
                latent_z = value.detach().cpu().numpy().reshape(
                    rows,
                    candidates,
                    stageb.FUTURE_STEPS,
                    stageb.CABLE_DIM,
                )
                predicted_z = predicted_x0.detach().cpu().numpy().reshape(
                    rows,
                    candidates,
                    stageb.FUTURE_STEPS,
                    stageb.CABLE_DIM,
                )
                latent = target_standardizer.denormalize(latent_z)
                predicted = target_standardizer.denormalize(predicted_z)
                snapshots[str(timestep_value)] = {
                    "latent": physical_decomposition(latent, geometry),
                    "predicted_x0":
                        physical_decomposition(predicted, geometry),
                    "latent_sha256": sha256_array(latent),
                    "predicted_x0_sha256": sha256_array(predicted),
                }

            current_alpha = float(alpha_bar[timestep_value])
            if timestep_value == 0:
                value = predicted_x0
                continue
            previous_alpha = float(alpha_bar[timestep_value - 1])
            epsilon = (
                value - math.sqrt(current_alpha) * predicted_x0
            ) / math.sqrt(max(1.0 - current_alpha, 1.0e-12))
            value = (
                math.sqrt(previous_alpha) * predicted_x0
                + math.sqrt(max(1.0 - previous_alpha, 0.0)) * epsilon
            )

    final_z = value.detach().cpu().numpy().reshape(
        rows,
        candidates,
        stageb.FUTURE_STEPS,
        stageb.CABLE_DIM,
    )
    final = target_standardizer.denormalize(final_z)
    snapshots["0"] = {
        "latent": physical_decomposition(final, geometry),
        "predicted_x0": physical_decomposition(final, geometry),
        "latent_sha256": sha256_array(final),
        "predicted_x0_sha256": sha256_array(final),
    }
    missing = sorted(str(value) for value in wanted - {0} if str(value) not in snapshots)
    if missing:
        raise StageCAttributionError(
            f"reverse trace is missing snapshots: {missing}"
        )
    return final, snapshots


def classify_attribution(
    *,
    spec: AttributionSpec,
    training_target: Mapping[str, Any],
    probe_target: Mapping[str, Any],
    one_step: Mapping[str, Mapping[str, Any]],
    reverse: Mapping[str, Any],
    historical_branch_support: float,
) -> Dict[str, Any]:
    simultaneous_failures = []
    if historical_branch_support < stageb.DiagnosticSpec().branch_support_min:
        simultaneous_failures.append("branch_support")
    if reverse["combined_candidate_rate"] < (
        stageb.DiagnosticSpec().physical_candidate_rate_min
    ):
        simultaneous_failures.append("physical_combined")
    if reverse["segment"]["candidate_all_pass_rate"] <= 0.0:
        simultaneous_failures.append("segment_all_candidate")

    training_rate = float(
        training_target["segment"]["candidate_all_pass_rate"]
    )
    probe_rate = float(
        probe_target["segment"]["candidate_all_pass_rate"]
    )
    one_step_10 = float(
        one_step["10"]["segment"]["candidate_all_pass_rate"]
    )
    reverse_rate = float(
        reverse["segment"]["candidate_all_pass_rate"]
    )

    if training_rate < spec.training_ground_truth_candidate_rate_min:
        root = (
            "phase314b_r256_stagec_segment_gate_self_calibration_failed"
        )
        next_path = (
            "RECALIBRATE_FROZEN_CABLE_SEGMENT_GATE_BEFORE_MODEL_OR_BRANCH_REPAIR"
        )
        locus = "physical_gate_contract"
    elif probe_rate < spec.probe_ground_truth_candidate_rate_min:
        root = (
            "phase314b_r256_stagec_segment_gate_probe_distribution_shift"
        )
        next_path = (
            "RECALIBRATE_OR_STRATIFY_CABLE_SEGMENT_GATE_BEFORE_MODEL_REPAIR"
        )
        locus = "probe_distribution_or_gate"
    elif one_step_10 < spec.one_step_segment_candidate_rate_min:
        root = (
            "phase314b_r256_stagec_cable_x0_segment_geometry_not_preserved"
        )
        next_path = (
            "ADD_ORDERED_SEGMENT_GEOMETRY_OBJECTIVE_TO_CABLE_X0_DENOISER"
        )
        locus = "x0_denoiser"
    elif reverse_rate < stageb.DiagnosticSpec().physical_candidate_rate_min:
        root = (
            "phase314b_r256_stagec_reverse_segment_geometry_collapse"
        )
        next_path = (
            "REPAIR_CABLE_REVERSE_SEGMENT_TRANSPORT_WITH_FROZEN_BRANCH_AUDIT"
        )
        locus = "reverse_transport"
    elif historical_branch_support < stageb.DiagnosticSpec().branch_support_min:
        root = (
            "phase314b_r256_stagec_physical_gate_passed_branch_support_failed"
        )
        next_path = "REPAIR_CABLE_ONLY_BRANCH_TRANSPORT_BEFORE_FORMAL_PILOT"
        locus = "branch_transport"
    else:
        root = (
            "phase314b_r256_stagec_physical_and_branch_attribution_supported"
        )
        next_path = (
            "RUN_FROZEN_CABLE_ONLY_FORMAL_PILOT_AND_COLLECT_ACTION_DIVERSE_IDM_DATA"
        )
        locus = "none"
    return {
        "root_cause": root,
        "required_next_path": next_path,
        "primary_failure_locus": locus,
        "simultaneous_historical_failures": simultaneous_failures,
        "historical_branch_failure_masked_physical_failure": bool(
            "branch_support" in simultaneous_failures
            and "physical_combined" in simultaneous_failures
        ),
        "training_target_segment_candidate_rate": training_rate,
        "probe_target_segment_candidate_rate": probe_rate,
        "one_step_t10_segment_candidate_rate": one_step_10,
        "reverse_segment_candidate_rate": reverse_rate,
    }


def run_attribution(
    *,
    root: Path,
    attribution_spec: Optional[AttributionSpec] = None,
) -> Dict[str, Any]:
    spec = AttributionSpec() if attribution_spec is None else attribution_spec
    spec.validate()
    repository_root = Path(root).resolve()
    immutable = validate_immutable_inputs(repository_root)
    source_audit = source_logic_audit(repository_root)

    stageb_spec = stageb.DiagnosticSpec()
    stageb_spec.validate()
    arrays = stageb.load_npz_strict(
        repository_root / stageb.STAGEA_TRAIN_VIEW,
        required_keys=stageb.REQUIRED_TRAIN_KEYS,
    )
    validation = stageb.validate_train_view(arrays)
    runtime = stageb.set_deterministic_runtime(stageb_spec.seed)

    condition = np.asarray(
        arrays["diffusion_condition_x"],
        dtype=np.float32,
    )
    target = np.asarray(
        arrays["diffusion_target_cable"],
        dtype=np.float32,
    )
    groups = np.asarray(arrays["episode_group_key"]).astype(str)
    train_mask, probe_mask, group_mapping = stageb.deterministic_group_split(
        groups,
        folds=stageb_spec.group_folds,
        probe_fold=stageb_spec.probe_fold,
    )
    condition_standardizer = stageb.fit_standardizer(condition[train_mask])
    target_standardizer = stageb.fit_standardizer(target[train_mask])
    geometry = stageb.fit_geometry_contract(target[train_mask])

    model, training, _ = stageb.train_model(
        condition=condition[train_mask],
        target=target[train_mask],
        spec=stageb_spec,
        condition_standardizer=condition_standardizer,
        target_standardizer=target_standardizer,
    )
    expected_identity = immutable["worker_evidence"]["worker_result"]["identity"]
    identity_checks = {
        "final_model_sha256": training["final_model_sha256"],
        "final_optimizer_sha256": training["final_optimizer_sha256"],
        "loss_history_sha256": training["loss_history_sha256"],
        "gradient_history_sha256": training["gradient_history_sha256"],
        "source_exposure_sha256": training["source_exposure_sha256"],
    }
    for key, observed in identity_checks.items():
        if observed != expected_identity[key]:
            raise StageCAttributionError(
                f"frozen Stage-B training replay differs: {key}: {observed}"
            )

    control_prediction = reproduce_train_control_prediction(
        model=model,
        train_condition=condition[train_mask],
        train_target=target[train_mask],
        spec=stageb_spec,
        condition_standardizer=condition_standardizer,
        target_standardizer=target_standardizer,
    )
    if sha256_array(control_prediction) != (
        EXPECTED_HISTORICAL_TRAIN_CONTROL_SHA256
    ):
        raise StageCAttributionError("train-control replay differs")

    one_step_predictions_by_t, one_step_sha = one_step_predictions(
        model=model,
        condition=condition[probe_mask],
        target=target[probe_mask],
        spec=stageb_spec,
        condition_standardizer=condition_standardizer,
        target_standardizer=target_standardizer,
        noise_seed=stageb_spec.seed + 2001,
    )
    if one_step_sha != EXPECTED_HISTORICAL_ONE_STEP_SHA256:
        raise StageCAttributionError("one-step replay differs")

    final_candidates, reverse_trace = reverse_sample_with_trace(
        model=model,
        condition=condition[probe_mask],
        spec=stageb_spec,
        condition_standardizer=condition_standardizer,
        target_standardizer=target_standardizer,
        geometry=geometry,
        noise_seed=stageb_spec.seed + 3001,
        snapshot_timesteps=spec.snapshot_timesteps,
    )
    if sha256_array(final_candidates) != EXPECTED_HISTORICAL_REVERSE_SHA256:
        raise StageCAttributionError("reverse candidate replay differs")

    probe_pair_key = np.asarray(arrays["pair_key"]).astype(str)[probe_mask]
    probe_condition_name = (
        np.asarray(arrays["condition_name"]).astype(str)[probe_mask]
    )
    historical_branch = stageb.branch_metrics(
        pair_key=probe_pair_key,
        condition_name=probe_condition_name,
        target=target[probe_mask],
        candidates=final_candidates,
    )
    if not math.isclose(
        historical_branch["row_own_branch_support_rate"],
        EXPECTED_HISTORICAL_BRANCH_SUPPORT,
        rel_tol=0.0,
        abs_tol=1.0e-15,
    ):
        raise StageCAttributionError("historical branch replay differs")

    training_target_physical = physical_decomposition(
        target[train_mask],
        geometry,
    )
    probe_target_physical = physical_decomposition(
        target[probe_mask],
        geometry,
    )
    last_state_physical = physical_decomposition(
        stageb.last_state_baseline(condition[probe_mask]),
        geometry,
    )
    constant_velocity_physical = physical_decomposition(
        stageb.constant_velocity_baseline(condition[probe_mask]),
        geometry,
    )
    one_step_physical = {
        str(timestep): physical_decomposition(prediction, geometry)
        for timestep, prediction in sorted(one_step_predictions_by_t.items())
    }
    final_physical = physical_decomposition(final_candidates, geometry)
    for key, expected_value in EXPECTED_HISTORICAL_PHYSICAL.items():
        mapped_key = {
            "finite_rate": "finite_candidate_rate",
            "coordinate_rate": "coordinate_candidate_rate",
            "segment_rate": None,
            "topology_rate": "topology_candidate_rate",
            "candidate_rate": "combined_candidate_rate",
            "row_any_rate": "combined_row_any_rate",
        }[key]
        observed = (
            final_physical["segment"]["candidate_all_pass_rate"]
            if key == "segment_rate"
            else final_physical[mapped_key]
        )
        if not math.isclose(
            float(observed),
            float(expected_value),
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ):
            raise StageCAttributionError(
                f"historical physical replay differs: {key}"
            )

    branch_curve = branch_prefix_curve(
        pair_key=probe_pair_key,
        condition_name=probe_condition_name,
        target=target[probe_mask],
        candidates=final_candidates,
        prefix_k=spec.branch_prefix_k,
    )
    branch_margins = branch_margin_records(
        pair_key=probe_pair_key,
        condition_name=probe_condition_name,
        target=target[probe_mask],
        candidates=final_candidates,
    )
    branch_ci = deterministic_pair_bootstrap_ci(
        pair_key=probe_pair_key,
        condition_name=probe_condition_name,
        target=target[probe_mask],
        candidates=final_candidates,
        resamples=spec.bootstrap_resamples,
        seed=spec.bootstrap_seed,
    )

    physical_valid_mask = stageb.physical_validity(
        final_candidates,
        geometry,
    )["valid"]
    physical_branch = physical_valid_branch_metrics(
        pair_key=probe_pair_key,
        condition_name=probe_condition_name,
        target=target[probe_mask],
        candidates=final_candidates,
        valid_mask=physical_valid_mask,
    )

    classification = classify_attribution(
        spec=spec,
        training_target=training_target_physical,
        probe_target=probe_target_physical,
        one_step=one_step_physical,
        reverse=final_physical,
        historical_branch_support=
            historical_branch["row_own_branch_support_rate"],
    )
    return {
        "phase": PHASE,
        "phase_id": PHASE_ID,
        "schema": "phase314b_r256_stagec_worker_result_v1",
        "verdict": "PASS",
        "scientific_status": "BLOCKED",
        "root_cause": classification["root_cause"],
        "required_next_path": classification["required_next_path"],
        "attribution_spec": asdict(spec),
        "frozen_stageb_spec": asdict(stageb_spec),
        "runtime": runtime,
        "immutable_inputs": immutable["file_sha256"],
        "source_logic_audit": source_audit,
        "train_view_validation": validation,
        "split": {
            "total_groups": len(group_mapping),
            "training_rows": int(np.sum(train_mask)),
            "probe_rows": int(np.sum(probe_mask)),
            "training_groups": len(set(groups[train_mask].tolist())),
            "probe_groups": len(set(groups[probe_mask].tolist())),
            "group_integrity_pass": True,
        },
        "frozen_replay": {
            "identity": {
                **identity_checks,
                "train_control_prediction_sha256":
                    sha256_array(control_prediction),
                "one_step_prediction_sha256": one_step_sha,
                "reverse_candidate_sha256":
                    sha256_array(final_candidates),
            },
            "matches_stageb": True,
            "historical_branch_support":
                historical_branch["row_own_branch_support_rate"],
            "historical_physical": final_physical,
        },
        "physical_attribution": {
            "training_ground_truth": training_target_physical,
            "probe_ground_truth": probe_target_physical,
            "last_state_baseline": last_state_physical,
            "constant_velocity_baseline":
                constant_velocity_physical,
            "one_step_predictions": one_step_physical,
            "reverse_final": final_physical,
            "reverse_trace": reverse_trace,
        },
        "branch_attribution": {
            "historical_k8":
                historical_branch["row_own_branch_support_rate"],
            "prefix_k_curve": branch_curve,
            "margin_records": branch_margins,
            "pair_bootstrap_ci": branch_ci,
            "physical_valid_only": physical_branch,
            "historical_threshold":
                stageb_spec.branch_support_min,
            "threshold_changed": False,
        },
        "classification": classification,
        "historical_evidence_modified": False,
        "model_changed": False,
        "scheduler_changed": False,
        "training_changed": False,
        "threshold_changed": False,
        "validation_targets_used": False,
        "formal_test_targets_used": False,
        "checkpoint_saved": False,
        "weights_persisted": False,
        "prediction_tensor_persisted": False,
        "formal_diffusion_training": False,
        "formal_reverse_sampling": False,
        "formal_idm_training": False,
        "action_diverse_data_collection": False,
        "candidate_execution": False,
        "phase4": False,
        "cps": False,
        "train_only_recommendation": None,
        "selected_configuration": None,
    }


def identity_projection(result: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "root_cause": result["root_cause"],
        "required_next_path": result["required_next_path"],
        "frozen_replay": result["frozen_replay"],
        "physical_attribution": result["physical_attribution"],
        "branch_attribution": result["branch_attribution"],
        "classification": result["classification"],
        "split": result["split"],
    }


def compare_worker_results(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> Dict[str, Any]:
    left_bytes = stageb.stable_json_bytes(identity_projection(left))
    right_bytes = stageb.stable_json_bytes(identity_projection(right))
    return {
        "exact": left_bytes == right_bytes,
        "left_sha256": hashlib.sha256(left_bytes).hexdigest(),
        "right_sha256": hashlib.sha256(right_bytes).hexdigest(),
        "frozen_identity_exact": {
            key: (
                left["frozen_replay"]["identity"][key]
                == right["frozen_replay"]["identity"][key]
            )
            for key in sorted(left["frozen_replay"]["identity"])
        },
    }
