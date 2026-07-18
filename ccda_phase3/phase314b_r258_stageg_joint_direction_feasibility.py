"""Phase3.14b-r2.5.8 Stage G joint direction-feasibility surrogate.

Stage F established three facts before this module is admitted:

* the structural-zero cable representation is explicit and translation stable;
* the clipping-aware float32 ULP policy is locked on objective-train with factor 1;
* grouped-OOF projected-direction signal exists, but no direction-only candidate
  forms an integrable scientific state witness under the frozen Stage-E
  constrained direct-x0 integrator.

Stage G therefore does not change the Stage-E integrator, Stage-F features,
projected-oracle target, grouped folds, or holdout boundary.  It adds one
train-only mechanism: for each deployable direction backbone, construct a
predeclared bank of scaled directions, run the frozen target-independent
integrator, and learn a fold-local feasibility/progress ranker from deployable
candidate descriptors.  The ranker selects one bank member per row before the
ground-truth target is opened for evaluation.

No diffusion model, inverse-dynamics model, action execution, environment
rollout, checkpoint, model weight, prediction tensor, candidate tensor, NPZ,
cache, image, or video is persisted.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ccda_phase3 import phase314b_r256_stageb_cable_diffusion as stageb
from ccda_phase3 import phase314b_r258_staged_direction_surrogate as staged258
from ccda_phase3 import phase314b_r258_stagee_constrained_integrator as stagee258
from ccda_phase3 import phase314b_r258_stagef_constraint_aware_surrogate as stagef


PHASE = "Phase3.14b-r2.5.8 Stage G"
PHASE_ID = "phase314b_r258_stageg"
BASE_EVIDENCE_COMMIT = "a00be9cb9ddd2b373fe38edad7742317641a3cb1"
BASE_IMPLEMENTATION_COMMIT = "5a7a910a46963c7bfa9de68aabedd398dc0f92ba"
EXPECTED_SUBMODULE_COMMIT = "633a88752445cf5d6776ed374fdbbdb35f93050c"
BASE_REPORT = "reports/phase3_14b_r258_stagef_consolidated_v8_ulp_policy_summary.json"
EXPECTED_BASE_REPORT_SHA256 = (
    "c913719d2bfd35d8d4353bbb451b23be5d857ae95a860da1675f1637c3a38e35"
)
EXPECTED_BASE_SINGLE_RUN_SHA256 = (
    "3db43d765f17ae94d566562f32b43b0b7cba0e03bbf0c5bea0718d489d4c546e"
)
EXPECTED_BASE_ROOT_CAUSE = (
    "phase314b_r258_stagef_constraint_aware_direction_still_not_integrable"
)
EXPECTED_BASE_NEXT_PATH = "CALIBRATE_JOINT_DIRECTION_AND_FEASIBILITY_SURROGATE"
EXPECTED_SELECTED_ULP_FACTOR = 1.0

BASE_DIRECTION_IDS: Tuple[str, ...] = (
    "anchor_point_rr32_all",
    "anchor_point_rr32_feasible",
    "full_point_rr32_all",
    "full_point_rr32_feasible",
    "full_point_rr64_feasible",
    "segment_target_rr32_all",
    "segment_target_rr32_feasible",
    "segment_target_rr64_feasible",
    "full_point_rff256_feasible",
)
SCALE_MULTIPLIERS: Tuple[float, ...] = (0.25, 0.5, 1.0, 2.0)


class JointDirectionFeasibilityError(RuntimeError):
    """Raised when Stage-G scientific or leakage invariants fail."""


@dataclass(frozen=True)
class JointDirectionFeasibilitySpec:
    timesteps: Tuple[int, ...] = stagef.TIMESTEPS
    grouped_cv_folds: int = 6
    scale_multipliers: Tuple[float, ...] = SCALE_MULTIPLIERS
    utility_ridge_alpha: float = 1.0
    feasibility_l2: float = 1.0
    feasibility_max_iterations: int = 48
    feasibility_tolerance: float = 1.0e-10
    probability_cutoff: float = 0.5
    probability_weight: float = 0.05
    beneficial_reduction_min: float = 0.0
    brier_improvement_min: float = 0.0
    utility_correlation_min: float = 0.10
    fixed_scale_reduction_margin_min: float = 0.005
    permutation_reduction_margin_min: float = 0.01
    holdout_reduction_mean_min: float = 0.0
    descriptor_epsilon: float = 1.0e-12
    ranker_seed: int = 260718

    def validate(self) -> None:
        if self.timesteps != stagef.TIMESTEPS:
            raise ValueError("Stage-G timestep population changed")
        if self.grouped_cv_folds != 6:
            raise ValueError("Stage-G grouped fold count changed")
        if self.scale_multipliers != SCALE_MULTIPLIERS:
            raise ValueError("Stage-G scale bank changed")
        if any(float(value) <= 0.0 for value in self.scale_multipliers):
            raise ValueError("Stage-G scale bank is not positive")
        if tuple(sorted(self.scale_multipliers)) != self.scale_multipliers:
            raise ValueError("Stage-G scale bank order changed")
        for value in (
            self.utility_ridge_alpha,
            self.feasibility_l2,
            self.feasibility_tolerance,
            self.probability_cutoff,
            self.probability_weight,
            self.utility_correlation_min,
            self.fixed_scale_reduction_margin_min,
            self.permutation_reduction_margin_min,
            self.descriptor_epsilon,
        ):
            if float(value) <= 0.0:
                raise ValueError("Stage-G positive policy value is invalid")
        if not 0.0 < self.probability_cutoff < 1.0:
            raise ValueError("Stage-G probability cutoff is invalid")
        if self.feasibility_max_iterations != 48:
            raise ValueError("Stage-G logistic iteration count changed")
        if self.ranker_seed != 260718:
            raise ValueError("Stage-G ranker seed changed")


@dataclass(frozen=True)
class JointCandidateDefinition:
    candidate_id: str
    base_direction_id: str

    def validate(self) -> None:
        if self.base_direction_id not in BASE_DIRECTION_IDS:
            raise ValueError("unknown Stage-G direction backbone")
        if self.candidate_id != self.base_direction_id + "__joint_ranker":
            raise ValueError("Stage-G candidate identifier changed")
        base = stagef.definition_by_id(self.base_direction_id)
        if base.role != "selectable":
            raise ValueError("Stage-G backbone is not deployable/selectable")


JOINT_CANDIDATE_DEFINITIONS: Tuple[JointCandidateDefinition, ...] = tuple(
    JointCandidateDefinition(
        candidate_id=base_id + "__joint_ranker",
        base_direction_id=base_id,
    )
    for base_id in BASE_DIRECTION_IDS
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(str(tuple(array.shape)).encode("utf-8"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite Stage-G value")
        return value
    raise TypeError("unsupported Stage-G JSON value: {!r}".format(type(value)))


def stable_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            jsonable(dict(value)),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _git_output(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=str(root), text=True
    ).strip()


def _assert_ancestor(root: Path, ancestor: str, descendant: str) -> None:
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=str(root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise JointDirectionFeasibilityError(
            "required commit is not an ancestor: {} -> {}".format(
                ancestor, descendant
            )
        )


def validate_base_evidence(root: Path) -> Dict[str, Any]:
    repository_root = Path(root).resolve()
    stagef_base = stagef.validate_base_evidence(repository_root)
    head = _git_output(repository_root, "rev-parse", "HEAD")
    _assert_ancestor(repository_root, BASE_EVIDENCE_COMMIT, head)
    _assert_ancestor(repository_root, BASE_IMPLEMENTATION_COMMIT, head)
    stagef_source_sha = stagef.assert_file_bound_to_commit(
        repository_root,
        "ccda_phase3/phase314b_r258_stagef_constraint_aware_surrogate.py",
        BASE_EVIDENCE_COMMIT,
    )
    report_sha = stagef.assert_file_bound_to_commit(
        repository_root,
        BASE_REPORT,
        BASE_EVIDENCE_COMMIT,
    )
    if report_sha != EXPECTED_BASE_REPORT_SHA256:
        raise JointDirectionFeasibilityError("Stage-F V8 report SHA changed")
    report = stagef.load_json(repository_root / BASE_REPORT)
    if report.get("execution_verdict") != "PASS":
        raise JointDirectionFeasibilityError("Stage-F V8 execution is not PASS")
    if report.get("scientific_status") != "BLOCKED":
        raise JointDirectionFeasibilityError("Stage-F V8 scientific status changed")
    if report.get("root_cause") != EXPECTED_BASE_ROOT_CAUSE:
        raise JointDirectionFeasibilityError("Stage-F V8 root cause changed")
    if report.get("required_next_path") != EXPECTED_BASE_NEXT_PATH:
        raise JointDirectionFeasibilityError("Stage-F V8 next path changed")
    execution = report.get("execution")
    if not isinstance(execution, Mapping):
        raise JointDirectionFeasibilityError("Stage-F V8 execution payload missing")
    if execution.get("single_run_result_sha256") != EXPECTED_BASE_SINGLE_RUN_SHA256:
        raise JointDirectionFeasibilityError("Stage-F V8 single-run SHA changed")
    scientific = report.get("scientific_result")
    if not isinstance(scientific, Mapping):
        raise JointDirectionFeasibilityError("Stage-F V8 scientific result missing")
    policy = scientific.get("ulp_admission_policy")
    if not isinstance(policy, Mapping):
        raise JointDirectionFeasibilityError("Stage-F V8 ULP policy missing")
    if float(policy.get("selected_factor", -1.0)) != EXPECTED_SELECTED_ULP_FACTOR:
        raise JointDirectionFeasibilityError("Stage-F V8 ULP factor changed")
    if policy.get("policy_pass") is not True:
        raise JointDirectionFeasibilityError("Stage-F V8 ULP policy is not PASS")
    if scientific.get("selection_holdout_evaluated") is not False:
        raise JointDirectionFeasibilityError("Stage-F V8 holdout boundary changed")
    if scientific.get("frozen_probe_accessed") is not False:
        raise JointDirectionFeasibilityError("Stage-F V8 frozen probe boundary changed")
    return {
        "stagef_base": stagef_base,
        "stagef_v8_report_sha256": report_sha,
        "stagef_v8_single_run_sha256": EXPECTED_BASE_SINGLE_RUN_SHA256,
        "stagef_source_sha256": stagef_source_sha,
        "selected_ulp_factor": EXPECTED_SELECTED_ULP_FACTOR,
        "head": head,
    }


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray, epsilon: float) -> np.ndarray:
    return np.asarray(numerator, dtype=np.float64) / np.maximum(
        np.asarray(denominator, dtype=np.float64), float(epsilon)
    )


def _row_norm(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    return np.linalg.norm(array.reshape(array.shape[0], -1), axis=1)


def row_target_distance_reduction(
    *,
    control: np.ndarray,
    candidate: np.ndarray,
    target: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    before = _row_norm(np.asarray(control, dtype=np.float64) - np.asarray(target, dtype=np.float64))
    after = _row_norm(np.asarray(candidate, dtype=np.float64) - np.asarray(target, dtype=np.float64))
    return (before - after) / np.maximum(before, float(epsilon))


def _quantile_stats(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("descriptor statistic input must be [rows, columns]")
    return np.stack(
        (
            np.min(array, axis=1),
            np.mean(array, axis=1),
            np.std(array, axis=1),
            np.max(array, axis=1),
        ),
        axis=1,
    )


def candidate_descriptors(
    *,
    control: np.ndarray,
    proposed_direction: np.ndarray,
    scale_multiplier: float,
    integration: Mapping[str, Any],
    context: Mapping[str, Any],
    epsilon: float,
) -> np.ndarray:
    """Build target-independent descriptors after frozen integration."""
    control_raw = np.asarray(control, dtype=np.float64)
    direction_raw = np.asarray(proposed_direction, dtype=np.float64)
    candidate = np.asarray(integration["candidate"], dtype=np.float64)
    rows = control_raw.shape[0]
    if direction_raw.shape != control_raw.shape or candidate.shape != control_raw.shape:
        raise ValueError("Stage-G candidate descriptor shape changed")
    selected_scale = np.asarray(integration["selected_scale"], dtype=np.float64).reshape(rows)
    control_points = stagef._control_points(control_raw)
    proposed_points = stagef._control_points(control_raw + direction_raw)
    candidate_points = stagef._control_points(candidate)
    control_segments = stagef.stable_segment_vectors(control_points)
    proposed_segments = stagef.stable_segment_vectors(proposed_points)
    candidate_segments = stagef.stable_segment_vectors(candidate_points)
    control_lengths = np.linalg.norm(control_segments, axis=-1)
    proposed_lengths = np.linalg.norm(proposed_segments, axis=-1)
    candidate_lengths = np.linalg.norm(candidate_segments, axis=-1)
    zero_mask = stagef.structural_zero_mask(control_raw)
    resolvable = ~zero_mask
    proposed_ratio = np.ones_like(proposed_lengths, dtype=np.float64)
    candidate_ratio = np.ones_like(candidate_lengths, dtype=np.float64)
    proposed_ratio[resolvable] = _safe_divide(
        proposed_lengths[resolvable], control_lengths[resolvable], epsilon
    )
    candidate_ratio[resolvable] = _safe_divide(
        candidate_lengths[resolvable], control_lengths[resolvable], epsilon
    )
    constraint_context = {"stage_d_contract": context["stage_d_contract"]}
    proposed_state = stagef.constraint_state_features(
        control_raw + direction_raw, constraint_context
    )
    candidate_state = stagef.constraint_state_features(
        candidate, constraint_context
    )
    proposed_z = np.asarray(
        proposed_state["constraint_z_resolvable"], dtype=np.float64
    )
    candidate_z = np.asarray(
        candidate_state["constraint_z_resolvable"], dtype=np.float64
    )
    proposed_zero = np.asarray(
        proposed_state["structural_zero_mask"], dtype=np.float64
    )
    candidate_zero = np.asarray(
        candidate_state["structural_zero_mask"], dtype=np.float64
    )
    blocks: List[np.ndarray] = []
    blocks.append(np.full((rows, 1), float(scale_multiplier), dtype=np.float64))
    blocks.append(selected_scale[:, None])
    blocks.append(_row_norm(direction_raw)[:, None])
    blocks.append(_row_norm(candidate - control_raw)[:, None])
    blocks.append(
        np.linalg.norm(direction_raw.reshape(rows, stageb.FUTURE_STEPS, -1), axis=2)
    )
    blocks.append(
        np.linalg.norm(
            (candidate - control_raw).reshape(rows, stageb.FUTURE_STEPS, -1),
            axis=2,
        )
    )
    for value in (
        proposed_ratio,
        candidate_ratio,
        np.abs(proposed_z),
        np.abs(candidate_z),
    ):
        blocks.append(
            np.concatenate(
                [
                    _quantile_stats(value[:, horizon, :])
                    for horizon in range(stageb.FUTURE_STEPS)
                ],
                axis=1,
            )
        )
    blocks.append(np.mean(proposed_zero, axis=2))
    blocks.append(np.mean(candidate_zero, axis=2))
    blocks.append(np.mean(np.not_equal(proposed_zero, zero_mask), axis=2))
    blocks.append(np.mean(np.not_equal(candidate_zero, zero_mask), axis=2))
    descriptor = np.concatenate(blocks, axis=1).astype(np.float64)
    if descriptor.shape[0] != rows or descriptor.shape[1] != 92:
        raise JointDirectionFeasibilityError(
            "Stage-G descriptor dimension changed: {}".format(descriptor.shape)
        )
    if not np.all(np.isfinite(descriptor)):
        raise JointDirectionFeasibilityError("Stage-G descriptor is non-finite")
    return descriptor


def _fit_standardizer(value: np.ndarray, epsilon: float) -> Dict[str, np.ndarray]:
    x = np.asarray(value, dtype=np.float64)
    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    active = scale > float(epsilon)
    safe_scale = np.where(active, scale, 1.0)
    return {
        "mean": mean.astype(np.float64),
        "scale": safe_scale.astype(np.float64),
        "active": active.astype(np.bool_),
    }


def _apply_standardizer(value: np.ndarray, model: Mapping[str, Any]) -> np.ndarray:
    x = np.asarray(value, dtype=np.float64)
    mean = np.asarray(model["mean"], dtype=np.float64)
    scale = np.asarray(model["scale"], dtype=np.float64)
    active = np.asarray(model["active"], dtype=np.bool_)
    result = (x - mean[None]) / scale[None]
    result[:, ~active] = 0.0
    return result.astype(np.float64)


def _fit_scalar_ridge(
    features: np.ndarray,
    target: np.ndarray,
    *,
    alpha: float,
    epsilon: float,
) -> Dict[str, Any]:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    standardizer = _fit_standardizer(x, epsilon)
    xz = _apply_standardizer(x, standardizer)
    design = np.concatenate((np.ones((xz.shape[0], 1)), xz), axis=1)
    gram = design.T @ design
    rhs = design.T @ y
    regularizer = float(alpha) * np.eye(design.shape[1], dtype=np.float64)
    regularizer[0, 0] = 0.0
    coefficient = np.linalg.solve(gram + regularizer, rhs)
    return {
        "standardizer": standardizer,
        "coefficient": coefficient.astype(np.float64),
        "identity": {
            "coefficient_sha256": sha256_array(coefficient),
            "feature_mean_sha256": sha256_array(standardizer["mean"]),
            "feature_scale_sha256": sha256_array(standardizer["scale"]),
            "feature_active_sha256": sha256_array(standardizer["active"]),
            "fit_rows": int(x.shape[0]),
        },
    }


def _predict_scalar_ridge(model: Mapping[str, Any], features: np.ndarray) -> np.ndarray:
    xz = _apply_standardizer(features, model["standardizer"])
    design = np.concatenate((np.ones((xz.shape[0], 1)), xz), axis=1)
    result = design @ np.asarray(model["coefficient"], dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise JointDirectionFeasibilityError("Stage-G utility prediction is non-finite")
    return result.astype(np.float64)


def _sigmoid(value: np.ndarray) -> np.ndarray:
    x = np.asarray(value, dtype=np.float64)
    result = np.empty_like(x)
    positive = x >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_value = np.exp(x[~positive])
    result[~positive] = exp_value / (1.0 + exp_value)
    return result


def _fit_logistic_irls(
    features: np.ndarray,
    target: np.ndarray,
    *,
    l2: float,
    max_iterations: int,
    tolerance: float,
    epsilon: float,
) -> Dict[str, Any]:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    unique = np.unique(y)
    if not np.all(np.isin(unique, np.asarray([0.0, 1.0]))):
        raise ValueError("Stage-G feasibility target is not binary")
    prevalence = float(np.mean(y))
    if unique.size == 1:
        probability = min(max(prevalence, float(epsilon)), 1.0 - float(epsilon))
        return {
            "mode": "constant",
            "probability": probability,
            "identity": {
                "mode": "constant",
                "fit_rows": int(x.shape[0]),
                "prevalence": prevalence,
            },
        }
    standardizer = _fit_standardizer(x, epsilon)
    xz = _apply_standardizer(x, standardizer)
    design = np.concatenate((np.ones((xz.shape[0], 1)), xz), axis=1)
    coefficient = np.zeros(design.shape[1], dtype=np.float64)
    coefficient[0] = math.log(prevalence / (1.0 - prevalence))
    regularizer = float(l2) * np.eye(design.shape[1], dtype=np.float64)
    regularizer[0, 0] = 0.0
    converged = False
    iterations = 0
    for iteration in range(int(max_iterations)):
        probability = np.clip(_sigmoid(design @ coefficient), epsilon, 1.0 - epsilon)
        weight = np.maximum(probability * (1.0 - probability), epsilon)
        gradient = design.T @ (probability - y) + regularizer @ coefficient
        hessian = design.T @ (weight[:, None] * design) + regularizer
        step = np.linalg.solve(hessian, gradient)
        next_coefficient = coefficient - step
        iterations = iteration + 1
        if float(np.max(np.abs(next_coefficient - coefficient))) <= float(tolerance):
            coefficient = next_coefficient
            converged = True
            break
        coefficient = next_coefficient
    if not converged:
        probability = np.clip(_sigmoid(design @ coefficient), epsilon, 1.0 - epsilon)
        gradient = design.T @ (probability - y) + regularizer @ coefficient
        if float(np.max(np.abs(gradient))) > 1.0e-6:
            raise JointDirectionFeasibilityError("Stage-G logistic IRLS did not converge")
    return {
        "mode": "logistic",
        "standardizer": standardizer,
        "coefficient": coefficient.astype(np.float64),
        "identity": {
            "mode": "logistic",
            "coefficient_sha256": sha256_array(coefficient),
            "feature_mean_sha256": sha256_array(standardizer["mean"]),
            "feature_scale_sha256": sha256_array(standardizer["scale"]),
            "feature_active_sha256": sha256_array(standardizer["active"]),
            "fit_rows": int(x.shape[0]),
            "prevalence": prevalence,
            "iterations": int(iterations),
            "converged": bool(converged),
        },
    }


def _predict_logistic(model: Mapping[str, Any], features: np.ndarray) -> np.ndarray:
    rows = np.asarray(features).shape[0]
    if model["mode"] == "constant":
        return np.full(rows, float(model["probability"]), dtype=np.float64)
    xz = _apply_standardizer(features, model["standardizer"])
    design = np.concatenate((np.ones((xz.shape[0], 1)), xz), axis=1)
    result = _sigmoid(design @ np.asarray(model["coefficient"], dtype=np.float64))
    return np.clip(result, 0.0, 1.0).astype(np.float64)


def _pearson(left: np.ndarray, right: np.ndarray, epsilon: float) -> float:
    x = np.asarray(left, dtype=np.float64).reshape(-1)
    y = np.asarray(right, dtype=np.float64).reshape(-1)
    xc = x - np.mean(x)
    yc = y - np.mean(y)
    denominator = float(np.linalg.norm(xc) * np.linalg.norm(yc))
    if denominator <= float(epsilon):
        return 0.0
    return float(np.dot(xc, yc) / denominator)


def binary_metrics(
    *,
    probability: np.ndarray,
    target: np.ndarray,
    cutoff: float,
) -> Dict[str, Any]:
    p = np.asarray(probability, dtype=np.float64).reshape(-1)
    y = np.asarray(target, dtype=np.bool_).reshape(-1)
    prevalence = float(np.mean(y))
    brier = float(np.mean((p - y.astype(np.float64)) ** 2))
    baseline_brier = float(np.mean((prevalence - y.astype(np.float64)) ** 2))
    predicted = p >= float(cutoff)
    positive = y
    negative = ~y
    tpr = float(np.mean(predicted[positive])) if np.any(positive) else 1.0
    tnr = float(np.mean(~predicted[negative])) if np.any(negative) else 1.0
    return {
        "prevalence": prevalence,
        "positive_count": int(np.sum(positive)),
        "negative_count": int(np.sum(negative)),
        "brier": brier,
        "baseline_brier": baseline_brier,
        "brier_improvement": baseline_brier - brier,
        "balanced_accuracy": 0.5 * (tpr + tnr),
        "true_positive_rate": tpr,
        "true_negative_rate": tnr,
    }


def _fit_direction_model(
    *,
    definition: stagef.ConstraintAwareDefinition,
    features: np.ndarray,
    control: np.ndarray,
    oracle_target: Mapping[str, Any],
    condition_name: Sequence[Any],
    fit_mask: np.ndarray,
    spec: stagef.ConstraintAwareSpec,
    permutation_seed: Optional[int],
) -> Mapping[str, Any]:
    encoded = stagef.encode_projected_target(
        control=control,
        oracle_candidate=oracle_target["candidate"],
        target_mode=definition.target_mode,
    )
    training_target = np.asarray(encoded, dtype=np.float64).copy()
    if permutation_seed is not None:
        indices = np.flatnonzero(fit_mask)
        random = np.random.RandomState(int(permutation_seed))
        training_target[indices] = training_target[indices[random.permutation(indices.size)]]
    return stagef.fit_constraint_model(
        definition=definition,
        features=features,
        encoded_target=training_target,
        condition_name=condition_name,
        fit_mask=fit_mask,
        spec=spec,
    )


def _predict_direction(
    *,
    definition: stagef.ConstraintAwareDefinition,
    model: Mapping[str, Any],
    features: np.ndarray,
    control: np.ndarray,
    condition_name: Sequence[Any],
) -> np.ndarray:
    encoded = stagef.predict_constraint_model(
        definition=definition,
        model=model,
        features=features,
        condition_name=condition_name,
    )
    return stagef.decode_projected_prediction(
        control=control,
        encoded=encoded,
        target_mode=definition.target_mode,
    )


def build_integrated_bank(
    *,
    control: np.ndarray,
    base_direction: np.ndarray,
    context: Mapping[str, Any],
    integrator_spec: stagee258.ConstrainedIntegratorSpec,
    spec: JointDirectionFeasibilitySpec,
) -> Dict[str, Any]:
    control_raw = np.asarray(control, dtype=np.float32)
    direction_raw = np.asarray(base_direction, dtype=np.float64)
    rows = control_raw.shape[0]
    directions = []
    descriptors = []
    selected_scales = []
    candidates = []
    observable = []
    integration_identities = []
    for multiplier in spec.scale_multipliers:
        proposed = direction_raw * float(multiplier)
        integration = stagee258.integrate_rowwise(
            control=control_raw,
            direction=proposed,
            definition=stagef.fixed_integrator_definition(),
            context=context,
            spec=integrator_spec,
        )
        selected_scale = np.asarray(
            integration["selected_scale"], dtype=np.float64
        ).reshape(rows)
        candidate = np.asarray(integration["candidate"], dtype=np.float32)
        descriptor = candidate_descriptors(
            control=control_raw,
            proposed_direction=proposed,
            scale_multiplier=float(multiplier),
            integration=integration,
            context=context,
            epsilon=spec.descriptor_epsilon,
        )
        directions.append(proposed.astype(np.float64))
        descriptors.append(descriptor)
        selected_scales.append(selected_scale)
        candidates.append(candidate)
        observable.append(selected_scale > 0.0)
        integration_identities.append(
            {
                "scale_multiplier": float(multiplier),
                "direction_sha256": sha256_array(proposed),
                "candidate_sha256": sha256_array(candidate),
                "selected_scale_sha256": sha256_array(selected_scale),
                "observable_feasible_rate": float(np.mean(selected_scale > 0.0)),
            }
        )
    return {
        "directions": np.stack(directions, axis=1),
        "descriptors": np.stack(descriptors, axis=1),
        "selected_scales": np.stack(selected_scales, axis=1),
        "candidates": np.stack(candidates, axis=1),
        "observable_feasible": np.stack(observable, axis=1),
        "integration_identities": integration_identities,
    }


def bank_training_labels(
    *,
    bank: Mapping[str, Any],
    control: np.ndarray,
    target: np.ndarray,
    spec: JointDirectionFeasibilitySpec,
) -> Dict[str, np.ndarray]:
    candidates = np.asarray(bank["candidates"], dtype=np.float64)
    rows, bank_size = candidates.shape[:2]
    reduction = np.empty((rows, bank_size), dtype=np.float64)
    for bank_index in range(bank_size):
        reduction[:, bank_index] = row_target_distance_reduction(
            control=control,
            candidate=candidates[:, bank_index],
            target=target,
            epsilon=spec.descriptor_epsilon,
        )
    observable = np.asarray(bank["observable_feasible"], dtype=np.bool_)
    beneficial = observable & (reduction > float(spec.beneficial_reduction_min))
    return {
        "utility": reduction,
        "beneficial": beneficial,
    }


def fit_feasibility_ranker(
    *,
    descriptors: np.ndarray,
    labels: Mapping[str, np.ndarray],
    spec: JointDirectionFeasibilitySpec,
    permutation_seed: Optional[int] = None,
) -> Dict[str, Any]:
    x = np.asarray(descriptors, dtype=np.float64)
    utility = np.asarray(labels["utility"], dtype=np.float64)
    beneficial = np.asarray(labels["beneficial"], dtype=np.bool_)
    if x.shape[:2] != utility.shape or utility.shape != beneficial.shape:
        raise ValueError("Stage-G ranker training population changed")
    flat_x = x.reshape(-1, x.shape[-1])
    flat_utility = utility.reshape(-1).copy()
    flat_beneficial = beneficial.reshape(-1).astype(np.float64).copy()
    if permutation_seed is not None:
        random = np.random.RandomState(int(permutation_seed))
        permutation = random.permutation(flat_utility.shape[0])
        flat_utility = flat_utility[permutation]
        flat_beneficial = flat_beneficial[permutation]
    utility_model = _fit_scalar_ridge(
        flat_x,
        flat_utility,
        alpha=spec.utility_ridge_alpha,
        epsilon=spec.descriptor_epsilon,
    )
    feasibility_model = _fit_logistic_irls(
        flat_x,
        flat_beneficial,
        l2=spec.feasibility_l2,
        max_iterations=spec.feasibility_max_iterations,
        tolerance=spec.feasibility_tolerance,
        epsilon=spec.descriptor_epsilon,
    )
    return {
        "utility_model": utility_model,
        "feasibility_model": feasibility_model,
        "identity": {
            "utility_model": copy.deepcopy(utility_model["identity"]),
            "feasibility_model": copy.deepcopy(feasibility_model["identity"]),
            "training_descriptor_sha256": sha256_array(flat_x),
            "training_utility_sha256": sha256_array(flat_utility),
            "training_beneficial_sha256": sha256_array(flat_beneficial),
            "permuted_labels": bool(permutation_seed is not None),
        },
    }


def score_integrated_bank(
    *,
    ranker: Mapping[str, Any],
    bank: Mapping[str, Any],
    spec: JointDirectionFeasibilitySpec,
) -> Dict[str, np.ndarray]:
    descriptors = np.asarray(bank["descriptors"], dtype=np.float64)
    rows, bank_size, dimension = descriptors.shape
    flat = descriptors.reshape(-1, dimension)
    utility = _predict_scalar_ridge(ranker["utility_model"], flat).reshape(rows, bank_size)
    probability = _predict_logistic(ranker["feasibility_model"], flat).reshape(rows, bank_size)
    observable = np.asarray(bank["observable_feasible"], dtype=np.bool_)
    score = utility + float(spec.probability_weight) * (probability - 0.5)
    score = np.where(observable, score, -np.inf)
    eligible = observable & (probability >= float(spec.probability_cutoff))
    selected = np.empty(rows, dtype=np.int64)
    fallback = np.zeros(rows, dtype=np.bool_)
    for row in range(rows):
        if np.any(eligible[row]):
            masked = np.where(eligible[row], score[row], -np.inf)
            selected[row] = int(np.argmax(masked))
        elif np.any(observable[row]):
            masked = np.where(observable[row], probability[row], -np.inf)
            selected[row] = int(np.argmax(masked))
            fallback[row] = True
        else:
            selected[row] = 0
            fallback[row] = True
    return {
        "predicted_utility": utility,
        "predicted_probability": probability,
        "score": score,
        "eligible": eligible,
        "selected_index": selected,
        "fallback": fallback,
    }


def gather_selected_direction(
    *,
    bank: Mapping[str, Any],
    selected_index: np.ndarray,
) -> np.ndarray:
    directions = np.asarray(bank["directions"], dtype=np.float64)
    index = np.asarray(selected_index, dtype=np.int64).reshape(-1)
    if directions.shape[0] != index.shape[0]:
        raise ValueError("Stage-G selected direction row count changed")
    return directions[np.arange(index.shape[0]), index].astype(np.float64)


def ranker_metrics(
    *,
    scoring: Mapping[str, np.ndarray],
    labels: Mapping[str, np.ndarray],
    bank: Mapping[str, Any],
    spec: JointDirectionFeasibilitySpec,
) -> Dict[str, Any]:
    utility = np.asarray(labels["utility"], dtype=np.float64)
    beneficial = np.asarray(labels["beneficial"], dtype=np.bool_)
    predicted_utility = np.asarray(scoring["predicted_utility"], dtype=np.float64)
    probability = np.asarray(scoring["predicted_probability"], dtype=np.float64)
    selected = np.asarray(scoring["selected_index"], dtype=np.int64)
    row = np.arange(selected.shape[0])
    selected_utility = utility[row, selected]
    oracle_best = np.max(utility, axis=1)
    fixed_means = np.mean(utility, axis=0)
    best_fixed_index = int(np.argmax(fixed_means))
    best_fixed_mean = float(fixed_means[best_fixed_index])
    binary = binary_metrics(
        probability=probability.reshape(-1),
        target=beneficial.reshape(-1),
        cutoff=spec.probability_cutoff,
    )
    selected_observable = np.asarray(
        bank["observable_feasible"], dtype=np.bool_
    )[row, selected]
    result = {
        "utility_correlation": _pearson(
            predicted_utility.reshape(-1),
            utility.reshape(-1),
            spec.descriptor_epsilon,
        ),
        "binary": binary,
        "selected_mean_reduction": float(np.mean(selected_utility)),
        "selected_median_reduction": float(np.median(selected_utility)),
        "selected_positive_reduction_rate": float(np.mean(selected_utility > 0.0)),
        "selected_observable_feasible_rate": float(np.mean(selected_observable)),
        "mean_top1_regret": float(np.mean(oracle_best - selected_utility)),
        "best_fixed_scale_index": best_fixed_index,
        "best_fixed_scale_multiplier": float(spec.scale_multipliers[best_fixed_index]),
        "best_fixed_mean_reduction": best_fixed_mean,
        "fixed_scale_mean_reduction": {
            str(float(multiplier)): float(fixed_means[index])
            for index, multiplier in enumerate(spec.scale_multipliers)
        },
        "ranker_margin_over_best_fixed": float(
            np.mean(selected_utility) - best_fixed_mean
        ),
        "fallback_rate": float(np.mean(scoring["fallback"])),
        "selected_index_sha256": sha256_array(selected),
    }
    gates = {
        "utility_signal": (
            result["utility_correlation"] >= spec.utility_correlation_min
        ),
        "feasibility_signal": (
            binary["brier_improvement"] >= spec.brier_improvement_min
        ),
        "fixed_scale_margin": (
            result["ranker_margin_over_best_fixed"]
            >= spec.fixed_scale_reduction_margin_min
        ),
        "observable_feasible": result["selected_observable_feasible_rate"] >= 0.95,
    }
    result["gates"] = {**gates, "all": bool(all(gates.values()))}
    return result


def _fit_predict_joint_fold(
    *,
    definition: JointCandidateDefinition,
    features: np.ndarray,
    control: np.ndarray,
    target: np.ndarray,
    oracle_target: Mapping[str, Any],
    condition_name: Sequence[Any],
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    context: Mapping[str, Any],
    stagef_spec: stagef.ConstraintAwareSpec,
    integrator_spec: stagee258.ConstrainedIntegratorSpec,
    spec: JointDirectionFeasibilitySpec,
    ranker_permutation_seed: Optional[int],
) -> Dict[str, Any]:
    base_definition = stagef.definition_by_id(definition.base_direction_id)
    fit_mask = np.asarray(train_mask, dtype=np.bool_).copy()
    if base_definition.fit_population == "oracle_feasible":
        fit_mask &= np.asarray(oracle_target["feasible_mask"], dtype=np.bool_)
    direction_model = _fit_direction_model(
        definition=base_definition,
        features=features,
        control=control,
        oracle_target=oracle_target,
        condition_name=condition_name,
        fit_mask=fit_mask,
        spec=stagef_spec,
        permutation_seed=None,
    )
    train_direction = _predict_direction(
        definition=base_definition,
        model=direction_model,
        features=features[train_mask],
        control=control[train_mask],
        condition_name=np.asarray(condition_name)[train_mask],
    )
    test_direction = _predict_direction(
        definition=base_definition,
        model=direction_model,
        features=features[test_mask],
        control=control[test_mask],
        condition_name=np.asarray(condition_name)[test_mask],
    )
    train_bank = build_integrated_bank(
        control=control[train_mask],
        base_direction=train_direction,
        context=context,
        integrator_spec=integrator_spec,
        spec=spec,
    )
    train_labels = bank_training_labels(
        bank=train_bank,
        control=control[train_mask],
        target=target[train_mask],
        spec=spec,
    )
    permutation_seed = None
    if ranker_permutation_seed is not None:
        permutation_seed = int(ranker_permutation_seed)
    ranker = fit_feasibility_ranker(
        descriptors=train_bank["descriptors"],
        labels=train_labels,
        spec=spec,
        permutation_seed=permutation_seed,
    )
    test_bank = build_integrated_bank(
        control=control[test_mask],
        base_direction=test_direction,
        context=context,
        integrator_spec=integrator_spec,
        spec=spec,
    )
    test_scoring = score_integrated_bank(ranker=ranker, bank=test_bank, spec=spec)
    test_labels = bank_training_labels(
        bank=test_bank,
        control=control[test_mask],
        target=target[test_mask],
        spec=spec,
    )
    selected_direction = gather_selected_direction(
        bank=test_bank,
        selected_index=test_scoring["selected_index"],
    )
    return {
        "selected_direction": selected_direction,
        "base_direction": test_direction,
        "ranker_metrics": ranker_metrics(
            scoring=test_scoring,
            labels=test_labels,
            bank=test_bank,
            spec=spec,
        ),
        "direction_model_identity": copy.deepcopy(direction_model["identity"]),
        "ranker_identity": copy.deepcopy(ranker["identity"]),
        "test_scoring_identity": {
            "predicted_utility_sha256": sha256_array(
                test_scoring["predicted_utility"]
            ),
            "predicted_probability_sha256": sha256_array(
                test_scoring["predicted_probability"]
            ),
            "selected_index_sha256": sha256_array(test_scoring["selected_index"]),
        },
        "train_bank_identity": copy.deepcopy(train_bank["integration_identities"]),
        "test_bank_identity": copy.deepcopy(test_bank["integration_identities"]),
    }


def fit_oof_joint_candidate(
    *,
    definition: JointCandidateDefinition,
    features: np.ndarray,
    control: np.ndarray,
    target: np.ndarray,
    oracle_target: Mapping[str, Any],
    condition_name: Sequence[Any],
    fold_assignment: np.ndarray,
    context: Mapping[str, Any],
    stagef_spec: stagef.ConstraintAwareSpec,
    integrator_spec: stagee258.ConstrainedIntegratorSpec,
    spec: JointDirectionFeasibilitySpec,
    ranker_permutation_seed: Optional[int] = None,
) -> Dict[str, Any]:
    definition.validate()
    rows = np.asarray(control).shape[0]
    selected_direction = np.empty_like(np.asarray(control), dtype=np.float64)
    base_direction = np.empty_like(np.asarray(control), dtype=np.float64)
    covered = np.zeros(rows, dtype=np.bool_)
    fold_records = []
    assignment = np.asarray(fold_assignment, dtype=np.int64)
    metric_accumulator: Dict[str, List[float]] = {
        "utility_correlation": [],
        "brier_improvement": [],
        "selected_mean_reduction": [],
        "best_fixed_mean_reduction": [],
        "ranker_margin": [],
        "observable_feasible_rate": [],
    }
    gate_values: List[bool] = []
    for fold in range(spec.grouped_cv_folds):
        test_mask = assignment == fold
        train_mask = ~test_mask
        if not np.any(test_mask) or not np.any(train_mask):
            raise JointDirectionFeasibilityError("Stage-G OOF fold is empty")
        fold_result = _fit_predict_joint_fold(
            definition=definition,
            features=features,
            control=control,
            target=target,
            oracle_target=oracle_target,
            condition_name=condition_name,
            train_mask=train_mask,
            test_mask=test_mask,
            context=context,
            stagef_spec=stagef_spec,
            integrator_spec=integrator_spec,
            spec=spec,
            ranker_permutation_seed=(
                None
                if ranker_permutation_seed is None
                else int(ranker_permutation_seed) + 1009 * int(fold)
            ),
        )
        selected_direction[test_mask] = fold_result["selected_direction"]
        base_direction[test_mask] = fold_result["base_direction"]
        covered[test_mask] = True
        metrics = fold_result["ranker_metrics"]
        metric_accumulator["utility_correlation"].append(
            float(metrics["utility_correlation"])
        )
        metric_accumulator["brier_improvement"].append(
            float(metrics["binary"]["brier_improvement"])
        )
        metric_accumulator["selected_mean_reduction"].append(
            float(metrics["selected_mean_reduction"])
        )
        metric_accumulator["best_fixed_mean_reduction"].append(
            float(metrics["best_fixed_mean_reduction"])
        )
        metric_accumulator["ranker_margin"].append(
            float(metrics["ranker_margin_over_best_fixed"])
        )
        metric_accumulator["observable_feasible_rate"].append(
            float(metrics["selected_observable_feasible_rate"])
        )
        gate_values.append(bool(metrics["gates"]["all"]))
        fold_records.append(
            {
                "fold": int(fold),
                "train_rows": int(np.sum(train_mask)),
                "test_rows": int(np.sum(test_mask)),
                "train_mask_sha256": sha256_array(train_mask),
                "test_mask_sha256": sha256_array(test_mask),
                "direction_model_identity": fold_result[
                    "direction_model_identity"
                ],
                "ranker_identity": fold_result["ranker_identity"],
                "ranker_metrics": metrics,
                "test_scoring_identity": fold_result["test_scoring_identity"],
                "ranker_training_labels_permuted": bool(
                    ranker_permutation_seed is not None
                ),
                "test_target_used_for_fit": False,
            }
        )
    if not np.all(covered):
        raise JointDirectionFeasibilityError("Stage-G OOF population incomplete")
    aggregate = {
        key: {
            "mean": float(np.mean(value)),
            "minimum": float(np.min(value)),
            "maximum": float(np.max(value)),
        }
        for key, value in metric_accumulator.items()
    }
    aggregate_gates = {
        "all_folds": bool(all(gate_values)),
        "minimum_utility_correlation": (
            aggregate["utility_correlation"]["minimum"]
            >= spec.utility_correlation_min
        ),
        "minimum_fixed_scale_margin": (
            aggregate["ranker_margin"]["minimum"]
            >= spec.fixed_scale_reduction_margin_min
        ),
        "minimum_observable_feasible": (
            aggregate["observable_feasible_rate"]["minimum"] >= 0.95
        ),
    }
    aggregate_gates["all"] = bool(all(aggregate_gates.values()))
    return {
        "selected_direction": selected_direction,
        "base_direction": base_direction,
        "selected_direction_sha256": sha256_array(selected_direction),
        "base_direction_sha256": sha256_array(base_direction),
        "fold_records": fold_records,
        "ranker_metrics": aggregate,
        "ranker_gates": aggregate_gates,
        "ranker_training_labels_permuted": bool(
            ranker_permutation_seed is not None
        ),
    }


def _direction_baselines(
    *,
    context: Mapping[str, Any],
    oracle_targets: Mapping[int, Mapping[str, Any]],
    direction_spec: staged258.DirectionSurrogateSpec,
    integrator_spec: stagee258.ConstrainedIntegratorSpec,
    stagef_spec: stagef.ConstraintAwareSpec,
) -> Tuple[Dict[int, float], Dict[int, float]]:
    placeholder = {int(timestep): -1.0 for timestep in stagef_spec.timesteps}
    global_record = stagef.candidate_record(
        definition=stagef.definition_by_id("projected_global_mean"),
        context=context,
        oracle_targets=oracle_targets,
        global_baselines=placeholder,
        condition_baselines=placeholder,
        direction_spec=direction_spec,
        integrator_spec=integrator_spec,
        spec=stagef_spec,
    )
    condition_record = stagef.candidate_record(
        definition=stagef.definition_by_id("projected_condition_mean"),
        context=context,
        oracle_targets=oracle_targets,
        global_baselines=placeholder,
        condition_baselines=placeholder,
        direction_spec=direction_spec,
        integrator_spec=integrator_spec,
        spec=stagef_spec,
    )
    global_baseline = {
        int(timestep): float(
            global_record["timestep_records"][str(timestep)]
            ["projected_direction_metrics"]["cosine"]["mean"]
        )
        for timestep in stagef_spec.timesteps
    }
    condition_baseline = {
        int(timestep): float(
            condition_record["timestep_records"][str(timestep)]
            ["projected_direction_metrics"]["cosine"]["mean"]
        )
        for timestep in stagef_spec.timesteps
    }
    return global_baseline, condition_baseline


def joint_candidate_record(
    *,
    definition: JointCandidateDefinition,
    context: Mapping[str, Any],
    oracle_targets: Mapping[int, Mapping[str, Any]],
    global_baselines: Mapping[int, float],
    condition_baselines: Mapping[int, float],
    direction_spec: staged258.DirectionSurrogateSpec,
    integrator_spec: stagee258.ConstrainedIntegratorSpec,
    stagef_spec: stagef.ConstraintAwareSpec,
    spec: JointDirectionFeasibilitySpec,
    ranker_permutation_seed: Optional[int] = None,
) -> Dict[str, Any]:
    definition.validate()
    base_definition = stagef.definition_by_id(definition.base_direction_id)
    timestep_records: Dict[str, Any] = {}
    for timestep in spec.timesteps:
        control = np.asarray(
            context["objective_control_predictions"][int(timestep)],
            dtype=np.float32,
        )
        target = np.asarray(context["objective_target"], dtype=np.float32)
        features = stagef.build_constraint_features(
            condition=context["objective_condition"],
            control=control,
            condition_name=context["objective_condition_name"],
            feature_mode=base_definition.feature_mode,
            context=context,
        )
        joint = fit_oof_joint_candidate(
            definition=definition,
            features=features,
            control=control,
            target=target,
            oracle_target=oracle_targets[int(timestep)],
            condition_name=context["objective_condition_name"],
            fold_assignment=context["objective_fold_assignment"],
            context=context,
            stagef_spec=stagef_spec,
            integrator_spec=integrator_spec,
            spec=spec,
            ranker_permutation_seed=ranker_permutation_seed,
        )
        metrics = staged258.row_direction_metrics(
            predicted_residual=joint["selected_direction"],
            target_residual=oracle_targets[int(timestep)]["projected_delta"],
            condition_name=context["objective_condition_name"],
            fold_assignment=context["objective_fold_assignment"],
            epsilon=stagef_spec.standardizer_epsilon,
        )
        direction_gates = stagef.projected_direction_gate(
            metrics=metrics,
            global_mean=float(global_baselines[int(timestep)]),
            condition_mean=float(condition_baselines[int(timestep)]),
            spec=stagef_spec,
        )
        integrated = stagef._fixed_integration_record(
            direction=joint["selected_direction"],
            control=control,
            target=target,
            groups=context["objective_groups"],
            condition_name=context["objective_condition_name"],
            timestep=int(timestep),
            context=context,
            direction_spec=direction_spec,
            integrator_spec=integrator_spec,
        )
        timestep_records[str(timestep)] = {
            "timestep": int(timestep),
            "feature_sha256": sha256_array(features),
            "joint_oof": {
                key: value
                for key, value in joint.items()
                if key not in ("selected_direction", "base_direction")
            },
            "projected_direction_metrics": metrics,
            "projected_direction_gates": direction_gates,
            **integrated,
        }
    eligible = bool(
        ranker_permutation_seed is None
        and all(
            item["projected_direction_gates"]["all"]
            and item["joint_oof"]["ranker_gates"]["all"]
            and item["scientific_pass"]
            for item in timestep_records.values()
        )
    )
    return {
        "candidate_id": definition.candidate_id,
        "definition": asdict(definition),
        "base_definition": asdict(base_definition),
        "timestep_records": timestep_records,
        "eligible": eligible,
        "ranker_training_labels_permuted": bool(
            ranker_permutation_seed is not None
        ),
        "holdout_used": False,
    }


def _selection_key(record: Mapping[str, Any]) -> Tuple[Any, ...]:
    timestep_records = list(record["timestep_records"].values())
    minimum_reduction = min(
        float(item["evaluation"]["target_distance_reduction_fraction"]["mean"])
        for item in timestep_records
    )
    maximum_nmse = max(
        float(item["evaluation"]["normalized_mse_ratio"])
        for item in timestep_records
    )
    minimum_ranker_margin = min(
        float(
            item["joint_oof"]["ranker_metrics"]["ranker_margin"]["minimum"]
        )
        for item in timestep_records
    )
    return (
        -minimum_reduction,
        maximum_nmse,
        -minimum_ranker_margin,
        record["candidate_id"],
    )


def select_joint_candidate(
    records: Sequence[Mapping[str, Any]],
) -> Optional[Dict[str, Any]]:
    eligible = [record for record in records if record["eligible"]]
    if not eligible:
        return None
    return copy.deepcopy(min(eligible, key=_selection_key))


def ranker_permutation_control(
    *,
    selected: Mapping[str, Any],
    context: Mapping[str, Any],
    oracle_targets: Mapping[int, Mapping[str, Any]],
    global_baselines: Mapping[int, float],
    condition_baselines: Mapping[int, float],
    direction_spec: staged258.DirectionSurrogateSpec,
    integrator_spec: stagee258.ConstrainedIntegratorSpec,
    stagef_spec: stagef.ConstraintAwareSpec,
    spec: JointDirectionFeasibilitySpec,
) -> Dict[str, Any]:
    definition = JointCandidateDefinition(**selected["definition"])
    permuted = joint_candidate_record(
        definition=definition,
        context=context,
        oracle_targets=oracle_targets,
        global_baselines=global_baselines,
        condition_baselines=condition_baselines,
        direction_spec=direction_spec,
        integrator_spec=integrator_spec,
        stagef_spec=stagef_spec,
        spec=spec,
        ranker_permutation_seed=spec.ranker_seed,
    )
    records: Dict[str, Any] = {}
    passes = []
    for timestep in spec.timesteps:
        original = selected["timestep_records"][str(timestep)]
        control = permuted["timestep_records"][str(timestep)]
        original_reduction = float(
            original["evaluation"]["target_distance_reduction_fraction"]["mean"]
        )
        permuted_reduction = float(
            control["evaluation"]["target_distance_reduction_fraction"]["mean"]
        )
        gates = {
            "reduction_margin": (
                original_reduction - permuted_reduction
                >= spec.permutation_reduction_margin_min
            ),
            "permuted_state_not_scientific": not bool(control["scientific_pass"]),
            "permuted_ranker_not_eligible": not bool(
                control["joint_oof"]["ranker_gates"]["all"]
            ),
        }
        records[str(timestep)] = {
            "original_reduction": original_reduction,
            "permuted_reduction": permuted_reduction,
            "reduction_margin": original_reduction - permuted_reduction,
            "gates": {**gates, "all": bool(all(gates.values()))},
            "permuted_record": control,
        }
        passes.append(bool(all(gates.values())))
    return {
        "candidate_id": selected["candidate_id"],
        "records": records,
        "all_pass": bool(all(passes)),
        "holdout_used": False,
    }


def _fit_full_joint_and_select(
    *,
    definition: JointCandidateDefinition,
    train_features: np.ndarray,
    train_control: np.ndarray,
    train_target: np.ndarray,
    train_oracle: Mapping[str, Any],
    train_names: Sequence[Any],
    test_features: np.ndarray,
    test_control: np.ndarray,
    test_names: Sequence[Any],
    context: Mapping[str, Any],
    stagef_spec: stagef.ConstraintAwareSpec,
    integrator_spec: stagee258.ConstrainedIntegratorSpec,
    spec: JointDirectionFeasibilitySpec,
) -> Dict[str, Any]:
    base_definition = stagef.definition_by_id(definition.base_direction_id)
    fit_mask = np.ones(train_control.shape[0], dtype=np.bool_)
    if base_definition.fit_population == "oracle_feasible":
        fit_mask &= np.asarray(train_oracle["feasible_mask"], dtype=np.bool_)
    model = _fit_direction_model(
        definition=base_definition,
        features=train_features,
        control=train_control,
        oracle_target=train_oracle,
        condition_name=train_names,
        fit_mask=fit_mask,
        spec=stagef_spec,
        permutation_seed=None,
    )
    train_direction = _predict_direction(
        definition=base_definition,
        model=model,
        features=train_features,
        control=train_control,
        condition_name=train_names,
    )
    test_direction = _predict_direction(
        definition=base_definition,
        model=model,
        features=test_features,
        control=test_control,
        condition_name=test_names,
    )
    train_bank = build_integrated_bank(
        control=train_control,
        base_direction=train_direction,
        context=context,
        integrator_spec=integrator_spec,
        spec=spec,
    )
    train_labels = bank_training_labels(
        bank=train_bank,
        control=train_control,
        target=train_target,
        spec=spec,
    )
    ranker = fit_feasibility_ranker(
        descriptors=train_bank["descriptors"],
        labels=train_labels,
        spec=spec,
    )
    test_bank = build_integrated_bank(
        control=test_control,
        base_direction=test_direction,
        context=context,
        integrator_spec=integrator_spec,
        spec=spec,
    )
    scoring = score_integrated_bank(ranker=ranker, bank=test_bank, spec=spec)
    selected_direction = gather_selected_direction(
        bank=test_bank,
        selected_index=scoring["selected_index"],
    )
    return {
        "selected_direction": selected_direction,
        "direction_model_identity": copy.deepcopy(model["identity"]),
        "ranker_identity": copy.deepcopy(ranker["identity"]),
        "test_selection": {
            "selected_index_sha256": sha256_array(scoring["selected_index"]),
            "predicted_utility_sha256": sha256_array(scoring["predicted_utility"]),
            "predicted_probability_sha256": sha256_array(
                scoring["predicted_probability"]
            ),
            "fallback_rate": float(np.mean(scoring["fallback"])),
        },
    }


def locked_holdout_evaluation(
    *,
    selected: Mapping[str, Any],
    context: Mapping[str, Any],
    objective_oracle_targets: Mapping[int, Mapping[str, Any]],
    direction_spec: staged258.DirectionSurrogateSpec,
    integrator_spec: stagee258.ConstrainedIntegratorSpec,
    stagef_spec: stagef.ConstraintAwareSpec,
    spec: JointDirectionFeasibilitySpec,
) -> Dict[str, Any]:
    definition = JointCandidateDefinition(**selected["definition"])
    base_definition = stagef.definition_by_id(definition.base_direction_id)
    records: Dict[str, Any] = {}
    passes = []
    for timestep in spec.timesteps:
        train_control = np.asarray(
            context["objective_control_predictions"][int(timestep)],
            dtype=np.float32,
        )
        holdout_control = np.asarray(
            context["control_predictions"][int(timestep)], dtype=np.float32
        )
        train_features = stagef.build_constraint_features(
            condition=context["objective_condition"],
            control=train_control,
            condition_name=context["objective_condition_name"],
            feature_mode=base_definition.feature_mode,
            context=context,
        )
        holdout_features = stagef.build_constraint_features(
            condition=context["holdout_condition"],
            control=holdout_control,
            condition_name=context["holdout_condition_name"],
            feature_mode=base_definition.feature_mode,
            context=context,
        )
        fitted = _fit_full_joint_and_select(
            definition=definition,
            train_features=train_features,
            train_control=train_control,
            train_target=np.asarray(context["objective_target"], dtype=np.float32),
            train_oracle=objective_oracle_targets[int(timestep)],
            train_names=context["objective_condition_name"],
            test_features=holdout_features,
            test_control=holdout_control,
            test_names=context["holdout_condition_name"],
            context=context,
            stagef_spec=stagef_spec,
            integrator_spec=integrator_spec,
            spec=spec,
        )
        selected_direction = np.asarray(
            fitted["selected_direction"], dtype=np.float64
        )
        # Candidate generation and ranker selection are complete before target access.
        holdout_target = np.asarray(context["holdout_target"], dtype=np.float32)
        integrated = stagef._fixed_integration_record(
            direction=selected_direction,
            control=holdout_control,
            target=holdout_target,
            groups=context["holdout_groups"],
            condition_name=context["holdout_condition_name"],
            timestep=int(timestep),
            context=context,
            direction_spec=direction_spec,
            integrator_spec=integrator_spec,
        )
        holdout_oracle = stagef.generate_projected_oracle_target(
            control=holdout_control,
            target=holdout_target,
            groups=context["holdout_groups"],
            condition_name=context["holdout_condition_name"],
            timestep=int(timestep),
            context=context,
            direction_spec=direction_spec,
            integrator_spec=integrator_spec,
            spec=stagef_spec,
        )
        metrics = staged258.row_direction_metrics(
            predicted_residual=selected_direction,
            target_residual=holdout_oracle["projected_delta"],
            condition_name=context["holdout_condition_name"],
            fold_assignment=None,
            epsilon=stagef_spec.standardizer_epsilon,
        )
        direction_gates = stagef.holdout_projected_direction_gate(
            metrics, spec=stagef_spec
        )
        reduction_mean = float(
            integrated["evaluation"]["target_distance_reduction_fraction"]["mean"]
        )
        scientific_pass = bool(
            direction_gates["all"]
            and integrated["scientific_pass"]
            and reduction_mean >= spec.holdout_reduction_mean_min
        )
        records[str(timestep)] = {
            "direction_model_identity": fitted["direction_model_identity"],
            "ranker_identity": fitted["ranker_identity"],
            "test_selection": fitted["test_selection"],
            "selected_direction_sha256": sha256_array(selected_direction),
            "holdout_target_used_for_fit": False,
            "holdout_target_used_for_ranker_selection": False,
            "holdout_target_used_after_locked_selection_for_evaluation": True,
            "projected_direction_metrics": metrics,
            "projected_direction_gates": direction_gates,
            **integrated,
            "scientific_pass": scientific_pass,
        }
        passes.append(scientific_pass)
    return {
        "candidate_id": selected["candidate_id"],
        "definition": copy.deepcopy(selected["definition"]),
        "timestep_records": records,
        "scientific_pass": bool(all(passes)),
        "selection_changed_after_holdout": False,
        "holdout_used_for_fit": False,
        "holdout_used_for_selection": False,
        "holdout_evaluated": True,
    }


def classify(
    *,
    records: Sequence[Mapping[str, Any]],
    oracle_targets: Mapping[int, Mapping[str, Any]],
    selected: Optional[Mapping[str, Any]],
    permutation: Optional[Mapping[str, Any]],
    projection_only: Optional[Mapping[str, Any]],
    holdout: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    ranker_signal = [
        record["candidate_id"]
        for record in records
        if all(
            item["joint_oof"]["ranker_gates"]["all"]
            for item in record["timestep_records"].values()
        )
    ]
    direction_signal = [
        record["candidate_id"]
        for record in records
        if all(
            item["projected_direction_gates"]["all"]
            for item in record["timestep_records"].values()
        )
    ]
    state_signal = [
        record["candidate_id"]
        for record in records
        if all(item["scientific_pass"] for item in record["timestep_records"].values())
    ]
    eligible = [record["candidate_id"] for record in records if record["eligible"]]
    oracle_all = all(
        bool(oracle_targets[int(timestep)]["evaluation"]["scientific_pass"])
        for timestep in stagef.TIMESTEPS
    )
    if not oracle_all:
        root = "phase314b_r258_stageg_projected_oracle_target_not_stable"
        next_path = "AUDIT_STAGEF_PROJECTED_ORACLE_CONTRACT"
        locus = "projected_oracle_target"
    elif selected is None:
        if not direction_signal:
            root = "phase314b_r258_stageg_direction_backbone_signal_not_reproduced"
            next_path = "AUDIT_STAGEF_TO_STAGEG_DIRECTION_IDENTITY"
            locus = "direction_backbone"
        elif not ranker_signal:
            root = "phase314b_r258_stageg_feasibility_progress_not_identifiable"
            next_path = "AUDIT_CANDIDATE_DESCRIPTOR_IDENTIFIABILITY"
            locus = "feasibility_progress_ranker"
        elif not state_signal:
            root = "phase314b_r258_stageg_joint_direction_feasibility_still_not_integrable"
            next_path = "CALIBRATE_MULTI_DIRECTION_FEASIBILITY_RANKER"
            locus = "joint_post_prediction_integrability"
        else:
            root = "phase314b_r258_stageg_joint_state_witness_fails_selection_policy"
            next_path = "AUDIT_JOINT_SURROGATE_SELECTION_POLICY"
            locus = "selection_policy"
    elif permutation is None or not permutation["all_pass"]:
        root = "phase314b_r258_stageg_ranker_permutation_control_failed"
        next_path = "AUDIT_JOINT_RANKER_TARGET_SHORTCUT"
        locus = "ranker_permutation"
    elif projection_only is None or not projection_only["pass"]:
        root = "phase314b_r258_stageg_projection_only_control_explains_witness"
        next_path = "AUDIT_STAGEG_PROJECTION_ONLY_SHORTCUT"
        locus = "projection_only_shortcut"
    elif holdout is None:
        raise JointDirectionFeasibilityError("selected Stage-G candidate lacks holdout")
    elif not holdout["scientific_pass"]:
        root = "phase314b_r258_stageg_joint_surrogate_does_not_generalize"
        next_path = "AUDIT_JOINT_SURROGATE_HOLDOUT_GENERALIZATION"
        locus = "holdout_generalization"
    else:
        root = "phase314b_r258_stageg_joint_surrogate_selected_and_holdout_validated"
        next_path = "INTEGRATE_SELECTED_JOINT_SURROGATE_IN_TRAIN_ONLY_DENOISER"
        locus = "joint_surrogate_validated"
    return {
        "root_cause": root,
        "required_next_path": next_path,
        "primary_failure_locus": locus,
        "ranker_signal_candidate_ids": ranker_signal,
        "direction_signal_candidate_ids": direction_signal,
        "state_signal_candidate_ids": state_signal,
        "eligible_candidate_ids": eligible,
        "projected_oracle_all_timesteps": bool(oracle_all),
        "selected_configuration": copy.deepcopy(selected),
        "permutation_pass": None if permutation is None else bool(permutation["all_pass"]),
        "projection_only_pass": None if projection_only is None else bool(projection_only["pass"]),
        "holdout_scientific_pass": None if holdout is None else bool(holdout["scientific_pass"]),
    }


def run_calibration(
    *,
    root: Path,
    environment: Mapping[str, Any],
    spec: Optional[JointDirectionFeasibilitySpec] = None,
    stagef_spec: Optional[stagef.ConstraintAwareSpec] = None,
    direction_spec: Optional[staged258.DirectionSurrogateSpec] = None,
    integrator_spec: Optional[stagee258.ConstrainedIntegratorSpec] = None,
) -> Dict[str, Any]:
    active = JointDirectionFeasibilitySpec() if spec is None else spec
    active_stagef = stagef.ConstraintAwareSpec() if stagef_spec is None else stagef_spec
    active_direction = (
        staged258.DirectionSurrogateSpec()
        if direction_spec is None
        else direction_spec
    )
    active_integrator = (
        stagee258.ConstrainedIntegratorSpec()
        if integrator_spec is None
        else integrator_spec
    )
    active.validate()
    active_stagef.validate()
    active_direction.validate()
    active_integrator.validate()
    if float(active_stagef.translation_float32_z_ulp_factor) != 8.0:
        raise JointDirectionFeasibilityError(
            "Stage-F default spec changed before policy lock"
        )
    repository_root = Path(root).resolve()
    base = validate_base_evidence(repository_root)
    stagef.stagec258.stagea258.validate_environment_payload(environment)
    if environment.get("compatibility_sha256") != stagef.EXPECTED_COMPATIBILITY_SHA256:
        raise JointDirectionFeasibilityError("portable compatibility SHA changed")
    cold = stagef.stagec258.stagea258.assert_cold_cuda_context_portable()
    captured = stagef.stageb258.capture_portable_control_model(root=repository_root)
    context = stagef.build_context(
        root=repository_root,
        captured=captured,
        direction_spec=active_direction,
    )
    ulp_policy = stagef.calibrate_float32_constraint_z_policy(
        controls_by_timestep={
            int(timestep): context["objective_control_predictions"][int(timestep)]
            for timestep in active_stagef.timesteps
        },
        context=context,
        spec=active_stagef,
    )
    if float(ulp_policy["selected_factor"]) != EXPECTED_SELECTED_ULP_FACTOR:
        raise JointDirectionFeasibilityError("Stage-G ULP policy replay changed")
    active_stagef = stagef.replace(
        active_stagef,
        translation_float32_z_ulp_factor=EXPECTED_SELECTED_ULP_FACTOR,
    )
    active_stagef.validate()
    translation = {
        str(int(timestep)): stagef.translation_invariance_audit(
            condition=context["objective_condition"],
            control=context["objective_control_predictions"][int(timestep)],
            condition_name=context["objective_condition_name"],
            context=context,
            spec=active_stagef,
        )
        for timestep in active_stagef.timesteps
    }
    oracle_targets: Dict[int, Mapping[str, Any]] = {}
    for timestep in active_stagef.timesteps:
        oracle_targets[int(timestep)] = stagef.generate_projected_oracle_target(
            control=context["objective_control_predictions"][int(timestep)],
            target=context["objective_target"],
            groups=context["objective_groups"],
            condition_name=context["objective_condition_name"],
            timestep=int(timestep),
            context=context,
            direction_spec=active_direction,
            integrator_spec=active_integrator,
            spec=active_stagef,
        )
    global_baselines, condition_baselines = _direction_baselines(
        context=context,
        oracle_targets=oracle_targets,
        direction_spec=active_direction,
        integrator_spec=active_integrator,
        stagef_spec=active_stagef,
    )
    records = [
        joint_candidate_record(
            definition=definition,
            context=context,
            oracle_targets=oracle_targets,
            global_baselines=global_baselines,
            condition_baselines=condition_baselines,
            direction_spec=active_direction,
            integrator_spec=active_integrator,
            stagef_spec=active_stagef,
            spec=active,
        )
        for definition in JOINT_CANDIDATE_DEFINITIONS
    ]
    order = {
        definition.candidate_id: index
        for index, definition in enumerate(JOINT_CANDIDATE_DEFINITIONS)
    }
    records.sort(key=lambda record: order[record["candidate_id"]])
    selected = select_joint_candidate(records)
    permutation = None
    projection = None
    holdout = None
    if selected is not None:
        permutation = ranker_permutation_control(
            selected=selected,
            context=context,
            oracle_targets=oracle_targets,
            global_baselines=global_baselines,
            condition_baselines=condition_baselines,
            direction_spec=active_direction,
            integrator_spec=active_integrator,
            stagef_spec=active_stagef,
            spec=active,
        )
        projection = stagef.projection_only_control(
            context=context,
            direction_spec=active_direction,
            integrator_spec=active_integrator,
            spec=active_stagef,
        )
        if permutation["all_pass"] and projection["pass"]:
            holdout = locked_holdout_evaluation(
                selected=selected,
                context=context,
                objective_oracle_targets=oracle_targets,
                direction_spec=active_direction,
                integrator_spec=active_integrator,
                stagef_spec=active_stagef,
                spec=active,
            )
    classification = classify(
        records=records,
        oracle_targets=oracle_targets,
        selected=selected,
        permutation=permutation,
        projection_only=projection,
        holdout=holdout,
    )
    validated = (
        copy.deepcopy(selected)
        if (
            selected is not None
            and permutation is not None
            and permutation["all_pass"]
            and projection is not None
            and projection["pass"]
            and holdout is not None
            and holdout["scientific_pass"]
        )
        else None
    )
    contract = {
        "schema": "phase314b_r258_stageg_joint_direction_feasibility_contract_v1",
        "phase": PHASE,
        "base_evidence_commit": BASE_EVIDENCE_COMMIT,
        "base_report_sha256": EXPECTED_BASE_REPORT_SHA256,
        "base_single_run_sha256": EXPECTED_BASE_SINGLE_RUN_SHA256,
        "environment": dict(environment),
        "cold_main_worker_context": cold,
        "control_capture": captured["control_identity"],
        "joint_spec": asdict(active),
        "stagef_spec": asdict(active_stagef),
        "direction_spec": asdict(active_direction),
        "integrator_spec": asdict(active_integrator),
        "fixed_integrator": asdict(stagef.fixed_integrator_definition()),
        "candidate_definitions": [
            asdict(definition) for definition in JOINT_CANDIDATE_DEFINITIONS
        ],
        "supervision_contract": {
            "direction_target": "Stage-F projected-oracle displacement",
            "ranker_utility_target": "objective-train candidate target-distance reduction",
            "ranker_beneficial_target": "positive reduction and positive frozen-integrator scale",
            "ranker_features": "control, predicted direction, integrated candidate, frozen integrator outputs",
            "ranker_target_available_at_deployment": False,
            "ranker_features_target_independent": True,
            "outer_test_target_used_for_fit": False,
            "selection_holdout_used_for_fit_or_selection": False,
            "frozen_probe_accessed": False,
        },
        "scale_bank": list(active.scale_multipliers),
        "selected_ulp_factor": EXPECTED_SELECTED_ULP_FACTOR,
        "stagef_scientific_source_modified": False,
        "stagee_integrator_modified": False,
        "diffusion_model_candidate_trained": False,
        "model_weights_persisted": False,
    }
    contract["contract_sha256"] = sha256_bytes(stable_json_bytes(contract))
    selection = {
        "schema": "phase314b_r258_stageg_joint_direction_feasibility_selection_v1",
        "selected_configuration": copy.deepcopy(selected),
        "validated_train_only_recommendation": copy.deepcopy(validated),
        "classification": classification,
        "candidate_ids": [record["candidate_id"] for record in records],
        "permutation_pass": None if permutation is None else permutation["all_pass"],
        "projection_only_pass": None if projection is None else projection["pass"],
        "holdout_scientific_pass": None if holdout is None else holdout["scientific_pass"],
    }
    selection["selection_sha256"] = sha256_bytes(stable_json_bytes(selection))
    ready = validated is not None
    return {
        "phase": PHASE,
        "schema": "phase314b_r258_stageg_joint_direction_feasibility_result_v1",
        "verdict": "PASS",
        "scientific_status": "READY" if ready else "BLOCKED",
        "root_cause": classification["root_cause"],
        "required_next_path": classification["required_next_path"],
        "immutable_inputs": {
            "base_evidence_commit": BASE_EVIDENCE_COMMIT,
            "base_report_sha256": EXPECTED_BASE_REPORT_SHA256,
            "base_single_run_sha256": EXPECTED_BASE_SINGLE_RUN_SHA256,
            "stagef_source_sha256": base["stagef_source_sha256"],
        },
        "environment": dict(environment),
        "cold_main_worker_context": cold,
        "control_capture": captured["control_identity"],
        "split": {
            "objective_train_rows": int(context["objective_target"].shape[0]),
            "objective_train_groups": int(
                len(set(context["objective_groups"].tolist()))
            ),
            "selection_holdout_rows": int(context["holdout_target"].shape[0]),
            "frozen_probe_rows": int(np.sum(context["frozen_probe_mask"])),
            "fold_assignment_sha256": sha256_array(
                context["objective_fold_assignment"]
            ),
        },
        "ulp_admission_policy": ulp_policy,
        "translation_invariance": translation,
        "projected_oracle_targets": {
            str(timestep): {
                key: value
                for key, value in oracle_targets[int(timestep)].items()
                if key not in ("candidate", "projected_delta", "feasible_mask")
            }
            for timestep in active.timesteps
        },
        "joint_candidate_records": records,
        "objective_train_selected_configuration": copy.deepcopy(selected),
        "ranker_permutation_control": permutation,
        "projection_only_control": projection,
        "locked_holdout_evaluation": holdout,
        "classification": classification,
        "joint_contract": contract,
        "selection": selection,
        "selected_configuration": copy.deepcopy(validated),
        "train_only_recommendation": copy.deepcopy(validated),
        "selection_holdout_evaluated": bool(holdout is not None),
        "selection_holdout_used_for_fit": False,
        "selection_holdout_used_for_selection": False,
        "frozen_probe_accessed": False,
        "new_diffusion_model_candidate_trained": False,
        "reverse_sampling_run": False,
        "formal_training_run": False,
        "idm_run": False,
        "candidate_execution": False,
        "deformable_ravens_executed": False,
        "phase4": False,
        "cps": False,
        "checkpoint_saved": False,
        "weights_persisted": False,
        "surrogate_weights_persisted": False,
        "prediction_tensor_persisted": False,
        "candidate_tensor_persisted": False,
        "npz_saved": False,
        "cache_saved": False,
        "image_saved": False,
        "video_saved": False,
    }
