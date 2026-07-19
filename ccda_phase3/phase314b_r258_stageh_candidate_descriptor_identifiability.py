"""Phase3.14b-r2.5.8 Stage H candidate-descriptor identifiability audit.

Stage G established that all nine frozen direction backbones retain grouped-OOF
projected-direction signal, while the feasibility/progress ranker has zero
utility correlation, zero fixed-scale margin, and zero observable-feasible rate
at t=10/25/50.  This module does not add another model.  It audits the complete
objective-train candidate pipeline to determine where identifiability is lost:

* before integration: are the OOF directions and scaled proposals distinct?
* inside the frozen Stage-E integrator: are proposals accepted, rejected, or
  collapsed to the same state?
* after integration: are candidate states and post-integrator descriptor blocks
  distinct across the frozen scale bank?
* in supervision: do utility and beneficial labels vary across rows/scales?
* in the Stage-G descriptor: is any apparent variation only proposal-side
  information that survives even when the integrated candidate is unchanged?

The audit uses objective-train only.  It does not fit a new feasibility ranker,
open selection holdout, access the frozen probe, modify the Stage-E integrator,
change the Stage-F ULP policy, execute actions, or persist model/candidate data.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ccda_phase3 import phase314b_r258_staged_direction_surrogate as staged258
from ccda_phase3 import phase314b_r258_stagee_constrained_integrator as stagee258
from ccda_phase3 import phase314b_r258_stagef_constraint_aware_surrogate as stagef
from ccda_phase3 import phase314b_r258_stageg_joint_direction_feasibility as stageg


PHASE = "Phase3.14b-r2.5.8 Stage H"
PHASE_ID = "phase314b_r258_stageh"
BASE_IMPLEMENTATION_COMMIT = "4db7d88c7b4dae98613e7950574c4f314d02dc66"
BASE_EVIDENCE_COMMIT = "8a49fc91e21eb12579c580f4b5f16a65c1b325f9"
EXPECTED_SUBMODULE_COMMIT = "633a88752445cf5d6776ed374fdbbdb35f93050c"
BASE_REPORT = "reports/phase3_14b_r258_stageg_joint_direction_feasibility_summary.json"
EXPECTED_BASE_REPORT_SHA256 = (
    "a726a77ed3beb06110b99ef53164e8e2e7996652d04c6879185565636b45f6bb"
)
EXPECTED_BASE_SINGLE_RUN_SHA256 = (
    "a362e846de59c519b41b6f71777369bbd7a211ac754ea6780076dd69d822950c"
)
EXPECTED_BASE_ROOT_CAUSE = (
    "phase314b_r258_stageg_feasibility_progress_not_identifiable"
)
EXPECTED_BASE_NEXT_PATH = "AUDIT_CANDIDATE_DESCRIPTOR_IDENTIFIABILITY"
EXPECTED_SELECTED_ULP_FACTOR = 1.0

BASE_DIRECTION_IDS: Tuple[str, ...] = stageg.BASE_DIRECTION_IDS
SCALE_MULTIPLIERS: Tuple[float, ...] = stageg.SCALE_MULTIPLIERS

# The Stage-G descriptor is frozen at 92 dimensions.  Separate the blocks that
# can vary before integration from those that can only vary after integration.
DESCRIPTOR_BLOCKS: Mapping[str, Tuple[int, int]] = {
    "nominal_scale": (0, 1),
    "selected_scale": (1, 2),
    "proposed_total_norm": (2, 3),
    "candidate_total_norm": (3, 4),
    "proposed_horizon_norm": (4, 8),
    "candidate_horizon_norm": (8, 12),
    "proposed_segment_ratio_stats": (12, 28),
    "candidate_segment_ratio_stats": (28, 44),
    "proposed_constraint_z_stats": (44, 60),
    "candidate_constraint_z_stats": (60, 76),
    "proposed_zero_fraction": (76, 80),
    "candidate_zero_fraction": (80, 84),
    "proposed_zero_change_fraction": (84, 88),
    "candidate_zero_change_fraction": (88, 92),
}
PROPOSAL_DESCRIPTOR_BLOCKS: Tuple[str, ...] = (
    "nominal_scale",
    "proposed_total_norm",
    "proposed_horizon_norm",
    "proposed_segment_ratio_stats",
    "proposed_constraint_z_stats",
    "proposed_zero_fraction",
    "proposed_zero_change_fraction",
)
INTEGRATED_DESCRIPTOR_BLOCKS: Tuple[str, ...] = (
    "selected_scale",
    "candidate_total_norm",
    "candidate_horizon_norm",
    "candidate_segment_ratio_stats",
    "candidate_constraint_z_stats",
    "candidate_zero_fraction",
    "candidate_zero_change_fraction",
)
INTEGRATION_DIAGNOSTIC_TOKENS: Tuple[str, ...] = (
    "accept",
    "eligible",
    "feasible",
    "gate",
    "pass",
    "reason",
    "reject",
    "retention",
    "scale",
    "valid",
)


class CandidateDescriptorIdentifiabilityError(RuntimeError):
    """Raised when Stage-H scientific, shape, or leakage invariants fail."""


@dataclass(frozen=True)
class CandidateDescriptorIdentifiabilitySpec:
    timesteps: Tuple[int, ...] = stagef.TIMESTEPS
    grouped_cv_folds: int = 6
    scale_multipliers: Tuple[float, ...] = SCALE_MULTIPLIERS
    numeric_identity_atol: float = 1.0e-12
    direction_nonzero_epsilon: float = 1.0e-12
    candidate_motion_epsilon: float = 1.0e-12
    utility_epsilon: float = 1.0e-12
    descriptor_variance_epsilon: float = 1.0e-14
    proposal_nonzero_rate_min: float = 0.95
    proposal_unique_count_min: float = 3.5
    diagnostic_leaf_limit: int = 96

    def validate(self) -> None:
        if self.timesteps != stagef.TIMESTEPS:
            raise ValueError("Stage-H timestep population changed")
        if self.grouped_cv_folds != 6:
            raise ValueError("Stage-H grouped fold count changed")
        if self.scale_multipliers != SCALE_MULTIPLIERS:
            raise ValueError("Stage-H scale bank changed")
        if len(self.scale_multipliers) != 4:
            raise ValueError("Stage-H scale bank size changed")
        for value in (
            self.numeric_identity_atol,
            self.direction_nonzero_epsilon,
            self.candidate_motion_epsilon,
            self.utility_epsilon,
            self.descriptor_variance_epsilon,
            self.proposal_nonzero_rate_min,
            self.proposal_unique_count_min,
        ):
            if float(value) <= 0.0:
                raise ValueError("Stage-H positive policy value is invalid")
        if not 0.0 < self.proposal_nonzero_rate_min <= 1.0:
            raise ValueError("Stage-H proposal nonzero rate is invalid")
        if not 1.0 <= self.proposal_unique_count_min <= len(self.scale_multipliers):
            raise ValueError("Stage-H proposal unique-count gate is invalid")
        if self.diagnostic_leaf_limit != 96:
            raise ValueError("Stage-H diagnostic leaf limit changed")
        validate_descriptor_layout()


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
            raise ValueError("non-finite Stage-H value")
        return value
    raise TypeError("unsupported Stage-H JSON value: {!r}".format(type(value)))


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
    return subprocess.check_output(["git", *args], cwd=str(root), text=True).strip()


def _assert_ancestor(root: Path, ancestor: str, descendant: str) -> None:
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=str(root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise CandidateDescriptorIdentifiabilityError(
            "required commit is not an ancestor: {} -> {}".format(
                ancestor, descendant
            )
        )


def validate_base_evidence(root: Path) -> Dict[str, Any]:
    repository_root = Path(root).resolve()
    stageg_base = stageg.validate_base_evidence(repository_root)
    head = _git_output(repository_root, "rev-parse", "HEAD")
    _assert_ancestor(repository_root, BASE_IMPLEMENTATION_COMMIT, head)
    _assert_ancestor(repository_root, BASE_EVIDENCE_COMMIT, head)
    stageg_source_sha = stagef.assert_file_bound_to_commit(
        repository_root,
        "ccda_phase3/phase314b_r258_stageg_joint_direction_feasibility.py",
        BASE_EVIDENCE_COMMIT,
    )
    report_sha = stagef.assert_file_bound_to_commit(
        repository_root,
        BASE_REPORT,
        BASE_EVIDENCE_COMMIT,
    )
    if report_sha != EXPECTED_BASE_REPORT_SHA256:
        raise CandidateDescriptorIdentifiabilityError("Stage-G report SHA changed")
    report = stagef.load_json(repository_root / BASE_REPORT)
    if report.get("execution_verdict") != "PASS":
        raise CandidateDescriptorIdentifiabilityError("Stage-G execution is not PASS")
    if report.get("scientific_status") != "BLOCKED":
        raise CandidateDescriptorIdentifiabilityError(
            "Stage-G scientific status changed"
        )
    if report.get("root_cause") != EXPECTED_BASE_ROOT_CAUSE:
        raise CandidateDescriptorIdentifiabilityError("Stage-G root cause changed")
    if report.get("required_next_path") != EXPECTED_BASE_NEXT_PATH:
        raise CandidateDescriptorIdentifiabilityError("Stage-G next path changed")
    execution = report.get("execution")
    if not isinstance(execution, Mapping):
        raise CandidateDescriptorIdentifiabilityError(
            "Stage-G execution payload missing"
        )
    if execution.get("single_run_result_sha256") != EXPECTED_BASE_SINGLE_RUN_SHA256:
        raise CandidateDescriptorIdentifiabilityError(
            "Stage-G single-run SHA changed"
        )
    scientific = report.get("scientific_result")
    if not isinstance(scientific, Mapping):
        raise CandidateDescriptorIdentifiabilityError(
            "Stage-G scientific result missing"
        )
    if scientific.get("selected_configuration") is not None:
        raise CandidateDescriptorIdentifiabilityError(
            "Stage-G selected configuration changed"
        )
    if scientific.get("train_only_recommendation") is not None:
        raise CandidateDescriptorIdentifiabilityError(
            "Stage-G train-only recommendation changed"
        )
    if scientific.get("selection_holdout_evaluated") is not False:
        raise CandidateDescriptorIdentifiabilityError(
            "Stage-G holdout boundary changed"
        )
    if scientific.get("frozen_probe_accessed") is not False:
        raise CandidateDescriptorIdentifiabilityError(
            "Stage-G frozen-probe boundary changed"
        )
    return {
        "stageg_base": stageg_base,
        "stageg_report_sha256": report_sha,
        "stageg_single_run_sha256": EXPECTED_BASE_SINGLE_RUN_SHA256,
        "stageg_source_sha256": stageg_source_sha,
        "head": head,
    }


def validate_descriptor_layout() -> None:
    cursor = 0
    for name, bounds in DESCRIPTOR_BLOCKS.items():
        start, stop = bounds
        if start != cursor or stop <= start:
            raise ValueError("Stage-G descriptor block layout changed: {}".format(name))
        cursor = stop
    if cursor != 92:
        raise ValueError("Stage-G descriptor dimension changed")
    if set(PROPOSAL_DESCRIPTOR_BLOCKS) & set(INTEGRATED_DESCRIPTOR_BLOCKS):
        raise ValueError("Stage-H descriptor block groups overlap")
    if set(PROPOSAL_DESCRIPTOR_BLOCKS) | set(INTEGRATED_DESCRIPTOR_BLOCKS) != set(
        DESCRIPTOR_BLOCKS
    ):
        raise ValueError("Stage-H descriptor block groups are incomplete")


def descriptor_block(value: np.ndarray, names: Sequence[str]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 3 or array.shape[2] != 92:
        raise ValueError("Stage-H descriptor tensor must be [rows, bank, 92]")
    blocks = [
        array[:, :, DESCRIPTOR_BLOCKS[name][0] : DESCRIPTOR_BLOCKS[name][1]]
        for name in names
    ]
    return np.concatenate(blocks, axis=2)


def _flatten_bank(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim < 2:
        raise ValueError("Stage-H bank tensor must have row and bank axes")
    return array.reshape(array.shape[0], array.shape[1], -1)


def row_unique_counts(value: np.ndarray, *, atol: float) -> np.ndarray:
    flat = _flatten_bank(value).astype(np.float64, copy=False)
    counts = np.empty(flat.shape[0], dtype=np.int64)
    for row in range(flat.shape[0]):
        representatives: List[np.ndarray] = []
        for item in flat[row]:
            if not any(
                np.all(np.abs(item - representative) <= float(atol))
                for representative in representatives
            ):
                representatives.append(item.copy())
        counts[row] = len(representatives)
    return counts


def pairwise_distance_summary(value: np.ndarray, *, epsilon: float) -> Dict[str, Any]:
    flat = _flatten_bank(value).astype(np.float64, copy=False)
    distances: List[np.ndarray] = []
    for left in range(flat.shape[1]):
        for right in range(left + 1, flat.shape[1]):
            distances.append(np.linalg.norm(flat[:, left] - flat[:, right], axis=1))
    if not distances:
        raise ValueError("Stage-H bank lacks pairwise comparisons")
    stacked = np.stack(distances, axis=1)
    return {
        "minimum": float(np.min(stacked)),
        "mean": float(np.mean(stacked)),
        "median": float(np.median(stacked)),
        "maximum": float(np.max(stacked)),
        "positive_pair_rate": float(np.mean(stacked > float(epsilon))),
        "per_row_minimum_mean": float(np.mean(np.min(stacked, axis=1))),
        "per_row_maximum_mean": float(np.mean(np.max(stacked, axis=1))),
    }


def _unique_float_count(value: np.ndarray, epsilon: float) -> int:
    array = np.sort(np.asarray(value, dtype=np.float64).reshape(-1))
    if array.size == 0:
        return 0
    return int(1 + np.sum(np.diff(array) > float(epsilon)))


def _binary_entropy(prevalence: float) -> float:
    p = float(prevalence)
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p)))


def _pearson(left: np.ndarray, right: np.ndarray, epsilon: float) -> float:
    x = np.asarray(left, dtype=np.float64).reshape(-1)
    y = np.asarray(right, dtype=np.float64).reshape(-1)
    xc = x - np.mean(x)
    yc = y - np.mean(y)
    denominator = float(np.linalg.norm(xc) * np.linalg.norm(yc))
    if denominator <= float(epsilon):
        return 0.0
    return float(np.dot(xc, yc) / denominator)


def maximum_feature_signal(
    descriptor: np.ndarray,
    target: np.ndarray,
    *,
    epsilon: float,
) -> Dict[str, Any]:
    x = np.asarray(descriptor, dtype=np.float64).reshape(-1, descriptor.shape[-1])
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    correlations = np.zeros(x.shape[1], dtype=np.float64)
    active = np.std(x, axis=0) > float(epsilon)
    if np.std(y) > float(epsilon):
        for index in np.flatnonzero(active):
            correlations[index] = abs(_pearson(x[:, index], y, epsilon))
    best = int(np.argmax(correlations)) if correlations.size else -1
    return {
        "active_dimension_count": int(np.sum(active)),
        "maximum_absolute_correlation": (
            0.0 if best < 0 else float(correlations[best])
        ),
        "best_dimension": best,
        "correlation_sha256": sha256_array(correlations),
    }


def _numeric_array_summary(value: Any) -> Dict[str, Any]:
    array = np.asarray(value)
    result: Dict[str, Any] = {
        "shape": [int(item) for item in array.shape],
        "dtype": str(array.dtype),
        "sha256": sha256_array(array),
        "element_count": int(array.size),
    }
    if array.size and (
        np.issubdtype(array.dtype, np.number) or array.dtype == np.bool_
    ):
        numeric = array.astype(np.float64, copy=False)
        finite = numeric[np.isfinite(numeric)]
        if finite.size:
            result.update(
                {
                    "minimum": float(np.min(finite)),
                    "mean": float(np.mean(finite)),
                    "maximum": float(np.max(finite)),
                    "zero_rate": float(np.mean(finite == 0.0)),
                    "positive_rate": float(np.mean(finite > 0.0)),
                }
            )
        if array.dtype == np.bool_:
            result["true_rate"] = float(np.mean(array))
    return result


def integration_diagnostic_leaves(
    integration: Mapping[str, Any],
    *,
    limit: int,
) -> Dict[str, Any]:
    leaves: Dict[str, Any] = {}

    def visit(value: Any, path: str, depth: int) -> None:
        if len(leaves) >= int(limit) or depth > 5:
            return
        if isinstance(value, Mapping):
            for key in sorted(value, key=lambda item: str(item)):
                key_text = str(key)
                if path == "$" and key_text in {"candidate", "direction"}:
                    continue
                visit(value[key], path + "." + key_text, depth + 1)
            return
        path_lower = path.lower()
        if not any(token in path_lower for token in INTEGRATION_DIAGNOSTIC_TOKENS):
            return
        if isinstance(value, (str, bool, int, float)) or value is None:
            leaves[path] = jsonable(value)
            return
        if isinstance(value, np.ndarray) or (
            hasattr(value, "shape") and hasattr(value, "dtype")
        ):
            leaves[path] = _numeric_array_summary(value)

    visit(integration, "$", 0)
    return leaves


def fit_oof_base_direction(
    *,
    base_direction_id: str,
    features: np.ndarray,
    control: np.ndarray,
    oracle_target: Mapping[str, Any],
    condition_name: Sequence[Any],
    fold_assignment: np.ndarray,
    stagef_spec: stagef.ConstraintAwareSpec,
    grouped_cv_folds: int,
) -> Dict[str, Any]:
    definition = stagef.definition_by_id(base_direction_id)
    if definition.role != "selectable":
        raise CandidateDescriptorIdentifiabilityError(
            "Stage-H backbone is not selectable"
        )
    rows = np.asarray(control).shape[0]
    prediction = np.empty_like(np.asarray(control), dtype=np.float64)
    covered = np.zeros(rows, dtype=np.bool_)
    assignment = np.asarray(fold_assignment, dtype=np.int64)
    fold_records = []
    for fold in range(int(grouped_cv_folds)):
        test_mask = assignment == fold
        train_mask = ~test_mask
        if not np.any(test_mask) or not np.any(train_mask):
            raise CandidateDescriptorIdentifiabilityError(
                "Stage-H OOF fold is empty"
            )
        fit_mask = train_mask.copy()
        if definition.fit_population == "oracle_feasible":
            fit_mask &= np.asarray(oracle_target["feasible_mask"], dtype=np.bool_)
        model = stageg._fit_direction_model(
            definition=definition,
            features=features,
            control=control,
            oracle_target=oracle_target,
            condition_name=condition_name,
            fit_mask=fit_mask,
            spec=stagef_spec,
            permutation_seed=None,
        )
        fold_prediction = stageg._predict_direction(
            definition=definition,
            model=model,
            features=features[test_mask],
            control=control[test_mask],
            condition_name=np.asarray(condition_name)[test_mask],
        )
        prediction[test_mask] = fold_prediction
        covered[test_mask] = True
        fold_records.append(
            {
                "fold": int(fold),
                "train_rows": int(np.sum(train_mask)),
                "fit_rows": int(np.sum(fit_mask)),
                "test_rows": int(np.sum(test_mask)),
                "train_mask_sha256": sha256_array(train_mask),
                "fit_mask_sha256": sha256_array(fit_mask),
                "test_mask_sha256": sha256_array(test_mask),
                "model_identity": copy.deepcopy(model["identity"]),
                "test_target_used_for_fit": False,
            }
        )
    if not np.all(covered):
        raise CandidateDescriptorIdentifiabilityError(
            "Stage-H OOF population is incomplete"
        )
    return {
        "prediction": prediction,
        "prediction_sha256": sha256_array(prediction),
        "fold_records": fold_records,
    }


def build_audit_bank(
    *,
    control: np.ndarray,
    base_direction: np.ndarray,
    context: Mapping[str, Any],
    integrator_spec: stagee258.ConstrainedIntegratorSpec,
    spec: CandidateDescriptorIdentifiabilitySpec,
) -> Dict[str, Any]:
    control_raw = np.asarray(control, dtype=np.float32)
    direction_raw = np.asarray(base_direction, dtype=np.float64)
    rows = control_raw.shape[0]
    proposals: List[np.ndarray] = []
    descriptors: List[np.ndarray] = []
    selected_scales: List[np.ndarray] = []
    candidates: List[np.ndarray] = []
    diagnostics: List[Mapping[str, Any]] = []
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
        descriptor = stageg.candidate_descriptors(
            control=control_raw,
            proposed_direction=proposed,
            scale_multiplier=float(multiplier),
            integration=integration,
            context=context,
            epsilon=spec.numeric_identity_atol,
        )
        proposals.append(proposed.astype(np.float64))
        descriptors.append(descriptor)
        selected_scales.append(selected_scale)
        candidates.append(candidate)
        diagnostics.append(
            {
                "scale_multiplier": float(multiplier),
                "mapping_keys": sorted(str(key) for key in integration.keys()),
                "diagnostic_leaves": integration_diagnostic_leaves(
                    integration, limit=spec.diagnostic_leaf_limit
                ),
            }
        )
    return {
        "proposals": np.stack(proposals, axis=1),
        "descriptors": np.stack(descriptors, axis=1),
        "selected_scales": np.stack(selected_scales, axis=1),
        "candidates": np.stack(candidates, axis=1),
        "observable_feasible": np.stack(selected_scales, axis=1) > 0.0,
        "integration_diagnostics": diagnostics,
    }


def proposal_audit(
    *,
    base_direction: np.ndarray,
    proposals: np.ndarray,
    spec: CandidateDescriptorIdentifiabilitySpec,
) -> Dict[str, Any]:
    base_norm = np.linalg.norm(
        np.asarray(base_direction, dtype=np.float64).reshape(base_direction.shape[0], -1),
        axis=1,
    )
    unique = row_unique_counts(proposals, atol=spec.numeric_identity_atol)
    result = {
        "base_direction_nonzero_rate": float(
            np.mean(base_norm > spec.direction_nonzero_epsilon)
        ),
        "base_direction_norm": {
            "minimum": float(np.min(base_norm)),
            "mean": float(np.mean(base_norm)),
            "maximum": float(np.max(base_norm)),
        },
        "per_row_unique_count": {
            "minimum": int(np.min(unique)),
            "mean": float(np.mean(unique)),
            "maximum": int(np.max(unique)),
            "full_bank_rate": float(np.mean(unique == len(spec.scale_multipliers))),
        },
        "pairwise_distance": pairwise_distance_summary(
            proposals, epsilon=spec.numeric_identity_atol
        ),
        "proposal_sha256": sha256_array(proposals),
    }
    result["non_degenerate"] = bool(
        result["base_direction_nonzero_rate"] >= spec.proposal_nonzero_rate_min
        and result["per_row_unique_count"]["mean"]
        >= spec.proposal_unique_count_min
    )
    return result


def candidate_audit(
    *,
    control: np.ndarray,
    bank: Mapping[str, Any],
    spec: CandidateDescriptorIdentifiabilitySpec,
) -> Dict[str, Any]:
    control_raw = np.asarray(control, dtype=np.float64)
    candidates = np.asarray(bank["candidates"], dtype=np.float64)
    selected_scale = np.asarray(bank["selected_scales"], dtype=np.float64)
    displacement = candidates - control_raw[:, None]
    displacement_norm = np.linalg.norm(
        displacement.reshape(displacement.shape[0], displacement.shape[1], -1),
        axis=2,
    )
    motion = displacement_norm > spec.candidate_motion_epsilon
    observable = selected_scale > 0.0
    unique = row_unique_counts(candidates, atol=spec.numeric_identity_atol)
    exact_control = np.all(candidates == control_raw[:, None], axis=tuple(range(2, candidates.ndim)))
    numeric_control = displacement_norm <= spec.candidate_motion_epsilon
    positive_without_motion = observable & ~motion
    motion_without_positive = motion & ~observable
    effective_scale = selected_scale * np.asarray(spec.scale_multipliers)[None]
    return {
        "selected_scale": {
            "positive_rate": float(np.mean(observable)),
            "zero_rate": float(np.mean(selected_scale == 0.0)),
            "minimum": float(np.min(selected_scale)),
            "mean": float(np.mean(selected_scale)),
            "maximum": float(np.max(selected_scale)),
            "per_multiplier_positive_rate": {
                str(float(multiplier)): float(np.mean(observable[:, index]))
                for index, multiplier in enumerate(spec.scale_multipliers)
            },
            "effective_base_scale_minimum": float(np.min(effective_scale)),
            "effective_base_scale_mean": float(np.mean(effective_scale)),
            "effective_base_scale_maximum": float(np.max(effective_scale)),
        },
        "candidate_motion": {
            "positive_rate": float(np.mean(motion)),
            "exact_control_rate": float(np.mean(exact_control)),
            "numeric_control_rate": float(np.mean(numeric_control)),
            "minimum_norm": float(np.min(displacement_norm)),
            "mean_norm": float(np.mean(displacement_norm)),
            "maximum_norm": float(np.max(displacement_norm)),
        },
        "per_row_unique_count": {
            "minimum": int(np.min(unique)),
            "mean": float(np.mean(unique)),
            "maximum": int(np.max(unique)),
            "single_candidate_rate": float(np.mean(unique == 1)),
        },
        "pairwise_distance": pairwise_distance_summary(
            candidates, epsilon=spec.numeric_identity_atol
        ),
        "observable_motion_consistency": {
            "positive_scale_without_motion_count": int(np.sum(positive_without_motion)),
            "motion_without_positive_scale_count": int(np.sum(motion_without_positive)),
            "consistent": bool(
                not np.any(positive_without_motion)
                and not np.any(motion_without_positive)
            ),
        },
        "candidate_sha256": sha256_array(candidates),
        "selected_scale_sha256": sha256_array(selected_scale),
    }


def descriptor_audit(
    *,
    descriptor: np.ndarray,
    utility: np.ndarray,
    beneficial: np.ndarray,
    spec: CandidateDescriptorIdentifiabilitySpec,
) -> Dict[str, Any]:
    descriptor_raw = np.asarray(descriptor, dtype=np.float64)
    proposal = descriptor_block(descriptor_raw, PROPOSAL_DESCRIPTOR_BLOCKS)
    integrated = descriptor_block(descriptor_raw, INTEGRATED_DESCRIPTOR_BLOCKS)

    def population(value: np.ndarray) -> Dict[str, Any]:
        unique = row_unique_counts(value, atol=spec.numeric_identity_atol)
        flat = value.reshape(-1, value.shape[-1])
        variance = np.var(flat, axis=0)
        return {
            "dimension": int(value.shape[-1]),
            "active_dimension_count": int(
                np.sum(variance > spec.descriptor_variance_epsilon)
            ),
            "per_row_unique_count": {
                "minimum": int(np.min(unique)),
                "mean": float(np.mean(unique)),
                "maximum": int(np.max(unique)),
                "single_descriptor_rate": float(np.mean(unique == 1)),
            },
            "pairwise_distance": pairwise_distance_summary(
                value, epsilon=spec.numeric_identity_atol
            ),
            "sha256": sha256_array(value),
        }

    return {
        "full": {
            **population(descriptor_raw),
            "utility_signal": maximum_feature_signal(
                descriptor_raw,
                utility,
                epsilon=spec.descriptor_variance_epsilon,
            ),
            "beneficial_signal": maximum_feature_signal(
                descriptor_raw,
                beneficial.astype(np.float64),
                epsilon=spec.descriptor_variance_epsilon,
            ),
        },
        "proposal_only": {
            **population(proposal),
            "utility_signal": maximum_feature_signal(
                proposal,
                utility,
                epsilon=spec.descriptor_variance_epsilon,
            ),
        },
        "post_integrator": {
            **population(integrated),
            "utility_signal": maximum_feature_signal(
                integrated,
                utility,
                epsilon=spec.descriptor_variance_epsilon,
            ),
        },
        "layout": {
            "blocks": {key: list(value) for key, value in DESCRIPTOR_BLOCKS.items()},
            "proposal_blocks": list(PROPOSAL_DESCRIPTOR_BLOCKS),
            "post_integrator_blocks": list(INTEGRATED_DESCRIPTOR_BLOCKS),
        },
    }


def label_audit(
    *,
    utility: np.ndarray,
    beneficial: np.ndarray,
    spec: CandidateDescriptorIdentifiabilitySpec,
) -> Dict[str, Any]:
    utility_raw = np.asarray(utility, dtype=np.float64)
    beneficial_raw = np.asarray(beneficial, dtype=np.bool_)
    row_range = np.max(utility_raw, axis=1) - np.min(utility_raw, axis=1)
    row_max = np.max(utility_raw, axis=1, keepdims=True)
    tie_count = np.sum(
        np.abs(utility_raw - row_max) <= spec.utility_epsilon,
        axis=1,
    )
    prevalence = float(np.mean(beneficial_raw))
    fixed_mean = np.mean(utility_raw, axis=0)
    return {
        "utility": {
            "minimum": float(np.min(utility_raw)),
            "mean": float(np.mean(utility_raw)),
            "maximum": float(np.max(utility_raw)),
            "standard_deviation": float(np.std(utility_raw)),
            "unique_count": _unique_float_count(
                utility_raw, spec.utility_epsilon
            ),
            "positive_rate": float(np.mean(utility_raw > spec.utility_epsilon)),
            "negative_rate": float(np.mean(utility_raw < -spec.utility_epsilon)),
            "zero_rate": float(np.mean(np.abs(utility_raw) <= spec.utility_epsilon)),
            "row_nonconstant_rate": float(np.mean(row_range > spec.utility_epsilon)),
            "oracle_best_tie_rate": float(np.mean(tie_count > 1)),
            "all_scales_tied_rate": float(
                np.mean(tie_count == utility_raw.shape[1])
            ),
            "fixed_scale_mean": {
                str(float(multiplier)): float(fixed_mean[index])
                for index, multiplier in enumerate(spec.scale_multipliers)
            },
            "sha256": sha256_array(utility_raw),
        },
        "beneficial": {
            "positive_count": int(np.sum(beneficial_raw)),
            "negative_count": int(beneficial_raw.size - np.sum(beneficial_raw)),
            "prevalence": prevalence,
            "entropy_bits": _binary_entropy(prevalence),
            "unique_count": int(np.unique(beneficial_raw).size),
            "sha256": sha256_array(beneficial_raw),
        },
    }


def identify_record_locus(
    *,
    proposal: Mapping[str, Any],
    candidate: Mapping[str, Any],
    descriptor: Mapping[str, Any],
    labels: Mapping[str, Any],
) -> str:
    consistency = candidate["observable_motion_consistency"]
    if consistency["consistent"] is not True:
        return "observable_feasibility_indicator_inconsistent"
    if proposal["non_degenerate"] is not True:
        return "proposal_bank_degenerate_before_integration"
    selected_positive = float(candidate["selected_scale"]["positive_rate"])
    motion_positive = float(candidate["candidate_motion"]["positive_rate"])
    if selected_positive == 0.0 and motion_positive == 0.0:
        return "frozen_integrator_rejects_all_scale_bank_candidates"
    if float(candidate["per_row_unique_count"]["mean"]) <= 1.0:
        return "frozen_integrator_collapses_scale_bank"
    utility = labels["utility"]
    beneficial = labels["beneficial"]
    if (
        int(utility["unique_count"]) <= 1
        and int(beneficial["unique_count"]) <= 1
    ):
        return "feasibility_progress_labels_degenerate"
    if (
        float(descriptor["post_integrator"]["per_row_unique_count"]["mean"])
        <= 1.0
    ):
        return "post_integrator_descriptor_collision"
    if (
        float(
            descriptor["post_integrator"]["utility_signal"][
                "maximum_absolute_correlation"
            ]
        )
        == 0.0
    ):
        return "post_integrator_descriptor_has_no_utility_signal"
    return "identifiable_population_present"


def audit_candidate_record(
    *,
    base_direction_id: str,
    timestep: int,
    context: Mapping[str, Any],
    oracle_target: Mapping[str, Any],
    stagef_spec: stagef.ConstraintAwareSpec,
    integrator_spec: stagee258.ConstrainedIntegratorSpec,
    spec: CandidateDescriptorIdentifiabilitySpec,
) -> Dict[str, Any]:
    definition = stagef.definition_by_id(base_direction_id)
    control = np.asarray(
        context["objective_control_predictions"][int(timestep)], dtype=np.float32
    )
    target = np.asarray(context["objective_target"], dtype=np.float32)
    features = stagef.build_constraint_features(
        condition=context["objective_condition"],
        control=control,
        condition_name=context["objective_condition_name"],
        feature_mode=definition.feature_mode,
        context=context,
    )
    oof = fit_oof_base_direction(
        base_direction_id=base_direction_id,
        features=features,
        control=control,
        oracle_target=oracle_target,
        condition_name=context["objective_condition_name"],
        fold_assignment=context["objective_fold_assignment"],
        stagef_spec=stagef_spec,
        grouped_cv_folds=spec.grouped_cv_folds,
    )
    bank = build_audit_bank(
        control=control,
        base_direction=oof["prediction"],
        context=context,
        integrator_spec=integrator_spec,
        spec=spec,
    )
    labels_raw = stageg.bank_training_labels(
        bank=bank,
        control=control,
        target=target,
        spec=stageg.JointDirectionFeasibilitySpec(),
    )
    proposal = proposal_audit(
        base_direction=oof["prediction"],
        proposals=bank["proposals"],
        spec=spec,
    )
    candidate = candidate_audit(
        control=control,
        bank=bank,
        spec=spec,
    )
    labels = label_audit(
        utility=labels_raw["utility"],
        beneficial=labels_raw["beneficial"],
        spec=spec,
    )
    descriptor = descriptor_audit(
        descriptor=bank["descriptors"],
        utility=labels_raw["utility"],
        beneficial=labels_raw["beneficial"],
        spec=spec,
    )
    locus = identify_record_locus(
        proposal=proposal,
        candidate=candidate,
        descriptor=descriptor,
        labels=labels,
    )
    return {
        "base_direction_id": base_direction_id,
        "timestep": int(timestep),
        "feature_mode": definition.feature_mode,
        "feature_sha256": sha256_array(features),
        "oof_direction": {
            "prediction_sha256": oof["prediction_sha256"],
            "fold_records": oof["fold_records"],
        },
        "proposal_audit": proposal,
        "candidate_audit": candidate,
        "descriptor_audit": descriptor,
        "label_audit": labels,
        "integration_diagnostics": bank["integration_diagnostics"],
        "primary_failure_locus": locus,
        "objective_train_rows": int(control.shape[0]),
        "holdout_used": False,
    }


def classify(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not records:
        raise CandidateDescriptorIdentifiabilityError(
            "Stage-H audit has no records"
        )
    loci: Dict[str, int] = {}
    for record in records:
        locus = str(record["primary_failure_locus"])
        loci[locus] = loci.get(locus, 0) + 1
    total = len(records)
    all_locus = lambda name: loci.get(name, 0) == total
    if loci.get("observable_feasibility_indicator_inconsistent", 0) > 0:
        root = (
            "phase314b_r258_stageh_observable_feasibility_indicator_inconsistent"
        )
        next_path = (
            "CORRECT_OBSERVABLE_FEASIBILITY_LABEL_WITHOUT_CHANGING_INTEGRATOR"
        )
        primary = "observable_feasibility_definition"
    elif all_locus("proposal_bank_degenerate_before_integration"):
        root = "phase314b_r258_stageh_oof_proposal_bank_degenerate"
        next_path = "AUDIT_OOF_DIRECTION_MAGNITUDE_AND_SCALE_BANK_COUPLING"
        primary = "pre_integration_proposals"
    elif all_locus("frozen_integrator_rejects_all_scale_bank_candidates"):
        root = (
            "phase314b_r258_stageh_frozen_integrator_rejects_entire_surrogate_scale_bank"
        )
        next_path = (
            "AUDIT_FROZEN_INTEGRATOR_REJECTION_PREDICATES_ON_OOF_DIRECTION_BANK"
        )
        primary = "frozen_integrator_rejection"
    elif all_locus("frozen_integrator_collapses_scale_bank"):
        root = "phase314b_r258_stageh_frozen_integrator_collapses_scale_bank"
        next_path = "AUDIT_SCALE_MULTIPLIER_AND_INTERNAL_SCALE_COUPLING"
        primary = "frozen_integrator_collapse"
    elif all_locus("feasibility_progress_labels_degenerate"):
        root = "phase314b_r258_stageh_feasibility_progress_labels_degenerate"
        next_path = (
            "REDESIGN_TARGET_INDEPENDENT_FEASIBILITY_SUPERVISION_FROM_FROZEN_CERTIFICATES"
        )
        primary = "supervision_labels"
    elif all_locus("post_integrator_descriptor_collision"):
        root = "phase314b_r258_stageh_post_integrator_descriptor_collision"
        next_path = "REDESIGN_POST_INTEGRATOR_CANDIDATE_DESCRIPTOR"
        primary = "candidate_descriptor"
    elif all_locus("post_integrator_descriptor_has_no_utility_signal"):
        root = "phase314b_r258_stageh_descriptor_has_no_progress_signal"
        next_path = "REDESIGN_TARGET_INDEPENDENT_FEASIBILITY_OBSERVATIONS"
        primary = "candidate_descriptor_signal"
    elif all_locus("identifiable_population_present"):
        root = "phase314b_r258_stageh_population_identifiable_ranker_path_suspect"
        next_path = "AUDIT_STAGEG_FOLD_LOCAL_RANKER_ASSEMBLY"
        primary = "stageg_ranker_assembly"
    else:
        root = "phase314b_r258_stageh_identifiability_failure_is_heterogeneous"
        next_path = "STRATIFY_INTEGRATOR_AND_DESCRIPTOR_FAILURE_BY_BACKBONE_TIMESTEP"
        primary = "heterogeneous_pipeline_failure"
    return {
        "root_cause": root,
        "required_next_path": next_path,
        "primary_failure_locus": primary,
        "record_locus_counts": dict(sorted(loci.items())),
        "record_count": total,
        "all_records_share_primary_locus": len(loci) == 1,
    }


def run_calibration(
    *,
    root: Path,
    environment: Mapping[str, Any],
    spec: Optional[CandidateDescriptorIdentifiabilitySpec] = None,
    stagef_spec: Optional[stagef.ConstraintAwareSpec] = None,
    direction_spec: Optional[staged258.DirectionSurrogateSpec] = None,
    integrator_spec: Optional[stagee258.ConstrainedIntegratorSpec] = None,
) -> Dict[str, Any]:
    active = CandidateDescriptorIdentifiabilitySpec() if spec is None else spec
    active_stagef = stagef.ConstraintAwareSpec() if stagef_spec is None else stagef_spec
    active_direction = (
        staged258.DirectionSurrogateSpec() if direction_spec is None else direction_spec
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
    repository_root = Path(root).resolve()
    base = validate_base_evidence(repository_root)
    stagef.stagec258.stagea258.validate_environment_payload(environment)
    if environment.get("compatibility_sha256") != stagef.EXPECTED_COMPATIBILITY_SHA256:
        raise CandidateDescriptorIdentifiabilityError(
            "portable compatibility SHA changed"
        )
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
        raise CandidateDescriptorIdentifiabilityError(
            "Stage-H ULP policy replay changed"
        )
    active_stagef = stagef.replace(
        active_stagef,
        translation_float32_z_ulp_factor=EXPECTED_SELECTED_ULP_FACTOR,
    )
    active_stagef.validate()
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
    records = [
        audit_candidate_record(
            base_direction_id=base_direction_id,
            timestep=int(timestep),
            context=context,
            oracle_target=oracle_targets[int(timestep)],
            stagef_spec=active_stagef,
            integrator_spec=active_integrator,
            spec=active,
        )
        for base_direction_id in BASE_DIRECTION_IDS
        for timestep in active.timesteps
    ]
    classification = classify(records)
    contract = {
        "schema": "phase314b_r258_stageh_candidate_descriptor_identifiability_contract_v1",
        "phase": PHASE,
        "base_evidence_commit": BASE_EVIDENCE_COMMIT,
        "base_report_sha256": EXPECTED_BASE_REPORT_SHA256,
        "base_single_run_sha256": EXPECTED_BASE_SINGLE_RUN_SHA256,
        "environment": dict(environment),
        "cold_main_worker_context": cold,
        "control_capture": captured["control_identity"],
        "audit_spec": asdict(active),
        "stagef_spec": asdict(active_stagef),
        "direction_spec": asdict(active_direction),
        "integrator_spec": asdict(active_integrator),
        "fixed_integrator": asdict(stagef.fixed_integrator_definition()),
        "base_direction_ids": list(BASE_DIRECTION_IDS),
        "scale_bank": list(SCALE_MULTIPLIERS),
        "descriptor_layout": {
            "blocks": {key: list(value) for key, value in DESCRIPTOR_BLOCKS.items()},
            "proposal_blocks": list(PROPOSAL_DESCRIPTOR_BLOCKS),
            "post_integrator_blocks": list(INTEGRATED_DESCRIPTOR_BLOCKS),
        },
        "data_boundary": {
            "objective_train_only": True,
            "selection_holdout_evaluated": False,
            "selection_holdout_used_for_fit_or_selection": False,
            "frozen_probe_accessed": False,
            "condition_label_used_for_descriptor_or_selection": False,
            "target_used_only_for_objective_train_label_audit": True,
        },
        "mechanism_boundary": {
            "new_ranker_fitted": False,
            "candidate_matrix_changed": False,
            "scale_bank_changed": False,
            "stagee_integrator_modified": False,
            "stagef_feature_definition_modified": False,
            "ulp_policy_changed": False,
        },
    }
    contract["contract_sha256"] = sha256_bytes(stable_json_bytes(contract))
    audit_payload = {
        "schema": "phase314b_r258_stageh_candidate_descriptor_identifiability_audit_v1",
        "records": records,
        "classification": classification,
    }
    audit_payload["audit_sha256"] = sha256_bytes(stable_json_bytes(audit_payload))
    return {
        "phase": PHASE,
        "schema": "phase314b_r258_stageh_candidate_descriptor_identifiability_result_v1",
        "verdict": "PASS",
        "scientific_status": "BLOCKED",
        "root_cause": classification["root_cause"],
        "required_next_path": classification["required_next_path"],
        "primary_failure_locus": classification["primary_failure_locus"],
        "selected_configuration": None,
        "train_only_recommendation": None,
        "immutable_inputs": {
            "base_evidence_commit": BASE_EVIDENCE_COMMIT,
            "base_report_sha256": EXPECTED_BASE_REPORT_SHA256,
            "base_single_run_sha256": EXPECTED_BASE_SINGLE_RUN_SHA256,
            "stageg_source_sha256": base["stageg_source_sha256"],
        },
        "contract": contract,
        "identifiability_audit": audit_payload,
        "candidate_descriptor_identifiability_completed": True,
        "scientific_calibration_completed": False,
        "selection_holdout_evaluated": False,
        "selection_holdout_used_for_fit_or_selection": False,
        "frozen_probe_accessed": False,
        "formal_training_run": False,
        "reverse_sampling_run": False,
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
