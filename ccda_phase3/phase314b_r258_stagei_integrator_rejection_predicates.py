"""Phase3.14b-r2.5.8 Stage I frozen-integrator rejection audit.

Stage H established that all nine grouped-OOF direction backbones produce
non-degenerate four-scale proposal banks, yet the frozen Stage-E integrator
returns control and selected_scale=0 for every proposal.  Stage I leaves the
integrator, direction models, scale bank, geometry contracts, ULP policy,
objective-train split, holdout, and frozen probe unchanged.  It audits the
actual rejection predicates returned by the frozen integrator and compares
three direction sources under the same target-independent execution path:

* grouped-OOF deployable surrogate direction;
* raw objective-train oracle residual (target - control), diagnostic only;
* frozen projected-oracle displacement, diagnostic only.

Ground truth is used only to construct the two objective-train diagnostic
controls.  It never enters a deployable descriptor, fit, selection policy,
holdout decision, or frozen-probe computation.  No direction, candidate,
predicate tensor, checkpoint, cache, NPZ, image, or video is persisted.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import inspect
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
from ccda_phase3 import phase314b_r258_stageh_candidate_descriptor_identifiability as stageh


PHASE = "Phase3.14b-r2.5.8 Stage I"
PHASE_ID = "phase314b_r258_stagei"
BASE_IMPLEMENTATION_COMMIT = "0203b661702b739fea0d68d08db026d7dae146f9"
BASE_EVIDENCE_COMMIT = "5d3a55ec3bef4204292266f6be0716e2cdfd8335"
INDEPENDENT_GOALS_COMMIT = "0ce0c9efa05c1b71d7e312bdf65bbe27eb094486"
EXPECTED_SUBMODULE_COMMIT = "633a88752445cf5d6776ed374fdbbdb35f93050c"
BASE_REPORT = (
    "reports/phase3_14b_r258_stageh_"
    "candidate_descriptor_identifiability_summary.json"
)
EXPECTED_BASE_REPORT_SHA256 = (
    "5ebf1e7e08bba862283b719be928545ac7e0e2ac04e721d694e31e2088b2597f"
)
EXPECTED_BASE_SINGLE_RUN_SHA256 = (
    "5b281368212b7446454915c9b758c7f87b40a39f706617b54009e04424b0b395"
)
EXPECTED_BASE_ROOT_CAUSE = (
    "phase314b_r258_stageh_frozen_integrator_rejects_entire_surrogate_scale_bank"
)
EXPECTED_BASE_NEXT_PATH = (
    "AUDIT_FROZEN_INTEGRATOR_REJECTION_PREDICATES_ON_OOF_DIRECTION_BANK"
)
EXPECTED_SELECTED_ULP_FACTOR = 1.0

BASE_DIRECTION_IDS: Tuple[str, ...] = stageh.BASE_DIRECTION_IDS
SCALE_MULTIPLIERS: Tuple[float, ...] = stageh.SCALE_MULTIPLIERS
SOURCE_IDS: Tuple[str, ...] = (
    "oof_surrogate",
    "raw_oracle",
    "projected_oracle",
)

# Families are ordered as an audit priority, not asserted to be the source-code
# execution order.  The module also records AST line evidence for observed leaf
# names so a claimed discriminator remains inspectable.
PREDICATE_FAMILY_ORDER: Tuple[str, ...] = (
    "direction_retention",
    "reconstruction",
    "lower_segment_geometry",
    "upper_segment_geometry",
    "segment_geometry",
    "finite_state",
    "displacement",
    "physical_geometry",
    "final_acceptance",
    "scale_selection",
)

PREDICATE_TOKENS: Tuple[str, ...] = (
    "accept",
    "eligible",
    "feasible",
    "finite",
    "gate",
    "geometry",
    "length",
    "lower",
    "motion",
    "pass",
    "physical",
    "reason",
    "reconstruct",
    "reject",
    "retention",
    "scale",
    "segment",
    "upper",
    "valid",
)

PASS_TOKENS: Tuple[str, ...] = (
    "accept",
    "eligible",
    "feasible",
    "finite",
    "gate",
    "pass",
    "valid",
)

SKIPPED_LARGE_KEYS: Tuple[str, ...] = (
    "candidate",
    "direction",
    "proposal",
    "projected",
    "reconstructed",
)

FAMILY_OUTCOMES: Mapping[str, Tuple[str, str]] = {
    "direction_retention": (
        "phase314b_r258_stagei_oof_direction_rejected_by_retention_predicate",
        "CALIBRATE_INTEGRATOR_COMPATIBLE_DIRECTION_BASIS_ON_OBJECTIVE_TRAIN_ONLY",
    ),
    "reconstruction": (
        "phase314b_r258_stagei_oof_direction_rejected_by_reconstruction_predicate",
        "AUDIT_SURROGATE_DIRECTION_RECONSTRUCTION_COMPATIBILITY",
    ),
    "lower_segment_geometry": (
        "phase314b_r258_stagei_oof_direction_violates_lower_segment_geometry",
        "CALIBRATE_LOCAL_SEGMENT_DIRECTION_CORRECTION_BEFORE_FROZEN_INTEGRATION",
    ),
    "upper_segment_geometry": (
        "phase314b_r258_stagei_oof_direction_violates_upper_segment_geometry",
        "CALIBRATE_LOCAL_SEGMENT_DIRECTION_CORRECTION_BEFORE_FROZEN_INTEGRATION",
    ),
    "segment_geometry": (
        "phase314b_r258_stagei_oof_direction_violates_segment_geometry",
        "CALIBRATE_LOCAL_SEGMENT_DIRECTION_CORRECTION_BEFORE_FROZEN_INTEGRATION",
    ),
    "finite_state": (
        "phase314b_r258_stagei_oof_direction_produces_nonfinite_reconstruction",
        "AUDIT_NUMERICAL_STABILITY_OF_SURROGATE_DIRECTION_RECONSTRUCTION",
    ),
    "displacement": (
        "phase314b_r258_stagei_oof_direction_fails_displacement_predicate",
        "CALIBRATE_EXECUTABLE_DIRECTION_MAGNITUDE_PARAMETERIZATION",
    ),
    "physical_geometry": (
        "phase314b_r258_stagei_oof_direction_fails_physical_geometry_predicate",
        "AUDIT_PHYSICAL_GEOMETRY_RESIDUAL_OF_OOF_DIRECTION_BANK",
    ),
    "final_acceptance": (
        "phase314b_r258_stagei_oof_direction_fails_unclassified_final_predicate",
        "EXPOSE_FINAL_INTEGRATOR_ACCEPTANCE_COMPONENTS_WITHOUT_CHANGING_DECISIONS",
    ),
    "scale_selection": (
        "phase314b_r258_stagei_integrator_scale_selection_contract_inconsistent",
        "CORRECT_FROZEN_INTEGRATOR_SCALE_SELECTION_OBSERVABILITY",
    ),
}


class IntegratorRejectionPredicateError(RuntimeError):
    """Raised when Stage-I scientific, shape, or leakage invariants fail."""


@dataclass(frozen=True)
class IntegratorRejectionPredicateSpec:
    timesteps: Tuple[int, ...] = stagef.TIMESTEPS
    grouped_cv_folds: int = 6
    scale_multipliers: Tuple[float, ...] = SCALE_MULTIPLIERS
    oracle_acceptance_rate_min: float = 0.95
    oof_rejection_rate_max: float = 0.05
    predicate_control_pass_rate_min: float = 0.90
    predicate_oof_pass_rate_max: float = 0.10
    direction_nonzero_epsilon: float = 1.0e-12
    candidate_motion_epsilon: float = 1.0e-12
    binary_tolerance: float = 1.0e-12
    numeric_epsilon: float = 1.0e-14
    maximum_leaf_depth: int = 8
    maximum_leaf_count: int = 128
    maximum_failing_leaf_count: int = 24

    def validate(self) -> None:
        if self.timesteps != stagef.TIMESTEPS:
            raise ValueError("Stage-I timestep population changed")
        if self.grouped_cv_folds != 6:
            raise ValueError("Stage-I grouped fold count changed")
        if self.scale_multipliers != SCALE_MULTIPLIERS:
            raise ValueError("Stage-I scale bank changed")
        if len(self.scale_multipliers) != 4:
            raise ValueError("Stage-I scale bank size changed")
        for name, value in (
            ("oracle_acceptance_rate_min", self.oracle_acceptance_rate_min),
            ("predicate_control_pass_rate_min", self.predicate_control_pass_rate_min),
        ):
            if not 0.0 < float(value) <= 1.0:
                raise ValueError("Stage-I {} is invalid".format(name))
        for name, value in (
            ("oof_rejection_rate_max", self.oof_rejection_rate_max),
            ("predicate_oof_pass_rate_max", self.predicate_oof_pass_rate_max),
        ):
            if not 0.0 <= float(value) < 1.0:
                raise ValueError("Stage-I {} is invalid".format(name))
        for value in (
            self.direction_nonzero_epsilon,
            self.candidate_motion_epsilon,
            self.binary_tolerance,
            self.numeric_epsilon,
        ):
            if float(value) <= 0.0:
                raise ValueError("Stage-I numeric policy is invalid")
        if self.maximum_leaf_depth != 8:
            raise ValueError("Stage-I maximum leaf depth changed")
        if self.maximum_leaf_count != 128:
            raise ValueError("Stage-I maximum leaf count changed")
        if self.maximum_failing_leaf_count != 24:
            raise ValueError("Stage-I failing leaf count changed")


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
            raise ValueError("non-finite Stage-I value")
        return value
    raise TypeError("unsupported Stage-I JSON value: {!r}".format(type(value)))


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
        raise IntegratorRejectionPredicateError(
            "required commit is not an ancestor: {} -> {}".format(
                ancestor, descendant
            )
        )


def _find_key(value: Any, key: str) -> List[Any]:
    found: List[Any] = []
    if isinstance(value, Mapping):
        for item_key, item in value.items():
            if str(item_key) == key:
                found.append(item)
            found.extend(_find_key(item, key))
    elif isinstance(value, (list, tuple)):
        for item in value:
            found.extend(_find_key(item, key))
    return found


def validate_base_evidence(root: Path) -> Dict[str, Any]:
    repository_root = Path(root).resolve()
    stageh_base = stageh.validate_base_evidence(repository_root)
    head = _git_output(repository_root, "rev-parse", "HEAD")
    for commit in (
        BASE_IMPLEMENTATION_COMMIT,
        INDEPENDENT_GOALS_COMMIT,
        BASE_EVIDENCE_COMMIT,
    ):
        _assert_ancestor(repository_root, commit, head)
    stageh_source_sha = stagef.assert_file_bound_to_commit(
        repository_root,
        "ccda_phase3/phase314b_r258_stageh_candidate_descriptor_identifiability.py",
        BASE_EVIDENCE_COMMIT,
    )
    goals_sha = stagef.assert_file_bound_to_commit(
        repository_root,
        "GOALS.md",
        INDEPENDENT_GOALS_COMMIT,
    )
    report_sha = stagef.assert_file_bound_to_commit(
        repository_root,
        BASE_REPORT,
        BASE_EVIDENCE_COMMIT,
    )
    if report_sha != EXPECTED_BASE_REPORT_SHA256:
        raise IntegratorRejectionPredicateError("Stage-H report SHA changed")
    report_path = repository_root / BASE_REPORT
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("execution_verdict") != "PASS":
        raise IntegratorRejectionPredicateError("Stage-H execution verdict changed")
    if report.get("scientific_status") != "BLOCKED":
        raise IntegratorRejectionPredicateError("Stage-H scientific status changed")
    if report.get("root_cause") != EXPECTED_BASE_ROOT_CAUSE:
        raise IntegratorRejectionPredicateError("Stage-H root cause changed")
    if report.get("required_next_path") != EXPECTED_BASE_NEXT_PATH:
        raise IntegratorRejectionPredicateError("Stage-H next path changed")
    single_run_values = _find_key(report, "single_run_result_sha256")
    if EXPECTED_BASE_SINGLE_RUN_SHA256 not in [str(value) for value in single_run_values]:
        raise IntegratorRejectionPredicateError("Stage-H single-run SHA changed")
    frozen_values = _find_key(report, "frozen_probe_accessed")
    if any(bool(value) for value in frozen_values):
        raise IntegratorRejectionPredicateError("Stage-H frozen probe boundary changed")
    holdout_values = _find_key(report, "selection_holdout_evaluated")
    if any(value is True for value in holdout_values):
        raise IntegratorRejectionPredicateError("Stage-H holdout boundary changed")
    return {
        "stageh_base": stageh_base,
        "head": head,
        "stageh_source_sha256": stageh_source_sha,
        "goals_sha256": goals_sha,
        "report_sha256": report_sha,
        "single_run_sha256": EXPECTED_BASE_SINGLE_RUN_SHA256,
    }


def predicate_family(path: str) -> Optional[str]:
    lowered = path.lower()
    if "retention" in lowered:
        return "direction_retention"
    if "reconstruct" in lowered:
        return "reconstruction"
    if "lower" in lowered and any(token in lowered for token in ("segment", "length", "gate", "pass", "valid")):
        return "lower_segment_geometry"
    if "upper" in lowered and any(token in lowered for token in ("segment", "length", "gate", "pass", "valid")):
        return "upper_segment_geometry"
    if any(token in lowered for token in ("segment", "constraint")) and any(
        token in lowered for token in ("gate", "pass", "valid", "feasible", "length")
    ):
        return "segment_geometry"
    if "finite" in lowered:
        return "finite_state"
    if any(token in lowered for token in ("displacement", "motion")):
        return "displacement"
    if any(token in lowered for token in ("physical", "geometry")):
        return "physical_geometry"
    if "scale" in lowered:
        return "scale_selection"
    if any(token in lowered for token in PASS_TOKENS):
        return "final_acceptance"
    return None


def _numeric_array_summary(
    value: Any,
    *,
    path: str,
    retention_minimum: float,
    binary_tolerance: float,
) -> Dict[str, Any]:
    array = np.asarray(value)
    result: Dict[str, Any] = {
        "path": path,
        "family": predicate_family(path),
        "shape": [int(item) for item in array.shape],
        "dtype": str(array.dtype),
        "sha256": sha256_array(array),
        "element_count": int(array.size),
        "pass_rate": None,
        "pass_rate_rule": None,
    }
    if array.size == 0:
        return result
    if array.dtype == np.bool_:
        result.update(
            {
                "true_rate": float(np.mean(array)),
                "pass_rate": float(np.mean(array)),
                "pass_rate_rule": "boolean_true",
            }
        )
        return result
    if not np.issubdtype(array.dtype, np.number):
        return result
    numeric = array.astype(np.float64, copy=False)
    finite_mask = np.isfinite(numeric)
    result["finite_rate"] = float(np.mean(finite_mask))
    if not np.any(finite_mask):
        return result
    finite = numeric[finite_mask]
    result.update(
        {
            "minimum": float(np.min(finite)),
            "mean": float(np.mean(finite)),
            "maximum": float(np.max(finite)),
            "zero_rate": float(np.mean(np.abs(finite) <= binary_tolerance)),
            "positive_rate": float(np.mean(finite > binary_tolerance)),
        }
    )
    lowered = path.lower()
    near_zero_or_one = np.logical_or(
        np.abs(finite) <= binary_tolerance,
        np.abs(finite - 1.0) <= binary_tolerance,
    )
    if np.all(near_zero_or_one) and any(token in lowered for token in PASS_TOKENS):
        pass_rate = float(np.mean(finite > 0.5))
        result["pass_rate"] = pass_rate
        result["pass_rate_rule"] = "binary_numeric_positive"
    elif "retention" in lowered:
        result["pass_rate"] = float(np.mean(finite >= float(retention_minimum)))
        result["pass_rate_rule"] = "retention_ge_minimum"
        result["retention_minimum"] = float(retention_minimum)
    elif "selected_scale" in lowered or lowered.endswith(".scale"):
        result["pass_rate"] = float(np.mean(finite > binary_tolerance))
        result["pass_rate_rule"] = "scale_positive"
    return result


def predicate_leaves(
    integration: Mapping[str, Any],
    *,
    retention_minimum: float,
    spec: IntegratorRejectionPredicateSpec,
) -> List[Dict[str, Any]]:
    leaves: List[Dict[str, Any]] = []

    def visit(value: Any, path: str, depth: int) -> None:
        if len(leaves) >= spec.maximum_leaf_count or depth > spec.maximum_leaf_depth:
            return
        if isinstance(value, Mapping):
            for key in sorted(value, key=lambda item: str(item)):
                text = str(key)
                lowered = text.lower()
                if path == "$" and any(token == lowered for token in SKIPPED_LARGE_KEYS):
                    continue
                visit(value[key], path + "." + text, depth + 1)
            return
        lowered_path = path.lower()
        if not any(token in lowered_path for token in PREDICATE_TOKENS):
            return
        if isinstance(value, (str, bool, int, float, np.generic)) or value is None:
            if isinstance(value, str) or value is None:
                leaves.append(
                    {
                        "path": path,
                        "family": predicate_family(path),
                        "value": jsonable(value),
                        "pass_rate": None,
                        "pass_rate_rule": None,
                    }
                )
            else:
                leaves.append(
                    _numeric_array_summary(
                        np.asarray(value),
                        path=path,
                        retention_minimum=retention_minimum,
                        binary_tolerance=spec.binary_tolerance,
                    )
                )
            return
        if isinstance(value, np.ndarray) or (
            hasattr(value, "shape") and hasattr(value, "dtype")
        ):
            leaves.append(
                _numeric_array_summary(
                    value,
                    path=path,
                    retention_minimum=retention_minimum,
                    binary_tolerance=spec.binary_tolerance,
                )
            )

    visit(integration, "$", 0)
    return leaves


def predicate_source_index(module: Any) -> Dict[str, Any]:
    path = Path(module.__file__).resolve()
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    identifiers: Dict[str, int] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            identifiers.setdefault(node.id.lower(), int(node.lineno))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            text = node.value.strip().lower()
            if text and len(text) <= 160:
                identifiers.setdefault(text, int(node.lineno))
    families: Dict[str, List[Dict[str, Any]]] = {
        name: [] for name in PREDICATE_FAMILY_ORDER
    }
    for name, line in sorted(identifiers.items(), key=lambda item: (item[1], item[0])):
        family = predicate_family(name)
        if family is not None:
            families[family].append({"identifier": name, "line": int(line)})
    integrate_source_sha: Optional[str]
    try:
        integrate_source_sha = sha256_bytes(
            inspect.getsource(module.integrate_rowwise).encode("utf-8")
        )
    except (OSError, TypeError):
        integrate_source_sha = None
    return {
        "module_path": str(path),
        "module_sha256": sha256_bytes(source.encode("utf-8")),
        "integrate_rowwise_source_sha256": integrate_source_sha,
        "family_identifiers": {
            family: values[:64]
            for family, values in families.items()
            if values
        },
    }


def summarize_predicate_families(
    leaves: Sequence[Mapping[str, Any]],
    *,
    source_index: Mapping[str, Any],
    spec: IntegratorRejectionPredicateSpec,
) -> Dict[str, Any]:
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for leaf in leaves:
        family = leaf.get("family")
        if isinstance(family, str):
            grouped.setdefault(family, []).append(leaf)
    source_families = source_index.get("family_identifiers", {})
    result: Dict[str, Any] = {}
    for family in PREDICATE_FAMILY_ORDER:
        family_leaves = grouped.get(family, [])
        pass_rates = [
            float(leaf["pass_rate"])
            for leaf in family_leaves
            if leaf.get("pass_rate") is not None
        ]
        failing = sorted(
            [
                leaf
                for leaf in family_leaves
                if leaf.get("pass_rate") is not None
                and float(leaf["pass_rate"]) < spec.predicate_control_pass_rate_min
            ],
            key=lambda leaf: (float(leaf["pass_rate"]), str(leaf["path"])),
        )[: spec.maximum_failing_leaf_count]
        source_lines = [
            int(item["line"])
            for item in source_families.get(family, [])
            if isinstance(item, Mapping) and "line" in item
        ]
        result[family] = {
            "observed_leaf_count": len(family_leaves),
            "pass_rate_leaf_count": len(pass_rates),
            "minimum_pass_rate": min(pass_rates) if pass_rates else None,
            "mean_pass_rate": (
                float(np.mean(np.asarray(pass_rates, dtype=np.float64)))
                if pass_rates
                else None
            ),
            "maximum_pass_rate": max(pass_rates) if pass_rates else None,
            "source_line_minimum": min(source_lines) if source_lines else None,
            "failing_leaves": [jsonable(dict(leaf)) for leaf in failing],
        }
    return result


def _row_norm(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    return np.linalg.norm(array.reshape(array.shape[0], -1), axis=1)


def direction_comparison(
    left: np.ndarray,
    right: np.ndarray,
    *,
    epsilon: float,
) -> Dict[str, Any]:
    x = np.asarray(left, dtype=np.float64).reshape(left.shape[0], -1)
    y = np.asarray(right, dtype=np.float64).reshape(right.shape[0], -1)
    x_norm = np.linalg.norm(x, axis=1)
    y_norm = np.linalg.norm(y, axis=1)
    denominator = np.maximum(x_norm * y_norm, float(epsilon))
    cosine = np.sum(x * y, axis=1) / denominator
    ratio = x_norm / np.maximum(y_norm, float(epsilon))
    return {
        "cosine": {
            "minimum": float(np.min(cosine)),
            "mean": float(np.mean(cosine)),
            "maximum": float(np.max(cosine)),
            "positive_rate": float(np.mean(cosine > 0.0)),
        },
        "norm_ratio": {
            "minimum": float(np.min(ratio)),
            "mean": float(np.mean(ratio)),
            "maximum": float(np.max(ratio)),
        },
        "left_sha256": sha256_array(left),
        "right_sha256": sha256_array(right),
    }


def _retention_minimum(integrator_spec: Any) -> float:
    for name in (
        "retention_min",
        "minimum_retention",
        "direction_retention_min",
    ):
        if hasattr(integrator_spec, name):
            return float(getattr(integrator_spec, name))
    return 0.25


def audit_direction_source(
    *,
    source_id: str,
    control: np.ndarray,
    base_direction: np.ndarray,
    context: Mapping[str, Any],
    integrator_spec: stagee258.ConstrainedIntegratorSpec,
    source_index: Mapping[str, Any],
    spec: IntegratorRejectionPredicateSpec,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if source_id not in SOURCE_IDS:
        raise ValueError("Stage-I direction source changed")
    control_raw = np.asarray(control, dtype=np.float32)
    direction_raw = np.asarray(base_direction, dtype=np.float64)
    if direction_raw.shape != control_raw.shape:
        raise IntegratorRejectionPredicateError(
            "Stage-I direction/control shape changed"
        )
    rows = control_raw.shape[0]
    selected_columns: List[np.ndarray] = []
    motion_columns: List[np.ndarray] = []
    multiplier_records: List[Dict[str, Any]] = []
    family_rates: Dict[str, List[Optional[float]]] = {
        family: [] for family in PREDICATE_FAMILY_ORDER
    }
    retention_minimum = _retention_minimum(integrator_spec)
    for multiplier in spec.scale_multipliers:
        proposed = direction_raw * float(multiplier)
        integration = stagee258.integrate_rowwise(
            control=control_raw,
            direction=proposed,
            definition=stagef.fixed_integrator_definition(),
            context=context,
            spec=integrator_spec,
        )
        if not isinstance(integration, Mapping):
            raise IntegratorRejectionPredicateError(
                "Stage-I integrator result is not a mapping"
            )
        if "selected_scale" not in integration or "candidate" not in integration:
            raise IntegratorRejectionPredicateError(
                "Stage-I integrator observable surface changed"
            )
        selected = np.asarray(integration["selected_scale"], dtype=np.float64).reshape(rows)
        candidate = np.asarray(integration["candidate"], dtype=np.float32)
        if candidate.shape != control_raw.shape:
            raise IntegratorRejectionPredicateError(
                "Stage-I candidate shape changed"
            )
        displacement = candidate.astype(np.float64) - control_raw.astype(np.float64)
        motion_norm = _row_norm(displacement)
        motion = motion_norm > spec.candidate_motion_epsilon
        exact_control = np.all(
            candidate == control_raw,
            axis=tuple(range(1, candidate.ndim)),
        )
        numeric_control = motion_norm <= spec.candidate_motion_epsilon
        selected_positive = selected > spec.binary_tolerance
        if np.any(selected_positive != motion):
            raise IntegratorRejectionPredicateError(
                "Stage-I selected-scale/motion observable is inconsistent"
            )
        leaves = predicate_leaves(
            integration,
            retention_minimum=retention_minimum,
            spec=spec,
        )
        family_summary = summarize_predicate_families(
            leaves,
            source_index=source_index,
            spec=spec,
        )
        for family in PREDICATE_FAMILY_ORDER:
            family_rates[family].append(
                family_summary[family]["minimum_pass_rate"]
            )
        selected_columns.append(selected)
        motion_columns.append(motion)
        multiplier_records.append(
            {
                "scale_multiplier": float(multiplier),
                "proposed_direction_sha256": sha256_array(proposed),
                "proposed_direction_norm": {
                    "minimum": float(np.min(_row_norm(proposed))),
                    "mean": float(np.mean(_row_norm(proposed))),
                    "maximum": float(np.max(_row_norm(proposed))),
                    "nonzero_rate": float(
                        np.mean(_row_norm(proposed) > spec.direction_nonzero_epsilon)
                    ),
                },
                "mapping_keys": sorted(str(key) for key in integration.keys()),
                "selected_scale": {
                    "minimum": float(np.min(selected)),
                    "mean": float(np.mean(selected)),
                    "maximum": float(np.max(selected)),
                    "positive_rate": float(np.mean(selected_positive)),
                    "sha256": sha256_array(selected),
                },
                "candidate_motion": {
                    "minimum_norm": float(np.min(motion_norm)),
                    "mean_norm": float(np.mean(motion_norm)),
                    "maximum_norm": float(np.max(motion_norm)),
                    "positive_rate": float(np.mean(motion)),
                    "exact_control_rate": float(np.mean(exact_control)),
                    "numeric_control_rate": float(np.mean(numeric_control)),
                    "candidate_sha256": sha256_array(candidate),
                },
                "predicate_families": family_summary,
            }
        )
    selected_bank = np.stack(selected_columns, axis=1)
    motion_bank = np.stack(motion_columns, axis=1)
    any_selected = np.any(selected_bank > spec.binary_tolerance, axis=1)
    all_rejected = ~any_selected
    summary = {
        "source_id": source_id,
        "row_count": int(rows),
        "direction_sha256": sha256_array(direction_raw),
        "base_direction_norm": {
            "minimum": float(np.min(_row_norm(direction_raw))),
            "mean": float(np.mean(_row_norm(direction_raw))),
            "maximum": float(np.max(_row_norm(direction_raw))),
            "nonzero_rate": float(
                np.mean(_row_norm(direction_raw) > spec.direction_nonzero_epsilon)
            ),
        },
        "scale_bank_acceptance": {
            "selected_scale_positive_rate": float(
                np.mean(selected_bank > spec.binary_tolerance)
            ),
            "candidate_motion_positive_rate": float(np.mean(motion_bank)),
            "rows_with_any_accepted_multiplier_rate": float(np.mean(any_selected)),
            "rows_with_all_multipliers_rejected_rate": float(np.mean(all_rejected)),
            "selected_scale_bank_sha256": sha256_array(selected_bank),
            "motion_bank_sha256": sha256_array(motion_bank),
        },
        "multiplier_records": multiplier_records,
    }
    internal = {
        "selected_bank": selected_bank,
        "motion_bank": motion_bank,
        "family_rates": family_rates,
    }
    return summary, internal


def _family_population_rate(
    internal: Mapping[str, Any],
    family: str,
) -> Optional[float]:
    values = [
        float(value)
        for value in internal["family_rates"].get(family, [])
        if value is not None
    ]
    if not values:
        return None
    return float(min(values))


def compare_rejection_predicates(
    *,
    oof_summary: Mapping[str, Any],
    oof_internal: Mapping[str, Any],
    raw_summary: Mapping[str, Any],
    raw_internal: Mapping[str, Any],
    projected_summary: Mapping[str, Any],
    projected_internal: Mapping[str, Any],
    spec: IntegratorRejectionPredicateSpec,
) -> Dict[str, Any]:
    oof_accept = float(
        oof_summary["scale_bank_acceptance"][
            "rows_with_any_accepted_multiplier_rate"
        ]
    )
    raw_accept = float(
        raw_summary["scale_bank_acceptance"][
            "rows_with_any_accepted_multiplier_rate"
        ]
    )
    projected_accept = float(
        projected_summary["scale_bank_acceptance"][
            "rows_with_any_accepted_multiplier_rate"
        ]
    )
    if projected_accept >= spec.oracle_acceptance_rate_min:
        comparator_id = "projected_oracle"
        comparator_internal = projected_internal
        comparator_accept = projected_accept
    elif raw_accept >= spec.oracle_acceptance_rate_min:
        comparator_id = "raw_oracle"
        comparator_internal = raw_internal
        comparator_accept = raw_accept
    else:
        return {
            "locus": "oracle_control_not_admitted",
            "discriminator_family": None,
            "comparator_source": None,
            "oof_acceptance_rate": oof_accept,
            "raw_oracle_acceptance_rate": raw_accept,
            "projected_oracle_acceptance_rate": projected_accept,
            "interpretation": (
                "Neither objective-train oracle control is admitted at the "
                "predeclared rate; the frozen integrator replay must be restored "
                "before attributing rejection to the surrogate direction."
            ),
        }
    if oof_accept > spec.oof_rejection_rate_max:
        return {
            "locus": "stageh_rejection_not_reproduced",
            "discriminator_family": None,
            "comparator_source": comparator_id,
            "oof_acceptance_rate": oof_accept,
            "raw_oracle_acceptance_rate": raw_accept,
            "projected_oracle_acceptance_rate": projected_accept,
            "interpretation": (
                "Stage-I observed accepted OOF rows despite Stage-H complete "
                "rejection; provenance or replay identity changed."
            ),
        }
    discriminator: Optional[str] = None
    family_comparisons: Dict[str, Any] = {}
    for family in PREDICATE_FAMILY_ORDER:
        oof_rate = _family_population_rate(oof_internal, family)
        control_rate = _family_population_rate(comparator_internal, family)
        discriminates = bool(
            oof_rate is not None
            and control_rate is not None
            and oof_rate <= spec.predicate_oof_pass_rate_max
            and control_rate >= spec.predicate_control_pass_rate_min
        )
        family_comparisons[family] = {
            "oof_minimum_pass_rate": oof_rate,
            "control_minimum_pass_rate": control_rate,
            "discriminates": discriminates,
        }
        if discriminator is None and discriminates:
            discriminator = family
    if discriminator is not None:
        locus = "oof_rejected_at_{}".format(discriminator)
        interpretation = (
            "The same frozen integrator admits the {} direction control but "
            "systematically rejects the OOF surrogate at the {} predicate "
            "family.".format(comparator_id, discriminator)
        )
    else:
        observed_oof_rates = [
            item["oof_minimum_pass_rate"]
            for item in family_comparisons.values()
            if item["oof_minimum_pass_rate"] is not None
        ]
        if observed_oof_rates and min(observed_oof_rates) >= spec.predicate_control_pass_rate_min:
            locus = "scale_selection_zero_despite_observed_predicate_pass"
            discriminator = "scale_selection"
            interpretation = (
                "Observed predicate leaves pass, yet selected_scale remains zero; "
                "the rejection surface or scale-selection observable is inconsistent."
            )
        else:
            locus = "predicate_surface_does_not_identify_rejection"
            interpretation = (
                "The frozen output surface does not expose a predicate family "
                "that separates accepted oracle controls from rejected OOF directions."
            )
    return {
        "locus": locus,
        "discriminator_family": discriminator,
        "comparator_source": comparator_id,
        "comparator_acceptance_rate": comparator_accept,
        "oof_acceptance_rate": oof_accept,
        "raw_oracle_acceptance_rate": raw_accept,
        "projected_oracle_acceptance_rate": projected_accept,
        "family_comparisons": family_comparisons,
        "interpretation": interpretation,
    }


def classify(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not records:
        raise IntegratorRejectionPredicateError("Stage-I audit has no OOF records")
    loci: Dict[str, int] = {}
    families: Dict[str, int] = {}
    for record in records:
        comparison = record["rejection_comparison"]
        locus = str(comparison["locus"])
        loci[locus] = loci.get(locus, 0) + 1
        family = comparison.get("discriminator_family")
        if isinstance(family, str):
            families[family] = families.get(family, 0) + 1
    total = len(records)
    if loci.get("oracle_control_not_admitted", 0) > 0:
        root = "phase314b_r258_stagei_frozen_integrator_oracle_replay_not_admitted"
        next_path = "RESTORE_STAGEE_ORACLE_REPLAY_BEFORE_SURROGATE_REPAIR"
        primary = "frozen_integrator_oracle_replay"
    elif loci.get("stageh_rejection_not_reproduced", 0) > 0:
        root = "phase314b_r258_stagei_stageh_rejection_not_reproduced"
        next_path = "AUDIT_STAGEH_STAGEI_REPLAY_IDENTITY"
        primary = "replay_identity"
    elif loci.get("predicate_surface_does_not_identify_rejection", 0) > 0:
        root = "phase314b_r258_stagei_rejection_predicate_surface_insufficient"
        next_path = "ADD_READ_ONLY_REJECTION_TELEMETRY_TO_FROZEN_INTEGRATOR"
        primary = "predicate_observability"
    elif loci.get("scale_selection_zero_despite_observed_predicate_pass", 0) > 0:
        root, next_path = FAMILY_OUTCOMES["scale_selection"]
        primary = "scale_selection"
    elif len(families) == 1 and sum(families.values()) == total:
        family = next(iter(families))
        root, next_path = FAMILY_OUTCOMES.get(
            family,
            (
                "phase314b_r258_stagei_unclassified_rejection_predicate",
                "AUDIT_UNCLASSIFIED_FINAL_INTEGRATOR_PREDICATE",
            ),
        )
        primary = family
    else:
        root = "phase314b_r258_stagei_rejection_predicates_are_heterogeneous"
        next_path = "STRATIFY_OOF_DIRECTION_REJECTION_BY_BACKBONE_TIMESTEP_PREDICATE"
        primary = "heterogeneous_rejection_predicates"
    return {
        "root_cause": root,
        "required_next_path": next_path,
        "primary_failure_locus": primary,
        "record_count": total,
        "record_locus_counts": dict(sorted(loci.items())),
        "discriminator_family_counts": dict(sorted(families.items())),
        "all_records_share_primary_locus": len(loci) == 1,
        "all_records_share_discriminator_family": (
            len(families) == 1 and sum(families.values()) == total
        ),
    }


def run_calibration(
    *,
    root: Path,
    environment: Mapping[str, Any],
    spec: Optional[IntegratorRejectionPredicateSpec] = None,
    stagef_spec: Optional[stagef.ConstraintAwareSpec] = None,
    direction_spec: Optional[staged258.DirectionSurrogateSpec] = None,
    integrator_spec: Optional[stagee258.ConstrainedIntegratorSpec] = None,
) -> Dict[str, Any]:
    active = IntegratorRejectionPredicateSpec() if spec is None else spec
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
        raise IntegratorRejectionPredicateError(
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
        raise IntegratorRejectionPredicateError(
            "Stage-I ULP policy replay changed"
        )
    active_stagef = stagef.replace(
        active_stagef,
        translation_float32_z_ulp_factor=EXPECTED_SELECTED_ULP_FACTOR,
    )
    active_stagef.validate()
    source_index = predicate_source_index(stagee258)
    oracle_targets: Dict[int, Mapping[str, Any]] = {}
    control_audits: Dict[int, Dict[str, Any]] = {}
    control_internals: Dict[int, Dict[str, Any]] = {}
    for timestep in active.timesteps:
        timestep_value = int(timestep)
        control = np.asarray(
            context["objective_control_predictions"][timestep_value],
            dtype=np.float32,
        )
        target = np.asarray(context["objective_target"], dtype=np.float32)
        oracle = stagef.generate_projected_oracle_target(
            control=control,
            target=target,
            groups=context["objective_groups"],
            condition_name=context["objective_condition_name"],
            timestep=timestep_value,
            context=context,
            direction_spec=active_direction,
            integrator_spec=active_integrator,
            spec=active_stagef,
        )
        oracle_targets[timestep_value] = oracle
        raw_direction = target.astype(np.float64) - control.astype(np.float64)
        projected_direction = (
            np.asarray(oracle["candidate"], dtype=np.float64)
            - control.astype(np.float64)
        )
        raw_summary, raw_internal = audit_direction_source(
            source_id="raw_oracle",
            control=control,
            base_direction=raw_direction,
            context=context,
            integrator_spec=active_integrator,
            source_index=source_index,
            spec=active,
        )
        projected_summary, projected_internal = audit_direction_source(
            source_id="projected_oracle",
            control=control,
            base_direction=projected_direction,
            context=context,
            integrator_spec=active_integrator,
            source_index=source_index,
            spec=active,
        )
        control_audits[timestep_value] = {
            "raw_oracle": raw_summary,
            "projected_oracle": projected_summary,
            "raw_to_projected_direction": direction_comparison(
                raw_direction,
                projected_direction,
                epsilon=active.numeric_epsilon,
            ),
        }
        control_internals[timestep_value] = {
            "raw_oracle": raw_internal,
            "projected_oracle": projected_internal,
        }
    records: List[Dict[str, Any]] = []
    for base_direction_id in BASE_DIRECTION_IDS:
        definition = stagef.definition_by_id(base_direction_id)
        for timestep in active.timesteps:
            timestep_value = int(timestep)
            control = np.asarray(
                context["objective_control_predictions"][timestep_value],
                dtype=np.float32,
            )
            features = stagef.build_constraint_features(
                condition=context["objective_condition"],
                control=control,
                condition_name=context["objective_condition_name"],
                feature_mode=definition.feature_mode,
                context=context,
            )
            oof = stageh.fit_oof_base_direction(
                base_direction_id=base_direction_id,
                features=features,
                control=control,
                oracle_target=oracle_targets[timestep_value],
                condition_name=context["objective_condition_name"],
                fold_assignment=context["objective_fold_assignment"],
                stagef_spec=active_stagef,
                grouped_cv_folds=active.grouped_cv_folds,
            )
            oof_summary, oof_internal = audit_direction_source(
                source_id="oof_surrogate",
                control=control,
                base_direction=oof["prediction"],
                context=context,
                integrator_spec=active_integrator,
                source_index=source_index,
                spec=active,
            )
            raw_summary = control_audits[timestep_value]["raw_oracle"]
            projected_summary = control_audits[timestep_value]["projected_oracle"]
            raw_internal = control_internals[timestep_value]["raw_oracle"]
            projected_internal = control_internals[timestep_value]["projected_oracle"]
            comparison = compare_rejection_predicates(
                oof_summary=oof_summary,
                oof_internal=oof_internal,
                raw_summary=raw_summary,
                raw_internal=raw_internal,
                projected_summary=projected_summary,
                projected_internal=projected_internal,
                spec=active,
            )
            raw_direction = (
                np.asarray(context["objective_target"], dtype=np.float64)
                - control.astype(np.float64)
            )
            projected_direction = (
                np.asarray(
                    oracle_targets[timestep_value]["candidate"], dtype=np.float64
                )
                - control.astype(np.float64)
            )
            records.append(
                {
                    "base_direction_id": base_direction_id,
                    "timestep": timestep_value,
                    "feature_mode": definition.feature_mode,
                    "feature_sha256": sha256_array(features),
                    "oof_direction": {
                        "prediction_sha256": oof["prediction_sha256"],
                        "fold_records": oof["fold_records"],
                        "vs_raw_oracle": direction_comparison(
                            oof["prediction"],
                            raw_direction,
                            epsilon=active.numeric_epsilon,
                        ),
                        "vs_projected_oracle": direction_comparison(
                            oof["prediction"],
                            projected_direction,
                            epsilon=active.numeric_epsilon,
                        ),
                    },
                    "oof_integrator_audit": oof_summary,
                    "rejection_comparison": comparison,
                    "objective_train_rows": int(control.shape[0]),
                    "holdout_used": False,
                }
            )
    classification = classify(records)
    contract = {
        "schema": "phase314b_r258_stagei_integrator_rejection_predicate_contract_v1",
        "phase": PHASE,
        "base_evidence_commit": BASE_EVIDENCE_COMMIT,
        "independent_goals_commit": INDEPENDENT_GOALS_COMMIT,
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
        "source_ids": list(SOURCE_IDS),
        "predicate_family_order": list(PREDICATE_FAMILY_ORDER),
        "integrator_source_index": source_index,
        "data_boundary": {
            "objective_train_only": True,
            "raw_oracle_is_diagnostic_only": True,
            "projected_oracle_is_diagnostic_only": True,
            "selection_holdout_evaluated": False,
            "selection_holdout_used_for_fit_or_selection": False,
            "frozen_probe_accessed": False,
            "condition_label_used_for_audit_selection": False,
            "target_used_only_for_objective_train_oracle_controls": True,
        },
        "mechanism_boundary": {
            "new_ranker_fitted": False,
            "candidate_matrix_changed": False,
            "scale_bank_changed": False,
            "stagee_integrator_modified": False,
            "stagef_feature_definition_modified": False,
            "ulp_policy_changed": False,
            "integrator_decisions_modified": False,
        },
    }
    contract["contract_sha256"] = sha256_bytes(stable_json_bytes(contract))
    audit_payload = {
        "schema": "phase314b_r258_stagei_integrator_rejection_predicate_audit_v1",
        "control_audits_by_timestep": {
            str(key): value for key, value in sorted(control_audits.items())
        },
        "oof_records": records,
        "classification": classification,
    }
    audit_payload["audit_sha256"] = sha256_bytes(stable_json_bytes(audit_payload))
    return {
        "phase": PHASE,
        "schema": "phase314b_r258_stagei_integrator_rejection_predicate_result_v1",
        "verdict": "PASS",
        "scientific_status": "BLOCKED",
        "root_cause": classification["root_cause"],
        "required_next_path": classification["required_next_path"],
        "primary_failure_locus": classification["primary_failure_locus"],
        "selected_configuration": None,
        "train_only_recommendation": None,
        "immutable_inputs": {
            "base_evidence_commit": BASE_EVIDENCE_COMMIT,
            "independent_goals_commit": INDEPENDENT_GOALS_COMMIT,
            "base_report_sha256": EXPECTED_BASE_REPORT_SHA256,
            "base_single_run_sha256": EXPECTED_BASE_SINGLE_RUN_SHA256,
            "stageh_source_sha256": base["stageh_source_sha256"],
            "goals_sha256": base["goals_sha256"],
        },
        "contract": contract,
        "integrator_rejection_audit": audit_payload,
        "integrator_rejection_predicate_audit_completed": True,
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
        "predicate_tensor_persisted": False,
        "npz_saved": False,
        "cache_saved": False,
        "image_saved": False,
        "video_saved": False,
    }
