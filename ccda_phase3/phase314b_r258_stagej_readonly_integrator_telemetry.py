"""Phase3.14b-r2.5.8 Stage J read-only frozen-integrator telemetry.

Stage I proved that the frozen Stage-E integrator accepts raw/projected oracle
controls while rejecting every grouped-OOF surrogate scale-bank proposal, but
its returned mapping does not expose the predicate that causes rejection.
Stage J preserves the Stage-E source and every numerical decision byte-for-byte.
It observes the *actual executed Python frames* of the frozen integrator with
``sys.settrace`` and records only bounded summaries of predicate-like locals.

Every traced call is paired with an untraced call using identical arguments.
The complete returned structures must be recursively byte-exact.  Therefore the
telemetry is admitted only when observation is demonstrably read-only.  The
objective-train population, grouped folds, direction backbones, scale bank,
Stage-F structural-zero representation, ULP factor, Stage-E integrator,
holdout boundary, and frozen probe remain unchanged.  No direction, candidate,
predicate tensor, checkpoint, cache, NPZ, image, or video is persisted.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

import numpy as np

from ccda_phase3 import phase314b_r258_staged_direction_surrogate as staged258
from ccda_phase3 import phase314b_r258_stagee_constrained_integrator as stagee258
from ccda_phase3 import phase314b_r258_stagef_constraint_aware_surrogate as stagef
from ccda_phase3 import phase314b_r258_stageh_candidate_descriptor_identifiability as stageh
from ccda_phase3 import phase314b_r258_stagei_integrator_rejection_predicates as stagei


PHASE = "Phase3.14b-r2.5.8 Stage J"
PHASE_ID = "phase314b_r258_stagej"
BASE_IMPLEMENTATION_COMMIT = "5950ea3355ee4a9789522bd6e77c58aaaba672b2"
BASE_EVIDENCE_COMMIT = "f4c511310fc40bfca398a684dd182f8ed5be8816"
INDEPENDENT_GOALS_COMMIT = "0ce0c9efa05c1b71d7e312bdf65bbe27eb094486"
EXPECTED_SUBMODULE_COMMIT = "633a88752445cf5d6776ed374fdbbdb35f93050c"
BASE_REPORT = "reports/phase3_14b_r258_stagei_integrator_rejection_predicates_summary.json"
EXPECTED_BASE_REPORT_SHA256 = "3e00eab4d6a88905b2fca4f8be2f38e58e3c60a8f8bc6aea66750f74c9827c83"
EXPECTED_BASE_SINGLE_RUN_SHA256 = "4a5f2fe7c8124626e9bd4546da53b7e62688795bab6e168ceb8b038bf5e1d0c3"
EXPECTED_BASE_ROOT_CAUSE = "phase314b_r258_stagei_rejection_predicate_surface_insufficient"
EXPECTED_BASE_NEXT_PATH = "ADD_READ_ONLY_REJECTION_TELEMETRY_TO_FROZEN_INTEGRATOR"
EXPECTED_SELECTED_ULP_FACTOR = 1.0

BASE_DIRECTION_IDS: Tuple[str, ...] = stagei.BASE_DIRECTION_IDS
SCALE_MULTIPLIERS: Tuple[float, ...] = stagei.SCALE_MULTIPLIERS
SOURCE_IDS: Tuple[str, ...] = stagei.SOURCE_IDS

PREDICATE_NAME_TOKENS: Tuple[str, ...] = (
    "accept",
    "eligible",
    "feasible",
    "finite",
    "gate",
    "geometry",
    "lower",
    "motion",
    "pass",
    "physical",
    "reconstruct",
    "reject",
    "retention",
    "scale",
    "segment",
    "upper",
    "valid",
)

SKIP_VALUE_TOKENS: Tuple[str, ...] = (
    "candidate",
    "control",
    "direction",
    "point",
    "proposal",
    "reconstructed",
    "target",
    "vector",
)

FAMILY_OUTCOMES: Mapping[str, Tuple[str, str]] = {
    "direction_retention": (
        "phase314b_r258_stagej_oof_rejected_by_direction_retention",
        "CALIBRATE_INTEGRATOR_COMPATIBLE_DIRECTION_BASIS_ON_OBJECTIVE_TRAIN_ONLY",
    ),
    "reconstruction": (
        "phase314b_r258_stagej_oof_rejected_by_reconstruction",
        "AUDIT_SURROGATE_DIRECTION_RECONSTRUCTION_COMPATIBILITY",
    ),
    "lower_segment_geometry": (
        "phase314b_r258_stagej_oof_rejected_by_lower_segment_geometry",
        "CALIBRATE_LOCAL_SEGMENT_DIRECTION_CORRECTION_BEFORE_FROZEN_INTEGRATION",
    ),
    "upper_segment_geometry": (
        "phase314b_r258_stagej_oof_rejected_by_upper_segment_geometry",
        "CALIBRATE_LOCAL_SEGMENT_DIRECTION_CORRECTION_BEFORE_FROZEN_INTEGRATION",
    ),
    "segment_geometry": (
        "phase314b_r258_stagej_oof_rejected_by_segment_geometry",
        "CALIBRATE_LOCAL_SEGMENT_DIRECTION_CORRECTION_BEFORE_FROZEN_INTEGRATION",
    ),
    "finite_state": (
        "phase314b_r258_stagej_oof_rejected_by_finite_state",
        "AUDIT_NUMERICAL_STABILITY_OF_SURROGATE_DIRECTION_RECONSTRUCTION",
    ),
    "displacement": (
        "phase314b_r258_stagej_oof_rejected_by_displacement",
        "CALIBRATE_EXECUTABLE_DIRECTION_MAGNITUDE_PARAMETERIZATION",
    ),
    "physical_geometry": (
        "phase314b_r258_stagej_oof_rejected_by_physical_geometry",
        "AUDIT_PHYSICAL_GEOMETRY_RESIDUAL_OF_OOF_DIRECTION_BANK",
    ),
    "final_acceptance": (
        "phase314b_r258_stagej_oof_rejected_by_final_acceptance",
        "AUDIT_FINAL_ACCEPTANCE_COMPONENTS_ON_OOF_DIRECTION_BANK",
    ),
    "scale_selection": (
        "phase314b_r258_stagej_oof_rejected_by_scale_selection",
        "AUDIT_SCALE_SEARCH_REJECTION_SEQUENCE_ON_OOF_DIRECTION_BANK",
    ),
}


class ReadOnlyIntegratorTelemetryError(RuntimeError):
    """Raised when Stage-J scientific, identity, or telemetry invariants fail."""


@dataclass(frozen=True)
class ReadOnlyIntegratorTelemetrySpec:
    timesteps: Tuple[int, ...] = stagef.TIMESTEPS
    grouped_cv_folds: int = 6
    scale_multipliers: Tuple[float, ...] = SCALE_MULTIPLIERS
    oracle_acceptance_rate_min: float = 0.95
    oof_acceptance_rate_max: float = 0.05
    oracle_predicate_pass_rate_min: float = 0.90
    oof_predicate_pass_rate_max: float = 0.10
    binary_tolerance: float = 1.0e-12
    candidate_motion_epsilon: float = 1.0e-12
    maximum_trace_events: int = 4096
    maximum_values_per_event: int = 32
    maximum_discriminator_records: int = 32

    def validate(self) -> None:
        if self.timesteps != stagef.TIMESTEPS:
            raise ValueError("Stage-J timestep population changed")
        if self.grouped_cv_folds != 6:
            raise ValueError("Stage-J grouped fold count changed")
        if self.scale_multipliers != SCALE_MULTIPLIERS:
            raise ValueError("Stage-J scale bank changed")
        if self.maximum_trace_events != 4096:
            raise ValueError("Stage-J trace event ceiling changed")
        if self.maximum_values_per_event != 32:
            raise ValueError("Stage-J event value ceiling changed")
        if self.maximum_discriminator_records != 32:
            raise ValueError("Stage-J discriminator ceiling changed")
        for value in (
            self.oracle_acceptance_rate_min,
            self.oracle_predicate_pass_rate_min,
        ):
            if not 0.0 < float(value) <= 1.0:
                raise ValueError("Stage-J positive pass threshold is invalid")
        for value in (
            self.oof_acceptance_rate_max,
            self.oof_predicate_pass_rate_max,
        ):
            if not 0.0 <= float(value) < 1.0:
                raise ValueError("Stage-J rejection threshold is invalid")
        if self.binary_tolerance <= 0.0 or self.candidate_motion_epsilon <= 0.0:
            raise ValueError("Stage-J numeric tolerance is invalid")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(str(tuple(array.shape)).encode("utf-8"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def stable_json_bytes(value: Any) -> bytes:
    return json.dumps(
        jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


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
            raise ValueError("non-finite value in Stage-J JSON payload")
        return value
    return repr(value)


def _extract_single_run_sha(payload: Mapping[str, Any]) -> Optional[str]:
    execution = payload.get("execution")
    if isinstance(execution, Mapping):
        value = execution.get("single_run_result_sha256")
        if isinstance(value, str):
            return value
    for key in ("single_run_result_sha256", "single_run_sha256"):
        value = payload.get(key)
        if isinstance(value, str):
            return value
    return None


def validate_base_evidence(root: Path) -> Dict[str, Any]:
    repository_root = Path(root).resolve()
    report_path = repository_root / BASE_REPORT
    if not report_path.is_file():
        raise ReadOnlyIntegratorTelemetryError("Stage-I base report is missing")
    report_bytes = report_path.read_bytes()
    if sha256_bytes(report_bytes) != EXPECTED_BASE_REPORT_SHA256:
        raise ReadOnlyIntegratorTelemetryError("Stage-I base report SHA changed")
    payload = json.loads(report_bytes.decode("utf-8"))
    if payload.get("execution_verdict") != "PASS":
        raise ReadOnlyIntegratorTelemetryError("Stage-I execution verdict changed")
    if payload.get("scientific_status") != "BLOCKED":
        raise ReadOnlyIntegratorTelemetryError("Stage-I scientific status changed")
    if payload.get("root_cause") != EXPECTED_BASE_ROOT_CAUSE:
        raise ReadOnlyIntegratorTelemetryError("Stage-I root cause changed")
    if payload.get("required_next_path") != EXPECTED_BASE_NEXT_PATH:
        raise ReadOnlyIntegratorTelemetryError("Stage-I next path changed")
    if _extract_single_run_sha(payload) != EXPECTED_BASE_SINGLE_RUN_SHA256:
        raise ReadOnlyIntegratorTelemetryError("Stage-I single-run SHA changed")
    stagee_path = Path(inspect.getsourcefile(stagee258) or "").resolve()
    stagei_path = Path(inspect.getsourcefile(stagei) or "").resolve()
    if not stagee_path.is_file() or not stagei_path.is_file():
        raise ReadOnlyIntegratorTelemetryError("Stage-E/Stage-I source is unavailable")
    return {
        "base_report_sha256": EXPECTED_BASE_REPORT_SHA256,
        "base_single_run_sha256": EXPECTED_BASE_SINGLE_RUN_SHA256,
        "stagee_source_path": str(stagee_path),
        "stagee_source_sha256": sha256_bytes(stagee_path.read_bytes()),
        "stagei_source_path": str(stagei_path),
        "stagei_source_sha256": sha256_bytes(stagei_path.read_bytes()),
    }


def _target_names(target: ast.AST) -> Set[str]:
    names: Set[str] = set()
    if isinstance(target, ast.Name):
        names.add(target.id)
    elif isinstance(target, (ast.Tuple, ast.List)):
        for item in target.elts:
            names.update(_target_names(item))
    return names


def _call_name(node: ast.Call) -> str:
    value = node.func
    parts: List[str] = []
    while isinstance(value, ast.Attribute):
        parts.append(value.attr)
        value = value.value
    if isinstance(value, ast.Name):
        parts.append(value.id)
    return ".".join(reversed(parts)).lower()


def _predicate_expression(node: Optional[ast.AST]) -> bool:
    if node is None:
        return False
    if isinstance(node, (ast.Compare, ast.BoolOp)):
        return True
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.Not, ast.Invert)):
        return True
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.BitAnd, ast.BitOr)):
        return True
    if isinstance(node, ast.Call):
        name = _call_name(node)
        return any(token in name for token in PREDICATE_NAME_TOKENS)
    return any(_predicate_expression(child) for child in ast.iter_child_nodes(node))


def _names_in_expression(node: ast.AST) -> Set[str]:
    return {item.id for item in ast.walk(node) if isinstance(item, ast.Name)}


def predicate_source_index(module: ModuleType) -> Dict[str, Any]:
    source_path = Path(inspect.getsourcefile(module) or "").resolve()
    if not source_path.is_file():
        raise ReadOnlyIntegratorTelemetryError("frozen integrator source is unavailable")
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    indexed: Dict[int, Set[str]] = {}
    branch_names: Set[str] = set()
    function_ranges: List[Dict[str, Any]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            function_ranges.append(
                {
                    "name": node.name,
                    "line": int(node.lineno),
                    "end_line": int(getattr(node, "end_lineno", node.lineno)),
                }
            )
        value: Optional[ast.AST] = None
        targets: Set[str] = set()
        if isinstance(node, ast.Assign):
            value = node.value
            for target in node.targets:
                targets.update(_target_names(target))
        elif isinstance(node, ast.AnnAssign):
            value = node.value
            targets.update(_target_names(node.target))
        elif isinstance(node, ast.NamedExpr):
            value = node.value
            targets.update(_target_names(node.target))
        if targets:
            predicate_targets = {
                name
                for name in targets
                if _predicate_expression(value)
                or any(token in name.lower() for token in PREDICATE_NAME_TOKENS)
            }
            if predicate_targets:
                indexed.setdefault(int(node.lineno), set()).update(predicate_targets)
        if isinstance(node, (ast.If, ast.While, ast.Assert)):
            test = node.test
            branch_names.update(_names_in_expression(test))
            indexed.setdefault(int(node.lineno), set()).update(_names_in_expression(test))
        if isinstance(node, ast.IfExp):
            branch_names.update(_names_in_expression(node.test))
            indexed.setdefault(int(node.lineno), set()).update(_names_in_expression(node.test))
    for line, names in list(indexed.items()):
        indexed[line] = {
            name
            for name in names
            if any(token in name.lower() for token in PREDICATE_NAME_TOKENS)
            or name in branch_names
        }
        if not indexed[line]:
            del indexed[line]
    return {
        "source_path": str(source_path),
        "source_sha256": sha256_bytes(source_path.read_bytes()),
        "integrate_rowwise_source_sha256": sha256_bytes(
            inspect.getsource(module.integrate_rowwise).encode("utf-8")
        ),
        "indexed_names_by_line": {
            str(line): sorted(names) for line, names in sorted(indexed.items())
        },
        "indexed_name_count": int(sum(len(names) for names in indexed.values())),
        "function_ranges": sorted(function_ranges, key=lambda item: item["line"]),
    }


def _family_for_name(name: str) -> Optional[str]:
    return stagei.predicate_family("$.{}".format(name))


def _numeric_summary(array: np.ndarray) -> Dict[str, Any]:
    flat = np.asarray(array, dtype=np.float64).reshape(-1)
    finite = np.isfinite(flat)
    result: Dict[str, Any] = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": sha256_array(array),
        "finite_rate": float(np.mean(finite)) if flat.size else 1.0,
    }
    if flat.size and np.any(finite):
        values = flat[finite]
        result.update(
            {
                "minimum": float(np.min(values)),
                "mean": float(np.mean(values)),
                "maximum": float(np.max(values)),
                "zero_rate": float(np.mean(values == 0.0)),
                "positive_rate": float(np.mean(values > 0.0)),
            }
        )
    return result


def summarize_trace_value(
    name: str,
    value: Any,
    *,
    rows: int,
    retention_minimum: float,
) -> Optional[Dict[str, Any]]:
    lower_name = name.lower()
    if any(token in lower_name for token in SKIP_VALUE_TOKENS) and not any(
        token in lower_name for token in PREDICATE_NAME_TOKENS
    ):
        return None
    if isinstance(value, (bool, np.bool_)):
        return {
            "kind": "boolean_scalar",
            "pass_rate": float(bool(value)),
            "pass_rule": "boolean_true",
            "value": bool(value),
        }
    if isinstance(value, (int, float, np.generic)) and not isinstance(value, complex):
        number = float(value)
        if not math.isfinite(number):
            return {
                "kind": "numeric_scalar",
                "finite": False,
            }
        result: Dict[str, Any] = {
            "kind": "numeric_scalar",
            "value": number,
            "finite": True,
        }
        if "retention" in lower_name:
            result.update(
                {
                    "pass_rate": float(number >= retention_minimum),
                    "pass_rule": "retention_ge_minimum",
                }
            )
        elif "scale" in lower_name or "motion" in lower_name:
            result.update(
                {
                    "pass_rate": float(number > 0.0),
                    "pass_rule": "positive_numeric",
                }
            )
        return result
    if not isinstance(value, np.ndarray):
        try:
            array = np.asarray(value)
        except Exception:
            return None
    else:
        array = value
    if array.dtype == object or array.size == 0:
        return None
    if array.ndim > 0 and rows > 0 and array.shape[0] not in (1, rows):
        # Non-row arrays are retained only when their names explicitly describe
        # a predicate.  This prevents point/candidate tensors from entering the
        # report while preserving scalar gate vectors.
        if not any(token in lower_name for token in PREDICATE_NAME_TOKENS):
            return None
    if np.issubdtype(array.dtype, np.bool_):
        bool_array = np.asarray(array, dtype=bool)
        return {
            "kind": "boolean_array",
            "shape": list(bool_array.shape),
            "dtype": str(array.dtype),
            "sha256": sha256_array(bool_array),
            "pass_rate": float(np.mean(bool_array)),
            "pass_rule": "boolean_true",
            "false_count": int(np.count_nonzero(~bool_array)),
            "true_count": int(np.count_nonzero(bool_array)),
        }
    if not np.issubdtype(array.dtype, np.number):
        return None
    result = _numeric_summary(array)
    result["kind"] = "numeric_array"
    numeric = np.asarray(array, dtype=np.float64)
    if "retention" in lower_name:
        result.update(
            {
                "pass_rate": float(np.mean(numeric >= retention_minimum)),
                "pass_rule": "retention_ge_minimum",
            }
        )
    elif "scale" in lower_name or "motion" in lower_name:
        result.update(
            {
                "pass_rate": float(np.mean(numeric > 0.0)),
                "pass_rule": "positive_numeric",
            }
        )
    elif np.all(np.isfinite(numeric)) and np.all(
        np.logical_or(np.isclose(numeric, 0.0), np.isclose(numeric, 1.0))
    ):
        result.update(
            {
                "pass_rate": float(np.mean(numeric > 0.5)),
                "pass_rule": "binary_numeric",
            }
        )
    return result


class _TraceCollector:
    def __init__(
        self,
        *,
        source_index: Mapping[str, Any],
        rows: int,
        retention_minimum: float,
        spec: ReadOnlyIntegratorTelemetrySpec,
    ) -> None:
        self.source_path = str(Path(source_index["source_path"]).resolve())
        self.indexed_names_by_line = {
            int(line): set(names)
            for line, names in source_index["indexed_names_by_line"].items()
        }
        self.rows = int(rows)
        self.retention_minimum = float(retention_minimum)
        self.spec = spec
        self.events: List[Dict[str, Any]] = []
        self._seen: Set[Tuple[str, int, str, str]] = set()
        self.truncated = False

    def __call__(self, frame: Any, event: str, arg: Any) -> Optional[Callable[..., Any]]:
        filename = str(Path(frame.f_code.co_filename).resolve())
        if filename != self.source_path:
            return None
        if event not in ("call", "line", "return", "exception"):
            return self
        if len(self.events) >= self.spec.maximum_trace_events:
            self.truncated = True
            return self
        line = int(frame.f_lineno)
        indexed_names: Set[str] = set(self.indexed_names_by_line.get(line, set()))
        # A line event occurs before the new line executes.  Include all indexed
        # locals already materialized in the frame, so assignments are observed
        # on the following line and again on return without source rewriting.
        all_indexed_names: Set[str] = set()
        for names in self.indexed_names_by_line.values():
            all_indexed_names.update(names)
        candidate_names = [
            name
            for name in frame.f_locals
            if name in indexed_names
            or name in all_indexed_names
            or any(token in name.lower() for token in PREDICATE_NAME_TOKENS)
        ]
        values: List[Dict[str, Any]] = []
        for name in sorted(candidate_names)[: self.spec.maximum_values_per_event]:
            summary = summarize_trace_value(
                name,
                frame.f_locals[name],
                rows=self.rows,
                retention_minimum=self.retention_minimum,
            )
            if summary is None:
                continue
            identity = str(summary.get("sha256", summary.get("value", summary.get("pass_rate"))))
            key = (frame.f_code.co_name, line, name, identity)
            if key in self._seen:
                continue
            self._seen.add(key)
            values.append(
                {
                    "name": name,
                    "family": _family_for_name(name),
                    "indexed_on_current_line": name in indexed_names,
                    **summary,
                }
            )
        if values or event in ("return", "exception"):
            self.events.append(
                {
                    "sequence": len(self.events),
                    "event": event,
                    "function": frame.f_code.co_name,
                    "line": line,
                    "values": values,
                    "exception_type": (
                        None
                        if event != "exception" or not isinstance(arg, tuple)
                        else getattr(arg[0], "__name__", str(arg[0]))
                    ),
                }
            )
        return self


def _exact_identity(left: Any, right: Any, path: str = "$") -> None:
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        if not isinstance(left, np.ndarray) or not isinstance(right, np.ndarray):
            raise ReadOnlyIntegratorTelemetryError(
                "telemetry changed result type at {}".format(path)
            )
        if left.dtype != right.dtype or left.shape != right.shape:
            raise ReadOnlyIntegratorTelemetryError(
                "telemetry changed array contract at {}".format(path)
            )
        if left.tobytes(order="C") != right.tobytes(order="C"):
            raise ReadOnlyIntegratorTelemetryError(
                "telemetry changed array bytes at {}".format(path)
            )
        return
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            raise ReadOnlyIntegratorTelemetryError(
                "telemetry changed mapping type at {}".format(path)
            )
        if list(left.keys()) != list(right.keys()):
            raise ReadOnlyIntegratorTelemetryError(
                "telemetry changed mapping keys/order at {}".format(path)
            )
        for key in left:
            _exact_identity(left[key], right[key], "{}.{}".format(path, key))
        return
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if type(left) is not type(right) or len(left) != len(right):
            raise ReadOnlyIntegratorTelemetryError(
                "telemetry changed sequence contract at {}".format(path)
            )
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            _exact_identity(left_item, right_item, "{}[{}]".format(path, index))
        return
    if isinstance(left, np.generic):
        left = left.item()
    if isinstance(right, np.generic):
        right = right.item()
    if isinstance(left, float) or isinstance(right, float):
        if type(left) is not type(right):
            raise ReadOnlyIntegratorTelemetryError(
                "telemetry changed scalar type at {}".format(path)
            )
        if math.isnan(float(left)) and math.isnan(float(right)):
            return
        if float(left).hex() != float(right).hex():
            raise ReadOnlyIntegratorTelemetryError(
                "telemetry changed float bits at {}".format(path)
            )
        return
    if type(left) is not type(right) or left != right:
        raise ReadOnlyIntegratorTelemetryError(
            "telemetry changed scalar value at {}".format(path)
        )


def traced_integrate_rowwise(
    *,
    integrate: Callable[..., Any],
    source_index: Mapping[str, Any],
    control: np.ndarray,
    direction: np.ndarray,
    definition: Any,
    context: Mapping[str, Any],
    integrator_spec: Any,
    telemetry_spec: ReadOnlyIntegratorTelemetrySpec,
) -> Tuple[Mapping[str, Any], Dict[str, Any]]:
    baseline = integrate(
        control=control,
        direction=direction,
        definition=definition,
        context=context,
        spec=integrator_spec,
    )
    collector = _TraceCollector(
        source_index=source_index,
        rows=int(np.asarray(control).shape[0]),
        retention_minimum=stagei._retention_minimum(integrator_spec),
        spec=telemetry_spec,
    )
    previous = sys.gettrace()
    if previous is not None:
        raise ReadOnlyIntegratorTelemetryError(
            "another Python trace function is already active"
        )
    try:
        sys.settrace(collector)
        traced = integrate(
            control=control,
            direction=direction,
            definition=definition,
            context=context,
            spec=integrator_spec,
        )
    finally:
        sys.settrace(previous)
    _exact_identity(baseline, traced)
    if not isinstance(traced, Mapping):
        raise ReadOnlyIntegratorTelemetryError(
            "frozen integrator result is not a mapping"
        )
    telemetry = {
        "schema": "phase314b_r258_stagej_readonly_trace_v1",
        "observer": "python_frame_line_trace",
        "stagee_source_sha256": source_index["source_sha256"],
        "integrate_rowwise_source_sha256": source_index[
            "integrate_rowwise_source_sha256"
        ],
        "returned_result_bit_exact_with_trace_disabled": True,
        "event_count": len(collector.events),
        "event_ceiling": telemetry_spec.maximum_trace_events,
        "truncated": collector.truncated,
        "events": collector.events,
    }
    if collector.truncated:
        raise ReadOnlyIntegratorTelemetryError("Stage-J trace event ceiling exceeded")
    if not collector.events:
        raise ReadOnlyIntegratorTelemetryError("Stage-J trace captured no integrator events")
    return traced, telemetry


def _row_norm(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    return np.linalg.norm(array.reshape(array.shape[0], -1), axis=1)


def _trace_leaf_records(telemetry: Mapping[str, Any]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for event in telemetry.get("events", []):
        for value in event.get("values", []):
            if value.get("pass_rate") is None:
                continue
            records.append(
                {
                    "key": "{}:{}:{}".format(
                        event["function"], event["line"], value["name"]
                    ),
                    "sequence": int(event["sequence"]),
                    "function": event["function"],
                    "line": int(event["line"]),
                    "name": value["name"],
                    "family": value.get("family"),
                    "pass_rate": float(value["pass_rate"]),
                    "pass_rule": value.get("pass_rule"),
                    "kind": value.get("kind"),
                    "sha256": value.get("sha256"),
                }
            )
    return records


def _aggregate_trace_leaves(telemetries: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for telemetry in telemetries:
        for record in _trace_leaf_records(telemetry):
            grouped.setdefault(record["key"], []).append(record)
    result: Dict[str, Dict[str, Any]] = {}
    for key, items in sorted(grouped.items()):
        rates = [float(item["pass_rate"]) for item in items]
        first = min(items, key=lambda item: (item["line"], item["sequence"]))
        result[key] = {
            "key": key,
            "function": first["function"],
            "line": first["line"],
            "name": first["name"],
            "family": first["family"],
            "minimum_pass_rate": float(min(rates)),
            "mean_pass_rate": float(np.mean(rates)),
            "maximum_pass_rate": float(max(rates)),
            "observation_count": len(items),
            "first_sequence": int(min(item["sequence"] for item in items)),
        }
    return result


def audit_direction_source(
    *,
    source_id: str,
    control: np.ndarray,
    base_direction: np.ndarray,
    context: Mapping[str, Any],
    integrator_spec: stagee258.ConstrainedIntegratorSpec,
    source_index: Mapping[str, Any],
    spec: ReadOnlyIntegratorTelemetrySpec,
) -> Dict[str, Any]:
    if source_id not in SOURCE_IDS:
        raise ValueError("Stage-J source ID changed")
    control_raw = np.asarray(control, dtype=np.float32)
    direction_raw = np.asarray(base_direction, dtype=np.float64)
    if direction_raw.shape != control_raw.shape:
        raise ReadOnlyIntegratorTelemetryError(
            "Stage-J direction/control shape changed"
        )
    rows = int(control_raw.shape[0])
    selected_columns: List[np.ndarray] = []
    multiplier_records: List[Dict[str, Any]] = []
    telemetries: List[Mapping[str, Any]] = []
    for multiplier in spec.scale_multipliers:
        proposed = direction_raw * float(multiplier)
        integration, telemetry = traced_integrate_rowwise(
            integrate=stagee258.integrate_rowwise,
            source_index=source_index,
            control=control_raw,
            direction=proposed,
            definition=stagef.fixed_integrator_definition(),
            context=context,
            integrator_spec=integrator_spec,
            telemetry_spec=spec,
        )
        if "selected_scale" not in integration or "candidate" not in integration:
            raise ReadOnlyIntegratorTelemetryError(
                "Stage-J integrator observable surface changed"
            )
        selected = np.asarray(integration["selected_scale"], dtype=np.float64).reshape(rows)
        candidate = np.asarray(integration["candidate"], dtype=np.float32)
        motion_norm = _row_norm(candidate.astype(np.float64) - control_raw.astype(np.float64))
        selected_positive = selected > spec.binary_tolerance
        motion_positive = motion_norm > spec.candidate_motion_epsilon
        if np.any(selected_positive != motion_positive):
            raise ReadOnlyIntegratorTelemetryError(
                "Stage-J selected-scale/motion observable is inconsistent"
            )
        selected_columns.append(selected)
        telemetries.append(telemetry)
        multiplier_records.append(
            {
                "scale_multiplier": float(multiplier),
                "proposed_direction_sha256": sha256_array(proposed),
                "selected_scale": {
                    "minimum": float(np.min(selected)),
                    "mean": float(np.mean(selected)),
                    "maximum": float(np.max(selected)),
                    "positive_rate": float(np.mean(selected_positive)),
                    "sha256": sha256_array(selected),
                },
                "candidate_motion": {
                    "minimum": float(np.min(motion_norm)),
                    "mean": float(np.mean(motion_norm)),
                    "maximum": float(np.max(motion_norm)),
                    "positive_rate": float(np.mean(motion_positive)),
                    "candidate_sha256": sha256_array(candidate),
                },
                "telemetry": {
                    "schema": telemetry["schema"],
                    "observer": telemetry["observer"],
                    "returned_result_bit_exact_with_trace_disabled": telemetry[
                        "returned_result_bit_exact_with_trace_disabled"
                    ],
                    "event_count": telemetry["event_count"],
                    "truncated": telemetry["truncated"],
                    "telemetry_sha256": sha256_bytes(stable_json_bytes(telemetry)),
                },
            }
        )
    selected_bank = np.stack(selected_columns, axis=1)
    accepted = selected_bank > spec.binary_tolerance
    return {
        "source_id": source_id,
        "objective_train_rows": rows,
        "direction_nonzero_rate": float(
            np.mean(_row_norm(direction_raw) > spec.candidate_motion_epsilon)
        ),
        "scale_bank_acceptance": {
            "rows_with_any_accepted_multiplier_rate": float(np.mean(np.any(accepted, axis=1))),
            "selected_scale_positive_rate": float(np.mean(accepted)),
            "selected_scale_bank_sha256": sha256_array(selected_bank),
        },
        "trace_leaf_index": _aggregate_trace_leaves(telemetries),
        "multiplier_records": multiplier_records,
    }


def compare_trace_surfaces(
    *,
    oof: Mapping[str, Any],
    raw: Mapping[str, Any],
    projected: Mapping[str, Any],
    spec: ReadOnlyIntegratorTelemetrySpec,
) -> Dict[str, Any]:
    oof_accept = float(
        oof["scale_bank_acceptance"]["rows_with_any_accepted_multiplier_rate"]
    )
    raw_accept = float(
        raw["scale_bank_acceptance"]["rows_with_any_accepted_multiplier_rate"]
    )
    projected_accept = float(
        projected["scale_bank_acceptance"]["rows_with_any_accepted_multiplier_rate"]
    )
    comparator_id = "projected_oracle" if projected_accept >= raw_accept else "raw_oracle"
    comparator = projected if comparator_id == "projected_oracle" else raw
    comparator_accept = max(raw_accept, projected_accept)
    if comparator_accept < spec.oracle_acceptance_rate_min:
        return {
            "locus": "oracle_control_not_admitted",
            "discriminator": None,
            "comparator_source": comparator_id,
            "oof_acceptance_rate": oof_accept,
            "raw_oracle_acceptance_rate": raw_accept,
            "projected_oracle_acceptance_rate": projected_accept,
            "candidate_discriminators": [],
        }
    if oof_accept > spec.oof_acceptance_rate_max:
        return {
            "locus": "stageh_rejection_not_reproduced",
            "discriminator": None,
            "comparator_source": comparator_id,
            "oof_acceptance_rate": oof_accept,
            "raw_oracle_acceptance_rate": raw_accept,
            "projected_oracle_acceptance_rate": projected_accept,
            "candidate_discriminators": [],
        }
    oof_index = oof["trace_leaf_index"]
    comparator_index = comparator["trace_leaf_index"]
    candidates: List[Dict[str, Any]] = []
    for key in sorted(set(oof_index).intersection(comparator_index)):
        oof_item = oof_index[key]
        control_item = comparator_index[key]
        oof_rate = float(oof_item["minimum_pass_rate"])
        control_rate = float(control_item["minimum_pass_rate"])
        family = oof_item.get("family") or control_item.get("family")
        if (
            control_rate >= spec.oracle_predicate_pass_rate_min
            and oof_rate <= spec.oof_predicate_pass_rate_max
        ):
            candidates.append(
                {
                    "key": key,
                    "function": oof_item["function"],
                    "line": int(oof_item["line"]),
                    "name": oof_item["name"],
                    "family": family,
                    "oof_minimum_pass_rate": oof_rate,
                    "oracle_minimum_pass_rate": control_rate,
                    "oof_first_sequence": int(oof_item["first_sequence"]),
                    "oracle_first_sequence": int(control_item["first_sequence"]),
                }
            )
    candidates.sort(
        key=lambda item: (
            item["line"],
            item["oof_first_sequence"],
            item["function"],
            item["name"],
        )
    )
    candidates = candidates[: spec.maximum_discriminator_records]
    discriminator = candidates[0] if candidates else None
    if discriminator is None:
        locus = "readonly_trace_does_not_identify_rejection"
    elif discriminator.get("family") is None:
        locus = "rejection_identified_without_named_family"
    else:
        locus = "oof_first_diverges_at_{}".format(discriminator["family"])
    return {
        "locus": locus,
        "discriminator": discriminator,
        "comparator_source": comparator_id,
        "oof_acceptance_rate": oof_accept,
        "raw_oracle_acceptance_rate": raw_accept,
        "projected_oracle_acceptance_rate": projected_accept,
        "candidate_discriminators": candidates,
    }


def classify(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not records:
        raise ReadOnlyIntegratorTelemetryError("Stage-J audit has no records")
    loci: Dict[str, int] = {}
    families: Dict[str, int] = {}
    keys: Dict[str, int] = {}
    for record in records:
        comparison = record["trace_comparison"]
        locus = str(comparison["locus"])
        loci[locus] = loci.get(locus, 0) + 1
        discriminator = comparison.get("discriminator")
        if isinstance(discriminator, Mapping):
            family = discriminator.get("family")
            key = discriminator.get("key")
            if isinstance(family, str):
                families[family] = families.get(family, 0) + 1
            if isinstance(key, str):
                keys[key] = keys.get(key, 0) + 1
    total = len(records)
    if loci.get("oracle_control_not_admitted", 0):
        root = "phase314b_r258_stagej_frozen_integrator_oracle_replay_not_admitted"
        next_path = "RESTORE_STAGEE_ORACLE_REPLAY_BEFORE_SURROGATE_REPAIR"
        primary = "oracle_replay"
    elif loci.get("stageh_rejection_not_reproduced", 0):
        root = "phase314b_r258_stagej_stageh_rejection_not_reproduced"
        next_path = "AUDIT_STAGEH_STAGEJ_REPLAY_IDENTITY"
        primary = "replay_identity"
    elif len(families) == 1 and sum(families.values()) == total:
        family = next(iter(families))
        root, next_path = FAMILY_OUTCOMES.get(
            family,
            (
                "phase314b_r258_stagej_rejection_identified_without_policy_mapping",
                "REVIEW_IDENTIFIED_INTEGRATOR_PREDICATE_BEFORE_MODEL_REPAIR",
            ),
        )
        primary = family
    elif loci.get("readonly_trace_does_not_identify_rejection", 0):
        root = "phase314b_r258_stagej_readonly_trace_surface_insufficient"
        next_path = "ADD_EXPLICIT_OPTIONAL_PREDICATE_CALLBACK_WITH_BIT_EXACT_OFF_ON_REPLAY"
        primary = "trace_observability"
    elif families:
        root = "phase314b_r258_stagej_rejection_predicates_are_heterogeneous"
        next_path = "STRATIFY_OOF_REJECTION_BY_TRACED_PREDICATE"
        primary = "heterogeneous_rejection_predicates"
    else:
        root = "phase314b_r258_stagej_rejection_identified_without_named_family"
        next_path = "MAP_TRACED_SOURCE_LOCATION_TO_FROZEN_INTEGRATOR_PREDICATE"
        primary = "source_location_without_family"
    return {
        "root_cause": root,
        "required_next_path": next_path,
        "primary_failure_locus": primary,
        "record_count": total,
        "record_locus_counts": dict(sorted(loci.items())),
        "discriminator_family_counts": dict(sorted(families.items())),
        "discriminator_key_counts": dict(sorted(keys.items())),
        "all_records_share_primary_locus": len(loci) == 1,
        "all_records_share_discriminator_family": (
            len(families) == 1 and sum(families.values()) == total
        ),
        "all_records_share_discriminator_key": (
            len(keys) == 1 and sum(keys.values()) == total
        ),
    }


def run_calibration(
    *,
    root: Path,
    environment: Mapping[str, Any],
    spec: Optional[ReadOnlyIntegratorTelemetrySpec] = None,
    stagef_spec: Optional[stagef.ConstraintAwareSpec] = None,
    direction_spec: Optional[staged258.DirectionSurrogateSpec] = None,
    integrator_spec: Optional[stagee258.ConstrainedIntegratorSpec] = None,
) -> Dict[str, Any]:
    active = ReadOnlyIntegratorTelemetrySpec() if spec is None else spec
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
        raise ReadOnlyIntegratorTelemetryError("portable compatibility SHA changed")
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
        raise ReadOnlyIntegratorTelemetryError("Stage-J ULP policy replay changed")
    active_stagef = stagef.replace(
        active_stagef,
        translation_float32_z_ulp_factor=EXPECTED_SELECTED_ULP_FACTOR,
    )
    active_stagef.validate()
    source_index = predicate_source_index(stagee258)

    oracle_targets: Dict[int, Mapping[str, Any]] = {}
    controls: Dict[int, Dict[str, Any]] = {}
    for timestep in active.timesteps:
        t = int(timestep)
        control = np.asarray(context["objective_control_predictions"][t], dtype=np.float32)
        target = np.asarray(context["objective_target"], dtype=np.float32)
        oracle = stagef.generate_projected_oracle_target(
            control=control,
            target=target,
            groups=context["objective_groups"],
            condition_name=context["objective_condition_name"],
            timestep=t,
            context=context,
            direction_spec=active_direction,
            integrator_spec=active_integrator,
            spec=active_stagef,
        )
        oracle_targets[t] = oracle
        raw_direction = target.astype(np.float64) - control.astype(np.float64)
        projected_direction = (
            np.asarray(oracle["candidate"], dtype=np.float64)
            - control.astype(np.float64)
        )
        controls[t] = {
            "raw_oracle": audit_direction_source(
                source_id="raw_oracle",
                control=control,
                base_direction=raw_direction,
                context=context,
                integrator_spec=active_integrator,
                source_index=source_index,
                spec=active,
            ),
            "projected_oracle": audit_direction_source(
                source_id="projected_oracle",
                control=control,
                base_direction=projected_direction,
                context=context,
                integrator_spec=active_integrator,
                source_index=source_index,
                spec=active,
            ),
        }

    records: List[Dict[str, Any]] = []
    for base_direction_id in BASE_DIRECTION_IDS:
        definition = stagef.definition_by_id(base_direction_id)
        for timestep in active.timesteps:
            t = int(timestep)
            control = np.asarray(context["objective_control_predictions"][t], dtype=np.float32)
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
                oracle_target=oracle_targets[t],
                condition_name=context["objective_condition_name"],
                fold_assignment=context["objective_fold_assignment"],
                stagef_spec=active_stagef,
                grouped_cv_folds=active.grouped_cv_folds,
            )
            oof_audit = audit_direction_source(
                source_id="oof_surrogate",
                control=control,
                base_direction=oof["prediction"],
                context=context,
                integrator_spec=active_integrator,
                source_index=source_index,
                spec=active,
            )
            comparison = compare_trace_surfaces(
                oof=oof_audit,
                raw=controls[t]["raw_oracle"],
                projected=controls[t]["projected_oracle"],
                spec=active,
            )
            records.append(
                {
                    "base_direction_id": base_direction_id,
                    "timestep": t,
                    "feature_mode": definition.feature_mode,
                    "feature_sha256": sha256_array(features),
                    "oof_prediction_sha256": oof["prediction_sha256"],
                    "oof_integrator_telemetry": oof_audit,
                    "trace_comparison": comparison,
                    "objective_train_rows": int(control.shape[0]),
                    "holdout_used": False,
                }
            )
    classification = classify(records)
    contract = {
        "schema": "phase314b_r258_stagej_readonly_integrator_telemetry_contract_v1",
        "phase": PHASE,
        "base_evidence_commit": BASE_EVIDENCE_COMMIT,
        "independent_goals_commit": INDEPENDENT_GOALS_COMMIT,
        "base_report_sha256": EXPECTED_BASE_REPORT_SHA256,
        "base_single_run_sha256": EXPECTED_BASE_SINGLE_RUN_SHA256,
        "environment": dict(environment),
        "cold_main_worker_context": cold,
        "control_capture": captured["control_identity"],
        "telemetry_spec": asdict(active),
        "stagef_spec": asdict(active_stagef),
        "direction_spec": asdict(active_direction),
        "integrator_spec": asdict(active_integrator),
        "fixed_integrator": asdict(stagef.fixed_integrator_definition()),
        "base_direction_ids": list(BASE_DIRECTION_IDS),
        "scale_bank": list(SCALE_MULTIPLIERS),
        "source_ids": list(SOURCE_IDS),
        "stagee_source_index": source_index,
        "data_boundary": {
            "objective_train_only": True,
            "selection_holdout_evaluated": False,
            "selection_holdout_used_for_fit_or_selection": False,
            "frozen_probe_accessed": False,
            "target_used_only_for_objective_train_oracle_controls": True,
        },
        "mechanism_boundary": {
            "stagee_source_modified": False,
            "integrator_decisions_modified": False,
            "telemetry_observer_is_external": True,
            "every_traced_result_compared_to_untraced_result": True,
            "new_ranker_fitted": False,
            "candidate_matrix_changed": False,
            "scale_bank_changed": False,
            "stagef_feature_definition_modified": False,
            "ulp_policy_changed": False,
        },
    }
    contract["contract_sha256"] = sha256_bytes(stable_json_bytes(contract))
    audit = {
        "schema": "phase314b_r258_stagej_readonly_integrator_telemetry_audit_v1",
        "control_telemetry_by_timestep": {
            str(key): value for key, value in sorted(controls.items())
        },
        "oof_records": records,
        "classification": classification,
    }
    audit["audit_sha256"] = sha256_bytes(stable_json_bytes(audit))
    return {
        "schema": "phase314b_r258_stagej_readonly_integrator_telemetry_result_v1",
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
            **base,
        },
        "contract": contract,
        "readonly_integrator_telemetry": audit,
        "readonly_telemetry_completed": True,
        "all_traced_results_bit_exact": True,
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
