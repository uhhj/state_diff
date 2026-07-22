"""Phase3.14b-r2.5.8 Stage K explicit frozen-integrator predicate callback.

Stage J proved that an external Python trace is read-only but cannot observe the
predicate surface that separates oracle directions from grouped-OOF surrogate
directions.  Stage K adds one optional, default-disabled callback to the frozen
Stage-E ``integrate_rowwise`` function.  The callback receives recursively
immutable scalar summaries emitted in the actual scale-search order.

Every callback-enabled integration is paired with a callback-disabled call using
identical arguments, and the complete returned mappings must be recursively
byte-exact.  The callback cannot alter candidate generation, scale selection,
thresholds, topology checks, or the returned object.  The audit remains
objective-train-only; holdout and the frozen probe remain closed.
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
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ccda_phase3 import phase314b_r258_staged_direction_surrogate as staged258
from ccda_phase3 import phase314b_r258_stagee_constrained_integrator as stagee258
from ccda_phase3 import phase314b_r258_stagef_constraint_aware_surrogate as stagef
from ccda_phase3 import phase314b_r258_stageh_candidate_descriptor_identifiability as stageh
from ccda_phase3 import phase314b_r258_stagej_readonly_integrator_telemetry as stagej


PHASE = "Phase3.14b-r2.5.8 Stage K"
PHASE_ID = "phase314b_r258_stagek"
BASE_IMPLEMENTATION_COMMIT = "3c6004de3418171966629c59f378c1cba6f6bc2d"
BASE_EVIDENCE_COMMIT = "90375c4a21e4a798539bc0fabea79787f2011966"
EXPECTED_SUBMODULE_COMMIT = "633a88752445cf5d6776ed374fdbbdb35f93050c"
BASE_REPORT = "reports/phase3_14b_r258_stagej_readonly_integrator_telemetry_summary.json"
EXPECTED_BASE_REPORT_SHA256 = "d09106590f5916f8aafb2c847cc296393c0e755bce79e2578505644172469bf1"
EXPECTED_BASE_SINGLE_RUN_SHA256 = "37c0bdedaa5e85dae989591cc8421b957414cd882939189ad4f057c120044d22"
EXPECTED_BASE_ROOT_CAUSE = "phase314b_r258_stagej_readonly_trace_surface_insufficient"
EXPECTED_BASE_NEXT_PATH = "ADD_EXPLICIT_OPTIONAL_PREDICATE_CALLBACK_WITH_BIT_EXACT_OFF_ON_REPLAY"
EXPECTED_PRE_CALLBACK_STAGEE_SHA256 = "d952e169196916b3ac9ad7a86d1944cfe48281599463f053718a27ea89238f8c"
EXPECTED_SELECTED_ULP_FACTOR = 1.0
STAGEE_RELATIVE_PATH = "ccda_phase3/phase314b_r258_stagee_constrained_integrator.py"

BASE_DIRECTION_IDS: Tuple[str, ...] = stageh.BASE_DIRECTION_IDS
SCALE_MULTIPLIERS: Tuple[float, ...] = stageh.SCALE_MULTIPLIERS
SOURCE_IDS: Tuple[str, ...] = ("raw_oracle", "projected_oracle", "oof_surrogate")
PREDICATE_ORDER: Tuple[str, ...] = tuple(stagee258.PREDICATE_ORDER)

PREDICATE_OUTCOMES: Mapping[str, Tuple[str, str]] = {
    "finite_state": (
        "phase314b_r258_stagek_oof_rejected_by_finite_state",
        "AUDIT_NUMERICAL_STABILITY_OF_SURROGATE_RECONSTRUCTION",
    ),
    "upper_segment_geometry": (
        "phase314b_r258_stagek_oof_rejected_by_upper_segment_geometry",
        "CALIBRATE_LOCAL_SEGMENT_DIRECTION_CORRECTION_BEFORE_FROZEN_INTEGRATION",
    ),
    "lower_segment_geometry": (
        "phase314b_r258_stagek_oof_rejected_by_lower_segment_geometry",
        "CALIBRATE_LOCAL_SEGMENT_DIRECTION_CORRECTION_BEFORE_FROZEN_INTEGRATION",
    ),
    "coordinate_recenter": (
        "phase314b_r258_stagek_oof_rejected_by_coordinate_recenter",
        "AUDIT_OOF_DIRECTION_LOCAL_FRAME_COMPATIBILITY",
    ),
    "coordinate_geometry": (
        "phase314b_r258_stagek_oof_rejected_by_coordinate_geometry",
        "AUDIT_OOF_DIRECTION_LOCAL_FRAME_COMPATIBILITY",
    ),
    "reconstruction_bounds": (
        "phase314b_r258_stagek_oof_rejected_by_reconstruction_bounds",
        "CALIBRATE_LOCAL_SEGMENT_DIRECTION_CORRECTION_BEFORE_FROZEN_INTEGRATION",
    ),
    "segment_geometry": (
        "phase314b_r258_stagek_oof_rejected_by_segment_geometry",
        "CALIBRATE_LOCAL_SEGMENT_DIRECTION_CORRECTION_BEFORE_FROZEN_INTEGRATION",
    ),
    "direction_retention": (
        "phase314b_r258_stagek_oof_rejected_by_direction_retention",
        "CALIBRATE_INTEGRATOR_COMPATIBLE_DIRECTION_BASIS_ON_OBJECTIVE_TRAIN_ONLY",
    ),
    "displacement": (
        "phase314b_r258_stagek_oof_rejected_by_displacement",
        "CALIBRATE_EXECUTABLE_DIRECTION_MAGNITUDE_PARAMETERIZATION",
    ),
    "topology": (
        "phase314b_r258_stagek_oof_rejected_by_topology",
        "CALIBRATE_TOPOLOGY_COMPATIBLE_DIRECTION_SURROGATE_ON_OBJECTIVE_TRAIN_ONLY",
    ),
}


class ExplicitPredicateCallbackError(RuntimeError):
    """Raised when Stage-K evidence, identity, or callback invariants fail."""


@dataclass(frozen=True)
class ExplicitPredicateCallbackSpec:
    timesteps: Tuple[int, ...] = stagef.TIMESTEPS
    grouped_cv_folds: int = 6
    scale_multipliers: Tuple[float, ...] = SCALE_MULTIPLIERS
    oracle_acceptance_rate_min: float = 0.95
    oof_acceptance_rate_max: float = 0.05
    oracle_predicate_pass_rate_min: float = 0.90
    oof_predicate_pass_rate_max: float = 0.10
    oof_first_failure_rate_min: float = 0.50
    oracle_first_failure_rate_max: float = 0.10
    binary_tolerance: float = 1.0e-12
    candidate_motion_epsilon: float = 1.0e-12
    maximum_events_per_call: int = 16

    def validate(self) -> None:
        if self.timesteps != stagef.TIMESTEPS:
            raise ValueError("Stage-K timestep population changed")
        if self.grouped_cv_folds != 6:
            raise ValueError("Stage-K grouped fold count changed")
        if self.scale_multipliers != SCALE_MULTIPLIERS:
            raise ValueError("Stage-K external scale bank changed")
        if self.maximum_events_per_call != 16:
            raise ValueError("Stage-K event ceiling changed")
        if tuple(stagee258.PREDICATE_ORDER) != PREDICATE_ORDER:
            raise ValueError("Stage-K predicate order changed")
        if set(PREDICATE_ORDER) != set(PREDICATE_OUTCOMES):
            raise ValueError("Stage-K predicate policy mapping is incomplete")
        for value in (
            self.oracle_acceptance_rate_min,
            self.oracle_predicate_pass_rate_min,
            self.oof_first_failure_rate_min,
        ):
            if not 0.0 < float(value) <= 1.0:
                raise ValueError("Stage-K positive threshold is invalid")
        for value in (
            self.oof_acceptance_rate_max,
            self.oof_predicate_pass_rate_max,
            self.oracle_first_failure_rate_max,
        ):
            if not 0.0 <= float(value) < 1.0:
                raise ValueError("Stage-K rejection threshold is invalid")
        if self.binary_tolerance <= 0.0 or self.candidate_motion_epsilon <= 0.0:
            raise ValueError("Stage-K numerical tolerance is invalid")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_array(value: np.ndarray) -> str:
    return stagej.sha256_array(value)


def stable_json_bytes(value: Any) -> bytes:
    return stagej.stable_json_bytes(value)


def jsonable(value: Any) -> Any:
    return stagej.jsonable(value)


def _git(root: Path, *args: str) -> bytes:
    result = subprocess.run(
        ["git", *args],
        cwd=str(root),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout


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


TELEMETRY_HELPERS = {
    "_freeze_predicate_telemetry_value",
    "_predicate_rate",
    "_predicate_telemetry_attempt_event",
    "_predicate_telemetry_final_event",
    "_emit_predicate_telemetry",
}
TELEMETRY_CONSTANTS = {
    "PREDICATE_CALLBACK_SCHEMA",
    "PREDICATE_ORDER",
    "PredicateTelemetryCallback",
}


class _StripCallbackEmission(ast.NodeTransformer):
    def visit_If(self, node: ast.If) -> Optional[ast.AST]:
        if (
            isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "predicate_callback"
        ):
            return None
        return self.generic_visit(node)


class _StripTelemetry(ast.NodeTransformer):
    def visit_ImportFrom(self, node: ast.ImportFrom) -> Optional[ast.AST]:
        if node.module == "types":
            names = [item for item in node.names if item.name != "MappingProxyType"]
            if not names:
                return None
            node.names = names
        if node.module == "typing":
            node.names = [item for item in node.names if item.name != "Callable"]
        return node

    def visit_Assign(self, node: ast.Assign) -> Optional[ast.AST]:
        names = {
            target.id
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        if names & TELEMETRY_CONSTANTS:
            return None
        return self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> Optional[ast.AST]:
        if isinstance(node.target, ast.Name) and node.target.id in TELEMETRY_CONSTANTS:
            return None
        return self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Optional[ast.AST]:
        if node.name in TELEMETRY_HELPERS:
            return None
        if node.name == "integrate_rowwise":
            pairs = list(zip(node.args.kwonlyargs, node.args.kw_defaults))
            pairs = [(arg, default) for arg, default in pairs if arg.arg != "predicate_callback"]
            node.args.kwonlyargs = [item[0] for item in pairs]
            node.args.kw_defaults = [item[1] for item in pairs]
            stripped = _StripCallbackEmission().visit(node)
            if not isinstance(stripped, ast.FunctionDef):
                raise ExplicitPredicateCallbackError(
                    "integrate_rowwise disappeared during telemetry normalization"
                )
            node = stripped
        return self.generic_visit(node)


def normalized_integrator_ast(source: str) -> str:
    tree = ast.parse(source)
    normalized = _StripTelemetry().visit(tree)
    ast.fix_missing_locations(normalized)
    return ast.dump(normalized, annotate_fields=True, include_attributes=False)


def validate_base_evidence(root: Path) -> Dict[str, Any]:
    repository_root = Path(root).resolve()
    report_path = repository_root / BASE_REPORT
    if not report_path.is_file():
        raise ExplicitPredicateCallbackError("Stage-J base report is missing")
    report_bytes = report_path.read_bytes()
    if sha256_bytes(report_bytes) != EXPECTED_BASE_REPORT_SHA256:
        raise ExplicitPredicateCallbackError("Stage-J base report SHA changed")
    payload = json.loads(report_bytes.decode("utf-8"))
    if payload.get("execution_verdict") != "PASS":
        raise ExplicitPredicateCallbackError("Stage-J execution verdict changed")
    if payload.get("scientific_status") != "BLOCKED":
        raise ExplicitPredicateCallbackError("Stage-J scientific status changed")
    if payload.get("root_cause") != EXPECTED_BASE_ROOT_CAUSE:
        raise ExplicitPredicateCallbackError("Stage-J root cause changed")
    if payload.get("required_next_path") != EXPECTED_BASE_NEXT_PATH:
        raise ExplicitPredicateCallbackError("Stage-J next path changed")
    if _extract_single_run_sha(payload) != EXPECTED_BASE_SINGLE_RUN_SHA256:
        raise ExplicitPredicateCallbackError("Stage-J single-run SHA changed")

    parent_source = _git(
        repository_root,
        "show",
        "{}:{}".format(BASE_EVIDENCE_COMMIT, STAGEE_RELATIVE_PATH),
    )
    if sha256_bytes(parent_source) != EXPECTED_PRE_CALLBACK_STAGEE_SHA256:
        raise ExplicitPredicateCallbackError("pre-callback Stage-E blob changed")
    current_path = repository_root / STAGEE_RELATIVE_PATH
    current_source = current_path.read_bytes()
    if normalized_integrator_ast(current_source.decode("utf-8")) != normalized_integrator_ast(
        parent_source.decode("utf-8")
    ):
        raise ExplicitPredicateCallbackError(
            "Stage-E changes are not telemetry-only after AST normalization"
        )
    callback_parameter = inspect.signature(stagee258.integrate_rowwise).parameters.get(
        "predicate_callback"
    )
    if callback_parameter is None or callback_parameter.default is not None:
        raise ExplicitPredicateCallbackError(
            "Stage-E predicate callback is not optional/default-disabled"
        )
    return {
        "base_report_sha256": EXPECTED_BASE_REPORT_SHA256,
        "base_single_run_sha256": EXPECTED_BASE_SINGLE_RUN_SHA256,
        "pre_callback_stagee_sha256": EXPECTED_PRE_CALLBACK_STAGEE_SHA256,
        "current_stagee_sha256": sha256_bytes(current_source),
        "telemetry_only_ast_equivalence": True,
        "predicate_callback_default_is_none": True,
    }


def _assert_scalar_tree(value: Any, path: str = "$") -> None:
    if isinstance(value, Mapping):
        if not isinstance(value, MappingProxyType):
            raise ExplicitPredicateCallbackError(
                "callback mapping is mutable at {}".format(path)
            )
        for key, item in value.items():
            if not isinstance(key, str):
                raise ExplicitPredicateCallbackError(
                    "callback key is not text at {}".format(path)
                )
            _assert_scalar_tree(item, "{}.{}".format(path, key))
        return
    if isinstance(value, tuple):
        for index, item in enumerate(value):
            _assert_scalar_tree(item, "{}[{}]".format(path, index))
        return
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float) and math.isfinite(value):
        return
    raise ExplicitPredicateCallbackError(
        "callback contains a non-scalar value at {}: {}".format(path, type(value))
    )


class PredicateEventCollector:
    def __init__(self, spec: ExplicitPredicateCallbackSpec) -> None:
        self.spec = spec
        self.events: List[Dict[str, Any]] = []

    def __call__(self, event: Mapping[str, Any]) -> None:
        _assert_scalar_tree(event)
        normalized = jsonable(event)
        if normalized.get("schema") != stagee258.PREDICATE_CALLBACK_SCHEMA:
            raise ExplicitPredicateCallbackError("callback event schema changed")
        event_type = normalized.get("event_type")
        if event_type not in ("scale_attempt", "final_summary"):
            raise ExplicitPredicateCallbackError("callback event type changed")
        if len(self.events) >= self.spec.maximum_events_per_call:
            raise ExplicitPredicateCallbackError("callback event ceiling exceeded")
        if event_type == "scale_attempt":
            if tuple(normalized.get("predicate_order", ())) != PREDICATE_ORDER:
                raise ExplicitPredicateCallbackError("callback predicate order changed")
            counts = normalized.get("first_failed_counts")
            if not isinstance(counts, Mapping):
                raise ExplicitPredicateCallbackError("callback lacks first-failure counts")
            active = int(normalized.get("active_row_count", -1))
            accepted = int(normalized.get("accepted_count", -1))
            if sum(int(counts[name]) for name in PREDICATE_ORDER) + accepted != active:
                raise ExplicitPredicateCallbackError("callback population does not close")
        self.events.append(normalized)

    def finalize(self) -> Dict[str, Any]:
        attempts = [item for item in self.events if item["event_type"] == "scale_attempt"]
        finals = [item for item in self.events if item["event_type"] == "final_summary"]
        if not attempts or len(finals) != 1:
            raise ExplicitPredicateCallbackError(
                "callback did not emit attempts plus one final summary"
            )
        attempted_scales = [float(item["attempted_scale"]) for item in attempts]
        if attempted_scales != sorted(attempted_scales, reverse=True):
            raise ExplicitPredicateCallbackError("callback scale order changed")
        return {
            "schema": "phase314b_r258_stagek_callback_capture_v1",
            "event_count": len(self.events),
            "attempt_count": len(attempts),
            "final_summary_count": len(finals),
            "attempted_scales": attempted_scales,
            "events": self.events,
            "events_sha256": sha256_bytes(stable_json_bytes(self.events)),
            "all_events_recursively_immutable": True,
            "all_events_scalar_only": True,
        }


def _row_norm(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    return np.sqrt(np.sum(array.reshape(array.shape[0], -1) ** 2, axis=1))


def callback_integrate_rowwise(
    *,
    control: np.ndarray,
    direction: np.ndarray,
    context: Mapping[str, Any],
    integrator_spec: stagee258.ConstrainedIntegratorSpec,
    callback_spec: ExplicitPredicateCallbackSpec,
) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
    definition = stagef.fixed_integrator_definition()
    baseline = stagee258.integrate_rowwise(
        control=control,
        direction=direction,
        definition=definition,
        context=context,
        spec=integrator_spec,
        predicate_callback=None,
    )
    collector = PredicateEventCollector(callback_spec)
    observed = stagee258.integrate_rowwise(
        control=control,
        direction=direction,
        definition=definition,
        context=context,
        spec=integrator_spec,
        predicate_callback=collector,
    )
    stagej._exact_identity(baseline, observed, "$.integrator")
    telemetry = collector.finalize()
    telemetry["callback_disabled_candidate_sha256"] = str(baseline["candidate_sha256"])
    telemetry["callback_enabled_candidate_sha256"] = str(observed["candidate_sha256"])
    telemetry["callback_disabled_selected_scale_sha256"] = str(
        baseline["selected_scale_sha256"]
    )
    telemetry["callback_enabled_selected_scale_sha256"] = str(
        observed["selected_scale_sha256"]
    )
    telemetry["returned_result_bit_exact"] = True
    return observed, telemetry


def aggregate_attempt_telemetry(captures: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    pass_numerators = {name: 0 for name in PREDICATE_ORDER}
    pass_denominators = {name: 0 for name in PREDICATE_ORDER}
    first_failures = {name: 0 for name in PREDICATE_ORDER}
    active_total = 0
    accepted_total = 0
    attempt_total = 0
    for capture in captures:
        for event in capture["events"]:
            if event["event_type"] != "scale_attempt":
                continue
            attempt_total += 1
            active = int(event["active_row_count"])
            active_total += active
            accepted_total += int(event["accepted_count"])
            counts = event["predicate_pass_counts"]
            failures = event["first_failed_counts"]
            topology_denominator = int(event["topology_checked_count"])
            for name in PREDICATE_ORDER:
                denominator = topology_denominator if name == "topology" else active
                pass_denominators[name] += denominator
                pass_numerators[name] += int(counts[name])
                first_failures[name] += int(failures[name])
    predicate_pass_rates = {
        name: (
            1.0
            if pass_denominators[name] == 0
            else float(pass_numerators[name] / pass_denominators[name])
        )
        for name in PREDICATE_ORDER
    }
    first_failure_rates = {
        name: (0.0 if active_total == 0 else float(first_failures[name] / active_total))
        for name in PREDICATE_ORDER
    }
    return {
        "attempt_count": attempt_total,
        "active_row_attempt_count": active_total,
        "accepted_row_attempt_count": accepted_total,
        "predicate_pass_counts": pass_numerators,
        "predicate_pass_denominators": pass_denominators,
        "predicate_pass_rates": predicate_pass_rates,
        "first_failed_counts": first_failures,
        "first_failed_rates": first_failure_rates,
    }


def audit_direction_source(
    *,
    source_id: str,
    control: np.ndarray,
    base_direction: np.ndarray,
    context: Mapping[str, Any],
    integrator_spec: stagee258.ConstrainedIntegratorSpec,
    spec: ExplicitPredicateCallbackSpec,
) -> Dict[str, Any]:
    control_raw = np.asarray(control, dtype=np.float32)
    direction_raw = np.asarray(base_direction, dtype=np.float64)
    rows = int(control_raw.shape[0])
    selected_columns: List[np.ndarray] = []
    captures: List[Mapping[str, Any]] = []
    multiplier_records: List[Dict[str, Any]] = []
    for multiplier in spec.scale_multipliers:
        proposed = direction_raw * float(multiplier)
        integration, telemetry = callback_integrate_rowwise(
            control=control_raw,
            direction=proposed,
            context=context,
            integrator_spec=integrator_spec,
            callback_spec=spec,
        )
        selected = np.asarray(integration["selected_scale"], dtype=np.float64).reshape(rows)
        candidate = np.asarray(integration["candidate"], dtype=np.float32)
        motion = _row_norm(candidate.astype(np.float64) - control_raw.astype(np.float64))
        selected_positive = selected > spec.binary_tolerance
        motion_positive = motion > spec.candidate_motion_epsilon
        if np.any(selected_positive != motion_positive):
            raise ExplicitPredicateCallbackError(
                "selected-scale and candidate-motion observables disagree"
            )
        selected_columns.append(selected)
        captures.append(telemetry)
        multiplier_records.append(
            {
                "scale_multiplier": float(multiplier),
                "proposed_direction_sha256": sha256_array(proposed),
                "selected_scale_positive_rate": float(np.mean(selected_positive)),
                "selected_scale_sha256": sha256_array(selected),
                "candidate_motion_positive_rate": float(np.mean(motion_positive)),
                "candidate_sha256": sha256_array(candidate),
                "callback_capture": {
                    key: value for key, value in telemetry.items() if key != "events"
                },
            }
        )
    selected_bank = np.stack(selected_columns, axis=1)
    accepted = selected_bank > spec.binary_tolerance
    aggregate = aggregate_attempt_telemetry(captures)
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
        "predicate_aggregate": aggregate,
        "multiplier_records": multiplier_records,
        "all_callback_results_bit_exact": True,
        "callback_pair_count": len(spec.scale_multipliers),
    }


def compare_predicate_surfaces(
    *,
    oof: Mapping[str, Any],
    raw: Mapping[str, Any],
    projected: Mapping[str, Any],
    spec: ExplicitPredicateCallbackSpec,
) -> Dict[str, Any]:
    oof_accept = float(oof["scale_bank_acceptance"]["rows_with_any_accepted_multiplier_rate"])
    raw_accept = float(raw["scale_bank_acceptance"]["rows_with_any_accepted_multiplier_rate"])
    projected_accept = float(
        projected["scale_bank_acceptance"]["rows_with_any_accepted_multiplier_rate"]
    )
    comparator_id = "projected_oracle" if projected_accept >= raw_accept else "raw_oracle"
    comparator = projected if comparator_id == "projected_oracle" else raw
    comparator_accept = max(raw_accept, projected_accept)
    if comparator_accept < spec.oracle_acceptance_rate_min:
        return {
            "locus": "oracle_control_not_admitted",
            "discriminator_predicate": None,
            "comparator_source": comparator_id,
            "oof_acceptance_rate": oof_accept,
            "raw_oracle_acceptance_rate": raw_accept,
            "projected_oracle_acceptance_rate": projected_accept,
            "predicate_comparisons": [],
        }
    if oof_accept > spec.oof_acceptance_rate_max:
        return {
            "locus": "stageh_rejection_not_reproduced",
            "discriminator_predicate": None,
            "comparator_source": comparator_id,
            "oof_acceptance_rate": oof_accept,
            "raw_oracle_acceptance_rate": raw_accept,
            "projected_oracle_acceptance_rate": projected_accept,
            "predicate_comparisons": [],
        }
    comparisons: List[Dict[str, Any]] = []
    discriminator: Optional[str] = None
    oof_aggregate = oof["predicate_aggregate"]
    oracle_aggregate = comparator["predicate_aggregate"]
    for predicate in PREDICATE_ORDER:
        oof_pass = float(oof_aggregate["predicate_pass_rates"][predicate])
        oracle_pass = float(oracle_aggregate["predicate_pass_rates"][predicate])
        oof_first = float(oof_aggregate["first_failed_rates"][predicate])
        oracle_first = float(oracle_aggregate["first_failed_rates"][predicate])
        pass_discriminator = (
            oracle_pass >= spec.oracle_predicate_pass_rate_min
            and oof_pass <= spec.oof_predicate_pass_rate_max
        )
        first_failure_discriminator = (
            oof_first >= spec.oof_first_failure_rate_min
            and oracle_first <= spec.oracle_first_failure_rate_max
        )
        comparisons.append(
            {
                "predicate": predicate,
                "oof_pass_rate": oof_pass,
                "oracle_pass_rate": oracle_pass,
                "oof_first_failure_rate": oof_first,
                "oracle_first_failure_rate": oracle_first,
                "pass_rate_discriminator": pass_discriminator,
                "first_failure_discriminator": first_failure_discriminator,
            }
        )
        if discriminator is None and (pass_discriminator or first_failure_discriminator):
            discriminator = predicate
    return {
        "locus": (
            "explicit_callback_does_not_identify_rejection"
            if discriminator is None
            else "oof_first_diverges_at_{}".format(discriminator)
        ),
        "discriminator_predicate": discriminator,
        "comparator_source": comparator_id,
        "oof_acceptance_rate": oof_accept,
        "raw_oracle_acceptance_rate": raw_accept,
        "projected_oracle_acceptance_rate": projected_accept,
        "predicate_comparisons": comparisons,
    }


def classify(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not records:
        raise ExplicitPredicateCallbackError("Stage-K audit has no records")
    loci: Dict[str, int] = {}
    predicates: Dict[str, int] = {}
    for record in records:
        comparison = record["predicate_comparison"]
        locus = str(comparison["locus"])
        loci[locus] = loci.get(locus, 0) + 1
        predicate = comparison.get("discriminator_predicate")
        if isinstance(predicate, str):
            predicates[predicate] = predicates.get(predicate, 0) + 1
    total = len(records)
    if loci.get("oracle_control_not_admitted", 0):
        root = "phase314b_r258_stagek_frozen_integrator_oracle_replay_not_admitted"
        next_path = "RESTORE_STAGEE_ORACLE_REPLAY_BEFORE_SURROGATE_REPAIR"
        primary = "oracle_replay"
    elif loci.get("stageh_rejection_not_reproduced", 0):
        root = "phase314b_r258_stagek_stageh_rejection_not_reproduced"
        next_path = "AUDIT_STAGEH_STAGEK_REPLAY_IDENTITY"
        primary = "replay_identity"
    elif len(predicates) == 1 and sum(predicates.values()) == total:
        primary = next(iter(predicates))
        root, next_path = PREDICATE_OUTCOMES[primary]
    elif predicates:
        root = "phase314b_r258_stagek_rejection_predicates_are_heterogeneous"
        next_path = "STRATIFY_OOF_REJECTION_BY_EXPLICIT_INTEGRATOR_PREDICATE"
        primary = "heterogeneous_rejection_predicates"
    else:
        root = "phase314b_r258_stagek_explicit_callback_surface_insufficient"
        next_path = "AUDIT_EXPLICIT_CALLBACK_PREDICATE_ASSEMBLY"
        primary = "callback_observability"
    return {
        "root_cause": root,
        "required_next_path": next_path,
        "primary_failure_locus": primary,
        "record_count": total,
        "record_locus_counts": dict(sorted(loci.items())),
        "discriminator_predicate_counts": dict(sorted(predicates.items())),
        "all_records_share_primary_locus": len(loci) == 1,
        "all_records_share_discriminator_predicate": (
            len(predicates) == 1 and sum(predicates.values()) == total
        ),
    }


def run_calibration(
    *,
    root: Path,
    environment: Mapping[str, Any],
    spec: Optional[ExplicitPredicateCallbackSpec] = None,
    stagef_spec: Optional[stagef.ConstraintAwareSpec] = None,
    direction_spec: Optional[staged258.DirectionSurrogateSpec] = None,
    integrator_spec: Optional[stagee258.ConstrainedIntegratorSpec] = None,
) -> Dict[str, Any]:
    active = ExplicitPredicateCallbackSpec() if spec is None else spec
    active_stagef = stagef.ConstraintAwareSpec() if stagef_spec is None else stagef_spec
    active_direction = (
        staged258.DirectionSurrogateSpec() if direction_spec is None else direction_spec
    )
    active_integrator = (
        stagee258.ConstrainedIntegratorSpec() if integrator_spec is None else integrator_spec
    )
    active.validate()
    active_stagef.validate()
    active_direction.validate()
    active_integrator.validate()
    repository_root = Path(root).resolve()
    base = validate_base_evidence(repository_root)
    stagef.stagec258.stagea258.validate_environment_payload(environment)
    if environment.get("compatibility_sha256") != stagef.EXPECTED_COMPATIBILITY_SHA256:
        raise ExplicitPredicateCallbackError("portable compatibility SHA changed")
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
        raise ExplicitPredicateCallbackError("Stage-K ULP policy replay changed")
    active_stagef = stagef.replace(
        active_stagef,
        translation_float32_z_ulp_factor=EXPECTED_SELECTED_ULP_FACTOR,
    )
    active_stagef.validate()

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
                spec=active,
            ),
            "projected_oracle": audit_direction_source(
                source_id="projected_oracle",
                control=control,
                base_direction=projected_direction,
                context=context,
                integrator_spec=active_integrator,
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
                spec=active,
            )
            comparison = compare_predicate_surfaces(
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
                    "oof_integrator_callback": oof_audit,
                    "predicate_comparison": comparison,
                    "objective_train_rows": int(control.shape[0]),
                    "holdout_used": False,
                }
            )
    classification = classify(records)
    contract = {
        "schema": "phase314b_r258_stagek_explicit_integrator_predicate_callback_contract_v1",
        "phase": PHASE,
        "base_evidence_commit": BASE_EVIDENCE_COMMIT,
        "base_report_sha256": EXPECTED_BASE_REPORT_SHA256,
        "base_single_run_sha256": EXPECTED_BASE_SINGLE_RUN_SHA256,
        "base_validation": base,
        "environment": dict(environment),
        "cold_main_worker_context": cold,
        "control_capture": captured["control_identity"],
        "callback_spec": asdict(active),
        "stagef_spec": asdict(active_stagef),
        "direction_spec": asdict(active_direction),
        "integrator_spec": asdict(active_integrator),
        "fixed_integrator": asdict(stagef.fixed_integrator_definition()),
        "predicate_order": list(PREDICATE_ORDER),
        "base_direction_ids": list(BASE_DIRECTION_IDS),
        "scale_bank": list(SCALE_MULTIPLIERS),
        "data_boundary": {
            "objective_train_only": True,
            "selection_holdout_evaluated": False,
            "selection_holdout_used_for_fit_or_selection": False,
            "frozen_probe_accessed": False,
            "target_used_only_for_objective_train_oracle_controls": True,
        },
        "mechanism_boundary": {
            "stagee_numerical_ast_unchanged_after_telemetry_stripping": True,
            "predicate_callback_default_disabled": True,
            "every_callback_result_compared_to_callback_disabled_result": True,
            "callback_payload_recursively_immutable": True,
            "callback_payload_scalar_only": True,
            "integrator_thresholds_changed": False,
            "integrator_scale_search_changed": False,
            "new_ranker_fitted": False,
            "candidate_matrix_changed": False,
            "external_scale_bank_changed": False,
            "stagef_feature_definition_modified": False,
            "ulp_policy_changed": False,
        },
    }
    contract["contract_sha256"] = sha256_bytes(stable_json_bytes(contract))
    audit = {
        "schema": "phase314b_r258_stagek_explicit_integrator_predicate_callback_audit_v1",
        "control_callback_by_timestep": {
            str(key): value for key, value in sorted(controls.items())
        },
        "oof_records": records,
        "classification": classification,
        "callback_off_on_pair_count": 132,
        "all_callback_results_bit_exact": True,
    }
    actual_pairs = sum(
        int(value["raw_oracle"]["callback_pair_count"])
        + int(value["projected_oracle"]["callback_pair_count"])
        for value in controls.values()
    ) + sum(int(record["oof_integrator_callback"]["callback_pair_count"]) for record in records)
    if actual_pairs != 132:
        raise ExplicitPredicateCallbackError(
            "Stage-K callback pair population changed: {}".format(actual_pairs)
        )
    audit["callback_off_on_pair_count"] = actual_pairs
    audit["audit_sha256"] = sha256_bytes(stable_json_bytes(audit))
    return {
        "schema": "phase314b_r258_stagek_explicit_integrator_predicate_callback_result_v1",
        "verdict": "PASS",
        "scientific_status": "BLOCKED",
        "root_cause": classification["root_cause"],
        "required_next_path": classification["required_next_path"],
        "primary_failure_locus": classification["primary_failure_locus"],
        "selected_configuration": None,
        "train_only_recommendation": None,
        "immutable_inputs": base,
        "contract": contract,
        "explicit_integrator_predicate_callback": audit,
        "explicit_callback_completed": True,
        "all_callback_results_bit_exact": True,
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
