"""Phase3.14b-r2.5.8 Stage L explicit-callback predicate assembly audit.

Stage K proved that the default-disabled Stage-E predicate callback is read-only
and that callback-enabled integrations are byte-exact with callback-disabled
integrations.  It did not identify a discriminator because its aggregate pass
rates use active-row denominators for predicates that are only meaningful after
previous predicates pass, while its first-failure rates use all active row
attempts across every internal scale and every external multiplier.

Stage L does not change Stage-E, Stage-K, any threshold, any direction model, or
the scale bank.  It replays the same objective-train-only 132 callback-off/on
pairs and audits how callback events are assembled.  It reconstructs the exact
sequential-reach population at each predicate, quantifies unreachable-row pass
contamination and global-denominator dilution, and compares oracle and grouped-
OOF sources with predeclared corrected assembly views.  No callback event,
row mask, direction, candidate, model, or tensor is persisted.
"""
from __future__ import annotations

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
from ccda_phase3 import phase314b_r258_stageh_candidate_descriptor_identifiability as stageh
from ccda_phase3 import phase314b_r258_stagek_explicit_integrator_predicate_callback as stagek


PHASE = "Phase3.14b-r2.5.8 Stage L"
PHASE_ID = "phase314b_r258_stagel"
BASE_IMPLEMENTATION_COMMIT = "9ff2379b142e79ee07aedb3c6ee5bf94c089a1d5"
BASE_EVIDENCE_COMMIT = "70f55e62aea557194737528f942bfde697f984b5"
EXPECTED_SUBMODULE_COMMIT = "633a88752445cf5d6776ed374fdbbdb35f93050c"
BASE_REPORT = (
    "reports/phase3_14b_r258_stagek_explicit_integrator_predicate_callback_summary.json"
)
EXPECTED_BASE_REPORT_SHA256 = (
    "2055949b86590e96781da5deb5929cf9207624d7835d8f902d6021630161b55e"
)
EXPECTED_BASE_SINGLE_RUN_SHA256 = (
    "d9a554fcb1f1aa3656dcb7e79c971a168421d9c2522919572bc0aa67db0782c1"
)
EXPECTED_BASE_ROOT_CAUSE = (
    "phase314b_r258_stagek_explicit_callback_surface_insufficient"
)
EXPECTED_BASE_NEXT_PATH = "AUDIT_EXPLICIT_CALLBACK_PREDICATE_ASSEMBLY"
EXPECTED_SELECTED_ULP_FACTOR = 1.0
EXPECTED_CALLBACK_PAIR_COUNT = 132

FROZEN_SOURCE_SHA256: Mapping[str, str] = {
    "ccda_phase3/phase314b_r258_stagee_constrained_integrator.py": (
        "37a9567391a308bfe804b95b4aa488ada40e82b2eb94bd03f2060be962cc9eb1"
    ),
    "ccda_phase3/phase314b_r258_stagek_explicit_integrator_predicate_callback.py": (
        "403b4242c6bf7c018e6220bf4e3920e26a9cd945e6bf3871792d88bd7e1f3139"
    ),
    "tests/test_phase3_14b_r258_stagek_explicit_integrator_predicate_callback.py": (
        "425b2bdc590e0c2db8a313df3d9046b1d3edeb6b1caf518d0c6f61b98eea9f30"
    ),
}

BASE_DIRECTION_IDS: Tuple[str, ...] = tuple(stagek.BASE_DIRECTION_IDS)
SCALE_MULTIPLIERS: Tuple[float, ...] = tuple(stagek.SCALE_MULTIPLIERS)
PREDICATE_ORDER: Tuple[str, ...] = tuple(stagek.PREDICATE_ORDER)


class PredicateAssemblyAuditError(RuntimeError):
    """Raised when Stage-L immutable or assembly invariants fail."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def stable_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    payload = (
        str(array.dtype).encode("ascii")
        + b"|"
        + repr(tuple(array.shape)).encode("ascii")
        + b"|"
        + array.tobytes(order="C")
    )
    return sha256_bytes(payload)


def _git(root: Path, *args: str) -> bytes:
    process = subprocess.run(
        ["git", *args],
        cwd=str(root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if process.returncode != 0:
        raise PredicateAssemblyAuditError(
            "git {} failed: {}".format(
                " ".join(args), process.stderr.decode("utf-8", "replace")
            )
        )
    return process.stdout


def _require_ancestor(root: Path, ancestor: str, descendant: str) -> None:
    process = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=str(root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if process.returncode != 0:
        raise PredicateAssemblyAuditError(
            "required commit is not an ancestor: {}".format(ancestor)
        )


def _extract_single_run_sha(value: Any) -> Optional[str]:
    if isinstance(value, Mapping):
        direct = value.get("single_run_result_sha256")
        if isinstance(direct, str):
            return direct
        execution = value.get("execution")
        if isinstance(execution, Mapping):
            nested = execution.get("single_run_result_sha256")
            if isinstance(nested, str):
                return nested
        for item in value.values():
            nested = _extract_single_run_sha(item)
            if nested is not None:
                return nested
    elif isinstance(value, (list, tuple)):
        for item in value:
            nested = _extract_single_run_sha(item)
            if nested is not None:
                return nested
    return None


def _require_false(mapping: Mapping[str, Any], keys: Iterable[str], prefix: str) -> None:
    for key in keys:
        if mapping.get(key) is not False:
            raise PredicateAssemblyAuditError(
                "{} boundary changed: {}={!r}".format(prefix, key, mapping.get(key))
            )


def validate_base_evidence(root: Path) -> Dict[str, Any]:
    repository_root = Path(root).resolve()
    head = _git(repository_root, "rev-parse", "HEAD").decode().strip()
    _require_ancestor(repository_root, BASE_IMPLEMENTATION_COMMIT, head)
    _require_ancestor(repository_root, BASE_EVIDENCE_COMMIT, head)

    report_path = repository_root / BASE_REPORT
    if not report_path.is_file():
        raise PredicateAssemblyAuditError("Stage-K evidence report is missing")
    report_bytes = report_path.read_bytes()
    if sha256_bytes(report_bytes) != EXPECTED_BASE_REPORT_SHA256:
        raise PredicateAssemblyAuditError("Stage-K report SHA changed")
    report = json.loads(report_bytes.decode("utf-8"))
    if report.get("execution_verdict") != "PASS":
        raise PredicateAssemblyAuditError("Stage-K execution verdict changed")
    if report.get("scientific_status") != "BLOCKED":
        raise PredicateAssemblyAuditError("Stage-K scientific status changed")
    if report.get("root_cause") != EXPECTED_BASE_ROOT_CAUSE:
        raise PredicateAssemblyAuditError("Stage-K root cause changed")
    if report.get("required_next_path") != EXPECTED_BASE_NEXT_PATH:
        raise PredicateAssemblyAuditError("Stage-K next path changed")
    if _extract_single_run_sha(report) != EXPECTED_BASE_SINGLE_RUN_SHA256:
        raise PredicateAssemblyAuditError("Stage-K single-run SHA changed")

    scientific = report.get("scientific_result")
    if not isinstance(scientific, Mapping):
        raise PredicateAssemblyAuditError("Stage-K scientific_result is missing")
    audit = scientific.get("explicit_integrator_predicate_callback")
    if not isinstance(audit, Mapping):
        raise PredicateAssemblyAuditError("Stage-K callback audit is missing")
    if int(audit.get("callback_off_on_pair_count", -1)) != EXPECTED_CALLBACK_PAIR_COUNT:
        raise PredicateAssemblyAuditError("Stage-K callback pair count changed")
    if audit.get("all_callback_results_bit_exact") is not True:
        raise PredicateAssemblyAuditError("Stage-K callback identity changed")
    classification = audit.get("classification")
    if not isinstance(classification, Mapping):
        raise PredicateAssemblyAuditError("Stage-K classification is missing")
    if classification.get("discriminator_predicate_counts") != {}:
        raise PredicateAssemblyAuditError("Stage-K unexpectedly identified a predicate")

    contract = scientific.get("contract")
    if not isinstance(contract, Mapping):
        raise PredicateAssemblyAuditError("Stage-K contract is missing")
    mechanism = contract.get("mechanism_boundary")
    if not isinstance(mechanism, Mapping):
        raise PredicateAssemblyAuditError("Stage-K mechanism boundary is missing")
    required_true = (
        "stagee_numerical_ast_unchanged_after_telemetry_stripping",
        "predicate_callback_default_disabled",
        "every_callback_result_compared_to_callback_disabled_result",
        "callback_payload_recursively_immutable",
        "callback_payload_scalar_only",
    )
    for key in required_true:
        if mechanism.get(key) is not True:
            raise PredicateAssemblyAuditError(
                "Stage-K mechanism proof changed: {}".format(key)
            )
    required_false = (
        "integrator_thresholds_changed",
        "integrator_scale_search_changed",
        "new_ranker_fitted",
        "candidate_matrix_changed",
        "external_scale_bank_changed",
        "stagef_feature_definition_modified",
        "ulp_policy_changed",
    )
    _require_false(mechanism, required_false, "Stage-K mechanism")

    _require_false(
        scientific,
        (
            "selection_holdout_evaluated",
            "selection_holdout_used_for_fit_or_selection",
            "frozen_probe_accessed",
            "formal_training_run",
            "reverse_sampling_run",
            "idm_run",
            "candidate_execution",
            "deformable_ravens_executed",
            "phase4",
            "cps",
            "checkpoint_saved",
            "weights_persisted",
            "surrogate_weights_persisted",
            "prediction_tensor_persisted",
            "candidate_tensor_persisted",
            "predicate_tensor_persisted",
            "npz_saved",
            "cache_saved",
            "image_saved",
            "video_saved",
        ),
        "Stage-K scientific",
    )

    source_sha: Dict[str, str] = {}
    for relative, expected in FROZEN_SOURCE_SHA256.items():
        current = (repository_root / relative).read_bytes()
        committed = _git(repository_root, "show", "{}:{}".format(BASE_EVIDENCE_COMMIT, relative))
        if current != committed:
            raise PredicateAssemblyAuditError(
                "frozen Stage-K path differs from evidence commit: {}".format(relative)
            )
        actual = sha256_bytes(current)
        if actual != expected:
            raise PredicateAssemblyAuditError(
                "frozen Stage-K source SHA changed: {}".format(relative)
            )
        source_sha[relative] = actual

    inherited = stagek.validate_base_evidence(repository_root)
    return {
        "base_implementation_commit": BASE_IMPLEMENTATION_COMMIT,
        "base_evidence_commit": BASE_EVIDENCE_COMMIT,
        "base_report_sha256": EXPECTED_BASE_REPORT_SHA256,
        "base_single_run_sha256": EXPECTED_BASE_SINGLE_RUN_SHA256,
        "base_callback_pair_count": EXPECTED_CALLBACK_PAIR_COUNT,
        "base_source_sha256": source_sha,
        "stagek_inherited_validation": inherited,
        "stagek_report_unmodified": True,
    }


@dataclass(frozen=True)
class PredicateAssemblyAuditSpec:
    timesteps: Tuple[int, ...] = (10, 25, 50)
    scale_multipliers: Tuple[float, ...] = SCALE_MULTIPLIERS
    grouped_cv_folds: int = 6
    oracle_acceptance_rate_min: float = 0.95
    oof_acceptance_rate_max: float = 0.05
    oracle_conditional_pass_rate_min: float = 0.90
    oof_conditional_pass_rate_max: float = 0.10
    oof_rejection_mass_min: float = 0.50
    oracle_rejection_mass_max: float = 0.10
    stratum_support_min: int = 2
    candidate_motion_epsilon: float = 1.0e-12
    binary_tolerance: float = 1.0e-12

    def validate(self) -> None:
        if self.timesteps != (10, 25, 50):
            raise PredicateAssemblyAuditError("Stage-L timesteps changed")
        if self.scale_multipliers != SCALE_MULTIPLIERS:
            raise PredicateAssemblyAuditError("Stage-L external scale bank changed")
        if self.grouped_cv_folds != 6:
            raise PredicateAssemblyAuditError("Stage-L grouped folds changed")
        for value in (
            self.oracle_acceptance_rate_min,
            self.oof_acceptance_rate_max,
            self.oracle_conditional_pass_rate_min,
            self.oof_conditional_pass_rate_max,
            self.oof_rejection_mass_min,
            self.oracle_rejection_mass_max,
        ):
            if not (0.0 <= float(value) <= 1.0):
                raise PredicateAssemblyAuditError("Stage-L rate threshold is invalid")
        if self.stratum_support_min < 1:
            raise PredicateAssemblyAuditError("Stage-L stratum support is invalid")


def _scale_key(value: float) -> str:
    return format(float(value), ".17g")


def _attempt_events(capture: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    events = capture.get("events")
    if not isinstance(events, Sequence):
        raise PredicateAssemblyAuditError("callback capture lacks events")
    attempts = [
        event
        for event in events
        if isinstance(event, Mapping) and event.get("event_type") == "scale_attempt"
    ]
    if not attempts:
        raise PredicateAssemblyAuditError("callback capture lacks scale attempts")
    return attempts


def aggregate_sequential_attempts(events: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not events:
        raise PredicateAssemblyAuditError("cannot aggregate an empty event population")
    reached = {name: 0 for name in PREDICATE_ORDER}
    sequential_pass = {name: 0 for name in PREDICATE_ORDER}
    first_failed = {name: 0 for name in PREDICATE_ORDER}
    legacy_pass = {name: 0 for name in PREDICATE_ORDER}
    legacy_denominator = {name: 0 for name in PREDICATE_ORDER}
    unreachable_pass_contamination = {name: 0 for name in PREDICATE_ORDER}
    active_total = 0
    accepted_total = 0

    for event in events:
        if tuple(event.get("predicate_order", ())) != PREDICATE_ORDER:
            raise PredicateAssemblyAuditError("predicate order changed in callback event")
        active = int(event.get("active_row_count", -1))
        accepted = int(event.get("accepted_count", -1))
        if active < 0 or accepted < 0 or accepted > active:
            raise PredicateAssemblyAuditError("invalid active/accepted callback population")
        failures = event.get("first_failed_counts")
        passes = event.get("predicate_pass_counts")
        if not isinstance(failures, Mapping) or not isinstance(passes, Mapping):
            raise PredicateAssemblyAuditError("callback event lacks predicate counts")
        topology_checked = int(event.get("topology_checked_count", -1))
        if topology_checked < 0:
            raise PredicateAssemblyAuditError("invalid topology checked count")

        active_total += active
        accepted_total += accepted
        unresolved = active
        for predicate in PREDICATE_ORDER:
            failure = int(failures.get(predicate, -1))
            if failure < 0 or failure > unresolved:
                raise PredicateAssemblyAuditError(
                    "first-failure count exceeds sequential reach: {}".format(predicate)
                )
            reached[predicate] += unresolved
            first_failed[predicate] += failure
            survivors = unresolved - failure
            sequential_pass[predicate] += survivors

            legacy_count = int(passes.get(predicate, -1))
            if legacy_count < 0:
                raise PredicateAssemblyAuditError("invalid legacy predicate pass count")
            legacy_pass[predicate] += legacy_count
            legacy_denominator[predicate] += (
                topology_checked if predicate == "topology" else active
            )
            contamination = legacy_count - survivors
            if contamination < 0:
                raise PredicateAssemblyAuditError(
                    "legacy pass count is below sequential survivor count: {}".format(
                        predicate
                    )
                )
            unreachable_pass_contamination[predicate] += contamination
            unresolved = survivors
        if unresolved != accepted:
            raise PredicateAssemblyAuditError(
                "sequential predicate population does not close to accepted rows"
            )

    rejected_total = active_total - accepted_total
    conditional_pass_rates = {
        name: (
            1.0 if reached[name] == 0 else float(sequential_pass[name] / reached[name])
        )
        for name in PREDICATE_ORDER
    }
    conditional_failure_rates = {
        name: (
            0.0 if reached[name] == 0 else float(first_failed[name] / reached[name])
        )
        for name in PREDICATE_ORDER
    }
    rejection_mass_rates = {
        name: (
            0.0 if rejected_total == 0 else float(first_failed[name] / rejected_total)
        )
        for name in PREDICATE_ORDER
    }
    global_active_first_failure_rates = {
        name: (
            0.0 if active_total == 0 else float(first_failed[name] / active_total)
        )
        for name in PREDICATE_ORDER
    }
    legacy_pass_rates = {
        name: (
            1.0
            if legacy_denominator[name] == 0
            else float(legacy_pass[name] / legacy_denominator[name])
        )
        for name in PREDICATE_ORDER
    }
    contamination_rates = {
        name: (
            0.0
            if legacy_pass[name] == 0
            else float(unreachable_pass_contamination[name] / legacy_pass[name])
        )
        for name in PREDICATE_ORDER
    }
    dominant = max(PREDICATE_ORDER, key=lambda name: first_failed[name])
    dominant_count = int(first_failed[dominant])
    if dominant_count == 0:
        dominant_predicate: Optional[str] = None
        dominance = 0.0
    else:
        dominant_predicate = dominant
        dominance = 0.0 if rejected_total == 0 else float(dominant_count / rejected_total)

    return {
        "event_count": len(events),
        "active_row_attempt_count": active_total,
        "accepted_row_attempt_count": accepted_total,
        "rejected_row_attempt_count": rejected_total,
        "sequential_reach_counts": reached,
        "sequential_pass_counts": sequential_pass,
        "first_failed_counts": first_failed,
        "conditional_pass_rates": conditional_pass_rates,
        "conditional_failure_rates": conditional_failure_rates,
        "rejection_mass_rates": rejection_mass_rates,
        "global_active_first_failure_rates": global_active_first_failure_rates,
        "legacy_unconditional_pass_counts": legacy_pass,
        "legacy_unconditional_pass_denominators": legacy_denominator,
        "legacy_unconditional_pass_rates": legacy_pass_rates,
        "unreachable_pass_contamination_counts": unreachable_pass_contamination,
        "unreachable_pass_contamination_rates": contamination_rates,
        "unreachable_pass_contamination_total": int(
            sum(unreachable_pass_contamination.values())
        ),
        "dominant_failure_predicate": dominant_predicate,
        "dominant_failure_mass": dominance,
        "sequential_population_closed": True,
    }


def _merge_event_groups(groups: Sequence[Sequence[Mapping[str, Any]]]) -> List[Mapping[str, Any]]:
    output: List[Mapping[str, Any]] = []
    for group in groups:
        output.extend(group)
    if not output:
        raise PredicateAssemblyAuditError("cannot merge an empty event population")
    return output


def _legacy_aggregate_matches(
    captures: Sequence[Mapping[str, Any]],
    assembly: Mapping[str, Any],
) -> bool:
    legacy = stagek.aggregate_attempt_telemetry(captures)
    for predicate in PREDICATE_ORDER:
        expected_pass = float(legacy["predicate_pass_rates"][predicate])
        actual_pass = float(assembly["legacy_unconditional_pass_rates"][predicate])
        expected_first = float(legacy["first_failed_rates"][predicate])
        actual_first = float(assembly["global_active_first_failure_rates"][predicate])
        if expected_pass != actual_pass or expected_first != actual_first:
            return False
    return (
        int(legacy["active_row_attempt_count"])
        == int(assembly["active_row_attempt_count"])
        and int(legacy["accepted_row_attempt_count"])
        == int(assembly["accepted_row_attempt_count"])
    )


def audit_direction_source_assembly(
    *,
    source_id: str,
    control: np.ndarray,
    base_direction: np.ndarray,
    context: Mapping[str, Any],
    integrator_spec: stagee258.ConstrainedIntegratorSpec,
    callback_spec: stagek.ExplicitPredicateCallbackSpec,
    spec: PredicateAssemblyAuditSpec,
) -> Dict[str, Any]:
    control_raw = np.asarray(control, dtype=np.float32)
    direction_raw = np.asarray(base_direction, dtype=np.float64)
    rows = int(control_raw.shape[0])
    selected_columns: List[np.ndarray] = []
    captures: List[Mapping[str, Any]] = []
    events_by_multiplier: Dict[str, List[Mapping[str, Any]]] = {}
    events_by_internal_scale: Dict[str, List[Mapping[str, Any]]] = {}
    multiplier_records: List[Dict[str, Any]] = []

    for multiplier in spec.scale_multipliers:
        proposed = direction_raw * float(multiplier)
        integration, capture = stagek.callback_integrate_rowwise(
            control=control_raw,
            direction=proposed,
            context=context,
            integrator_spec=integrator_spec,
            callback_spec=callback_spec,
        )
        selected = np.asarray(integration["selected_scale"], dtype=np.float64).reshape(rows)
        candidate = np.asarray(integration["candidate"], dtype=np.float32)
        motion = stagek._row_norm(
            candidate.astype(np.float64) - control_raw.astype(np.float64)
        )
        selected_positive = selected > spec.binary_tolerance
        motion_positive = motion > spec.candidate_motion_epsilon
        if np.any(selected_positive != motion_positive):
            raise PredicateAssemblyAuditError(
                "selected-scale and candidate-motion observables disagree"
            )
        selected_columns.append(selected)
        captures.append(capture)
        attempts = _attempt_events(capture)
        multiplier_key = _scale_key(multiplier)
        events_by_multiplier[multiplier_key] = list(attempts)
        for event in attempts:
            internal_key = _scale_key(float(event["attempted_scale"]))
            events_by_internal_scale.setdefault(internal_key, []).append(event)
        multiplier_assembly = aggregate_sequential_attempts(attempts)
        multiplier_records.append(
            {
                "scale_multiplier": float(multiplier),
                "proposed_direction_sha256": sha256_array(proposed),
                "selected_scale_positive_rate": float(np.mean(selected_positive)),
                "selected_scale_sha256": sha256_array(selected),
                "candidate_motion_positive_rate": float(np.mean(motion_positive)),
                "candidate_sha256": sha256_array(candidate),
                "assembly": multiplier_assembly,
                "callback_capture_sha256": str(capture["events_sha256"]),
                "callback_result_bit_exact": bool(capture["returned_result_bit_exact"]),
            }
        )

    selected_bank = np.stack(selected_columns, axis=1)
    accepted = selected_bank > spec.binary_tolerance
    all_events = _merge_event_groups([_attempt_events(item) for item in captures])
    overall = aggregate_sequential_attempts(all_events)
    if not _legacy_aggregate_matches(captures, overall):
        raise PredicateAssemblyAuditError(
            "Stage-L failed to reproduce the Stage-K legacy aggregate exactly"
        )
    return {
        "source_id": source_id,
        "objective_train_rows": rows,
        "direction_nonzero_rate": float(
            np.mean(stagek._row_norm(direction_raw) > spec.candidate_motion_epsilon)
        ),
        "scale_bank_acceptance": {
            "rows_with_any_accepted_multiplier_rate": float(
                np.mean(np.any(accepted, axis=1))
            ),
            "selected_scale_positive_rate": float(np.mean(accepted)),
            "selected_scale_bank_sha256": sha256_array(selected_bank),
        },
        "assembly_overall": overall,
        "assembly_by_external_multiplier": {
            key: aggregate_sequential_attempts(value)
            for key, value in sorted(events_by_multiplier.items())
        },
        "assembly_by_internal_scale": {
            key: aggregate_sequential_attempts(value)
            for key, value in sorted(events_by_internal_scale.items())
        },
        "multiplier_records": multiplier_records,
        "stagek_legacy_aggregate_reproduced_exactly": True,
        "all_callback_results_bit_exact": True,
        "callback_pair_count": len(spec.scale_multipliers),
        "callback_events_persisted": False,
    }


def _predicate_comparison(
    *,
    oof: Mapping[str, Any],
    oracle: Mapping[str, Any],
    spec: PredicateAssemblyAuditSpec,
) -> Tuple[List[Dict[str, Any]], Optional[str], Optional[str]]:
    comparisons: List[Dict[str, Any]] = []
    corrected: Optional[str] = None
    failure_mode: Optional[str] = None
    for predicate in PREDICATE_ORDER:
        oof_conditional = float(oof["conditional_pass_rates"][predicate])
        oracle_conditional = float(oracle["conditional_pass_rates"][predicate])
        oof_mass = float(oof["rejection_mass_rates"][predicate])
        oracle_mass = float(oracle["rejection_mass_rates"][predicate])
        oof_legacy_pass = float(oof["legacy_unconditional_pass_rates"][predicate])
        oracle_legacy_pass = float(oracle["legacy_unconditional_pass_rates"][predicate])
        oof_legacy_first = float(oof["global_active_first_failure_rates"][predicate])
        oracle_legacy_first = float(oracle["global_active_first_failure_rates"][predicate])
        conditional_discriminator = (
            oracle_conditional >= spec.oracle_conditional_pass_rate_min
            and oof_conditional <= spec.oof_conditional_pass_rate_max
        )
        rejection_mass_discriminator = (
            oof_mass >= spec.oof_rejection_mass_min
            and oracle_mass <= spec.oracle_rejection_mass_max
        )
        legacy_pass_discriminator = (
            oracle_legacy_pass >= spec.oracle_conditional_pass_rate_min
            and oof_legacy_pass <= spec.oof_conditional_pass_rate_max
        )
        legacy_first_discriminator = (
            oof_legacy_first >= spec.oof_rejection_mass_min
            and oracle_legacy_first <= spec.oracle_rejection_mass_max
        )
        comparisons.append(
            {
                "predicate": predicate,
                "oof_conditional_pass_rate": oof_conditional,
                "oracle_conditional_pass_rate": oracle_conditional,
                "oof_rejection_mass_rate": oof_mass,
                "oracle_rejection_mass_rate": oracle_mass,
                "oof_legacy_unconditional_pass_rate": oof_legacy_pass,
                "oracle_legacy_unconditional_pass_rate": oracle_legacy_pass,
                "oof_legacy_global_first_failure_rate": oof_legacy_first,
                "oracle_legacy_global_first_failure_rate": oracle_legacy_first,
                "conditional_discriminator": conditional_discriminator,
                "rejection_mass_discriminator": rejection_mass_discriminator,
                "legacy_pass_discriminator": legacy_pass_discriminator,
                "legacy_first_failure_discriminator": legacy_first_discriminator,
            }
        )
        if corrected is None and (
            conditional_discriminator or rejection_mass_discriminator
        ):
            corrected = predicate
            if conditional_discriminator and not legacy_pass_discriminator:
                failure_mode = "unconditional_pass_denominator_contamination"
            elif rejection_mass_discriminator and not legacy_first_discriminator:
                failure_mode = "global_first_failure_denominator_dilution"
            else:
                failure_mode = "sequential_assembly_recovers_discriminator"
    return comparisons, corrected, failure_mode


def _stratum_discriminators(
    *,
    oof: Mapping[str, Mapping[str, Any]],
    oracle: Mapping[str, Mapping[str, Any]],
    spec: PredicateAssemblyAuditSpec,
) -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {}
    for key in sorted(set(oof) & set(oracle)):
        comparisons, discriminator, mode = _predicate_comparison(
            oof=oof[key], oracle=oracle[key], spec=spec
        )
        if discriminator is not None:
            counts[discriminator] = counts.get(discriminator, 0) + 1
        records.append(
            {
                "stratum": key,
                "discriminator_predicate": discriminator,
                "assembly_failure_mode": mode,
                "predicate_comparisons": comparisons,
            }
        )
    shared = None
    if len(counts) == 1 and sum(counts.values()) >= spec.stratum_support_min:
        shared = next(iter(counts))
    return {
        "stratum_count": len(records),
        "records": records,
        "discriminator_counts": dict(sorted(counts.items())),
        "shared_discriminator_predicate": shared,
    }


def compare_assembly_surfaces(
    *,
    oof: Mapping[str, Any],
    raw: Mapping[str, Any],
    projected: Mapping[str, Any],
    spec: PredicateAssemblyAuditSpec,
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
            "corrected_discriminator_predicate": None,
            "assembly_failure_mode": None,
            "comparator_source": comparator_id,
            "oof_acceptance_rate": oof_accept,
            "raw_oracle_acceptance_rate": raw_accept,
            "projected_oracle_acceptance_rate": projected_accept,
            "predicate_comparisons": [],
        }
    if oof_accept > spec.oof_acceptance_rate_max:
        return {
            "locus": "stageh_rejection_not_reproduced",
            "corrected_discriminator_predicate": None,
            "assembly_failure_mode": None,
            "comparator_source": comparator_id,
            "oof_acceptance_rate": oof_accept,
            "raw_oracle_acceptance_rate": raw_accept,
            "projected_oracle_acceptance_rate": projected_accept,
            "predicate_comparisons": [],
        }

    comparisons, discriminator, mode = _predicate_comparison(
        oof=oof["assembly_overall"],
        oracle=comparator["assembly_overall"],
        spec=spec,
    )
    external = _stratum_discriminators(
        oof=oof["assembly_by_external_multiplier"],
        oracle=comparator["assembly_by_external_multiplier"],
        spec=spec,
    )
    internal = _stratum_discriminators(
        oof=oof["assembly_by_internal_scale"],
        oracle=comparator["assembly_by_internal_scale"],
        spec=spec,
    )
    if discriminator is not None:
        locus = "corrected_sequential_assembly_identifies_{}".format(discriminator)
    elif external["shared_discriminator_predicate"] is not None:
        locus = "external_multiplier_stratification_identifies_{}".format(
            external["shared_discriminator_predicate"]
        )
    elif internal["shared_discriminator_predicate"] is not None:
        locus = "internal_scale_stratification_identifies_{}".format(
            internal["shared_discriminator_predicate"]
        )
    elif external["discriminator_counts"] or internal["discriminator_counts"]:
        locus = "predicate_assembly_is_scale_stratified_and_heterogeneous"
    else:
        contamination = int(
            oof["assembly_overall"]["unreachable_pass_contamination_total"]
        )
        locus = (
            "scalar_callback_cannot_link_rows_across_scale_search"
            if contamination > 0
            else "callback_predicate_masks_do_not_discriminate"
        )
    return {
        "locus": locus,
        "corrected_discriminator_predicate": discriminator,
        "assembly_failure_mode": mode,
        "comparator_source": comparator_id,
        "oof_acceptance_rate": oof_accept,
        "raw_oracle_acceptance_rate": raw_accept,
        "projected_oracle_acceptance_rate": projected_accept,
        "predicate_comparisons": comparisons,
        "external_multiplier_strata": external,
        "internal_scale_strata": internal,
    }


def classify(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not records:
        raise PredicateAssemblyAuditError("Stage-L audit has no records")
    loci: Dict[str, int] = {}
    predicates: Dict[str, int] = {}
    modes: Dict[str, int] = {}
    external_predicates: Dict[str, int] = {}
    internal_predicates: Dict[str, int] = {}
    for record in records:
        comparison = record["assembly_comparison"]
        locus = str(comparison["locus"])
        loci[locus] = loci.get(locus, 0) + 1
        predicate = comparison.get("corrected_discriminator_predicate")
        if isinstance(predicate, str):
            predicates[predicate] = predicates.get(predicate, 0) + 1
        mode = comparison.get("assembly_failure_mode")
        if isinstance(mode, str):
            modes[mode] = modes.get(mode, 0) + 1
        for field, output in (
            ("external_multiplier_strata", external_predicates),
            ("internal_scale_strata", internal_predicates),
        ):
            value = comparison.get(field)
            if isinstance(value, Mapping):
                shared = value.get("shared_discriminator_predicate")
                if isinstance(shared, str):
                    output[shared] = output.get(shared, 0) + 1

    total = len(records)
    if loci.get("oracle_control_not_admitted", 0):
        root = "phase314b_r258_stagel_frozen_integrator_oracle_replay_not_admitted"
        next_path = "RESTORE_STAGEE_ORACLE_REPLAY_BEFORE_ASSEMBLY_AUDIT"
        primary = "oracle_replay"
    elif loci.get("stageh_rejection_not_reproduced", 0):
        root = "phase314b_r258_stagel_stageh_rejection_not_reproduced"
        next_path = "AUDIT_STAGEH_STAGEL_REPLAY_IDENTITY"
        primary = "replay_identity"
    elif len(predicates) == 1 and sum(predicates.values()) == total:
        predicate = next(iter(predicates))
        root = "phase314b_r258_stagel_stagek_assembly_masked_{}".format(predicate)
        next_path = (
            "CORRECT_STAGEK_PREDICATE_AGGREGATION_WITH_SEQUENTIAL_REACH_"
            "THEN_REPLAY_ON_OBJECTIVE_TRAIN_ONLY"
        )
        primary = "predicate_assembly::{}".format(predicate)
    elif predicates:
        root = "phase314b_r258_stagel_corrected_predicates_are_heterogeneous"
        next_path = "STRATIFY_OOF_REJECTION_BY_SEQUENTIAL_PREDICATE_AND_SCALE"
        primary = "heterogeneous_corrected_predicates"
    elif external_predicates or internal_predicates:
        root = "phase314b_r258_stagel_rejection_signal_is_scale_stratified"
        next_path = "STRATIFY_OOF_REJECTION_BY_CALLBACK_SCALE_AND_PREDICATE"
        primary = "scale_stratified_predicate_assembly"
    elif loci.get("scalar_callback_cannot_link_rows_across_scale_search", 0):
        root = "phase314b_r258_stagel_scalar_callback_lacks_cross_scale_row_linkage"
        next_path = "ADD_READ_ONLY_ROW_COHORT_HASHES_TO_EXPLICIT_CALLBACK"
        primary = "cross_scale_row_linkage"
    else:
        root = "phase314b_r258_stagel_callback_predicate_masks_not_discriminative"
        next_path = "AUDIT_CALLBACK_PREDICATE_MASKS_AGAINST_FAST_FEASIBLE_CONJUNCTION"
        primary = "predicate_mask_semantics"
    return {
        "root_cause": root,
        "required_next_path": next_path,
        "primary_failure_locus": primary,
        "record_count": total,
        "record_locus_counts": dict(sorted(loci.items())),
        "corrected_discriminator_predicate_counts": dict(sorted(predicates.items())),
        "assembly_failure_mode_counts": dict(sorted(modes.items())),
        "external_stratum_shared_predicate_counts": dict(
            sorted(external_predicates.items())
        ),
        "internal_stratum_shared_predicate_counts": dict(
            sorted(internal_predicates.items())
        ),
        "all_records_share_corrected_discriminator": (
            len(predicates) == 1 and sum(predicates.values()) == total
        ),
    }


def run_calibration(
    *,
    root: Path,
    environment: Mapping[str, Any],
    spec: Optional[PredicateAssemblyAuditSpec] = None,
    callback_spec: Optional[stagek.ExplicitPredicateCallbackSpec] = None,
    stagef_spec: Optional[stagef.ConstraintAwareSpec] = None,
    direction_spec: Optional[staged258.DirectionSurrogateSpec] = None,
    integrator_spec: Optional[stagee258.ConstrainedIntegratorSpec] = None,
) -> Dict[str, Any]:
    active = PredicateAssemblyAuditSpec() if spec is None else spec
    active_callback = (
        stagek.ExplicitPredicateCallbackSpec()
        if callback_spec is None
        else callback_spec
    )
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
    active_callback.validate()
    active_stagef.validate()
    active_direction.validate()
    active_integrator.validate()
    if tuple(active_callback.timesteps) != active.timesteps:
        raise PredicateAssemblyAuditError("Stage-K/Stage-L timesteps disagree")
    if tuple(active_callback.scale_multipliers) != active.scale_multipliers:
        raise PredicateAssemblyAuditError("Stage-K/Stage-L scale banks disagree")

    repository_root = Path(root).resolve()
    base = validate_base_evidence(repository_root)
    stagef.stagec258.stagea258.validate_environment_payload(environment)
    if environment.get("compatibility_sha256") != stagef.EXPECTED_COMPATIBILITY_SHA256:
        raise PredicateAssemblyAuditError("portable compatibility SHA changed")
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
            for timestep in active.timesteps
        },
        context=context,
        spec=active_stagef,
    )
    if float(ulp_policy["selected_factor"]) != EXPECTED_SELECTED_ULP_FACTOR:
        raise PredicateAssemblyAuditError("Stage-L ULP policy replay changed")
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
            "raw_oracle": audit_direction_source_assembly(
                source_id="raw_oracle",
                control=control,
                base_direction=raw_direction,
                context=context,
                integrator_spec=active_integrator,
                callback_spec=active_callback,
                spec=active,
            ),
            "projected_oracle": audit_direction_source_assembly(
                source_id="projected_oracle",
                control=control,
                base_direction=projected_direction,
                context=context,
                integrator_spec=active_integrator,
                callback_spec=active_callback,
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
            oof_audit = audit_direction_source_assembly(
                source_id="oof_surrogate",
                control=control,
                base_direction=oof["prediction"],
                context=context,
                integrator_spec=active_integrator,
                callback_spec=active_callback,
                spec=active,
            )
            comparison = compare_assembly_surfaces(
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
                    "oof_predicate_assembly": oof_audit,
                    "assembly_comparison": comparison,
                    "objective_train_rows": int(control.shape[0]),
                    "holdout_used": False,
                }
            )

    classification = classify(records)
    actual_pairs = sum(
        int(value["raw_oracle"]["callback_pair_count"])
        + int(value["projected_oracle"]["callback_pair_count"])
        for value in controls.values()
    ) + sum(
        int(record["oof_predicate_assembly"]["callback_pair_count"])
        for record in records
    )
    if actual_pairs != EXPECTED_CALLBACK_PAIR_COUNT:
        raise PredicateAssemblyAuditError(
            "Stage-L callback pair population changed: {}".format(actual_pairs)
        )

    contract = {
        "schema": "phase314b_r258_stagel_predicate_assembly_contract_v1",
        "phase": PHASE,
        "base_implementation_commit": BASE_IMPLEMENTATION_COMMIT,
        "base_evidence_commit": BASE_EVIDENCE_COMMIT,
        "base_report_sha256": EXPECTED_BASE_REPORT_SHA256,
        "base_single_run_sha256": EXPECTED_BASE_SINGLE_RUN_SHA256,
        "base_validation": base,
        "environment": dict(environment),
        "cold_main_worker_context": cold,
        "control_capture": captured["control_identity"],
        "assembly_spec": asdict(active),
        "callback_spec": asdict(active_callback),
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
            "stagee_source_modified": False,
            "stagek_source_modified": False,
            "callback_schema_modified": False,
            "callback_payload_modified": False,
            "integrator_thresholds_changed": False,
            "integrator_scale_search_changed": False,
            "new_ranker_fitted": False,
            "candidate_matrix_changed": False,
            "external_scale_bank_changed": False,
            "stagef_feature_definition_modified": False,
            "ulp_policy_changed": False,
            "assembly_only_replay": True,
            "stagek_legacy_aggregate_reproduced_exactly": True,
        },
    }
    contract["contract_sha256"] = sha256_bytes(stable_json_bytes(contract))
    audit = {
        "schema": "phase314b_r258_stagel_predicate_assembly_audit_v1",
        "control_assembly_by_timestep": {
            str(key): value for key, value in sorted(controls.items())
        },
        "oof_records": records,
        "classification": classification,
        "callback_off_on_pair_count": actual_pairs,
        "all_callback_results_bit_exact": True,
        "callback_events_persisted": False,
        "stagek_legacy_aggregate_reproduced_exactly": True,
    }
    audit["audit_sha256"] = sha256_bytes(stable_json_bytes(audit))
    return {
        "schema": "phase314b_r258_stagel_predicate_assembly_result_v1",
        "verdict": "PASS",
        "scientific_status": "BLOCKED",
        "root_cause": classification["root_cause"],
        "required_next_path": classification["required_next_path"],
        "primary_failure_locus": classification["primary_failure_locus"],
        "selected_configuration": None,
        "train_only_recommendation": None,
        "immutable_inputs": base,
        "contract": contract,
        "predicate_assembly_audit": audit,
        "predicate_assembly_audit_completed": True,
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
        "callback_event_persisted": False,
        "npz_saved": False,
        "cache_saved": False,
        "image_saved": False,
        "video_saved": False,
    }
