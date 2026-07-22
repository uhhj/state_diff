"""Phase3.14b-r2.5.8 Stage M external-multiplier predicate stratification.

Stage-L Resume2 corrected a top-level classifier defect and established that all
27 grouped-OOF records contain external-multiplier-stratified heterogeneous
predicate evidence.  The immutable Stage-L Resume1 report already persists the
scalar assembly summaries needed to resolve that evidence by multiplier; it is
therefore unnecessary and prohibited to rerun Stage-L science or the 132
callback-off/on pairs.

This module performs a report-only, stratum-local audit.  For every
backbone × timestep × external multiplier cell it:

* binds the frozen Stage-L comparator source;
* verifies the persisted Stage-L ungated predicate comparison;
* requires local oracle admission and local OOF rejection before accepting a
  predicate discriminator;
* aggregates predicate support by multiplier, timestep, and backbone; and
* classifies whether upper-segment rejection is a single-multiplier boundary,
  a monotone magnitude boundary, or a model/timestep-dependent effect.

No target, direction, candidate, callback event, model, tensor, holdout, frozen
probe, CUDA context, DeformableRavens execution, or new training is accessed.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


PHASE = "Phase3.14b-r2.5.8 Stage M"
SCHEMA = "phase314b_r258_stagem_external_multiplier_predicate_stratification_v1"

BASE_EVIDENCE_COMMIT = "7c6f096e90ab7ad1b53dbca40a2e58731465ace4"
BASE_IMPLEMENTATION_COMMIT = "9f2b5b793640dd115485bcad6a144c57fdcb6e4f"
STAGEL_RESUME1_EVIDENCE_COMMIT = "c8c9bd0139ab7249689855c55be1b491d759a3f4"
STAGEL_RESUME1_IMPLEMENTATION_COMMIT = "9fd9751188d011c8a4fba0457ed03fd067e41f7f"
STAGEL_BLOCKED_PROVENANCE_COMMIT = "77953e4bdf8779fb94ba3f59de886133b9477321"
STAGEL_IMPLEMENTATION_COMMIT = "cd7fe38b1013b13ca84086bedd64dfa5eaacff47"
EXPECTED_REMOTE = "6758ea7ad800667a436b0243d3b1f6c63256d854"
EXPECTED_SUBMODULE = "633a88752445cf5d6776ed374fdbbdb35f93050c"

STAGEL_RESUME1_REPORT = (
    "reports/phase3_14b_r258_stagel_resume1_predicate_assembly_audit_summary.json"
)
EXPECTED_STAGEL_RESUME1_REPORT_SHA256 = (
    "120cf31f649342589e74e3a5b8ce97bb50f4aac33f865fae54ebebef02f79eed"
)
EXPECTED_STAGEL_SINGLE_RUN_SHA256 = (
    "193a6514fd0e3fac76e3cb913eaa10be90b86d400183fe348190056c30080beb"
)
STAGEL_RESUME2_REPORT = (
    "reports/phase3_14b_r258_stagel_resume2_stratified_classifier_repair_summary.json"
)
EXPECTED_STAGEL_RESUME2_REPORT_SHA256 = (
    "2f84b2b12378e0a87aa938ba566ac985581075542b5c93ca7d5043d45b5cd222"
)

SUCCESS_REPORT = (
    "reports/phase3_14b_r258_stagem_external_multiplier_predicate_stratification_summary.json"
)
BLOCKED_REPORT = (
    "reports/phase3_14b_r258_stagem_external_multiplier_predicate_stratification_blocked_summary.json"
)

IMPLEMENTATION_SUBJECT = (
    "Phase3.14b-r2.5.8 Stage M: stratify external multiplier rejection predicates"
)
EVIDENCE_SUBJECT = (
    "Record Phase3.14b-r2.5.8 Stage M external multiplier stratification evidence"
)
BLOCKED_EVIDENCE_SUBJECT = (
    "Record Phase3.14b-r2.5.8 Stage M blocked evidence"
)
IMPLEMENTATION_PATHS: Tuple[Tuple[str, str], ...] = (
    (
        "A",
        "ccda_phase3/phase314b_r258_stagem_external_multiplier_predicate_stratification.py",
    ),
    ("A", "scripts/phase3_14b_r258_stagem_finalize.py"),
    (
        "A",
        "tests/test_phase3_14b_r258_stagem_external_multiplier_predicate_stratification.py",
    ),
)

EXPECTED_RECORD_COUNT = 27
EXPECTED_CALLBACK_PAIRS = 132
EXPECTED_TIMESTEPS: Tuple[int, ...] = (10, 25, 50)
EXPECTED_EXTERNAL_MULTIPLIERS: Tuple[float, ...] = (0.25, 0.5, 1.0, 2.0)
EXPECTED_BASE_DIRECTION_COUNT = 9
PREDICATE_ORDER: Tuple[str, ...] = (
    "finite_state",
    "upper_segment_geometry",
    "lower_segment_geometry",
    "coordinate_recenter",
    "coordinate_geometry",
    "reconstruction_bounds",
    "segment_geometry",
    "direction_retention",
    "displacement",
    "topology",
)

FALSE_BOUNDARIES: Tuple[str, ...] = (
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
    "callback_event_persisted",
    "npz_saved",
    "cache_saved",
    "image_saved",
    "video_saved",
)


class StageMError(RuntimeError):
    """Fail-closed Stage-M validation error."""


@dataclass(frozen=True)
class StageMSpec:
    oracle_local_acceptance_min: float = 0.95
    oof_local_acceptance_max: float = 0.05
    oracle_conditional_pass_min: float = 0.90
    oof_conditional_pass_max: float = 0.10
    oof_rejection_mass_min: float = 0.50
    oracle_rejection_mass_max: float = 0.10
    stable_record_support_min: int = 25
    sparse_other_multiplier_support_max: int = 2
    timestep_support_min: int = 8
    monotonic_tolerance: float = 1.0e-12

    def validate(self) -> None:
        rates = (
            self.oracle_local_acceptance_min,
            self.oof_local_acceptance_max,
            self.oracle_conditional_pass_min,
            self.oof_conditional_pass_max,
            self.oof_rejection_mass_min,
            self.oracle_rejection_mass_max,
        )
        if any(not (0.0 <= float(value) <= 1.0) for value in rates):
            raise StageMError("Stage-M rate threshold is outside [0, 1]")
        if self.stable_record_support_min != 25:
            raise StageMError("Stage-M stable support changed")
        if self.sparse_other_multiplier_support_max != 2:
            raise StageMError("Stage-M sparse-support threshold changed")
        if self.timestep_support_min != 8:
            raise StageMError("Stage-M timestep support changed")
        if self.monotonic_tolerance < 0.0:
            raise StageMError("Stage-M monotonic tolerance is negative")


def stable_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise StageMError(f"JSON root is not a mapping: {path}")
    return value


def write_once(path: Path, payload: bytes) -> None:
    if path.exists():
        raise StageMError(f"refusing to overwrite report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise StageMError(f"stale temporary report exists: {temporary}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StageMError(f"{label} is not a mapping")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise StageMError(f"{label} is not a sequence")
    return value


def _finite_rate(value: Any, label: str) -> float:
    output = float(value)
    if not math.isfinite(output) or not (0.0 <= output <= 1.0):
        raise StageMError(f"{label} is not a finite rate: {value!r}")
    return output


def _nonnegative_int(value: Any, label: str) -> int:
    output = int(value)
    if output < 0:
        raise StageMError(f"{label} is negative")
    return output


def _scale_key(value: float) -> str:
    return format(float(value), ".17g")


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=str(root),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise StageMError(
            "git command failed: git {}\n{}".format(" ".join(args), completed.stderr)
        )
    return completed.stdout.strip()


def commit_parent(root: Path, commit: str) -> str:
    return _git(root, "rev-parse", f"{commit}^")


def commit_subject(root: Path, commit: str) -> str:
    return _git(root, "show", "-s", "--format=%s", commit)


def commit_name_status(root: Path, commit: str) -> Tuple[Tuple[str, str], ...]:
    output = _git(
        root,
        "diff-tree",
        "--no-commit-id",
        "--name-status",
        "-r",
        commit,
    )
    rows: List[Tuple[str, str]] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            raise StageMError(f"unexpected name-status row: {line}")
        rows.append((parts[0], parts[1]))
    return tuple(sorted(rows))


def require_clean(root: Path, label: str) -> None:
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise StageMError(f"{label} is not clean:\n{status}")


def require_false(mapping: Mapping[str, Any], keys: Iterable[str], label: str) -> None:
    for key in keys:
        if mapping.get(key) is not False:
            raise StageMError(f"{label}.{key} is not false")


def validate_repository(root: Path) -> Dict[str, Any]:
    repo = Path(root).resolve()
    if _git(repo, "branch", "--show-current") != "Experiment1":
        raise StageMError("Stage M requires Experiment1")
    head = _git(repo, "rev-parse", "HEAD")
    if commit_parent(repo, head) != BASE_EVIDENCE_COMMIT:
        raise StageMError("Stage-M implementation parent changed")
    if commit_subject(repo, head) != IMPLEMENTATION_SUBJECT:
        raise StageMError("Stage-M implementation subject changed")
    if commit_name_status(repo, head) != tuple(sorted(IMPLEMENTATION_PATHS)):
        raise StageMError("Stage-M implementation path population changed")
    if _git(repo, "rev-parse", "refs/remotes/origin/Experiment1") != EXPECTED_REMOTE:
        raise StageMError("origin/Experiment1 changed")
    if _git(repo, "rev-parse", "HEAD:external/deformable-ravens") != EXPECTED_SUBMODULE:
        raise StageMError("submodule gitlink changed")
    submodule = repo / "external/deformable-ravens"
    if _git(submodule, "rev-parse", "HEAD") != EXPECTED_SUBMODULE:
        raise StageMError("submodule worktree commit changed")
    require_clean(repo, "main worktree")
    require_clean(submodule, "submodule worktree")

    expected_subjects = {
        BASE_EVIDENCE_COMMIT: (
            "Record Phase3.14b-r2.5.8 Stage L Resume2 classification evidence"
        ),
        BASE_IMPLEMENTATION_COMMIT: (
            "Phase3.14b-r2.5.8 Stage L Resume2: repair stratified classification"
        ),
        STAGEL_RESUME1_EVIDENCE_COMMIT: (
            "Record Phase3.14b-r2.5.8 Stage L Resume1 predicate assembly evidence"
        ),
        STAGEL_RESUME1_IMPLEMENTATION_COMMIT: (
            "Phase3.14b-r2.5.8 Stage L Resume1: recover deterministic execution"
        ),
        STAGEL_BLOCKED_PROVENANCE_COMMIT: (
            "Record Phase3.14b-r2.5.8 Stage L deterministic-environment blocked provenance"
        ),
        STAGEL_IMPLEMENTATION_COMMIT: (
            "Phase3.14b-r2.5.8 Stage L: audit explicit callback predicate assembly"
        ),
    }
    for commit, expected in expected_subjects.items():
        if commit_subject(repo, commit) != expected:
            raise StageMError(f"commit subject changed: {commit}")
    if commit_parent(repo, BASE_EVIDENCE_COMMIT) != BASE_IMPLEMENTATION_COMMIT:
        raise StageMError("Stage-L Resume2 evidence parent changed")
    if commit_parent(repo, BASE_IMPLEMENTATION_COMMIT) != STAGEL_RESUME1_EVIDENCE_COMMIT:
        raise StageMError("Stage-L Resume2 implementation parent changed")

    expected_reports = (
        (
            STAGEL_RESUME1_REPORT,
            EXPECTED_STAGEL_RESUME1_REPORT_SHA256,
            STAGEL_RESUME1_EVIDENCE_COMMIT,
        ),
        (
            STAGEL_RESUME2_REPORT,
            EXPECTED_STAGEL_RESUME2_REPORT_SHA256,
            BASE_EVIDENCE_COMMIT,
        ),
    )
    report_bindings: Dict[str, str] = {}
    for relative, expected_sha, commit in expected_reports:
        path = repo / relative
        if not path.is_file():
            raise StageMError(f"required report missing: {relative}")
        observed_sha = sha256_path(path)
        if observed_sha != expected_sha:
            raise StageMError(f"report SHA changed: {relative}")
        blob = subprocess.run(
            ["git", "show", f"{commit}:{relative}"],
            cwd=str(repo),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if blob.returncode != 0 or blob.stdout != path.read_bytes():
            raise StageMError(f"report differs from committed blob: {relative}")
        report_bindings[relative] = observed_sha

    for relative in (SUCCESS_REPORT, BLOCKED_REPORT):
        if (repo / relative).exists():
            raise StageMError(f"Stage-M output already exists: {relative}")
    return {
        "head": head,
        "parent": BASE_EVIDENCE_COMMIT,
        "origin_experiment1": EXPECTED_REMOTE,
        "submodule_commit": EXPECTED_SUBMODULE,
        "report_sha256": report_bindings,
    }


def extract_stage_l_scientific_result(
    resume1_report: Mapping[str, Any],
) -> Mapping[str, Any]:
    if resume1_report.get("execution_verdict") != "PASS":
        raise StageMError("Stage-L Resume1 execution is not PASS")
    if resume1_report.get("scientific_status") != "BLOCKED":
        raise StageMError("Stage-L Resume1 scientific status changed")
    recovery = _mapping(resume1_report.get("recovery_contract"), "recovery_contract")
    if (
        recovery.get("underlying_stage_l_single_run_result_sha256")
        != EXPECTED_STAGEL_SINGLE_RUN_SHA256
    ):
        raise StageMError("Stage-L single-run SHA changed")
    if int(recovery.get("underlying_callback_pair_count", -1)) != EXPECTED_CALLBACK_PAIRS:
        raise StageMError("Stage-L callback pair count changed")
    if recovery.get("stage_l_science_modified") is not False:
        raise StageMError("Resume1 says Stage-L science was modified")
    if recovery.get("existing_test_gates_rerun") is not False:
        raise StageMError("Resume1 says historical tests were rerun")
    stage_l_result = _mapping(resume1_report.get("stage_l_result"), "stage_l_result")
    if stage_l_result.get("execution_verdict") != "PASS":
        raise StageMError("underlying Stage-L execution is not PASS")
    scientific = _mapping(stage_l_result.get("scientific_result"), "scientific_result")
    require_false(scientific, FALSE_BOUNDARIES, "scientific_result")
    if scientific.get("selected_configuration") is not None:
        raise StageMError("Stage-L selected_configuration is not null")
    if scientific.get("train_only_recommendation") is not None:
        raise StageMError("Stage-L train_only_recommendation is not null")
    audit = _mapping(scientific.get("predicate_assembly_audit"), "predicate_assembly_audit")
    if int(audit.get("callback_off_on_pair_count", -1)) != EXPECTED_CALLBACK_PAIRS:
        raise StageMError("Stage-L callback pair count changed")
    if audit.get("all_callback_results_bit_exact") is not True:
        raise StageMError("Stage-L callback identity changed")
    if audit.get("callback_events_persisted") is not False:
        raise StageMError("Stage-L callback events were persisted")
    return scientific


def validate_resume2_report(resume2_report: Mapping[str, Any]) -> Mapping[str, Any]:
    if resume2_report.get("execution_verdict") != "PASS":
        raise StageMError("Stage-L Resume2 execution is not PASS")
    if resume2_report.get("scientific_status") != "BLOCKED":
        raise StageMError("Stage-L Resume2 scientific status changed")
    if (
        resume2_report.get("root_cause")
        != "phase314b_r258_stagel_rejection_signal_is_scale_stratified_and_heterogeneous"
    ):
        raise StageMError("Stage-L Resume2 root cause changed")
    if (
        resume2_report.get("required_next_path")
        != "STRATIFY_OOF_REJECTION_BY_CALLBACK_SCALE_AND_PREDICATE"
    ):
        raise StageMError("Stage-L Resume2 next path changed")
    corrected = _mapping(
        resume2_report.get("corrected_classification"), "corrected_classification"
    )
    if int(corrected.get("record_count", -1)) != EXPECTED_RECORD_COUNT:
        raise StageMError("Stage-L Resume2 record count changed")
    if int(corrected.get("external_heterogeneous_record_count", -1)) != EXPECTED_RECORD_COUNT:
        raise StageMError("Stage-L Resume2 external record count changed")
    if int(corrected.get("internal_heterogeneous_record_count", -1)) != 0:
        raise StageMError("Stage-L Resume2 internal heterogeneity changed")
    if corrected.get("external_stratum_discriminator_counts") != {
        "upper_segment_geometry": EXPECTED_RECORD_COUNT
    }:
        raise StageMError("Stage-L Resume2 external discriminator summary changed")
    contract = _mapping(resume2_report.get("correction_contract"), "correction_contract")
    for key in (
        "stage_l_science_rerun",
        "callback_pairs_rerun",
        "existing_tests_rerun",
        "base_report_modified",
    ):
        if contract.get(key) is not False:
            raise StageMError(f"Stage-L Resume2 {key} is not false")
    return corrected


def _multiplier_record_map(audit: Mapping[str, Any], label: str) -> Dict[str, Mapping[str, Any]]:
    records = _sequence(audit.get("multiplier_records"), f"{label}.multiplier_records")
    output: Dict[str, Mapping[str, Any]] = {}
    for index, value in enumerate(records):
        record = _mapping(value, f"{label}.multiplier_records[{index}]")
        multiplier = float(record.get("scale_multiplier"))
        key = _scale_key(multiplier)
        if key in output:
            raise StageMError(f"duplicate multiplier record: {label}.{key}")
        output[key] = record
    expected = {_scale_key(value) for value in EXPECTED_EXTERNAL_MULTIPLIERS}
    if set(output) != expected:
        raise StageMError(f"external multiplier population changed: {label}")
    return output


def _assembly_map(audit: Mapping[str, Any], label: str) -> Mapping[str, Any]:
    value = _mapping(
        audit.get("assembly_by_external_multiplier"),
        f"{label}.assembly_by_external_multiplier",
    )
    expected = {_scale_key(item) for item in EXPECTED_EXTERNAL_MULTIPLIERS}
    if set(value) != expected:
        raise StageMError(f"external assembly population changed: {label}")
    return value


def _stored_stratum_map(comparison: Mapping[str, Any], label: str) -> Dict[str, Mapping[str, Any]]:
    external = _mapping(
        comparison.get("external_multiplier_strata"),
        f"{label}.external_multiplier_strata",
    )
    records = _sequence(external.get("records"), f"{label}.external records")
    output: Dict[str, Mapping[str, Any]] = {}
    for index, value in enumerate(records):
        record = _mapping(value, f"{label}.external records[{index}]")
        key = str(record.get("stratum"))
        if key in output:
            raise StageMError(f"duplicate stored stratum: {label}.{key}")
        output[key] = record
    expected = {_scale_key(item) for item in EXPECTED_EXTERNAL_MULTIPLIERS}
    if set(output) != expected:
        raise StageMError(f"stored external strata changed: {label}")
    return output


def _predicate_discriminator(
    *,
    oof: Mapping[str, Any],
    oracle: Mapping[str, Any],
    spec: StageMSpec,
) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    comparisons: List[Dict[str, Any]] = []
    discriminator: Optional[str] = None
    for predicate in PREDICATE_ORDER:
        oof_conditional = _finite_rate(
            _mapping(oof.get("conditional_pass_rates"), "oof conditional").get(predicate),
            f"oof conditional.{predicate}",
        )
        oracle_conditional = _finite_rate(
            _mapping(oracle.get("conditional_pass_rates"), "oracle conditional").get(predicate),
            f"oracle conditional.{predicate}",
        )
        oof_mass = _finite_rate(
            _mapping(oof.get("rejection_mass_rates"), "oof rejection mass").get(predicate),
            f"oof rejection mass.{predicate}",
        )
        oracle_mass = _finite_rate(
            _mapping(oracle.get("rejection_mass_rates"), "oracle rejection mass").get(predicate),
            f"oracle rejection mass.{predicate}",
        )
        conditional = (
            oracle_conditional >= spec.oracle_conditional_pass_min
            and oof_conditional <= spec.oof_conditional_pass_max
        )
        rejection_mass = (
            oof_mass >= spec.oof_rejection_mass_min
            and oracle_mass <= spec.oracle_rejection_mass_max
        )
        comparisons.append(
            {
                "predicate": predicate,
                "oof_conditional_pass_rate": oof_conditional,
                "oracle_conditional_pass_rate": oracle_conditional,
                "oof_rejection_mass_rate": oof_mass,
                "oracle_rejection_mass_rate": oracle_mass,
                "conditional_discriminator": conditional,
                "rejection_mass_discriminator": rejection_mass,
            }
        )
        if discriminator is None and (conditional or rejection_mass):
            discriminator = predicate
    return discriminator, comparisons


def _assembly_metrics(assembly: Mapping[str, Any], predicate: str) -> Dict[str, Any]:
    reached = _mapping(assembly.get("sequential_reach_counts"), "sequential_reach_counts")
    failed = _mapping(assembly.get("first_failed_counts"), "first_failed_counts")
    conditional_failure = _mapping(
        assembly.get("conditional_failure_rates"), "conditional_failure_rates"
    )
    rejection_mass = _mapping(assembly.get("rejection_mass_rates"), "rejection_mass_rates")
    return {
        "active_row_attempt_count": _nonnegative_int(
            assembly.get("active_row_attempt_count"), "active_row_attempt_count"
        ),
        "accepted_row_attempt_count": _nonnegative_int(
            assembly.get("accepted_row_attempt_count"), "accepted_row_attempt_count"
        ),
        "sequential_reach_count": _nonnegative_int(
            reached.get(predicate), f"sequential_reach_counts.{predicate}"
        ),
        "first_failed_count": _nonnegative_int(
            failed.get(predicate), f"first_failed_counts.{predicate}"
        ),
        "conditional_failure_rate": _finite_rate(
            conditional_failure.get(predicate), f"conditional_failure_rates.{predicate}"
        ),
        "rejection_mass_rate": _finite_rate(
            rejection_mass.get(predicate), f"rejection_mass_rates.{predicate}"
        ),
    }


def build_record_strata(
    *,
    record: Mapping[str, Any],
    controls: Mapping[str, Any],
    spec: StageMSpec,
) -> Dict[str, Any]:
    base_direction_id = str(record.get("base_direction_id"))
    timestep = int(record.get("timestep"))
    if timestep not in EXPECTED_TIMESTEPS:
        raise StageMError(f"unexpected timestep: {timestep}")
    comparison = _mapping(record.get("assembly_comparison"), "assembly_comparison")
    comparator_source = str(comparison.get("comparator_source"))
    if comparator_source not in ("raw_oracle", "projected_oracle"):
        raise StageMError("invalid comparator source")
    oof_audit = _mapping(record.get("oof_predicate_assembly"), "oof_predicate_assembly")
    timestep_controls = _mapping(controls.get(str(timestep)), f"controls[{timestep}]")
    comparator_audit = _mapping(
        timestep_controls.get(comparator_source),
        f"controls[{timestep}].{comparator_source}",
    )
    raw_audit = _mapping(timestep_controls.get("raw_oracle"), f"controls[{timestep}].raw")
    projected_audit = _mapping(
        timestep_controls.get("projected_oracle"), f"controls[{timestep}].projected"
    )

    oof_records = _multiplier_record_map(oof_audit, "oof")
    comparator_records = _multiplier_record_map(comparator_audit, "comparator")
    raw_records = _multiplier_record_map(raw_audit, "raw")
    projected_records = _multiplier_record_map(projected_audit, "projected")
    oof_assemblies = _assembly_map(oof_audit, "oof")
    comparator_assemblies = _assembly_map(comparator_audit, "comparator")
    stored = _stored_stratum_map(comparison, "comparison")

    strata: List[Dict[str, Any]] = []
    stored_mismatch_count = 0
    local_ineligible_prior_discriminator_count = 0
    for multiplier in EXPECTED_EXTERNAL_MULTIPLIERS:
        key = _scale_key(multiplier)
        oof_acceptance = _finite_rate(
            oof_records[key].get("selected_scale_positive_rate"),
            f"oof acceptance {key}",
        )
        comparator_acceptance = _finite_rate(
            comparator_records[key].get("selected_scale_positive_rate"),
            f"comparator acceptance {key}",
        )
        raw_acceptance = _finite_rate(
            raw_records[key].get("selected_scale_positive_rate"),
            f"raw acceptance {key}",
        )
        projected_acceptance = _finite_rate(
            projected_records[key].get("selected_scale_positive_rate"),
            f"projected acceptance {key}",
        )
        ungated, predicate_comparisons = _predicate_discriminator(
            oof=_mapping(oof_assemblies[key], f"oof assembly {key}"),
            oracle=_mapping(comparator_assemblies[key], f"oracle assembly {key}"),
            spec=spec,
        )
        stored_predicate = stored[key].get("discriminator_predicate")
        if stored_predicate is not None:
            stored_predicate = str(stored_predicate)
        if stored_predicate != ungated:
            stored_mismatch_count += 1

        oracle_admitted = comparator_acceptance >= spec.oracle_local_acceptance_min
        oof_rejected = oof_acceptance <= spec.oof_local_acceptance_max
        locally_eligible = oracle_admitted and oof_rejected
        local_predicate = ungated if locally_eligible else None
        if stored_predicate is not None and not locally_eligible:
            local_ineligible_prior_discriminator_count += 1
        if not oracle_admitted:
            locus = "oracle_not_admitted_at_external_multiplier"
        elif not oof_rejected:
            locus = "oof_not_rejected_at_external_multiplier"
        elif local_predicate is None:
            locus = "no_local_predicate_discriminator"
        else:
            locus = f"local_discriminator::{local_predicate}"

        upper_oof = _assembly_metrics(
            _mapping(oof_assemblies[key], f"oof assembly {key}"),
            "upper_segment_geometry",
        )
        upper_oracle = _assembly_metrics(
            _mapping(comparator_assemblies[key], f"oracle assembly {key}"),
            "upper_segment_geometry",
        )
        strata.append(
            {
                "external_multiplier": float(multiplier),
                "stratum_key": key,
                "comparator_source": comparator_source,
                "oof_acceptance_rate": oof_acceptance,
                "comparator_oracle_acceptance_rate": comparator_acceptance,
                "raw_oracle_acceptance_rate": raw_acceptance,
                "projected_oracle_acceptance_rate": projected_acceptance,
                "oracle_locally_admitted": oracle_admitted,
                "oof_locally_rejected": oof_rejected,
                "locally_eligible": locally_eligible,
                "stage_l_stored_discriminator_predicate": stored_predicate,
                "recomputed_ungated_discriminator_predicate": ungated,
                "local_discriminator_predicate": local_predicate,
                "locus": locus,
                "upper_segment_oof": upper_oof,
                "upper_segment_oracle": upper_oracle,
                "predicate_comparisons": predicate_comparisons,
            }
        )

    upper_failure_rates = [
        float(item["upper_segment_oof"]["conditional_failure_rate"]) for item in strata
    ]
    upper_mass_rates = [
        float(item["upper_segment_oof"]["rejection_mass_rate"]) for item in strata
    ]
    tolerance = spec.monotonic_tolerance
    upper_failure_nondecreasing = all(
        left <= right + tolerance
        for left, right in zip(upper_failure_rates, upper_failure_rates[1:])
    )
    upper_mass_nondecreasing = all(
        left <= right + tolerance
        for left, right in zip(upper_mass_rates, upper_mass_rates[1:])
    )
    return {
        "base_direction_id": base_direction_id,
        "timestep": timestep,
        "feature_mode": record.get("feature_mode"),
        "comparator_source": comparator_source,
        "strata": strata,
        "stored_stratum_discriminator_mismatch_count": stored_mismatch_count,
        "prior_discriminator_not_locally_eligible_count": (
            local_ineligible_prior_discriminator_count
        ),
        "upper_failure_rate_nondecreasing_with_multiplier": (
            upper_failure_nondecreasing
        ),
        "upper_rejection_mass_nondecreasing_with_multiplier": (
            upper_mass_nondecreasing
        ),
    }


def _count_nested(
    target: MutableMapping[str, int], key: Optional[str], increment: int = 1
) -> None:
    if key is not None:
        target[key] = target.get(key, 0) + increment


def aggregate_by_multiplier(
    records: Sequence[Mapping[str, Any]], spec: StageMSpec
) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for multiplier in EXPECTED_EXTERNAL_MULTIPLIERS:
        key = _scale_key(multiplier)
        cells = [
            _mapping(
                next(
                    item
                    for item in _sequence(record.get("strata"), "record.strata")
                    if _scale_key(float(_mapping(item, "stratum").get("external_multiplier")))
                    == key
                ),
                "selected stratum",
            )
            for record in records
        ]
        predicate_counts: Dict[str, int] = {}
        stored_counts: Dict[str, int] = {}
        eligible = 0
        oracle_admitted = 0
        oof_rejected = 0
        timestep_support: Dict[str, Dict[str, int]] = {
            str(timestep): {} for timestep in EXPECTED_TIMESTEPS
        }
        backbone_support: Dict[str, Dict[str, int]] = {}
        for record, cell in zip(records, cells):
            oracle_admitted += int(cell.get("oracle_locally_admitted") is True)
            oof_rejected += int(cell.get("oof_locally_rejected") is True)
            eligible += int(cell.get("locally_eligible") is True)
            local = cell.get("local_discriminator_predicate")
            stored = cell.get("stage_l_stored_discriminator_predicate")
            if isinstance(local, str):
                _count_nested(predicate_counts, local)
                timestep_key = str(int(record.get("timestep")))
                _count_nested(timestep_support[timestep_key], local)
                backbone = str(record.get("base_direction_id"))
                backbone_support.setdefault(backbone, {})
                _count_nested(backbone_support[backbone], local)
            if isinstance(stored, str):
                _count_nested(stored_counts, stored)
        oof_acceptance = [float(cell["oof_acceptance_rate"]) for cell in cells]
        oracle_acceptance = [
            float(cell["comparator_oracle_acceptance_rate"]) for cell in cells
        ]
        upper_failure = [
            float(cell["upper_segment_oof"]["conditional_failure_rate"])
            for cell in cells
        ]
        upper_mass = [
            float(cell["upper_segment_oof"]["rejection_mass_rate"])
            for cell in cells
        ]
        output[key] = {
            "external_multiplier": float(multiplier),
            "record_count": len(cells),
            "oracle_locally_admitted_count": oracle_admitted,
            "oof_locally_rejected_count": oof_rejected,
            "locally_eligible_count": eligible,
            "local_discriminator_counts": dict(sorted(predicate_counts.items())),
            "stage_l_stored_discriminator_counts": dict(sorted(stored_counts.items())),
            "local_upper_segment_geometry_count": predicate_counts.get(
                "upper_segment_geometry", 0
            ),
            "oof_acceptance_rate_mean": mean(oof_acceptance),
            "oof_acceptance_rate_median": median(oof_acceptance),
            "oracle_acceptance_rate_mean": mean(oracle_acceptance),
            "oracle_acceptance_rate_median": median(oracle_acceptance),
            "oof_upper_conditional_failure_rate_mean": mean(upper_failure),
            "oof_upper_conditional_failure_rate_median": median(upper_failure),
            "oof_upper_rejection_mass_rate_mean": mean(upper_mass),
            "oof_upper_rejection_mass_rate_median": median(upper_mass),
            "predicate_support_by_timestep": {
                item: dict(sorted(counts.items()))
                for item, counts in sorted(timestep_support.items())
            },
            "predicate_support_by_backbone": {
                item: dict(sorted(counts.items()))
                for item, counts in sorted(backbone_support.items())
            },
        }
    return output


def classify_stratification(
    *,
    records: Sequence[Mapping[str, Any]],
    by_multiplier: Mapping[str, Any],
    spec: StageMSpec,
) -> Dict[str, Any]:
    mismatch_count = sum(
        int(record.get("stored_stratum_discriminator_mismatch_count", 0))
        for record in records
    )
    ineligible_prior_count = sum(
        int(record.get("prior_discriminator_not_locally_eligible_count", 0))
        for record in records
    )
    monotone_failure_records = sum(
        int(record.get("upper_failure_rate_nondecreasing_with_multiplier") is True)
        for record in records
    )
    monotone_mass_records = sum(
        int(record.get("upper_rejection_mass_nondecreasing_with_multiplier") is True)
        for record in records
    )

    upper_counts: Dict[str, int] = {
        key: int(_mapping(value, f"by_multiplier[{key}]").get(
            "local_upper_segment_geometry_count", 0
        ))
        for key, value in by_multiplier.items()
    }
    accepted_record_counts: Dict[str, int] = {
        key: sum(
            int(
                next(
                    item
                    for item in _sequence(record.get("strata"), "record.strata")
                    if _scale_key(float(_mapping(item, "stratum").get("external_multiplier")))
                    == key
                ).get("oof_acceptance_rate", 0.0)
                > spec.oof_local_acceptance_max
            )
            for record in records
        )
        for key in by_multiplier
    }
    dominant_key = max(upper_counts, key=lambda key: (upper_counts[key], float(key)))
    dominant_count = upper_counts[dominant_key]
    other_max = max(
        (count for key, count in upper_counts.items() if key != dominant_key),
        default=0,
    )
    stable_upper_keys = [
        key
        for key, count in upper_counts.items()
        if count >= spec.stable_record_support_min
    ]
    timestep_coverage = _mapping(
        _mapping(by_multiplier[dominant_key], "dominant multiplier").get(
            "predicate_support_by_timestep"
        ),
        "dominant timestep support",
    )
    dominant_timestep_stable = all(
        int(_mapping(timestep_coverage.get(str(t)), f"timestep {t}").get(
            "upper_segment_geometry", 0
        ))
        >= spec.timestep_support_min
        for t in EXPECTED_TIMESTEPS
    )
    locally_eligible_dominant = int(
        _mapping(by_multiplier[dominant_key], "dominant multiplier").get(
            "locally_eligible_count", 0
        )
    )

    if mismatch_count:
        root = "phase314b_r258_stagem_persisted_stratum_comparison_not_reproducible"
        next_path = "AUDIT_STAGE_L_STRATUM_SUMMARY_IDENTITY"
        primary = "stratum_summary_identity"
    elif ineligible_prior_count and dominant_count < spec.stable_record_support_min:
        root = "phase314b_r258_stagem_prior_stratum_signal_lacks_local_oracle_admission"
        next_path = "AUDIT_ORACLE_ADMISSION_BY_EXTERNAL_MULTIPLIER"
        primary = "stratum_local_oracle_admission"
    elif any(count >= spec.stable_record_support_min for count in accepted_record_counts.values()):
        root = "phase314b_r258_stagem_external_multiplier_restores_oof_acceptance"
        next_path = "LOCK_EXTERNAL_MULTIPLIER_FOR_OBJECTIVE_TRAIN_ONLY_CONFIRMATION"
        primary = "oof_acceptance_emergence"
    elif (
        dominant_count >= spec.stable_record_support_min
        and other_max <= spec.sparse_other_multiplier_support_max
        and dominant_timestep_stable
        and locally_eligible_dominant >= spec.stable_record_support_min
    ):
        root = "phase314b_r258_stagem_upper_segment_geometry_is_single_multiplier_boundary"
        next_path = "AUDIT_LOWER_MULTIPLIER_REJECTION_AFTER_UPPER_SEGMENT_GATE"
        primary = "single_external_multiplier_upper_segment_boundary"
    elif (
        len(stable_upper_keys) >= 2
        and monotone_failure_records >= spec.stable_record_support_min
        and monotone_mass_records >= spec.stable_record_support_min
    ):
        root = "phase314b_r258_stagem_upper_segment_geometry_tracks_external_direction_magnitude"
        next_path = "CALIBRATE_EXTERNAL_DIRECTION_MAGNITUDE_ON_OBJECTIVE_TRAIN_ONLY"
        primary = "monotone_external_magnitude_upper_segment_boundary"
    elif dominant_count >= spec.stable_record_support_min:
        root = "phase314b_r258_stagem_upper_segment_boundary_depends_on_backbone_or_timestep"
        next_path = "STRATIFY_UPPER_SEGMENT_REJECTION_BY_BACKBONE_AND_TIMESTEP"
        primary = "backbone_or_timestep_dependent_upper_segment_boundary"
    else:
        root = "phase314b_r258_stagem_external_multiplier_stratification_has_no_stable_predicate"
        next_path = "AUDIT_LOW_MULTIPLIER_NONUPPER_REJECTION_PREDICATES"
        primary = "external_multiplier_stratification_insufficient"

    return {
        "root_cause": root,
        "required_next_path": next_path,
        "primary_failure_locus": primary,
        "record_count": len(records),
        "stratum_count": len(records) * len(EXPECTED_EXTERNAL_MULTIPLIERS),
        "stored_stratum_discriminator_mismatch_count": mismatch_count,
        "prior_discriminator_not_locally_eligible_count": ineligible_prior_count,
        "upper_segment_geometry_support_by_multiplier": dict(sorted(upper_counts.items())),
        "oof_acceptance_emergence_record_count_by_multiplier": dict(
            sorted(accepted_record_counts.items())
        ),
        "dominant_upper_multiplier": float(dominant_key),
        "dominant_upper_support_count": dominant_count,
        "maximum_other_multiplier_upper_support_count": other_max,
        "stable_upper_multiplier_keys": sorted(stable_upper_keys, key=float),
        "dominant_multiplier_has_timestep_coverage": dominant_timestep_stable,
        "upper_failure_rate_monotone_record_count": monotone_failure_records,
        "upper_rejection_mass_monotone_record_count": monotone_mass_records,
        "classification_thresholds": asdict(spec),
    }


def build_report(
    *,
    repository: Mapping[str, Any],
    resume1_report: Mapping[str, Any],
    resume2_report: Mapping[str, Any],
    spec: Optional[StageMSpec] = None,
) -> Mapping[str, Any]:
    active = StageMSpec() if spec is None else spec
    active.validate()
    scientific = extract_stage_l_scientific_result(resume1_report)
    validate_resume2_report(resume2_report)
    audit = _mapping(scientific.get("predicate_assembly_audit"), "predicate_assembly_audit")
    controls = _mapping(
        audit.get("control_assembly_by_timestep"), "control_assembly_by_timestep"
    )
    if set(controls) != {str(value) for value in EXPECTED_TIMESTEPS}:
        raise StageMError("control timestep population changed")
    raw_records = _sequence(audit.get("oof_records"), "oof_records")
    if len(raw_records) != EXPECTED_RECORD_COUNT:
        raise StageMError("Stage-L OOF record count changed")
    records = [
        build_record_strata(
            record=_mapping(value, f"oof_records[{index}]"),
            controls=controls,
            spec=active,
        )
        for index, value in enumerate(raw_records)
    ]
    base_ids = sorted({str(record["base_direction_id"]) for record in records})
    if len(base_ids) != EXPECTED_BASE_DIRECTION_COUNT:
        raise StageMError("base-direction population changed")
    population = sorted((str(record["base_direction_id"]), int(record["timestep"])) for record in records)
    expected_population = sorted(
        (base_id, timestep) for base_id in base_ids for timestep in EXPECTED_TIMESTEPS
    )
    if population != expected_population:
        raise StageMError("backbone × timestep population is incomplete")

    by_multiplier = aggregate_by_multiplier(records, active)
    classification = classify_stratification(
        records=records,
        by_multiplier=by_multiplier,
        spec=active,
    )
    report: Dict[str, Any] = {
        "phase": PHASE,
        "schema": SCHEMA,
        "execution_verdict": "PASS",
        "scientific_status": "BLOCKED",
        "root_cause": classification["root_cause"],
        "required_next_path": classification["required_next_path"],
        "primary_failure_locus": classification["primary_failure_locus"],
        "selected_configuration": None,
        "train_only_recommendation": None,
        "repository": dict(repository),
        "immutable_inputs": {
            "stage_l_resume1_report": STAGEL_RESUME1_REPORT,
            "stage_l_resume1_report_sha256": EXPECTED_STAGEL_RESUME1_REPORT_SHA256,
            "stage_l_single_run_sha256": EXPECTED_STAGEL_SINGLE_RUN_SHA256,
            "stage_l_resume2_report": STAGEL_RESUME2_REPORT,
            "stage_l_resume2_report_sha256": EXPECTED_STAGEL_RESUME2_REPORT_SHA256,
            "callback_off_on_pair_count": EXPECTED_CALLBACK_PAIRS,
            "callback_pairs_rerun": False,
            "stage_l_science_rerun": False,
            "historical_tests_rerun": False,
        },
        "stratification_spec": asdict(active),
        "base_direction_ids": base_ids,
        "timesteps": list(EXPECTED_TIMESTEPS),
        "external_multipliers": list(EXPECTED_EXTERNAL_MULTIPLIERS),
        "predicate_order": list(PREDICATE_ORDER),
        "record_strata": records,
        "aggregate_by_external_multiplier": by_multiplier,
        "classification": classification,
        "source_support_limitations": {
            "proposal_displacement_magnitude_available": False,
            "selected_internal_scale_histogram_available": False,
            "reason": (
                "Stage-L persisted scalar assembly and acceptance summaries but did not "
                "persist direction tensors, proposal norms, or selected-scale histograms."
            ),
            "no_missing_value_was_inferred": True,
        },
        "mechanism_boundary": {
            "stagee_modified": False,
            "stagek_modified": False,
            "stagel_modified": False,
            "stagel_resume1_report_modified": False,
            "stagel_resume2_report_modified": False,
            "callback_schema_modified": False,
            "integrator_thresholds_changed": False,
            "integrator_scale_search_changed": False,
            "external_scale_bank_changed": False,
            "direction_model_refit": False,
            "new_ranker_fitted": False,
            "report_only_stratification": True,
        },
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
    return report


def blocked_payload(error: BaseException, repository: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    return {
        "phase": PHASE,
        "schema": "phase314b_r258_stagem_blocked_v1",
        "execution_verdict": "BLOCKED",
        "scientific_status": "BLOCKED",
        "root_cause": "phase314b_r258_stagem_finalizer_failed_before_completion",
        "required_next_path": "RESTORE_STAGEM_REPORT_ONLY_STRATIFICATION_FINALIZER",
        "error_type": type(error).__name__,
        "error": str(error),
        "repository": None if repository is None else dict(repository),
        "stage_l_science_rerun": False,
        "callback_pairs_rerun": False,
        "historical_tests_rerun": False,
        "selected_configuration": None,
        "train_only_recommendation": None,
        **{key: False for key in FALSE_BOUNDARIES},
    }
