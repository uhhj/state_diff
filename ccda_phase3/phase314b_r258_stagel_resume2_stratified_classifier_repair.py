"""Phase3.14b-r2.5.8 Stage-L Resume2 stratified-classifier repair.

Stage-L Resume1 completed the unchanged predicate-assembly audit.  Its 27/27
record-level comparisons reported
``predicate_assembly_is_scale_stratified_and_heterogeneous`` while the top-level
classifier incorrectly emitted ``callback_predicate_masks_not_discriminative``.

The defect is purely classificatory: ``classify`` only accumulates a stratum's
``shared_discriminator_predicate`` and ignores non-empty heterogeneous
``discriminator_counts``.  This module validates the immutable Resume1 report,
recomputes the classification from its persisted record summaries, and emits a
write-once correction report.  It does not import or execute Stage-E/K/L
science, touch CUDA, reconstruct callback events, or access holdout/probe data.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


PHASE = "Phase3.14b-r2.5.8 Stage L Resume2"
SCHEMA = "phase314b_r258_stagel_resume2_stratified_classifier_repair_v1"

BASE_EVIDENCE_COMMIT = "c8c9bd0139ab7249689855c55be1b491d759a3f4"
BASE_IMPLEMENTATION_COMMIT = "9fd9751188d011c8a4fba0457ed03fd067e41f7f"
BASE_BLOCKED_PROVENANCE_COMMIT = "77953e4bdf8779fb94ba3f59de886133b9477321"
STAGEL_IMPLEMENTATION_COMMIT = "cd7fe38b1013b13ca84086bedd64dfa5eaacff47"
EXPECTED_REMOTE = "6758ea7ad800667a436b0243d3b1f6c63256d854"
EXPECTED_SUBMODULE = "633a88752445cf5d6776ed374fdbbdb35f93050c"

BASE_REPORT = "reports/phase3_14b_r258_stagel_resume1_predicate_assembly_audit_summary.json"
EXPECTED_BASE_REPORT_SHA256 = "120cf31f649342589e74e3a5b8ce97bb50f4aac33f865fae54ebebef02f79eed"
EXPECTED_BASE_SINGLE_RUN_SHA256 = "193a6514fd0e3fac76e3cb913eaa10be90b86d400183fe348190056c30080beb"
EXPECTED_BASE_ROOT_CAUSE = "phase314b_r258_stagel_callback_predicate_masks_not_discriminative"
EXPECTED_BASE_NEXT_PATH = "AUDIT_CALLBACK_PREDICATE_MASKS_AGAINST_FAST_FEASIBLE_CONJUNCTION"
EXPECTED_RECORD_LOCUS = "predicate_assembly_is_scale_stratified_and_heterogeneous"
EXPECTED_RECORD_COUNT = 27
EXPECTED_CALLBACK_PAIRS = 132

SUCCESS_REPORT = "reports/phase3_14b_r258_stagel_resume2_stratified_classifier_repair_summary.json"
BLOCKED_REPORT = "reports/phase3_14b_r258_stagel_resume2_stratified_classifier_repair_blocked_summary.json"

CORRECTED_ROOT_CAUSE = "phase314b_r258_stagel_rejection_signal_is_scale_stratified_and_heterogeneous"
CORRECTED_NEXT_PATH = "STRATIFY_OOF_REJECTION_BY_CALLBACK_SCALE_AND_PREDICATE"
CORRECTED_PRIMARY_LOCUS = "scale_stratified_heterogeneous_predicate_assembly"

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

IMPLEMENTATION_SUBJECT = "Phase3.14b-r2.5.8 Stage L Resume2: repair stratified classification"
EVIDENCE_SUBJECT = "Record Phase3.14b-r2.5.8 Stage L Resume2 classification evidence"
BLOCKED_EVIDENCE_SUBJECT = "Record Phase3.14b-r2.5.8 Stage L Resume2 blocked evidence"

IMPLEMENTATION_PATHS: Tuple[Tuple[str, str], ...] = (
    ("A", "ccda_phase3/phase314b_r258_stagel_resume2_stratified_classifier_repair.py"),
    ("A", "scripts/phase3_14b_r258_stagel_resume2_finalize.py"),
    ("A", "tests/test_phase3_14b_r258_stagel_resume2_stratified_classifier_repair.py"),
)


class StageLResume2Error(RuntimeError):
    """Fail-closed Resume2 validation error."""


@dataclass(frozen=True)
class StratifiedSignal:
    external_record_count: int
    internal_record_count: int
    external_predicate_counts: Mapping[str, int]
    internal_predicate_counts: Mapping[str, int]
    records_with_any_stratified_signal: int


def stable_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
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
        raise StageLResume2Error(f"JSON root is not a mapping: {path}")
    return value


def write_once(path: Path, payload: bytes) -> None:
    if path.exists():
        raise StageLResume2Error(f"refusing to overwrite report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise StageLResume2Error(f"stale temporary report exists: {temporary}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=str(root),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise StageLResume2Error(
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
            raise StageLResume2Error(f"unexpected name-status row: {line}")
        rows.append((parts[0], parts[1]))
    return tuple(sorted(rows))


def require_clean(root: Path, label: str) -> None:
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise StageLResume2Error(f"{label} is not clean:\n{status}")


def require_false(mapping: Mapping[str, Any], keys: Iterable[str], label: str) -> None:
    for key in keys:
        if mapping.get(key) is not False:
            raise StageLResume2Error(f"{label}.{key} is not false")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StageLResume2Error(f"{label} is not a mapping")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise StageLResume2Error(f"{label} is not a sequence")
    return value


def validate_repository(root: Path) -> Dict[str, Any]:
    repo = Path(root).resolve()
    if _git(repo, "branch", "--show-current") != "Experiment1":
        raise StageLResume2Error("Stage-L Resume2 requires Experiment1")
    head = _git(repo, "rev-parse", "HEAD")
    if commit_parent(repo, head) != BASE_EVIDENCE_COMMIT:
        raise StageLResume2Error("Resume2 implementation parent changed")
    if commit_subject(repo, head) != IMPLEMENTATION_SUBJECT:
        raise StageLResume2Error("Resume2 implementation subject changed")
    if commit_name_status(repo, head) != tuple(sorted(IMPLEMENTATION_PATHS)):
        raise StageLResume2Error("Resume2 implementation path population changed")
    if _git(repo, "rev-parse", "refs/remotes/origin/Experiment1") != EXPECTED_REMOTE:
        raise StageLResume2Error("origin/Experiment1 changed")
    if _git(repo, "rev-parse", "HEAD:external/deformable-ravens") != EXPECTED_SUBMODULE:
        raise StageLResume2Error("submodule gitlink changed")
    submodule = repo / "external/deformable-ravens"
    if _git(submodule, "rev-parse", "HEAD") != EXPECTED_SUBMODULE:
        raise StageLResume2Error("submodule worktree commit changed")
    require_clean(repo, "main worktree")
    require_clean(submodule, "submodule worktree")

    expected_subjects = {
        BASE_EVIDENCE_COMMIT: "Record Phase3.14b-r2.5.8 Stage L Resume1 predicate assembly evidence",
        BASE_IMPLEMENTATION_COMMIT: "Phase3.14b-r2.5.8 Stage L Resume1: recover deterministic execution",
        BASE_BLOCKED_PROVENANCE_COMMIT: "Record Phase3.14b-r2.5.8 Stage L deterministic-environment blocked provenance",
        STAGEL_IMPLEMENTATION_COMMIT: "Phase3.14b-r2.5.8 Stage L: audit explicit callback predicate assembly",
    }
    for commit, subject in expected_subjects.items():
        if commit_subject(repo, commit) != subject:
            raise StageLResume2Error(f"commit subject changed: {commit}")
    if commit_parent(repo, BASE_EVIDENCE_COMMIT) != BASE_IMPLEMENTATION_COMMIT:
        raise StageLResume2Error("Resume1 evidence parent changed")
    if commit_parent(repo, BASE_IMPLEMENTATION_COMMIT) != BASE_BLOCKED_PROVENANCE_COMMIT:
        raise StageLResume2Error("Resume1 implementation parent changed")
    if commit_parent(repo, BASE_BLOCKED_PROVENANCE_COMMIT) != STAGEL_IMPLEMENTATION_COMMIT:
        raise StageLResume2Error("blocked provenance parent changed")

    expected_evidence_path = (("A", BASE_REPORT),)
    if commit_name_status(repo, BASE_EVIDENCE_COMMIT) != expected_evidence_path:
        raise StageLResume2Error("Resume1 evidence path population changed")

    report = repo / BASE_REPORT
    if not report.is_file():
        raise StageLResume2Error("Resume1 report missing")
    observed_sha = sha256_path(report)
    if observed_sha != EXPECTED_BASE_REPORT_SHA256:
        raise StageLResume2Error("Resume1 report SHA changed")
    blob = subprocess.run(
        ["git", "show", f"{BASE_EVIDENCE_COMMIT}:{BASE_REPORT}"],
        cwd=str(repo),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if blob.returncode != 0 or blob.stdout != report.read_bytes():
        raise StageLResume2Error("Resume1 report differs from committed blob")
    for relative in (SUCCESS_REPORT, BLOCKED_REPORT):
        if (repo / relative).exists():
            raise StageLResume2Error(f"Resume2 output already exists: {relative}")
    return {
        "head": head,
        "parent": BASE_EVIDENCE_COMMIT,
        "origin_experiment1": EXPECTED_REMOTE,
        "submodule_commit": EXPECTED_SUBMODULE,
        "base_report": BASE_REPORT,
        "base_report_sha256": observed_sha,
    }


def validate_classifier_source_bug(source_text: str) -> Dict[str, Any]:
    """Prove that the old top-level classifier ignores heterogeneous counts."""
    required = (
        'shared = value.get("shared_discriminator_predicate")',
        "elif external_predicates or internal_predicates:",
        'root = "phase314b_r258_stagel_callback_predicate_masks_not_discriminative"',
    )
    for fragment in required:
        if fragment not in source_text:
            raise StageLResume2Error(
                f"Stage-L classifier source no longer matches expected defect: {fragment}"
            )
    classify_start = source_text.find("def classify(")
    classify_end = source_text.find("\ndef run_calibration(", classify_start)
    if classify_start < 0 or classify_end < 0:
        raise StageLResume2Error("cannot isolate Stage-L classify function")
    classify_source = source_text[classify_start:classify_end]
    if '["discriminator_counts"]' in classify_source or ".get(\"discriminator_counts\")" in classify_source:
        raise StageLResume2Error("Stage-L classifier already consumes discriminator_counts")
    return {
        "old_classifier_reads_shared_only": True,
        "old_classifier_ignores_heterogeneous_discriminator_counts": True,
        "old_classifier_source_sha256": sha256_bytes(source_text.encode("utf-8")),
    }


def extract_stage_l_scientific_result(resume1_report: Mapping[str, Any]) -> Mapping[str, Any]:
    if resume1_report.get("execution_verdict") != "PASS":
        raise StageLResume2Error("Resume1 execution is not PASS")
    if resume1_report.get("scientific_status") != "BLOCKED":
        raise StageLResume2Error("Resume1 scientific status changed")
    if resume1_report.get("root_cause") != EXPECTED_BASE_ROOT_CAUSE:
        raise StageLResume2Error("unexpected Resume1 root cause")
    if resume1_report.get("required_next_path") != EXPECTED_BASE_NEXT_PATH:
        raise StageLResume2Error("unexpected Resume1 next path")
    if resume1_report.get("selected_configuration") is not None:
        raise StageLResume2Error("Resume1 selected_configuration is not null")
    if resume1_report.get("train_only_recommendation") is not None:
        raise StageLResume2Error("Resume1 train_only_recommendation is not null")

    recovery = _mapping(resume1_report.get("recovery_contract"), "recovery_contract")
    if recovery.get("underlying_stage_l_single_run_result_sha256") != EXPECTED_BASE_SINGLE_RUN_SHA256:
        raise StageLResume2Error("underlying Stage-L single-run SHA changed")
    if int(recovery.get("underlying_callback_pair_count", -1)) != EXPECTED_CALLBACK_PAIRS:
        raise StageLResume2Error("underlying callback pair count changed")
    if recovery.get("stage_l_science_modified") is not False:
        raise StageLResume2Error("Resume1 says Stage-L science was modified")
    if recovery.get("existing_test_gates_rerun") is not False:
        raise StageLResume2Error("Resume1 says existing tests were rerun")

    stage_l_result = _mapping(resume1_report.get("stage_l_result"), "stage_l_result")
    if stage_l_result.get("execution_verdict") != "PASS":
        raise StageLResume2Error("underlying Stage-L execution is not PASS")
    scientific = _mapping(stage_l_result.get("scientific_result"), "scientific_result")
    if scientific.get("root_cause") != EXPECTED_BASE_ROOT_CAUSE:
        raise StageLResume2Error("underlying scientific root changed")
    if scientific.get("required_next_path") != EXPECTED_BASE_NEXT_PATH:
        raise StageLResume2Error("underlying scientific next path changed")
    if scientific.get("selected_configuration") is not None:
        raise StageLResume2Error("underlying selected_configuration is not null")
    if scientific.get("train_only_recommendation") is not None:
        raise StageLResume2Error("underlying train_only_recommendation is not null")
    require_false(scientific, FALSE_BOUNDARIES, "scientific_result")
    return scientific


def _merge_counts(target: MutableMapping[str, int], source: Mapping[str, Any]) -> None:
    for key, value in source.items():
        count = int(value)
        if count < 0:
            raise StageLResume2Error("negative discriminator count")
        if count:
            target[str(key)] = target.get(str(key), 0) + count


def collect_stratified_signal(records: Sequence[Mapping[str, Any]]) -> StratifiedSignal:
    external_records = 0
    internal_records = 0
    any_records = 0
    external_counts: Dict[str, int] = {}
    internal_counts: Dict[str, int] = {}
    for index, record in enumerate(records):
        comparison = _mapping(record.get("assembly_comparison"), f"records[{index}].assembly_comparison")
        if comparison.get("locus") != EXPECTED_RECORD_LOCUS:
            raise StageLResume2Error(
                f"record {index} locus changed: {comparison.get('locus')}"
            )
        if comparison.get("corrected_discriminator_predicate") is not None:
            raise StageLResume2Error(f"record {index} unexpectedly has overall discriminator")
        found = False
        for field, counter, which in (
            ("external_multiplier_strata", external_counts, "external"),
            ("internal_scale_strata", internal_counts, "internal"),
        ):
            stratum = _mapping(comparison.get(field), f"records[{index}].{field}")
            counts = _mapping(stratum.get("discriminator_counts"), f"records[{index}].{field}.discriminator_counts")
            if counts:
                _merge_counts(counter, counts)
                found = True
                if which == "external":
                    external_records += 1
                else:
                    internal_records += 1
            if stratum.get("shared_discriminator_predicate") is not None:
                raise StageLResume2Error(
                    f"record {index} {field} is not heterogeneous"
                )
        if not found:
            raise StageLResume2Error(
                f"record {index} claims scale-stratified heterogeneity without counts"
            )
        any_records += 1
    return StratifiedSignal(
        external_record_count=external_records,
        internal_record_count=internal_records,
        external_predicate_counts=dict(sorted(external_counts.items())),
        internal_predicate_counts=dict(sorted(internal_counts.items())),
        records_with_any_stratified_signal=any_records,
    )


def corrected_classification(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if len(records) != EXPECTED_RECORD_COUNT:
        raise StageLResume2Error(
            f"Stage-L record count changed: expected {EXPECTED_RECORD_COUNT}, got {len(records)}"
        )
    signal = collect_stratified_signal(records)
    if signal.records_with_any_stratified_signal != EXPECTED_RECORD_COUNT:
        raise StageLResume2Error("not all Stage-L records contain stratified signal")
    return {
        "root_cause": CORRECTED_ROOT_CAUSE,
        "required_next_path": CORRECTED_NEXT_PATH,
        "primary_failure_locus": CORRECTED_PRIMARY_LOCUS,
        "record_count": EXPECTED_RECORD_COUNT,
        "record_locus_counts": {EXPECTED_RECORD_LOCUS: EXPECTED_RECORD_COUNT},
        "corrected_discriminator_predicate_counts": {},
        "external_heterogeneous_record_count": signal.external_record_count,
        "internal_heterogeneous_record_count": signal.internal_record_count,
        "records_with_any_stratified_signal": signal.records_with_any_stratified_signal,
        "external_stratum_discriminator_counts": dict(signal.external_predicate_counts),
        "internal_stratum_discriminator_counts": dict(signal.internal_predicate_counts),
        "all_records_scale_stratified_and_heterogeneous": True,
        "classification_bug_corrected": True,
    }


def build_corrected_report(
    *,
    root: Path,
    repository: Mapping[str, Any],
    base_report: Mapping[str, Any],
) -> Mapping[str, Any]:
    scientific = extract_stage_l_scientific_result(base_report)
    audit = _mapping(scientific.get("predicate_assembly_audit"), "predicate_assembly_audit")
    if int(audit.get("callback_off_on_pair_count", -1)) != EXPECTED_CALLBACK_PAIRS:
        raise StageLResume2Error("callback pair count changed")
    if audit.get("all_callback_results_bit_exact") is not True:
        raise StageLResume2Error("callback identity changed")
    records = _sequence(audit.get("oof_records"), "predicate_assembly_audit.oof_records")
    typed_records = [_mapping(value, f"oof_records[{index}]") for index, value in enumerate(records)]
    corrected = corrected_classification(typed_records)

    stage_l_source = root / "ccda_phase3/phase314b_r258_stagel_predicate_assembly_audit.py"
    if not stage_l_source.is_file():
        raise StageLResume2Error("Stage-L source missing")
    source_bug = validate_classifier_source_bug(stage_l_source.read_text(encoding="utf-8"))

    result: Dict[str, Any] = {
        "phase": PHASE,
        "schema": SCHEMA,
        "execution_verdict": "PASS",
        "scientific_status": "BLOCKED",
        "root_cause": corrected["root_cause"],
        "required_next_path": corrected["required_next_path"],
        "primary_failure_locus": corrected["primary_failure_locus"],
        "selected_configuration": None,
        "train_only_recommendation": None,
        "repository": dict(repository),
        "correction_contract": {
            "schema": "phase314b_r258_stagel_resume2_correction_contract_v1",
            "correction_scope": "classification_only",
            "base_report": BASE_REPORT,
            "base_report_sha256": EXPECTED_BASE_REPORT_SHA256,
            "base_single_run_result_sha256": EXPECTED_BASE_SINGLE_RUN_SHA256,
            "base_root_cause": EXPECTED_BASE_ROOT_CAUSE,
            "base_required_next_path": EXPECTED_BASE_NEXT_PATH,
            "base_report_modified": False,
            "stage_l_science_rerun": False,
            "callback_pairs_rerun": False,
            "existing_tests_rerun": False,
            "stagee_source_modified": False,
            "stagek_source_modified": False,
            "stagel_source_modified": False,
            "callback_schema_modified": False,
            "integrator_numerics_modified": False,
            "source_bug_proof": source_bug,
        },
        "corrected_classification": corrected,
        "original_classification": dict(
            _mapping(audit.get("classification"), "predicate_assembly_audit.classification")
        ),
        "callback_off_on_pair_count": EXPECTED_CALLBACK_PAIRS,
        "all_callback_results_bit_exact": True,
        "boundaries": {key: False for key in FALSE_BOUNDARIES},
    }
    result["correction_sha256"] = sha256_bytes(stable_json_bytes(result))
    return result


def blocked_payload(error: BaseException, repository: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    return {
        "phase": PHASE,
        "schema": "phase314b_r258_stagel_resume2_stratified_classifier_repair_blocked_v1",
        "execution_verdict": "BLOCKED",
        "scientific_status": "BLOCKED",
        "root_cause": "phase314b_r258_stagel_resume2_classifier_repair_failed",
        "required_next_path": "INSPECT_STAGEL_RESUME2_WITHOUT_RERUNNING_STAGE_L",
        "error_type": f"{type(error).__module__}.{type(error).__qualname__}",
        "error": str(error),
        "repository": repository,
        "selected_configuration": None,
        "train_only_recommendation": None,
        "stage_l_science_rerun": False,
        "callback_pairs_rerun": False,
        "existing_tests_rerun": False,
        "boundaries": {key: False for key in FALSE_BOUNDARIES},
    }
