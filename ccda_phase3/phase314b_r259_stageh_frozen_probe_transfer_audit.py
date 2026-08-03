"""Read-only Stage-H audit of Stage-G frozen-probe transfer failure.

Stage G consumed the one and only frozen-probe evaluation and persisted bounded
JSON evidence.  Stage H does not load the dataset, initialize CUDA, fit a model,
generate a candidate, evaluate a recipe, or access any split target.  It reads
only the committed Stage-G environment-probe, worker and controller-summary
reports.

The audit separates what is identifiable from those reports from what is not:

* objective-train risk prevalence versus frozen-probe raw-candidate adverse
  prevalence;
* the exact Brier penalty attributable to prevalence shift;
* model Brier excess beyond the training-prevalence constant predictor;
* raw candidate admission, risk conditional-keep rate, and the effect of
  abstention relative to the persisted no-abstention baseline;
* preservation or regression of the aligned geometry gate.

Row-level probabilities, labels, descriptors and accept masks were intentionally
not persisted by Stage G.  AUROC, ECE, reliability diagrams, score-distribution
shift, descriptor-distribution shift and row-level attribution are therefore
reported as unavailable.  The audit never treats the consumed frozen probe as a
retuning surface.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

PHASE = "Phase3.14b-r2.5.9 Stage H"
SCHEMA = "phase314b_r259_stageh_frozen_probe_transfer_failure_audit_v1"
BLOCKED_SCHEMA = SCHEMA + "_blocked_v1"

BASE_HEAD = "127cf4c59edcd0a6052cb04c969f86f393b23616"
BASE_IMPLEMENTATION = "048adcb8c375fd8431b14c9130612fb837f8509f"
EXPECTED_SUBMODULE = "633a88752445cf5d6776ed374fdbbdb35f93050c"

BASE_IMPLEMENTATION_SUBJECT = (
    "Phase3.14b-r2.5.9 Stage G: evaluate canonical risk repair on frozen probe"
)
BASE_EVIDENCE_SUBJECT = (
    "Record Phase3.14b-r2.5.9 Stage G one-shot frozen-probe evidence"
)
IMPLEMENTATION_SUBJECT = (
    "Phase3.14b-r2.5.9 Stage H: audit frozen-probe risk transfer failure"
)
EVIDENCE_SUBJECT = (
    "Record Phase3.14b-r2.5.9 Stage H frozen-probe transfer audit"
)
BLOCKED_EVIDENCE_SUBJECT = (
    "Record Phase3.14b-r2.5.9 Stage H blocked evidence"
)

IMPLEMENTATION_PATHS: Tuple[Tuple[str, str], ...] = (
    ("A", "ccda_phase3/phase314b_r259_stageh_frozen_probe_transfer_audit.py"),
    ("A", "scripts/phase3_14b_r259_stageh_frozen_probe_transfer_audit.py"),
    ("A", "tests/test_phase3_14b_r259_stageh_frozen_probe_transfer_audit.py"),
)

SOURCE_PROBE = "reports/phase3_14b_r259_stageg_environment_probe_evidence.json"
SOURCE_WORKER = "reports/phase3_14b_r259_stageg_frozen_probe_worker_evidence.json"
SOURCE_SUMMARY = "reports/phase3_14b_r259_stageg_one_shot_frozen_probe_summary.json"
SOURCE_PROBE_SCHEMA = (
    "phase314b_r259_stageg_one_shot_frozen_probe_v1_environment_probe_v1"
)
SOURCE_WORKER_SCHEMA = "phase314b_r259_stageg_one_shot_frozen_probe_v1_worker_v1"
SOURCE_SUMMARY_SCHEMA = "phase314b_r259_stageg_one_shot_frozen_probe_v1"

SUCCESS_REPORT = "reports/phase3_14b_r259_stageh_frozen_probe_transfer_failure_audit_summary.json"
BLOCKED_REPORT = "reports/phase3_14b_r259_stageh_frozen_probe_transfer_failure_audit_blocked_summary.json"

LOCKED_TIMESTEPS: Tuple[int, ...] = (10, 25, 50)
EXPECTED_CANONICAL_RECIPE = (
    "desc_compact_v1__l2_4.00__temp_1.00__shrink_1.00__risk_0.50"
)
EXPECTED_FROZEN_PROBE_ROWS = 126
FLOAT_TOLERANCE = 1.0e-10

FALSE_BOUNDARIES: Tuple[str, ...] = (
    "environment_probe_run",
    "science_worker_run",
    "cuda_used",
    "dataset_loaded",
    "direction_fit_run",
    "risk_fit_run",
    "candidate_generation_run",
    "recipe_evaluation_run",
    "selection_holdout_reaccessed",
    "selection_holdout_target_loaded",
    "selection_holdout_used_for_fit_or_selection",
    "frozen_probe_reaccessed",
    "frozen_probe_target_loaded",
    "frozen_probe_used_for_fit_or_selection",
    "formal_training_run",
    "reverse_sampling_run",
    "idm_run",
    "candidate_execution",
    "deformable_ravens_executed",
    "phase4",
    "cps",
    "checkpoint_saved",
    "weights_persisted",
    "prediction_tensor_persisted",
    "candidate_tensor_persisted",
    "descriptor_tensor_persisted",
    "risk_probability_tensor_persisted",
    "npz_saved",
    "cache_saved",
    "image_saved",
    "video_saved",
)


class StageHAuditError(RuntimeError):
    """Fail-closed Stage-H audit error."""


def stable_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        indent=2,
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise StageHAuditError("JSON root is not a mapping: {}".format(path))
    return value


def atomic_write_once(path: Path, payload: bytes) -> None:
    target = Path(path)
    if target.exists():
        raise StageHAuditError("write-once output exists: {}".format(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(target)
        descriptor = os.open(str(target.parent), os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=str(root), text=True, stderr=subprocess.STDOUT
    ).strip()


def _git_bytes(root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=str(root))


def commit_name_status(root: Path, commit: str) -> Tuple[Tuple[str, str], ...]:
    output = _git(root, "diff-tree", "--no-commit-id", "--name-status", "-r", commit)
    records: List[Tuple[str, str]] = []
    for line in output.splitlines():
        fields = line.split("\t")
        if len(fields) != 2:
            raise StageHAuditError("unexpected commit path record: {!r}".format(line))
        records.append((fields[0], fields[1]))
    return tuple(sorted(records))


def assert_clean_worktree(root: Path, label: str) -> None:
    if _git(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise StageHAuditError("{} worktree is dirty".format(label))


def validate_repository(root: Path, implementation_commit: str) -> Mapping[str, Any]:
    repo = Path(root).resolve()
    checks = {
        "branch": (_git(repo, "branch", "--show-current"), "Experiment1"),
        "HEAD": (_git(repo, "rev-parse", "HEAD"), implementation_commit),
        "Stage-H parent": (_git(repo, "rev-parse", "HEAD^"), BASE_HEAD),
        "Stage-G evidence parent": (_git(repo, "rev-parse", BASE_HEAD + "^"), BASE_IMPLEMENTATION),
        "Stage-G implementation subject": (
            _git(repo, "show", "-s", "--format=%s", BASE_IMPLEMENTATION),
            BASE_IMPLEMENTATION_SUBJECT,
        ),
        "Stage-G evidence subject": (
            _git(repo, "show", "-s", "--format=%s", BASE_HEAD),
            BASE_EVIDENCE_SUBJECT,
        ),
        "Stage-H implementation subject": (
            _git(repo, "show", "-s", "--format=%s", implementation_commit),
            IMPLEMENTATION_SUBJECT,
        ),
        "submodule gitlink": (
            _git(repo, "rev-parse", "HEAD:external/deformable-ravens"),
            EXPECTED_SUBMODULE,
        ),
    }
    for label, (actual, expected) in checks.items():
        if actual != expected:
            raise StageHAuditError(
                "{} changed: expected={} actual={}".format(label, expected, actual)
            )
    if commit_name_status(repo, implementation_commit) != tuple(sorted(IMPLEMENTATION_PATHS)):
        raise StageHAuditError("Stage-H implementation population changed")
    submodule = repo / "external/deformable-ravens"
    if _git(submodule, "rev-parse", "HEAD") != EXPECTED_SUBMODULE:
        raise StageHAuditError("DeformableRavens worktree commit changed")
    assert_clean_worktree(submodule, "DeformableRavens")
    assert_clean_worktree(repo, "Stage-H")
    for relative in (SUCCESS_REPORT, BLOCKED_REPORT):
        if (repo / relative).exists():
            raise StageHAuditError("Stage-H terminal report already exists")
    return {
        "root": str(repo),
        "head": implementation_commit,
        "parent": BASE_HEAD,
        "stage_g_implementation": BASE_IMPLEMENTATION,
        "stage_g_evidence": BASE_HEAD,
        "submodule_commit": EXPECTED_SUBMODULE,
        "remote_identity_used_as_gate": False,
    }


def _validate_self_hash(
    payload: Mapping[str, Any], field: str, label: str
) -> str:
    observed = payload.get(field)
    if not isinstance(observed, str) or len(observed) != 64:
        raise StageHAuditError("{} self-hash is missing".format(label))
    expected = sha256_bytes(
        stable_json_bytes({key: value for key, value in payload.items() if key != field})
    )
    if observed != expected:
        raise StageHAuditError("{} self-hash changed".format(label))
    return observed


def _require_committed_blob(root: Path, relative: str) -> Path:
    repo = Path(root).resolve()
    path = repo / relative
    if not path.is_file():
        raise StageHAuditError("source evidence is missing: {}".format(relative))
    committed = _git_bytes(repo, "show", "{}:{}".format(BASE_HEAD, relative))
    if committed != path.read_bytes():
        raise StageHAuditError("source evidence differs from Stage-G evidence commit: {}".format(relative))
    return path


def validate_source_evidence(
    root: Path,
) -> Tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    probe_path = _require_committed_blob(root, SOURCE_PROBE)
    worker_path = _require_committed_blob(root, SOURCE_WORKER)
    summary_path = _require_committed_blob(root, SOURCE_SUMMARY)
    probe = load_json(probe_path)
    worker = load_json(worker_path)
    summary = load_json(summary_path)

    if probe.get("schema") != SOURCE_PROBE_SCHEMA or probe.get("execution_verdict") != "PASS":
        raise StageHAuditError("Stage-G environment-probe schema/verdict changed")
    _validate_self_hash(probe, "probe_evidence_sha256", "Stage-G environment probe")
    environment_audit = probe.get("portable_environment_audit")
    if not isinstance(environment_audit, Mapping) or environment_audit.get("portable_contract_passed") is not True:
        raise StageHAuditError("Stage-G portable environment contract did not pass")
    if probe.get("frozen_probe_accessed") is not False:
        raise StageHAuditError("Stage-G environment probe accessed frozen probe")

    if worker.get("schema") != SOURCE_WORKER_SCHEMA or worker.get("execution_verdict") != "PASS":
        raise StageHAuditError("Stage-G worker schema/verdict changed")
    _validate_self_hash(worker, "worker_result_sha256", "Stage-G worker")
    expected_worker = {
        "scientific_status": "BLOCKED",
        "root_cause": "phase314b_r259_stageg_canonical_joint_policy_fails_one_shot_frozen_probe",
        "required_next_path": "AUDIT_R259_FROZEN_PROBE_TRANSFER_FAILURE_WITHOUT_REACCESS_OR_RETUNING",
        "frozen_probe_evaluation_count_added": 1,
        "cumulative_frozen_probe_evaluation_count": 1,
        "frozen_probe_accessed": True,
        "selection_holdout_evaluation_count_added": 0,
        "cumulative_selection_holdout_evaluation_count": 1,
        "rerun_authorized": False,
        "selected_configuration": None,
        "train_only_recommendation": None,
    }
    for key, expected in expected_worker.items():
        if worker.get(key) != expected:
            raise StageHAuditError("Stage-G worker field changed: {}".format(key))
    contract = worker.get("preregistration_contract")
    if not isinstance(contract, Mapping):
        raise StageHAuditError("Stage-G preregistration contract is missing")
    if contract.get("canonical_recipe_id") != EXPECTED_CANONICAL_RECIPE:
        raise StageHAuditError("Stage-G canonical recipe changed")
    if contract.get("one_shot_frozen_probe") is not True or contract.get("rerun_forbidden") is not True:
        raise StageHAuditError("Stage-G one-shot contract changed")

    if summary.get("schema") != SOURCE_SUMMARY_SCHEMA or summary.get("execution_verdict") != "PASS":
        raise StageHAuditError("Stage-G summary schema/verdict changed")
    _validate_self_hash(summary, "summary_sha256", "Stage-G summary")
    expected_summary = {
        "scientific_status": "BLOCKED",
        "root_cause": "phase314b_r259_stageg_canonical_joint_policy_fails_one_shot_frozen_probe",
        "required_next_path": "AUDIT_R259_FROZEN_PROBE_TRANSFER_FAILURE_WITHOUT_REACCESS_OR_RETUNING",
        "all_timesteps_pass": False,
        "frozen_probe_evaluation_count_added": 1,
        "cumulative_frozen_probe_evaluation_count": 1,
        "frozen_probe_accessed": True,
        "selection_holdout_evaluation_count_added": 0,
        "cumulative_selection_holdout_evaluation_count": 1,
        "rerun_authorized": False,
        "selected_configuration": None,
        "train_only_recommendation": None,
    }
    for key, expected in expected_summary.items():
        if summary.get(key) != expected:
            raise StageHAuditError("Stage-G summary field changed: {}".format(key))

    if summary.get("frozen_probe_timestep_records") != worker.get("frozen_probe_timestep_records"):
        raise StageHAuditError("Stage-G worker/summary timestep records differ")
    if summary.get("training_diagnostics") != worker.get("training_diagnostics"):
        raise StageHAuditError("Stage-G worker/summary training diagnostics differ")
    if summary.get("procedure_gate_totals") != worker.get("procedure_gate_totals"):
        raise StageHAuditError("Stage-G worker/summary gate totals differ")
    protocol = summary.get("durable_evidence_protocol")
    if not isinstance(protocol, Mapping):
        raise StageHAuditError("Stage-G durable evidence protocol missing")
    if protocol.get("worker_result_sha256") != worker.get("worker_result_sha256"):
        raise StageHAuditError("Stage-G summary references a different worker result")
    if protocol.get("worker_file_sha256") != sha256_file(worker_path):
        raise StageHAuditError("Stage-G worker file SHA reference changed")
    if protocol.get("probe_file_sha256") != sha256_file(probe_path):
        raise StageHAuditError("Stage-G probe file SHA reference changed")
    return probe, worker, summary


def _finite(record: Mapping[str, Any], key: str) -> float:
    value = float(record[key])
    if not math.isfinite(value):
        raise StageHAuditError("non-finite metric: {}".format(key))
    return value


def _by_timestep(mapping: Mapping[str, Any], timestep: int) -> Mapping[str, Any]:
    value = mapping.get(str(int(timestep)))
    if value is None:
        value = mapping.get(int(timestep))  # type: ignore[arg-type]
    if not isinstance(value, Mapping):
        raise StageHAuditError("missing timestep record: {}".format(timestep))
    return value


def _safe_ratio(numerator: float, denominator: float) -> Optional[float]:
    if abs(float(denominator)) <= 1.0e-15:
        return None
    return float(numerator) / float(denominator)


def _verify_constant_brier(
    *, probe_prevalence: float, training_prevalence: float, observed: float
) -> float:
    expected = (
        probe_prevalence * (1.0 - training_prevalence) ** 2
        + (1.0 - probe_prevalence) * training_prevalence**2
    )
    if abs(expected - observed) > FLOAT_TOLERANCE:
        raise StageHAuditError(
            "persisted constant Brier is inconsistent with prevalence values"
        )
    return expected


def _timestep_audit(
    timestep: int,
    training: Mapping[str, Any],
    timestep_record: Mapping[str, Any],
) -> Mapping[str, Any]:
    record = timestep_record.get("record")
    baseline = timestep_record.get("no_abstention_baseline")
    eligibility = timestep_record.get("eligibility")
    gate_counts = timestep_record.get("gate_counts")
    if not all(isinstance(item, Mapping) for item in (record, baseline, eligibility, gate_counts)):
        raise StageHAuditError("Stage-G timestep payload is incomplete: {}".format(timestep))
    record = record  # type: ignore[assignment]
    baseline = baseline  # type: ignore[assignment]
    eligibility = eligibility  # type: ignore[assignment]
    gate_counts = gate_counts  # type: ignore[assignment]

    row_count = int(record["row_count"])
    raw_rows = int(record["raw_accepted_row_count"])
    selected_rows = int(record["selected_row_count"])
    adverse_rows = int(record["adverse_row_count"])
    if row_count != EXPECTED_FROZEN_PROBE_ROWS:
        raise StageHAuditError("frozen-probe row population changed")
    if raw_rows == 0:
        raise StageHAuditError("no raw candidate rows are available for Brier audit")
    if not (0 <= adverse_rows <= raw_rows <= row_count):
        raise StageHAuditError("frozen-probe row counts are inconsistent")
    if not (0 <= selected_rows <= raw_rows):
        raise StageHAuditError("risk-selected row count exceeds raw candidate rows")

    training_prevalence = _finite(training, "risk_prevalence")
    objective_adverse_rate = _finite(training, "adverse_label_rate")
    probe_prevalence = adverse_rows / float(raw_rows)
    probe_optimal_constant_brier = probe_prevalence * (1.0 - probe_prevalence)
    observed_training_constant_brier = _finite(record, "risk_constant_brier_score")
    recomputed_training_constant_brier = _verify_constant_brier(
        probe_prevalence=probe_prevalence,
        training_prevalence=training_prevalence,
        observed=observed_training_constant_brier,
    )
    model_brier = _finite(record, "risk_brier_score")
    prevalence_shift = probe_prevalence - training_prevalence
    prevalence_shift_penalty = prevalence_shift**2
    if abs(
        (recomputed_training_constant_brier - probe_optimal_constant_brier)
        - prevalence_shift_penalty
    ) > FLOAT_TOLERANCE:
        raise StageHAuditError("Brier prevalence-shift identity changed")

    model_excess_over_training_constant = model_brier - observed_training_constant_brier
    model_excess_over_probe_optimal = model_brier - probe_optimal_constant_brier
    failed_checks = sorted(
        key for key, value in dict(eligibility.get("checks", {})).items() if value is not True
    )
    geometry_pass = all(int(value) == 0 for value in gate_counts.values())

    baseline_adverse = _finite(baseline, "adverse_sse_mass")
    record_adverse = _finite(record, "adverse_sse_mass")
    baseline_beneficial = _finite(baseline, "beneficial_sse_mass")
    record_beneficial = _finite(record, "beneficial_sse_mass")
    baseline_cvar = _finite(baseline["group_tail"], "worst_fraction_cvar_mse_ratio")
    record_cvar = _finite(record["group_tail"], "worst_fraction_cvar_mse_ratio")

    return {
        "timestep": int(timestep),
        "risk_path": timestep_record.get("risk_path"),
        "risk_l2": float(timestep_record.get("risk_l2")),
        "training": {
            "objective_adverse_label_rate": objective_adverse_rate,
            "risk_fit_row_count": int(training["risk_fit_row_count"]),
            "risk_model_prevalence": training_prevalence,
        },
        "frozen_probe_raw_candidate_population": {
            "row_count": row_count,
            "raw_candidate_row_count": raw_rows,
            "raw_candidate_adverse_row_count": adverse_rows,
            "raw_candidate_admission_rate": raw_rows / float(row_count),
            "raw_candidate_adverse_prevalence": probe_prevalence,
        },
        "prevalence_transfer": {
            "signed_probe_minus_training": prevalence_shift,
            "absolute_gap": abs(prevalence_shift),
            "brier_penalty_from_prevalence_shift": prevalence_shift_penalty,
            "probe_optimal_constant_brier": probe_optimal_constant_brier,
            "training_prevalence_constant_brier": observed_training_constant_brier,
            "constant_brier_recomputed": recomputed_training_constant_brier,
        },
        "probability_transfer": {
            "model_brier": model_brier,
            "training_prevalence_constant_brier": observed_training_constant_brier,
            "model_brier_excess_over_training_constant": model_excess_over_training_constant,
            "model_brier_excess_over_probe_optimal_constant": model_excess_over_probe_optimal,
            "brier_skill_vs_training_constant": (
                1.0 - model_brier / observed_training_constant_brier
                if observed_training_constant_brier > 0.0
                else None
            ),
            "training_constant_beaten": bool(model_brier <= observed_training_constant_brier + 1.0e-12),
            "base_rate_shift_alone_explains_model_failure": bool(
                model_excess_over_training_constant <= FLOAT_TOLERANCE
            ),
        },
        "abstention": {
            "selected_row_count": selected_rows,
            "acceptance_rate": _finite(record, "acceptance_rate"),
            "conditional_keep_rate_given_raw_candidate": selected_rows / float(raw_rows),
            "conditional_rejection_rate_given_raw_candidate": 1.0 - selected_rows / float(raw_rows),
        },
        "candidate_and_policy_effect": {
            "no_abstention_overall_mse_ratio": _finite(baseline, "overall_mse_ratio"),
            "risk_policy_overall_mse_ratio": _finite(record, "overall_mse_ratio"),
            "overall_mse_gain_from_abstention": (
                _finite(baseline, "overall_mse_ratio") - _finite(record, "overall_mse_ratio")
            ),
            "no_abstention_positive_reduction_rate": _finite(
                baseline, "positive_distance_reduction_rate"
            ),
            "risk_policy_positive_reduction_rate": _finite(
                record, "positive_distance_reduction_rate"
            ),
            "adverse_sse_reduction_fraction": (
                1.0 - record_adverse / baseline_adverse if baseline_adverse > 0.0 else None
            ),
            "beneficial_sse_retention_fraction": _safe_ratio(
                record_beneficial, baseline_beneficial
            ),
            "net_sse_gain_from_abstention": (
                _finite(record, "net_sse_reduction_mass")
                - _finite(baseline, "net_sse_reduction_mass")
            ),
            "group_cvar_gain_from_abstention": baseline_cvar - record_cvar,
            "raw_candidate_mean_improvement": bool(
                _finite(baseline, "overall_mse_ratio") < 1.0
            ),
        },
        "eligibility": {
            "pass": eligibility.get("pass") is True,
            "failed_checks": failed_checks,
        },
        "geometry": {
            "gate_counts": copy.deepcopy(dict(gate_counts)),
            "all_aligned_gate_counts_zero": geometry_pass,
        },
        "persisted_identity_only": {
            "descriptor_sha256": timestep_record.get("descriptor_sha256"),
            "risk_probability_sha256": timestep_record.get("risk_probability_sha256"),
            "candidate_sha256": timestep_record.get("candidate_sha256"),
            "selected_scale_sha256": timestep_record.get("selected_scale_sha256"),
        },
    }


def build_audit(
    probe: Mapping[str, Any],
    worker: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> Mapping[str, Any]:
    training = worker.get("training_diagnostics")
    records = worker.get("frozen_probe_timestep_records")
    if not isinstance(training, Mapping) or not isinstance(records, Mapping):
        raise StageHAuditError("Stage-G training or frozen-probe records are missing")
    timestep_audits: Dict[str, Mapping[str, Any]] = {}
    for timestep in LOCKED_TIMESTEPS:
        timestep_audits[str(timestep)] = _timestep_audit(
            timestep,
            _by_timestep(training, timestep),
            _by_timestep(records, timestep),
        )

    all_geometry_pass = all(
        item["geometry"]["all_aligned_gate_counts_zero"] is True
        for item in timestep_audits.values()
    )
    all_brier_fail = all(
        item["probability_transfer"]["training_constant_beaten"] is False
        for item in timestep_audits.values()
    )
    all_incremental_excess_positive = all(
        float(item["probability_transfer"]["model_brier_excess_over_training_constant"])
        > FLOAT_TOLERANCE
        for item in timestep_audits.values()
    )
    all_raw_candidates_improve_mean = all(
        item["candidate_and_policy_effect"]["raw_candidate_mean_improvement"] is True
        for item in timestep_audits.values()
    )
    any_acceptance_fail = any(
        "acceptance" in item["eligibility"]["failed_checks"]
        for item in timestep_audits.values()
    )

    if not all_geometry_pass:
        classification = "frozen_probe_transfer_includes_geometry_gate_regression"
        root_cause = "phase314b_r259_stageh_frozen_probe_transfer_failure_includes_geometry_regression"
        next_path = "AUDIT_R259_FROZEN_PROBE_GEOMETRY_TRANSFER_FAILURE_WITHOUT_REACCESS"
    elif all_brier_fail and all_incremental_excess_positive and all_raw_candidates_improve_mean:
        classification = (
            "risk_probability_transfer_failure_beyond_base_rate_shift_with_"
            "raw_candidate_mean_improvement_preserved"
        )
        root_cause = (
            "phase314b_r259_stageh_risk_probability_transfer_fails_beyond_"
            "prevalence_shift_while_raw_candidates_preserve_mean_improvement"
        )
        next_path = (
            "DESIGN_R259_RISK_TRANSFER_REPAIR_ON_OBJECTIVE_TRAIN_WITH_FRESH_"
            "UNTOUCHED_EVALUATION_SET"
        )
    elif all_brier_fail and all_incremental_excess_positive:
        classification = "risk_probability_transfer_failure_beyond_base_rate_shift_with_mixed_candidate_transfer"
        root_cause = "phase314b_r259_stageh_risk_probability_transfer_fails_beyond_prevalence_shift_with_mixed_candidate_transfer"
        next_path = (
            "DESIGN_R259_JOINT_CANDIDATE_AND_RISK_TRANSFER_RESEARCH_WITH_FRESH_"
            "UNTOUCHED_EVALUATION_SET"
        )
    elif all_brier_fail:
        classification = "risk_probability_transfer_failure_consistent_with_prevalence_shift_but_not_fully_attributable"
        root_cause = "phase314b_r259_stageh_risk_probability_transfer_failure_requires_fresh_data_attribution"
        next_path = (
            "DESIGN_R259_RISK_TRANSFER_REPAIR_ON_OBJECTIVE_TRAIN_WITH_FRESH_"
            "UNTOUCHED_EVALUATION_SET"
        )
    else:
        classification = "mixed_frozen_probe_policy_transfer_failure"
        root_cause = "phase314b_r259_stageh_frozen_probe_policy_transfer_failure_is_mixed_across_timesteps"
        next_path = (
            "DESIGN_R259_MIXED_TRANSFER_RESEARCH_WITH_FRESH_UNTOUCHED_EVALUATION_SET"
        )

    return {
        "classification": classification,
        "root_cause": root_cause,
        "required_next_path": next_path,
        "joint_findings": {
            "all_aligned_geometry_gate_counts_zero": all_geometry_pass,
            "all_timesteps_risk_brier_worse_than_training_prevalence_constant": all_brier_fail,
            "all_timesteps_model_brier_has_incremental_excess_beyond_base_rate_shift": all_incremental_excess_positive,
            "all_timesteps_raw_candidate_mean_improvement_preserved": all_raw_candidates_improve_mean,
            "any_timestep_acceptance_gate_failed": any_acceptance_fail,
            "canonical_joint_policy_passed": summary.get("all_timesteps_pass") is True,
        },
        "timestep_audits": timestep_audits,
        "source_evidence_limitations": {
            "row_level_risk_probabilities_persisted": False,
            "row_level_adverse_labels_persisted": False,
            "row_level_descriptors_persisted": False,
            "row_level_accept_masks_persisted": False,
            "auroc_computable": False,
            "average_precision_computable": False,
            "ece_computable": False,
            "reliability_bins_computable": False,
            "risk_score_distribution_shift_computable": False,
            "descriptor_distribution_shift_computable": False,
            "accept_mask_hamming_attribution_computable": False,
            "calibration_vs_ranking_failure_separable": False,
            "reason": (
                "Stage G persisted bounded aggregate metrics and array identities, "
                "not row-level probabilities, labels, descriptors or masks"
            ),
        },
        "governance_conclusion": {
            "frozen_probe_access_consumed": True,
            "frozen_probe_reaccess_authorized": False,
            "retuning_against_frozen_probe_authorized": False,
            "recipe_fallback_authorized": False,
            "timestep_cherry_pick_authorized": False,
            "future_final_evaluation_requires_fresh_untouched_data": True,
        },
        "environment_not_primary_failure": {
            "portable_contract_passed": probe["portable_environment_audit"][
                "portable_contract_passed"
            ],
            "required_operation_passed": probe["required_operation_dry_run"]["pass"],
            "probe_did_not_access_frozen_probe": probe["frozen_probe_accessed"] is False,
        },
    }


def build_summary(
    repository: Mapping[str, Any],
    probe: Mapping[str, Any],
    worker: Mapping[str, Any],
    source_summary: Mapping[str, Any],
) -> Mapping[str, Any]:
    audit = build_audit(probe, worker, source_summary)
    payload: Dict[str, Any] = {
        "phase": PHASE,
        "schema": SCHEMA,
        "execution_verdict": "PASS",
        "scientific_status": "BLOCKED",
        "root_cause": audit["root_cause"],
        "required_next_path": audit["required_next_path"],
        "primary_failure_locus": "frozen_probe_risk_transfer_failure_audit",
        "repository": copy.deepcopy(dict(repository)),
        "source_evidence": {
            "probe_path": SOURCE_PROBE,
            "probe_file_sha256": sha256_file(Path(repository["root"]) / SOURCE_PROBE),
            "probe_self_sha256": probe["probe_evidence_sha256"],
            "worker_path": SOURCE_WORKER,
            "worker_file_sha256": sha256_file(Path(repository["root"]) / SOURCE_WORKER),
            "worker_result_sha256": worker["worker_result_sha256"],
            "summary_path": SOURCE_SUMMARY,
            "summary_file_sha256": sha256_file(Path(repository["root"]) / SOURCE_SUMMARY),
            "summary_self_sha256": source_summary["summary_sha256"],
        },
        "audit": audit,
        "execution_counts": {
            "report_read_count": 3,
            "environment_probe_count": 0,
            "science_worker_count": 0,
            "cuda_operation_count": 0,
            "dataset_load_count": 0,
            "direction_fit_count": 0,
            "risk_fit_count": 0,
            "candidate_generation_count": 0,
            "recipe_evaluation_count": 0,
            "frozen_probe_evaluation_count": 0,
        },
        "selected_configuration": None,
        "train_only_recommendation": None,
        "selection_holdout_evaluation_count_added": 0,
        "cumulative_selection_holdout_evaluation_count": 1,
        "frozen_probe_evaluation_count_added": 0,
        "cumulative_frozen_probe_evaluation_count": 1,
        "frozen_probe_accessed_historically": True,
        "frozen_probe_reaccessed": False,
        "rerun_authorized": False,
        **{key: False for key in FALSE_BOUNDARIES},
    }
    payload["summary_sha256"] = sha256_bytes(stable_json_bytes(payload))
    validate_summary(payload)
    return payload


def validate_summary(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != SCHEMA or payload.get("execution_verdict") != "PASS":
        raise StageHAuditError("Stage-H summary schema/verdict changed")
    if payload.get("scientific_status") != "BLOCKED":
        raise StageHAuditError("Stage-H audit must remain scientifically BLOCKED")
    audit = payload.get("audit")
    if not isinstance(audit, Mapping):
        raise StageHAuditError("Stage-H audit payload is missing")
    if payload.get("root_cause") != audit.get("root_cause"):
        raise StageHAuditError("Stage-H root cause differs from audit")
    if payload.get("required_next_path") != audit.get("required_next_path"):
        raise StageHAuditError("Stage-H next path differs from audit")
    records = audit.get("timestep_audits")
    if not isinstance(records, Mapping) or sorted(records) != ["10", "25", "50"]:
        raise StageHAuditError("Stage-H timestep audit population changed")
    if payload.get("selected_configuration") is not None:
        raise StageHAuditError("Stage-H selected a configuration")
    if payload.get("train_only_recommendation") is not None:
        raise StageHAuditError("Stage-H retained a recommendation")
    if payload.get("selection_holdout_evaluation_count_added") != 0:
        raise StageHAuditError("Stage-H re-accessed selection holdout")
    if payload.get("cumulative_selection_holdout_evaluation_count") != 1:
        raise StageHAuditError("Stage-H selection-holdout count changed")
    if payload.get("frozen_probe_evaluation_count_added") != 0:
        raise StageHAuditError("Stage-H re-accessed frozen probe")
    if payload.get("cumulative_frozen_probe_evaluation_count") != 1:
        raise StageHAuditError("Stage-H frozen-probe count changed")
    if payload.get("frozen_probe_accessed_historically") is not True:
        raise StageHAuditError("Stage-H lost historical frozen-probe access")
    if payload.get("frozen_probe_reaccessed") is not False:
        raise StageHAuditError("Stage-H recorded frozen-probe reaccess")
    if payload.get("rerun_authorized") is not False:
        raise StageHAuditError("Stage-H authorized rerun")
    if any(payload.get(key) is not False for key in FALSE_BOUNDARIES):
        raise StageHAuditError("Stage-H crossed a forbidden boundary")
    expected_counts = {
        "report_read_count": 3,
        "environment_probe_count": 0,
        "science_worker_count": 0,
        "cuda_operation_count": 0,
        "dataset_load_count": 0,
        "direction_fit_count": 0,
        "risk_fit_count": 0,
        "candidate_generation_count": 0,
        "recipe_evaluation_count": 0,
        "frozen_probe_evaluation_count": 0,
    }
    if payload.get("execution_counts") != expected_counts:
        raise StageHAuditError("Stage-H execution counts changed")
    expected = sha256_bytes(
        stable_json_bytes(
            {key: value for key, value in payload.items() if key != "summary_sha256"}
        )
    )
    if payload.get("summary_sha256") != expected:
        raise StageHAuditError("Stage-H summary self-hash changed")


def blocked_report(
    repository: Optional[Mapping[str, Any]], error: BaseException
) -> Mapping[str, Any]:
    return {
        "phase": PHASE,
        "schema": BLOCKED_SCHEMA,
        "execution_verdict": "BLOCKED",
        "scientific_status": "BLOCKED",
        "root_cause": "phase314b_r259_stageh_read_only_transfer_audit_execution_failed",
        "required_next_path": "AUDIT_R259_STAGEH_EXECUTION_FAILURE_WITHOUT_FROZEN_PROBE_REACCESS",
        "primary_failure_locus": "stageh_read_only_audit_execution_contract",
        "repository": None if repository is None else dict(repository),
        "error_type": type(error).__name__,
        "error_message": str(error),
        "selected_configuration": None,
        "train_only_recommendation": None,
        "selection_holdout_evaluation_count_added": 0,
        "cumulative_selection_holdout_evaluation_count": 1,
        "frozen_probe_evaluation_count_added": 0,
        "cumulative_frozen_probe_evaluation_count": 1,
        "frozen_probe_accessed_historically": True,
        "frozen_probe_reaccessed": False,
        "rerun_authorized": False,
        **{key: False for key in FALSE_BOUNDARIES},
    }
