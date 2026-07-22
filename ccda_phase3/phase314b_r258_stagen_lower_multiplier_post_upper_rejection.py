"""Phase3.14b-r2.5.8 Stage N lower-multiplier post-upper rejection audit.

Stage M established a single external-multiplier upper-segment boundary at 0.5:
all 27 backbone × timestep records identified ``upper_segment_geometry`` at
multiplier 0.5, while no other multiplier carried that discriminator.  The
immutable Stage-M report also persists complete per-predicate conditional-pass
and rejection-mass comparisons for every external multiplier.  Stage N therefore
performs a report-only audit of the preregistered lower multiplier 0.25.

For each of the 27 records Stage N verifies local oracle admission, OOF rejection,
and clearance of the upper-segment gate at 0.25.  It then scans the frozen
predicate order strictly after ``upper_segment_geometry`` and identifies the
first locally discriminative post-upper predicate.  Results are aggregated by
predicate, timestep, and backbone.  No Stage-L science, callback pair, model,
tensor, candidate, holdout, frozen probe, CUDA operation, or DeformableRavens
execution is repeated.
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


PHASE = "Phase3.14b-r2.5.8 Stage N"
SCHEMA = "phase314b_r258_stagen_lower_multiplier_post_upper_rejection_v1"

BASE_EVIDENCE_COMMIT = "2803d0131f4bc398abdc340f48cee8bdee769ae4"
BASE_IMPLEMENTATION_COMMIT = "d888241a581fb53ae006cfd15e58f1d68dd40078"
EXPECTED_REMOTE = "6758ea7ad800667a436b0243d3b1f6c63256d854"
EXPECTED_SUBMODULE = "633a88752445cf5d6776ed374fdbbdb35f93050c"

STAGEM_REPORT = (
    "reports/phase3_14b_r258_stagem_external_multiplier_predicate_stratification_summary.json"
)
EXPECTED_STAGEM_REPORT_SHA256 = (
    "8588fc5aa480bbaffb4ee3327a407d8751dddf048d8588fc5971a0e9540d7cb4"
)
EXPECTED_STAGEM_ROOT_CAUSE = (
    "phase314b_r258_stagem_upper_segment_geometry_is_single_multiplier_boundary"
)
EXPECTED_STAGEM_NEXT_PATH = "AUDIT_LOWER_MULTIPLIER_REJECTION_AFTER_UPPER_SEGMENT_GATE"

SUCCESS_REPORT = (
    "reports/phase3_14b_r258_stagen_lower_multiplier_post_upper_rejection_summary.json"
)
BLOCKED_REPORT = (
    "reports/phase3_14b_r258_stagen_lower_multiplier_post_upper_rejection_blocked_summary.json"
)

IMPLEMENTATION_SUBJECT = (
    "Phase3.14b-r2.5.8 Stage N: audit lower-multiplier post-upper rejection"
)
EVIDENCE_SUBJECT = (
    "Record Phase3.14b-r2.5.8 Stage N lower-multiplier rejection evidence"
)
BLOCKED_EVIDENCE_SUBJECT = "Record Phase3.14b-r2.5.8 Stage N blocked evidence"
IMPLEMENTATION_PATHS: Tuple[Tuple[str, str], ...] = (
    (
        "A",
        "ccda_phase3/phase314b_r258_stagen_lower_multiplier_post_upper_rejection.py",
    ),
    ("A", "scripts/phase3_14b_r258_stagen_finalize.py"),
    (
        "A",
        "tests/test_phase3_14b_r258_stagen_lower_multiplier_post_upper_rejection.py",
    ),
)

EXPECTED_RECORD_COUNT = 27
EXPECTED_STRATUM_COUNT = 108
EXPECTED_TIMESTEPS: Tuple[int, ...] = (10, 25, 50)
EXPECTED_BACKBONE_COUNT = 9
EXPECTED_EXTERNAL_MULTIPLIERS: Tuple[float, ...] = (0.25, 0.5, 1.0, 2.0)
LOWER_MULTIPLIER = 0.25
UPPER_BOUNDARY_MULTIPLIER = 0.5
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
POST_UPPER_PREDICATES: Tuple[str, ...] = PREDICATE_ORDER[
    PREDICATE_ORDER.index("upper_segment_geometry") + 1 :
]

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

PREDICATE_NEXT_PATH: Mapping[str, Tuple[str, str]] = {
    "lower_segment_geometry": (
        "phase314b_r258_stagen_lower_segment_geometry_is_post_upper_boundary",
        "AUDIT_LOWER_SEGMENT_GEOMETRY_AT_LOWER_EXTERNAL_MULTIPLIER",
    ),
    "coordinate_recenter": (
        "phase314b_r258_stagen_coordinate_recenter_is_post_upper_boundary",
        "AUDIT_LOCAL_FRAME_RECENTERING_AT_LOWER_EXTERNAL_MULTIPLIER",
    ),
    "coordinate_geometry": (
        "phase314b_r258_stagen_coordinate_geometry_is_post_upper_boundary",
        "AUDIT_LOCAL_FRAME_COMPATIBILITY_AT_LOWER_EXTERNAL_MULTIPLIER",
    ),
    "reconstruction_bounds": (
        "phase314b_r258_stagen_reconstruction_bounds_are_post_upper_boundary",
        "AUDIT_SEGMENT_RECONSTRUCTION_BOUNDS_AT_LOWER_EXTERNAL_MULTIPLIER",
    ),
    "segment_geometry": (
        "phase314b_r258_stagen_segment_geometry_is_post_upper_boundary",
        "AUDIT_SEGMENT_VECTOR_COMPATIBILITY_AT_LOWER_EXTERNAL_MULTIPLIER",
    ),
    "direction_retention": (
        "phase314b_r258_stagen_direction_retention_is_post_upper_boundary",
        "CALIBRATE_INTEGRATOR_COMPATIBLE_DIRECTION_BASIS_ON_OBJECTIVE_TRAIN_ONLY",
    ),
    "displacement": (
        "phase314b_r258_stagen_displacement_is_post_upper_boundary",
        "AUDIT_MINIMUM_DISPLACEMENT_AT_LOWER_EXTERNAL_MULTIPLIER",
    ),
    "topology": (
        "phase314b_r258_stagen_topology_is_post_upper_boundary",
        "CALIBRATE_TOPOLOGY_COMPATIBLE_DIRECTION_SURROGATE_ON_OBJECTIVE_TRAIN_ONLY",
    ),
}


class StageNError(RuntimeError):
    """Fail-closed Stage-N validation error."""


@dataclass(frozen=True)
class StageNSpec:
    oracle_local_acceptance_min: float = 0.95
    oof_local_acceptance_max: float = 0.05
    upper_oof_conditional_pass_min: float = 0.90
    upper_oof_rejection_mass_max: float = 0.10
    upper_oracle_conditional_pass_min: float = 0.90
    upper_oracle_rejection_mass_max: float = 0.10
    stable_record_support_min: int = 25
    sparse_other_predicate_support_max: int = 2
    timestep_support_min: int = 8
    tolerance: float = 1.0e-12

    def validate(self) -> None:
        rates = (
            self.oracle_local_acceptance_min,
            self.oof_local_acceptance_max,
            self.upper_oof_conditional_pass_min,
            self.upper_oof_rejection_mass_max,
            self.upper_oracle_conditional_pass_min,
            self.upper_oracle_rejection_mass_max,
        )
        if any(not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0 for value in rates):
            raise StageNError("Stage-N rate threshold is outside [0, 1]")
        if self.stable_record_support_min != 25:
            raise StageNError("Stage-N stable support changed")
        if self.sparse_other_predicate_support_max != 2:
            raise StageNError("Stage-N sparse support changed")
        if self.timestep_support_min != 8:
            raise StageNError("Stage-N timestep support changed")
        if self.tolerance < 0.0:
            raise StageNError("Stage-N tolerance is negative")


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


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise StageNError(f"JSON root is not a mapping: {path}")
    return value


def write_once(path: Path, data: bytes) -> None:
    output = Path(path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite write-once output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"temporary output already exists: {temporary}")
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StageNError(f"{label} is not a mapping")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise StageNError(f"{label} is not a sequence")
    return value


def _finite_rate(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0 or result > 1.0:
        raise StageNError(f"{label} is not a finite rate")
    return result


def _nonnegative_int(value: Any, label: str) -> int:
    result = int(value)
    if result < 0 or result != value:
        raise StageNError(f"{label} is not a nonnegative integer")
    return result


def _scale_key(value: float) -> str:
    return format(float(value), ".12g")


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=str(root),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


def _commit_paths(root: Path, commit: str) -> Tuple[Tuple[str, str], ...]:
    completed = subprocess.run(
        ["git", "diff-tree", "--no-commit-id", "--name-status", "-r", commit],
        cwd=str(root),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    output: List[Tuple[str, str]] = []
    for line in completed.stdout.splitlines():
        fields = line.split("\t")
        if len(fields) != 2:
            raise StageNError(f"unexpected diff-tree row: {line!r}")
        output.append((fields[0], fields[1]))
    return tuple(sorted(output))


def validate_repository_for_implementation(root: Path) -> Mapping[str, Any]:
    repo = Path(root).resolve()
    if _git(repo, "branch", "--show-current") != "Experiment1":
        raise StageNError("Stage N requires Experiment1")
    if _git(repo, "rev-parse", "HEAD") != BASE_EVIDENCE_COMMIT:
        raise StageNError("Stage-N base HEAD changed")
    if _git(repo, "rev-parse", "origin/Experiment1") != EXPECTED_REMOTE:
        raise StageNError("origin/Experiment1 changed")
    if _git(repo, "status", "--porcelain", "--untracked-files=all"):
        raise StageNError("main repository is not clean before Stage N")
    submodule = repo / "external/deformable-ravens"
    if _git(submodule, "rev-parse", "HEAD") != EXPECTED_SUBMODULE:
        raise StageNError("DeformableRavens commit changed")
    if _git(submodule, "status", "--porcelain", "--untracked-files=all"):
        raise StageNError("DeformableRavens worktree is dirty")
    report = repo / STAGEM_REPORT
    if not report.is_file() or sha256_file(report) != EXPECTED_STAGEM_REPORT_SHA256:
        raise StageNError("Stage-M report binding changed")
    for relative in (SUCCESS_REPORT, BLOCKED_REPORT):
        if (repo / relative).exists():
            raise StageNError(f"Stage-N output already exists: {relative}")
    return {
        "branch": "Experiment1",
        "base_evidence_commit": BASE_EVIDENCE_COMMIT,
        "base_implementation_commit": BASE_IMPLEMENTATION_COMMIT,
        "origin_experiment1": EXPECTED_REMOTE,
        "submodule_commit": EXPECTED_SUBMODULE,
        "stage_m_report": STAGEM_REPORT,
        "stage_m_report_sha256": EXPECTED_STAGEM_REPORT_SHA256,
    }


def validate_implementation_commit(root: Path) -> Mapping[str, Any]:
    repo = Path(root).resolve()
    head = _git(repo, "rev-parse", "HEAD")
    if _git(repo, "rev-parse", f"{head}^") != BASE_EVIDENCE_COMMIT:
        raise StageNError("Stage-N implementation parent changed")
    if _git(repo, "show", "-s", "--format=%s", head) != IMPLEMENTATION_SUBJECT:
        raise StageNError("Stage-N implementation subject changed")
    if _commit_paths(repo, head) != tuple(sorted(IMPLEMENTATION_PATHS)):
        raise StageNError("Stage-N implementation paths changed")
    if _git(repo, "rev-parse", "origin/Experiment1") != EXPECTED_REMOTE:
        raise StageNError("origin/Experiment1 changed")
    status = _git(repo, "status", "--porcelain", "--untracked-files=all")
    if status:
        raise StageNError(f"unexpected Stage-N worktree state: {status}")
    submodule = repo / "external/deformable-ravens"
    if _git(submodule, "rev-parse", "HEAD") != EXPECTED_SUBMODULE:
        raise StageNError("DeformableRavens commit changed")
    if _git(submodule, "status", "--porcelain", "--untracked-files=all"):
        raise StageNError("DeformableRavens worktree is dirty")
    report = repo / STAGEM_REPORT
    if sha256_file(report) != EXPECTED_STAGEM_REPORT_SHA256:
        raise StageNError("Stage-M report changed after implementation commit")
    return {
        "implementation_commit": head,
        "implementation_parent": BASE_EVIDENCE_COMMIT,
        "implementation_subject": IMPLEMENTATION_SUBJECT,
        "implementation_paths": [list(value) for value in IMPLEMENTATION_PATHS],
        "origin_experiment1": EXPECTED_REMOTE,
        "submodule_commit": EXPECTED_SUBMODULE,
    }


def validate_stage_m_report(report: Mapping[str, Any]) -> Mapping[str, Any]:
    if report.get("execution_verdict") != "PASS":
        raise StageNError("Stage-M execution is not PASS")
    if report.get("scientific_status") != "BLOCKED":
        raise StageNError("Stage-M scientific status changed")
    if report.get("root_cause") != EXPECTED_STAGEM_ROOT_CAUSE:
        raise StageNError("Stage-M root cause changed")
    if report.get("required_next_path") != EXPECTED_STAGEM_NEXT_PATH:
        raise StageNError("Stage-M next path changed")
    if report.get("selected_configuration") is not None:
        raise StageNError("Stage-M selected configuration changed")
    if report.get("train_only_recommendation") is not None:
        raise StageNError("Stage-M recommendation changed")
    classification = _mapping(report.get("classification"), "classification")
    if int(classification.get("record_count", -1)) != EXPECTED_RECORD_COUNT:
        raise StageNError("Stage-M record count changed")
    if int(classification.get("stratum_count", -1)) != EXPECTED_STRATUM_COUNT:
        raise StageNError("Stage-M stratum count changed")
    if float(classification.get("dominant_upper_multiplier")) != UPPER_BOUNDARY_MULTIPLIER:
        raise StageNError("Stage-M dominant upper multiplier changed")
    if int(classification.get("dominant_upper_support_count", -1)) != EXPECTED_RECORD_COUNT:
        raise StageNError("Stage-M dominant upper support changed")
    if int(classification.get("maximum_other_multiplier_upper_support_count", -1)) != 0:
        raise StageNError("Stage-M non-dominant upper support changed")
    if int(classification.get("upper_failure_rate_monotone_record_count", -1)) != EXPECTED_RECORD_COUNT:
        raise StageNError("Stage-M upper failure monotonicity changed")
    if int(classification.get("upper_rejection_mass_monotone_record_count", -1)) != EXPECTED_RECORD_COUNT:
        raise StageNError("Stage-M upper mass monotonicity changed")
    immutable = _mapping(report.get("immutable_inputs"), "immutable_inputs")
    for key in ("callback_pairs_rerun", "stage_l_science_rerun", "historical_tests_rerun"):
        if immutable.get(key) is not False:
            raise StageNError(f"Stage-M {key} is not false")
    mechanism = _mapping(report.get("mechanism_boundary"), "mechanism_boundary")
    if mechanism.get("report_only_stratification") is not True:
        raise StageNError("Stage-M is not report-only")
    for key in (
        "stagee_modified",
        "stagek_modified",
        "stagel_modified",
        "callback_schema_modified",
        "integrator_thresholds_changed",
        "integrator_scale_search_changed",
        "external_scale_bank_changed",
        "direction_model_refit",
        "new_ranker_fitted",
    ):
        if mechanism.get(key) is not False:
            raise StageNError(f"Stage-M mechanism boundary changed: {key}")
    for key in FALSE_BOUNDARIES:
        if report.get(key) is not False:
            raise StageNError(f"Stage-M forbidden boundary changed: {key}")
    records = _sequence(report.get("record_strata"), "record_strata")
    if len(records) != EXPECTED_RECORD_COUNT:
        raise StageNError("Stage-M record population changed")
    return classification


def _strata_map(record: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    values = _sequence(record.get("strata"), "record.strata")
    output: Dict[str, Mapping[str, Any]] = {}
    for index, value in enumerate(values):
        cell = _mapping(value, f"record.strata[{index}]")
        multiplier = float(cell.get("external_multiplier"))
        key = _scale_key(multiplier)
        if key in output:
            raise StageNError(f"duplicate multiplier in record: {key}")
        output[key] = cell
    expected = {_scale_key(value) for value in EXPECTED_EXTERNAL_MULTIPLIERS}
    if set(output) != expected:
        raise StageNError("Stage-M multiplier population changed")
    return output


def _comparison_map(cell: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    values = _sequence(cell.get("predicate_comparisons"), "predicate_comparisons")
    output: Dict[str, Mapping[str, Any]] = {}
    for index, value in enumerate(values):
        comparison = _mapping(value, f"predicate_comparisons[{index}]")
        predicate = str(comparison.get("predicate"))
        if predicate in output:
            raise StageNError(f"duplicate predicate comparison: {predicate}")
        output[predicate] = comparison
    if tuple(output) != PREDICATE_ORDER:
        raise StageNError("predicate comparison order changed")
    return output


def _upper_metrics(cell: Mapping[str, Any], source: str) -> Mapping[str, Any]:
    metrics = _mapping(cell.get(source), source)
    return {
        "conditional_failure_rate": _finite_rate(
            metrics.get("conditional_failure_rate"), f"{source}.conditional_failure_rate"
        ),
        "rejection_mass_rate": _finite_rate(
            metrics.get("rejection_mass_rate"), f"{source}.rejection_mass_rate"
        ),
        "sequential_reach_count": _nonnegative_int(
            metrics.get("sequential_reach_count"), f"{source}.sequential_reach_count"
        ),
        "first_failed_count": _nonnegative_int(
            metrics.get("first_failed_count"), f"{source}.first_failed_count"
        ),
    }


def _predicate_candidates(
    comparisons: Mapping[str, Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], Optional[str], Optional[str]]:
    candidates: List[Dict[str, Any]] = []
    first_predicate: Optional[str] = None
    first_mode: Optional[str] = None
    for predicate in POST_UPPER_PREDICATES:
        value = _mapping(comparisons.get(predicate), f"comparison.{predicate}")
        conditional = value.get("conditional_discriminator") is True
        rejection_mass = value.get("rejection_mass_discriminator") is True
        if conditional or rejection_mass:
            mode = "both" if conditional and rejection_mass else (
                "conditional" if conditional else "rejection_mass"
            )
            candidates.append(
                {
                    "predicate": predicate,
                    "mode": mode,
                    "oof_conditional_pass_rate": _finite_rate(
                        value.get("oof_conditional_pass_rate"),
                        f"{predicate}.oof_conditional_pass_rate",
                    ),
                    "oracle_conditional_pass_rate": _finite_rate(
                        value.get("oracle_conditional_pass_rate"),
                        f"{predicate}.oracle_conditional_pass_rate",
                    ),
                    "oof_rejection_mass_rate": _finite_rate(
                        value.get("oof_rejection_mass_rate"),
                        f"{predicate}.oof_rejection_mass_rate",
                    ),
                    "oracle_rejection_mass_rate": _finite_rate(
                        value.get("oracle_rejection_mass_rate"),
                        f"{predicate}.oracle_rejection_mass_rate",
                    ),
                }
            )
            if first_predicate is None:
                first_predicate = predicate
                first_mode = mode
    return candidates, first_predicate, first_mode


def build_record_audit(
    record: Mapping[str, Any], spec: Optional[StageNSpec] = None
) -> Mapping[str, Any]:
    active = StageNSpec() if spec is None else spec
    active.validate()
    base_direction_id = str(record.get("base_direction_id"))
    timestep = int(record.get("timestep"))
    if timestep not in EXPECTED_TIMESTEPS:
        raise StageNError(f"unexpected timestep: {timestep}")
    strata = _strata_map(record)
    lower = strata[_scale_key(LOWER_MULTIPLIER)]
    boundary = strata[_scale_key(UPPER_BOUNDARY_MULTIPLIER)]

    if boundary.get("local_discriminator_predicate") != "upper_segment_geometry":
        raise StageNError("0.5 upper-segment boundary changed")
    if boundary.get("locally_eligible") is not True:
        raise StageNError("0.5 upper-segment boundary is not locally eligible")

    oracle_acceptance = _finite_rate(
        lower.get("comparator_oracle_acceptance_rate"), "lower oracle acceptance"
    )
    oof_acceptance = _finite_rate(
        lower.get("oof_acceptance_rate"), "lower OOF acceptance"
    )
    oracle_admitted = oracle_acceptance >= active.oracle_local_acceptance_min
    oof_rejected = oof_acceptance <= active.oof_local_acceptance_max

    comparisons = _comparison_map(lower)
    upper_comparison = _mapping(
        comparisons["upper_segment_geometry"], "upper_segment_geometry comparison"
    )
    oof_upper_pass = _finite_rate(
        upper_comparison.get("oof_conditional_pass_rate"), "oof upper pass"
    )
    oracle_upper_pass = _finite_rate(
        upper_comparison.get("oracle_conditional_pass_rate"), "oracle upper pass"
    )
    oof_upper_mass = _finite_rate(
        upper_comparison.get("oof_rejection_mass_rate"), "oof upper mass"
    )
    oracle_upper_mass = _finite_rate(
        upper_comparison.get("oracle_rejection_mass_rate"), "oracle upper mass"
    )
    upper_cleared = (
        oof_upper_pass >= active.upper_oof_conditional_pass_min
        and oof_upper_mass <= active.upper_oof_rejection_mass_max
        and oracle_upper_pass >= active.upper_oracle_conditional_pass_min
        and oracle_upper_mass <= active.upper_oracle_rejection_mass_max
        and lower.get("local_discriminator_predicate") != "upper_segment_geometry"
    )

    candidates, first_predicate, first_mode = _predicate_candidates(comparisons)
    eligible_for_post_upper = oracle_admitted and oof_rejected and upper_cleared
    selected_predicate = first_predicate if eligible_for_post_upper else None
    selected_mode = first_mode if eligible_for_post_upper else None

    if not oracle_admitted:
        locus = "lower_multiplier_oracle_not_admitted"
    elif not oof_rejected:
        locus = "lower_multiplier_oof_not_rejected"
    elif not upper_cleared:
        locus = "lower_multiplier_does_not_clear_upper_segment_gate"
    elif selected_predicate is None:
        locus = "lower_multiplier_has_no_post_upper_discriminator"
    else:
        locus = f"lower_multiplier_post_upper::{selected_predicate}"

    return {
        "base_direction_id": base_direction_id,
        "timestep": timestep,
        "feature_mode": record.get("feature_mode"),
        "lower_multiplier": LOWER_MULTIPLIER,
        "upper_boundary_multiplier": UPPER_BOUNDARY_MULTIPLIER,
        "lower_multiplier_oracle_acceptance_rate": oracle_acceptance,
        "lower_multiplier_oof_acceptance_rate": oof_acceptance,
        "lower_multiplier_oracle_admitted": oracle_admitted,
        "lower_multiplier_oof_rejected": oof_rejected,
        "lower_multiplier_upper_gate_cleared": upper_cleared,
        "eligible_for_post_upper_audit": eligible_for_post_upper,
        "post_upper_discriminator_predicate": selected_predicate,
        "post_upper_discriminator_mode": selected_mode,
        "post_upper_discriminator_candidates": candidates,
        "locus": locus,
        "lower_upper_segment_oof": _upper_metrics(lower, "upper_segment_oof"),
        "lower_upper_segment_oracle": _upper_metrics(lower, "upper_segment_oracle"),
        "boundary_upper_segment_oof": _upper_metrics(boundary, "upper_segment_oof"),
        "boundary_upper_segment_oracle": _upper_metrics(boundary, "upper_segment_oracle"),
    }


def _increment(target: MutableMapping[str, int], key: Optional[str]) -> None:
    if isinstance(key, str):
        target[key] = target.get(key, 0) + 1


def aggregate_records(records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    predicates: Dict[str, int] = {}
    modes: Dict[str, int] = {}
    loci: Dict[str, int] = {}
    by_timestep: Dict[str, Dict[str, int]] = {str(value): {} for value in EXPECTED_TIMESTEPS}
    by_backbone: Dict[str, Dict[str, int]] = {}
    oracle_acceptance: List[float] = []
    oof_acceptance: List[float] = []
    for record in records:
        predicate = record.get("post_upper_discriminator_predicate")
        mode = record.get("post_upper_discriminator_mode")
        locus = str(record.get("locus"))
        _increment(predicates, predicate if isinstance(predicate, str) else None)
        _increment(modes, mode if isinstance(mode, str) else None)
        loci[locus] = loci.get(locus, 0) + 1
        if isinstance(predicate, str):
            timestep = str(int(record.get("timestep")))
            _increment(by_timestep[timestep], predicate)
            backbone = str(record.get("base_direction_id"))
            by_backbone.setdefault(backbone, {})
            _increment(by_backbone[backbone], predicate)
        oracle_acceptance.append(float(record["lower_multiplier_oracle_acceptance_rate"]))
        oof_acceptance.append(float(record["lower_multiplier_oof_acceptance_rate"]))
    return {
        "record_count": len(records),
        "oracle_admitted_record_count": sum(
            int(record.get("lower_multiplier_oracle_admitted") is True) for record in records
        ),
        "oof_rejected_record_count": sum(
            int(record.get("lower_multiplier_oof_rejected") is True) for record in records
        ),
        "upper_gate_cleared_record_count": sum(
            int(record.get("lower_multiplier_upper_gate_cleared") is True) for record in records
        ),
        "post_upper_eligible_record_count": sum(
            int(record.get("eligible_for_post_upper_audit") is True) for record in records
        ),
        "post_upper_discriminator_counts": dict(sorted(predicates.items())),
        "post_upper_discriminator_mode_counts": dict(sorted(modes.items())),
        "record_locus_counts": dict(sorted(loci.items())),
        "predicate_support_by_timestep": {
            key: dict(sorted(value.items())) for key, value in sorted(by_timestep.items())
        },
        "predicate_support_by_backbone": {
            key: dict(sorted(value.items())) for key, value in sorted(by_backbone.items())
        },
        "lower_oracle_acceptance_rate_mean": mean(oracle_acceptance),
        "lower_oracle_acceptance_rate_median": median(oracle_acceptance),
        "lower_oof_acceptance_rate_mean": mean(oof_acceptance),
        "lower_oof_acceptance_rate_median": median(oof_acceptance),
    }


def classify(
    records: Sequence[Mapping[str, Any]],
    aggregate: Mapping[str, Any],
    spec: Optional[StageNSpec] = None,
) -> Mapping[str, Any]:
    active = StageNSpec() if spec is None else spec
    active.validate()
    if len(records) != EXPECTED_RECORD_COUNT:
        raise StageNError("Stage-N record count changed")
    oracle_admitted = int(aggregate.get("oracle_admitted_record_count", -1))
    oof_rejected = int(aggregate.get("oof_rejected_record_count", -1))
    upper_cleared = int(aggregate.get("upper_gate_cleared_record_count", -1))
    eligible = int(aggregate.get("post_upper_eligible_record_count", -1))
    counts = {
        str(key): int(value)
        for key, value in _mapping(
            aggregate.get("post_upper_discriminator_counts"),
            "post_upper_discriminator_counts",
        ).items()
    }
    dominant_predicate: Optional[str] = None
    dominant_count = 0
    other_max = 0
    timestep_stable = False
    if counts:
        dominant_predicate = max(counts, key=lambda key: (counts[key], -PREDICATE_ORDER.index(key)))
        dominant_count = counts[dominant_predicate]
        other_max = max((value for key, value in counts.items() if key != dominant_predicate), default=0)
        by_timestep = _mapping(
            aggregate.get("predicate_support_by_timestep"), "predicate_support_by_timestep"
        )
        timestep_stable = all(
            int(_mapping(by_timestep.get(str(timestep)), f"timestep {timestep}").get(
                dominant_predicate, 0
            ))
            >= active.timestep_support_min
            for timestep in EXPECTED_TIMESTEPS
        )

    accepted_count = EXPECTED_RECORD_COUNT - oof_rejected
    if oracle_admitted < active.stable_record_support_min:
        root = "phase314b_r258_stagen_lower_multiplier_oracle_not_admitted"
        next_path = "AUDIT_ORACLE_ADMISSION_AT_LOWER_EXTERNAL_MULTIPLIER"
        primary = "lower_multiplier_oracle_admission"
    elif accepted_count >= active.stable_record_support_min:
        root = "phase314b_r258_stagen_lower_multiplier_restores_oof_acceptance"
        next_path = "LOCK_LOWER_MULTIPLIER_FOR_OBJECTIVE_TRAIN_ONLY_CONFIRMATION"
        primary = "lower_multiplier_acceptance_emergence"
    elif upper_cleared < active.stable_record_support_min:
        root = "phase314b_r258_stagen_lower_multiplier_does_not_clear_upper_segment_gate"
        next_path = "AUDIT_UPPER_GATE_TRANSITION_BETWEEN_0.25_AND_0.5"
        primary = "lower_multiplier_upper_gate_clearance"
    elif (
        dominant_predicate is not None
        and dominant_count >= active.stable_record_support_min
        and other_max <= active.sparse_other_predicate_support_max
        and timestep_stable
        and eligible >= active.stable_record_support_min
    ):
        root, next_path = PREDICATE_NEXT_PATH[dominant_predicate]
        primary = f"post_upper::{dominant_predicate}"
    elif counts:
        root = "phase314b_r258_stagen_post_upper_rejection_is_backbone_or_timestep_dependent"
        next_path = "STRATIFY_LOWER_MULTIPLIER_POST_UPPER_REJECTION_BY_BACKBONE_AND_TIMESTEP"
        primary = "heterogeneous_post_upper_rejection"
    else:
        root = "phase314b_r258_stagen_scalar_summary_has_no_post_upper_discriminator"
        next_path = "ADD_READ_ONLY_ROW_COHORT_HASHES_FOR_LOWER_MULTIPLIER_REJECTION"
        primary = "post_upper_scalar_summary_insufficient"

    return {
        "root_cause": root,
        "required_next_path": next_path,
        "primary_failure_locus": primary,
        "record_count": len(records),
        "lower_multiplier": LOWER_MULTIPLIER,
        "upper_boundary_multiplier": UPPER_BOUNDARY_MULTIPLIER,
        "oracle_admitted_record_count": oracle_admitted,
        "oof_rejected_record_count": oof_rejected,
        "oof_accepted_record_count": accepted_count,
        "upper_gate_cleared_record_count": upper_cleared,
        "post_upper_eligible_record_count": eligible,
        "dominant_post_upper_predicate": dominant_predicate,
        "dominant_post_upper_support_count": dominant_count,
        "maximum_other_post_upper_support_count": other_max,
        "dominant_predicate_has_timestep_coverage": timestep_stable,
        "post_upper_discriminator_counts": dict(sorted(counts.items())),
        "classification_thresholds": asdict(active),
    }


def build_report(
    *,
    repository: Mapping[str, Any],
    stage_m_report: Mapping[str, Any],
    spec: Optional[StageNSpec] = None,
) -> Mapping[str, Any]:
    active = StageNSpec() if spec is None else spec
    active.validate()
    validate_stage_m_report(stage_m_report)
    raw_records = _sequence(stage_m_report.get("record_strata"), "record_strata")
    records = [
        build_record_audit(_mapping(value, f"record_strata[{index}]"), active)
        for index, value in enumerate(raw_records)
    ]
    backbones = sorted({str(record["base_direction_id"]) for record in records})
    if len(backbones) != EXPECTED_BACKBONE_COUNT:
        raise StageNError("Stage-N backbone population changed")
    population = sorted((str(record["base_direction_id"]), int(record["timestep"])) for record in records)
    expected = sorted((backbone, timestep) for backbone in backbones for timestep in EXPECTED_TIMESTEPS)
    if population != expected:
        raise StageNError("Stage-N backbone × timestep population is incomplete")
    aggregate = aggregate_records(records)
    classification = classify(records, aggregate, active)
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
            "stage_m_report": STAGEM_REPORT,
            "stage_m_report_sha256": EXPECTED_STAGEM_REPORT_SHA256,
            "stage_m_evidence_commit": BASE_EVIDENCE_COMMIT,
            "stage_m_science_rerun": False,
            "stage_l_science_rerun": False,
            "callback_pairs_rerun": False,
            "historical_tests_rerun": False,
        },
        "audit_spec": asdict(active),
        "predicate_order": list(PREDICATE_ORDER),
        "post_upper_predicate_order": list(POST_UPPER_PREDICATES),
        "backbone_ids": backbones,
        "timesteps": list(EXPECTED_TIMESTEPS),
        "lower_multiplier": LOWER_MULTIPLIER,
        "upper_boundary_multiplier": UPPER_BOUNDARY_MULTIPLIER,
        "record_audits": records,
        "aggregate": aggregate,
        "classification": classification,
        "source_support_limitations": {
            "proposal_displacement_magnitude_available": False,
            "selected_internal_scale_histogram_available": False,
            "row_level_cross_multiplier_identity_available": False,
            "post_upper_analysis_uses_persisted_sequential_conditional_rates": True,
            "no_missing_value_was_inferred": True,
        },
        "mechanism_boundary": {
            "stagee_modified": False,
            "stagek_modified": False,
            "stagel_modified": False,
            "stagem_modified": False,
            "callback_schema_modified": False,
            "integrator_thresholds_changed": False,
            "integrator_scale_search_changed": False,
            "external_scale_bank_changed": False,
            "direction_model_refit": False,
            "new_ranker_fitted": False,
            "report_only_post_upper_audit": True,
        },
        **{key: False for key in FALSE_BOUNDARIES},
    }
    report["report_identity_sha256"] = sha256_bytes(stable_json_bytes(report))
    return report


def blocked_report(
    *, repository: Optional[Mapping[str, Any]], error: BaseException
) -> Mapping[str, Any]:
    return {
        "phase": PHASE,
        "schema": "phase314b_r258_stagen_blocked_v1",
        "execution_verdict": "BLOCKED",
        "scientific_status": "BLOCKED",
        "root_cause": "phase314b_r258_stagen_report_only_finalizer_failed",
        "required_next_path": "RESTORE_STAGEN_REPORT_ONLY_FINALIZER",
        "primary_failure_locus": "execution_contract",
        "selected_configuration": None,
        "train_only_recommendation": None,
        "repository": None if repository is None else dict(repository),
        "error_type": type(error).__name__,
        "error_message": str(error),
        "stage_m_science_rerun": False,
        "stage_l_science_rerun": False,
        "callback_pairs_rerun": False,
        "historical_tests_rerun": False,
        **{key: False for key in FALSE_BOUNDARIES},
    }
