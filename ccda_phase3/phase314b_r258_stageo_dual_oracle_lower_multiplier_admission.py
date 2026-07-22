"""Phase3.14b-r2.5.8 Stage O dual-oracle lower-multiplier admission audit.

Stage N classified the external multiplier 0.25 as oracle-not-admitted, but its
27 record count repeats the same comparator oracle control for every one of the
nine OOF backbones.  The underlying independent oracle population is only three
comparator cells, one per timestep.  In addition, the immutable Stage-L report
stores internal-scale summaries pooled across all four external multipliers, so
it cannot identify the internal-scale rejection mechanism specifically at 0.25.

Stage O therefore performs one tightly bounded objective-train-only replay:
raw oracle and projected oracle at timesteps 10/25/50, external multiplier 0.25,
for exactly six callback-off/on pairs.  It does not fit or evaluate an OOF
surrogate, does not access holdout or the frozen probe, and does not modify the
integrator, callback, thresholds, scale search, or external multiplier bank.

Callback events and arrays remain in memory.  The report persists scalar
acceptance, displacement statistics, selected-internal-scale histograms, and
sequential predicate aggregates only.  Every replay result is checked against
the matching immutable Stage-L 0.25 control summary before classification.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np


PHASE = "Phase3.14b-r2.5.8 Stage O"
SCHEMA = "phase314b_r258_stageo_dual_oracle_lower_multiplier_admission_v1"
BLOCKED_SCHEMA = "phase314b_r258_stageo_dual_oracle_lower_multiplier_admission_blocked_v1"

BASE_EVIDENCE_COMMIT = "9fb03b578bce6f680043f0ff38d0c20592713e6a"
BASE_IMPLEMENTATION_COMMIT = "a9b90d1f451b4444bd4d226213d847b5b13d84d2"
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
STAGEM_REPORT = (
    "reports/phase3_14b_r258_stagem_external_multiplier_predicate_stratification_summary.json"
)
EXPECTED_STAGEM_REPORT_SHA256 = (
    "8588fc5aa480bbaffb4ee3327a407d8751dddf048d8588fc5971a0e9540d7cb4"
)
STAGEN_REPORT = (
    "reports/phase3_14b_r258_stagen_lower_multiplier_post_upper_rejection_summary.json"
)
EXPECTED_STAGEN_REPORT_SHA256 = (
    "eade1273a6fc3cd66e203247c3b3d8f9e861e0d7560683c6d958c5ca723db23e"
)

SUCCESS_REPORT = (
    "reports/phase3_14b_r258_stageo_dual_oracle_lower_multiplier_admission_summary.json"
)
BLOCKED_REPORT = (
    "reports/phase3_14b_r258_stageo_dual_oracle_lower_multiplier_admission_blocked_summary.json"
)

IMPLEMENTATION_SUBJECT = (
    "Phase3.14b-r2.5.8 Stage O: audit dual-oracle lower-multiplier admission"
)
EVIDENCE_SUBJECT = (
    "Record Phase3.14b-r2.5.8 Stage O dual-oracle admission evidence"
)
BLOCKED_EVIDENCE_SUBJECT = (
    "Record Phase3.14b-r2.5.8 Stage O blocked evidence"
)
IMPLEMENTATION_PATHS: Tuple[Tuple[str, str], ...] = (
    (
        "A",
        "ccda_phase3/phase314b_r258_stageo_dual_oracle_lower_multiplier_admission.py",
    ),
    ("A", "scripts/phase3_14b_r258_stageo_execute.py"),
    (
        "A",
        "tests/test_phase3_14b_r258_stageo_dual_oracle_lower_multiplier_admission.py",
    ),
)

LOWER_MULTIPLIER = 0.25
EXPECTED_TIMESTEPS: Tuple[int, ...] = (10, 25, 50)
ORACLE_SOURCES: Tuple[str, ...] = ("raw_oracle", "projected_oracle")
EXPECTED_REPLAY_PAIR_COUNT = 6
EXPECTED_STAGEN_RECORD_COUNT = 27
EXPECTED_UNIQUE_COMPARATOR_CELL_COUNT = 3
EXPECTED_BACKBONE_REPLICATION = 9
EXPECTED_SELECTED_ULP_FACTOR = 1.0

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

EXPECTED_ENV: Mapping[str, str] = {
    "PYTHONHASHSEED": "0",
    "PYTHONNOUSERSITE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}

FROZEN_SOURCE_SHA256: Mapping[str, str] = {
    "ccda_phase3/phase314b_r258_stagee_constrained_integrator.py": (
        "37a9567391a308bfe804b95b4aa488ada40e82b2eb94bd03f2060be962cc9eb1"
    ),
    "ccda_phase3/phase314b_r258_stagek_explicit_integrator_predicate_callback.py": (
        "403b4242c6bf7c018e6220bf4e3920e26a9cd945e6bf3871792d88bd7e1f3139"
    ),
    "ccda_phase3/phase314b_r258_stagel_predicate_assembly_audit.py": (
        "b6d5c3ae42ced096a715e40de7c692eb4a313c7c4a5e31ad3807773d7963fcbd"
    ),
    "ccda_phase3/phase314b_r258_stagem_external_multiplier_predicate_stratification.py": (
        "b22d659cdbe0f3f093bbde8dd935d378fda784670827b234eb915f7370d83adf"
    ),
    "ccda_phase3/phase314b_r258_stagen_lower_multiplier_post_upper_rejection.py": (
        "d5169560a2f2cd9d347b3e2646766e90c70c1ed76d8d5ceb439642002745d74c"
    ),
}

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


class StageOError(RuntimeError):
    """Fail-closed Stage-O error."""


@dataclass(frozen=True)
class StageOSpec:
    lower_multiplier: float = LOWER_MULTIPLIER
    oracle_admission_rate_min: float = 0.95
    dominant_failure_mass_min: float = 0.50
    stable_cell_support_min: int = 5
    sparse_other_cell_support_max: int = 1
    binary_tolerance: float = 1.0e-12
    candidate_motion_epsilon: float = 1.0e-12

    def validate(self) -> None:
        if float(self.lower_multiplier) != LOWER_MULTIPLIER:
            raise StageOError("Stage-O lower multiplier changed")
        for value in (
            self.oracle_admission_rate_min,
            self.dominant_failure_mass_min,
        ):
            if not (0.0 <= float(value) <= 1.0):
                raise StageOError("Stage-O rate threshold is outside [0, 1]")
        if self.stable_cell_support_min != 5:
            raise StageOError("Stage-O stable support changed")
        if self.sparse_other_cell_support_max != 1:
            raise StageOError("Stage-O sparse support changed")
        if self.binary_tolerance < 0.0 or self.candidate_motion_epsilon < 0.0:
            raise StageOError("Stage-O numerical tolerance is negative")


def stable_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StageOError(f"{label} is not a mapping")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise StageOError(f"{label} is not a sequence")
    return value


def _finite_rate(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or not (0.0 <= result <= 1.0):
        raise StageOError(f"{label} is not a finite rate")
    return result


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise StageOError(f"{label} is boolean")
    result = int(value)
    if result < 0 or float(result) != float(value):
        raise StageOError(f"{label} is not a nonnegative integer")
    return result


def _scale_key(value: float) -> str:
    return format(float(value), ".17g")


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=str(root),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise StageOError(
            "git {} failed: {}".format(" ".join(args), completed.stderr.strip())
        )
    return completed.stdout.strip()


def _git_bytes(root: Path, *args: str) -> bytes:
    completed = subprocess.run(
        ["git", *args],
        cwd=str(root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise StageOError(
            "git {} failed: {}".format(
                " ".join(args), completed.stderr.decode("utf-8", "replace")
            )
        )
    return completed.stdout


def commit_name_status(root: Path, commit: str) -> Tuple[Tuple[str, str], ...]:
    output = _git(root, "diff-tree", "--no-commit-id", "--name-status", "-r", commit)
    records: List[Tuple[str, str]] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) != 2:
            raise StageOError("unexpected diff-tree record")
        records.append((fields[0], fields[1]))
    return tuple(sorted(records))


def status_paths(root: Path) -> Tuple[str, ...]:
    output = _git(root, "status", "--porcelain", "--untracked-files=all")
    paths: List[str] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        value = line[3:].strip()
        if " -> " in value:
            value = value.split(" -> ", 1)[1]
        paths.append(value)
    return tuple(sorted(paths))


def require_false(mapping: Mapping[str, Any], keys: Iterable[str], label: str) -> None:
    for key in keys:
        if mapping.get(key) is not False:
            raise StageOError(f"{label} boundary changed: {key}={mapping.get(key)!r}")


def validate_environment_variables() -> Mapping[str, str]:
    mismatch = {
        key: {"expected": expected, "actual": os.environ.get(key)}
        for key, expected in EXPECTED_ENV.items()
        if os.environ.get(key) != expected
    }
    if mismatch:
        raise StageOError(f"deterministic environment mismatch: {mismatch}")
    return dict(EXPECTED_ENV)


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise StageOError(f"cannot load JSON {path}: {error}") from error
    return _mapping(value, str(path))


def _validate_report_blob(root: Path, relative: str, expected_sha: str) -> Mapping[str, Any]:
    path = root / relative
    if not path.is_file():
        raise StageOError(f"required report is missing: {relative}")
    if sha256_file(path) != expected_sha:
        raise StageOError(f"report SHA changed: {relative}")
    if _git_bytes(root, "show", f"HEAD:{relative}") != path.read_bytes():
        raise StageOError(f"report differs from committed blob: {relative}")
    return _load_json(path)


def validate_stage_n_report(report: Mapping[str, Any]) -> Mapping[str, Any]:
    if report.get("execution_verdict") != "PASS":
        raise StageOError("Stage-N execution is not PASS")
    if report.get("scientific_status") != "BLOCKED":
        raise StageOError("Stage-N scientific status changed")
    if report.get("root_cause") != "phase314b_r258_stagen_lower_multiplier_oracle_not_admitted":
        raise StageOError("Stage-N root cause changed")
    if report.get("required_next_path") != "AUDIT_ORACLE_ADMISSION_AT_LOWER_EXTERNAL_MULTIPLIER":
        raise StageOError("Stage-N next path changed")
    classification = _mapping(report.get("classification"), "Stage-N classification")
    if int(classification.get("record_count", -1)) != EXPECTED_STAGEN_RECORD_COUNT:
        raise StageOError("Stage-N record population changed")
    if int(classification.get("oracle_admitted_record_count", -1)) != 0:
        raise StageOError("Stage-N oracle-admitted count changed")
    if int(classification.get("upper_gate_cleared_record_count", -1)) != 0:
        raise StageOError("Stage-N upper-clear count changed")
    if report.get("selected_configuration") is not None:
        raise StageOError("Stage-N selected configuration is not null")
    if report.get("train_only_recommendation") is not None:
        raise StageOError("Stage-N recommendation is not null")
    require_false(report, FALSE_BOUNDARIES, "Stage-N")
    return classification


def validate_stage_m_report(report: Mapping[str, Any]) -> None:
    if report.get("execution_verdict") != "PASS":
        raise StageOError("Stage-M execution is not PASS")
    if report.get("root_cause") != "phase314b_r258_stagem_upper_segment_geometry_is_single_multiplier_boundary":
        raise StageOError("Stage-M root cause changed")
    if report.get("required_next_path") != "AUDIT_LOWER_MULTIPLIER_REJECTION_AFTER_UPPER_SEGMENT_GATE":
        raise StageOError("Stage-M next path changed")
    require_false(report, FALSE_BOUNDARIES, "Stage-M")


def extract_stage_l_scientific_result(resume1_report: Mapping[str, Any]) -> Mapping[str, Any]:
    if resume1_report.get("execution_verdict") != "PASS":
        raise StageOError("Stage-L Resume1 execution is not PASS")
    if resume1_report.get("scientific_status") != "BLOCKED":
        raise StageOError("Stage-L Resume1 scientific status changed")
    recovery = _mapping(resume1_report.get("recovery_contract"), "recovery_contract")
    if recovery.get("underlying_stage_l_single_run_result_sha256") != EXPECTED_STAGEL_SINGLE_RUN_SHA256:
        raise StageOError("Stage-L single-run SHA changed")
    if int(recovery.get("underlying_callback_pair_count", -1)) != 132:
        raise StageOError("Stage-L callback pair count changed")
    if recovery.get("stage_l_science_modified") is not False:
        raise StageOError("Stage-L science was modified")
    stage_l_result = _mapping(resume1_report.get("stage_l_result"), "stage_l_result")
    scientific = _mapping(stage_l_result.get("scientific_result"), "scientific_result")
    require_false(scientific, FALSE_BOUNDARIES, "Stage-L scientific result")
    audit = _mapping(scientific.get("predicate_assembly_audit"), "predicate_assembly_audit")
    if int(audit.get("callback_off_on_pair_count", -1)) != 132:
        raise StageOError("Stage-L callback pair count changed")
    if audit.get("all_callback_results_bit_exact") is not True:
        raise StageOError("Stage-L callback identity changed")
    if audit.get("callback_events_persisted") is not False:
        raise StageOError("Stage-L callback events were persisted")
    return scientific


def unique_stage_n_comparator_cells(
    stage_n_report: Mapping[str, Any], stage_m_report: Mapping[str, Any]
) -> Mapping[str, Any]:
    records = _sequence(stage_n_report.get("record_audits"), "Stage-N record_audits")
    if len(records) != EXPECTED_STAGEN_RECORD_COUNT:
        raise StageOError("Stage-N record count changed")
    stage_m_records = _sequence(stage_m_report.get("record_strata"), "Stage-M record_strata")
    comparator_by_cell: Dict[Tuple[str, int], str] = {}
    backbones: set[str] = set()
    for index, raw in enumerate(stage_m_records):
        record = _mapping(raw, f"Stage-M record {index}")
        backbone = str(record.get("base_direction_id"))
        timestep = int(record.get("timestep"))
        comparator = str(record.get("comparator_source"))
        backbones.add(backbone)
        key = (backbone, timestep)
        comparator_by_cell[key] = comparator
    if len(backbones) != EXPECTED_BACKBONE_REPLICATION:
        raise StageOError("Stage-M backbone population changed")
    unique = set()
    for index, raw in enumerate(records):
        record = _mapping(raw, f"Stage-N record {index}")
        backbone = str(record.get("base_direction_id"))
        timestep = int(record.get("timestep"))
        key = (backbone, timestep)
        if key not in comparator_by_cell:
            raise StageOError("Stage-M/Stage-N backbone × timestep population differs")
        unique.add((timestep, comparator_by_cell[key]))
    if len(unique) != EXPECTED_UNIQUE_COMPARATOR_CELL_COUNT:
        raise StageOError("Stage-N comparator cell population changed")
    return {
        "stage_n_record_count": len(records),
        "unique_comparator_oracle_cell_count": len(unique),
        "backbone_replication_factor": len(records) // len(unique),
        "unique_comparator_cells": [
            {"timestep": timestep, "source_id": source}
            for timestep, source in sorted(unique)
        ],
        "stage_n_27_records_are_not_independent_oracle_controls": True,
    }


def validate_repository(root: Path) -> Mapping[str, Any]:
    repo = Path(root).resolve()
    if _git(repo, "branch", "--show-current") != "Experiment1":
        raise StageOError("Stage O requires Experiment1")
    head = _git(repo, "rev-parse", "HEAD")
    parent = _git(repo, "rev-parse", f"{head}^")
    if parent != BASE_EVIDENCE_COMMIT:
        raise StageOError("Stage-O implementation parent changed")
    if _git(repo, "show", "-s", "--format=%s", head) != IMPLEMENTATION_SUBJECT:
        raise StageOError("Stage-O implementation subject changed")
    if commit_name_status(repo, head) != tuple(sorted(IMPLEMENTATION_PATHS)):
        raise StageOError("Stage-O implementation path population changed")
    if _git(repo, "rev-parse", "origin/Experiment1") != EXPECTED_REMOTE:
        raise StageOError("origin/Experiment1 changed")
    submodule = repo / "external/deformable-ravens"
    if _git(submodule, "rev-parse", "HEAD") != EXPECTED_SUBMODULE:
        raise StageOError("DeformableRavens commit changed")
    if status_paths(submodule):
        raise StageOError("DeformableRavens worktree is dirty")
    if status_paths(repo):
        raise StageOError("Stage-O worktree must be clean before execution")

    resume1 = _validate_report_blob(
        repo, STAGEL_RESUME1_REPORT, EXPECTED_STAGEL_RESUME1_REPORT_SHA256
    )
    stage_m = _validate_report_blob(repo, STAGEM_REPORT, EXPECTED_STAGEM_REPORT_SHA256)
    stage_n = _validate_report_blob(repo, STAGEN_REPORT, EXPECTED_STAGEN_REPORT_SHA256)
    extract_stage_l_scientific_result(resume1)
    validate_stage_m_report(stage_m)
    validate_stage_n_report(stage_n)

    source_sha: Dict[str, str] = {}
    for relative, expected in FROZEN_SOURCE_SHA256.items():
        path = repo / relative
        actual = sha256_file(path)
        if actual != expected:
            raise StageOError(f"frozen source SHA changed: {relative}")
        source_sha[relative] = actual

    for relative in (SUCCESS_REPORT, BLOCKED_REPORT):
        if (repo / relative).exists():
            raise StageOError(f"Stage-O output already exists: {relative}")
    return {
        "root": str(repo),
        "head": head,
        "parent": parent,
        "origin_experiment1": EXPECTED_REMOTE,
        "submodule_commit": EXPECTED_SUBMODULE,
        "frozen_source_sha256": source_sha,
        "report_sha256": {
            STAGEL_RESUME1_REPORT: EXPECTED_STAGEL_RESUME1_REPORT_SHA256,
            STAGEM_REPORT: EXPECTED_STAGEM_REPORT_SHA256,
            STAGEN_REPORT: EXPECTED_STAGEN_REPORT_SHA256,
        },
    }


def aggregate_sequential_attempts(events: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not events:
        raise StageOError("cannot aggregate an empty attempt population")
    reached = {name: 0 for name in PREDICATE_ORDER}
    passed = {name: 0 for name in PREDICATE_ORDER}
    failed = {name: 0 for name in PREDICATE_ORDER}
    active_total = 0
    accepted_total = 0
    for event in events:
        if tuple(event.get("predicate_order", ())) != PREDICATE_ORDER:
            raise StageOError("predicate order changed")
        active = _nonnegative_int(event.get("active_row_count"), "active_row_count")
        accepted = _nonnegative_int(event.get("accepted_count"), "accepted_count")
        if accepted > active:
            raise StageOError("accepted count exceeds active count")
        failures = _mapping(event.get("first_failed_counts"), "first_failed_counts")
        active_total += active
        accepted_total += accepted
        unresolved = active
        for predicate in PREDICATE_ORDER:
            failure = _nonnegative_int(failures.get(predicate), f"failure[{predicate}]")
            if failure > unresolved:
                raise StageOError("first-failure count exceeds sequential reach")
            reached[predicate] += unresolved
            failed[predicate] += failure
            survivors = unresolved - failure
            passed[predicate] += survivors
            unresolved = survivors
        if unresolved != accepted:
            raise StageOError("attempt population does not close to accepted rows")
    rejected_total = active_total - accepted_total
    conditional_pass = {
        key: 1.0 if reached[key] == 0 else float(passed[key] / reached[key])
        for key in PREDICATE_ORDER
    }
    conditional_failure = {
        key: 0.0 if reached[key] == 0 else float(failed[key] / reached[key])
        for key in PREDICATE_ORDER
    }
    rejection_mass = {
        key: 0.0 if rejected_total == 0 else float(failed[key] / rejected_total)
        for key in PREDICATE_ORDER
    }
    dominant = max(PREDICATE_ORDER, key=lambda key: failed[key])
    dominant_count = failed[dominant]
    return {
        "event_count": len(events),
        "active_row_attempt_count": active_total,
        "accepted_row_attempt_count": accepted_total,
        "rejected_row_attempt_count": rejected_total,
        "sequential_reach_counts": reached,
        "sequential_pass_counts": passed,
        "first_failed_counts": failed,
        "conditional_pass_rates": conditional_pass,
        "conditional_failure_rates": conditional_failure,
        "rejection_mass_rates": rejection_mass,
        "dominant_failure_predicate": None if dominant_count == 0 else dominant,
        "dominant_failure_mass": (
            0.0 if rejected_total == 0 else float(dominant_count / rejected_total)
        ),
        "sequential_population_closed": True,
    }


def group_attempts_by_internal_scale(
    events: Sequence[Mapping[str, Any]],
) -> Mapping[str, Sequence[Mapping[str, Any]]]:
    output: Dict[str, List[Mapping[str, Any]]] = {}
    previous: Optional[float] = None
    for event in events:
        scale = float(event.get("attempted_scale"))
        if not math.isfinite(scale) or scale <= 0.0:
            raise StageOError("invalid attempted internal scale")
        if previous is not None and scale <= previous:
            raise StageOError("internal scale attempt order changed")
        previous = scale
        output.setdefault(_scale_key(scale), []).append(event)
    return {key: tuple(value) for key, value in output.items()}


def scalar_stats(value: np.ndarray) -> Mapping[str, Any]:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise StageOError("cannot summarize an empty or non-finite array")
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "p05": float(np.percentile(array, 5.0)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95.0)),
        "max": float(np.max(array)),
    }


def selected_scale_histogram(selected: np.ndarray) -> Mapping[str, int]:
    array = np.asarray(selected, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise StageOError("selected scale contains invalid values")
    output: Dict[str, int] = {}
    for value in array:
        key = _scale_key(float(value))
        output[key] = output.get(key, 0) + 1
    if sum(output.values()) != int(array.size):
        raise StageOError("selected-scale histogram does not close")
    return dict(sorted(output.items(), key=lambda item: float(item[0])))


def _attempt_events(capture: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    values = _sequence(capture.get("events"), "callback events")
    attempts = [
        _mapping(value, f"callback event {index}")
        for index, value in enumerate(values)
        if isinstance(value, Mapping) and value.get("event_type") == "scale_attempt"
    ]
    if not attempts:
        raise StageOError("callback capture lacks scale attempts")
    return attempts


def _persisted_multiplier_record(
    persisted: Mapping[str, Any], multiplier: float
) -> Mapping[str, Any]:
    values = _sequence(persisted.get("multiplier_records"), "multiplier_records")
    matches = [
        _mapping(value, f"multiplier record {index}")
        for index, value in enumerate(values)
        if float(_mapping(value, f"multiplier record {index}").get("scale_multiplier"))
        == float(multiplier)
    ]
    if len(matches) != 1:
        raise StageOError("persisted multiplier record population changed")
    return matches[0]


def compare_replay_to_persisted(
    *,
    observed: Mapping[str, Any],
    persisted: Mapping[str, Any],
) -> Mapping[str, Any]:
    expected = _persisted_multiplier_record(persisted, LOWER_MULTIPLIER)
    exact_fields = (
        "selected_scale_sha256",
        "candidate_sha256",
        "callback_capture_sha256",
        "callback_result_bit_exact",
    )
    for field in exact_fields:
        if observed.get(field) != expected.get(field):
            raise StageOError(f"targeted replay differs from Stage-L: {field}")
    rate_fields = (
        "selected_scale_positive_rate",
        "candidate_motion_positive_rate",
    )
    for field in rate_fields:
        if float(observed.get(field)) != float(expected.get(field)):
            raise StageOError(f"targeted replay rate differs from Stage-L: {field}")
    if stable_json_bytes(observed.get("stage_l_assembly_overall")) != stable_json_bytes(
        expected.get("assembly")
    ):
        raise StageOError("targeted replay assembly differs from Stage-L")
    return {
        "matched_stage_l_025_multiplier_record": True,
        "persisted_selected_scale_sha256": expected.get("selected_scale_sha256"),
        "persisted_candidate_sha256": expected.get("candidate_sha256"),
        "persisted_callback_capture_sha256": expected.get("callback_capture_sha256"),
    }


def audit_oracle_direction(
    *,
    source_id: str,
    timestep: int,
    control: np.ndarray,
    base_direction: np.ndarray,
    context: Mapping[str, Any],
    persisted: Mapping[str, Any],
    runtime: Mapping[str, Any],
    spec: StageOSpec,
) -> Mapping[str, Any]:
    stagek = runtime["stagek"]
    integrator_spec = runtime["integrator_spec"]
    callback_spec = runtime["callback_spec"]
    control_raw = np.asarray(control, dtype=np.float32)
    direction_raw = np.asarray(base_direction, dtype=np.float64)
    if control_raw.shape != direction_raw.shape:
        raise StageOError("control and oracle direction shape mismatch")
    proposed = direction_raw * float(spec.lower_multiplier)
    integration, capture = stagek.callback_integrate_rowwise(
        control=control_raw,
        direction=proposed,
        context=context,
        integrator_spec=integrator_spec,
        callback_spec=callback_spec,
    )
    if capture.get("returned_result_bit_exact") is not True:
        raise StageOError("callback-off/on replay is not byte-exact")
    rows = int(control_raw.shape[0])
    selected = np.asarray(integration["selected_scale"], dtype=np.float64).reshape(rows)
    candidate = np.asarray(integration["candidate"], dtype=np.float32)
    motion = stagek._row_norm(candidate.astype(np.float64) - control_raw.astype(np.float64))
    selected_positive = selected > spec.binary_tolerance
    motion_positive = motion > spec.candidate_motion_epsilon
    if np.any(selected_positive != motion_positive):
        raise StageOError("selected-scale and candidate-motion observables disagree")
    events = _attempt_events(capture)
    overall = aggregate_sequential_attempts(events)
    stage_l_overall = runtime["stagel"].aggregate_sequential_attempts(events)
    by_scale_events = group_attempts_by_internal_scale(events)
    by_scale = {
        key: aggregate_sequential_attempts(value)
        for key, value in by_scale_events.items()
    }
    internal_dominants = {
        key: value.get("dominant_failure_predicate")
        for key, value in by_scale.items()
    }
    nonnull_internal_dominants = sorted(
        {value for value in internal_dominants.values() if isinstance(value, str)},
        key=PREDICATE_ORDER.index,
    )
    internal_scale_heterogeneous = len(nonnull_internal_dominants) > 1
    stable_internal_predicate = (
        nonnull_internal_dominants[0]
        if len(nonnull_internal_dominants) == 1
        else None
    )
    observed: Dict[str, Any] = {
        "source_id": source_id,
        "timestep": int(timestep),
        "external_multiplier": float(spec.lower_multiplier),
        "objective_train_rows": rows,
        "oracle_acceptance_rate": float(np.mean(selected_positive)),
        "oracle_admitted": bool(
            float(np.mean(selected_positive)) >= spec.oracle_admission_rate_min
        ),
        "selected_scale_positive_rate": float(np.mean(selected_positive)),
        "selected_scale_sha256": sha256_array(selected),
        "selected_internal_scale_histogram": selected_scale_histogram(selected),
        "candidate_motion_positive_rate": float(np.mean(motion_positive)),
        "candidate_sha256": sha256_array(candidate),
        "callback_capture_sha256": str(capture.get("events_sha256")),
        "callback_result_bit_exact": True,
        "proposal_row_norm_stats": scalar_stats(stagek._row_norm(proposed)),
        "candidate_motion_row_norm_stats": scalar_stats(motion),
        "assembly_overall": overall,
        "stage_l_assembly_overall": stage_l_overall,
        "assembly_by_internal_scale": by_scale,
        "internal_scale_keys": list(by_scale),
        "dominant_failure_predicate_by_internal_scale": internal_dominants,
        "dominant_failure_predicates_across_internal_scales": nonnull_internal_dominants,
        "internal_scale_failure_heterogeneous": internal_scale_heterogeneous,
        "stable_internal_scale_failure_predicate": stable_internal_predicate,
        "first_accepting_internal_scale": next(
            (
                float(key)
                for key, value in by_scale.items()
                if int(value["accepted_row_attempt_count"]) > 0
            ),
            None,
        ),
        "callback_events_persisted": False,
        "direction_tensor_persisted": False,
        "candidate_tensor_persisted": False,
    }
    observed["stage_l_identity_check"] = compare_replay_to_persisted(
        observed=observed,
        persisted=persisted,
    )
    return observed


def _runtime_modules() -> Mapping[str, Any]:
    from ccda_phase3 import phase314b_r258_stagel_predicate_assembly_audit as stagel
    from ccda_phase3 import phase314b_r258_stagek_explicit_integrator_predicate_callback as stagek

    return {"stagel": stagel, "stagek": stagek}


def probe_environment(root: Path) -> Mapping[str, Any]:
    runtime = _runtime_modules()
    stagel = runtime["stagel"]
    return stagel.stagef.stagec258.stagea258.probe_portable_environment(root=Path(root).resolve())


def _prepare_runtime(
    root: Path,
    environment: Mapping[str, Any],
) -> Mapping[str, Any]:
    modules = _runtime_modules()
    stagel = modules["stagel"]
    stagek = modules["stagek"]
    stagef = stagel.stagef
    staged258 = stagel.staged258

    stagel.validate_base_evidence(Path(root).resolve())
    stagef.stagec258.stagea258.validate_environment_payload(environment)
    if environment.get("compatibility_sha256") != stagef.EXPECTED_COMPATIBILITY_SHA256:
        raise StageOError("portable compatibility SHA changed")
    cold = stagef.stagec258.stagea258.assert_cold_cuda_context_portable()
    captured = stagef.stageb258.capture_portable_control_model(root=Path(root).resolve())
    direction_spec = staged258.DirectionSurrogateSpec()
    stagef_spec = stagef.ConstraintAwareSpec()
    integrator_spec = stagel.stagee258.ConstrainedIntegratorSpec()
    callback_spec = stagek.ExplicitPredicateCallbackSpec()
    direction_spec.validate()
    stagef_spec.validate()
    integrator_spec.validate()
    callback_spec.validate()
    context = stagef.build_context(
        root=Path(root).resolve(),
        captured=captured,
        direction_spec=direction_spec,
    )
    ulp_policy = stagef.calibrate_float32_constraint_z_policy(
        controls_by_timestep={
            int(timestep): context["objective_control_predictions"][int(timestep)]
            for timestep in EXPECTED_TIMESTEPS
        },
        context=context,
        spec=stagef_spec,
    )
    if float(ulp_policy["selected_factor"]) != EXPECTED_SELECTED_ULP_FACTOR:
        raise StageOError("Stage-O ULP policy replay changed")
    stagef_spec = stagef.replace(
        stagef_spec,
        translation_float32_z_ulp_factor=EXPECTED_SELECTED_ULP_FACTOR,
    )
    stagef_spec.validate()
    return {
        **modules,
        "stagef": stagef,
        "direction_spec": direction_spec,
        "stagef_spec": stagef_spec,
        "integrator_spec": integrator_spec,
        "callback_spec": callback_spec,
        "context": context,
        "cold_main_worker_context": cold,
        "control_capture": captured["control_identity"],
        "ulp_policy": ulp_policy,
    }


def classify_oracle_admission(
    cells: Sequence[Mapping[str, Any]],
    spec: Optional[StageOSpec] = None,
) -> Mapping[str, Any]:
    active = StageOSpec() if spec is None else spec
    active.validate()
    if len(cells) != EXPECTED_REPLAY_PAIR_COUNT:
        raise StageOError("Stage-O cell population changed")
    source_admission: Dict[str, int] = {source: 0 for source in ORACLE_SOURCES}
    source_timesteps: Dict[str, List[int]] = {source: [] for source in ORACLE_SOURCES}
    dominant_counts: Dict[str, int] = {}
    source_dominant: Dict[str, Dict[str, int]] = {source: {} for source in ORACLE_SOURCES}
    by_timestep: Dict[str, Dict[str, int]] = {str(t): {} for t in EXPECTED_TIMESTEPS}
    internal_scale_heterogeneous_cell_count = 0
    overall_internal_mismatch_cell_count = 0
    for cell in cells:
        source = str(cell.get("source_id"))
        timestep = int(cell.get("timestep"))
        if source not in ORACLE_SOURCES or timestep not in EXPECTED_TIMESTEPS:
            raise StageOError("unexpected source/timestep cell")
        if cell.get("oracle_admitted") is True:
            source_admission[source] += 1
            source_timesteps[source].append(timestep)
        assembly = _mapping(cell.get("assembly_overall"), "cell assembly")
        predicate = assembly.get("dominant_failure_predicate")
        mass = _finite_rate(assembly.get("dominant_failure_mass"), "dominant failure mass")
        internal_heterogeneous = cell.get("internal_scale_failure_heterogeneous") is True
        stable_internal = cell.get("stable_internal_scale_failure_predicate")
        if internal_heterogeneous:
            internal_scale_heterogeneous_cell_count += 1
        if (
            isinstance(predicate, str)
            and isinstance(stable_internal, str)
            and predicate != stable_internal
        ):
            overall_internal_mismatch_cell_count += 1
        cell_stable = (
            not internal_heterogeneous
            and (
                stable_internal is None
                or (isinstance(predicate, str) and stable_internal == predicate)
            )
        )
        if (
            isinstance(predicate, str)
            and mass >= active.dominant_failure_mass_min
            and cell_stable
        ):
            dominant_counts[predicate] = dominant_counts.get(predicate, 0) + 1
            source_dominant[source][predicate] = source_dominant[source].get(predicate, 0) + 1
            by_timestep[str(timestep)][predicate] = (
                by_timestep[str(timestep)].get(predicate, 0) + 1
            )

    fully_admitted = [
        source for source, count in source_admission.items() if count == len(EXPECTED_TIMESTEPS)
    ]
    partially_admitted = [
        source for source, count in source_admission.items() if 0 < count < len(EXPECTED_TIMESTEPS)
    ]
    dominant_predicate: Optional[str] = None
    dominant_support = 0
    other_max = 0
    timestep_covered = False
    if dominant_counts:
        dominant_predicate = max(
            dominant_counts,
            key=lambda key: (dominant_counts[key], -PREDICATE_ORDER.index(key)),
        )
        dominant_support = dominant_counts[dominant_predicate]
        other_max = max(
            (count for key, count in dominant_counts.items() if key != dominant_predicate),
            default=0,
        )
        timestep_covered = all(
            int(by_timestep[str(t)].get(dominant_predicate, 0)) >= 1
            for t in EXPECTED_TIMESTEPS
        )

    predicate_next: Mapping[str, Tuple[str, str]] = {
        "finite_state": (
            "phase314b_r258_stageo_lower_multiplier_oracle_nonfinite",
            "AUDIT_ORACLE_FINITE_STATE_AT_LOWER_EXTERNAL_MULTIPLIER",
        ),
        "upper_segment_geometry": (
            "phase314b_r258_stageo_lower_multiplier_oracle_upper_segment_rejection",
            "AUDIT_ORACLE_UPPER_SEGMENT_GATE_AT_LOWER_EXTERNAL_MULTIPLIER",
        ),
        "lower_segment_geometry": (
            "phase314b_r258_stageo_lower_multiplier_oracle_lower_segment_rejection",
            "AUDIT_ORACLE_LOWER_SEGMENT_GATE_AT_LOWER_EXTERNAL_MULTIPLIER",
        ),
        "coordinate_recenter": (
            "phase314b_r258_stageo_lower_multiplier_oracle_coordinate_recenter_rejection",
            "AUDIT_ORACLE_LOCAL_FRAME_COMPATIBILITY_AT_LOWER_MULTIPLIER",
        ),
        "coordinate_geometry": (
            "phase314b_r258_stageo_lower_multiplier_oracle_coordinate_geometry_rejection",
            "AUDIT_ORACLE_LOCAL_FRAME_COMPATIBILITY_AT_LOWER_MULTIPLIER",
        ),
        "reconstruction_bounds": (
            "phase314b_r258_stageo_lower_multiplier_oracle_reconstruction_rejection",
            "AUDIT_ORACLE_SEGMENT_RECONSTRUCTION_AT_LOWER_MULTIPLIER",
        ),
        "segment_geometry": (
            "phase314b_r258_stageo_lower_multiplier_oracle_segment_geometry_rejection",
            "AUDIT_ORACLE_SEGMENT_RECONSTRUCTION_AT_LOWER_MULTIPLIER",
        ),
        "direction_retention": (
            "phase314b_r258_stageo_lower_multiplier_oracle_direction_retention_rejection",
            "CALIBRATE_INTEGRATOR_COMPATIBLE_ORACLE_DIRECTION_BASIS_AT_LOWER_MULTIPLIER",
        ),
        "displacement": (
            "phase314b_r258_stageo_lower_multiplier_oracle_displacement_rejection",
            "AUDIT_MINIMUM_DISPLACEMENT_THRESHOLD_AT_LOWER_EXTERNAL_MULTIPLIER",
        ),
        "topology": (
            "phase314b_r258_stageo_lower_multiplier_oracle_topology_rejection",
            "AUDIT_ORACLE_TOPOLOGY_COMPATIBILITY_AT_LOWER_MULTIPLIER",
        ),
    }

    if len(fully_admitted) == 2:
        root = "phase314b_r258_stageo_both_lower_multiplier_oracle_sources_admitted"
        next_path = "REOPEN_LOWER_MULTIPLIER_OOF_POST_UPPER_AUDIT_WITH_DUAL_ORACLE_CONTROLS"
        primary = "dual_oracle_admission"
    elif len(fully_admitted) == 1 and not partially_admitted:
        root = "phase314b_r258_stageo_lower_multiplier_admission_is_oracle_source_specific"
        next_path = "REOPEN_LOWER_MULTIPLIER_OOF_POST_UPPER_AUDIT_USING_ADMITTED_ORACLE_SOURCE"
        primary = f"admitted_oracle_source::{fully_admitted[0]}"
    elif fully_admitted or partially_admitted:
        root = "phase314b_r258_stageo_lower_multiplier_oracle_admission_is_source_or_timestep_dependent"
        next_path = "STRATIFY_LOWER_MULTIPLIER_ORACLE_ADMISSION_BY_SOURCE_AND_TIMESTEP"
        primary = "heterogeneous_oracle_admission"
    elif internal_scale_heterogeneous_cell_count > 0 or overall_internal_mismatch_cell_count > 0:
        root = "phase314b_r258_stageo_oracle_rejection_is_internal_scale_dependent"
        next_path = "STRATIFY_LOWER_MULTIPLIER_ORACLE_REJECTION_BY_SOURCE_TIMESTEP_AND_INTERNAL_SCALE"
        primary = "internal_scale_dependent_oracle_rejection"
    elif (
        dominant_predicate is not None
        and dominant_support >= active.stable_cell_support_min
        and other_max <= active.sparse_other_cell_support_max
        and timestep_covered
    ):
        root, next_path = predicate_next[dominant_predicate]
        primary = f"oracle_rejection::{dominant_predicate}"
    elif dominant_counts:
        root = "phase314b_r258_stageo_oracle_rejection_is_source_timestep_or_scale_dependent"
        next_path = "STRATIFY_LOWER_MULTIPLIER_ORACLE_REJECTION_BY_SOURCE_TIMESTEP_AND_INTERNAL_SCALE"
        primary = "heterogeneous_oracle_rejection"
    else:
        root = "phase314b_r258_stageo_scalar_oracle_rejection_surface_insufficient"
        next_path = "ADD_READ_ONLY_ROW_COHORT_HASHES_FOR_LOWER_MULTIPLIER_ORACLE_REJECTION"
        primary = "oracle_scalar_surface_insufficient"

    return {
        "root_cause": root,
        "required_next_path": next_path,
        "primary_failure_locus": primary,
        "cell_count": len(cells),
        "source_admission_counts": source_admission,
        "source_admitted_timesteps": {
            key: sorted(value) for key, value in source_timesteps.items()
        },
        "fully_admitted_oracle_sources": fully_admitted,
        "partially_admitted_oracle_sources": partially_admitted,
        "dominant_failure_predicate": dominant_predicate,
        "dominant_failure_support_count": dominant_support,
        "maximum_other_failure_support_count": other_max,
        "dominant_failure_has_timestep_coverage": timestep_covered,
        "dominant_failure_counts": dict(sorted(dominant_counts.items())),
        "internal_scale_heterogeneous_cell_count": internal_scale_heterogeneous_cell_count,
        "overall_internal_dominant_mismatch_cell_count": overall_internal_mismatch_cell_count,
        "dominant_failure_counts_by_source": {
            key: dict(sorted(value.items())) for key, value in source_dominant.items()
        },
        "dominant_failure_counts_by_timestep": {
            key: dict(sorted(value.items())) for key, value in by_timestep.items()
        },
        "classification_thresholds": asdict(active),
    }


def run_targeted_replay(
    *,
    root: Path,
    environment: Mapping[str, Any],
    repository: Mapping[str, Any],
    spec: Optional[StageOSpec] = None,
) -> Mapping[str, Any]:
    active = StageOSpec() if spec is None else spec
    active.validate()
    validate_environment_variables()
    repo = Path(root).resolve()
    resume1_report = _load_json(repo / STAGEL_RESUME1_REPORT)
    stage_m_report = _load_json(repo / STAGEM_REPORT)
    stage_n_report = _load_json(repo / STAGEN_REPORT)
    scientific = extract_stage_l_scientific_result(resume1_report)
    validate_stage_m_report(stage_m_report)
    validate_stage_n_report(stage_n_report)
    independence = unique_stage_n_comparator_cells(stage_n_report, stage_m_report)
    audit = _mapping(scientific.get("predicate_assembly_audit"), "predicate_assembly_audit")
    persisted_controls = _mapping(
        audit.get("control_assembly_by_timestep"), "control_assembly_by_timestep"
    )
    if set(persisted_controls) != {str(value) for value in EXPECTED_TIMESTEPS}:
        raise StageOError("Stage-L control timestep population changed")

    runtime = _prepare_runtime(repo, environment)
    stagef = runtime["stagef"]
    context = runtime["context"]
    cells: List[Mapping[str, Any]] = []
    for timestep in EXPECTED_TIMESTEPS:
        control = np.asarray(
            context["objective_control_predictions"][int(timestep)], dtype=np.float32
        )
        target = np.asarray(context["objective_target"], dtype=np.float32)
        oracle = stagef.generate_projected_oracle_target(
            control=control,
            target=target,
            groups=context["objective_groups"],
            condition_name=context["objective_condition_name"],
            timestep=int(timestep),
            context=context,
            direction_spec=runtime["direction_spec"],
            integrator_spec=runtime["integrator_spec"],
            spec=runtime["stagef_spec"],
        )
        directions = {
            "raw_oracle": target.astype(np.float64) - control.astype(np.float64),
            "projected_oracle": (
                np.asarray(oracle["candidate"], dtype=np.float64)
                - control.astype(np.float64)
            ),
        }
        persisted_timestep = _mapping(
            persisted_controls[str(timestep)], f"persisted controls[{timestep}]"
        )
        for source_id in ORACLE_SOURCES:
            cells.append(
                audit_oracle_direction(
                    source_id=source_id,
                    timestep=timestep,
                    control=control,
                    base_direction=directions[source_id],
                    context=context,
                    persisted=_mapping(
                        persisted_timestep.get(source_id),
                        f"persisted controls[{timestep}].{source_id}",
                    ),
                    runtime=runtime,
                    spec=active,
                )
            )
    if len(cells) != EXPECTED_REPLAY_PAIR_COUNT:
        raise StageOError("Stage-O replay pair population changed")
    classification = classify_oracle_admission(cells, active)
    result: Dict[str, Any] = {
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
        "environment": copy.deepcopy(dict(environment)),
        "cold_main_worker_context": copy.deepcopy(runtime["cold_main_worker_context"]),
        "control_capture": copy.deepcopy(runtime["control_capture"]),
        "ulp_policy": copy.deepcopy(runtime["ulp_policy"]),
        "audit_spec": asdict(active),
        "stage_n_independence_correction": independence,
        "targeted_replay": {
            "external_multiplier": LOWER_MULTIPLIER,
            "oracle_sources": list(ORACLE_SOURCES),
            "timesteps": list(EXPECTED_TIMESTEPS),
            "callback_off_on_pair_count": len(cells),
            "all_callback_results_bit_exact": all(
                cell.get("callback_result_bit_exact") is True for cell in cells
            ),
            "all_cells_match_stage_l_025_identity": all(
                _mapping(cell.get("stage_l_identity_check"), "identity check").get(
                    "matched_stage_l_025_multiplier_record"
                )
                is True
                for cell in cells
            ),
            "cell_records": cells,
            "classification": classification,
            "callback_events_persisted": False,
            "direction_tensors_persisted": False,
            "candidate_tensors_persisted": False,
        },
        "immutable_inputs": {
            "stage_l_resume1_report_sha256": EXPECTED_STAGEL_RESUME1_REPORT_SHA256,
            "stage_l_single_run_sha256": EXPECTED_STAGEL_SINGLE_RUN_SHA256,
            "stage_m_report_sha256": EXPECTED_STAGEM_REPORT_SHA256,
            "stage_n_report_sha256": EXPECTED_STAGEN_REPORT_SHA256,
            "stage_l_science_rerun": False,
            "stage_l_132_callback_pairs_rerun": False,
            "oof_surrogate_refit": False,
            "oof_surrogate_evaluated": False,
            "targeted_oracle_callback_pairs_run": EXPECTED_REPLAY_PAIR_COUNT,
        },
        "mechanism_boundary": {
            "stagee_modified": False,
            "stagek_modified": False,
            "stagel_modified": False,
            "stagem_modified": False,
            "stagen_modified": False,
            "callback_schema_modified": False,
            "integrator_thresholds_changed": False,
            "integrator_scale_search_changed": False,
            "external_scale_bank_changed": False,
            "direction_model_refit": False,
            "new_ranker_fitted": False,
            "objective_train_oracle_only_replay": True,
            "raw_and_projected_oracles_audited_separately": True,
            "internal_scale_isolated_at_external_multiplier_025": True,
        },
        **{key: False for key in FALSE_BOUNDARIES},
    }
    result["scientific_result_sha256"] = sha256_bytes(stable_json_bytes(result))
    return result


def blocked_report(
    *, repository: Optional[Mapping[str, Any]], error: BaseException
) -> Mapping[str, Any]:
    return {
        "phase": PHASE,
        "schema": BLOCKED_SCHEMA,
        "execution_verdict": "BLOCKED",
        "scientific_status": "BLOCKED",
        "root_cause": "phase314b_r258_stageo_targeted_oracle_replay_failed",
        "required_next_path": "RESTORE_STAGEO_TARGETED_ORACLE_REPLAY",
        "primary_failure_locus": "execution_contract",
        "selected_configuration": None,
        "train_only_recommendation": None,
        "repository": None if repository is None else dict(repository),
        "error_type": type(error).__name__,
        "error_message": str(error),
        "stage_l_science_rerun": False,
        "stage_l_132_callback_pairs_rerun": False,
        "oof_surrogate_refit": False,
        "oof_surrogate_evaluated": False,
        **{key: False for key in FALSE_BOUNDARIES},
    }
