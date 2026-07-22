"""Phase3.14b-r2.5.8 Stage Q tolerance-aligned upper-gate shadow audit.

Stage P proved that every observed ``upper_segment_geometry`` violation at
external multiplier 0.25 was inherited from the control state and remained
inside the ``upper + 5 * segment_tolerance`` bound already accepted by the
frozen segment reconstruction contract.  Stage Q does not modify Stage E.  It
runs the same six raw/projected oracle cells, keeps the legacy strict upper gate
as an immutable control, and evaluates a shadow gate defined in the same length
space as reconstruction.

The shadow gate must satisfy two independent representations:

* length-space: ``segment_length <= frozen_upper + 5 * tolerance``;
* position-wise log-z: ``z <= (log(frozen_upper + 5*tolerance)-center)/scale``.

The two boolean masks are compared element by element.  Candidate generation,
scale order, all non-upper predicates, topology evaluation, oracle sources, and
all immutable Stage-O/Stage-P identities remain unchanged.  No holdout, frozen
probe, OOF fit, training, reverse, IDM, candidate execution, DeformableRavens,
Phase4, or CPS operation is allowed.
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
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


PHASE = "Phase3.14b-r2.5.8 Stage Q"
SCHEMA = "phase314b_r258_stageq_tolerance_aligned_upper_gate_shadow_v1"
BLOCKED_SCHEMA = "phase314b_r258_stageq_tolerance_aligned_upper_gate_shadow_blocked_v1"

BASE_EVIDENCE_COMMIT = "d0a06346e33e4638e98e129a052b8f42f4da92bb"
BASE_IMPLEMENTATION_COMMIT = "901654fb2b72f61a681568274f4fffb69595563d"
EXPECTED_REMOTE = "6758ea7ad800667a436b0243d3b1f6c63256d854"
EXPECTED_SUBMODULE = "633a88752445cf5d6776ed374fdbbdb35f93050c"

BASE_REPORT = "reports/phase3_14b_r258_stagep_oracle_upper_segment_gate_provenance_summary.json"
EXPECTED_BASE_REPORT_SHA256 = "e4a7aba0715a0bb6b8da72654f0bd55d9bda0c67e52f532620f26d6f4f68898c"
EXPECTED_BASE_SCIENTIFIC_RESULT_SHA256 = "1663c33b2430c97ef3f408a3c5de1244e15ee435f02bcd47480e9b919f646031"
EXPECTED_BASE_ROOT_CAUSE = (
    "phase314b_r258_stagep_upper_gate_is_stricter_than_"
    "reconstruction_bound_tolerance"
)
EXPECTED_BASE_NEXT_PATH = "ALIGN_RECONSTRUCTION_BOUND_TOLERANCE_WITH_UPPER_SEGMENT_GATE"
EXPECTED_BASE_CELL_COUNT = 6
EXPECTED_BASE_ATTEMPT_COUNT = 42
EXPECTED_BASE_VIOLATION_COUNT = 619626

SUCCESS_REPORT = "reports/phase3_14b_r258_stageq_tolerance_aligned_upper_gate_shadow_summary.json"
BLOCKED_REPORT = "reports/phase3_14b_r258_stageq_tolerance_aligned_upper_gate_shadow_blocked_summary.json"

IMPLEMENTATION_SUBJECT = (
    "Phase3.14b-r2.5.8 Stage Q: audit tolerance-aligned upper gate"
)
EVIDENCE_SUBJECT = (
    "Record Phase3.14b-r2.5.8 Stage Q tolerance-aligned upper-gate evidence"
)
BLOCKED_EVIDENCE_SUBJECT = (
    "Record Phase3.14b-r2.5.8 Stage Q blocked evidence"
)
BASE_IMPLEMENTATION_SUBJECT = (
    "Phase3.14b-r2.5.8 Stage P: audit oracle upper-segment gate provenance"
)
BASE_EVIDENCE_SUBJECT = (
    "Record Phase3.14b-r2.5.8 Stage P oracle upper-segment evidence"
)

IMPLEMENTATION_PATHS: Tuple[Tuple[str, str], ...] = (
    (
        "A",
        "ccda_phase3/phase314b_r258_stageq_tolerance_aligned_upper_gate_shadow.py",
    ),
    ("A", "scripts/phase3_14b_r258_stageq_execute.py"),
    (
        "A",
        "tests/test_phase3_14b_r258_stageq_tolerance_aligned_upper_gate_shadow.py",
    ),
)
BASE_IMPLEMENTATION_PATHS: Tuple[Tuple[str, str], ...] = (
    (
        "A",
        "ccda_phase3/phase314b_r258_stagep_oracle_upper_segment_gate_provenance.py",
    ),
    ("A", "scripts/phase3_14b_r258_stagep_execute.py"),
    (
        "A",
        "tests/test_phase3_14b_r258_stagep_oracle_upper_segment_gate_provenance.py",
    ),
)
BASE_EVIDENCE_PATHS: Tuple[Tuple[str, str], ...] = (("A", BASE_REPORT),)

LOWER_MULTIPLIER = 0.25
EXPECTED_TIMESTEPS: Tuple[int, ...] = (10, 25, 50)
ORACLE_SOURCES: Tuple[str, ...] = ("raw_oracle", "projected_oracle")
EXPECTED_PAIR_COUNT = 6
EXPECTED_POSITION_SHAPE = (4, 23)

FROZEN_SOURCE_SHA256: Mapping[str, str] = {
    "ccda_phase3/phase314b_r258_stagee_constrained_integrator.py": (
        "37a9567391a308bfe804b95b4aa488ada40e82b2eb94bd03f2060be962cc9eb1"
    ),
    "ccda_phase3/phase314b_r258_stagek_explicit_integrator_predicate_callback.py": (
        "403b4242c6bf7c018e6220bf4e3920e26a9cd945e6bf3871792d88bd7e1f3139"
    ),
    "ccda_phase3/phase314b_r258_stageo_dual_oracle_lower_multiplier_admission.py": (
        "7cdceb909eb599ccd04a7e907940a2ed5b32fd38ae115682ef1bac85f2aa411e"
    ),
    "ccda_phase3/phase314b_r258_stageo_resume1_descending_attempt_order_recovery.py": (
        "c591c4c2d550b3fbdd6877912166131d5bfd9df521ac7f2bb5cac8c8b8c36087"
    ),
    "ccda_phase3/phase314b_r258_stagep_oracle_upper_segment_gate_provenance.py": (
        "3aeea655a68fc08cabae23a1c51dd18efcf0ce8645f3298682e9d8bf50335b4f"
    ),
}

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


class StageQError(RuntimeError):
    """Fail-closed Stage-Q error."""


@dataclass(frozen=True)
class StageQSpec:
    external_multiplier: float = LOWER_MULTIPLIER
    reconstruction_tolerance_factor: float = 5.0
    oracle_admission_rate_min: float = 0.95
    stable_cell_support_min: int = 5
    sparse_other_cell_support_max: int = 1
    binary_tolerance: float = 1.0e-12

    def validate(self) -> None:
        if float(self.external_multiplier) != LOWER_MULTIPLIER:
            raise StageQError("Stage-Q external multiplier changed")
        if float(self.reconstruction_tolerance_factor) != 5.0:
            raise StageQError("reconstruction tolerance factor changed")
        if not 0.0 < float(self.oracle_admission_rate_min) <= 1.0:
            raise StageQError("oracle admission threshold is invalid")
        if int(self.stable_cell_support_min) != 5:
            raise StageQError("stable-cell support changed")
        if int(self.sparse_other_cell_support_max) != 1:
            raise StageQError("sparse-other support changed")
        if float(self.binary_tolerance) < 0.0:
            raise StageQError("binary tolerance is negative")


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
        raise StageQError(f"{label} is not a mapping")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise StageQError(f"{label} is not a sequence")
    return value


def _finite(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise StageQError(f"{label} is not finite")
    return result


def _finite_rate(value: Any, label: str) -> float:
    result = _finite(value, label)
    if not 0.0 <= result <= 1.0:
        raise StageQError(f"{label} is outside [0,1]")
    return result


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise StageQError(f"{label} is boolean")
    result = int(value)
    if result < 0 or float(result) != float(value):
        raise StageQError(f"{label} is not a nonnegative integer")
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
        raise StageQError(
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
        raise StageQError(
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
            raise StageQError("unexpected diff-tree record")
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
            raise StageQError(f"{label} boundary changed: {key}={mapping.get(key)!r}")


def validate_environment_variables() -> Mapping[str, str]:
    mismatch = {
        key: {"expected": expected, "actual": os.environ.get(key)}
        for key, expected in EXPECTED_ENV.items()
        if os.environ.get(key) != expected
    }
    if mismatch:
        raise StageQError(f"deterministic environment mismatch: {mismatch}")
    return dict(EXPECTED_ENV)


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise StageQError(f"cannot load JSON {path}: {error}") from error
    return _mapping(value, str(path))


def _validate_base_report(root: Path) -> Mapping[str, Any]:
    path = Path(root) / BASE_REPORT
    if not path.is_file():
        raise StageQError("Stage-P report is missing")
    if sha256_file(path) != EXPECTED_BASE_REPORT_SHA256:
        raise StageQError("Stage-P report SHA changed")
    if _git_bytes(root, "show", f"{BASE_EVIDENCE_COMMIT}:{BASE_REPORT}") != path.read_bytes():
        raise StageQError("Stage-P report differs from evidence blob")
    report = _load_json(path)
    if report.get("execution_verdict") != "PASS":
        raise StageQError("Stage-P execution is not PASS")
    if report.get("scientific_status") != "BLOCKED":
        raise StageQError("Stage-P scientific status changed")
    if report.get("root_cause") != EXPECTED_BASE_ROOT_CAUSE:
        raise StageQError("Stage-P root cause changed")
    if report.get("required_next_path") != EXPECTED_BASE_NEXT_PATH:
        raise StageQError("Stage-P next path changed")
    if report.get("scientific_result_sha256") != EXPECTED_BASE_SCIENTIFIC_RESULT_SHA256:
        raise StageQError("Stage-P scientific-result SHA changed")
    if report.get("selected_configuration") is not None:
        raise StageQError("Stage-P selected configuration is not null")
    if report.get("train_only_recommendation") is not None:
        raise StageQError("Stage-P recommendation is not null")
    audit = _mapping(
        report.get("upper_segment_gate_provenance"),
        "Stage-P upper-segment provenance",
    )
    classification = _mapping(audit.get("classification"), "Stage-P classification")
    checks = {
        "cell_count": int(classification.get("cell_count", -1)) == EXPECTED_BASE_CELL_COUNT,
        "attempt_count": int(classification.get("internal_scale_attempt_count", -1))
        == EXPECTED_BASE_ATTEMPT_COUNT,
        "callback_cells": int(
            classification.get("callback_upper_failure_cell_count", -1)
        )
        == EXPECTED_BASE_CELL_COUNT,
        "candidate_violation": int(
            classification.get("candidate_gate_violation_count", -1)
        )
        == EXPECTED_BASE_VIOLATION_COUNT,
        "already_control": int(
            classification.get("candidate_fail_already_in_control_count", -1)
        )
        == EXPECTED_BASE_VIOLATION_COUNT,
        "new_violation": int(
            classification.get("candidate_fail_not_in_control_count", -1)
        )
        == 0,
        "within_tolerance": int(
            classification.get(
                "candidate_gate_violation_within_reconstruction_tolerance_count",
                -1,
            )
        )
        == EXPECTED_BASE_VIOLATION_COUNT,
        "beyond_tolerance": int(
            classification.get(
                "candidate_gate_violation_beyond_reconstruction_tolerance_count",
                -1,
            )
        )
        == 0,
        "direct_callback_exact": audit.get(
            "direct_upper_masks_match_stagek_callback"
        )
        is True,
    }
    if not all(checks.values()):
        failed = sorted(key for key, value in checks.items() if not value)
        raise StageQError(f"Stage-P frozen result changed: {failed}")
    require_false(report, FALSE_BOUNDARIES, "Stage-P")
    return report


def _base_cell_map(report: Mapping[str, Any]) -> Mapping[Tuple[str, int], Mapping[str, Any]]:
    audit = _mapping(
        report.get("upper_segment_gate_provenance"),
        "Stage-P upper-segment provenance",
    )
    raw_cells = _sequence(audit.get("cell_records"), "Stage-P cell records")
    output: Dict[Tuple[str, int], Mapping[str, Any]] = {}
    for index, raw in enumerate(raw_cells):
        cell = _mapping(raw, f"Stage-P cell {index}")
        key = (str(cell.get("source_id")), int(cell.get("timestep")))
        if key in output:
            raise StageQError("duplicate Stage-P cell")
        output[key] = cell
    expected = {
        (source, timestep)
        for source in ORACLE_SOURCES
        for timestep in EXPECTED_TIMESTEPS
    }
    if set(output) != expected:
        raise StageQError("Stage-P cell population changed")
    return output


def validate_repository(root: Path) -> Mapping[str, Any]:
    repo = Path(root).resolve()
    if _git(repo, "branch", "--show-current") != "Experiment1":
        raise StageQError("Stage-Q requires Experiment1")
    head = _git(repo, "rev-parse", "HEAD")
    parent = _git(repo, "rev-parse", f"{head}^")
    if parent != BASE_EVIDENCE_COMMIT:
        raise StageQError("Stage-Q implementation parent changed")
    if _git(repo, "show", "-s", "--format=%s", head) != IMPLEMENTATION_SUBJECT:
        raise StageQError("Stage-Q implementation subject changed")
    if commit_name_status(repo, head) != tuple(sorted(IMPLEMENTATION_PATHS)):
        raise StageQError("Stage-Q implementation path population changed")
    if _git(repo, "rev-parse", f"{BASE_EVIDENCE_COMMIT}^") != BASE_IMPLEMENTATION_COMMIT:
        raise StageQError("Stage-P evidence parent changed")
    if _git(repo, "show", "-s", "--format=%s", BASE_IMPLEMENTATION_COMMIT) != (
        BASE_IMPLEMENTATION_SUBJECT
    ):
        raise StageQError("Stage-P implementation subject changed")
    if commit_name_status(repo, BASE_IMPLEMENTATION_COMMIT) != tuple(
        sorted(BASE_IMPLEMENTATION_PATHS)
    ):
        raise StageQError("Stage-P implementation paths changed")
    if _git(repo, "show", "-s", "--format=%s", BASE_EVIDENCE_COMMIT) != (
        BASE_EVIDENCE_SUBJECT
    ):
        raise StageQError("Stage-P evidence subject changed")
    if commit_name_status(repo, BASE_EVIDENCE_COMMIT) != tuple(
        sorted(BASE_EVIDENCE_PATHS)
    ):
        raise StageQError("Stage-P evidence paths changed")
    if _git(repo, "rev-parse", "origin/Experiment1") != EXPECTED_REMOTE:
        raise StageQError("origin/Experiment1 changed")
    submodule = repo / "external/deformable-ravens"
    if _git(submodule, "rev-parse", "HEAD") != EXPECTED_SUBMODULE:
        raise StageQError("DeformableRavens commit changed")
    if status_paths(submodule):
        raise StageQError("DeformableRavens worktree is dirty")
    if status_paths(repo):
        raise StageQError("Stage-Q worktree must be clean before execution")
    source_sha: Dict[str, str] = {}
    for relative, expected in FROZEN_SOURCE_SHA256.items():
        path = repo / relative
        if not path.is_file():
            raise StageQError(f"frozen source is missing: {relative}")
        actual = sha256_file(path)
        if actual != expected:
            raise StageQError(f"frozen source SHA changed: {relative}")
        source_sha[relative] = actual
    base_report = _validate_base_report(repo)
    for relative in (SUCCESS_REPORT, BLOCKED_REPORT):
        if (repo / relative).exists():
            raise StageQError(f"Stage-Q output already exists: {relative}")
    return {
        "root": str(repo),
        "head": head,
        "parent": parent,
        "origin_experiment1": EXPECTED_REMOTE,
        "submodule_commit": EXPECTED_SUBMODULE,
        "frozen_source_sha256": source_sha,
        "base_report_sha256": EXPECTED_BASE_REPORT_SHA256,
        "base_scientific_result_sha256": EXPECTED_BASE_SCIENTIFIC_RESULT_SHA256,
        "base_cell_count": len(_base_cell_map(base_report)),
    }


def scalar_stats(value: np.ndarray) -> Mapping[str, Any]:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise StageQError("cannot summarize an empty or non-finite array")
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


def zero_stats() -> Mapping[str, Any]:
    return {
        "count": 0,
        "mean": 0.0,
        "std": 0.0,
        "min": 0.0,
        "p05": 0.0,
        "median": 0.0,
        "p95": 0.0,
        "max": 0.0,
    }


def tolerant_upper_contract(
    *,
    upper_bound: np.ndarray,
    center_log: np.ndarray,
    scale_log: np.ndarray,
    segment_tolerance: float,
    tolerance_factor: float,
) -> Mapping[str, Any]:
    upper = np.asarray(upper_bound, dtype=np.float64)
    center = np.asarray(center_log, dtype=np.float64)
    scale = np.asarray(scale_log, dtype=np.float64)
    if upper.shape != EXPECTED_POSITION_SHAPE:
        raise StageQError(f"upper-bound shape changed: {upper.shape}")
    if center.shape != upper.shape or scale.shape != upper.shape:
        raise StageQError("reference/upper-bound shape differs")
    if not np.all(np.isfinite(upper)) or np.any(upper <= 0.0):
        raise StageQError("upper bound is invalid")
    if not np.all(np.isfinite(center)):
        raise StageQError("reference center is invalid")
    if not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
        raise StageQError("reference scale is invalid")
    tolerance = float(segment_tolerance) * float(tolerance_factor)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise StageQError("aligned upper tolerance is invalid")
    tolerant_length = upper + tolerance
    if not np.all(np.isfinite(tolerant_length)) or np.any(tolerant_length <= upper):
        raise StageQError("tolerant upper bound is invalid")
    z_threshold = (np.log(tolerant_length) - center) / scale
    if not np.all(np.isfinite(z_threshold)):
        raise StageQError("position-wise tolerant z threshold is non-finite")
    return {
        "length_upper": tolerant_length,
        "position_z_threshold": z_threshold,
        "length_upper_sha256": sha256_array(tolerant_length),
        "position_z_threshold_sha256": sha256_array(z_threshold),
        "length_upper_stats": scalar_stats(tolerant_length),
        "position_z_threshold_stats": scalar_stats(z_threshold),
        "length_tolerance": tolerance,
    }


def aligned_upper_masks(
    *,
    lengths: np.ndarray,
    z_scores: np.ndarray,
    contract: Mapping[str, Any],
) -> Mapping[str, Any]:
    length = np.asarray(lengths, dtype=np.float64)
    z = np.asarray(z_scores, dtype=np.float64)
    if length.ndim != 3 or length.shape[1:] != EXPECTED_POSITION_SHAPE:
        raise StageQError(f"segment-length shape changed: {length.shape}")
    if z.shape == (length.shape[0], 1, *EXPECTED_POSITION_SHAPE):
        z = z[:, 0]
    if z.shape != length.shape:
        raise StageQError(f"z/length shape differs: {z.shape} vs {length.shape}")
    length_upper = np.asarray(contract["length_upper"], dtype=np.float64)
    z_threshold = np.asarray(contract["position_z_threshold"], dtype=np.float64)
    if length_upper.shape != EXPECTED_POSITION_SHAPE:
        raise StageQError("aligned length-upper shape changed")
    if z_threshold.shape != EXPECTED_POSITION_SHAPE:
        raise StageQError("aligned z-threshold shape changed")
    length_element_pass = length <= length_upper[None]
    z_element_pass = z <= z_threshold[None]
    length_row_pass = np.all(length_element_pass, axis=(1, 2))
    z_row_pass = np.all(z_element_pass, axis=(1, 2))
    mismatch = length_element_pass != z_element_pass
    return {
        "length_element_pass": length_element_pass,
        "z_element_pass": z_element_pass,
        "length_row_pass": length_row_pass,
        "z_row_pass": z_row_pass,
        "element_equivalence": bool(not np.any(mismatch)),
        "row_equivalence": bool(np.array_equal(length_row_pass, z_row_pass)),
        "element_mismatch_count": int(np.count_nonzero(mismatch)),
        "row_mismatch_count": int(np.count_nonzero(length_row_pass != z_row_pass)),
        "length_row_pass_sha256": sha256_array(length_row_pass.astype(np.bool_)),
        "z_row_pass_sha256": sha256_array(z_row_pass.astype(np.bool_)),
        "length_element_pass_sha256": sha256_array(
            length_element_pass.astype(np.bool_)
        ),
        "z_element_pass_sha256": sha256_array(z_element_pass.astype(np.bool_)),
    }


def _predicate_masks(
    *,
    metrics: Mapping[str, Any],
    aligned_upper_pass: np.ndarray,
    bound_pass: Optional[np.ndarray],
    coordinate_possible: Optional[np.ndarray],
) -> Mapping[str, np.ndarray]:
    rows = int(np.asarray(aligned_upper_pass).shape[0])
    coordinate_recenter = (
        np.ones(rows, dtype=np.bool_)
        if coordinate_possible is None
        else np.asarray(coordinate_possible, dtype=np.bool_)
    )
    reconstruction_bounds = (
        np.ones(rows, dtype=np.bool_)
        if bound_pass is None
        else np.asarray(bound_pass, dtype=np.bool_)
    )
    coordinate_geometry = np.where(
        coordinate_recenter,
        np.asarray(metrics["coordinate_pass"], dtype=np.bool_),
        True,
    )
    segment_geometry = np.where(
        reconstruction_bounds,
        np.asarray(metrics["segment_pass"], dtype=np.bool_),
        True,
    )
    return {
        "finite_state": np.asarray(metrics["finite"], dtype=np.bool_),
        "upper_segment_geometry": np.asarray(aligned_upper_pass, dtype=np.bool_),
        "lower_segment_geometry": np.asarray(metrics["lower_pass"], dtype=np.bool_),
        "coordinate_recenter": coordinate_recenter,
        "coordinate_geometry": coordinate_geometry,
        "reconstruction_bounds": reconstruction_bounds,
        "segment_geometry": segment_geometry,
        "direction_retention": np.asarray(
            metrics["retention_pass"], dtype=np.bool_
        ),
        "displacement": np.asarray(metrics["nonzero_move"], dtype=np.bool_),
    }


def sequential_attempt_summary(
    *,
    active: np.ndarray,
    predicate_masks: Mapping[str, np.ndarray],
    topology_pass: np.ndarray,
) -> Mapping[str, Any]:
    active_mask = np.asarray(active, dtype=np.bool_)
    rows = int(active_mask.shape[0])
    if set(predicate_masks) != set(PREDICATE_ORDER[:-1]):
        raise StageQError("shadow predicate population changed")
    unresolved = active_mask.copy()
    first_failed: Dict[str, int] = {}
    reached: Dict[str, int] = {}
    passed: Dict[str, int] = {}
    for name in PREDICATE_ORDER[:-1]:
        mask = np.asarray(predicate_masks[name], dtype=np.bool_)
        if mask.shape != (rows,):
            raise StageQError(f"shadow predicate shape changed: {name}")
        reached[name] = int(np.count_nonzero(unresolved))
        failures = unresolved & ~mask
        first_failed[name] = int(np.count_nonzero(failures))
        unresolved &= mask
        passed[name] = int(np.count_nonzero(unresolved))
    topology = np.asarray(topology_pass, dtype=np.bool_)
    if topology.shape != (rows,):
        raise StageQError("shadow topology shape changed")
    reached["topology"] = int(np.count_nonzero(unresolved))
    topology_fail = unresolved & ~topology
    first_failed["topology"] = int(np.count_nonzero(topology_fail))
    accepted = unresolved & topology
    passed["topology"] = int(np.count_nonzero(accepted))
    active_count = int(np.count_nonzero(active_mask))
    accepted_count = int(np.count_nonzero(accepted))
    if sum(first_failed.values()) + accepted_count != active_count:
        raise StageQError("shadow sequential population does not close")
    return {
        "active_row_count": active_count,
        "accepted_row_count": accepted_count,
        "first_failed_counts": first_failed,
        "sequential_reach_counts": reached,
        "sequential_pass_counts": passed,
        "accepted_mask": accepted,
        "population_closed": True,
    }


def _attempt_events(capture: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    values = _sequence(capture.get("events"), "callback events")
    attempts = [
        _mapping(value, f"callback event {index}")
        for index, value in enumerate(values)
        if isinstance(value, Mapping) and value.get("event_type") == "scale_attempt"
    ]
    if not attempts:
        raise StageQError("callback capture lacks scale attempts")
    return attempts


def _expected_internal_order(runtime: Mapping[str, Any]) -> Tuple[float, ...]:
    integrator_spec = runtime["integrator_spec"]
    definition = runtime["stagef"].fixed_integrator_definition()
    eligible = [
        float(scale)
        for scale in integrator_spec.scale_grid
        if float(scale)
        <= float(definition.maximum_scale)
        + float(integrator_spec.standardizer_epsilon)
    ]
    expected = tuple(sorted(eligible, reverse=True))
    if not expected or len(set(expected)) != len(expected):
        raise StageQError("frozen internal-scale population is invalid")
    return expected


def _legacy_scale_candidate_map(integration: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
    records = _sequence(integration.get("scale_candidates"), "legacy scale candidates")
    output: Dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(records):
        record = _mapping(raw, f"legacy scale candidate {index}")
        key = _scale_key(float(record.get("scale")))
        if key in output:
            raise StageQError("duplicate legacy scale candidate")
        output[key] = record
    return output


def _base_identity_check(
    *,
    source_id: str,
    timestep: int,
    integration: Mapping[str, Any],
    capture: Mapping[str, Any],
    base_cell: Mapping[str, Any],
    stageo: Any,
) -> Mapping[str, Any]:
    selected = np.asarray(integration["selected_scale"], dtype=np.float64)
    candidate = np.asarray(integration["candidate"], dtype=np.float32)
    checks = {
        "source_id": source_id == str(base_cell.get("source_id")),
        "timestep": int(timestep) == int(base_cell.get("timestep")),
        "selected_scale_sha256": stageo.sha256_array(selected)
        == base_cell.get("final_selected_scale_sha256"),
        "candidate_sha256": stageo.sha256_array(candidate)
        == base_cell.get("final_candidate_sha256"),
        "callback_capture_sha256": str(capture.get("events_sha256"))
        == str(base_cell.get("callback_capture_sha256")),
        "callback_result_bit_exact": capture.get("returned_result_bit_exact") is True,
    }
    if not all(checks.values()):
        failed = sorted(key for key, value in checks.items() if not value)
        raise StageQError(f"Stage-Q replay differs from Stage-P: {failed}")
    return {"all_exact": True, "checks": checks}


def audit_shadow_cell(
    *,
    source_id: str,
    timestep: int,
    control: np.ndarray,
    base_direction: np.ndarray,
    context: Mapping[str, Any],
    runtime: Mapping[str, Any],
    base_cell: Mapping[str, Any],
    spec: StageQSpec,
) -> Mapping[str, Any]:
    stageo = runtime["stageo"]
    stagek = runtime["stagek"]
    stagee = runtime["stagel"].stagee258
    staged = stagee.staged
    stageb = stagee.stageb
    stagep = runtime["stagep"]
    integrator_spec = runtime["integrator_spec"]
    callback_spec = runtime["callback_spec"]
    definition = runtime["stagef"].fixed_integrator_definition()
    definition.validate()

    control_raw = np.asarray(control, dtype=np.float32)
    direction_raw = np.asarray(base_direction, dtype=np.float64)
    if control_raw.shape != direction_raw.shape:
        raise StageQError("control/oracle direction shape changed")
    proposed_direction = direction_raw * float(spec.external_multiplier)
    legacy_integration, capture = stagek.callback_integrate_rowwise(
        control=control_raw,
        direction=proposed_direction,
        context=context,
        integrator_spec=integrator_spec,
        callback_spec=callback_spec,
    )
    identity = _base_identity_check(
        source_id=source_id,
        timestep=timestep,
        integration=legacy_integration,
        capture=capture,
        base_cell=base_cell,
        stageo=stageo,
    )
    attempts = _attempt_events(capture)
    expected_order = _expected_internal_order(runtime)
    observed_order = tuple(float(item.get("attempted_scale")) for item in attempts)
    if observed_order != expected_order:
        raise StageQError("callback attempt order differs from frozen Stage-E order")
    base_attempts = _sequence(base_cell.get("attempt_records"), "Stage-P attempts")
    if len(base_attempts) != len(expected_order):
        raise StageQError("Stage-P internal-scale population changed")

    legacy_selected_scale = np.asarray(
        legacy_integration["selected_scale"], dtype=np.float64
    ).reshape(control_raw.shape[0])
    legacy_scale_candidates = _legacy_scale_candidate_map(legacy_integration)
    bounds = stagee.segment_bounds(definition=definition, context=context)
    upper_bound = np.asarray(bounds["upper"], dtype=np.float64)
    lower_bound = np.asarray(bounds["lower"], dtype=np.float64)
    reference = context["stage_d_contract"].reference
    tolerance_contract = tolerant_upper_contract(
        upper_bound=upper_bound,
        center_log=np.asarray(reference.center_log, dtype=np.float64),
        scale_log=np.asarray(reference.scale_log, dtype=np.float64),
        segment_tolerance=float(integrator_spec.segment_tolerance),
        tolerance_factor=float(spec.reconstruction_tolerance_factor),
    )
    strict_threshold = float(context["upper_gate"].upper_threshold)

    control_lengths = stageb.segment_lengths(control_raw).astype(np.float64)
    control_scores = staged.segment_scores(control_raw, reference)
    control_aligned = aligned_upper_masks(
        lengths=control_lengths,
        z_scores=np.asarray(control_scores["z"], dtype=np.float64),
        contract=tolerance_contract,
    )
    control_strict = (
        np.asarray(control_scores["upper"], dtype=np.float64)[:, 0]
        <= strict_threshold
    )
    control_strict_fail_aligned_pass = (~control_strict) & np.asarray(
        control_aligned["length_row_pass"], dtype=np.bool_
    )
    control_strict_pass_aligned_fail = control_strict & ~np.asarray(
        control_aligned["length_row_pass"], dtype=np.bool_
    )

    shadow_selected = np.zeros(control_raw.shape[0], dtype=np.bool_)
    shadow_selected_scale = np.zeros(control_raw.shape[0], dtype=np.float64)
    shadow_output = control_raw.copy()
    attempt_records: List[Mapping[str, Any]] = []
    cell_first_failure_counts = {name: 0 for name in PREDICATE_ORDER}
    strict_element_failures = 0
    aligned_element_failures = 0
    equivalence_mismatches = 0
    strict_pass_aligned_fail_rows = 0
    strict_fail_aligned_pass_rows = 0

    for index, (event, scale, raw_base_attempt) in enumerate(
        zip(attempts, expected_order, base_attempts)
    ):
        base_attempt = _mapping(raw_base_attempt, f"Stage-P attempt {index}")
        if float(base_attempt.get("internal_scale")) != float(scale):
            raise StageQError("Stage-P attempt order changed")
        raw_proposal = (
            control_raw.astype(np.float64) + float(scale) * proposed_direction
        ).astype(np.float32)
        reconstruction = stagee.reconstruct_segment_vectors(
            proposed=raw_proposal,
            control=control_raw,
            lower=lower_bound,
            upper=upper_bound,
            coordinate_abs_max=float(
                context["historical_geometry"].coordinate_abs_max
            ),
            epsilon=float(integrator_spec.segment_tolerance),
        )
        candidate = np.asarray(reconstruction["candidate"], dtype=np.float32)
        bound_pass = np.asarray(reconstruction["bound_pass"], dtype=np.bool_)
        coordinate_possible = np.asarray(
            reconstruction["coordinate_possible"], dtype=np.bool_
        )
        candidate_sha = stagee.sha256_array(candidate)
        scale_key = _scale_key(scale)
        legacy_scale = _mapping(
            legacy_scale_candidates.get(scale_key),
            f"legacy scale candidate {scale_key}",
        )
        candidate_checks = {
            "legacy_scale_candidate_sha": candidate_sha
            == legacy_scale.get("candidate_sha256"),
            "stagep_candidate_sha": candidate_sha
            == base_attempt.get("reconstructed_candidate_sha256"),
            "raw_proposal_sha": sha256_array(raw_proposal)
            == base_attempt.get("raw_proposal_sha256"),
        }
        if not all(candidate_checks.values()):
            failed = sorted(key for key, value in candidate_checks.items() if not value)
            raise StageQError(f"candidate generation differs from frozen replay: {failed}")

        metrics = stagee.observable_row_metrics(
            candidate=candidate,
            raw_proposal=raw_proposal,
            control=control_raw,
            direction=proposed_direction,
            definition=definition,
            context=context,
            spec=integrator_spec,
            bound_pass=bound_pass,
            coordinate_possible=coordinate_possible,
            include_topology=False,
        )
        candidate_lengths = stageb.segment_lengths(candidate).astype(np.float64)
        candidate_scores = staged.segment_scores(candidate, reference)
        aligned = aligned_upper_masks(
            lengths=candidate_lengths,
            z_scores=np.asarray(candidate_scores["z"], dtype=np.float64),
            contract=tolerance_contract,
        )
        aligned_upper_pass = np.asarray(
            aligned["length_row_pass"], dtype=np.bool_
        )
        strict_upper_pass = np.asarray(metrics["upper_pass"], dtype=np.bool_)
        strict_element_pass = (
            np.asarray(candidate_scores["z"], dtype=np.float64)[:, 0]
            <= strict_threshold
        )
        aligned_element_pass = np.asarray(
            aligned["length_element_pass"], dtype=np.bool_
        )
        strict_element_failure_count = int(np.count_nonzero(~strict_element_pass))
        aligned_element_failure_count = int(np.count_nonzero(~aligned_element_pass))
        strict_element_failures += strict_element_failure_count
        aligned_element_failures += aligned_element_failure_count
        equivalence_mismatches += int(aligned["element_mismatch_count"])
        strict_pass_aligned_fail = strict_upper_pass & ~aligned_upper_pass
        strict_fail_aligned_pass = ~strict_upper_pass & aligned_upper_pass
        strict_pass_aligned_fail_rows += int(
            np.count_nonzero(strict_pass_aligned_fail)
        )
        strict_fail_aligned_pass_rows += int(
            np.count_nonzero(strict_fail_aligned_pass)
        )

        legacy_selected_before = legacy_selected_scale > float(scale)
        callback_failures = _mapping(
            event.get("first_failed_counts"), "callback first-failed counts"
        )
        legacy_direct_upper_fail = (
            (~legacy_selected_before)
            & np.asarray(metrics["finite"], dtype=np.bool_)
            & ~strict_upper_pass
        )
        if int(callback_failures.get("upper_segment_geometry", -1)) != int(
            np.count_nonzero(legacy_direct_upper_fail)
        ):
            raise StageQError("legacy callback upper first-failure count changed")

        active = ~shadow_selected
        masks = _predicate_masks(
            metrics=metrics,
            aligned_upper_pass=aligned_upper_pass,
            bound_pass=bound_pass,
            coordinate_possible=coordinate_possible,
        )
        fast = active.copy()
        for name in PREDICATE_ORDER[:-1]:
            fast &= np.asarray(masks[name], dtype=np.bool_)
        topology_pass = np.zeros(control_raw.shape[0], dtype=np.bool_)
        checked = np.flatnonzero(fast)
        if checked.size:
            topology_pass[checked] = stageb.physical_validity(
                candidate[checked, None],
                context["historical_geometry"],
            )["topology"][:, 0]
        sequential = sequential_attempt_summary(
            active=active,
            predicate_masks=masks,
            topology_pass=topology_pass,
        )
        choose = np.asarray(sequential["accepted_mask"], dtype=np.bool_)
        if not np.array_equal(choose, fast & topology_pass):
            raise StageQError("shadow topology/acceptance decomposition changed")
        shadow_output[choose] = candidate[choose]
        shadow_selected_scale[choose] = float(scale)
        shadow_selected[choose] = True
        for name, count in _mapping(
            sequential["first_failed_counts"], "shadow first-failure counts"
        ).items():
            cell_first_failure_counts[str(name)] += int(count)

        attempt_records.append(
            {
                "internal_scale": float(scale),
                "effective_multiplier": float(scale * spec.external_multiplier),
                "candidate_sha256": candidate_sha,
                "candidate_generation_identity": candidate_checks,
                "legacy_active_row_count": int(
                    np.count_nonzero(~legacy_selected_before)
                ),
                "shadow_active_row_count": int(np.count_nonzero(active)),
                "legacy_strict_upper_row_pass_count": int(
                    np.count_nonzero(strict_upper_pass & ~legacy_selected_before)
                ),
                "shadow_aligned_upper_row_pass_count": int(
                    np.count_nonzero(aligned_upper_pass & active)
                ),
                "strict_upper_element_failure_count": strict_element_failure_count,
                "aligned_upper_element_failure_count": aligned_element_failure_count,
                "strict_fail_aligned_pass_row_count": int(
                    np.count_nonzero(strict_fail_aligned_pass)
                ),
                "strict_pass_aligned_fail_row_count": int(
                    np.count_nonzero(strict_pass_aligned_fail)
                ),
                "length_log_z_element_equivalence": aligned[
                    "element_equivalence"
                ],
                "length_log_z_row_equivalence": aligned["row_equivalence"],
                "length_log_z_element_mismatch_count": aligned[
                    "element_mismatch_count"
                ],
                "length_log_z_row_mismatch_count": aligned[
                    "row_mismatch_count"
                ],
                "aligned_length_row_pass_sha256": aligned[
                    "length_row_pass_sha256"
                ],
                "aligned_z_row_pass_sha256": aligned["z_row_pass_sha256"],
                "shadow_topology_checked_count": int(checked.size),
                "shadow_accepted_count": int(np.count_nonzero(choose)),
                "shadow_first_failed_counts": dict(
                    sequential["first_failed_counts"]
                ),
                "shadow_population_closed": True,
                "callback_events_persisted": False,
                "candidate_tensor_persisted": False,
                "predicate_mask_persisted": False,
            }
        )

    shadow_acceptance = float(np.mean(shadow_selected))
    nonupper_counts = {
        key: value
        for key, value in cell_first_failure_counts.items()
        if key != "upper_segment_geometry"
    }
    dominant = max(nonupper_counts, key=nonupper_counts.get)
    dominant_count = int(nonupper_counts[dominant])
    rejected_attempt_rows = int(sum(cell_first_failure_counts.values()))
    dominant_mass = (
        0.0
        if rejected_attempt_rows == 0
        else float(dominant_count / rejected_attempt_rows)
    )

    return {
        "source_id": source_id,
        "timestep": int(timestep),
        "external_multiplier": float(spec.external_multiplier),
        "objective_train_rows": int(control_raw.shape[0]),
        "legacy_identity": identity,
        "legacy_callback_result_bit_exact": True,
        "legacy_selected_scale_sha256": stageo.sha256_array(legacy_selected_scale),
        "legacy_candidate_sha256": str(legacy_integration["candidate_sha256"]),
        "legacy_callback_capture_sha256": str(capture["events_sha256"]),
        "legacy_oracle_acceptance_rate": float(
            np.mean(legacy_selected_scale > spec.binary_tolerance)
        ),
        "aligned_oracle_acceptance_rate": shadow_acceptance,
        "aligned_oracle_admitted": bool(
            shadow_acceptance >= spec.oracle_admission_rate_min
        ),
        "aligned_selected_scale_sha256": sha256_array(shadow_selected_scale),
        "aligned_candidate_sha256": sha256_array(shadow_output),
        "internal_scale_attempt_order": list(expected_order),
        "tolerance_contract": {
            key: value
            for key, value in tolerance_contract.items()
            if not isinstance(value, np.ndarray)
        },
        "control_gate_comparison": {
            "strict_upper_failure_count": int(np.count_nonzero(~control_strict)),
            "aligned_upper_failure_count": int(
                np.count_nonzero(~control_aligned["length_row_pass"])
            ),
            "strict_fail_aligned_pass_count": int(
                np.count_nonzero(control_strict_fail_aligned_pass)
            ),
            "strict_pass_aligned_fail_count": int(
                np.count_nonzero(control_strict_pass_aligned_fail)
            ),
            "length_log_z_element_equivalence": control_aligned[
                "element_equivalence"
            ],
            "length_log_z_element_mismatch_count": control_aligned[
                "element_mismatch_count"
            ],
        },
        "strict_upper_element_failure_count": strict_element_failures,
        "aligned_upper_element_failure_count": aligned_element_failures,
        "length_log_z_element_mismatch_count": equivalence_mismatches,
        "strict_pass_aligned_fail_row_count": strict_pass_aligned_fail_rows,
        "strict_fail_aligned_pass_row_count": strict_fail_aligned_pass_rows,
        "shadow_first_failed_counts": cell_first_failure_counts,
        "dominant_post_upper_failure_predicate": (
            None if dominant_count == 0 else dominant
        ),
        "dominant_post_upper_failure_count": dominant_count,
        "dominant_post_upper_failure_mass": dominant_mass,
        "attempt_records": attempt_records,
        "callback_events_persisted": False,
        "row_identity_persisted": False,
        "direction_tensor_persisted": False,
        "proposal_tensor_persisted": False,
        "candidate_tensor_persisted": False,
        "predicate_mask_persisted": False,
    }


def classify_shadow_alignment(
    cells: Sequence[Mapping[str, Any]],
    spec: Optional[StageQSpec] = None,
) -> Mapping[str, Any]:
    active = StageQSpec() if spec is None else spec
    active.validate()
    if len(cells) != EXPECTED_PAIR_COUNT:
        raise StageQError("Stage-Q cell population changed")

    element_mismatch_count = 0
    strict_pass_aligned_fail_count = 0
    strict_fail_aligned_pass_count = 0
    strict_element_failures = 0
    aligned_element_failures = 0
    aligned_admitted_cells = 0
    aligned_upper_failure_cells = 0
    dominant_counts: Dict[str, int] = {}
    by_timestep: Dict[str, Dict[str, int]] = {
        str(timestep): {} for timestep in EXPECTED_TIMESTEPS
    }
    by_source: Dict[str, Dict[str, int]] = {
        source: {} for source in ORACLE_SOURCES
    }
    for cell in cells:
        source = str(cell.get("source_id"))
        timestep = int(cell.get("timestep"))
        if source not in ORACLE_SOURCES or timestep not in EXPECTED_TIMESTEPS:
            raise StageQError("unexpected Stage-Q source/timestep")
        element_mismatch_count += _nonnegative_int(
            cell.get("length_log_z_element_mismatch_count"),
            "length/log-z mismatch count",
        )
        strict_pass_aligned_fail_count += _nonnegative_int(
            cell.get("strict_pass_aligned_fail_row_count"),
            "strict-pass aligned-fail count",
        )
        strict_fail_aligned_pass_count += _nonnegative_int(
            cell.get("strict_fail_aligned_pass_row_count"),
            "strict-fail aligned-pass count",
        )
        strict_element_failures += _nonnegative_int(
            cell.get("strict_upper_element_failure_count"),
            "strict upper element failures",
        )
        aligned_element_failures += _nonnegative_int(
            cell.get("aligned_upper_element_failure_count"),
            "aligned upper element failures",
        )
        if cell.get("aligned_oracle_admitted") is True:
            aligned_admitted_cells += 1
        if int(cell.get("aligned_upper_element_failure_count", 0)) > 0:
            aligned_upper_failure_cells += 1
        predicate = cell.get("dominant_post_upper_failure_predicate")
        if isinstance(predicate, str):
            dominant_counts[predicate] = dominant_counts.get(predicate, 0) + 1
            by_source[source][predicate] = by_source[source].get(predicate, 0) + 1
            by_timestep[str(timestep)][predicate] = (
                by_timestep[str(timestep)].get(predicate, 0) + 1
            )

    if strict_element_failures != EXPECTED_BASE_VIOLATION_COUNT:
        raise StageQError("Stage-Q aggregate strict violation count changed")

    dominant_predicate: Optional[str] = None
    dominant_support = 0
    other_max = 0
    timestep_covered = False
    if dominant_counts:
        dominant_predicate = max(
            dominant_counts,
            key=lambda key: (
                dominant_counts[key],
                -PREDICATE_ORDER.index(key),
            ),
        )
        dominant_support = int(dominant_counts[dominant_predicate])
        other_max = max(
            (
                count
                for key, count in dominant_counts.items()
                if key != dominant_predicate
            ),
            default=0,
        )
        timestep_covered = all(
            int(by_timestep[str(t)].get(dominant_predicate, 0)) >= 1
            for t in EXPECTED_TIMESTEPS
        )

    predicate_outcomes: Mapping[str, Tuple[str, str]] = {
        "finite_state": (
            "phase314b_r258_stageq_tolerance_alignment_exposes_nonfinite_state",
            "AUDIT_ORACLE_FINITE_STATE_AFTER_UPPER_TOLERANCE_ALIGNMENT",
        ),
        "lower_segment_geometry": (
            "phase314b_r258_stageq_tolerance_alignment_exposes_lower_segment_gate",
            "AUDIT_LOWER_SEGMENT_GATE_AFTER_UPPER_TOLERANCE_ALIGNMENT",
        ),
        "coordinate_recenter": (
            "phase314b_r258_stageq_tolerance_alignment_exposes_coordinate_recenter",
            "AUDIT_ORACLE_LOCAL_FRAME_AFTER_UPPER_TOLERANCE_ALIGNMENT",
        ),
        "coordinate_geometry": (
            "phase314b_r258_stageq_tolerance_alignment_exposes_coordinate_geometry",
            "AUDIT_ORACLE_LOCAL_FRAME_AFTER_UPPER_TOLERANCE_ALIGNMENT",
        ),
        "reconstruction_bounds": (
            "phase314b_r258_stageq_tolerance_alignment_exposes_reconstruction_bounds",
            "AUDIT_RECONSTRUCTION_BOUND_CLOSURE_AFTER_UPPER_TOLERANCE_ALIGNMENT",
        ),
        "segment_geometry": (
            "phase314b_r258_stageq_tolerance_alignment_exposes_strict_segment_geometry_gate",
            "ALIGN_HISTORICAL_SEGMENT_BOUND_TOLERANCE_WITH_RECONSTRUCTION_CONTRACT",
        ),
        "direction_retention": (
            "phase314b_r258_stageq_tolerance_alignment_exposes_direction_retention",
            "AUDIT_ORACLE_DIRECTION_RETENTION_AFTER_UPPER_TOLERANCE_ALIGNMENT",
        ),
        "displacement": (
            "phase314b_r258_stageq_tolerance_alignment_exposes_displacement_gate",
            "AUDIT_ORACLE_DISPLACEMENT_AFTER_UPPER_TOLERANCE_ALIGNMENT",
        ),
        "topology": (
            "phase314b_r258_stageq_tolerance_alignment_exposes_topology_gate",
            "AUDIT_ORACLE_TOPOLOGY_AFTER_UPPER_TOLERANCE_ALIGNMENT",
        ),
    }

    if element_mismatch_count > 0:
        root = "phase314b_r258_stageq_length_and_log_z_tolerance_masks_not_equivalent"
        next_path = "AUDIT_LENGTH_LOG_Z_UPPER_TOLERANCE_NUMERICS"
        primary = "length_log_z_boolean_mismatch"
    elif strict_pass_aligned_fail_count > 0:
        root = "phase314b_r258_stageq_tolerance_alignment_introduces_upper_gate_regression"
        next_path = "AUDIT_TOLERANCE_ALIGNED_UPPER_GATE_REGRESSION"
        primary = "strict_pass_aligned_fail_regression"
    elif aligned_upper_failure_cells > 0 or aligned_element_failures > 0:
        root = (
            "phase314b_r258_stageq_aligned_upper_gate_does_not_clear_"
            "reconstruction_tolerance_population"
        )
        next_path = "AUDIT_TOLERANCE_ALIGNED_UPPER_GATE_IMPLEMENTATION"
        primary = "aligned_upper_violation_remains"
    elif aligned_admitted_cells == EXPECTED_PAIR_COUNT:
        root = (
            "phase314b_r258_stageq_tolerance_alignment_restores_"
            "dual_oracle_admission"
        )
        next_path = (
            "REOPEN_LOWER_MULTIPLIER_OOF_POST_UPPER_AUDIT_WITH_"
            "TOLERANCE_ALIGNED_GATE"
        )
        primary = "dual_oracle_admission_restored"
    elif 0 < aligned_admitted_cells < EXPECTED_PAIR_COUNT:
        root = (
            "phase314b_r258_stageq_tolerance_aligned_oracle_admission_is_"
            "source_or_timestep_dependent"
        )
        next_path = (
            "STRATIFY_TOLERANCE_ALIGNED_ORACLE_ADMISSION_BY_SOURCE_AND_TIMESTEP"
        )
        primary = "heterogeneous_aligned_oracle_admission"
    elif (
        dominant_predicate is not None
        and dominant_support >= active.stable_cell_support_min
        and other_max <= active.sparse_other_cell_support_max
        and timestep_covered
    ):
        root, next_path = predicate_outcomes[dominant_predicate]
        primary = f"post_upper_rejection::{dominant_predicate}"
    elif dominant_counts:
        root = (
            "phase314b_r258_stageq_post_upper_rejection_is_source_or_"
            "timestep_dependent"
        )
        next_path = (
            "STRATIFY_TOLERANCE_ALIGNED_POST_UPPER_REJECTION_BY_SOURCE_AND_TIMESTEP"
        )
        primary = "heterogeneous_post_upper_rejection"
    else:
        root = "phase314b_r258_stageq_tolerance_aligned_scalar_surface_insufficient"
        next_path = "ADD_READ_ONLY_ROW_COHORT_HASHES_AFTER_UPPER_TOLERANCE_ALIGNMENT"
        primary = "aligned_scalar_surface_insufficient"

    return {
        "root_cause": root,
        "required_next_path": next_path,
        "primary_failure_locus": primary,
        "cell_count": len(cells),
        "strict_upper_element_failure_count": strict_element_failures,
        "aligned_upper_element_failure_count": aligned_element_failures,
        "strict_fail_aligned_pass_row_count": strict_fail_aligned_pass_count,
        "strict_pass_aligned_fail_row_count": strict_pass_aligned_fail_count,
        "length_log_z_element_mismatch_count": element_mismatch_count,
        "aligned_upper_failure_cell_count": aligned_upper_failure_cells,
        "aligned_oracle_admitted_cell_count": aligned_admitted_cells,
        "dominant_post_upper_failure_predicate": dominant_predicate,
        "dominant_post_upper_failure_support_count": dominant_support,
        "maximum_other_post_upper_failure_support_count": other_max,
        "dominant_post_upper_failure_has_timestep_coverage": timestep_covered,
        "post_upper_failure_counts": dict(sorted(dominant_counts.items())),
        "post_upper_failure_counts_by_source": {
            key: dict(sorted(value.items())) for key, value in by_source.items()
        },
        "post_upper_failure_counts_by_timestep": {
            key: dict(sorted(value.items())) for key, value in by_timestep.items()
        },
        "classification_spec": asdict(active),
    }


def probe_environment(root: Path) -> Mapping[str, Any]:
    from ccda_phase3 import (
        phase314b_r258_stageo_dual_oracle_lower_multiplier_admission as stageo,
    )

    return stageo.probe_environment(Path(root).resolve())


def run_shadow_alignment(
    *,
    root: Path,
    environment: Mapping[str, Any],
    repository: Mapping[str, Any],
    spec: Optional[StageQSpec] = None,
) -> Mapping[str, Any]:
    active = StageQSpec() if spec is None else spec
    active.validate()
    validate_environment_variables()
    repo = Path(root).resolve()
    base_report = _validate_base_report(repo)
    base_cells = _base_cell_map(base_report)

    from ccda_phase3 import (
        phase314b_r258_stageo_dual_oracle_lower_multiplier_admission as stageo,
    )
    from ccda_phase3 import (
        phase314b_r258_stagep_oracle_upper_segment_gate_provenance as stagep,
    )

    runtime = dict(stageo._prepare_runtime(repo, environment))
    runtime["stageo"] = stageo
    runtime["stagep"] = stagep
    stagef = runtime["stagef"]
    context = runtime["context"]
    cells: List[Mapping[str, Any]] = []

    for timestep in EXPECTED_TIMESTEPS:
        control = np.asarray(
            context["objective_control_predictions"][int(timestep)],
            dtype=np.float32,
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
        for source_id in ORACLE_SOURCES:
            key = (source_id, int(timestep))
            cells.append(
                audit_shadow_cell(
                    source_id=source_id,
                    timestep=int(timestep),
                    control=control,
                    base_direction=directions[source_id],
                    context=context,
                    runtime=runtime,
                    base_cell=base_cells[key],
                    spec=active,
                )
            )

    if len(cells) != EXPECTED_PAIR_COUNT:
        raise StageQError("Stage-Q replay pair population changed")
    classification = classify_shadow_alignment(cells, active)
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
        "repository": copy.deepcopy(dict(repository)),
        "environment": copy.deepcopy(dict(environment)),
        "audit_spec": asdict(active),
        "base_stagep": {
            "report_sha256": EXPECTED_BASE_REPORT_SHA256,
            "scientific_result_sha256": EXPECTED_BASE_SCIENTIFIC_RESULT_SHA256,
            "root_cause": EXPECTED_BASE_ROOT_CAUSE,
            "required_next_path": EXPECTED_BASE_NEXT_PATH,
            "candidate_gate_violation_count": EXPECTED_BASE_VIOLATION_COUNT,
            "violations_already_present_in_control": EXPECTED_BASE_VIOLATION_COUNT,
            "new_violations_absent_from_control": 0,
            "violations_within_reconstruction_tolerance": (
                EXPECTED_BASE_VIOLATION_COUNT
            ),
            "violations_beyond_reconstruction_tolerance": 0,
        },
        "tolerance_aligned_upper_gate_shadow": {
            "external_multiplier": LOWER_MULTIPLIER,
            "oracle_sources": list(ORACLE_SOURCES),
            "timesteps": list(EXPECTED_TIMESTEPS),
            "legacy_callback_off_on_pair_count": len(cells),
            "legacy_gate_preserved_as_control": True,
            "aligned_gate_is_shadow_only": True,
            "aligned_gate_written_into_stagee": False,
            "all_legacy_callback_results_bit_exact": all(
                cell.get("legacy_callback_result_bit_exact") is True
                for cell in cells
            ),
            "all_cells_match_stagep_identity": all(
                _mapping(cell.get("legacy_identity"), "legacy identity").get(
                    "all_exact"
                )
                is True
                for cell in cells
            ),
            "cell_records": cells,
            "classification": classification,
            "row_level_arrays_persisted": False,
            "callback_events_persisted": False,
            "direction_tensors_persisted": False,
            "proposal_tensors_persisted": False,
            "candidate_tensors_persisted": False,
            "predicate_masks_persisted": False,
        },
        "immutable_inputs": {
            "stagep_report_sha256": EXPECTED_BASE_REPORT_SHA256,
            "stagep_scientific_result_sha256": (
                EXPECTED_BASE_SCIENTIFIC_RESULT_SHA256
            ),
            "stagep_report_rewritten": False,
            "stage_l_132_callback_pairs_rerun": False,
            "oof_surrogate_refit": False,
            "oof_surrogate_evaluated": False,
            "targeted_oracle_callback_pairs_run": EXPECTED_PAIR_COUNT,
        },
        "mechanism_boundary": {
            "stagee_modified": False,
            "stagek_modified": False,
            "stagel_modified": False,
            "stagem_modified": False,
            "stagen_modified": False,
            "stageo_modified": False,
            "stageo_resume1_modified": False,
            "stagep_modified": False,
            "legacy_upper_gate_changed": False,
            "segment_reconstruction_changed": False,
            "integrator_thresholds_changed": False,
            "integrator_scale_search_changed": False,
            "external_scale_bank_changed": False,
            "shadow_gate_uses_reconstruction_length_tolerance": True,
            "positionwise_log_z_equivalence_audited": True,
            "objective_train_oracle_only_replay": True,
        },
        **{key: False for key in FALSE_BOUNDARIES},
    }
    result["scientific_result_sha256"] = sha256_bytes(stable_json_bytes(result))
    return result


def blocked_report(
    *,
    repository: Optional[Mapping[str, Any]],
    error: BaseException,
) -> Mapping[str, Any]:
    return {
        "phase": PHASE,
        "schema": BLOCKED_SCHEMA,
        "execution_verdict": "BLOCKED",
        "scientific_status": "BLOCKED",
        "root_cause": "phase314b_r258_stageq_shadow_alignment_execution_failed",
        "required_next_path": "RESTORE_STAGEQ_TOLERANCE_ALIGNED_UPPER_GATE_SHADOW",
        "primary_failure_locus": "execution_contract",
        "selected_configuration": None,
        "train_only_recommendation": None,
        "repository": None if repository is None else dict(repository),
        "error_type": type(error).__name__,
        "error_message": str(error),
        "stagep_report_preserved": True,
        "stagep_report_sha256": EXPECTED_BASE_REPORT_SHA256,
        "stage_l_132_callback_pairs_rerun": False,
        "oof_surrogate_refit": False,
        "oof_surrogate_evaluated": False,
        **{key: False for key in FALSE_BOUNDARIES},
    }
