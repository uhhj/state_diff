"""Phase3.14b-r2.5.8 Stage P oracle upper-segment gate provenance audit.

Stage-O Resume1 established that raw/projected oracle directions at external
multiplier 0.25 are rejected in all six source × timestep cells and that the
stable first-failure predicate is ``upper_segment_geometry``.  Stage P performs
one bounded objective-train-only replay of those same six callback-off/on pairs
and adds direct, read-only provenance for every frozen internal scale:

* control, raw-proposal, and reconstructed-candidate upper scores;
* per-position upper failure counts over 4 horizons × 23 ordered segments;
* reconstruction-bound overshoot and its relation to the frozen 5×epsilon
  bound tolerance;
* exact agreement between the direct candidate upper mask and the Stage-K
  callback first-failure count;
* exact final selected-scale/candidate/callback identity against Stage-O
  Resume1.

No row-level arrays, callback events, directions, proposals, candidates, or
predicate masks are persisted.  Stage-E/K/L/M/N/O/O-Resume1 remain unchanged.
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
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np


PHASE = "Phase3.14b-r2.5.8 Stage P"
SCHEMA = "phase314b_r258_stagep_oracle_upper_segment_gate_provenance_v1"
BLOCKED_SCHEMA = "phase314b_r258_stagep_oracle_upper_segment_gate_provenance_blocked_v1"

BASE_EVIDENCE_COMMIT = "c165dbb7882de4789612747fa4e396b01bfc5d4b"
BASE_IMPLEMENTATION_COMMIT = "26dab50abb93c8ab816a06b6f7439a9e37d641a1"
EXPECTED_STAGEO_BLOCKED_EVIDENCE_COMMIT = "8dd4a129708db0141d3d328e39270f9b4b8d8ebc"
EXPECTED_REMOTE = "6758ea7ad800667a436b0243d3b1f6c63256d854"
EXPECTED_SUBMODULE = "633a88752445cf5d6776ed374fdbbdb35f93050c"

BASE_REPORT = (
    "reports/phase3_14b_r258_stageo_resume1_dual_oracle_lower_multiplier_"
    "admission_summary.json"
)
EXPECTED_BASE_REPORT_SHA256 = "34c9cd1250b80af9311e8242bce4da1f583a335305ea34154db68b4694e100bd"
EXPECTED_RECOVERED_STAGEO_PAYLOAD_SHA256 = "e6c062aa4fb4d03794cf5b9e54cd5e375a09b4bd7af6a25c98d3032e4a4e579e"
EXPECTED_STAGEO_SCIENTIFIC_RESULT_SHA256 = "7f1306197a68291f4927c203b38219638855720c1c579e6908ad4b3abb529a2d"
EXPECTED_BASE_ROOT_CAUSE = "phase314b_r258_stageo_lower_multiplier_oracle_upper_segment_rejection"
EXPECTED_BASE_NEXT_PATH = "AUDIT_ORACLE_UPPER_SEGMENT_GATE_AT_LOWER_EXTERNAL_MULTIPLIER"

ORIGINAL_STAGEO_BLOCKED_REPORT = (
    "reports/phase3_14b_r258_stageo_dual_oracle_lower_multiplier_"
    "admission_blocked_summary.json"
)
EXPECTED_ORIGINAL_STAGEO_BLOCKED_REPORT_SHA256 = (
    "9d3368304d3217e420b17564cc7aa103884ef970ba20916937b4f397c29167ad"
)

SUCCESS_REPORT = "reports/phase3_14b_r258_stagep_oracle_upper_segment_gate_provenance_summary.json"
BLOCKED_REPORT = "reports/phase3_14b_r258_stagep_oracle_upper_segment_gate_provenance_blocked_summary.json"

IMPLEMENTATION_SUBJECT = "Phase3.14b-r2.5.8 Stage P: audit oracle upper-segment gate provenance"
EVIDENCE_SUBJECT = "Record Phase3.14b-r2.5.8 Stage P oracle upper-segment evidence"
BLOCKED_EVIDENCE_SUBJECT = "Record Phase3.14b-r2.5.8 Stage P blocked evidence"
BASE_IMPLEMENTATION_SUBJECT = "Phase3.14b-r2.5.8 Stage O Resume1: restore descending attempt order"
BASE_EVIDENCE_SUBJECT = "Record Phase3.14b-r2.5.8 Stage O Resume1 dual-oracle evidence"

IMPLEMENTATION_PATHS: Tuple[Tuple[str, str], ...] = (
    ("A", "ccda_phase3/phase314b_r258_stagep_oracle_upper_segment_gate_provenance.py"),
    ("A", "scripts/phase3_14b_r258_stagep_execute.py"),
    ("A", "tests/test_phase3_14b_r258_stagep_oracle_upper_segment_gate_provenance.py"),
)

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
}

LOWER_MULTIPLIER = 0.25
EXPECTED_TIMESTEPS: Tuple[int, ...] = (10, 25, 50)
ORACLE_SOURCES: Tuple[str, ...] = ("raw_oracle", "projected_oracle")
EXPECTED_PAIR_COUNT = 6
EXPECTED_POSITION_SHAPE = (4, 23)
EXPECTED_POSITION_COUNT = 92

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
    "row_identity_persisted",
    "npz_saved",
    "cache_saved",
    "image_saved",
    "video_saved",
)
BASE_FALSE_BOUNDARIES: Tuple[str, ...] = tuple(
    key for key in FALSE_BOUNDARIES if key != "row_identity_persisted"
)


class StagePError(RuntimeError):
    """Fail-closed Stage-P error."""


@dataclass(frozen=True)
class StagePSpec:
    external_multiplier: float = LOWER_MULTIPLIER
    top_position_count: int = 12
    binary_tolerance: float = 1.0e-12

    def validate(self) -> None:
        if float(self.external_multiplier) != LOWER_MULTIPLIER:
            raise StagePError("Stage-P external multiplier changed")
        if self.top_position_count != 12:
            raise StagePError("Stage-P top-position count changed")
        if not math.isfinite(self.binary_tolerance) or self.binary_tolerance < 0.0:
            raise StagePError("Stage-P binary tolerance is invalid")


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
        raise StagePError(f"{label} is not a mapping")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise StagePError(f"{label} is not a sequence")
    return value


def _finite(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise StagePError(f"{label} is not finite")
    return result


def _finite_rate(value: Any, label: str) -> float:
    result = _finite(value, label)
    if not 0.0 <= result <= 1.0:
        raise StagePError(f"{label} is not a rate")
    return result


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise StagePError(f"{label} is boolean")
    result = int(value)
    if result < 0 or float(result) != float(value):
        raise StagePError(f"{label} is not a nonnegative integer")
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
        raise StagePError(
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
        raise StagePError(
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
            raise StagePError("unexpected diff-tree record")
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
            raise StagePError(f"{label} boundary changed: {key}={mapping.get(key)!r}")


def validate_environment_variables() -> Mapping[str, str]:
    mismatch = {
        key: {"expected": expected, "actual": os.environ.get(key)}
        for key, expected in EXPECTED_ENV.items()
        if os.environ.get(key) != expected
    }
    if mismatch:
        raise StagePError(f"deterministic environment mismatch: {mismatch}")
    return dict(EXPECTED_ENV)


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise StagePError(f"cannot load JSON {path}: {error}") from error
    return _mapping(value, str(path))


def _validate_base_report(root: Path) -> Mapping[str, Any]:
    path = Path(root) / BASE_REPORT
    if not path.is_file():
        raise StagePError("Stage-O Resume1 report is missing")
    if sha256_file(path) != EXPECTED_BASE_REPORT_SHA256:
        raise StagePError("Stage-O Resume1 report SHA changed")
    if _git_bytes(root, "show", f"{BASE_EVIDENCE_COMMIT}:{BASE_REPORT}") != path.read_bytes():
        raise StagePError("Stage-O Resume1 report differs from committed blob")
    report = _load_json(path)
    if report.get("execution_verdict") != "PASS":
        raise StagePError("Stage-O Resume1 execution is not PASS")
    if report.get("scientific_status") != "BLOCKED":
        raise StagePError("Stage-O Resume1 scientific status changed")
    if report.get("root_cause") != EXPECTED_BASE_ROOT_CAUSE:
        raise StagePError("Stage-O Resume1 root cause changed")
    if report.get("required_next_path") != EXPECTED_BASE_NEXT_PATH:
        raise StagePError("Stage-O Resume1 next path changed")
    if report.get("stage_o_result_sha256") != EXPECTED_RECOVERED_STAGEO_PAYLOAD_SHA256:
        raise StagePError("recovered Stage-O payload SHA changed")
    stage_o = _mapping(report.get("stage_o_result"), "Stage-O result")
    if stage_o.get("scientific_result_sha256") != EXPECTED_STAGEO_SCIENTIFIC_RESULT_SHA256:
        raise StagePError("Stage-O scientific-result SHA changed")
    if stage_o.get("root_cause") != EXPECTED_BASE_ROOT_CAUSE:
        raise StagePError("nested Stage-O root cause changed")
    targeted = _mapping(stage_o.get("targeted_replay"), "targeted replay")
    if int(targeted.get("callback_off_on_pair_count", -1)) != EXPECTED_PAIR_COUNT:
        raise StagePError("Stage-O pair population changed")
    if targeted.get("all_callback_results_bit_exact") is not True:
        raise StagePError("Stage-O callback identity changed")
    if targeted.get("all_cells_match_stage_l_025_identity") is not True:
        raise StagePError("Stage-O/Stage-L identity changed")
    cells = _sequence(targeted.get("cell_records"), "Stage-O cell records")
    if len(cells) != EXPECTED_PAIR_COUNT:
        raise StagePError("Stage-O cell count changed")
    observed = set()
    for index, raw in enumerate(cells):
        cell = _mapping(raw, f"Stage-O cell {index}")
        source = str(cell.get("source_id"))
        timestep = int(cell.get("timestep"))
        if source not in ORACLE_SOURCES or timestep not in EXPECTED_TIMESTEPS:
            raise StagePError("Stage-O source/timestep population changed")
        observed.add((source, timestep))
        if float(cell.get("oracle_acceptance_rate", -1.0)) != 0.0:
            raise StagePError("Stage-O oracle admission changed")
        if cell.get("stable_internal_scale_failure_predicate") != (
            "upper_segment_geometry"
        ):
            raise StagePError("Stage-O stable failure predicate changed")
    expected = {(source, timestep) for source in ORACLE_SOURCES for timestep in EXPECTED_TIMESTEPS}
    if observed != expected:
        raise StagePError("Stage-O six-cell population changed")
    if report.get("selected_configuration") is not None:
        raise StagePError("Stage-O Resume1 selected a configuration")
    if report.get("train_only_recommendation") is not None:
        raise StagePError("Stage-O Resume1 emitted a recommendation")
    require_false(report, BASE_FALSE_BOUNDARIES, "Stage-O Resume1")
    require_false(stage_o, BASE_FALSE_BOUNDARIES, "nested Stage-O")
    return report


def _cell_map(base_report: Mapping[str, Any]) -> Mapping[Tuple[str, int], Mapping[str, Any]]:
    stage_o = _mapping(base_report.get("stage_o_result"), "Stage-O result")
    targeted = _mapping(stage_o.get("targeted_replay"), "targeted replay")
    values = _sequence(targeted.get("cell_records"), "Stage-O cell records")
    output: Dict[Tuple[str, int], Mapping[str, Any]] = {}
    for index, raw in enumerate(values):
        cell = _mapping(raw, f"Stage-O cell {index}")
        key = (str(cell.get("source_id")), int(cell.get("timestep")))
        if key in output:
            raise StagePError("duplicate Stage-O cell")
        output[key] = cell
    return output


def validate_repository(root: Path) -> Mapping[str, Any]:
    repo = Path(root).resolve()
    if _git(repo, "branch", "--show-current") != "Experiment1":
        raise StagePError("Stage P requires Experiment1")
    head = _git(repo, "rev-parse", "HEAD")
    parent = _git(repo, "rev-parse", f"{head}^")
    if parent != BASE_EVIDENCE_COMMIT:
        raise StagePError("Stage-P implementation parent changed")
    if _git(repo, "show", "-s", "--format=%s", head) != IMPLEMENTATION_SUBJECT:
        raise StagePError("Stage-P implementation subject changed")
    if commit_name_status(repo, head) != tuple(sorted(IMPLEMENTATION_PATHS)):
        raise StagePError("Stage-P implementation path population changed")
    if _git(repo, "rev-parse", f"{BASE_EVIDENCE_COMMIT}^") != BASE_IMPLEMENTATION_COMMIT:
        raise StagePError("Stage-O Resume1 evidence parent changed")
    if _git(repo, "show", "-s", "--format=%s", BASE_IMPLEMENTATION_COMMIT) != (
        BASE_IMPLEMENTATION_SUBJECT
    ):
        raise StagePError("Stage-O Resume1 implementation subject changed")
    if _git(repo, "show", "-s", "--format=%s", BASE_EVIDENCE_COMMIT) != (
        BASE_EVIDENCE_SUBJECT
    ):
        raise StagePError("Stage-O Resume1 evidence subject changed")
    if _git(repo, "rev-parse", "origin/Experiment1") != EXPECTED_REMOTE:
        raise StagePError("origin/Experiment1 changed")
    submodule = repo / "external/deformable-ravens"
    if _git(submodule, "rev-parse", "HEAD") != EXPECTED_SUBMODULE:
        raise StagePError("DeformableRavens commit changed")
    if status_paths(submodule):
        raise StagePError("DeformableRavens worktree is dirty")
    if status_paths(repo):
        raise StagePError("Stage-P worktree must be clean before execution")

    source_sha: Dict[str, str] = {}
    for relative, expected in FROZEN_SOURCE_SHA256.items():
        path = repo / relative
        if not path.is_file():
            raise StagePError(f"frozen source is missing: {relative}")
        actual = sha256_file(path)
        if actual != expected:
            raise StagePError(f"frozen source SHA changed: {relative}")
        source_sha[relative] = actual

    original_blocked = repo / ORIGINAL_STAGEO_BLOCKED_REPORT
    if not original_blocked.is_file():
        raise StagePError("original Stage-O blocked report is missing")
    if sha256_file(original_blocked) != EXPECTED_ORIGINAL_STAGEO_BLOCKED_REPORT_SHA256:
        raise StagePError("original Stage-O blocked report SHA changed")

    base_report = _validate_base_report(repo)
    for relative in (SUCCESS_REPORT, BLOCKED_REPORT):
        if (repo / relative).exists():
            raise StagePError(f"Stage-P output already exists: {relative}")
    return {
        "root": str(repo),
        "head": head,
        "parent": parent,
        "origin_experiment1": EXPECTED_REMOTE,
        "submodule_commit": EXPECTED_SUBMODULE,
        "frozen_source_sha256": source_sha,
        "base_report_sha256": EXPECTED_BASE_REPORT_SHA256,
        "base_recovered_stageo_payload_sha256": EXPECTED_RECOVERED_STAGEO_PAYLOAD_SHA256,
        "base_stageo_scientific_result_sha256": EXPECTED_STAGEO_SCIENTIFIC_RESULT_SHA256,
        "original_stageo_blocked_report_sha256": EXPECTED_ORIGINAL_STAGEO_BLOCKED_REPORT_SHA256,
        "base_cell_count": len(_cell_map(base_report)),
    }


def scalar_stats(value: np.ndarray) -> Mapping[str, Any]:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise StagePError("cannot summarize an empty or non-finite array")
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


def integer_matrix(value: np.ndarray, label: str) -> List[List[int]]:
    array = np.asarray(value)
    if array.shape != EXPECTED_POSITION_SHAPE:
        raise StagePError(f"{label} shape changed: {array.shape}")
    if np.any(array < 0) or not np.all(np.equal(array, np.floor(array))):
        raise StagePError(f"{label} is not a nonnegative integer matrix")
    return [[int(item) for item in row] for row in array]


def float_matrix(value: np.ndarray, label: str) -> List[List[float]]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != EXPECTED_POSITION_SHAPE:
        raise StagePError(f"{label} shape changed: {array.shape}")
    if not np.all(np.isfinite(array)):
        raise StagePError(f"{label} contains non-finite values")
    return [[float(item) for item in row] for row in array]


def upper_surface_summary(
    *,
    scores: Mapping[str, Any],
    threshold: float,
    top_position_count: int,
) -> Mapping[str, Any]:
    z = np.asarray(scores["z"], dtype=np.float64)
    upper = np.asarray(scores["upper"], dtype=np.float64)
    upper_element = np.asarray(scores["upper_element"], dtype=np.float64)
    if z.ndim != 4 or z.shape[1:] != (1, *EXPECTED_POSITION_SHAPE):
        raise StagePError(f"upper z shape changed: {z.shape}")
    if upper.shape != (z.shape[0], 1):
        raise StagePError("upper row-score shape changed")
    if upper_element.shape != z.shape:
        raise StagePError("upper-element shape changed")
    row_score = upper[:, 0]
    element = upper_element[:, 0]
    violation = element > float(threshold)
    row_pass = row_score <= float(threshold)
    failure_counts = np.sum(violation, axis=0, dtype=np.int64)
    max_z = np.max(z[:, 0], axis=0)
    max_overshoot = np.max(np.maximum(z[:, 0] - float(threshold), 0.0), axis=0)
    positions: List[Mapping[str, Any]] = []
    for horizon in range(EXPECTED_POSITION_SHAPE[0]):
        for segment in range(EXPECTED_POSITION_SHAPE[1]):
            positions.append(
                {
                    "horizon_index": horizon,
                    "segment_index": segment,
                    "failure_count": int(failure_counts[horizon, segment]),
                    "failure_rate": float(
                        failure_counts[horizon, segment] / z.shape[0]
                    ),
                    "maximum_z": float(max_z[horizon, segment]),
                    "maximum_z_overshoot": float(max_overshoot[horizon, segment]),
                }
            )
    worst = sorted(
        positions,
        key=lambda item: (
            -int(item["failure_count"]),
            -float(item["maximum_z_overshoot"]),
            int(item["horizon_index"]),
            int(item["segment_index"]),
        ),
    )[:top_position_count]
    margin = float(threshold) - row_score
    return {
        "row_count": int(z.shape[0]),
        "threshold": float(threshold),
        "row_pass_rate": float(np.mean(row_pass)),
        "row_failure_count": int(np.count_nonzero(~row_pass)),
        "row_score_stats": scalar_stats(row_score),
        "row_margin_stats": scalar_stats(margin),
        "element_count": int(np.prod(z[:, 0].shape)),
        "element_failure_count": int(np.count_nonzero(violation)),
        "element_failure_rate": float(np.mean(violation)),
        "failure_count_by_position": integer_matrix(
            failure_counts, "upper failure-count matrix"
        ),
        "maximum_z_by_position": float_matrix(max_z, "maximum-z matrix"),
        "maximum_z_overshoot_by_position": float_matrix(
            max_overshoot, "maximum-z-overshoot matrix"
        ),
        "worst_positions": worst,
        "row_upper_pass_sha256": sha256_array(row_pass.astype(np.bool_)),
        "element_upper_violation_sha256": sha256_array(violation.astype(np.bool_)),
    }


def reconstruction_boundary_summary(
    *,
    raw_lengths: np.ndarray,
    candidate_lengths: np.ndarray,
    upper_bound: np.ndarray,
    candidate_scores: Mapping[str, Any],
    threshold: float,
    segment_tolerance: float,
) -> Mapping[str, Any]:
    raw = np.asarray(raw_lengths, dtype=np.float64)
    candidate = np.asarray(candidate_lengths, dtype=np.float64)
    upper = np.asarray(upper_bound, dtype=np.float64)
    if raw.shape != candidate.shape or raw.ndim != 3:
        raise StagePError("raw/candidate segment-length shape changed")
    if raw.shape[1:] != EXPECTED_POSITION_SHAPE or upper.shape != EXPECTED_POSITION_SHAPE:
        raise StagePError("segment-bound position shape changed")
    z = np.asarray(candidate_scores["z"], dtype=np.float64)[:, 0]
    if z.shape != candidate.shape:
        raise StagePError("candidate z/length shape changed")
    gate_violation = z > float(threshold)
    length_overshoot = candidate - upper[None]
    above_bound = length_overshoot > 0.0
    tolerance = 5.0 * float(segment_tolerance)
    within_tolerance = above_bound & (length_overshoot <= tolerance)
    beyond_tolerance = length_overshoot > tolerance
    gate_fail_within_bound = gate_violation & ~above_bound
    raw_above = raw > upper[None]
    candidate_fail_at_raw_clipped = gate_violation & raw_above
    raw_pass_candidate_fail = gate_violation & ~(raw > upper[None])
    positive_overshoot = length_overshoot[length_overshoot > 0.0]
    overshoot_stats = (
        scalar_stats(positive_overshoot)
        if positive_overshoot.size
        else {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "p05": 0.0,
            "median": 0.0,
            "p95": 0.0,
            "max": 0.0,
        }
    )
    return {
        "bound_shape": list(upper.shape),
        "upper_bound_sha256": sha256_array(upper),
        "upper_bound_stats": scalar_stats(upper),
        "reconstruction_bound_tolerance": tolerance,
        "raw_length_above_bound_count": int(np.count_nonzero(raw_above)),
        "candidate_length_above_bound_count": int(np.count_nonzero(above_bound)),
        "candidate_length_above_bound_within_tolerance_count": int(
            np.count_nonzero(within_tolerance)
        ),
        "candidate_length_above_bound_beyond_tolerance_count": int(
            np.count_nonzero(beyond_tolerance)
        ),
        "candidate_gate_violation_count": int(np.count_nonzero(gate_violation)),
        "candidate_gate_violation_within_reconstruction_tolerance_count": int(
            np.count_nonzero(gate_violation & within_tolerance)
        ),
        "candidate_gate_violation_beyond_reconstruction_tolerance_count": int(
            np.count_nonzero(gate_violation & beyond_tolerance)
        ),
        "candidate_gate_fail_while_length_not_above_bound_count": int(
            np.count_nonzero(gate_fail_within_bound)
        ),
        "candidate_gate_fail_at_raw_clipped_position_count": int(
            np.count_nonzero(candidate_fail_at_raw_clipped)
        ),
        "raw_bound_pass_candidate_gate_fail_count": int(
            np.count_nonzero(raw_pass_candidate_fail)
        ),
        "candidate_length_overshoot_stats": overshoot_stats,
        "maximum_candidate_length_overshoot": float(
            max(0.0, np.max(length_overshoot))
        ),
        "maximum_candidate_z_overshoot": float(
            max(0.0, np.max(z - float(threshold)))
        ),
        "all_candidate_length_overshoots_within_reconstruction_tolerance": bool(
            not np.any(beyond_tolerance)
        ),
        "all_candidate_gate_violations_are_length_overshoots": bool(
            not np.any(gate_fail_within_bound)
        ),
        "candidate_length_above_bound_count_by_position": integer_matrix(
            np.sum(above_bound, axis=0, dtype=np.int64),
            "candidate above-bound count matrix",
        ),
        "candidate_gate_violation_count_by_position": integer_matrix(
            np.sum(gate_violation, axis=0, dtype=np.int64),
            "candidate gate-violation count matrix",
        ),
    }


def transition_summary(
    *,
    control_scores: Mapping[str, Any],
    raw_scores: Mapping[str, Any],
    candidate_scores: Mapping[str, Any],
    threshold: float,
) -> Mapping[str, Any]:
    control_z = np.asarray(control_scores["z"], dtype=np.float64)[:, 0]
    raw_z = np.asarray(raw_scores["z"], dtype=np.float64)[:, 0]
    candidate_z = np.asarray(candidate_scores["z"], dtype=np.float64)[:, 0]
    if not (control_z.shape == raw_z.shape == candidate_z.shape):
        raise StagePError("upper transition surface shapes differ")
    control_fail = control_z > float(threshold)
    raw_fail = raw_z > float(threshold)
    candidate_fail = candidate_z > float(threshold)
    return {
        "element_count": int(control_z.size),
        "control_fail_raw_fail_candidate_fail_count": int(
            np.count_nonzero(control_fail & raw_fail & candidate_fail)
        ),
        "control_pass_raw_fail_candidate_fail_count": int(
            np.count_nonzero(~control_fail & raw_fail & candidate_fail)
        ),
        "raw_fail_candidate_pass_count": int(np.count_nonzero(raw_fail & ~candidate_fail)),
        "raw_pass_candidate_fail_count": int(np.count_nonzero(~raw_fail & candidate_fail)),
        "candidate_fail_already_in_control_count": int(
            np.count_nonzero(control_fail & candidate_fail)
        ),
        "candidate_fail_not_in_control_count": int(
            np.count_nonzero(~control_fail & candidate_fail)
        ),
        "control_failure_sha256": sha256_array(control_fail.astype(np.bool_)),
        "raw_failure_sha256": sha256_array(raw_fail.astype(np.bool_)),
        "candidate_failure_sha256": sha256_array(candidate_fail.astype(np.bool_)),
    }


def _attempt_events(capture: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    values = _sequence(capture.get("events"), "callback events")
    attempts = [
        _mapping(item, f"callback event {index}")
        for index, item in enumerate(values)
        if isinstance(item, Mapping) and item.get("event_type") == "scale_attempt"
    ]
    if not attempts:
        raise StagePError("callback capture lacks scale attempts")
    return attempts


def _expected_internal_order(runtime: Mapping[str, Any]) -> Tuple[float, ...]:
    integrator_spec = runtime["integrator_spec"]
    definition = runtime["stagef"].fixed_integrator_definition()
    eligible = [
        float(scale)
        for scale in integrator_spec.scale_grid
        if float(scale)
        <= float(definition.maximum_scale) + float(integrator_spec.standardizer_epsilon)
    ]
    expected = tuple(sorted(eligible, reverse=True))
    if not expected or len(set(expected)) != len(expected):
        raise StagePError("frozen internal-scale population is invalid")
    return expected


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
        "selected_scale_sha256": (
            stageo.sha256_array(selected) == base_cell.get("selected_scale_sha256")
        ),
        "candidate_sha256": (
            stageo.sha256_array(candidate) == base_cell.get("candidate_sha256")
        ),
        "callback_capture_sha256": (
            str(capture.get("events_sha256"))
            == str(base_cell.get("callback_capture_sha256"))
        ),
        "callback_result_bit_exact": capture.get("returned_result_bit_exact") is True,
        "selected_scale_positive_rate": (
            float(np.mean(selected > 1.0e-12))
            == float(base_cell.get("selected_scale_positive_rate"))
        ),
    }
    if not all(checks.values()):
        failed = sorted(key for key, value in checks.items() if not value)
        raise StagePError(f"Stage-P replay differs from Stage-O Resume1: {failed}")
    return {"all_exact": True, "checks": checks}


def audit_upper_gate_cell(
    *,
    source_id: str,
    timestep: int,
    control: np.ndarray,
    base_direction: np.ndarray,
    context: Mapping[str, Any],
    runtime: Mapping[str, Any],
    base_cell: Mapping[str, Any],
    spec: StagePSpec,
) -> Mapping[str, Any]:
    stageo = runtime["stageo"]
    stagek = runtime["stagek"]
    stagee = runtime["stagel"].stagee258
    staged = stagee.staged
    stageb = stagee.stageb
    integrator_spec = runtime["integrator_spec"]
    callback_spec = runtime["callback_spec"]
    definition = runtime["stagef"].fixed_integrator_definition()
    definition.validate()

    control_raw = np.asarray(control, dtype=np.float32)
    direction_raw = np.asarray(base_direction, dtype=np.float64)
    if control_raw.shape != direction_raw.shape:
        raise StagePError("control/oracle direction shape changed")
    proposed_direction = direction_raw * float(spec.external_multiplier)
    integration, capture = stagek.callback_integrate_rowwise(
        control=control_raw,
        direction=proposed_direction,
        context=context,
        integrator_spec=integrator_spec,
        callback_spec=callback_spec,
    )
    identity = _base_identity_check(
        source_id=source_id,
        timestep=timestep,
        integration=integration,
        capture=capture,
        base_cell=base_cell,
        stageo=stageo,
    )
    attempts = _attempt_events(capture)
    expected_order = _expected_internal_order(runtime)
    observed_order = tuple(float(item.get("attempted_scale")) for item in attempts)
    if observed_order != expected_order:
        raise StagePError(
            "Stage-P callback attempt order differs from frozen Stage-E order"
        )

    selected_scale = np.asarray(integration["selected_scale"], dtype=np.float64).reshape(
        control_raw.shape[0]
    )
    bounds = stagee.segment_bounds(definition=definition, context=context)
    upper_bound = np.asarray(bounds["upper"], dtype=np.float64)
    threshold = float(context["upper_gate"].upper_threshold)
    tolerance = float(integrator_spec.segment_tolerance)
    reference = context["stage_d_contract"].reference
    control_scores = staged.segment_scores(control_raw, reference)
    control_surface = upper_surface_summary(
        scores=control_scores,
        threshold=threshold,
        top_position_count=spec.top_position_count,
    )
    control_lengths = stageb.segment_lengths(control_raw).astype(np.float64)

    attempt_records: List[Mapping[str, Any]] = []
    for event, scale in zip(attempts, expected_order):
        raw_proposal = (
            control_raw.astype(np.float64) + float(scale) * proposed_direction
        ).astype(np.float32)
        if definition.integration_mode == "linear":
            candidate = raw_proposal
            bound_pass = None
            coordinate_possible = None
            reconstruction = {
                "clipped_segment_rate": 0.0,
                "degenerate_segment_rate": 0.0,
            }
        elif definition.integration_mode == "segment_reconstruct":
            reconstruction = stagee.reconstruct_segment_vectors(
                proposed=raw_proposal,
                control=control_raw,
                lower=np.asarray(bounds["lower"], dtype=np.float64),
                upper=upper_bound,
                coordinate_abs_max=float(
                    context["historical_geometry"].coordinate_abs_max
                ),
                epsilon=tolerance,
            )
            candidate = np.asarray(reconstruction["candidate"], dtype=np.float32)
            bound_pass = np.asarray(reconstruction["bound_pass"], dtype=np.bool_)
            coordinate_possible = np.asarray(
                reconstruction["coordinate_possible"], dtype=np.bool_
            )
        else:
            raise StagePError("fixed integrator mode changed")

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
        raw_scores = staged.segment_scores(raw_proposal, reference)
        candidate_scores = staged.segment_scores(candidate, reference)
        direct_candidate_upper = (
            np.asarray(candidate_scores["upper"], dtype=np.float64)[:, 0]
            <= threshold
        )
        metric_candidate_upper = np.asarray(metrics["upper_pass"], dtype=np.bool_)
        if not np.array_equal(direct_candidate_upper, metric_candidate_upper):
            raise StagePError("direct/candidate upper mask differs from Stage-E metrics")

        selected_before = selected_scale > float(scale)
        active = ~selected_before
        finite = np.asarray(metrics["finite"], dtype=np.bool_)
        direct_upper_first_failure = active & finite & ~metric_candidate_upper
        direct_upper_pass_over_active = active & metric_candidate_upper
        callback_failures = _mapping(
            event.get("first_failed_counts"), "callback first-failed counts"
        )
        callback_pass = _mapping(
            event.get("predicate_pass_counts"), "callback predicate-pass counts"
        )
        callback_checks = {
            "attempted_scale": float(event.get("attempted_scale")) == float(scale),
            "active_row_count": int(event.get("active_row_count", -1))
            == int(np.count_nonzero(active)),
            "already_selected_count": int(event.get("already_selected_count", -1))
            == int(np.count_nonzero(selected_before)),
            "upper_first_failed_count": int(
                callback_failures.get("upper_segment_geometry", -1)
            )
            == int(np.count_nonzero(direct_upper_first_failure)),
            "upper_pass_count": int(
                callback_pass.get("upper_segment_geometry", -1)
            )
            == int(np.count_nonzero(direct_upper_pass_over_active)),
        }
        if not all(callback_checks.values()):
            failed = sorted(key for key, value in callback_checks.items() if not value)
            raise StagePError(f"direct upper provenance differs from callback: {failed}")

        raw_lengths = stageb.segment_lengths(raw_proposal).astype(np.float64)
        candidate_lengths = stageb.segment_lengths(candidate).astype(np.float64)
        raw_surface = upper_surface_summary(
            scores=raw_scores,
            threshold=threshold,
            top_position_count=spec.top_position_count,
        )
        candidate_surface = upper_surface_summary(
            scores=candidate_scores,
            threshold=threshold,
            top_position_count=spec.top_position_count,
        )
        boundary = reconstruction_boundary_summary(
            raw_lengths=raw_lengths,
            candidate_lengths=candidate_lengths,
            upper_bound=upper_bound,
            candidate_scores=candidate_scores,
            threshold=threshold,
            segment_tolerance=tolerance,
        )
        transition = transition_summary(
            control_scores=control_scores,
            raw_scores=raw_scores,
            candidate_scores=candidate_scores,
            threshold=threshold,
        )
        attempt_records.append(
            {
                "internal_scale": float(scale),
                "effective_multiplier": float(scale * spec.external_multiplier),
                "active_row_count": int(np.count_nonzero(active)),
                "already_selected_row_count": int(np.count_nonzero(selected_before)),
                "callback_upper_first_failed_count": int(
                    callback_failures["upper_segment_geometry"]
                ),
                "callback_upper_pass_count": int(
                    callback_pass["upper_segment_geometry"]
                ),
                "callback_direct_upper_exact": True,
                "raw_proposal_sha256": sha256_array(raw_proposal),
                "reconstructed_candidate_sha256": sha256_array(candidate),
                "clipped_segment_rate": float(
                    reconstruction["clipped_segment_rate"]
                ),
                "degenerate_segment_rate": float(
                    reconstruction["degenerate_segment_rate"]
                ),
                "raw_upper": raw_surface,
                "candidate_upper": candidate_surface,
                "reconstruction_boundary": boundary,
                "control_raw_candidate_transition": transition,
                "callback_checks": callback_checks,
                "raw_tensor_persisted": False,
                "candidate_tensor_persisted": False,
                "predicate_mask_persisted": False,
            }
        )

    return {
        "source_id": source_id,
        "timestep": int(timestep),
        "external_multiplier": float(spec.external_multiplier),
        "objective_train_rows": int(control_raw.shape[0]),
        "upper_threshold": threshold,
        "segment_tolerance": tolerance,
        "reconstruction_bound_tolerance": 5.0 * tolerance,
        "internal_scale_attempt_order": list(expected_order),
        "control_sha256": sha256_array(control_raw),
        "base_direction_sha256": sha256_array(direction_raw),
        "proposed_direction_sha256": sha256_array(proposed_direction),
        "final_selected_scale_sha256": stageo.sha256_array(selected_scale),
        "final_candidate_sha256": str(integration["candidate_sha256"]),
        "callback_capture_sha256": str(capture["events_sha256"]),
        "callback_result_bit_exact": True,
        "stageo_resume1_identity": identity,
        "control_upper": control_surface,
        "control_segment_length_stats": scalar_stats(control_lengths),
        "upper_bound_sha256": str(bounds["upper_sha256"]),
        "upper_bound_stats": scalar_stats(upper_bound),
        "attempt_records": attempt_records,
        "callback_events_persisted": False,
        "row_identity_persisted": False,
        "direction_tensor_persisted": False,
        "proposal_tensor_persisted": False,
        "candidate_tensor_persisted": False,
    }


def classify_upper_gate_provenance(
    cells: Sequence[Mapping[str, Any]],
    spec: Optional[StagePSpec] = None,
) -> Mapping[str, Any]:
    active = StagePSpec() if spec is None else spec
    active.validate()
    if len(cells) != EXPECTED_PAIR_COUNT:
        raise StagePError("Stage-P cell population changed")

    candidate_violation_count = 0
    violation_within_tolerance_count = 0
    violation_beyond_tolerance_count = 0
    violation_while_not_above_bound_count = 0
    raw_pass_candidate_fail_count = 0
    raw_fail_candidate_fail_count = 0
    candidate_fail_already_in_control_count = 0
    candidate_fail_not_in_control_count = 0
    control_invalid_cell_count = 0
    callback_upper_failure_cells = 0
    attempt_count = 0
    source_counts: Dict[str, Dict[str, int]] = {
        source: {} for source in ORACLE_SOURCES
    }
    timestep_counts: Dict[str, Dict[str, int]] = {
        str(timestep): {} for timestep in EXPECTED_TIMESTEPS
    }
    internal_scale_counts: Dict[str, Dict[str, int]] = {}

    for cell in cells:
        source = str(cell.get("source_id"))
        timestep = int(cell.get("timestep"))
        if source not in ORACLE_SOURCES or timestep not in EXPECTED_TIMESTEPS:
            raise StagePError("unexpected Stage-P source/timestep")
        control = _mapping(cell.get("control_upper"), "control upper")
        if int(control.get("row_failure_count", 0)) > 0:
            control_invalid_cell_count += 1
        attempts = _sequence(cell.get("attempt_records"), "attempt records")
        if not attempts:
            raise StagePError("Stage-P cell has no internal-scale attempts")
        cell_callback_failure = 0
        for raw_attempt in attempts:
            attempt = _mapping(raw_attempt, "attempt")
            attempt_count += 1
            scale_key = _scale_key(float(attempt.get("internal_scale")))
            boundary = _mapping(
                attempt.get("reconstruction_boundary"), "reconstruction boundary"
            )
            transition = _mapping(
                attempt.get("control_raw_candidate_transition"), "transition"
            )
            total = _nonnegative_int(
                boundary.get("candidate_gate_violation_count"),
                "candidate gate violation count",
            )
            within = _nonnegative_int(
                boundary.get(
                    "candidate_gate_violation_within_reconstruction_tolerance_count"
                ),
                "within-tolerance gate violation count",
            )
            beyond = _nonnegative_int(
                boundary.get(
                    "candidate_gate_violation_beyond_reconstruction_tolerance_count"
                ),
                "beyond-tolerance gate violation count",
            )
            not_above = _nonnegative_int(
                boundary.get(
                    "candidate_gate_fail_while_length_not_above_bound_count"
                ),
                "within-bound gate violation count",
            )
            if within + beyond + not_above != total:
                raise StagePError("candidate gate-violation categories do not close")
            candidate_violation_count += total
            violation_within_tolerance_count += within
            violation_beyond_tolerance_count += beyond
            violation_while_not_above_bound_count += not_above
            raw_pass_candidate_fail_count += _nonnegative_int(
                transition.get("raw_pass_candidate_fail_count"),
                "raw-pass candidate-fail count",
            )
            raw_fail_candidate_fail_count += (
                _nonnegative_int(
                    transition.get("control_fail_raw_fail_candidate_fail_count"),
                    "control/raw/candidate fail count",
                )
                + _nonnegative_int(
                    transition.get("control_pass_raw_fail_candidate_fail_count"),
                    "raw/candidate fail count",
                )
            )
            candidate_fail_already_in_control_count += _nonnegative_int(
                transition.get("candidate_fail_already_in_control_count"),
                "candidate fail already in control count",
            )
            candidate_fail_not_in_control_count += _nonnegative_int(
                transition.get("candidate_fail_not_in_control_count"),
                "candidate fail not in control count",
            )
            callback_fail = _nonnegative_int(
                attempt.get("callback_upper_first_failed_count"),
                "callback upper first-failure count",
            )
            cell_callback_failure += callback_fail
            categories = source_counts[source]
            categories["candidate_violation"] = categories.get(
                "candidate_violation", 0
            ) + total
            timestep_category = timestep_counts[str(timestep)]
            timestep_category["candidate_violation"] = timestep_category.get(
                "candidate_violation", 0
            ) + total
            scale_category = internal_scale_counts.setdefault(scale_key, {})
            scale_category["candidate_violation"] = scale_category.get(
                "candidate_violation", 0
            ) + total
            scale_category["within_tolerance"] = scale_category.get(
                "within_tolerance", 0
            ) + within
            scale_category["beyond_tolerance"] = scale_category.get(
                "beyond_tolerance", 0
            ) + beyond
            scale_category["while_not_above_bound"] = scale_category.get(
                "while_not_above_bound", 0
            ) + not_above
        if cell_callback_failure > 0:
            callback_upper_failure_cells += 1

    if callback_upper_failure_cells != EXPECTED_PAIR_COUNT:
        raise StagePError("Stage-P did not reproduce upper first-failure in all cells")
    if candidate_violation_count <= 0:
        raise StagePError("Stage-P direct upper surface has no violations")

    if violation_beyond_tolerance_count > 0:
        root = (
            "phase314b_r258_stagep_reconstruction_materially_exceeds_"
            "frozen_upper_bound"
        )
        next_path = "AUDIT_SEGMENT_RECONSTRUCTION_UPPER_BOUND_PRESERVATION"
        primary = "candidate_length_above_bound_beyond_reconstruction_tolerance"
    elif violation_within_tolerance_count == candidate_violation_count:
        root = (
            "phase314b_r258_stagep_upper_gate_is_stricter_than_"
            "reconstruction_bound_tolerance"
        )
        next_path = "ALIGN_RECONSTRUCTION_BOUND_TOLERANCE_WITH_UPPER_SEGMENT_GATE"
        primary = "reconstruction_tolerance_upper_gate_contract_mismatch"
    elif violation_while_not_above_bound_count == candidate_violation_count:
        root = "phase314b_r258_stagep_log_exp_upper_boundary_roundoff"
        next_path = "ALIGN_LOG_SEGMENT_UPPER_GATE_WITH_FROZEN_BOUNDARY_NUMERICS"
        primary = "log_exp_boundary_roundoff"
    else:
        root = "phase314b_r258_stagep_upper_boundary_numerical_contract_is_mixed"
        next_path = "ALIGN_RECONSTRUCTION_AND_LOG_UPPER_BOUNDARY_NUMERICS"
        primary = "mixed_upper_boundary_numerics"

    return {
        "root_cause": root,
        "required_next_path": next_path,
        "primary_failure_locus": primary,
        "cell_count": len(cells),
        "internal_scale_attempt_count": attempt_count,
        "callback_upper_failure_cell_count": callback_upper_failure_cells,
        "control_invalid_cell_count": control_invalid_cell_count,
        "candidate_gate_violation_count": candidate_violation_count,
        "candidate_gate_violation_within_reconstruction_tolerance_count": (
            violation_within_tolerance_count
        ),
        "candidate_gate_violation_beyond_reconstruction_tolerance_count": (
            violation_beyond_tolerance_count
        ),
        "candidate_gate_fail_while_length_not_above_bound_count": (
            violation_while_not_above_bound_count
        ),
        "raw_pass_candidate_fail_count": raw_pass_candidate_fail_count,
        "raw_fail_candidate_fail_count": raw_fail_candidate_fail_count,
        "candidate_fail_already_in_control_count": (
            candidate_fail_already_in_control_count
        ),
        "candidate_fail_not_in_control_count": candidate_fail_not_in_control_count,
        "candidate_violation_counts_by_source": {
            key: dict(sorted(value.items())) for key, value in source_counts.items()
        },
        "candidate_violation_counts_by_timestep": {
            key: dict(sorted(value.items())) for key, value in timestep_counts.items()
        },
        "candidate_violation_counts_by_internal_scale": {
            key: dict(sorted(value.items()))
            for key, value in sorted(
                internal_scale_counts.items(), key=lambda item: -float(item[0])
            )
        },
        "classification_spec": asdict(active),
    }


def probe_environment(root: Path) -> Mapping[str, Any]:
    from ccda_phase3 import (
        phase314b_r258_stageo_dual_oracle_lower_multiplier_admission as stageo,
    )

    return stageo.probe_environment(Path(root).resolve())


def run_upper_gate_provenance(
    *,
    root: Path,
    environment: Mapping[str, Any],
    repository: Mapping[str, Any],
    spec: Optional[StagePSpec] = None,
) -> Mapping[str, Any]:
    active = StagePSpec() if spec is None else spec
    active.validate()
    validate_environment_variables()
    repo = Path(root).resolve()
    base_report = _validate_base_report(repo)
    base_cells = _cell_map(base_report)

    from ccda_phase3 import (
        phase314b_r258_stageo_dual_oracle_lower_multiplier_admission as stageo,
    )

    runtime = dict(stageo._prepare_runtime(repo, environment))
    runtime["stageo"] = stageo
    stagef = runtime["stagef"]
    context = runtime["context"]
    cell_records: List[Mapping[str, Any]] = []

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
            if key not in base_cells:
                raise StagePError(f"base Stage-O cell is missing: {key}")
            cell_records.append(
                audit_upper_gate_cell(
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

    if len(cell_records) != EXPECTED_PAIR_COUNT:
        raise StagePError("Stage-P replay pair population changed")
    classification = classify_upper_gate_provenance(cell_records, active)
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
        "base_stageo_resume1": {
            "report_sha256": EXPECTED_BASE_REPORT_SHA256,
            "recovered_stageo_payload_sha256": (
                EXPECTED_RECOVERED_STAGEO_PAYLOAD_SHA256
            ),
            "stageo_scientific_result_sha256": (
                EXPECTED_STAGEO_SCIENTIFIC_RESULT_SHA256
            ),
            "root_cause": EXPECTED_BASE_ROOT_CAUSE,
            "required_next_path": EXPECTED_BASE_NEXT_PATH,
            "six_oracle_cells_have_zero_admission": True,
            "six_oracle_cells_have_stable_upper_failure": True,
        },
        "upper_segment_gate_provenance": {
            "external_multiplier": LOWER_MULTIPLIER,
            "oracle_sources": list(ORACLE_SOURCES),
            "timesteps": list(EXPECTED_TIMESTEPS),
            "callback_off_on_pair_count": len(cell_records),
            "all_callback_results_bit_exact": all(
                cell.get("callback_result_bit_exact") is True
                for cell in cell_records
            ),
            "all_cells_match_stageo_resume1_identity": all(
                _mapping(
                    cell.get("stageo_resume1_identity"),
                    "Stage-O Resume1 identity",
                ).get("all_exact")
                is True
                for cell in cell_records
            ),
            "direct_upper_masks_match_stagek_callback": all(
                all(
                    _mapping(attempt, "attempt").get(
                        "callback_direct_upper_exact"
                    )
                    is True
                    for attempt in _sequence(
                        cell.get("attempt_records"), "attempt records"
                    )
                )
                for cell in cell_records
            ),
            "cell_records": cell_records,
            "classification": classification,
            "row_level_arrays_persisted": False,
            "callback_events_persisted": False,
            "direction_tensors_persisted": False,
            "proposal_tensors_persisted": False,
            "candidate_tensors_persisted": False,
            "predicate_masks_persisted": False,
            "only_aggregate_4x23_position_counts_and_scalar_summaries_persisted": True,
        },
        "immutable_inputs": {
            "stageo_resume1_report_sha256": EXPECTED_BASE_REPORT_SHA256,
            "recovered_stageo_payload_sha256": (
                EXPECTED_RECOVERED_STAGEO_PAYLOAD_SHA256
            ),
            "stageo_scientific_result_sha256": (
                EXPECTED_STAGEO_SCIENTIFIC_RESULT_SHA256
            ),
            "original_stageo_blocked_report_sha256": (
                EXPECTED_ORIGINAL_STAGEO_BLOCKED_REPORT_SHA256
            ),
            "stage_l_132_callback_pairs_rerun": False,
            "stage_o_resume1_report_rewritten": False,
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
            "integrator_thresholds_changed": False,
            "integrator_scale_search_changed": False,
            "external_scale_bank_changed": False,
            "segment_reconstruction_changed": False,
            "upper_gate_changed": False,
            "oracle_source_population_changed": False,
            "timestep_population_changed": False,
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
        "root_cause": "phase314b_r258_stagep_upper_gate_provenance_execution_failed",
        "required_next_path": "RESTORE_STAGEP_ORACLE_UPPER_GATE_PROVENANCE_AUDIT",
        "primary_failure_locus": "execution_contract",
        "selected_configuration": None,
        "train_only_recommendation": None,
        "repository": None if repository is None else dict(repository),
        "error_type": type(error).__name__,
        "error_message": str(error),
        "stageo_resume1_report_preserved": True,
        "stageo_resume1_report_sha256": EXPECTED_BASE_REPORT_SHA256,
        "original_stageo_blocked_report_preserved": True,
        "original_stageo_blocked_report_sha256": (
            EXPECTED_ORIGINAL_STAGEO_BLOCKED_REPORT_SHA256
        ),
        "stage_l_132_callback_pairs_rerun": False,
        "oof_surrogate_refit": False,
        "oof_surrogate_evaluated": False,
        **{key: False for key in FALSE_BOUNDARIES},
    }
