"""Phase3.14b-r2.5.6 Stage D segment-gate recalibration.

Stage C established that the historical segment gate is not self-calibrated:
the diagnostic-training ground truth itself passed at only 0.94508 under a
nominal 0.95 candidate-all threshold.  This stage therefore changes only the
segment-gate statistical contract.

The Stage-B cable model, scheduler, training split, normalization, objective,
seed, candidate count and generated candidates are replayed exactly.  The new
gate is fitted and calibrated exclusively from the Stage-B diagnostic-training
groups.  The frozen probe groups and all model predictions are evaluation-only.

No model, checkpoint, prediction tensor, candidate tensor or NPZ is persisted.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from ccda_phase3 import phase314b_r256_stageb_cable_diffusion as stageb
from ccda_phase3 import phase314b_r256_stagec_reverse_attribution as stagec

PHASE = "Phase3.14b-r2.5.6 Stage D"
PHASE_ID = "phase314b_r256_staged"
BASE_EVIDENCE_COMMIT = "07fc6da908fba91ed4008d4ff385950c5aec5940"
EXPECTED_SUBMODULE_COMMIT = "633a88752445cf5d6776ed374fdbbdb35f93050c"

STAGEC_SOURCE = "ccda_phase3/phase314b_r256_stagec_reverse_attribution.py"
STAGEC_TEST_GATE = "reports/phase3_14b_r256_stagec_test_gate_summary.json"
STAGEC_WORKER_EVIDENCE = (
    "reports/phase3_14b_r256_stagec_worker_evidence.json"
)
STAGEC_SUMMARY = "reports/phase3_14b_r256_stagec_summary.json"
STAGEC_REPORT = "reports/phase3_14b_r256_stagec_report.md"

BASE_BOUND_FILES = (
    STAGEC_SOURCE,
    STAGEC_TEST_GATE,
    STAGEC_WORKER_EVIDENCE,
    STAGEC_SUMMARY,
    STAGEC_REPORT,
)

SNAPSHOT_TIMESTEPS = (99, 75, 50, 25, 10, 0)


class SegmentGateRecalibrationError(RuntimeError):
    """Raised when frozen replay or recalibration invariants fail."""


@dataclass(frozen=True)
class RecalibrationSpec:
    """Pre-registered segment-gate recalibration contract."""

    fit_modulus: int = 2
    fit_remainder: int = 0
    calibration_remainder: int = 1

    robust_scale_floor: float = 1.0e-3
    within_group_row_coverage: float = 0.95
    conformal_group_coverage: float = 0.95

    probe_row_acceptance_min: float = 0.90
    probe_group_acceptance_min: float = 0.80
    probe_condition_acceptance_min: float = 0.85

    one_step_t10_segment_acceptance_min: float = 0.50
    reverse_segment_row_any_min: float = 0.95
    reverse_combined_row_any_min: float = 0.95
    physical_branch_eligible_row_min: float = 0.95
    physical_branch_support_min: float = 0.75

    bootstrap_resamples: int = 4096
    bootstrap_seed: int = 256400
    snapshot_timesteps: Tuple[int, ...] = SNAPSHOT_TIMESTEPS
    physical_branch_prefix_k: Tuple[int, ...] = (1, 2, 4, 8)

    def validate(self) -> None:
        if self.fit_modulus < 2:
            raise ValueError("fit/calibration modulus must be at least two")
        remainders = {
            int(self.fit_remainder),
            int(self.calibration_remainder),
        }
        if len(remainders) != 2:
            raise ValueError("fit and calibration remainders must differ")
        if any(value < 0 or value >= self.fit_modulus for value in remainders):
            raise ValueError("fit/calibration remainder is outside modulus")
        for value in (
            self.within_group_row_coverage,
            self.conformal_group_coverage,
            self.probe_row_acceptance_min,
            self.probe_group_acceptance_min,
            self.probe_condition_acceptance_min,
            self.one_step_t10_segment_acceptance_min,
            self.reverse_segment_row_any_min,
            self.reverse_combined_row_any_min,
            self.physical_branch_eligible_row_min,
            self.physical_branch_support_min,
        ):
            if not 0.0 < float(value) <= 1.0:
                raise ValueError("coverage threshold is outside (0,1]")
        if self.robust_scale_floor <= 0.0:
            raise ValueError("robust scale floor must be positive")
        if self.bootstrap_resamples <= 0:
            raise ValueError("bootstrap count must be positive")
        if tuple(sorted(set(self.snapshot_timesteps), reverse=True)) != (
            self.snapshot_timesteps
        ):
            raise ValueError("snapshot timesteps must be unique and descending")
        if self.snapshot_timesteps[0] != 99 or self.snapshot_timesteps[-1] != 0:
            raise ValueError("snapshot contract must cover t=99 through t=0")
        if self.physical_branch_prefix_k[-1] != (
            stageb.DiagnosticSpec().reverse_candidates
        ):
            raise ValueError("physical branch K curve must end at frozen K")


@dataclass(frozen=True)
class SegmentReference:
    """Robust per-horizon/per-segment log-length reference."""

    center_log: np.ndarray
    scale_log: np.ndarray
    mad_scale: np.ndarray
    iqr_scale: np.ndarray
    scale_floor: float

    def validate(self) -> None:
        expected = (stageb.FUTURE_STEPS, stageb.BEADS - 1)
        for name, value in (
            ("center_log", self.center_log),
            ("scale_log", self.scale_log),
            ("mad_scale", self.mad_scale),
            ("iqr_scale", self.iqr_scale),
        ):
            array = np.asarray(value)
            if array.shape != expected:
                raise ValueError(f"{name} shape {array.shape} != {expected}")
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} contains NaN or Inf")
        if np.any(self.scale_log < float(self.scale_floor)):
            raise ValueError("reference scale is below its registered floor")

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return {
            "center_log": self.center_log.tolist(),
            "scale_log": self.scale_log.tolist(),
            "mad_scale": self.mad_scale.tolist(),
            "iqr_scale": self.iqr_scale.tolist(),
            "scale_floor": float(self.scale_floor),
            "center_log_sha256": sha256_array(self.center_log),
            "scale_log_sha256": sha256_array(self.scale_log),
            "mad_scale_sha256": sha256_array(self.mad_scale),
            "iqr_scale_sha256": sha256_array(self.iqr_scale),
        }


@dataclass(frozen=True)
class SegmentGateContract:
    """Cluster-aware split-conformal candidate-level segment gate."""

    reference: SegmentReference
    joint_threshold: float
    lower_threshold: float
    upper_threshold: float
    within_group_row_coverage: float
    conformal_group_coverage: float
    fit_group_count: int
    calibration_group_count: int
    fit_group_sha256: str
    calibration_group_sha256: str

    def validate(self) -> None:
        self.reference.validate()
        for name, value in (
            ("joint_threshold", self.joint_threshold),
            ("lower_threshold", self.lower_threshold),
            ("upper_threshold", self.upper_threshold),
        ):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} is invalid")
        if self.fit_group_count <= 0 or self.calibration_group_count <= 0:
            raise ValueError("gate split is empty")
        if not 0.0 < self.within_group_row_coverage <= 1.0:
            raise ValueError("within-group coverage is invalid")
        if not 0.0 < self.conformal_group_coverage <= 1.0:
            raise ValueError("conformal group coverage is invalid")

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return {
            "schema": "phase314b_r256_staged_segment_gate_contract_v1",
            "score": (
                "maximum absolute robust log-segment z-score over "
                "4 horizons x 23 ordered segments"
            ),
            "primary_gate": "joint_score <= joint_threshold",
            "candidate_positions": int(
                stageb.FUTURE_STEPS * (stageb.BEADS - 1)
            ),
            "joint_threshold": float(self.joint_threshold),
            "lower_threshold": float(self.lower_threshold),
            "upper_threshold": float(self.upper_threshold),
            "within_group_row_coverage":
                float(self.within_group_row_coverage),
            "conformal_group_coverage":
                float(self.conformal_group_coverage),
            "fit_group_count": int(self.fit_group_count),
            "calibration_group_count": int(self.calibration_group_count),
            "fit_group_sha256": self.fit_group_sha256,
            "calibration_group_sha256": self.calibration_group_sha256,
            "reference": self.reference.to_dict(),
            "uses_condition_label": False,
            "uses_probe_target_for_fit": False,
            "uses_model_candidate_for_fit": False,
        }


def git_output(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=str(root),
        text=True,
    ).strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(json.dumps(list(array.shape)).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def sha256_strings(values: Sequence[Any]) -> str:
    payload = "\n".join(str(value) for value in values).encode("utf-8")
    return sha256_bytes(payload)


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        if value.dtype.kind in "fc" and not np.all(np.isfinite(value)):
            raise ValueError("non-finite array cannot be serialized")
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
        if not np.isfinite(value):
            raise ValueError("non-finite float cannot be serialized")
        return value
    raise TypeError(f"unsupported JSON value: {type(value)!r}")


def stable_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            jsonable(dict(payload)),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def atomic_write_once(path: Path, payload: bytes, *, mode: int = 0o644) -> None:
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite write-once output: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.{os.getpid()}.tmp"
    if temporary.exists():
        temporary.unlink()
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, target)
        directory_fd = os.open(str(target.parent), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SegmentGateRecalibrationError(
            f"JSON root is not an object: {path}"
        )
    return value


def assert_base_file_bound(root: Path, relative: str) -> str:
    path = Path(root) / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    committed = subprocess.check_output(
        ["git", "show", f"{BASE_EVIDENCE_COMMIT}:{relative}"],
        cwd=str(root),
    )
    observed = path.read_bytes()
    if committed != observed:
        raise SegmentGateRecalibrationError(
            f"base-bound file differs from {BASE_EVIDENCE_COMMIT}: {relative}"
        )
    return sha256_bytes(observed)


def validate_immutable_inputs(root: Path) -> Dict[str, Any]:
    repository_root = Path(root).resolve()
    base_files = {
        relative: assert_base_file_bound(repository_root, relative)
        for relative in BASE_BOUND_FILES
    }
    summary = load_json(repository_root / STAGEC_SUMMARY)
    if summary.get("verdict") != "PASS":
        raise SegmentGateRecalibrationError("Stage-C verdict is not PASS")
    if summary.get("scientific_status") != "BLOCKED":
        raise SegmentGateRecalibrationError(
            "Stage-C scientific status changed"
        )
    if summary.get("root_cause") != (
        "phase314b_r256_stagec_segment_gate_self_calibration_failed"
    ):
        raise SegmentGateRecalibrationError(
            "Stage-C root cause changed"
        )
    if summary.get("required_next_path") != (
        "RECALIBRATE_FROZEN_CABLE_SEGMENT_GATE_BEFORE_MODEL_OR_BRANCH_REPAIR"
    ):
        raise SegmentGateRecalibrationError(
            "Stage-C next path changed"
        )
    if summary.get("train_only_recommendation") is not None:
        raise SegmentGateRecalibrationError(
            "Stage-C selected a train-only recommendation"
        )
    if summary.get("selected_configuration") is not None:
        raise SegmentGateRecalibrationError(
            "Stage-C selected a configuration"
        )

    worker_evidence = load_json(repository_root / STAGEC_WORKER_EVIDENCE)
    if worker_evidence.get("workers_exact") is not True:
        raise SegmentGateRecalibrationError(
            "Stage-C workers were not exact"
        )
    worker_result = worker_evidence.get("worker_result")
    if not isinstance(worker_result, dict):
        raise SegmentGateRecalibrationError(
            "Stage-C worker result is missing"
        )
    if worker_result.get("root_cause") != summary.get("root_cause"):
        raise SegmentGateRecalibrationError(
            "Stage-C summary/worker root causes differ"
        )
    identity = worker_result.get("frozen_replay", {}).get("identity")
    if not isinstance(identity, dict):
        raise SegmentGateRecalibrationError(
            "Stage-C frozen identity is missing"
        )
    if identity.get("final_model_sha256") != (
        stagec.EXPECTED_HISTORICAL_MODEL_SHA256
    ):
        raise SegmentGateRecalibrationError(
            "Stage-C model identity changed"
        )
    if identity.get("reverse_candidate_sha256") != (
        stagec.EXPECTED_HISTORICAL_REVERSE_SHA256
    ):
        raise SegmentGateRecalibrationError(
            "Stage-C reverse identity changed"
        )

    # Reuse Stage C's complete Stage-B immutable-input validation.
    stageb_immutable = stagec.validate_immutable_inputs(repository_root)
    return {
        "base_commit": BASE_EVIDENCE_COMMIT,
        "base_file_sha256": base_files,
        "stagec_summary": summary,
        "stagec_worker_evidence": worker_evidence,
        "stageb_immutable": stageb_immutable["file_sha256"],
    }


def source_logic_audit(root: Path) -> Dict[str, Any]:
    stageb_text = (
        Path(root) / "ccda_phase3/phase314b_r256_stageb_cable_diffusion.py"
    ).read_text(encoding="utf-8")
    stagec_text = (
        Path(root) / STAGEC_SOURCE
    ).read_text(encoding="utf-8")
    checks = {
        "historical_gate_uses_positionwise_bounds": (
            "contract.segment_lower[None, None]" in stageb_text
            and "contract.segment_upper[None, None]" in stageb_text
        ),
        "historical_gate_requires_all_92_positions": (
            "axis=(2, 3)" in stageb_text
            and '"segment": segment' in stageb_text
        ),
        "stagec_detected_self_calibration_failure": (
            "phase314b_r256_stagec_segment_gate_self_calibration_failed"
            in stagec_text
        ),
        "stagec_did_not_change_threshold": (
            '"threshold_changed": False' in stagec_text
        ),
        "stagec_branch_audit_unavailable_without_valid_candidates": (
            "no physical-valid reverse candidate under historical gate"
            in stagec_text
        ),
    }
    return {
        "checks": checks,
        "all_confirmed": bool(all(checks.values())),
        "stageb_source_sha256": sha256_file(
            Path(root) /
            "ccda_phase3/phase314b_r256_stageb_cable_diffusion.py"
        ),
        "stagec_source_sha256": sha256_file(Path(root) / STAGEC_SOURCE),
    }


def _higher_quantile(values: np.ndarray, coverage: float) -> float:
    array = np.sort(np.asarray(values, dtype=np.float64).reshape(-1))
    if array.size == 0:
        raise ValueError("quantile input is empty")
    if not 0.0 < float(coverage) <= 1.0:
        raise ValueError("coverage is outside (0,1]")
    rank = int(math.ceil(float(coverage) * array.size))
    rank = min(max(rank, 1), int(array.size))
    return float(array[rank - 1])


def _conformal_quantile(values: np.ndarray, coverage: float) -> Dict[str, Any]:
    array = np.sort(np.asarray(values, dtype=np.float64).reshape(-1))
    if array.size == 0:
        raise ValueError("conformal input is empty")
    rank = int(math.ceil((array.size + 1) * float(coverage)))
    rank = min(max(rank, 1), int(array.size))
    threshold = float(array[rank - 1])
    return {
        "threshold": threshold,
        "sample_count": int(array.size),
        "order_statistic_rank": rank,
        "empirical_acceptance": float(np.mean(array <= threshold)),
        "score_sha256": sha256_array(array),
    }


def deterministic_gate_split(
    groups: Sequence[Any],
    stageb_train_mask: np.ndarray,
    *,
    spec: RecalibrationSpec,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    values = np.asarray(groups).astype(str)
    train_mask = np.asarray(stageb_train_mask, dtype=np.bool_)
    if values.shape != train_mask.shape:
        raise ValueError("group array and Stage-B train mask differ")
    train_groups = sorted(set(values[train_mask].tolist()))
    if len(train_groups) < 4:
        raise SegmentGateRecalibrationError(
            "insufficient Stage-B training groups for recalibration"
        )
    assignment = {
        group: index % int(spec.fit_modulus)
        for index, group in enumerate(train_groups)
    }
    fit_groups = [
        group
        for group in train_groups
        if assignment[group] == int(spec.fit_remainder)
    ]
    calibration_groups = [
        group
        for group in train_groups
        if assignment[group] == int(spec.calibration_remainder)
    ]
    if not fit_groups or not calibration_groups:
        raise SegmentGateRecalibrationError(
            "fit/calibration gate split is empty"
        )
    fit_set = set(fit_groups)
    calibration_set = set(calibration_groups)
    if fit_set.intersection(calibration_set):
        raise AssertionError("gate fit/calibration groups overlap")
    fit_mask = train_mask & np.asarray(
        [group in fit_set for group in values],
        dtype=np.bool_,
    )
    calibration_mask = train_mask & np.asarray(
        [group in calibration_set for group in values],
        dtype=np.bool_,
    )
    if np.any(fit_mask & calibration_mask):
        raise AssertionError("gate fit/calibration rows overlap")
    if not np.array_equal(fit_mask | calibration_mask, train_mask):
        unused = train_mask & ~(fit_mask | calibration_mask)
        if np.any(unused):
            raise SegmentGateRecalibrationError(
                "registered modulus left Stage-B training groups unused"
            )
    return fit_mask, calibration_mask, {
        "fit_group_count": len(fit_groups),
        "calibration_group_count": len(calibration_groups),
        "fit_row_count": int(np.sum(fit_mask)),
        "calibration_row_count": int(np.sum(calibration_mask)),
        "fit_group_sha256": sha256_strings(fit_groups),
        "calibration_group_sha256":
            sha256_strings(calibration_groups),
        "group_overlap": 0,
        "row_overlap": 0,
        "all_stageb_training_rows_assigned": True,
    }


def fit_segment_reference(
    target: np.ndarray,
    *,
    scale_floor: float,
) -> SegmentReference:
    value = np.asarray(target, dtype=np.float32)
    if value.ndim != 3 or value.shape[1:] != (
        stageb.FUTURE_STEPS,
        stageb.CABLE_DIM,
    ):
        raise ValueError("reference target must be [N,4,48]")
    lengths = stageb.segment_lengths(value).astype(np.float64)
    if np.any(lengths <= 0.0) or not np.all(np.isfinite(lengths)):
        raise SegmentGateRecalibrationError(
            "reference segment lengths are invalid"
        )
    log_lengths = np.log(lengths)
    center = np.median(log_lengths, axis=0)
    absolute_deviation = np.abs(log_lengths - center[None])
    mad_scale = 1.4826 * np.median(absolute_deviation, axis=0)
    q25 = np.percentile(log_lengths, 25.0, axis=0)
    q75 = np.percentile(log_lengths, 75.0, axis=0)
    iqr_scale = (q75 - q25) / 1.349
    scale = np.maximum.reduce(
        [
            mad_scale,
            iqr_scale,
            np.full_like(mad_scale, float(scale_floor)),
        ]
    )
    result = SegmentReference(
        center_log=center.astype(np.float32),
        scale_log=scale.astype(np.float32),
        mad_scale=mad_scale.astype(np.float32),
        iqr_scale=iqr_scale.astype(np.float32),
        scale_floor=float(scale_floor),
    )
    result.validate()
    return result


def as_candidates(value: np.ndarray) -> np.ndarray:
    return stagec.as_candidates(value)


def segment_scores(
    value: np.ndarray,
    reference: SegmentReference,
) -> Dict[str, np.ndarray]:
    reference.validate()
    candidates = as_candidates(value)
    lengths = stageb.segment_lengths(candidates).astype(np.float64)
    nonpositive = lengths <= 0.0
    safe_lengths = np.maximum(lengths, np.finfo(np.float64).tiny)
    log_lengths = np.log(safe_lengths)
    z = (
        log_lengths - reference.center_log[None, None].astype(np.float64)
    ) / reference.scale_log[None, None].astype(np.float64)
    lower_element = np.maximum(-z, 0.0)
    upper_element = np.maximum(z, 0.0)
    lower_score = np.max(lower_element, axis=(2, 3))
    upper_score = np.max(upper_element, axis=(2, 3))
    joint_score = np.maximum(lower_score, upper_score)
    return {
        "joint": joint_score,
        "lower": lower_score,
        "upper": upper_score,
        "z": z,
        "lower_element": lower_element,
        "upper_element": upper_element,
        "nonpositive": nonpositive,
    }


def _group_score_distribution(
    row_scores: np.ndarray,
    groups: Sequence[Any],
    *,
    within_group_coverage: float,
) -> Tuple[np.ndarray, List[str]]:
    score = np.asarray(row_scores, dtype=np.float64)
    if score.ndim != 1:
        raise ValueError("group calibration requires one score per row")
    group_values = np.asarray(groups).astype(str)
    if group_values.shape != score.shape:
        raise ValueError("group calibration shape mismatch")
    records: MutableMapping[str, List[float]] = {}
    for group, value in zip(group_values, score):
        records.setdefault(group, []).append(float(value))
    ordered_groups = sorted(records)
    group_scores = np.asarray(
        [
            _higher_quantile(
                np.asarray(records[group], dtype=np.float64),
                within_group_coverage,
            )
            for group in ordered_groups
        ],
        dtype=np.float64,
    )
    return group_scores, ordered_groups


def fit_segment_gate(
    *,
    fit_target: np.ndarray,
    calibration_target: np.ndarray,
    calibration_groups: Sequence[Any],
    split_record: Mapping[str, Any],
    spec: RecalibrationSpec,
) -> Tuple[SegmentGateContract, Dict[str, Any]]:
    reference = fit_segment_reference(
        fit_target,
        scale_floor=spec.robust_scale_floor,
    )
    scores = segment_scores(calibration_target, reference)
    if scores["joint"].shape[1] != 1:
        raise ValueError("ground-truth calibration must have K=1")
    row_joint = scores["joint"][:, 0]
    row_lower = scores["lower"][:, 0]
    row_upper = scores["upper"][:, 0]

    group_joint, ordered_groups = _group_score_distribution(
        row_joint,
        calibration_groups,
        within_group_coverage=spec.within_group_row_coverage,
    )
    group_lower, ordered_lower = _group_score_distribution(
        row_lower,
        calibration_groups,
        within_group_coverage=spec.within_group_row_coverage,
    )
    group_upper, ordered_upper = _group_score_distribution(
        row_upper,
        calibration_groups,
        within_group_coverage=spec.within_group_row_coverage,
    )
    if not (
        ordered_groups == ordered_lower == ordered_upper
    ):
        raise AssertionError("one-sided calibration group orders differ")

    joint_quantile = _conformal_quantile(
        group_joint,
        spec.conformal_group_coverage,
    )
    lower_quantile = _conformal_quantile(
        group_lower,
        spec.conformal_group_coverage,
    )
    upper_quantile = _conformal_quantile(
        group_upper,
        spec.conformal_group_coverage,
    )
    contract = SegmentGateContract(
        reference=reference,
        joint_threshold=joint_quantile["threshold"],
        lower_threshold=lower_quantile["threshold"],
        upper_threshold=upper_quantile["threshold"],
        within_group_row_coverage=spec.within_group_row_coverage,
        conformal_group_coverage=spec.conformal_group_coverage,
        fit_group_count=int(split_record["fit_group_count"]),
        calibration_group_count=int(
            split_record["calibration_group_count"]
        ),
        fit_group_sha256=str(split_record["fit_group_sha256"]),
        calibration_group_sha256=str(
            split_record["calibration_group_sha256"]
        ),
    )
    contract.validate()
    return contract, {
        "joint": joint_quantile,
        "lower": lower_quantile,
        "upper": upper_quantile,
        "calibration_group_score_sha256":
            sha256_array(group_joint),
        "calibration_group_count": len(ordered_groups),
        "calibration_group_order_sha256":
            sha256_strings(ordered_groups),
        "calibration_row_joint_score": _safe_stats(row_joint),
        "calibration_row_lower_score": _safe_stats(row_lower),
        "calibration_row_upper_score": _safe_stats(row_upper),
    }


def _safe_stats(value: np.ndarray) -> Dict[str, float]:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "p05": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "minimum": 0.0,
            "maximum": 0.0,
        }
    if not np.all(np.isfinite(array)):
        raise SegmentGateRecalibrationError(
            "statistics input contains NaN or Inf"
        )
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "p05": float(np.percentile(array, 5.0)),
        "p50": float(np.percentile(array, 50.0)),
        "p95": float(np.percentile(array, 95.0)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def _group_acceptance(
    row_pass: np.ndarray,
    groups: Sequence[Any],
    *,
    within_group_coverage: float,
) -> Dict[str, Any]:
    passed = np.asarray(row_pass, dtype=np.bool_)
    group_values = np.asarray(groups).astype(str)
    if passed.shape != group_values.shape:
        raise ValueError("group acceptance shape mismatch")
    records: MutableMapping[str, List[bool]] = {}
    for group, value in zip(group_values, passed):
        records.setdefault(group, []).append(bool(value))
    ordered = sorted(records)
    fractions = np.asarray(
        [
            float(np.mean(records[group]))
            for group in ordered
        ],
        dtype=np.float64,
    )
    group_pass = fractions >= float(within_group_coverage)
    return {
        "group_count": len(ordered),
        "accepted_group_count": int(np.sum(group_pass)),
        "group_acceptance_rate": float(np.mean(group_pass)),
        "within_group_row_coverage":
            float(within_group_coverage),
        "group_row_fraction": _safe_stats(fractions),
        "group_order_sha256": sha256_strings(ordered),
        "group_fraction_sha256": sha256_array(fractions),
    }


def recalibrated_physical_decomposition(
    *,
    value: np.ndarray,
    contract: SegmentGateContract,
    historical_geometry: stageb.GeometryContract,
    groups: Sequence[Any],
    condition_name: Sequence[Any],
) -> Dict[str, Any]:
    contract.validate()
    candidates = as_candidates(value)
    scores = segment_scores(candidates, contract.reference)
    segment_pass = scores["joint"] <= float(contract.joint_threshold)

    historical = stageb.physical_validity(
        candidates,
        historical_geometry,
    )
    finite = np.asarray(historical["finite"], dtype=np.bool_)
    coordinate = np.asarray(historical["coordinate"], dtype=np.bool_)
    topology = np.asarray(historical["topology"], dtype=np.bool_)
    combined = finite & coordinate & segment_pass & topology

    row_segment_any = np.any(segment_pass, axis=1)
    row_combined_any = np.any(combined, axis=1)
    group_values = np.asarray(groups).astype(str)
    conditions = np.asarray(condition_name).astype(str)
    if group_values.shape != (candidates.shape[0],):
        raise ValueError("population group shape mismatch")
    if conditions.shape != group_values.shape:
        raise ValueError("population condition shape mismatch")

    z = scores["z"]
    absolute_z = np.abs(z)
    element_pass = absolute_z <= float(contract.joint_threshold)
    position_rate = np.mean(element_pass, axis=(0, 1))
    worst_positions = []
    order = np.argsort(position_rate.reshape(-1))
    for flat_index in order[: min(12, order.size)]:
        horizon, segment_index = np.unravel_index(
            int(flat_index),
            position_rate.shape,
        )
        worst_positions.append(
            {
                "horizon": int(horizon),
                "segment_index": int(segment_index),
                "element_pass_rate":
                    float(position_rate[horizon, segment_index]),
                "absolute_z": _safe_stats(
                    absolute_z[..., horizon, segment_index]
                ),
            }
        )

    condition_segment_rate = {}
    condition_combined_rate = {}
    for name in sorted(set(conditions.tolist())):
        selected = conditions == name
        condition_segment_rate[name] = float(
            np.mean(row_segment_any[selected])
        )
        condition_combined_rate[name] = float(
            np.mean(row_combined_any[selected])
        )

    return {
        "shape": list(candidates.shape),
        "sha256": sha256_array(candidates),
        "segment": {
            "candidate_rate": float(np.mean(segment_pass)),
            "row_any_rate": float(np.mean(row_segment_any)),
            "element_rate": float(np.mean(element_pass)),
            "row_joint_score": _safe_stats(
                np.min(scores["joint"], axis=1)
            ),
            "all_candidate_joint_score": _safe_stats(
                scores["joint"]
            ),
            "lower_score": _safe_stats(scores["lower"]),
            "upper_score": _safe_stats(scores["upper"]),
            "candidate_count": int(segment_pass.size),
            "accepted_candidate_count": int(np.sum(segment_pass)),
            "nonpositive_element_count":
                int(np.sum(scores["nonpositive"])),
            "condition_row_any_rate": condition_segment_rate,
            "group_acceptance": _group_acceptance(
                row_segment_any,
                group_values,
                within_group_coverage=
                    contract.within_group_row_coverage,
            ),
            "worst_horizon_segment_positions": worst_positions,
        },
        "combined": {
            "candidate_rate": float(np.mean(combined)),
            "row_any_rate": float(np.mean(row_combined_any)),
            "candidate_count": int(combined.size),
            "accepted_candidate_count": int(np.sum(combined)),
            "condition_row_any_rate": condition_combined_rate,
            "group_acceptance": _group_acceptance(
                row_combined_any,
                group_values,
                within_group_coverage=
                    contract.within_group_row_coverage,
            ),
        },
        "unchanged_subgates": {
            "finite_candidate_rate": float(np.mean(finite)),
            "coordinate_candidate_rate": float(np.mean(coordinate)),
            "topology_candidate_rate": float(np.mean(topology)),
            "historical_segment_candidate_rate":
                float(np.mean(historical["segment"])),
            "historical_combined_candidate_rate":
                float(np.mean(historical["valid"])),
        },
        "segment_valid_mask": segment_pass,
        "combined_valid_mask": combined,
    }


def public_population_record(
    decomposition: Mapping[str, Any],
) -> Dict[str, Any]:
    """Drop non-JSON masks while retaining every metric and digest."""
    return {
        key: value
        for key, value in decomposition.items()
        if key not in {"segment_valid_mask", "combined_valid_mask"}
    }


def reverse_sample_with_raw_trace(
    *,
    model: Any,
    condition: np.ndarray,
    spec: stageb.DiagnosticSpec,
    condition_standardizer: stageb.ArrayStandardizer,
    target_standardizer: stageb.ArrayStandardizer,
    noise_seed: int,
    snapshot_timesteps: Sequence[int],
) -> Tuple[np.ndarray, Dict[str, Dict[str, np.ndarray]]]:
    """Replay the exact Stage-B deterministic reverse and retain RAM-only snapshots."""
    torch, _ = stageb._torch_imports()
    device = next(model.parameters()).device
    scheduler = stageb.scheduler_arrays(spec)
    alpha_bar = scheduler["alpha_bar"].astype(np.float64)
    rows = condition.shape[0]
    candidates = spec.reverse_candidates
    initial = stageb._fixed_noise(
        (rows, candidates, stageb.FUTURE_STEPS, stageb.CABLE_DIM),
        seed=noise_seed,
    )
    value = torch.as_tensor(
        initial.reshape(
            rows * candidates,
            stageb.FUTURE_STEPS,
            stageb.CABLE_DIM,
        ),
        dtype=torch.float32,
        device=device,
    )
    condition_z = condition_standardizer.normalize(condition)
    condition_repeated = np.repeat(
        condition_z[:, None, :],
        candidates,
        axis=1,
    ).reshape(rows * candidates, stageb.CONDITION_DIM)
    condition_tensor = torch.as_tensor(
        condition_repeated,
        dtype=torch.float32,
        device=device,
    )
    wanted = set(int(item) for item in snapshot_timesteps)
    snapshots: Dict[str, Dict[str, np.ndarray]] = {}
    model.eval()
    with torch.no_grad():
        for timestep_value in reversed(range(spec.reverse_steps)):
            timestep = torch.full(
                (rows * candidates,),
                timestep_value,
                dtype=torch.long,
                device=device,
            )
            predicted_x0 = model(value, timestep, condition_tensor)
            if timestep_value in wanted and timestep_value != 0:
                latent_z = value.detach().cpu().numpy().reshape(
                    rows,
                    candidates,
                    stageb.FUTURE_STEPS,
                    stageb.CABLE_DIM,
                )
                predicted_z = predicted_x0.detach().cpu().numpy().reshape(
                    rows,
                    candidates,
                    stageb.FUTURE_STEPS,
                    stageb.CABLE_DIM,
                )
                snapshots[str(timestep_value)] = {
                    "latent": target_standardizer.denormalize(latent_z),
                    "predicted_x0":
                        target_standardizer.denormalize(predicted_z),
                }

            current_alpha = float(alpha_bar[timestep_value])
            if timestep_value == 0:
                value = predicted_x0
                continue
            previous_alpha = float(alpha_bar[timestep_value - 1])
            epsilon = (
                value - math.sqrt(current_alpha) * predicted_x0
            ) / math.sqrt(max(1.0 - current_alpha, 1.0e-12))
            value = (
                math.sqrt(previous_alpha) * predicted_x0
                + math.sqrt(max(1.0 - previous_alpha, 0.0)) * epsilon
            )

    final_z = value.detach().cpu().numpy().reshape(
        rows,
        candidates,
        stageb.FUTURE_STEPS,
        stageb.CABLE_DIM,
    )
    final = target_standardizer.denormalize(final_z)
    snapshots["0"] = {
        "latent": final,
        "predicted_x0": final,
    }
    missing = sorted(
        str(item)
        for item in wanted
        if str(item) not in snapshots
    )
    if missing:
        raise SegmentGateRecalibrationError(
            f"raw reverse trace is missing snapshots: {missing}"
        )
    return final, snapshots


def physical_branch_prefix_curve(
    *,
    pair_key: Sequence[Any],
    condition_name: Sequence[Any],
    target: np.ndarray,
    candidates: np.ndarray,
    combined_valid_mask: np.ndarray,
    prefix_k: Sequence[int],
) -> Dict[str, Any]:
    pred = as_candidates(candidates)
    valid = np.asarray(combined_valid_mask, dtype=np.bool_)
    if valid.shape != pred.shape[:2]:
        raise ValueError("physical branch mask shape mismatch")
    curve = {}
    for k in prefix_k:
        prefix = int(k)
        if not 1 <= prefix <= pred.shape[1]:
            raise ValueError(f"invalid physical branch K={prefix}")
        curve[str(prefix)] = stagec.physical_valid_branch_metrics(
            pair_key=pair_key,
            condition_name=condition_name,
            target=target,
            candidates=pred[:, :prefix],
            valid_mask=valid[:, :prefix],
        )
    return curve


def deterministic_eligible_branch_bootstrap(
    physical_branch: Mapping[str, Any],
    *,
    resamples: int,
    seed: int,
) -> Dict[str, Any]:
    rows = physical_branch.get("rows")
    if not isinstance(rows, list):
        raise ValueError("physical branch rows are missing")
    eligible_values = [
        float(record["supported"])
        for record in rows
        if record.get("eligible") is True
    ]
    if not eligible_values:
        return {
            "available": False,
            "eligible_rows": 0,
            "point_estimate": None,
            "ci95": None,
            "bootstrap_resamples": int(resamples),
            "bootstrap_seed": int(seed),
            "bootstrap_sha256": None,
        }
    values = np.asarray(eligible_values, dtype=np.float64)
    rng = np.random.RandomState(int(seed))
    sampled = np.empty(int(resamples), dtype=np.float64)
    for index in range(int(resamples)):
        selection = rng.randint(
            0,
            values.shape[0],
            size=values.shape[0],
        )
        sampled[index] = float(np.mean(values[selection]))
    return {
        "available": True,
        "eligible_rows": int(values.shape[0]),
        "point_estimate": float(np.mean(values)),
        "ci95": [
            float(np.percentile(sampled, 2.5)),
            float(np.percentile(sampled, 97.5)),
        ],
        "bootstrap_resamples": int(resamples),
        "bootstrap_seed": int(seed),
        "bootstrap_sha256": sha256_array(sampled),
    }


def classify_recalibration(
    *,
    spec: RecalibrationSpec,
    calibration_ground_truth: Mapping[str, Any],
    probe_ground_truth: Mapping[str, Any],
    one_step: Mapping[str, Mapping[str, Any]],
    reverse_final: Mapping[str, Any],
    physical_branch: Mapping[str, Any],
) -> Dict[str, Any]:
    calibration_group_rate = float(
        calibration_ground_truth["segment"]["group_acceptance"][
            "group_acceptance_rate"
        ]
    )
    probe_row_rate = float(
        probe_ground_truth["segment"]["row_any_rate"]
    )
    probe_group_rate = float(
        probe_ground_truth["segment"]["group_acceptance"][
            "group_acceptance_rate"
        ]
    )
    condition_rates = probe_ground_truth["segment"][
        "condition_row_any_rate"
    ]
    minimum_condition_rate = float(min(condition_rates.values()))

    one_step_t10 = float(
        one_step["10"]["segment"]["row_any_rate"]
    )
    reverse_segment_row_any = float(
        reverse_final["segment"]["row_any_rate"]
    )
    reverse_combined_row_any = float(
        reverse_final["combined"]["row_any_rate"]
    )
    eligible_rate = float(physical_branch["eligible_row_rate"])
    eligible_support = physical_branch["support_rate_among_eligible"]
    support_value = (
        float(eligible_support)
        if eligible_support is not None
        else 0.0
    )

    if calibration_group_rate < spec.conformal_group_coverage:
        root = (
            "phase314b_r256_staged_segment_gate_internal_"
            "self_calibration_failed"
        )
        next_path = (
            "REPAIR_GROUPED_CONFORMAL_SEGMENT_GATE_IMPLEMENTATION"
        )
        locus = "gate_implementation"
    elif (
        probe_row_rate < spec.probe_row_acceptance_min
        or probe_group_rate < spec.probe_group_acceptance_min
        or minimum_condition_rate <
            spec.probe_condition_acceptance_min
    ):
        root = (
            "phase314b_r256_staged_recalibrated_segment_gate_"
            "probe_generalization_failed"
        )
        next_path = (
            "STRATIFY_OR_REDEFINE_SEGMENT_GATE_WITH_TRAIN_ONLY_"
            "GROUPED_CALIBRATION"
        )
        locus = "gate_generalization"
    elif one_step_t10 < spec.one_step_t10_segment_acceptance_min:
        root = (
            "phase314b_r256_staged_recalibrated_gate_exposes_"
            "cable_x0_segment_failure"
        )
        next_path = (
            "ADD_ORDERED_SEGMENT_GEOMETRY_OBJECTIVE_TO_"
            "CABLE_X0_DENOISER"
        )
        locus = "x0_denoiser"
    elif reverse_segment_row_any < spec.reverse_segment_row_any_min:
        root = (
            "phase314b_r256_staged_recalibrated_gate_exposes_"
            "reverse_segment_coverage_failure"
        )
        next_path = (
            "REPAIR_CABLE_REVERSE_SEGMENT_TRANSPORT_WITH_"
            "FROZEN_RECALIBRATED_GATE"
        )
        locus = "reverse_segment_transport"
    elif reverse_combined_row_any < spec.reverse_combined_row_any_min:
        root = (
            "phase314b_r256_staged_recalibrated_segment_gate_"
            "passed_nonsegment_physical_failure_remains"
        )
        next_path = (
            "ATTRIBUTE_COORDINATE_OR_TOPOLOGY_FAILURE_WITH_"
            "FROZEN_SEGMENT_GATE"
        )
        locus = "nonsegment_physical_gate"
    elif eligible_rate < spec.physical_branch_eligible_row_min:
        root = (
            "phase314b_r256_staged_physical_candidate_"
            "row_coverage_insufficient"
        )
        next_path = (
            "REPAIR_PHYSICAL_CANDIDATE_COVERAGE_BEFORE_BRANCH_REPAIR"
        )
        locus = "physical_candidate_coverage"
    elif support_value < spec.physical_branch_support_min:
        root = (
            "phase314b_r256_staged_physical_valid_branch_support_failed"
        )
        next_path = (
            "REPAIR_CABLE_ONLY_BRANCH_TRANSPORT_WITH_"
            "FROZEN_RECALIBRATED_GATE"
        )
        locus = "branch_transport"
    else:
        root = (
            "phase314b_r256_staged_recalibrated_segment_gate_"
            "and_physical_branch_supported"
        )
        next_path = (
            "RUN_FROZEN_CABLE_ONLY_FORMAL_PILOT_AND_COLLECT_"
            "ACTION_DIVERSE_IDM_DATA"
        )
        locus = "none"

    return {
        "root_cause": root,
        "required_next_path": next_path,
        "primary_failure_locus": locus,
        "calibration_group_acceptance_rate":
            calibration_group_rate,
        "probe_segment_row_acceptance_rate": probe_row_rate,
        "probe_segment_group_acceptance_rate": probe_group_rate,
        "probe_minimum_condition_acceptance_rate":
            minimum_condition_rate,
        "one_step_t10_segment_acceptance_rate": one_step_t10,
        "reverse_segment_row_any_rate": reverse_segment_row_any,
        "reverse_combined_row_any_rate": reverse_combined_row_any,
        "physical_branch_eligible_row_rate": eligible_rate,
        "physical_branch_support_among_eligible":
            eligible_support,
    }


def run_recalibration(
    *,
    root: Path,
    spec: Optional[RecalibrationSpec] = None,
) -> Dict[str, Any]:
    active_spec = RecalibrationSpec() if spec is None else spec
    active_spec.validate()
    repository_root = Path(root).resolve()

    immutable = validate_immutable_inputs(repository_root)
    source_audit = source_logic_audit(repository_root)
    if not source_audit["all_confirmed"]:
        raise SegmentGateRecalibrationError(
            "historical source assumptions changed"
        )

    stageb_spec = stageb.DiagnosticSpec()
    stageb_spec.validate()
    arrays = stageb.load_npz_strict(
        repository_root / stageb.STAGEA_TRAIN_VIEW,
        required_keys=stageb.REQUIRED_TRAIN_KEYS,
    )
    validation = stageb.validate_train_view(arrays)
    runtime = stageb.set_deterministic_runtime(stageb_spec.seed)

    condition = np.asarray(
        arrays["diffusion_condition_x"],
        dtype=np.float32,
    )
    target = np.asarray(
        arrays["diffusion_target_cable"],
        dtype=np.float32,
    )
    groups = np.asarray(arrays["episode_group_key"]).astype(str)
    conditions = np.asarray(arrays["condition_name"]).astype(str)
    pair_key = np.asarray(arrays["pair_key"]).astype(str)

    train_mask, probe_mask, stageb_mapping = (
        stageb.deterministic_group_split(
            groups,
            folds=stageb_spec.group_folds,
            probe_fold=stageb_spec.probe_fold,
        )
    )
    fit_mask, calibration_mask, gate_split = (
        deterministic_gate_split(
            groups,
            train_mask,
            spec=active_spec,
        )
    )

    condition_standardizer = stageb.fit_standardizer(
        condition[train_mask]
    )
    target_standardizer = stageb.fit_standardizer(
        target[train_mask]
    )
    historical_geometry = stageb.fit_geometry_contract(
        target[train_mask]
    )

    model, training, _ = stageb.train_model(
        condition=condition[train_mask],
        target=target[train_mask],
        spec=stageb_spec,
        condition_standardizer=condition_standardizer,
        target_standardizer=target_standardizer,
    )
    expected_identity = immutable["stagec_worker_evidence"][
        "worker_result"
    ]["frozen_replay"]["identity"]
    training_identity = {
        "final_model_sha256": training["final_model_sha256"],
        "final_optimizer_sha256":
            training["final_optimizer_sha256"],
        "loss_history_sha256":
            training["loss_history_sha256"],
        "gradient_history_sha256":
            training["gradient_history_sha256"],
        "source_exposure_sha256":
            training["source_exposure_sha256"],
    }
    for key, observed in training_identity.items():
        if observed != expected_identity[key]:
            raise SegmentGateRecalibrationError(
                f"frozen Stage-B replay differs: {key}"
            )

    control_prediction = stagec.reproduce_train_control_prediction(
        model=model,
        train_condition=condition[train_mask],
        train_target=target[train_mask],
        spec=stageb_spec,
        condition_standardizer=condition_standardizer,
        target_standardizer=target_standardizer,
    )
    control_sha = sha256_array(control_prediction)
    if control_sha != expected_identity[
        "train_control_prediction_sha256"
    ]:
        raise SegmentGateRecalibrationError(
            "frozen train-control prediction differs"
        )

    one_step_predictions, one_step_sha = stagec.one_step_predictions(
        model=model,
        condition=condition[probe_mask],
        target=target[probe_mask],
        spec=stageb_spec,
        condition_standardizer=condition_standardizer,
        target_standardizer=target_standardizer,
        noise_seed=stageb_spec.seed + 2001,
    )
    if one_step_sha != expected_identity[
        "one_step_prediction_sha256"
    ]:
        raise SegmentGateRecalibrationError(
            "frozen one-step predictions differ"
        )

    final_candidates, raw_trace = reverse_sample_with_raw_trace(
        model=model,
        condition=condition[probe_mask],
        spec=stageb_spec,
        condition_standardizer=condition_standardizer,
        target_standardizer=target_standardizer,
        noise_seed=stageb_spec.seed + 3001,
        snapshot_timesteps=active_spec.snapshot_timesteps,
    )
    reverse_sha = sha256_array(final_candidates)
    if reverse_sha != expected_identity["reverse_candidate_sha256"]:
        raise SegmentGateRecalibrationError(
            "frozen reverse candidates differ"
        )

    gate, calibration_fit = fit_segment_gate(
        fit_target=target[fit_mask],
        calibration_target=target[calibration_mask],
        calibration_groups=groups[calibration_mask],
        split_record=gate_split,
        spec=active_spec,
    )

    def evaluate_population(
        value: np.ndarray,
        selected_mask: np.ndarray,
    ) -> Dict[str, Any]:
        return recalibrated_physical_decomposition(
            value=value,
            contract=gate,
            historical_geometry=historical_geometry,
            groups=groups[selected_mask],
            condition_name=conditions[selected_mask],
        )

    fit_ground_truth = evaluate_population(
        target[fit_mask],
        fit_mask,
    )
    calibration_ground_truth = evaluate_population(
        target[calibration_mask],
        calibration_mask,
    )
    training_ground_truth = evaluate_population(
        target[train_mask],
        train_mask,
    )
    probe_ground_truth = evaluate_population(
        target[probe_mask],
        probe_mask,
    )
    last_state = evaluate_population(
        stageb.last_state_baseline(condition[probe_mask]),
        probe_mask,
    )
    constant_velocity = evaluate_population(
        stageb.constant_velocity_baseline(condition[probe_mask]),
        probe_mask,
    )
    one_step_metrics = {
        str(timestep): evaluate_population(
            prediction,
            probe_mask,
        )
        for timestep, prediction in sorted(
            one_step_predictions.items()
        )
    }

    reverse_trace_metrics = {}
    for timestep in active_spec.snapshot_timesteps:
        record = raw_trace[str(timestep)]
        reverse_trace_metrics[str(timestep)] = {
            "latent": public_population_record(
                evaluate_population(record["latent"], probe_mask)
            ),
            "predicted_x0": public_population_record(
                evaluate_population(
                    record["predicted_x0"],
                    probe_mask,
                )
            ),
            "latent_sha256": sha256_array(record["latent"]),
            "predicted_x0_sha256":
                sha256_array(record["predicted_x0"]),
        }

    reverse_final_full = evaluate_population(
        final_candidates,
        probe_mask,
    )
    combined_valid_mask = np.asarray(
        reverse_final_full["combined_valid_mask"],
        dtype=np.bool_,
    )
    physical_branch = stagec.physical_valid_branch_metrics(
        pair_key=pair_key[probe_mask],
        condition_name=conditions[probe_mask],
        target=target[probe_mask],
        candidates=final_candidates,
        valid_mask=combined_valid_mask,
    )
    physical_branch_curve = physical_branch_prefix_curve(
        pair_key=pair_key[probe_mask],
        condition_name=conditions[probe_mask],
        target=target[probe_mask],
        candidates=final_candidates,
        combined_valid_mask=combined_valid_mask,
        prefix_k=active_spec.physical_branch_prefix_k,
    )
    branch_bootstrap = deterministic_eligible_branch_bootstrap(
        physical_branch,
        resamples=active_spec.bootstrap_resamples,
        seed=active_spec.bootstrap_seed,
    )

    classification = classify_recalibration(
        spec=active_spec,
        calibration_ground_truth=calibration_ground_truth,
        probe_ground_truth=probe_ground_truth,
        one_step=one_step_metrics,
        reverse_final=reverse_final_full,
        physical_branch=physical_branch,
    )

    gate_contract = gate.to_dict()
    gate_contract["contract_sha256"] = sha256_bytes(
        stable_json_bytes(gate_contract)
    )
    return {
        "phase": PHASE,
        "phase_id": PHASE_ID,
        "schema": "phase314b_r256_staged_worker_result_v1",
        "verdict": "PASS",
        "scientific_status": "BLOCKED",
        "root_cause": classification["root_cause"],
        "required_next_path": classification["required_next_path"],
        "recalibration_spec": asdict(active_spec),
        "frozen_stageb_spec": asdict(stageb_spec),
        "runtime": runtime,
        "immutable_inputs": {
            "base_commit": immutable["base_commit"],
            "base_file_sha256": immutable["base_file_sha256"],
            "stageb_immutable": immutable["stageb_immutable"],
        },
        "source_logic_audit": source_audit,
        "train_view_validation": validation,
        "split": {
            "stageb_total_groups": len(stageb_mapping),
            "stageb_training_rows": int(np.sum(train_mask)),
            "stageb_probe_rows": int(np.sum(probe_mask)),
            "stageb_training_groups":
                len(set(groups[train_mask].tolist())),
            "stageb_probe_groups":
                len(set(groups[probe_mask].tolist())),
            "gate_split": gate_split,
            "probe_group_sha256":
                sha256_strings(sorted(set(groups[probe_mask].tolist()))),
            "all_group_boundaries_intact": True,
        },
        "frozen_replay": {
            "identity": {
                **training_identity,
                "train_control_prediction_sha256": control_sha,
                "one_step_prediction_sha256": one_step_sha,
                "reverse_candidate_sha256": reverse_sha,
            },
            "matches_stageb_and_stagec": True,
            "model_changed": False,
            "scheduler_changed": False,
            "split_changed": False,
            "normalization_changed": False,
            "objective_changed": False,
            "candidate_count_changed": False,
        },
        "gate_contract": gate_contract,
        "calibration_fit": calibration_fit,
        "ground_truth_calibration": {
            "fit": public_population_record(fit_ground_truth),
            "calibration":
                public_population_record(calibration_ground_truth),
            "combined_stageb_training":
                public_population_record(training_ground_truth),
            "frozen_probe":
                public_population_record(probe_ground_truth),
        },
        "frozen_prediction_evaluation": {
            "last_state_baseline":
                public_population_record(last_state),
            "constant_velocity_baseline":
                public_population_record(constant_velocity),
            "one_step_predictions": {
                key: public_population_record(value)
                for key, value in one_step_metrics.items()
            },
            "reverse_final":
                public_population_record(reverse_final_full),
            "reverse_trace": reverse_trace_metrics,
        },
        "physical_branch_attribution": {
            "final_k8": physical_branch,
            "prefix_k_curve": physical_branch_curve,
            "eligible_bootstrap": branch_bootstrap,
            "historical_unfiltered_k8":
                stagec.EXPECTED_HISTORICAL_BRANCH_SUPPORT,
        },
        "classification": classification,
        "gate_parameters_persisted_as_json_only": True,
        "model_candidate_used_for_gate_fit": False,
        "probe_target_used_for_gate_fit": False,
        "condition_label_used_for_gate_fit": False,
        "historical_evidence_modified": False,
        "checkpoint_saved": False,
        "weights_persisted": False,
        "prediction_tensor_persisted": False,
        "candidate_tensor_persisted": False,
        "npz_saved": False,
        "formal_diffusion_training": False,
        "formal_reverse_sampling": False,
        "formal_idm_training": False,
        "action_diverse_data_collection": False,
        "candidate_execution": False,
        "phase4": False,
        "cps": False,
        "train_only_recommendation": None,
        "selected_configuration": None,
    }


def identity_projection(result: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "root_cause": result["root_cause"],
        "required_next_path": result["required_next_path"],
        "frozen_replay": result["frozen_replay"],
        "gate_contract": result["gate_contract"],
        "calibration_fit": result["calibration_fit"],
        "ground_truth_calibration":
            result["ground_truth_calibration"],
        "frozen_prediction_evaluation":
            result["frozen_prediction_evaluation"],
        "physical_branch_attribution":
            result["physical_branch_attribution"],
        "classification": result["classification"],
        "split": result["split"],
    }


def compare_worker_results(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> Dict[str, Any]:
    left_payload = stable_json_bytes(identity_projection(left))
    right_payload = stable_json_bytes(identity_projection(right))
    return {
        "exact": left_payload == right_payload,
        "left_sha256": sha256_bytes(left_payload),
        "right_sha256": sha256_bytes(right_payload),
        "identity_categories_exact": {
            key: left["frozen_replay"]["identity"][key]
            == right["frozen_replay"]["identity"][key]
            for key in sorted(left["frozen_replay"]["identity"])
        },
        "gate_contract_exact": (
            left["gate_contract"] == right["gate_contract"]
        ),
        "classification_exact": (
            left["classification"] == right["classification"]
        ),
    }
