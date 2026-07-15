"""Phase3.14b-r2.5.6 Stage D.1 asymmetric segment-gate audit.

Stage D fitted independent lower and upper directional thresholds, but its
primary segment gate used only::

    max(lower_score, upper_score) <= joint_threshold

The frozen evidence has joint_threshold == lower_threshold and an upper
threshold that is orders of magnitude smaller.  Consequently, the extreme
lower-tail calibration value also relaxes the upper-expansion side.

This stage is diagnostic only.  It:

* binds the complete Stage-D implementation and evidence to commit 0ade020;
* exactly replays the frozen Stage-B model and candidate identities;
* attributes the lower-tail threshold to exact groups, rows and segment
  positions;
* compares the historical Stage-D joint gate, the already-calibrated naive
  asymmetric gate, and a family-wise Bonferroni asymmetric gate;
* recomputes physical-valid branch support for each gate without selecting or
  modifying a gate.

No model, threshold, checkpoint, tensor, NPZ, image or video is persisted.
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
from ccda_phase3 import phase314b_r256_staged_segment_recalibration as staged

PHASE = "Phase3.14b-r2.5.6 Stage D.1"
PHASE_ID = "phase314b_r256_staged1"
BASE_EVIDENCE_COMMIT = "0ade020497eb8c40a422ddd5f2a14b7ec2ceddaf"
EXPECTED_SUBMODULE_COMMIT = "633a88752445cf5d6776ed374fdbbdb35f93050c"

STAGED_SOURCE = (
    "ccda_phase3/phase314b_r256_staged_segment_recalibration.py"
)
STAGED_TEST_GATE = "reports/phase3_14b_r256_staged_test_gate_summary.json"
STAGED_GATE_CONTRACT = (
    "reports/phase3_14b_r256_staged_segment_gate_contract.json"
)
STAGED_WORKER_EVIDENCE = (
    "reports/phase3_14b_r256_staged_worker_evidence.json"
)
STAGED_SUMMARY = "reports/phase3_14b_r256_staged_summary.json"
STAGED_REPORT = "reports/phase3_14b_r256_staged_report.md"

BASE_BOUND_FILES = (
    STAGED_SOURCE,
    STAGED_TEST_GATE,
    STAGED_GATE_CONTRACT,
    STAGED_WORKER_EVIDENCE,
    STAGED_SUMMARY,
    STAGED_REPORT,
)

EXTRA_TRAIN_KEYS = (
    "source_row_index",
    "visible_seed",
    "pair_group",
    "pair_key",
    "episode_group_key",
    "condition_name",
    "window_t",
)

CONTRACT_PROVENANCE_KEYS = (
    "source_file",
    "source_pickle_sha256",
    "episode_index",
    "row_index",
    "visible_seed",
    "pair_group",
    "pair_key",
    "episode_group_key",
    "condition_name",
    "window_t",
)


class AsymmetricGateAuditError(RuntimeError):
    """Raised when a frozen identity or directional-audit invariant fails."""


@dataclass(frozen=True)
class AuditSpec:
    """Pre-registered diagnostic thresholds.

    These values classify evidence only.  They do not modify the frozen gate.
    """

    familywise_coverage: float = 0.95
    one_sided_bonferroni_coverage: float = 0.975

    severe_collapse_ratio: float = 0.25
    moderate_collapse_ratio: float = 0.50
    mild_collapse_ratio: float = 0.75
    expansion_ratio_101: float = 1.01
    expansion_ratio_105: float = 1.05

    directional_coupling_ratio_min: float = 10.0
    scale_floor_atol: float = 1.0e-9

    probe_row_acceptance_min: float = 0.90
    probe_group_acceptance_min: float = 0.80
    probe_condition_acceptance_min: float = 0.85

    reverse_combined_row_any_min: float = 0.95
    physical_branch_eligible_row_min: float = 0.95
    physical_branch_support_min: float = 0.75

    branch_prefix_k: Tuple[int, ...] = (1, 2, 4, 8)
    bootstrap_resamples: int = 4096
    bootstrap_seed: int = 256410

    def validate(self) -> None:
        if not 0.0 < self.familywise_coverage < 1.0:
            raise ValueError("family-wise coverage must be in (0,1)")
        expected_one_sided = 1.0 - (
            1.0 - self.familywise_coverage
        ) / 2.0
        if abs(
            self.one_sided_bonferroni_coverage - expected_one_sided
        ) > 1.0e-12:
            raise ValueError(
                "one-sided Bonferroni coverage does not match alpha/2"
            )
        ratios = (
            self.severe_collapse_ratio,
            self.moderate_collapse_ratio,
            self.mild_collapse_ratio,
        )
        if not 0.0 < ratios[0] < ratios[1] < ratios[2] < 1.0:
            raise ValueError("collapse-ratio thresholds are invalid")
        if not 1.0 < self.expansion_ratio_101 < self.expansion_ratio_105:
            raise ValueError("expansion thresholds are invalid")
        if self.directional_coupling_ratio_min <= 1.0:
            raise ValueError("coupling-ratio threshold must exceed one")
        if self.scale_floor_atol <= 0.0:
            raise ValueError("scale-floor tolerance must be positive")
        if tuple(sorted(set(self.branch_prefix_k))) != self.branch_prefix_k:
            raise ValueError("branch K values must be ordered and unique")
        if self.branch_prefix_k[-1] != (
            stageb.DiagnosticSpec().reverse_candidates
        ):
            raise ValueError("branch K curve must end at frozen K")
        for value in (
            self.probe_row_acceptance_min,
            self.probe_group_acceptance_min,
            self.probe_condition_acceptance_min,
            self.reverse_combined_row_any_min,
            self.physical_branch_eligible_row_min,
            self.physical_branch_support_min,
        ):
            if not 0.0 < float(value) <= 1.0:
                raise ValueError("registered rate is outside (0,1]")
        if self.bootstrap_resamples <= 0:
            raise ValueError("bootstrap count must be positive")


@dataclass(frozen=True)
class DirectionalThresholds:
    name: str
    lower: float
    upper: float
    calibration_coverage: float
    source: str

    def validate(self) -> None:
        if not self.name:
            raise ValueError("directional gate name is empty")
        if not np.isfinite(self.lower) or self.lower < 0.0:
            raise ValueError("lower threshold is invalid")
        if not np.isfinite(self.upper) or self.upper < 0.0:
            raise ValueError("upper threshold is invalid")
        if not 0.0 < self.calibration_coverage <= 1.0:
            raise ValueError("calibration coverage is invalid")

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return {
            "name": self.name,
            "lower_threshold": float(self.lower),
            "upper_threshold": float(self.upper),
            "calibration_coverage": float(self.calibration_coverage),
            "source": self.source,
            "primary_gate": (
                "lower_score <= lower_threshold AND "
                "upper_score <= upper_threshold"
            ),
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
    return sha256_bytes(
        "\n".join(str(value) for value in values).encode("utf-8")
    )


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
            raise ValueError("non-finite scalar cannot be serialized")
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
        raise FileExistsError(f"refusing to overwrite: {target}")
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
        raise AsymmetricGateAuditError(
            f"JSON root is not an object: {path}"
        )
    return value


def _safe_stats(value: np.ndarray) -> Dict[str, float]:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "p01": 0.0,
            "p05": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "minimum": 0.0,
            "maximum": 0.0,
        }
    if not np.all(np.isfinite(array)):
        raise AsymmetricGateAuditError(
            "statistics input contains NaN or Inf"
        )
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "p01": float(np.percentile(array, 1.0)),
        "p05": float(np.percentile(array, 5.0)),
        "p50": float(np.percentile(array, 50.0)),
        "p95": float(np.percentile(array, 95.0)),
        "p99": float(np.percentile(array, 99.0)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


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
        raise AsymmetricGateAuditError(
            f"base-bound file changed after Stage D: {relative}"
        )
    return sha256_bytes(observed)


def validate_immutable_inputs(root: Path) -> Dict[str, Any]:
    repository_root = Path(root).resolve()
    bound = {
        relative: assert_base_file_bound(repository_root, relative)
        for relative in BASE_BOUND_FILES
    }
    summary = load_json(repository_root / STAGED_SUMMARY)
    if summary.get("verdict") != "PASS":
        raise AsymmetricGateAuditError("Stage-D verdict is not PASS")
    if summary.get("scientific_status") != "BLOCKED":
        raise AsymmetricGateAuditError(
            "Stage-D scientific status changed"
        )
    if summary.get("root_cause") != (
        "phase314b_r256_staged_physical_valid_branch_support_failed"
    ):
        raise AsymmetricGateAuditError("Stage-D root cause changed")
    if summary.get("required_next_path") != (
        "REPAIR_CABLE_ONLY_BRANCH_TRANSPORT_WITH_FROZEN_RECALIBRATED_GATE"
    ):
        raise AsymmetricGateAuditError("Stage-D next path changed")
    if summary.get("train_only_recommendation") is not None:
        raise AsymmetricGateAuditError(
            "Stage D selected a train-only recommendation"
        )
    if summary.get("selected_configuration") is not None:
        raise AsymmetricGateAuditError(
            "Stage D selected a configuration"
        )

    evidence = load_json(repository_root / STAGED_WORKER_EVIDENCE)
    if evidence.get("workers_exact") is not True:
        raise AsymmetricGateAuditError(
            "Stage-D workers were not exact"
        )
    result = evidence.get("worker_result")
    if not isinstance(result, dict):
        raise AsymmetricGateAuditError(
            "Stage-D worker result is missing"
        )
    identity = result.get("frozen_replay", {}).get("identity")
    if not isinstance(identity, dict):
        raise AsymmetricGateAuditError(
            "Stage-D frozen identity is missing"
        )
    if identity.get("final_model_sha256") != (
        stagec.EXPECTED_HISTORICAL_MODEL_SHA256
    ):
        raise AsymmetricGateAuditError(
            "Stage-D model identity changed"
        )
    if identity.get("reverse_candidate_sha256") != (
        stagec.EXPECTED_HISTORICAL_REVERSE_SHA256
    ):
        raise AsymmetricGateAuditError(
            "Stage-D reverse identity changed"
        )

    staged_immutable = staged.validate_immutable_inputs(repository_root)
    return {
        "base_commit": BASE_EVIDENCE_COMMIT,
        "base_file_sha256": bound,
        "staged_summary": summary,
        "staged_worker_evidence": evidence,
        "upstream_immutable": staged_immutable,
    }


def source_logic_audit(root: Path) -> Dict[str, Any]:
    text = (Path(root) / STAGED_SOURCE).read_text(encoding="utf-8")
    checks = {
        "directional_thresholds_are_fitted": (
            "lower_threshold=lower_quantile" in text
            and "upper_threshold=upper_quantile" in text
        ),
        "primary_gate_uses_only_joint_threshold": (
            'segment_pass = scores["joint"] <= float(contract.joint_threshold)'
            in text
        ),
        "element_reporting_uses_joint_threshold": (
            "element_pass = absolute_z <= float(contract.joint_threshold)"
            in text
        ),
        "contract_declares_joint_primary_gate": (
            '"primary_gate": "joint_score <= joint_threshold"' in text
        ),
        "condition_label_not_used_for_fit": (
            '"uses_condition_label": False' in text
        ),
    }
    return {
        "checks": checks,
        "all_confirmed": bool(all(checks.values())),
        "source_sha256": sha256_file(Path(root) / STAGED_SOURCE),
    }


def load_stage_d_gate(path: Path) -> Tuple[staged.SegmentGateContract, Dict[str, Any]]:
    payload = load_json(path)
    reference_payload = payload.get("reference")
    if not isinstance(reference_payload, dict):
        raise AsymmetricGateAuditError(
            "Stage-D gate reference is missing"
        )
    reference = staged.SegmentReference(
        center_log=np.asarray(
            reference_payload["center_log"],
            dtype=np.float32,
        ),
        scale_log=np.asarray(
            reference_payload["scale_log"],
            dtype=np.float32,
        ),
        mad_scale=np.asarray(
            reference_payload["mad_scale"],
            dtype=np.float32,
        ),
        iqr_scale=np.asarray(
            reference_payload["iqr_scale"],
            dtype=np.float32,
        ),
        scale_floor=float(reference_payload["scale_floor"]),
    )
    contract = staged.SegmentGateContract(
        reference=reference,
        joint_threshold=float(payload["joint_threshold"]),
        lower_threshold=float(payload["lower_threshold"]),
        upper_threshold=float(payload["upper_threshold"]),
        within_group_row_coverage=float(
            payload["within_group_row_coverage"]
        ),
        conformal_group_coverage=float(
            payload["conformal_group_coverage"]
        ),
        fit_group_count=int(payload["fit_group_count"]),
        calibration_group_count=int(
            payload["calibration_group_count"]
        ),
        fit_group_sha256=str(payload["fit_group_sha256"]),
        calibration_group_sha256=str(
            payload["calibration_group_sha256"]
        ),
    )
    contract.validate()
    internal = contract.to_dict()
    observed_internal_sha = sha256_bytes(stable_json_bytes(internal))
    if observed_internal_sha != str(payload["contract_sha256"]):
        raise AsymmetricGateAuditError(
            "Stage-D internal gate contract SHA changed"
        )
    if payload.get("primary_gate") != (
        "joint_score <= joint_threshold"
    ):
        raise AsymmetricGateAuditError(
            "Stage-D primary gate declaration changed"
        )
    return contract, payload


def _group_pass_summary(
    row_pass: np.ndarray,
    groups: Sequence[Any],
    *,
    within_group_coverage: float,
) -> Dict[str, Any]:
    passed = np.asarray(row_pass, dtype=np.bool_)
    group_values = np.asarray(groups).astype(str)
    if passed.shape != group_values.shape:
        raise ValueError("group-pass shape mismatch")
    records: MutableMapping[str, List[bool]] = {}
    for group, value in zip(group_values, passed):
        records.setdefault(group, []).append(bool(value))
    ordered = sorted(records)
    fractions = np.asarray(
        [float(np.mean(records[group])) for group in ordered],
        dtype=np.float64,
    )
    accepted = fractions >= float(within_group_coverage)
    return {
        "group_count": len(ordered),
        "accepted_group_count": int(np.sum(accepted)),
        "group_acceptance_rate": float(np.mean(accepted)),
        "within_group_row_coverage":
            float(within_group_coverage),
        "group_fraction": _safe_stats(fractions),
        "group_order_sha256": sha256_strings(ordered),
        "group_fraction_sha256": sha256_array(fractions),
    }


def stage_d_masks(
    scores: Mapping[str, np.ndarray],
    contract: staged.SegmentGateContract,
) -> Dict[str, np.ndarray]:
    return {
        "stage_d_joint": np.asarray(
            scores["joint"] <= float(contract.joint_threshold),
            dtype=np.bool_,
        ),
        "stage_d_naive_asymmetric": np.asarray(
            (scores["lower"] <= float(contract.lower_threshold))
            & (scores["upper"] <= float(contract.upper_threshold)),
            dtype=np.bool_,
        ),
    }


def bonferroni_thresholds(
    *,
    calibration_scores: Mapping[str, np.ndarray],
    calibration_groups: Sequence[Any],
    contract: staged.SegmentGateContract,
    spec: AuditSpec,
) -> Tuple[DirectionalThresholds, Dict[str, Any]]:
    if calibration_scores["lower"].shape[1] != 1:
        raise ValueError("ground-truth calibration must have K=1")
    lower_group, lower_order = staged._group_score_distribution(
        calibration_scores["lower"][:, 0],
        calibration_groups,
        within_group_coverage=contract.within_group_row_coverage,
    )
    upper_group, upper_order = staged._group_score_distribution(
        calibration_scores["upper"][:, 0],
        calibration_groups,
        within_group_coverage=contract.within_group_row_coverage,
    )
    if lower_order != upper_order:
        raise AssertionError("directional group orders differ")
    lower_quantile = staged._conformal_quantile(
        lower_group,
        spec.one_sided_bonferroni_coverage,
    )
    upper_quantile = staged._conformal_quantile(
        upper_group,
        spec.one_sided_bonferroni_coverage,
    )
    thresholds = DirectionalThresholds(
        name="bonferroni_asymmetric",
        lower=float(lower_quantile["threshold"]),
        upper=float(upper_quantile["threshold"]),
        calibration_coverage=float(spec.familywise_coverage),
        source=(
            "alpha/2 one-sided grouped split-conformal thresholds "
            "computed on the frozen Stage-D calibration groups"
        ),
    )
    thresholds.validate()
    return thresholds, {
        "one_sided_coverage":
            spec.one_sided_bonferroni_coverage,
        "lower": lower_quantile,
        "upper": upper_quantile,
        "group_count": len(lower_order),
        "group_order_sha256": sha256_strings(lower_order),
        "lower_group_score_sha256": sha256_array(lower_group),
        "upper_group_score_sha256": sha256_array(upper_group),
    }


def apply_directional_thresholds(
    scores: Mapping[str, np.ndarray],
    thresholds: DirectionalThresholds,
) -> np.ndarray:
    thresholds.validate()
    return np.asarray(
        (scores["lower"] <= float(thresholds.lower))
        & (scores["upper"] <= float(thresholds.upper)),
        dtype=np.bool_,
    )


def directional_element_pass(
    scores: Mapping[str, np.ndarray],
    *,
    lower_threshold: float,
    upper_threshold: float,
) -> np.ndarray:
    return np.asarray(
        (scores["lower_element"] <= float(lower_threshold))
        & (scores["upper_element"] <= float(upper_threshold)),
        dtype=np.bool_,
    )


def evaluate_gate_mask(
    *,
    value: np.ndarray,
    segment_mask: np.ndarray,
    element_mask: np.ndarray,
    historical_geometry: stageb.GeometryContract,
    groups: Sequence[Any],
    condition_name: Sequence[Any],
) -> Dict[str, Any]:
    candidates = staged.as_candidates(value)
    mask = np.asarray(segment_mask, dtype=np.bool_)
    elements = np.asarray(element_mask, dtype=np.bool_)
    if mask.shape != candidates.shape[:2]:
        raise ValueError("segment candidate mask shape mismatch")
    expected_element_shape = (
        candidates.shape[0],
        candidates.shape[1],
        stageb.FUTURE_STEPS,
        stageb.BEADS - 1,
    )
    if elements.shape != expected_element_shape:
        raise ValueError("segment element mask shape mismatch")

    historical = stageb.physical_validity(
        candidates,
        historical_geometry,
    )
    finite = np.asarray(historical["finite"], dtype=np.bool_)
    coordinate = np.asarray(historical["coordinate"], dtype=np.bool_)
    topology = np.asarray(historical["topology"], dtype=np.bool_)
    combined = finite & coordinate & topology & mask
    row_segment_any = np.any(mask, axis=1)
    row_combined_any = np.any(combined, axis=1)

    group_values = np.asarray(groups).astype(str)
    conditions = np.asarray(condition_name).astype(str)
    if group_values.shape != (candidates.shape[0],):
        raise ValueError("group metadata shape mismatch")
    if conditions.shape != group_values.shape:
        raise ValueError("condition metadata shape mismatch")

    segment_condition = {}
    combined_condition = {}
    for name in sorted(set(conditions.tolist())):
        selected = conditions == name
        segment_condition[name] = float(
            np.mean(row_segment_any[selected])
        )
        combined_condition[name] = float(
            np.mean(row_combined_any[selected])
        )

    return {
        "shape": list(candidates.shape),
        "candidate_sha256": sha256_array(candidates),
        "segment": {
            "candidate_rate": float(np.mean(mask)),
            "row_any_rate": float(np.mean(row_segment_any)),
            "element_rate": float(np.mean(elements)),
            "accepted_candidate_count": int(np.sum(mask)),
            "candidate_count": int(mask.size),
            "condition_row_any_rate": segment_condition,
            "group_acceptance": _group_pass_summary(
                row_segment_any,
                group_values,
                within_group_coverage=0.95,
            ),
        },
        "combined": {
            "candidate_rate": float(np.mean(combined)),
            "row_any_rate": float(np.mean(row_combined_any)),
            "accepted_candidate_count": int(np.sum(combined)),
            "candidate_count": int(combined.size),
            "condition_row_any_rate": combined_condition,
            "group_acceptance": _group_pass_summary(
                row_combined_any,
                group_values,
                within_group_coverage=0.95,
            ),
        },
        "unchanged_subgates": {
            "finite_candidate_rate": float(np.mean(finite)),
            "coordinate_candidate_rate": float(np.mean(coordinate)),
            "topology_candidate_rate": float(np.mean(topology)),
        },
        "segment_valid_mask": mask,
        "combined_valid_mask": combined,
    }


def public_evaluation(value: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: item
        for key, item in value.items()
        if key not in {"segment_valid_mask", "combined_valid_mask"}
    }


def ratio_audit(
    *,
    target: np.ndarray,
    reference: staged.SegmentReference,
    metadata: Mapping[str, np.ndarray],
    spec: AuditSpec,
    top_rows: int = 24,
) -> Dict[str, Any]:
    value = np.asarray(target, dtype=np.float32)
    lengths = stageb.segment_lengths(value).astype(np.float64)
    reference_length = np.exp(
        reference.center_log.astype(np.float64)
    )
    ratio = lengths / reference_length[None]
    if not np.all(np.isfinite(ratio)) or np.any(ratio < 0.0):
        raise AsymmetricGateAuditError(
            "relative segment ratio is invalid"
        )
    row_min = np.min(ratio, axis=(1, 2))
    row_max = np.max(ratio, axis=(1, 2))
    row_argmin = np.argmin(ratio.reshape(ratio.shape[0], -1), axis=1)
    order = np.argsort(row_min)
    records = []
    for row in order[: min(int(top_rows), row_min.size)]:
        horizon, segment_index = np.unravel_index(
            int(row_argmin[row]),
            (stageb.FUTURE_STEPS, stageb.BEADS - 1),
        )
        record = {
            "local_row": int(row),
            "minimum_ratio": float(row_min[row]),
            "maximum_ratio": float(row_max[row]),
            "horizon": int(horizon),
            "segment_index": int(segment_index),
            "segment_length": float(
                lengths[row, horizon, segment_index]
            ),
            "reference_length": float(
                reference_length[horizon, segment_index]
            ),
        }
        for key, values in metadata.items():
            item = np.asarray(values)[row]
            record[key] = jsonable(item)
        records.append(record)

    thresholds = {
        "below_severe": int(
            np.sum(ratio < float(spec.severe_collapse_ratio))
        ),
        "below_moderate": int(
            np.sum(ratio < float(spec.moderate_collapse_ratio))
        ),
        "below_mild": int(
            np.sum(ratio < float(spec.mild_collapse_ratio))
        ),
        "above_1p01": int(
            np.sum(ratio > float(spec.expansion_ratio_101))
        ),
        "above_1p05": int(
            np.sum(ratio > float(spec.expansion_ratio_105))
        ),
        "nonpositive": int(np.sum(lengths <= 0.0)),
    }
    return {
        "shape": list(ratio.shape),
        "ratio_sha256": sha256_array(ratio),
        "ratio": _safe_stats(ratio),
        "row_minimum_ratio": _safe_stats(row_min),
        "row_maximum_ratio": _safe_stats(row_max),
        "threshold_counts": thresholds,
        "row_count_with_severe_collapse": int(
            np.sum(row_min < float(spec.severe_collapse_ratio))
        ),
        "row_count_with_moderate_collapse": int(
            np.sum(row_min < float(spec.moderate_collapse_ratio))
        ),
        "lowest_rows": records,
    }


def threshold_driver_audit(
    *,
    calibration_target: np.ndarray,
    calibration_groups: Sequence[Any],
    calibration_metadata: Mapping[str, np.ndarray],
    contract: staged.SegmentGateContract,
    spec: AuditSpec,
) -> Dict[str, Any]:
    scores = staged.segment_scores(
        calibration_target,
        contract.reference,
    )
    lower_row = scores["lower"][:, 0]
    upper_row = scores["upper"][:, 0]
    group_values = np.asarray(calibration_groups).astype(str)

    grouped: MutableMapping[str, List[int]] = {}
    for index, group in enumerate(group_values):
        grouped.setdefault(group, []).append(index)
    group_records = []
    for group in sorted(grouped):
        rows = np.asarray(grouped[group], dtype=np.int64)
        lower_group = staged._higher_quantile(
            lower_row[rows],
            contract.within_group_row_coverage,
        )
        upper_group = staged._higher_quantile(
            upper_row[rows],
            contract.within_group_row_coverage,
        )
        lower_source = int(rows[np.argmax(lower_row[rows])])
        upper_source = int(rows[np.argmax(upper_row[rows])])
        group_records.append(
            {
                "group": group,
                "lower_group_score": float(lower_group),
                "upper_group_score": float(upper_group),
                "maximum_lower_row": lower_source,
                "maximum_upper_row": upper_source,
            }
        )

    lower_sorted = sorted(
        group_records,
        key=lambda record: record["lower_group_score"],
        reverse=True,
    )
    upper_sorted = sorted(
        group_records,
        key=lambda record: record["upper_group_score"],
        reverse=True,
    )

    ratio = stageb.segment_lengths(
        calibration_target
    ).astype(np.float64) / np.exp(
        contract.reference.center_log.astype(np.float64)
    )[None]

    def expand_driver(record: Mapping[str, Any], direction: str) -> Dict[str, Any]:
        row_key = (
            "maximum_lower_row"
            if direction == "lower"
            else "maximum_upper_row"
        )
        row = int(record[row_key])
        z = scores["z"][row, 0]
        if direction == "lower":
            flat = int(np.argmin(z.reshape(-1)))
        else:
            flat = int(np.argmax(z.reshape(-1)))
        horizon, segment_index = np.unravel_index(
            flat,
            (stageb.FUTURE_STEPS, stageb.BEADS - 1),
        )
        expanded = dict(record)
        expanded.update(
            {
                "direction": direction,
                "local_row": row,
                "horizon": int(horizon),
                "segment_index": int(segment_index),
                "z": float(z[horizon, segment_index]),
                "relative_length_ratio": float(
                    ratio[row, horizon, segment_index]
                ),
            }
        )
        for key, values in calibration_metadata.items():
            expanded[key] = jsonable(np.asarray(values)[row])
        return expanded

    lower_drivers = [
        expand_driver(record, "lower")
        for record in lower_sorted[:8]
    ]
    upper_drivers = [
        expand_driver(record, "upper")
        for record in upper_sorted[:8]
    ]
    lower_implied_ratio = np.exp(
        -float(contract.lower_threshold)
        * contract.reference.scale_log.astype(np.float64)
    )
    upper_implied_ratio = np.exp(
        float(contract.upper_threshold)
        * contract.reference.scale_log.astype(np.float64)
    )
    joint_upper_implied_ratio = np.exp(
        float(contract.joint_threshold)
        * contract.reference.scale_log.astype(np.float64)
    )
    floor_mask = np.isclose(
        contract.reference.scale_log,
        float(contract.reference.scale_floor),
        atol=float(spec.scale_floor_atol),
        rtol=0.0,
    )
    coupling_ratio = float(
        contract.joint_threshold
        / max(float(contract.upper_threshold), 1.0e-12)
    )
    return {
        "directional_thresholds": {
            "joint": float(contract.joint_threshold),
            "lower": float(contract.lower_threshold),
            "upper": float(contract.upper_threshold),
            "joint_to_upper_ratio": coupling_ratio,
            "directionally_coupled": bool(
                coupling_ratio >= spec.directional_coupling_ratio_min
            ),
        },
        "reference_scale": {
            "floor": float(contract.reference.scale_floor),
            "floor_saturated_positions": int(np.sum(floor_mask)),
            "total_positions": int(floor_mask.size),
            "floor_saturation_rate": float(np.mean(floor_mask)),
            "scale": _safe_stats(contract.reference.scale_log),
            "mad_scale": _safe_stats(contract.reference.mad_scale),
            "iqr_scale": _safe_stats(contract.reference.iqr_scale),
        },
        "implied_relative_length_ratios": {
            "lower_threshold": _safe_stats(lower_implied_ratio),
            "upper_threshold": _safe_stats(upper_implied_ratio),
            "joint_threshold_used_as_upper":
                _safe_stats(joint_upper_implied_ratio),
        },
        "lower_threshold_drivers": lower_drivers,
        "upper_threshold_drivers": upper_drivers,
        "lower_group_score": _safe_stats(
            np.asarray(
                [record["lower_group_score"] for record in group_records]
            )
        ),
        "upper_group_score": _safe_stats(
            np.asarray(
                [record["upper_group_score"] for record in group_records]
            )
        ),
        "lowest_driver_ratio": float(
            min(record["relative_length_ratio"] for record in lower_drivers)
        ),
        "severe_collapse_drives_lower_tail": bool(
            min(
                record["relative_length_ratio"]
                for record in lower_drivers
            )
            < float(spec.severe_collapse_ratio)
        ),
    }


def gate_difference(
    left: np.ndarray,
    right: np.ndarray,
) -> Dict[str, Any]:
    left_mask = np.asarray(left, dtype=np.bool_)
    right_mask = np.asarray(right, dtype=np.bool_)
    if left_mask.shape != right_mask.shape:
        raise ValueError("gate mask shapes differ")
    return {
        "shape": list(left_mask.shape),
        "left_only": int(np.sum(left_mask & ~right_mask)),
        "right_only": int(np.sum(right_mask & ~left_mask)),
        "both": int(np.sum(left_mask & right_mask)),
        "neither": int(np.sum(~left_mask & ~right_mask)),
        "different": int(np.sum(left_mask != right_mask)),
        "different_rate": float(np.mean(left_mask != right_mask)),
    }


def physical_branch_curve(
    *,
    pair_key: Sequence[Any],
    condition_name: Sequence[Any],
    target: np.ndarray,
    candidates: np.ndarray,
    valid_mask: np.ndarray,
    prefix_k: Sequence[int],
) -> Dict[str, Any]:
    pred = staged.as_candidates(candidates)
    valid = np.asarray(valid_mask, dtype=np.bool_)
    if valid.shape != pred.shape[:2]:
        raise ValueError("branch valid-mask shape mismatch")
    result = {}
    for value in prefix_k:
        k = int(value)
        result[str(k)] = stagec.physical_valid_branch_metrics(
            pair_key=pair_key,
            condition_name=condition_name,
            target=target,
            candidates=pred[:, :k],
            valid_mask=valid[:, :k],
        )
    return result


def deterministic_eligible_bootstrap(
    branch: Mapping[str, Any],
    *,
    resamples: int,
    seed: int,
) -> Dict[str, Any]:
    rows = branch.get("rows")
    if not isinstance(rows, list):
        raise ValueError("branch row records are missing")
    values = [
        float(record["supported"])
        for record in rows
        if record.get("eligible") is True
    ]
    if not values:
        return {
            "available": False,
            "eligible_rows": 0,
            "point_estimate": None,
            "ci95": None,
            "bootstrap_resamples": int(resamples),
            "bootstrap_seed": int(seed),
            "bootstrap_sha256": None,
        }
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.RandomState(int(seed))
    sampled = np.empty(int(resamples), dtype=np.float64)
    for index in range(int(resamples)):
        selected = rng.randint(0, array.size, size=array.size)
        sampled[index] = float(np.mean(array[selected]))
    return {
        "available": True,
        "eligible_rows": int(array.size),
        "point_estimate": float(np.mean(array)),
        "ci95": [
            float(np.percentile(sampled, 2.5)),
            float(np.percentile(sampled, 97.5)),
        ],
        "bootstrap_resamples": int(resamples),
        "bootstrap_seed": int(seed),
        "bootstrap_sha256": sha256_array(sampled),
    }


def _metadata_for_mask(
    arrays: Mapping[str, np.ndarray],
    mask: np.ndarray,
) -> Dict[str, np.ndarray]:
    return {
        key: np.asarray(arrays[key])[mask]
        for key in EXTRA_TRAIN_KEYS
        if key in arrays and key != "source_row_index"
    }


def attach_source_provenance(
    *,
    train_arrays: Mapping[str, np.ndarray],
    full_contract: Mapping[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    source_index = np.asarray(
        train_arrays["source_row_index"],
        dtype=np.int64,
    )
    result: Dict[str, np.ndarray] = {}
    for key in CONTRACT_PROVENANCE_KEYS:
        if key in full_contract:
            result[key] = np.asarray(full_contract[key])[source_index]
    result["source_row_index"] = source_index
    return result


def classify_audit(
    *,
    spec: AuditSpec,
    threshold_driver: Mapping[str, Any],
    calibration_results: Mapping[str, Mapping[str, Any]],
    probe_results: Mapping[str, Mapping[str, Any]],
    reverse_results: Mapping[str, Mapping[str, Any]],
    branch_results: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    coupling = bool(
        threshold_driver["directional_thresholds"][
            "directionally_coupled"
        ]
    )
    collapse = bool(
        threshold_driver["severe_collapse_drives_lower_tail"]
    )

    bonf_cal = calibration_results["bonferroni_asymmetric"]
    bonf_probe = probe_results["bonferroni_asymmetric"]
    bonf_reverse = reverse_results["bonferroni_asymmetric"]
    bonf_branch = branch_results["bonferroni_asymmetric"]["k8"]

    calibration_group_rate = float(
        bonf_cal["segment"]["group_acceptance"][
            "group_acceptance_rate"
        ]
    )
    probe_row_rate = float(bonf_probe["segment"]["row_any_rate"])
    probe_group_rate = float(
        bonf_probe["segment"]["group_acceptance"][
            "group_acceptance_rate"
        ]
    )
    probe_condition_rate = float(
        min(bonf_probe["segment"]["condition_row_any_rate"].values())
    )
    reverse_row_any = float(
        bonf_reverse["combined"]["row_any_rate"]
    )
    eligible_rate = float(bonf_branch["eligible_row_rate"])
    support = bonf_branch["support_rate_among_eligible"]
    support_value = float(support) if support is not None else 0.0

    if coupling and collapse:
        root = (
            "phase314b_r256_staged1_lower_tail_ground_truth_"
            "collapse_drives_directional_gate_coupling"
        )
        next_path = (
            "AUDIT_CABLE_XY_SEGMENT_COLLAPSE_PROVENANCE_"
            "BEFORE_GATE_OR_BRANCH_REPAIR"
        )
        locus = "ground_truth_segment_collapse_provenance"
    elif coupling and calibration_group_rate < spec.familywise_coverage:
        root = (
            "phase314b_r256_staged1_bonferroni_asymmetric_"
            "gate_internal_calibration_failed"
        )
        next_path = (
            "REDESIGN_DIRECTIONAL_SEGMENT_GATE_WITH_"
            "TRAIN_ONLY_GROUPED_CALIBRATION"
        )
        locus = "directional_gate_calibration"
    elif (
        coupling
        and (
            probe_row_rate < spec.probe_row_acceptance_min
            or probe_group_rate < spec.probe_group_acceptance_min
            or probe_condition_rate <
                spec.probe_condition_acceptance_min
        )
    ):
        root = (
            "phase314b_r256_staged1_bonferroni_asymmetric_"
            "gate_probe_generalization_failed"
        )
        next_path = (
            "ATTRIBUTE_DIRECTIONAL_GATE_PROBE_SHIFT_"
            "WITHOUT_USING_PROBE_FOR_CALIBRATION"
        )
        locus = "directional_gate_generalization"
    elif reverse_row_any < spec.reverse_combined_row_any_min:
        root = (
            "phase314b_r256_staged1_asymmetric_gate_"
            "physical_candidate_coverage_failed"
        )
        next_path = (
            "REPAIR_PHYSICAL_CANDIDATE_COVERAGE_WITH_"
            "FROZEN_DIRECTIONAL_GATE"
        )
        locus = "physical_candidate_coverage"
    elif eligible_rate < spec.physical_branch_eligible_row_min:
        root = (
            "phase314b_r256_staged1_asymmetric_gate_"
            "branch_eligibility_failed"
        )
        next_path = (
            "REPAIR_PHYSICAL_CANDIDATE_ELIGIBILITY_"
            "BEFORE_BRANCH_TRANSPORT"
        )
        locus = "branch_eligibility"
    elif support_value < spec.physical_branch_support_min:
        root = (
            "phase314b_r256_staged1_asymmetric_gate_"
            "physical_valid_branch_support_failed"
        )
        next_path = (
            "REPAIR_CABLE_ONLY_BRANCH_TRANSPORT_WITH_"
            "FROZEN_ASYMMETRIC_GATE"
        )
        locus = "branch_transport"
    else:
        root = (
            "phase314b_r256_staged1_asymmetric_gate_"
            "and_physical_branch_supported"
        )
        next_path = (
            "FREEZE_DIRECTIONAL_GATE_FOR_FORMAL_PILOT_"
            "AND_COLLECT_ACTION_DIVERSE_IDM_DATA"
        )
        locus = "none"

    return {
        "root_cause": root,
        "required_next_path": next_path,
        "primary_failure_locus": locus,
        "directional_coupling_confirmed": coupling,
        "severe_collapse_drives_lower_tail": collapse,
        "bonferroni_calibration_group_acceptance":
            calibration_group_rate,
        "bonferroni_probe_row_acceptance": probe_row_rate,
        "bonferroni_probe_group_acceptance": probe_group_rate,
        "bonferroni_probe_min_condition_acceptance":
            probe_condition_rate,
        "bonferroni_reverse_combined_row_any": reverse_row_any,
        "bonferroni_branch_eligible_row_rate": eligible_rate,
        "bonferroni_branch_support_among_eligible": support,
    }


def run_audit(
    *,
    root: Path,
    spec: Optional[AuditSpec] = None,
) -> Dict[str, Any]:
    active_spec = AuditSpec() if spec is None else spec
    active_spec.validate()
    repository_root = Path(root).resolve()

    immutable = validate_immutable_inputs(repository_root)
    source_audit = source_logic_audit(repository_root)
    if not source_audit["all_confirmed"]:
        raise AsymmetricGateAuditError(
            "Stage-D source assumptions changed"
        )
    gate, gate_file = load_stage_d_gate(
        repository_root / STAGED_GATE_CONTRACT
    )

    required_train_keys = tuple(
        sorted(set(stageb.REQUIRED_TRAIN_KEYS + EXTRA_TRAIN_KEYS))
    )
    arrays = stageb.load_npz_strict(
        repository_root / stageb.STAGEA_TRAIN_VIEW,
        required_keys=required_train_keys,
    )
    validation = stageb.validate_train_view(arrays)
    full_contract = stageb.load_npz_strict(
        repository_root / stageb.STAGEA_CONTRACT,
        required_keys=CONTRACT_PROVENANCE_KEYS,
    )
    provenance = attach_source_provenance(
        train_arrays=arrays,
        full_contract=full_contract,
    )

    stageb_spec = stageb.DiagnosticSpec()
    stageb_spec.validate()
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
        staged.deterministic_gate_split(
            groups,
            train_mask,
            spec=staged.RecalibrationSpec(),
        )
    )
    if gate_split["fit_group_sha256"] != gate.fit_group_sha256:
        raise AsymmetricGateAuditError(
            "Stage-D fit-group identity changed"
        )
    if (
        gate_split["calibration_group_sha256"]
        != gate.calibration_group_sha256
    ):
        raise AsymmetricGateAuditError(
            "Stage-D calibration-group identity changed"
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
    expected_identity = immutable["staged_worker_evidence"][
        "worker_result"
    ]["frozen_replay"]["identity"]
    replay_identity = {
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
    for key, observed in replay_identity.items():
        if observed != expected_identity[key]:
            raise AsymmetricGateAuditError(
                f"frozen model replay differs: {key}"
            )

    train_control = stagec.reproduce_train_control_prediction(
        model=model,
        train_condition=condition[train_mask],
        train_target=target[train_mask],
        spec=stageb_spec,
        condition_standardizer=condition_standardizer,
        target_standardizer=target_standardizer,
    )
    train_control_sha = sha256_array(train_control)
    if train_control_sha != expected_identity[
        "train_control_prediction_sha256"
    ]:
        raise AsymmetricGateAuditError(
            "frozen train-control prediction differs"
        )

    one_step, one_step_sha = stagec.one_step_predictions(
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
        raise AsymmetricGateAuditError(
            "frozen one-step prediction differs"
        )

    candidates, _ = staged.reverse_sample_with_raw_trace(
        model=model,
        condition=condition[probe_mask],
        spec=stageb_spec,
        condition_standardizer=condition_standardizer,
        target_standardizer=target_standardizer,
        noise_seed=stageb_spec.seed + 3001,
        snapshot_timesteps=(0,),
    )
    reverse_sha = sha256_array(candidates)
    if reverse_sha != expected_identity["reverse_candidate_sha256"]:
        raise AsymmetricGateAuditError(
            "frozen reverse candidates differ"
        )

    calibration_scores = staged.segment_scores(
        target[calibration_mask],
        gate.reference,
    )
    bonferroni, bonferroni_fit = bonferroni_thresholds(
        calibration_scores=calibration_scores,
        calibration_groups=groups[calibration_mask],
        contract=gate,
        spec=active_spec,
    )
    naive = DirectionalThresholds(
        name="stage_d_naive_asymmetric",
        lower=float(gate.lower_threshold),
        upper=float(gate.upper_threshold),
        calibration_coverage=float(gate.conformal_group_coverage),
        source=(
            "the independent Stage-D lower/upper 95% thresholds "
            "that were fitted but not used by its primary gate"
        ),
    )

    threshold_driver = threshold_driver_audit(
        calibration_target=target[calibration_mask],
        calibration_groups=groups[calibration_mask],
        calibration_metadata={
            **_metadata_for_mask(arrays, calibration_mask),
            **{
                key: np.asarray(value)[calibration_mask]
                for key, value in provenance.items()
            },
        },
        contract=gate,
        spec=active_spec,
    )

    population_masks = {
        "fit": fit_mask,
        "calibration": calibration_mask,
        "training": train_mask,
        "probe": probe_mask,
    }
    ratio_audits = {}
    for name, mask in population_masks.items():
        ratio_audits[name] = ratio_audit(
            target=target[mask],
            reference=gate.reference,
            metadata={
                **_metadata_for_mask(arrays, mask),
                **{
                    key: np.asarray(value)[mask]
                    for key, value in provenance.items()
                },
            },
            spec=active_spec,
        )

    gate_definitions = {
        "stage_d_joint": {
            "type": "joint",
            "joint_threshold": float(gate.joint_threshold),
            "primary_gate": "joint_score <= joint_threshold",
        },
        "stage_d_naive_asymmetric": naive.to_dict(),
        "bonferroni_asymmetric": bonferroni.to_dict(),
    }

    def evaluate_value(
        value: np.ndarray,
        selected_mask: np.ndarray,
    ) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
        scores = staged.segment_scores(value, gate.reference)
        masks = stage_d_masks(scores, gate)
        masks["bonferroni_asymmetric"] = (
            apply_directional_thresholds(scores, bonferroni)
        )
        element_masks = {
            "stage_d_joint": (
                np.abs(scores["z"])
                <= float(gate.joint_threshold)
            ),
            "stage_d_naive_asymmetric":
                directional_element_pass(
                    scores,
                    lower_threshold=naive.lower,
                    upper_threshold=naive.upper,
                ),
            "bonferroni_asymmetric":
                directional_element_pass(
                    scores,
                    lower_threshold=bonferroni.lower,
                    upper_threshold=bonferroni.upper,
                ),
        }
        evaluations = {}
        for key in sorted(masks):
            evaluations[key] = evaluate_gate_mask(
                value=value,
                segment_mask=masks[key],
                element_mask=element_masks[key],
                historical_geometry=historical_geometry,
                groups=groups[selected_mask],
                condition_name=conditions[selected_mask],
            )
        return evaluations, masks

    fit_eval, fit_masks = evaluate_value(target[fit_mask], fit_mask)
    calibration_eval, calibration_masks = evaluate_value(
        target[calibration_mask],
        calibration_mask,
    )
    training_eval, training_masks = evaluate_value(
        target[train_mask],
        train_mask,
    )
    probe_eval, probe_masks = evaluate_value(
        target[probe_mask],
        probe_mask,
    )
    one_step_eval = {}
    for timestep, prediction in sorted(one_step.items()):
        one_step_eval[str(timestep)], _ = evaluate_value(
            prediction,
            probe_mask,
        )
    reverse_eval, reverse_masks = evaluate_value(
        candidates,
        probe_mask,
    )

    branch_results = {}
    for gate_name in sorted(reverse_masks):
        combined = np.asarray(
            reverse_eval[gate_name]["combined_valid_mask"],
            dtype=np.bool_,
        )
        k_curve = physical_branch_curve(
            pair_key=pair_key[probe_mask],
            condition_name=conditions[probe_mask],
            target=target[probe_mask],
            candidates=candidates,
            valid_mask=combined,
            prefix_k=active_spec.branch_prefix_k,
        )
        k8 = k_curve[str(active_spec.branch_prefix_k[-1])]
        branch_results[gate_name] = {
            "k_curve": k_curve,
            "k8": k8,
            "eligible_bootstrap": deterministic_eligible_bootstrap(
                k8,
                resamples=active_spec.bootstrap_resamples,
                seed=active_spec.bootstrap_seed,
            ),
        }

    differences = {
        "reverse_joint_vs_naive": gate_difference(
            reverse_masks["stage_d_joint"],
            reverse_masks["stage_d_naive_asymmetric"],
        ),
        "reverse_joint_vs_bonferroni": gate_difference(
            reverse_masks["stage_d_joint"],
            reverse_masks["bonferroni_asymmetric"],
        ),
        "reverse_naive_vs_bonferroni": gate_difference(
            reverse_masks["stage_d_naive_asymmetric"],
            reverse_masks["bonferroni_asymmetric"],
        ),
        "probe_gt_joint_vs_naive": gate_difference(
            probe_masks["stage_d_joint"],
            probe_masks["stage_d_naive_asymmetric"],
        ),
        "probe_gt_joint_vs_bonferroni": gate_difference(
            probe_masks["stage_d_joint"],
            probe_masks["bonferroni_asymmetric"],
        ),
    }

    public_fit = {
        key: public_evaluation(value)
        for key, value in fit_eval.items()
    }
    public_calibration = {
        key: public_evaluation(value)
        for key, value in calibration_eval.items()
    }
    public_training = {
        key: public_evaluation(value)
        for key, value in training_eval.items()
    }
    public_probe = {
        key: public_evaluation(value)
        for key, value in probe_eval.items()
    }
    public_one_step = {
        timestep: {
            key: public_evaluation(value)
            for key, value in evaluation.items()
        }
        for timestep, evaluation in one_step_eval.items()
    }
    public_reverse = {
        key: public_evaluation(value)
        for key, value in reverse_eval.items()
    }

    classification = classify_audit(
        spec=active_spec,
        threshold_driver=threshold_driver,
        calibration_results=public_calibration,
        probe_results=public_probe,
        reverse_results=public_reverse,
        branch_results=branch_results,
    )

    audit_contract = {
        "schema": "phase314b_r256_staged1_gate_audit_contract_v1",
        "frozen_stage_d_contract_sha256":
            gate_file["contract_sha256"],
        "stage_d_gate": {
            "joint_threshold": float(gate.joint_threshold),
            "lower_threshold": float(gate.lower_threshold),
            "upper_threshold": float(gate.upper_threshold),
            "primary_gate": gate_file["primary_gate"],
        },
        "candidate_gate_definitions": gate_definitions,
        "bonferroni_fit": bonferroni_fit,
        "no_gate_selected": True,
        "no_threshold_modified": True,
    }
    audit_contract["contract_sha256"] = sha256_bytes(
        stable_json_bytes(audit_contract)
    )

    return {
        "phase": PHASE,
        "phase_id": PHASE_ID,
        "schema": "phase314b_r256_staged1_worker_result_v1",
        "verdict": "PASS",
        "scientific_status": "BLOCKED",
        "root_cause": classification["root_cause"],
        "required_next_path": classification["required_next_path"],
        "audit_spec": asdict(active_spec),
        "runtime": runtime,
        "immutable_inputs": {
            "base_commit": immutable["base_commit"],
            "base_file_sha256": immutable["base_file_sha256"],
        },
        "source_logic_audit": source_audit,
        "train_view_validation": validation,
        "split": {
            "stageb_group_count": len(stageb_mapping),
            "training_rows": int(np.sum(train_mask)),
            "probe_rows": int(np.sum(probe_mask)),
            "fit_rows": int(np.sum(fit_mask)),
            "calibration_rows": int(np.sum(calibration_mask)),
            "fit_groups": gate_split["fit_group_count"],
            "calibration_groups":
                gate_split["calibration_group_count"],
            "group_boundaries_intact": True,
        },
        "frozen_replay": {
            "identity": {
                **replay_identity,
                "train_control_prediction_sha256":
                    train_control_sha,
                "one_step_prediction_sha256": one_step_sha,
                "reverse_candidate_sha256": reverse_sha,
            },
            "matches_stage_b_c_d": True,
            "model_changed": False,
            "scheduler_changed": False,
            "split_changed": False,
            "normalization_changed": False,
            "objective_changed": False,
            "candidate_count_changed": False,
        },
        "audit_contract": audit_contract,
        "threshold_driver_attribution": threshold_driver,
        "ground_truth_ratio_audit": ratio_audits,
        "gate_evaluation": {
            "fit_ground_truth": public_fit,
            "calibration_ground_truth": public_calibration,
            "combined_training_ground_truth": public_training,
            "probe_ground_truth": public_probe,
            "one_step_predictions": public_one_step,
            "reverse_candidates": public_reverse,
            "mask_differences": differences,
        },
        "physical_branch_attribution": branch_results,
        "classification": classification,
        "deformable_ravens_modified": False,
        "gate_selected": False,
        "threshold_changed": False,
        "model_or_branch_repaired": False,
        "formal_pilot_run": False,
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
        "audit_contract": result["audit_contract"],
        "threshold_driver_attribution":
            result["threshold_driver_attribution"],
        "ground_truth_ratio_audit":
            result["ground_truth_ratio_audit"],
        "gate_evaluation": result["gate_evaluation"],
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
        "frozen_identity_exact": {
            key: left["frozen_replay"]["identity"][key]
            == right["frozen_replay"]["identity"][key]
            for key in sorted(left["frozen_replay"]["identity"])
        },
        "audit_contract_exact": (
            left["audit_contract"] == right["audit_contract"]
        ),
        "classification_exact": (
            left["classification"] == right["classification"]
        ),
    }
