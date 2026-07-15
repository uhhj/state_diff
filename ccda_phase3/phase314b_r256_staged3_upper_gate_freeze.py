"""Phase3.14b-r2.5.6 Stage D.3 one-sided XY upper-gate freeze.

Stage D.2 established, with exact raw-pickle provenance, that every severe
ground-truth XY segment contraction is a projection artifact: ordered adjacent
beads retain normal XYZ distance while their XY projection can approach zero.
Consequently, a lower bound on ordered XY segment length is not a valid physical
constraint for this representation.

This stage therefore freezes exactly one segment-validity direction:

    upper_score <= grouped_split_conformal_upper_threshold

The threshold is recomputed exclusively from the frozen Stage-D calibration
groups at 95% grouped split-conformal coverage.  It must be bit-exact with the
independently fitted Stage-D upper threshold.  Lower scores remain diagnostic
only and never enter candidate validity.

The Stage-B model, scheduler, normalization, grouped train/probe split, seed,
objective, training path, one-step noises, reverse noises and K=8 candidate
ordering are replayed exactly.  No checkpoint, prediction tensor, candidate
tensor, NPZ, cache, image or video is persisted.
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
from ccda_phase3 import phase314b_r256_staged1_asymmetric_gate_audit as staged1
from ccda_phase3 import phase314b_r256_staged2_collapse_provenance as staged2

PHASE = "Phase3.14b-r2.5.6 Stage D.3"
PHASE_ID = "phase314b_r256_staged3"
BASE_EVIDENCE_COMMIT = "306575ee46edd9d163550ad310544f10ed1258fe"
EXPECTED_SUBMODULE_COMMIT = "633a88752445cf5d6776ed374fdbbdb35f93050c"

STAGE_D_SOURCE = (
    "ccda_phase3/phase314b_r256_staged_segment_recalibration.py"
)
STAGE_D1_SOURCE = (
    "ccda_phase3/phase314b_r256_staged1_asymmetric_gate_audit.py"
)
STAGE_D2_SOURCE = (
    "ccda_phase3/phase314b_r256_staged2_collapse_provenance.py"
)
STAGE_D_GATE = "reports/phase3_14b_r256_staged_segment_gate_contract.json"
STAGE_D2_TEST_GATE = (
    "reports/phase3_14b_r256_staged2_test_gate_summary.json"
)
STAGE_D2_PROVENANCE = (
    "reports/phase3_14b_r256_staged2_source_provenance.json"
)
STAGE_D2_WORKER = (
    "reports/phase3_14b_r256_staged2_worker_evidence.json"
)
STAGE_D2_SUMMARY = "reports/phase3_14b_r256_staged2_summary.json"
STAGE_D2_REPORT = "reports/phase3_14b_r256_staged2_report.md"

BASE_BOUND_FILES = (
    STAGE_D_SOURCE,
    STAGE_D1_SOURCE,
    STAGE_D2_SOURCE,
    STAGE_D_GATE,
    STAGE_D2_TEST_GATE,
    STAGE_D2_PROVENANCE,
    STAGE_D2_WORKER,
    STAGE_D2_SUMMARY,
    STAGE_D2_REPORT,
)

SNAPSHOT_TIMESTEPS = (99, 75, 50, 25, 10, 0)


class UpperGateFreezeError(RuntimeError):
    """Raised when an immutable identity or gate-freeze invariant fails."""


@dataclass(frozen=True)
class UpperGateSpec:
    """Pre-registered one-sided contract and diagnostic decision gates."""

    within_group_row_coverage: float = 0.95
    conformal_group_coverage: float = 0.95

    probe_row_acceptance_min: float = 0.90
    probe_group_acceptance_min: float = 0.80
    probe_condition_acceptance_min: float = 0.85

    one_step_t10_upper_acceptance_min: float = 0.75
    reverse_upper_row_any_min: float = 0.95
    reverse_combined_row_any_min: float = 0.95

    physical_branch_eligible_row_min: float = 0.95
    physical_branch_support_min: float = 0.75

    branch_prefix_k: Tuple[int, ...] = (1, 2, 4, 8)
    bootstrap_resamples: int = 4096
    bootstrap_seed: int = 256430
    snapshot_timesteps: Tuple[int, ...] = SNAPSHOT_TIMESTEPS

    def validate(self) -> None:
        for value in (
            self.within_group_row_coverage,
            self.conformal_group_coverage,
            self.probe_row_acceptance_min,
            self.probe_group_acceptance_min,
            self.probe_condition_acceptance_min,
            self.one_step_t10_upper_acceptance_min,
            self.reverse_upper_row_any_min,
            self.reverse_combined_row_any_min,
            self.physical_branch_eligible_row_min,
            self.physical_branch_support_min,
        ):
            if not 0.0 < float(value) <= 1.0:
                raise ValueError("registered coverage/rate is outside (0,1]")
        if tuple(sorted(set(self.branch_prefix_k))) != self.branch_prefix_k:
            raise ValueError("branch K values must be ordered and unique")
        if self.branch_prefix_k[-1] != (
            stageb.DiagnosticSpec().reverse_candidates
        ):
            raise ValueError("branch K curve must end at frozen K")
        if tuple(sorted(set(self.snapshot_timesteps), reverse=True)) != (
            self.snapshot_timesteps
        ):
            raise ValueError("snapshot timesteps must be unique and descending")
        if self.snapshot_timesteps[0] != 99 or self.snapshot_timesteps[-1] != 0:
            raise ValueError("snapshot trace must cover t=99 through t=0")
        if self.bootstrap_resamples <= 0:
            raise ValueError("bootstrap count must be positive")


@dataclass(frozen=True)
class OneSidedUpperContract:
    """Immutable upper-only ordered-XY segment validity contract."""

    reference: staged.SegmentReference
    upper_threshold: float
    within_group_row_coverage: float
    conformal_group_coverage: float
    fit_group_count: int
    calibration_group_count: int
    fit_group_sha256: str
    calibration_group_sha256: str
    calibration_group_score_sha256: str
    stage_d_internal_contract_sha256: str
    stage_d2_raw_xyz_sha256: str
    stage_d2_projection_artifact_fraction: float

    def validate(self) -> None:
        self.reference.validate()
        if not np.isfinite(self.upper_threshold) or self.upper_threshold < 0.0:
            raise ValueError("upper threshold is invalid")
        if not 0.0 < self.within_group_row_coverage <= 1.0:
            raise ValueError("within-group coverage is invalid")
        if not 0.0 < self.conformal_group_coverage <= 1.0:
            raise ValueError("conformal coverage is invalid")
        if self.fit_group_count <= 0 or self.calibration_group_count <= 0:
            raise ValueError("gate split is empty")
        if self.stage_d2_projection_artifact_fraction != 1.0:
            raise ValueError(
                "one-sided freeze requires complete projection attribution"
            )
        for value in (
            self.fit_group_sha256,
            self.calibration_group_sha256,
            self.calibration_group_score_sha256,
            self.stage_d_internal_contract_sha256,
            self.stage_d2_raw_xyz_sha256,
        ):
            if len(str(value)) != 64:
                raise ValueError("contract SHA256 field is malformed")

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return {
            "schema": "phase314b_r256_staged3_one_sided_upper_contract_v1",
            "representation": "ordered cable XY, [4,48]",
            "score": (
                "maximum positive robust log-segment z-score over "
                "4 horizons x 23 ordered segments"
            ),
            "primary_gate": "upper_score <= upper_threshold",
            "lower_score_role": "diagnostic_only",
            "candidate_positions": int(
                stageb.FUTURE_STEPS * (stageb.BEADS - 1)
            ),
            "upper_threshold": float(self.upper_threshold),
            "within_group_row_coverage":
                float(self.within_group_row_coverage),
            "conformal_group_coverage":
                float(self.conformal_group_coverage),
            "fit_group_count": int(self.fit_group_count),
            "calibration_group_count": int(self.calibration_group_count),
            "fit_group_sha256": self.fit_group_sha256,
            "calibration_group_sha256": self.calibration_group_sha256,
            "calibration_group_score_sha256":
                self.calibration_group_score_sha256,
            "stage_d_internal_contract_sha256":
                self.stage_d_internal_contract_sha256,
            "stage_d2_raw_xyz_sha256":
                self.stage_d2_raw_xyz_sha256,
            "stage_d2_projection_artifact_fraction":
                float(self.stage_d2_projection_artifact_fraction),
            "reference": self.reference.to_dict(),
            "uses_lower_score_for_validity": False,
            "uses_probe_target_for_fit": False,
            "uses_model_candidate_for_fit": False,
            "uses_condition_label_for_fit": False,
            "historical_gate_modified": False,
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
        return {
            str(key): jsonable(item)
            for key, item in value.items()
        }
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


def atomic_write_once(
    path: Path,
    payload: bytes,
    *,
    mode: int = 0o644,
) -> None:
    target = Path(path)
    if target.exists():
        raise FileExistsError(
            f"refusing to overwrite write-once output: {target}"
        )
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
        raise UpperGateFreezeError(
            f"JSON root is not an object: {path}"
        )
    return value


def _safe_stats(value: np.ndarray) -> Dict[str, float]:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return {
            "count": 0,
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
        raise UpperGateFreezeError(
            "statistics input contains NaN or Inf"
        )
    return {
        "count": int(array.size),
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


def float64_bits(value: float) -> int:
    return int(
        np.asarray(float(value), dtype=np.float64).view(np.uint64)
    )


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
        raise UpperGateFreezeError(
            f"base-bound file changed after Stage D.2: {relative}"
        )
    return sha256_bytes(observed)


def validate_immutable_inputs(root: Path) -> Dict[str, Any]:
    repository_root = Path(root).resolve()
    bound = {
        relative: assert_base_file_bound(repository_root, relative)
        for relative in BASE_BOUND_FILES
    }
    summary = load_json(repository_root / STAGE_D2_SUMMARY)
    if summary.get("verdict") != "PASS":
        raise UpperGateFreezeError(
            "Stage-D.2 verdict is not PASS"
        )
    if summary.get("scientific_status") != "BLOCKED":
        raise UpperGateFreezeError(
            "Stage-D.2 scientific status changed"
        )
    if summary.get("root_cause") != (
        "phase314b_r256_staged2_xy_projection_collapse_confirmed"
    ):
        raise UpperGateFreezeError(
            "Stage-D.2 root cause changed"
        )
    if summary.get("required_next_path") != (
        "FREEZE_ONE_SIDED_XY_UPPER_SEGMENT_CONTRACT_"
        "AND_REAUDIT_FROZEN_REVERSE"
    ):
        raise UpperGateFreezeError(
            "Stage-D.2 next path changed"
        )
    if summary.get("gate_selected") is not False:
        raise UpperGateFreezeError(
            "Stage-D.2 unexpectedly selected a gate"
        )
    if summary.get("threshold_changed") is not False:
        raise UpperGateFreezeError(
            "Stage-D.2 unexpectedly changed a threshold"
        )
    if summary.get("train_only_recommendation") is not None:
        raise UpperGateFreezeError(
            "Stage-D.2 selected a train-only recommendation"
        )
    if summary.get("selected_configuration") is not None:
        raise UpperGateFreezeError(
            "Stage-D.2 selected a model configuration"
        )

    provenance = load_json(repository_root / STAGE_D2_PROVENANCE)
    classification = provenance.get("classification")
    if not isinstance(classification, dict):
        raise UpperGateFreezeError(
            "Stage-D.2 classification is missing"
        )
    if classification.get("primary_failure_locus") != (
        "xy_projection_contract"
    ):
        raise UpperGateFreezeError(
            "Stage-D.2 failure locus changed"
        )
    calibration = provenance.get(
        "population_geometry_audit", {}
    ).get("gate_calibration")
    if not isinstance(calibration, dict):
        raise UpperGateFreezeError(
            "Stage-D.2 calibration provenance is missing"
        )
    severe_fractions = calibration.get(
        "severe_event_fractions"
    )
    if not isinstance(severe_fractions, dict):
        raise UpperGateFreezeError(
            "Stage-D.2 severe-event fractions are missing"
        )
    if float(severe_fractions.get("projection_artifact", -1.0)) != 1.0:
        raise UpperGateFreezeError(
            "Stage-D.2 projection-artifact fraction changed"
        )
    if float(severe_fractions.get("true_xyz_collapse", -1.0)) != 0.0:
        raise UpperGateFreezeError(
            "Stage-D.2 true-XYZ fraction changed"
        )
    if float(severe_fractions.get("partial_xyz_collapse", -1.0)) != 0.0:
        raise UpperGateFreezeError(
            "Stage-D.2 partial-XYZ fraction changed"
        )

    worker = load_json(repository_root / STAGE_D2_WORKER)
    if worker.get("workers_exact") is not True:
        raise UpperGateFreezeError(
            "Stage-D.2 workers were not exact"
        )
    worker_result = worker.get("worker_result")
    if not isinstance(worker_result, dict):
        raise UpperGateFreezeError(
            "Stage-D.2 worker result is missing"
        )
    reconstruction = worker_result.get(
        "raw_reconstruction_audit"
    )
    if not isinstance(reconstruction, dict):
        raise UpperGateFreezeError(
            "Stage-D.2 reconstruction is missing"
        )
    if float(
        reconstruction.get("raw_xy_float32_exact_rate", -1.0)
    ) != 1.0:
        raise UpperGateFreezeError(
            "Stage-D.2 raw-XY/cache exactness changed"
        )
    if reconstruction.get(
        "bead_id_order_stable_across_target_horizons"
    ) is not True:
        raise UpperGateFreezeError(
            "Stage-D.2 bead-order status changed"
        )

    upstream = staged2.validate_immutable_inputs(repository_root)
    return {
        "base_commit": BASE_EVIDENCE_COMMIT,
        "base_file_sha256": bound,
        "stage_d2_summary": summary,
        "stage_d2_provenance": provenance,
        "stage_d2_worker": worker,
        "stage_d2_raw_xyz_sha256":
            reconstruction["raw_target_xyz_sha256"],
        "stage_d2_projection_artifact_fraction":
            severe_fractions["projection_artifact"],
        "upstream_immutable": upstream,
    }


def source_logic_audit(root: Path) -> Dict[str, Any]:
    repository_root = Path(root).resolve()
    stage_d_text = (
        repository_root / STAGE_D_SOURCE
    ).read_text(encoding="utf-8")
    stage_d1_text = (
        repository_root / STAGE_D1_SOURCE
    ).read_text(encoding="utf-8")
    stage_d2_text = (
        repository_root / STAGE_D2_SOURCE
    ).read_text(encoding="utf-8")
    checks = {
        "stage_d_fits_upper_group_scores": (
            'group_upper, ordered_upper = _group_score_distribution('
            in stage_d_text
        ),
        "stage_d_fits_upper_conformal_quantile": (
            'upper_quantile = _conformal_quantile(' in stage_d_text
        ),
        "stage_d_stores_upper_threshold": (
            'upper_threshold=upper_quantile["threshold"]'
            in stage_d_text
        ),
        "stage_d_primary_gate_did_not_use_upper_threshold": (
            'segment_pass = scores["joint"] <= float(contract.joint_threshold)'
            in stage_d_text
        ),
        "stage_d1_confirmed_directional_gate_difference": (
            "stage_d_naive_asymmetric" in stage_d1_text
            and "bonferroni_asymmetric" in stage_d1_text
        ),
        "stage_d2_classifies_projection_artifact": (
            '"xy_projection_artifact"' in stage_d2_text
            and "xyz_ratio >= spec.xyz_normal_ratio"
            in stage_d2_text
        ),
        "stage_d2_requires_raw_xy_cache_exact": (
            "raw_xy_float32_exact_required" in stage_d2_text
            and "np.array_equal(raw_xy32, cached_xy32)"
            in stage_d2_text
        ),
    }
    return {
        "checks": checks,
        "all_confirmed": bool(all(checks.values())),
        "source_sha256": {
            STAGE_D_SOURCE:
                sha256_file(repository_root / STAGE_D_SOURCE),
            STAGE_D1_SOURCE:
                sha256_file(repository_root / STAGE_D1_SOURCE),
            STAGE_D2_SOURCE:
                sha256_file(repository_root / STAGE_D2_SOURCE),
        },
        "contract_change": (
            "replace directionally coupled joint segment validity "
            "with upper-only validity; retain lower score for diagnostics"
        ),
    }


def recompute_upper_contract(
    *,
    target: np.ndarray,
    groups: Sequence[Any],
    fit_mask: np.ndarray,
    calibration_mask: np.ndarray,
    gate_split: Mapping[str, Any],
    stage_d_contract: staged.SegmentGateContract,
    stage_d_contract_payload: Mapping[str, Any],
    immutable: Mapping[str, Any],
    spec: UpperGateSpec,
) -> Tuple[OneSidedUpperContract, Dict[str, Any]]:
    reference = staged.fit_segment_reference(
        target[fit_mask],
        scale_floor=stage_d_contract.reference.scale_floor,
    )
    if reference.to_dict() != stage_d_contract.reference.to_dict():
        raise UpperGateFreezeError(
            "recomputed Stage-D reference differs"
        )

    scores = staged.segment_scores(
        target[calibration_mask],
        reference,
    )
    if scores["upper"].shape[1] != 1:
        raise ValueError("ground-truth calibration must have K=1")
    row_upper = scores["upper"][:, 0]
    group_upper, ordered_groups = staged._group_score_distribution(
        row_upper,
        np.asarray(groups)[calibration_mask],
        within_group_coverage=spec.within_group_row_coverage,
    )
    quantile = staged._conformal_quantile(
        group_upper,
        spec.conformal_group_coverage,
    )
    recomputed = float(quantile["threshold"])
    historical = float(stage_d_contract.upper_threshold)
    if float64_bits(recomputed) != float64_bits(historical):
        raise UpperGateFreezeError(
            "recomputed upper threshold is not bit-exact with Stage D: "
            f"{recomputed!r} vs {historical!r}"
        )
    if ordered_groups != sorted(
        set(np.asarray(groups)[calibration_mask].astype(str).tolist())
    ):
        raise AssertionError("upper calibration group order is unstable")
    if (
        str(gate_split["fit_group_sha256"])
        != stage_d_contract.fit_group_sha256
    ):
        raise UpperGateFreezeError(
            "Stage-D fit group identity changed"
        )
    if (
        str(gate_split["calibration_group_sha256"])
        != stage_d_contract.calibration_group_sha256
    ):
        raise UpperGateFreezeError(
            "Stage-D calibration group identity changed"
        )

    contract = OneSidedUpperContract(
        reference=reference,
        upper_threshold=recomputed,
        within_group_row_coverage=spec.within_group_row_coverage,
        conformal_group_coverage=spec.conformal_group_coverage,
        fit_group_count=int(gate_split["fit_group_count"]),
        calibration_group_count=int(
            gate_split["calibration_group_count"]
        ),
        fit_group_sha256=str(gate_split["fit_group_sha256"]),
        calibration_group_sha256=str(
            gate_split["calibration_group_sha256"]
        ),
        calibration_group_score_sha256=sha256_array(group_upper),
        stage_d_internal_contract_sha256=str(
            stage_d_contract_payload["contract_sha256"]
        ),
        stage_d2_raw_xyz_sha256=str(
            immutable["stage_d2_raw_xyz_sha256"]
        ),
        stage_d2_projection_artifact_fraction=float(
            immutable["stage_d2_projection_artifact_fraction"]
        ),
    )
    contract.validate()
    return contract, {
        "threshold": quantile,
        "recomputed_upper_threshold": recomputed,
        "stage_d_upper_threshold": historical,
        "threshold_bit_exact": True,
        "upper_row_score": _safe_stats(row_upper),
        "upper_row_score_sha256": sha256_array(row_upper),
        "upper_group_score": _safe_stats(group_upper),
        "upper_group_score_sha256": sha256_array(group_upper),
        "upper_group_order_sha256": sha256_strings(ordered_groups),
        "calibration_row_count": int(np.sum(calibration_mask)),
        "calibration_group_count": len(ordered_groups),
        "probe_used_for_fit": False,
        "model_candidate_used_for_fit": False,
        "condition_label_used_for_fit": False,
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
        [float(np.mean(records[group])) for group in ordered],
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


def upper_only_decomposition(
    *,
    value: np.ndarray,
    contract: OneSidedUpperContract,
    stage_d_contract: staged.SegmentGateContract,
    historical_geometry: stageb.GeometryContract,
    groups: Sequence[Any],
    condition_name: Sequence[Any],
) -> Dict[str, Any]:
    contract.validate()
    candidates = staged.as_candidates(value)
    scores = staged.segment_scores(candidates, contract.reference)

    upper_pass = np.asarray(
        scores["upper"] <= float(contract.upper_threshold),
        dtype=np.bool_,
    )
    upper_element_pass = np.asarray(
        scores["upper_element"] <= float(contract.upper_threshold),
        dtype=np.bool_,
    )

    # Lower score is retained only to expose projection diagnostics.
    historical_lower_pass = np.asarray(
        scores["lower"] <= float(stage_d_contract.lower_threshold),
        dtype=np.bool_,
    )
    historical_joint_pass = np.asarray(
        scores["joint"] <= float(stage_d_contract.joint_threshold),
        dtype=np.bool_,
    )

    historical = stageb.physical_validity(
        candidates,
        historical_geometry,
    )
    finite = np.asarray(historical["finite"], dtype=np.bool_)
    coordinate = np.asarray(historical["coordinate"], dtype=np.bool_)
    topology = np.asarray(historical["topology"], dtype=np.bool_)
    combined = finite & coordinate & topology & upper_pass

    row_upper_any = np.any(upper_pass, axis=1)
    row_combined_any = np.any(combined, axis=1)
    group_values = np.asarray(groups).astype(str)
    conditions = np.asarray(condition_name).astype(str)
    if group_values.shape != (candidates.shape[0],):
        raise ValueError("population group shape mismatch")
    if conditions.shape != group_values.shape:
        raise ValueError("population condition shape mismatch")

    upper_condition = {}
    combined_condition = {}
    for name in sorted(set(conditions.tolist())):
        selected = conditions == name
        upper_condition[name] = float(
            np.mean(row_upper_any[selected])
        )
        combined_condition[name] = float(
            np.mean(row_combined_any[selected])
        )

    position_rate = np.mean(upper_element_pass, axis=(0, 1))
    position_score = np.max(
        np.asarray(scores["upper_element"], dtype=np.float64),
        axis=(0, 1),
    )
    order = np.argsort(position_rate.reshape(-1))
    worst_positions = []
    for flat in order[: min(16, order.size)]:
        horizon, segment = np.unravel_index(
            int(flat),
            position_rate.shape,
        )
        worst_positions.append(
            {
                "horizon": int(horizon),
                "segment_index": int(segment),
                "upper_element_pass_rate":
                    float(position_rate[horizon, segment]),
                "maximum_upper_element_score":
                    float(position_score[horizon, segment]),
                "upper_element_score": _safe_stats(
                    scores["upper_element"][
                        ..., horizon, segment
                    ]
                ),
            }
        )

    upper_only_invalid = ~upper_pass & historical_lower_pass
    lower_only_diagnostic_invalid = (
        upper_pass & ~historical_lower_pass
    )
    both_direction_invalid = (
        ~upper_pass & ~historical_lower_pass
    )
    return {
        "shape": list(candidates.shape),
        "sha256": sha256_array(candidates),
        "upper_segment": {
            "candidate_rate": float(np.mean(upper_pass)),
            "row_any_rate": float(np.mean(row_upper_any)),
            "element_rate": float(np.mean(upper_element_pass)),
            "accepted_candidate_count": int(np.sum(upper_pass)),
            "candidate_count": int(upper_pass.size),
            "row_best_upper_score": _safe_stats(
                np.min(scores["upper"], axis=1)
            ),
            "all_candidate_upper_score": _safe_stats(
                scores["upper"]
            ),
            "condition_row_any_rate": upper_condition,
            "group_acceptance": _group_acceptance(
                row_upper_any,
                group_values,
                within_group_coverage=
                    contract.within_group_row_coverage,
            ),
            "worst_horizon_segment_positions": worst_positions,
        },
        "combined": {
            "candidate_rate": float(np.mean(combined)),
            "row_any_rate": float(np.mean(row_combined_any)),
            "accepted_candidate_count": int(np.sum(combined)),
            "candidate_count": int(combined.size),
            "condition_row_any_rate": combined_condition,
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
        },
        "lower_diagnostic": {
            "historical_lower_candidate_rate":
                float(np.mean(historical_lower_pass)),
            "historical_joint_candidate_rate":
                float(np.mean(historical_joint_pass)),
            "lower_score": _safe_stats(scores["lower"]),
            "upper_only_invalid_count":
                int(np.sum(upper_only_invalid)),
            "lower_only_diagnostic_invalid_count":
                int(np.sum(lower_only_diagnostic_invalid)),
            "both_direction_invalid_count":
                int(np.sum(both_direction_invalid)),
            "nonpositive_element_count":
                int(np.sum(scores["nonpositive"])),
            "used_for_validity": False,
        },
        "upper_valid_mask": upper_pass,
        "combined_valid_mask": combined,
    }


def public_decomposition(
    value: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        key: item
        for key, item in value.items()
        if key not in {"upper_valid_mask", "combined_valid_mask"}
    }


def branch_prefix_curve(
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
        raise ValueError("branch valid mask shape mismatch")
    result = {}
    for value in prefix_k:
        k = int(value)
        if not 1 <= k <= pred.shape[1]:
            raise ValueError(f"invalid branch K={k}")
        result[str(k)] = stagec.physical_valid_branch_metrics(
            pair_key=pair_key,
            condition_name=condition_name,
            target=target,
            candidates=pred[:, :k],
            valid_mask=valid[:, :k],
        )
    return result


def classify_result(
    *,
    spec: UpperGateSpec,
    calibration_gt: Mapping[str, Any],
    probe_gt: Mapping[str, Any],
    one_step: Mapping[str, Mapping[str, Any]],
    reverse_final: Mapping[str, Any],
    branch_k8: Mapping[str, Any],
) -> Dict[str, Any]:
    calibration_group = float(
        calibration_gt["upper_segment"]["group_acceptance"][
            "group_acceptance_rate"
        ]
    )
    probe_row = float(
        probe_gt["upper_segment"]["row_any_rate"]
    )
    probe_group = float(
        probe_gt["upper_segment"]["group_acceptance"][
            "group_acceptance_rate"
        ]
    )
    probe_condition = float(
        min(
            probe_gt["upper_segment"][
                "condition_row_any_rate"
            ].values()
        )
    )
    t10 = float(
        one_step["10"]["upper_segment"]["row_any_rate"]
    )
    reverse_upper = float(
        reverse_final["upper_segment"]["row_any_rate"]
    )
    reverse_combined = float(
        reverse_final["combined"]["row_any_rate"]
    )
    eligible = float(branch_k8["eligible_row_rate"])
    support = branch_k8["support_rate_among_eligible"]
    support_value = float(support) if support is not None else 0.0

    if calibration_group < spec.conformal_group_coverage:
        root = (
            "phase314b_r256_staged3_one_sided_upper_gate_"
            "internal_calibration_failed"
        )
        next_path = (
            "REPAIR_ONE_SIDED_GROUPED_CONFORMAL_UPPER_GATE"
        )
        locus = "upper_gate_calibration"
    elif (
        probe_row < spec.probe_row_acceptance_min
        or probe_group < spec.probe_group_acceptance_min
        or probe_condition < spec.probe_condition_acceptance_min
    ):
        root = (
            "phase314b_r256_staged3_one_sided_upper_gate_"
            "probe_generalization_failed"
        )
        next_path = (
            "ATTRIBUTE_ONE_SIDED_UPPER_GATE_PROBE_SHIFT_"
            "WITHOUT_USING_PROBE_FOR_CALIBRATION"
        )
        locus = "upper_gate_generalization"
    elif t10 < spec.one_step_t10_upper_acceptance_min:
        root = (
            "phase314b_r256_staged3_cable_x0_upper_"
            "segment_expansion_failed"
        )
        next_path = (
            "ADD_ORDERED_SEGMENT_UPPER_EXPANSION_OBJECTIVE_"
            "TO_CABLE_X0_DENOISER"
        )
        locus = "x0_upper_expansion"
    elif reverse_upper < spec.reverse_upper_row_any_min:
        root = (
            "phase314b_r256_staged3_reverse_upper_"
            "segment_transport_failed"
        )
        next_path = (
            "REPAIR_REVERSE_ORDERED_SEGMENT_EXPANSION_"
            "WITH_FROZEN_UPPER_GATE"
        )
        locus = "reverse_upper_transport"
    elif reverse_combined < spec.reverse_combined_row_any_min:
        root = (
            "phase314b_r256_staged3_upper_gate_passed_"
            "nonsegment_physical_failure_remains"
        )
        next_path = (
            "ATTRIBUTE_COORDINATE_OR_TOPOLOGY_FAILURE_"
            "WITH_FROZEN_UPPER_GATE"
        )
        locus = "nonsegment_physical_gate"
    elif eligible < spec.physical_branch_eligible_row_min:
        root = (
            "phase314b_r256_staged3_physical_candidate_"
            "branch_eligibility_failed"
        )
        next_path = (
            "REPAIR_PHYSICAL_CANDIDATE_ELIGIBILITY_"
            "BEFORE_BRANCH_TRANSPORT"
        )
        locus = "branch_eligibility"
    elif support_value < spec.physical_branch_support_min:
        root = (
            "phase314b_r256_staged3_physical_valid_"
            "branch_support_failed"
        )
        next_path = (
            "REPAIR_CABLE_ONLY_BRANCH_TRANSPORT_WITH_"
            "FROZEN_ONE_SIDED_GATE"
        )
        locus = "branch_transport"
    else:
        root = (
            "phase314b_r256_staged3_one_sided_upper_gate_"
            "and_branch_supported"
        )
        next_path = (
            "RUN_FROZEN_CABLE_ONLY_FORMAL_PILOT_AND_"
            "COLLECT_ACTION_DIVERSE_IDM_DATA"
        )
        locus = "none"

    return {
        "root_cause": root,
        "required_next_path": next_path,
        "primary_failure_locus": locus,
        "calibration_group_acceptance": calibration_group,
        "probe_upper_row_acceptance": probe_row,
        "probe_upper_group_acceptance": probe_group,
        "probe_minimum_condition_acceptance": probe_condition,
        "one_step_t10_upper_acceptance": t10,
        "reverse_upper_row_any": reverse_upper,
        "reverse_combined_row_any": reverse_combined,
        "physical_branch_eligible_row_rate": eligible,
        "physical_branch_support_among_eligible": support,
    }


def run_freeze_and_reaudit(
    *,
    root: Path,
    spec: Optional[UpperGateSpec] = None,
) -> Dict[str, Any]:
    active_spec = UpperGateSpec() if spec is None else spec
    active_spec.validate()
    repository_root = Path(root).resolve()

    immutable = validate_immutable_inputs(repository_root)
    logic = source_logic_audit(repository_root)
    if not logic["all_confirmed"]:
        raise UpperGateFreezeError(
            "one-sided gate source assumptions changed"
        )

    stage_d_contract, stage_d_payload = (
        staged1.load_stage_d_gate(
            repository_root / STAGE_D_GATE
        )
    )
    stageb_spec = stageb.DiagnosticSpec()
    stageb_spec.validate()

    arrays = stageb.load_npz_strict(
        repository_root / stageb.STAGEA_TRAIN_VIEW,
        required_keys=stageb.REQUIRED_TRAIN_KEYS,
    )
    validation = stageb.validate_train_view(arrays)
    runtime = stageb.set_deterministic_runtime(
        stageb_spec.seed
    )

    condition = np.asarray(
        arrays["diffusion_condition_x"],
        dtype=np.float32,
    )
    target = np.asarray(
        arrays["diffusion_target_cable"],
        dtype=np.float32,
    )
    groups = np.asarray(
        arrays["episode_group_key"]
    ).astype(str)
    conditions = np.asarray(
        arrays["condition_name"]
    ).astype(str)
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
    upper_contract, upper_fit = recompute_upper_contract(
        target=target,
        groups=groups,
        fit_mask=fit_mask,
        calibration_mask=calibration_mask,
        gate_split=gate_split,
        stage_d_contract=stage_d_contract,
        stage_d_contract_payload=stage_d_payload,
        immutable=immutable,
        spec=active_spec,
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
    expected_identity = immutable["stage_d2_worker"][
        "worker_result"
    ].get("frozen_replay", {}).get("identity")
    if not isinstance(expected_identity, dict):
        # Stage D.2 intentionally did not replay the model.  Bind to the
        # already immutable Stage-D.1 upstream identity instead.
        expected_identity = immutable["upstream_immutable"][
            "staged1_worker_evidence"
        ]["worker_result"]["frozen_replay"]["identity"]

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
            raise UpperGateFreezeError(
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
        raise UpperGateFreezeError(
            "frozen train-control prediction differs"
        )

    one_step_predictions, one_step_sha = (
        stagec.one_step_predictions(
            model=model,
            condition=condition[probe_mask],
            target=target[probe_mask],
            spec=stageb_spec,
            condition_standardizer=condition_standardizer,
            target_standardizer=target_standardizer,
            noise_seed=stageb_spec.seed + 2001,
        )
    )
    if one_step_sha != expected_identity[
        "one_step_prediction_sha256"
    ]:
        raise UpperGateFreezeError(
            "frozen one-step prediction differs"
        )

    candidates, trace = staged.reverse_sample_with_raw_trace(
        model=model,
        condition=condition[probe_mask],
        spec=stageb_spec,
        condition_standardizer=condition_standardizer,
        target_standardizer=target_standardizer,
        noise_seed=stageb_spec.seed + 3001,
        snapshot_timesteps=active_spec.snapshot_timesteps,
    )
    reverse_sha = sha256_array(candidates)
    if reverse_sha != expected_identity[
        "reverse_candidate_sha256"
    ]:
        raise UpperGateFreezeError(
            "frozen reverse candidates differ"
        )

    def evaluate(
        value: np.ndarray,
        mask: np.ndarray,
    ) -> Dict[str, Any]:
        return upper_only_decomposition(
            value=value,
            contract=upper_contract,
            stage_d_contract=stage_d_contract,
            historical_geometry=historical_geometry,
            groups=groups[mask],
            condition_name=conditions[mask],
        )

    fit_gt = evaluate(target[fit_mask], fit_mask)
    calibration_gt = evaluate(
        target[calibration_mask],
        calibration_mask,
    )
    training_gt = evaluate(target[train_mask], train_mask)
    probe_gt = evaluate(target[probe_mask], probe_mask)
    last_state = evaluate(
        stageb.last_state_baseline(condition[probe_mask]),
        probe_mask,
    )
    constant_velocity = evaluate(
        stageb.constant_velocity_baseline(
            condition[probe_mask]
        ),
        probe_mask,
    )
    one_step = {
        str(timestep): evaluate(prediction, probe_mask)
        for timestep, prediction in sorted(
            one_step_predictions.items()
        )
    }

    trace_metrics = {}
    for timestep in active_spec.snapshot_timesteps:
        record = trace[str(timestep)]
        trace_metrics[str(timestep)] = {
            "latent": public_decomposition(
                evaluate(record["latent"], probe_mask)
            ),
            "predicted_x0": public_decomposition(
                evaluate(
                    record["predicted_x0"],
                    probe_mask,
                )
            ),
            "latent_sha256": sha256_array(record["latent"]),
            "predicted_x0_sha256":
                sha256_array(record["predicted_x0"]),
        }

    reverse_full = evaluate(candidates, probe_mask)
    valid_mask = np.asarray(
        reverse_full["combined_valid_mask"],
        dtype=np.bool_,
    )
    branch_curve = branch_prefix_curve(
        pair_key=pair_key[probe_mask],
        condition_name=conditions[probe_mask],
        target=target[probe_mask],
        candidates=candidates,
        valid_mask=valid_mask,
        prefix_k=active_spec.branch_prefix_k,
    )
    branch_k8 = branch_curve[
        str(active_spec.branch_prefix_k[-1])
    ]
    branch_bootstrap = staged1.deterministic_eligible_bootstrap(
        branch_k8,
        resamples=active_spec.bootstrap_resamples,
        seed=active_spec.bootstrap_seed,
    )

    classification = classify_result(
        spec=active_spec,
        calibration_gt=calibration_gt,
        probe_gt=probe_gt,
        one_step=one_step,
        reverse_final=reverse_full,
        branch_k8=branch_k8,
    )

    contract_payload = upper_contract.to_dict()
    contract_payload["contract_sha256"] = sha256_bytes(
        stable_json_bytes(contract_payload)
    )
    return {
        "phase": PHASE,
        "phase_id": PHASE_ID,
        "schema": "phase314b_r256_staged3_worker_result_v1",
        "verdict": "PASS",
        "scientific_status": "BLOCKED",
        "root_cause": classification["root_cause"],
        "required_next_path":
            classification["required_next_path"],
        "upper_gate_spec": asdict(active_spec),
        "frozen_stageb_spec": asdict(stageb_spec),
        "runtime": runtime,
        "immutable_inputs": {
            "base_commit": immutable["base_commit"],
            "base_file_sha256":
                immutable["base_file_sha256"],
            "stage_d2_raw_xyz_sha256":
                immutable["stage_d2_raw_xyz_sha256"],
            "stage_d2_projection_artifact_fraction":
                immutable[
                    "stage_d2_projection_artifact_fraction"
                ],
        },
        "source_logic_audit": logic,
        "train_view_validation": validation,
        "split": {
            "stageb_training_rows": int(np.sum(train_mask)),
            "stageb_probe_rows": int(np.sum(probe_mask)),
            "reference_fit_rows": int(np.sum(fit_mask)),
            "gate_calibration_rows":
                int(np.sum(calibration_mask)),
            "stageb_group_count": len(stageb_mapping),
            "reference_fit_groups":
                int(gate_split["fit_group_count"]),
            "gate_calibration_groups":
                int(gate_split["calibration_group_count"]),
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
            "matches_stage_b_c_d_d1": True,
            "model_changed": False,
            "scheduler_changed": False,
            "split_changed": False,
            "normalization_changed": False,
            "objective_changed": False,
            "candidate_count_changed": False,
        },
        "upper_gate_contract": contract_payload,
        "upper_gate_fit": upper_fit,
        "ground_truth_evaluation": {
            "reference_fit": public_decomposition(fit_gt),
            "gate_calibration":
                public_decomposition(calibration_gt),
            "combined_stageb_training":
                public_decomposition(training_gt),
            "frozen_probe": public_decomposition(probe_gt),
        },
        "frozen_prediction_evaluation": {
            "last_state_baseline":
                public_decomposition(last_state),
            "constant_velocity_baseline":
                public_decomposition(constant_velocity),
            "one_step_predictions": {
                key: public_decomposition(value)
                for key, value in one_step.items()
            },
            "reverse_final":
                public_decomposition(reverse_full),
            "reverse_trace": trace_metrics,
        },
        "physical_branch_attribution": {
            "prefix_k_curve": branch_curve,
            "k8": branch_k8,
            "eligible_bootstrap": branch_bootstrap,
        },
        "classification": classification,
        "one_sided_contract_frozen": True,
        "selected_gate_contract":
            "one_sided_xy_upper_grouped_conformal_v1",
        "gate_selected": True,
        "threshold_recomputed": True,
        "threshold_bit_exact_with_stage_d_upper": True,
        "threshold_value_changed_from_stage_d_upper": False,
        "historical_gate_modified": False,
        "lower_score_used_for_validity": False,
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


def identity_projection(
    result: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "root_cause": result["root_cause"],
        "required_next_path": result["required_next_path"],
        "frozen_replay": result["frozen_replay"],
        "upper_gate_contract": result["upper_gate_contract"],
        "upper_gate_fit": result["upper_gate_fit"],
        "ground_truth_evaluation":
            result["ground_truth_evaluation"],
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
        "frozen_identity_exact": {
            key: left["frozen_replay"]["identity"][key]
            == right["frozen_replay"]["identity"][key]
            for key in sorted(left["frozen_replay"]["identity"])
        },
        "upper_contract_exact": (
            left["upper_gate_contract"]
            == right["upper_gate_contract"]
        ),
        "classification_exact": (
            left["classification"] == right["classification"]
        ),
    }
