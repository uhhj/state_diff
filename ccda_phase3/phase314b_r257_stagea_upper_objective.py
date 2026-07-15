"""Phase3.14b-r2.5.7 Stage A upper-expansion objective calibration.

r2.5.6 Stage D.3 Resume1 froze a valid one-sided ordered-XY segment
contract and established that the frozen cable x0 denoiser violates it at
every one-step evaluation timestep.  The Stage-B model directly predicts
normalized x0; its training loss is MSE against normalized clean cable
future.  This stage therefore introduces a train-only raw-geometry loss on
that direct x0 prediction.

The objective is aligned with the frozen upper gate but intentionally avoids
dividing by the 1e-3 robust scale floor.  For each ordered XY segment:

    log_excess = relu(log(length) - log(allowed_upper_length))

where:

    log(allowed_upper_length)
        = center_log + frozen_upper_threshold * scale_log

The loss combines a row-wise maximum Huber penalty with a smaller element-wise
Huber penalty.  It has the same zero set as the frozen upper gate, while keeping
gradients in physical log-ratio units rather than robust-z units in the
thousands.

Hyperparameter selection is leakage-controlled:
* the frozen Stage-B training partition is split again by episode group;
* one control and three nonzero lambda candidates use exactly the same
  initialization, mini-batch indices, timesteps and noise;
* configuration selection uses only the internal train-only holdout;
* the frozen probe target is not accessed until a nonzero configuration is
  selected;
* the selected configuration is retrained from scratch on all 874 Stage-B
  training rows before the one-time frozen-probe audit.

No checkpoint, tensor, NPZ, cache, image or video is persisted.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from ccda_phase3 import phase314b_r256_stageb_cable_diffusion as stageb
from ccda_phase3 import phase314b_r256_stagec_reverse_attribution as stagec
from ccda_phase3 import phase314b_r256_staged_segment_recalibration as staged
from ccda_phase3 import phase314b_r256_staged1_asymmetric_gate_audit as staged1
from ccda_phase3 import phase314b_r256_staged3_resume1_upper_gate_freeze as staged3

PHASE = "Phase3.14b-r2.5.7 Stage A"
PHASE_ID = "phase314b_r257_stagea"
BASE_EVIDENCE_COMMIT = "f188c6abb8986526b18404d61e421cfdce0632fe"
EXPECTED_SUBMODULE_COMMIT = "633a88752445cf5d6776ed374fdbbdb35f93050c"

STAGEB_SOURCE = "ccda_phase3/phase314b_r256_stageb_cable_diffusion.py"
STAGEC_SOURCE = "ccda_phase3/phase314b_r256_stagec_reverse_attribution.py"
STAGED_SOURCE = "ccda_phase3/phase314b_r256_staged_segment_recalibration.py"
STAGED3_SOURCE = (
    "ccda_phase3/phase314b_r256_staged3_resume1_upper_gate_freeze.py"
)
STAGED3_CONTRACT = (
    "reports/"
    "phase3_14b_r256_staged3_resume1_one_sided_upper_contract.json"
)
STAGED3_TEST_GATE = (
    "reports/phase3_14b_r256_staged3_resume1_test_gate_summary.json"
)
STAGED3_WORKER = (
    "reports/phase3_14b_r256_staged3_resume1_worker_evidence.json"
)
STAGED3_SUMMARY = (
    "reports/phase3_14b_r256_staged3_resume1_summary.json"
)
STAGED3_REPORT = (
    "reports/phase3_14b_r256_staged3_resume1_report.md"
)

BASE_BOUND_FILES = (
    STAGEB_SOURCE,
    STAGEC_SOURCE,
    STAGED_SOURCE,
    STAGED3_SOURCE,
    STAGED3_CONTRACT,
    STAGED3_TEST_GATE,
    STAGED3_WORKER,
    STAGED3_SUMMARY,
    STAGED3_REPORT,
)

EXPECTED_FROZEN_IDENTITY = {
    "final_model_sha256":
        "4a9ebcdb9d1995602a501c8820e07c0efcb757f1e3efff4ac0f2c5a4a23421d2",
    "final_optimizer_sha256":
        "ff8feee0e677c2720a7807ad19984764f2d69e1e0c296a9c4056f426d2f22601",
    "loss_history_sha256":
        "9f8db5bf1563f34e4c94557eb9fb729728f3e16c9beb9c49d746f9ee4eba4d4d",
    "gradient_history_sha256":
        "e434d77c214d52b918b57ed4445d4ac990bb0fb03afd674c2e76b2a6b96d84aa",
    "source_exposure_sha256":
        "571adb6f5b59428b74b40171169c8e9b7381075baf1deecb398971478be4393a",
    "train_control_prediction_sha256":
        "bb9b91da6eb324b24af10322ee20509ad426e54d851c194d62f71e63c68eb076",
    "one_step_prediction_sha256":
        "d647ac6b547a42212bbc82582fabed565f3f8340cd78a09d6c61ebb8ab6742ef",
    "reverse_candidate_sha256":
        "1ab9f8d0026f215bf9e3b8aafeb4f8799bc1f8d329d8773e415413699a6c6d2d",
}

SNAPSHOT_TIMESTEPS = (99, 75, 50, 25, 10, 0)


class UpperObjectiveCalibrationError(RuntimeError):
    """Raised when a frozen identity or leakage boundary is violated."""


@dataclass(frozen=True)
class UpperObjectiveSpec:
    """Pre-registered train-only objective-calibration contract."""

    candidate_lambdas: Tuple[float, ...] = (
        0.0,
        1.0e-3,
        1.0e-2,
        1.0e-1,
    )
    selection_group_folds: int = 4
    selection_holdout_fold: int = 0

    huber_delta_log_ratio: float = 0.10
    element_loss_weight: float = 0.25
    segment_length_epsilon: float = 1.0e-8

    selection_t10_upper_row_min: float = 0.75
    selection_t25_upper_row_min: float = 0.60
    selection_t50_upper_row_min: float = 0.50
    selection_nmse_ratio_max: float = 1.25
    selection_train_control_ratio_max: float = 1.25

    final_train_control_nmse_max: float = 0.005
    final_train_control_ratio_max: float = 1.25
    final_one_step_nmse_ratio_max: float = 1.25
    final_reverse_nmse_ratio_max: float = 1.25
    final_diversity_ratio_min: float = 0.50

    final_t10_upper_row_min: float = 0.75
    final_reverse_upper_row_min: float = 0.95
    final_reverse_combined_row_min: float = 0.95
    final_branch_eligible_row_min: float = 0.95
    final_branch_support_min: float = 0.75

    branch_prefix_k: Tuple[int, ...] = (1, 2, 4, 8)
    bootstrap_resamples: int = 4096
    bootstrap_seed: int = 257100
    snapshot_timesteps: Tuple[int, ...] = SNAPSHOT_TIMESTEPS

    def validate(self) -> None:
        if self.candidate_lambdas[0] != 0.0:
            raise ValueError("first candidate must be the zero-weight control")
        if tuple(sorted(set(self.candidate_lambdas))) != self.candidate_lambdas:
            raise ValueError("candidate lambdas must be ordered and unique")
        if len(self.candidate_lambdas) < 2:
            raise ValueError("at least one nonzero lambda is required")
        if any(value < 0.0 for value in self.candidate_lambdas):
            raise ValueError("candidate lambda is negative")
        if self.selection_group_folds < 2:
            raise ValueError("selection group split requires at least two folds")
        if not 0 <= self.selection_holdout_fold < self.selection_group_folds:
            raise ValueError("selection holdout fold is invalid")
        if self.huber_delta_log_ratio <= 0.0:
            raise ValueError("Huber delta must be positive")
        if not 0.0 <= self.element_loss_weight <= 1.0:
            raise ValueError("element loss weight is invalid")
        if self.segment_length_epsilon <= 0.0:
            raise ValueError("segment epsilon must be positive")
        for value in (
            self.selection_t10_upper_row_min,
            self.selection_t25_upper_row_min,
            self.selection_t50_upper_row_min,
            self.selection_nmse_ratio_max,
            self.selection_train_control_ratio_max,
            self.final_train_control_nmse_max,
            self.final_train_control_ratio_max,
            self.final_one_step_nmse_ratio_max,
            self.final_reverse_nmse_ratio_max,
            self.final_diversity_ratio_min,
            self.final_t10_upper_row_min,
            self.final_reverse_upper_row_min,
            self.final_reverse_combined_row_min,
            self.final_branch_eligible_row_min,
            self.final_branch_support_min,
        ):
            if float(value) <= 0.0:
                raise ValueError("registered decision threshold is not positive")
        if tuple(sorted(set(self.branch_prefix_k))) != self.branch_prefix_k:
            raise ValueError("branch K values must be ordered and unique")
        if self.branch_prefix_k[-1] != stageb.DiagnosticSpec().reverse_candidates:
            raise ValueError("branch K curve must end at frozen K")
        if tuple(sorted(set(self.snapshot_timesteps), reverse=True)) != (
            self.snapshot_timesteps
        ):
            raise ValueError("snapshot timesteps must be descending and unique")
        if self.bootstrap_resamples <= 0:
            raise ValueError("bootstrap count must be positive")


@dataclass(frozen=True)
class UpperObjectiveContract:
    """Frozen differentiable training objective."""

    upper_gate_contract_sha256: str
    upper_threshold: float
    reference_center_log: np.ndarray
    reference_scale_log: np.ndarray
    allowed_upper_log_length: np.ndarray
    huber_delta_log_ratio: float
    element_loss_weight: float
    segment_length_epsilon: float

    def validate(self) -> None:
        expected = (stageb.FUTURE_STEPS, stageb.BEADS - 1)
        for name, value in (
            ("reference_center_log", self.reference_center_log),
            ("reference_scale_log", self.reference_scale_log),
            ("allowed_upper_log_length", self.allowed_upper_log_length),
        ):
            array = np.asarray(value)
            if array.shape != expected:
                raise ValueError(f"{name} shape changed: {array.shape}")
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} contains NaN or Inf")
        expected_allowed = (
            np.asarray(self.reference_center_log, dtype=np.float64)
            + float(self.upper_threshold)
            * np.asarray(self.reference_scale_log, dtype=np.float64)
        )
        if not np.array_equal(
            expected_allowed.astype(np.float32),
            np.asarray(self.allowed_upper_log_length, dtype=np.float32),
        ):
            raise ValueError("allowed upper log-length is inconsistent")
        if len(self.upper_gate_contract_sha256) != 64:
            raise ValueError("upper gate SHA is malformed")
        if self.huber_delta_log_ratio <= 0.0:
            raise ValueError("Huber delta is invalid")
        if not 0.0 <= self.element_loss_weight <= 1.0:
            raise ValueError("element loss weight is invalid")
        if self.segment_length_epsilon <= 0.0:
            raise ValueError("segment epsilon is invalid")

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return {
            "schema": "phase314b_r257_stagea_upper_objective_contract_v1",
            "model_parameterization": "direct_normalized_x0_prediction",
            "raw_geometry_space": "ordered cable XY [B,4,24,2]",
            "gate_zero_set": (
                "all ordered segment log lengths <= "
                "center_log + upper_threshold * scale_log"
            ),
            "element_excess": (
                "relu(log(segment_length) - allowed_upper_log_length)"
            ),
            "row_term": "mean(huber(max_element_excess_per_row))",
            "element_term": (
                "element_loss_weight * mean(huber(element_excess))"
            ),
            "upper_gate_contract_sha256": self.upper_gate_contract_sha256,
            "upper_threshold": float(self.upper_threshold),
            "huber_delta_log_ratio": float(self.huber_delta_log_ratio),
            "element_loss_weight": float(self.element_loss_weight),
            "segment_length_epsilon": float(self.segment_length_epsilon),
            "reference_center_log":
                np.asarray(self.reference_center_log).tolist(),
            "reference_scale_log":
                np.asarray(self.reference_scale_log).tolist(),
            "allowed_upper_log_length":
                np.asarray(self.allowed_upper_log_length).tolist(),
            "reference_center_log_sha256":
                sha256_array(self.reference_center_log),
            "reference_scale_log_sha256":
                sha256_array(self.reference_scale_log),
            "allowed_upper_log_length_sha256":
                sha256_array(self.allowed_upper_log_length),
            "uses_robust_z_as_loss": False,
            "uses_lower_xy_constraint": False,
            "uses_probe_for_selection": False,
            "uses_formal_validation_or_test": False,
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
        raise UpperObjectiveCalibrationError(
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
        raise UpperObjectiveCalibrationError(
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
        raise UpperObjectiveCalibrationError(
            f"base-bound r2.5.6 file changed: {relative}"
        )
    return sha256_bytes(observed)


def validate_immutable_inputs(root: Path) -> Dict[str, Any]:
    repository_root = Path(root).resolve()
    bound = {
        relative: assert_base_file_bound(repository_root, relative)
        for relative in BASE_BOUND_FILES
    }
    summary = load_json(repository_root / STAGED3_SUMMARY)
    if summary.get("verdict") != "PASS":
        raise UpperObjectiveCalibrationError(
            "Stage-D.3 Resume1 verdict is not PASS"
        )
    if summary.get("scientific_status") != "BLOCKED":
        raise UpperObjectiveCalibrationError(
            "Stage-D.3 Resume1 scientific status changed"
        )
    if summary.get("root_cause") != (
        "phase314b_r256_staged3_cable_x0_upper_segment_expansion_failed"
    ):
        raise UpperObjectiveCalibrationError(
            "Stage-D.3 Resume1 root cause changed"
        )
    if summary.get("required_next_path") != (
        "ADD_ORDERED_SEGMENT_UPPER_EXPANSION_OBJECTIVE_TO_CABLE_X0_DENOISER"
    ):
        raise UpperObjectiveCalibrationError(
            "Stage-D.3 Resume1 next path changed"
        )
    if summary.get("gate_selected") is not True:
        raise UpperObjectiveCalibrationError(
            "one-sided upper gate is not selected"
        )
    if summary.get("lower_score_used_for_validity") is not False:
        raise UpperObjectiveCalibrationError(
            "lower XY score re-entered validity"
        )
    if summary.get("train_only_recommendation") is not None:
        raise UpperObjectiveCalibrationError(
            "Stage-D.3 Resume1 selected a model recommendation"
        )
    if summary.get("selected_configuration") is not None:
        raise UpperObjectiveCalibrationError(
            "Stage-D.3 Resume1 selected a model configuration"
        )

    worker = load_json(repository_root / STAGED3_WORKER)
    if worker.get("workers_exact") is not True:
        raise UpperObjectiveCalibrationError(
            "Stage-D.3 Resume1 workers were not exact"
        )
    worker_result = worker.get("worker_result")
    if not isinstance(worker_result, dict):
        raise UpperObjectiveCalibrationError(
            "Stage-D.3 Resume1 worker result is missing"
        )
    frozen_identity = worker_result.get(
        "frozen_replay", {}
    ).get("identity")
    if frozen_identity != EXPECTED_FROZEN_IDENTITY:
        raise UpperObjectiveCalibrationError(
            "frozen r2.5.6 identity changed"
        )

    contract_payload = load_json(
        repository_root / STAGED3_CONTRACT
    )
    if contract_payload.get("primary_gate") != (
        "upper_score <= upper_threshold"
    ):
        raise UpperObjectiveCalibrationError(
            "one-sided contract primary gate changed"
        )
    if contract_payload.get("lower_score_role") != "diagnostic_only":
        raise UpperObjectiveCalibrationError(
            "one-sided contract lower role changed"
        )
    if contract_payload.get("uses_lower_score_for_validity") is not False:
        raise UpperObjectiveCalibrationError(
            "one-sided contract uses lower score"
        )
    if float(contract_payload.get("upper_threshold", -1.0)) != (
        3.937946393999212
    ):
        raise UpperObjectiveCalibrationError(
            "frozen upper threshold changed"
        )
    internal_sha = contract_payload.get("contract_sha256")
    if internal_sha != (
        "24f5ce7aa5c6d3c45ab6d4e659fa1c4719ffb486b61022db453b289e990b2781"
    ):
        raise UpperObjectiveCalibrationError(
            "one-sided internal contract SHA changed"
        )

    upstream = staged3.validate_immutable_inputs(repository_root)
    return {
        "base_commit": BASE_EVIDENCE_COMMIT,
        "base_file_sha256": bound,
        "staged3_summary": summary,
        "staged3_worker": worker,
        "staged3_contract": contract_payload,
        "frozen_identity": frozen_identity,
        "upstream_immutable": upstream,
    }


def source_logic_audit(root: Path) -> Dict[str, Any]:
    repository_root = Path(root).resolve()
    stageb_text = (repository_root / STAGEB_SOURCE).read_text(
        encoding="utf-8"
    )
    staged3_text = (repository_root / STAGED3_SOURCE).read_text(
        encoding="utf-8"
    )
    checks = {
        "model_is_named_x0_denoiser": (
            "class CableX0Denoiser" in stageb_text
        ),
        "model_output_is_direct_x0_shape": (
            "return predicted.reshape(" in stageb_text
            and "FUTURE_STEPS" in stageb_text
            and "CABLE_DIM" in stageb_text
        ),
        "training_target_is_clean_normalized_x0": (
            "loss = torch.mean((predicted - clean) ** 2)"
            in stageb_text
        ),
        "reverse_treats_model_output_as_predicted_x0": (
            "predicted_x0 = model(value, timestep, condition_tensor)"
            in stageb_text
        ),
        "target_standardizer_has_denormalize": (
            "def denormalize(self, value: np.ndarray)"
            in stageb_text
        ),
        "one_sided_gate_uses_upper_only": (
            "upper_score <= upper_threshold" in staged3_text
            and '"lower_score_role": "diagnostic_only"' in staged3_text
        ),
        "stageb_training_uses_dedicated_generator": (
            "generator.manual_seed(spec.seed + 17)" in stageb_text
        ),
    }
    return {
        "checks": checks,
        "all_confirmed": bool(all(checks.values())),
        "source_sha256": {
            STAGEB_SOURCE: sha256_file(repository_root / STAGEB_SOURCE),
            STAGEC_SOURCE: sha256_file(repository_root / STAGEC_SOURCE),
            STAGED_SOURCE: sha256_file(repository_root / STAGED_SOURCE),
            STAGED3_SOURCE: sha256_file(repository_root / STAGED3_SOURCE),
        },
        "objective_placement": (
            "direct model x0 output after torch denormalization to raw XY"
        ),
        "epsilon_reconstruction_required": False,
    }


def load_upper_gate_contract(
    payload: Mapping[str, Any],
) -> staged3.OneSidedUpperContract:
    reference_payload = payload.get("reference")
    if not isinstance(reference_payload, Mapping):
        raise UpperObjectiveCalibrationError(
            "one-sided reference is missing"
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
    contract = staged3.OneSidedUpperContract(
        reference=reference,
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
        calibration_group_score_sha256=str(
            payload["calibration_group_score_sha256"]
        ),
        stage_d_internal_contract_sha256=str(
            payload["stage_d_internal_contract_sha256"]
        ),
        stage_d2_raw_xyz_sha256=str(
            payload["stage_d2_raw_xyz_sha256"]
        ),
        stage_d2_projection_artifact_fraction=float(
            payload["stage_d2_projection_artifact_fraction"]
        ),
    )
    contract.validate()
    return contract


def build_objective_contract(
    *,
    upper_gate: staged3.OneSidedUpperContract,
    upper_gate_payload: Mapping[str, Any],
    spec: UpperObjectiveSpec,
) -> UpperObjectiveContract:
    allowed = (
        upper_gate.reference.center_log.astype(np.float64)
        + float(upper_gate.upper_threshold)
        * upper_gate.reference.scale_log.astype(np.float64)
    ).astype(np.float32)
    result = UpperObjectiveContract(
        upper_gate_contract_sha256=str(
            upper_gate_payload["contract_sha256"]
        ),
        upper_threshold=float(upper_gate.upper_threshold),
        reference_center_log=
            upper_gate.reference.center_log.copy(),
        reference_scale_log=
            upper_gate.reference.scale_log.copy(),
        allowed_upper_log_length=allowed,
        huber_delta_log_ratio=float(
            spec.huber_delta_log_ratio
        ),
        element_loss_weight=float(
            spec.element_loss_weight
        ),
        segment_length_epsilon=float(
            spec.segment_length_epsilon
        ),
    )
    result.validate()
    return result


def deterministic_selection_split(
    groups: Sequence[Any],
    stageb_train_mask: np.ndarray,
    *,
    folds: int,
    holdout_fold: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    values = np.asarray(groups).astype(str)
    train = np.asarray(stageb_train_mask, dtype=np.bool_)
    if values.shape != train.shape:
        raise ValueError("selection split shape mismatch")
    unique = sorted(set(values[train].tolist()))
    if len(unique) < folds:
        raise UpperObjectiveCalibrationError(
            "insufficient Stage-B training groups for selection split"
        )
    mapping = {
        group: index % int(folds)
        for index, group in enumerate(unique)
    }
    assignment = np.full(values.shape, -1, dtype=np.int64)
    for index in np.flatnonzero(train):
        assignment[index] = mapping[values[index]]
    holdout = train & (assignment == int(holdout_fold))
    objective_train = train & (assignment != int(holdout_fold))
    if not np.any(holdout) or not np.any(objective_train):
        raise UpperObjectiveCalibrationError(
            "objective selection split is empty"
        )
    for group in unique:
        observed = set(
            assignment[(values == group) & train].tolist()
        )
        if len(observed) != 1:
            raise AssertionError(
                "episode group crossed objective selection split"
            )
    if np.any(holdout & objective_train):
        raise AssertionError("selection rows overlap")
    if not np.array_equal(
        holdout | objective_train,
        train,
    ):
        raise AssertionError("selection split lost Stage-B rows")
    train_groups = sorted(set(values[objective_train].tolist()))
    holdout_groups = sorted(set(values[holdout].tolist()))
    return objective_train, holdout, {
        "folds": int(folds),
        "holdout_fold": int(holdout_fold),
        "objective_train_rows": int(np.sum(objective_train)),
        "selection_holdout_rows": int(np.sum(holdout)),
        "objective_train_groups": len(train_groups),
        "selection_holdout_groups": len(holdout_groups),
        "objective_train_group_sha256":
            sha256_strings(train_groups),
        "selection_holdout_group_sha256":
            sha256_strings(holdout_groups),
        "group_overlap": 0,
        "row_overlap": 0,
        "all_stageb_training_rows_assigned": True,
    }


def huber_positive_torch(
    value: Any,
    *,
    delta: float,
) -> Any:
    torch, _ = stageb._torch_imports()
    positive = torch.clamp(value, min=0.0)
    delta_value = torch.as_tensor(
        float(delta),
        dtype=positive.dtype,
        device=positive.device,
    )
    return torch.where(
        positive <= delta_value,
        0.5 * positive * positive / delta_value,
        positive - 0.5 * delta_value,
    )


def upper_objective_terms_torch(
    predicted_x0_z: Any,
    *,
    target_standardizer: stageb.ArrayStandardizer,
    contract: UpperObjectiveContract,
) -> Dict[str, Any]:
    torch, _ = stageb._torch_imports()
    contract.validate()
    if tuple(predicted_x0_z.shape[1:]) != (
        stageb.FUTURE_STEPS,
        stageb.CABLE_DIM,
    ):
        raise ValueError("predicted x0 must be [B,4,48]")
    device = predicted_x0_z.device
    mean = torch.as_tensor(
        target_standardizer.mean,
        dtype=predicted_x0_z.dtype,
        device=device,
    )
    scale = torch.as_tensor(
        target_standardizer.scale,
        dtype=predicted_x0_z.dtype,
        device=device,
    )
    predicted_raw = predicted_x0_z * scale + mean
    points = predicted_raw.reshape(
        predicted_raw.shape[0],
        stageb.FUTURE_STEPS,
        stageb.BEADS,
        2,
    )
    delta_xy = points[:, :, 1:, :] - points[:, :, :-1, :]
    lengths = torch.sqrt(
        torch.sum(delta_xy * delta_xy, dim=-1)
        + float(contract.segment_length_epsilon) ** 2
    )
    log_lengths = torch.log(lengths)
    allowed = torch.as_tensor(
        contract.allowed_upper_log_length,
        dtype=predicted_x0_z.dtype,
        device=device,
    )
    excess = torch.relu(log_lengths - allowed)
    row_max = torch.amax(excess, dim=(1, 2))
    row_huber = huber_positive_torch(
        row_max,
        delta=contract.huber_delta_log_ratio,
    )
    element_huber = huber_positive_torch(
        excess,
        delta=contract.huber_delta_log_ratio,
    )
    row_loss = torch.mean(row_huber)
    element_loss = torch.mean(element_huber)
    total = (
        row_loss
        + float(contract.element_loss_weight) * element_loss
    )
    return {
        "total": total,
        "row": row_loss,
        "element": element_loss,
        "maximum_excess": torch.max(excess),
        "positive_element_rate":
            torch.mean((excess > 0.0).to(predicted_x0_z.dtype)),
        "row_violation_rate":
            torch.mean((row_max > 0.0).to(predicted_x0_z.dtype)),
    }


def train_model_with_upper_objective(
    *,
    condition: np.ndarray,
    target: np.ndarray,
    stageb_spec: stageb.DiagnosticSpec,
    objective_contract: UpperObjectiveContract,
    lambda_upper: float,
    condition_standardizer: stageb.ArrayStandardizer,
    target_standardizer: stageb.ArrayStandardizer,
) -> Tuple[Any, Dict[str, Any]]:
    torch, _ = stageb._torch_imports()
    if lambda_upper < 0.0:
        raise ValueError("lambda_upper is negative")
    device = torch.device("cuda:0")
    model = stageb.make_model(stageb_spec).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=stageb_spec.learning_rate,
        weight_decay=stageb_spec.weight_decay,
    )
    scheduler = stageb.scheduler_arrays(stageb_spec)
    alpha_bar = torch.as_tensor(
        scheduler["alpha_bar"],
        dtype=torch.float32,
        device=device,
    )
    condition_z = torch.as_tensor(
        condition_standardizer.normalize(condition),
        dtype=torch.float32,
        device=device,
    )
    target_z = torch.as_tensor(
        target_standardizer.normalize(target),
        dtype=torch.float32,
        device=device,
    )
    initial_model_sha = stageb.tensor_state_sha256(model)
    initial_optimizer_sha = stageb.optimizer_state_sha256(optimizer)

    generator = torch.Generator(device=device)
    generator.manual_seed(stageb_spec.seed + 17)
    total_values: List[float] = []
    diffusion_values: List[float] = []
    geometry_values: List[float] = []
    geometry_row_values: List[float] = []
    geometry_element_values: List[float] = []
    violation_rate_values: List[float] = []
    maximum_excess_values: List[float] = []
    gradient_values: List[float] = []
    exposure = np.zeros(condition.shape[0], dtype=np.int64)

    model.train()
    for _step in range(stageb_spec.train_steps):
        indices = torch.randint(
            low=0,
            high=condition.shape[0],
            size=(stageb_spec.batch_size,),
            generator=generator,
            device=device,
        )
        timestep = torch.randint(
            low=0,
            high=stageb_spec.train_timesteps,
            size=(stageb_spec.batch_size,),
            generator=generator,
            device=device,
        )
        noise = torch.randn(
            (
                stageb_spec.batch_size,
                stageb.FUTURE_STEPS,
                stageb.CABLE_DIM,
            ),
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        clean = target_z[indices]
        alpha = alpha_bar[timestep].reshape(-1, 1, 1)
        noisy = (
            torch.sqrt(alpha) * clean
            + torch.sqrt(1.0 - alpha) * noise
        )
        predicted = model(
            noisy,
            timestep,
            condition_z[indices],
        )
        diffusion_loss = torch.mean((predicted - clean) ** 2)
        if float(lambda_upper) == 0.0:
            geometry_terms = None
            loss = diffusion_loss
        else:
            geometry_terms = upper_objective_terms_torch(
                predicted,
                target_standardizer=target_standardizer,
                contract=objective_contract,
            )
            loss = (
                diffusion_loss
                + float(lambda_upper) * geometry_terms["total"]
            )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=10.0,
        )
        optimizer.step()

        total_values.append(float(loss.detach().cpu()))
        diffusion_values.append(
            float(diffusion_loss.detach().cpu())
        )
        if geometry_terms is None:
            geometry_values.append(0.0)
            geometry_row_values.append(0.0)
            geometry_element_values.append(0.0)
            violation_rate_values.append(0.0)
            maximum_excess_values.append(0.0)
        else:
            geometry_values.append(
                float(geometry_terms["total"].detach().cpu())
            )
            geometry_row_values.append(
                float(geometry_terms["row"].detach().cpu())
            )
            geometry_element_values.append(
                float(geometry_terms["element"].detach().cpu())
            )
            violation_rate_values.append(
                float(
                    geometry_terms["row_violation_rate"]
                    .detach()
                    .cpu()
                )
            )
            maximum_excess_values.append(
                float(
                    geometry_terms["maximum_excess"]
                    .detach()
                    .cpu()
                )
            )
        gradient_values.append(
            float(gradient_norm.detach().cpu())
        )
        np.add.at(
            exposure,
            indices.detach().cpu().numpy(),
            1,
        )

    model.eval()
    total_array = np.asarray(total_values, dtype=np.float64)
    diffusion_array = np.asarray(
        diffusion_values,
        dtype=np.float64,
    )
    geometry_array = np.asarray(
        geometry_values,
        dtype=np.float64,
    )
    gradient_array = np.asarray(
        gradient_values,
        dtype=np.float64,
    )
    exposure_array = np.asarray(exposure, dtype=np.int64)
    records = {
        "lambda_upper": float(lambda_upper),
        "initial_model_sha256": initial_model_sha,
        "final_model_sha256":
            stageb.tensor_state_sha256(model),
        "initial_optimizer_sha256":
            initial_optimizer_sha,
        "final_optimizer_sha256":
            stageb.optimizer_state_sha256(optimizer),
        "total_loss_history_sha256":
            sha256_array(total_array),
        "diffusion_loss_history_sha256":
            sha256_array(diffusion_array),
        "geometry_loss_history_sha256":
            sha256_array(geometry_array),
        "gradient_history_sha256":
            sha256_array(gradient_array),
        "source_exposure_sha256":
            sha256_array(exposure_array),
        "total_loss_first": float(total_array[0]),
        "total_loss_final": float(total_array[-1]),
        "total_loss_tail_mean":
            float(np.mean(total_array[-100:])),
        "diffusion_loss_tail_mean":
            float(np.mean(diffusion_array[-100:])),
        "geometry_loss_tail_mean":
            float(np.mean(geometry_array[-100:])),
        "geometry_row_loss_tail_mean":
            float(np.mean(geometry_row_values[-100:])),
        "geometry_element_loss_tail_mean":
            float(np.mean(geometry_element_values[-100:])),
        "batch_row_violation_rate_tail_mean":
            float(np.mean(violation_rate_values[-100:])),
        "maximum_log_excess_tail_max":
            float(np.max(maximum_excess_values[-100:])),
        "gradient_tail_mean":
            float(np.mean(gradient_array[-100:])),
        "loss_finite": bool(
            np.all(np.isfinite(total_array))
            and np.all(np.isfinite(diffusion_array))
            and np.all(np.isfinite(geometry_array))
        ),
        "gradient_finite":
            bool(np.all(np.isfinite(gradient_array))),
        "training_rows": int(condition.shape[0]),
        "training_steps": int(stageb_spec.train_steps),
    }
    return model, records


def one_step_audit(
    *,
    model: Any,
    condition: np.ndarray,
    target: np.ndarray,
    groups: Sequence[Any],
    condition_name: Sequence[Any],
    stageb_spec: stageb.DiagnosticSpec,
    upper_gate: staged3.OneSidedUpperContract,
    stage_d_contract: staged.SegmentGateContract,
    historical_geometry: stageb.GeometryContract,
    condition_standardizer: stageb.ArrayStandardizer,
    target_standardizer: stageb.ArrayStandardizer,
    noise_seed: int,
) -> Dict[str, Any]:
    predictions, prediction_sha = stagec.one_step_predictions(
        model=model,
        condition=condition,
        target=target,
        spec=stageb_spec,
        condition_standardizer=condition_standardizer,
        target_standardizer=target_standardizer,
        noise_seed=noise_seed,
    )
    records = {}
    for timestep, prediction in sorted(predictions.items()):
        decomposition = staged3.upper_only_decomposition(
            value=prediction,
            contract=upper_gate,
            stage_d_contract=stage_d_contract,
            historical_geometry=historical_geometry,
            groups=groups,
            condition_name=condition_name,
        )
        records[str(timestep)] = {
            "normalized_mse": stageb.normalized_mse(
                prediction,
                target,
                target_standardizer,
            ),
            "raw_rmse": stageb.rmse(prediction, target),
            "upper": staged3.public_decomposition(
                decomposition
            ),
            "prediction_sha256":
                sha256_array(prediction),
        }
    return {
        "prediction_sha256": prediction_sha,
        "timesteps": records,
    }


def train_control_audit(
    *,
    model: Any,
    condition: np.ndarray,
    target: np.ndarray,
    stageb_spec: stageb.DiagnosticSpec,
    condition_standardizer: stageb.ArrayStandardizer,
    target_standardizer: stageb.ArrayStandardizer,
) -> Dict[str, Any]:
    prediction = stagec.reproduce_train_control_prediction(
        model=model,
        train_condition=condition,
        train_target=target,
        spec=stageb_spec,
        condition_standardizer=condition_standardizer,
        target_standardizer=target_standardizer,
    )
    rows = prediction.shape[0]
    return {
        "rows": int(rows),
        "normalized_mse": stageb.normalized_mse(
            prediction,
            target[:rows],
            target_standardizer,
        ),
        "raw_rmse": stageb.rmse(
            prediction,
            target[:rows],
        ),
        "prediction_sha256": sha256_array(prediction),
    }


def selection_record(
    *,
    lambda_upper: float,
    training: Mapping[str, Any],
    train_control: Mapping[str, Any],
    one_step: Mapping[str, Any],
    control: Optional[Mapping[str, Any]],
    spec: UpperObjectiveSpec,
) -> Dict[str, Any]:
    timestep_records = one_step["timesteps"]
    t10_upper = float(
        timestep_records["10"]["upper"]["upper_segment"][
            "row_any_rate"
        ]
    )
    t25_upper = float(
        timestep_records["25"]["upper"]["upper_segment"][
            "row_any_rate"
        ]
    )
    t50_upper = float(
        timestep_records["50"]["upper"]["upper_segment"][
            "row_any_rate"
        ]
    )
    if control is None:
        ratios = {
            "train_control": 1.0,
            "t10_nmse": 1.0,
            "t25_nmse": 1.0,
            "t50_nmse": 1.0,
        }
    else:
        control_step = control["one_step"]["timesteps"]
        ratios = {
            "train_control": float(
                train_control["normalized_mse"]
                / max(
                    float(control["train_control"]["normalized_mse"]),
                    1.0e-12,
                )
            ),
            "t10_nmse": float(
                timestep_records["10"]["normalized_mse"]
                / max(
                    float(control_step["10"]["normalized_mse"]),
                    1.0e-12,
                )
            ),
            "t25_nmse": float(
                timestep_records["25"]["normalized_mse"]
                / max(
                    float(control_step["25"]["normalized_mse"]),
                    1.0e-12,
                )
            ),
            "t50_nmse": float(
                timestep_records["50"]["normalized_mse"]
                / max(
                    float(control_step["50"]["normalized_mse"]),
                    1.0e-12,
                )
            ),
        }
    gates = {
        "training_loss_finite":
            bool(training["loss_finite"]),
        "training_gradient_finite":
            bool(training["gradient_finite"]),
        "t10_upper_row":
            t10_upper >= spec.selection_t10_upper_row_min,
        "t25_upper_row":
            t25_upper >= spec.selection_t25_upper_row_min,
        "t50_upper_row":
            t50_upper >= spec.selection_t50_upper_row_min,
        "train_control_ratio":
            ratios["train_control"]
            <= spec.selection_train_control_ratio_max,
        "t10_nmse_ratio":
            ratios["t10_nmse"]
            <= spec.selection_nmse_ratio_max,
        "t25_nmse_ratio":
            ratios["t25_nmse"]
            <= spec.selection_nmse_ratio_max,
        "t50_nmse_ratio":
            ratios["t50_nmse"]
            <= spec.selection_nmse_ratio_max,
    }
    feasible = (
        float(lambda_upper) > 0.0
        and bool(all(gates.values()))
    )
    return {
        "lambda_upper": float(lambda_upper),
        "training": dict(training),
        "train_control": dict(train_control),
        "one_step": dict(one_step),
        "relative_to_control": ratios,
        "selection_gates": gates,
        "feasible_nonzero_configuration": feasible,
    }


def select_configuration(
    records: Sequence[Mapping[str, Any]],
) -> Optional[Dict[str, Any]]:
    feasible = [
        record
        for record in records
        if record["feasible_nonzero_configuration"]
    ]
    if not feasible:
        return None
    selected = sorted(
        feasible,
        key=lambda record: (
            float(record["lambda_upper"]),
            float(
                record["one_step"]["timesteps"]["10"][
                    "normalized_mse"
                ]
            ),
            str(record["training"]["final_model_sha256"]),
        ),
    )[0]
    return {
        "lambda_upper": float(selected["lambda_upper"]),
        "selection_rule": (
            "smallest nonzero lambda passing all pre-registered "
            "train-only holdout gates"
        ),
        "pilot_model_sha256":
            selected["training"]["final_model_sha256"],
        "selection_record_sha256":
            sha256_bytes(stable_json_bytes(selected)),
    }


def candidate_diversity(
    candidates: np.ndarray,
    standardizer: stageb.ArrayStandardizer,
) -> Dict[str, Any]:
    value = np.asarray(candidates, dtype=np.float32)
    if value.ndim != 4:
        raise ValueError("candidates must be [N,K,4,48]")
    normalized = standardizer.normalize(value)
    pair_values = []
    row_means = []
    near_duplicate = 0
    pair_count = 0
    for left in range(value.shape[1]):
        for right in range(left + 1, value.shape[1]):
            distance = np.mean(
                (
                    normalized[:, left].astype(np.float64)
                    - normalized[:, right].astype(np.float64)
                )
                ** 2,
                axis=(1, 2),
            )
            pair_values.append(distance)
            near_duplicate += int(np.sum(distance <= 1.0e-10))
            pair_count += int(distance.size)
    if pair_values:
        matrix = np.stack(pair_values, axis=1)
        row_means = np.mean(matrix, axis=1)
        flat = matrix.reshape(-1)
    else:
        matrix = np.empty((value.shape[0], 0), dtype=np.float64)
        row_means = np.zeros(value.shape[0], dtype=np.float64)
        flat = np.empty(0, dtype=np.float64)
    row_variance = np.mean(
        np.var(normalized.astype(np.float64), axis=1),
        axis=(1, 2),
    )
    return {
        "candidate_count": int(value.shape[1]),
        "pairwise_distance": _safe_stats(flat),
        "row_mean_pairwise_distance": _safe_stats(row_means),
        "row_candidate_variance": _safe_stats(row_variance),
        "near_duplicate_pair_rate": (
            float(near_duplicate / pair_count)
            if pair_count
            else 0.0
        ),
        "pairwise_distance_sha256": sha256_array(matrix),
    }


def reverse_audit(
    *,
    model: Any,
    condition: np.ndarray,
    target: np.ndarray,
    groups: Sequence[Any],
    condition_name: Sequence[Any],
    pair_key: Sequence[Any],
    stageb_spec: stageb.DiagnosticSpec,
    upper_gate: staged3.OneSidedUpperContract,
    stage_d_contract: staged.SegmentGateContract,
    historical_geometry: stageb.GeometryContract,
    condition_standardizer: stageb.ArrayStandardizer,
    target_standardizer: stageb.ArrayStandardizer,
    noise_seed: int,
    snapshot_timesteps: Sequence[int],
    branch_prefix_k: Sequence[int],
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> Dict[str, Any]:
    candidates, trace = staged.reverse_sample_with_raw_trace(
        model=model,
        condition=condition,
        spec=stageb_spec,
        condition_standardizer=condition_standardizer,
        target_standardizer=target_standardizer,
        noise_seed=noise_seed,
        snapshot_timesteps=snapshot_timesteps,
    )
    final_full = staged3.upper_only_decomposition(
        value=candidates,
        contract=upper_gate,
        stage_d_contract=stage_d_contract,
        historical_geometry=historical_geometry,
        groups=groups,
        condition_name=condition_name,
    )
    valid_mask = np.asarray(
        final_full["combined_valid_mask"],
        dtype=np.bool_,
    )
    branch_curve = staged3.branch_prefix_curve(
        pair_key=pair_key,
        condition_name=condition_name,
        target=target,
        candidates=candidates,
        valid_mask=valid_mask,
        prefix_k=branch_prefix_k,
    )
    k8 = branch_curve[str(branch_prefix_k[-1])]
    bootstrap = staged1.deterministic_eligible_bootstrap(
        k8,
        resamples=bootstrap_resamples,
        seed=bootstrap_seed,
    )
    trace_records = {}
    for timestep in snapshot_timesteps:
        item = trace[str(timestep)]
        trace_records[str(timestep)] = {
            "latent": staged3.public_decomposition(
                staged3.upper_only_decomposition(
                    value=item["latent"],
                    contract=upper_gate,
                    stage_d_contract=stage_d_contract,
                    historical_geometry=historical_geometry,
                    groups=groups,
                    condition_name=condition_name,
                )
            ),
            "predicted_x0": staged3.public_decomposition(
                staged3.upper_only_decomposition(
                    value=item["predicted_x0"],
                    contract=upper_gate,
                    stage_d_contract=stage_d_contract,
                    historical_geometry=historical_geometry,
                    groups=groups,
                    condition_name=condition_name,
                )
            ),
            "latent_sha256": sha256_array(item["latent"]),
            "predicted_x0_sha256":
                sha256_array(item["predicted_x0"]),
        }
    best_nmse, best_index = stageb.best_of_k_nmse(
        candidates,
        target,
        target_standardizer,
    )
    return {
        "candidate_sha256": sha256_array(candidates),
        "best_of_k_normalized_mse": best_nmse,
        "best_index_sha256": sha256_array(best_index),
        "diversity": candidate_diversity(
            candidates,
            target_standardizer,
        ),
        "final": staged3.public_decomposition(final_full),
        "trace": trace_records,
        "physical_branch": {
            "prefix_k_curve": branch_curve,
            "k8": k8,
            "eligible_bootstrap": bootstrap,
        },
    }


def classify_final(
    *,
    spec: UpperObjectiveSpec,
    selected: Optional[Mapping[str, Any]],
    baseline: Optional[Mapping[str, Any]],
    repaired: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    if selected is None:
        return {
            "root_cause":
                "phase314b_r257_stagea_no_train_only_upper_objective_configuration",
            "required_next_path":
                "CALIBRATE_STRONGER_OR_ALTERNATIVE_UPPER_OBJECTIVE_ON_TRAIN_ONLY_SPLIT",
            "primary_failure_locus": "train_only_objective_selection",
        }
    if baseline is None or repaired is None:
        raise ValueError("selected configuration requires final audits")

    baseline_control = float(
        baseline["train_control"]["normalized_mse"]
    )
    repaired_control = float(
        repaired["train_control"]["normalized_mse"]
    )
    control_ratio = repaired_control / max(
        baseline_control,
        1.0e-12,
    )
    one_step_ratios = {}
    for timestep in ("10", "25", "50"):
        one_step_ratios[timestep] = float(
            repaired["one_step"]["timesteps"][timestep][
                "normalized_mse"
            ]
            / max(
                baseline["one_step"]["timesteps"][timestep][
                    "normalized_mse"
                ],
                1.0e-12,
            )
        )
    t10_upper = float(
        repaired["one_step"]["timesteps"]["10"]["upper"][
            "upper_segment"
        ]["row_any_rate"]
    )
    reverse_upper = float(
        repaired["reverse"]["final"]["upper_segment"][
            "row_any_rate"
        ]
    )
    reverse_combined = float(
        repaired["reverse"]["final"]["combined"][
            "row_any_rate"
        ]
    )
    reverse_nmse_ratio = float(
        repaired["reverse"]["best_of_k_normalized_mse"]
        / max(
            baseline["reverse"]["best_of_k_normalized_mse"],
            1.0e-12,
        )
    )
    baseline_diversity = float(
        baseline["reverse"]["diversity"][
            "row_mean_pairwise_distance"
        ]["mean"]
    )
    repaired_diversity = float(
        repaired["reverse"]["diversity"][
            "row_mean_pairwise_distance"
        ]["mean"]
    )
    diversity_ratio = repaired_diversity / max(
        baseline_diversity,
        1.0e-12,
    )
    branch = repaired["reverse"]["physical_branch"]["k8"]
    eligible = float(branch["eligible_row_rate"])
    support = branch["support_rate_among_eligible"]
    support_value = float(support) if support is not None else 0.0

    if (
        repaired_control > spec.final_train_control_nmse_max
        or control_ratio > spec.final_train_control_ratio_max
    ):
        root = (
            "phase314b_r257_stagea_upper_objective_"
            "train_control_regression"
        )
        next_path = (
            "REDESIGN_UPPER_OBJECTIVE_WEIGHTING_TO_PRESERVE_"
            "TRAIN_CONTROL_RECONSTRUCTION"
        )
        locus = "train_control_tradeoff"
    elif t10_upper < spec.final_t10_upper_row_min:
        root = (
            "phase314b_r257_stagea_upper_objective_"
            "failed_on_frozen_probe_x0"
        )
        next_path = (
            "AUDIT_DIRECT_X0_GEOMETRY_GRADIENT_AND_INCREASE_"
            "TRAIN_ONLY_OBJECTIVE_CAPACITY"
        )
        locus = "x0_upper_expansion"
    elif max(one_step_ratios.values()) > (
        spec.final_one_step_nmse_ratio_max
    ):
        root = (
            "phase314b_r257_stagea_upper_objective_"
            "one_step_prediction_tradeoff_failed"
        )
        next_path = (
            "REDESIGN_LOCAL_UPPER_OBJECTIVE_WITH_SCALE_OR_"
            "TIMESTEP_WEIGHTING"
        )
        locus = "one_step_fidelity_tradeoff"
    elif reverse_upper < spec.final_reverse_upper_row_min:
        root = (
            "phase314b_r257_stagea_reverse_transport_failed_"
            "after_valid_x0_objective"
        )
        next_path = (
            "REPAIR_REVERSE_TRANSPORT_WITH_TRAINED_"
            "UPPER_VALID_X0_MODEL"
        )
        locus = "reverse_upper_transport"
    elif reverse_combined < spec.final_reverse_combined_row_min:
        root = (
            "phase314b_r257_stagea_nonsegment_physical_"
            "failure_after_upper_repair"
        )
        next_path = (
            "ATTRIBUTE_COORDINATE_OR_TOPOLOGY_FAILURE_AFTER_"
            "UPPER_OBJECTIVE_REPAIR"
        )
        locus = "nonsegment_physical_gate"
    elif reverse_nmse_ratio > spec.final_reverse_nmse_ratio_max:
        root = (
            "phase314b_r257_stagea_upper_objective_"
            "reverse_prediction_tradeoff_failed"
        )
        next_path = (
            "REDESIGN_UPPER_OBJECTIVE_TO_PRESERVE_REVERSE_"
            "BEST_OF_K_UTILITY"
        )
        locus = "reverse_fidelity_tradeoff"
    elif diversity_ratio < spec.final_diversity_ratio_min:
        root = (
            "phase314b_r257_stagea_upper_objective_"
            "candidate_diversity_collapsed"
        )
        next_path = (
            "ADD_DIVERSITY_PRESERVATION_OR_REDUCE_"
            "UPPER_OBJECTIVE_STRENGTH"
        )
        locus = "candidate_diversity"
    elif eligible < spec.final_branch_eligible_row_min:
        root = (
            "phase314b_r257_stagea_physical_candidate_"
            "eligibility_failed"
        )
        next_path = (
            "REPAIR_REMAINING_PHYSICAL_CANDIDATE_COVERAGE_"
            "BEFORE_BRANCH_AUDIT"
        )
        locus = "branch_eligibility"
    elif support_value < spec.final_branch_support_min:
        root = (
            "phase314b_r257_stagea_physical_valid_"
            "branch_transport_failed"
        )
        next_path = (
            "REPAIR_CONTACT_CONDITIONAL_BRANCH_TRANSPORT_"
            "WITH_UPPER_VALID_X0_MODEL"
        )
        locus = "branch_transport"
    else:
        root = (
            "phase314b_r257_stagea_upper_objective_"
            "and_physical_branch_supported"
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
        "train_control_nmse": repaired_control,
        "train_control_ratio": control_ratio,
        "one_step_nmse_ratio": one_step_ratios,
        "one_step_t10_upper_row_any": t10_upper,
        "reverse_upper_row_any": reverse_upper,
        "reverse_combined_row_any": reverse_combined,
        "reverse_nmse_ratio": reverse_nmse_ratio,
        "diversity_ratio": diversity_ratio,
        "branch_eligible_row_rate": eligible,
        "branch_support_among_eligible": support,
    }


def run_calibration(
    *,
    root: Path,
    spec: Optional[UpperObjectiveSpec] = None,
) -> Dict[str, Any]:
    active_spec = UpperObjectiveSpec() if spec is None else spec
    active_spec.validate()
    repository_root = Path(root).resolve()

    immutable = validate_immutable_inputs(repository_root)
    source_audit = source_logic_audit(repository_root)
    if not source_audit["all_confirmed"]:
        raise UpperObjectiveCalibrationError(
            "r2.5.7 source assumptions changed"
        )

    upper_gate_payload = immutable["staged3_contract"]
    upper_gate = load_upper_gate_contract(
        upper_gate_payload
    )
    objective_contract = build_objective_contract(
        upper_gate=upper_gate,
        upper_gate_payload=upper_gate_payload,
        spec=active_spec,
    )
    stage_d_contract, _stage_d_payload = (
        staged1.load_stage_d_gate(
            repository_root / staged3.STAGE_D_GATE
        )
    )

    stageb_spec = stageb.DiagnosticSpec()
    stageb_spec.validate()
    arrays = stageb.load_npz_strict(
        repository_root / stageb.STAGEA_TRAIN_VIEW,
        required_keys=stageb.REQUIRED_TRAIN_KEYS,
    )
    validation = stageb.validate_train_view(arrays)
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
    condition_name = np.asarray(
        arrays["condition_name"]
    ).astype(str)
    pair_key = np.asarray(arrays["pair_key"]).astype(str)

    stageb_train, frozen_probe, stageb_mapping = (
        stageb.deterministic_group_split(
            groups,
            folds=stageb_spec.group_folds,
            probe_fold=stageb_spec.probe_fold,
        )
    )
    objective_train, selection_holdout, selection_split = (
        deterministic_selection_split(
            groups,
            stageb_train,
            folds=active_spec.selection_group_folds,
            holdout_fold=active_spec.selection_holdout_fold,
        )
    )
    if np.any(frozen_probe & (objective_train | selection_holdout)):
        raise AssertionError("frozen probe crossed selection split")

    historical_geometry = stageb.fit_geometry_contract(
        target[stageb_train]
    )

    # Exact zero-weight equivalence on the full Stage-B training population.
    stageb.set_deterministic_runtime(stageb_spec.seed)
    frozen_model, frozen_training, _ = stageb.train_model(
        condition=condition[stageb_train],
        target=target[stageb_train],
        spec=stageb_spec,
        condition_standardizer=stageb.fit_standardizer(
            condition[stageb_train]
        ),
        target_standardizer=stageb.fit_standardizer(
            target[stageb_train]
        ),
    )
    for key in (
        "final_model_sha256",
        "final_optimizer_sha256",
        "loss_history_sha256",
        "gradient_history_sha256",
        "source_exposure_sha256",
    ):
        if frozen_training[key] != EXPECTED_FROZEN_IDENTITY[key]:
            raise UpperObjectiveCalibrationError(
                f"zero-weight baseline replay changed: {key}"
            )

    pilot_condition_standardizer = stageb.fit_standardizer(
        condition[objective_train]
    )
    pilot_target_standardizer = stageb.fit_standardizer(
        target[objective_train]
    )
    pilot_records: List[Dict[str, Any]] = []
    control_bundle: Optional[Dict[str, Any]] = None

    # Frozen probe is intentionally not indexed in this loop.
    for lambda_upper in active_spec.candidate_lambdas:
        stageb.set_deterministic_runtime(stageb_spec.seed)
        pilot_model, pilot_training = (
            train_model_with_upper_objective(
                condition=condition[objective_train],
                target=target[objective_train],
                stageb_spec=stageb_spec,
                objective_contract=objective_contract,
                lambda_upper=float(lambda_upper),
                condition_standardizer=
                    pilot_condition_standardizer,
                target_standardizer=
                    pilot_target_standardizer,
            )
        )
        pilot_control = train_control_audit(
            model=pilot_model,
            condition=condition[objective_train],
            target=target[objective_train],
            stageb_spec=stageb_spec,
            condition_standardizer=
                pilot_condition_standardizer,
            target_standardizer=
                pilot_target_standardizer,
        )
        pilot_one_step = one_step_audit(
            model=pilot_model,
            condition=condition[selection_holdout],
            target=target[selection_holdout],
            groups=groups[selection_holdout],
            condition_name=
                condition_name[selection_holdout],
            stageb_spec=stageb_spec,
            upper_gate=upper_gate,
            stage_d_contract=stage_d_contract,
            historical_geometry=historical_geometry,
            condition_standardizer=
                pilot_condition_standardizer,
            target_standardizer=
                pilot_target_standardizer,
            noise_seed=stageb_spec.seed + 4101,
        )
        record = selection_record(
            lambda_upper=float(lambda_upper),
            training=pilot_training,
            train_control=pilot_control,
            one_step=pilot_one_step,
            control=control_bundle,
            spec=active_spec,
        )
        if float(lambda_upper) == 0.0:
            control_bundle = {
                "training": pilot_training,
                "train_control": pilot_control,
                "one_step": pilot_one_step,
            }
            record = selection_record(
                lambda_upper=0.0,
                training=pilot_training,
                train_control=pilot_control,
                one_step=pilot_one_step,
                control=None,
                spec=active_spec,
            )
        pilot_records.append(record)
        del pilot_model

    if control_bundle is None:
        raise AssertionError("zero-weight pilot control is missing")
    selected = select_configuration(pilot_records)
    selection_projection = {
        "split": selection_split,
        "candidate_records": pilot_records,
        "selected_configuration": selected,
        "selection_uses_frozen_probe": False,
    }
    selection_sha = sha256_bytes(
        stable_json_bytes(selection_projection)
    )

    baseline_final = None
    repaired_final = None
    probe_accessed = False

    if selected is not None:
        probe_accessed = True
        full_condition_standardizer = stageb.fit_standardizer(
            condition[stageb_train]
        )
        full_target_standardizer = stageb.fit_standardizer(
            target[stageb_train]
        )

        # The frozen baseline was already replayed exactly before selection.
        baseline_control = train_control_audit(
            model=frozen_model,
            condition=condition[stageb_train],
            target=target[stageb_train],
            stageb_spec=stageb_spec,
            condition_standardizer=
                full_condition_standardizer,
            target_standardizer=
                full_target_standardizer,
        )
        if baseline_control["prediction_sha256"] != (
            EXPECTED_FROZEN_IDENTITY[
                "train_control_prediction_sha256"
            ]
        ):
            raise UpperObjectiveCalibrationError(
                "frozen baseline train-control changed"
            )
        baseline_one_step = one_step_audit(
            model=frozen_model,
            condition=condition[frozen_probe],
            target=target[frozen_probe],
            groups=groups[frozen_probe],
            condition_name=condition_name[frozen_probe],
            stageb_spec=stageb_spec,
            upper_gate=upper_gate,
            stage_d_contract=stage_d_contract,
            historical_geometry=historical_geometry,
            condition_standardizer=
                full_condition_standardizer,
            target_standardizer=
                full_target_standardizer,
            noise_seed=stageb_spec.seed + 2001,
        )
        if baseline_one_step["prediction_sha256"] != (
            EXPECTED_FROZEN_IDENTITY[
                "one_step_prediction_sha256"
            ]
        ):
            raise UpperObjectiveCalibrationError(
                "frozen baseline one-step changed"
            )
        baseline_reverse = reverse_audit(
            model=frozen_model,
            condition=condition[frozen_probe],
            target=target[frozen_probe],
            groups=groups[frozen_probe],
            condition_name=condition_name[frozen_probe],
            pair_key=pair_key[frozen_probe],
            stageb_spec=stageb_spec,
            upper_gate=upper_gate,
            stage_d_contract=stage_d_contract,
            historical_geometry=historical_geometry,
            condition_standardizer=
                full_condition_standardizer,
            target_standardizer=
                full_target_standardizer,
            noise_seed=stageb_spec.seed + 3001,
            snapshot_timesteps=
                active_spec.snapshot_timesteps,
            branch_prefix_k=active_spec.branch_prefix_k,
            bootstrap_resamples=
                active_spec.bootstrap_resamples,
            bootstrap_seed=active_spec.bootstrap_seed,
        )
        if baseline_reverse["candidate_sha256"] != (
            EXPECTED_FROZEN_IDENTITY[
                "reverse_candidate_sha256"
            ]
        ):
            raise UpperObjectiveCalibrationError(
                "frozen baseline reverse candidates changed"
            )
        baseline_final = {
            "train_control": baseline_control,
            "one_step": baseline_one_step,
            "reverse": baseline_reverse,
        }

        stageb.set_deterministic_runtime(stageb_spec.seed)
        repaired_model, repaired_training = (
            train_model_with_upper_objective(
                condition=condition[stageb_train],
                target=target[stageb_train],
                stageb_spec=stageb_spec,
                objective_contract=objective_contract,
                lambda_upper=float(
                    selected["lambda_upper"]
                ),
                condition_standardizer=
                    full_condition_standardizer,
                target_standardizer=
                    full_target_standardizer,
            )
        )
        repaired_control = train_control_audit(
            model=repaired_model,
            condition=condition[stageb_train],
            target=target[stageb_train],
            stageb_spec=stageb_spec,
            condition_standardizer=
                full_condition_standardizer,
            target_standardizer=
                full_target_standardizer,
        )
        repaired_one_step = one_step_audit(
            model=repaired_model,
            condition=condition[frozen_probe],
            target=target[frozen_probe],
            groups=groups[frozen_probe],
            condition_name=condition_name[frozen_probe],
            stageb_spec=stageb_spec,
            upper_gate=upper_gate,
            stage_d_contract=stage_d_contract,
            historical_geometry=historical_geometry,
            condition_standardizer=
                full_condition_standardizer,
            target_standardizer=
                full_target_standardizer,
            noise_seed=stageb_spec.seed + 2001,
        )
        repaired_reverse = reverse_audit(
            model=repaired_model,
            condition=condition[frozen_probe],
            target=target[frozen_probe],
            groups=groups[frozen_probe],
            condition_name=condition_name[frozen_probe],
            pair_key=pair_key[frozen_probe],
            stageb_spec=stageb_spec,
            upper_gate=upper_gate,
            stage_d_contract=stage_d_contract,
            historical_geometry=historical_geometry,
            condition_standardizer=
                full_condition_standardizer,
            target_standardizer=
                full_target_standardizer,
            noise_seed=stageb_spec.seed + 3001,
            snapshot_timesteps=
                active_spec.snapshot_timesteps,
            branch_prefix_k=active_spec.branch_prefix_k,
            bootstrap_resamples=
                active_spec.bootstrap_resamples,
            bootstrap_seed=active_spec.bootstrap_seed,
        )
        repaired_final = {
            "training": repaired_training,
            "train_control": repaired_control,
            "one_step": repaired_one_step,
            "reverse": repaired_reverse,
        }

    classification = classify_final(
        spec=active_spec,
        selected=selected,
        baseline=baseline_final,
        repaired=repaired_final,
    )
    objective_payload = objective_contract.to_dict()
    objective_payload["contract_sha256"] = sha256_bytes(
        stable_json_bytes(objective_payload)
    )

    return {
        "phase": PHASE,
        "phase_id": PHASE_ID,
        "schema": "phase314b_r257_stagea_worker_result_v1",
        "verdict": "PASS",
        "scientific_status": "BLOCKED",
        "root_cause": classification["root_cause"],
        "required_next_path":
            classification["required_next_path"],
        "objective_spec": asdict(active_spec),
        "frozen_stageb_spec": asdict(stageb_spec),
        "immutable_inputs": {
            "base_commit": immutable["base_commit"],
            "base_file_sha256":
                immutable["base_file_sha256"],
            "frozen_identity": immutable["frozen_identity"],
        },
        "source_logic_audit": source_audit,
        "train_view_validation": validation,
        "split": {
            "stageb_training_rows": int(np.sum(stageb_train)),
            "frozen_probe_rows": int(np.sum(frozen_probe)),
            "stageb_group_count": len(stageb_mapping),
            **selection_split,
        },
        "objective_contract": objective_payload,
        "zero_weight_full_replay": {
            "identity": {
                "final_model_sha256":
                    frozen_training["final_model_sha256"],
                "final_optimizer_sha256":
                    frozen_training["final_optimizer_sha256"],
                "loss_history_sha256":
                    frozen_training["loss_history_sha256"],
                "gradient_history_sha256":
                    frozen_training["gradient_history_sha256"],
                "source_exposure_sha256":
                    frozen_training["source_exposure_sha256"],
            },
            "matches_r256_frozen_identity": True,
        },
        "train_only_selection": {
            **selection_projection,
            "selection_sha256": selection_sha,
            "probe_accessed_during_selection": False,
        },
        "final_frozen_probe_audit": {
            "performed": selected is not None,
            "probe_accessed_after_selection": probe_accessed,
            "baseline": baseline_final,
            "repaired": repaired_final,
        },
        "classification": classification,
        "selected_configuration": selected,
        "train_only_recommendation": selected,
        "model_objective_changed": True,
        "scheduler_changed": False,
        "split_leakage_detected": False,
        "normalization_contract_changed": False,
        "candidate_count_changed": False,
        "upper_gate_changed": False,
        "lower_score_used_for_validity": False,
        "formal_pilot_run": False,
        "checkpoint_saved": False,
        "weights_persisted": False,
        "prediction_tensor_persisted": False,
        "candidate_tensor_persisted": False,
        "npz_saved": False,
        "cache_saved": False,
        "formal_diffusion_training": False,
        "formal_reverse_sampling": False,
        "formal_idm_training": False,
        "action_diverse_data_collection": False,
        "candidate_execution": False,
        "phase4": False,
        "cps": False,
    }


def identity_projection(
    result: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "root_cause": result["root_cause"],
        "required_next_path": result["required_next_path"],
        "split": result["split"],
        "objective_contract": result["objective_contract"],
        "zero_weight_full_replay":
            result["zero_weight_full_replay"],
        "train_only_selection":
            result["train_only_selection"],
        "final_frozen_probe_audit":
            result["final_frozen_probe_audit"],
        "classification": result["classification"],
        "selected_configuration":
            result["selected_configuration"],
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
        "objective_contract_exact": (
            left["objective_contract"]
            == right["objective_contract"]
        ),
        "selection_exact": (
            left["train_only_selection"]
            == right["train_only_selection"]
        ),
        "final_audit_exact": (
            left["final_frozen_probe_audit"]
            == right["final_frozen_probe_audit"]
        ),
        "classification_exact": (
            left["classification"]
            == right["classification"]
        ),
    }
