"""Phase3.14b-r2.5.7 Stage C top-k quadratic objective calibration.

Stage B proved that the Stage-A row-max + element Huber objective was connected
to the direct-x0 model and was not underpowered at lambda=0.1, but 96.8% of
positive violations were in the Huber linear region. Every nonzero Stage-A
candidate worsened the train-only holdout continuous upper excess.

This stage replaces the saturated Huber objective with a train-only top-k
quadratic objective.  It does not manually reuse the Stage-A lambda scale.
For each top-k variant, lambda is calibrated on the frozen Stage-B diagnostic
batch at the common initial model so that:

    lambda * ||g_geometry|| / ||g_diffusion||

targets 0.25, 0.50, or 1.00.

Candidate selection uses only the same 638-row objective-training population
and the same 236-row grouped selection holdout used by Stage A.  The frozen
126-row probe, reverse sampler, formal pilot, IDM, environment and CPS remain
closed.  A selected configuration is only a train-only recommendation for the
next stage; no full-874-row repaired model is trained here.

No checkpoint, weights, prediction tensor, candidate tensor, NPZ, cache,
image, or video is persisted.
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
from ccda_phase3 import phase314b_r256_staged3_resume1_upper_gate_freeze as staged3
from ccda_phase3 import phase314b_r257_stagea_upper_objective as stagea
from ccda_phase3 import phase314b_r257_stageb_mechanism_audit as stageb_mechanism

PHASE = "Phase3.14b-r2.5.7 Stage C"
PHASE_ID = "phase314b_r257_stagec"
BASE_EVIDENCE_COMMIT = "37b20e24afe21d403bb4fcd48ecf0c1a78a77141"
EXPECTED_SUBMODULE_COMMIT = "633a88752445cf5d6776ed374fdbbdb35f93050c"

STAGEB_MECHANISM_SOURCE = (
    "ccda_phase3/phase314b_r257_stageb_mechanism_audit.py"
)
STAGEB_MECHANISM_TEST = (
    "tests/test_phase3_14b_r257_stageb_mechanism_audit.py"
)
STAGEB_MECHANISM_CONTRACT = (
    "reports/phase3_14b_r257_stageb_mechanism_contract.json"
)
STAGEB_MECHANISM_TEST_GATE = (
    "reports/phase3_14b_r257_stageb_test_gate_summary.json"
)
STAGEB_MECHANISM_WORKER = (
    "reports/phase3_14b_r257_stageb_worker_evidence.json"
)
STAGEB_MECHANISM_SUMMARY = (
    "reports/phase3_14b_r257_stageb_summary.json"
)
STAGEB_MECHANISM_REPORT = (
    "reports/phase3_14b_r257_stageb_report.md"
)

BASE_BOUND_FILES = (
    STAGEB_MECHANISM_SOURCE,
    STAGEB_MECHANISM_TEST,
    STAGEB_MECHANISM_CONTRACT,
    STAGEB_MECHANISM_TEST_GATE,
    STAGEB_MECHANISM_WORKER,
    STAGEB_MECHANISM_SUMMARY,
    STAGEB_MECHANISM_REPORT,
)

EXPECTED_STAGEB_WORKER_SHA256 = (
    "17bfea56e69b46946af40ffc590ea83b42fd360b598d99649aec9339b9ab274c"
)
EXPECTED_STAGEB_MECHANISM_CONTRACT_SHA256 = (
    "829082531e2b52298bf2bd2a5b5ccb32509c5d4b8bd430c837aa610a95a4ec22"
)
EXPECTED_STAGEB_MECHANISM_CONTRACT_FILE_SHA256 = (
    "fb5eddb8fbdae36fee095ee204d46036039b2c8188849f9521a1cec835188ae2"
)

CONTROL_CANDIDATE_ID = "control_diffusion_only"


class TopKCalibrationError(RuntimeError):
    """Raised when an immutable identity or selection boundary fails."""


@dataclass(frozen=True)
class TopKCalibrationSpec:
    """Pre-registered top-k quadratic calibration contract."""

    top_k_values: Tuple[int, ...] = (8, 16)
    target_gradient_ratios: Tuple[float, ...] = (
        0.25,
        0.50,
        1.00,
    )

    completed_step_checkpoints: Tuple[int, ...] = (
        1,
        100,
        500,
        2000,
        8000,
    )
    holdout_noise_seed_offset: int = 4101

    t10_mean_excess_reduction_min: float = 0.20
    t25_mean_excess_reduction_min: float = 0.15
    t50_mean_excess_reduction_min: float = 0.10
    top8_mean_excess_reduction_min: float = 0.0
    p95_excess_ratio_max: float = 1.0
    positive_segment_count_ratio_max: float = 1.0

    train_control_nmse_ratio_max: float = 1.20
    one_step_nmse_ratio_max: float = 1.20

    binary_t10_upper_row_min: float = 0.75
    binary_t25_upper_row_min: float = 0.60
    binary_t50_upper_row_min: float = 0.50

    gradient_clip_frequency_max: float = 0.25
    largest_row_contribution_max: float = 0.25
    top_five_percent_contribution_max: float = 0.60
    severe_gradient_conflict_cosine: float = -0.10
    severe_gradient_conflict_fraction: float = 0.60

    row_loss_epsilon: float = 1.0e-12

    def validate(self) -> None:
        if tuple(sorted(set(self.top_k_values))) != self.top_k_values:
            raise ValueError("top-k values must be ordered and unique")
        if not self.top_k_values:
            raise ValueError("top-k values are empty")
        maximum_positions = (
            stageb.FUTURE_STEPS * (stageb.BEADS - 1)
        )
        if any(
            int(value) <= 0 or int(value) > maximum_positions
            for value in self.top_k_values
        ):
            raise ValueError("top-k value is outside ordered geometry")
        if tuple(sorted(set(self.target_gradient_ratios))) != (
            self.target_gradient_ratios
        ):
            raise ValueError(
                "target gradient ratios must be ordered and unique"
            )
        if not self.target_gradient_ratios:
            raise ValueError("target gradient ratios are empty")
        if any(
            not 0.0 < float(value) <= 1.0
            for value in self.target_gradient_ratios
        ):
            raise ValueError(
                "target gradient ratio is outside (0,1]"
            )
        if self.completed_step_checkpoints[-1] != (
            stageb.DiagnosticSpec().train_steps
        ):
            raise ValueError(
                "final checkpoint must equal frozen training steps"
            )
        if tuple(sorted(set(self.completed_step_checkpoints))) != (
            self.completed_step_checkpoints
        ):
            raise ValueError(
                "completed-step checkpoints must be ordered"
            )
        if self.completed_step_checkpoints[0] <= 0:
            raise ValueError(
                "checkpoint must follow at least one update"
            )
        for value in (
            self.t10_mean_excess_reduction_min,
            self.t25_mean_excess_reduction_min,
            self.t50_mean_excess_reduction_min,
            self.p95_excess_ratio_max,
            self.positive_segment_count_ratio_max,
            self.train_control_nmse_ratio_max,
            self.one_step_nmse_ratio_max,
            self.binary_t10_upper_row_min,
            self.binary_t25_upper_row_min,
            self.binary_t50_upper_row_min,
            self.gradient_clip_frequency_max,
            self.largest_row_contribution_max,
            self.top_five_percent_contribution_max,
        ):
            if float(value) <= 0.0:
                raise ValueError(
                    "registered decision threshold is not positive"
                )
        if self.top8_mean_excess_reduction_min < 0.0:
            raise ValueError(
                "top-8 reduction threshold is negative"
            )
        if not -1.0 <= self.severe_gradient_conflict_cosine <= 1.0:
            raise ValueError(
                "gradient conflict cosine is invalid"
            )
        if not 0.0 <= self.severe_gradient_conflict_fraction <= 1.0:
            raise ValueError(
                "gradient conflict fraction is invalid"
            )
        if self.row_loss_epsilon <= 0.0:
            raise ValueError("row-loss epsilon must be positive")


@dataclass(frozen=True)
class CalibratedCandidate:
    candidate_id: str
    top_k: int
    target_gradient_ratio: float
    lambda_upper: float
    initial_model_sha256: str
    diffusion_gradient_norm: float
    geometry_gradient_norm: float
    geometry_gradient_cosine: float
    geometry_gradient_conflict_fraction: float

    def validate(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate id is empty")
        if self.top_k <= 0:
            raise ValueError("candidate top-k is invalid")
        if not 0.0 < self.target_gradient_ratio <= 1.0:
            raise ValueError("candidate target ratio is invalid")
        if not np.isfinite(self.lambda_upper):
            raise ValueError("candidate lambda is non-finite")
        if self.lambda_upper <= 0.0:
            raise ValueError("candidate lambda is not positive")
        if len(self.initial_model_sha256) != 64:
            raise ValueError("initial model SHA is malformed")
        if self.diffusion_gradient_norm <= 0.0:
            raise ValueError(
                "diffusion gradient norm is not positive"
            )
        if self.geometry_gradient_norm <= 0.0:
            raise ValueError(
                "geometry gradient norm is not positive"
            )

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return asdict(self)


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
    raise TypeError(
        f"unsupported JSON value: {type(value)!r}"
    )


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
    temporary = target.parent / (
        f".{target.name}.{os.getpid()}.tmp"
    )
    if temporary.exists():
        temporary.unlink()
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, target)
        directory_fd = os.open(
            str(target.parent),
            os.O_RDONLY,
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(
        Path(path).read_text(encoding="utf-8")
    )
    if not isinstance(value, dict):
        raise TopKCalibrationError(
            f"JSON root is not an object: {path}"
        )
    return value


def safe_stats(value: np.ndarray) -> Dict[str, float]:
    return stageb_mechanism.safe_stats(value)


def assert_base_file_bound(
    root: Path,
    relative: str,
) -> str:
    path = Path(root) / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    committed = subprocess.check_output(
        [
            "git",
            "show",
            f"{BASE_EVIDENCE_COMMIT}:{relative}",
        ],
        cwd=str(root),
    )
    observed = path.read_bytes()
    if committed != observed:
        raise TopKCalibrationError(
            f"Stage-B bound file changed: {relative}"
        )
    return sha256_bytes(observed)


def worker_identity_sha(
    worker: Mapping[str, Any],
) -> str:
    comparison = worker.get("comparison")
    if not isinstance(comparison, Mapping):
        raise TopKCalibrationError(
            "Stage-B worker comparison is missing"
        )
    left = comparison.get("left_sha256")
    right = comparison.get("right_sha256")
    if left != right or not isinstance(left, str):
        raise TopKCalibrationError(
            "Stage-B worker identity hashes differ"
        )
    return left


def validate_immutable_inputs(
    root: Path,
) -> Dict[str, Any]:
    repository_root = Path(root).resolve()
    bound = {
        relative: assert_base_file_bound(
            repository_root,
            relative,
        )
        for relative in BASE_BOUND_FILES
    }
    summary = load_json(
        repository_root / STAGEB_MECHANISM_SUMMARY
    )
    if summary.get("verdict") != "PASS":
        raise TopKCalibrationError(
            "Stage-B execution verdict is not PASS"
        )
    if summary.get("scientific_status") != "BLOCKED":
        raise TopKCalibrationError(
            "Stage-B scientific status changed"
        )
    if summary.get("root_cause") != (
        "phase314b_r257_stageb_upper_objective_"
        "huber_linear_saturation"
    ):
        raise TopKCalibrationError(
            "Stage-B root cause changed"
        )
    if summary.get("required_next_path") != (
        "CALIBRATE_TOPK_QUADRATIC_OR_SOFTPLUS_"
        "UPPER_OBJECTIVE_ON_TRAIN_ONLY_SPLIT"
    ):
        raise TopKCalibrationError(
            "Stage-B required next path changed"
        )
    if summary.get("selected_configuration") is not None:
        raise TopKCalibrationError(
            "Stage-B unexpectedly selected a configuration"
        )
    if summary.get("train_only_recommendation") is not None:
        raise TopKCalibrationError(
            "Stage-B unexpectedly selected a recommendation"
        )
    if summary.get("frozen_probe_accessed") is not False:
        raise TopKCalibrationError(
            "Stage-B accessed frozen probe"
        )
    if summary.get("new_hyperparameter_candidate_run") is not False:
        raise TopKCalibrationError(
            "Stage-B unexpectedly ran a new hyperparameter"
        )
    if summary.get("new_objective_variant_run") is not False:
        raise TopKCalibrationError(
            "Stage-B unexpectedly ran a new objective"
        )
    if summary.get("reverse_sampling_run") is not False:
        raise TopKCalibrationError(
            "Stage-B unexpectedly ran reverse sampling"
        )

    worker = load_json(
        repository_root / STAGEB_MECHANISM_WORKER
    )
    if worker.get("workers_exact") is not True:
        raise TopKCalibrationError(
            "Stage-B workers were not exact"
        )
    observed_worker_sha = worker_identity_sha(worker)
    if observed_worker_sha != EXPECTED_STAGEB_WORKER_SHA256:
        raise TopKCalibrationError(
            "Stage-B worker identity SHA changed"
        )
    worker_result = worker.get("worker_result")
    if not isinstance(worker_result, Mapping):
        raise TopKCalibrationError(
            "Stage-B worker result is missing"
        )
    mechanism_contract = worker_result.get(
        "mechanism_contract"
    )
    if not isinstance(mechanism_contract, Mapping):
        raise TopKCalibrationError(
            "Stage-B mechanism contract is missing"
        )
    if mechanism_contract.get("contract_sha256") != (
        EXPECTED_STAGEB_MECHANISM_CONTRACT_SHA256
    ):
        raise TopKCalibrationError(
            "Stage-B internal contract SHA changed"
        )
    if worker_result.get("frozen_probe_accessed") is not False:
        raise TopKCalibrationError(
            "Stage-B worker accessed frozen probe"
        )
    if worker_result.get("selected_configuration") is not None:
        raise TopKCalibrationError(
            "Stage-B worker selected a configuration"
        )
    if worker_result.get("train_only_recommendation") is not None:
        raise TopKCalibrationError(
            "Stage-B worker selected a recommendation"
        )

    contract_path = (
        repository_root / STAGEB_MECHANISM_CONTRACT
    )
    if sha256_file(contract_path) != (
        EXPECTED_STAGEB_MECHANISM_CONTRACT_FILE_SHA256
    ):
        raise TopKCalibrationError(
            "Stage-B mechanism contract file SHA changed"
        )
    contract = load_json(contract_path)
    if contract.get("contract_sha256") != (
        EXPECTED_STAGEB_MECHANISM_CONTRACT_SHA256
    ):
        raise TopKCalibrationError(
            "Stage-B contract payload changed"
        )
    if contract.get("uses_frozen_probe") is not False:
        raise TopKCalibrationError(
            "Stage-B contract used frozen probe"
        )
    if contract.get("changes_training_objective") is not False:
        raise TopKCalibrationError(
            "Stage-B contract changed training objective"
        )

    upstream = stageb_mechanism.validate_immutable_inputs(
        repository_root
    )
    return {
        "base_commit": BASE_EVIDENCE_COMMIT,
        "base_file_sha256": bound,
        "stageb_summary": summary,
        "stageb_worker": worker,
        "stageb_worker_result": worker_result,
        "stageb_contract": contract,
        "worker_identity_sha256": observed_worker_sha,
        "upstream_immutable": upstream,
    }


def source_logic_audit(root: Path) -> Dict[str, Any]:
    repository_root = Path(root).resolve()
    stagea_text = (
        repository_root / stageb_mechanism.STAGEA_SOURCE
    ).read_text(encoding="utf-8")
    stageb_text = (
        repository_root / STAGEB_MECHANISM_SOURCE
    ).read_text(encoding="utf-8")
    checks = {
        "stagea_direct_x0_objective": (
            "predicted_x0_z" in stagea_text
            and "target_standardizer" in stagea_text
        ),
        "stagea_uses_frozen_allowed_log": (
            "allowed_upper_log_length" in stagea_text
        ),
        "stageb_confirms_huber_saturation": (
            '"huber_linear_saturation"' in stageb_text
            and "huber_linear_fraction" in stageb_text
        ),
        "stageb_continuous_profile_available": (
            "def upper_violation_profile(" in stageb_text
        ),
        "stageb_gradient_metrics_available": (
            "def _gradient_pair_metrics(" in stageb_text
        ),
        "stageb_fixed_batch_available": (
            "def fixed_diagnostic_batch(" in stageb_text
        ),
        "stageb_model_has_no_dropout_or_batchnorm": (
            '"model_has_no_dropout"' in stageb_text
            and '"model_has_no_batchnorm"' in stageb_text
        ),
    }
    return {
        "checks": checks,
        "all_confirmed": bool(all(checks.values())),
        "source_sha256": {
            stageb_mechanism.STAGEA_SOURCE:
                sha256_file(
                    repository_root
                    / stageb_mechanism.STAGEA_SOURCE
                ),
            STAGEB_MECHANISM_SOURCE:
                sha256_file(
                    repository_root
                    / STAGEB_MECHANISM_SOURCE
                ),
        },
        "softplus_selected": False,
        "reason_softplus_not_primary": (
            "standard softplus is asymptotically linear for "
            "large positive excess and does not directly remove "
            "the confirmed Huber linear-saturation mechanism"
        ),
    }


def topk_quadratic_terms_torch(
    predicted_x0_z: Any,
    *,
    target_standardizer: stageb.ArrayStandardizer,
    objective_contract: stagea.UpperObjectiveContract,
    top_k: int,
) -> Dict[str, Any]:
    torch, _ = stageb._torch_imports()
    excess_record = stageb_mechanism.upper_excess_torch(
        predicted_x0_z,
        target_standardizer=target_standardizer,
        contract=objective_contract,
    )
    excess = excess_record["excess"]
    flattened = excess.reshape(excess.shape[0], -1)
    actual_k = min(int(top_k), int(flattened.shape[1]))
    selected, indices = torch.topk(
        flattened,
        k=actual_k,
        dim=1,
        largest=True,
        sorted=True,
    )
    row_loss = torch.mean(selected * selected, dim=1)
    total = torch.mean(row_loss)
    return {
        "total": total,
        "row_loss": row_loss,
        "selected_excess": selected,
        "selected_indices": indices,
        "maximum_excess": torch.max(excess),
        "positive_element_rate":
            torch.mean((excess > 0.0).to(predicted_x0_z.dtype)),
        "row_violation_rate":
            torch.mean(
                (excess_record["row_max"] > 0.0)
                .to(predicted_x0_z.dtype)
            ),
        "excess": excess,
    }


def row_contribution_profile(
    row_loss: np.ndarray,
    *,
    epsilon: float,
) -> Dict[str, Any]:
    values = np.asarray(
        row_loss,
        dtype=np.float64,
    ).reshape(-1)
    if values.size == 0:
        raise ValueError("row-loss population is empty")
    if not np.all(np.isfinite(values)):
        raise ValueError("row loss contains NaN or Inf")
    nonnegative = np.maximum(values, 0.0)
    total = float(np.sum(nonnegative))
    ordered = np.sort(nonnegative)[::-1]
    count_one = max(
        1,
        int(math.ceil(values.size * 0.01)),
    )
    count_five = max(
        1,
        int(math.ceil(values.size * 0.05)),
    )
    denominator = max(total, float(epsilon))
    return {
        "row_loss": safe_stats(nonnegative),
        "row_loss_sha256":
            sha256_array(nonnegative.astype(np.float64)),
        "largest_row_contribution_fraction":
            float(ordered[0] / denominator),
        "top_one_percent_contribution_fraction":
            float(np.sum(ordered[:count_one]) / denominator),
        "top_five_percent_contribution_fraction":
            float(np.sum(ordered[:count_five]) / denominator),
        "total_row_loss": total,
        "positive_row_fraction":
            float(np.mean(nonnegative > 0.0)),
    }


def selected_position_profile(
    indices: np.ndarray,
) -> Dict[str, Any]:
    value = np.asarray(indices, dtype=np.int64)
    if value.ndim != 2:
        raise ValueError(
            "selected top-k indices must be [B,K]"
        )
    flattened = value.reshape(-1)
    unique, counts = np.unique(
        flattened,
        return_counts=True,
    )
    total = int(flattened.size)
    records = []
    horizon_counts = np.zeros(
        stageb.FUTURE_STEPS,
        dtype=np.int64,
    )
    segment_counts = np.zeros(
        stageb.BEADS - 1,
        dtype=np.int64,
    )
    for flat, count in zip(
        unique.tolist(),
        counts.tolist(),
    ):
        horizon, segment = np.unravel_index(
            int(flat),
            (
                stageb.FUTURE_STEPS,
                stageb.BEADS - 1,
            ),
        )
        horizon_counts[horizon] += int(count)
        segment_counts[segment] += int(count)
        records.append(
            {
                "horizon": int(horizon),
                "segment_index": int(segment),
                "count": int(count),
                "fraction": (
                    float(count / total)
                    if total
                    else 0.0
                ),
            }
        )
    records.sort(
        key=lambda item: (
            -item["count"],
            item["horizon"],
            item["segment_index"],
        )
    )
    return {
        "selected_index_sha256": sha256_array(value),
        "top_positions": records[:32],
        "top_position_fraction": (
            records[0]["fraction"]
            if records
            else 0.0
        ),
        "horizon_selection_fraction": (
            horizon_counts.astype(np.float64)
            / max(total, 1)
        ).tolist(),
        "segment_selection_fraction": (
            segment_counts.astype(np.float64)
            / max(total, 1)
        ).tolist(),
    }


def fixed_batch_losses_and_gradients(
    *,
    model: Any,
    diagnostic_batch: Mapping[str, Any],
    target_standardizer: stageb.ArrayStandardizer,
    objective_contract: stagea.UpperObjectiveContract,
    top_k: int,
    lambda_upper: float,
    row_loss_epsilon: float,
) -> Dict[str, Any]:
    torch, _ = stageb._torch_imports()
    device = torch.device("cuda:0")
    noisy = torch.as_tensor(
        diagnostic_batch["noisy_z"],
        dtype=torch.float32,
        device=device,
    )
    timestep = torch.as_tensor(
        diagnostic_batch["timesteps"],
        dtype=torch.long,
        device=device,
    )
    condition = torch.as_tensor(
        diagnostic_batch["condition_z"],
        dtype=torch.float32,
        device=device,
    )
    clean = torch.as_tensor(
        diagnostic_batch["target_z"],
        dtype=torch.float32,
        device=device,
    )
    named_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    parameters = [
        parameter
        for _name, parameter in named_parameters
    ]

    predicted_diffusion = model(
        noisy,
        timestep,
        condition,
    )
    diffusion_loss = torch.mean(
        (predicted_diffusion - clean) ** 2
    )
    diffusion_gradients = torch.autograd.grad(
        diffusion_loss,
        parameters,
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )

    predicted_geometry = model(
        noisy,
        timestep,
        condition,
    )
    geometry_terms = topk_quadratic_terms_torch(
        predicted_geometry,
        target_standardizer=target_standardizer,
        objective_contract=objective_contract,
        top_k=int(top_k),
    )
    geometry_gradients = torch.autograd.grad(
        geometry_terms["total"],
        parameters,
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )
    gradient = stageb_mechanism._gradient_pair_metrics(
        named_parameters,
        diffusion_gradients,
        geometry_gradients,
        lambda_upper=float(lambda_upper),
    )
    row_loss = (
        geometry_terms["row_loss"]
        .detach()
        .cpu()
        .numpy()
    )
    selected_indices = (
        geometry_terms["selected_indices"]
        .detach()
        .cpu()
        .numpy()
    )
    selected_excess = (
        geometry_terms["selected_excess"]
        .detach()
        .cpu()
        .numpy()
    )
    model.zero_grad(set_to_none=True)
    return {
        "diffusion_loss":
            float(diffusion_loss.detach().cpu()),
        "geometry_loss":
            float(geometry_terms["total"].detach().cpu()),
        "gradient": gradient,
        "row_contribution": row_contribution_profile(
            row_loss,
            epsilon=float(row_loss_epsilon),
        ),
        "selected_positions":
            selected_position_profile(selected_indices),
        "selected_excess":
            safe_stats(selected_excess),
        "selected_excess_sha256":
            sha256_array(selected_excess),
        "predicted_x0_sha256":
            sha256_array(
                predicted_geometry.detach().cpu().numpy()
            ),
    }


def calibrate_candidates(
    *,
    stageb_spec: stageb.DiagnosticSpec,
    diagnostic_batch: Mapping[str, Any],
    target_standardizer: stageb.ArrayStandardizer,
    objective_contract: stagea.UpperObjectiveContract,
    spec: TopKCalibrationSpec,
) -> Tuple[List[CalibratedCandidate], Dict[str, Any]]:
    stageb.set_deterministic_runtime(stageb_spec.seed)
    model = stageb.make_model(stageb_spec).to("cuda:0")
    initial_model_sha = stageb.tensor_state_sha256(model)
    raw_records: MutableMapping[str, Any] = {}
    candidates = []
    for top_k in spec.top_k_values:
        raw = fixed_batch_losses_and_gradients(
            model=model,
            diagnostic_batch=diagnostic_batch,
            target_standardizer=target_standardizer,
            objective_contract=objective_contract,
            top_k=int(top_k),
            lambda_upper=1.0,
            row_loss_epsilon=spec.row_loss_epsilon,
        )
        diffusion_norm = float(
            raw["gradient"]["diffusion_norm"]
        )
        geometry_norm = float(
            raw["gradient"]["geometry_norm"]
        )
        if (
            not np.isfinite(diffusion_norm)
            or not np.isfinite(geometry_norm)
            or diffusion_norm <= 0.0
            or geometry_norm <= 0.0
        ):
            raise TopKCalibrationError(
                f"invalid initial gradient norms for k={top_k}"
            )
        raw_records[str(top_k)] = raw
        for target_ratio in spec.target_gradient_ratios:
            lambda_upper = (
                float(target_ratio)
                * diffusion_norm
                / geometry_norm
            )
            candidate_id = (
                f"topk{k_token(int(top_k))}_"
                f"r{ratio_token(float(target_ratio))}"
            )
            candidate = CalibratedCandidate(
                candidate_id=candidate_id,
                top_k=int(top_k),
                target_gradient_ratio=
                    float(target_ratio),
                lambda_upper=float(lambda_upper),
                initial_model_sha256=initial_model_sha,
                diffusion_gradient_norm=diffusion_norm,
                geometry_gradient_norm=geometry_norm,
                geometry_gradient_cosine=float(
                    raw["gradient"]["cosine"]
                ),
                geometry_gradient_conflict_fraction=float(
                    raw["gradient"][
                        "conflict_fraction_among_active"
                    ]
                ),
            )
            candidate.validate()
            candidates.append(candidate)
    del model
    return candidates, {
        "initial_model_sha256": initial_model_sha,
        "raw_k_records": raw_records,
        "candidate_count": len(candidates),
        "candidate_order": [
            candidate.candidate_id
            for candidate in candidates
        ],
        "calibration_sha256": sha256_bytes(
            stable_json_bytes(
                {
                    "initial_model_sha256":
                        initial_model_sha,
                    "candidates": [
                        candidate.to_dict()
                        for candidate in candidates
                    ],
                    "raw_k_records": raw_records,
                }
            )
        ),
    }


def k_token(value: int) -> str:
    return str(int(value))


def ratio_token(value: float) -> str:
    return (
        f"{float(value):.2f}"
        .replace(".", "p")
        .replace("-", "m")
    )


def compatible_training_record(
    record: Mapping[str, Any],
) -> Dict[str, Any]:
    keys = (
        "lambda_upper",
        "initial_model_sha256",
        "final_model_sha256",
        "initial_optimizer_sha256",
        "final_optimizer_sha256",
        "total_loss_history_sha256",
        "diffusion_loss_history_sha256",
        "geometry_loss_history_sha256",
        "gradient_history_sha256",
        "source_exposure_sha256",
        "total_loss_first",
        "total_loss_final",
        "total_loss_tail_mean",
        "diffusion_loss_tail_mean",
        "geometry_loss_tail_mean",
        "geometry_row_loss_tail_mean",
        "geometry_element_loss_tail_mean",
        "batch_row_violation_rate_tail_mean",
        "maximum_log_excess_tail_max",
        "gradient_tail_mean",
        "loss_finite",
        "gradient_finite",
        "training_rows",
        "training_steps",
    )
    return {
        key: record[key]
        for key in keys
    }


def train_candidate(
    *,
    condition: np.ndarray,
    target: np.ndarray,
    stageb_spec: stageb.DiagnosticSpec,
    objective_contract: stagea.UpperObjectiveContract,
    condition_standardizer: stageb.ArrayStandardizer,
    target_standardizer: stageb.ArrayStandardizer,
    candidate: Optional[CalibratedCandidate],
    diagnostic_batch: Mapping[str, Any],
    spec: TopKCalibrationSpec,
) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
    torch, _ = stageb._torch_imports()
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
    initial_optimizer_sha = (
        stageb.optimizer_state_sha256(optimizer)
    )

    if candidate is not None:
        candidate.validate()
        if candidate.initial_model_sha256 != initial_model_sha:
            raise TopKCalibrationError(
                "candidate initial model SHA changed"
            )
        lambda_upper = float(candidate.lambda_upper)
        top_k = int(candidate.top_k)
        candidate_id = candidate.candidate_id
    else:
        lambda_upper = 0.0
        top_k = int(spec.top_k_values[0])
        candidate_id = CONTROL_CANDIDATE_ID

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
    clipping_values: List[bool] = []
    exposure = np.zeros(
        condition.shape[0],
        dtype=np.int64,
    )
    checkpoint_records: MutableMapping[str, Any] = {}

    model.train()
    for step_index in range(stageb_spec.train_steps):
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
        diffusion_loss = torch.mean(
            (predicted - clean) ** 2
        )
        if candidate is None:
            geometry_terms = None
            loss = diffusion_loss
        else:
            geometry_terms = topk_quadratic_terms_torch(
                predicted,
                target_standardizer=
                    target_standardizer,
                objective_contract=objective_contract,
                top_k=top_k,
            )
            loss = (
                diffusion_loss
                + lambda_upper * geometry_terms["total"]
            )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=10.0,
        )
        optimizer.step()

        gradient_float = float(
            gradient_norm.detach().cpu()
        )
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
                float(
                    geometry_terms["total"]
                    .detach()
                    .cpu()
                )
            )
            geometry_row_values.append(
                float(
                    torch.mean(
                        geometry_terms["row_loss"]
                    )
                    .detach()
                    .cpu()
                )
            )
            geometry_element_values.append(0.0)
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
        gradient_values.append(gradient_float)
        clipping_values.append(
            bool(gradient_float > 10.0)
        )
        np.add.at(
            exposure,
            indices.detach().cpu().numpy(),
            1,
        )

        completed_step = step_index + 1
        if completed_step in (
            spec.completed_step_checkpoints
        ):
            checkpoint_records[str(completed_step)] = (
                fixed_batch_losses_and_gradients(
                    model=model,
                    diagnostic_batch=diagnostic_batch,
                    target_standardizer=
                        target_standardizer,
                    objective_contract=
                        objective_contract,
                    top_k=top_k,
                    lambda_upper=lambda_upper,
                    row_loss_epsilon=
                        spec.row_loss_epsilon,
                )
            )

    model.eval()
    total_array = np.asarray(
        total_values,
        dtype=np.float64,
    )
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
    exposure_array = np.asarray(
        exposure,
        dtype=np.int64,
    )
    training_record = {
        "lambda_upper": lambda_upper,
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
    diagnostics = {
        "candidate_id": candidate_id,
        "top_k": top_k,
        "target_gradient_ratio": (
            None
            if candidate is None
            else candidate.target_gradient_ratio
        ),
        "lambda_upper": lambda_upper,
        "gradient_clip_count":
            int(np.sum(clipping_values)),
        "gradient_clip_frequency":
            float(np.mean(clipping_values)),
        "checkpoint_records": checkpoint_records,
        "checkpoint_record_sha256":
            sha256_bytes(
                stable_json_bytes(
                    checkpoint_records
                )
            ),
    }
    return model, training_record, diagnostics


def holdout_profiles(
    predictions: Mapping[int, np.ndarray],
    *,
    upper_gate: staged3.OneSidedUpperContract,
    objective_contract: stagea.UpperObjectiveContract,
) -> Dict[str, Any]:
    return {
        str(timestep):
            stageb_mechanism.upper_violation_profile(
                prediction,
                upper_gate=upper_gate,
                objective_contract=objective_contract,
                top_k=(1, 4, 8, 16),
            )
        for timestep, prediction in sorted(
            predictions.items()
        )
    }


def reduction(
    candidate: float,
    control: float,
) -> float:
    return (
        1.0 - float(candidate) / float(control)
        if float(control) > 0.0
        else 0.0
    )


def ratio(
    candidate: float,
    control: float,
) -> float:
    return (
        float(candidate) / float(control)
        if float(control) > 0.0
        else 0.0
    )


def final_checkpoint(
    diagnostics: Mapping[str, Any],
) -> Mapping[str, Any]:
    checkpoints = diagnostics["checkpoint_records"]
    key = str(
        max(int(value) for value in checkpoints)
    )
    return checkpoints[key]


def selection_record(
    *,
    candidate: Optional[CalibratedCandidate],
    training: Mapping[str, Any],
    training_diagnostics: Mapping[str, Any],
    train_control: Mapping[str, Any],
    one_step: Mapping[str, Any],
    profiles: Mapping[str, Any],
    control: Optional[Mapping[str, Any]],
    spec: TopKCalibrationSpec,
) -> Dict[str, Any]:
    if candidate is None:
        return {
            "candidate_id": CONTROL_CANDIDATE_ID,
            "candidate": None,
            "training": dict(training),
            "training_diagnostics":
                dict(training_diagnostics),
            "train_control": dict(train_control),
            "one_step": dict(one_step),
            "holdout_profiles": dict(profiles),
            "continuous_geometry_gates": {},
            "fidelity_gates": {},
            "binary_gates": {},
            "stability_gates": {},
            "continuous_geometry_pass": False,
            "fidelity_pass": True,
            "binary_pass": False,
            "stability_pass": True,
            "eligible_for_selection": False,
            "relative_to_control": {
                "train_control_nmse_ratio": 1.0,
                "t10_nmse_ratio": 1.0,
                "t25_nmse_ratio": 1.0,
                "t50_nmse_ratio": 1.0,
            },
            "continuous_response": {
                "t10_mean_excess_reduction": 0.0,
                "t25_mean_excess_reduction": 0.0,
                "t50_mean_excess_reduction": 0.0,
                "t10_top8_reduction": 0.0,
                "t25_top8_reduction": 0.0,
                "t50_top8_reduction": 0.0,
            },
        }
    if control is None:
        raise ValueError(
            "non-control candidate requires control"
        )

    continuous = {}
    p95_ratios = {}
    positive_count_ratios = {}
    top8_reductions = {}
    for timestep in ("10", "25", "50"):
        candidate_profile = profiles[timestep]
        control_profile = control[
            "holdout_profiles"
        ][timestep]
        continuous[timestep] = reduction(
            candidate_profile[
                "row_best_max_log_excess"
            ]["mean"],
            control_profile[
                "row_best_max_log_excess"
            ]["mean"],
        )
        p95_ratios[timestep] = ratio(
            candidate_profile[
                "row_best_max_log_excess"
            ]["p95"],
            control_profile[
                "row_best_max_log_excess"
            ]["p95"],
        )
        positive_count_ratios[timestep] = ratio(
            candidate_profile[
                "candidate_positive_segment_count"
            ]["mean"],
            control_profile[
                "candidate_positive_segment_count"
            ]["mean"],
        )
        top8_reductions[timestep] = reduction(
            candidate_profile[
                "top_k_mean_log_excess"
            ]["8"]["mean"],
            control_profile[
                "top_k_mean_log_excess"
            ]["8"]["mean"],
        )

    continuous_gates = {
        "t10_mean_excess_reduction":
            continuous["10"]
            >= spec.t10_mean_excess_reduction_min,
        "t25_mean_excess_reduction":
            continuous["25"]
            >= spec.t25_mean_excess_reduction_min,
        "t50_mean_excess_reduction":
            continuous["50"]
            >= spec.t50_mean_excess_reduction_min,
        "all_top8_mean_excess_reduction":
            all(
                value
                > spec.top8_mean_excess_reduction_min
                for value in top8_reductions.values()
            ),
        "all_p95_nonincreasing":
            all(
                value <= spec.p95_excess_ratio_max
                for value in p95_ratios.values()
            ),
        "all_positive_count_nonincreasing":
            all(
                value
                <= spec.positive_segment_count_ratio_max
                for value
                in positive_count_ratios.values()
            ),
    }

    one_step_ratio = {
        timestep: ratio(
            one_step["timesteps"][timestep][
                "normalized_mse"
            ],
            control["one_step"]["timesteps"][timestep][
                "normalized_mse"
            ],
        )
        for timestep in ("10", "25", "50")
    }
    control_ratio = ratio(
        train_control["normalized_mse"],
        control["train_control"]["normalized_mse"],
    )
    fidelity_gates = {
        "loss_finite": bool(training["loss_finite"]),
        "gradient_finite":
            bool(training["gradient_finite"]),
        "train_control_nmse_ratio":
            control_ratio
            <= spec.train_control_nmse_ratio_max,
        "all_one_step_nmse_ratios":
            all(
                value
                <= spec.one_step_nmse_ratio_max
                for value in one_step_ratio.values()
            ),
    }

    binary_rate = {
        timestep: float(
            profiles[timestep]["row_any_rate"]
        )
        for timestep in ("10", "25", "50")
    }
    binary_gates = {
        "t10_upper_row_any":
            binary_rate["10"]
            >= spec.binary_t10_upper_row_min,
        "t25_upper_row_any":
            binary_rate["25"]
            >= spec.binary_t25_upper_row_min,
        "t50_upper_row_any":
            binary_rate["50"]
            >= spec.binary_t50_upper_row_min,
    }

    last = final_checkpoint(training_diagnostics)
    row_contribution = last["row_contribution"]
    stability_gates = {
        "gradient_clip_frequency":
            float(
                training_diagnostics[
                    "gradient_clip_frequency"
                ]
            )
            <= spec.gradient_clip_frequency_max,
        "largest_row_contribution":
            float(
                row_contribution[
                    "largest_row_contribution_fraction"
                ]
            )
            <= spec.largest_row_contribution_max,
        "top_five_percent_contribution":
            float(
                row_contribution[
                    "top_five_percent_contribution_fraction"
                ]
            )
            <= spec.top_five_percent_contribution_max,
    }

    continuous_pass = bool(
        all(continuous_gates.values())
    )
    fidelity_pass = bool(all(fidelity_gates.values()))
    binary_pass = bool(all(binary_gates.values()))
    stability_pass = bool(all(stability_gates.values()))
    return {
        "candidate_id": candidate.candidate_id,
        "candidate": candidate.to_dict(),
        "training": dict(training),
        "training_diagnostics":
            dict(training_diagnostics),
        "train_control": dict(train_control),
        "one_step": dict(one_step),
        "holdout_profiles": dict(profiles),
        "continuous_geometry_gates":
            continuous_gates,
        "fidelity_gates": fidelity_gates,
        "binary_gates": binary_gates,
        "stability_gates": stability_gates,
        "continuous_geometry_pass": continuous_pass,
        "fidelity_pass": fidelity_pass,
        "binary_pass": binary_pass,
        "stability_pass": stability_pass,
        "eligible_for_selection": bool(
            continuous_pass
            and fidelity_pass
            and binary_pass
            and stability_pass
        ),
        "relative_to_control": {
            "train_control_nmse_ratio": control_ratio,
            "t10_nmse_ratio": one_step_ratio["10"],
            "t25_nmse_ratio": one_step_ratio["25"],
            "t50_nmse_ratio": one_step_ratio["50"],
            "p95_excess_ratio": p95_ratios,
            "positive_segment_count_ratio":
                positive_count_ratios,
        },
        "continuous_response": {
            "t10_mean_excess_reduction":
                continuous["10"],
            "t25_mean_excess_reduction":
                continuous["25"],
            "t50_mean_excess_reduction":
                continuous["50"],
            "t10_top8_reduction":
                top8_reductions["10"],
            "t25_top8_reduction":
                top8_reductions["25"],
            "t50_top8_reduction":
                top8_reductions["50"],
        },
        "binary_rate": binary_rate,
    }


def select_configuration(
    records: Sequence[Mapping[str, Any]],
) -> Optional[Dict[str, Any]]:
    eligible = [
        record
        for record in records
        if record.get("eligible_for_selection") is True
    ]
    if not eligible:
        return None
    selected = sorted(
        eligible,
        key=lambda record: (
            -float(record["binary_rate"]["10"]),
            float(
                record["holdout_profiles"]["10"][
                    "row_best_max_log_excess"
                ]["mean"]
            ),
            float(
                record["candidate"][
                    "target_gradient_ratio"
                ]
            ),
            int(record["candidate"]["top_k"]),
            str(record["candidate_id"]),
        ),
    )[0]
    return {
        "candidate_id": selected["candidate_id"],
        "top_k": int(selected["candidate"]["top_k"]),
        "target_gradient_ratio": float(
            selected["candidate"][
                "target_gradient_ratio"
            ]
        ),
        "lambda_upper": float(
            selected["candidate"]["lambda_upper"]
        ),
        "pilot_model_sha256":
            selected["training"]["final_model_sha256"],
        "selection_rule": (
            "highest train-only t10 binary upper rate, "
            "then lowest t10 row-best log excess, "
            "then lowest target gradient ratio, "
            "then smallest top-k"
        ),
        "selection_record_sha256":
            sha256_bytes(stable_json_bytes(selected)),
    }


def candidate_conflict_severe(
    record: Mapping[str, Any],
    spec: TopKCalibrationSpec,
) -> bool:
    checkpoint = final_checkpoint(
        record["training_diagnostics"]
    )
    gradient = checkpoint["gradient"]
    return bool(
        float(gradient["cosine"])
        <= spec.severe_gradient_conflict_cosine
        or float(
            gradient[
                "conflict_fraction_among_active"
            ]
        )
        >= spec.severe_gradient_conflict_fraction
    )


def classify_selection(
    *,
    records: Sequence[Mapping[str, Any]],
    selected: Optional[Mapping[str, Any]],
    spec: TopKCalibrationSpec,
) -> Dict[str, Any]:
    candidates = [
        record
        for record in records
        if record["candidate_id"]
        != CONTROL_CANDIDATE_ID
    ]
    if selected is not None:
        root = (
            "phase314b_r257_stagec_topk_quadratic_"
            "train_only_configuration_selected"
        )
        next_path = (
            "RETRAIN_SELECTED_TOPK_OBJECTIVE_ON_FULL_"
            "STAGEB_TRAIN_AND_OPEN_FROZEN_PROBE"
        )
        locus = "none"
    else:
        continuous = [
            record
            for record in candidates
            if record["continuous_geometry_pass"]
        ]
        if not continuous:
            root = (
                "phase314b_r257_stagec_topk_quadratic_"
                "no_geometry_response"
            )
            next_path = (
                "AUDIT_DIRECT_X0_RESIDUAL_AND_RAW_"
                "GEOMETRY_PARAMETERIZATION"
            )
            locus = "geometry_response"
        else:
            fidelity = [
                record
                for record in continuous
                if record["fidelity_pass"]
            ]
            if not fidelity:
                best = max(
                    continuous,
                    key=lambda record: (
                        float(
                            record["continuous_response"][
                                "t10_mean_excess_reduction"
                            ]
                        ),
                        -float(
                            record["relative_to_control"][
                                "t10_nmse_ratio"
                            ]
                        ),
                    ),
                )
                if candidate_conflict_severe(best, spec):
                    root = (
                        "phase314b_r257_stagec_"
                        "topk_quadratic_gradient_conflict"
                    )
                    next_path = (
                        "CALIBRATE_CONFLICT_PROJECTED_OR_"
                        "TIMESTEP_GATED_TOPK_OBJECTIVE"
                    )
                    locus = "gradient_conflict"
                else:
                    root = (
                        "phase314b_r257_stagec_"
                        "topk_quadratic_fidelity_tradeoff"
                    )
                    next_path = (
                        "CALIBRATE_TIMESTEP_GATED_TOPK_"
                        "QUADRATIC_OBJECTIVE"
                    )
                    locus = "fidelity_tradeoff"
            else:
                stable = [
                    record
                    for record in fidelity
                    if record["stability_pass"]
                ]
                if not stable:
                    root = (
                        "phase314b_r257_stagec_"
                        "topk_quadratic_outlier_dominance"
                    )
                    next_path = (
                        "CALIBRATE_CLIPPED_TOPK_QUADRATIC_"
                        "OBJECTIVE_ON_TRAIN_ONLY_SPLIT"
                    )
                    locus = "quadratic_outlier_dominance"
                else:
                    root = (
                        "phase314b_r257_stagec_continuous_"
                        "geometry_response_without_gate_crossing"
                    )
                    next_path = (
                        "CALIBRATE_STRONGER_GRADIENT_BALANCED_"
                        "TOPK_OBJECTIVE_ON_TRAIN_ONLY_SPLIT"
                    )
                    locus = "binary_gate_crossing"

    best = min(
        candidates,
        key=lambda record: (
            float(
                record["holdout_profiles"]["10"][
                    "row_best_max_log_excess"
                ]["mean"]
            ),
            str(record["candidate_id"]),
        ),
    )
    return {
        "root_cause": root,
        "required_next_path": next_path,
        "primary_failure_locus": locus,
        "best_candidate_by_t10_continuous_excess":
            best["candidate_id"],
        "best_t10_mean_excess": float(
            best["holdout_profiles"]["10"][
                "row_best_max_log_excess"
            ]["mean"]
        ),
        "best_t10_mean_excess_reduction": float(
            best["continuous_response"][
                "t10_mean_excess_reduction"
            ]
        ),
        "best_t10_binary_upper_row_any": float(
            best["binary_rate"]["10"]
        ),
        "selected_configuration": (
            None
            if selected is None
            else dict(selected)
        ),
    }


def run_calibration(
    *,
    root: Path,
    spec: Optional[TopKCalibrationSpec] = None,
) -> Dict[str, Any]:
    active_spec = (
        TopKCalibrationSpec()
        if spec is None
        else spec
    )
    active_spec.validate()
    repository_root = Path(root).resolve()

    immutable = validate_immutable_inputs(
        repository_root
    )
    logic = source_logic_audit(repository_root)
    if not logic["all_confirmed"]:
        raise TopKCalibrationError(
            "Stage-C source assumptions changed"
        )

    stageb_spec = stageb.DiagnosticSpec()
    stageb_spec.validate()
    upstream = (
        stageb_mechanism.validate_immutable_inputs(
            repository_root
        )
    )
    objective_contract = (
        stageb_mechanism.load_objective_contract(
            upstream["stagea_contract"]
        )
    )
    upper_gate = stagea.load_upper_gate_contract(
        upstream["upstream_immutable"][
            "staged3_contract"
        ]
    )
    stage_d_contract, _ = staged1.load_stage_d_gate(
        repository_root / staged3.STAGE_D_GATE
    )

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

    stageb_train, frozen_probe, _ = (
        stageb.deterministic_group_split(
            groups,
            folds=stageb_spec.group_folds,
            probe_fold=stageb_spec.probe_fold,
        )
    )
    objective_train, selection_holdout, split = (
        stagea.deterministic_selection_split(
            groups,
            stageb_train,
            folds=
                stagea.UpperObjectiveSpec()
                .selection_group_folds,
            holdout_fold=
                stagea.UpperObjectiveSpec()
                .selection_holdout_fold,
        )
    )
    stageb_split = immutable[
        "stageb_worker_result"
    ]["split"]
    for key in (
        "objective_train_rows",
        "selection_holdout_rows",
        "objective_train_groups",
        "selection_holdout_groups",
        "objective_train_group_sha256",
        "selection_holdout_group_sha256",
        "group_overlap",
        "row_overlap",
    ):
        if split[key] != stageb_split[key]:
            raise TopKCalibrationError(
                f"train-only split changed: {key}"
            )
    if np.any(
        frozen_probe
        & (objective_train | selection_holdout)
    ):
        raise AssertionError(
            "frozen probe crossed train-only split"
        )

    condition_standardizer = stageb.fit_standardizer(
        condition[objective_train]
    )
    target_standardizer = stageb.fit_standardizer(
        target[objective_train]
    )
    diagnostic_batch = (
        stageb_mechanism.fixed_diagnostic_batch(
            condition=condition[objective_train],
            target=target[objective_train],
            condition_standardizer=
                condition_standardizer,
            target_standardizer=
                target_standardizer,
            stageb_spec=stageb_spec,
            spec=stageb_mechanism.MechanismAuditSpec(),
        )
    )
    expected_batch_sha = immutable[
        "stageb_contract"
    ]["diagnostic_batch_sha256"]
    observed_batch_sha = {
        key: value
        for key, value in diagnostic_batch.items()
        if key.endswith("_sha256")
    }
    if observed_batch_sha != expected_batch_sha:
        raise TopKCalibrationError(
            "Stage-B diagnostic batch changed"
        )

    candidates, calibration = calibrate_candidates(
        stageb_spec=stageb_spec,
        diagnostic_batch=diagnostic_batch,
        target_standardizer=target_standardizer,
        objective_contract=objective_contract,
        spec=active_spec,
    )

    stagea_candidates = immutable[
        "upstream_immutable"
    ]["candidate_records"]
    stagea_control = next(
        record
        for record in stagea_candidates
        if float(record["lambda_upper"]) == 0.0
    )

    records = []
    control_bundle = None
    ordered_candidates: List[
        Optional[CalibratedCandidate]
    ] = [None] + list(candidates)
    for candidate in ordered_candidates:
        stageb.set_deterministic_runtime(
            stageb_spec.seed
        )
        model, training, diagnostics = train_candidate(
            condition=condition[objective_train],
            target=target[objective_train],
            stageb_spec=stageb_spec,
            objective_contract=objective_contract,
            condition_standardizer=
                condition_standardizer,
            target_standardizer=
                target_standardizer,
            candidate=candidate,
            diagnostic_batch=diagnostic_batch,
            spec=active_spec,
        )

        train_control = stagea.train_control_audit(
            model=model,
            condition=condition[objective_train],
            target=target[objective_train],
            stageb_spec=stageb_spec,
            condition_standardizer=
                condition_standardizer,
            target_standardizer=
                target_standardizer,
        )
        one_step = stagea.one_step_audit(
            model=model,
            condition=condition[selection_holdout],
            target=target[selection_holdout],
            groups=groups[selection_holdout],
            condition_name=
                condition_name[selection_holdout],
            stageb_spec=stageb_spec,
            upper_gate=upper_gate,
            stage_d_contract=stage_d_contract,
            historical_geometry=
                stageb.fit_geometry_contract(
                    target[stageb_train]
                ),
            condition_standardizer=
                condition_standardizer,
            target_standardizer=
                target_standardizer,
            noise_seed=(
                stageb_spec.seed
                + active_spec.holdout_noise_seed_offset
            ),
        )
        predictions, aggregate_sha = (
            stagec.one_step_predictions(
                model=model,
                condition=condition[selection_holdout],
                target=target[selection_holdout],
                spec=stageb_spec,
                condition_standardizer=
                    condition_standardizer,
                target_standardizer=
                    target_standardizer,
                noise_seed=(
                    stageb_spec.seed
                    + active_spec.holdout_noise_seed_offset
                ),
            )
        )
        if aggregate_sha != one_step[
            "prediction_sha256"
        ]:
            raise TopKCalibrationError(
                "one-step aggregate SHA mismatch"
            )
        profiles = holdout_profiles(
            predictions,
            upper_gate=upper_gate,
            objective_contract=objective_contract,
        )

        if candidate is None:
            if compatible_training_record(training) != (
                stagea_control["training"]
            ):
                raise TopKCalibrationError(
                    "Stage-C control training differs from Stage A"
                )
            if train_control != stagea_control[
                "train_control"
            ]:
                raise TopKCalibrationError(
                    "Stage-C control train-control differs"
                )
            if one_step != stagea_control["one_step"]:
                raise TopKCalibrationError(
                    "Stage-C control one-step differs"
                )
            record = selection_record(
                candidate=None,
                training=training,
                training_diagnostics=diagnostics,
                train_control=train_control,
                one_step=one_step,
                profiles=profiles,
                control=None,
                spec=active_spec,
            )
            control_bundle = record
        else:
            if control_bundle is None:
                raise AssertionError(
                    "control candidate was not run first"
                )
            record = selection_record(
                candidate=candidate,
                training=training,
                training_diagnostics=diagnostics,
                train_control=train_control,
                one_step=one_step,
                profiles=profiles,
                control=control_bundle,
                spec=active_spec,
            )
        records.append(record)
        del model

    if control_bundle is None:
        raise AssertionError(
            "control candidate is missing"
        )
    selected = select_configuration(records)
    classification = classify_selection(
        records=records,
        selected=selected,
        spec=active_spec,
    )
    selection_payload = {
        "candidate_records": records,
        "selected_configuration": selected,
        "selection_uses_frozen_probe": False,
        "selection_rule": (
            "continuous geometry, fidelity, stability and "
            "binary gates on the frozen train-only holdout"
        ),
    }
    selection_sha = sha256_bytes(
        stable_json_bytes(selection_payload)
    )

    calibration_contract = {
        "schema":
            "phase314b_r257_stagec_topk_quadratic_contract_v1",
        "base_evidence_commit":
            BASE_EVIDENCE_COMMIT,
        "objective_variant":
            "topk_quadratic_raw_log_excess",
        "top_k_values":
            list(active_spec.top_k_values),
        "target_gradient_ratios":
            list(active_spec.target_gradient_ratios),
        "lambda_calibration": (
            "lambda = target_ratio * "
            "||g_diffusion|| / ||g_geometry|| "
            "at the common initial model on the frozen "
            "Stage-B diagnostic batch"
        ),
        "calibration": calibration,
        "selection_thresholds": {
            "t10_mean_excess_reduction_min":
                active_spec
                .t10_mean_excess_reduction_min,
            "t25_mean_excess_reduction_min":
                active_spec
                .t25_mean_excess_reduction_min,
            "t50_mean_excess_reduction_min":
                active_spec
                .t50_mean_excess_reduction_min,
            "top8_mean_excess_reduction_min":
                active_spec
                .top8_mean_excess_reduction_min,
            "p95_excess_ratio_max":
                active_spec.p95_excess_ratio_max,
            "positive_segment_count_ratio_max":
                active_spec
                .positive_segment_count_ratio_max,
            "train_control_nmse_ratio_max":
                active_spec
                .train_control_nmse_ratio_max,
            "one_step_nmse_ratio_max":
                active_spec.one_step_nmse_ratio_max,
            "binary_t10_upper_row_min":
                active_spec.binary_t10_upper_row_min,
            "binary_t25_upper_row_min":
                active_spec.binary_t25_upper_row_min,
            "binary_t50_upper_row_min":
                active_spec.binary_t50_upper_row_min,
            "gradient_clip_frequency_max":
                active_spec.gradient_clip_frequency_max,
            "largest_row_contribution_max":
                active_spec
                .largest_row_contribution_max,
            "top_five_percent_contribution_max":
                active_spec
                .top_five_percent_contribution_max,
        },
        "uses_frozen_probe": False,
        "runs_reverse_sampling": False,
        "trains_full_stageb_model": False,
        "softplus_variant_run": False,
    }
    calibration_contract["contract_sha256"] = (
        sha256_bytes(
            stable_json_bytes(
                calibration_contract
            )
        )
    )

    return {
        "phase": PHASE,
        "phase_id": PHASE_ID,
        "schema":
            "phase314b_r257_stagec_worker_result_v1",
        "verdict": "PASS",
        "scientific_status": "BLOCKED",
        "root_cause": classification["root_cause"],
        "required_next_path":
            classification["required_next_path"],
        "calibration_spec": asdict(active_spec),
        "immutable_inputs": {
            "base_commit": immutable["base_commit"],
            "base_file_sha256":
                immutable["base_file_sha256"],
            "stageb_worker_identity_sha256":
                immutable["worker_identity_sha256"],
            "stageb_mechanism_contract_sha256":
                EXPECTED_STAGEB_MECHANISM_CONTRACT_SHA256,
            "stageb_mechanism_contract_file_sha256":
                EXPECTED_STAGEB_MECHANISM_CONTRACT_FILE_SHA256,
        },
        "source_logic_audit": logic,
        "train_view_validation": validation,
        "split": {
            "stageb_training_rows":
                int(np.sum(stageb_train)),
            "objective_train_rows":
                int(np.sum(objective_train)),
            "selection_holdout_rows":
                int(np.sum(selection_holdout)),
            "frozen_probe_rows":
                int(np.sum(frozen_probe)),
            **split,
            "frozen_probe_accessed": False,
        },
        "calibration_contract":
            calibration_contract,
        "selection": {
            **selection_payload,
            "selection_sha256": selection_sha,
        },
        "classification": classification,
        "selected_configuration": selected,
        "train_only_recommendation": selected,
        "new_hyperparameter_candidate_run": True,
        "new_objective_variant_run": True,
        "objective_variant":
            "topk_quadratic_raw_log_excess",
        "control_replay_exact": True,
        "frozen_probe_accessed": False,
        "reverse_sampling_run": False,
        "full_stageb_repaired_model_trained": False,
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
        "deformable_ravens_executed": False,
        "phase4": False,
        "cps": False,
    }


def identity_projection(
    result: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "root_cause": result["root_cause"],
        "required_next_path":
            result["required_next_path"],
        "split": result["split"],
        "calibration_contract":
            result["calibration_contract"],
        "selection": result["selection"],
        "classification": result["classification"],
        "selected_configuration":
            result["selected_configuration"],
    }


def compare_worker_results(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> Dict[str, Any]:
    left_payload = stable_json_bytes(
        identity_projection(left)
    )
    right_payload = stable_json_bytes(
        identity_projection(right)
    )
    return {
        "exact": left_payload == right_payload,
        "left_sha256": sha256_bytes(left_payload),
        "right_sha256": sha256_bytes(right_payload),
        "calibration_contract_exact": (
            left["calibration_contract"]
            == right["calibration_contract"]
        ),
        "selection_exact": (
            left["selection"]
            == right["selection"]
        ),
        "classification_exact": (
            left["classification"]
            == right["classification"]
        ),
    }
