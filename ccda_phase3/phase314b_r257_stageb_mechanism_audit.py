"""Phase3.14b-r2.5.7 Stage B train-only objective mechanism audit.

Stage A calibrated lambda_upper in {0, 1e-3, 1e-2, 1e-1} on a grouped
holdout inside the frozen Stage-B training partition.  All candidates retained
zero one-step upper-valid row coverage while larger lambda values degraded
reconstruction.  The binary gate alone does not establish whether the
objective:

* reduced continuous upper violation but failed to cross the strict gate;
* produced a geometry gradient too small relative to diffusion MSE;
* conflicted with the diffusion gradient;
* concentrated row-max gradient on too few segment positions;
* saturated the Huber loss in its linear region; or
* was disconnected from the direct-x0 model graph.

This source-only/train-only stage deterministically replays the four Stage-A
pilot models.  It binds each replay to the Stage-A model, optimizer, loss,
gradient, exposure, train-control and one-step identities.  During the exact
training replay it performs read-only gradient diagnostics at registered
completed-step checkpoints on a separate deterministic batch.  The diagnostic
forward/backward calls use no random state, do not update parameters or
optimizer state, and the final replay identities must remain exact.

The frozen 126-row probe, reverse sampler, formal pilot, IDM, environment and
CPS are not accessed.
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

PHASE = "Phase3.14b-r2.5.7 Stage B"
PHASE_ID = "phase314b_r257_stageb"
BASE_EVIDENCE_COMMIT = "a15aa9d6880767f02ff0db30486f06d625e4d0a3"
EXPECTED_SUBMODULE_COMMIT = "633a88752445cf5d6776ed374fdbbdb35f93050c"

STAGEA_SOURCE = "ccda_phase3/phase314b_r257_stagea_upper_objective.py"
STAGEA_TEST = "tests/test_phase3_14b_r257_stagea_upper_objective.py"
STAGEA_CONTRACT = (
    "reports/phase3_14b_r257_stagea_upper_objective_contract.json"
)
STAGEA_TEST_GATE = (
    "reports/phase3_14b_r257_stagea_test_gate_summary.json"
)
STAGEA_WORKER = (
    "reports/phase3_14b_r257_stagea_worker_evidence.json"
)
STAGEA_SUMMARY = "reports/phase3_14b_r257_stagea_summary.json"
STAGEA_REPORT = "reports/phase3_14b_r257_stagea_report.md"

BASE_BOUND_FILES = (
    STAGEA_SOURCE,
    STAGEA_TEST,
    STAGEA_CONTRACT,
    STAGEA_TEST_GATE,
    STAGEA_WORKER,
    STAGEA_SUMMARY,
    STAGEA_REPORT,
)

EXPECTED_STAGEA_WORKER_SHA256 = (
    "ab6a626a284a1da6c1cf88b75f26388f9c31413cd4a112f11def3c610f7fde3c"
)
EXPECTED_STAGEA_OBJECTIVE_SHA256 = (
    "58f9e3e21e3580a317ae2f1ada1e7d99be672e2da1259bc1a6d4800909bf199f"
)
EXPECTED_STAGEA_SELECTION_SHA256 = (
    "ded47097a184fd6f2ed5fb36bb4e1085a255f75f1fa1d9524b193f1c2aba0169"
)


class MechanismAuditError(RuntimeError):
    """Raised when an immutable identity or attribution boundary fails."""


@dataclass(frozen=True)
class MechanismAuditSpec:
    """Pre-registered train-only mechanism-attribution contract."""

    candidate_lambdas: Tuple[float, ...] = (
        0.0,
        1.0e-3,
        1.0e-2,
        1.0e-1,
    )
    completed_step_checkpoints: Tuple[int, ...] = (
        1,
        100,
        500,
        2000,
        8000,
    )
    diagnostic_batch_size: int = 64
    diagnostic_timesteps: Tuple[int, ...] = (10, 25, 50, 75)
    diagnostic_seed_offset: int = 5201
    holdout_noise_seed_offset: int = 4101
    top_k: Tuple[int, ...] = (1, 4, 8)

    minimum_mean_excess_reduction: float = 0.05
    substantial_mean_excess_reduction: float = 0.20
    weak_scaled_gradient_ratio: float = 0.10
    conflicting_gradient_cosine: float = -0.10
    rowmax_position_concentration: float = 0.35
    huber_linear_fraction: float = 0.90
    graph_gradient_norm_floor: float = 1.0e-12

    def validate(self) -> None:
        if self.candidate_lambdas != stagea.UpperObjectiveSpec().candidate_lambdas:
            raise ValueError("candidate lambdas differ from Stage A")
        if self.completed_step_checkpoints[-1] != (
            stageb.DiagnosticSpec().train_steps
        ):
            raise ValueError("final checkpoint must equal training steps")
        if tuple(sorted(set(self.completed_step_checkpoints))) != (
            self.completed_step_checkpoints
        ):
            raise ValueError("completed-step checkpoints must be ordered")
        if self.completed_step_checkpoints[0] <= 0:
            raise ValueError("checkpoint must follow at least one update")
        if self.diagnostic_batch_size <= 0:
            raise ValueError("diagnostic batch size must be positive")
        if not self.diagnostic_timesteps:
            raise ValueError("diagnostic timesteps are empty")
        if any(
            not 0 <= int(value) < stageb.DiagnosticSpec().train_timesteps
            for value in self.diagnostic_timesteps
        ):
            raise ValueError("diagnostic timestep is outside training range")
        if tuple(sorted(set(self.top_k))) != self.top_k:
            raise ValueError("top-k values must be ordered and unique")
        if self.top_k[-1] > (
            stageb.FUTURE_STEPS * (stageb.BEADS - 1)
        ):
            raise ValueError("top-k exceeds ordered segment count")
        for value in (
            self.minimum_mean_excess_reduction,
            self.substantial_mean_excess_reduction,
            self.weak_scaled_gradient_ratio,
            self.rowmax_position_concentration,
            self.huber_linear_fraction,
        ):
            if not 0.0 < float(value) <= 1.0:
                raise ValueError("registered rate is outside (0,1]")
        if self.substantial_mean_excess_reduction < (
            self.minimum_mean_excess_reduction
        ):
            raise ValueError("substantial reduction is below minimum")
        if not -1.0 <= self.conflicting_gradient_cosine <= 1.0:
            raise ValueError("gradient cosine threshold is invalid")
        if self.graph_gradient_norm_floor <= 0.0:
            raise ValueError("gradient norm floor must be positive")


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
        raise MechanismAuditError(
            f"JSON root is not an object: {path}"
        )
    return value


def safe_stats(value: np.ndarray) -> Dict[str, float]:
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
        raise MechanismAuditError(
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
        raise MechanismAuditError(
            f"Stage-A bound file changed: {relative}"
        )
    return sha256_bytes(observed)


def _worker_identity_sha(worker: Mapping[str, Any]) -> str:
    comparison = worker.get("comparison")
    if not isinstance(comparison, Mapping):
        raise MechanismAuditError(
            "Stage-A worker comparison is missing"
        )
    left = comparison.get("left_sha256")
    right = comparison.get("right_sha256")
    if left != right or not isinstance(left, str):
        raise MechanismAuditError(
            "Stage-A worker identity hashes differ"
        )
    return left


def validate_immutable_inputs(root: Path) -> Dict[str, Any]:
    repository_root = Path(root).resolve()
    bound = {
        relative: assert_base_file_bound(repository_root, relative)
        for relative in BASE_BOUND_FILES
    }
    summary = load_json(repository_root / STAGEA_SUMMARY)
    if summary.get("verdict") != "PASS":
        raise MechanismAuditError(
            "Stage-A execution verdict is not PASS"
        )
    if summary.get("scientific_status") != "BLOCKED":
        raise MechanismAuditError(
            "Stage-A scientific status changed"
        )
    if summary.get("root_cause") != (
        "phase314b_r257_stagea_no_train_only_upper_objective_configuration"
    ):
        raise MechanismAuditError(
            "Stage-A root cause changed"
        )
    if summary.get("required_next_path") != (
        "CALIBRATE_STRONGER_OR_ALTERNATIVE_UPPER_OBJECTIVE_ON_TRAIN_ONLY_SPLIT"
    ):
        raise MechanismAuditError(
            "Stage-A required next path changed"
        )
    if summary.get("selected_configuration") is not None:
        raise MechanismAuditError(
            "Stage-A unexpectedly selected a configuration"
        )
    if summary.get("train_only_recommendation") is not None:
        raise MechanismAuditError(
            "Stage-A unexpectedly selected a recommendation"
        )
    if summary.get("split_leakage_detected") is not False:
        raise MechanismAuditError(
            "Stage-A split leakage status changed"
        )

    worker = load_json(repository_root / STAGEA_WORKER)
    if worker.get("workers_exact") is not True:
        raise MechanismAuditError(
            "Stage-A workers were not exact"
        )
    worker_sha = _worker_identity_sha(worker)
    if worker_sha != EXPECTED_STAGEA_WORKER_SHA256:
        raise MechanismAuditError(
            "Stage-A worker identity SHA changed"
        )
    result = worker.get("worker_result")
    if not isinstance(result, Mapping):
        raise MechanismAuditError(
            "Stage-A worker result is missing"
        )
    selection = result.get("train_only_selection")
    if not isinstance(selection, Mapping):
        raise MechanismAuditError(
            "Stage-A train-only selection is missing"
        )
    if selection.get("selection_sha256") != (
        EXPECTED_STAGEA_SELECTION_SHA256
    ):
        raise MechanismAuditError(
            "Stage-A selection SHA changed"
        )
    if selection.get("probe_accessed_during_selection") is not False:
        raise MechanismAuditError(
            "Stage-A accessed frozen probe during selection"
        )
    if selection.get("selected_configuration") is not None:
        raise MechanismAuditError(
            "Stage-A selection is no longer empty"
        )
    final_audit = result.get("final_frozen_probe_audit")
    if not isinstance(final_audit, Mapping):
        raise MechanismAuditError(
            "Stage-A final audit record is missing"
        )
    if final_audit.get("performed") is not False:
        raise MechanismAuditError(
            "Stage-A unexpectedly performed frozen-probe audit"
        )
    if final_audit.get("probe_accessed_after_selection") is not False:
        raise MechanismAuditError(
            "Stage-A unexpectedly accessed frozen probe"
        )

    contract = load_json(repository_root / STAGEA_CONTRACT)
    if contract.get("contract_sha256") != (
        EXPECTED_STAGEA_OBJECTIVE_SHA256
    ):
        raise MechanismAuditError(
            "Stage-A objective contract SHA changed"
        )
    if contract.get("selected_configuration") is not None:
        raise MechanismAuditError(
            "Stage-A contract selected a configuration"
        )
    if contract.get("selection_sha256") != (
        EXPECTED_STAGEA_SELECTION_SHA256
    ):
        raise MechanismAuditError(
            "Stage-A contract selection SHA changed"
        )

    candidate_records = selection.get("candidate_records")
    if not isinstance(candidate_records, list):
        raise MechanismAuditError(
            "Stage-A candidate records are missing"
        )
    lambdas = tuple(
        float(record["lambda_upper"])
        for record in candidate_records
    )
    if lambdas != stagea.UpperObjectiveSpec().candidate_lambdas:
        raise MechanismAuditError(
            "Stage-A candidate lambda order changed"
        )
    if any(
        record.get("feasible_nonzero_configuration") is True
        for record in candidate_records
    ):
        raise MechanismAuditError(
            "Stage-A candidate feasibility changed"
        )

    upstream = stagea.validate_immutable_inputs(repository_root)
    return {
        "base_commit": BASE_EVIDENCE_COMMIT,
        "base_file_sha256": bound,
        "stagea_summary": summary,
        "stagea_worker": worker,
        "stagea_result": result,
        "stagea_contract": contract,
        "candidate_records": candidate_records,
        "worker_identity_sha256": worker_sha,
        "upstream_immutable": upstream,
    }


def source_logic_audit(root: Path) -> Dict[str, Any]:
    repository_root = Path(root).resolve()
    stagea_text = (repository_root / STAGEA_SOURCE).read_text(
        encoding="utf-8"
    )
    stageb_text = (
        repository_root / stagea.STAGEB_SOURCE
    ).read_text(encoding="utf-8")
    checks = {
        "stagea_uses_row_max_huber": (
            "row_max = torch.amax(excess, dim=(1, 2))"
            in stagea_text
        ),
        "stagea_uses_element_huber": (
            "element_loss = torch.mean(element_huber)"
            in stagea_text
        ),
        "stagea_records_geometry_tail_only": (
            '"geometry_loss_tail_mean"' in stagea_text
            and '"maximum_log_excess_tail_max"' in stagea_text
        ),
        "stagea_does_not_persist_models": (
            '"weights_persisted": False' in stagea_text
            and '"checkpoint_saved": False' in stagea_text
        ),
        "stagea_uses_fixed_training_generator": (
            "generator.manual_seed(stageb_spec.seed + 17)"
            in stagea_text
        ),
        "stagea_selection_uses_train_only_holdout": (
            "condition=condition[selection_holdout]"
            in stagea_text
            and "probe_accessed_during_selection" in stagea_text
        ),
        "model_has_no_dropout": (
            "nn.Dropout" not in stageb_text
        ),
        "model_has_no_batchnorm": (
            "BatchNorm" not in stageb_text
        ),
        "model_directly_predicts_x0": (
            "class CableX0Denoiser" in stageb_text
            and "loss = torch.mean((predicted - clean) ** 2)"
            in stageb_text
        ),
    }
    return {
        "checks": checks,
        "all_confirmed": bool(all(checks.values())),
        "source_sha256": {
            STAGEA_SOURCE:
                sha256_file(repository_root / STAGEA_SOURCE),
            stagea.STAGEB_SOURCE:
                sha256_file(repository_root / stagea.STAGEB_SOURCE),
        },
        "diagnostic_replay_safe": bool(
            checks["model_has_no_dropout"]
            and checks["model_has_no_batchnorm"]
        ),
    }


def load_objective_contract(
    payload: Mapping[str, Any],
) -> stagea.UpperObjectiveContract:
    contract = stagea.UpperObjectiveContract(
        upper_gate_contract_sha256=str(
            payload["upper_gate_contract_sha256"]
        ),
        upper_threshold=float(payload["upper_threshold"]),
        reference_center_log=np.asarray(
            payload["reference_center_log"],
            dtype=np.float32,
        ),
        reference_scale_log=np.asarray(
            payload["reference_scale_log"],
            dtype=np.float32,
        ),
        allowed_upper_log_length=np.asarray(
            payload["allowed_upper_log_length"],
            dtype=np.float32,
        ),
        huber_delta_log_ratio=float(
            payload["huber_delta_log_ratio"]
        ),
        element_loss_weight=float(
            payload["element_loss_weight"]
        ),
        segment_length_epsilon=float(
            payload["segment_length_epsilon"]
        ),
    )
    contract.validate()
    return contract


def fixed_diagnostic_batch(
    *,
    condition: np.ndarray,
    target: np.ndarray,
    condition_standardizer: stageb.ArrayStandardizer,
    target_standardizer: stageb.ArrayStandardizer,
    stageb_spec: stageb.DiagnosticSpec,
    spec: MechanismAuditSpec,
) -> Dict[str, Any]:
    rows = int(condition.shape[0])
    batch_size = min(int(spec.diagnostic_batch_size), rows)
    if batch_size <= 0:
        raise ValueError("diagnostic population is empty")
    rng = np.random.RandomState(
        int(stageb_spec.seed + spec.diagnostic_seed_offset)
    )
    indices = rng.permutation(rows)[:batch_size].astype(np.int64)
    timesteps = np.resize(
        np.asarray(spec.diagnostic_timesteps, dtype=np.int64),
        batch_size,
    )
    noise = rng.standard_normal(
        (
            batch_size,
            stageb.FUTURE_STEPS,
            stageb.CABLE_DIM,
        )
    ).astype(np.float32)
    condition_z = condition_standardizer.normalize(
        condition[indices]
    ).astype(np.float32)
    target_z = target_standardizer.normalize(
        target[indices]
    ).astype(np.float32)
    scheduler = stageb.scheduler_arrays(stageb_spec)
    alpha = scheduler["alpha_bar"][timesteps].reshape(-1, 1, 1)
    noisy = (
        np.sqrt(alpha).astype(np.float32) * target_z
        + np.sqrt(1.0 - alpha).astype(np.float32) * noise
    ).astype(np.float32)
    return {
        "indices": indices,
        "timesteps": timesteps,
        "noise": noise,
        "condition_z": condition_z,
        "target_z": target_z,
        "noisy_z": noisy,
        "indices_sha256": sha256_array(indices),
        "timesteps_sha256": sha256_array(timesteps),
        "noise_sha256": sha256_array(noise),
        "condition_sha256": sha256_array(condition_z),
        "target_sha256": sha256_array(target_z),
        "noisy_sha256": sha256_array(noisy),
    }


def upper_excess_torch(
    predicted_x0_z: Any,
    *,
    target_standardizer: stageb.ArrayStandardizer,
    contract: stagea.UpperObjectiveContract,
) -> Dict[str, Any]:
    torch, _ = stageb._torch_imports()
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
    raw = predicted_x0_z * scale + mean
    points = raw.reshape(
        raw.shape[0],
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
    return {
        "raw": raw,
        "lengths": lengths,
        "log_lengths": log_lengths,
        "excess": excess,
        "row_max": torch.amax(excess, dim=(1, 2)),
    }


def _gradient_pair_metrics(
    named_parameters: Sequence[Tuple[str, Any]],
    diffusion_gradients: Sequence[Optional[Any]],
    geometry_gradients: Sequence[Optional[Any]],
    *,
    lambda_upper: float,
) -> Dict[str, Any]:
    torch, _ = stageb._torch_imports()
    diffusion_sq = 0.0
    geometry_sq = 0.0
    dot = 0.0
    total_elements = 0
    active_elements = 0
    conflict_elements = 0
    zero_geometry_elements = 0
    layers: MutableMapping[str, Any] = {}

    for (name, parameter), grad_d, grad_g in zip(
        named_parameters,
        diffusion_gradients,
        geometry_gradients,
    ):
        if grad_d is None:
            grad_d = torch.zeros_like(parameter)
        if grad_g is None:
            grad_g = torch.zeros_like(parameter)
        d = grad_d.detach()
        g = grad_g.detach()
        d_sq = float(torch.sum(d * d).cpu())
        g_sq = float(torch.sum(g * g).cpu())
        d_dot_g = float(torch.sum(d * g).cpu())
        diffusion_sq += d_sq
        geometry_sq += g_sq
        dot += d_dot_g
        total_elements += int(g.numel())
        zero_geometry_elements += int(
            torch.sum(g == 0.0).cpu()
        )
        active = (d != 0.0) & (g != 0.0)
        active_count = int(torch.sum(active).cpu())
        active_elements += active_count
        conflict_elements += int(
            torch.sum((d * g < 0.0) & active).cpu()
        )
        d_norm = math.sqrt(max(d_sq, 0.0))
        g_norm = math.sqrt(max(g_sq, 0.0))
        denominator = d_norm * g_norm
        layers[name] = {
            "shape": list(parameter.shape),
            "diffusion_norm": d_norm,
            "geometry_norm": g_norm,
            "scaled_geometry_norm":
                float(lambda_upper) * g_norm,
            "cosine": (
                d_dot_g / denominator
                if denominator > 0.0
                else 0.0
            ),
            "active_element_count": active_count,
            "conflict_fraction_among_active": (
                float(
                    torch.sum((d * g < 0.0) & active).cpu()
                )
                / active_count
                if active_count
                else 0.0
            ),
            "zero_geometry_fraction": float(
                torch.mean((g == 0.0).to(torch.float32)).cpu()
            ),
        }

    diffusion_norm = math.sqrt(max(diffusion_sq, 0.0))
    geometry_norm = math.sqrt(max(geometry_sq, 0.0))
    denominator = diffusion_norm * geometry_norm
    combined_sq = (
        diffusion_sq
        + 2.0 * float(lambda_upper) * dot
        + float(lambda_upper) ** 2 * geometry_sq
    )
    return {
        "diffusion_norm": diffusion_norm,
        "geometry_norm": geometry_norm,
        "scaled_geometry_norm":
            float(lambda_upper) * geometry_norm,
        "scaled_geometry_to_diffusion_ratio": (
            float(lambda_upper) * geometry_norm
            / diffusion_norm
            if diffusion_norm > 0.0
            else 0.0
        ),
        "cosine": (
            dot / denominator
            if denominator > 0.0
            else 0.0
        ),
        "dot_product": dot,
        "combined_norm":
            math.sqrt(max(combined_sq, 0.0)),
        "active_element_count": active_elements,
        "conflict_fraction_among_active": (
            conflict_elements / active_elements
            if active_elements
            else 0.0
        ),
        "zero_geometry_fraction": (
            zero_geometry_elements / total_elements
            if total_elements
            else 0.0
        ),
        "per_parameter": layers,
    }


def fixed_batch_attribution(
    *,
    model: Any,
    batch: Mapping[str, Any],
    lambda_upper: float,
    target_standardizer: stageb.ArrayStandardizer,
    objective_contract: stagea.UpperObjectiveContract,
) -> Dict[str, Any]:
    torch, _ = stageb._torch_imports()
    device = torch.device("cuda:0")
    noisy = torch.as_tensor(
        batch["noisy_z"],
        dtype=torch.float32,
        device=device,
    )
    timestep = torch.as_tensor(
        batch["timesteps"],
        dtype=torch.long,
        device=device,
    )
    condition = torch.as_tensor(
        batch["condition_z"],
        dtype=torch.float32,
        device=device,
    )
    clean = torch.as_tensor(
        batch["target_z"],
        dtype=torch.float32,
        device=device,
    )
    named_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    parameters = [parameter for _name, parameter in named_parameters]

    predicted_diffusion = model(noisy, timestep, condition)
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

    predicted_geometry = model(noisy, timestep, condition)
    geometry_terms = stagea.upper_objective_terms_torch(
        predicted_geometry,
        target_standardizer=target_standardizer,
        contract=objective_contract,
    )
    geometry_gradients = torch.autograd.grad(
        geometry_terms["total"],
        parameters,
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )
    pair = _gradient_pair_metrics(
        named_parameters,
        diffusion_gradients,
        geometry_gradients,
        lambda_upper=float(lambda_upper),
    )

    excess_record = upper_excess_torch(
        predicted_geometry.detach(),
        target_standardizer=target_standardizer,
        contract=objective_contract,
    )
    excess = excess_record["excess"].detach()
    positive = excess > 0.0
    delta = float(objective_contract.huber_delta_log_ratio)
    quadratic = positive & (excess <= delta)
    linear = excess > delta
    positive_count = int(torch.sum(positive).cpu())
    flattened = excess.reshape(excess.shape[0], -1)
    row_argmax = torch.argmax(flattened, dim=1)
    unique, counts = torch.unique(
        row_argmax,
        sorted=True,
        return_counts=True,
    )
    frequencies = []
    for flat, count in zip(
        unique.detach().cpu().numpy().tolist(),
        counts.detach().cpu().numpy().tolist(),
    ):
        horizon, segment = np.unravel_index(
            int(flat),
            (stageb.FUTURE_STEPS, stageb.BEADS - 1),
        )
        frequencies.append(
            {
                "horizon": int(horizon),
                "segment_index": int(segment),
                "count": int(count),
                "fraction": float(
                    count / excess.shape[0]
                ),
            }
        )
    frequencies.sort(
        key=lambda item: (
            -item["count"],
            item["horizon"],
            item["segment_index"],
        )
    )

    model.zero_grad(set_to_none=True)
    return {
        "diffusion_loss": float(
            diffusion_loss.detach().cpu()
        ),
        "geometry_loss": float(
            geometry_terms["total"].detach().cpu()
        ),
        "geometry_row_loss": float(
            geometry_terms["row"].detach().cpu()
        ),
        "geometry_element_loss": float(
            geometry_terms["element"].detach().cpu()
        ),
        "gradient": pair,
        "huber": {
            "positive_element_count": positive_count,
            "positive_element_rate": float(
                torch.mean(
                    positive.to(torch.float32)
                ).cpu()
            ),
            "quadratic_fraction_among_positive": (
                float(torch.sum(quadratic).cpu())
                / positive_count
                if positive_count
                else 0.0
            ),
            "linear_fraction_among_positive": (
                float(torch.sum(linear).cpu())
                / positive_count
                if positive_count
                else 0.0
            ),
            "row_violation_rate": float(
                torch.mean(
                    (excess_record["row_max"] > 0.0)
                    .to(torch.float32)
                ).cpu()
            ),
            "maximum_log_excess": float(
                torch.max(excess).cpu()
            ),
            "row_max_log_excess": safe_stats(
                excess_record["row_max"]
                .detach()
                .cpu()
                .numpy()
            ),
            "row_argmax_position_frequency":
                frequencies[:16],
            "top_position_fraction": (
                frequencies[0]["fraction"]
                if frequencies
                else 0.0
            ),
        },
        "predicted_x0_sha256": sha256_array(
            predicted_geometry.detach().cpu().numpy()
        ),
    }


def replay_pilot_with_attribution(
    *,
    condition: np.ndarray,
    target: np.ndarray,
    stageb_spec: stageb.DiagnosticSpec,
    objective_contract: stagea.UpperObjectiveContract,
    lambda_upper: float,
    condition_standardizer: stageb.ArrayStandardizer,
    target_standardizer: stageb.ArrayStandardizer,
    diagnostic_batch: Mapping[str, Any],
    spec: MechanismAuditSpec,
) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
    """Replay Stage-A training exactly while adding read-only diagnostics."""
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
    initial_optimizer_sha = stageb.optimizer_state_sha256(
        optimizer
    )

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
        diffusion_loss = torch.mean((predicted - clean) ** 2)
        if float(lambda_upper) == 0.0:
            geometry_terms = None
            loss = diffusion_loss
        else:
            geometry_terms = stagea.upper_objective_terms_torch(
                predicted,
                target_standardizer=target_standardizer,
                contract=objective_contract,
            )
            loss = (
                diffusion_loss
                + float(lambda_upper)
                * geometry_terms["total"]
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
                float(
                    geometry_terms["element"].detach().cpu()
                )
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

        completed_step = step_index + 1
        if completed_step in spec.completed_step_checkpoints:
            checkpoint_records[str(completed_step)] = (
                fixed_batch_attribution(
                    model=model,
                    batch=diagnostic_batch,
                    lambda_upper=float(lambda_upper),
                    target_standardizer=target_standardizer,
                    objective_contract=objective_contract,
                )
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
    training_record = {
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
    attribution = {
        "completed_step_checkpoints":
            list(spec.completed_step_checkpoints),
        "diagnostic_batch": {
            key: value
            for key, value in diagnostic_batch.items()
            if key.endswith("_sha256")
        },
        "checkpoint_records": checkpoint_records,
        "checkpoint_record_sha256": sha256_bytes(
            stable_json_bytes(checkpoint_records)
        ),
    }
    return model, training_record, attribution


def _top_k_mean(
    value: np.ndarray,
    k: int,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim < 1:
        raise ValueError("top-k input is scalar")
    flattened = array.reshape(*array.shape[:-2], -1)
    actual = min(int(k), flattened.shape[-1])
    partition = np.partition(
        flattened,
        flattened.shape[-1] - actual,
        axis=-1,
    )[..., -actual:]
    return np.mean(partition, axis=-1)


def upper_violation_profile(
    value: np.ndarray,
    *,
    upper_gate: staged3.OneSidedUpperContract,
    objective_contract: stagea.UpperObjectiveContract,
    top_k: Sequence[int],
) -> Dict[str, Any]:
    candidates = staged.as_candidates(value).astype(np.float32)
    points = candidates.reshape(
        candidates.shape[0],
        candidates.shape[1],
        stageb.FUTURE_STEPS,
        stageb.BEADS,
        2,
    )
    delta_xy = points[:, :, :, 1:, :] - points[:, :, :, :-1, :]
    lengths = np.sqrt(
        np.sum(
            delta_xy.astype(np.float64) ** 2,
            axis=-1,
        )
        + float(objective_contract.segment_length_epsilon) ** 2
    )
    log_lengths = np.log(lengths)
    allowed = np.asarray(
        objective_contract.allowed_upper_log_length,
        dtype=np.float64,
    )
    excess = np.maximum(log_lengths - allowed, 0.0)
    center = np.asarray(
        upper_gate.reference.center_log,
        dtype=np.float64,
    )
    scale = np.asarray(
        upper_gate.reference.scale_log,
        dtype=np.float64,
    )
    upper_z_element = np.maximum(
        (log_lengths - center) / scale,
        0.0,
    )
    candidate_max_excess = np.max(
        excess,
        axis=(2, 3),
    )
    candidate_max_z = np.max(
        upper_z_element,
        axis=(2, 3),
    )
    row_best_excess = np.min(
        candidate_max_excess,
        axis=1,
    )
    row_best_z = np.min(candidate_max_z, axis=1)
    positive_count = np.sum(
        excess > 0.0,
        axis=(2, 3),
    )
    candidate_pass = (
        candidate_max_z <= float(upper_gate.upper_threshold)
    )
    row_any = np.any(candidate_pass, axis=1)

    top_records = {}
    for k in top_k:
        top_records[str(k)] = safe_stats(
            _top_k_mean(excess, int(k))
        )

    horizon_rate = np.mean(excess > 0.0, axis=(0, 1, 3))
    segment_rate = np.mean(excess > 0.0, axis=(0, 1, 2))
    position_rate = np.mean(excess > 0.0, axis=(0, 1))
    position_mean = np.mean(excess, axis=(0, 1))
    position_max = np.max(excess, axis=(0, 1))
    flattened = excess.reshape(
        excess.shape[0] * excess.shape[1],
        -1,
    )
    argmax = np.argmax(flattened, axis=1)
    unique, counts = np.unique(argmax, return_counts=True)
    frequencies = []
    for flat, count in zip(unique.tolist(), counts.tolist()):
        horizon, segment = np.unravel_index(
            int(flat),
            (stageb.FUTURE_STEPS, stageb.BEADS - 1),
        )
        frequencies.append(
            {
                "horizon": int(horizon),
                "segment_index": int(segment),
                "count": int(count),
                "fraction": float(
                    count / flattened.shape[0]
                ),
                "violation_rate": float(
                    position_rate[horizon, segment]
                ),
                "mean_log_excess": float(
                    position_mean[horizon, segment]
                ),
                "maximum_log_excess": float(
                    position_max[horizon, segment]
                ),
            }
        )
    frequencies.sort(
        key=lambda item: (
            -item["count"],
            item["horizon"],
            item["segment_index"],
        )
    )

    delta = float(objective_contract.huber_delta_log_ratio)
    positive = excess > 0.0
    positive_total = int(np.sum(positive))
    linear = excess > delta
    quadratic = positive & ~linear
    rows_with_at_most = {
        str(limit): float(
            np.mean(np.min(positive_count, axis=1) <= limit)
        )
        for limit in (0, 1, 2, 4, 8)
    }
    return {
        "shape": list(candidates.shape),
        "candidate_sha256": sha256_array(candidates),
        "length_sha256": sha256_array(lengths),
        "excess_sha256": sha256_array(excess),
        "upper_z_sha256": sha256_array(upper_z_element),
        "candidate_max_log_excess":
            safe_stats(candidate_max_excess),
        "row_best_max_log_excess":
            safe_stats(row_best_excess),
        "candidate_max_upper_z":
            safe_stats(candidate_max_z),
        "row_best_upper_z":
            safe_stats(row_best_z),
        "candidate_positive_segment_count":
            safe_stats(positive_count),
        "candidate_pass_rate":
            float(np.mean(candidate_pass)),
        "row_any_rate": float(np.mean(row_any)),
        "rows_with_at_most_violations": rows_with_at_most,
        "top_k_mean_log_excess": top_records,
        "huber_region": {
            "positive_element_count": positive_total,
            "quadratic_fraction_among_positive": (
                float(np.sum(quadratic)) / positive_total
                if positive_total
                else 0.0
            ),
            "linear_fraction_among_positive": (
                float(np.sum(linear)) / positive_total
                if positive_total
                else 0.0
            ),
        },
        "horizon_violation_rate": [
            float(value) for value in horizon_rate
        ],
        "segment_index_violation_rate": [
            float(value) for value in segment_rate
        ],
        "row_argmax_position_frequency": frequencies[:32],
        "top_position_fraction": (
            frequencies[0]["fraction"]
            if frequencies
            else 0.0
        ),
    }


def _median_checkpoint_value(
    attribution: Mapping[str, Any],
    path: Sequence[str],
) -> float:
    values = []
    for checkpoint in attribution["checkpoint_records"].values():
        current: Any = checkpoint
        for key in path:
            current = current[key]
        values.append(float(current))
    if not values:
        return 0.0
    return float(np.median(np.asarray(values, dtype=np.float64)))


def summarize_mechanism(
    records: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    by_lambda = {
        float(record["lambda_upper"]): record
        for record in records
    }
    control = by_lambda[0.0]
    control_mean = float(
        control["holdout_profiles"]["10"][
            "row_best_max_log_excess"
        ]["mean"]
    )
    nonzero = [
        record
        for value, record in sorted(by_lambda.items())
        if value > 0.0
    ]
    best = min(
        nonzero,
        key=lambda record: (
            float(
                record["holdout_profiles"]["10"][
                    "row_best_max_log_excess"
                ]["mean"]
            ),
            float(record["lambda_upper"]),
        ),
    )
    best_mean = float(
        best["holdout_profiles"]["10"][
            "row_best_max_log_excess"
        ]["mean"]
    )
    reduction = (
        1.0 - best_mean / control_mean
        if control_mean > 0.0
        else 0.0
    )
    strongest = by_lambda[max(by_lambda)]
    strongest_attribution = strongest["training_attribution"]
    return {
        "control_t10_mean_row_best_log_excess": control_mean,
        "best_lambda_by_t10_mean_excess":
            float(best["lambda_upper"]),
        "best_t10_mean_row_best_log_excess": best_mean,
        "best_t10_mean_excess_reduction": reduction,
        "strongest_lambda": float(strongest["lambda_upper"]),
        "strongest_scaled_gradient_ratio_median":
            _median_checkpoint_value(
                strongest_attribution,
                (
                    "gradient",
                    "scaled_geometry_to_diffusion_ratio",
                ),
            ),
        "strongest_gradient_cosine_median":
            _median_checkpoint_value(
                strongest_attribution,
                ("gradient", "cosine"),
            ),
        "strongest_gradient_conflict_fraction_median":
            _median_checkpoint_value(
                strongest_attribution,
                (
                    "gradient",
                    "conflict_fraction_among_active",
                ),
            ),
        "strongest_zero_geometry_fraction_median":
            _median_checkpoint_value(
                strongest_attribution,
                (
                    "gradient",
                    "zero_geometry_fraction",
                ),
            ),
        "strongest_geometry_norm_median":
            _median_checkpoint_value(
                strongest_attribution,
                ("gradient", "geometry_norm"),
            ),
        "strongest_huber_linear_fraction_median":
            _median_checkpoint_value(
                strongest_attribution,
                (
                    "huber",
                    "linear_fraction_among_positive",
                ),
            ),
        "strongest_rowmax_top_position_fraction_median":
            _median_checkpoint_value(
                strongest_attribution,
                ("huber", "top_position_fraction"),
            ),
        "strongest_t10_top_position_fraction": float(
            strongest["holdout_profiles"]["10"][
                "top_position_fraction"
            ]
        ),
        "strongest_t10_positive_segment_count_mean": float(
            strongest["holdout_profiles"]["10"][
                "candidate_positive_segment_count"
            ]["mean"]
        ),
        "strongest_t10_nmse_ratio": float(
            strongest["stagea_record"][
                "relative_to_control"
            ]["t10_nmse"]
        ),
    }


def classify_mechanism(
    *,
    summary: Mapping[str, Any],
    spec: MechanismAuditSpec,
) -> Dict[str, Any]:
    geometry_norm = float(
        summary["strongest_geometry_norm_median"]
    )
    reduction = float(
        summary["best_t10_mean_excess_reduction"]
    )
    scaled_ratio = float(
        summary["strongest_scaled_gradient_ratio_median"]
    )
    cosine = float(
        summary["strongest_gradient_cosine_median"]
    )
    linear = float(
        summary["strongest_huber_linear_fraction_median"]
    )
    concentration = max(
        float(
            summary[
                "strongest_rowmax_top_position_fraction_median"
            ]
        ),
        float(
            summary["strongest_t10_top_position_fraction"]
        ),
    )
    positive_count = float(
        summary["strongest_t10_positive_segment_count_mean"]
    )
    nmse_ratio = float(
        summary["strongest_t10_nmse_ratio"]
    )

    if geometry_norm <= spec.graph_gradient_norm_floor:
        root = (
            "phase314b_r257_stageb_upper_objective_"
            "gradient_disconnected"
        )
        next_path = (
            "REPAIR_UPPER_OBJECTIVE_GRAPH_CONNECTIVITY_"
            "BEFORE_RECALIBRATION"
        )
        locus = "objective_graph"
    elif (
        reduction < spec.minimum_mean_excess_reduction
        and scaled_ratio < spec.weak_scaled_gradient_ratio
    ):
        root = (
            "phase314b_r257_stageb_upper_objective_"
            "gradient_underpowered"
        )
        next_path = (
            "CALIBRATE_GRADIENT_BALANCED_TOPK_UPPER_OBJECTIVE_"
            "ON_TRAIN_ONLY_SPLIT"
        )
        locus = "gradient_scale"
    elif (
        cosine <= spec.conflicting_gradient_cosine
        and nmse_ratio > 1.0
    ):
        root = (
            "phase314b_r257_stageb_upper_objective_"
            "diffusion_gradient_conflict"
        )
        next_path = (
            "CALIBRATE_TIMESTEP_GATED_CONFLICT_BALANCED_"
            "UPPER_OBJECTIVE_ON_TRAIN_ONLY_SPLIT"
        )
        locus = "gradient_conflict"
    elif (
        concentration >= spec.rowmax_position_concentration
        and positive_count > 4.0
    ):
        root = (
            "phase314b_r257_stageb_rowmax_upper_objective_"
            "gradient_concentration"
        )
        next_path = (
            "CALIBRATE_TOPK_SMOOTH_UPPER_OBJECTIVE_"
            "ON_TRAIN_ONLY_SPLIT"
        )
        locus = "rowmax_concentration"
    elif linear >= spec.huber_linear_fraction:
        root = (
            "phase314b_r257_stageb_upper_objective_"
            "huber_linear_saturation"
        )
        next_path = (
            "CALIBRATE_TOPK_QUADRATIC_OR_SOFTPLUS_"
            "UPPER_OBJECTIVE_ON_TRAIN_ONLY_SPLIT"
        )
        locus = "huber_saturation"
    elif reduction >= spec.substantial_mean_excess_reduction:
        root = (
            "phase314b_r257_stageb_upper_objective_"
            "continuous_response_but_gate_gap_remains"
        )
        next_path = (
            "CALIBRATE_STRONGER_TOPK_UPPER_OBJECTIVE_"
            "WITH_FIDELITY_GATES_ON_TRAIN_ONLY_SPLIT"
        )
        locus = "insufficient_gate_crossing"
    elif reduction >= spec.minimum_mean_excess_reduction:
        root = (
            "phase314b_r257_stageb_upper_objective_"
            "weak_continuous_response"
        )
        next_path = (
            "CALIBRATE_TOPK_GRADIENT_BALANCED_UPPER_OBJECTIVE_"
            "ON_TRAIN_ONLY_SPLIT"
        )
        locus = "weak_objective_response"
    else:
        root = (
            "phase314b_r257_stageb_upper_objective_"
            "no_holdout_geometry_response"
        )
        next_path = (
            "AUDIT_DIRECT_X0_RESIDUAL_PARAMETERIZATION_AND_"
            "OBJECTIVE_LOCALIZATION_BEFORE_RECALIBRATION"
        )
        locus = "objective_response"

    return {
        "root_cause": root,
        "required_next_path": next_path,
        "primary_failure_locus": locus,
        **dict(summary),
    }


def run_mechanism_audit(
    *,
    root: Path,
    spec: Optional[MechanismAuditSpec] = None,
) -> Dict[str, Any]:
    active_spec = MechanismAuditSpec() if spec is None else spec
    active_spec.validate()
    repository_root = Path(root).resolve()

    immutable = validate_immutable_inputs(repository_root)
    logic = source_logic_audit(repository_root)
    if not logic["all_confirmed"]:
        raise MechanismAuditError(
            "Stage-B mechanism source assumptions changed"
        )

    stagea_result = immutable["stagea_result"]
    stagea_contract_payload = immutable["stagea_contract"]
    objective_contract = load_objective_contract(
        stagea_contract_payload
    )
    upstream = stagea.validate_immutable_inputs(repository_root)
    upper_gate = stagea.load_upper_gate_contract(
        upstream["staged3_contract"]
    )
    stage_d_contract, _ = staged1.load_stage_d_gate(
        repository_root / staged3.STAGE_D_GATE
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

    stageb_train, frozen_probe, _ = (
        stageb.deterministic_group_split(
            groups,
            folds=stageb_spec.group_folds,
            probe_fold=stageb_spec.probe_fold,
        )
    )
    objective_train, selection_holdout, selection_split = (
        stagea.deterministic_selection_split(
            groups,
            stageb_train,
            folds=stagea.UpperObjectiveSpec().selection_group_folds,
            holdout_fold=
                stagea.UpperObjectiveSpec().selection_holdout_fold,
        )
    )
    expected_split = stagea_result["split"]
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
        if selection_split[key] != expected_split[key]:
            raise MechanismAuditError(
                f"Stage-A selection split changed: {key}"
            )
    if np.any(
        frozen_probe & (objective_train | selection_holdout)
    ):
        raise AssertionError("frozen probe crossed train-only audit")

    condition_standardizer = stageb.fit_standardizer(
        condition[objective_train]
    )
    target_standardizer = stageb.fit_standardizer(
        target[objective_train]
    )
    diagnostic_batch = fixed_diagnostic_batch(
        condition=condition[objective_train],
        target=target[objective_train],
        condition_standardizer=condition_standardizer,
        target_standardizer=target_standardizer,
        stageb_spec=stageb_spec,
        spec=active_spec,
    )
    expected_by_lambda = {
        float(record["lambda_upper"]): record
        for record in immutable["candidate_records"]
    }

    audit_records = []
    for lambda_upper in active_spec.candidate_lambdas:
        expected = expected_by_lambda[float(lambda_upper)]
        stageb.set_deterministic_runtime(stageb_spec.seed)
        model, training, attribution = (
            replay_pilot_with_attribution(
                condition=condition[objective_train],
                target=target[objective_train],
                stageb_spec=stageb_spec,
                objective_contract=objective_contract,
                lambda_upper=float(lambda_upper),
                condition_standardizer=condition_standardizer,
                target_standardizer=target_standardizer,
                diagnostic_batch=diagnostic_batch,
                spec=active_spec,
            )
        )
        if training != expected["training"]:
            differing = sorted(
                key
                for key in set(training) | set(expected["training"])
                if training.get(key) != expected["training"].get(key)
            )
            raise MechanismAuditError(
                "instrumented replay differs from Stage A training: "
                f"lambda={lambda_upper}, keys={differing}"
            )

        train_control = stagea.train_control_audit(
            model=model,
            condition=condition[objective_train],
            target=target[objective_train],
            stageb_spec=stageb_spec,
            condition_standardizer=condition_standardizer,
            target_standardizer=target_standardizer,
        )
        if train_control != expected["train_control"]:
            raise MechanismAuditError(
                "instrumented replay train-control differs: "
                f"lambda={lambda_upper}"
            )

        one_step_audit = stagea.one_step_audit(
            model=model,
            condition=condition[selection_holdout],
            target=target[selection_holdout],
            groups=groups[selection_holdout],
            condition_name=condition_name[selection_holdout],
            stageb_spec=stageb_spec,
            upper_gate=upper_gate,
            stage_d_contract=stage_d_contract,
            historical_geometry=stageb.fit_geometry_contract(
                target[stageb_train]
            ),
            condition_standardizer=condition_standardizer,
            target_standardizer=target_standardizer,
            noise_seed=(
                stageb_spec.seed
                + active_spec.holdout_noise_seed_offset
            ),
        )
        if one_step_audit != expected["one_step"]:
            raise MechanismAuditError(
                "instrumented replay one-step audit differs: "
                f"lambda={lambda_upper}"
            )

        predictions, aggregate_sha = (
            stagec.one_step_predictions(
                model=model,
                condition=condition[selection_holdout],
                target=target[selection_holdout],
                spec=stageb_spec,
                condition_standardizer=condition_standardizer,
                target_standardizer=target_standardizer,
                noise_seed=(
                    stageb_spec.seed
                    + active_spec.holdout_noise_seed_offset
                ),
            )
        )
        if aggregate_sha != expected["one_step"][
            "prediction_sha256"
        ]:
            raise MechanismAuditError(
                "holdout prediction aggregate SHA differs"
            )
        profiles = {
            str(timestep): upper_violation_profile(
                prediction,
                upper_gate=upper_gate,
                objective_contract=objective_contract,
                top_k=active_spec.top_k,
            )
            for timestep, prediction in sorted(
                predictions.items()
            )
        }
        audit_records.append(
            {
                "lambda_upper": float(lambda_upper),
                "stagea_record": expected,
                "training_replay_exact": True,
                "train_control_replay_exact": True,
                "one_step_replay_exact": True,
                "training_attribution": attribution,
                "holdout_profiles": profiles,
            }
        )
        del model

    mechanism_summary = summarize_mechanism(audit_records)
    classification = classify_mechanism(
        summary=mechanism_summary,
        spec=active_spec,
    )
    mechanism_contract = {
        "schema": "phase314b_r257_stageb_mechanism_contract_v1",
        "candidate_lambdas":
            list(active_spec.candidate_lambdas),
        "completed_step_checkpoints":
            list(active_spec.completed_step_checkpoints),
        "diagnostic_batch_size":
            active_spec.diagnostic_batch_size,
        "diagnostic_timesteps":
            list(active_spec.diagnostic_timesteps),
        "diagnostic_batch_sha256": {
            key: value
            for key, value in diagnostic_batch.items()
            if key.endswith("_sha256")
        },
        "top_k": list(active_spec.top_k),
        "classification_thresholds": {
            "minimum_mean_excess_reduction":
                active_spec.minimum_mean_excess_reduction,
            "substantial_mean_excess_reduction":
                active_spec.substantial_mean_excess_reduction,
            "weak_scaled_gradient_ratio":
                active_spec.weak_scaled_gradient_ratio,
            "conflicting_gradient_cosine":
                active_spec.conflicting_gradient_cosine,
            "rowmax_position_concentration":
                active_spec.rowmax_position_concentration,
            "huber_linear_fraction":
                active_spec.huber_linear_fraction,
            "graph_gradient_norm_floor":
                active_spec.graph_gradient_norm_floor,
        },
        "uses_frozen_probe": False,
        "uses_reverse_sampling": False,
        "selects_new_lambda": False,
        "changes_training_objective": False,
    }
    mechanism_contract["contract_sha256"] = sha256_bytes(
        stable_json_bytes(mechanism_contract)
    )
    return {
        "phase": PHASE,
        "phase_id": PHASE_ID,
        "schema": "phase314b_r257_stageb_worker_result_v1",
        "verdict": "PASS",
        "scientific_status": "BLOCKED",
        "root_cause": classification["root_cause"],
        "required_next_path":
            classification["required_next_path"],
        "mechanism_spec": asdict(active_spec),
        "immutable_inputs": {
            "base_commit": immutable["base_commit"],
            "base_file_sha256":
                immutable["base_file_sha256"],
            "stagea_worker_identity_sha256":
                immutable["worker_identity_sha256"],
            "stagea_objective_contract_sha256":
                EXPECTED_STAGEA_OBJECTIVE_SHA256,
            "stagea_selection_sha256":
                EXPECTED_STAGEA_SELECTION_SHA256,
        },
        "source_logic_audit": logic,
        "train_view_validation": validation,
        "split": {
            "stageb_training_rows": int(np.sum(stageb_train)),
            "frozen_probe_rows": int(np.sum(frozen_probe)),
            **selection_split,
            "frozen_probe_accessed": False,
        },
        "mechanism_contract": mechanism_contract,
        "candidate_audits": audit_records,
        "mechanism_summary": mechanism_summary,
        "classification": classification,
        "selected_configuration": None,
        "train_only_recommendation": None,
        "mechanism_recommendation":
            classification["required_next_path"],
        "training_performed": True,
        "training_is_exact_stagea_replay": True,
        "new_hyperparameter_candidate_run": False,
        "new_objective_variant_run": False,
        "frozen_probe_accessed": False,
        "reverse_sampling_run": False,
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
        "required_next_path": result["required_next_path"],
        "split": result["split"],
        "mechanism_contract": result["mechanism_contract"],
        "candidate_audits": result["candidate_audits"],
        "mechanism_summary": result["mechanism_summary"],
        "classification": result["classification"],
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
        "mechanism_contract_exact": (
            left["mechanism_contract"]
            == right["mechanism_contract"]
        ),
        "candidate_audits_exact": (
            left["candidate_audits"]
            == right["candidate_audits"]
        ),
        "classification_exact": (
            left["classification"]
            == right["classification"]
        ),
    }
