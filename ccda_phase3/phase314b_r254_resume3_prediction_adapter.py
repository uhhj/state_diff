"""Pure-memory prior-prediction adapter for Phase3.14b-r2.5.4 Resume3.

This module repairs only the Resume2 assumption that ``fit_shared_prior_snapshot``
returns ``_prior_prediction_z``.  The historical fit API is left unchanged.  A
prediction fingerprint is reconstructed from the retained ``_prior_state`` by
using the exact Resume1 model-instantiation and ``predict_base_x0`` path.

No residual model is trained here and no tensor payload is serialized.
"""
from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import pickle
import random
import subprocess
from typing import Any, Callable, Dict, Iterator, Optional, Tuple

import numpy as np
import torch

from ccda_phase3.phase314b_r251_gradient_calibration import (
    instantiate_snapshot_model,
    tensor_state_sha256,
)
from ccda_phase3.phase314b_r254_resume2_functional_prior import (
    CURRENT_DEVICE_PRIOR_PREDICTION_SHA256,
    CURRENT_DEVICE_PRIOR_STATE_SHA256,
    CURRENT_DEVICE_PRIOR_Z_MSE,
    EXPECTED_CACHE_SHA256,
    EXPECTED_CONTRACT_SHA256,
    EXPECTED_CURRENT_GPU,
    EXPECTED_R253_PILOT_SHA256,
    EXPECTED_SUBMODULE_COMMIT,
    HISTORICAL_PRIOR_STATE_SHA256,
    RESUME1_AUDIT_PATH,
    RESUME1_EVIDENCE_PATH,
    RESUME1_SUMMARY_PATH,
    R253_PILOT_PATH,
    FunctionalPriorSpec,
    assert_only_allowed_worktree_paths,
    assert_paths_match_commit,
    load_json,
    prediction_sha256,
    require_ancestor,
    sha256_file,
    source_sha256 as resume2_source_sha256,
    symmetric_relative_error,
    verify_resume1_functional_contract,
)

PHASE = "Phase3.14b-r2.5.4-Resume3"
RESUME_GENERATION = 3
RESUME3_SCHEMA = "phase314b_r254_resume3_in_memory_prior_prediction_v1"

EXPECTED_BASELINE_COMMIT = "60cf3fe9b9840280a5bdcdc3bac41f11ae3530c1"
EXPECTED_RESUME2_IMPLEMENTATION_COMMIT = "b0585f56ae948bb59583d0b20d71587da731b649"
EXPECTED_RESUME1_REPORT_COMMIT = "2ae075b20be4a0e960287d05264a46882665db15"
EXPECTED_R254_IMPLEMENTATION_COMMIT = "d315ac52f193eb6c0d5fc195b2cfa229dc7391e1"
EXPECTED_R254_BLOCKED_COMMIT = "d75988089ae6ba5e3fe65e44fae28eadcf02cfdc"
EXPECTED_R253_FINAL_COMMIT = "c35c374f61f9b4dc2ca4cdaa76b12ddda71be994"

RESUME2_PREFLIGHT_PATH = "reports/phase3_14b_r254_resume2_preflight_summary.json"
RESUME2_BLOCKED_SUMMARY_PATH = "reports/phase3_14b_r254_resume2_blocked_summary.json"
RESUME2_BLOCKED_REPORT_PATH = "reports/phase3_14b_r254_resume2_blocked_report.md"
RESUME2_PILOT_PATH = "reports/phase3_14b_r254_resume2_pilot_summary.json"

RESUME3_PREFLIGHT_PATH = "reports/phase3_14b_r254_resume3_preflight_summary.json"
RESUME3_PILOT_PATH = "reports/phase3_14b_r254_resume3_pilot_summary.json"
RESUME3_BLOCKED_SUMMARY_PATH = "reports/phase3_14b_r254_resume3_blocked_summary.json"
RESUME3_BLOCKED_REPORT_PATH = "reports/phase3_14b_r254_resume3_blocked_report.md"
FINAL_SUMMARY_PATH = "reports/phase3_14b_r254_summary.json"
FINAL_REPORT_PATH = "reports/phase3_14b_r254_report.md"
ORIGINAL_R254_PREFLIGHT_PATH = "reports/phase3_14b_r254_preflight_summary.json"
ORIGINAL_R254_STANDARD_PILOT_PATH = "reports/phase3_14b_r254_pilot_summary.json"

IMMUTABLE_PATHS_AT_BASELINE = (
    "ccda_phase3/phase314b_r254_robot_proxy_attribution.py",
    "scripts/phase3_14b_r254_preflight.py",
    "scripts/phase3_14b_r254_run_pilot.py",
    "scripts/phase3_14b_r254_finalize.py",
    "scripts/phase3_14b_r254_run.sh",
    "tests/test_phase314b_r254_robot_proxy_attribution.py",
    "reports/phase3_14b_r254_preflight_summary.json",
    "reports/phase3_14b_r254_blocked_summary.json",
    "reports/phase3_14b_r254_blocked_report.md",
    "ccda_phase3/phase314b_r254_resume1_prior_determinism.py",
    "scripts/phase3_14b_r254_resume1_prior_worker.py",
    "scripts/phase3_14b_r254_resume1_preflight.py",
    "scripts/phase3_14b_r254_resume1_run_audit.py",
    "scripts/phase3_14b_r254_resume1_finalize.py",
    "scripts/phase3_14b_r254_resume1_run.sh",
    "tests/test_phase314b_r254_resume1_prior_determinism.py",
    "ccda_phase3/phase314b_r254_resume2_functional_prior.py",
    "scripts/phase3_14b_r254_resume2_preflight.py",
    "scripts/phase3_14b_r254_resume2_run_pilot.py",
    "scripts/phase3_14b_r254_resume2_finalize.py",
    "scripts/phase3_14b_r254_resume2_blocked.py",
    "scripts/phase3_14b_r254_resume2_run.sh",
    "tests/test_phase314b_r254_resume2_functional_prior.py",
    RESUME1_SUMMARY_PATH,
    RESUME1_AUDIT_PATH,
    RESUME1_EVIDENCE_PATH,
    RESUME2_PREFLIGHT_PATH,
    RESUME2_BLOCKED_SUMMARY_PATH,
    RESUME2_BLOCKED_REPORT_PATH,
)

SOURCE_PATHS = (
    "ccda_phase3/phase314b_r254_resume3_prediction_adapter.py",
    "scripts/phase3_14b_r254_resume3_preflight.py",
    "scripts/phase3_14b_r254_resume3_run_pilot.py",
    "scripts/phase3_14b_r254_resume3_finalize.py",
    "scripts/phase3_14b_r254_resume3_blocked.py",
    "scripts/phase3_14b_r254_resume3_run.sh",
    "tests/test_phase314b_r254_resume3_prediction_adapter.py",
)


@dataclass(frozen=True)
class Resume3PredictionSpec:
    prior_z_mse_absolute_tolerance: float = 1.0e-10
    prior_z_mse_relative_tolerance: float = 1.0e-8
    require_rng_restoration: bool = True

    def validate(self) -> None:
        for key, value in asdict(self).items():
            if isinstance(value, bool):
                continue
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"invalid Resume3 tolerance {key}: {value}")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def source_sha256(root: Path) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for relative in SOURCE_PATHS:
        path = Path(root) / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        result[relative] = sha256_file(path)
    return result


def git_output(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=Path(root), text=True, stderr=subprocess.STDOUT
    ).strip()


def _tensor_bytes(value: torch.Tensor) -> bytes:
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    return tensor.numpy().tobytes(order="C")


def _numpy_state_bytes(state: Tuple[Any, ...]) -> bytes:
    name, keys, position, has_gauss, cached = state
    payload = bytearray()
    payload.extend(str(name).encode("utf-8"))
    payload.extend(np.ascontiguousarray(keys).tobytes(order="C"))
    payload.extend(np.asarray([position, has_gauss], dtype=np.int64).tobytes())
    payload.extend(np.asarray([cached], dtype=np.float64).tobytes())
    return bytes(payload)


def stochastic_state_fingerprint() -> str:
    """Hash Python, NumPy, CPU/CUDA RNG state and deterministic backend flags."""
    digest = hashlib.sha256()
    digest.update(pickle.dumps(random.getstate(), protocol=4))
    digest.update(_numpy_state_bytes(np.random.get_state()))
    digest.update(_tensor_bytes(torch.get_rng_state()))
    if torch.cuda.is_available():
        for state in torch.cuda.get_rng_state_all():
            digest.update(_tensor_bytes(state))
    flags = {
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
    }
    digest.update(json.dumps(flags, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


@contextmanager
def preserve_stochastic_state() -> Iterator[None]:
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cpu_state = torch.get_rng_state().clone()
    cuda_states = (
        [value.clone() for value in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else None
    )
    deterministic_algorithms = bool(torch.are_deterministic_algorithms_enabled())
    cudnn_deterministic = bool(torch.backends.cudnn.deterministic)
    cudnn_benchmark = bool(torch.backends.cudnn.benchmark)
    cuda_matmul_allow_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
    cudnn_allow_tf32 = bool(torch.backends.cudnn.allow_tf32)
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        torch.use_deterministic_algorithms(deterministic_algorithms)
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = cudnn_benchmark
        torch.backends.cuda.matmul.allow_tf32 = cuda_matmul_allow_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_allow_tf32


def reconstruct_prior_prediction(
    snapshot: Mapping[str, Any],
    *,
    scheduler: Any,
    condition_z: torch.Tensor,
    seed: int,
    instantiate_fn: Callable[..., Any] = instantiate_snapshot_model,
) -> Tuple[str, Tuple[int, ...], str]:
    """Reproduce the Resume1 prediction fingerprint without mutating snapshot.

    Returns ``(prediction_sha256, shape, dtype)``.  The prediction tensor is
    never inserted into the snapshot and is released before this function
    returns.
    """
    state = snapshot.get("_prior_state")
    if not isinstance(state, Mapping) or not state:
        raise RuntimeError("fresh prior snapshot lacks retained _prior_state")
    if not isinstance(condition_z, torch.Tensor):
        raise TypeError("condition_z must be a torch.Tensor")
    if condition_z.ndim != 2 or condition_z.shape[0] <= 0:
        raise ValueError(f"condition_z must have shape [N,D], got {condition_z.shape}")
    if scheduler is None or not hasattr(scheduler, "alphas_cumprod"):
        raise ValueError("scheduler must expose alphas_cumprod")

    state_sha_before = tensor_state_sha256(state)
    expected_state_sha = str(snapshot.get("prior_state_sha256", ""))
    if state_sha_before != expected_state_sha:
        raise RuntimeError("retained prior state does not match snapshot SHA")

    rng_before = stochastic_state_fingerprint()
    prediction_sha = ""
    prediction_shape: Tuple[int, ...] = ()
    prediction_dtype = ""
    with preserve_stochastic_state():
        model = instantiate_fn(
            scheduler=scheduler,
            condition_dim=int(condition_z.shape[1]),
            device=condition_z.device,
            snapshot=snapshot,
            seed=int(seed),
        )
        model.eval()
        with torch.no_grad():
            prediction = model.predict_base_x0(condition_z)
        if condition_z.is_cuda:
            torch.cuda.synchronize(condition_z.device)
        prediction_np = np.ascontiguousarray(
            prediction.detach().cpu().numpy().astype(np.float32, copy=False),
            dtype=np.float32,
        )
        prediction_sha = prediction_sha256(prediction_np)
        prediction_shape = tuple(int(value) for value in prediction_np.shape)
        prediction_dtype = str(prediction_np.dtype)
        del prediction_np
        del prediction
        del model
    rng_after = stochastic_state_fingerprint()
    if rng_before != rng_after:
        raise RuntimeError("prior prediction reconstruction changed stochastic state")
    if tensor_state_sha256(state) != state_sha_before:
        raise RuntimeError("prior prediction reconstruction mutated retained state")
    return prediction_sha, prediction_shape, prediction_dtype


def validate_fresh_prior_snapshot(
    snapshot: MutableMapping[str, Any],
    *,
    scheduler: Any,
    condition_z: torch.Tensor,
    seed: int,
    spec: Resume3PredictionSpec = Resume3PredictionSpec(),
    instantiate_fn: Callable[..., Any] = instantiate_snapshot_model,
) -> Dict[str, Any]:
    """Validate the fresh prior via an exact in-memory prediction reconstruction."""
    spec.validate()
    if "_prior_prediction_z" in snapshot:
        raise RuntimeError(
            "Resume3 requires the historical snapshot API; unexpected _prior_prediction_z"
        )
    observed_state_sha = str(snapshot.get("prior_state_sha256", ""))
    if observed_state_sha != CURRENT_DEVICE_PRIOR_STATE_SHA256:
        raise RuntimeError(
            "fresh prior state SHA mismatch: "
            f"observed={observed_state_sha}, expected={CURRENT_DEVICE_PRIOR_STATE_SHA256}"
        )

    prediction_sha, prediction_shape, prediction_dtype = reconstruct_prior_prediction(
        snapshot,
        scheduler=scheduler,
        condition_z=condition_z,
        seed=int(seed),
        instantiate_fn=instantiate_fn,
    )
    if prediction_sha != CURRENT_DEVICE_PRIOR_PREDICTION_SHA256:
        raise RuntimeError(
            "fresh prior prediction SHA mismatch: "
            f"observed={prediction_sha}, expected={CURRENT_DEVICE_PRIOR_PREDICTION_SHA256}"
        )

    observed_mse = float(snapshot.get("prior_z_mse", float("nan")))
    if not math.isfinite(observed_mse):
        raise RuntimeError("fresh prior z-MSE is non-finite")
    absolute_error = abs(observed_mse - CURRENT_DEVICE_PRIOR_Z_MSE)
    relative_error = symmetric_relative_error(observed_mse, CURRENT_DEVICE_PRIOR_Z_MSE)
    if not (
        absolute_error <= spec.prior_z_mse_absolute_tolerance
        or relative_error <= spec.prior_z_mse_relative_tolerance
    ):
        raise RuntimeError(
            "fresh prior z-MSE mismatch: "
            f"observed={observed_mse}, expected={CURRENT_DEVICE_PRIOR_Z_MSE}"
        )

    result = {
        "schema": RESUME3_SCHEMA,
        "historical_cross_device_state_sha256": HISTORICAL_PRIOR_STATE_SHA256,
        "same_device_expected_state_sha256": CURRENT_DEVICE_PRIOR_STATE_SHA256,
        "observed_state_sha256": observed_state_sha,
        "state_sha_exact": True,
        "same_device_expected_prediction_sha256": CURRENT_DEVICE_PRIOR_PREDICTION_SHA256,
        "observed_prediction_sha256": prediction_sha,
        "prediction_sha_exact": True,
        "prediction_shape": list(prediction_shape),
        "prediction_dtype": prediction_dtype,
        "expected_prior_z_mse": CURRENT_DEVICE_PRIOR_Z_MSE,
        "observed_prior_z_mse": observed_mse,
        "prior_z_mse_absolute_error": absolute_error,
        "prior_z_mse_relative_error": relative_error,
        "prior_z_mse_pass": True,
        "prediction_reconstructed_from_prior_state": True,
        "prediction_tensor_persisted": False,
        "snapshot_api_changed": False,
        "rng_state_restored": True,
        "functional_prior_contract_pass": True,
    }
    snapshot["prior_prediction_sha256"] = prediction_sha
    snapshot["resume3_functional_prior_validation"] = dict(result)
    if "_prior_prediction_z" in snapshot:
        raise RuntimeError("prediction tensor leaked into fresh prior snapshot")
    return result


def augment_pilot_payload(
    payload: Mapping[str, Any],
    *,
    functional_contract: Mapping[str, Any],
    fresh_prior_validation: Mapping[str, Any],
    resume3_source_hashes: Mapping[str, str],
    resume3_preflight_path: str = RESUME3_PREFLIGHT_PATH,
) -> Dict[str, Any]:
    result = dict(payload)
    result["resume_generation"] = RESUME_GENERATION
    result["resume3_schema"] = RESUME3_SCHEMA
    result["resume3_preflight_report"] = resume3_preflight_path
    result["resume3_source_sha256"] = dict(resume3_source_hashes)
    result["functional_prior_contract"] = dict(functional_contract)
    result["fresh_prior_validation"] = dict(fresh_prior_validation)
    result["historical_exact_prior_sha_required_for_cross_device"] = False
    result["same_device_exact_prior_sha_required"] = True
    result["prior_prediction_reconstructed_in_memory"] = True
    result["prior_prediction_tensor_persisted"] = False
    result["robot_proxy_attribution_run"] = True
    result["reverse_sampling_rerun"] = False
    result["train_only_recommendation"] = None
    result["selected_configuration"] = None
    return result


def validate_resume3_pilot_contract(payload: Mapping[str, Any]) -> None:
    if int(payload.get("resume_generation", -1)) != RESUME_GENERATION:
        raise RuntimeError("Resume3 pilot generation mismatch")
    if payload.get("resume3_schema") != RESUME3_SCHEMA:
        raise RuntimeError("Resume3 pilot schema mismatch")
    functional = payload.get("functional_prior_contract")
    if not isinstance(functional, Mapping) or not bool(
        functional.get("functional_prior_contract_supported")
    ):
        raise RuntimeError("Resume3 pilot lacks committed functional-prior contract")
    fresh = payload.get("fresh_prior_validation")
    if not isinstance(fresh, Mapping) or not bool(
        fresh.get("functional_prior_contract_pass")
    ):
        raise RuntimeError("Resume3 fresh-prior validation failed")
    required = (
        "state_sha_exact",
        "prediction_sha_exact",
        "prior_z_mse_pass",
        "prediction_reconstructed_from_prior_state",
        "rng_state_restored",
    )
    for key in required:
        if not bool(fresh.get(key)):
            raise RuntimeError(f"Resume3 fresh-prior contract failed: {key}")
    if bool(fresh.get("prediction_tensor_persisted")):
        raise RuntimeError("Resume3 prediction tensor persistence is forbidden")
    shared = payload.get("shared_prior")
    if not isinstance(shared, Mapping):
        raise RuntimeError("Resume3 shared_prior is missing")
    if str(shared.get("prior_state_sha256")) != CURRENT_DEVICE_PRIOR_STATE_SHA256:
        raise RuntimeError("Resume3 shared-prior state SHA mismatch")
    if str(shared.get("prior_prediction_sha256")) != CURRENT_DEVICE_PRIOR_PREDICTION_SHA256:
        raise RuntimeError("Resume3 shared-prior prediction SHA mismatch")
    if payload.get("train_only_recommendation") is not None:
        raise RuntimeError("Resume3 must not recommend a configuration")
    if payload.get("selected_configuration") is not None:
        raise RuntimeError("Resume3 must not select a configuration")
    if bool(payload.get("reverse_sampling_rerun")):
        raise RuntimeError("Resume3 must not rerun reverse sampling")


def compact_functional_prior_contract(payload: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "resume_generation": RESUME_GENERATION,
        "schema": RESUME3_SCHEMA,
        "resume1_root_cause": payload.get("resume1_root_cause"),
        "functional_prior_contract_supported": bool(
            payload.get("functional_prior_contract_supported")
        ),
        "historical_prior_state_sha256": payload.get("historical_prior_state_sha256"),
        "current_device_prior_state_sha256": payload.get(
            "current_device_prior_state_sha256"
        ),
        "current_device_prior_prediction_sha256": payload.get(
            "current_device_prior_prediction_sha256"
        ),
        "current_device_prior_z_mse": float(
            payload.get("current_device_prior_z_mse", float("nan"))
        ),
        "same_device_exact_state_sha": bool(payload.get("same_device_exact_state_sha")),
        "same_device_exact_prediction_sha": bool(
            payload.get("same_device_exact_prediction_sha")
        ),
        "same_device_functional_equivalence": bool(
            payload.get("same_device_functional_equivalence")
        ),
        "historical_functional_fingerprint": bool(
            payload.get("historical_functional_fingerprint")
        ),
    }


def verify_resume2_blocked_contract(payload: Mapping[str, Any]) -> Dict[str, Any]:
    if payload.get("verdict") != "BLOCKED":
        raise RuntimeError("Resume2 blocked verdict mismatch")
    if payload.get("root_cause") != "phase314b_r254_resume2_execution_failed":
        raise RuntimeError("Resume2 blocked root cause mismatch")
    if payload.get("pilot_report") is not None or payload.get("pilot_sha256") is not None:
        raise RuntimeError("Resume2 unexpectedly produced a pilot")
    if not payload.get("preflight_report") or not payload.get("preflight_sha256"):
        raise RuntimeError("Resume2 preflight evidence is missing")
    if bool(payload.get("robot_proxy_attribution_interpretable")):
        raise RuntimeError("Resume2 incorrectly marked attribution interpretable")
    if payload.get("train_only_recommendation") is not None:
        raise RuntimeError("Resume2 blocked evidence contains a recommendation")
    if payload.get("selected_configuration") is not None:
        raise RuntimeError("Resume2 blocked evidence contains a selection")
    for key in (
        "reverse_sampling_rerun",
        "validation_targets_used",
        "formal_test_read",
        "formal_training",
        "idm",
        "candidate_execution",
        "phase4",
        "cps",
    ):
        if bool(payload.get(key)):
            raise RuntimeError(f"Resume2 blocked boundary crossed: {key}")
    return {
        "verdict": "BLOCKED",
        "root_cause": "phase314b_r254_resume2_execution_failed",
        "pilot_created": False,
        "robot_proxy_attribution_interpretable": False,
        "boundaries_closed": True,
    }


__all__ = [
    "CURRENT_DEVICE_PRIOR_PREDICTION_SHA256",
    "CURRENT_DEVICE_PRIOR_STATE_SHA256",
    "CURRENT_DEVICE_PRIOR_Z_MSE",
    "EXPECTED_BASELINE_COMMIT",
    "EXPECTED_CACHE_SHA256",
    "EXPECTED_CONTRACT_SHA256",
    "EXPECTED_CURRENT_GPU",
    "EXPECTED_R253_PILOT_SHA256",
    "EXPECTED_SUBMODULE_COMMIT",
    "FINAL_REPORT_PATH",
    "FINAL_SUMMARY_PATH",
    "IMMUTABLE_PATHS_AT_BASELINE",
    "ORIGINAL_R254_PREFLIGHT_PATH",
    "ORIGINAL_R254_STANDARD_PILOT_PATH",
    "RESUME1_AUDIT_PATH",
    "RESUME1_EVIDENCE_PATH",
    "RESUME1_SUMMARY_PATH",
    "RESUME2_BLOCKED_REPORT_PATH",
    "RESUME2_BLOCKED_SUMMARY_PATH",
    "RESUME2_PILOT_PATH",
    "RESUME2_PREFLIGHT_PATH",
    "RESUME3_BLOCKED_REPORT_PATH",
    "RESUME3_BLOCKED_SUMMARY_PATH",
    "RESUME3_PILOT_PATH",
    "RESUME3_PREFLIGHT_PATH",
    "RESUME3_SCHEMA",
    "R253_PILOT_PATH",
    "Resume3PredictionSpec",
    "assert_only_allowed_worktree_paths",
    "assert_paths_match_commit",
    "augment_pilot_payload",
    "compact_functional_prior_contract",
    "load_json",
    "reconstruct_prior_prediction",
    "require_ancestor",
    "resume2_source_sha256",
    "sha256_file",
    "source_sha256",
    "stochastic_state_fingerprint",
    "validate_fresh_prior_snapshot",
    "validate_resume3_pilot_contract",
    "verify_resume1_functional_contract",
    "verify_resume2_blocked_contract",
]
