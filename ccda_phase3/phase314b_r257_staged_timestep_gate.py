"""Phase3.14b-r2.5.7 Stage D timestep-gated K16 calibration.

Stage C established that gradient-balanced K16 quadratic upper-expansion
penalties produce a real train-only continuous-geometry response, but all
responsive configurations fail the frozen fidelity gates.  The strongest K16
candidate reduced row-best upper excess at t=10/25/50 by 23.5/19.2/13.7%
while increasing train-control NMSE by 3.689x.  This stage tests the registered
causal repair: apply the same raw-XY K16 quadratic objective only to low-noise
training timesteps.

The objective is evaluated only on samples whose sampled diffusion timestep is
at or below a registered hard cutoff.  Active rows are averaged conditionally;
inactive rows do not dilute the per-active-row geometry gradient.  Candidate
lambda values are calibrated at the common initial model on a deterministic
100-row batch containing every training timestep exactly once.  For each gate,
lambda targets a conditional geometry/diffusion gradient ratio of 0.50 or 1.00.

Candidate selection remains entirely inside the frozen Stage-A split:
638 objective-training rows and 236 grouped selection-holdout rows.  The
126-row frozen probe, reverse sampler, formal pilot, IDM, environment and CPS
remain closed.  A selected configuration is only a train-only recommendation
for the next stage; no full-874-row repaired model is trained here.

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
from ccda_phase3 import phase314b_r256_staged1_asymmetric_gate_audit as staged1
from ccda_phase3 import phase314b_r256_staged3_resume1_upper_gate_freeze as staged3
from ccda_phase3 import phase314b_r257_stagea_upper_objective as stagea
from ccda_phase3 import phase314b_r257_stageb_mechanism_audit as stageb_mechanism
from ccda_phase3 import phase314b_r257_stagec_topk_quadratic as stagec_topk

PHASE = "Phase3.14b-r2.5.7 Stage D"
PHASE_ID = "phase314b_r257_staged"
BASE_EVIDENCE_COMMIT = "a002246558e66029bdd6279e09c95e1303e2356c"
EXPECTED_SUBMODULE_COMMIT = "633a88752445cf5d6776ed374fdbbdb35f93050c"

STAGEC_SOURCE = "ccda_phase3/phase314b_r257_stagec_topk_quadratic.py"
STAGEC_TEST = "tests/test_phase3_14b_r257_stagec_topk_quadratic.py"
STAGEC_CONTRACT = "reports/phase3_14b_r257_stagec_topk_quadratic_contract.json"
STAGEC_TEST_GATE = "reports/phase3_14b_r257_stagec_test_gate_summary.json"
STAGEC_WORKER = "reports/phase3_14b_r257_stagec_worker_evidence.json"
STAGEC_SUMMARY = "reports/phase3_14b_r257_stagec_summary.json"
STAGEC_REPORT = "reports/phase3_14b_r257_stagec_report.md"

BASE_BOUND_FILES = (
    STAGEC_SOURCE,
    STAGEC_TEST,
    STAGEC_CONTRACT,
    STAGEC_TEST_GATE,
    STAGEC_WORKER,
    STAGEC_SUMMARY,
    STAGEC_REPORT,
)

EXPECTED_STAGEC_WORKER_SHA256 = (
    "ee221172af8367fa14570aba53f6acbb8af1d8713664b44c8986e458f1b3e754"
)
EXPECTED_STAGEC_CONTRACT_SHA256 = (
    "ccf8db5daa183b871521f1e3237192400ad04e7cb1058c5e5cac1d5b5d180deb"
)
EXPECTED_STAGEC_CONTRACT_FILE_SHA256 = (
    "bd54179b6c81702734a7485bf78cc33468257dc2bd9ba4e65f5fd37d86ded01a"
)
EXPECTED_STAGEC_SELECTION_SHA256 = (
    "858d8ca83e1bba9740829546878b7d5b2e91344a2ecefe2aea8100aecc307464"
)
EXPECTED_STAGEC_INITIAL_MODEL_SHA256 = (
    "4f1c102d9f39ba59544f5565b4c01fe03e073fca596aefdc822b707cd2f2c7fd"
)

CONTROL_CANDIDATE_ID = "control_diffusion_only"


class TimestepGateCalibrationError(RuntimeError):
    """Raised when an immutable identity or train-only boundary fails."""


@dataclass(frozen=True)
class TimestepGateSpec:
    """Pre-registered hard-timestep-gate calibration contract."""

    top_k: int = 16
    timestep_cutoffs: Tuple[int, ...] = (10, 25, 50)
    target_gradient_ratios: Tuple[float, ...] = (0.50, 1.00)

    calibration_batch_size: int = 100
    calibration_seed_offset: int = 6301
    holdout_noise_seed_offset: int = 4101
    completed_step_checkpoints: Tuple[int, ...] = (1, 100, 500, 2000, 8000)

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
    active_fraction_tolerance: float = 0.03
    row_loss_epsilon: float = 1.0e-12

    def validate(self) -> None:
        maximum_positions = stageb.FUTURE_STEPS * (stageb.BEADS - 1)
        if not 0 < int(self.top_k) <= maximum_positions:
            raise ValueError("top-k is outside ordered geometry")
        if tuple(sorted(set(self.timestep_cutoffs))) != self.timestep_cutoffs:
            raise ValueError("timestep cutoffs must be ordered and unique")
        if not self.timestep_cutoffs:
            raise ValueError("timestep cutoffs are empty")
        if any(
            int(value) < 0 or int(value) >= stageb.DiagnosticSpec().train_timesteps
            for value in self.timestep_cutoffs
        ):
            raise ValueError("timestep cutoff is outside training range")
        if tuple(sorted(set(self.target_gradient_ratios))) != self.target_gradient_ratios:
            raise ValueError("target gradient ratios must be ordered and unique")
        if not self.target_gradient_ratios:
            raise ValueError("target gradient ratios are empty")
        if any(not 0.0 < float(value) <= 1.0 for value in self.target_gradient_ratios):
            raise ValueError("target gradient ratio is outside (0,1]")
        if self.calibration_batch_size != stageb.DiagnosticSpec().train_timesteps:
            raise ValueError("calibration batch must contain each training timestep once")
        if self.completed_step_checkpoints[-1] != stageb.DiagnosticSpec().train_steps:
            raise ValueError("final checkpoint must equal frozen training steps")
        if tuple(sorted(set(self.completed_step_checkpoints))) != self.completed_step_checkpoints:
            raise ValueError("completed-step checkpoints must be ordered")
        if self.completed_step_checkpoints[0] <= 0:
            raise ValueError("checkpoint must follow at least one update")
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
            self.active_fraction_tolerance,
        ):
            if float(value) <= 0.0:
                raise ValueError("registered decision threshold is not positive")
        if self.top8_mean_excess_reduction_min < 0.0:
            raise ValueError("top-8 reduction threshold is negative")
        if not -1.0 <= self.severe_gradient_conflict_cosine <= 1.0:
            raise ValueError("gradient conflict cosine is invalid")
        if not 0.0 <= self.severe_gradient_conflict_fraction <= 1.0:
            raise ValueError("gradient conflict fraction is invalid")
        if self.row_loss_epsilon <= 0.0:
            raise ValueError("row-loss epsilon must be positive")


@dataclass(frozen=True)
class GatedCandidate:
    candidate_id: str
    timestep_cutoff: int
    top_k: int
    target_gradient_ratio: float
    lambda_upper: float
    initial_model_sha256: str
    diffusion_gradient_norm: float
    gated_geometry_gradient_norm: float
    gated_geometry_gradient_cosine: float
    gated_geometry_gradient_conflict_fraction: float
    calibration_active_count: int
    calibration_active_fraction: float

    def validate(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate id is empty")
        if self.timestep_cutoff < 0:
            raise ValueError("candidate timestep cutoff is invalid")
        if self.top_k <= 0:
            raise ValueError("candidate top-k is invalid")
        if not 0.0 < self.target_gradient_ratio <= 1.0:
            raise ValueError("candidate target ratio is invalid")
        if not np.isfinite(self.lambda_upper) or self.lambda_upper <= 0.0:
            raise ValueError("candidate lambda is invalid")
        if len(self.initial_model_sha256) != 64:
            raise ValueError("initial model SHA is malformed")
        if self.diffusion_gradient_norm <= 0.0:
            raise ValueError("diffusion gradient norm is not positive")
        if self.gated_geometry_gradient_norm <= 0.0:
            raise ValueError("gated geometry gradient norm is not positive")
        if self.calibration_active_count <= 0:
            raise ValueError("calibration gate has no active rows")
        if not 0.0 < self.calibration_active_fraction <= 1.0:
            raise ValueError("calibration active fraction is invalid")

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return asdict(self)


def git_output(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=str(root), text=True).strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_array(value: np.ndarray) -> str:
    return stagec_topk.sha256_array(value)


def jsonable(value: Any) -> Any:
    return stagec_topk.jsonable(value)


def stable_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return stagec_topk.stable_json_bytes(payload)


def atomic_write_once(path: Path, payload: bytes, *, mode: int = 0o644) -> None:
    stagec_topk.atomic_write_once(path, payload, mode=mode)


def load_json(path: Path) -> Dict[str, Any]:
    return stagec_topk.load_json(path)


def safe_stats(value: np.ndarray) -> Dict[str, float]:
    return stageb_mechanism.safe_stats(value)


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
        raise TimestepGateCalibrationError(f"Stage-C bound file changed: {relative}")
    return sha256_bytes(observed)


def worker_identity_sha(worker: Mapping[str, Any]) -> str:
    comparison = worker.get("comparison")
    if not isinstance(comparison, Mapping):
        raise TimestepGateCalibrationError("Stage-C worker comparison is missing")
    left = comparison.get("left_sha256")
    right = comparison.get("right_sha256")
    if left != right or not isinstance(left, str):
        raise TimestepGateCalibrationError("Stage-C worker identity hashes differ")
    return left


def validate_immutable_inputs(root: Path) -> Dict[str, Any]:
    repository_root = Path(root).resolve()
    bound = {
        relative: assert_base_file_bound(repository_root, relative)
        for relative in BASE_BOUND_FILES
    }
    summary = load_json(repository_root / STAGEC_SUMMARY)
    if summary.get("verdict") != "PASS":
        raise TimestepGateCalibrationError("Stage-C execution verdict is not PASS")
    if summary.get("scientific_status") != "BLOCKED":
        raise TimestepGateCalibrationError("Stage-C scientific status changed")
    if summary.get("root_cause") != (
        "phase314b_r257_stagec_topk_quadratic_fidelity_tradeoff"
    ):
        raise TimestepGateCalibrationError("Stage-C root cause changed")
    if summary.get("required_next_path") != (
        "CALIBRATE_TIMESTEP_GATED_TOPK_QUADRATIC_OBJECTIVE"
    ):
        raise TimestepGateCalibrationError("Stage-C required next path changed")
    if summary.get("selected_configuration") is not None:
        raise TimestepGateCalibrationError("Stage-C unexpectedly selected a configuration")
    if summary.get("train_only_recommendation") is not None:
        raise TimestepGateCalibrationError("Stage-C unexpectedly selected a recommendation")
    for key in (
        "frozen_probe_accessed",
        "reverse_sampling_run",
        "full_stageb_repaired_model_trained",
        "formal_pilot_run",
    ):
        if summary.get(key) is not False:
            raise TimestepGateCalibrationError(f"Stage-C boundary changed: {key}")
    if summary.get("control_replay_exact") is not True:
        raise TimestepGateCalibrationError("Stage-C control replay was not exact")

    worker = load_json(repository_root / STAGEC_WORKER)
    if worker.get("workers_exact") is not True:
        raise TimestepGateCalibrationError("Stage-C workers were not exact")
    observed_worker_sha = worker_identity_sha(worker)
    if observed_worker_sha != EXPECTED_STAGEC_WORKER_SHA256:
        raise TimestepGateCalibrationError("Stage-C worker identity SHA changed")
    worker_result = worker.get("worker_result")
    if not isinstance(worker_result, Mapping):
        raise TimestepGateCalibrationError("Stage-C worker result is missing")
    calibration_contract = worker_result.get("calibration_contract")
    if not isinstance(calibration_contract, Mapping):
        raise TimestepGateCalibrationError("Stage-C calibration contract is missing")
    if calibration_contract.get("contract_sha256") != EXPECTED_STAGEC_CONTRACT_SHA256:
        raise TimestepGateCalibrationError("Stage-C internal contract SHA changed")
    if worker_result.get("selected_configuration") is not None:
        raise TimestepGateCalibrationError("Stage-C worker selected a configuration")
    if worker_result.get("train_only_recommendation") is not None:
        raise TimestepGateCalibrationError("Stage-C worker selected a recommendation")
    if worker_result.get("frozen_probe_accessed") is not False:
        raise TimestepGateCalibrationError("Stage-C worker accessed frozen probe")
    selection = worker_result.get("selection")
    if not isinstance(selection, Mapping):
        raise TimestepGateCalibrationError("Stage-C selection is missing")
    if selection.get("selection_sha256") != EXPECTED_STAGEC_SELECTION_SHA256:
        raise TimestepGateCalibrationError("Stage-C selection SHA changed")
    if selection.get("selected_configuration") is not None:
        raise TimestepGateCalibrationError("Stage-C selection is no longer empty")
    candidate_records = selection.get("candidate_records")
    if not isinstance(candidate_records, list) or len(candidate_records) != 7:
        raise TimestepGateCalibrationError("Stage-C candidate population changed")
    control = candidate_records[0]
    if control.get("candidate_id") != stagec_topk.CONTROL_CANDIDATE_ID:
        raise TimestepGateCalibrationError("Stage-C control ordering changed")
    best = worker_result.get("classification", {}).get(
        "best_candidate_by_t10_continuous_excess"
    )
    if best != "topk16_r1p00":
        raise TimestepGateCalibrationError("Stage-C best continuous candidate changed")
    best_record = next(
        record for record in candidate_records if record.get("candidate_id") == best
    )
    if best_record.get("continuous_geometry_pass") is not True:
        raise TimestepGateCalibrationError("Stage-C best candidate lost continuous pass")
    if best_record.get("fidelity_pass") is not False:
        raise TimestepGateCalibrationError("Stage-C best candidate fidelity status changed")
    if best_record.get("binary_pass") is not False:
        raise TimestepGateCalibrationError("Stage-C best candidate binary status changed")

    contract_path = repository_root / STAGEC_CONTRACT
    if sha256_file(contract_path) != EXPECTED_STAGEC_CONTRACT_FILE_SHA256:
        raise TimestepGateCalibrationError("Stage-C contract file SHA changed")
    contract = load_json(contract_path)
    if contract.get("contract_sha256") != EXPECTED_STAGEC_CONTRACT_SHA256:
        raise TimestepGateCalibrationError("Stage-C contract payload changed")
    if contract.get("selection_sha256") != EXPECTED_STAGEC_SELECTION_SHA256:
        raise TimestepGateCalibrationError("Stage-C contract selection SHA changed")
    if contract.get("selected_configuration") is not None:
        raise TimestepGateCalibrationError("Stage-C contract selected a configuration")
    if contract.get("uses_frozen_probe") is not False:
        raise TimestepGateCalibrationError("Stage-C contract used frozen probe")
    if contract.get("runs_reverse_sampling") is not False:
        raise TimestepGateCalibrationError("Stage-C contract ran reverse sampling")
    initial_model = contract.get("calibration", {}).get("initial_model_sha256")
    if initial_model != EXPECTED_STAGEC_INITIAL_MODEL_SHA256:
        raise TimestepGateCalibrationError("Stage-C initial model SHA changed")

    upstream = stagec_topk.validate_immutable_inputs(repository_root)
    return {
        "base_commit": BASE_EVIDENCE_COMMIT,
        "base_file_sha256": bound,
        "stagec_summary": summary,
        "stagec_worker": worker,
        "stagec_worker_result": worker_result,
        "stagec_contract": contract,
        "candidate_records": candidate_records,
        "control_record": control,
        "worker_identity_sha256": observed_worker_sha,
        "upstream_immutable": upstream,
    }


def source_logic_audit(root: Path) -> Dict[str, Any]:
    repository_root = Path(root).resolve()
    stageb_text = (repository_root / stageb_mechanism.STAGEA_SOURCE).read_text(
        encoding="utf-8"
    )
    stagec_text = (repository_root / STAGEC_SOURCE).read_text(encoding="utf-8")
    checks = {
        "model_directly_predicts_x0": (
            "predicted_x0_z" in stageb_text and "target_standardizer" in stageb_text
        ),
        "stagec_uses_k16_candidate": (
            "top_k_values: Tuple[int, ...] = (8, 16)" in stagec_text
        ),
        "stagec_uses_quadratic_penalty": (
            "row_loss = torch.mean(selected * selected, dim=1)" in stagec_text
        ),
        "stagec_calibrates_gradient_ratio": (
            "* diffusion_norm" in stagec_text and "/ geometry_norm" in stagec_text
        ),
        "stagec_control_has_exact_gate": (
            "Stage-C control training differs from Stage A" in stagec_text
        ),
        "model_has_no_dropout_or_batchnorm": (
            '"model_has_no_dropout"' in (repository_root / stageb_mechanism.STAGEA_SOURCE).read_text(encoding="utf-8")
            or "nn.Dropout" not in stageb_text
        ),
    }
    return {
        "checks": checks,
        "all_confirmed": bool(all(checks.values())),
        "source_sha256": {
            stageb_mechanism.STAGEA_SOURCE: sha256_file(
                repository_root / stageb_mechanism.STAGEA_SOURCE
            ),
            STAGEC_SOURCE: sha256_file(repository_root / STAGEC_SOURCE),
        },
        "gate_type": "hard_low_noise_timestep_cutoff",
        "gate_normalization": "conditional_mean_over_active_rows",
        "top_k": 16,
    }


def gate_weights_torch(timestep: Any, cutoff: int, *, dtype: Any) -> Any:
    return (timestep <= int(cutoff)).to(dtype)


def gated_topk_quadratic_terms_torch(
    predicted_x0_z: Any,
    timestep: Any,
    *,
    target_standardizer: stageb.ArrayStandardizer,
    objective_contract: stagea.UpperObjectiveContract,
    top_k: int,
    timestep_cutoff: int,
) -> Dict[str, Any]:
    torch, _ = stageb._torch_imports()
    base = stagec_topk.topk_quadratic_terms_torch(
        predicted_x0_z,
        target_standardizer=target_standardizer,
        objective_contract=objective_contract,
        top_k=int(top_k),
    )
    weights = gate_weights_torch(
        timestep,
        int(timestep_cutoff),
        dtype=predicted_x0_z.dtype,
    )
    active_count = torch.sum(weights)
    denominator = torch.clamp(active_count, min=1.0)
    total = torch.sum(base["row_loss"] * weights) / denominator
    return {
        **base,
        "ungated_total": base["total"],
        "total": total,
        "gate_weights": weights,
        "active_count": active_count,
        "active_fraction": torch.mean(weights),
        "zero_active": active_count <= 0.0,
        "timestep_cutoff": int(timestep_cutoff),
    }


def stratified_calibration_batch(
    *,
    condition: np.ndarray,
    target: np.ndarray,
    condition_standardizer: stageb.ArrayStandardizer,
    target_standardizer: stageb.ArrayStandardizer,
    stageb_spec: stageb.DiagnosticSpec,
    spec: TimestepGateSpec,
) -> Dict[str, Any]:
    rows = int(condition.shape[0])
    if rows < spec.calibration_batch_size:
        raise ValueError("objective-training population is too small")
    rng = np.random.RandomState(stageb_spec.seed + spec.calibration_seed_offset)
    indices = rng.permutation(rows)[: spec.calibration_batch_size].astype(np.int64)
    timesteps = np.arange(stageb_spec.train_timesteps, dtype=np.int64)
    if timesteps.shape[0] != spec.calibration_batch_size:
        raise AssertionError("calibration timestep shape changed")
    noise = rng.standard_normal(
        (spec.calibration_batch_size, stageb.FUTURE_STEPS, stageb.CABLE_DIM)
    ).astype(np.float32)
    condition_z = condition_standardizer.normalize(condition[indices]).astype(np.float32)
    target_z = target_standardizer.normalize(target[indices]).astype(np.float32)
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


def fixed_batch_gated_gradients(
    *,
    model: Any,
    calibration_batch: Mapping[str, Any],
    target_standardizer: stageb.ArrayStandardizer,
    objective_contract: stagea.UpperObjectiveContract,
    top_k: int,
    timestep_cutoff: int,
    lambda_upper: float,
    row_loss_epsilon: float,
) -> Dict[str, Any]:
    torch, _ = stageb._torch_imports()
    device = torch.device("cuda:0")
    noisy = torch.as_tensor(calibration_batch["noisy_z"], dtype=torch.float32, device=device)
    timestep = torch.as_tensor(calibration_batch["timesteps"], dtype=torch.long, device=device)
    condition = torch.as_tensor(
        calibration_batch["condition_z"], dtype=torch.float32, device=device
    )
    clean = torch.as_tensor(calibration_batch["target_z"], dtype=torch.float32, device=device)
    named_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    parameters = [parameter for _name, parameter in named_parameters]

    predicted_diffusion = model(noisy, timestep, condition)
    diffusion_loss = torch.mean((predicted_diffusion - clean) ** 2)
    diffusion_gradients = torch.autograd.grad(
        diffusion_loss,
        parameters,
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )

    predicted_geometry = model(noisy, timestep, condition)
    geometry = gated_topk_quadratic_terms_torch(
        predicted_geometry,
        timestep,
        target_standardizer=target_standardizer,
        objective_contract=objective_contract,
        top_k=int(top_k),
        timestep_cutoff=int(timestep_cutoff),
    )
    geometry_gradients = torch.autograd.grad(
        geometry["total"],
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

    weights = geometry["gate_weights"].detach().cpu().numpy().astype(np.float64)
    active = weights > 0.0
    row_loss = geometry["row_loss"].detach().cpu().numpy()
    indices = geometry["selected_indices"].detach().cpu().numpy()
    selected = geometry["selected_excess"].detach().cpu().numpy()
    if not np.any(active):
        raise TimestepGateCalibrationError("calibration gate has no active rows")
    model.zero_grad(set_to_none=True)
    return {
        "diffusion_loss": float(diffusion_loss.detach().cpu()),
        "geometry_loss": float(geometry["total"].detach().cpu()),
        "ungated_geometry_loss": float(geometry["ungated_total"].detach().cpu()),
        "gradient": gradient,
        "gate": {
            "timestep_cutoff": int(timestep_cutoff),
            "active_count": int(np.sum(active)),
            "active_fraction": float(np.mean(active)),
            "expected_active_fraction": float((int(timestep_cutoff) + 1) / 100.0),
            "active_timestep_min": int(np.min(np.asarray(calibration_batch["timesteps"])[active])),
            "active_timestep_max": int(np.max(np.asarray(calibration_batch["timesteps"])[active])),
        },
        "row_contribution": stagec_topk.row_contribution_profile(
            row_loss[active], epsilon=float(row_loss_epsilon)
        ),
        "selected_positions": stagec_topk.selected_position_profile(indices[active]),
        "selected_excess": safe_stats(selected[active]),
        "selected_excess_sha256": sha256_array(selected[active]),
        "predicted_x0_sha256": sha256_array(
            predicted_geometry.detach().cpu().numpy()
        ),
    }


def cutoff_token(value: int) -> str:
    return str(int(value))


def ratio_token(value: float) -> str:
    return f"{float(value):.2f}".replace(".", "p").replace("-", "m")


def calibrate_candidates(
    *,
    stageb_spec: stageb.DiagnosticSpec,
    calibration_batch: Mapping[str, Any],
    target_standardizer: stageb.ArrayStandardizer,
    objective_contract: stagea.UpperObjectiveContract,
    spec: TimestepGateSpec,
) -> Tuple[List[GatedCandidate], Dict[str, Any]]:
    stageb.set_deterministic_runtime(stageb_spec.seed)
    model = stageb.make_model(stageb_spec).to("cuda:0")
    initial_model_sha = stageb.tensor_state_sha256(model)
    if initial_model_sha != EXPECTED_STAGEC_INITIAL_MODEL_SHA256:
        raise TimestepGateCalibrationError("common initial model SHA changed")
    raw_records: MutableMapping[str, Any] = {}
    candidates: List[GatedCandidate] = []
    for cutoff in spec.timestep_cutoffs:
        raw = fixed_batch_gated_gradients(
            model=model,
            calibration_batch=calibration_batch,
            target_standardizer=target_standardizer,
            objective_contract=objective_contract,
            top_k=spec.top_k,
            timestep_cutoff=int(cutoff),
            lambda_upper=1.0,
            row_loss_epsilon=spec.row_loss_epsilon,
        )
        diffusion_norm = float(raw["gradient"]["diffusion_norm"])
        geometry_norm = float(raw["gradient"]["geometry_norm"])
        if (
            not np.isfinite(diffusion_norm)
            or not np.isfinite(geometry_norm)
            or diffusion_norm <= 0.0
            or geometry_norm <= 0.0
        ):
            raise TimestepGateCalibrationError(
                f"invalid initial gradient norms for cutoff={cutoff}"
            )
        raw_records[str(cutoff)] = raw
        for target_ratio in spec.target_gradient_ratios:
            lambda_upper = float(target_ratio) * diffusion_norm / geometry_norm
            candidate_id = (
                f"k16_t{cutoff_token(int(cutoff))}_"
                f"r{ratio_token(float(target_ratio))}"
            )
            candidate = GatedCandidate(
                candidate_id=candidate_id,
                timestep_cutoff=int(cutoff),
                top_k=int(spec.top_k),
                target_gradient_ratio=float(target_ratio),
                lambda_upper=float(lambda_upper),
                initial_model_sha256=initial_model_sha,
                diffusion_gradient_norm=diffusion_norm,
                gated_geometry_gradient_norm=geometry_norm,
                gated_geometry_gradient_cosine=float(raw["gradient"]["cosine"]),
                gated_geometry_gradient_conflict_fraction=float(
                    raw["gradient"]["conflict_fraction_among_active"]
                ),
                calibration_active_count=int(raw["gate"]["active_count"]),
                calibration_active_fraction=float(raw["gate"]["active_fraction"]),
            )
            candidate.validate()
            candidates.append(candidate)
    del model
    payload = {
        "initial_model_sha256": initial_model_sha,
        "raw_cutoff_records": raw_records,
        "candidate_count": len(candidates),
        "candidate_order": [candidate.candidate_id for candidate in candidates],
        "candidates": [candidate.to_dict() for candidate in candidates],
    }
    return candidates, {
        **payload,
        "calibration_sha256": sha256_bytes(stable_json_bytes(payload)),
    }


def train_gated_candidate(
    *,
    condition: np.ndarray,
    target: np.ndarray,
    stageb_spec: stageb.DiagnosticSpec,
    objective_contract: stagea.UpperObjectiveContract,
    condition_standardizer: stageb.ArrayStandardizer,
    target_standardizer: stageb.ArrayStandardizer,
    candidate: GatedCandidate,
    calibration_batch: Mapping[str, Any],
    spec: TimestepGateSpec,
) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
    torch, _ = stageb._torch_imports()
    candidate.validate()
    device = torch.device("cuda:0")
    model = stageb.make_model(stageb_spec).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=stageb_spec.learning_rate,
        weight_decay=stageb_spec.weight_decay,
    )
    scheduler = stageb.scheduler_arrays(stageb_spec)
    alpha_bar = torch.as_tensor(scheduler["alpha_bar"], dtype=torch.float32, device=device)
    condition_z = torch.as_tensor(
        condition_standardizer.normalize(condition), dtype=torch.float32, device=device
    )
    target_z = torch.as_tensor(
        target_standardizer.normalize(target), dtype=torch.float32, device=device
    )
    initial_model_sha = stageb.tensor_state_sha256(model)
    initial_optimizer_sha = stageb.optimizer_state_sha256(optimizer)
    if initial_model_sha != candidate.initial_model_sha256:
        raise TimestepGateCalibrationError("candidate initial model SHA changed")

    generator = torch.Generator(device=device)
    generator.manual_seed(stageb_spec.seed + 17)
    total_values: List[float] = []
    diffusion_values: List[float] = []
    geometry_values: List[float] = []
    gradient_values: List[float] = []
    clipping_values: List[bool] = []
    active_counts: List[int] = []
    active_fractions: List[float] = []
    zero_active_values: List[bool] = []
    violation_rate_values: List[float] = []
    maximum_excess_values: List[float] = []
    exposure = np.zeros(condition.shape[0], dtype=np.int64)
    timestep_exposure = np.zeros(stageb_spec.train_timesteps, dtype=np.int64)
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
            (stageb_spec.batch_size, stageb.FUTURE_STEPS, stageb.CABLE_DIM),
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        clean = target_z[indices]
        alpha = alpha_bar[timestep].reshape(-1, 1, 1)
        noisy = torch.sqrt(alpha) * clean + torch.sqrt(1.0 - alpha) * noise
        predicted = model(noisy, timestep, condition_z[indices])
        diffusion_loss = torch.mean((predicted - clean) ** 2)
        geometry = gated_topk_quadratic_terms_torch(
            predicted,
            timestep,
            target_standardizer=target_standardizer,
            objective_contract=objective_contract,
            top_k=candidate.top_k,
            timestep_cutoff=candidate.timestep_cutoff,
        )
        loss = diffusion_loss + candidate.lambda_upper * geometry["total"]

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        gradient_float = float(gradient_norm.detach().cpu())
        active_count = int(geometry["active_count"].detach().cpu())
        active_fraction = float(geometry["active_fraction"].detach().cpu())
        total_values.append(float(loss.detach().cpu()))
        diffusion_values.append(float(diffusion_loss.detach().cpu()))
        geometry_values.append(float(geometry["total"].detach().cpu()))
        gradient_values.append(gradient_float)
        clipping_values.append(bool(gradient_float > 10.0))
        active_counts.append(active_count)
        active_fractions.append(active_fraction)
        zero_active_values.append(active_count == 0)
        violation_rate_values.append(float(geometry["row_violation_rate"].detach().cpu()))
        maximum_excess_values.append(float(geometry["maximum_excess"].detach().cpu()))
        index_np = indices.detach().cpu().numpy()
        timestep_np = timestep.detach().cpu().numpy()
        np.add.at(exposure, index_np, 1)
        np.add.at(timestep_exposure, timestep_np, 1)

        completed_step = step_index + 1
        if completed_step in spec.completed_step_checkpoints:
            checkpoint_records[str(completed_step)] = fixed_batch_gated_gradients(
                model=model,
                calibration_batch=calibration_batch,
                target_standardizer=target_standardizer,
                objective_contract=objective_contract,
                top_k=candidate.top_k,
                timestep_cutoff=candidate.timestep_cutoff,
                lambda_upper=candidate.lambda_upper,
                row_loss_epsilon=spec.row_loss_epsilon,
            )

    model.eval()
    total_array = np.asarray(total_values, dtype=np.float64)
    diffusion_array = np.asarray(diffusion_values, dtype=np.float64)
    geometry_array = np.asarray(geometry_values, dtype=np.float64)
    gradient_array = np.asarray(gradient_values, dtype=np.float64)
    exposure_array = np.asarray(exposure, dtype=np.int64)
    timestep_exposure_array = np.asarray(timestep_exposure, dtype=np.int64)
    active_count_array = np.asarray(active_counts, dtype=np.int64)
    active_fraction_array = np.asarray(active_fractions, dtype=np.float64)
    training_record = {
        "candidate_id": candidate.candidate_id,
        "lambda_upper": float(candidate.lambda_upper),
        "timestep_cutoff": int(candidate.timestep_cutoff),
        "top_k": int(candidate.top_k),
        "initial_model_sha256": initial_model_sha,
        "final_model_sha256": stageb.tensor_state_sha256(model),
        "initial_optimizer_sha256": initial_optimizer_sha,
        "final_optimizer_sha256": stageb.optimizer_state_sha256(optimizer),
        "total_loss_history_sha256": sha256_array(total_array),
        "diffusion_loss_history_sha256": sha256_array(diffusion_array),
        "geometry_loss_history_sha256": sha256_array(geometry_array),
        "gradient_history_sha256": sha256_array(gradient_array),
        "source_exposure_sha256": sha256_array(exposure_array),
        "timestep_exposure_sha256": sha256_array(timestep_exposure_array),
        "active_count_history_sha256": sha256_array(active_count_array),
        "active_fraction_history_sha256": sha256_array(active_fraction_array),
        "total_loss_first": float(total_array[0]),
        "total_loss_final": float(total_array[-1]),
        "total_loss_tail_mean": float(np.mean(total_array[-100:])),
        "diffusion_loss_tail_mean": float(np.mean(diffusion_array[-100:])),
        "geometry_loss_tail_mean": float(np.mean(geometry_array[-100:])),
        "gradient_tail_mean": float(np.mean(gradient_array[-100:])),
        "batch_row_violation_rate_tail_mean": float(
            np.mean(violation_rate_values[-100:])
        ),
        "maximum_log_excess_tail_max": float(np.max(maximum_excess_values[-100:])),
        "loss_finite": bool(
            np.all(np.isfinite(total_array))
            and np.all(np.isfinite(diffusion_array))
            and np.all(np.isfinite(geometry_array))
        ),
        "gradient_finite": bool(np.all(np.isfinite(gradient_array))),
        "training_rows": int(condition.shape[0]),
        "training_steps": int(stageb_spec.train_steps),
    }
    expected_active_fraction = (candidate.timestep_cutoff + 1) / stageb_spec.train_timesteps
    diagnostics = {
        "candidate": candidate.to_dict(),
        "gradient_clip_count": int(np.sum(clipping_values)),
        "gradient_clip_frequency": float(np.mean(clipping_values)),
        "gate_active_count": safe_stats(active_count_array),
        "gate_active_fraction": safe_stats(active_fraction_array),
        "expected_gate_active_fraction": float(expected_active_fraction),
        "observed_global_active_fraction": float(
            np.sum(timestep_exposure_array[: candidate.timestep_cutoff + 1])
            / np.sum(timestep_exposure_array)
        ),
        "zero_active_batch_count": int(np.sum(zero_active_values)),
        "zero_active_batch_fraction": float(np.mean(zero_active_values)),
        "timestep_exposure_sha256": sha256_array(timestep_exposure_array),
        "timestep_exposure": timestep_exposure_array.tolist(),
        "checkpoint_records": checkpoint_records,
        "checkpoint_record_sha256": sha256_bytes(stable_json_bytes(checkpoint_records)),
    }
    return model, training_record, diagnostics


def reduction(candidate: float, control: float) -> float:
    return stagec_topk.reduction(candidate, control)


def ratio(candidate: float, control: float) -> float:
    return stagec_topk.ratio(candidate, control)


def final_checkpoint(diagnostics: Mapping[str, Any]) -> Mapping[str, Any]:
    checkpoints = diagnostics["checkpoint_records"]
    key = str(max(int(value) for value in checkpoints))
    return checkpoints[key]


def selection_record(
    *,
    candidate: GatedCandidate,
    training: Mapping[str, Any],
    training_diagnostics: Mapping[str, Any],
    train_control: Mapping[str, Any],
    one_step: Mapping[str, Any],
    profiles: Mapping[str, Any],
    control: Mapping[str, Any],
    spec: TimestepGateSpec,
) -> Dict[str, Any]:
    continuous: MutableMapping[str, float] = {}
    p95_ratios: MutableMapping[str, float] = {}
    positive_count_ratios: MutableMapping[str, float] = {}
    top8_reductions: MutableMapping[str, float] = {}
    for timestep in ("10", "25", "50"):
        candidate_profile = profiles[timestep]
        control_profile = control["holdout_profiles"][timestep]
        continuous[timestep] = reduction(
            candidate_profile["row_best_max_log_excess"]["mean"],
            control_profile["row_best_max_log_excess"]["mean"],
        )
        p95_ratios[timestep] = ratio(
            candidate_profile["row_best_max_log_excess"]["p95"],
            control_profile["row_best_max_log_excess"]["p95"],
        )
        positive_count_ratios[timestep] = ratio(
            candidate_profile["candidate_positive_segment_count"]["mean"],
            control_profile["candidate_positive_segment_count"]["mean"],
        )
        top8_reductions[timestep] = reduction(
            candidate_profile["top_k_mean_log_excess"]["8"]["mean"],
            control_profile["top_k_mean_log_excess"]["8"]["mean"],
        )

    continuous_gates = {
        "t10_mean_excess_reduction": (
            continuous["10"] >= spec.t10_mean_excess_reduction_min
        ),
        "t25_mean_excess_reduction": (
            continuous["25"] >= spec.t25_mean_excess_reduction_min
        ),
        "t50_mean_excess_reduction": (
            continuous["50"] >= spec.t50_mean_excess_reduction_min
        ),
        "all_top8_mean_excess_reduction": all(
            value > spec.top8_mean_excess_reduction_min
            for value in top8_reductions.values()
        ),
        "all_p95_nonincreasing": all(
            value <= spec.p95_excess_ratio_max for value in p95_ratios.values()
        ),
        "all_positive_count_nonincreasing": all(
            value <= spec.positive_segment_count_ratio_max
            for value in positive_count_ratios.values()
        ),
    }

    one_step_ratios = {
        timestep: ratio(
            one_step["timesteps"][timestep]["normalized_mse"],
            control["one_step"]["timesteps"][timestep]["normalized_mse"],
        )
        for timestep in ("10", "25", "50")
    }
    control_ratio = ratio(
        train_control["normalized_mse"], control["train_control"]["normalized_mse"]
    )
    fidelity_gates = {
        "loss_finite": bool(training["loss_finite"]),
        "gradient_finite": bool(training["gradient_finite"]),
        "train_control_nmse_ratio": (
            control_ratio <= spec.train_control_nmse_ratio_max
        ),
        "all_one_step_nmse_ratios": all(
            value <= spec.one_step_nmse_ratio_max
            for value in one_step_ratios.values()
        ),
    }

    binary_rate = {
        timestep: float(profiles[timestep]["row_any_rate"])
        for timestep in ("10", "25", "50")
    }
    binary_gates = {
        "t10_upper_row_any": binary_rate["10"] >= spec.binary_t10_upper_row_min,
        "t25_upper_row_any": binary_rate["25"] >= spec.binary_t25_upper_row_min,
        "t50_upper_row_any": binary_rate["50"] >= spec.binary_t50_upper_row_min,
    }

    checkpoint = final_checkpoint(training_diagnostics)
    row_contribution = checkpoint["row_contribution"]
    active_fraction_error = abs(
        float(training_diagnostics["observed_global_active_fraction"])
        - float(training_diagnostics["expected_gate_active_fraction"])
    )
    stability_gates = {
        "gradient_clip_frequency": (
            float(training_diagnostics["gradient_clip_frequency"])
            <= spec.gradient_clip_frequency_max
        ),
        "largest_row_contribution": (
            float(row_contribution["largest_row_contribution_fraction"])
            <= spec.largest_row_contribution_max
        ),
        "top_five_percent_contribution": (
            float(row_contribution["top_five_percent_contribution_fraction"])
            <= spec.top_five_percent_contribution_max
        ),
        "active_fraction_contract": (
            active_fraction_error <= spec.active_fraction_tolerance
        ),
    }

    continuous_pass = bool(all(continuous_gates.values()))
    fidelity_pass = bool(all(fidelity_gates.values()))
    binary_pass = bool(all(binary_gates.values()))
    stability_pass = bool(all(stability_gates.values()))
    return {
        "candidate_id": candidate.candidate_id,
        "candidate": candidate.to_dict(),
        "training": dict(training),
        "training_diagnostics": dict(training_diagnostics),
        "train_control": dict(train_control),
        "one_step": dict(one_step),
        "holdout_profiles": dict(profiles),
        "continuous_geometry_gates": continuous_gates,
        "fidelity_gates": fidelity_gates,
        "binary_gates": binary_gates,
        "stability_gates": stability_gates,
        "continuous_geometry_pass": continuous_pass,
        "fidelity_pass": fidelity_pass,
        "binary_pass": binary_pass,
        "stability_pass": stability_pass,
        "eligible_for_selection": bool(
            continuous_pass and fidelity_pass and binary_pass and stability_pass
        ),
        "relative_to_control": {
            "train_control_nmse_ratio": control_ratio,
            "t10_nmse_ratio": one_step_ratios["10"],
            "t25_nmse_ratio": one_step_ratios["25"],
            "t50_nmse_ratio": one_step_ratios["50"],
            "p95_excess_ratio": p95_ratios,
            "positive_segment_count_ratio": positive_count_ratios,
        },
        "continuous_response": {
            "t10_mean_excess_reduction": continuous["10"],
            "t25_mean_excess_reduction": continuous["25"],
            "t50_mean_excess_reduction": continuous["50"],
            "t10_top8_reduction": top8_reductions["10"],
            "t25_top8_reduction": top8_reductions["25"],
            "t50_top8_reduction": top8_reductions["50"],
        },
        "binary_rate": binary_rate,
        "active_fraction_error": active_fraction_error,
    }


def select_configuration(records: Sequence[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    eligible = [record for record in records if record.get("eligible_for_selection") is True]
    if not eligible:
        return None
    selected = sorted(
        eligible,
        key=lambda record: (
            -float(record["binary_rate"]["10"]),
            float(record["holdout_profiles"]["10"]["row_best_max_log_excess"]["mean"]),
            int(record["candidate"]["timestep_cutoff"]),
            float(record["candidate"]["target_gradient_ratio"]),
            str(record["candidate_id"]),
        ),
    )[0]
    return {
        "candidate_id": selected["candidate_id"],
        "timestep_cutoff": int(selected["candidate"]["timestep_cutoff"]),
        "top_k": int(selected["candidate"]["top_k"]),
        "target_gradient_ratio": float(selected["candidate"]["target_gradient_ratio"]),
        "lambda_upper": float(selected["candidate"]["lambda_upper"]),
        "pilot_model_sha256": selected["training"]["final_model_sha256"],
        "selection_rule": (
            "highest train-only t10 binary upper rate, then lowest t10 row-best "
            "log excess, then narrowest timestep cutoff, then lowest target ratio"
        ),
        "selection_record_sha256": sha256_bytes(stable_json_bytes(selected)),
    }


def candidate_conflict_severe(
    record: Mapping[str, Any], spec: TimestepGateSpec
) -> bool:
    gradient = final_checkpoint(record["training_diagnostics"])["gradient"]
    return bool(
        float(gradient["cosine"]) <= spec.severe_gradient_conflict_cosine
        or float(gradient["conflict_fraction_among_active"])
        >= spec.severe_gradient_conflict_fraction
    )


def classify_selection(
    *,
    records: Sequence[Mapping[str, Any]],
    selected: Optional[Mapping[str, Any]],
    spec: TimestepGateSpec,
) -> Dict[str, Any]:
    if not records:
        raise ValueError("candidate records are empty")
    if selected is not None:
        root = (
            "phase314b_r257_staged_timestep_gated_k16_"
            "train_only_configuration_selected"
        )
        next_path = (
            "RETRAIN_SELECTED_TIMESTEP_GATED_K16_OBJECTIVE_ON_FULL_"
            "STAGEB_TRAIN_AND_OPEN_FROZEN_PROBE"
        )
        locus = "none"
    else:
        continuous = [record for record in records if record["continuous_geometry_pass"]]
        fidelity = [record for record in records if record["fidelity_pass"]]
        continuous_fidelity = [
            record
            for record in records
            if record["continuous_geometry_pass"] and record["fidelity_pass"]
        ]
        if not continuous:
            if fidelity:
                root = (
                    "phase314b_r257_staged_timestep_gating_restored_fidelity_"
                    "but_geometry_underpowered"
                )
                next_path = (
                    "CALIBRATE_LOW_NOISE_K16_CURRICULUM_WITH_INTERMEDIATE_"
                    "GRADIENT_RATIO_ON_TRAIN_ONLY_SPLIT"
                )
                locus = "geometry_underpowered"
            else:
                root = (
                    "phase314b_r257_staged_timestep_gated_k16_"
                    "no_geometry_or_fidelity_solution"
                )
                next_path = (
                    "CALIBRATE_CONFLICT_PROJECTED_LOW_NOISE_K16_OBJECTIVE_"
                    "ON_TRAIN_ONLY_SPLIT"
                )
                locus = "joint_tradeoff"
        elif not continuous_fidelity:
            best = max(
                continuous,
                key=lambda record: (
                    float(record["continuous_response"]["t10_mean_excess_reduction"]),
                    -float(record["relative_to_control"]["train_control_nmse_ratio"]),
                ),
            )
            if candidate_conflict_severe(best, spec):
                root = (
                    "phase314b_r257_staged_timestep_gated_k16_"
                    "persistent_gradient_conflict"
                )
                next_path = (
                    "CALIBRATE_CONFLICT_PROJECTED_LOW_NOISE_K16_OBJECTIVE_"
                    "ON_TRAIN_ONLY_SPLIT"
                )
                locus = "gradient_conflict"
            else:
                root = (
                    "phase314b_r257_staged_timestep_gated_k16_"
                    "persistent_fidelity_tradeoff"
                )
                next_path = (
                    "CALIBRATE_LOW_NOISE_K16_CURRICULUM_WITH_DECAYED_"
                    "GEOMETRY_WEIGHT"
                )
                locus = "fidelity_tradeoff"
        else:
            stable = [record for record in continuous_fidelity if record["stability_pass"]]
            if not stable:
                root = (
                    "phase314b_r257_staged_timestep_gated_k16_"
                    "outlier_or_gate_instability"
                )
                next_path = (
                    "CALIBRATE_CLIPPED_TIMESTEP_GATED_K16_OBJECTIVE_"
                    "ON_TRAIN_ONLY_SPLIT"
                )
                locus = "stability"
            else:
                root = (
                    "phase314b_r257_staged_timestep_gated_k16_continuous_"
                    "and_fidelity_response_without_gate_crossing"
                )
                next_path = (
                    "CALIBRATE_CURRICULUM_OR_STRONGER_TIMESTEP_GATED_K16_"
                    "OBJECTIVE_ON_TRAIN_ONLY_SPLIT"
                )
                locus = "binary_gate_crossing"

    best = min(
        records,
        key=lambda record: (
            float(record["holdout_profiles"]["10"]["row_best_max_log_excess"]["mean"]),
            float(record["relative_to_control"]["train_control_nmse_ratio"]),
            str(record["candidate_id"]),
        ),
    )
    best_fidelity = min(
        records,
        key=lambda record: (
            float(record["relative_to_control"]["train_control_nmse_ratio"]),
            -float(record["continuous_response"]["t10_mean_excess_reduction"]),
            str(record["candidate_id"]),
        ),
    )
    return {
        "root_cause": root,
        "required_next_path": next_path,
        "primary_failure_locus": locus,
        "best_candidate_by_t10_continuous_excess": best["candidate_id"],
        "best_t10_mean_excess_reduction": float(
            best["continuous_response"]["t10_mean_excess_reduction"]
        ),
        "best_t10_binary_upper_row_any": float(best["binary_rate"]["10"]),
        "best_fidelity_candidate": best_fidelity["candidate_id"],
        "best_train_control_nmse_ratio": float(
            best_fidelity["relative_to_control"]["train_control_nmse_ratio"]
        ),
        "selected_configuration": None if selected is None else dict(selected),
    }


def run_calibration(
    *, root: Path, spec: Optional[TimestepGateSpec] = None
) -> Dict[str, Any]:
    active_spec = TimestepGateSpec() if spec is None else spec
    active_spec.validate()
    repository_root = Path(root).resolve()

    immutable = validate_immutable_inputs(repository_root)
    logic = source_logic_audit(repository_root)
    if not logic["all_confirmed"]:
        raise TimestepGateCalibrationError("Stage-D source assumptions changed")

    stageb_spec = stageb.DiagnosticSpec()
    stageb_spec.validate()
    upstream = stagec_topk.validate_immutable_inputs(repository_root)
    objective_contract = stageb_mechanism.load_objective_contract(
        upstream["upstream_immutable"]["stagea_contract"]
    )
    upper_gate = stagea.load_upper_gate_contract(
        upstream["upstream_immutable"]["upstream_immutable"]["staged3_contract"]
    )
    stage_d_contract, _ = staged1.load_stage_d_gate(
        repository_root / staged3.STAGE_D_GATE
    )

    arrays = stageb.load_npz_strict(
        repository_root / stageb.STAGEA_TRAIN_VIEW,
        required_keys=stageb.REQUIRED_TRAIN_KEYS,
    )
    validation = stageb.validate_train_view(arrays)
    condition = np.asarray(arrays["diffusion_condition_x"], dtype=np.float32)
    target = np.asarray(arrays["diffusion_target_cable"], dtype=np.float32)
    groups = np.asarray(arrays["episode_group_key"]).astype(str)
    condition_name = np.asarray(arrays["condition_name"]).astype(str)

    stageb_train, frozen_probe, _ = stageb.deterministic_group_split(
        groups,
        folds=stageb_spec.group_folds,
        probe_fold=stageb_spec.probe_fold,
    )
    objective_train, selection_holdout, split = stagea.deterministic_selection_split(
        groups,
        stageb_train,
        folds=stagea.UpperObjectiveSpec().selection_group_folds,
        holdout_fold=stagea.UpperObjectiveSpec().selection_holdout_fold,
    )
    stagec_split = immutable["stagec_worker_result"]["split"]
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
        if split[key] != stagec_split[key]:
            raise TimestepGateCalibrationError(f"train-only split changed: {key}")
    if np.any(frozen_probe & (objective_train | selection_holdout)):
        raise AssertionError("frozen probe crossed train-only split")

    condition_standardizer = stageb.fit_standardizer(condition[objective_train])
    target_standardizer = stageb.fit_standardizer(target[objective_train])
    calibration_batch = stratified_calibration_batch(
        condition=condition[objective_train],
        target=target[objective_train],
        condition_standardizer=condition_standardizer,
        target_standardizer=target_standardizer,
        stageb_spec=stageb_spec,
        spec=active_spec,
    )
    candidates, calibration = calibrate_candidates(
        stageb_spec=stageb_spec,
        calibration_batch=calibration_batch,
        target_standardizer=target_standardizer,
        objective_contract=objective_contract,
        spec=active_spec,
    )

    control_expected = immutable["control_record"]
    stageb.set_deterministic_runtime(stageb_spec.seed)
    control_model, control_training, control_diagnostics = stagec_topk.train_candidate(
        condition=condition[objective_train],
        target=target[objective_train],
        stageb_spec=stageb_spec,
        objective_contract=objective_contract,
        condition_standardizer=condition_standardizer,
        target_standardizer=target_standardizer,
        candidate=None,
        diagnostic_batch=stageb_mechanism.fixed_diagnostic_batch(
            condition=condition[objective_train],
            target=target[objective_train],
            condition_standardizer=condition_standardizer,
            target_standardizer=target_standardizer,
            stageb_spec=stageb_spec,
            spec=stageb_mechanism.MechanismAuditSpec(),
        ),
        spec=stagec_topk.TopKCalibrationSpec(),
    )
    if stagec_topk.compatible_training_record(control_training) != control_expected["training"]:
        raise TimestepGateCalibrationError("Stage-D control training differs from Stage C")
    control_train_control = stagea.train_control_audit(
        model=control_model,
        condition=condition[objective_train],
        target=target[objective_train],
        stageb_spec=stageb_spec,
        condition_standardizer=condition_standardizer,
        target_standardizer=target_standardizer,
    )
    if control_train_control != control_expected["train_control"]:
        raise TimestepGateCalibrationError("Stage-D control train-control differs")
    control_one_step = stagea.one_step_audit(
        model=control_model,
        condition=condition[selection_holdout],
        target=target[selection_holdout],
        groups=groups[selection_holdout],
        condition_name=condition_name[selection_holdout],
        stageb_spec=stageb_spec,
        upper_gate=upper_gate,
        stage_d_contract=stage_d_contract,
        historical_geometry=stageb.fit_geometry_contract(target[stageb_train]),
        condition_standardizer=condition_standardizer,
        target_standardizer=target_standardizer,
        noise_seed=stageb_spec.seed + active_spec.holdout_noise_seed_offset,
    )
    if control_one_step != control_expected["one_step"]:
        raise TimestepGateCalibrationError("Stage-D control one-step differs")
    control_predictions, control_prediction_sha = stagec.one_step_predictions(
        model=control_model,
        condition=condition[selection_holdout],
        target=target[selection_holdout],
        spec=stageb_spec,
        condition_standardizer=condition_standardizer,
        target_standardizer=target_standardizer,
        noise_seed=stageb_spec.seed + active_spec.holdout_noise_seed_offset,
    )
    if control_prediction_sha != control_one_step["prediction_sha256"]:
        raise TimestepGateCalibrationError("control prediction aggregate SHA differs")
    control_profiles = stagec_topk.holdout_profiles(
        control_predictions,
        upper_gate=upper_gate,
        objective_contract=objective_contract,
    )
    if control_profiles != control_expected["holdout_profiles"]:
        raise TimestepGateCalibrationError("Stage-D control continuous profiles differ")
    control_bundle = {
        "candidate_id": CONTROL_CANDIDATE_ID,
        "candidate": None,
        "training": control_expected["training"],
        "training_diagnostics": control_diagnostics,
        "train_control": control_train_control,
        "one_step": control_one_step,
        "holdout_profiles": control_profiles,
    }
    del control_model

    records: List[Dict[str, Any]] = []
    for candidate in candidates:
        stageb.set_deterministic_runtime(stageb_spec.seed)
        model, training, diagnostics = train_gated_candidate(
            condition=condition[objective_train],
            target=target[objective_train],
            stageb_spec=stageb_spec,
            objective_contract=objective_contract,
            condition_standardizer=condition_standardizer,
            target_standardizer=target_standardizer,
            candidate=candidate,
            calibration_batch=calibration_batch,
            spec=active_spec,
        )
        train_control = stagea.train_control_audit(
            model=model,
            condition=condition[objective_train],
            target=target[objective_train],
            stageb_spec=stageb_spec,
            condition_standardizer=condition_standardizer,
            target_standardizer=target_standardizer,
        )
        one_step = stagea.one_step_audit(
            model=model,
            condition=condition[selection_holdout],
            target=target[selection_holdout],
            groups=groups[selection_holdout],
            condition_name=condition_name[selection_holdout],
            stageb_spec=stageb_spec,
            upper_gate=upper_gate,
            stage_d_contract=stage_d_contract,
            historical_geometry=stageb.fit_geometry_contract(target[stageb_train]),
            condition_standardizer=condition_standardizer,
            target_standardizer=target_standardizer,
            noise_seed=stageb_spec.seed + active_spec.holdout_noise_seed_offset,
        )
        predictions, prediction_sha = stagec.one_step_predictions(
            model=model,
            condition=condition[selection_holdout],
            target=target[selection_holdout],
            spec=stageb_spec,
            condition_standardizer=condition_standardizer,
            target_standardizer=target_standardizer,
            noise_seed=stageb_spec.seed + active_spec.holdout_noise_seed_offset,
        )
        if prediction_sha != one_step["prediction_sha256"]:
            raise TimestepGateCalibrationError("candidate prediction aggregate SHA differs")
        profiles = stagec_topk.holdout_profiles(
            predictions,
            upper_gate=upper_gate,
            objective_contract=objective_contract,
        )
        records.append(
            selection_record(
                candidate=candidate,
                training=training,
                training_diagnostics=diagnostics,
                train_control=train_control,
                one_step=one_step,
                profiles=profiles,
                control=control_bundle,
                spec=active_spec,
            )
        )
        del model

    selected = select_configuration(records)
    classification = classify_selection(records=records, selected=selected, spec=active_spec)
    selection_payload = {
        "control": control_bundle,
        "candidate_records": records,
        "selected_configuration": selected,
        "selection_uses_frozen_probe": False,
        "selection_rule": (
            "continuous geometry, fidelity, stability and binary gates on the "
            "frozen train-only holdout"
        ),
    }
    selection_sha = sha256_bytes(stable_json_bytes(selection_payload))

    contract = {
        "schema": "phase314b_r257_staged_timestep_gate_contract_v1",
        "base_evidence_commit": BASE_EVIDENCE_COMMIT,
        "objective_variant": "hard_timestep_gated_k16_quadratic_raw_log_excess",
        "top_k": int(active_spec.top_k),
        "timestep_cutoffs": list(active_spec.timestep_cutoffs),
        "target_gradient_ratios": list(active_spec.target_gradient_ratios),
        "gate_definition": "active iff sampled diffusion timestep <= cutoff",
        "gate_normalization": "mean top-k row loss over active rows only",
        "lambda_calibration": (
            "lambda = target_ratio * ||g_diffusion|| / ||g_gated_geometry|| "
            "at the common initial model on a deterministic t=0..99 batch"
        ),
        "calibration_batch_sha256": {
            key: value
            for key, value in calibration_batch.items()
            if key.endswith("_sha256")
        },
        "calibration": calibration,
        "selection_thresholds": {
            key: value
            for key, value in asdict(active_spec).items()
            if key.endswith("_min") or key.endswith("_max") or key.endswith("_tolerance")
        },
        "uses_frozen_probe": False,
        "runs_reverse_sampling": False,
        "trains_full_stageb_model": False,
        "soft_gate_run": False,
        "curriculum_run": False,
    }
    contract["contract_sha256"] = sha256_bytes(stable_json_bytes(contract))

    return {
        "phase": PHASE,
        "phase_id": PHASE_ID,
        "schema": "phase314b_r257_staged_worker_result_v1",
        "verdict": "PASS",
        "scientific_status": "BLOCKED",
        "root_cause": classification["root_cause"],
        "required_next_path": classification["required_next_path"],
        "gate_spec": asdict(active_spec),
        "immutable_inputs": {
            "base_commit": immutable["base_commit"],
            "base_file_sha256": immutable["base_file_sha256"],
            "stagec_worker_identity_sha256": immutable["worker_identity_sha256"],
            "stagec_contract_sha256": EXPECTED_STAGEC_CONTRACT_SHA256,
            "stagec_contract_file_sha256": EXPECTED_STAGEC_CONTRACT_FILE_SHA256,
            "stagec_selection_sha256": EXPECTED_STAGEC_SELECTION_SHA256,
        },
        "source_logic_audit": logic,
        "train_view_validation": validation,
        "split": {
            "stageb_training_rows": int(np.sum(stageb_train)),
            "objective_train_rows": int(np.sum(objective_train)),
            "selection_holdout_rows": int(np.sum(selection_holdout)),
            "frozen_probe_rows": int(np.sum(frozen_probe)),
            **split,
            "frozen_probe_accessed": False,
        },
        "calibration_contract": contract,
        "selection": {
            **selection_payload,
            "selection_sha256": selection_sha,
        },
        "classification": classification,
        "selected_configuration": selected,
        "train_only_recommendation": selected,
        "new_hyperparameter_candidate_run": True,
        "new_objective_variant_run": True,
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


def identity_projection(result: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "root_cause": result["root_cause"],
        "required_next_path": result["required_next_path"],
        "split": result["split"],
        "calibration_contract": result["calibration_contract"],
        "selection": result["selection"],
        "classification": result["classification"],
        "selected_configuration": result["selected_configuration"],
    }


def compare_worker_results(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> Dict[str, Any]:
    left_payload = stable_json_bytes(identity_projection(left))
    right_payload = stable_json_bytes(identity_projection(right))
    return {
        "exact": left_payload == right_payload,
        "left_sha256": sha256_bytes(left_payload),
        "right_sha256": sha256_bytes(right_payload),
        "calibration_contract_exact": (
            left["calibration_contract"] == right["calibration_contract"]
        ),
        "selection_exact": left["selection"] == right["selection"],
        "classification_exact": left["classification"] == right["classification"],
    }
