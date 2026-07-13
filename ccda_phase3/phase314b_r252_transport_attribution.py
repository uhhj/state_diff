"""Paired one-step and reverse-trajectory attribution for Phase3.14b-r2.5.2.

This module is additive and train-only.  It does not alter the r2.5.1 model,
loss, rows, seeds, scheduler, or gates.  It retrains three fixed diagnostic
variants and attributes the mismatch between paired low/mid one-step failure
and full-reverse candidate quality.
"""

from __future__ import annotations

import hashlib
import inspect
import math
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from ccda_phase3.phase314b_r22_geometry import torch_inverse_standardize
from ccda_phase3.phase314b_r241_multirow import LabeledTupleBank
from ccda_phase3.phase314b_r242_frozen_prior import (
    FactorizedAnalyticX0SkipDenoiser,
    paired_branch_audit,
)
from ccda_phase3.phase314b_r25_ordered_geometry import (
    GeometryScales,
    TorchGeometryContract,
    paired_full_reverse_branch_support,
)
import ccda_phase3.phase314b_r251_gradient_calibration as r251
from ccda_phase3.phase314b_r2_contract import REPAIR_CONFIGS
from ccda_phase3.phase314b_r2_diffusion import (
    predict_original_sample,
    training_target,
)
from ccda_phase3.schema_v2 import DEFAULT_TF, STATE_DIM

PHASE = "phase3_14b_r252"
BASE_REPORT_COMMIT = "45ce0e388d325882a49d63db94e5fa5bf30cf82e"
BASE_IMPLEMENTATION_COMMIT = "fd4b07b173bf88844d7ac8678bf9d957974135da"
EXPECTED_SUBMODULE_COMMIT = "633a88752445cf5d6776ed374fdbbdb35f93050c"
EXPECTED_CACHE_SHA256 = (
    "3cc512f650557c81c3b81f4f128a5b55360b77d3f664fdd6607666936285fbe8"
)
EXPECTED_CONTRACT_SHA256 = (
    "fa2725ca40da2499008360f13d291d2ce8694e6393910b0522fe800f4f37cdcc"
)
EXPECTED_R251_ROOT_CAUSE = (
    "phase314b_r251_paired_low_mid_geometry_transport_failed"
)
EXPECTED_R251_RECOMMENDATION = None
EXPECTED_PAIRED_ROWS = r251.EXPECTED_PAIRED_ROWS
EXPECTED_PAIRED_CONDITIONS = r251.EXPECTED_PAIRED_CONDITIONS

DIAGNOSTIC_OBJECTIVE_NAMES = (
    "v_only_frozen_control",
    "ordered_mean_raw_g100",
    "ordered_cvar_contract_g010",
)
PAIR_TIMESTEPS = (10, 25, 50)
TRAJECTORY_TIMESTEPS = (99, 90, 75, 50, 25, 10, 0)
EVALUATION_NOISE_SEEDS = tuple(99000 + index for index in range(8))
PER_TIMESTEP_GRADIENT_NOISE_SEEDS = tuple(120000 + index for index in range(8))
FULL_REVERSE_K = 16
FULL_REVERSE_STEPS = 100
COMMON_PAIRED_PRIOR_SEED = r251.COMMON_PAIRED_PRIOR_SEED
COMMON_PAIRED_TRAINING_SEED = r251.COMMON_PAIRED_TRAINING_SEED
COMMON_REVERSE_SEED = r251.COMMON_REVERSE_SEED

ATTRIBUTION_SCHEMA = "phase314b_r252_paired_transport_attribution_v1"
TRAJECTORY_SCHEMA = "phase314b_r252_predicted_x0_trajectory_v1"
PER_TIMESTEP_GRADIENT_SCHEMA = "phase314b_r252_per_timestep_gradient_v1"

SOURCE_PATHS = (
    "ccda_phase3/phase314b_r252_transport_attribution.py",
    "scripts/phase3_14b_r252_preflight.py",
    "scripts/phase3_14b_r252_run_pilot.py",
    "scripts/phase3_14b_r252_finalize.py",
    "scripts/phase3_14b_r252_run.sh",
    "tests/test_phase314b_r252_transport_attribution.py",
)

DEPENDENCY_PATHS = (
    "ccda_phase3/phase314b_r242_frozen_prior.py",
    "ccda_phase3/phase314b_r25_ordered_geometry.py",
    "ccda_phase3/phase314b_r251_gradient_calibration.py",
    "scripts/phase3_14b_r251_run_pilot.py",
    "scripts/phase3_14b_r251_finalize.py",
    "reports/phase3_14b_r251_pilot_summary.json",
    "reports/phase3_14b_r251_summary.json",
    "reports/phase3_14b_r251_report.md",
)


@dataclass(frozen=True)
class AttributionGate:
    legacy_own_fraction: float = 0.90
    legacy_separation_ratio: float = 0.50
    legacy_delta_cosine: float = 0.50
    decode_tolerance: float = 2.0e-5
    matched_noise_tolerance: float = 1.0e-7
    forward_ratio_tolerance: float = 2.0e-5
    prior_pair_tolerance: float = 2.0e-6
    reverse_valid_query_min: float = 0.875
    reproduction_rate_tolerance: float = 0.03
    reproduction_relative_tolerance: float = 0.05

    def validate(self) -> None:
        values = asdict(self)
        for name, value in values.items():
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"invalid attribution gate {name}")
        if not 0 <= self.legacy_own_fraction <= 1:
            raise ValueError("legacy_own_fraction must lie in [0,1]")
        if not 0 <= self.reverse_valid_query_min <= 1:
            raise ValueError("reverse_valid_query_min must lie in [0,1]")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_sha256(root: Path) -> Dict[str, str]:
    base = Path(root)
    return {path: sha256_file(base / path) for path in SOURCE_PATHS}


def dependency_sha256(root: Path) -> Dict[str, str]:
    base = Path(root)
    return {path: sha256_file(base / path) for path in DEPENDENCY_PATHS}


def git_output(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=Path(root), text=True).strip()


def assert_only_allowed_worktree_paths(
    root: Path,
    allowed_paths: Sequence[str],
) -> None:
    allowed = {str(Path(path).as_posix()) for path in allowed_paths}
    output = git_output(root, "status", "--porcelain", "--untracked-files=all")
    observed = set()
    for line in output.splitlines():
        if not line:
            continue
        value = line[3:]
        if " -> " in value:
            value = value.split(" -> ", 1)[1]
        observed.add(str(Path(value).as_posix()))
    unexpected = sorted(observed - allowed)
    if unexpected:
        raise RuntimeError("unexpected worktree changes: " + ", ".join(unexpected))


def diagnostic_objectives() -> Tuple[r251.CalibratedGeometryObjective, ...]:
    by_name = {
        objective.name: objective
        for objective in r251.CALIBRATED_GEOMETRY_OBJECTIVES
    }
    missing = [name for name in DIAGNOSTIC_OBJECTIVE_NAMES if name not in by_name]
    if missing:
        raise RuntimeError(f"diagnostic objectives missing: {missing}")
    values = tuple(by_name[name] for name in DIAGNOSTIC_OBJECTIVE_NAMES)
    if values[0].selectable:
        raise RuntimeError("diagnostic control unexpectedly selectable")
    if not all(objective.selectable for objective in values[1:]):
        raise RuntimeError("diagnostic geometry objective is not selectable")
    return values


def _stats(values: Sequence[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("stats values must be a non-empty vector")
    if not np.isfinite(array).all():
        raise ValueError("stats values must be finite")
    return {
        "min": float(np.min(array)),
        "p05": float(np.percentile(array, 5)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def _cosine(reference: np.ndarray, value: np.ndarray) -> float:
    left = np.asarray(reference, dtype=np.float64).reshape(-1)
    right = np.asarray(value, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1.0e-12:
        return 0.0
    return float(np.dot(left, right) / denominator)


def _rmse(value: np.ndarray) -> float:
    array = np.asarray(value, dtype=np.float64)
    return float(np.sqrt(np.mean(array ** 2)))


def _active_standardized_flat(
    value: torch.Tensor,
    active_mask: torch.Tensor,
) -> np.ndarray:
    active = active_mask.to(device=value.device, dtype=torch.bool).reshape(-1)
    flat = value.reshape(value.shape[0], -1)[:, active]
    return flat.detach().cpu().numpy().astype(np.float64)


def _raw_xy_flat(value: torch.Tensor) -> np.ndarray:
    if value.ndim != 3 or tuple(value.shape[1:]) != (DEFAULT_TF, STATE_DIM):
        raise ValueError("raw future must be [N,4,87]")
    return (
        value[..., :48]
        .reshape(value.shape[0], -1)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )


def _group_records_by_timestep(
    records: Sequence[Mapping[str, Any]],
    *,
    gate: AttributionGate,
) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for timestep in PAIR_TIMESTEPS:
        selected = [item for item in records if int(item["timestep"]) == timestep]
        if not selected:
            raise RuntimeError(f"missing paired transport records at t={timestep}")
        own_flags: List[float] = []
        own_margins: List[float] = []
        for item in selected:
            own_flags.extend(item["own_target_closer"])
            own_margins.extend(item["normalized_own_margin"])
        separation = _stats([float(item["model_separation_ratio"]) for item in selected])
        cosine = _stats([float(item["model_delta_cosine"]) for item in selected])
        noisy_ratio = _stats([float(item["noisy_separation_ratio"]) for item in selected])
        residual_gain = _stats([float(item["residual_delta_gain"]) for item in selected])
        residual_cosine = _stats([float(item["residual_delta_cosine"]) for item in selected])
        required_norm = _stats([float(item["required_residual_delta_rms"]) for item in selected])
        predicted_norm = _stats([float(item["predicted_residual_delta_rms"]) for item in selected])
        own_fraction = float(np.mean(np.asarray(own_flags, dtype=np.float64)))
        expected_alpha = float(selected[0]["expected_forward_alpha"])
        alpha_error = max(
            abs(float(item["noisy_separation_ratio"]) - expected_alpha)
            for item in selected
        )
        output[str(timestep)] = {
            "comparison_count": int(len(selected)),
            "own_target_closer_fraction": own_fraction,
            "normalized_own_margin": _stats(own_margins),
            "model_separation_ratio": separation,
            "model_delta_cosine": cosine,
            "noisy_separation_ratio": noisy_ratio,
            "expected_forward_alpha": expected_alpha,
            "forward_alpha_max_abs_error": float(alpha_error),
            "residual_delta_gain": residual_gain,
            "residual_delta_cosine": residual_cosine,
            "required_residual_delta_rms": required_norm,
            "predicted_residual_delta_rms": predicted_norm,
            "legacy_gate_pass": bool(
                own_fraction >= gate.legacy_own_fraction
                and separation["p50"] >= gate.legacy_separation_ratio
                and cosine["p50"] >= gate.legacy_delta_cosine
            ),
            "forward_signal_contract_pass": bool(
                alpha_error <= gate.forward_ratio_tolerance
            ),
        }
    return output


def paired_one_step_attribution(
    *,
    model: FactorizedAnalyticX0SkipDenoiser,
    scheduler,
    bank: LabeledTupleBank,
    source_pair_ids: Sequence[int],
    active_mask: torch.Tensor,
    future_mean: torch.Tensor,
    future_scale: torch.Tensor,
    gate: AttributionGate = AttributionGate(),
) -> Dict[str, Any]:
    """Decompose paired one-step transport into signal, prior and residual terms."""
    gate.validate()
    bank.validate()
    pair_ids = np.asarray(source_pair_ids, dtype=np.int64)
    source_count = int(torch.max(bank.source_ids).item()) + 1
    if pair_ids.shape != (source_count,):
        raise ValueError("source_pair_ids shape mismatch")
    counts = {int(value): int(np.sum(pair_ids == value)) for value in np.unique(pair_ids)}
    if any(value != 2 for value in counts.values()):
        raise ValueError("each source pair must contain exactly two rows")

    active = active_mask.to(device=bank.noisy.device, dtype=torch.bool)
    repair = REPAIR_CONFIGS["v_prediction_cosine"]
    model.eval()
    with torch.no_grad():
        output = model(
            bank.noisy,
            bank.timesteps,
            bank.condition_z,
            detach_prior_for_v=True,
        )
        output = torch.where(active[None], output, torch.zeros_like(output))
        predicted_z = predict_original_sample(
            scheduler=scheduler,
            config=repair,
            sample=bank.noisy,
            model_output=output,
            timesteps=bank.timesteps,
        )
        predicted_z = torch.where(active[None], predicted_z, torch.zeros_like(predicted_z))
        target_v = training_target(
            scheduler=scheduler,
            config=repair,
            clean_sample=bank.clean_z,
            noise=bank.noise,
            timesteps=bank.timesteps,
        )
        oracle_z = predict_original_sample(
            scheduler=scheduler,
            config=repair,
            sample=bank.noisy,
            model_output=target_v,
            timesteps=bank.timesteps,
        )
        oracle_z = torch.where(active[None], oracle_z, torch.zeros_like(oracle_z))
        prior_z = model.predict_base_x0(bank.condition_z)
        prior_z = torch.where(active[None], prior_z, torch.zeros_like(prior_z))
        residual_v = model.predict_residual_v(
            bank.noisy,
            bank.timesteps,
            bank.condition_z,
        )
        residual_v = torch.where(active[None], residual_v, torch.zeros_like(residual_v))
        alpha, sigma = model.coefficients(bank.timesteps, bank.noisy.dtype)
        analytic_v = (alpha * bank.noisy - prior_z) / sigma
        required_residual_v = target_v - analytic_v
        reconstructed_z = prior_z - sigma * residual_v

    decode_error = float(torch.max(torch.abs(oracle_z - bank.clean_z)).cpu())
    analytic_identity_error = float(
        torch.max(torch.abs(predicted_z - reconstructed_z)).cpu()
    )
    condition = bank.condition_z.detach().cpu().numpy().astype(np.float64)
    noise = bank.noise.detach().cpu().numpy().astype(np.float64)
    source = bank.source_ids.detach().cpu().numpy().astype(np.int64)
    timestep = bank.timesteps.detach().cpu().numpy().astype(np.int64)
    noise_id = bank.noise_ids.detach().cpu().numpy().astype(np.int64)
    sigma_np = sigma.reshape(-1).detach().cpu().numpy().astype(np.float64)
    alpha_np = alpha.reshape(-1).detach().cpu().numpy().astype(np.float64)

    target_raw = torch_inverse_standardize(bank.clean_z, future_mean, future_scale)
    noisy_raw = torch_inverse_standardize(bank.noisy, future_mean, future_scale)
    prior_raw = torch_inverse_standardize(prior_z, future_mean, future_scale)
    predicted_raw = torch_inverse_standardize(predicted_z, future_mean, future_scale)
    oracle_raw = torch_inverse_standardize(oracle_z, future_mean, future_scale)
    target_xy = _raw_xy_flat(target_raw)
    noisy_xy = _raw_xy_flat(noisy_raw)
    prior_xy = _raw_xy_flat(prior_raw)
    predicted_xy = _raw_xy_flat(predicted_raw)
    oracle_xy = _raw_xy_flat(oracle_raw)
    required_residual = _active_standardized_flat(required_residual_v, active)
    predicted_residual = _active_standardized_flat(residual_v, active)

    records: List[Dict[str, Any]] = []
    max_condition_difference = 0.0
    max_noise_difference = 0.0
    for pair_id in sorted(counts):
        source_members = np.flatnonzero(pair_ids == int(pair_id))
        for local_timestep in PAIR_TIMESTEPS:
            for local_noise in sorted(set(noise_id.tolist())):
                selected = np.flatnonzero(
                    np.isin(source, source_members)
                    & (timestep == int(local_timestep))
                    & (noise_id == int(local_noise))
                )
                if selected.size != 2:
                    raise RuntimeError(
                        "incomplete paired bank for "
                        f"pair={pair_id}, t={local_timestep}, noise={local_noise}"
                    )
                a, b = int(selected[0]), int(selected[1])
                condition_difference = float(np.max(np.abs(condition[a] - condition[b])))
                noise_difference = float(np.max(np.abs(noise[a] - noise[b])))
                max_condition_difference = max(max_condition_difference, condition_difference)
                max_noise_difference = max(max_noise_difference, noise_difference)

                target_delta = target_xy[a] - target_xy[b]
                target_separation = _rmse(target_delta)
                if target_separation <= 1.0e-12:
                    raise RuntimeError(f"paired target separation is zero for pair {pair_id}")
                noisy_delta = noisy_xy[a] - noisy_xy[b]
                prior_delta = prior_xy[a] - prior_xy[b]
                model_delta = predicted_xy[a] - predicted_xy[b]
                oracle_delta = oracle_xy[a] - oracle_xy[b]
                required_delta = required_residual[a] - required_residual[b]
                predicted_delta = predicted_residual[a] - predicted_residual[b]
                required_delta_rms = _rmse(required_delta)
                predicted_delta_rms = _rmse(predicted_delta)

                own_a = _rmse(predicted_xy[a] - target_xy[a])
                own_b = _rmse(predicted_xy[b] - target_xy[b])
                cross_a = _rmse(predicted_xy[a] - target_xy[b])
                cross_b = _rmse(predicted_xy[b] - target_xy[a])
                records.append(
                    {
                        "pair_id": int(pair_id),
                        "timestep": int(local_timestep),
                        "noise_id": int(local_noise),
                        "sigma": float(sigma_np[a]),
                        "expected_forward_alpha": float(alpha_np[a]),
                        "target_separation": target_separation,
                        "noisy_separation_ratio": float(
                            _rmse(noisy_delta) / target_separation
                        ),
                        "prior_separation_ratio": float(
                            _rmse(prior_delta) / target_separation
                        ),
                        "model_separation_ratio": float(
                            _rmse(model_delta) / target_separation
                        ),
                        "oracle_separation_ratio": float(
                            _rmse(oracle_delta) / target_separation
                        ),
                        "model_delta_cosine": _cosine(target_delta, model_delta),
                        "oracle_delta_cosine": _cosine(target_delta, oracle_delta),
                        "own_target_closer": [
                            float(own_a < cross_a),
                            float(own_b < cross_b),
                        ],
                        "normalized_own_margin": [
                            float((cross_a - own_a) / target_separation),
                            float((cross_b - own_b) / target_separation),
                        ],
                        "required_residual_delta_rms": required_delta_rms,
                        "predicted_residual_delta_rms": predicted_delta_rms,
                        "residual_delta_gain": float(
                            predicted_delta_rms / max(required_delta_rms, 1.0e-12)
                        ),
                        "residual_delta_cosine": _cosine(
                            required_delta,
                            predicted_delta,
                        ),
                    }
                )

    by_timestep = _group_records_by_timestep(records, gate=gate)
    model_legacy = paired_branch_audit(
        predicted_z=predicted_z,
        bank=bank,
        source_pair_ids=source_pair_ids,
        future_mean=future_mean,
        future_scale=future_scale,
    )
    oracle_legacy = paired_branch_audit(
        predicted_z=oracle_z,
        bank=bank,
        source_pair_ids=source_pair_ids,
        future_mean=future_mean,
        future_scale=future_scale,
    )
    prior_ratios = [float(item["prior_separation_ratio"]) for item in records]
    oracle_ratios = [float(item["oracle_separation_ratio"]) for item in records]
    oracle_cosines = [float(item["oracle_delta_cosine"]) for item in records]
    forward_contract = all(
        bool(value["forward_signal_contract_pass"])
        for value in by_timestep.values()
    )
    pipeline_pass = bool(
        decode_error <= gate.decode_tolerance
        and analytic_identity_error <= gate.decode_tolerance
        and max_condition_difference <= gate.matched_noise_tolerance
        and max_noise_difference <= gate.matched_noise_tolerance
        and max(prior_ratios) <= gate.prior_pair_tolerance
        and max(abs(value - 1.0) for value in oracle_ratios) <= gate.decode_tolerance
        and min(oracle_cosines) >= 1.0 - 1.0e-5
        and bool(oracle_legacy["pass"])
        and forward_contract
    )
    threshold_sweep = {}
    for own_threshold in (0.75, 0.80, 0.90):
        for separation_threshold in (0.25, 0.50):
            for cosine_threshold in (0.25, 0.50):
                name = (
                    f"own{int(100 * own_threshold)}_"
                    f"sep{int(100 * separation_threshold)}_"
                    f"cos{int(100 * cosine_threshold)}"
                )
                threshold_sweep[name] = bool(
                    float(model_legacy["own_target_closer_fraction"])
                    >= own_threshold
                    and float(model_legacy["separation_ratio"]["p50"])
                    >= separation_threshold
                    and float(model_legacy["branch_delta_cosine"]["p50"])
                    >= cosine_threshold
                )

    return {
        "schema": ATTRIBUTION_SCHEMA,
        "record_count": int(len(records)),
        "pipeline_controls": {
            "oracle_decode_max_abs": decode_error,
            "analytic_x0_identity_max_abs": analytic_identity_error,
            "paired_condition_max_abs_difference": max_condition_difference,
            "matched_noise_max_abs_difference": max_noise_difference,
            "prior_pair_separation_ratio_max": float(max(prior_ratios)),
            "oracle_separation_ratio": _stats(oracle_ratios),
            "oracle_delta_cosine": _stats(oracle_cosines),
            "oracle_legacy_gate_pass": bool(oracle_legacy["pass"]),
            "forward_signal_contract_pass": bool(forward_contract),
            "pass": pipeline_pass,
        },
        "legacy_model_audit": model_legacy,
        "legacy_oracle_audit": oracle_legacy,
        "by_timestep": by_timestep,
        "threshold_sweep": threshold_sweep,
        "records": records,
        "_predicted_z": predicted_z,
        "_oracle_z": oracle_z,
        "_prior_z": prior_z,
    }


def _repeat_balanced_rows(
    value: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    source_count = int(value.shape[0])
    if source_count <= 0 or batch_size <= 0:
        raise ValueError("source_count and batch_size must be positive")
    index = torch.arange(batch_size, device=value.device, dtype=torch.long)
    index = torch.remainder(index, source_count)
    return value.index_select(0, index)


def _fixed_timestep_ratio_samples(
    *,
    model: FactorizedAnalyticX0SkipDenoiser,
    scheduler,
    condition_z: torch.Tensor,
    clean_z: torch.Tensor,
    clean_raw: torch.Tensor,
    active_mask: torch.Tensor,
    future_mean: torch.Tensor,
    future_scale: torch.Tensor,
    torch_contract: TorchGeometryContract,
    scales: GeometryScales,
    objective: r251.CalibratedGeometryObjective,
    multiplier: float,
    timestep: int,
    batch_size: int,
    noise_seeds: Sequence[int],
) -> Dict[str, Any]:
    if not objective.selectable:
        return {
            "timestep": int(timestep),
            "weighted_ratios": [],
            "unweighted_ratios": [],
            "v_gradient_norms": [],
            "geometry_gradient_norms": [],
            "tracking": {
                "target_gradient_ratio": 0.0,
                "observed_ratio_median": 0.0,
                "observed_ratio_p95": 0.0,
                "pass": True,
            },
        }
    device = condition_z.device
    active = active_mask.to(device=device, dtype=torch.bool)
    condition_batch = _repeat_balanced_rows(condition_z, batch_size)
    clean_batch = _repeat_balanced_rows(clean_z, batch_size)
    raw_batch = _repeat_balanced_rows(clean_raw, batch_size)
    residual_parameters = model.residual_parameters()
    repair = REPAIR_CONFIGS["v_prediction_cosine"]
    v_norms: List[float] = []
    geometry_norms: List[float] = []
    weighted_ratios: List[float] = []
    for noise_seed in noise_seeds:
        generator = torch.Generator(device=device).manual_seed(
            int(noise_seed) + 1000 * int(timestep)
        )
        noise = torch.randn(
            clean_batch.shape,
            generator=generator,
            device=device,
            dtype=clean_batch.dtype,
        )
        noise = torch.where(active[None], noise, torch.zeros_like(noise))
        timesteps = torch.full(
            (batch_size,),
            int(timestep),
            device=device,
            dtype=torch.long,
        )
        losses = r251._forward_losses(
            model=model,
            scheduler=scheduler,
            repair=repair,
            condition_batch=condition_batch,
            clean_batch=clean_batch,
            raw_batch=raw_batch,
            active_mask=active,
            future_mean=future_mean,
            future_scale=future_scale,
            torch_contract=torch_contract,
            scales=scales,
            objective=objective,
            timesteps=timesteps,
            noise=noise,
            geometry_timestep_max=50,
        )
        v_norm = r251._gradient_norm(
            losses["v_loss"], residual_parameters, retain_graph=True
        )
        geometry_norm = r251._gradient_norm(
            losses["geometry_loss"], residual_parameters, retain_graph=False
        )
        if not math.isfinite(v_norm) or not math.isfinite(geometry_norm):
            raise RuntimeError("non-finite fixed-timestep gradient norm")
        if v_norm <= 0 or geometry_norm <= 0:
            raise RuntimeError("non-positive fixed-timestep gradient norm")
        v_norms.append(float(v_norm))
        geometry_norms.append(float(geometry_norm))
        weighted_ratios.append(float(multiplier * geometry_norm / v_norm))
    tracking = r251.gradient_tracking_gate(
        target_ratio=float(objective.target_gradient_ratio),
        observed_ratios=weighted_ratios,
        spec=r251.GradientCalibrationSpec(),
    )
    return {
        "timestep": int(timestep),
        "weighted_ratios": weighted_ratios,
        "unweighted_ratios": [
            float(geometry / v)
            for v, geometry in zip(v_norms, geometry_norms)
        ],
        "v_gradient_norms": v_norms,
        "geometry_gradient_norms": geometry_norms,
        "tracking": tracking,
    }


def per_timestep_gradient_audit(
    *,
    initial_model: FactorizedAnalyticX0SkipDenoiser,
    final_model: FactorizedAnalyticX0SkipDenoiser,
    scheduler,
    condition_z: torch.Tensor,
    clean_z: torch.Tensor,
    clean_raw: torch.Tensor,
    active_mask: torch.Tensor,
    future_mean: torch.Tensor,
    future_scale: torch.Tensor,
    torch_contract: TorchGeometryContract,
    scales: GeometryScales,
    objective: r251.CalibratedGeometryObjective,
    multiplier: float,
    batch_size: int,
    noise_seeds: Sequence[int] = PER_TIMESTEP_GRADIENT_NOISE_SEEDS,
) -> Dict[str, Any]:
    output: Dict[str, Any] = {
        "schema": PER_TIMESTEP_GRADIENT_SCHEMA,
        "objective": objective.name,
        "target_gradient_ratio": float(objective.target_gradient_ratio),
        "multiplier": float(multiplier),
        "initial": {},
        "final": {},
    }
    for label, model in (("initial", initial_model), ("final", final_model)):
        for timestep in PAIR_TIMESTEPS:
            output[label][str(timestep)] = _fixed_timestep_ratio_samples(
                model=model,
                scheduler=scheduler,
                condition_z=condition_z,
                clean_z=clean_z,
                clean_raw=clean_raw,
                active_mask=active_mask,
                future_mean=future_mean,
                future_scale=future_scale,
                torch_contract=torch_contract,
                scales=scales,
                objective=objective,
                multiplier=multiplier,
                timestep=timestep,
                batch_size=batch_size,
                noise_seeds=noise_seeds,
            )
    output["initial_all_timesteps_pass"] = bool(
        all(
            value["tracking"]["pass"]
            for value in output["initial"].values()
        )
    )
    output["final_all_timesteps_pass"] = bool(
        all(
            value["tracking"]["pass"]
            for value in output["final"].values()
        )
    )
    return output


def _set_scheduler_timesteps(scheduler, count: int, device: torch.device) -> None:
    signature = inspect.signature(scheduler.set_timesteps)
    if "device" in signature.parameters:
        scheduler.set_timesteps(int(count), device=device)
    else:
        scheduler.set_timesteps(int(count))


def reverse_predicted_x0_trajectory(
    *,
    model: FactorizedAnalyticX0SkipDenoiser,
    scheduler,
    condition_z: torch.Tensor,
    active_mask: torch.Tensor,
    sample_count: int = FULL_REVERSE_K,
    inference_steps: int = FULL_REVERSE_STEPS,
    seed: int = COMMON_REVERSE_SEED,
    checkpoint_timesteps: Sequence[int] = TRAJECTORY_TIMESTEPS,
) -> Dict[str, Any]:
    """Run the official reverse process and retain only in-memory x0 estimates."""
    if sample_count <= 0 or inference_steps <= 0:
        raise ValueError("sample_count and inference_steps must be positive")
    model.eval()
    device = condition_z.device
    active = active_mask.to(device=device, dtype=torch.bool)
    query_count = int(condition_z.shape[0])
    generator = torch.Generator(device=device).manual_seed(int(seed))
    base = torch.randn(
        (sample_count, 1, DEFAULT_TF, STATE_DIM),
        generator=generator,
        device=device,
        dtype=condition_z.dtype,
    )
    sample = base.expand(-1, query_count, -1, -1).clone()
    sample = torch.where(active[None, None], sample, torch.zeros_like(sample))
    flat_condition = condition_z[None].expand(sample_count, -1, -1).reshape(
        sample_count * query_count, -1
    )
    flat = sample.reshape(sample_count * query_count, DEFAULT_TF, STATE_DIM)
    _set_scheduler_timesteps(scheduler, inference_steps, device)
    scheduler_values = [
        int(value.item()) if torch.is_tensor(value) else int(value)
        for value in scheduler.timesteps
    ]
    requested = tuple(int(value) for value in checkpoint_timesteps)
    missing = sorted(set(requested) - set(scheduler_values))
    if missing:
        raise RuntimeError(f"scheduler is missing trajectory timesteps: {missing}")
    step_signature = inspect.signature(scheduler.step)
    use_generator = "generator" in step_signature.parameters
    repair = REPAIR_CONFIGS["v_prediction_cosine"]
    checkpoints: Dict[str, torch.Tensor] = {}
    iterations: Dict[str, int] = {}
    with torch.no_grad():
        for iteration, timestep_value in enumerate(scheduler_values, start=1):
            timesteps = torch.full(
                (flat.shape[0],),
                int(timestep_value),
                device=device,
                dtype=torch.long,
            )
            output = model(
                flat,
                timesteps,
                flat_condition,
                detach_prior_for_v=True,
            )
            output = torch.where(active[None], output, torch.zeros_like(output))
            predicted_x0 = predict_original_sample(
                scheduler=scheduler,
                config=repair,
                sample=flat,
                model_output=output,
                timesteps=timesteps,
            )
            predicted_x0 = torch.where(
                active[None], predicted_x0, torch.zeros_like(predicted_x0)
            )
            if int(timestep_value) in set(requested):
                checkpoints[str(int(timestep_value))] = predicted_x0.reshape(
                    sample_count,
                    query_count,
                    DEFAULT_TF,
                    STATE_DIM,
                ).detach().clone()
                iterations[str(int(timestep_value))] = int(iteration)
            kwargs = {"generator": generator} if use_generator else {}
            flat = scheduler.step(
                output,
                int(timestep_value),
                flat,
                **kwargs,
            ).prev_sample
            flat = torch.where(active[None], flat, torch.zeros_like(flat))
            if not bool(torch.isfinite(flat).all()):
                raise RuntimeError(
                    f"non-finite reverse sample at timestep {timestep_value}"
                )
    if set(checkpoints) != {str(value) for value in requested}:
        raise RuntimeError("trajectory checkpoint capture is incomplete")
    endpoint = flat.reshape(sample_count, query_count, DEFAULT_TF, STATE_DIM)
    return {
        "schema": TRAJECTORY_SCHEMA,
        "scheduler_timesteps": scheduler_values,
        "checkpoint_iterations": iterations,
        "checkpoint_predicted_x0": checkpoints,
        "endpoint_sample": endpoint,
    }


def _compact_reverse_metrics(value: Mapping[str, Any]) -> Dict[str, Any]:
    calibrated = value["calibrated"]
    return {
        "sample_count": int(value["sample_count"]),
        "query_count": int(value["query_count"]),
        "best_ordered_rmse_mean": float(value["best_ordered_rmse_mean"]),
        "best_ordered_rmse_p95": float(value["best_ordered_rmse_p95"]),
        "k1_ordered_rmse_mean": float(value["k1_ordered_rmse_mean"]),
        "pool_diversity": float(value["pool_diversity"]),
        "nearest_inversion_mean": float(value["nearest_inversion_mean"]),
        "nearest_inversion_p95": float(value["nearest_inversion_p95"]),
        "nearest_inversion_max": float(value["nearest_inversion_max"]),
        "nearest_unique_fraction_mean": float(
            value["nearest_unique_fraction_mean"]
        ),
        "nearest_unique_fraction_p05": float(
            value["nearest_unique_fraction_p05"]
        ),
        "sample_validity_rate": float(calibrated["sample_validity_rate"]),
        "query_has_valid_candidate_rate": float(
            calibrated["query_has_valid_candidate_rate"]
        ),
        "segment_score_p95": float(calibrated["segment_score_p95"]),
        "chain_score_p95": float(calibrated["chain_score_p95"]),
    }


def evaluate_reverse_trajectory(
    *,
    trajectory: Mapping[str, Any],
    paired_target_raw: torch.Tensor,
    future_mean: torch.Tensor,
    future_scale: torch.Tensor,
    physical_contract,
    reference_threshold: float,
) -> Dict[str, Any]:
    target = paired_target_raw
    if target.ndim != 4 or tuple(target.shape[1:]) != (2, DEFAULT_TF, STATE_DIM):
        raise ValueError("paired_target_raw must be [P,2,4,87]")
    target_flat = target.reshape(-1, DEFAULT_TF, STATE_DIM).detach().cpu().numpy()
    checkpoints = trajectory.get("checkpoint_predicted_x0")
    if not isinstance(checkpoints, Mapping):
        raise ValueError("trajectory checkpoints are missing")
    output: Dict[str, Any] = {
        "schema": TRAJECTORY_SCHEMA,
        "checkpoint_iterations": dict(trajectory["checkpoint_iterations"]),
        "checkpoints": {},
    }
    ordered_timesteps = [value for value in TRAJECTORY_TIMESTEPS]
    for timestep in ordered_timesteps:
        pool_z = checkpoints[str(timestep)]
        raw_metrics = r251.paired_reverse_pool_metrics_with_inversion(
            pool_z=pool_z,
            paired_target_raw=target,
            future_mean=future_mean,
            future_scale=future_scale,
            physical_contract=physical_contract,
        )
        branch = paired_full_reverse_branch_support(
            pool_raw=raw_metrics["_pool_raw"],
            paired_target_raw=target_flat,
            reference_threshold=reference_threshold,
        )
        output["checkpoints"][str(timestep)] = {
            "metrics": _compact_reverse_metrics(raw_metrics),
            "branch_support": branch,
        }
    endpoint_raw = r251.paired_reverse_pool_metrics_with_inversion(
        pool_z=trajectory["endpoint_sample"],
        paired_target_raw=target,
        future_mean=future_mean,
        future_scale=future_scale,
        physical_contract=physical_contract,
    )
    endpoint_branch = paired_full_reverse_branch_support(
        pool_raw=endpoint_raw["_pool_raw"],
        paired_target_raw=target_flat,
        reference_threshold=reference_threshold,
    )
    output["endpoint"] = {
        "metrics": _compact_reverse_metrics(endpoint_raw),
        "branch_support": endpoint_branch,
    }
    first_branch_timestep: Optional[int] = None
    first_valid_timestep: Optional[int] = None
    for timestep in ordered_timesteps:
        item = output["checkpoints"][str(timestep)]
        if first_branch_timestep is None and bool(item["branch_support"]["pass"]):
            first_branch_timestep = int(timestep)
        if (
            first_valid_timestep is None
            and float(item["metrics"]["query_has_valid_candidate_rate"]) >= 0.875
        ):
            first_valid_timestep = int(timestep)
    early = output["checkpoints"][str(ordered_timesteps[0])]["metrics"]
    endpoint = output["endpoint"]["metrics"]
    output["attribution"] = {
        "first_branch_support_timestep": first_branch_timestep,
        "first_valid_query_timestep": first_valid_timestep,
        "validity_delta_from_t99": float(
            endpoint["sample_validity_rate"] - early["sample_validity_rate"]
        ),
        "best_ordered_rmse_ratio_to_t99": float(
            endpoint["best_ordered_rmse_mean"]
            / max(early["best_ordered_rmse_mean"], 1.0e-12)
        ),
        "inversion_p95_delta_from_t99": float(
            endpoint["nearest_inversion_p95"] - early["nearest_inversion_p95"]
        ),
        "endpoint_branch_support_pass": bool(
            output["endpoint"]["branch_support"]["pass"]
        ),
        "endpoint_valid_query_pass": bool(
            endpoint["query_has_valid_candidate_rate"] >= 0.875
        ),
    }
    return output


def compare_endpoint_to_r251(
    *,
    observed: Mapping[str, Any],
    expected: Mapping[str, Any],
    gate: AttributionGate = AttributionGate(),
) -> Dict[str, Any]:
    gate.validate()
    metrics = observed["metrics"]
    branch = observed["branch_support"]
    checks = {
        "sample_validity_rate": (
            float(metrics["sample_validity_rate"]),
            float(expected["reverse_sample_validity_rate"]),
            gate.reproduction_rate_tolerance,
            "absolute",
        ),
        "query_has_valid_candidate_rate": (
            float(metrics["query_has_valid_candidate_rate"]),
            float(expected["reverse_query_has_valid_candidate_rate"]),
            gate.reproduction_rate_tolerance,
            "absolute",
        ),
        "best_ordered_rmse_mean": (
            float(metrics["best_ordered_rmse_mean"]),
            float(expected["reverse_best_ordered_rmse_mean"]),
            gate.reproduction_relative_tolerance,
            "relative",
        ),
        "nearest_inversion_p95": (
            float(metrics["nearest_inversion_p95"]),
            float(expected["reverse_nearest_inversion_p95"]),
            gate.reproduction_rate_tolerance,
            "absolute",
        ),
        "both_branch_support_rate": (
            float(branch["both_branch_support_rate"]),
            float(expected["both_branch_support_rate"]),
            gate.reproduction_rate_tolerance,
            "absolute",
        ),
    }
    details: Dict[str, Any] = {}
    passed = True
    for name, (actual, target, tolerance, mode) in checks.items():
        if mode == "relative":
            error = abs(actual - target) / max(abs(target), 1.0e-12)
        else:
            error = abs(actual - target)
        item_pass = bool(error <= tolerance)
        details[name] = {
            "observed": actual,
            "expected": target,
            "error": float(error),
            "tolerance": float(tolerance),
            "mode": mode,
            "pass": item_pass,
        }
        passed = passed and item_pass
    return {"checks": details, "pass": bool(passed)}


def strip_runtime_objects(value: Any) -> Any:
    if torch.is_tensor(value):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        raise TypeError("runtime tensor must not be serialized")
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {
            str(key): strip_runtime_objects(item)
            for key, item in value.items()
            if not str(key).startswith("_")
        }
    if isinstance(value, tuple):
        return [strip_runtime_objects(item) for item in value]
    if isinstance(value, list):
        return [strip_runtime_objects(item) for item in value]
    return value


def _variant_mechanisms(value: Mapping[str, Any]) -> List[str]:
    mechanisms: List[str] = []
    gate = value.get("one_step_gate_decomposition", {})
    attribution = value.get("one_step_attribution", {})
    pipeline = attribution.get("pipeline_controls", {})
    by_timestep = attribution.get("by_timestep", {})
    per_timestep_gradient = value.get("per_timestep_gradient", {})
    trajectory = value.get("trajectory", {})
    endpoint_attribution = trajectory.get("attribution", {})
    if (
        bool(pipeline.get("pass"))
        and bool(gate.get("branch_audit_pass"))
        and not bool(gate.get("composite_one_step_pass"))
    ):
        mechanisms.append("composite_gate_conflation")
    if bool(value.get("geometry_objective_selectable")):
        final_values = per_timestep_gradient.get("final", {})
        if final_values and not bool(
            per_timestep_gradient.get("final_all_timesteps_pass")
        ):
            mechanisms.append("per_timestep_gradient_miscalibration")
    poor_timesteps = 0
    for item in by_timestep.values():
        if (
            bool(item.get("forward_signal_contract_pass"))
            and (
                float(item.get("model_separation_ratio", {}).get("p50", 0.0)) < 0.50
                or float(item.get("residual_delta_cosine", {}).get("p50", 0.0)) < 0.50
            )
        ):
            poor_timesteps += 1
    if bool(pipeline.get("pass")) and poor_timesteps >= 2:
        mechanisms.append("sigma_scaled_residual_gain_deficiency")
    if (
        not bool(gate.get("branch_audit_pass"))
        and bool(endpoint_attribution.get("endpoint_branch_support_pass"))
        and bool(endpoint_attribution.get("endpoint_valid_query_pass"))
    ):
        mechanisms.append("reverse_trajectory_recovery")
    return mechanisms


def classify_attribution(report: Mapping[str, Any]) -> Dict[str, Any]:
    variants = report.get("variants", {})
    if set(variants) != set(DIAGNOSTIC_OBJECTIVE_NAMES):
        return {
            "root_cause": "phase314b_r252_diagnostic_matrix_incomplete",
            "supported_mechanisms": [],
            "next_stage": "repair the fixed attribution matrix",
            "train_only_recommendation": None,
        }
    if any(not bool(value.get("training_completed")) for value in variants.values()):
        return {
            "root_cause": "phase314b_r252_training_reproduction_failed",
            "supported_mechanisms": [],
            "next_stage": "repair deterministic paired-model reproduction",
            "train_only_recommendation": None,
        }
    if any(
        not bool(
            value.get("one_step_attribution", {})
            .get("pipeline_controls", {})
            .get("pass")
        )
        for value in variants.values()
    ):
        return {
            "root_cause": "phase314b_r252_metric_pipeline_contract_failed",
            "supported_mechanisms": [],
            "next_stage": "repair one-step decode, pairing, or matched-noise instrumentation",
            "train_only_recommendation": None,
        }
    if any(
        not bool(value.get("r251_endpoint_reproduction", {}).get("pass"))
        for value in variants.values()
    ):
        return {
            "root_cause": "phase314b_r252_r251_reverse_reproduction_failed",
            "supported_mechanisms": [],
            "next_stage": "repair deterministic r2.5.1 reverse reproduction",
            "train_only_recommendation": None,
        }

    mechanisms_by_name = {
        name: _variant_mechanisms(value)
        for name, value in variants.items()
    }
    supported: List[str] = []
    for name in DIAGNOSTIC_OBJECTIVE_NAMES:
        for mechanism in mechanisms_by_name[name]:
            if mechanism not in supported:
                supported.append(mechanism)

    selectable_names = [
        name
        for name in DIAGNOSTIC_OBJECTIVE_NAMES
        if bool(variants[name].get("geometry_objective_selectable"))
    ]
    if not selectable_names:
        return {
            "root_cause": "phase314b_r252_diagnostic_matrix_incomplete",
            "supported_mechanisms": supported,
            "next_stage": "restore the fixed selectable attribution objectives",
            "train_only_recommendation": None,
        }
    selectable_mechanisms = {
        name: set(mechanisms_by_name[name])
        for name in selectable_names
    }
    composite_supported = any(
        "composite_gate_conflation" in value
        for value in selectable_mechanisms.values()
    )
    gradients_bad_for_all_selectable = all(
        "per_timestep_gradient_miscalibration" in value
        for value in selectable_mechanisms.values()
    )
    sigma_deficiency_supported = any(
        "sigma_scaled_residual_gain_deficiency" in value
        and "per_timestep_gradient_miscalibration" not in value
        for value in selectable_mechanisms.values()
    )
    reverse_recovery_supported = any(
        "reverse_trajectory_recovery" in value
        for value in selectable_mechanisms.values()
    )

    if composite_supported:
        root = "phase314b_r252_composite_one_step_gate_conflation_supported"
        next_stage = "separate exact reconstruction and branch-transport gates without changing training"
    elif gradients_bad_for_all_selectable:
        root = "phase314b_r252_timestep_gradient_miscalibration_supported"
        next_stage = "test timestep-stratified gradient calibration under the fixed frozen model"
    elif sigma_deficiency_supported:
        root = "phase314b_r252_sigma_scaled_residual_gain_deficiency_supported"
        next_stage = "test a train-only low-noise x0 correction or sigma-compensated residual objective"
    elif reverse_recovery_supported:
        root = "phase314b_r252_reverse_trajectory_recovery_supported"
        next_stage = "replicate trajectory recovery before revising the one-step gate"
    else:
        root = "phase314b_r252_paired_transport_failure_confirmed"
        next_stage = "reformulate paired low/mid transport under the fixed frozen architecture"
    return {
        "root_cause": root,
        "supported_mechanisms": supported,
        "next_stage": next_stage,
        "train_only_recommendation": None,
    }
