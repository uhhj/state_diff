"""Phase3.14b-r1 DDPM sampling-scale diagnostics.

This module is diagnostic-only. It does not modify the immutable cache,
Phase3.14b checkpoints, formal data, or the checkpoint-bound Phase3.14b
source files.
"""
from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from ccda_phase3.phase314a_contract import (
    BEAD_XY_DIM,
    CACHE_ROWS,
    Standardizer,
    load_npz_no_pickle,
    sha256_file,
    strict_json_dump,
    strict_json_load,
)
from ccda_phase3.phase314a_metrics import chamfer_xy
from ccda_phase3.phase314b_contract import (
    CACHE_SHA256,
    fixed_balanced_eval_indices,
    future_standardizer,
    input_values_and_standardizer,
    load_locked_cache,
)
from ccda_phase3.phase314b_diffusion import make_ddpm_scheduler
from ccda_phase3.phase314b_models import build_denoiser
from ccda_phase3.schema_v2 import DEFAULT_TF, STATE_DIM


PHASE = "phase3_14b_r1"
SELECTED_CHECKPOINT_SHA256 = (
    "28d876028d46d9efc71ae70c16b44a6933806654784538a366b95c89275d8e2a"
)
TIMESTEPS = (0, 10, 25, 50, 75, 90, 99)
TIMESTEP_BINS = (
    (0, 9),
    (10, 24),
    (25, 49),
    (50, 74),
    (75, 89),
    (90, 99),
)
TRACE_TIMESTEPS = frozenset((99, 90, 75, 50, 25, 10, 0))
PARTIAL_START_TIMESTEPS = (10, 25, 50, 75, 99)

LOCKED_PHASE314B_PATHS = (
    "ccda_phase3/phase314b_contract.py",
    "ccda_phase3/phase314b_models.py",
    "ccda_phase3/phase314b_diffusion.py",
    "ccda_phase3/phase314b_metrics.py",
    "scripts/phase3_14b_preflight.py",
    "scripts/phase3_14b_train_ddpm.py",
    "scripts/phase3_14b_eval_ddpm.py",
    "scripts/phase3_14b_analyze.py",
    "scripts/phase3_14b_run.sh",
)


def finite_stats(value: np.ndarray) -> Dict[str, float]:
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("finite_stats requires a non-empty finite array")
    percentiles = np.percentile(
        array,
        [0.1, 1.0, 50.0, 99.0, 99.9],
    )
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "rms": float(np.sqrt(np.mean(array * array))),
        "abs_max": float(np.max(np.abs(array))),
        "p0_1": float(percentiles[0]),
        "p1": float(percentiles[1]),
        "p50": float(percentiles[2]),
        "p99": float(percentiles[3]),
        "p99_9": float(percentiles[4]),
        "abs_p99": float(np.percentile(np.abs(array), 99.0)),
        "abs_p99_9": float(np.percentile(np.abs(array), 99.9)),
    }


def active_values(
    value: np.ndarray,
    active_mask: np.ndarray,
) -> np.ndarray:
    array = np.asarray(value)
    mask = np.asarray(active_mask, dtype=np.bool_)
    if array.shape[-2:] != mask.shape:
        raise ValueError(
            f"value tail {array.shape[-2:]} != active mask {mask.shape}"
        )
    expanded = np.broadcast_to(mask, array.shape)
    return np.asarray(array[expanded])


def standardization_audit(
    arrays: Mapping[str, np.ndarray],
) -> Dict[str, Any]:
    standardizer = future_standardizer(arrays)
    future = np.asarray(arrays["y_state"], dtype=np.float32)
    transformed = standardizer.transform(future)
    reconstructed = standardizer.inverse(transformed)
    roundtrip_error = float(
        np.max(np.abs(reconstructed.astype(np.float64) - future))
    )

    active = np.asarray(standardizer.active, dtype=np.bool_)
    scales = np.asarray(standardizer.scale, dtype=np.float64)
    active_scales = scales[active]
    if active_scales.size == 0:
        raise RuntimeError("future active mask is empty")
    if np.any(active_scales <= 0) or not np.all(np.isfinite(active_scales)):
        raise RuntimeError("future active scales are invalid")

    split_values = np.asarray(arrays["split_name"]).astype(str)
    blocks: Dict[str, Any] = {}
    for split in ("train", "val", "test"):
        selected = split_values == split
        if not np.any(selected):
            raise RuntimeError(f"empty split {split}")
        raw = future[selected]
        z = transformed[selected]
        blocks[split] = {
            "raw_all": finite_stats(raw),
            "z_active": finite_stats(active_values(z, active)),
            "raw_bead_xy": finite_stats(raw[..., :BEAD_XY_DIM]),
            "z_bead_xy_active": finite_stats(
                active_values(
                    z[..., :BEAD_XY_DIM],
                    active[:, :BEAD_XY_DIM],
                )
            ),
            "raw_robot": finite_stats(raw[..., BEAD_XY_DIM:]),
            "z_robot_active": finite_stats(
                active_values(
                    z[..., BEAD_XY_DIM:],
                    active[:, BEAD_XY_DIM:],
                )
            ),
        }

    horizons = {}
    for horizon in range(DEFAULT_TF):
        horizons[str(horizon)] = {
            "raw": finite_stats(future[:, horizon]),
            "z_active": finite_stats(
                transformed[:, horizon, active[horizon]]
            ),
            "active_dimensions": int(np.sum(active[horizon])),
            "scale_active": finite_stats(scales[horizon, active[horizon]]),
        }

    return {
        "roundtrip_max_abs": roundtrip_error,
        "active_dimensions": int(np.sum(active)),
        "inactive_dimensions": int(active.size - np.sum(active)),
        "active_scale_min": float(np.min(active_scales)),
        "active_scale_max": float(np.max(active_scales)),
        "active_scale_ratio": float(
            np.max(active_scales) / np.min(active_scales)
        ),
        "splits": blocks,
        "horizons": horizons,
    }


def scheduler_coefficients(scheduler) -> List[Dict[str, float]]:
    alpha_bar = scheduler.alphas_cumprod.detach().cpu().double().numpy()
    beta = scheduler.betas.detach().cpu().double().numpy()
    rows = []
    for timestep in range(len(alpha_bar)):
        value = float(alpha_bar[timestep])
        rows.append(
            {
                "timestep": timestep,
                "beta": float(beta[timestep]),
                "alpha_bar": value,
                "sqrt_alpha_bar": float(math.sqrt(value)),
                "sqrt_one_minus_alpha_bar": float(
                    math.sqrt(max(0.0, 1.0 - value))
                ),
                "x0_error_amplification": float(
                    1.0 / math.sqrt(max(value, np.finfo(np.float64).tiny))
                ),
            }
        )
    return rows


def predict_x0_from_epsilon(
    sample: torch.Tensor,
    epsilon: torch.Tensor,
    alpha_bar_t: torch.Tensor,
) -> torch.Tensor:
    alpha = alpha_bar_t.to(
        device=sample.device,
        dtype=sample.dtype,
    )
    while alpha.ndim < sample.ndim:
        alpha = alpha.unsqueeze(-1)
    return (
        sample - torch.sqrt(1.0 - alpha) * epsilon
    ) / torch.sqrt(alpha)


def true_epsilon_scheduler_oracle(
    scheduler,
    *,
    x0: torch.Tensor,
    noise: torch.Tensor,
    timesteps: Sequence[int] = TIMESTEPS,
) -> List[Dict[str, float]]:
    if x0.shape != noise.shape:
        raise ValueError("x0/noise shape mismatch")
    rows: List[Dict[str, float]] = []
    for raw_timestep in timesteps:
        timestep = int(raw_timestep)
        batch_t = torch.full(
            (x0.shape[0],),
            timestep,
            device=x0.device,
            dtype=torch.long,
        )
        xt = scheduler.add_noise(x0, noise, batch_t)
        result = scheduler.step(
            noise,
            timestep,
            xt,
            generator=torch.Generator(device=x0.device).manual_seed(
                910000 + timestep
            ),
        )
        manual = predict_x0_from_epsilon(
            xt,
            noise,
            scheduler.alphas_cumprod[timestep],
        )
        rows.append(
            {
                "timestep": timestep,
                "diffusers_x0_max_abs_error": float(
                    torch.max(
                        torch.abs(result.pred_original_sample - x0)
                    ).item()
                ),
                "manual_x0_max_abs_error": float(
                    torch.max(torch.abs(manual - x0)).item()
                ),
                "diffusers_vs_manual_max_abs": float(
                    torch.max(
                        torch.abs(result.pred_original_sample - manual)
                    ).item()
                ),
                "xt_abs_max": float(torch.max(torch.abs(xt)).item()),
                "x0_abs_max": float(torch.max(torch.abs(x0)).item()),
                "pred_x0_abs_max": float(
                    torch.max(
                        torch.abs(result.pred_original_sample)
                    ).item()
                ),
            }
        )
    return rows


def manual_ddpm_step(
    *,
    scheduler,
    epsilon: torch.Tensor,
    timestep: int,
    sample: torch.Tensor,
    variance_noise: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Diffusers 0.11.1 fixed-small DDPM step with explicit variance noise."""
    t = int(timestep)
    alpha_prod_t = scheduler.alphas_cumprod[t].to(
        device=sample.device,
        dtype=sample.dtype,
    )
    alpha_prod_prev = (
        scheduler.alphas_cumprod[t - 1].to(
            device=sample.device,
            dtype=sample.dtype,
        )
        if t > 0
        else scheduler.one.to(
            device=sample.device,
            dtype=sample.dtype,
        )
    )
    beta_prod_t = 1.0 - alpha_prod_t
    beta_prod_prev = 1.0 - alpha_prod_prev
    current_alpha_t = alpha_prod_t / alpha_prod_prev
    current_beta_t = 1.0 - current_alpha_t

    pred_x0 = predict_x0_from_epsilon(
        sample,
        epsilon,
        alpha_prod_t,
    )
    pred_x0_coeff = (
        torch.sqrt(alpha_prod_prev)
        * current_beta_t
        / beta_prod_t
    )
    current_coeff = (
        torch.sqrt(current_alpha_t)
        * beta_prod_prev
        / beta_prod_t
    )
    mean = pred_x0_coeff * pred_x0 + current_coeff * sample

    if t == 0:
        return mean, pred_x0
    if variance_noise is None:
        raise ValueError("variance_noise is required for t > 0")
    variance = (
        beta_prod_prev / beta_prod_t * current_beta_t
    )
    variance = torch.clamp(variance, min=1e-20)
    return mean + torch.sqrt(variance) * variance_noise, pred_x0


def manual_step_equivalence(
    scheduler,
    *,
    epsilon: torch.Tensor,
    sample: torch.Tensor,
    timesteps: Sequence[int] = TIMESTEPS,
) -> List[Dict[str, float]]:
    rows = []
    for raw_timestep in timesteps:
        timestep = int(raw_timestep)
        seed = 920000 + timestep
        generator_a = torch.Generator(
            device=sample.device
        ).manual_seed(seed)
        generator_b = torch.Generator(
            device=sample.device
        ).manual_seed(seed)
        official = scheduler.step(
            epsilon,
            timestep,
            sample,
            generator=generator_a,
        )
        variance_noise = (
            torch.randn(
                sample.shape,
                generator=generator_b,
                device=sample.device,
                dtype=sample.dtype,
            )
            if timestep > 0
            else None
        )
        manual_prev, manual_x0 = manual_ddpm_step(
            scheduler=scheduler,
            epsilon=epsilon,
            timestep=timestep,
            sample=sample,
            variance_noise=variance_noise,
        )
        rows.append(
            {
                "timestep": timestep,
                "prev_sample_max_abs": float(
                    torch.max(
                        torch.abs(
                            official.prev_sample - manual_prev
                        )
                    ).item()
                ),
                "pred_x0_max_abs": float(
                    torch.max(
                        torch.abs(
                            official.pred_original_sample - manual_x0
                        )
                    ).item()
                ),
            }
        )
    return rows


def load_selected_model(
    *,
    root: Path,
    arrays: Mapping[str, np.ndarray],
    device: torch.device,
    use_ema: bool,
) -> Tuple[torch.nn.Module, np.ndarray, Dict[str, Any]]:
    selection = strict_json_load(
        root / "reports/phase3_14b_selected_model.json"
    )
    checkpoint_path = root / str(selection["checkpoint"])
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    actual = sha256_file(checkpoint_path)
    if actual != SELECTED_CHECKPOINT_SHA256:
        raise RuntimeError(
            f"selected checkpoint changed: {actual}"
        )
    if actual != selection["checkpoint_sha256"]:
        raise RuntimeError("selection/checkpoint SHA256 mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("phase") != "phase3_14b":
        raise RuntimeError("selected checkpoint is not Phase3.14b")
    if checkpoint.get("cache_sha256") != CACHE_SHA256:
        raise RuntimeError("selected checkpoint used another cache")
    for relative, expected in checkpoint["source_sha256"].items():
        actual_source = sha256_file(root / relative)
        if actual_source != expected:
            raise RuntimeError(
                f"checkpoint-bound source changed: {relative}"
            )

    variant = str(checkpoint["input_variant"])
    x_raw, x_standardizer = input_values_and_standardizer(
        arrays,
        variant,
    )
    condition_z = x_standardizer.transform(x_raw)
    model = build_denoiser(
        str(checkpoint["model_family"]),
        condition_dim=condition_z.shape[1],
    ).to(device)
    state = (
        checkpoint["ema_state_dict"]["shadow"]
        if use_ema
        else checkpoint["model_state_dict"]
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    checkpoint_metadata = {
        key: value
        for key, value in checkpoint.items()
        if key not in {"model_state_dict", "ema_state_dict"}
    }
    return model, condition_z, {
        "selection": selection,
        "checkpoint": checkpoint_metadata,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": actual,
        "weights": "ema" if use_ema else "raw",
    }


@torch.no_grad()
def timestep_model_audit(
    *,
    model: torch.nn.Module,
    scheduler,
    x0_z: torch.Tensor,
    condition_z: torch.Tensor,
    active_mask: torch.Tensor,
    rows_per_timestep: int,
    seed: int,
) -> List[Dict[str, float]]:
    active = active_mask.to(device=x0_z.device, dtype=torch.bool)
    generator = torch.Generator(
        device=x0_z.device
    ).manual_seed(int(seed))
    rows: List[Dict[str, float]] = []
    for bin_start, bin_end in TIMESTEP_BINS:
        count = int(rows_per_timestep)
        selected = torch.arange(count, device=x0_z.device) % x0_z.shape[0]
        x0 = x0_z[selected]
        cond = condition_z[selected]
        timesteps = torch.randint(
            int(bin_start),
            int(bin_end) + 1,
            (count,),
            generator=generator,
            device=x0_z.device,
            dtype=torch.long,
        )
        noise = torch.randn(
            x0.shape,
            generator=generator,
            device=x0.device,
            dtype=x0.dtype,
        )
        noise = torch.where(
            active[None, :, :],
            noise,
            torch.zeros_like(noise),
        )
        xt = scheduler.add_noise(x0, noise, timesteps)
        prediction = model(xt, timesteps, cond)
        prediction = torch.where(
            active[None, :, :],
            prediction,
            torch.zeros_like(prediction),
        )
        alpha_bar = scheduler.alphas_cumprod[
            timesteps
        ].to(device=x0.device, dtype=x0.dtype)
        pred_x0 = predict_x0_from_epsilon(
            xt,
            prediction,
            alpha_bar,
        )
        active_expanded = active[None, :, :].expand_as(prediction)
        eps_error = (prediction - noise)[active_expanded]
        x0_error = (pred_x0 - x0)[active_expanded]
        rows.append(
            {
                "bin": f"{bin_start}-{bin_end}",
                "rows": count,
                "epsilon_mse": float(
                    torch.mean(eps_error.square()).item()
                ),
                "epsilon_mae": float(
                    torch.mean(torch.abs(eps_error)).item()
                ),
                "pred_x0_mse": float(
                    torch.mean(x0_error.square()).item()
                ),
                "pred_x0_mae": float(
                    torch.mean(torch.abs(x0_error)).item()
                ),
                "pred_x0_abs_p99": float(
                    torch.quantile(
                        torch.abs(pred_x0[active_expanded]),
                        0.99,
                    ).item()
                ),
                "pred_x0_abs_max": float(
                    torch.max(
                        torch.abs(pred_x0[active_expanded])
                    ).item()
                ),
                "true_x0_abs_max": float(
                    torch.max(torch.abs(x0[active_expanded])).item()
                ),
            }
        )
    return rows


def _raw_xy_abs_max(
    future_z: torch.Tensor,
    standardizer: Standardizer,
) -> float:
    raw = standardizer.inverse(
        future_z.detach().cpu().numpy().astype(np.float32)
    )
    return float(np.max(np.abs(raw[..., :BEAD_XY_DIM])))


@torch.no_grad()
def reverse_chain_trace(
    *,
    model: torch.nn.Module,
    scheduler,
    condition_z: torch.Tensor,
    active_mask: torch.Tensor,
    future_std: Standardizer,
    seed: int,
    posterior_noise: bool,
) -> Tuple[torch.Tensor, List[Dict[str, float]]]:
    device = condition_z.device
    active = active_mask.to(device=device, dtype=torch.bool)
    generator = torch.Generator(device=device).manual_seed(int(seed))
    current = torch.randn(
        (
            condition_z.shape[0],
            DEFAULT_TF,
            STATE_DIM,
        ),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    current = torch.where(
        active[None, :, :],
        current,
        torch.zeros_like(current),
    )
    trace: List[Dict[str, float]] = []
    scheduler.set_timesteps(100, device=device)

    for scheduler_timestep in scheduler.timesteps:
        timestep = int(scheduler_timestep.item())
        batch_t = torch.full(
            (condition_z.shape[0],),
            timestep,
            device=device,
            dtype=torch.long,
        )
        epsilon = model(current, batch_t, condition_z)
        epsilon = torch.where(
            active[None, :, :],
            epsilon,
            torch.zeros_like(epsilon),
        )
        variance_noise = (
            torch.randn(
                current.shape,
                generator=generator,
                device=device,
                dtype=current.dtype,
            )
            if posterior_noise and timestep > 0
            else (
                torch.zeros_like(current)
                if timestep > 0
                else None
            )
        )
        previous, pred_x0 = manual_ddpm_step(
            scheduler=scheduler,
            epsilon=epsilon,
            timestep=timestep,
            sample=current,
            variance_noise=variance_noise,
        )
        previous = torch.where(
            active[None, :, :],
            previous,
            torch.zeros_like(previous),
        )
        if timestep in TRACE_TIMESTEPS:
            active_current = current[
                active[None, :, :].expand_as(current)
            ]
            active_eps = epsilon[
                active[None, :, :].expand_as(epsilon)
            ]
            active_x0 = pred_x0[
                active[None, :, :].expand_as(pred_x0)
            ]
            trace.append(
                {
                    "timestep": timestep,
                    "current_rms": float(
                        torch.sqrt(
                            torch.mean(active_current.square())
                        ).item()
                    ),
                    "current_abs_max": float(
                        torch.max(torch.abs(active_current)).item()
                    ),
                    "epsilon_rms": float(
                        torch.sqrt(
                            torch.mean(active_eps.square())
                        ).item()
                    ),
                    "epsilon_abs_max": float(
                        torch.max(torch.abs(active_eps)).item()
                    ),
                    "pred_x0_rms": float(
                        torch.sqrt(
                            torch.mean(active_x0.square())
                        ).item()
                    ),
                    "pred_x0_abs_max": float(
                        torch.max(torch.abs(active_x0)).item()
                    ),
                    "pred_x0_raw_xy_abs_max": _raw_xy_abs_max(
                        pred_x0,
                        future_std,
                    ),
                }
            )
        current = previous
    return current, trace


def final_chamfer_rows(
    prediction_raw: np.ndarray,
    target_raw: np.ndarray,
) -> np.ndarray:
    prediction = np.asarray(prediction_raw, dtype=np.float32)
    target = np.asarray(target_raw, dtype=np.float32)
    if prediction.shape != target.shape:
        raise ValueError("prediction/target shape mismatch")
    values = []
    for pred, truth in zip(prediction, target):
        values.append(
            chamfer_xy(
                pred[-1, :BEAD_XY_DIM].reshape(-1, 2),
                truth[-1, :BEAD_XY_DIM].reshape(-1, 2),
            )
        )
    return np.asarray(values, dtype=np.float64)


@torch.no_grad()
def partial_denoise_audit(
    *,
    model: torch.nn.Module,
    scheduler,
    x0_z: torch.Tensor,
    target_raw: np.ndarray,
    condition_z: torch.Tensor,
    active_mask: torch.Tensor,
    future_std: Standardizer,
    seed: int,
) -> List[Dict[str, float]]:
    device = x0_z.device
    active = active_mask.to(device=device, dtype=torch.bool)
    rows: List[Dict[str, float]] = []
    for start_timestep in PARTIAL_START_TIMESTEPS:
        generator = torch.Generator(
            device=device
        ).manual_seed(int(seed) + int(start_timestep))
        forward_noise = torch.randn(
            x0_z.shape,
            generator=generator,
            device=device,
            dtype=x0_z.dtype,
        )
        forward_noise = torch.where(
            active[None, :, :],
            forward_noise,
            torch.zeros_like(forward_noise),
        )
        batch_t = torch.full(
            (x0_z.shape[0],),
            int(start_timestep),
            device=device,
            dtype=torch.long,
        )
        current = scheduler.add_noise(
            x0_z,
            forward_noise,
            batch_t,
        )
        for timestep in range(int(start_timestep), -1, -1):
            model_t = torch.full(
                (x0_z.shape[0],),
                timestep,
                device=device,
                dtype=torch.long,
            )
            epsilon = model(current, model_t, condition_z)
            epsilon = torch.where(
                active[None, :, :],
                epsilon,
                torch.zeros_like(epsilon),
            )
            variance_noise = (
                torch.zeros_like(current)
                if timestep > 0
                else None
            )
            current, _ = manual_ddpm_step(
                scheduler=scheduler,
                epsilon=epsilon,
                timestep=timestep,
                sample=current,
                variance_noise=variance_noise,
            )
            current = torch.where(
                active[None, :, :],
                current,
                torch.zeros_like(current),
            )

        active_expanded = active[None, :, :].expand_as(current)
        z_error = (current - x0_z)[active_expanded]
        raw_prediction = future_std.inverse(
            current.detach().cpu().numpy().astype(np.float32)
        )
        chamfer = final_chamfer_rows(
            raw_prediction,
            target_raw,
        )
        rows.append(
            {
                "start_timestep": int(start_timestep),
                "z_mse": float(torch.mean(z_error.square()).item()),
                "z_mae": float(torch.mean(torch.abs(z_error)).item()),
                "z_abs_max": float(
                    torch.max(torch.abs(current[active_expanded])).item()
                ),
                "final_chamfer_mean": float(np.mean(chamfer)),
                "final_chamfer_median": float(np.median(chamfer)),
                "raw_xy_abs_max": float(
                    np.max(np.abs(raw_prediction[..., :BEAD_XY_DIM]))
                ),
            }
        )
    return rows


def compare_weight_sets(
    ema: Mapping[str, Any],
    raw: Mapping[str, Any],
) -> Dict[str, Any]:
    ema_bins = {item["bin"]: item for item in ema["timestep_bins"]}
    raw_bins = {item["bin"]: item for item in raw["timestep_bins"]}
    high = "90-99"
    low = "0-9"
    return {
        "ema_high_x0_mse": ema_bins[high]["pred_x0_mse"],
        "raw_high_x0_mse": raw_bins[high]["pred_x0_mse"],
        "ema_low_x0_mse": ema_bins[low]["pred_x0_mse"],
        "raw_low_x0_mse": raw_bins[low]["pred_x0_mse"],
        "ema_high_to_low_ratio": float(
            ema_bins[high]["pred_x0_mse"]
            / max(ema_bins[low]["pred_x0_mse"], 1e-12)
        ),
        "raw_high_to_low_ratio": float(
            raw_bins[high]["pred_x0_mse"]
            / max(raw_bins[low]["pred_x0_mse"], 1e-12)
        ),
    }
