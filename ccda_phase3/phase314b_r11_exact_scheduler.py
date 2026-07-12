"""Phase3.14b-r1.1 exact Diffusers-0.11.1 scheduler equivalence.

This module does not modify the immutable cache, DDPM checkpoint, formal
dataset, or any checkpoint-bound Phase3.14b source.  It mirrors the installed
Diffusers 0.11.1 DDPMScheduler.step arithmetic and reruns the affected
reverse-chain diagnostics with the same selected checkpoint.
"""
from __future__ import annotations

import hashlib
import inspect
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from ccda_phase3.phase314a_contract import (
    BEAD_XY_DIM,
    Standardizer,
    sha256_file,
    strict_json_load,
)
from ccda_phase3.phase314b_contract import (
    CACHE_SHA256,
    fixed_balanced_eval_indices,
    future_standardizer,
    load_locked_cache,
)
from ccda_phase3.phase314b_diffusion import make_ddpm_scheduler
from ccda_phase3.phase314b_r1_diagnostics import (
    SELECTED_CHECKPOINT_SHA256,
    compare_weight_sets,
    final_chamfer_rows,
    finite_stats,
    load_selected_model,
    scheduler_coefficients,
    standardization_audit,
    timestep_model_audit,
    true_epsilon_scheduler_oracle,
)
from ccda_phase3.schema_v2 import DEFAULT_TF, STATE_DIM


PHASE = "phase3_14b_r11"
DIFFUSERS_REQUIRED_VERSION = "0.11.1"
TIMESTEPS = (0, 10, 25, 50, 75, 90, 99)
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


def scheduler_source_contract() -> Dict[str, Any]:
    import diffusers
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

    version = str(diffusers.__version__)
    if version != DIFFUSERS_REQUIRED_VERSION:
        raise RuntimeError(
            f"Diffusers version {version} != {DIFFUSERS_REQUIRED_VERSION}"
        )

    step_source = inspect.getsource(DDPMScheduler.step)
    variance_source = inspect.getsource(DDPMScheduler._get_variance)
    required_step_tokens = (
        "self.betas[t]",
        "self.alphas[t]",
        "self._get_variance",
        "pred_original_sample_coeff",
        "current_sample_coeff",
    )
    missing = [token for token in required_step_tokens if token not in step_source]
    if missing:
        raise RuntimeError(
            f"installed Diffusers step source lacks required tokens: {missing}"
        )
    required_variance_tokens = (
        "self.betas[t]",
        "torch.clamp(variance, min=1e-20)",
    )
    missing_variance = [
        token for token in required_variance_tokens
        if token not in variance_source
    ]
    if missing_variance:
        raise RuntimeError(
            "installed Diffusers variance source lacks required tokens: "
            f"{missing_variance}"
        )

    source_path = Path(inspect.getsourcefile(DDPMScheduler) or "")
    if not source_path.is_file():
        raise RuntimeError("cannot resolve installed DDPMScheduler source file")
    return {
        "diffusers_version": version,
        "scheduler_class": (
            f"{DDPMScheduler.__module__}.{DDPMScheduler.__name__}"
        ),
        "source_path": str(source_path),
        "source_file_sha256": sha256_file(source_path),
        "step_source_sha256": hashlib.sha256(
            step_source.encode("utf-8")
        ).hexdigest(),
        "variance_source_sha256": hashlib.sha256(
            variance_source.encode("utf-8")
        ).hexdigest(),
        "step_required_tokens": list(required_step_tokens),
        "variance_required_tokens": list(required_variance_tokens),
    }


def _assert_scheduler_contract(scheduler) -> None:
    if str(scheduler.config.prediction_type) != "epsilon":
        raise RuntimeError("formal scheduler must predict epsilon")
    if str(scheduler.config.variance_type) != "fixed_small":
        raise RuntimeError("formal scheduler must use fixed_small variance")
    if bool(scheduler.config.clip_sample):
        raise RuntimeError("formal scheduler clip_sample must be false")
    if bool(getattr(scheduler.config, "thresholding", False)):
        raise RuntimeError("formal scheduler thresholding must be false")


def exact_diffusers_0111_components(
    *,
    scheduler,
    model_output: torch.Tensor,
    timestep: int,
    sample: torch.Tensor,
    variance_noise: Optional[torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Mirror Diffusers 0.11.1 DDPMScheduler.step operation-by-operation.

    Do not replace scheduler.betas[t] or scheduler.alphas[t] with algebraically
    equivalent expressions.  The purpose is floating-point implementation
    equivalence, not only mathematical equivalence.
    """
    _assert_scheduler_contract(scheduler)
    t = int(timestep)
    if model_output.shape != sample.shape:
        raise ValueError("model_output/sample shape mismatch")

    # Exact order from Diffusers v0.11.1 scheduling_ddpm.py.
    alpha_prod_t = scheduler.alphas_cumprod[t]
    alpha_prod_t_prev = (
        scheduler.alphas_cumprod[t - 1]
        if t > 0
        else scheduler.one
    )
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev

    pred_original_sample = (
        sample
        - beta_prod_t ** 0.5 * model_output
    ) / alpha_prod_t ** 0.5

    pred_original_sample_coeff = (
        alpha_prod_t_prev ** 0.5
        * scheduler.betas[t]
        / beta_prod_t
    )
    current_sample_coeff = (
        scheduler.alphas[t] ** 0.5
        * beta_prod_t_prev
        / beta_prod_t
    )
    posterior_mean = (
        pred_original_sample_coeff * pred_original_sample
        + current_sample_coeff * sample
    )

    if t > 0:
        if variance_noise is None:
            raise ValueError("variance_noise is required when timestep > 0")
        variance = scheduler._get_variance(t)
        variance_term = variance ** 0.5 * variance_noise
    else:
        variance = torch.zeros(
            (),
            device=sample.device,
            dtype=sample.dtype,
        )
        variance_term = torch.zeros_like(sample)

    return {
        "pred_original_sample": pred_original_sample,
        "pred_original_sample_coeff": pred_original_sample_coeff,
        "current_sample_coeff": current_sample_coeff,
        "posterior_mean": posterior_mean,
        "variance": variance,
        "variance_term": variance_term,
        "prev_sample": posterior_mean + variance_term,
    }


def legacy_algebraic_components(
    *,
    scheduler,
    model_output: torch.Tensor,
    timestep: int,
    sample: torch.Tensor,
    variance_noise: Optional[torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Reproduce the r1 algebraically equivalent but non-exact implementation."""
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
    pred_x0 = (
        sample
        - torch.sqrt(beta_prod_t) * model_output
    ) / torch.sqrt(alpha_prod_t)
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
    if t > 0:
        if variance_noise is None:
            raise ValueError("variance_noise required")
        variance = (
            beta_prod_prev / beta_prod_t * current_beta_t
        )
        variance = torch.clamp(variance, min=1e-20)
        variance_term = torch.sqrt(variance) * variance_noise
    else:
        variance_term = torch.zeros_like(sample)
    return {
        "pred_original_sample": pred_x0,
        "posterior_mean": mean,
        "variance_term": variance_term,
        "prev_sample": mean + variance_term,
    }


def tensor_difference(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> Dict[str, Any]:
    if actual.shape != expected.shape:
        raise ValueError("tensor shape mismatch")
    difference = (actual - expected).detach()
    absolute = torch.abs(difference)
    max_flat = int(torch.argmax(absolute).item())
    unravel = np.unravel_index(max_flat, tuple(absolute.shape))
    rms = float(torch.sqrt(torch.mean(difference.double() ** 2)).item())
    expected_rms = float(
        torch.sqrt(torch.mean(expected.detach().double() ** 2)).item()
    )
    max_abs = float(torch.max(absolute).item())
    expected_abs_max = float(torch.max(torch.abs(expected)).item())
    return {
        "max_abs": max_abs,
        "rms": rms,
        "relative_rms": float(rms / max(expected_rms, 1e-30)),
        "relative_max": float(
            max_abs / max(expected_abs_max, 1e-30)
        ),
        "expected_rms": expected_rms,
        "expected_abs_max": expected_abs_max,
        "max_index": [int(value) for value in unravel],
        "allclose_rtol_1e_6_atol_1e_6": bool(
            torch.allclose(
                actual,
                expected,
                rtol=1e-6,
                atol=1e-6,
            )
        ),
        "bitwise_equal": bool(torch.equal(actual, expected)),
    }


def exact_step_equivalence(
    *,
    device: torch.device,
    shape: Tuple[int, ...] = (4, DEFAULT_TF, STATE_DIM),
    timesteps: Sequence[int] = TIMESTEPS,
    seed: int = 970000,
) -> List[Dict[str, Any]]:
    scheduler = make_ddpm_scheduler()
    scheduler.set_timesteps(100, device=device)
    _assert_scheduler_contract(scheduler)

    model_output = torch.randn(
        shape,
        generator=torch.Generator(device=device).manual_seed(seed),
        device=device,
        dtype=torch.float32,
    )
    sample = torch.randn(
        shape,
        generator=torch.Generator(device=device).manual_seed(seed + 1),
        device=device,
        dtype=torch.float32,
    )

    rows: List[Dict[str, Any]] = []
    for raw_timestep in timesteps:
        timestep = int(raw_timestep)
        random_seed = seed + 1000 + timestep
        official_generator = torch.Generator(
            device=device
        ).manual_seed(random_seed)
        manual_generator = torch.Generator(
            device=device
        ).manual_seed(random_seed)
        official = scheduler.step(
            model_output,
            timestep,
            sample,
            generator=official_generator,
        )
        variance_noise = (
            torch.randn(
                model_output.shape,
                generator=manual_generator,
                device=device,
                dtype=model_output.dtype,
            )
            if timestep > 0
            else None
        )
        exact = exact_diffusers_0111_components(
            scheduler=scheduler,
            model_output=model_output,
            timestep=timestep,
            sample=sample,
            variance_noise=variance_noise,
        )
        legacy = legacy_algebraic_components(
            scheduler=scheduler,
            model_output=model_output,
            timestep=timestep,
            sample=sample,
            variance_noise=variance_noise,
        )

        official_variance_term = (
            official.prev_sample - exact["posterior_mean"]
        )
        rows.append(
            {
                "device": str(device),
                "dtype": str(sample.dtype),
                "timestep": timestep,
                "pred_x0": tensor_difference(
                    exact["pred_original_sample"],
                    official.pred_original_sample,
                ),
                "posterior_mean_plus_variance": tensor_difference(
                    exact["prev_sample"],
                    official.prev_sample,
                ),
                "variance_term": tensor_difference(
                    exact["variance_term"],
                    official_variance_term,
                ),
                "legacy_prev_vs_official": tensor_difference(
                    legacy["prev_sample"],
                    official.prev_sample,
                ),
                "legacy_pred_x0_vs_official": tensor_difference(
                    legacy["pred_original_sample"],
                    official.pred_original_sample,
                ),
                "exact_beta": float(scheduler.betas[timestep].item()),
                "exact_alpha": float(scheduler.alphas[timestep].item()),
                "exact_variance": float(
                    exact["variance"].detach().cpu().item()
                ),
            }
        )
    return rows


@torch.no_grad()
def exact_reverse_chain_trace(
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
        (condition_z.shape[0], DEFAULT_TF, STATE_DIM),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    current = torch.where(
        active[None, :, :],
        current,
        torch.zeros_like(current),
    )
    scheduler.set_timesteps(100, device=device)
    trace: List[Dict[str, float]] = []

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
        exact = exact_diffusers_0111_components(
            scheduler=scheduler,
            model_output=epsilon,
            timestep=timestep,
            sample=current,
            variance_noise=variance_noise,
        )
        previous = torch.where(
            active[None, :, :],
            exact["prev_sample"],
            torch.zeros_like(current),
        )
        if timestep in TRACE_TIMESTEPS:
            expanded = active[None, :, :].expand_as(current)
            pred_raw = future_std.inverse(
                exact["pred_original_sample"]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            trace.append(
                {
                    "timestep": timestep,
                    "current_rms": float(
                        torch.sqrt(
                            torch.mean(current[expanded].square())
                        ).item()
                    ),
                    "current_abs_max": float(
                        torch.max(torch.abs(current[expanded])).item()
                    ),
                    "epsilon_rms": float(
                        torch.sqrt(
                            torch.mean(epsilon[expanded].square())
                        ).item()
                    ),
                    "epsilon_abs_max": float(
                        torch.max(torch.abs(epsilon[expanded])).item()
                    ),
                    "pred_x0_rms": float(
                        torch.sqrt(
                            torch.mean(
                                exact["pred_original_sample"][
                                    expanded
                                ].square()
                            )
                        ).item()
                    ),
                    "pred_x0_abs_max": float(
                        torch.max(
                            torch.abs(
                                exact["pred_original_sample"][expanded]
                            )
                        ).item()
                    ),
                    "pred_x0_raw_xy_abs_max": float(
                        np.max(
                            np.abs(
                                pred_raw[..., :BEAD_XY_DIM]
                            )
                        )
                    ),
                }
            )
        current = previous
    return current, trace


@torch.no_grad()
def exact_partial_denoise_audit(
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
        current = scheduler.add_noise(x0_z, forward_noise, batch_t)

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
            exact = exact_diffusers_0111_components(
                scheduler=scheduler,
                model_output=epsilon,
                timestep=timestep,
                sample=current,
                variance_noise=(
                    torch.zeros_like(current)
                    if timestep > 0
                    else None
                ),
            )
            current = torch.where(
                active[None, :, :],
                exact["prev_sample"],
                torch.zeros_like(current),
            )

        expanded = active[None, :, :].expand_as(current)
        z_error = (current - x0_z)[expanded]
        raw_prediction = future_std.inverse(
            current.detach().cpu().numpy().astype(np.float32)
        )
        chamfer = final_chamfer_rows(raw_prediction, target_raw)
        rows.append(
            {
                "start_timestep": int(start_timestep),
                "z_mse": float(torch.mean(z_error.square()).item()),
                "z_mae": float(torch.mean(torch.abs(z_error)).item()),
                "z_abs_max": float(
                    torch.max(torch.abs(current[expanded])).item()
                ),
                "final_chamfer_mean": float(np.mean(chamfer)),
                "final_chamfer_median": float(np.median(chamfer)),
                "raw_xy_abs_max": float(
                    np.max(
                        np.abs(
                            raw_prediction[..., :BEAD_XY_DIM]
                        )
                    )
                ),
            }
        )
    return rows


def final_result_stats(
    *,
    final_z: torch.Tensor,
    future_std: Standardizer,
    target_raw: np.ndarray,
    active_mask: np.ndarray,
) -> Dict[str, Any]:
    active = torch.from_numpy(active_mask).to(
        device=final_z.device,
        dtype=torch.bool,
    )
    expanded = active[None, :, :].expand_as(final_z)
    raw = future_std.inverse(
        final_z.detach().cpu().numpy().astype(np.float32)
    )
    chamfer = final_chamfer_rows(raw, target_raw)
    return {
        "z": finite_stats(
            final_z[expanded].detach().cpu().numpy()
        ),
        "raw_xy": finite_stats(raw[..., :BEAD_XY_DIM]),
        "final_chamfer_mean": float(np.mean(chamfer)),
        "final_chamfer_median": float(np.median(chamfer)),
        "final_chamfer_max": float(np.max(chamfer)),
    }


def summarize_equivalence(
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    exact_prev = [
        row["posterior_mean_plus_variance"] for row in rows
    ]
    pred_x0 = [row["pred_x0"] for row in rows]
    legacy = [row["legacy_prev_vs_official"] for row in rows]
    return {
        "rows": len(rows),
        "all_exact_prev_close": bool(
            all(item["allclose_rtol_1e_6_atol_1e_6"] for item in exact_prev)
        ),
        "all_pred_x0_close": bool(
            all(item["allclose_rtol_1e_6_atol_1e_6"] for item in pred_x0)
        ),
        "exact_prev_max_abs": float(
            max(item["max_abs"] for item in exact_prev)
        ),
        "exact_prev_max_relative_rms": float(
            max(item["relative_rms"] for item in exact_prev)
        ),
        "pred_x0_max_abs": float(
            max(item["max_abs"] for item in pred_x0)
        ),
        "legacy_prev_max_abs": float(
            max(item["max_abs"] for item in legacy)
        ),
        "legacy_prev_max_relative_rms": float(
            max(item["relative_rms"] for item in legacy)
        ),
    }
