"""Phase3.14b-r2 objective-aware schedulers, targets, and sampling."""
from __future__ import annotations

import copy
import inspect
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import torch

from ccda_phase3.phase314b_diffusion import ExponentialMovingAverage
from ccda_phase3.phase314b_r2_contract import (
    DIFFUSERS_VERSION,
    RepairConfig,
    cosine_betas,
    schedule_stats_from_betas,
)
from ccda_phase3.schema_v2 import DEFAULT_TF, STATE_DIM


def require_diffusers_0111() -> str:
    import diffusers

    version = str(diffusers.__version__)
    if version != DIFFUSERS_VERSION:
        raise RuntimeError(
            f"Diffusers {version} != required {DIFFUSERS_VERSION}"
        )
    return version


def make_repair_scheduler(config: RepairConfig):
    require_diffusers_0111()
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

    config.validate()
    common = {
        "num_train_timesteps": int(config.num_train_timesteps),
        "variance_type": "fixed_small",
        "clip_sample": False,
        "prediction_type": str(config.prediction_type),
    }
    if config.schedule_kind == "cosine_original":
        scheduler = DDPMScheduler(
            beta_schedule="squaredcos_cap_v2",
            **common,
        )
    elif config.schedule_kind == "cosine_capped":
        trained_betas = cosine_betas(
            config.num_train_timesteps,
            max_beta=float(config.max_beta),
        )
        scheduler = DDPMScheduler(
            trained_betas=trained_betas,
            **common,
        )
    else:
        raise ValueError(config.schedule_kind)

    if bool(scheduler.config.clip_sample):
        raise RuntimeError("z-score sample clipping must remain disabled")
    if str(scheduler.config.prediction_type) != config.prediction_type:
        raise RuntimeError("scheduler prediction_type mismatch")
    if str(scheduler.config.variance_type) != "fixed_small":
        raise RuntimeError("scheduler variance_type mismatch")
    return scheduler


def scheduler_contract(
    scheduler,
    config: RepairConfig,
) -> Dict[str, Any]:
    betas = scheduler.betas.detach().cpu().numpy().astype(np.float32)
    return {
        "repair_config": config.name,
        "prediction_type": config.prediction_type,
        "schedule_kind": config.schedule_kind,
        "max_beta": config.max_beta,
        "variance_type": str(scheduler.config.variance_type),
        "clip_sample": bool(scheduler.config.clip_sample),
        "thresholding": bool(
            getattr(scheduler.config, "thresholding", False)
        ),
        **schedule_stats_from_betas(betas),
    }


def training_target(
    *,
    scheduler,
    config: RepairConfig,
    clean_sample: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    if clean_sample.shape != noise.shape:
        raise ValueError("clean sample/noise shape mismatch")
    if config.prediction_type == "epsilon":
        return noise
    if config.prediction_type == "sample":
        return clean_sample
    if config.prediction_type == "v_prediction":
        return scheduler.get_velocity(
            clean_sample,
            noise,
            timesteps,
        )
    raise ValueError(config.prediction_type)


def active_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    active_mask: torch.Tensor,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError("prediction/target shape mismatch")
    if active_mask.shape != prediction.shape[1:]:
        raise ValueError("active mask shape mismatch")
    expanded = active_mask[None, :, :].expand_as(prediction)
    if not torch.any(expanded):
        raise ValueError("active target mask is empty")
    return ((prediction - target) ** 2)[expanded].mean()


def predict_original_sample(
    *,
    scheduler,
    config: RepairConfig,
    sample: torch.Tensor,
    model_output: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    if sample.shape != model_output.shape:
        raise ValueError("sample/model_output shape mismatch")
    alpha_bar = scheduler.alphas_cumprod.to(
        device=sample.device,
        dtype=sample.dtype,
    )[timesteps]
    while alpha_bar.ndim < sample.ndim:
        alpha_bar = alpha_bar.unsqueeze(-1)
    beta_bar = 1.0 - alpha_bar
    if config.prediction_type == "epsilon":
        return (
            sample - torch.sqrt(beta_bar) * model_output
        ) / torch.sqrt(alpha_bar)
    if config.prediction_type == "sample":
        return model_output
    if config.prediction_type == "v_prediction":
        return (
            torch.sqrt(alpha_bar) * sample
            - torch.sqrt(beta_bar) * model_output
        )
    raise ValueError(config.prediction_type)


@torch.no_grad()
def sample_future_z(
    *,
    model: torch.nn.Module,
    scheduler,
    config: RepairConfig,
    condition_z: torch.Tensor,
    active_mask: torch.Tensor,
    num_samples: int,
    seed: int,
    num_inference_steps: int = 100,
    row_batch_size: int = 128,
) -> torch.Tensor:
    if condition_z.ndim != 2:
        raise ValueError("condition_z must be [N,C]")
    if active_mask.shape != (DEFAULT_TF, STATE_DIM):
        raise ValueError("active mask must be [4,87]")
    if int(num_samples) <= 0:
        raise ValueError("num_samples must be positive")

    device = condition_z.device
    active = active_mask.to(device=device, dtype=torch.bool)
    scheduler.set_timesteps(
        int(num_inference_steps),
        device=device,
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    outputs = []

    for _sample_index in range(int(num_samples)):
        row_outputs = []
        for start in range(
            0,
            condition_z.shape[0],
            int(row_batch_size),
        ):
            condition = condition_z[
                start:start + int(row_batch_size)
            ]
            batch = condition.shape[0]
            current = torch.randn(
                (batch, DEFAULT_TF, STATE_DIM),
                generator=generator,
                device=device,
                dtype=torch.float32,
            )
            current = torch.where(
                active[None, :, :],
                current,
                torch.zeros_like(current),
            )
            for scheduler_timestep in scheduler.timesteps:
                timestep = int(scheduler_timestep.item())
                batch_t = torch.full(
                    (batch,),
                    timestep,
                    device=device,
                    dtype=torch.long,
                )
                model_output = model(
                    current,
                    batch_t,
                    condition,
                )
                model_output = torch.where(
                    active[None, :, :],
                    model_output,
                    torch.zeros_like(model_output),
                )
                result = scheduler.step(
                    model_output,
                    timestep,
                    current,
                    generator=generator,
                )
                current = torch.where(
                    active[None, :, :],
                    result.prev_sample,
                    torch.zeros_like(current),
                )
            row_outputs.append(current.cpu())
        outputs.append(torch.cat(row_outputs, dim=0))
    return torch.stack(outputs, dim=0)


@torch.no_grad()
def partial_denoise(
    *,
    model: torch.nn.Module,
    scheduler,
    config: RepairConfig,
    clean_z: torch.Tensor,
    condition_z: torch.Tensor,
    active_mask: torch.Tensor,
    start_timestep: int,
    seed: int,
) -> torch.Tensor:
    device = clean_z.device
    active = active_mask.to(device=device, dtype=torch.bool)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    noise = torch.randn(
        clean_z.shape,
        generator=generator,
        device=device,
        dtype=clean_z.dtype,
    )
    noise = torch.where(
        active[None, :, :],
        noise,
        torch.zeros_like(noise),
    )
    batch_t = torch.full(
        (clean_z.shape[0],),
        int(start_timestep),
        device=device,
        dtype=torch.long,
    )
    current = scheduler.add_noise(clean_z, noise, batch_t)

    # Posterior mean only isolates objective/schedule conditioning.
    for timestep in range(int(start_timestep), -1, -1):
        current_t = torch.full(
            (clean_z.shape[0],),
            timestep,
            device=device,
            dtype=torch.long,
        )
        model_output = model(current, current_t, condition_z)
        model_output = torch.where(
            active[None, :, :],
            model_output,
            torch.zeros_like(model_output),
        )

        alpha_prod_t = scheduler.alphas_cumprod[timestep]
        alpha_prod_prev = (
            scheduler.alphas_cumprod[timestep - 1]
            if timestep > 0
            else scheduler.one
        )
        beta_prod_t = 1.0 - alpha_prod_t
        beta_prod_prev = 1.0 - alpha_prod_prev
        pred_x0 = predict_original_sample(
            scheduler=scheduler,
            config=config,
            sample=current,
            model_output=model_output,
            timesteps=current_t,
        )
        pred_x0_coeff = (
            alpha_prod_prev ** 0.5
            * scheduler.betas[timestep]
            / beta_prod_t
        )
        current_coeff = (
            scheduler.alphas[timestep] ** 0.5
            * beta_prod_prev
            / beta_prod_t
        )
        current = pred_x0_coeff * pred_x0 + current_coeff * current
        current = torch.where(
            active[None, :, :],
            current,
            torch.zeros_like(current),
        )
    return current
