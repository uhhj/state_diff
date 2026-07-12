"""Version-tolerant DDPM scheduler, EMA, loss, and sampling."""
from __future__ import annotations

import copy
import inspect
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import torch

from ccda_phase3.schema_v2 import DEFAULT_TF, STATE_DIM


def make_ddpm_scheduler(
    *,
    num_train_timesteps: int = 100,
    beta_schedule: str = "squaredcos_cap_v2",
    prediction_type: str = "epsilon",
    variance_type: str = "fixed_small",
):
    try:
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    except Exception as exc:
        raise RuntimeError(
            "diffusers is required; install it inside coord_bimanual"
        ) from exc

    signature = inspect.signature(DDPMScheduler.__init__)
    kwargs: Dict[str, Any] = {
        "num_train_timesteps": int(num_train_timesteps),
        "beta_schedule": str(beta_schedule),
        "prediction_type": str(prediction_type),
        "variance_type": str(variance_type),
        "clip_sample": False,
    }
    if "thresholding" in signature.parameters:
        kwargs["thresholding"] = False
    if "timestep_spacing" in signature.parameters:
        kwargs["timestep_spacing"] = "leading"
    scheduler = DDPMScheduler(**kwargs)

    if getattr(scheduler.config, "clip_sample", None) is not False:
        raise RuntimeError("DDPM scheduler unexpectedly clips z-score samples")
    if bool(getattr(scheduler.config, "thresholding", False)):
        raise RuntimeError("DDPM scheduler thresholding must be disabled")
    if str(getattr(scheduler.config, "prediction_type", "")) != "epsilon":
        raise RuntimeError("DDPM prediction_type must be epsilon")
    return scheduler


class ExponentialMovingAverage:
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        decay: float = 0.999,
    ):
        self.decay = float(decay)
        if not 0.0 < self.decay < 1.0:
            raise ValueError("EMA decay must lie in (0,1)")
        self.shadow = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        self.updates = 0

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        state = model.state_dict()
        for key, value in state.items():
            current = value.detach().cpu()
            if not torch.is_floating_point(current):
                self.shadow[key] = current.clone()
                continue
            self.shadow[key].mul_(self.decay).add_(
                current,
                alpha=1.0 - self.decay,
            )
        self.updates += 1

    def state_dict(self) -> Dict[str, Any]:
        return {
            "decay": self.decay,
            "updates": self.updates,
            "shadow": {
                key: value.clone()
                for key, value in self.shadow.items()
            },
        }

    @classmethod
    def from_state_dict(
        cls,
        model: torch.nn.Module,
        state: Mapping[str, Any],
    ) -> "ExponentialMovingAverage":
        result = cls(model, decay=float(state["decay"]))
        result.updates = int(state["updates"])
        result.shadow = {
            key: value.detach().cpu().clone()
            for key, value in state["shadow"].items()
        }
        return result

    def copy_to(self, model: torch.nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=True)

    def averaged_model(
        self,
        model: torch.nn.Module,
        *,
        device: torch.device,
    ) -> torch.nn.Module:
        averaged = copy.deepcopy(model).to(device)
        self.copy_to(averaged)
        averaged.eval()
        return averaged


def active_noise_loss(
    prediction: torch.Tensor,
    noise: torch.Tensor,
    active_mask: torch.Tensor,
) -> torch.Tensor:
    if prediction.shape != noise.shape:
        raise ValueError("prediction/noise shape mismatch")
    if active_mask.shape != prediction.shape[1:]:
        raise ValueError(
            f"active mask {active_mask.shape} != "
            f"{prediction.shape[1:]}"
        )
    active = active_mask[None, :, :].expand_as(prediction)
    if not torch.any(active):
        raise ValueError("future active mask is empty")
    squared = (prediction - noise) ** 2
    return squared[active].mean()


def _scheduler_step(
    scheduler,
    model_output: torch.Tensor,
    timestep,
    sample: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    try:
        result = scheduler.step(
            model_output,
            timestep,
            sample,
            generator=generator,
        )
    except TypeError:
        result = scheduler.step(
            model_output,
            timestep,
            sample,
        )
    return result.prev_sample


@torch.no_grad()
def sample_future_z(
    *,
    model: torch.nn.Module,
    scheduler,
    condition_z: torch.Tensor,
    active_mask: torch.Tensor,
    num_samples: int,
    seed: int,
    num_inference_steps: int = 100,
    sample_batch_size: int = 256,
) -> torch.Tensor:
    if condition_z.ndim != 2:
        raise ValueError("condition_z must be [N,C]")
    if active_mask.shape != (DEFAULT_TF, STATE_DIM):
        raise ValueError("future active mask shape mismatch")
    if int(num_samples) <= 0:
        raise ValueError("num_samples must be positive")

    device = condition_z.device
    scheduler.set_timesteps(int(num_inference_steps))
    active = active_mask.to(device=device, dtype=torch.bool)
    all_samples = []
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))

    row_count = condition_z.shape[0]
    for sample_start in range(0, int(num_samples)):
        # Generate one nested sample at a time so K=1,4,8,16,32 are
        # prefixes of the same deterministic pool.
        row_outputs = []
        for start in range(0, row_count, int(sample_batch_size)):
            condition_batch = condition_z[
                start:start + int(sample_batch_size)
            ]
            batch = condition_batch.shape[0]
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
            for timestep in scheduler.timesteps:
                if torch.is_tensor(timestep):
                    scalar = int(timestep.item())
                else:
                    scalar = int(timestep)
                timesteps = torch.full(
                    (batch,),
                    scalar,
                    device=device,
                    dtype=torch.long,
                )
                epsilon = model(
                    current,
                    timesteps,
                    condition_batch,
                )
                epsilon = torch.where(
                    active[None, :, :],
                    epsilon,
                    torch.zeros_like(epsilon),
                )
                current = _scheduler_step(
                    scheduler,
                    epsilon,
                    timestep,
                    current,
                    generator,
                )
                current = torch.where(
                    active[None, :, :],
                    current,
                    torch.zeros_like(current),
                )
            row_outputs.append(current.cpu())
        all_samples.append(torch.cat(row_outputs, dim=0))
    return torch.stack(all_samples, dim=0)


def scheduler_contract(scheduler) -> Dict[str, Any]:
    return {
        "class": scheduler.__class__.__name__,
        "num_train_timesteps": int(
            scheduler.config.num_train_timesteps
        ),
        "beta_schedule": str(scheduler.config.beta_schedule),
        "prediction_type": str(scheduler.config.prediction_type),
        "variance_type": str(scheduler.config.variance_type),
        "clip_sample": bool(scheduler.config.clip_sample),
        "thresholding": bool(
            getattr(scheduler.config, "thresholding", False)
        ),
    }
