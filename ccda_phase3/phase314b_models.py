"""Phase3.14b denoisers with a common [B,T,D] interface."""
from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn

from ccda_phase3.schema_v2 import DEFAULT_TF, STATE_DIM


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dimension: int = 128):
        super().__init__()
        self.dimension = int(dimension)

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        value = timestep.float().reshape(-1)
        half = self.dimension // 2
        frequency = torch.exp(
            -math.log(10000.0)
            * torch.arange(
                half,
                device=value.device,
                dtype=torch.float32,
            )
            / max(half - 1, 1)
        )
        argument = value[:, None] * frequency[None, :]
        embedding = torch.cat(
            [torch.sin(argument), torch.cos(argument)],
            dim=-1,
        )
        if embedding.shape[-1] < self.dimension:
            embedding = torch.nn.functional.pad(
                embedding,
                (0, self.dimension - embedding.shape[-1]),
            )
        return embedding


class MLPFutureDenoiser(nn.Module):
    family = "mlp_ddpm"

    def __init__(
        self,
        condition_dim: int,
        hidden_dim: int = 512,
        time_dim: int = 128,
    ):
        super().__init__()
        self.condition_dim = int(condition_dim)
        self.future_shape = (DEFAULT_TF, STATE_DIM)
        self.time = SinusoidalTimeEmbedding(time_dim)
        flat_future = DEFAULT_TF * STATE_DIM
        self.net = nn.Sequential(
            nn.Linear(
                self.condition_dim + flat_future + int(time_dim),
                int(hidden_dim),
            ),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), flat_future),
        )

    def forward(
        self,
        noisy_future_z: torch.Tensor,
        timestep: torch.Tensor,
        condition_z: torch.Tensor,
    ) -> torch.Tensor:
        if noisy_future_z.ndim != 3:
            raise ValueError("future sample must be [B,T,D]")
        batch = noisy_future_z.shape[0]
        joined = torch.cat(
            [
                condition_z,
                noisy_future_z.reshape(batch, -1),
                self.time(timestep),
            ],
            dim=-1,
        )
        return self.net(joined).reshape(
            batch,
            DEFAULT_TF,
            STATE_DIM,
        )


class TemporalUNetFutureDenoiser(nn.Module):
    family = "temporal_unet_ddpm"

    def __init__(
        self,
        condition_dim: int,
        diffusion_step_embed_dim: int = 128,
        down_dims: Sequence[int] = (128, 256),
        kernel_size: int = 3,
        n_groups: int = 8,
        cond_predict_scale: bool = True,
    ):
        super().__init__()
        from state_diff.model.diffusion.conditional_unet1d import (
            ConditionalUnet1D,
        )

        self.condition_dim = int(condition_dim)
        self.future_shape = (DEFAULT_TF, STATE_DIM)
        self.unet = ConditionalUnet1D(
            input_dim=STATE_DIM,
            output_dim=STATE_DIM,
            local_cond_dim=None,
            global_cond_dim=self.condition_dim,
            diffusion_step_embed_dim=int(
                diffusion_step_embed_dim
            ),
            down_dims=list(int(value) for value in down_dims),
            kernel_size=int(kernel_size),
            n_groups=int(n_groups),
            cond_predict_scale=bool(cond_predict_scale),
        )

    def forward(
        self,
        noisy_future_z: torch.Tensor,
        timestep: torch.Tensor,
        condition_z: torch.Tensor,
    ) -> torch.Tensor:
        output = self.unet(
            noisy_future_z,
            timestep,
            local_cond=None,
            global_cond=condition_z,
        )
        if output.shape != noisy_future_z.shape:
            raise RuntimeError(
                f"Temporal U-Net changed shape: "
                f"{output.shape} != {noisy_future_z.shape}"
            )
        return output


def build_denoiser(
    family: str,
    condition_dim: int,
) -> nn.Module:
    if family == "mlp_ddpm":
        return MLPFutureDenoiser(condition_dim=condition_dim)
    if family == "temporal_unet_ddpm":
        return TemporalUNetFutureDenoiser(
            condition_dim=condition_dim
        )
    raise ValueError(f"unsupported denoiser family: {family}")


def parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))
