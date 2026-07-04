#!/usr/bin/env python3
from __future__ import annotations

import math

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None
    nn = None


if torch is not None:
    class SinusoidalTimeEmbedding(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.dim = dim

        def forward(self, t):
            half = self.dim // 2
            freqs = torch.exp(
                -math.log(10000.0)
                * torch.arange(half, device=t.device).float()
                / max(half - 1, 1)
            )
            args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
            emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
            if emb.shape[-1] < self.dim:
                emb = torch.nn.functional.pad(emb, (0, self.dim - emb.shape[-1]))
            return emb


    class ConditionalStateDDPM(nn.Module):
        """Conditional DDPM over future state trajectories."""

        def __init__(
            self,
            x_dim: int,
            y_dim: int,
            hidden_dim: int = 512,
            time_dim: int = 128,
            diffusion_steps: int = 100,
            beta_start: float = 1e-4,
            beta_end: float = 0.02,
        ):
            super().__init__()
            self.x_dim = int(x_dim)
            self.y_dim = int(y_dim)
            self.hidden_dim = int(hidden_dim)
            self.time_dim = int(time_dim)
            self.diffusion_steps = int(diffusion_steps)

            self.time = SinusoidalTimeEmbedding(time_dim)
            self.net = nn.Sequential(
                nn.Linear(x_dim + y_dim + time_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, y_dim),
            )

            betas = torch.linspace(beta_start, beta_end, diffusion_steps)
            alphas = 1.0 - betas
            alpha_bar = torch.cumprod(alphas, dim=0)

            self.register_buffer("betas", betas)
            self.register_buffer("alphas", alphas)
            self.register_buffer("alpha_bar", alpha_bar)

            self.register_buffer("x_mean", torch.zeros(x_dim))
            self.register_buffer("x_std", torch.ones(x_dim))
            self.register_buffer("y_mean", torch.zeros(y_dim))
            self.register_buffer("y_std", torch.ones(y_dim))

        def set_standardizers(self, x_mean, x_std, y_mean, y_std) -> None:
            device = self.x_mean.device
            self.x_mean.data = torch.as_tensor(x_mean, dtype=torch.float32, device=device).reshape(-1)
            self.x_std.data = torch.as_tensor(x_std, dtype=torch.float32, device=device).reshape(-1).clamp_min(1e-6)
            self.y_mean.data = torch.as_tensor(y_mean, dtype=torch.float32, device=device).reshape(-1)
            self.y_std.data = torch.as_tensor(y_std, dtype=torch.float32, device=device).reshape(-1).clamp_min(1e-6)

        def standardize_x(self, x_raw):
            return (x_raw - self.x_mean) / self.x_std.clamp_min(1e-6)

        def standardize_y(self, y_raw):
            return (y_raw - self.y_mean) / self.y_std.clamp_min(1e-6)

        def unstandardize_y(self, y_z):
            return y_z * self.y_std + self.y_mean

        def forward(self, x_cond_raw, y_noisy_z, t):
            x_z = self.standardize_x(x_cond_raw)
            h = torch.cat([x_z, y_noisy_z, self.time(t)], dim=-1)
            return self.net(h)

        def q_sample(self, y0_z, t, noise):
            ab = self.alpha_bar[t].view(-1, 1)
            return torch.sqrt(ab) * y0_z + torch.sqrt(1.0 - ab) * noise

        @torch.no_grad()
        def sample(self, x_cond_raw, n_samples: int = 1):
            self.eval()
            device = next(self.parameters()).device
            x_cond_raw = x_cond_raw.to(device)
            batch = x_cond_raw.shape[0]
            x_rep = x_cond_raw.repeat_interleave(n_samples, dim=0)
            y_z = torch.randn((batch * n_samples, self.y_dim), device=device)

            for step in reversed(range(self.diffusion_steps)):
                t = torch.full((y_z.shape[0],), step, device=device, dtype=torch.long)
                eps = self.forward(x_rep, y_z, t)
                beta = self.betas[step]
                alpha = self.alphas[step]
                abar = self.alpha_bar[step]
                mean = (1.0 / torch.sqrt(alpha)) * (y_z - beta / torch.sqrt(1.0 - abar) * eps)
                if step > 0:
                    y_z = mean + torch.sqrt(beta) * torch.randn_like(y_z)
                else:
                    y_z = mean

            y_raw = self.unstandardize_y(y_z)
            return y_raw.reshape(batch, n_samples, self.y_dim)


    class InverseDynamicsMLP(nn.Module):
        def __init__(self, x_dim: int, action_dim: int, hidden_dim: int = 256):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(x_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
            )

        def forward(self, x):
            return self.net(x)


    class DirectActionMLP(InverseDynamicsMLP):
        pass

else:
    class ConditionalStateDDPM:
        def __init__(self, *args, **kwargs):
            raise ImportError("torch is required for ConditionalStateDDPM")

    class InverseDynamicsMLP:
        def __init__(self, *args, **kwargs):
            raise ImportError("torch is required for InverseDynamicsMLP")

    class DirectActionMLP:
        def __init__(self, *args, **kwargs):
            raise ImportError("torch is required for DirectActionMLP")
