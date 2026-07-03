"""Phase3 model definitions.

The primary implementation is a PyTorch conditional DDPM when torch is
available. The training scripts also provide a NumPy ridge surrogate fallback
for environments where torch is not installed; checkpoints record the backend.
"""

import math
from typing import Optional

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover - exercised on minimal envs
    torch = None
    nn = None


if torch is not None:
    class SinusoidalTimeEmbedding(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.dim = dim

        def forward(self, t):
            half = self.dim // 2
            freqs = torch.exp(torch.arange(half, device=t.device).float() * -(math.log(10000.0) / max(half - 1, 1)))
            args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
            emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
            if emb.shape[-1] < self.dim:
                emb = torch.nn.functional.pad(emb, (0, self.dim - emb.shape[-1]))
            return emb


    class ConditionalStateDDPM(nn.Module):
        """MLP conditional DDPM over future world-state trajectories."""

        def __init__(self, x_dim: int, y_dim: int, hidden_dim: int = 256, time_dim: int = 64, diffusion_steps: int = 100):
            super().__init__()
            self.x_dim = x_dim
            self.y_dim = y_dim
            self.diffusion_steps = diffusion_steps
            self.time = SinusoidalTimeEmbedding(time_dim)
            self.net = nn.Sequential(
                nn.Linear(x_dim + y_dim + time_dim, hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, y_dim),
            )
            betas = torch.linspace(1e-4, 0.02, diffusion_steps)
            alphas = 1.0 - betas
            alpha_bar = torch.cumprod(alphas, dim=0)
            self.register_buffer("betas", betas)
            self.register_buffer("alphas", alphas)
            self.register_buffer("alpha_bar", alpha_bar)
            self.register_buffer("x_mean", torch.zeros(x_dim))
            self.register_buffer("x_std", torch.ones(x_dim))
            self.register_buffer("y_mean", torch.zeros(y_dim))
            self.register_buffer("y_std", torch.ones(y_dim))

        def forward(self, x_cond, y_noisy, t):
            xz = (x_cond - self.x_mean) / self.x_std.clamp_min(1e-6)
            yz = (y_noisy - self.y_mean) / self.y_std.clamp_min(1e-6)
            return self.net(torch.cat([xz, yz, self.time(t)], dim=-1))

        def sample(self, x_cond, n_samples: int = 1):
            self.eval()
            device = next(self.parameters()).device
            x_cond = x_cond.to(device)
            b = x_cond.shape[0]
            x_rep = x_cond.repeat_interleave(n_samples, dim=0)
            y = torch.randn((b * n_samples, self.y_dim), device=device)
            for step in reversed(range(self.diffusion_steps)):
                t = torch.full((y.shape[0],), step, device=device, dtype=torch.long)
                eps = self.forward(x_rep, y, t)
                beta = self.betas[step]
                alpha = self.alphas[step]
                abar = self.alpha_bar[step]
                y = (1 / torch.sqrt(alpha)) * (y - beta / torch.sqrt(1 - abar) * eps)
                if step > 0:
                    y = y + torch.sqrt(beta) * torch.randn_like(y)
            y = y * self.y_std + self.y_mean
            return y.reshape(b, n_samples, self.y_dim)


    class InverseDynamicsMLP(nn.Module):
        """Predict action_t from state history and predicted future state."""

        def __init__(self, x_dim: int, action_dim: int, hidden_dim: int = 256):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(x_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
            )

        def forward(self, x):
            return self.net(x)


    class DirectActionMLP(InverseDynamicsMLP):
        """Optional sanity-check direct action imitation baseline."""
        pass
else:
    class ConditionalStateDDPM:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError("torch is required for ConditionalStateDDPM")

    class InverseDynamicsMLP:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError("torch is required for InverseDynamicsMLP")

    class DirectActionMLP:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError("torch is required for DirectActionMLP")
