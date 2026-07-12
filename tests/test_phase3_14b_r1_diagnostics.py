from __future__ import annotations

import math

import numpy as np
import torch

from ccda_phase3.phase314b_r1_diagnostics import (
    active_values,
    finite_stats,
    manual_ddpm_step,
    predict_x0_from_epsilon,
)


class FakeScheduler:
    def __init__(self):
        self.betas = torch.tensor(
            [0.01, 0.02, 0.03],
            dtype=torch.float32,
        )
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(
            self.alphas,
            dim=0,
        )
        self.one = torch.tensor(1.0)


def test_finite_stats_reports_abs_tail():
    result = finite_stats(np.array([-3.0, -1.0, 2.0, 4.0]))
    assert result["count"] == 4
    assert result["min"] == -3.0
    assert result["max"] == 4.0
    assert result["abs_max"] == 4.0
    assert result["rms"] > 0.0


def test_active_values_broadcasts_time_dimension():
    value = np.arange(2 * 4 * 3).reshape(2, 4, 3)
    mask = np.array(
        [
            [True, False, False],
            [False, True, False],
            [False, False, True],
            [True, True, False],
        ]
    )
    selected = active_values(value, mask)
    assert selected.shape == (2 * int(np.sum(mask)),)
    assert 0 in selected


def test_true_epsilon_reconstructs_x0():
    scheduler = FakeScheduler()
    x0 = torch.randn(3, 4, 5)
    noise = torch.randn_like(x0)
    timestep = 2
    alpha_bar = scheduler.alphas_cumprod[timestep]
    xt = (
        torch.sqrt(alpha_bar) * x0
        + torch.sqrt(1.0 - alpha_bar) * noise
    )
    reconstructed = predict_x0_from_epsilon(
        xt,
        noise,
        alpha_bar,
    )
    assert torch.max(torch.abs(reconstructed - x0)).item() < 1e-5


def test_manual_ddpm_t0_returns_predicted_x0():
    scheduler = FakeScheduler()
    x0 = torch.randn(2, 4, 5)
    noise = torch.randn_like(x0)
    alpha_bar = scheduler.alphas_cumprod[0]
    xt = (
        torch.sqrt(alpha_bar) * x0
        + torch.sqrt(1.0 - alpha_bar) * noise
    )
    previous, predicted_x0 = manual_ddpm_step(
        scheduler=scheduler,
        epsilon=noise,
        timestep=0,
        sample=xt,
        variance_noise=None,
    )
    assert torch.max(torch.abs(predicted_x0 - x0)).item() < 1e-5
    assert torch.max(torch.abs(previous - x0)).item() < 1e-5


def test_manual_ddpm_is_finite_at_terminal_step():
    scheduler = FakeScheduler()
    sample = torch.randn(2, 4, 5)
    epsilon = torch.randn_like(sample)
    variance_noise = torch.randn_like(sample)
    previous, predicted_x0 = manual_ddpm_step(
        scheduler=scheduler,
        epsilon=epsilon,
        timestep=2,
        sample=sample,
        variance_noise=variance_noise,
    )
    assert torch.all(torch.isfinite(previous))
    assert torch.all(torch.isfinite(predicted_x0))
