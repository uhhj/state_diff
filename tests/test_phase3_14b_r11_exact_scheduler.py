from __future__ import annotations

import importlib.util
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ccda_phase3.phase314b_r11_exact_scheduler import (
    exact_diffusers_0111_components,
    exact_step_equivalence,
    legacy_algebraic_components,
    tensor_difference,
)


class FakeOutput:
    def __init__(self, prev_sample, pred_original_sample):
        self.prev_sample = prev_sample
        self.pred_original_sample = pred_original_sample


class FakeScheduler:
    def __init__(self):
        # Use the formal 100-step cosine schedule. At intermediate timesteps,
        # recomputing beta from cumulative alpha differs from the stored beta
        # in float32 even though the real-number formulas are equivalent.
        def alpha_bar(value):
            return math.cos(
                (value + 0.008) / 1.008 * math.pi / 2
            ) ** 2

        self.betas = torch.tensor(
            [
                min(
                    1
                    - alpha_bar((index + 1) / 100)
                    / alpha_bar(index / 100),
                    0.999,
                )
                for index in range(100)
            ],
            dtype=torch.float32,
        )
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(
            self.alphas,
            dim=0,
        )
        self.one = torch.tensor(1.0)
        self.config = SimpleNamespace(
            prediction_type="epsilon",
            variance_type="fixed_small",
            clip_sample=False,
            thresholding=False,
        )

    def _get_variance(self, t, predicted_variance=None):
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = (
            self.alphas_cumprod[t - 1]
            if t > 0
            else self.one
        )
        variance = (
            (1 - alpha_prod_t_prev)
            / (1 - alpha_prod_t)
            * self.betas[t]
        )
        return torch.clamp(variance, min=1e-20)

    def step(self, model_output, timestep, sample, generator=None):
        t = int(timestep)
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = (
            self.alphas_cumprod[t - 1]
            if t > 0
            else self.one
        )
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        pred_original_sample = (
            sample
            - beta_prod_t ** 0.5 * model_output
        ) / alpha_prod_t ** 0.5
        pred_original_sample_coeff = (
            alpha_prod_t_prev ** 0.5
            * self.betas[t]
            / beta_prod_t
        )
        current_sample_coeff = (
            self.alphas[t] ** 0.5
            * beta_prod_t_prev
            / beta_prod_t
        )
        previous = (
            pred_original_sample_coeff * pred_original_sample
            + current_sample_coeff * sample
        )
        if t > 0:
            noise = torch.randn(
                model_output.shape,
                generator=generator,
                device=model_output.device,
                dtype=model_output.dtype,
            )
            previous = (
                previous
                + self._get_variance(t) ** 0.5 * noise
            )
        return FakeOutput(previous, pred_original_sample)


def test_exact_components_match_official_operation_order():
    scheduler = FakeScheduler()
    sample = torch.randn(2, 4, 3)
    output = torch.randn_like(sample)
    seed = 123
    official = scheduler.step(
        output,
        75,
        sample,
        generator=torch.Generator().manual_seed(seed),
    )
    variance_noise = torch.randn(
        output.shape,
        generator=torch.Generator().manual_seed(seed),
        dtype=output.dtype,
    )
    exact = exact_diffusers_0111_components(
        scheduler=scheduler,
        model_output=output,
        timestep=75,
        sample=sample,
        variance_noise=variance_noise,
    )
    assert torch.equal(
        exact["pred_original_sample"],
        official.pred_original_sample,
    )
    assert torch.equal(exact["prev_sample"], official.prev_sample)


def test_legacy_algebraic_path_can_differ_in_float32():
    scheduler = FakeScheduler()
    sample = torch.randn(16, 4, 87)
    output = torch.randn_like(sample)
    variance_noise = torch.randn_like(sample)
    exact = exact_diffusers_0111_components(
        scheduler=scheduler,
        model_output=output,
        timestep=75,
        sample=sample,
        variance_noise=variance_noise,
    )
    legacy = legacy_algebraic_components(
        scheduler=scheduler,
        model_output=output,
        timestep=4,
        sample=sample,
        variance_noise=variance_noise,
    )
    difference = torch.max(
        torch.abs(exact["prev_sample"] - legacy["prev_sample"])
    ).item()
    assert difference > 0.0


def test_tensor_difference_reports_relative_and_location():
    expected = torch.tensor([[1.0, 2.0]])
    actual = torch.tensor([[1.0, 2.001]])
    result = tensor_difference(actual, expected)
    assert result["max_abs"] > 0.0
    assert result["relative_rms"] > 0.0
    assert result["max_index"] == [0, 1]
    assert not result["bitwise_equal"]


@pytest.mark.skipif(
    importlib.util.find_spec("diffusers") is None,
    reason="diffusers unavailable",
)
def test_installed_diffusers_exact_equivalence():
    import diffusers

    if diffusers.__version__ != "0.11.1":
        pytest.skip("formal integration requires Diffusers 0.11.1")
    rows = exact_step_equivalence(
        device=torch.device("cpu"),
        shape=(2, 4, 87),
        timesteps=(0, 50, 99),
        seed=555,
    )
    assert all(
        row["posterior_mean_plus_variance"][
            "allclose_rtol_1e_6_atol_1e_6"
        ]
        for row in rows
    )
    assert all(
        row["pred_x0"]["allclose_rtol_1e_6_atol_1e_6"]
        for row in rows
    )
