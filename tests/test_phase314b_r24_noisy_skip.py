from __future__ import annotations

import numpy as np
import pytest
import torch

from ccda_phase3.phase314b_r24_noisy_skip import (
    AnalyticX0SkipDenoiser,
    PilotModelSpec,
    analytic_formula_oracle,
    classify_pilot,
    make_multirow_bank,
    select_train_only_recommendation,
)


class FakeScheduler:
    def __init__(self) -> None:
        self.alphas_cumprod = torch.linspace(0.99, 0.01, 100)

    def add_noise(
        self,
        clean: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        alpha_bar = self.alphas_cumprod.to(
            device=clean.device,
            dtype=clean.dtype,
        )[timesteps.to(clean.device)]
        alpha = torch.sqrt(alpha_bar).reshape(-1, 1, 1)
        sigma = torch.sqrt(1.0 - alpha_bar).reshape(-1, 1, 1)
        return alpha * clean + sigma * noise

    def get_velocity(
        self,
        sample: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Mirror Diffusers DDPMScheduler.get_velocity()."""
        alpha_bar = self.alphas_cumprod.to(
            device=sample.device,
            dtype=sample.dtype,
        )[timesteps.to(sample.device)]
        sqrt_alpha = torch.sqrt(alpha_bar).flatten()
        sqrt_one_minus_alpha = torch.sqrt(1.0 - alpha_bar).flatten()

        while sqrt_alpha.ndim < sample.ndim:
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
        while sqrt_one_minus_alpha.ndim < sample.ndim:
            sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)

        return sqrt_alpha * noise - sqrt_one_minus_alpha * sample


def source_tensors(row_count: int = 2):
    condition = torch.randn(row_count, 261)
    clean = torch.randn(row_count, 4, 87)
    raw = clean.clone()
    active = torch.ones(4, 87, dtype=torch.bool)
    return condition, clean, raw, active


def test_model_spec_rejects_invalid_kind() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        PilotModelSpec(name="bad", kind="unknown").validate()




def test_analytic_formula_oracle_passes() -> None:
    scheduler = FakeScheduler()
    condition, clean, raw, active = source_tensors(1)
    bank = make_multirow_bank(
        scheduler=scheduler,
        condition_z=condition,
        clean_z=clean,
        clean_raw=raw,
        active_mask=active,
        timesteps=(10, 50, 90),
        noise_seeds=(13, 14),
    )
    result = analytic_formula_oracle(
        scheduler=scheduler,
        bank=bank,
        active_mask=active,
    )
    assert result["pass"] is True
    assert result["v_target_max_abs"] <= 2.0e-5
    assert result["x0_reconstruction_max_abs"] <= 2.0e-5


def test_analytic_model_shape_and_finite() -> None:
    scheduler = FakeScheduler()
    model = AnalyticX0SkipDenoiser(
        condition_dim=261,
        alpha_bar=scheduler.alphas_cumprod,
        use_residual=True,
    )
    noisy = torch.randn(3, 4, 87)
    timestep = torch.tensor([10, 50, 90])
    condition = torch.randn(3, 261)
    output = model(noisy, timestep, condition)
    assert output.shape == noisy.shape
    assert torch.isfinite(output).all()


def test_analytic_model_exact_when_prior_is_clean() -> None:
    scheduler = FakeScheduler()
    model = AnalyticX0SkipDenoiser(
        condition_dim=261,
        alpha_bar=scheduler.alphas_cumprod,
        hidden_dim=8,
        use_residual=False,
    )
    clean_value = torch.randn(4, 87)
    for parameter in model.x0_prior.parameters():
        parameter.data.zero_()
    model.x0_prior[-1].bias.data.copy_(clean_value.reshape(-1))

    condition = torch.zeros(2, 261)
    clean = clean_value[None].repeat(2, 1, 1)
    noise = torch.randn_like(clean)
    timestep = torch.tensor([25, 75])
    noisy = scheduler.add_noise(clean, noise, timestep)
    output = model(noisy, timestep, condition)

    alpha_bar = scheduler.alphas_cumprod[timestep]
    alpha = torch.sqrt(alpha_bar).reshape(-1, 1, 1)
    sigma = torch.sqrt(1.0 - alpha_bar).reshape(-1, 1, 1)
    expected_v = alpha * noise - sigma * clean
    assert torch.allclose(output, expected_v, atol=1.0e-5, rtol=1.0e-5)


def test_analytic_no_residual_jacobian_matches_theory() -> None:
    scheduler = FakeScheduler()
    model = AnalyticX0SkipDenoiser(
        condition_dim=261,
        alpha_bar=scheduler.alphas_cumprod,
        hidden_dim=8,
        use_residual=False,
    )
    noisy = torch.randn(1, 4, 87, requires_grad=True)
    condition = torch.randn(1, 261)
    timestep = torch.tensor([50])
    direction = torch.randn_like(noisy)

    def function(value: torch.Tensor) -> torch.Tensor:
        return model(value, timestep, condition)

    _, jvp = torch.autograd.functional.jvp(function, noisy, direction)
    alpha_bar = scheduler.alphas_cumprod[50]
    gain = torch.sqrt(alpha_bar / (1.0 - alpha_bar))
    assert torch.allclose(jvp, gain * direction, atol=1.0e-5, rtol=1.0e-5)


def test_multirow_bank_count_and_matched_noise() -> None:
    scheduler = FakeScheduler()
    condition, clean, raw, active = source_tensors(2)
    bank = make_multirow_bank(
        scheduler=scheduler,
        condition_z=condition,
        clean_z=clean,
        clean_raw=raw,
        active_mask=active,
        timesteps=(10, 50),
        noise_seeds=(7, 8, 9),
    )
    assert bank.row_count == 12
    # Layout is row, timestep, seed. Same seed at same timestep is matched.
    first_row_noise = bank.noise[0]
    second_row_noise = bank.noise[6]
    assert torch.equal(first_row_noise, second_row_noise)


def fake_run(passed: bool):
    return {
        "evaluations": {
            "fresh_noise_all_t": {"gate_pass": passed},
            "fresh_noise_low_mid": {"gate_pass": passed},
        },
        "jvp": {"t50_gate_pass": passed},
    }


def test_recommendation_prefers_simple_analytic_skip() -> None:
    stages = {
        "one_row": {
            "analytic_x0_skip_v": fake_run(True),
            "analytic_x0_skip_residual_v": fake_run(True),
        },
        "unique_free_16": {
            "analytic_x0_skip_v": fake_run(True),
            "analytic_x0_skip_residual_v": fake_run(True),
        },
        "paired_16": {
            "analytic_x0_skip_v": fake_run(True),
            "analytic_x0_skip_residual_v": fake_run(True),
        },
    }
    assert select_train_only_recommendation(stages) == "analytic_x0_skip_v"


def test_classifier_supports_analytic_skip() -> None:
    stages = {
        "one_row": {"analytic_x0_skip_v": fake_run(True)},
        "unique_free_16": {"analytic_x0_skip_v": fake_run(True)},
        "paired_16": {"analytic_x0_skip_v": fake_run(True)},
    }
    result = classify_pilot({"stages": stages})
    assert result["root_cause"] == (
        "phase314b_r24_train_only_analytic_noisy_skip_supported"
    )
    assert result["train_only_recommendation"] == "analytic_x0_skip_v"


def test_classifier_stops_after_one_row_failure() -> None:
    stages = {
        "one_row": {"analytic_x0_skip_v": fake_run(False)},
        "unique_free_16": {},
        "paired_16": {},
    }
    result = classify_pilot({"stages": stages})
    assert result["root_cause"] == (
        "phase314b_r24_random_stream_skip_pilot_failed"
    )