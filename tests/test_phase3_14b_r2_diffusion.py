from __future__ import annotations

from types import SimpleNamespace
import importlib.util

import pytest
import torch

from ccda_phase3.phase314b_r2_contract import RepairConfig
from ccda_phase3.phase314b_r2_diffusion import (
    active_mse,
    predict_original_sample,
    training_target,
)


class FakeScheduler:
    def __init__(self):
        self.alphas_cumprod = torch.tensor(
            [0.9, 0.25],
            dtype=torch.float32,
        )

    def get_velocity(self, sample, noise, timesteps):
        alpha = self.alphas_cumprod[timesteps]
        while alpha.ndim < sample.ndim:
            alpha = alpha.unsqueeze(-1)
        return torch.sqrt(alpha) * noise - torch.sqrt(1-alpha) * sample


def config(prediction_type: str) -> RepairConfig:
    return RepairConfig(
        name=prediction_type,
        prediction_type=prediction_type,
        schedule_kind="cosine_original",
    )


def test_training_targets_match_objective_contract():
    scheduler = FakeScheduler()
    clean = torch.randn(2, 4, 3)
    noise = torch.randn_like(clean)
    timesteps = torch.tensor([0, 1])
    epsilon = training_target(
        scheduler=scheduler,
        config=config("epsilon"),
        clean_sample=clean,
        noise=noise,
        timesteps=timesteps,
    )
    sample = training_target(
        scheduler=scheduler,
        config=config("sample"),
        clean_sample=clean,
        noise=noise,
        timesteps=timesteps,
    )
    velocity = training_target(
        scheduler=scheduler,
        config=config("v_prediction"),
        clean_sample=clean,
        noise=noise,
        timesteps=timesteps,
    )
    assert torch.equal(epsilon, noise)
    assert torch.equal(sample, clean)
    assert velocity.shape == clean.shape


def test_predict_original_sample_is_exact_for_all_objectives():
    scheduler = FakeScheduler()
    clean = torch.randn(2, 4, 3)
    noise = torch.randn_like(clean)
    timesteps = torch.tensor([0, 1])
    alpha = scheduler.alphas_cumprod[timesteps]
    while alpha.ndim < clean.ndim:
        alpha = alpha.unsqueeze(-1)
    noisy = torch.sqrt(alpha) * clean + torch.sqrt(1-alpha) * noise

    eps_x0 = predict_original_sample(
        scheduler=scheduler,
        config=config("epsilon"),
        sample=noisy,
        model_output=noise,
        timesteps=timesteps,
    )
    sample_x0 = predict_original_sample(
        scheduler=scheduler,
        config=config("sample"),
        sample=noisy,
        model_output=clean,
        timesteps=timesteps,
    )
    velocity = scheduler.get_velocity(clean, noise, timesteps)
    velocity_x0 = predict_original_sample(
        scheduler=scheduler,
        config=config("v_prediction"),
        sample=noisy,
        model_output=velocity,
        timesteps=timesteps,
    )
    assert torch.max(torch.abs(eps_x0-clean)).item() < 1e-5
    assert torch.equal(sample_x0, clean)
    assert torch.max(torch.abs(velocity_x0-clean)).item() < 1e-5


def test_active_mse_ignores_inactive_dimension():
    prediction = torch.zeros(2, 4, 3)
    target = torch.zeros_like(prediction)
    prediction[..., -1] = 1000
    active = torch.ones(4, 3, dtype=torch.bool)
    active[..., -1] = False
    assert active_mse(prediction, target, active).item() == 0.0


@pytest.mark.skipif(
    importlib.util.find_spec("diffusers") is None,
    reason="diffusers is unavailable",
)
def test_formal_diffusers_0111_repair_schedulers():
    import diffusers
    from ccda_phase3.phase314b_r2_contract import REPAIR_CONFIGS
    from ccda_phase3.phase314b_r2_diffusion import (
        make_repair_scheduler,
        scheduler_contract,
    )

    if diffusers.__version__ != "0.11.1":
        pytest.skip("formal integration requires Diffusers 0.11.1")
    for config in REPAIR_CONFIGS.values():
        scheduler = make_repair_scheduler(config)
        contract = scheduler_contract(scheduler, config)
        assert contract["prediction_type"] == config.prediction_type
        assert contract["clip_sample"] is False
        assert contract["terminal_alpha_bar"] > 0.0
