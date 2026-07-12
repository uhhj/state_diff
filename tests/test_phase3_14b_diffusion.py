from __future__ import annotations

import importlib.util

import pytest
import torch

from ccda_phase3.phase314b_diffusion import (
    ExponentialMovingAverage,
    active_noise_loss,
    make_ddpm_scheduler,
)


def test_active_noise_loss_ignores_inactive_dimensions():
    prediction = torch.zeros(1, 4, 87)
    noise = torch.zeros_like(prediction)
    prediction[:, :, -1] = 1000.0
    active = torch.ones(4, 87, dtype=torch.bool)
    active[:, -1] = False
    assert active_noise_loss(prediction, noise, active).item() == 0.0


def test_ema_round_trip():
    model = torch.nn.Linear(2, 2)
    ema = ExponentialMovingAverage(model, decay=0.9)
    with torch.no_grad():
        model.weight.add_(1.0)
    ema.update(model)
    restored = torch.nn.Linear(2, 2)
    ema.copy_to(restored)
    for key, value in restored.state_dict().items():
        assert torch.equal(value, ema.shadow[key])


@pytest.mark.skipif(
    importlib.util.find_spec("diffusers") is None,
    reason="diffusers is unavailable",
)
def test_scheduler_never_clips_zscore_samples():
    scheduler = make_ddpm_scheduler()
    assert scheduler.config.clip_sample is False
    assert getattr(scheduler.config, "thresholding", False) is False
    assert scheduler.config.prediction_type == "epsilon"
