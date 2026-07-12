from __future__ import annotations

import importlib.util

import pytest
import torch

from ccda_phase3.phase314b_models import (
    MLPFutureDenoiser,
    TemporalUNetFutureDenoiser,
)


def test_mlp_shape_and_backward():
    model = MLPFutureDenoiser(condition_dim=303)
    sample = torch.randn(2, 4, 87, requires_grad=True)
    timestep = torch.tensor([1, 20], dtype=torch.long)
    condition = torch.randn(2, 303)
    output = model(sample, timestep, condition)
    assert output.shape == sample.shape
    output.square().mean().backward()
    assert sample.grad is not None
    assert torch.all(torch.isfinite(sample.grad))


@pytest.mark.skipif(
    importlib.util.find_spec("state_diff") is None,
    reason="repository state_diff package is unavailable",
)
def test_temporal_unet_shape_and_backward():
    model = TemporalUNetFutureDenoiser(condition_dim=261)
    sample = torch.randn(2, 4, 87, requires_grad=True)
    timestep = torch.tensor([1, 20], dtype=torch.long)
    condition = torch.randn(2, 261)
    output = model(sample, timestep, condition)
    assert output.shape == sample.shape
    output.square().mean().backward()
    assert sample.grad is not None
    assert torch.all(torch.isfinite(sample.grad))
