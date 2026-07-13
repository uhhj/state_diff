from __future__ import annotations

import numpy as np
import pytest
import torch

from ccda_phase3.phase314b_r232_controls import (
    ModelSpec,
    ResidualMLPFutureDenoiser,
    TimeAffineFutureDenoiser,
    classify_controls,
    make_tuple_bank,
    oracle_v_from_x0,
    timestep_embedding_audit,
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
        alpha_bar = self.alphas_cumprod.to(clean.device)[timesteps]
        alpha = torch.sqrt(alpha_bar).reshape(-1, 1, 1)
        sigma = torch.sqrt(1.0 - alpha_bar).reshape(-1, 1, 1)
        return alpha * clean + sigma * noise


def one_row_tensors():
    condition = torch.zeros(1, 261)
    clean = torch.randn(1, 4, 87)
    raw = clean.clone()
    active = torch.ones(4, 87, dtype=torch.bool)
    return condition, clean, raw, active


def test_model_shapes() -> None:
    noisy = torch.randn(3, 4, 87)
    timestep = torch.tensor([10, 50, 90])
    condition = torch.randn(3, 261)
    for model in (
        ResidualMLPFutureDenoiser(261),
        TimeAffineFutureDenoiser(261),
    ):
        output = model(noisy, timestep, condition)
        assert output.shape == noisy.shape
        assert torch.isfinite(output).all()


def test_model_spec_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        ModelSpec("unknown").validate()


def test_tuple_bank_fixed_timestep_counts_and_determinism() -> None:
    scheduler = FakeScheduler()
    condition, clean, raw, active = one_row_tensors()
    kwargs = dict(
        scheduler=scheduler,
        condition_z=condition,
        clean_z=clean,
        clean_raw=raw,
        active_mask=active,
        timesteps=(50,),
        noise_seeds=(1, 2, 3),
        mode="fixed_timestep",
    )
    first = make_tuple_bank(**kwargs)
    second = make_tuple_bank(**kwargs)
    assert first.row_count == 3
    assert first.timesteps.tolist() == [50, 50, 50]
    assert torch.equal(first.noise, second.noise)
    assert torch.equal(first.noisy, second.noisy)


def test_tuple_bank_cartesian_counts() -> None:
    scheduler = FakeScheduler()
    condition, clean, raw, active = one_row_tensors()
    bank = make_tuple_bank(
        scheduler=scheduler,
        condition_z=condition,
        clean_z=clean,
        clean_raw=raw,
        active_mask=active,
        timesteps=(10, 50, 90),
        noise_seeds=(7, 8),
        mode="cartesian",
    )
    assert bank.row_count == 6
    assert sorted(set(bank.timesteps.tolist())) == [10, 50, 90]
    assert sorted(set(bank.noise_ids.tolist())) == [7, 8]


def test_tuple_bank_rejects_invalid_fixed_timestep() -> None:
    scheduler = FakeScheduler()
    condition, clean, raw, active = one_row_tensors()
    with pytest.raises(ValueError, match="one timestep"):
        make_tuple_bank(
            scheduler=scheduler,
            condition_z=condition,
            clean_z=clean,
            clean_raw=raw,
            active_mask=active,
            timesteps=(10, 50),
            noise_seeds=(1,),
            mode="fixed_timestep",
        )


def test_oracle_v_recovers_manual_velocity() -> None:
    scheduler = FakeScheduler()
    condition, clean, raw, active = one_row_tensors()
    bank = make_tuple_bank(
        scheduler=scheduler,
        condition_z=condition,
        clean_z=clean,
        clean_raw=raw,
        active_mask=active,
        timesteps=(50,),
        noise_seeds=(11,),
        mode="fixed_timestep",
    )
    alpha_bar = scheduler.alphas_cumprod[50]
    expected = torch.sqrt(alpha_bar) * bank.noise - torch.sqrt(
        1.0 - alpha_bar
    ) * bank.clean_z
    actual = oracle_v_from_x0(
        scheduler=scheduler,
        noisy=bank.noisy,
        clean_z=bank.clean_z,
        timesteps=bank.timesteps,
    )
    assert torch.allclose(actual, expected, atol=1.0e-5, rtol=1.0e-5)


def test_timestep_embedding_has_no_exact_collision() -> None:
    audit = timestep_embedding_audit(time_dim=128, device=torch.device("cpu"))
    assert audit["pass"] is True
    assert audit["exact_duplicate"] is False
    assert audit["minimum_pairwise_l2"] > 0


def synthetic_report(default: bool = True):
    def bank_run(**values):
        return {
            "evaluations": {
                key: {"gate_pass": value}
                for key, value in values.items()
            }
        }

    return {
        "oracle_parity": {"pass": True},
        "timestep_embedding_audit": {"pass": True},
        "runs": {
            "fixed_t50_baseline": bank_run(
                seen_noise_bank=default,
                heldout_noise_bank=default,
            ),
            "fixed_t50_residual": bank_run(
                seen_noise_bank=default,
                heldout_noise_bank=default,
            ),
            "fixed_t50_time_affine": bank_run(
                seen_noise_bank=default,
                heldout_noise_bank=default,
            ),
            "fixed_noise_all_t_baseline": bank_run(
                seen_timestep_bank=default,
            ),
            "fixed_noise_all_t_residual": bank_run(
                seen_timestep_bank=default,
            ),
            "cartesian_baseline": bank_run(
                seen_cartesian=default,
                heldout_cartesian=default,
            ),
            "cartesian_residual": bank_run(
                seen_cartesian=default,
                heldout_cartesian=default,
            ),
            "stream_baseline": {"heldout": {"gate_pass": default}},
            "stream_residual": {"heldout": {"gate_pass": default}},
        },
    }


def test_classifier_detects_skip_path_deficiency() -> None:
    report = synthetic_report(True)
    report["runs"]["fixed_t50_baseline"]["evaluations"][
        "heldout_noise_bank"
    ]["gate_pass"] = False
    result = classify_controls(report)
    assert result["root_cause"] == (
        "phase314b_r232_noisy_input_skip_path_deficiency_supported"
    )


def test_classifier_detects_old_batch1_coverage_failure() -> None:
    result = classify_controls(synthetic_report(True))
    assert result["root_cause"] == (
        "phase314b_r232_r231_batch1_coverage_failure_supported"
    )


def test_classifier_prioritizes_oracle_failure() -> None:
    report = synthetic_report(True)
    report["oracle_parity"]["pass"] = False
    result = classify_controls(report)
    assert result["root_cause"] == (
        "phase314b_r232_v_target_or_x0_oracle_parity_failed"
    )
