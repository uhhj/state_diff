from __future__ import annotations

import numpy as np

from ccda_phase3.phase314b_r2_metrics import (
    evaluate_pool,
    fit_validity_contract,
    run_stability_gate,
    sample_validity,
)
from ccda_phase3.schema_v2 import STATE_DIM


def make_future(offset: float = 0.0) -> np.ndarray:
    state = np.zeros((4, STATE_DIM), dtype=np.float32)
    xy = np.zeros((4, 24, 2), dtype=np.float32)
    xy[..., 0] = np.linspace(0.0, 0.23, 24)
    xy[..., 1] = float(offset)
    state[:, :48] = xy.reshape(4, -1)
    state[:, 83:87] = [0, 0, 0, 1]
    return state


def test_validity_contract_accepts_in_distribution_and_rejects_explosion():
    train = np.stack(
        [make_future(value) for value in np.linspace(-0.01, 0.01, 20)],
        axis=0,
    )
    contract = fit_validity_contract(train)
    good = train[:2][None, ...]
    bad = good.copy()
    bad[..., :48] += 100.0
    assert sample_validity(good, contract)["sample_validity_rate"] == 1.0
    assert sample_validity(bad, contract)["sample_validity_rate"] == 0.0


def test_stability_gate_passes_reasonable_pool():
    train = np.stack(
        [make_future(value) for value in np.linspace(-0.01, 0.01, 20)],
        axis=0,
    )
    target = train[:4]
    pool_raw = np.stack([target, target], axis=0)
    pool_z = np.zeros_like(pool_raw)
    active = np.ones((4, STATE_DIM), dtype=np.bool_)
    metrics = evaluate_pool(
        sample_pool_z=pool_z,
        sample_pool_raw=pool_raw,
        target_raw=target,
        active_mask=active,
        validity_contract=fit_validity_contract(train),
        k_values=(1, 4, 8) if False else (1, 2),
    )
    # Adapt the synthetic K=2 result to the formal K=8 gate interface.
    metrics["k_metrics"]["4"] = metrics["k_metrics"]["2"]
    metrics["k_metrics"]["8"] = metrics["k_metrics"]["2"]
    gate = run_stability_gate(
        pool_metrics=metrics,
        partial_t10_chamfer=0.01,
        partial_t99_chamfer=0.02,
    )
    assert gate["stable"]
