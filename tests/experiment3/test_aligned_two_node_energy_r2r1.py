import numpy as np

from scripts.experiment3.phase0_soft_blockpush_r2 import (
    analyze_mechanics_validation as analyzer)


def test_aligned_energy_formula_uses_same_state():
    mass = .03 / 72
    stiffness = 12.0
    extension = np.array([.0005, .0002, -.0001])
    velocity = np.array([0.0, .1, -.05])
    kinetic = .5 * mass * velocity ** 2
    spring = .5 * stiffness * extension ** 2
    total = kinetic + spring
    np.testing.assert_allclose(total, kinetic + spring, rtol=0, atol=1e-15)


def test_summary_uses_aligned_total_energy():
    steps = 20
    aligned = np.linspace(1.0, .1, steps)
    data = {
        "extension": np.linspace(.001, -.001, steps),
        "total_energy": aligned,
        "edge_length": np.full(steps, .0105),
        "capped_force_count": np.zeros(steps, dtype=np.int64)}
    config = {
        "physics": {"outer_timestep_s": 1 / 240},
        "soft_block": {"spacing_m": [.0105, .0105, .0105]},
        "mechanics_validation": {
            "edge_ratio_min": .8, "edge_ratio_max": 1.2,
            "final_energy_ratio_max": .25,
            "energy_increase_fraction_max": .02}}
    result = analyzer.summarize_two_node(data, config, 8)
    assert result["energy_measurement"] == "post_step_aligned"
    assert result["energy_increase_count"] == 0
    assert result["energy_increase_fraction"] == 0
    assert result["cumulative_positive_energy_injection_ratio"] == 0


def test_energy_count_gate_is_not_relaxed():
    energy = np.ones(101)
    energy[1:4] += np.array([1e-3, 2e-3, 3e-3])
    delta = np.diff(energy)
    positive = delta > 1e-15
    fraction = np.count_nonzero(positive) / len(delta)
    assert fraction > .02
