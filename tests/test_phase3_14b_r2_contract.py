from __future__ import annotations

import math

import numpy as np

from ccda_phase3.phase314b_r2_contract import (
    REPAIR_CONFIGS,
    cosine_betas,
    default_linear_100_stats,
    schedule_stats_from_betas,
)


def test_repair_matrix_has_three_distinct_repairs():
    assert set(REPAIR_CONFIGS) == {
        "epsilon_cosine_cap_0p5",
        "sample_cosine",
        "v_prediction_cosine",
    }
    assert REPAIR_CONFIGS[
        "epsilon_cosine_cap_0p5"
    ].prediction_type == "epsilon"
    assert REPAIR_CONFIGS["sample_cosine"].prediction_type == "sample"
    assert (
        REPAIR_CONFIGS["v_prediction_cosine"].prediction_type
        == "v_prediction"
    )


def test_capped_cosine_reduces_terminal_amplification():
    original = cosine_betas(100, max_beta=0.999)
    capped = cosine_betas(100, max_beta=0.5)
    original_stats = schedule_stats_from_betas(original)
    capped_stats = schedule_stats_from_betas(capped)
    assert original_stats["epsilon_x0_error_amplification"] > 2000
    assert 50 < capped_stats["epsilon_x0_error_amplification"] < 70
    assert capped_stats["terminal_signal_coefficient"] < 0.02


def test_default_linear_100_is_excluded_for_terminal_prior_mismatch():
    stats = default_linear_100_stats()
    assert stats["excluded_from_matrix"]
    assert stats["terminal_signal_coefficient"] > 0.5
    assert stats["terminal_alpha_bar"] > 0.3
