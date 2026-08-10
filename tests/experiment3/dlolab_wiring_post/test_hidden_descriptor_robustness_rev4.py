import json
from pathlib import Path

import numpy as np
import pytest

from scripts.experiment3.dlolab_wiring_post.hidden_descriptor_robustness_rev4 import (
    contact_proxy_metrics,
    deadband_sensitivity_rows,
    replay_locator,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
REV4_REPLAY = json.loads(
    (
        REPO_ROOT
        / "configs/experiment3/published_benchmark/"
        "dlolab_wiring_post_pb0s_rev4.json"
    ).read_text(encoding="utf-8")
)["replay"]


def _descriptor(
        clearance_a,
        clearance_b,
        *,
        wrap=0.0,
        global_angle=0.0,
        nearest_a=20,
        nearest_b=20):
    return {
        "descriptor_a": [{
            "surface_clearance_m": clearance_a,
            "contact_like": clearance_a <= 0.003,
            "nearest_vertex": nearest_a,
        }],
        "descriptor_b": [{
            "surface_clearance_m": clearance_b,
            "contact_like": clearance_b <= 0.003,
            "nearest_vertex": nearest_b,
        }],
        "roi_wrap_difference_rad": [wrap],
        "global_angular_difference_rad": [global_angle],
    }


def test_micron_boundary_flip_has_tiny_symmetric_margin():
    metrics = contact_proxy_metrics(
        _descriptor(
            0.0029983,
            0.0030026,
        ),
        0.003,
        [0.0, 0.000005],
        0.75,
        0.75,
    )

    assert metrics["reason"] == "contact_proxy_only"
    assert (
        metrics["max_symmetric_boundary_margin_m"]
        < 0.000005
    )
    assert metrics["deadband_survival"]["0.000000000"] is True
    assert metrics["deadband_survival"]["0.000005000"] is False


def test_large_contact_proxy_separation_survives_deadband():
    metrics = contact_proxy_metrics(
        _descriptor(
            0.0020,
            0.0040,
        ),
        0.003,
        [0.00025, 0.0005, 0.0011],
        0.75,
        0.75,
    )

    assert np.isclose(
        metrics["max_symmetric_boundary_margin_m"],
        0.001,
    )
    assert metrics["deadband_survival"]["0.000250000"] is True
    assert metrics["deadband_survival"]["0.000500000"] is True
    assert metrics["deadband_survival"]["0.001100000"] is False


def test_wrap_support_is_independent_of_contact_proxy():
    metrics = contact_proxy_metrics(
        _descriptor(
            0.004,
            0.0045,
            wrap=0.8,
        ),
        0.003,
        [0.0],
        0.75,
        0.75,
    )

    assert metrics["reason"] == "original_wrap_only"
    assert metrics["has_contact_proxy_mismatch"] is False
    assert metrics["has_original_wrap_support"] is True


def test_global_angle_is_diagnostic_not_original_wrap_support():
    metrics = contact_proxy_metrics(
        _descriptor(
            0.004,
            0.0045,
            global_angle=0.9,
        ),
        0.003,
        [0.0],
        0.75,
        0.75,
    )

    assert metrics["has_original_wrap_support"] is False
    assert metrics["has_global_angle_support_diagnostic"] is True


@pytest.mark.parametrize(
    ("rollout_id", "batch_index", "env_index", "batch_seed"),
    [
        (0, 0, 0, 123),
        (31, 0, 31, 123),
        (32, 1, 0, 124),
        (63, 1, 31, 124),
        (64, 2, 0, 125),
        (127, 3, 31, 126),
        (218, 6, 26, 129),
        (255, 7, 31, 130),
        (256, 8, 0, 131),
        (426, 13, 10, 136),
        (511, 15, 31, 138),
    ],
)
def test_replay_locator_matches_frozen_scaleup_collection(
        rollout_id,
        batch_index,
        env_index,
        batch_seed):
    result = replay_locator(
        rollout_id,
        REV4_REPLAY["n_envs_per_batch"],
        REV4_REPLAY["base_seed"],
    )
    assert result == {
        "rollout_id": rollout_id,
        "batch_index": batch_index,
        "env_index": env_index,
        "batch_seed": batch_seed,
    }


def test_deadband_denominators_use_contact_proxy_supported_pairs():
    contact_supported, rows = deadband_sensitivity_rows(
        [0.0, 0.0005],
        {
            "0.000000000": 6,
            "0.000500000": 3,
        },
        {
            "contact_proxy_only": 4,
            "original_wrap_only": 4,
            "contact_and_original_wrap": 2,
        },
        10,
    )

    assert contact_supported == 6
    assert rows[0]["surviving_count"] == 6
    assert np.isclose(
        rows[0]["surviving_fraction_of_original_hidden"],
        0.6,
    )
    assert np.isclose(
        rows[0][
            "surviving_fraction_of_contact_proxy_supported"
        ],
        1.0,
    )
    assert np.isclose(
        rows[1]["surviving_fraction_of_original_hidden"],
        0.3,
    )
    assert np.isclose(
        rows[1][
            "surviving_fraction_of_contact_proxy_supported"
        ],
        0.5,
    )


def test_deadband_contact_proxy_denominator_zero_is_not_applicable():
    contact_supported, rows = deadband_sensitivity_rows(
        [0.0],
        {"0.000000000": 0},
        {
            "contact_proxy_only": 0,
            "original_wrap_only": 5,
            "contact_and_original_wrap": 0,
        },
        5,
    )

    assert contact_supported == 0
    assert rows[0][
        "surviving_fraction_of_contact_proxy_supported"
    ] is None
