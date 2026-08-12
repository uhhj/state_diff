import numpy as np

from scripts.experiment3.dlolab_wrapping.replay_alignment_root_cause_pb3r1 import (
    classify_history_dependence,
    spatial_error_metrics,
    worker_plan,
)


def test_uniform_translation_is_removed_by_centered_metric():
    frozen = np.zeros((50, 3), dtype=np.float64)
    live = frozen + np.array([1e-5, -2e-5, 3e-5], dtype=np.float64)

    row = spatial_error_metrics(live, frozen)

    assert row["translation_removed_coordinate_rmse_m"] < 1e-15
    assert np.isclose(
        row["mean_translation_norm_m"],
        np.linalg.norm([1e-5, -2e-5, 3e-5]),
    )


def test_local_vertex_error_survives_translation_removal():
    frozen = np.zeros((50, 3), dtype=np.float64)
    live = frozen.copy()
    live[28] = np.array([5e-5, 0.0, 0.0])

    row = spatial_error_metrics(live, frozen)

    assert row["argmax_vertex"] == 28
    assert row["translation_removed_fraction"] > 0.95


def _summary(coordinate_rmse, max_abs, *, all_pass):
    return {
        "median_coordinate_rmse_m": coordinate_rmse,
        "median_rope_max_abs_coordinate_m": max_abs,
        "all_t0_alignment_pass": True,
        "all_original_pb3_alignment_pass": all_pass,
    }


def test_history_dependence_requires_twofold_reduction_and_original_pass():
    config = {
        "diagnostic_decision": {
            "history_dependence_material_reduction_ratio": 0.5,
        }
    }
    summaries = {
        "m0_sequential_history": _summary(
            2e-6, 2e-5, all_pass=True
        ),
        "m1_isolated_collector": _summary(
            8e-6, 6e-5, all_pass=False
        ),
    }

    row = classify_history_dependence(config, summaries)

    assert row["history_dependence_confirmed"] is True


def test_small_improvement_does_not_confirm_history_dependence():
    config = {
        "diagnostic_decision": {
            "history_dependence_material_reduction_ratio": 0.5,
        }
    }
    summaries = {
        "m0_sequential_history": _summary(
            6e-6, 4.8e-5, all_pass=True
        ),
        "m1_isolated_collector": _summary(
            8e-6, 5.2e-5, all_pass=False
        ),
    }

    row = classify_history_dependence(config, summaries)

    assert row["history_dependence_confirmed"] is False


def test_original_alignment_failure_blocks_history_confirmation():
    config = {
        "diagnostic_decision": {
            "history_dependence_material_reduction_ratio": 0.5,
        }
    }
    summaries = {
        "m0_sequential_history": _summary(
            1e-6, 1e-5, all_pass=False
        ),
        "m1_isolated_collector": _summary(
            8e-6, 6e-5, all_pass=False
        ),
    }

    row = classify_history_dependence(config, summaries)

    assert row["history_dependence_confirmed"] is False


def test_worker_plan_is_cost_bounded_to_seven_fresh_processes():
    config = {
        "replay": {
            "modes": {
                "m0_sequential_history": {"repeats": 3},
                "m1_isolated_collector": {"repeats": 3},
                "m2_build_state_restore": {"repeats": 1},
            }
        }
    }

    plan = worker_plan(config)

    assert len(plan) == 7
    assert plan == [
        ("m0_sequential_history", 0),
        ("m0_sequential_history", 1),
        ("m0_sequential_history", 2),
        ("m1_isolated_collector", 0),
        ("m1_isolated_collector", 1),
        ("m1_isolated_collector", 2),
        ("m2_build_state_restore", 0),
    ]



def test_t0_failure_prevents_history_confirmation():
    config = {
        "diagnostic_decision": {
            "history_dependence_material_reduction_ratio": 0.5,
        }
    }
    m0 = _summary(1e-6, 1e-5, all_pass=True)
    m1 = _summary(8e-6, 6e-5, all_pass=False)
    m0["all_t0_alignment_pass"] = False

    row = classify_history_dependence(
        config,
        {
            "m0_sequential_history": m0,
            "m1_isolated_collector": m1,
        },
    )

    assert row["history_dependence_confirmed"] is False
    assert row["t0_reconstruction_valid_for_m0_m1"] is False
