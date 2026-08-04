import copy

import numpy as np
import pytest

from scripts.experiment2.phase0.phase0e_common import (
    contact_feature,
    preload_return_residual,
    rank_phase0e,
    summarize_phase0e_candidate,
)


def trace_fixture():
    return {
        "phase": np.asarray(["no_action", "no_action", "preload", "preload"]),
        "bead_positions": np.asarray(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[0.1, 0.0, 0.0], [1.1, 0.0, 0.0]],
                [[0.2, 0.0, 0.0], [1.2, 0.0, 0.0]],
                [[0.4, 0.0, 0.0], [1.4, 0.0, 0.0]],
            ]
        ),
        "contact_force_norm": np.asarray([0.0, 0.0, 2.0, 4.0]),
        "contact_max_force_norm": np.asarray([0.0, 0.0, 3.0, 5.0]),
        "contact_active_beads": np.asarray([0, 0, 1, 2]),
        "contact_mean_speed": np.asarray([0.0, 0.0, 0.2, 0.4]),
    }


def targets():
    return {
        "max_initial_abs_xy": 1e-12,
        "max_arm_jump": 1e-12,
        "max_median_no_action_drift": 0.0015,
        "max_median_preload_visible_difference": 0.004,
        "min_median_contact_impulse_gap": 0.08,
        "min_median_main_branch_ade": 0.004,
        "min_median_main_branch_fde": 0.01,
        "min_median_branch_amplification": 2.0,
        "max_grouped_vision_accuracy": 0.625,
        "min_grouped_contact_accuracy": 0.875,
        "min_seed_pass_fraction": 0.625,
    }


def pair_row():
    return {
        "action_hash_match": True,
        "free_base_state_hash_match": True,
        "hidden_base_state_hash_match": True,
        "free_hidden_initial_hash_match": True,
        "max_initial_state_difference": 0.0,
        "free_arm_max_abs_jump": 0.0,
        "hidden_arm_max_abs_jump": 0.0,
        "free_no_action_drift": 0.0001,
        "hidden_no_action_drift": 0.0001,
        "preload_end_max_abs_xy": 0.002,
        "contact_impulse_gap": 0.12,
        "main_branch_ade": 0.008,
        "main_branch_fde": 0.02,
        "branch_amplification": 10.0,
        "free_preload_return_residual": 0.001,
        "hidden_preload_return_residual": 0.001,
    }


def classifier(accuracy):
    return {"accuracy": accuracy, "predictions": [], "groups": ["a", "b", "c"]}


def summarize(vision=0.5, contact=1.0):
    return summarize_phase0e_candidate(
        "candidate",
        {"id": "candidate", "action": {"protocol": "direct"}},
        [copy.deepcopy(pair_row()) for _ in range(8)],
        targets(),
        classifier(vision),
        classifier(vision),
        classifier(contact),
    )


def test_contact_impulse_and_preload_return_residual():
    feature = contact_feature(trace_fixture(), hz=10.0, trace_stride=2)
    assert feature[0] == pytest.approx(1.2)
    assert feature[1] == 4.0
    assert feature[2] == 5.0
    assert preload_return_residual(trace_fixture()) == pytest.approx(0.3)


def test_candidate_rejects_visual_leakage_and_weak_contact_classifier():
    assert not summarize(vision=0.75, contact=1.0)["eligible"]
    assert not summarize(vision=0.5, contact=0.75)["eligible"]


def test_ranking_places_eligible_before_higher_scoring_review():
    eligible = summarize()
    review = copy.deepcopy(eligible)
    review["candidate_id"] = "review"
    review["eligible"] = False
    review["score"] = eligible["score"] + 100.0
    assert rank_phase0e([review, eligible])[0]["candidate_id"] == "candidate"
