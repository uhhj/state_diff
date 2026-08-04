from scripts.experiment2.phase0.calibration_common import (
    build_candidate_config,
    rank_candidates,
    summarize_candidate,
)


def _targets():
    return {
        "max_initial_abs_xy": 1e-12,
        "max_arm_jump": 1e-12,
        "max_median_no_action_drift": 0.0015,
        "max_median_preload_visible_difference": 0.006,
        "min_median_contact_impulse_gap": 0.08,
        "min_median_main_branch_ade": 0.004,
        "min_median_main_branch_fde": 0.01,
        "min_median_branch_amplification": 1.8,
    }


def _row():
    return {
        "action_hash_match": True,
        "free_base_state_hash_match": True,
        "hidden_base_state_hash_match": True,
        "free_hidden_initial_hash_match": True,
        "max_initial_state_difference": 0.0,
        "free_arm_max_abs_jump": 0.0,
        "hidden_arm_max_abs_jump": 0.0,
        "free_no_action_drift": 0.0004,
        "hidden_no_action_drift": 0.0005,
        "preload_end_max_abs_xy": 0.004,
        "contact_impulse_gap": 0.12,
        "main_branch_ade": 0.006,
        "main_branch_fde": 0.012,
    }


def test_build_candidate_config_does_not_mutate_base():
    base = {
        "action": {
            "preload_distance": 0.006,
            "main_pull_distance": 0.09,
        },
        "friction": {
            "selected_count": 5,
            "patch_radius": 0.045,
        },
    }
    candidate = {
        "id": "example",
        "action": {"preload_distance": 0.003},
        "friction": {"selected_count": 3},
    }
    result = build_candidate_config(base, candidate)
    assert result["action"]["preload_distance"] == 0.003
    assert result["action"]["main_pull_distance"] == 0.09
    assert result["friction"]["selected_count"] == 3
    assert result["friction"]["patch_radius"] == 0.045
    assert base["action"]["preload_distance"] == 0.006


def test_summarize_candidate_marks_good_rows_eligible():
    summary = summarize_candidate(
        "good",
        {"id": "good"},
        [_row(), _row(), _row()],
        _targets(),
    )
    assert summary["eligible"] is True
    assert summary["checks"]["exact_pairing"] is True
    assert summary["median"]["branch_amplification"] == 3.0


def test_rank_candidates_prefers_eligible_candidate():
    eligible = summarize_candidate(
        "eligible",
        {"id": "eligible"},
        [_row()],
        _targets(),
    )
    bad = _row()
    bad["preload_end_max_abs_xy"] = 0.02
    ineligible = summarize_candidate(
        "ineligible",
        {"id": "ineligible"},
        [bad],
        _targets(),
    )
    assert rank_candidates(
        [ineligible, eligible]
    )[0]["candidate_id"] == "eligible"
