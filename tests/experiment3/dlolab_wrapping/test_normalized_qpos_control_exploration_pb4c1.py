import copy

import numpy as np
import pytest

import scripts.experiment3.dlolab_wrapping.normalized_qpos_control_exploration_pb4c1 as pb4c1


def _alpha():
    return 0.2546772721216831


def _actions(alpha=None):
    a = _alpha() if alpha is None else float(alpha)
    ids = pb4c1.EXPECTED_ACTION_IDS
    masks = [
        (-a, -a),
        (-a, 0.0),
        (-a, a),
        (0.0, -a),
        (0.0, 0.0),
        (0.0, a),
        (a, -a),
        (a, 0.0),
        (a, a),
    ]
    return [
        {"action_id": action_id, "arm1_mask": m1, "arm2_mask": m2}
        for action_id, (m1, m2) in zip(ids, masks)
    ]


def _config():
    return {
        "classification": "EXPLORATORY",
        "family_lock": {
            "recompute_alpha": False,
            "round_alpha": False,
            "rescale_alpha": False,
            "clip_qpos_targets": False,
            "modify_action_ids": False,
            "modify_action_order": False,
            "modify_horizon": False,
            "require_zero_hard_limit_violations_pre_gpu": True,
            "source_of_truth": "configs/experiment3/published_benchmark/dlolab_wrapping_pb4c_family.json",
        },
        "control_rules": {
            "inherit_from_pb4": True,
            "primary_horizon_microsteps": 20,
            "repeat_count": 3,
            "absolute_regret_min": 0.05,
            "repeat_floor_multiplier": 5.0,
            "minimum_passing_pairs": 3,
            "minimum_passing_winding_strata": 2,
            "winding_strata_source":
                "PB4-C1 fresh live revalidation winding indices at branch time",
        },
        "exploratory_scope": {
            "same_cohort_reused_after_pb4_outcome_seen": True,
            "formal_condition5_confirmed": False,
            "condition5_confirmation_allowed": False,
            "positive_verdict": pb4c1.SIGNAL_FOUND,
            "negative_verdict": pb4c1.SIGNAL_NOT_FOUND,
        },
    }


def _shortlist():
    pairs = []
    for index in range(10):
        if index < 5:
            wa, wb = [0, 0, 0], [0, 1, 0]
        else:
            wa, wb = [0, 1, 0], [1, 1, 0]
        pairs.append(
            {
                "pair_id": f"p{index + 1}",
                "source_rank": index + 1,
                "rollout_a": 2 * index,
                "rollout_b": 2 * index + 1,
                "time_index": 13,
                "winding_index_a": wa,
                "winding_index_b": wb,
            }
        )
    return {"pairs": pairs}


def _family(alpha=None):
    a = _alpha() if alpha is None else float(alpha)
    return {
        "phase": "PB4-C0",
        "verdict": "PB4C0_FEASIBILITY_NORMALIZED_FAMILY_FROZEN",
        "classification": "CPU_ONLY_ACTION_FAMILY_DERIVATION",
        "formal_condition5_confirmed": False,
        "gpu_executed": False,
        "intervention_space": "per_arm_joint_position_target_qpos",
        "cartesian_motion_reversal_claim": False,
        "branch_times": [13],
        "primary_horizon_microsteps": 20,
        "position_limits": {
            "source": "pinned Panda hard limits",
            "comparison_tolerance_native_units": 1e-12,
        },
        "derivation": {"alpha_formal": a},
        "formal_family": {
            "mask_values": [-a, 0.0, a],
            "actions": _actions(a),
            "allow_clipping": False,
            "allow_per_joint_scale": False,
            "allow_per_arm_scale": False,
            "allow_per_pair_scale": False,
            "allow_horizon_change": False,
            "joint_limit_validation": True,
            "out_of_limit_command_count": 0,
        },
    }


def test_static_config_freezes_original_pb4_rules():
    pb4c1.validate_static_config(_config())

    changed = copy.deepcopy(_config())
    changed["control_rules"]["absolute_regret_min"] = 0.02
    with pytest.raises(RuntimeError, match="absolute_regret_min"):
        pb4c1.validate_static_config(changed)


def test_static_config_forbids_alpha_rescale_and_condition5_claim():
    changed = copy.deepcopy(_config())
    changed["family_lock"]["rescale_alpha"] = True
    with pytest.raises(RuntimeError, match="rescale_alpha"):
        pb4c1.validate_static_config(changed)

    changed = copy.deepcopy(_config())
    changed["exploratory_scope"]["formal_condition5_confirmed"] = True
    with pytest.raises(RuntimeError, match="Condition 5"):
        pb4c1.validate_static_config(changed)


def test_frozen_family_requires_exact_alpha_action_grid(monkeypatch):
    family = _family()
    evidence = {
        "scientific": {
            "family": copy.deepcopy(family),
            "formal_condition5_confirmed": False,
            "gpu_executed": False,
        }
    }
    shortlist = _shortlist()
    qpos = np.zeros((60, 18), dtype=np.float64)

    monkeypatch.setattr(pb4c1.pb4c0, "branch_times", lambda _: [13])
    monkeypatch.setattr(pb4c1.pb4c0, "load_pinned_panda_limits", lambda: {})
    monkeypatch.setattr(
        pb4c1.pb4c0,
        "count_grid_violations",
        lambda *args, **kwargs: {
            "generated_command_count": 180,
            "out_of_limit_command_count": 0,
            "joint_limit_violation_event_count": 0,
            "first_violation": None,
        },
    )
    monkeypatch.setattr(
        pb4c1.pb4c0,
        "min_limit_margins",
        lambda *args, **kwargs: {
            "minimum_revolute_joint_limit_margin_rad": 0.004,
            "minimum_prismatic_finger_limit_margin_m": 0.0,
        },
    )

    result = pb4c1.validate_frozen_family(
        family,
        evidence,
        shortlist,
        qpos,
    )
    assert result["alpha_formal"] == _alpha()
    assert result["actions"] == _actions()
    assert result["pre_gpu_hard_limit_validation"]["out_of_limit_command_count"] == 0


def test_frozen_family_rejects_alpha_action_mismatch(monkeypatch):
    family = _family()
    family["formal_family"]["actions"][0]["arm1_mask"] = -0.2
    evidence = {
        "scientific": {
            "family": copy.deepcopy(family),
            "formal_condition5_confirmed": False,
            "gpu_executed": False,
        }
    }
    with pytest.raises(RuntimeError, match="action masks"):
        pb4c1.validate_frozen_family(
            family,
            evidence,
            _shortlist(),
            np.zeros((60, 18), dtype=np.float64),
        )


def test_pre_gpu_hard_limit_drift_blocks_protocol(monkeypatch):
    family = _family()
    evidence = {
        "scientific": {
            "family": copy.deepcopy(family),
            "formal_condition5_confirmed": False,
            "gpu_executed": False,
        }
    }
    monkeypatch.setattr(pb4c1.pb4c0, "branch_times", lambda _: [13])
    monkeypatch.setattr(pb4c1.pb4c0, "load_pinned_panda_limits", lambda: {})
    monkeypatch.setattr(
        pb4c1.pb4c0,
        "count_grid_violations",
        lambda *args, **kwargs: {
            "generated_command_count": 180,
            "out_of_limit_command_count": 1,
            "joint_limit_violation_event_count": 1,
            "first_violation": {"fake": True},
        },
    )
    with pytest.raises(RuntimeError, match="hard-limit violation"):
        pb4c1.validate_frozen_family(
            family,
            evidence,
            _shortlist(),
            np.zeros((60, 18), dtype=np.float64),
        )


def test_runtime_config_uses_exact_frozen_family_without_recomputing():
    pb4_config = {
        "action_library": {},
        "control_relevance": {
            "repeat_count": 3,
            "best_action_statistic": "median_of_three_task_scores",
            "repeat_floor": "max_pairwise_absolute_score_difference_over_repeats",
            "absolute_regret_min": 0.05,
            "repeat_floor_multiplier": 5.0,
            "unique_best_rule": "frozen",
            "robust_cross_regret": "frozen",
            "pair_pass_rule": "frozen",
            "minimum_passing_pairs": 3,
            "minimum_passing_winding_strata": 2,
            "historical_pb3b1_stratum_is_formal_gate": False,
        },
        "outputs": {},
        "provenance": {},
    }
    config = _config()
    config.update(
        {
            "phase_name": "pb4c1-test",
            "outputs": {"protocol": "x", "raw_root": "y", "committed_report_dir": "z"},
            "provenance": {"starting_main_sha": "sha", "dlolab_revision": "dlo"},
        }
    )
    sources = {
        "pb4_config": pb4_config,
        "family_validation": {
            "alpha_formal": _alpha(),
            "mask_values": [-_alpha(), 0.0, _alpha()],
            "actions": _actions(),
        },
    }
    runtime = pb4c1.build_runtime_config(config, sources)
    assert runtime["action_library"]["alpha_formal"] == _alpha()
    assert runtime["action_library"]["actions"] == _actions()
    assert runtime["action_library"]["recompute_alpha"] is False
    assert runtime["action_library"]["allow_clipping"] is False


def _fake_live_records(shortlist):
    rows = []
    for pair in shortlist["pairs"]:
        rows.append(
            {
                "pair_id": pair["pair_id"],
                "valid": True,
                "live_winding_index_a": pair["winding_index_a"],
                "live_winding_index_b": pair["winding_index_b"],
            }
        )
    return rows


def _branch_records(rollout_id, t, best_action):
    rows = []
    for action_id in pb4c1.EXPECTED_ACTION_IDS:
        score = 1.0 if action_id == best_action else 0.7
        for repeat in range(3):
            rows.append(
                {
                    "rollout_id": rollout_id,
                    "time_index": t,
                    "action_id": action_id,
                    "repeat_index": repeat,
                    "task_score": score,
                    "survived_horizon": True,
                }
            )
    return rows


def test_exploratory_signal_keeps_three_pair_two_live_strata_rule():
    shortlist = _shortlist()
    records = []
    passing = {0, 1, 5}
    for index, pair in enumerate(shortlist["pairs"]):
        if index in passing:
            best_a = "arm1_negative_arm2_positive"
            best_b = "arm1_positive_arm2_negative"
        else:
            best_a = best_b = "both_positive"
        records.extend(_branch_records(pair["rollout_a"], 13, best_a))
        records.extend(_branch_records(pair["rollout_b"], 13, best_b))

    runtime = {
        "action_library": {"actions": _actions()},
        "control_relevance": {
            "absolute_regret_min": 0.05,
            "repeat_floor_multiplier": 5.0,
            "minimum_passing_pairs": 3,
            "minimum_passing_winding_strata": 2,
        },
    }
    audit = pb4c1.pb4.analyze_control_relevance(
        runtime,
        shortlist,
        records,
        _fake_live_records(shortlist),
    )
    assert audit["confirmed"] is True
    assert audit["passing_pair_count"] == 3
    assert audit["passing_winding_strata"] == 2


def test_complete_action_matrix_requires_540_unique_keys():
    shortlist = _shortlist()
    runtime = {"action_library": {"actions": _actions()}}
    records = []
    for pair in shortlist["pairs"]:
        for rollout_key in ("rollout_a", "rollout_b"):
            rollout_id = pair[rollout_key]
            for action in _actions():
                for repeat in range(3):
                    records.append(
                        {
                            "rollout_id": rollout_id,
                            "time_index": pair["time_index"],
                            "action_id": action["action_id"],
                            "repeat_index": repeat,
                        }
                    )
    assert pb4c1.validate_complete_action_matrix(runtime, shortlist, records) == 540


def _scientific_stub(live_valid, snapshot_valid, restores, audit, blocked):
    return {
        "live_pair_barrier": {"valid": live_valid},
        "snapshot_restore_barrier": {"valid": snapshot_valid},
        "snapshot_restore_records": restores,
        "audit": audit,
        "blocked_details": blocked,
    }


def test_blocked_result_boundaries_are_preserved():
    live = _scientific_stub(
        False, False, [], None, {"failure_component": "live_cohort_revalidation"}
    )
    assert pb4c1.validate_result_boundaries(pb4c1.LIVE_FAILED, live, []) == "blocked_live"

    snapshot = _scientific_stub(
        True,
        False,
        [{"valid": False}],
        None,
        {"failure_component": "snapshot_restore_validation"},
    )
    assert pb4c1.validate_result_boundaries(
        pb4c1.SNAPSHOT_FAILED, snapshot, []
    ) == "blocked_snapshot"

    action = _scientific_stub(
        True,
        True,
        [{"valid": True}] * 60,
        None,
        {"failure_component": "incomplete_action_matrix"},
    )
    assert pb4c1.validate_result_boundaries(
        pb4c1.ACTION_FAILED, action, []
    ) == "blocked_action"
