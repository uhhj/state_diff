"""PB4 — Wrapping control relevance / action regret audit.

Scientific question
-------------------
Given the already-confirmed PB3 same-action future bifurcation cohort, does the
hidden winding branch change which control suffix is best, with control-relevant
regret that exceeds deterministic repeat variability?

This phase:
1. freezes a symmetric 4-action arm continue/hold library before GPU use;
2. revalidates the same frozen PB3-B1 10-pair cohort live;
3. validates native snapshot restore before action evaluation;
4. evaluates all 20 branches x 4 actions x 3 repeats;
5. applies the pre-registered robust cross-regret rule.

The action library is derived only from the verified official best_qpos suffix:
for each 9-DOF arm block, either continue the official suffix or hold the branch
time qpos target. No hidden winding label, PB3 future metric, or PB4 outcome is
used to construct or modify actions.

No StateDiff/CFPM training and no sensor experiment is run here.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import itertools
import json
import os
from pathlib import Path

import numpy as np

from scripts.experiment3.dlolab_wrapping.paths import REPO_ROOT, official_log_dir
from scripts.experiment3.dlolab_wrapping.prospective_causal_audit_pb3b2 import (
    _capture_snapshots,
    _revalidate_pairs,
    _validate_all_snapshot_restores,
    validate_and_normalize_cohort,
)
from scripts.experiment3.dlolab_wrapping.replay_winding_audit import (
    _dual_arm_command,
    build_env,
    load_json,
    normalize_qpos,
    sample_env,
    to_numpy,
)
from scripts.experiment3.dlolab_wrapping.snapshot_causal_audit_pb3 import (
    PB3Blocked,
    _jsonable,
    extract_env_sample,
    git,
    group_sides_by_batch_time,
    load_frozen_rollouts,
    required_side_records,
    seed_everything,
)
from utils.domain_randomization import wrapping_args


PROTOCOL_VERDICT = "PB4_CONTROL_RELEVANCE_PROTOCOL_PREREGISTERED"
POSITIVE_VERDICT = "PB4_CONTROL_RELEVANCE_CONFIRMED"
NEGATIVE_VERDICT = "PB4_CONTROL_RELEVANCE_NOT_CONFIRMED"
LIVE_FAILED = "PB4_LIVE_COHORT_REVALIDATION_FAILED"
SNAPSHOT_FAILED = "PB4_SNAPSHOT_RESTORE_FAILED"
ACTION_FAILED = "PB4_ACTION_EVALUATION_FAILED"


@dataclass
class PB4Blocked(Exception):
    verdict: str
    details: dict


def _source_paths(config):
    source = config["source"]
    return {
        "cohort": REPO_ROOT / source["cohort"],
        "pb3b1_evidence": REPO_ROOT / source["pb3b1_evidence"],
        "pb3b1_raw_rollouts": Path(source["pb3b1_raw_rollouts"]),
        "pb3r3_alignment_rule": REPO_ROOT / source["pb3r3_alignment_rule"],
        "pb3b2_evidence": REPO_ROOT / source["pb3b2_evidence"],
    }


def validate_sources(config):
    paths = _source_paths(config)
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(str(path))

    raw = load_frozen_rollouts(paths["pb3b1_raw_rollouts"])
    shortlist, cohort_validation, rule = validate_and_normalize_cohort(
        config,
        raw,
    )

    pb3b2 = load_json(paths["pb3b2_evidence"])
    expected = config["source"]["expected_pb3b2_verdict"]
    if pb3b2.get("verdict") != expected:
        raise RuntimeError(
            f"PB3-B2 verdict mismatch: {pb3b2.get('verdict')} != {expected}"
        )
    scientific = pb3b2["scientific"]
    if scientific.get("causal_future_status") != "TESTED":
        raise RuntimeError("PB3-B2 causal future status is not TESTED")
    audit = scientific.get("audit")
    if not isinstance(audit, dict) or not bool(audit.get("confirmed")):
        raise RuntimeError("PB3-B2 audit is not confirmed")
    if int(audit.get("passing_pair_count", -1)) != 10:
        raise RuntimeError("PB3-B2 must have 10/10 Gate-4 passing pairs")
    if len(audit.get("passing_winding_strata", [])) < 2:
        raise RuntimeError("PB3-B2 must cover at least two passing strata")
    if scientific.get("boundaries", {}).get("pb4_started"):
        raise RuntimeError("PB3-B2 evidence unexpectedly says PB4 already started")

    if _jsonable(dict(wrapping_args)) != config["replay"]["expected_wrapping_args"]:
        raise RuntimeError("Pinned wrapping_args mismatch")

    snapshot_threshold = float(
        config["barriers"]["snapshot_restore_rope_max_abs_m"]
    )
    frozen_snapshot_threshold = float(
        rule["snapshot_restore_alignment"]["threshold_m"]
    )
    if snapshot_threshold != frozen_snapshot_threshold:
        raise RuntimeError(
            "PB4 snapshot threshold differs from frozen PB3-R3 rule"
        )

    if (
        not config["barriers"]["require_all_10_live_pairs"]
        or not config["barriers"]["require_20_unique_branches"]
    ):
        raise RuntimeError("PB4 live/snapshot barriers must remain enabled")
    if (
        config["barriers"]["allow_pair_drop"]
        or config["barriers"]["allow_pair_replacement"]
    ):
        raise RuntimeError("PB4 must forbid pair drop/replacement")
    if not config["barriers"].get("live_prefix_validity_required", False):
        raise RuntimeError("PB4 must require prefix validity through branch time")
    if config["control_relevance"].get(
            "historical_pb3b1_stratum_is_formal_gate", True):
        raise RuntimeError(
            "PB4 phase criterion must use fresh live winding strata"
        )
    if config["control_relevance"].get("winding_strata_source") != (
        "PB4 fresh live revalidation winding indices at branch time"
    ):
        raise RuntimeError("PB4 winding strata source changed")
    if not config["task_score"].get("store_score_contributions", False):
        raise RuntimeError("PB4 must store per-microstep score contributions")
    if int(config["task_score"].get(
            "score_contribution_count_per_record", -1)) != 20:
        raise RuntimeError("PB4 must store exactly 20 score contributions")

    qpos_path = official_log_dir() / "best_qpos.npy"
    if not qpos_path.is_file():
        raise FileNotFoundError(str(qpos_path))
    qpos = normalize_qpos(np.load(qpos_path))
    if not np.array_equal(qpos, raw["common_qpos_replay"]):
        raise RuntimeError(
            "Official best_qpos differs from frozen PB3-B1 common replay"
        )
    if qpos.ndim != 2 or qpos.shape[1] != 18:
        raise RuntimeError(f"Expected best_qpos [T,18], got {qpos.shape}")

    return raw, shortlist, cohort_validation, rule, pb3b2, qpos


def validate_action_library(config, qpos, shortlist):
    lib = config["action_library"]
    horizon = int(lib["primary_horizon_microsteps"])
    block = int(lib["arm_dof_block"])
    actions = lib["actions"]

    if horizon != 20:
        raise RuntimeError("PB4 primary action horizon must remain 20 microsteps")
    if int(lib["macro_step_equivalents"]) != 2:
        raise RuntimeError("PB4 horizon must remain two CMA-ES macro-step equivalents")
    if block != 9 or qpos.shape[1] != 2 * block:
        raise RuntimeError("PB4 expects two 9-DOF qpos arm blocks")

    expected = [
        ("both_continue", 1.0, 1.0),
        ("arm1_continue_arm2_hold", 1.0, 0.0),
        ("arm1_hold_arm2_continue", 0.0, 1.0),
        ("hold_both", 0.0, 0.0),
    ]
    actual = [
        (
            row["action_id"],
            float(row["arm1_mask"]),
            float(row["arm2_mask"]),
        )
        for row in actions
    ]
    if actual != expected:
        raise RuntimeError(f"PB4 frozen action library changed: {actual}")

    if (
        lib["uses_hidden_winding_to_construct_action"]
        or lib["uses_pb3_pair_future_metrics_to_construct_action"]
        or lib["uses_pb4_outcomes_to_construct_action"]
        or lib["allow_action_addition_after_gpu"]
        or lib["allow_action_removal_after_gpu"]
    ):
        raise RuntimeError("PB4 action-library leakage/mutation flag is enabled")

    max_t = max(int(pair["time_index"]) for pair in shortlist["pairs"])
    if max_t + horizon >= len(qpos):
        raise RuntimeError(
            f"PB4 action horizon exceeds qpos: max_t={max_t}, H={horizon}, "
            f"len={len(qpos)}"
        )
    return actions


def masked_qpos_command(qpos, time_index, relative_step, action):
    """Construct one counterfactual joint-position command.

    Each arm independently either follows the official qpos displacement from
    q_t or holds q_t. The two branch states of a pair therefore receive exactly
    the same action sequence.
    """
    qpos = np.asarray(qpos, dtype=np.float64)
    t = int(time_index)
    r = int(relative_step)
    base = qpos[t]
    target = qpos[t + r]
    if base.shape != (18,) or target.shape != (18,):
        raise ValueError("Expected 18D qpos commands")

    out = base.copy()
    m1 = float(action["arm1_mask"])
    m2 = float(action["arm2_mask"])
    out[:9] = base[:9] + m1 * (target[:9] - base[:9])
    out[9:] = base[9:] + m2 * (target[9:] - base[9:])
    if not np.isfinite(out).all():
        raise ValueError("Non-finite PB4 action command")
    return out


def build_protocol(config, shortlist, cohort_validation, rule, pb3b2, qpos):
    actions = validate_action_library(config, qpos, shortlist)
    cr = config["control_relevance"]

    if int(cr["repeat_count"]) != 3:
        raise RuntimeError("PB4 repeat_count must remain 3")
    if float(cr["absolute_regret_min"]) != 0.05:
        raise RuntimeError("PB4 absolute regret threshold must remain 0.05")
    if float(cr["repeat_floor_multiplier"]) != 5.0:
        raise RuntimeError("PB4 repeat-floor multiplier must remain 5")
    if int(cr["minimum_passing_pairs"]) != 3:
        raise RuntimeError("PB4 minimum passing pair count must remain 3")
    if int(cr["minimum_passing_winding_strata"]) != 2:
        raise RuntimeError("PB4 minimum passing winding strata must remain 2")

    return {
        "phase": "PB4",
        "verdict": PROTOCOL_VERDICT,
        "scientific_question": (
            "Does the hidden winding branch change the preferred control suffix "
            "with repeat-robust task regret?"
        ),
        "source": {
            "pb3b1_cohort_pair_count": cohort_validation["pair_count"],
            "pb3b1_unique_rollout_count":
                cohort_validation["unique_rollout_count"],
            "pb3b1_live_winding_strata_count":
                cohort_validation["live_winding_strata_count"],
            "pb3b2_verdict": pb3b2["verdict"],
            "pb3b2_gate4_passing_pairs":
                pb3b2["scientific"]["audit"]["passing_pair_count"],
            "pb3b2_passing_winding_strata":
                pb3b2["scientific"]["audit"]["passing_winding_strata"],
        },
        "action_library": {
            "interface": config["action_library"]["interface"],
            "primary_horizon_microsteps":
                config["action_library"]["primary_horizon_microsteps"],
            "macro_step_equivalents":
                config["action_library"]["macro_step_equivalents"],
            "arm_dof_block": config["action_library"]["arm_dof_block"],
            "construction": config["action_library"]["construction"],
            "actions": actions,
            "uses_hidden_winding_to_construct_action": False,
            "uses_pb3_pair_future_metrics_to_construct_action": False,
            "uses_pb4_outcomes_to_construct_action": False,
            "allow_action_addition_after_gpu": False,
            "allow_action_removal_after_gpu": False,
        },
        "task_score": config["task_score"],
        "barriers": {
            "all_10_live_pairs_before_snapshot": True,
            "live_prefix_validity_required": True,
            "live_prefix_validity_rule":
                config["barriers"]["live_prefix_validity_rule"],
            "all_20_branches_restore_before_actions": True,
            "snapshot_restore_rope_max_abs_m":
                float(rule["snapshot_restore_alignment"]["threshold_m"]),
            "pair_drop_allowed": False,
            "pair_replacement_allowed": False,
        },
        "control_relevance": config["control_relevance"],
        "qpos": {
            "shape": [int(v) for v in qpos.shape],
            "action_horizon_fits_all_pairs": True,
        },
        "phase_boundary": {
            "deployable_sensor_test_started": False,
            "state_diff_training_started": False,
            "cfpm_training_started": False,
        },
    }


def freeze_protocol(config_path):
    config = load_json(config_path)
    raw, shortlist, validation, rule, pb3b2, qpos = validate_sources(config)
    protocol = build_protocol(
        config, shortlist, validation, rule, pb3b2, qpos
    )
    path = REPO_ROOT / config["outputs"]["protocol"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    print(f"verdict={PROTOCOL_VERDICT}")
    print(f"protocol={path}")
    print(
        "actions="
        + ",".join(row["action_id"] for row in protocol["action_library"]["actions"])
    )
    return protocol


def load_and_validate_protocol(
        config, shortlist, validation, rule, pb3b2, qpos):
    path = REPO_ROOT / config["outputs"]["protocol"]
    if not path.is_file():
        raise FileNotFoundError(str(path))
    actual = load_json(path)
    expected = build_protocol(
        config, shortlist, validation, rule, pb3b2, qpos
    )
    if actual != expected:
        raise RuntimeError(
            "Committed PB4 protocol differs from deterministic pre-GPU derivation"
        )
    return actual


def _winding_stratum(index_a, index_b):
    a = ",".join(str(int(v)) for v in index_a)
    b = ",".join(str(int(v)) for v in index_b)
    return "<->".join(sorted((a, b)))


def prefix_validity_from_failure_steps(
        time_index,
        first_stretch_step,
        first_rope_nan_step):
    """Return PB3-B1-compatible physical prefix validity at branch time."""
    t = int(time_index)
    stretch = int(first_stretch_step)
    rope_nan = int(first_rope_nan_step)
    first_fail = min(stretch, rope_nan)
    return {
        "prefix_valid_through_t": bool(first_fail > t),
        "first_fail_step": None if first_fail >= 10**9 else int(first_fail),
        "failed_stretch_through_t": bool(stretch <= t),
        "failed_rope_nan_through_t": bool(rope_nan <= t),
    }


def _current_physical_failure_steps(env, stretch_limit):
    rope = np.asarray(env.rope.get_all_verts())
    rope_nan = np.isnan(rope).any(axis=(1, 2))
    distance = to_numpy(
        env.rope.get_geodesic_distance(
            env.control_idx[0],
            env.control_idx[1],
        )
    ).reshape(env.n_envs)
    control_init = to_numpy(
        env.control_dist_init
    ).reshape(env.n_envs)
    stretch = distance / control_init > float(stretch_limit)
    return np.asarray(stretch, dtype=bool), np.asarray(rope_nan, dtype=bool)


def _collect_live_histories_with_prefix_validity(
        env, config, shortlist, qpos, build_state):
    """PB4 fresh replay with explicit PB3-B1 prefix-failure tracking.

    Selection/history semantics are still delegated to the already-frozen
    PB3 live revalidation helper. This collector adds only the missing physical
    prefix condition:
      no stretch / rope-NaN failure at or before branch time t.
    """
    sides = required_side_records(shortlist)
    groups = group_sides_by_batch_time(sides)
    by_batch = {}
    for (batch, time_index), members in groups.items():
        by_batch.setdefault(batch, {})[time_index] = members

    n_intervals = env.steps_interval // env._cmaes_n_steps_sub
    stretch_limit = float(config["replay"]["official_stretch_ratio_limit"])
    sentinel = 10**9
    output = {key: {"meta": row} for key, row in sides.items()}

    for batch in sorted(by_batch):
        env.scene.reset(state=build_state)
        seed_everything(int(config["replay"]["base_seed"]) + int(batch))
        env.use_qpos = True
        env.reset()

        times = sorted(by_batch[batch])
        max_time = max(times)
        history_steps = {
            step
            for t in times
            for step in (t - 2, t - 1, t)
        }
        samples = {}

        first_stretch = np.full(env.n_envs, sentinel, dtype=np.int64)
        first_rope_nan = np.full(env.n_envs, sentinel, dtype=np.int64)

        stretch_now, nan_now = _current_physical_failure_steps(
            env,
            stretch_limit,
        )
        first_stretch[stretch_now] = 0
        first_rope_nan[nan_now] = 0

        for step in range(1, max_time + 1):
            _dual_arm_command(env, qpos[step])
            for _ in range(n_intervals):
                env.scene.step()

            stretch_now, nan_now = _current_physical_failure_steps(
                env,
                stretch_limit,
            )
            new_stretch = stretch_now & (first_stretch == sentinel)
            new_nan = nan_now & (first_rope_nan == sentinel)
            first_stretch[new_stretch] = int(step)
            first_rope_nan[new_nan] = int(step)

            if step in history_steps:
                samples[step] = sample_env(env)

        for t in times:
            for key, row in by_batch[batch][t]:
                env_index = int(row["env_index"])
                output[key]["live_history"] = [
                    extract_env_sample(
                        samples[step],
                        env_index,
                    )
                    for step in (t - 2, t - 1, t)
                ]
                output[key]["common_action_history"] = np.asarray(
                    qpos[t - 2:t + 1]
                ).copy()
                output[key]["prefix_validity"] = (
                    prefix_validity_from_failure_steps(
                        t,
                        first_stretch[env_index],
                        first_rope_nan[env_index],
                    )
                )

    return output, groups, by_batch, n_intervals


def _live_winding_index(history):
    turns = np.asarray(
        history[-1]["signed_winding_turns"],
        dtype=np.float64,
    )
    if turns.shape != (3,) or not np.isfinite(turns).all():
        raise ValueError(
            f"Expected finite 3D winding turns, got shape={turns.shape}"
        )
    return [int(v) for v in np.rint(turns).astype(np.int8)]


def _revalidate_pairs_with_prefix(shortlist, live, rule):
    records = _revalidate_pairs(
        shortlist,
        live,
        rule,
    )
    if len(records) != len(shortlist["pairs"]):
        raise RuntimeError("PB4 live revalidation record count mismatch")

    for pair, record in zip(shortlist["pairs"], records):
        t = int(pair["time_index"])
        key_a = (int(pair["rollout_a"]), t)
        key_b = (int(pair["rollout_b"]), t)

        prefix_a = dict(live[key_a]["prefix_validity"])
        prefix_b = dict(live[key_b]["prefix_validity"])
        live_index_a = _live_winding_index(
            live[key_a]["live_history"]
        )
        live_index_b = _live_winding_index(
            live[key_b]["live_history"]
        )

        live_stratum = _winding_stratum(
            live_index_a,
            live_index_b,
        )
        frozen_stratum = _winding_stratum(
            pair["winding_index_a"],
            pair["winding_index_b"],
        )

        record["prefix_validity_a"] = prefix_a
        record["prefix_validity_b"] = prefix_b
        record["prefix_valid_through_t"] = bool(
            prefix_a["prefix_valid_through_t"]
            and prefix_b["prefix_valid_through_t"]
        )
        record["live_winding_index_a"] = live_index_a
        record["live_winding_index_b"] = live_index_b
        record["pb4_live_winding_stratum"] = live_stratum
        record["pb3b1_frozen_winding_stratum"] = frozen_stratum
        record["stratum_match"] = bool(live_stratum == frozen_stratum)

        # Preserve the already-frozen live observation test, but strengthen it
        # with the PB3-B1 physical prefix condition.
        record["valid"] = bool(
            record.get("valid", False)
            and record["prefix_valid_through_t"]
            and live_index_a != live_index_b
        )

    return records


def _live_barrier(records):
    if len(records) != 10 or not all(bool(row.get("valid")) for row in records):
        raise PB4Blocked(
            LIVE_FAILED,
            {
                "failure_component": "live_cohort_revalidation",
                "expected_pair_count": 10,
                "actual_pair_count": len(records),
                "failed_pairs": [
                    row for row in records if not bool(row.get("valid"))
                ],
                "action_evaluation_started": False,
            },
        )


def _snapshot_barrier(
        env, config, by_batch, snapshots, references, side_output, records):
    """Validate native restores while preserving partial blocked evidence."""
    try:
        _validate_all_snapshot_restores(
            env,
            config,
            by_batch,
            snapshots,
            references,
            side_output,
            records,
        )
    except PB3Blocked as exc:
        raise PB4Blocked(
            SNAPSHOT_FAILED,
            {
                "failure_component": "snapshot_restore_validation",
                "pb3_helper_details": exc.details,
                "actual_restore_records": len(records),
                "failed_restores": [
                    row for row in records if not bool(row.get("valid"))
                ],
                "action_evaluation_started": False,
            },
        ) from exc

    if len(records) != 60 or not all(bool(row["valid"]) for row in records):
        raise PB4Blocked(
            SNAPSHOT_FAILED,
            {
                "failure_component": "snapshot_restore_validation",
                "actual_restore_records": len(records),
                "failed_restores": [
                    row for row in records if not bool(row["valid"])
                ],
                "action_evaluation_started": False,
            },
        )
    return records



def published_score_transition(
        alive, rope_nan, stretch_failed, reward):
    """One PB4 score step matching pinned Wrapping cumulative semantics.

    Physical failure (rope NaN / stretch) permanently clears alive.
    Reward NaN alone contributes zero for this microstep but does not clear
    alive.
    """
    alive = bool(alive)
    rope_nan = bool(rope_nan)
    stretch_failed = bool(stretch_failed)
    reward_finite = bool(np.isfinite(reward))

    physical_failure = bool(
        alive and (rope_nan or stretch_failed)
    )
    if physical_failure:
        alive = False

    contribution = (
        float(reward) + 1.0
        if alive and reward_finite
        else 0.0
    )
    return {
        "alive": alive,
        "contribution": float(contribution),
        "physical_failure": physical_failure,
        "reward_nan": bool(not reward_finite),
    }


def recompute_task_score(score_contributions, horizon):
    values = np.asarray(score_contributions, dtype=np.float64)
    if values.shape != (int(horizon),):
        raise ValueError(
            f"Expected {horizon} score contributions, got {values.shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError("Non-finite score contribution")
    return float(np.sum(values, dtype=np.float64) / float(horizon))


def _evaluate_actions(
        env,
        config,
        by_batch,
        snapshots,
        qpos,
        n_intervals,
        records):
    actions = config["action_library"]["actions"]
    horizon = int(config["action_library"]["primary_horizon_microsteps"])
    repeats = int(config["control_relevance"]["repeat_count"])
    stretch_limit = float(config["replay"]["official_stretch_ratio_limit"])

    for batch in sorted(by_batch):
        for t in sorted(by_batch[batch]):
            group = (batch, t)
            members = by_batch[batch][t]

            for action in actions:
                action_id = action["action_id"]
                for repeat in range(repeats):
                    env.scene.reset(state=snapshots[group])

                    state = {}
                    for _, row in members:
                        key = (int(row["rollout_id"]), int(t))
                        state[key] = {
                            "alive": True,
                            "failure_step": None,
                            "last_reward": None,
                            "reward_nan_steps": [],
                            "score_contributions": [],
                        }

                    for relative in range(1, horizon + 1):
                        command = masked_qpos_command(
                            qpos,
                            t,
                            relative,
                            action,
                        )
                        _dual_arm_command(env, command)
                        for _ in range(n_intervals):
                            env.scene.step()

                        rope = np.asarray(env.rope.get_all_verts())
                        distance = to_numpy(
                            env.rope.get_geodesic_distance(
                                env.control_idx[0],
                                env.control_idx[1],
                            )
                        ).reshape(env.n_envs)
                        control_init = to_numpy(
                            env.control_dist_init
                        ).reshape(env.n_envs)
                        stretch = distance / control_init
                        reward = np.asarray(env.reward(), dtype=np.float64)

                        for _, row in members:
                            key = (int(row["rollout_id"]), int(t))
                            item = state[key]
                            env_index = int(row["env_index"])

                            transition = published_score_transition(
                                item["alive"],
                                np.isnan(rope[env_index]).any(),
                                stretch[env_index] > stretch_limit,
                                reward[env_index],
                            )
                            if (
                                transition["physical_failure"]
                                and item["failure_step"] is None
                            ):
                                item["failure_step"] = int(relative)

                            if transition["reward_nan"]:
                                item["reward_nan_steps"].append(int(relative))

                            item["alive"] = bool(transition["alive"])
                            item["score_contributions"].append(
                                float(transition["contribution"])
                            )
                            item["last_reward"] = (
                                float(reward[env_index])
                                if np.isfinite(reward[env_index])
                                else None
                            )

                    final_sample = sample_env(env)

                    for _, row in members:
                        key = (int(row["rollout_id"]), int(t))
                        item = state[key]
                        score = recompute_task_score(
                            item["score_contributions"],
                            horizon,
                        )

                        env_index = int(row["env_index"])
                        turns = np.asarray(
                            final_sample[
                                "signed_winding_turns"
                            ][env_index],
                            dtype=np.float64,
                        )
                        terminal_winding = (
                            [float(v) for v in turns]
                            if np.isfinite(turns).all()
                            else None
                        )

                        records.append(
                            {
                                "rollout_id": key[0],
                                "time_index": key[1],
                                "batch_index": int(batch),
                                "env_index": env_index,
                                "action_id": action_id,
                                "arm1_mask": float(action["arm1_mask"]),
                                "arm2_mask": float(action["arm2_mask"]),
                                "repeat_index": int(repeat),
                                "horizon_microsteps": horizon,
                                "score_contributions": [
                                    float(v)
                                    for v in item["score_contributions"]
                                ],
                                "task_score": score,
                                "survived_horizon": bool(item["alive"]),
                                "failure_step": item["failure_step"],
                                "reward_nan_steps": item["reward_nan_steps"],
                                "terminal_reward": item["last_reward"],
                                "terminal_signed_winding_turns":
                                    terminal_winding,
                            }
                        )

    expected = 20 * len(actions) * repeats
    keys = {
        (
            int(row["rollout_id"]),
            int(row["time_index"]),
            row["action_id"],
            int(row["repeat_index"]),
        )
        for row in records
    }
    if len(records) != expected or len(keys) != expected:
        raise PB4Blocked(
            ACTION_FAILED,
            {
                "failure_component": "incomplete_action_matrix",
                "expected_records": expected,
                "actual_records": len(records),
                "unique_records": len(keys),
            },
        )

    for row in records:
        recomputed = recompute_task_score(
            row["score_contributions"],
            horizon,
        )
        if not np.isclose(
            float(row["task_score"]),
            recomputed,
            rtol=0.0,
            atol=1e-12,
        ):
            raise PB4Blocked(
                ACTION_FAILED,
                {
                    "failure_component": "task_score_reconstruction_mismatch",
                    "record": row,
                    "recomputed_task_score": recomputed,
                },
            )
    return records


def max_pairwise_abs(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape != (3,):
        raise ValueError(f"Expected exactly 3 repeat values, got {arr.shape}")
    return float(
        max(
            abs(float(arr[i]) - float(arr[j]))
            for i, j in itertools.combinations(range(3), 2)
        )
    )


def summarize_branch(action_records, actions, absolute_min, floor_multiplier):
    by_action = {}
    for action in actions:
        action_id = action["action_id"]
        rows = sorted(
            [row for row in action_records if row["action_id"] == action_id],
            key=lambda row: int(row["repeat_index"]),
        )
        if len(rows) != 3 or [int(row["repeat_index"]) for row in rows] != [0, 1, 2]:
            raise RuntimeError(f"Incomplete repeats for action {action_id}")
        scores = [float(row["task_score"]) for row in rows]
        by_action[action_id] = {
            "scores": scores,
            "median_score": float(np.median(scores)),
            "repeat_floor": max_pairwise_abs(scores),
            "survived_repeat_count": int(
                sum(bool(row["survived_horizon"]) for row in rows)
            ),
        }

    ordered = sorted(
        by_action.items(),
        key=lambda kv: (
            -float(kv[1]["median_score"]),
            [a["action_id"] for a in actions].index(kv[0]),
        ),
    )
    best_id, best_row = ordered[0]
    second_id, second_row = ordered[1]
    branch_floor = float(
        max(row["repeat_floor"] for row in by_action.values())
    )
    threshold = float(max(absolute_min, floor_multiplier * branch_floor))
    top_gap = float(
        best_row["median_score"] - second_row["median_score"]
    )
    unique_best = bool(top_gap >= threshold)

    return {
        "actions": by_action,
        "ranking": [action_id for action_id, _ in ordered],
        "best_action_id": best_id,
        "second_action_id": second_id,
        "top_median_gap": top_gap,
        "repeat_floor_max": branch_floor,
        "unique_best_threshold": threshold,
        "unique_best": unique_best,
    }


def _scores(branch_summary, action_id):
    return np.asarray(
        branch_summary["actions"][action_id]["scores"],
        dtype=np.float64,
    )


def analyze_control_relevance(
        config,
        shortlist,
        action_records,
        live_records):
    actions = config["action_library"]["actions"]
    cr = config["control_relevance"]
    absolute_min = float(cr["absolute_regret_min"])
    multiplier = float(cr["repeat_floor_multiplier"])

    by_branch = {}
    for row in action_records:
        key = (int(row["rollout_id"]), int(row["time_index"]))
        by_branch.setdefault(key, []).append(row)

    live_by_pair = {
        row["pair_id"]: row
        for row in live_records
    }
    if len(live_by_pair) != 10:
        raise RuntimeError(
            f"Expected 10 PB4 live pair records, got {len(live_by_pair)}"
        )

    pair_results = []
    for pair in shortlist["pairs"]:
        t = int(pair["time_index"])
        key_a = (int(pair["rollout_a"]), t)
        key_b = (int(pair["rollout_b"]), t)
        live = live_by_pair[pair["pair_id"]]

        if not bool(live.get("valid")):
            raise RuntimeError(
                f"PB4 control analysis received invalid live pair {pair['pair_id']}"
            )

        branch_a = summarize_branch(
            by_branch[key_a],
            actions,
            absolute_min,
            multiplier,
        )
        branch_b = summarize_branch(
            by_branch[key_b],
            actions,
            absolute_min,
            multiplier,
        )

        best_a = branch_a["best_action_id"]
        best_b = branch_b["best_action_id"]
        different = bool(best_a != best_b)
        pair_floor = float(
            max(
                branch_a["repeat_floor_max"],
                branch_b["repeat_floor_max"],
            )
        )
        formal_threshold = float(
            max(
                absolute_min,
                multiplier * pair_floor,
            )
        )

        if different:
            regret_a = float(
                np.min(_scores(branch_a, best_a))
                - np.max(_scores(branch_a, best_b))
            )
            regret_b = float(
                np.min(_scores(branch_b, best_b))
                - np.max(_scores(branch_b, best_a))
            )
        else:
            regret_a = 0.0
            regret_b = 0.0

        pair_pass = bool(
            branch_a["unique_best"]
            and branch_b["unique_best"]
            and different
            and regret_a >= formal_threshold
            and regret_b >= formal_threshold
        )

        live_index_a = [
            int(v) for v in live["live_winding_index_a"]
        ]
        live_index_b = [
            int(v) for v in live["live_winding_index_b"]
        ]
        live_stratum = _winding_stratum(
            live_index_a,
            live_index_b,
        )
        frozen_stratum = _winding_stratum(
            pair["winding_index_a"],
            pair["winding_index_b"],
        )

        pair_results.append(
            {
                "pair_id": pair["pair_id"],
                "source_rank": int(pair["source_rank"]),
                "rollout_a": int(pair["rollout_a"]),
                "rollout_b": int(pair["rollout_b"]),
                "time_index": t,
                "pb3b1_frozen_winding_index_a": [
                    int(v) for v in pair["winding_index_a"]
                ],
                "pb3b1_frozen_winding_index_b": [
                    int(v) for v in pair["winding_index_b"]
                ],
                "pb3b1_frozen_winding_stratum": frozen_stratum,
                "pb4_live_winding_index_a": live_index_a,
                "pb4_live_winding_index_b": live_index_b,
                "pb4_live_winding_stratum": live_stratum,
                "stratum_match": bool(live_stratum == frozen_stratum),
                "branch_a": branch_a,
                "branch_b": branch_b,
                "best_action_switch": different,
                "robust_cross_regret_a": regret_a,
                "robust_cross_regret_b": regret_b,
                "pair_repeat_floor_max": pair_floor,
                "formal_regret_threshold": formal_threshold,
                "pair_control_relevance_pass": pair_pass,
            }
        )

    passing = [
        row for row in pair_results
        if row["pair_control_relevance_pass"]
    ]
    passing_strata = sorted(
        {
            row["pb4_live_winding_stratum"]
            for row in passing
        }
    )
    confirmed = bool(
        len(passing) >= int(cr["minimum_passing_pairs"])
        and len(passing_strata)
        >= int(cr["minimum_passing_winding_strata"])
    )
    return {
        "confirmed": confirmed,
        "pair_count": len(pair_results),
        "passing_pair_count": len(passing),
        "passing_winding_strata": len(passing_strata),
        "passing_winding_stratum_labels": passing_strata,
        "winding_strata_source": "PB4 fresh live revalidation",
        "minimum_passing_pairs": int(cr["minimum_passing_pairs"]),
        "minimum_passing_winding_strata":
            int(cr["minimum_passing_winding_strata"]),
        "pairs": pair_results,
    }


def _save_score_npz(config, shortlist, action_records):
    actions = config["action_library"]["actions"]
    action_ids = [row["action_id"] for row in actions]
    scores = np.empty((10, 2, len(actions), 3), dtype=np.float64)
    survived = np.zeros((10, 2, len(actions), 3), dtype=bool)
    failure_step = np.full((10, 2, len(actions), 3), -1, dtype=np.int16)

    lookup = {
        (
            int(row["rollout_id"]),
            int(row["time_index"]),
            row["action_id"],
            int(row["repeat_index"]),
        ): row
        for row in action_records
    }
    for p_idx, pair in enumerate(shortlist["pairs"]):
        t = int(pair["time_index"])
        for side_idx, key_name in enumerate(("rollout_a", "rollout_b")):
            rollout_id = int(pair[key_name])
            for a_idx, action_id in enumerate(action_ids):
                for repeat in range(3):
                    row = lookup[(rollout_id, t, action_id, repeat)]
                    scores[p_idx, side_idx, a_idx, repeat] = float(
                        row["task_score"]
                    )
                    survived[p_idx, side_idx, a_idx, repeat] = bool(
                        row["survived_horizon"]
                    )
                    if row["failure_step"] is not None:
                        failure_step[p_idx, side_idx, a_idx, repeat] = int(
                            row["failure_step"]
                        )

    raw_root = Path(config["outputs"]["raw_root"])
    raw_root.mkdir(parents=True, exist_ok=True)
    path = raw_root / "ACTION_SCORES.npz"
    np.savez_compressed(
        path,
        scores=scores,
        survived=survived,
        failure_step=failure_step,
        action_ids=np.asarray(action_ids, dtype="<U64"),
    )
    return path


def write_outputs(
        config,
        validation,
        protocol,
        verdict,
        live_records,
        restore_records,
        action_records,
        *,
        audit=None,
        blocked=None):
    raw_root = Path(config["outputs"]["raw_root"])
    report_dir = REPO_ROOT / config["outputs"]["committed_report_dir"]
    raw_root.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "phase": "PB4",
        "verdict": verdict,
        "control_relevance_status": (
            "TESTED" if audit is not None else "UNTESTED"
        ),
        "cohort_validation": validation,
        "protocol": protocol,
        "live_pair_revalidation_records": live_records,
        "live_pair_barrier": {
            "valid": len(live_records) == 10
            and all(bool(row.get("valid")) for row in live_records),
            "valid_pair_count": int(
                sum(bool(row.get("valid")) for row in live_records)
            ),
            "required_pair_count": 10,
        },
        "snapshot_restore_records": restore_records,
        "snapshot_restore_barrier": {
            "valid": len(restore_records) == 60
            and all(bool(row.get("valid")) for row in restore_records),
            "restore_record_count": len(restore_records),
            "required_restore_record_count": 60,
        },
        "action_evaluation_record_count": len(action_records),
        "expected_action_evaluation_record_count": 240,
        "audit": audit,
        "blocked_details": blocked,
        "boundaries": {
            "pair_drop": False,
            "pair_replacement": False,
            "action_library_modified_after_gpu": False,
            "deployable_sensor_test_started": False,
            "state_diff_training_started": False,
            "cfpm_training_started": False,
        },
    }

    (raw_root / "AUDIT.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    (raw_root / "ACTION_EVALUATIONS.json").write_text(
        json.dumps(
            {
                "verdict": verdict,
                "records": action_records,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    evidence = {
        "phase_name": config["phase_name"],
        "verdict": verdict,
        "repository": {
            "starting_main_sha": config["provenance"]["starting_main_sha"],
            "ending_main_sha": git("rev-parse", "HEAD"),
            "dlolab_gitlink": git(
                "rev-parse", "HEAD:external/dlo-lab"
            ),
        },
        "scientific": result,
    }
    (report_dir / "EVIDENCE.json").write_text(
        json.dumps(evidence, indent=2) + "\n",
        encoding="utf-8",
    )
    (report_dir / "PAIR_REGRET.json").write_text(
        json.dumps(
            {
                "verdict": verdict,
                "audit": audit,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# PB4 Wrapping Control Relevance / Action Regret",
        "",
        f"Verdict: `{verdict}`",
        "",
        "## Frozen action library",
        "",
    ]
    for row in protocol["action_library"]["actions"]:
        lines.append(
            f"- {row['action_id']}: "
            f"arm1={row['arm1_mask']}, arm2={row['arm2_mask']}"
        )
    lines += [
        "",
        "## Barriers",
        "",
        (
            "- Live pair revalidation: "
            f"{result['live_pair_barrier']['valid_pair_count']}/10"
        ),
        (
            "- Snapshot restore records: "
            f"{result['snapshot_restore_barrier']['restore_record_count']}/60"
        ),
        (
            "- Action evaluation records: "
            f"{len(action_records)}/240"
        ),
    ]
    if audit is not None:
        lines += [
            "",
            "## Control relevance",
            "",
            (
                "- Passing pairs: "
                f"{audit['passing_pair_count']}/{audit['pair_count']}"
            ),
            (
                "- Passing winding strata: "
                f"{audit['passing_winding_strata']}"
            ),
            (
                "- Formal absolute regret floor: "
                f"{config['control_relevance']['absolute_regret_min']}"
            ),
        ]
    lines += [
        "",
        "## Boundary",
        "",
        (
            "PB4 establishes only control relevance under the frozen finite "
            "action library. It does not establish deployable sensing or model "
            "performance. No StateDiff/CFPM training was started."
        ),
        "",
    ]
    (report_dir / "RESULT.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    return result


def execute(config, shortlist, rule, qpos):
    env = build_env(
        n_envs=int(config["replay"]["n_envs"]),
        n_steps_sub=int(config["replay"]["n_steps_sub"]),
        log_dir=official_log_dir(),
    )
    env.init_domain_randomization(**wrapping_args)
    build_state = env.scene.get_state()

    live_records = []
    restore_records = []
    action_records = []
    side_output = None
    try:
        live, _, by_batch, n_intervals = (
            _collect_live_histories_with_prefix_validity(
                env,
                config,
                shortlist,
                qpos,
                build_state,
            )
        )
        live_records = _revalidate_pairs_with_prefix(
            shortlist,
            live,
            rule,
        )
        _live_barrier(live_records)

        sides = required_side_records(shortlist)
        side_output = {
            key: {
                "meta": row,
                "alignment": None,
                "live_history": live[key]["live_history"],
                "common_action_history": live[key]["common_action_history"],
                "snapshot_restore_errors_m": [],
                "repeats": [],
            }
            for key, row in sides.items()
        }

        snapshots, references = _capture_snapshots(
            env,
            config,
            by_batch,
            qpos,
            build_state,
            n_intervals,
        )
        if len(sides) != 20:
            raise PB4Blocked(
                SNAPSHOT_FAILED,
                {
                    "failure_component": "snapshot_capture_count",
                    "expected_branch_count": 20,
                    "actual_branch_count": len(sides),
                },
            )

        _snapshot_barrier(
            env,
            config,
            by_batch,
            snapshots,
            references,
            side_output,
            restore_records,
        )

        _evaluate_actions(
            env,
            config,
            by_batch,
            snapshots,
            qpos,
            n_intervals,
            action_records,
        )
    finally:
        env.stop()

    return live_records, restore_records, action_records


def run(config_path):
    config = load_json(config_path)
    raw, shortlist, validation, rule, pb3b2, qpos = validate_sources(config)
    protocol = load_and_validate_protocol(
        config,
        shortlist,
        validation,
        rule,
        pb3b2,
        qpos,
    )

    live_records = []
    restore_records = []
    action_records = []
    try:
        live_records, restore_records, action_records = execute(
            config,
            shortlist,
            rule,
            qpos,
        )
        audit = analyze_control_relevance(
            config,
            shortlist,
            action_records,
            live_records,
        )
        verdict = POSITIVE_VERDICT if audit["confirmed"] else NEGATIVE_VERDICT
        _save_score_npz(
            config,
            shortlist,
            action_records,
        )
        result = write_outputs(
            config,
            validation,
            protocol,
            verdict,
            live_records,
            restore_records,
            action_records,
            audit=audit,
        )
    except PB4Blocked as exc:
        result = write_outputs(
            config,
            validation,
            protocol,
            exc.verdict,
            live_records,
            restore_records,
            action_records,
            blocked=exc.details,
        )

    print(f"verdict={result['verdict']}")
    return result


def validate_action_record_scores(action_records, horizon):
    keys = set()
    for row in action_records:
        key = (
            int(row["rollout_id"]),
            int(row["time_index"]),
            row["action_id"],
            int(row["repeat_index"]),
        )
        if key in keys:
            raise RuntimeError(f"Duplicate PB4 action record key: {key}")
        keys.add(key)

        recomputed = recompute_task_score(
            row["score_contributions"],
            horizon,
        )
        if not np.isclose(
            float(row["task_score"]),
            recomputed,
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError(
                "PB4 task score does not reconstruct from score_contributions: "
                f"key={key}, stored={row['task_score']}, recomputed={recomputed}"
            )
    return len(keys)


def validate_result_boundaries(
        verdict,
        scientific,
        action_records):
    audit = scientific.get("audit")
    live_barrier = scientific["live_pair_barrier"]
    snapshot_barrier = scientific["snapshot_restore_barrier"]
    blocked = scientific.get("blocked_details")

    if verdict in (POSITIVE_VERDICT, NEGATIVE_VERDICT):
        if not live_barrier["valid"]:
            raise RuntimeError("Completed PB4 result lacks valid live barrier")
        if not snapshot_barrier["valid"]:
            raise RuntimeError("Completed PB4 result lacks valid snapshot barrier")
        if len(action_records) != 240:
            raise RuntimeError(
                f"Completed PB4 result requires 240 actions, got {len(action_records)}"
            )
        if audit is None:
            raise RuntimeError("Completed PB4 result requires non-null audit")
        if blocked is not None:
            raise RuntimeError("Completed PB4 result must not contain blocked details")
        return "complete"

    if verdict == LIVE_FAILED:
        if live_barrier["valid"]:
            raise RuntimeError("LIVE_FAILED cannot have a valid live barrier")
        if len(scientific["snapshot_restore_records"]) != 0:
            raise RuntimeError("LIVE_FAILED must not create snapshot restore records")
        if len(action_records) != 0:
            raise RuntimeError("LIVE_FAILED must not evaluate actions")
        if audit is not None:
            raise RuntimeError("LIVE_FAILED must have audit=null")
        if not blocked or blocked.get("failure_component") != "live_cohort_revalidation":
            raise RuntimeError("LIVE_FAILED blocked_details mismatch")
        return "blocked_live"

    if verdict == SNAPSHOT_FAILED:
        if not live_barrier["valid"]:
            raise RuntimeError("SNAPSHOT_FAILED requires passed live barrier")
        if snapshot_barrier["valid"]:
            raise RuntimeError("SNAPSHOT_FAILED cannot have passed snapshot barrier")
        if len(action_records) != 0:
            raise RuntimeError("SNAPSHOT_FAILED must not evaluate actions")
        if audit is not None:
            raise RuntimeError("SNAPSHOT_FAILED must have audit=null")
        if not blocked or blocked.get("failure_component") not in (
            "snapshot_restore_validation",
            "snapshot_capture_count",
        ):
            raise RuntimeError("SNAPSHOT_FAILED blocked_details mismatch")
        return "blocked_snapshot"

    if verdict == ACTION_FAILED:
        if not live_barrier["valid"]:
            raise RuntimeError("ACTION_FAILED requires passed live barrier")
        if not snapshot_barrier["valid"]:
            raise RuntimeError("ACTION_FAILED requires passed snapshot barrier")
        if audit is not None:
            raise RuntimeError("ACTION_FAILED must have audit=null")
        if not blocked:
            raise RuntimeError("ACTION_FAILED requires blocked_details")
        if len(action_records) > 240:
            raise RuntimeError("ACTION_FAILED cannot have >240 action records")
        return "blocked_action"

    raise RuntimeError(f"Unsupported PB4 formal verdict: {verdict}")


def validate_results(config_path):
    config = load_json(config_path)
    _, shortlist, validation, rule, pb3b2, qpos = validate_sources(config)
    protocol = load_and_validate_protocol(
        config,
        shortlist,
        validation,
        rule,
        pb3b2,
        qpos,
    )

    raw_root = Path(config["outputs"]["raw_root"])
    action_path = raw_root / "ACTION_EVALUATIONS.json"
    evidence_path = (
        REPO_ROOT
        / config["outputs"]["committed_report_dir"]
        / "EVIDENCE.json"
    )
    regret_path = (
        REPO_ROOT
        / config["outputs"]["committed_report_dir"]
        / "PAIR_REGRET.json"
    )
    for path in (action_path, evidence_path, regret_path):
        if not path.is_file():
            raise FileNotFoundError(str(path))

    action_payload = load_json(action_path)
    evidence = load_json(evidence_path)
    regret = load_json(regret_path)

    verdict = evidence.get("verdict")
    scientific = evidence["scientific"]
    action_records = action_payload["records"]

    if action_payload.get("verdict") != verdict:
        raise RuntimeError("PB4 raw action verdict differs from evidence")
    if regret.get("verdict") != verdict:
        raise RuntimeError("PB4 PAIR_REGRET verdict differs from evidence")
    if scientific["protocol"] != protocol:
        raise RuntimeError("PB4 evidence protocol mismatch")
    if int(scientific["action_evaluation_record_count"]) != len(action_records):
        raise RuntimeError("PB4 action record count differs from evidence")

    horizon = int(config["action_library"]["primary_horizon_microsteps"])
    validate_action_record_scores(
        action_records,
        horizon,
    )
    mode = validate_result_boundaries(
        verdict,
        scientific,
        action_records,
    )

    if mode == "complete":
        live_records = scientific["live_pair_revalidation_records"]
        expected_audit = analyze_control_relevance(
            config,
            shortlist,
            action_records,
            live_records,
        )
        if scientific["audit"] != expected_audit:
            raise RuntimeError(
                "PB4 evidence audit differs from deterministic analysis"
            )
        if regret.get("audit") != expected_audit:
            raise RuntimeError(
                "PB4 PAIR_REGRET differs from deterministic analysis"
            )
        expected_verdict = (
            POSITIVE_VERDICT
            if expected_audit["confirmed"]
            else NEGATIVE_VERDICT
        )
        if verdict != expected_verdict:
            raise RuntimeError(
                f"PB4 verdict mismatch: {verdict} != {expected_verdict}"
            )
    else:
        if regret.get("audit") is not None:
            raise RuntimeError("Blocked PB4 result must have PAIR_REGRET audit=null")
        if scientific.get("audit") is not None:
            raise RuntimeError("Blocked PB4 result must have evidence audit=null")

    print("PB4 result validation: PASS")
    print(f"verdict={verdict}")
    print(f"validation_mode={mode}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "command",
        choices=("freeze-protocol", "run", "validate-results"),
    )
    args = parser.parse_args()

    if args.command == "freeze-protocol":
        freeze_protocol(args.config)
    elif args.command == "run":
        run(args.config)
    else:
        validate_results(args.config)


if __name__ == "__main__":
    main()
