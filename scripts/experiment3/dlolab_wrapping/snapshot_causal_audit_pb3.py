"""PB3 — Wrapping snapshot same-action causal bifurcation audit.

The formal primary metric is differential deformation, not raw full-state
distance:

    delta_X(h) = X(t+h) - X(t)

For a PB2-C pair A/B we compare the ordered rope displacement fields
delta_X_A(h) and delta_X_B(h). This removes static hidden-geometry differences
at the branch point and directly tests whether the same future qpos suffix
induces different deformation dynamics.

Every formal branch state is:
- from an official-valid PB2-C rollout;
- replay-aligned to the frozen PB2-C raw rollout;
- snapshotted with pinned Genesis Scene.get_state();
- restored with Scene.reset(state=snapshot);
- repeated three times under the identical official future qpos suffix.

PB2-C partial-state thresholds are not reused as Gate 4.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import itertools
import json
from pathlib import Path
import random
import subprocess

import numpy as np
import torch

from scripts.experiment3.dlolab_wrapping.paths import (
    REPO_ROOT,
    official_log_dir,
)
from scripts.experiment3.dlolab_wrapping.replay_winding_audit import (
    _dual_arm_command,
    build_env,
    load_json,
    normalize_qpos,
    sample_env,
    to_numpy,
)
from scripts.experiment3.dlolab_wrapping.pair_discovery_pb2c import (
    batch_symmetric_chamfer,
    compute_visibility_mask,
    signed_winding_index,
)
from scripts.experiment3.dlolab_wrapping.preregister_alignment_pb3r3 import (
    live_pair_revalidation,
    targeted_replay_alignment,
    validate_generated_rule,
)
from utils.domain_randomization import wrapping_args  # noqa: E402


@dataclass
class PB3Blocked(Exception):
    verdict: str
    details: dict


def git(*args):
    return subprocess.check_output(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
    ).strip()


def _jsonable(value):
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {
            key: _jsonable(v)
            for key, v in value.items()
        }
    return value


def seed_everything(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def winding_stratum(index_a, index_b):
    a = ",".join(str(int(v)) for v in index_a)
    b = ",".join(str(int(v)) for v in index_b)
    return "<->".join(sorted((a, b)))


def derive_rank_order_greedy_shortlist(candidates, target_pair_count):
    """Recompute the frozen shortlist directly from PB2-C rank order.

    Rule:
      scan top_candidates in stored order;
      accept iff neither rollout has appeared in a previously accepted pair;
      stop after target_pair_count accepted pairs.

    No shortlist file fields are consulted here.
    """
    top = candidates.get("top_candidates", [])
    used_rollouts = set()
    expected = []

    for source_rank, row in enumerate(top, start=1):
        rollout_a = int(row["rollout_a"])
        rollout_b = int(row["rollout_b"])

        if (
            rollout_a in used_rollouts
            or rollout_b in used_rollouts
        ):
            continue

        expected.append(
            {
                "source_rank": int(source_rank),
                "rollout_a": rollout_a,
                "rollout_b": rollout_b,
                "time_index": int(row["time_index"]),
                "winding_index_a": [
                    int(v)
                    for v in row["winding_index_a"]
                ],
                "winding_index_b": [
                    int(v)
                    for v in row["winding_index_b"]
                ],
                "differing_post_indices": [
                    int(v)
                    for v in row["differing_post_indices"]
                ],
                "visible_rope_history_chamfer_m":
                    float(row["visible_rope_history_chamfer_m"]),
                "pair_max_winding_integer_residual":
                    float(row["pair_max_winding_integer_residual"]),
                "replay_a": {
                    "batch_index":
                        int(row["replay_a"]["batch_index"]),
                    "env_index":
                        int(row["replay_a"]["env_index"]),
                    "seed":
                        int(row["replay_a"]["seed"]),
                },
                "replay_b": {
                    "batch_index":
                        int(row["replay_b"]["batch_index"]),
                    "env_index":
                        int(row["replay_b"]["env_index"]),
                    "seed":
                        int(row["replay_b"]["seed"]),
                },
            }
        )

        used_rollouts.update(
            (rollout_a, rollout_b)
        )

        if len(expected) == int(target_pair_count):
            break

    if len(expected) != int(target_pair_count):
        raise RuntimeError(
            "PB2-C top_candidates do not contain enough rollout-disjoint "
            f"pairs for target_pair_count={target_pair_count}; "
            f"derived={len(expected)}"
        )

    return expected


def _normalized_shortlist_pair(pair):
    return {
        "source_rank":
            int(pair["source_rank"]),
        "rollout_a":
            int(pair["rollout_a"]),
        "rollout_b":
            int(pair["rollout_b"]),
        "time_index":
            int(pair["time_index"]),
        "winding_index_a":
            [int(v) for v in pair["winding_index_a"]],
        "winding_index_b":
            [int(v) for v in pair["winding_index_b"]],
        "differing_post_indices":
            [int(v) for v in pair["differing_post_indices"]],
        "visible_rope_history_chamfer_m":
            float(pair["visible_rope_history_chamfer_m"]),
        "pair_max_winding_integer_residual":
            float(pair["pair_max_winding_integer_residual"]),
        "replay_a": {
            "batch_index":
                int(pair["replay_a"]["batch_index"]),
            "env_index":
                int(pair["replay_a"]["env_index"]),
            "seed":
                int(pair["replay_a"]["seed"]),
        },
        "replay_b": {
            "batch_index":
                int(pair["replay_b"]["batch_index"]),
            "env_index":
                int(pair["replay_b"]["env_index"]),
            "seed":
                int(pair["replay_b"]["seed"]),
        },
    }


def validate_shortlist_against_pb2c(shortlist, candidates):
    selection_rule = shortlist.get("selection_rule", {})

    if bool(selection_rule.get("future_information_used")):
        raise RuntimeError(
            "PB3 shortlist must not use future information"
        )

    target_pair_count = int(
        selection_rule.get(
            "target_pair_count",
            -1,
        )
    )
    if target_pair_count != 10:
        raise RuntimeError(
            "PB3 formal target_pair_count must remain 10"
        )

    if int(
        selection_rule.get(
            "max_rollout_use_count",
            -1,
        )
    ) != 1:
        raise RuntimeError(
            "PB3 formal max_rollout_use_count must remain 1"
        )

    expected = derive_rank_order_greedy_shortlist(
        candidates,
        target_pair_count,
    )

    actual_pairs = shortlist.get("pairs", [])
    if len(actual_pairs) != target_pair_count:
        raise RuntimeError(
            "PB3 shortlist pair count does not match frozen target"
        )

    actual = [
        _normalized_shortlist_pair(pair)
        for pair in actual_pairs
    ]

    # This is the key invariant: the shortlist must equal the result of
    # replaying the rank-order greedy scan from the committed PB2-C list.
    # Monotonic source ranks or membership alone are insufficient.
    for index, (actual_row, expected_row) in enumerate(
            zip(actual, expected)):
        if actual_row != expected_row:
            raise RuntimeError(
                "PB3 shortlist does not equal the frozen PB2-C rank-order "
                "greedy scan at accepted-pair index "
                f"{index}: actual={actual_row}, expected={expected_row}"
            )

    used = [
        rollout_id
        for row in actual
        for rollout_id in (
            row["rollout_a"],
            row["rollout_b"],
        )
    ]
    if len(used) != 20 or len(set(used)) != 20:
        raise RuntimeError(
            "PB3 formal shortlist must contain 10 pairs / 20 unique rollouts"
        )

    expected_ranks = [
        row["source_rank"]
        for row in expected
    ]
    actual_ranks = [
        row["source_rank"]
        for row in actual
    ]

    return {
        "rank_order_scan_verified": True,
        "derived_source_ranks": expected_ranks,
        "actual_source_ranks": actual_ranks,
        "target_pair_count": target_pair_count,
        "unique_rollout_count": len(set(used)),
    }


def validate_sources(config):
    evidence_path = REPO_ROOT / config["source"]["pb2c_evidence"]
    candidate_path = REPO_ROOT / config["source"]["pb2c_candidates"]
    shortlist_path = REPO_ROOT / config["source"]["shortlist"]
    raw_path = Path(config["source"]["pb2c_raw_rollouts"])

    for path in (
        evidence_path,
        candidate_path,
        shortlist_path,
        raw_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(str(path))

    evidence = load_json(evidence_path)
    expected_verdict = config["source"]["expected_pb2c_verdict"]
    if evidence.get("verdict") != expected_verdict:
        raise RuntimeError(
            "PB2-C verdict mismatch: "
            f"{evidence.get('verdict')} != {expected_verdict}"
        )

    candidates = load_json(candidate_path)
    shortlist = load_json(shortlist_path)
    shortlist_validation = validate_shortlist_against_pb2c(
        shortlist,
        candidates,
    )

    actual_args = _jsonable(dict(wrapping_args))
    expected_args = config["replay"]["expected_wrapping_args"]
    if actual_args != expected_args:
        raise RuntimeError(
            f"Pinned wrapping_args mismatch: {actual_args} != {expected_args}"
        )

    result = {
        "evidence": evidence,
        "candidates": candidates,
        "shortlist": shortlist,
        "shortlist_validation": shortlist_validation,
        "raw_path": raw_path,
    }

    r3_source = config["source"].get("pb3r3")
    if r3_source is not None:
        r3_config_path = REPO_ROOT / r3_source["config"]
        r3_rule_path = REPO_ROOT / r3_source["alignment_rule"]
        r3_evidence_path = REPO_ROOT / r3_source["evidence"]
        for path in (r3_config_path, r3_rule_path, r3_evidence_path):
            if not path.is_file():
                raise FileNotFoundError(str(path))

        # Recompute the independent PB3-R2 state envelope before accepting
        # the committed R3 rule for a formal Resume.
        validate_generated_rule(r3_config_path)
        rule = load_json(r3_rule_path)
        r3_evidence = load_json(r3_evidence_path)
        expected_r3_verdict = r3_source["expected_verdict"]
        if (
            rule.get("verdict") != expected_r3_verdict
            or r3_evidence.get("verdict") != expected_r3_verdict
        ):
            raise RuntimeError("PB3-R3 rule/evidence verdict mismatch")

        threshold = float(
            rule["targeted_replay_engineering_alignment"]["rope"][
                "threshold_m"
            ]
        )
        evidence_threshold = float(
            r3_evidence["scientific"][
                "derived_rope_coordinate_rmse_threshold_m"
            ]
        )
        if not np.isclose(
            threshold,
            evidence_threshold,
            rtol=1e-12,
            atol=1e-15,
        ):
            raise RuntimeError("PB3-R3 rule/evidence threshold mismatch")

        if rule["live_pair_revalidation"].get("allow_pair_drop"):
            raise RuntimeError("PB3 Resume must not allow pair dropping")
        if rule["live_pair_revalidation"].get("allow_pair_replacement"):
            raise RuntimeError("PB3 Resume must not allow pair replacement")
        if not rule["live_pair_revalidation"].get(
            "all_10_formal_pairs_required"
        ):
            raise RuntimeError("PB3 Resume requires all 10 formal pairs")
        if not rule["targeted_replay_engineering_alignment"].get(
            "all_20_formal_branch_states_required"
        ):
            raise RuntimeError("PB3 Resume requires all 20 branch states")

        frozen_gate = rule["gate4"]
        for key in (
            "absolute_effect_min_m",
            "repeat_floor_multiplier",
            "minimum_passing_pairs",
            "minimum_passing_winding_strata",
        ):
            if float(frozen_gate[key]) != float(config["gate4"][key]):
                raise RuntimeError(f"PB3-R3 Gate 4 mismatch at {key}")

        result["r3_rule"] = rule
        result["r3_evidence"] = r3_evidence

    return result


def load_frozen_rollouts(path):
    with np.load(path) as handle:
        data = {
            key: np.asarray(handle[key])
            for key in handle.files
        }

    required = {
        "rollout_id",
        "batch_index",
        "env_index",
        "batch_seed",
        "official_rollout_valid",
        "rope_xyz",
        "ee1_pos",
        "ee2_pos",
        "motor_qpos_1",
        "motor_qpos_2",
        "signed_winding_turns",
        "common_qpos_replay",
    }
    missing = sorted(required - set(data))
    if missing:
        raise RuntimeError(
            f"Frozen PB2-C rollout file missing fields: {missing}"
        )

    return data


def frozen_index_by_rollout(data):
    ids = [int(v) for v in data["rollout_id"]]
    if len(set(ids)) != len(ids):
        raise RuntimeError("Frozen PB2-C rollout IDs are not unique")
    return {
        rollout_id: index
        for index, rollout_id in enumerate(ids)
    }


def extract_env_sample(sample, env_index):
    keys = (
        "rope_xyz",
        "post_xyz",
        "ee1_pos",
        "ee1_quat",
        "ee2_pos",
        "ee2_quat",
        "motor_qpos_1",
        "motor_qpos_2",
        "reward",
        "signed_winding_turns",
    )
    return {
        key: np.asarray(sample[key][env_index]).copy()
        for key in keys
    }


def compare_live_to_frozen(
        live,
        frozen,
        frozen_index,
        time_index,
        config):
    align = config["alignment"]

    rope_error = float(
        np.max(
            np.abs(
                np.asarray(live["rope_xyz"])
                - frozen["rope_xyz"][
                    frozen_index,
                    time_index,
                ]
            )
        )
    )

    ee_error = float(
        max(
            np.max(
                np.abs(
                    np.asarray(live["ee1_pos"])
                    - frozen["ee1_pos"][
                        frozen_index,
                        time_index,
                    ]
                )
            ),
            np.max(
                np.abs(
                    np.asarray(live["ee2_pos"])
                    - frozen["ee2_pos"][
                        frozen_index,
                        time_index,
                    ]
                )
            ),
        )
    )

    qpos_error = float(
        max(
            np.max(
                np.abs(
                    np.asarray(live["motor_qpos_1"])
                    - frozen["motor_qpos_1"][
                        frozen_index,
                        time_index,
                    ]
                )
            ),
            np.max(
                np.abs(
                    np.asarray(live["motor_qpos_2"])
                    - frozen["motor_qpos_2"][
                        frozen_index,
                        time_index,
                    ]
                )
            ),
        )
    )

    live_index = signed_winding_index(
        np.asarray(live["signed_winding_turns"])
    ).tolist()

    frozen_winding_index = signed_winding_index(
        frozen["signed_winding_turns"][
            frozen_index,
            time_index,
        ]
    ).tolist()

    valid = (
        rope_error <= float(align["rope_max_abs_m"])
        and ee_error <= float(align["ee_max_abs_m"])
        and qpos_error
        <= float(align["motor_qpos_max_abs_rad"])
    )

    if align["require_winding_index_exact"]:
        valid = (
            valid
            and live_index == frozen_winding_index
        )

    return {
        "valid": bool(valid),
        "rope_max_abs_m": rope_error,
        "ee_max_abs_m": ee_error,
        "motor_qpos_max_abs_rad": qpos_error,
        "live_winding_index": live_index,
        "frozen_winding_index": frozen_winding_index,
    }


def ordered_displacement_field_rmse(
        a0,
        ah,
        b0,
        bh):
    a0 = np.asarray(a0, dtype=np.float64)
    ah = np.asarray(ah, dtype=np.float64)
    b0 = np.asarray(b0, dtype=np.float64)
    bh = np.asarray(bh, dtype=np.float64)

    delta = (
        (ah - a0)
        - (bh - b0)
    )
    return float(
        np.sqrt(
            np.mean(
                np.sum(
                    delta * delta,
                    axis=-1,
                )
            )
        )
    )


def ordered_position_rmse(a, b):
    delta = (
        np.asarray(a, dtype=np.float64)
        - np.asarray(b, dtype=np.float64)
    )
    return float(
        np.sqrt(
            np.mean(
                np.sum(
                    delta * delta,
                    axis=-1,
                )
            )
        )
    )


def partial_chamfer_m(
        rope_a,
        post_a,
        rope_b,
        post_b,
        radius_m):
    rope_a = np.asarray(rope_a, dtype=np.float32)
    rope_b = np.asarray(rope_b, dtype=np.float32)
    post_a = np.asarray(post_a, dtype=np.float32)
    post_b = np.asarray(post_b, dtype=np.float32)

    mask_a = compute_visibility_mask(
        rope_a[None, None],
        post_a[None, None],
        radius_m,
    )[0, 0]

    mask_b = compute_visibility_mask(
        rope_b[None, None],
        post_b[None, None],
        radius_m,
    )[0, 0]

    if not mask_a.any() or not mask_b.any():
        return None

    value = batch_symmetric_chamfer(
        rope_a[None],
        mask_a[None],
        rope_b[None],
        mask_b[None],
    )
    return float(value[0])


def gate4_threshold(
        repeat_floor,
        *,
        absolute_effect_min_m,
        repeat_floor_multiplier):
    return float(
        max(
            float(absolute_effect_min_m),
            float(repeat_floor_multiplier)
            * float(repeat_floor),
        )
    )


def pairwise_repeat_floor(repeat_samples, horizon):
    values = []
    for i, j in itertools.combinations(
            range(len(repeat_samples)),
            2):
        values.append(
            ordered_displacement_field_rmse(
                repeat_samples[i][0]["rope_xyz"],
                repeat_samples[i][horizon]["rope_xyz"],
                repeat_samples[j][0]["rope_xyz"],
                repeat_samples[j][horizon]["rope_xyz"],
            )
        )

    return (
        0.0
        if not values
        else float(max(values))
    )


def cross_branch_values(
        repeats_a,
        repeats_b,
        horizon):
    values = []
    for repeat_a in repeats_a:
        for repeat_b in repeats_b:
            values.append(
                ordered_displacement_field_rmse(
                    repeat_a[0]["rope_xyz"],
                    repeat_a[horizon]["rope_xyz"],
                    repeat_b[0]["rope_xyz"],
                    repeat_b[horizon]["rope_xyz"],
                )
            )
    return np.asarray(values, dtype=np.float64)


def cross_partial_chamfer_values(
        repeats_a,
        repeats_b,
        horizon,
        radius_m):
    values = []
    for repeat_a in repeats_a:
        for repeat_b in repeats_b:
            value = partial_chamfer_m(
                repeat_a[horizon]["rope_xyz"],
                repeat_a[horizon]["post_xyz"],
                repeat_b[horizon]["rope_xyz"],
                repeat_b[horizon]["post_xyz"],
                radius_m,
            )
            if value is not None:
                values.append(value)
    return np.asarray(values, dtype=np.float64)


def required_side_records(shortlist):
    rows = {}
    for pair in shortlist["pairs"]:
        time_index = int(pair["time_index"])
        for side in ("a", "b"):
            rollout_id = int(pair[f"rollout_{side}"])
            replay = pair[f"replay_{side}"]
            key = (rollout_id, time_index)
            rows[key] = {
                "rollout_id": rollout_id,
                "time_index": time_index,
                "batch_index": int(replay["batch_index"]),
                "env_index": int(replay["env_index"]),
                "seed": int(replay["seed"]),
            }
    return rows


def group_sides_by_batch_time(side_records):
    groups = {}
    for key, row in side_records.items():
        group_key = (
            int(row["batch_index"]),
            int(row["time_index"]),
        )
        groups.setdefault(
            group_key,
            [],
        ).append(
            (key, row)
        )
    return groups


def run_snapshot_replays(
        config,
        shortlist,
        frozen,
        qpos):
    side_records = required_side_records(
        shortlist
    )
    groups = group_sides_by_batch_time(
        side_records
    )

    frozen_lookup = frozen_index_by_rollout(
        frozen
    )

    n_envs = int(
        config["replay"]["n_envs"]
    )

    env = build_env(
        n_envs=n_envs,
        n_steps_sub=config["replay"][
            "n_steps_sub"
        ],
        log_dir=official_log_dir(),
    )

    env.init_domain_randomization(
        **wrapping_args
    )

    # Pinned Genesis reset(state=...) updates the scene's registered initial
    # state. Preserve the pristine built state so every new PB2-C batch starts
    # from the same base before repository position randomization is applied.
    build_state = env.scene.get_state()

    n_intervals = (
        env.steps_interval
        // env._cmaes_n_steps_sub
    )
    if n_intervals <= 0:
        env.stop()
        raise RuntimeError(
            "Invalid Wrapping replay interval count"
        )

    horizons = [
        int(v)
        for v in config["replay"][
            "horizons"
        ]
    ]
    max_horizon = max(horizons)
    repeat_count = int(
        config["replay"][
            "snapshot_repeat_count"
        ]
    )

    side_output = {
        key: {
            "meta": row,
            "alignment": None,
            "snapshot_restore_errors_m": [],
            "repeats": [],
        }
        for key, row in side_records.items()
    }

    alignment_records = []

    groups_by_batch = {}
    for (
            batch_index,
            time_index), members in groups.items():
        groups_by_batch.setdefault(
            batch_index,
            {},
        )[time_index] = members

    try:
        for batch_index in sorted(
                groups_by_batch):
            batch_groups = groups_by_batch[
                batch_index
            ]

            env.scene.reset(
                state=build_state
            )

            seed = (
                int(
                    config["replay"][
                        "base_seed"
                    ]
                )
                + int(batch_index)
            )

            seed_everything(seed)
            env.use_qpos = True
            env.reset()

            target_times = sorted(
                int(v)
                for v in batch_groups
            )
            max_target = max(target_times)

            snapshots = {}
            branch_samples = {}

            if 0 in target_times:
                branch_samples[0] = sample_env(
                    env
                )
                snapshots[0] = (
                    env.scene.get_state()
                )

            for step_index in range(
                    1,
                    max_target + 1):
                _dual_arm_command(
                    env,
                    qpos[step_index],
                )
                for _ in range(
                        n_intervals):
                    env.scene.step()

                if step_index in batch_groups:
                    branch_samples[
                        step_index
                    ] = sample_env(
                        env
                    )
                    snapshots[
                        step_index
                    ] = env.scene.get_state()

            # Alignment is a precondition for causal interpretation.
            for time_index in target_times:
                live_batch = branch_samples[
                    time_index
                ]

                for key, row in (
                        batch_groups[
                            time_index
                        ]):
                    frozen_index = (
                        frozen_lookup[
                            int(
                                row[
                                    "rollout_id"
                                ]
                            )
                        ]
                    )

                    if not bool(
                        frozen[
                            "official_rollout_valid"
                        ][
                            frozen_index
                        ]
                    ):
                        raise RuntimeError(
                            "PB3 shortlist contains a frozen "
                            "official-invalid rollout"
                        )

                    live = extract_env_sample(
                        live_batch,
                        int(
                            row[
                                "env_index"
                            ]
                        ),
                    )

                    alignment = (
                        compare_live_to_frozen(
                            live,
                            frozen,
                            frozen_index,
                            time_index,
                            config,
                        )
                    )

                    side_output[
                        key
                    ][
                        "alignment"
                    ] = alignment

                    alignment_records.append(
                        {
                            "rollout_id":
                                int(
                                    row[
                                        "rollout_id"
                                    ]
                                ),
                            "time_index":
                                int(time_index),
                            **alignment,
                        }
                    )

                    if not alignment[
                        "valid"
                    ]:
                        raise PB3Blocked(
                            "PB3_TARGETED_REPLAY_ALIGNMENT_FAILED",
                            {
                                "failed_alignment":
                                    alignment_records[
                                        -1
                                    ]
                            },
                        )

            # Snapshot suffix repeats are run only after all branch states in
            # this batch have aligned to the frozen PB2-C acquisition.
            for time_index in target_times:
                snapshot = snapshots[
                    time_index
                ]

                reference_batch = (
                    branch_samples[
                        time_index
                    ]
                )

                members = batch_groups[
                    time_index
                ]

                for repeat_index in range(
                        repeat_count):
                    env.scene.reset(
                        state=snapshot
                    )

                    restored_batch = (
                        sample_env(
                            env
                        )
                    )

                    repeat_store = {}
                    suffix_valid = {
                        key: True
                        for key, _ in members
                    }

                    for key, row in members:
                        env_index = int(
                            row["env_index"]
                        )

                        reference = (
                            extract_env_sample(
                                reference_batch,
                                env_index,
                            )
                        )
                        restored = (
                            extract_env_sample(
                                restored_batch,
                                env_index,
                            )
                        )

                        restore_error = float(
                            np.max(
                                np.abs(
                                    restored[
                                        "rope_xyz"
                                    ]
                                    - reference[
                                        "rope_xyz"
                                    ]
                                )
                            )
                        )

                        side_output[
                            key
                        ][
                            "snapshot_restore_errors_m"
                        ].append(
                            restore_error
                        )

                        if (
                            restore_error
                            > float(
                                config[
                                    "alignment"
                                ][
                                    "snapshot_restore_rope_max_abs_m"
                                ]
                            )
                        ):
                            raise PB3Blocked(
                                "PB3_SNAPSHOT_RESTORE_ALIGNMENT_FAILED",
                                {
                                    "rollout_id":
                                        int(
                                            row[
                                                "rollout_id"
                                            ]
                                        ),
                                    "time_index":
                                        int(
                                            time_index
                                        ),
                                    "repeat_index":
                                        int(
                                            repeat_index
                                        ),
                                    "rope_max_abs_m":
                                        restore_error,
                                },
                            )

                        repeat_store[
                            key
                        ] = {
                            0: restored
                        }

                    for relative_step in range(
                            1,
                            max_horizon + 1):
                        command_index = (
                            int(time_index)
                            + relative_step
                        )

                        if command_index >= len(
                                qpos):
                            raise RuntimeError(
                                "PB3 horizon exceeds best_qpos "
                                f"length: {command_index}"
                            )

                        _dual_arm_command(
                            env,
                            qpos[
                                command_index
                            ],
                        )
                        for _ in range(
                                n_intervals):
                            env.scene.step()

                        rope_now = np.asarray(
                            env.rope.get_all_verts()
                        )
                        dist_now = to_numpy(
                            env.rope.get_geodesic_distance(
                                env.control_idx[0],
                                env.control_idx[1],
                            )
                        ).reshape(
                            env.n_envs
                        )

                        stretch_ratio = (
                            dist_now
                            / to_numpy(
                                env.control_dist_init
                            ).reshape(
                                env.n_envs
                            )
                        )

                        for key, row in members:
                            env_index = int(
                                row["env_index"]
                            )

                            if (
                                np.isnan(
                                    rope_now[
                                        env_index
                                    ]
                                ).any()
                                or stretch_ratio[
                                    env_index
                                ]
                                > float(
                                    config[
                                        "replay"
                                    ][
                                        "official_stretch_ratio_limit"
                                    ]
                                )
                            ):
                                suffix_valid[
                                    key
                                ] = False

                        if relative_step in horizons:
                            sampled = sample_env(
                                env
                            )

                            for key, row in members:
                                repeat_store[
                                    key
                                ][
                                    relative_step
                                ] = (
                                    extract_env_sample(
                                        sampled,
                                        int(
                                            row[
                                                "env_index"
                                            ]
                                        ),
                                    )
                                )

                    for key, row in members:
                        if not suffix_valid[
                            key
                        ]:
                            raise PB3Blocked(
                                "PB3_REPEAT_SUFFIX_INVALID",
                                {
                                    "rollout_id":
                                        int(
                                            row[
                                                "rollout_id"
                                            ]
                                        ),
                                    "time_index":
                                        int(
                                            time_index
                                        ),
                                    "repeat_index":
                                        int(
                                            repeat_index
                                        ),
                                },
                            )

                        side_output[
                            key
                        ][
                            "repeats"
                        ].append(
                            repeat_store[
                                key
                            ]
                        )

    finally:
        env.stop()

    return (
        side_output,
        alignment_records,
    )


def enforce_targeted_alignment_barrier(alignment_records):
    """Require all 20 formal branch states before any future suffix."""
    keys = {
        (int(row["rollout_id"]), int(row["time_index"]))
        for row in alignment_records
    }
    failures = [row for row in alignment_records if not row.get("valid", False)]
    if len(alignment_records) != 20 or len(keys) != 20 or failures:
        raise PB3Blocked(
            "PB3_R3_LIVE_BRANCH_ALIGNMENT_FAILED",
            {
                "failure_component": "targeted_replay_alignment",
                "expected_branch_state_count": 20,
                "actual_branch_state_count": len(alignment_records),
                "unique_branch_state_count": len(keys),
                "failed_alignments": failures,
                "future_suffix_executed": False,
            },
        )


def enforce_live_pair_barrier(pair_records):
    """Require all 10 frozen formal pairs before any snapshot/future work."""
    pair_ids = [row["pair_id"] for row in pair_records]
    failures = [row for row in pair_records if not row.get("valid", False)]
    if len(pair_records) != 10 or len(set(pair_ids)) != 10 or failures:
        raise PB3Blocked(
            "PB3_R3_LIVE_BRANCH_ALIGNMENT_FAILED",
            {
                "failure_component": "live_pair_revalidation",
                "expected_pair_count": 10,
                "actual_pair_count": len(pair_records),
                "unique_pair_count": len(set(pair_ids)),
                "failed_pairs": failures,
                "future_suffix_executed": False,
            },
        )


def _frozen_branch_sample(frozen, frozen_index, time_index):
    keys = (
        "rope_xyz",
        "ee1_pos",
        "ee2_pos",
        "motor_qpos_1",
        "motor_qpos_2",
        "signed_winding_turns",
    )
    return {
        key: np.asarray(frozen[key][frozen_index, time_index]).copy()
        for key in keys
    }


def run_snapshot_replays_r3(
        config,
        shortlist,
        frozen,
        qpos,
        rule,
        alignment_records=None,
        pair_revalidation_records=None):
    """Run PB3 Resume with a strict all-branches/all-pairs future barrier."""
    side_records = required_side_records(shortlist)
    groups = group_sides_by_batch_time(side_records)
    frozen_lookup = frozen_index_by_rollout(frozen)
    n_envs = int(config["replay"]["n_envs"])

    env = build_env(
        n_envs=n_envs,
        n_steps_sub=config["replay"]["n_steps_sub"],
        log_dir=official_log_dir(),
    )
    env.init_domain_randomization(**wrapping_args)
    build_state = env.scene.get_state()

    n_intervals = env.steps_interval // env._cmaes_n_steps_sub
    if n_intervals <= 0:
        env.stop()
        raise RuntimeError("Invalid Wrapping replay interval count")

    horizons = [int(v) for v in config["replay"]["horizons"]]
    max_horizon = max(horizons)
    repeat_count = int(config["replay"]["snapshot_repeat_count"])

    side_output = {
        key: {
            "meta": row,
            "alignment": None,
            "live_history": None,
            "common_action_history": None,
            "snapshot_restore_errors_m": [],
            "repeats": [],
        }
        for key, row in side_records.items()
    }
    if alignment_records is None:
        alignment_records = []
    if pair_revalidation_records is None:
        pair_revalidation_records = []
    groups_by_batch = {}
    for (batch_index, time_index), members in groups.items():
        groups_by_batch.setdefault(batch_index, {})[time_index] = members

    try:
        # Phase 1: replay every required formal branch history.  No snapshot
        # restore or future command is executed in this phase.
        for batch_index in sorted(groups_by_batch):
            batch_groups = groups_by_batch[batch_index]
            env.scene.reset(state=build_state)
            seed = int(config["replay"]["base_seed"]) + int(batch_index)
            seed_everything(seed)
            env.use_qpos = True
            env.reset()

            target_times = sorted(int(v) for v in batch_groups)
            if min(target_times) < 2:
                raise RuntimeError("PB3 live pair history requires t >= 2")
            history_times = {
                step
                for time_index in target_times
                for step in (time_index - 2, time_index - 1, time_index)
            }
            max_target = max(target_times)
            sampled_batches = {}
            if 0 in history_times:
                sampled_batches[0] = sample_env(env)

            for step_index in range(1, max_target + 1):
                _dual_arm_command(env, qpos[step_index])
                for _ in range(n_intervals):
                    env.scene.step()
                if step_index in history_times:
                    sampled_batches[step_index] = sample_env(env)

            for time_index in target_times:
                members = batch_groups[time_index]
                reference_batch = sampled_batches[time_index]
                for key, row in members:
                    rollout_id = int(row["rollout_id"])
                    frozen_index = frozen_lookup[rollout_id]
                    if not bool(frozen["official_rollout_valid"][frozen_index]):
                        raise RuntimeError(
                            "PB3 shortlist contains a frozen official-invalid rollout"
                        )

                    env_index = int(row["env_index"])
                    live = extract_env_sample(reference_batch, env_index)
                    frozen_sample = _frozen_branch_sample(
                        frozen,
                        frozen_index,
                        time_index,
                    )
                    alignment = targeted_replay_alignment(
                        live,
                        frozen_sample,
                        rule,
                    )
                    side_output[key]["alignment"] = alignment
                    side_output[key]["live_history"] = [
                        extract_env_sample(sampled_batches[step], env_index)
                        for step in (time_index - 2, time_index - 1, time_index)
                    ]
                    side_output[key]["common_action_history"] = np.asarray(
                        qpos[time_index - 2: time_index + 1]
                    ).copy()
                    alignment_records.append(
                        {
                            "rollout_id": rollout_id,
                            "time_index": int(time_index),
                            **alignment,
                        }
                    )

        # This barrier is evaluated only after all 20 states were attempted.
        enforce_targeted_alignment_barrier(alignment_records)

        # Phase 2: re-evaluate every frozen formal pair on the actual live
        # three-frame histories.  Still no future command has been executed.
        for pair in shortlist["pairs"]:
            time_index = int(pair["time_index"])
            key_a = (int(pair["rollout_a"]), time_index)
            key_b = (int(pair["rollout_b"]), time_index)
            action_a = side_output[key_a]["common_action_history"]
            action_b = side_output[key_b]["common_action_history"]
            pair_result = live_pair_revalidation(
                side_output[key_a]["live_history"],
                side_output[key_b]["live_history"],
                rule,
                same_time=(
                    side_output[key_a]["meta"]["time_index"]
                    == side_output[key_b]["meta"]["time_index"]
                    == time_index
                ),
                common_action_history_equal=np.array_equal(action_a, action_b),
            )
            pair_revalidation_records.append(
                {
                    "pair_id": pair["pair_id"],
                    "rollout_a": int(pair["rollout_a"]),
                    "rollout_b": int(pair["rollout_b"]),
                    "time_index": time_index,
                    **pair_result,
                }
            )

        enforce_live_pair_barrier(pair_revalidation_records)

        # Phase 3: only after both complete pre-future barriers pass do we
        # replay the frozen recipe again to each branch time, create the
        # snapshot, and execute its identical future suffix.  This second
        # deterministic pass avoids creating or retaining any snapshot before
        # the global 20-state/10-pair admission decision.
        for batch_index in sorted(groups_by_batch):
            batch_groups = groups_by_batch[batch_index]
            env.scene.reset(state=build_state)
            seed = int(config["replay"]["base_seed"]) + int(batch_index)
            seed_everything(seed)
            env.use_qpos = True
            env.reset()

            target_times = sorted(int(v) for v in batch_groups)
            max_target = max(target_times)
            for step_index in range(1, max_target + 1):
                _dual_arm_command(env, qpos[step_index])
                for _ in range(n_intervals):
                    env.scene.step()
                if step_index not in batch_groups:
                    continue

                time_index = int(step_index)
                members = batch_groups[time_index]
                reference_batch = sample_env(env)
                snapshot = env.scene.get_state()

                for repeat_index in range(repeat_count):
                    env.scene.reset(state=snapshot)
                    restored_batch = sample_env(env)
                    repeat_store = {}
                    suffix_valid = {key: True for key, _ in members}

                    for key, row in members:
                        env_index = int(row["env_index"])
                        reference = extract_env_sample(
                            reference_batch,
                            env_index,
                        )
                        restored = extract_env_sample(
                            restored_batch,
                            env_index,
                        )
                        restore_error = float(
                            np.max(
                                np.abs(
                                    restored["rope_xyz"]
                                    - reference["rope_xyz"]
                                )
                            )
                        )
                        side_output[key]["snapshot_restore_errors_m"].append(
                            restore_error
                        )
                        if restore_error > float(
                            rule["snapshot_restore_alignment"]["threshold_m"]
                        ):
                            raise PB3Blocked(
                                "PB3_SNAPSHOT_RESTORE_ALIGNMENT_FAILED",
                                {
                                    "rollout_id": int(row["rollout_id"]),
                                    "time_index": time_index,
                                    "repeat_index": int(repeat_index),
                                    "rope_max_abs_m": restore_error,
                                },
                            )
                        repeat_store[key] = {0: restored}

                    for relative_step in range(1, max_horizon + 1):
                        command_index = time_index + relative_step
                        if command_index >= len(qpos):
                            raise RuntimeError(
                                "PB3 horizon exceeds best_qpos length: "
                                f"{command_index}"
                            )
                        _dual_arm_command(env, qpos[command_index])
                        for _ in range(n_intervals):
                            env.scene.step()

                        rope_now = np.asarray(env.rope.get_all_verts())
                        dist_now = to_numpy(
                            env.rope.get_geodesic_distance(
                                env.control_idx[0],
                                env.control_idx[1],
                            )
                        ).reshape(env.n_envs)
                        stretch_ratio = dist_now / to_numpy(
                            env.control_dist_init
                        ).reshape(env.n_envs)
                        for key, row in members:
                            env_index = int(row["env_index"])
                            if (
                                np.isnan(rope_now[env_index]).any()
                                or stretch_ratio[env_index]
                                > float(
                                    config["replay"][
                                        "official_stretch_ratio_limit"
                                    ]
                                )
                            ):
                                suffix_valid[key] = False

                        if relative_step in horizons:
                            sampled = sample_env(env)
                            for key, row in members:
                                repeat_store[key][relative_step] = (
                                    extract_env_sample(
                                        sampled,
                                        int(row["env_index"]),
                                    )
                                )

                    for key, row in members:
                        if not suffix_valid[key]:
                            raise PB3Blocked(
                                "PB3_REPEAT_SUFFIX_INVALID",
                                {
                                    "rollout_id": int(row["rollout_id"]),
                                    "time_index": time_index,
                                    "repeat_index": int(repeat_index),
                                },
                            )
                        side_output[key]["repeats"].append(repeat_store[key])

                # Resume the canonical branch-history trajectory before
                # advancing to a later branch time in the same batch.
                env.scene.reset(state=snapshot)

    finally:
        env.stop()

    return side_output, alignment_records, pair_revalidation_records


def analyze_pairs(
        config,
        shortlist,
        side_output):
    horizons = [
        int(v)
        for v in config[
            "replay"
        ][
            "horizons"
        ]
    ]

    gate = config["gate4"]
    radius = float(
        config["diagnostics"][
            "partial_observation_occlusion_radius_m"
        ]
    )

    pair_rows = []

    for pair in shortlist["pairs"]:
        time_index = int(
            pair["time_index"]
        )

        key_a = (
            int(
                pair[
                    "rollout_a"
                ]
            ),
            time_index,
        )
        key_b = (
            int(
                pair[
                    "rollout_b"
                ]
            ),
            time_index,
        )

        repeats_a = side_output[
            key_a
        ][
            "repeats"
        ]
        repeats_b = side_output[
            key_b
        ][
            "repeats"
        ]

        current_partial = (
            cross_partial_chamfer_values(
                repeats_a,
                repeats_b,
                0,
                radius,
            )
        )

        horizon_rows = []

        for horizon in horizons:
            branch_values = (
                cross_branch_values(
                    repeats_a,
                    repeats_b,
                    horizon,
                )
            )

            repeat_a = (
                pairwise_repeat_floor(
                    repeats_a,
                    horizon,
                )
            )
            repeat_b = (
                pairwise_repeat_floor(
                    repeats_b,
                    horizon,
                )
            )

            repeat_floor = float(
                max(
                    repeat_a,
                    repeat_b,
                )
            )

            formal_threshold = (
                gate4_threshold(
                    repeat_floor,
                    absolute_effect_min_m=
                        gate[
                            "absolute_effect_min_m"
                        ],
                    repeat_floor_multiplier=
                        gate[
                            "repeat_floor_multiplier"
                        ],
                )
            )

            branch_min = float(
                np.min(
                    branch_values
                )
            )
            branch_median = float(
                np.median(
                    branch_values
                )
            )
            branch_max = float(
                np.max(
                    branch_values
                )
            )

            partial_values = (
                cross_partial_chamfer_values(
                    repeats_a,
                    repeats_b,
                    horizon,
                    radius,
                )
            )

            partial_median = (
                None
                if partial_values.size == 0
                else float(
                    np.median(
                        partial_values
                    )
                )
            )

            current_partial_median = (
                None
                if current_partial.size == 0
                else float(
                    np.median(
                        current_partial
                    )
                )
            )

            partial_growth = (
                None
                if partial_median is None
                or current_partial_median is None
                else float(
                    partial_median
                    - current_partial_median
                )
            )

            raw_position_values = []
            winding_delta_values = []

            for repeat_a_row in repeats_a:
                for repeat_b_row in repeats_b:
                    raw_position_values.append(
                        ordered_position_rmse(
                            repeat_a_row[
                                horizon
                            ][
                                "rope_xyz"
                            ],
                            repeat_b_row[
                                horizon
                            ][
                                "rope_xyz"
                            ],
                        )
                    )

                    winding_delta_values.append(
                        float(
                            np.max(
                                np.abs(
                                    np.asarray(
                                        repeat_a_row[
                                            horizon
                                        ][
                                            "signed_winding_turns"
                                        ],
                                        dtype=np.float64,
                                    )
                                    - np.asarray(
                                        repeat_b_row[
                                            horizon
                                        ][
                                            "signed_winding_turns"
                                        ],
                                        dtype=np.float64,
                                    )
                                )
                            )
                        )
                    )

            horizon_rows.append(
                {
                    "horizon":
                        int(
                            horizon
                        ),

                    "branch_displacement_rmse_min_m":
                        branch_min,

                    "branch_displacement_rmse_median_m":
                        branch_median,

                    "branch_displacement_rmse_max_m":
                        branch_max,

                    "repeat_floor_a_max_m":
                        repeat_a,

                    "repeat_floor_b_max_m":
                        repeat_b,

                    "repeat_floor_max_m":
                        repeat_floor,

                    "formal_threshold_m":
                        formal_threshold,

                    "gate4_pass":
                        bool(
                            branch_min
                            >= formal_threshold
                        ),

                    "raw_full_rope_position_rmse_median_m":
                        float(
                            np.median(
                                np.asarray(
                                    raw_position_values
                                )
                            )
                        ),

                    "partial_future_chamfer_median_m":
                        partial_median,

                    "partial_chamfer_growth_from_h0_m":
                        partial_growth,

                    "signed_winding_delta_linf_median_turns":
                        float(
                            np.median(
                                np.asarray(
                                    winding_delta_values
                                )
                            )
                        ),
                }
            )

        passing_horizons = [
            int(
                row[
                    "horizon"
                ]
            )
            for row in horizon_rows
            if row[
                "gate4_pass"
            ]
        ]

        max_row = max(
            horizon_rows,
            key=lambda row: row[
                "branch_displacement_rmse_median_m"
            ],
        )

        pair_rows.append(
            {
                "pair_id":
                    pair[
                        "pair_id"
                    ],

                "source_rank":
                    int(
                        pair[
                            "source_rank"
                        ]
                    ),

                "rollout_a":
                    int(
                        pair[
                            "rollout_a"
                        ]
                    ),

                "rollout_b":
                    int(
                        pair[
                            "rollout_b"
                        ]
                    ),

                "time_index":
                    time_index,

                "winding_index_a":
                    pair[
                        "winding_index_a"
                    ],

                "winding_index_b":
                    pair[
                        "winding_index_b"
                    ],

                "winding_stratum":
                    winding_stratum(
                        pair[
                            "winding_index_a"
                        ],
                        pair[
                            "winding_index_b"
                        ],
                    ),

                "pb2c_visible_rope_history_chamfer_m":
                    float(
                        pair[
                            "visible_rope_history_chamfer_m"
                        ]
                    ),

                "pb2c_pair_max_winding_integer_residual":
                    float(
                        pair[
                            "pair_max_winding_integer_residual"
                        ]
                    ),

                "pair_gate4_pass":
                    bool(
                        passing_horizons
                    ),

                "passing_horizons":
                    passing_horizons,

                "first_passing_horizon":
                    (
                        None
                        if not passing_horizons
                        else min(
                            passing_horizons
                        )
                    ),

                "horizon_of_maximum_sampled_displacement_divergence":
                    int(
                        max_row[
                            "horizon"
                        ]
                    ),

                "maximum_sampled_branch_displacement_rmse_median_m":
                    float(
                        max_row[
                            "branch_displacement_rmse_median_m"
                        ]
                    ),

                "horizons":
                    horizon_rows,
            }
        )

    passing = [
        row
        for row in pair_rows
        if row[
            "pair_gate4_pass"
        ]
    ]

    passing_strata = sorted(
        {
            row[
                "winding_stratum"
            ]
            for row in passing
        }
    )

    confirmed = (
        len(passing)
        >= int(
            config[
                "gate4"
            ][
                "minimum_passing_pairs"
            ]
        )
        and len(
            passing_strata
        )
        >= int(
            config[
                "gate4"
            ][
                "minimum_passing_winding_strata"
            ]
        )
    )

    return {
        "pair_count":
            int(
                len(
                    pair_rows
                )
            ),

        "passing_pair_count":
            int(
                len(
                    passing
                )
            ),

        "passing_pair_ids":
            [
                row[
                    "pair_id"
                ]
                for row in passing
            ],

        "passing_winding_strata":
            passing_strata,

        "confirmed":
            bool(
                confirmed
            ),

        "pairs":
            pair_rows,
    }


def save_trajectory_npz(
        config,
        shortlist,
        side_output):
    raw_root = Path(
        config["outputs"][
            "raw_root"
        ]
    )
    raw_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    side_keys = []
    seen = set()

    for pair in shortlist["pairs"]:
        for side in (
                "a",
                "b"):
            key = (
                int(
                    pair[
                        f"rollout_{side}"
                    ]
                ),
                int(
                    pair[
                        "time_index"
                    ]
                ),
            )
            if key not in seen:
                side_keys.append(
                    key
                )
                seen.add(key)

    horizons = [
        0,
        *[
            int(v)
            for v in config[
                "replay"
            ][
                "horizons"
            ]
        ],
    ]

    repeat_count = int(
        config["replay"][
            "snapshot_repeat_count"
        ]
    )

    fields = (
        "rope_xyz",
        "post_xyz",
        "ee1_pos",
        "ee1_quat",
        "ee2_pos",
        "ee2_quat",
        "motor_qpos_1",
        "motor_qpos_2",
        "reward",
        "signed_winding_turns",
    )

    arrays = {}
    for field in fields:
        arrays[field] = np.stack(
            [
                np.stack(
                    [
                        np.stack(
                            [
                                side_output[
                                    key
                                ][
                                    "repeats"
                                ][
                                    repeat_index
                                ][
                                    horizon
                                ][
                                    field
                                ]
                                for horizon in horizons
                            ],
                            axis=0,
                        )
                        for repeat_index in range(
                            repeat_count
                        )
                    ],
                    axis=0,
                )
                for key in side_keys
            ],
            axis=0,
        )

    side_meta = [
        side_output[
            key
        ][
            "meta"
        ]
        for key in side_keys
    ]

    output = (
        raw_root
        / "PB3_TRAJECTORIES.npz"
    )

    np.savez_compressed(
        output,
        rollout_id=np.asarray(
            [
                int(
                    row[
                        "rollout_id"
                    ]
                )
                for row in side_meta
            ],
            dtype=np.int32,
        ),
        time_index=np.asarray(
            [
                int(
                    row[
                        "time_index"
                    ]
                )
                for row in side_meta
            ],
            dtype=np.int16,
        ),
        batch_index=np.asarray(
            [
                int(
                    row[
                        "batch_index"
                    ]
                )
                for row in side_meta
            ],
            dtype=np.int16,
        ),
        env_index=np.asarray(
            [
                int(
                    row[
                        "env_index"
                    ]
                )
                for row in side_meta
            ],
            dtype=np.int16,
        ),
        seed=np.asarray(
            [
                int(
                    row[
                        "seed"
                    ]
                )
                for row in side_meta
            ],
            dtype=np.int32,
        ),
        horizons=np.asarray(
            horizons,
            dtype=np.int16,
        ),
        **arrays,
    )

    return output


def write_outputs(
        config,
        shortlist,
        *,
        shortlist_validation,
        verdict,
        alignment_records,
        pair_revalidation_records=None,
        r3_rule=None,
        audit=None,
        blocked_details=None,
        trajectories_path=None):
    raw_root = Path(
        config["outputs"][
            "raw_root"
        ]
    )
    raw_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    committed = (
        REPO_ROOT
        / config["outputs"][
            "committed_report_dir"
        ]
    )
    committed.mkdir(
        parents=True,
        exist_ok=True,
    )

    pair_revalidation_records = pair_revalidation_records or []
    pre_future_valid = bool(
        len(alignment_records) == 20
        and all(row.get("valid", False) for row in alignment_records)
        and len(pair_revalidation_records) == 10
        and all(row.get("valid", False) for row in pair_revalidation_records)
    )
    future_suffix_executed = bool(
        verdict in (
            "PB3_CAUSAL_FUTURE_BIFURCATION_CONFIRMED",
            "PB3_CAUSAL_FUTURE_BIFURCATION_NOT_CONFIRMED",
            "PB3_REPEAT_SUFFIX_INVALID",
        )
    )

    result = {
        "verdict":
            verdict,

        "shortlist_pair_count":
            int(
                len(
                    shortlist[
                        "pairs"
                    ]
                )
            ),

        "shortlist_unique_rollout_count":
            int(
                len(
                    {
                        int(
                            pair[
                                key
                            ]
                        )
                        for pair in shortlist[
                            "pairs"
                        ]
                        for key in (
                            "rollout_a",
                            "rollout_b",
                        )
                    }
                )
            ),

        "shortlist_selection":
            shortlist[
                "selection_rule"
            ],

        "shortlist_validation":
            shortlist_validation,

        "horizons":
            config["replay"][
                "horizons"
            ],

        "snapshot_repeat_count":
            int(
                config[
                    "replay"
                ][
                    "snapshot_repeat_count"
                ]
            ),

        "gate4": {
            **config[
                "gate4"
            ],
            "pb2c_discovery_thresholds_reused":
                False,
        },

        "alignment_records":
            alignment_records,

        "live_pair_revalidation_records":
            pair_revalidation_records,

        "pre_future_barrier": {
            "all_20_targeted_replay_alignments_valid": bool(
                len(alignment_records) == 20
                and all(row.get("valid", False) for row in alignment_records)
            ),
            "all_10_live_pairs_valid": bool(
                len(pair_revalidation_records) == 10
                and all(
                    row.get("valid", False)
                    for row in pair_revalidation_records
                )
            ),
            "valid": pre_future_valid,
            "future_suffix_executed": future_suffix_executed,
        },

        "pb3r3_alignment_rule": r3_rule,

        "audit":
            audit,

        "blocked_details":
            blocked_details,

        "trajectories_path":
            (
                None
                if trajectories_path
                is None
                else str(
                    trajectories_path
                )
            ),
    }

    (
        raw_root
        / "AUDIT.json"
    ).write_text(
        json.dumps(
            result,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    evidence = {
        "phase_name":
            config[
                "phase_name"
            ],

        "verdict":
            verdict,

        "repository": {
            "starting_main_sha":
                config[
                    "provenance"
                ][
                    "starting_main_sha"
                ],

            "ending_main_sha":
                git(
                    "rev-parse",
                    "HEAD",
                ),

            "dlolab_gitlink":
                git(
                    "rev-parse",
                    "HEAD:external/dlo-lab",
                ),
        },

        "scientific":
            result,
    }

    (
        committed
        / "EVIDENCE.json"
    ).write_text(
        json.dumps(
            evidence,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    pair_metrics = {
        "verdict":
            verdict,

        "pair_metrics":
            (
                []
                if audit is None
                else audit[
                    "pairs"
                ]
            ),
    }

    (
        committed
        / "PAIR_METRICS.json"
    ).write_text(
        json.dumps(
            pair_metrics,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    if verdict == (
        "PB3_CAUSAL_FUTURE_BIFURCATION_CONFIRMED"
    ):
        next_action = (
            "Proceed to PB4 control-relevance/action-regret audit. "
            "Do not start StateDiff training before PB4."
        )
    elif verdict == (
        "PB3_CAUSAL_FUTURE_BIFURCATION_NOT_CONFIRMED"
    ):
        next_action = (
            "Do not lower Gate-4 criteria post hoc. Inspect the frozen "
            "pair-level/horizon results, then reassess whether Wrapping "
            "provides a control-relevant CCDA branch before further data "
            "or model work."
        )
    elif verdict == (
        "PB3_TARGETED_REPLAY_ALIGNMENT_FAILED"
    ):
        next_action = (
            "Fix exact PB2-C targeted replay alignment only. Do not "
            "interpret future bifurcation and do not loosen the alignment "
            "tolerance automatically."
        )
    elif verdict == "PB3_R3_LIVE_BRANCH_ALIGNMENT_FAILED":
        next_action = (
            "Stop before all snapshot/future work. Preserve the frozen R3 "
            "rule and formal 10-pair shortlist; do not drop or replace pairs."
        )
    elif verdict == (
        "PB3_SNAPSHOT_RESTORE_ALIGNMENT_FAILED"
    ):
        next_action = (
            "Fix pinned Genesis snapshot restore usage only. Do not "
            "interpret future bifurcation."
        )
    else:
        next_action = (
            "Fix the repeat-suffix validity issue only before causal "
            "interpretation."
        )

    lines = [
        "# PB3 DLO-Lab Wrapping Same-Action Causal Bifurcation Audit",
        "",
        f"Verdict: `{verdict}`",
        "",
        "## Formal design",
        "",
        "- Shortlist: 10 PB2-C pairs / 20 unique rollouts",
        "- Shortlist selected without future information: True",
        "- Snapshot repeats per branch state: "
        f"{config['replay']['snapshot_repeat_count']}",
        "- Horizons: "
        f"`{config['replay']['horizons']}`",
        (
            "- Primary metric: ordered rope displacement-field RMSE "
            "`(X(t+h)-X(t))_A vs (X(t+h)-X(t))_B`"
        ),
        (
            "- Absolute effect minimum: "
            f"{1000 * config['gate4']['absolute_effect_min_m']:.3f} mm"
        ),
        (
            "- Repeat-floor multiplier: "
            f"{config['gate4']['repeat_floor_multiplier']:.3f}x"
        ),
        "",
        "## Replay / snapshot",
        "",
        (
            "- Targeted replay alignment valid: "
            f"{all(row.get('valid', False) for row in alignment_records) if alignment_records else False}"
        ),
        (
            "- Live PB2-C pair revalidation valid: "
            f"{all(row.get('valid', False) for row in pair_revalidation_records) if pair_revalidation_records else False}"
        ),
        f"- Pre-future barrier passed: {pre_future_valid}",
        f"- Future suffix executed: {future_suffix_executed}",
    ]

    if audit is not None:
        lines.extend(
            [
                "",
                "## Gate 4 result",
                "",
                (
                    "- Passing pairs: "
                    f"{audit['passing_pair_count']} / {audit['pair_count']}"
                ),
                (
                    "- Passing winding strata: "
                    f"{audit['passing_winding_strata']}"
                ),
                "",
            ]
        )

    if blocked_details is not None:
        lines.extend(
            [
                "",
                "## Blocked details",
                "",
                "```json",
                json.dumps(
                    blocked_details,
                    indent=2,
                ),
                "```",
                "",
            ]
        )

    lines.extend(
        [
            "## Next action",
            "",
            next_action,
            "",
            "## Boundary",
            "",
            (
                "PB3 tests same-action future dynamics only. It does not "
                "establish control relevance, deployable sensing, or model "
                "performance."
            ),
            "",
        ]
    )

    (
        committed
        / "RESULT.md"
    ).write_text(
        "\n".join(
            lines
        ),
        encoding="utf-8",
    )

    return result


def run(config_path):
    config = load_json(
        config_path
    )

    sources = validate_sources(
        config
    )

    frozen = load_frozen_rollouts(
        sources[
            "raw_path"
        ]
    )

    qpos_path = (
        official_log_dir()
        / "best_qpos.npy"
    )
    if not qpos_path.is_file():
        raise FileNotFoundError(
            str(
                qpos_path
            )
        )

    qpos = normalize_qpos(
        np.load(
            qpos_path
        )
    )

    if not np.array_equal(
        qpos,
        frozen[
            "common_qpos_replay"
        ],
    ):
        raise RuntimeError(
            "PB3 official best_qpos differs from frozen PB2-C common replay"
        )

    alignment_records = []
    pair_revalidation_records = []
    r3_rule = sources.get("r3_rule")

    try:
        if r3_rule is None:
            side_output, alignment_records = run_snapshot_replays(
                config,
                sources["shortlist"],
                frozen,
                qpos,
            )
        else:
            side_output, alignment_records, pair_revalidation_records = (
                run_snapshot_replays_r3(
                    config,
                    sources["shortlist"],
                    frozen,
                    qpos,
                    r3_rule,
                    alignment_records=alignment_records,
                    pair_revalidation_records=pair_revalidation_records,
                )
            )

        audit = analyze_pairs(
            config,
            sources[
                "shortlist"
            ],
            side_output,
        )

        trajectories_path = (
            save_trajectory_npz(
                config,
                sources[
                    "shortlist"
                ],
                side_output,
            )
        )

        verdict = (
            "PB3_CAUSAL_FUTURE_BIFURCATION_CONFIRMED"
            if audit[
                "confirmed"
            ]
            else
            "PB3_CAUSAL_FUTURE_BIFURCATION_NOT_CONFIRMED"
        )

        result = write_outputs(
            config,
            sources[
                "shortlist"
            ],
            shortlist_validation=
                sources[
                    "shortlist_validation"
                ],
            verdict=verdict,
            alignment_records=
                alignment_records,
            pair_revalidation_records=pair_revalidation_records,
            r3_rule=r3_rule,
            audit=audit,
            trajectories_path=
                trajectories_path,
        )

    except PB3Blocked as blocked:
        result = write_outputs(
            config,
            sources[
                "shortlist"
            ],
            shortlist_validation=
                sources[
                    "shortlist_validation"
                ],
            verdict=blocked.verdict,
            alignment_records=
                alignment_records,
            pair_revalidation_records=pair_revalidation_records,
            r3_rule=r3_rule,
            blocked_details=
                blocked.details,
        )

    print(
        "verdict={}".format(
            result[
                "verdict"
            ]
        )
    )

    return result


def validate_resume_sources(config_path):
    """CPU-only validation of frozen PB3 Resume inputs and R3 rule."""
    config = load_json(config_path)
    sources = validate_sources(config)
    if "r3_rule" not in sources:
        raise RuntimeError("PB3 Resume config must provide a validated R3 rule")
    frozen = load_frozen_rollouts(sources["raw_path"])
    qpos_path = official_log_dir() / "best_qpos.npy"
    if not qpos_path.is_file():
        raise FileNotFoundError(str(qpos_path))
    qpos = normalize_qpos(np.load(qpos_path))
    if not np.array_equal(qpos, frozen["common_qpos_replay"]):
        raise RuntimeError(
            "PB3 official best_qpos differs from frozen PB2-C common replay"
        )
    print("PB3 Resume source validation: PASS")
    print(
        "formal_pairs={} formal_rollouts={} rope_threshold_m={:.17g}".format(
            len(sources["shortlist"]["pairs"]),
            sources["shortlist_validation"]["unique_rollout_count"],
            float(
                sources["r3_rule"][
                    "targeted_replay_engineering_alignment"
                ]["rope"]["threshold_m"]
            ),
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
    )
    args = parser.parse_args()
    if args.validate_only:
        validate_resume_sources(args.config)
    else:
        run(args.config)


if __name__ == "__main__":
    main()
