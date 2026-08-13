"""PB3-B1 — prospective live-pair cohort discovery for DLO-Lab Wrapping.

Purpose
-------
Replace the blocked historical-frozen PB3 cohort with a prospective cohort
whose scientific admission is evaluated on a fresh live replay itself.

This phase has two steps:
1. CPU-only protocol freeze before any GPU replay.
2. Fresh live screening after the protocol is committed.

This phase NEVER:
- creates branch snapshots for causal suffixes;
- runs a same-action future suffix;
- computes Gate 4;
- uses future divergence, force, sensor, or reward to admit pairs;
- drops/replaces pairs after the first 10 accepted live-valid pairs.

The next phase (PB3-B2) may use the committed cohort, but must revalidate its
live pair semantics before running any future suffix.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from scripts.experiment3.dlolab_wrapping.collect_natural_rollouts_pb2c import (
    replay_batch,
)
from scripts.experiment3.dlolab_wrapping.pair_discovery_pb2c import (
    batch_symmetric_chamfer,
    chamfer_for_pairs_at_time,
    compute_visibility_mask,
    load_datasets,
    occlusion_radius,
    robot_history_metrics,
    signed_winding_index,
    winding_integer_residual,
)
from scripts.experiment3.dlolab_wrapping.paths import (
    REPO_ROOT,
    official_log_dir,
)
from scripts.experiment3.dlolab_wrapping.replay_winding_audit import (
    build_env,
    load_json,
    normalize_qpos,
)
from utils.domain_randomization import wrapping_args  # noqa: E402


PROTOCOL_VERDICT = "PB3B1_PROTOCOL_PREREGISTERED"
SUCCESS_VERDICT = "PB3B1_PROSPECTIVE_LIVE_COHORT_FROZEN"
INSUFFICIENT_VERDICT = "PB3B1_PROSPECTIVE_LIVE_COHORT_INSUFFICIENT"
STRATA_INSUFFICIENT_VERDICT = (
    "PB3B1_PROSPECTIVE_LIVE_COHORT_STRATA_INSUFFICIENT"
)


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


def _candidate_signature(row):
    return {
        "rollout_a": int(row["rollout_a"]),
        "rollout_b": int(row["rollout_b"]),
        "time_index": int(row["time_index"]),
        "visible_rope_history_chamfer_m": float(
            row["visible_rope_history_chamfer_m"]
        ),
        "dual_ee_position_history_mean_m": float(
            row["robot_observation"][
                "dual_ee_position_history_mean_m"
            ]
        ),
        "dual_ee_quaternion_geodesic_history_mean_rad": float(
            row["robot_observation"][
                "dual_ee_quaternion_geodesic_history_mean_rad"
            ]
        ),
        "dual_motor_qpos_history_rms_rad": float(
            row["robot_observation"][
                "dual_motor_qpos_history_rms_rad"
            ]
        ),
        "winding_index_a": [
            int(v)
            for v in row["winding_index_a"]
        ],
        "winding_index_b": [
            int(v)
            for v in row["winding_index_b"]
        ],
        "replay_a": {
            "batch_index": int(row["replay_a"]["batch_index"]),
            "env_index": int(row["replay_a"]["env_index"]),
            "seed": int(row["replay_a"]["seed"]),
        },
        "replay_b": {
            "batch_index": int(row["replay_b"]["batch_index"]),
            "env_index": int(row["replay_b"]["env_index"]),
            "seed": int(row["replay_b"]["seed"]),
        },
    }


def _candidate_signature_close(a, b):
    if (
        a["rollout_a"] != b["rollout_a"]
        or a["rollout_b"] != b["rollout_b"]
        or a["time_index"] != b["time_index"]
        or a["winding_index_a"] != b["winding_index_a"]
        or a["winding_index_b"] != b["winding_index_b"]
        or a["replay_a"] != b["replay_a"]
        or a["replay_b"] != b["replay_b"]
    ):
        return False

    for key in (
        "visible_rope_history_chamfer_m",
        "dual_ee_position_history_mean_m",
        "dual_ee_quaternion_geodesic_history_mean_rad",
        "dual_motor_qpos_history_rms_rad",
    ):
        if not np.isclose(
            float(a[key]),
            float(b[key]),
            rtol=1e-12,
            atol=1e-15,
        ):
            return False
    return True


def load_sources(config):
    source = config["source"]

    repo_paths = {
        "pb2c_config": REPO_ROOT / source["pb2c_config"],
        "pb2c_candidates": REPO_ROOT / source["pb2c_candidates"],
        "pb2c_evidence": REPO_ROOT / source["pb2c_evidence"],
        "historical_pb3_shortlist":
            REPO_ROOT / source["historical_pb3_shortlist"],
        "historical_pb3_resume_evidence":
            REPO_ROOT / source["historical_pb3_resume_evidence"],
        "pb3r2_evidence": REPO_ROOT / source["pb3r2_evidence"],
        "pb3r3_rule": REPO_ROOT / source["pb3r3_rule"],
    }

    raw_path = Path(source["pb2c_raw_rollouts"])

    for path in list(repo_paths.values()) + [raw_path]:
        if not path.is_file():
            raise FileNotFoundError(str(path))

    pb2c_config = load_json(repo_paths["pb2c_config"])
    pb2c_candidates = load_json(repo_paths["pb2c_candidates"])
    pb2c_evidence = load_json(repo_paths["pb2c_evidence"])
    shortlist = load_json(repo_paths["historical_pb3_shortlist"])
    historical_resume = load_json(
        repo_paths["historical_pb3_resume_evidence"]
    )
    pb3r2 = load_json(repo_paths["pb3r2_evidence"])
    pb3r3 = load_json(repo_paths["pb3r3_rule"])

    if pb2c_evidence.get("verdict") != source["expected_pb2c_verdict"]:
        raise RuntimeError("PB2-C verdict mismatch")

    if (
        historical_resume.get("verdict")
        != source["expected_historical_pb3_resume_verdict"]
    ):
        raise RuntimeError("Historical PB3 Resume verdict mismatch")

    scientific = historical_resume["scientific"]
    if scientific.get("trajectories_path") is not None:
        raise RuntimeError(
            "Historical blocked PB3 must not contain causal trajectories"
        )
    if scientific.get("audit") is not None:
        raise RuntimeError(
            "Historical blocked PB3 must remain pre-Gate-4"
        )
    if bool(
        scientific["pre_future_barrier"].get("future_suffix_executed")
    ):
        raise RuntimeError(
            "Historical blocked PB3 unexpectedly executed future suffix"
        )

    if pb3r2.get("verdict") != source["expected_pb3r2_verdict"]:
        raise RuntimeError("PB3-R2 verdict mismatch")

    if pb3r3.get("verdict") != source["expected_pb3r3_verdict"]:
        raise RuntimeError("PB3-R3 verdict mismatch")

    expected_args = config["live_collection"]["expected_wrapping_args"]
    actual_args = _jsonable(dict(wrapping_args))
    if actual_args != expected_args:
        raise RuntimeError(
            f"Pinned wrapping_args mismatch: {actual_args} != {expected_args}"
        )

    if len(shortlist.get("pairs", [])) != 10:
        raise RuntimeError(
            "Historical formal PB3 shortlist must remain 10 pairs"
        )

    formal_rollouts = sorted(
        {
            int(pair[key])
            for pair in shortlist["pairs"]
            for key in ("rollout_a", "rollout_b")
        }
    )
    if len(formal_rollouts) != 20:
        raise RuntimeError(
            "Historical formal PB3 shortlist must contain 20 unique rollouts"
        )

    calibration_states = (
        pb3r2["scientific"]["calibration_set"]["states"]
    )
    calibration_rollouts = sorted(
        {
            int(row["rollout_id"])
            for row in calibration_states
        }
    )
    if len(calibration_rollouts) != 20:
        raise RuntimeError(
            "PB3-R2 calibration must contain 20 unique rollouts"
        )

    if set(formal_rollouts) & set(calibration_rollouts):
        raise RuntimeError(
            "Historical formal PB3 and PB3-R2 calibration rollouts overlap"
        )

    exclusion_union = sorted(
        set(formal_rollouts) | set(calibration_rollouts)
    )
    expected_union = int(
        config["candidate_queue"]["expected_exclusion_union_size"]
    )
    if len(exclusion_union) != expected_union:
        raise RuntimeError(
            f"Expected exclusion union {expected_union}, got "
            f"{len(exclusion_union)}"
        )

    frozen_data = load_datasets([raw_path])

    return {
        "pb2c_config": pb2c_config,
        "pb2c_candidates": pb2c_candidates,
        "pb2c_evidence": pb2c_evidence,
        "shortlist": shortlist,
        "historical_resume": historical_resume,
        "pb3r2": pb3r2,
        "pb3r3": pb3r3,
        "formal_rollouts": formal_rollouts,
        "calibration_rollouts": calibration_rollouts,
        "exclusion_union": exclusion_union,
        "frozen_data": frozen_data,
    }


def prefix_valid_indices_at_time(data, time_index, history):
    """Return rollouts whose selection history is valid through branch time.

    This intentionally does NOT use official_rollout_valid, final reward, or
    any post-branch survival information. A rollout that fails after the branch
    time remains eligible for that branch point.
    """
    time_index = int(time_index)
    start = time_index - int(history) + 1
    if start < 0:
        return np.asarray([], dtype=np.int64)

    failed = (
        np.asarray(
            data["official_failed_rope_nan"],
            dtype=bool,
        )
        | np.asarray(
            data["official_failed_stretch"],
            dtype=bool,
        )
    )
    first_fail = np.asarray(
        data["official_first_fail_step"],
        dtype=np.int64,
    )

    survived_through_branch = (
        (~failed)
        | (first_fail > time_index)
    )

    selection_keys = (
        "rope_xyz",
        "post_xyz",
        "ee1_pos",
        "ee1_quat",
        "ee2_pos",
        "ee2_quat",
        "motor_qpos_1",
        "motor_qpos_2",
        "signed_winding_turns",
    )

    finite = np.ones(
        len(survived_through_branch),
        dtype=bool,
    )
    for key in selection_keys:
        values = np.asarray(
            data[key][
                :,
                start:time_index + 1,
            ]
        )
        finite &= np.isfinite(
            values.reshape(
                values.shape[0],
                -1,
            )
        ).all(axis=1)

    return np.flatnonzero(
        survived_through_branch
        & finite
    )


def mine_prefix_only_candidate_queue(data, pb2c_config):
    """Rebuild PB2-C-style candidates without future-survival conditioning.

    Candidate validity is evaluated separately at each branch time using only
    t-history+1:t state and failure status through t. Full-rollout
    official_rollout_valid is never used for queue admission.
    """
    rope_xyz = np.asarray(
        data["rope_xyz"],
        dtype=np.float32,
    )
    post_xyz = np.asarray(
        data["post_xyz"],
        dtype=np.float32,
    )
    signed_turns = np.asarray(
        data["signed_winding_turns"],
        dtype=np.float64,
    )

    obs_cfg = pb2c_config[
        "partial_state_observation"
    ]
    rope_cfg = obs_cfg["rope_component"]
    robot_cfg = obs_cfg["robot_component"]
    history = int(
        obs_cfg["history_samples"]
    )
    n_times = int(
        rope_xyz.shape[1]
    )

    chamfer_threshold = float(
        rope_cfg[
            "max_visible_history_chamfer_m"
        ]
    )
    ee_pos_threshold = float(
        robot_cfg[
            "max_dual_ee_position_history_mean_m"
        ]
    )
    ee_quat_threshold = float(
        robot_cfg[
            "max_dual_ee_quaternion_geodesic_history_mean_rad"
        ]
    )
    qpos_threshold = float(
        robot_cfg[
            "max_dual_motor_qpos_history_rms_rad"
        ]
    )
    pair_batch_size = int(
        pb2c_config[
            "discovery"
        ][
            "pair_batch_size"
        ]
    )

    radius = occlusion_radius(
        pb2c_config
    )
    visible = compute_visibility_mask(
        rope_xyz,
        post_xyz,
        radius,
    )
    visible_counts = visible.sum(
        axis=2
    )
    winding_index = signed_winding_index(
        signed_turns
    )
    residual = winding_integer_residual(
        signed_turns
    )

    candidates = []
    funnel = {
        "same_time_prefix_valid_pair_comparisons": 0,
        "hidden_winding_index_different": 0,
        "nonempty_partial_rope_history": 0,
        "robot_observation_pass": 0,
        "visible_history_chamfer_pass": 0,
        "candidate_count": 0,
    }

    safe_single_frame_bound = (
        history
        * chamfer_threshold
    )

    for time_index in range(
        history - 1,
        n_times,
    ):
        valid_indices = (
            prefix_valid_indices_at_time(
                data,
                time_index,
                history,
            )
        )
        if len(valid_indices) < 2:
            continue

        local_a, local_b = np.triu_indices(
            len(valid_indices),
            k=1,
        )
        upper_a = valid_indices[
            local_a
        ]
        upper_b = valid_indices[
            local_b
        ]

        funnel[
            "same_time_prefix_valid_pair_comparisons"
        ] += int(
            len(upper_a)
        )

        hidden_diff = np.any(
            winding_index[
                upper_a,
                time_index,
            ]
            != winding_index[
                upper_b,
                time_index,
            ],
            axis=1,
        )
        ia = upper_a[
            hidden_diff
        ]
        ib = upper_b[
            hidden_diff
        ]
        funnel[
            "hidden_winding_index_different"
        ] += int(
            len(ia)
        )
        if len(ia) == 0:
            continue

        start = (
            time_index
            - history
            + 1
        )
        hist_ok = (
            np.all(
                visible_counts[
                    ia,
                    start:time_index + 1,
                ] > 0,
                axis=1,
            )
            & np.all(
                visible_counts[
                    ib,
                    start:time_index + 1,
                ] > 0,
                axis=1,
            )
        )
        ia = ia[
            hist_ok
        ]
        ib = ib[
            hist_ok
        ]
        funnel[
            "nonempty_partial_rope_history"
        ] += int(
            len(ia)
        )
        if len(ia) == 0:
            continue

        robot = robot_history_metrics(
            data,
            ia,
            ib,
            time_index,
            history,
        )
        robot_pass = (
            robot[
                "dual_ee_position_history_mean_m"
            ]
            <= ee_pos_threshold
        )
        robot_pass &= (
            robot[
                "dual_ee_quaternion_geodesic_history_mean_rad"
            ]
            <= ee_quat_threshold
        )
        robot_pass &= (
            robot[
                "dual_motor_qpos_history_rms_rad"
            ]
            <= qpos_threshold
        )

        ia = ia[
            robot_pass
        ]
        ib = ib[
            robot_pass
        ]
        robot = {
            key:
                value[
                    robot_pass
                ]
            for key, value in robot.items()
        }
        funnel[
            "robot_observation_pass"
        ] += int(
            len(ia)
        )
        if len(ia) == 0:
            continue

        current_chamfer = (
            chamfer_for_pairs_at_time(
                rope_xyz,
                visible,
                ia,
                ib,
                time_index,
                pair_batch_size,
            )
        )
        safe = (
            current_chamfer
            <= safe_single_frame_bound
        )
        ia = ia[
            safe
        ]
        ib = ib[
            safe
        ]
        current_chamfer = (
            current_chamfer[
                safe
            ]
        )
        robot = {
            key:
                value[
                    safe
                ]
            for key, value in robot.items()
        }
        if len(ia) == 0:
            continue

        history_sum = (
            current_chamfer.astype(
                np.float64
            )
        )
        for past_time in range(
            start,
            time_index,
        ):
            history_sum += (
                chamfer_for_pairs_at_time(
                    rope_xyz,
                    visible,
                    ia,
                    ib,
                    past_time,
                    pair_batch_size,
                )
            )

        history_chamfer = (
            history_sum
            / history
        )
        chamfer_pass = (
            history_chamfer
            <= chamfer_threshold
        )

        ia = ia[
            chamfer_pass
        ]
        ib = ib[
            chamfer_pass
        ]
        history_chamfer = (
            history_chamfer[
                chamfer_pass
            ]
        )
        robot = {
            key:
                value[
                    chamfer_pass
                ]
            for key, value in robot.items()
        }
        funnel[
            "visible_history_chamfer_pass"
        ] += int(
            len(ia)
        )

        for local_index in range(
            len(ia)
        ):
            a = int(
                ia[
                    local_index
                ]
            )
            b = int(
                ib[
                    local_index
                ]
            )
            index_a = winding_index[
                a,
                time_index,
            ]
            index_b = winding_index[
                b,
                time_index,
            ]
            residual_a = residual[
                a,
                time_index,
            ]
            residual_b = residual[
                b,
                time_index,
            ]
            differing_posts = (
                np.flatnonzero(
                    index_a
                    != index_b
                )
            )

            candidates.append(
                {
                    "rollout_a":
                        int(
                            data[
                                "rollout_id"
                            ][a]
                        ),
                    "rollout_b":
                        int(
                            data[
                                "rollout_id"
                            ][b]
                        ),
                    "time_index":
                        int(
                            time_index
                        ),
                    "visible_rope_history_chamfer_m":
                        float(
                            history_chamfer[
                                local_index
                            ]
                        ),
                    "robot_observation": {
                        "dual_ee_position_history_mean_m":
                            float(
                                robot[
                                    "dual_ee_position_history_mean_m"
                                ][local_index]
                            ),
                        "dual_ee_quaternion_geodesic_history_mean_rad":
                            float(
                                robot[
                                    "dual_ee_quaternion_geodesic_history_mean_rad"
                                ][local_index]
                            ),
                        "dual_motor_qpos_history_rms_rad":
                            float(
                                robot[
                                    "dual_motor_qpos_history_rms_rad"
                                ][local_index]
                            ),
                    },
                    "winding_index_a":
                        [
                            int(v)
                            for v in index_a
                        ],
                    "winding_index_b":
                        [
                            int(v)
                            for v in index_b
                        ],
                    "pair_max_winding_integer_residual":
                        float(
                            max(
                                np.max(
                                    residual_a
                                ),
                                np.max(
                                    residual_b
                                ),
                            )
                        ),
                    "differing_post_count":
                        int(
                            len(
                                differing_posts
                            )
                        ),
                    "replay_a": {
                        "batch_index":
                            int(
                                data[
                                    "batch_index"
                                ][a]
                            ),
                        "env_index":
                            int(
                                data[
                                    "env_index"
                                ][a]
                            ),
                        "seed":
                            int(
                                data[
                                    "batch_seed"
                                ][a]
                            ),
                    },
                    "replay_b": {
                        "batch_index":
                            int(
                                data[
                                    "batch_index"
                                ][b]
                            ),
                        "env_index":
                            int(
                                data[
                                    "env_index"
                                ][b]
                            ),
                        "seed":
                            int(
                                data[
                                    "batch_seed"
                                ][b]
                            ),
                    },
                    "prefix_valid_a":
                        True,
                    "prefix_valid_b":
                        True,
                    "full_rollout_valid_a_diagnostic":
                        bool(
                            data[
                                "official_rollout_valid"
                            ][a]
                        ),
                    "full_rollout_valid_b_diagnostic":
                        bool(
                            data[
                                "official_rollout_valid"
                            ][b]
                        ),
                    "selection_used_full_rollout_survival":
                        False,
                    "selection_used_final_reward":
                        False,
                    "selection_used_future_divergence":
                        False,
                }
            )

    candidates.sort(
        key=lambda row: (
            row[
                "visible_rope_history_chamfer_m"
            ],
            row[
                "robot_observation"
            ][
                "dual_ee_position_history_mean_m"
            ],
            row[
                "robot_observation"
            ][
                "dual_ee_quaternion_geodesic_history_mean_rad"
            ],
            row[
                "robot_observation"
            ][
                "dual_motor_qpos_history_rms_rad"
            ],
            -row[
                "differing_post_count"
            ],
        )
    )
    funnel[
        "candidate_count"
    ] = int(
        len(
            candidates
        )
    )
    return {
        "candidate_count":
            int(
                len(
                    candidates
                )
            ),
        "candidates":
            candidates,
        "funnel":
            funnel,
    }


def validate_survivor_subset_against_pb2c(
        prefix_candidates,
        sources):
    """Regression-only check; never used to form the prospective queue."""
    full_valid_ids = {
        int(
            sources[
                "frozen_data"
            ][
                "rollout_id"
            ][index]
        )
        for index in np.flatnonzero(
            np.asarray(
                sources[
                    "frozen_data"
                ][
                    "official_rollout_valid"
                ],
                dtype=bool,
            )
        )
    }

    survivor_subset = [
        row
        for row in prefix_candidates
        if (
            int(
                row[
                    "rollout_a"
                ]
            )
            in full_valid_ids
            and int(
                row[
                    "rollout_b"
                ]
            )
            in full_valid_ids
        )
    ]

    committed = sources[
        "pb2c_candidates"
    ]
    if len(
        survivor_subset
    ) != int(
        committed[
            "candidate_count"
        ]
    ):
        raise RuntimeError(
            "Prefix-only miner does not reproduce historical PB2-C "
            "candidate count on the full-survivor subset"
        )

    committed_top = committed[
        "top_candidates"
    ]
    for index, (
            actual,
            expected) in enumerate(
                zip(
                    survivor_subset[
                        :len(
                            committed_top
                        )
                    ],
                    committed_top,
                ),
                start=1,
            ):
        if not _candidate_signature_close(
            _candidate_signature(
                actual
            ),
            _candidate_signature(
                expected
            ),
        ):
            raise RuntimeError(
                "Prefix-only miner differs from historical PB2-C on the "
                f"full-survivor subset at rank {index}"
            )

    return {
        "historical_survivor_candidate_count":
            int(
                len(
                    survivor_subset
                )
            ),
        "historical_committed_candidate_count":
            int(
                committed[
                    "candidate_count"
                ]
            ),
        "historical_top_k_validated":
            True,
    }


def recompute_prefix_candidate_ranking(config, sources):
    prefix = mine_prefix_only_candidate_queue(
        sources[
            "frozen_data"
        ],
        sources[
            "pb2c_config"
        ],
    )

    survivor_validation = (
        validate_survivor_subset_against_pb2c(
            prefix[
                "candidates"
            ],
            sources,
        )
    )

    excluded = set(
        sources[
            "exclusion_union"
        ]
    )
    eligible = []

    for source_rank, row in enumerate(
            prefix[
                "candidates"
            ],
            start=1):
        rollout_a = int(
            row[
                "rollout_a"
            ]
        )
        rollout_b = int(
            row[
                "rollout_b"
            ]
        )

        if (
            rollout_a in excluded
            or rollout_b in excluded
        ):
            continue

        eligible.append(
            {
                "source_rank":
                    int(
                        source_rank
                    ),
                "rollout_a":
                    rollout_a,
                "rollout_b":
                    rollout_b,
                "time_index":
                    int(
                        row[
                            "time_index"
                        ]
                    ),
                "replay_a":
                    row[
                        "replay_a"
                    ],
                "replay_b":
                    row[
                        "replay_b"
                    ],
                "prefix_candidate_diagnostic": {
                    "visible_rope_history_chamfer_m":
                        float(
                            row[
                                "visible_rope_history_chamfer_m"
                            ]
                        ),
                    "winding_index_a":
                        row[
                            "winding_index_a"
                        ],
                    "winding_index_b":
                        row[
                            "winding_index_b"
                        ],
                    "full_rollout_valid_a_diagnostic":
                        bool(
                            row[
                                "full_rollout_valid_a_diagnostic"
                            ]
                        ),
                    "full_rollout_valid_b_diagnostic":
                        bool(
                            row[
                                "full_rollout_valid_b_diagnostic"
                            ]
                        ),
                },
            }
        )

    if len(
        eligible
    ) < int(
        config[
            "cohort"
        ][
            "target_pair_count"
        ]
    ):
        raise RuntimeError(
            "Fixed exclusions leave fewer than the target number of "
            "prefix-only candidate pairs"
        )

    return {
        "prefix_candidate_count":
            int(
                prefix[
                    "candidate_count"
                ]
            ),
        "prefix_funnel":
            prefix[
                "funnel"
            ],
        "survivor_subset_validation":
            survivor_validation,
        "eligible_candidate_count":
            int(
                len(
                    eligible
                )
            ),
        "eligible_candidates":
            eligible,
        "selection_used_full_rollout_survival":
            False,
        "selection_used_final_reward":
            False,
    }


def build_protocol(config, sources, queue_info):
    pb2c = sources["pb2c_config"]
    obs = pb2c["partial_state_observation"]
    rope = obs["rope_component"]
    robot = obs["robot_component"]

    protocol = {
        "phase": "PB3-B1",
        "verdict": PROTOCOL_VERDICT,
        "purpose": (
            "Prospective live-pair cohort discovery only; no causal future "
            "suffix in this phase."
        ),
        "source_state": {
            "historical_pb3_resume_verdict":
                sources["historical_resume"]["verdict"],
            "historical_pb3_future_suffix_executed": False,
            "pb3r2_verdict": sources["pb3r2"]["verdict"],
            "pb3r3_verdict": sources["pb3r3"]["verdict"],
        },
        "fixed_exclusions": {
            "historical_formal_pb3_rollout_ids":
                sources["formal_rollouts"],
            "pb3r2_calibration_rollout_ids":
                sources["calibration_rollouts"],
            "union_count": len(sources["exclusion_union"]),
            "union_rollout_ids": sources["exclusion_union"],
            "reason": (
                "Keep the prospective cohort independent of historical formal "
                "PB3 branch outcomes and PB3-R2 threshold-calibration states."
            ),
        },
        "candidate_queue": {
            "source":
                "prefix-only frozen PB2-C-style candidate ranking",
            "prefix_candidate_count":
                queue_info["prefix_candidate_count"],
            "historical_survivor_subset_validation":
                queue_info["survivor_subset_validation"],
            "selection_used_full_rollout_survival":
                False,
            "selection_used_final_reward":
                False,
            "eligible_candidate_count":
                queue_info["eligible_candidate_count"],
            "ranking_rule":
                config["candidate_queue"]["ranking_rule"],
            "source_rank_preserved_after_exclusion": True,
            "future_information_used": False,
        },
        "live_collection": {
            "batch_indices":
                config["live_collection"]["batch_indices"],
            "n_envs": int(config["live_collection"]["n_envs"]),
            "n_steps_sub": int(
                config["live_collection"]["n_steps_sub"]
            ),
            "base_seed": int(
                config["live_collection"]["base_seed"]
            ),
            "wrapping_args":
                config["live_collection"]["expected_wrapping_args"],
            "common_official_best_qpos": True,
            "reuse_original_pb2c_replay_batch": True,
        },
        "live_admission": {
            "history_samples": int(obs["history_samples"]),
            "branch_validity": (
                "finite selection fields and no rope-NaN/stretch failure "
                "at or before the candidate branch time"
            ),
            "post_branch_validity_used": False,
            "partial_rope": {
                "occlusion_radius_m": float(occlusion_radius(pb2c)),
                "max_visible_history_chamfer_m": float(
                    rope["max_visible_history_chamfer_m"]
                ),
                "require_nonempty_visible_rope_each_frame": True,
            },
            "robot": {
                "max_dual_ee_position_history_mean_m": float(
                    robot["max_dual_ee_position_history_mean_m"]
                ),
                "max_dual_ee_quaternion_geodesic_history_mean_rad":
                    float(
                        robot[
                            "max_dual_ee_quaternion_geodesic_history_mean_rad"
                        ]
                    ),
                "max_dual_motor_qpos_history_rms_rad": float(
                    robot["max_dual_motor_qpos_history_rms_rad"]
                ),
            },
            "hidden_state": {
                "descriptor":
                    "rounded_signed_task_native_winding_index",
                "require_live_index_vector_different": True,
            },
            "same_time_required": True,
            "common_action_history_equal_by_construction": True,
            "future_divergence_used": False,
            "force_or_sensor_used": False,
            "reward_used": False,
            "historical_frozen_to_live_alignment_used_as_gate": False,
            "pb3r3_61_945756736um_rule_used_as_gate": False,
        },
        "cohort_selection": {
            "target_pair_count": int(
                config["cohort"]["target_pair_count"]
            ),
            "max_rollout_use_count": int(
                config["cohort"]["max_rollout_use_count"]
            ),
            "scan_rule": config["cohort"]["scan_rule"],
            "stop_after_first_target_pair_count": True,
            "minimum_live_winding_strata_to_proceed": int(
                config["cohort"][
                    "minimum_live_winding_strata_to_proceed"
                ]
            ),
            "continue_scanning_to_rescue_strata_after_first_10": False,
            "allow_pair_drop": False,
            "allow_pair_replacement": False,
        },
        "phase_boundary": {
            "branch_snapshot_created": False,
            "future_suffix_executed": False,
            "gate4_executed": False,
            "pb4_started": False,
            "training_started": False,
            "next_if_success": (
                "PB3-B2: revalidate the frozen prospective cohort live, then "
                "run the unchanged snapshot same-action multi-horizon Gate 4."
            ),
        },
    }
    return protocol


def freeze_protocol(config_path):
    config = load_json(config_path)
    sources = load_sources(config)
    queue_info = recompute_prefix_candidate_ranking(
        config,
        sources,
    )
    protocol = build_protocol(
        config,
        sources,
        queue_info,
    )

    output = REPO_ROOT / config["outputs"]["protocol"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(protocol, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"verdict={PROTOCOL_VERDICT}")
    print(
        "prefix_candidate_count="
        f"{queue_info['prefix_candidate_count']}"
    )
    print(
        "eligible_candidate_count="
        f"{queue_info['eligible_candidate_count']}"
    )
    print(f"protocol={output}")
    return protocol


def load_and_validate_protocol(config, sources, queue_info):
    path = REPO_ROOT / config["outputs"]["protocol"]
    if not path.is_file():
        raise FileNotFoundError(str(path))

    actual = load_json(path)
    expected = build_protocol(
        config,
        sources,
        queue_info,
    )
    if actual != expected:
        raise RuntimeError(
            "Committed PB3-B1 protocol differs from the deterministic "
            "pre-GPU derivation"
        )
    return actual


def validate_qpos_against_frozen(sources):
    qpos_path = official_log_dir() / "best_qpos.npy"
    if not qpos_path.is_file():
        raise FileNotFoundError(str(qpos_path))

    qpos = normalize_qpos(np.load(qpos_path))
    frozen_qpos = np.asarray(
        sources["frozen_data"]["common_qpos_replay"]
    )
    if not np.array_equal(qpos, frozen_qpos):
        raise RuntimeError(
            "Official best_qpos differs from frozen PB2-C common_qpos_replay"
        )
    return qpos


def collect_fresh_live_data(config, qpos):
    n_envs = int(config["live_collection"]["n_envs"])
    n_steps_sub = int(
        config["live_collection"]["n_steps_sub"]
    )

    env = build_env(
        n_envs=n_envs,
        n_steps_sub=n_steps_sub,
        log_dir=official_log_dir(),
    )
    env.init_domain_randomization(**wrapping_args)

    batch_rows = []
    rollout_ids = []
    batch_indices = []
    env_indices = []
    seeds = []

    try:
        for batch_index in config["live_collection"]["batch_indices"]:
            batch_index = int(batch_index)
            seed = (
                int(config["live_collection"]["base_seed"])
                + batch_index
            )
            row = replay_batch(
                env,
                qpos,
                seed,
                float(
                    config["live_collection"][
                        "official_stretch_ratio_limit"
                    ]
                ),
            )
            batch_rows.append(row)

            for env_index in range(n_envs):
                rollout_ids.append(
                    batch_index * n_envs + env_index
                )
                batch_indices.append(batch_index)
                env_indices.append(env_index)
                seeds.append(seed)
    finally:
        env.stop()

    merged = {
        key: np.concatenate(
            [row[key] for row in batch_rows],
            axis=0,
        )
        for key in batch_rows[0]
    }
    merged["rollout_id"] = np.asarray(
        rollout_ids,
        dtype=np.int32,
    )
    merged["batch_index"] = np.asarray(
        batch_indices,
        dtype=np.int16,
    )
    merged["env_index"] = np.asarray(
        env_indices,
        dtype=np.int16,
    )
    merged["batch_seed"] = np.asarray(
        seeds,
        dtype=np.int32,
    )
    merged["common_qpos_replay"] = np.asarray(qpos).copy()

    ids = np.asarray(merged["rollout_id"], dtype=np.int64)
    if len(ids) != 128 or len(np.unique(ids)) != 128:
        raise RuntimeError(
            "Fresh live collection must contain 128 unique rollouts"
        )

    return merged


def atomic_save_npz(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(
            handle,
            **data,
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def rollout_index_lookup(data):
    ids = [
        int(v)
        for v in data["rollout_id"]
    ]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Live rollout IDs are not unique")
    return {
        rollout_id: index
        for index, rollout_id in enumerate(ids)
    }


def valid_through_branch(data, row_index, time_index):
    failed_nan = bool(
        data["official_failed_rope_nan"][row_index]
    )
    failed_stretch = bool(
        data["official_failed_stretch"][row_index]
    )
    first_fail = int(
        data["official_first_fail_step"][row_index]
    )

    if (
        (failed_nan or failed_stretch)
        and first_fail <= int(time_index)
    ):
        return False

    return True


def winding_stratum(index_a, index_b):
    a = ",".join(
        str(int(v))
        for v in index_a
    )
    b = ",".join(
        str(int(v))
        for v in index_b
    )
    return "<->".join(sorted((a, b)))


def prepare_live_screen_cache(data, pb2c_config):
    radius = float(occlusion_radius(pb2c_config))
    visible = compute_visibility_mask(
        np.asarray(data["rope_xyz"], dtype=np.float32),
        np.asarray(data["post_xyz"], dtype=np.float32),
        radius,
    )
    winding = signed_winding_index(
        np.asarray(
            data["signed_winding_turns"],
            dtype=np.float64,
        )
    )
    return {
        "visible": visible,
        "winding_index": winding,
        "rollout_lookup": rollout_index_lookup(data),
        "occlusion_radius_m": radius,
    }


def evaluate_live_candidate(
        candidate,
        live_data,
        pb2c_config,
        cache):
    ia = cache["rollout_lookup"][
        int(candidate["rollout_a"])
    ]
    ib = cache["rollout_lookup"][
        int(candidate["rollout_b"])
    ]
    time_index = int(candidate["time_index"])

    obs = pb2c_config["partial_state_observation"]
    history = int(obs["history_samples"])
    start = time_index - history + 1

    if start < 0:
        raise RuntimeError(
            "Candidate branch time is shorter than the frozen history window"
        )

    valid_a = valid_through_branch(
        live_data,
        ia,
        time_index,
    )
    valid_b = valid_through_branch(
        live_data,
        ib,
        time_index,
    )

    keys = (
        "rope_xyz",
        "post_xyz",
        "ee1_pos",
        "ee1_quat",
        "ee2_pos",
        "ee2_quat",
        "motor_qpos_1",
        "motor_qpos_2",
        "signed_winding_turns",
    )
    finite = True
    for key in keys:
        a_values = np.asarray(
            live_data[key][ia, start:time_index + 1]
        )
        b_values = np.asarray(
            live_data[key][ib, start:time_index + 1]
        )
        if (
            not np.isfinite(a_values).all()
            or not np.isfinite(b_values).all()
        ):
            finite = False
            break

    branch_valid = bool(valid_a and valid_b and finite)

    index_a = (
        cache["winding_index"][ia, time_index]
    )
    index_b = (
        cache["winding_index"][ib, time_index]
    )
    hidden_different = bool(
        np.any(index_a != index_b)
    )

    visible = cache["visible"]
    counts_a = visible[
        ia,
        start:time_index + 1,
    ].sum(axis=1)
    counts_b = visible[
        ib,
        start:time_index + 1,
    ].sum(axis=1)
    nonempty = bool(
        np.all(counts_a > 0)
        and np.all(counts_b > 0)
    )

    robot = robot_history_metrics(
        live_data,
        np.asarray([ia], dtype=np.int64),
        np.asarray([ib], dtype=np.int64),
        time_index,
        history,
    )
    ee_pos = float(
        robot["dual_ee_position_history_mean_m"][0]
    )
    quat = float(
        robot[
            "dual_ee_quaternion_geodesic_history_mean_rad"
        ][0]
    )
    qpos = float(
        robot["dual_motor_qpos_history_rms_rad"][0]
    )

    rope_cfg = obs["rope_component"]
    robot_cfg = obs["robot_component"]

    chamfer_values = []
    if nonempty:
        for frame in range(start, time_index + 1):
            value = batch_symmetric_chamfer(
                np.asarray(
                    live_data["rope_xyz"][ia, frame],
                    dtype=np.float32,
                )[None],
                visible[ia, frame][None],
                np.asarray(
                    live_data["rope_xyz"][ib, frame],
                    dtype=np.float32,
                )[None],
                visible[ib, frame][None],
            )
            chamfer_values.append(float(value[0]))

    visible_history_chamfer = (
        float(np.mean(chamfer_values))
        if chamfer_values
        else float("inf")
    )

    passes = {
        "branch_valid_through_time": branch_valid,
        "live_winding_index_different": hidden_different,
        "nonempty_visible_rope_history": nonempty,
        "ee_position": bool(
            ee_pos
            <= float(
                robot_cfg[
                    "max_dual_ee_position_history_mean_m"
                ]
            )
        ),
        "ee_quaternion": bool(
            quat
            <= float(
                robot_cfg[
                    "max_dual_ee_quaternion_geodesic_history_mean_rad"
                ]
            )
        ),
        "motor_qpos": bool(
            qpos
            <= float(
                robot_cfg[
                    "max_dual_motor_qpos_history_rms_rad"
                ]
            )
        ),
        "visible_rope_history_chamfer": bool(
            visible_history_chamfer
            <= float(
                rope_cfg["max_visible_history_chamfer_m"]
            )
        ),
    }
    valid = bool(all(passes.values()))

    return {
        "valid": valid,
        "passes": passes,
        "live_metrics": {
            "visible_rope_history_chamfer_m":
                visible_history_chamfer,
            "dual_ee_position_history_mean_m": ee_pos,
            "dual_ee_quaternion_geodesic_history_mean_rad":
                quat,
            "dual_motor_qpos_history_rms_rad": qpos,
            "winding_index_a": [
                int(v)
                for v in index_a
            ],
            "winding_index_b": [
                int(v)
                for v in index_b
            ],
            "winding_stratum": winding_stratum(
                index_a,
                index_b,
            ),
            "visible_vertex_counts_a": [
                int(v)
                for v in counts_a
            ],
            "visible_vertex_counts_b": [
                int(v)
                for v in counts_b
            ],
        },
    }


def scan_live_cohort(
        config,
        sources,
        protocol,
        queue_info,
        live_data):
    target = int(
        config["cohort"]["target_pair_count"]
    )
    max_use = int(
        config["cohort"]["max_rollout_use_count"]
    )
    if max_use != 1:
        raise RuntimeError(
            "PB3-B1 currently requires max_rollout_use_count=1"
        )

    cache = prepare_live_screen_cache(
        live_data,
        sources["pb2c_config"],
    )

    used_rollouts = set()
    accepted = []
    scan_log = []

    funnel = {
        "eligible_queue_candidates": int(
            queue_info["eligible_candidate_count"]
        ),
        "scanned_until_stop": 0,
        "rejected_rollout_already_used": 0,
        "rejected_branch_invalid_through_time": 0,
        "rejected_live_hidden_same": 0,
        "rejected_empty_visible_history": 0,
        "rejected_ee_position": 0,
        "rejected_ee_quaternion": 0,
        "rejected_motor_qpos": 0,
        "rejected_visible_chamfer": 0,
        "live_valid_before_rollout_uniqueness": 0,
        "accepted_pairs": 0,
    }

    for candidate in queue_info["eligible_candidates"]:
        if len(accepted) == target:
            break

        funnel["scanned_until_stop"] += 1

        rollout_a = int(candidate["rollout_a"])
        rollout_b = int(candidate["rollout_b"])

        if (
            rollout_a in used_rollouts
            or rollout_b in used_rollouts
        ):
            funnel["rejected_rollout_already_used"] += 1
            scan_log.append(
                {
                    "source_rank": int(candidate["source_rank"]),
                    "rollout_a": rollout_a,
                    "rollout_b": rollout_b,
                    "time_index": int(candidate["time_index"]),
                    "decision": "reject_rollout_already_used",
                }
            )
            continue

        evaluated = evaluate_live_candidate(
            candidate,
            live_data,
            sources["pb2c_config"],
            cache,
        )

        if not evaluated["valid"]:
            passes = evaluated["passes"]
            if not passes["branch_valid_through_time"]:
                reason = "reject_branch_invalid_through_time"
                funnel[
                    "rejected_branch_invalid_through_time"
                ] += 1
            elif not passes["live_winding_index_different"]:
                reason = "reject_live_hidden_same"
                funnel["rejected_live_hidden_same"] += 1
            elif not passes["nonempty_visible_rope_history"]:
                reason = "reject_empty_visible_history"
                funnel["rejected_empty_visible_history"] += 1
            elif not passes["ee_position"]:
                reason = "reject_ee_position"
                funnel["rejected_ee_position"] += 1
            elif not passes["ee_quaternion"]:
                reason = "reject_ee_quaternion"
                funnel["rejected_ee_quaternion"] += 1
            elif not passes["motor_qpos"]:
                reason = "reject_motor_qpos"
                funnel["rejected_motor_qpos"] += 1
            else:
                reason = "reject_visible_chamfer"
                funnel["rejected_visible_chamfer"] += 1

            scan_log.append(
                {
                    "source_rank": int(candidate["source_rank"]),
                    "rollout_a": rollout_a,
                    "rollout_b": rollout_b,
                    "time_index": int(candidate["time_index"]),
                    "decision": reason,
                    "live_metrics": evaluated["live_metrics"],
                }
            )
            continue

        funnel["live_valid_before_rollout_uniqueness"] += 1

        accepted_row = {
            "cohort_rank": int(len(accepted) + 1),
            "source_rank": int(candidate["source_rank"]),
            "rollout_a": rollout_a,
            "rollout_b": rollout_b,
            "time_index": int(candidate["time_index"]),
            "replay_a": candidate["replay_a"],
            "replay_b": candidate["replay_b"],
            "live_metrics": evaluated["live_metrics"],
            "action_history_equal_by_construction": True,
            "selection_used_future_divergence": False,
            "selection_used_force_or_sensor": False,
            "selection_used_reward": False,
        }
        accepted.append(accepted_row)
        used_rollouts.update((rollout_a, rollout_b))
        funnel["accepted_pairs"] += 1

        scan_log.append(
            {
                "source_rank": int(candidate["source_rank"]),
                "rollout_a": rollout_a,
                "rollout_b": rollout_b,
                "time_index": int(candidate["time_index"]),
                "decision": "accept",
                "cohort_rank": int(len(accepted)),
                "live_metrics": evaluated["live_metrics"],
            }
        )

    strata = sorted(
        {
            row["live_metrics"]["winding_stratum"]
            for row in accepted
        }
    )

    if len(accepted) < target:
        verdict = INSUFFICIENT_VERDICT
    elif len(strata) < int(
        config["cohort"]["minimum_live_winding_strata_to_proceed"]
    ):
        verdict = STRATA_INSUFFICIENT_VERDICT
    else:
        verdict = SUCCESS_VERDICT

    cohort = {
        "phase": "PB3-B1",
        "verdict": verdict,
        "protocol_verdict": protocol["verdict"],
        "selection": {
            "target_pair_count": target,
            "accepted_pair_count": int(len(accepted)),
            "unique_rollout_count": int(len(used_rollouts)),
            "source_rank_scan_preserved": True,
            "first_10_live_valid_rollout_disjoint_pairs": True,
            "continued_after_first_10_to_rescue_strata": False,
            "pair_drop_allowed": False,
            "pair_replacement_allowed": False,
            "future_information_used": False,
        },
        "live_winding_strata": strata,
        "live_winding_strata_count": int(len(strata)),
        "pairs": accepted,
        "phase_boundary": {
            "branch_snapshot_created": False,
            "future_suffix_executed": False,
            "gate4_executed": False,
            "pb4_started": False,
            "training_started": False,
        },
    }

    return cohort, funnel, scan_log


def load_live_npz(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(str(path))
    with np.load(path) as handle:
        return {
            key: np.asarray(handle[key])
            for key in handle.files
        }


def write_screen_outputs(
        config,
        protocol,
        queue_info,
        cohort,
        funnel,
        scan_log):
    raw_root = Path(config["outputs"]["raw_root"])
    raw_root.mkdir(parents=True, exist_ok=True)

    (raw_root / "SCREEN_LOG.json").write_text(
        json.dumps(
            {
                "verdict": cohort["verdict"],
                "funnel": funnel,
                "scan_log": scan_log,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    cohort_path = REPO_ROOT / config["outputs"]["cohort"]
    cohort_path.parent.mkdir(parents=True, exist_ok=True)
    cohort_path.write_text(
        json.dumps(cohort, indent=2) + "\n",
        encoding="utf-8",
    )

    report_dir = (
        REPO_ROOT
        / config["outputs"]["committed_report_dir"]
    )
    report_dir.mkdir(parents=True, exist_ok=True)

    scientific = {
        "verdict": cohort["verdict"],
        "protocol": protocol,
        "prefix_candidate_count":
            queue_info["prefix_candidate_count"],
        "historical_survivor_subset_validation":
            queue_info["survivor_subset_validation"],
        "eligible_candidate_count_after_fixed_exclusions":
            queue_info["eligible_candidate_count"],
        "screen_funnel": funnel,
        "cohort": cohort,
        "non_claims": [
            (
                "PB3-B1 is prospective pair discovery only; it does not "
                "establish same-action future bifurcation."
            ),
            (
                "The artificial partial-rope observation remains a discovery "
                "surrogate and is not claimed deployable."
            ),
            (
                "Historical frozen-to-live reconstruction error is not a "
                "formal PB3-B1 admission gate."
            ),
        ],
    }

    evidence = {
        "phase_name": config["phase_name"],
        "verdict": cohort["verdict"],
        "repository": {
            "starting_main_sha":
                config["provenance"]["starting_main_sha"],
            "ending_main_sha":
                git("rev-parse", "HEAD"),
            "dlolab_gitlink":
                git("rev-parse", "HEAD:external/dlo-lab"),
        },
        "scientific": scientific,
    }
    (report_dir / "EVIDENCE.json").write_text(
        json.dumps(evidence, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# PB3-B1 Prospective Live-Pair Cohort Discovery",
        "",
        f"Verdict: `{cohort['verdict']}`",
        "",
        "## Frozen protocol",
        "",
        f"- Prefix-only frozen candidates: {queue_info['prefix_candidate_count']}",
        (
            "- Historical full-survivor subset regression count: "
            f"{queue_info['survivor_subset_validation']['historical_survivor_candidate_count']}"
        ),
        (
            "- Eligible after historical formal + PB3-R2 calibration "
            f"rollout exclusion: {queue_info['eligible_candidate_count']}"
        ),
        "- Historical formal PB3 rollouts reused: 0",
        "- PB3-R2 calibration rollouts reused: 0",
        "",
        "## Live screening",
        "",
        (
            "- Candidates scanned until stop: "
            f"{funnel['scanned_until_stop']}"
        ),
        (
            "- Accepted pairs: "
            f"{cohort['selection']['accepted_pair_count']}/"
            f"{cohort['selection']['target_pair_count']}"
        ),
        (
            "- Unique accepted rollouts: "
            f"{cohort['selection']['unique_rollout_count']}"
        ),
        (
            "- Live winding strata: "
            f"{cohort['live_winding_strata_count']}"
        ),
        "",
        "## Selection boundary",
        "",
        "- Live history only: Yes",
        "- Future divergence used: No",
        "- Post-branch validity used for admission: No",
        "- Force/sensor used: No",
        "- Reward used: No",
        "- Historical R3 reconstruction threshold used: No",
        "- Branch snapshot created: No",
        "- Future suffix executed: No",
        "- Gate 4 executed: No",
        "",
        "## Next action",
        "",
    ]

    if cohort["verdict"] == SUCCESS_VERDICT:
        lines.append(
            "Commit the frozen prospective cohort and stop. Next phase is "
            "PB3-B2: revalidate these same 10 live pairs in the causal run, "
            "then apply the unchanged snapshot same-action Gate 4."
        )
    else:
        lines.append(
            "Stop. Do not run PB3-B2 or lower live admission thresholds. "
            "This bounded prospective screen did not produce a cohort that "
            "can test the frozen two-stratum Gate-4 phase criterion."
        )

    lines.append("")
    (report_dir / "RESULT.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    return evidence


def screen(config_path):
    config = load_json(config_path)
    sources = load_sources(config)
    queue_info = recompute_prefix_candidate_ranking(
        config,
        sources,
    )
    protocol = load_and_validate_protocol(
        config,
        sources,
        queue_info,
    )
    qpos = validate_qpos_against_frozen(sources)

    live_data = collect_fresh_live_data(
        config,
        qpos,
    )

    raw_root = Path(config["outputs"]["raw_root"])
    live_path = raw_root / "LIVE_SCREEN_ROLLOUTS.npz"
    atomic_save_npz(live_path, live_data)

    cohort, funnel, scan_log = scan_live_cohort(
        config,
        sources,
        protocol,
        queue_info,
        live_data,
    )
    write_screen_outputs(
        config,
        protocol,
        queue_info,
        cohort,
        funnel,
        scan_log,
    )

    print(f"verdict={cohort['verdict']}")
    print(
        "accepted_pair_count="
        f"{cohort['selection']['accepted_pair_count']}"
    )
    print(
        "live_winding_strata_count="
        f"{cohort['live_winding_strata_count']}"
    )
    return cohort


def validate_cohort(config_path):
    config = load_json(config_path)
    sources = load_sources(config)
    queue_info = recompute_prefix_candidate_ranking(
        config,
        sources,
    )
    protocol = load_and_validate_protocol(
        config,
        sources,
        queue_info,
    )

    live_path = (
        Path(config["outputs"]["raw_root"])
        / "LIVE_SCREEN_ROLLOUTS.npz"
    )
    live_data = load_live_npz(live_path)

    expected, _, _ = scan_live_cohort(
        config,
        sources,
        protocol,
        queue_info,
        live_data,
    )

    cohort_path = REPO_ROOT / config["outputs"]["cohort"]
    if not cohort_path.is_file():
        raise FileNotFoundError(str(cohort_path))
    actual = load_json(cohort_path)

    if actual != expected:
        raise RuntimeError(
            "Committed/generated PB3-B1 cohort differs from deterministic "
            "re-screening of the saved live dataset"
        )

    report = (
        REPO_ROOT
        / config["outputs"]["committed_report_dir"]
        / "EVIDENCE.json"
    )
    if not report.is_file():
        raise FileNotFoundError(str(report))
    evidence = load_json(report)
    if evidence.get("verdict") != actual["verdict"]:
        raise RuntimeError(
            "PB3-B1 cohort/evidence verdict mismatch"
        )

    print("PB3-B1 cohort validation: PASS")
    print(f"verdict={actual['verdict']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
    )
    parser.add_argument(
        "command",
        choices=(
            "freeze-protocol",
            "screen",
            "validate-cohort",
        ),
    )
    args = parser.parse_args()

    if args.command == "freeze-protocol":
        freeze_protocol(args.config)
    elif args.command == "screen":
        screen(args.config)
    else:
        validate_cohort(args.config)


if __name__ == "__main__":
    main()
