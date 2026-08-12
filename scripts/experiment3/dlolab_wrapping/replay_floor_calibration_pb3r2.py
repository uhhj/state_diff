"""PB3-R2 — independent replay reconstruction-floor calibration.

This phase measures the replay reconstruction floor on official-valid PB2-C
states that are disjoint from the 20 formal PB3 shortlist rollouts.

It does not:
- run PB3 future suffixes;
- compute Gate 4;
- modify the frozen 50 um PB3 alignment reference;
- modify the PB3 shortlist;
- choose a new alignment threshold.

Calibration selection is deterministic and independent of state values,
future behavior, replay error, winding, reward, force, or sensor data.

Execution is cost-bounded:
    4 batches x 3 fresh processes = 12 fresh Genesis processes.
Each worker replays only through t=20 and evaluates five calibration states.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from scripts.experiment3.dlolab_wrapping.collect_natural_rollouts_pb2c import (
    _initial_validity,
    _update_official_validity,
)
from scripts.experiment3.dlolab_wrapping.paths import (
    REPO_ROOT,
    official_log_dir,
)
from scripts.experiment3.dlolab_wrapping.replay_alignment_root_cause_pb3r1 import (
    _jsonable,
    extract_env,
    robot_error_metrics,
    seed_everything,
    spatial_error_metrics,
)
from scripts.experiment3.dlolab_wrapping.replay_winding_audit import (
    _dual_arm_command,
    build_env,
    load_json,
    normalize_qpos,
    sample_env,
)
from utils.domain_randomization import wrapping_args  # noqa: E402


MODULE = (
    "scripts.experiment3.dlolab_wrapping."
    "replay_floor_calibration_pb3r2"
)


class PB3R2Blocked(RuntimeError):
    def __init__(self, verdict, details):
        super().__init__(verdict)
        self.verdict = verdict
        self.details = details


def git(*args):
    return subprocess.check_output(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
    ).strip()


def validate_source_chain(config):
    pb3_path = REPO_ROOT / config["source"]["pb3_evidence"]
    pb3r1_path = REPO_ROOT / config["source"]["pb3r1_evidence"]

    if not pb3_path.is_file():
        raise FileNotFoundError(str(pb3_path))
    if not pb3r1_path.is_file():
        raise FileNotFoundError(str(pb3r1_path))

    pb3 = load_json(pb3_path)
    pb3r1 = load_json(pb3r1_path)

    if pb3.get("verdict") != config["source"]["expected_pb3_verdict"]:
        raise RuntimeError(
            "PB3 archived verdict mismatch: "
            f"{pb3.get('verdict')} != "
            f"{config['source']['expected_pb3_verdict']}"
        )

    if (
        pb3r1.get("verdict")
        != config["source"]["expected_pb3r1_verdict"]
    ):
        raise RuntimeError(
            "PB3-R1 verdict mismatch: "
            f"{pb3r1.get('verdict')} != "
            f"{config['source']['expected_pb3r1_verdict']}"
        )

    if pb3["scientific"].get("audit") is not None:
        raise RuntimeError(
            "PB3-R2 requires PB3 to remain blocked before Gate 4"
        )

    if pb3["scientific"].get("trajectories_path") is not None:
        raise RuntimeError(
            "PB3-R2 requires no PB3 future trajectory artifact"
        )

    scope = pb3r1["scientific"].get("scope", {})
    if bool(scope.get("future_suffix_executed")):
        raise RuntimeError(
            "PB3-R1 unexpectedly executed future suffix"
        )
    if bool(scope.get("gate4_executed")):
        raise RuntimeError(
            "PB3-R1 unexpectedly executed Gate 4"
        )

    actual_args = _jsonable(dict(wrapping_args))
    expected_args = config["replay"]["expected_wrapping_args"]
    if actual_args != expected_args:
        raise RuntimeError(
            f"Pinned wrapping_args mismatch: {actual_args} != {expected_args}"
        )


def load_frozen(config):
    path = Path(config["source"]["pb2c_raw_rollouts"])
    if not path.is_file():
        raise FileNotFoundError(str(path))

    with np.load(path) as handle:
        frozen = {
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
    missing = sorted(required - set(frozen))
    if missing:
        raise RuntimeError(
            f"PB2-C frozen raw file missing fields: {missing}"
        )

    return frozen


def load_shortlist(config):
    path = REPO_ROOT / config["source"]["pb3_shortlist"]
    if not path.is_file():
        raise FileNotFoundError(str(path))

    shortlist = load_json(path)

    formal_rollouts = sorted(
        {
            int(pair[key])
            for pair in shortlist["pairs"]
            for key in ("rollout_a", "rollout_b")
        }
    )

    if len(shortlist["pairs"]) != 10:
        raise RuntimeError(
            "Expected 10 formal PB3 shortlist pairs"
        )
    if len(formal_rollouts) != 20:
        raise RuntimeError(
            "Expected 20 unique formal PB3 shortlist rollouts"
        )

    return shortlist, formal_rollouts


def load_qpos_and_validate_frozen(frozen):
    qpos_path = official_log_dir() / "best_qpos.npy"
    if not qpos_path.is_file():
        raise FileNotFoundError(str(qpos_path))

    qpos = normalize_qpos(
        np.load(qpos_path)
    )

    if not np.array_equal(
        qpos,
        frozen["common_qpos_replay"],
    ):
        raise RuntimeError(
            "Official best_qpos differs from frozen PB2-C common replay"
        )

    return qpos


def evenly_spaced_positions(n, k):
    if int(k) < 1:
        raise ValueError("k must be positive")
    if int(n) < int(k):
        raise ValueError(
            f"Need n >= k, got n={n}, k={k}"
        )
    if int(k) == 1:
        return [0]

    return [
        (j * (int(n) - 1)) // (int(k) - 1)
        for j in range(int(k))
    ]


def derive_calibration_set(config, frozen, formal_rollouts):
    selection = config["calibration_selection"]
    batch_indices = [
        int(v)
        for v in selection["batch_indices"]
    ]
    per_batch = int(selection["states_per_batch"])

    if batch_indices != [0, 1, 2, 3]:
        raise RuntimeError(
            "PB3-R2 calibration batches must remain [0,1,2,3]"
        )
    if per_batch != 5:
        raise RuntimeError(
            "PB3-R2 states_per_batch must remain 5"
        )

    rollout_ids = np.asarray(
        frozen["rollout_id"],
        dtype=np.int64,
    )
    batch_index = np.asarray(
        frozen["batch_index"],
        dtype=np.int64,
    )
    env_index = np.asarray(
        frozen["env_index"],
        dtype=np.int64,
    )
    valid = np.asarray(
        frozen["official_rollout_valid"],
        dtype=bool,
    )

    formal = set(
        int(v)
        for v in formal_rollouts
    )

    states = []

    even_pattern = [
        int(v)
        for v in selection["time_pattern_even_batch"]
    ]
    odd_pattern = [
        int(v)
        for v in selection["time_pattern_odd_batch"]
    ]

    if len(even_pattern) != per_batch or len(odd_pattern) != per_batch:
        raise RuntimeError(
            "PB3-R2 time patterns must have states_per_batch entries"
        )

    for batch in batch_indices:
        eligible_rows = []

        for index in np.flatnonzero(
                (batch_index == batch) & valid):
            rollout_id = int(
                rollout_ids[index]
            )
            if rollout_id in formal:
                continue

            eligible_rows.append(
                {
                    "frozen_index":
                        int(index),

                    "rollout_id":
                        rollout_id,

                    "batch_index":
                        int(batch),

                    "env_index":
                        int(
                            env_index[index]
                        ),

                    "batch_seed":
                        int(
                            frozen[
                                "batch_seed"
                            ][index]
                        ),
                }
            )

        eligible_rows.sort(
            key=lambda row: row["env_index"]
        )

        positions = evenly_spaced_positions(
            len(eligible_rows),
            per_batch,
        )

        chosen = [
            eligible_rows[position]
            for position in positions
        ]

        pattern = (
            even_pattern
            if batch % 2 == 0
            else odd_pattern
        )

        for slot, (row, target_time) in enumerate(
                zip(chosen, pattern)):
            states.append(
                {
                    "calibration_id":
                        f"pb3r2_b{batch}_s{slot}",

                    "rollout_id":
                        int(
                            row[
                                "rollout_id"
                            ]
                        ),

                    "batch_index":
                        int(batch),

                    "env_index":
                        int(
                            row[
                                "env_index"
                            ]
                        ),

                    "batch_seed":
                        int(
                            row[
                                "batch_seed"
                            ]
                        ),

                    "time_index":
                        int(
                            target_time
                        ),

                    "selection_slot":
                        int(slot),

                    "eligible_order_position":
                        int(
                            positions[slot]
                        ),

                    "eligible_count_in_batch":
                        int(
                            len(
                                eligible_rows
                            )
                        ),
                }
            )

    if len(states) != int(selection["total_states"]):
        raise RuntimeError(
            "Derived calibration state count mismatch"
        )

    if len(
        {
            state["rollout_id"]
            for state in states
        }
    ) != len(states):
        raise RuntimeError(
            "PB3-R2 calibration set reuses a rollout"
        )

    overlap = sorted(
        {
            state["rollout_id"]
            for state in states
        }
        & formal
    )

    if overlap:
        raise RuntimeError(
            f"PB3-R2 calibration overlaps formal PB3 rollouts: {overlap}"
        )

    time_counts = {
        str(time):
            int(
                sum(
                    state[
                        "time_index"
                    ]
                    == time
                    for state in states
                )
            )
        for time in sorted(
            {
                state[
                    "time_index"
                ]
                for state in states
            }
        )
    }

    if time_counts != {
        "13": 10,
        "20": 10,
    }:
        raise RuntimeError(
            f"Expected 10 t13 and 10 t20 states, got {time_counts}"
        )

    return {
        "phase":
            "PB3-R2",

        "selection_rule":
            selection[
                "selection_rule"
            ],

        "selection_used_state_value":
            False,

        "selection_used_replay_error":
            False,

        "selection_used_future":
            False,

        "selection_used_winding":
            False,

        "selection_used_reward_force_or_sensor":
            False,

        "formal_pb3_rollout_ids_excluded":
            formal_rollouts,

        "state_count":
            int(
                len(
                    states
                )
            ),

        "unique_rollout_count":
            int(
                len(
                    states
                )
            ),

        "time_counts":
            time_counts,

        "states":
            states,
    }


def normalized_calibration_set(row):
    return {
        "phase":
            row["phase"],

        "selection_rule":
            row["selection_rule"],

        "selection_used_state_value":
            bool(
                row[
                    "selection_used_state_value"
                ]
            ),

        "selection_used_replay_error":
            bool(
                row[
                    "selection_used_replay_error"
                ]
            ),

        "selection_used_future":
            bool(
                row[
                    "selection_used_future"
                ]
            ),

        "selection_used_winding":
            bool(
                row[
                    "selection_used_winding"
                ]
            ),

        "selection_used_reward_force_or_sensor":
            bool(
                row[
                    "selection_used_reward_force_or_sensor"
                ]
            ),

        "formal_pb3_rollout_ids_excluded":
            [
                int(v)
                for v in row[
                    "formal_pb3_rollout_ids_excluded"
                ]
            ],

        "state_count":
            int(
                row[
                    "state_count"
                ]
            ),

        "unique_rollout_count":
            int(
                row[
                    "unique_rollout_count"
                ]
            ),

        "time_counts":
            {
                str(k):
                    int(v)
                for k, v
                in row[
                    "time_counts"
                ].items()
            },

        "states":
            [
                {
                    "calibration_id":
                        state[
                            "calibration_id"
                        ],

                    "rollout_id":
                        int(
                            state[
                                "rollout_id"
                            ]
                        ),

                    "batch_index":
                        int(
                            state[
                                "batch_index"
                            ]
                        ),

                    "env_index":
                        int(
                            state[
                                "env_index"
                            ]
                        ),

                    "batch_seed":
                        int(
                            state[
                                "batch_seed"
                            ]
                        ),

                    "time_index":
                        int(
                            state[
                                "time_index"
                            ]
                        ),

                    "selection_slot":
                        int(
                            state[
                                "selection_slot"
                            ]
                        ),

                    "eligible_order_position":
                        int(
                            state[
                                "eligible_order_position"
                            ]
                        ),

                    "eligible_count_in_batch":
                        int(
                            state[
                                "eligible_count_in_batch"
                            ]
                        ),
                }
                for state in row[
                    "states"
                ]
            ],
    }


def freeze_set(config_path, output_path):
    config = load_json(config_path)
    validate_source_chain(config)

    frozen = load_frozen(config)
    _, formal_rollouts = load_shortlist(
        config
    )

    expected = derive_calibration_set(
        config,
        frozen,
        formal_rollouts,
    )

    output = Path(output_path)
    if not output.is_absolute():
        output = REPO_ROOT / output

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.write_text(
        json.dumps(
            expected,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        f"calibration_set={output}"
    )
    print(
        f"state_count={expected['state_count']}"
    )
    print(
        f"time_counts={expected['time_counts']}"
    )

    return expected


def load_and_validate_calibration_set(
        config,
        frozen,
        formal_rollouts):
    path = (
        REPO_ROOT
        / config[
            "calibration_selection"
        ][
            "calibration_set_path"
        ]
    )

    if not path.is_file():
        raise FileNotFoundError(
            str(path)
        )

    actual = normalized_calibration_set(
        load_json(
            path
        )
    )

    expected = normalized_calibration_set(
        derive_calibration_set(
            config,
            frozen,
            formal_rollouts,
        )
    )

    if actual != expected:
        raise RuntimeError(
            "Committed PB3-R2 calibration set does not equal the "
            "deterministic frozen selection rule"
        )

    return actual


def frozen_index_lookup(frozen):
    ids = [
        int(v)
        for v in frozen[
            "rollout_id"
        ]
    ]

    if len(set(ids)) != len(ids):
        raise RuntimeError(
            "PB2-C frozen rollout IDs are not unique"
        )

    return {
        rollout_id:
            index
        for index, rollout_id
        in enumerate(ids)
    }



def sample_finiteness(
        live,
        frozen,
        frozen_index,
        time_index):
    """Verify every field used by PB3-R2 reconstruction is finite.

    This is a hard success precondition for every one of the 60 target
    reconstruction samples. Both the fresh replay state and its frozen PB2-C
    reference must be finite.
    """
    live_fields = (
        "rope_xyz",
        "ee1_pos",
        "ee2_pos",
        "motor_qpos_1",
        "motor_qpos_2",
        "signed_winding_turns",
    )

    frozen_fields = live_fields

    live_field_finite = {
        key:
            bool(
                np.isfinite(
                    np.asarray(
                        live[key]
                    )
                ).all()
            )
        for key in live_fields
    }

    frozen_field_finite = {
        key:
            bool(
                np.isfinite(
                    np.asarray(
                        frozen[key][
                            frozen_index,
                            time_index,
                        ]
                    )
                ).all()
            )
        for key in frozen_fields
    }

    live_all = bool(
        all(
            live_field_finite.values()
        )
    )

    frozen_all = bool(
        all(
            frozen_field_finite.values()
        )
    )

    return {
        "live_field_finite":
            live_field_finite,

        "frozen_field_finite":
            frozen_field_finite,

        "live_all_finite":
            live_all,

        "frozen_all_finite":
            frozen_all,

        "all_finite":
            bool(
                live_all
                and frozen_all
            ),
    }


def full_alignment_metrics(
        live,
        frozen,
        frozen_index,
        time_index,
        config):
    spatial = spatial_error_metrics(
        live["rope_xyz"],
        frozen[
            "rope_xyz"
        ][
            frozen_index,
            time_index,
        ],
    )

    robot = robot_error_metrics(
        live,
        frozen,
        frozen_index,
        time_index,
    )

    reference = config[
        "frozen_pb3_alignment_reference"
    ]

    old_50um_rope_reference_pass = bool(
        spatial[
            "rope_max_abs_coordinate_m"
        ]
        <= float(
            reference[
                "rope_max_abs_m"
            ]
        )
    )

    old_full_pb3_alignment_pass = bool(
        old_50um_rope_reference_pass
        and robot[
            "ee_max_abs_m"
        ]
        <= float(
            reference[
                "ee_max_abs_m"
            ]
        )
        and robot[
            "motor_qpos_max_abs_rad"
        ]
        <= float(
            reference[
                "motor_qpos_max_abs_rad"
            ]
        )
        and (
            not reference[
                "require_winding_index_exact"
            ]
            or robot[
                "winding_index_exact"
            ]
        )
    )

    return {
        # Pure rope-only coverage field. This is the only field allowed in
        # the PB3-R2 "50 um coverage" statistic.
        "old_50um_rope_reference_pass":
            old_50um_rope_reference_pass,

        # Full historical PB3 alignment remains available only for t0
        # engineering identity checks and diagnostics.
        "old_full_pb3_alignment_pass":
            old_full_pb3_alignment_pass,

        **spatial,
        **robot,
    }


def worker_batch(
        config_path,
        calibration_set_path,
        batch_index,
        repeat_index,
        output_root):
    config = load_json(
        config_path
    )
    validate_source_chain(
        config
    )

    frozen = load_frozen(
        config
    )
    _, formal_rollouts = load_shortlist(
        config
    )

    calibration_set = (
        load_and_validate_calibration_set(
            config,
            frozen,
            formal_rollouts,
        )
    )

    qpos = load_qpos_and_validate_frozen(
        frozen
    )

    batch_index = int(
        batch_index
    )
    repeat_index = int(
        repeat_index
    )

    states = [
        state
        for state in calibration_set[
            "states"
        ]
        if int(
            state[
                "batch_index"
            ]
        )
        == batch_index
    ]

    if len(states) != 5:
        raise RuntimeError(
            f"Expected 5 calibration states in batch {batch_index}"
        )

    max_target_time = max(
        int(
            state[
                "time_index"
            ]
        )
        for state in states
    )

    if max_target_time > int(
        config[
            "replay"
        ][
            "max_target_time"
        ]
    ):
        raise RuntimeError(
            "Calibration target exceeds frozen max_target_time"
        )

    lookup = frozen_index_lookup(
        frozen
    )

    output_root = Path(
        output_root
    )
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    stem = (
        f"batch{batch_index}_repeat"
        f"{repeat_index}"
    )

    json_path = (
        output_root
        / f"{stem}.json"
    )

    env = build_env(
        n_envs=int(
            config[
                "replay"
            ][
                "n_envs"
            ]
        ),
        n_steps_sub=int(
            config[
                "replay"
            ][
                "n_steps_sub"
            ]
        ),
        log_dir=official_log_dir(),
    )

    env.init_domain_randomization(
        **wrapping_args
    )

    try:
        seed = (
            int(
                config[
                    "replay"
                ][
                    "base_seed"
                ]
            )
            + batch_index
        )

        seed_everything(
            seed
        )

        env.use_qpos = True
        env.reset()

        total_micro_steps = int(
            qpos.shape[0]
            - 1
        )

        validity = _initial_validity(
            env,
            total_micro_steps,
        )

        # HARD PRECONDITION: evaluate t0 immediately after reset, before
        # issuing even one future command. If t0 fails, this worker writes a
        # blocked record and returns without stepping the simulator.
        t0_row = sample_env(
            env
        )

        t0_metrics = []
        t0_failures = []

        for state in states:
            frozen_index = lookup[
                int(
                    state[
                        "rollout_id"
                    ]
                )
            ]

            live = extract_env(
                t0_row,
                int(
                    state[
                        "env_index"
                    ]
                ),
            )

            finite = sample_finiteness(
                live,
                frozen,
                frozen_index,
                0,
            )

            metrics = full_alignment_metrics(
                live,
                frozen,
                frozen_index,
                0,
                config,
            )

            row = {
                "calibration_id":
                    state[
                        "calibration_id"
                    ],

                "rollout_id":
                    int(
                        state[
                            "rollout_id"
                        ]
                    ),

                "finiteness":
                    finite,

                "metrics":
                    metrics,
            }

            t0_metrics.append(
                row
            )

            if (
                not finite[
                    "all_finite"
                ]
                or not metrics[
                    "old_full_pb3_alignment_pass"
                ]
            ):
                t0_failures.append(
                    row
                )

        if t0_failures:
            result = {
                "worker_verdict":
                    "PB3R2_T0_RECONSTRUCTION_FAILED",

                "batch_index":
                    batch_index,

                "repeat_index":
                    repeat_index,

                "fresh_process":
                    True,

                "seed":
                    seed,

                "t0_checked_before_any_future_step":
                    True,

                "future_steps_executed":
                    0,

                "t0_metrics":
                    t0_metrics,

                "target_metrics":
                    [],

                "blocked_details": {
                    "failure_count":
                        int(
                            len(
                                t0_failures
                            )
                        ),

                    "first_failure":
                        t0_failures[
                            0
                        ],
                },
            }

            json_path.write_text(
                json.dumps(
                    result,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            print(
                "worker_verdict="
                "PB3R2_T0_RECONSTRUCTION_FAILED"
            )
            return result

        n_intervals = (
            env.steps_interval
            // env._cmaes_n_steps_sub
        )

        if n_intervals <= 0:
            raise RuntimeError(
                "Invalid Wrapping replay interval count"
            )

        states_by_time = {}

        for state in states:
            states_by_time.setdefault(
                int(
                    state[
                        "time_index"
                    ]
                ),
                [],
            ).append(
                state
            )

        target_metrics = []
        target_ropes = []
        target_ids = []
        target_times = []

        for global_step in range(
                1,
                max_target_time + 1):
            _dual_arm_command(
                env,
                qpos[
                    global_step
                ],
            )

            for _ in range(
                    n_intervals):
                env.scene.step()

            _update_official_validity(
                env,
                validity,
                global_step=
                    global_step,
                stretch_ratio_limit=
                    float(
                        config[
                            "replay"
                        ][
                            "official_stretch_ratio_limit"
                        ]
                    ),
            )

            row = sample_env(
                env
            )

            if global_step not in states_by_time:
                continue

            for state in states_by_time[
                    global_step]:
                env_index = int(
                    state[
                        "env_index"
                    ]
                )

                if not bool(
                    validity[
                        "alive"
                    ][
                        env_index
                    ]
                ):
                    result = {
                        "worker_verdict":
                            "PB3R2_CALIBRATION_REPLAY_INVALID",

                        "batch_index":
                            batch_index,

                        "repeat_index":
                            repeat_index,

                        "fresh_process":
                            True,

                        "seed":
                            seed,

                        "t0_checked_before_any_future_step":
                            True,

                        "future_steps_executed":
                            int(
                                global_step
                            ),

                        "t0_metrics":
                            t0_metrics,

                        "target_metrics":
                            target_metrics,

                        "blocked_details": {
                            "calibration_id":
                                state[
                                    "calibration_id"
                                ],

                            "rollout_id":
                                int(
                                    state[
                                        "rollout_id"
                                    ]
                                ),

                            "time_index":
                                int(
                                    global_step
                                ),
                        },
                    }

                    json_path.write_text(
                        json.dumps(
                            result,
                            indent=2,
                        )
                        + "\n",
                        encoding="utf-8",
                    )

                    print(
                        "worker_verdict="
                        "PB3R2_CALIBRATION_REPLAY_INVALID"
                    )
                    return result

                frozen_index = lookup[
                    int(
                        state[
                            "rollout_id"
                        ]
                    )
                ]

                live = extract_env(
                    row,
                    env_index,
                )

                finite = sample_finiteness(
                    live,
                    frozen,
                    frozen_index,
                    global_step,
                )

                # HARD SUCCESS PRECONDITION: no non-finite target sample is
                # allowed into the 60-sample calibration distribution.
                if not finite[
                    "all_finite"
                ]:
                    result = {
                        "worker_verdict":
                            "PB3R2_TARGET_NONFINITE",

                        "batch_index":
                            batch_index,

                        "repeat_index":
                            repeat_index,

                        "fresh_process":
                            True,

                        "seed":
                            seed,

                        "t0_checked_before_any_future_step":
                            True,

                        "future_steps_executed":
                            int(
                                global_step
                            ),

                        "t0_metrics":
                            t0_metrics,

                        "target_metrics":
                            target_metrics,

                        "blocked_details": {
                            "calibration_id":
                                state[
                                    "calibration_id"
                                ],

                            "rollout_id":
                                int(
                                    state[
                                        "rollout_id"
                                    ]
                                ),

                            "time_index":
                                int(
                                    global_step
                                ),

                            "finiteness":
                                finite,
                        },
                    }

                    json_path.write_text(
                        json.dumps(
                            result,
                            indent=2,
                        )
                        + "\n",
                        encoding="utf-8",
                    )

                    print(
                        "worker_verdict="
                        "PB3R2_TARGET_NONFINITE"
                    )
                    return result

                metrics = full_alignment_metrics(
                    live,
                    frozen,
                    frozen_index,
                    global_step,
                    config,
                )

                # Metrics themselves must also be finite. This is separate
                # from checking the underlying state arrays.
                numeric_metric_names = (
                    "coordinate_rmse_m",
                    "rope_max_abs_coordinate_m",
                    "max_vertex_l2_m",
                    "median_vertex_l2_m",
                    "min_vertex_l2_m",
                    "mean_translation_norm_m",
                    "translation_removed_coordinate_rmse_m",
                    "translation_removed_fraction",
                    "ee_max_abs_m",
                    "motor_qpos_max_abs_rad",
                )

                metrics_all_finite = bool(
                    all(
                        np.isfinite(
                            float(
                                metrics[
                                    name
                                ]
                            )
                        )
                        for name in numeric_metric_names
                    )
                )

                if not metrics_all_finite:
                    result = {
                        "worker_verdict":
                            "PB3R2_TARGET_NONFINITE",

                        "batch_index":
                            batch_index,

                        "repeat_index":
                            repeat_index,

                        "fresh_process":
                            True,

                        "seed":
                            seed,

                        "t0_checked_before_any_future_step":
                            True,

                        "future_steps_executed":
                            int(
                                global_step
                            ),

                        "t0_metrics":
                            t0_metrics,

                        "target_metrics":
                            target_metrics,

                        "blocked_details": {
                            "calibration_id":
                                state[
                                    "calibration_id"
                                ],

                            "rollout_id":
                                int(
                                    state[
                                        "rollout_id"
                                    ]
                                ),

                            "time_index":
                                int(
                                    global_step
                                ),

                            "reason":
                                "derived_metric_nonfinite",
                        },
                    }

                    json_path.write_text(
                        json.dumps(
                            result,
                            indent=2,
                        )
                        + "\n",
                        encoding="utf-8",
                    )

                    print(
                        "worker_verdict="
                        "PB3R2_TARGET_NONFINITE"
                    )
                    return result

                target_metrics.append(
                    {
                        "calibration_id":
                            state[
                                "calibration_id"
                            ],

                        "rollout_id":
                            int(
                                state[
                                    "rollout_id"
                                ]
                            ),

                        "batch_index":
                            batch_index,

                        "env_index":
                            env_index,

                        "time_index":
                            int(
                                global_step
                            ),

                        "repeat_index":
                            repeat_index,

                        "finiteness":
                            finite,

                        "metrics_all_finite":
                            True,

                        "metrics":
                            metrics,
                    }
                )

                target_ropes.append(
                    np.asarray(
                        live[
                            "rope_xyz"
                        ]
                    ).copy()
                )

                target_ids.append(
                    int(
                        state[
                            "rollout_id"
                        ]
                    )
                )

                target_times.append(
                    int(
                        global_step
                    )
                )

        if len(
            target_metrics
        ) != len(
            states
        ):
            raise RuntimeError(
                "Worker did not collect all five calibration targets"
            )

        npz_path = (
            output_root
            / f"{stem}.npz"
        )

        np.savez_compressed(
            npz_path,
            rollout_id=np.asarray(
                target_ids,
                dtype=np.int32,
            ),
            time_index=np.asarray(
                target_times,
                dtype=np.int16,
            ),
            rope_xyz=np.stack(
                target_ropes,
                axis=0,
            ),
        )

        result = {
            "worker_verdict":
                "PB3R2_WORKER_COMPLETE",

            "batch_index":
                batch_index,

            "repeat_index":
                repeat_index,

            "fresh_process":
                True,

            "seed":
                seed,

            "t0_checked_before_any_future_step":
                True,

            "future_steps_executed":
                int(
                    max_target_time
                ),

            "t0_metrics":
                t0_metrics,

            "target_metrics":
                target_metrics,

            "all_target_samples_finite":
                True,

            "npz_path":
                str(
                    npz_path
                ),
        }

        json_path.write_text(
            json.dumps(
                result,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        print(
            json.dumps(
                {
                    "worker_verdict":
                        "PB3R2_WORKER_COMPLETE",

                    "batch_index":
                        batch_index,

                    "repeat_index":
                        repeat_index,

                    "targets":
                        len(
                            target_metrics
                        ),

                    "rope_50um_passes":
                        sum(
                            row[
                                "metrics"
                            ][
                                "old_50um_rope_reference_pass"
                            ]
                            for row in target_metrics
                        ),
                }
            )
        )

        return result

    finally:
        env.stop()


def quantile_summary(values):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if values.size == 0:
        raise ValueError(
            "Cannot summarize empty calibration values"
        )

    return {
        "count":
            int(
                values.size
            ),

        "min":
            float(
                np.min(
                    values
                )
            ),

        "median":
            float(
                np.quantile(
                    values,
                    0.50,
                )
            ),

        "p95":
            float(
                np.quantile(
                    values,
                    0.95,
                )
            ),

        "p99":
            float(
                np.quantile(
                    values,
                    0.99,
                )
            ),

        "max":
            float(
                np.max(
                    values
                )
            ),
    }


def summarize_metric_rows(rows):
    metric_names = (
        "coordinate_rmse_m",
        "rope_max_abs_coordinate_m",
        "max_vertex_l2_m",
        "translation_removed_coordinate_rmse_m",
        "ee_max_abs_m",
        "motor_qpos_max_abs_rad",
    )

    return {
        metric_name:
            quantile_summary(
                [
                    row[
                        "metrics"
                    ][
                        metric_name
                    ]
                    for row in rows
                ]
            )
        for metric_name
        in metric_names
    }


def load_worker_ropes(worker_records):
    rope_map = {}

    for worker in worker_records:
        with np.load(
                worker[
                    "npz_path"
                ]) as handle:
            ids = np.asarray(
                handle[
                    "rollout_id"
                ],
                dtype=np.int64,
            )
            times = np.asarray(
                handle[
                    "time_index"
                ],
                dtype=np.int64,
            )
            ropes = np.asarray(
                handle[
                    "rope_xyz"
                ]
            )

        repeat_index = int(
            worker[
                "repeat_index"
            ]
        )

        for rollout_id, time_index, rope in zip(
                ids,
                times,
                ropes):
            key = (
                int(
                    rollout_id
                ),
                int(
                    time_index
                ),
                repeat_index,
            )

            if key in rope_map:
                raise RuntimeError(
                    f"Duplicate calibration rope key: {key}"
                )

            rope_map[
                key
            ] = np.asarray(
                rope
            ).copy()

    return rope_map


def fresh_repeat_rows(
        calibration_set,
        rope_map,
        repeat_count):
    rows = []

    for state in calibration_set[
            "states"]:
        rollout_id = int(
            state[
                "rollout_id"
            ]
        )
        time_index = int(
            state[
                "time_index"
            ]
        )

        for left, right in itertools.combinations(
                range(
                    int(
                        repeat_count
                    )
                ),
                2):
            rope_left = rope_map[
                (
                    rollout_id,
                    time_index,
                    left,
                )
            ]
            rope_right = rope_map[
                (
                    rollout_id,
                    time_index,
                    right,
                )
            ]

            rows.append(
                {
                    "calibration_id":
                        state[
                            "calibration_id"
                        ],

                    "rollout_id":
                        rollout_id,

                    "batch_index":
                        int(
                            state[
                                "batch_index"
                            ]
                        ),

                    "time_index":
                        time_index,

                    "repeat_pair":
                        [
                            int(left),
                            int(right),
                        ],

                    "metrics":
                        spatial_error_metrics(
                            rope_left,
                            rope_right,
                        ),
                }
            )

    return rows


def argmax_histogram(rows):
    histogram = {}

    for row in rows:
        vertex = str(
            int(
                row[
                    "metrics"
                ][
                    "argmax_vertex"
                ]
            )
        )

        histogram[
            vertex
        ] = (
            histogram.get(
                vertex,
                0,
            )
            + 1
        )

    return dict(
        sorted(
            histogram.items(),
            key=lambda item: (
                -item[
                    1
                ],
                int(
                    item[
                        0
                    ]
                ),
            ),
        )
    )


def grouped_summary(rows, key_name):
    groups = {}

    for row in rows:
        key = str(
            int(
                row[
                    key_name
                ]
            )
        )

        groups.setdefault(
            key,
            [],
        ).append(
            row
        )

    return {
        key:
            summarize_metric_rows(
                group_rows
            )
        for key, group_rows
        in sorted(
            groups.items(),
            key=lambda item: int(
                item[
                    0
                ]
            ),
        )
    }


def write_blocked(
        config,
        verdict,
        details):
    raw_root = Path(
        config[
            "outputs"
        ][
            "raw_root"
        ]
    )
    raw_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    result = {
        "verdict":
            verdict,

        "scope": {
            "future_suffix_executed":
                False,

            "gate4_executed":
                False,

            "pb3_alignment_threshold_modified":
                False,

            "pb3_shortlist_modified":
                False,

            "new_alignment_threshold_selected":
                False,
        },

        "blocked_details":
            details,
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

    committed = (
        REPO_ROOT
        / config[
            "outputs"
        ][
            "committed_report_dir"
        ]
    )
    committed.mkdir(
        parents=True,
        exist_ok=True,
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

    (
        committed
        / "RESULT.md"
    ).write_text(
        (
            "# PB3-R2 Replay Floor Calibration\n\n"
            f"Verdict: `{verdict}`\n\n"
            "No new PB3 alignment threshold was selected.\n\n"
            "```json\n"
            + json.dumps(
                details,
                indent=2,
            )
            + "\n```\n"
        ),
        encoding="utf-8",
    )

    return result


def aggregate(config_path):
    config = load_json(
        config_path
    )
    validate_source_chain(
        config
    )

    frozen = load_frozen(
        config
    )
    _, formal_rollouts = load_shortlist(
        config
    )

    calibration_set = (
        load_and_validate_calibration_set(
            config,
            frozen,
            formal_rollouts,
        )
    )

    load_qpos_and_validate_frozen(
        frozen
    )

    raw_root = Path(
        config[
            "outputs"
        ][
            "raw_root"
        ]
    )
    raw_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    repeat_count = int(
        config[
            "replay"
        ][
            "fresh_process_repeats"
        ]
    )

    worker_records = []

    try:
        for batch_index in config[
                "calibration_selection"
        ][
            "batch_indices"
        ]:
            for repeat_index in range(
                    repeat_count):
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        MODULE,
                        "worker",
                        "--config",
                        str(
                            config_path
                        ),
                        "--calibration-set",
                        str(
                            REPO_ROOT
                            / config[
                                "calibration_selection"
                            ][
                                "calibration_set_path"
                            ]
                        ),
                        "--batch-index",
                        str(
                            batch_index
                        ),
                        "--repeat-index",
                        str(
                            repeat_index
                        ),
                        "--output-root",
                        str(
                            raw_root
                        ),
                    ],
                    cwd=REPO_ROOT,
                    check=True,
                )

                path = (
                    raw_root
                    / (
                        f"batch{batch_index}_repeat"
                        f"{repeat_index}.json"
                    )
                )

                worker_record = load_json(
                    path
                )

                worker_verdict = worker_record.get(
                    "worker_verdict"
                )

                if worker_verdict != "PB3R2_WORKER_COMPLETE":
                    return write_blocked(
                        config,
                        worker_verdict,
                        {
                            "batch_index":
                                int(
                                    batch_index
                                ),

                            "repeat_index":
                                int(
                                    repeat_index
                                ),

                            "t0_checked_before_any_future_step":
                                bool(
                                    worker_record.get(
                                        "t0_checked_before_any_future_step",
                                        False,
                                    )
                                ),

                            "future_steps_executed":
                                int(
                                    worker_record.get(
                                        "future_steps_executed",
                                        0,
                                    )
                                ),

                            "worker_blocked_details":
                                worker_record.get(
                                    "blocked_details"
                                ),
                        },
                    )

                worker_records.append(
                    worker_record
                )

    except subprocess.CalledProcessError as exc:
        return write_blocked(
            config,
            "PB3R2_WORKER_EXECUTION_FAILED",
            {
                "returncode":
                    int(
                        exc.returncode
                    ),
            },
        )

    reconstruction_rows = [
        row
        for worker in worker_records
        for row in worker[
            "target_metrics"
        ]
    ]

    if len(
        reconstruction_rows
    ) != (
        int(
            config[
                "calibration_selection"
            ][
                "total_states"
            ]
        )
        * repeat_count
    ):
        return write_blocked(
            config,
            "PB3R2_CALIBRATION_SAMPLE_COUNT_MISMATCH",
            {
                "actual":
                    int(
                        len(
                            reconstruction_rows
                        )
                    ),
            },
        )

    nonfinite_target_rows = [
        row
        for row in reconstruction_rows
        if (
            not bool(
                row.get(
                    "finiteness",
                    {}
                ).get(
                    "all_finite",
                    False,
                )
            )
            or not bool(
                row.get(
                    "metrics_all_finite",
                    False,
                )
            )
        )
    ]

    if nonfinite_target_rows:
        return write_blocked(
            config,
            "PB3R2_TARGET_NONFINITE",
            {
                "failure_count":
                    int(
                        len(
                            nonfinite_target_rows
                        )
                    ),

                "first_failure":
                    nonfinite_target_rows[
                        0
                    ],
            },
        )

    all_60_target_samples_finite = bool(
        len(
            reconstruction_rows
        )
        == 60
        and not nonfinite_target_rows
    )

    if not all_60_target_samples_finite:
        return write_blocked(
            config,
            "PB3R2_TARGET_FINITE_COUNT_NOT_SATISFIED",
            {
                "target_sample_count":
                    int(
                        len(
                            reconstruction_rows
                        )
                    ),

                "expected":
                    60,
            },
        )

    rope_map = load_worker_ropes(
        worker_records
    )

    repeat_rows = fresh_repeat_rows(
        calibration_set,
        rope_map,
        repeat_count,
    )

    frozen_summary = summarize_metric_rows(
        reconstruction_rows
    )

    repeat_summary = {
        metric_name:
            quantile_summary(
                [
                    row[
                        "metrics"
                    ][
                        metric_name
                    ]
                    for row in repeat_rows
                ]
            )
        for metric_name in (
            "coordinate_rmse_m",
            "rope_max_abs_coordinate_m",
            "max_vertex_l2_m",
            "translation_removed_coordinate_rmse_m",
        )
    }

    old_reference = float(
        config[
            "frozen_pb3_alignment_reference"
        ][
            "rope_max_abs_m"
        ]
    )

    old_pass_count = int(
        sum(
            row[
                "metrics"
            ][
                "old_50um_rope_reference_pass"
            ]
            for row in reconstruction_rows
        )
    )

    reconstruction_count = int(
        len(
            reconstruction_rows
        )
    )

    old_pass_rate = float(
        old_pass_count
        / reconstruction_count
    )

    result = {
        "verdict":
            "PB3R2_REPLAY_FLOOR_CALIBRATED",

        "scope": {
            "future_suffix_executed":
                False,

            "gate4_executed":
                False,

            "pb3_alignment_threshold_modified":
                False,

            "pb3_shortlist_modified":
                False,

            "new_alignment_threshold_selected":
                False,

            "calibration_independent_of_formal_pb3_rollouts":
                True,
        },

        "calibration_set":
            calibration_set,

        "sample_counts": {
            "calibration_states":
                int(
                    calibration_set[
                        "state_count"
                    ]
                ),

            "fresh_process_repeats_per_state":
                repeat_count,

            "frozen_reconstruction_samples":
                reconstruction_count,

            "fresh_repeat_pair_samples":
                int(
                    len(
                        repeat_rows
                    )
                ),

            "fresh_genesis_processes":
                int(
                    len(
                        worker_records
                    )
                ),
        },

        "frozen_reconstruction_floor":
            frozen_summary,

        "fresh_process_repeat_floor":
            repeat_summary,

        "target_finiteness": {
            "expected_target_sample_count":
                60,

            "actual_target_sample_count":
                reconstruction_count,

            "all_60_target_samples_finite":
                all_60_target_samples_finite,
        },

        "original_50um_rope_reference": {
            "definition":
                "rope_max_abs_coordinate_m <= 5e-5 m only",

            "rope_max_abs_m":
                old_reference,

            "pass_count":
                old_pass_count,

            "sample_count":
                reconstruction_count,

            "pass_rate":
                old_pass_rate,

            "exceedance_count":
                int(
                    reconstruction_count
                    - old_pass_count
                ),
        },

        "by_time":
            grouped_summary(
                reconstruction_rows,
                "time_index",
            ),

        "by_batch":
            grouped_summary(
                reconstruction_rows,
                "batch_index",
            ),

        "argmax_vertex_histogram":
            argmax_histogram(
                reconstruction_rows
            ),

        "interpretation_boundary": {
            "pb3r2_only_measures_replay_floor":
                True,

            "does_not_select_new_threshold":
                True,

            "next_step":
                (
                    "Use only this independent calibration evidence to "
                    "pre-register PB3-R3 alignment handling. Do not run the "
                    "frozen PB3 future audit until that rule is committed."
                ),
        },
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

    committed = (
        REPO_ROOT
        / config[
            "outputs"
        ][
            "committed_report_dir"
        ]
    )
    committed.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        committed
        / "CALIBRATION_SUMMARY.json"
    ).write_text(
        json.dumps(
            {
                "verdict":
                    result[
                        "verdict"
                    ],

                "sample_counts":
                    result[
                        "sample_counts"
                    ],

                "frozen_reconstruction_floor":
                    result[
                        "frozen_reconstruction_floor"
                    ],

                "fresh_process_repeat_floor":
                    result[
                        "fresh_process_repeat_floor"
                    ],

                "target_finiteness":
                    result[
                        "target_finiteness"
                    ],

                "original_50um_rope_reference":
                    result[
                        "original_50um_rope_reference"
                    ],

                "by_time":
                    result[
                        "by_time"
                    ],

                "by_batch":
                    result[
                        "by_batch"
                    ],

                "argmax_vertex_histogram":
                    result[
                        "argmax_vertex_histogram"
                    ],
            },
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
            result[
                "verdict"
            ],

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

    rope_summary = result[
        "frozen_reconstruction_floor"
    ][
        "rope_max_abs_coordinate_m"
    ]

    repeat_rope_summary = result[
        "fresh_process_repeat_floor"
    ][
        "rope_max_abs_coordinate_m"
    ]

    lines = [
        "# PB3-R2 Independent Replay Floor Calibration",
        "",
        "Verdict: `PB3R2_REPLAY_FLOOR_CALIBRATED`",
        "",
        "## Scope",
        "",
        "- Formal PB3 remains blocked: Yes",
        "- Future suffix executed: No",
        "- Gate 4 executed: No",
        "- Frozen 50 um alignment reference modified: No",
        "- New alignment threshold selected: No",
        "- Formal PB3 shortlist modified: No",
        "",
        "## Calibration design",
        "",
        (
            "- Calibration states: "
            f"{result['sample_counts']['calibration_states']}"
        ),
        (
            "- Formal-shortlist overlap: 0 rollouts"
        ),
        (
            "- Fresh-process repeats/state: "
            f"{repeat_count}"
        ),
        (
            "- Frozen reconstruction samples: "
            f"{reconstruction_count}"
        ),
        (
            "- Fresh-process repeat-pair samples: "
            f"{len(repeat_rows)}"
        ),
        (
            "- Fresh Genesis processes: "
            f"{len(worker_records)}"
        ),
        (
            "- All 60 target samples finite: "
            f"{all_60_target_samples_finite}"
        ),
        "",
        "## Frozen reconstruction — rope max abs",
        "",
        (
            "- median: "
            f"{rope_summary['median']:.9e} m"
        ),
        (
            "- P95: "
            f"{rope_summary['p95']:.9e} m"
        ),
        (
            "- P99: "
            f"{rope_summary['p99']:.9e} m"
        ),
        (
            "- max: "
            f"{rope_summary['max']:.9e} m"
        ),
        "",
        "## Fresh-process repeat — rope max abs",
        "",
        (
            "- median: "
            f"{repeat_rope_summary['median']:.9e} m"
        ),
        (
            "- P95: "
            f"{repeat_rope_summary['p95']:.9e} m"
        ),
        (
            "- P99: "
            f"{repeat_rope_summary['p99']:.9e} m"
        ),
        (
            "- max: "
            f"{repeat_rope_summary['max']:.9e} m"
        ),
        "",
        "## Original 50 um rope-only reference",
        "",
        (
            "- Pass rate: "
            f"{old_pass_count}/{reconstruction_count} "
            f"({100.0 * old_pass_rate:.2f}%)"
        ),
        (
            "- Exceedances: "
            f"{reconstruction_count - old_pass_count}"
        ),
        "",
        "## Next action",
        "",
        (
            "Use this independent calibration distribution to pre-register "
            "PB3-R3 alignment handling. Do not choose the new rule from the "
            "formal 10-pair PB3 outcomes, and do not resume Gate 4 before "
            "that rule is committed."
        ),
        "",
    ]

    (
        committed
        / "RESULT.md"
    ).write_text(
        "\n".join(
            lines
        ),
        encoding="utf-8",
    )

    print(
        "verdict=PB3R2_REPLAY_FLOOR_CALIBRATED"
    )
    print(
        "old_50um_pass_rate="
        f"{old_pass_count}/{reconstruction_count}"
    )

    return result


def main():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    freeze = subparsers.add_parser(
        "freeze-set"
    )
    freeze.add_argument(
        "--config",
        required=True,
    )
    freeze.add_argument(
        "--output",
        required=True,
    )

    worker = subparsers.add_parser(
        "worker"
    )
    worker.add_argument(
        "--config",
        required=True,
    )
    worker.add_argument(
        "--calibration-set",
        required=True,
    )
    worker.add_argument(
        "--batch-index",
        type=int,
        required=True,
    )
    worker.add_argument(
        "--repeat-index",
        type=int,
        required=True,
    )
    worker.add_argument(
        "--output-root",
        required=True,
    )

    run = subparsers.add_parser(
        "run"
    )
    run.add_argument(
        "--config",
        required=True,
    )

    args = parser.parse_args()

    if args.command == "freeze-set":
        freeze_set(
            args.config,
            args.output,
        )

    elif args.command == "worker":
        # --calibration-set is accepted explicitly for transparent invocation;
        # the file path is also frozen in config and validated there.
        expected_path = (
            REPO_ROOT
            / load_json(
                args.config
            )[
                "calibration_selection"
            ][
                "calibration_set_path"
            ]
        )

        if Path(
            args.calibration_set
        ).resolve() != expected_path.resolve():
            raise RuntimeError(
                "--calibration-set does not match frozen config path"
            )

        worker_batch(
            args.config,
            args.calibration_set,
            args.batch_index,
            args.repeat_index,
            args.output_root,
        )

    elif args.command == "run":
        aggregate(
            args.config
        )


if __name__ == "__main__":
    main()
