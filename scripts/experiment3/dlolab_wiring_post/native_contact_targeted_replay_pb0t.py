"""PB0-T: targeted simulator-native contact confirmation for Wiring-post.

Purpose
-------
Replay only the REV4 shortlisted rollout states and query DLO-Lab's native ROD
collision state. No benchmark physics, reward, geometry, pair thresholds, or
model code is changed.

Native signals used
-------------------
1) ROD <-> Rigid:
   ``rope._solver.vertices_collision`` stores per-vertex ``collided``,
   ``penetration`` and ``geom_idx``. DLO-Lab's own dataset generator reads the
   same fields for rope-world collision diagnostics.

2) ROD <-> ROD:
   ``rope._solver.rr_constraints`` stores the rod-rod collision constraint
   penetration for valid edge pairs. Wiring-post contains fixed hidden ROD
   extensions for both posts, so this captures that published-task collider
   path without inventing a new geometric proxy.

The phase qualifies a pair when the native simulator contact signatures of
state A and state B differ at the original current state (t=18). Agreement
with the old 3 mm proxy orientation is diagnostic only and never controls
qualification.

The t=17..19 pattern and per-command scene-step occupancy are supporting
diagnostics only. `stable_core_support` requires the same unequal A/B native
contact direction on the same post across the full core window; it is
independent of the proxy orientation and is not a new formal gate.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiment3.dlolab_wiring_post.collect_rollouts import (  # noqa: E402
    build_env,
    normalize_qpos,
    to_numpy,
)
from scripts.experiment3.dlolab_wiring_post.paths import (  # noqa: E402
    official_log_dir,
)

# collect_rollouts already installs the pinned DLO-Lab import path.
import genesis as gs  # noqa: E402
from utils.domain_randomization import wiring_post_args  # noqa: E402


def load_json(path):
    return json.loads(
        Path(path).read_text(
            encoding="utf-8"
        )
    )


def git(*args):
    return subprocess.check_output(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
    ).strip()


def entity_geom_indices(entity):
    indices = []
    for link in entity.links:
        for geom in link._geoms:
            indices.append(
                int(geom.idx)
            )
    return sorted(set(indices))


def build_native_metadata(env):
    solver = env.rope._solver

    collision = solver.vertices_collision
    for field_name in (
            "collided",
            "penetration",
            "geom_idx"):
        if not hasattr(
                collision,
                field_name):
            raise RuntimeError(
                "Pinned DLO-Lab native ROD-rigid collision field "
                "is missing: {}".format(field_name)
            )

    post_geom_indices = [
        entity_geom_indices(env.stick1),
        entity_geom_indices(env.stick2),
    ]

    if any(
            not values
            for values
            in post_geom_indices):
        raise RuntimeError(
            "Could not map Wiring-post rigid cylinders to rigid geom indices"
        )

    if set(post_geom_indices[0]) & set(post_geom_indices[1]):
        raise RuntimeError(
            "Wiring-post rigid post geom indices overlap unexpectedly"
        )

    rr_pairs = np.asarray(
        solver.rr_constraint_info.valid_pair.to_numpy(),
        dtype=np.int64,
    )
    rod_ids = np.asarray(
        solver.vertices_info.rod_idx.to_numpy(),
        dtype=np.int64,
    ).reshape(-1)

    if rr_pairs.ndim != 2 or rr_pairs.shape[1] != 2:
        raise RuntimeError(
            "Unexpected rr_constraint_info.valid_pair shape: {}".format(
                rr_pairs.shape
            )
        )

    pair_rod_ids = rod_ids[rr_pairs]

    main_rod_id = int(
        env.rope._rod_idx
    )
    hidden_rod_ids = [
        int(env.stick1_hidden._rod_idx),
        int(env.stick2_hidden._rod_idx),
    ]

    rr_post_masks = []
    for hidden_rod_id in hidden_rod_ids:
        mask = (
            (
                pair_rod_ids[:, 0]
                == main_rod_id
            )
            & (
                pair_rod_ids[:, 1]
                == hidden_rod_id
            )
        ) | (
            (
                pair_rod_ids[:, 1]
                == main_rod_id
            )
            & (
                pair_rod_ids[:, 0]
                == hidden_rod_id
            )
        )
        rr_post_masks.append(mask)

    if any(
            not np.any(mask)
            for mask
            in rr_post_masks):
        raise RuntimeError(
            "Could not map main rope <-> hidden-post ROD edge pairs"
        )

    rope_v_start = int(
        env.rope._v_start
    )
    rope_n_vertices = int(
        env.rope.n_vertices
    )

    return {
        "post_geom_indices":
            post_geom_indices,

        "rr_pairs":
            rr_pairs,

        "pair_rod_ids":
            pair_rod_ids,

        "rr_post_masks":
            rr_post_masks,

        "main_rod_id":
            main_rod_id,

        "hidden_rod_ids":
            hidden_rod_ids,

        "rope_v_start":
            rope_v_start,

        "rope_n_vertices":
            rope_n_vertices,
    }


def _rigid_collision_arrays(
        env,
        metadata):
    collision = (
        env.rope._solver
        .vertices_collision
    )

    collided = np.asarray(
        collision.collided.to_numpy()
    ).T.astype(bool, copy=False)

    geom_idx = np.asarray(
        collision.geom_idx.to_numpy()
    ).T.astype(np.int64, copy=False)

    penetration = np.asarray(
        collision.penetration.to_numpy()
    ).T.astype(np.float64, copy=False)

    v_start = int(
        metadata["rope_v_start"]
    )
    n_vertices = int(
        metadata["rope_n_vertices"]
    )
    sl = slice(
        v_start,
        v_start + n_vertices,
    )

    return (
        collided[:, sl],
        geom_idx[:, sl],
        penetration[:, sl],
    )


def _rod_rod_penetration_arrays(
        env,
        metadata):
    raw = np.asarray(
        env.rope._solver
        .rr_constraints
        .penetration
        .to_numpy(),
        dtype=np.float64,
    )

    n_pairs = int(
        metadata["rr_pairs"].shape[0]
    )
    n_envs = int(
        env.n_envs
    )

    # The pinned ROD solver constructs rr_constraints with
    # shape (substep_frame, valid_edge_pair, batch).
    if (
        raw.ndim != 3
        or raw.shape[1] != n_pairs
        or raw.shape[2] != n_envs
    ):
        raise RuntimeError(
            "Unexpected rr_constraints.penetration shape: {} "
            "(expected [frame, {}, {}])".format(
                raw.shape,
                n_pairs,
                n_envs,
            )
        )

    # process_input() clears contact state for the step. Taking the maximum
    # over the solver's local substep frames gives this scene.step()'s native
    # rod-rod constraint penetration.
    return np.max(
        raw,
        axis=0,
    ).T


def _main_rope_local_edge_ids(
        active_pair_ids,
        metadata):
    rr_pairs = metadata[
        "rr_pairs"
    ]
    pair_rod_ids = metadata[
        "pair_rod_ids"
    ]
    main_rod_id = int(
        metadata["main_rod_id"]
    )
    v_start = int(
        metadata["rope_v_start"]
    )

    local_edges = []
    for pair_id in active_pair_ids:
        pair_id = int(pair_id)
        pair = rr_pairs[pair_id]
        rods = pair_rod_ids[pair_id]

        if int(rods[0]) == main_rod_id:
            rope_edge_start = int(
                pair[0]
            )
        elif int(rods[1]) == main_rod_id:
            rope_edge_start = int(
                pair[1]
            )
        else:
            continue

        local_edges.append(
            rope_edge_start
            - v_start
        )

    return sorted(set(local_edges))


def read_native_step(
        env,
        metadata,
        env_indices):
    (
        collided,
        geom_idx,
        rigid_penetration,
    ) = _rigid_collision_arrays(
        env,
        metadata,
    )

    rr_penetration = (
        _rod_rod_penetration_arrays(
            env,
            metadata,
        )
    )

    result = {}

    for env_index in env_indices:
        env_index = int(
            env_index
        )
        posts = []

        for post_index in range(2):
            rigid_ids = metadata[
                "post_geom_indices"
            ][post_index]

            rigid_mask = (
                collided[env_index]
                & np.isin(
                    geom_idx[env_index],
                    rigid_ids,
                )
            )

            rigid_local_vertices = (
                np.flatnonzero(
                    rigid_mask
                )
                .astype(int)
                .tolist()
            )

            rigid_max_penetration = (
                float(
                    np.max(
                        rigid_penetration[
                            env_index
                        ][rigid_mask]
                    )
                )
                if np.any(
                    rigid_mask
                )
                else 0.0
            )

            rr_mask = metadata[
                "rr_post_masks"
            ][post_index]

            rr_values = (
                rr_penetration[
                    env_index,
                    rr_mask
                ]
            )

            rr_active_local = (
                np.flatnonzero(
                    rr_values > 0.0
                )
            )

            rr_global_pair_ids = (
                np.flatnonzero(
                    rr_mask
                )[
                    rr_active_local
                ]
            )

            rr_local_edges = (
                _main_rope_local_edge_ids(
                    rr_global_pair_ids,
                    metadata,
                )
            )

            rr_max_penetration = (
                float(
                    np.max(
                        rr_values
                    )
                )
                if rr_values.size
                else 0.0
            )

            rigid_any = bool(
                np.any(
                    rigid_mask
                )
            )
            rr_any = bool(
                np.any(
                    rr_values > 0.0
                )
            )

            posts.append(
                {
                    "rigid_post_collision":
                        rigid_any,

                    "hidden_rod_collision":
                        rr_any,

                    "combined_native_post_contact":
                        bool(
                            rigid_any
                            or rr_any
                        ),

                    "rigid_collision_vertex_ids":
                        rigid_local_vertices,

                    "hidden_rod_collision_rope_edge_ids":
                        rr_local_edges,

                    "rigid_max_penetration_m":
                        rigid_max_penetration,

                    "hidden_rod_max_constraint_penetration_m":
                        rr_max_penetration,
                }
            )

        result[str(env_index)] = {
            "posts":
                posts
        }

    return result


def init_interval_aggregate():
    return {
        "scene_steps":
            0,

        "posts": [
            {
                "rigid_contact_scene_steps":
                    0,

                "hidden_rod_contact_scene_steps":
                    0,

                "combined_contact_scene_steps":
                    0,

                "rigid_max_penetration_m":
                    0.0,

                "hidden_rod_max_constraint_penetration_m":
                    0.0,

                "rigid_collision_vertex_ids":
                    set(),

                "hidden_rod_collision_rope_edge_ids":
                    set(),
            }
            for _ in range(2)
        ],
    }


def update_interval_aggregate(
        aggregate,
        step_record):
    aggregate[
        "scene_steps"
    ] += 1

    for post_index, post in enumerate(
            step_record[
                "posts"
            ]):
        target = aggregate[
            "posts"
        ][post_index]

        if post[
            "rigid_post_collision"
        ]:
            target[
                "rigid_contact_scene_steps"
            ] += 1

        if post[
            "hidden_rod_collision"
        ]:
            target[
                "hidden_rod_contact_scene_steps"
            ] += 1

        if post[
            "combined_native_post_contact"
        ]:
            target[
                "combined_contact_scene_steps"
            ] += 1

        target[
            "rigid_max_penetration_m"
        ] = max(
            float(
                target[
                    "rigid_max_penetration_m"
                ]
            ),
            float(
                post[
                    "rigid_max_penetration_m"
                ]
            ),
        )

        target[
            "hidden_rod_max_constraint_penetration_m"
        ] = max(
            float(
                target[
                    "hidden_rod_max_constraint_penetration_m"
                ]
            ),
            float(
                post[
                    "hidden_rod_max_constraint_penetration_m"
                ]
            ),
        )

        target[
            "rigid_collision_vertex_ids"
        ].update(
            post[
                "rigid_collision_vertex_ids"
            ]
        )

        target[
            "hidden_rod_collision_rope_edge_ids"
        ].update(
            post[
                "hidden_rod_collision_rope_edge_ids"
            ]
        )


def finalize_interval_aggregate(
        aggregate):
    n = int(
        aggregate[
            "scene_steps"
        ]
    )
    posts = []

    for post in aggregate[
            "posts"]:
        posts.append(
            {
                "rigid_contact_scene_steps":
                    int(
                        post[
                            "rigid_contact_scene_steps"
                        ]
                    ),

                "hidden_rod_contact_scene_steps":
                    int(
                        post[
                            "hidden_rod_contact_scene_steps"
                        ]
                    ),

                "combined_contact_scene_steps":
                    int(
                        post[
                            "combined_contact_scene_steps"
                        ]
                    ),

                "combined_contact_scene_step_fraction":
                    (
                        float(
                            post[
                                "combined_contact_scene_steps"
                            ]
                            / n
                        )
                        if n
                        else None
                    ),

                "rigid_max_penetration_m":
                    float(
                        post[
                            "rigid_max_penetration_m"
                        ]
                    ),

                "hidden_rod_max_constraint_penetration_m":
                    float(
                        post[
                            "hidden_rod_max_constraint_penetration_m"
                        ]
                    ),

                "rigid_collision_vertex_ids":
                    sorted(
                        int(v)
                        for v
                        in post[
                            "rigid_collision_vertex_ids"
                        ]
                    ),

                "hidden_rod_collision_rope_edge_ids":
                    sorted(
                        int(v)
                        for v
                        in post[
                            "hidden_rod_collision_rope_edge_ids"
                        ]
                    ),
            }
        )

    return {
        "scene_steps":
            n,

        "posts":
            posts,
    }


def replay_locator(
        rollout_id,
        n_envs,
        base_seed):
    rollout_id = int(
        rollout_id
    )
    n_envs = int(
        n_envs
    )
    batch = (
        rollout_id
        // n_envs
    )
    env_index = (
        rollout_id
        % n_envs
    )
    return {
        "rollout_id":
            rollout_id,

        "batch_index":
            int(batch),

        "env_index":
            int(env_index),

        "batch_seed":
            int(
                base_seed
            )
            + int(
                batch
            ),
    }


def selected_rollouts(
        shortlist,
        pb0_cfg):
    n_envs = int(
        pb0_cfg[
            "collection"
        ][
            "n_envs"
        ]
    )
    base_seed = int(
        pb0_cfg[
            "collection"
        ][
            "base_seed"
        ]
    )

    rollout_ids = set()

    for candidate in shortlist[
            "candidates"]:
        rollout_ids.add(
            int(
                candidate[
                    "rollout_a"
                ]
            )
        )
        rollout_ids.add(
            int(
                candidate[
                    "rollout_b"
                ]
            )
        )

    return {
        rollout_id:
            replay_locator(
                rollout_id,
                n_envs,
                base_seed,
            )
        for rollout_id
        in sorted(
            rollout_ids
        )
    }


def group_by_batch(
        locators):
    groups = defaultdict(
        list
    )
    for rollout_id, locator in (
            locators.items()):
        groups[
            int(
                locator[
                    "batch_index"
                ]
            )
        ].append(
            int(
                rollout_id
            )
        )

    return {
        batch:
            sorted(
                rollout_ids
            )
        for batch, rollout_ids
        in sorted(
            groups.items()
        )
    }


def _control_command(
        env,
        command):
    command_batch = np.repeat(
        np.asarray(
            command,
            dtype=np.float32,
        )[None, :],
        env.n_envs,
        axis=0,
    )
    command_tc = torch.tensor(
        command_batch,
        dtype=gs.tc_float,
    )

    env.c1.robot.control_dofs_position(
        command_tc[
            ...,
            :-2
        ],
        env.c1.motors_dof,
    )

    env.c1.robot.control_dofs_position(
        command_tc[
            ...,
            -2:
        ],
        env.c1.fingers_dof,
    )


def run_targeted_replay(
        env,
        qpos,
        *,
        batch_index,
        rollout_ids,
        pb0_cfg,
        phase_cfg,
        metadata,
        frozen_rope):
    n_envs = int(
        pb0_cfg[
            "collection"
        ][
            "n_envs"
        ]
    )
    base_seed = int(
        pb0_cfg[
            "collection"
        ][
            "base_seed"
        ]
    )

    seed = (
        base_seed
        + int(
            batch_index
        )
    )

    np.random.seed(
        int(
            seed
        )
    )
    torch.manual_seed(
        int(
            seed
        )
    )

    env.use_qpos = True
    env.reset()

    window = set(
        int(v)
        for v
        in phase_cfg[
            "replay"
        ][
            "window_samples"
        ]
    )
    stop_after = int(
        phase_cfg[
            "replay"
        ][
            "stop_after_sample"
        ]
    )
    tolerance = float(
        phase_cfg[
            "replay"
        ][
            "replay_match_max_abs_tol_m"
        ]
    )

    env_indices = [
        int(
            rollout_id
            % n_envs
        )
        for rollout_id
        in rollout_ids
    ]

    records = {
        int(
            rollout_id
        ): {
            "locator":
                replay_locator(
                    rollout_id,
                    n_envs,
                    base_seed,
                ),

            "samples":
                {},

            "replay_match_max_abs_m":
                0.0,

            "replay_match_within_tolerance":
                True,
        }
        for rollout_id
        in rollout_ids
    }

    n_intervals = (
        env.steps_interval
        // env._cmaes_n_steps_sub
    )

    if n_intervals <= 0:
        raise RuntimeError(
            "Invalid replay n_intervals"
        )

    for sample_index in range(
            1,
            stop_after + 1):
        _control_command(
            env,
            qpos[
                sample_index
            ],
        )

        interval_by_env = {
            env_index:
                init_interval_aggregate()
            for env_index
            in env_indices
        }

        last_step = None

        for _ in range(
                n_intervals):
            env.scene.step()

            if sample_index in window:
                last_step = read_native_step(
                    env,
                    metadata,
                    env_indices,
                )

                for env_index in env_indices:
                    update_interval_aggregate(
                        interval_by_env[
                            env_index
                        ],
                        last_step[
                            str(
                                env_index
                            )
                        ],
                    )

        if sample_index not in window:
            continue

        if last_step is None:
            last_step = read_native_step(
                env,
                metadata,
                env_indices,
            )

        rope_xyz = to_numpy(
            env.rope.get_all_verts()
        ).astype(
            np.float64,
            copy=False,
        )

        for rollout_id in rollout_ids:
            env_index = int(
                rollout_id
                % n_envs
            )

            replay_state = rope_xyz[
                env_index
            ]

            frozen_state = np.asarray(
                frozen_rope[
                    rollout_id,
                    sample_index
                ],
                dtype=np.float64,
            )

            max_abs = float(
                np.max(
                    np.abs(
                        replay_state
                        - frozen_state
                    )
                )
            )

            record = records[
                int(
                    rollout_id
                )
            ]

            record[
                "replay_match_max_abs_m"
            ] = max(
                float(
                    record[
                        "replay_match_max_abs_m"
                    ]
                ),
                max_abs,
            )

            if max_abs > tolerance:
                record[
                    "replay_match_within_tolerance"
                ] = False

            record[
                "samples"
            ][
                str(
                    sample_index
                )
            ] = {
                "replay_vs_frozen_max_abs_m":
                    max_abs,

                "end":
                    last_step[
                        str(
                            env_index
                        )
                    ],

                "interval":
                    finalize_interval_aggregate(
                        interval_by_env[
                            env_index
                        ]
                    ),
            }

    return records


def merge_record_dicts(
        target,
        source):
    for key, value in source.items():
        if key in target:
            raise RuntimeError(
                "Duplicate targeted rollout record: {}".format(
                    key
                )
            )
        target[key] = value


def native_post_bool(
        records,
        rollout_id,
        sample_index,
        post_index):
    return bool(
        records[
            int(
                rollout_id
            )
        ][
            "samples"
        ][
            str(
                int(
                    sample_index
                )
            )
        ][
            "end"
        ][
            "posts"
        ][
            int(
                post_index
            )
        ][
            "combined_native_post_contact"
        ]
    )


def _proxy_orientation_by_post(
        candidate,
        contact_threshold_m):
    result = {}
    for mismatch in candidate[
            "hidden_robustness"
        ][
            "mismatch_posts"]:
        post_index = int(
            mismatch[
                "post_index"
            ]
        )
        result[
            post_index
        ] = {
            "proxy_contact_a":
                bool(
                    float(
                        mismatch[
                            "clearance_a_m"
                        ]
                    )
                    <= float(
                        contact_threshold_m
                    )
                ),

            "proxy_contact_b":
                bool(
                    float(
                        mismatch[
                            "clearance_b_m"
                        ]
                    )
                    <= float(
                        contact_threshold_m
                    )
                ),

            "clearance_a_m":
                float(
                    mismatch[
                        "clearance_a_m"
                    ]
                ),

            "clearance_b_m":
                float(
                    mismatch[
                        "clearance_b_m"
                    ]
                ),
        }

    return result


def _stable_native_direction(
        native_pairs):
    """
    Supporting diagnostic only.

    True iff the same post has an unequal native contact state across every
    core sample AND the A/B direction is unchanged across the whole core
    window, e.g. (True, False) at 17, 18 and 19.

    This is intentionally independent of the old 3 mm proxy orientation.
    """
    if not native_pairs:
        return False

    first = tuple(
        bool(v)
        for v
        in native_pairs[0]
    )

    if first[0] == first[1]:
        return False

    return all(
        tuple(
            bool(v)
            for v
            in pair
        )
        == first
        for pair
        in native_pairs
    )


def pair_confirmation(
        candidate,
        records,
        *,
        contact_threshold_m,
        target_sample,
        core_samples):
    """
    Confirm simulator-native hidden contact mismatch.

    Critical rule:
    ----------------
    A native mismatch is defined only from the simulator-native contact
    signatures of state A and state B. It does NOT need to agree with the old
    geometric proxy's A/B direction.

    The old proxy is retained only as a retrospective diagnostic:
      - which post(s) originally triggered candidate mining;
      - whether native contact happens to agree with that old orientation.

    A mismatch on either published Wiring-post post is sufficient to establish
    a simulator-native current-state contact difference, provided exact replay
    is valid.
    """
    a = int(
        candidate[
            "rollout_a"
        ]
    )
    b = int(
        candidate[
            "rollout_b"
        ]
    )

    replay_valid = bool(
        records[a][
            "replay_match_within_tolerance"
        ]
        and records[b][
            "replay_match_within_tolerance"
        ]
    )

    proxy_by_post = _proxy_orientation_by_post(
        candidate,
        contact_threshold_m,
    )

    post_results = []

    # Evaluate BOTH published Wiring-post posts independently of which post
    # happened to trigger the old geometric proxy.
    for post_index in range(2):
        native_a = native_post_bool(
            records,
            a,
            target_sample,
            post_index,
        )
        native_b = native_post_bool(
            records,
            b,
            target_sample,
            post_index,
        )

        target_native_mismatch = bool(
            native_a
            != native_b
        )

        proxy = proxy_by_post.get(
            post_index
        )

        if proxy is None:
            proxy_orientation_agreement = None
        else:
            proxy_orientation_agreement = bool(
                native_a
                == proxy[
                    "proxy_contact_a"
                ]
                and native_b
                == proxy[
                    "proxy_contact_b"
                ]
            )

        core_pattern = []
        native_pairs = []

        for sample in core_samples:
            sample_a = native_post_bool(
                records,
                a,
                sample,
                post_index,
            )
            sample_b = native_post_bool(
                records,
                b,
                sample,
                post_index,
            )

            native_pairs.append(
                (
                    sample_a,
                    sample_b,
                )
            )

            core_pattern.append(
                {
                    "sample":
                        int(
                            sample
                        ),

                    "native_contact_a":
                        sample_a,

                    "native_contact_b":
                        sample_b,

                    "native_mismatch":
                        bool(
                            sample_a
                            != sample_b
                        ),

                    "proxy_orientation_agreement":
                        (
                            None
                            if proxy is None
                            else bool(
                                sample_a
                                == proxy[
                                    "proxy_contact_a"
                                ]
                                and sample_b
                                == proxy[
                                    "proxy_contact_b"
                                ]
                            )
                        ),
                }
            )

        persistent_core_mismatch_any_direction = bool(
            core_pattern
            and all(
                row[
                    "native_mismatch"
                ]
                for row
                in core_pattern
            )
        )

        stable_native_mismatch_same_direction = (
            _stable_native_direction(
                native_pairs
            )
        )

        sample_a = records[a][
            "samples"
        ][
            str(
                target_sample
            )
        ]
        sample_b = records[b][
            "samples"
        ][
            str(
                target_sample
            )
        ]

        post_results.append(
            {
                "post_index":
                    post_index,

                "original_proxy_implicated_post":
                    bool(
                        proxy is not None
                    ),

                "proxy_contact_a":
                    (
                        None
                        if proxy is None
                        else proxy[
                            "proxy_contact_a"
                        ]
                    ),

                "proxy_contact_b":
                    (
                        None
                        if proxy is None
                        else proxy[
                            "proxy_contact_b"
                        ]
                    ),

                "proxy_clearance_a_m":
                    (
                        None
                        if proxy is None
                        else proxy[
                            "clearance_a_m"
                        ]
                    ),

                "proxy_clearance_b_m":
                    (
                        None
                        if proxy is None
                        else proxy[
                            "clearance_b_m"
                        ]
                    ),

                "native_contact_a":
                    native_a,

                "native_contact_b":
                    native_b,

                "target_native_mismatch":
                    target_native_mismatch,

                # Diagnostic only. Never used for qualification.
                "proxy_orientation_agreement_at_target":
                    proxy_orientation_agreement,

                # Supporting temporal diagnostics, independent of proxy.
                "persistent_core_mismatch_any_direction":
                    persistent_core_mismatch_any_direction,

                "stable_native_mismatch_same_direction":
                    stable_native_mismatch_same_direction,

                "core_pattern":
                    core_pattern,

                "target_interval_fraction_a":
                    sample_a[
                        "interval"
                    ][
                        "posts"
                    ][
                        post_index
                    ][
                        "combined_contact_scene_step_fraction"
                    ],

                "target_interval_fraction_b":
                    sample_b[
                        "interval"
                    ][
                        "posts"
                    ][
                        post_index
                    ][
                        "combined_contact_scene_step_fraction"
                    ],

                "target_end_sources_a": {
                    "rigid":
                        sample_a[
                            "end"
                        ][
                            "posts"
                        ][
                            post_index
                        ][
                            "rigid_post_collision"
                        ],

                    "hidden_rod":
                        sample_a[
                            "end"
                        ][
                            "posts"
                        ][
                            post_index
                        ][
                            "hidden_rod_collision"
                        ],
                },

                "target_end_sources_b": {
                    "rigid":
                        sample_b[
                            "end"
                        ][
                            "posts"
                        ][
                            post_index
                        ][
                            "rigid_post_collision"
                        ],

                    "hidden_rod":
                        sample_b[
                            "end"
                        ][
                            "posts"
                        ][
                            post_index
                        ][
                            "hidden_rod_collision"
                        ],
                },
            }
        )

    native_mismatch_confirmed = bool(
        replay_valid
        and any(
            row[
                "target_native_mismatch"
            ]
            for row
            in post_results
        )
    )

    stable_support = bool(
        native_mismatch_confirmed
        and any(
            row[
                "stable_native_mismatch_same_direction"
            ]
            for row
            in post_results
        )
    )

    if not replay_valid:
        status = (
            "REPLAY_ALIGNMENT_INVALID"
        )
    elif native_mismatch_confirmed:
        status = (
            "NATIVE_POST_CONTACT_MISMATCH_CONFIRMED_AT_TARGET"
        )
    else:
        status = (
            "NATIVE_POST_CONTACT_MISMATCH_NOT_CONFIRMED"
        )

    proxy_agreement_posts = [
        int(
            row[
                "post_index"
            ]
        )
        for row
        in post_results
        if row[
            "proxy_orientation_agreement_at_target"
        ] is True
    ]

    proxy_disagreement_posts = [
        int(
            row[
                "post_index"
            ]
        )
        for row
        in post_results
        if row[
            "proxy_orientation_agreement_at_target"
        ] is False
    ]

    native_mismatch_posts = [
        int(
            row[
                "post_index"
            ]
        )
        for row
        in post_results
        if row[
            "target_native_mismatch"
        ]
    ]

    stable_native_mismatch_posts = [
        int(
            row[
                "post_index"
            ]
        )
        for row
        in post_results
        if row[
            "stable_native_mismatch_same_direction"
        ]
    ]

    return {
        "rollout_a":
            a,

        "rollout_b":
            b,

        "time_index":
            int(
                candidate[
                    "time_index"
                ]
            ),

        "observable_chamfer_m":
            float(
                candidate[
                    "observable_chamfer_m"
                ]
            ),

        "ee_position_distance_m":
            float(
                candidate[
                    "ee_position_distance_m"
                ]
            ),

        "replay_valid":
            replay_valid,

        "status":
            status,

        "native_mismatch_confirmed_at_target":
            native_mismatch_confirmed,

        "native_mismatch_post_indices":
            native_mismatch_posts,

        "stable_core_support":
            stable_support,

        "stable_native_mismatch_post_indices":
            stable_native_mismatch_posts,

        # Old proxy agreement is diagnostic only.
        "proxy_orientation_agreement_post_indices":
            proxy_agreement_posts,

        "proxy_orientation_disagreement_post_indices":
            proxy_disagreement_posts,

        "post_results":
            post_results,
    }


def build_confirmed_shortlist(
        pair_results,
        k,
        *,
        global_replay_alignment_valid):
    """
    Build the PB1 shortlist only when the targeted replay is globally valid.

    PB0-T treats any selected-rollout alignment failure as an experiment-level
    validity failure. In that case no native contact result may be promoted,
    even if some individual pairs happen to involve only aligned rollouts.
    """
    if not bool(
            global_replay_alignment_valid):
        return {
            "selection_uses_future_divergence":
                False,

            "selection_uses_robot_force_or_sensor_signal":
                False,

            "selection_uses_native_current_contact":
                True,

            "global_replay_alignment_valid":
                False,

            "blocked_reason":
                "PB0T_TARGETED_REPLAY_ALIGNMENT_FAILED",

            "purpose":
                (
                    "No PB1 candidate is emitted because the formal "
                    "targeted replay failed experiment-level alignment."
                ),

            "candidates":
                [],
        }

    confirmed = [
        row
        for row
        in pair_results
        if row[
            "native_mismatch_confirmed_at_target"
        ]
    ]

    confirmed.sort(
        key=lambda row: (
            not row[
                "stable_core_support"
            ],
            row[
                "observable_chamfer_m"
            ],
            row[
                "ee_position_distance_m"
            ],
        )
    )

    return {
        "selection_uses_future_divergence":
            False,

        "selection_uses_robot_force_or_sensor_signal":
            False,

        "selection_uses_native_current_contact":
            True,

        "global_replay_alignment_valid":
            True,

        "blocked_reason":
            None,

        "purpose":
            (
                "Candidates qualified for PB1-A snapshot "
                "same-action multi-horizon causal audit."
            ),

        "candidates":
            confirmed[
                :int(
                    k
                )
            ],
    }


def build_report(
        phase_cfg,
        pb0_cfg,
        metadata,
        records,
        pair_results,
        confirmed_shortlist):
    replay_invalid = [
        rollout_id
        for rollout_id, record
        in records.items()
        if not record[
            "replay_match_within_tolerance"
        ]
    ]

    global_replay_alignment_valid = bool(
        not replay_invalid
    )

    if (
        not global_replay_alignment_valid
        and confirmed_shortlist[
            "candidates"
        ]
    ):
        raise RuntimeError(
            "PB0-T invariant violated: alignment-failed run "
            "must not emit a confirmed shortlist"
        )

    confirmed_count = int(
        sum(
            row[
                "native_mismatch_confirmed_at_target"
            ]
            for row
            in pair_results
        )
    )

    stable_count = int(
        sum(
            row[
                "stable_core_support"
            ]
            for row
            in pair_results
        )
    )

    if replay_invalid:
        verdict = (
            "PB0T_TARGETED_REPLAY_ALIGNMENT_FAILED"
        )
        next_action = (
            "Do not interpret or promote any native-contact result "
            "from this formal run. CONFIRMED_SHORTLIST must be empty. "
            "Fix exact replay reproduction first; do not loosen the "
            "frozen replay alignment tolerance automatically."
        )
    elif confirmed_count > 0:
        verdict = (
            "PB0T_NATIVE_POST_CONTACT_MISMATCH_CONFIRMED"
        )
        next_action = (
            "Proceed to PB1-A using only the confirmed shortlist. "
            "Restore/branch from each qualified state, impose an "
            "identical future command, and compare pair divergence "
            "against deterministic repeat/uncertainty at multiple "
            "horizons. Do not reuse future/current>=2."
        )
    else:
        verdict = (
            "PB0T_NATIVE_POST_CONTACT_MISMATCH_NOT_CONFIRMED"
        )
        next_action = (
            "Do not send REV4 contact-proxy-only pairs to Gate 4. "
            "The targeted published-task replay did not find a "
            "simulator-native post-contact mismatch between the two "
            "current states. Stop treating the 3 mm "
            "proxy as ground truth. From a project-efficiency "
            "standpoint, move to another published task such as "
            "DLO-Lab Wrapping rather than adding more geometric "
            "proxy engineering."
        )

    result = {
        "verdict":
            verdict,

        "formal_pb0_verdict":
            phase_cfg[
                "source"
            ][
                "formal_pb0_verdict"
            ],

        "formal_pb0_verdict_changed":
            False,

        "new_targeted_simulation":
            True,

        "new_large_scale_collection":
            False,

        "model_training_started":
            False,

        "native_signal_contract": {
            "rod_rigid":
                (
                    "RODSolver vertices_collision: collided, "
                    "penetration, rigid geom_idx"
                ),

            "rod_rod":
                (
                    "RODSolver rr_constraints penetration for "
                    "main-rope <-> fixed hidden-post ROD pairs"
                ),

            "post_geom_indices":
                metadata[
                    "post_geom_indices"
                ],

            "main_rod_id":
                metadata[
                    "main_rod_id"
                ],

            "hidden_post_rod_ids":
                metadata[
                    "hidden_rod_ids"
                ],
        },

        "targeted_rollout_count":
            int(
                len(
                    records
                )
            ),

        "targeted_pair_count":
            int(
                len(
                    pair_results
                )
            ),

        "replay_alignment_tolerance_m":
            float(
                phase_cfg[
                    "replay"
                ][
                    "replay_match_max_abs_tol_m"
                ]
            ),

        "global_replay_alignment_valid":
            global_replay_alignment_valid,

        "replay_invalid_rollout_ids":
            [
                int(v)
                for v
                in replay_invalid
            ],

        "native_confirmed_pair_count":
            confirmed_count,

        "native_confirmed_with_stable_core_support_count":
            stable_count,

        "pair_results":
            pair_results,

        "confirmed_shortlist":
            confirmed_shortlist,

        "targeted_records":
            records,

        "next_action":
            next_action,

        "non_claims": [
            (
                "The original 3 mm geometric proxy is not "
                "redefined as simulator-native ground truth."
            ),
            (
                "The t=17..19 stability pattern is supporting "
                "evidence only, not a formal gate."
            ),
            (
                "Per-command scene-step contact occupancy is "
                "supporting evidence only."
            ),
            (
                "No future divergence is used to qualify the "
                "current hidden-state difference."
            ),
        ],
    }

    raw_root = Path(
        phase_cfg[
            "outputs"
        ][
            "raw_root"
        ]
    )
    raw_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        raw_root
        / "NATIVE_CONTACT_TARGETED_REPLAY.json"
    ).write_text(
        json.dumps(
            result,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    (
        raw_root
        / "CONFIRMED_SHORTLIST.json"
    ).write_text(
        json.dumps(
            confirmed_shortlist,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    committed = (
        REPO_ROOT
        / phase_cfg[
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
            phase_cfg[
                "phase_name"
            ],

        "verdict":
            verdict,

        "repository": {
            "starting_main_sha":
                phase_cfg[
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
        / "CONFIRMED_SHORTLIST.json"
    ).write_text(
        json.dumps(
            confirmed_shortlist,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    lines = [
        "# PB0-T Wiring-post Simulator-Native Contact Confirmation",
        "",
        "Verdict: `{}`".format(
            verdict
        ),
        "",
        "## Scope",
        "",
        "- Targeted simulation only: Yes",
        "- Large-scale recollection: No",
        "- Model training: No",
        "- Formal PB0 verdict changed: No",
        "",
        "## Native signal",
        "",
        "- ROD-rigid: `vertices_collision.collided / penetration / geom_idx`",
        "- ROD-ROD hidden post: `rr_constraints.penetration`",
        "",
        "## Replay",
        "",
        "- Unique targeted rollouts: {}".format(
            len(records)
        ),
        "- Targeted pairs: {}".format(
            len(pair_results)
        ),
        "- Global replay alignment valid: {}".format(
            global_replay_alignment_valid
        ),
        "- Replay-invalid rollouts: {}".format(
            len(replay_invalid)
        ),
        "- Confirmed shortlist blocked by alignment failure: {}".format(
            bool(
                replay_invalid
                and not confirmed_shortlist[
                    "candidates"
                ]
            )
        ),
        "- Alignment tolerance (engineering only): {:.9g} m".format(
            phase_cfg[
                "replay"
            ][
                "replay_match_max_abs_tol_m"
            ]
        ),
        "",
        "## Native confirmation",
        "",
        "- Confirmed at target sample: {}".format(
            confirmed_count
        ),
        "- Also stable across core t=17..19: {}".format(
            stable_count
        ),
        "- Confirmed shortlist size: {}".format(
            len(
                confirmed_shortlist[
                    "candidates"
                ]
            )
        ),
        "",
        "## Next action",
        "",
        next_action,
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

    return result


def run(config_path):
    phase_cfg = load_json(
        config_path
    )

    pb0_cfg = load_json(
        REPO_ROOT
        / phase_cfg[
            "source"
        ][
            "pb0_config"
        ]
    )

    shortlist = load_json(
        REPO_ROOT
        / phase_cfg[
            "source"
        ][
            "rev4_shortlist"
        ]
    )

    expected = int(
        phase_cfg[
            "source"
        ][
            "expected_shortlist_count"
        ]
    )

    if len(
            shortlist[
                "candidates"
            ]
    ) != expected:
        raise RuntimeError(
            "Unexpected REV4 shortlist size: {} != {}".format(
                len(
                    shortlist[
                        "candidates"
                    ]
                ),
                expected,
            )
        )

    dataset_path = Path(
        phase_cfg[
            "source"
        ][
            "frozen_dataset"
        ]
    )

    if not dataset_path.is_file():
        raise FileNotFoundError(
            str(
                dataset_path
            )
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

    stop_after = int(
        phase_cfg[
            "replay"
        ][
            "stop_after_sample"
        ]
    )

    if stop_after >= qpos.shape[0]:
        raise RuntimeError(
            "PB0-T stop_after_sample exceeds qpos replay"
        )

    locators = selected_rollouts(
        shortlist,
        pb0_cfg,
    )

    groups = group_by_batch(
        locators
    )

    n_envs = int(
        pb0_cfg[
            "collection"
        ][
            "n_envs"
        ]
    )

    env = build_env(
        n_envs,
        pb0_cfg[
            "collection"
        ][
            "n_steps_sub"
        ],
        official_log_dir(),
    )

    if pb0_cfg[
        "collection"
    ][
        "activate_repo_wiring_post_position_randomization"
    ]:
        env.init_domain_randomization(
            **wiring_post_args
        )

    metadata = build_native_metadata(
        env
    )

    all_records = {}

    try:
        with np.load(
                dataset_path
        ) as frozen:
            frozen_rope = frozen[
                "rope_xyz"
            ]

            for batch_index, rollout_ids in (
                    groups.items()):
                batch_records = run_targeted_replay(
                    env,
                    qpos,
                    batch_index=batch_index,
                    rollout_ids=rollout_ids,
                    pb0_cfg=pb0_cfg,
                    phase_cfg=phase_cfg,
                    metadata=metadata,
                    frozen_rope=frozen_rope,
                )

                merge_record_dicts(
                    all_records,
                    batch_records,
                )
    finally:
        env.stop()

    contact_threshold = float(
        pb0_cfg[
            "oracle_pair_descriptor"
        ][
            "contact_proxy_margin_m"
        ]
    )

    target_sample = int(
        phase_cfg[
            "replay"
        ][
            "target_sample"
        ]
    )

    core_samples = [
        int(v)
        for v
        in phase_cfg[
            "replay"
        ][
            "core_stability_samples"
        ]
    ]

    pair_results = [
        pair_confirmation(
            candidate,
            all_records,
            contact_threshold_m=
                contact_threshold,
            target_sample=
                target_sample,
            core_samples=
                core_samples,
        )
        for candidate
        in shortlist[
            "candidates"
        ]
    ]

    global_replay_alignment_valid = bool(
        all(
            record[
                "replay_match_within_tolerance"
            ]
            for record
            in all_records.values()
        )
    )

    confirmed_shortlist = (
        build_confirmed_shortlist(
            pair_results,
            phase_cfg[
                "outputs"
            ][
                "confirmed_shortlist_k"
            ],
            global_replay_alignment_valid=
                global_replay_alignment_valid,
        )
    )

    return build_report(
        phase_cfg,
        pb0_cfg,
        metadata,
        all_records,
        pair_results,
        confirmed_shortlist,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
    )
    args = parser.parse_args()

    result = run(
        args.config
    )

    print(
        "verdict={}".format(
            result[
                "verdict"
            ]
        )
    )

    print(
        "native_confirmed_pair_count={}".format(
            result[
                "native_confirmed_pair_count"
            ]
        )
    )

    print(
        "stable_core_support_count={}".format(
            result[
                "native_confirmed_with_stable_core_support_count"
            ]
        )
    )


if __name__ == "__main__":
    main()
