"""PB2-A/B: reproduce official DLO-Lab Wrapping and audit task-native winding.

The phase deliberately stops before pair mining.

PB2-A asks only whether the pinned published Wrapping task and its official
CMA-ES output run successfully.

PB2-B asks whether the privileged winding quantity is genuinely task-native,
finite, numerically reproducible from the published definition, and dynamically
excited by the official replay.

No benchmark physics, geometry, reward, observation mask, pair threshold,
StateDiff model, sensor model, or CCDA gate is changed here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from scripts.experiment3.dlolab_wrapping.paths import (
    REPO_ROOT,
    add_dlolab_to_path,
    official_log_dir,
)

add_dlolab_to_path()

import genesis as gs  # noqa: E402
from omegaconf import DictConfig  # noqa: E402
from envs.env_wrapping import Train_Env_Wrapping  # noqa: E402


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


def to_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def normalize_qpos(qpos):
    qpos = np.asarray(
        qpos,
        dtype=np.float32,
    )
    if qpos.ndim == 3:
        if qpos.shape[0] != 1:
            raise ValueError(
                "best_qpos.npy has unexpected leading dimension"
            )
        qpos = qpos[0]
    if qpos.ndim != 2:
        raise ValueError(
            "best_qpos.npy must be [T,D] or [1,T,D]"
        )
    return qpos


def signed_winding_turns_np(
        rope_verts,
        posts_pos):
    """Exact signed-turn intermediate used by published Wrapping winding loss.

    Wrapping uses a closed circular rod, so the published implementation
    intentionally closes the vertex sequence with np.roll.
    """
    # Preserve the published input dtype. The official function performs
    # these NumPy operations directly on the simulator arrays, so avoiding a
    # silent float64 promotion makes the identity check genuinely like-for-like.
    rope_xy = np.asarray(
        rope_verts
    )[:, :2]

    posts_xy = np.asarray(
        posts_pos
    )[:, :2]

    rel_pos = (
        rope_xy[:, None, :]
        - posts_xy[None, :, :]
    )

    current = rel_pos
    next_ = np.roll(
        rel_pos,
        shift=-1,
        axis=0,
    )

    cross = (
        current[..., 0]
        * next_[..., 1]
        - current[..., 1]
        * next_[..., 0]
    )

    dot = np.sum(
        current * next_,
        axis=-1,
    )

    angle_steps = np.arctan2(
        cross,
        dot,
    )

    total_winding = np.sum(
        angle_steps,
        axis=0,
    )

    return (
        total_winding
        / (2.0 * np.pi)
    )


def winding_loss_from_signed_turns(
        signed_turns):
    signed_turns = np.asarray(
        signed_turns
    )
    return float(
        np.mean(
            (
                np.abs(
                    signed_turns
                )
                - 1.0
            )
            ** 2
        )
    )


def build_env(
        n_envs,
        n_steps_sub,
        log_dir):
    cfg = DictConfig(
        {
            "task":
                "wrapping",

            "log_dir":
                str(
                    log_dir
                ),

            "n_envs":
                int(
                    n_envs
                ),

            "GUI":
                False,

            "camera":
                False,

            "raytracer":
                False,

            "requires_grad":
                False,
        }
    )

    env = Train_Env_Wrapping(
        config=cfg
    )

    env.init_cmaes_env(
        n_steps_sub=int(
            n_steps_sub
        )
    )

    return env


def _dual_arm_command(
        env,
        command):
    command = np.asarray(
        command,
        dtype=np.float32,
    )

    if command.shape != (18,):
        raise ValueError(
            "Wrapping qpos command must have 18 values"
        )

    command_batch = np.repeat(
        command[None, :],
        env.n_envs,
        axis=0,
    )

    command_tc = torch.tensor(
        command_batch,
        dtype=gs.tc_float,
    )

    first = command_tc[
        ...,
        :9
    ]

    second = command_tc[
        ...,
        9:
    ]

    env.c1.robot.control_dofs_position(
        first[
            ...,
            :-2
        ],
        env.c1.motors_dof,
    )

    env.c1.robot.control_dofs_position(
        first[
            ...,
            -2:
        ],
        env.c1.fingers_dof,
    )

    env.c2.robot.control_dofs_position(
        second[
            ...,
            :-2
        ],
        env.c2.motors_dof,
    )

    env.c2.robot.control_dofs_position(
        second[
            ...,
            -2:
        ],
        env.c2.fingers_dof,
    )


def sample_env(env):
    rope_xyz = to_numpy(
        env.rope.get_all_verts()
    ).astype(
        np.float32,
        copy=False,
    )

    rope_vel = to_numpy(
        env.rope.get_all_vels_tc()
    ).astype(
        np.float32,
        copy=False,
    )

    post_xyz = np.stack(
        [
            to_numpy(
                env.post1.get_pos()
            ),
            to_numpy(
                env.post2.get_pos()
            ),
            to_numpy(
                env.post3.get_pos()
            ),
        ],
        axis=1,
    ).astype(
        np.float32,
        copy=False,
    )

    ee1_pos = to_numpy(
        env.c1.ef.get_pos()
    ).astype(
        np.float32,
        copy=False,
    )

    ee1_quat = to_numpy(
        env.c1.ef.get_quat()
    ).astype(
        np.float32,
        copy=False,
    )

    ee2_pos = to_numpy(
        env.c2.ef.get_pos()
    ).astype(
        np.float32,
        copy=False,
    )

    ee2_quat = to_numpy(
        env.c2.ef.get_quat()
    ).astype(
        np.float32,
        copy=False,
    )

    motor_qpos_1 = to_numpy(
        env.c1.robot.get_dofs_position(
            env.c1.motors_dof
        )
    ).astype(
        np.float32,
        copy=False,
    )

    motor_qpos_2 = to_numpy(
        env.c2.robot.get_dofs_position(
            env.c2.motors_dof
        )
    ).astype(
        np.float32,
        copy=False,
    )

    motor_force_1 = to_numpy(
        env.c1.robot.get_dofs_force(
            env.c1.motors_dof
        )
    ).astype(
        np.float32,
        copy=False,
    )

    motor_force_2 = to_numpy(
        env.c2.robot.get_dofs_force(
            env.c2.motors_dof
        )
    ).astype(
        np.float32,
        copy=False,
    )

    motor_control_force_1 = to_numpy(
        env.c1.robot.get_dofs_control_force(
            env.c1.motors_dof
        )
    ).astype(
        np.float32,
        copy=False,
    )

    motor_control_force_2 = to_numpy(
        env.c2.robot.get_dofs_control_force(
            env.c2.motors_dof
        )
    ).astype(
        np.float32,
        copy=False,
    )

    reward = np.asarray(
        env.reward(),
        dtype=np.float64,
    )

    signed_turns = np.stack(
        [
            signed_winding_turns_np(
                rope_xyz[index],
                post_xyz[index],
            )
            for index
            in range(
                env.n_envs
            )
        ],
        axis=0,
    )

    winding_loss_reconstructed = np.asarray(
        [
            winding_loss_from_signed_turns(
                signed_turns[
                    index
                ]
            )
            for index
            in range(
                env.n_envs
            )
        ],
        dtype=np.float64,
    )

    winding_loss_official = np.asarray(
        [
            env._winding_angle_loss_np(
                rope_verts=
                    rope_xyz[
                        index
                    ],

                posts_pos=
                    post_xyz[
                        index
                    ],
            )
            for index
            in range(
                env.n_envs
            )
        ],
        dtype=np.float64,
    )

    return {
        "rope_xyz":
            rope_xyz,

        "rope_vel":
            rope_vel,

        "post_xyz":
            post_xyz,

        "ee1_pos":
            ee1_pos,

        "ee1_quat":
            ee1_quat,

        "ee2_pos":
            ee2_pos,

        "ee2_quat":
            ee2_quat,

        "motor_qpos_1":
            motor_qpos_1,

        "motor_qpos_2":
            motor_qpos_2,

        "motor_force_1":
            motor_force_1,

        "motor_force_2":
            motor_force_2,

        "motor_control_force_1":
            motor_control_force_1,

        "motor_control_force_2":
            motor_control_force_2,

        "reward":
            reward,

        "signed_winding_turns":
            signed_turns,

        "winding_magnitude_turns":
            np.abs(
                signed_turns
            ),

        "winding_loss_reconstructed":
            winding_loss_reconstructed,

        "winding_loss_official":
            winding_loss_official,
    }


def _stack_time(
        rows,
        key):
    return np.stack(
        [
            row[
                key
            ]
            for row
            in rows
        ],
        axis=1,
    )


def replay_best_qpos(
        env,
        qpos):
    env.use_qpos = True
    env.reset()

    rows = [
        sample_env(
            env
        )
    ]

    n_intervals = (
        env.steps_interval
        // env._cmaes_n_steps_sub
    )

    if n_intervals <= 0:
        raise RuntimeError(
            "Invalid Wrapping replay interval count"
        )

    for command in qpos[
            1:]:
        _dual_arm_command(
            env,
            command,
        )

        for _ in range(
                n_intervals):
            env.scene.step()

        rows.append(
            sample_env(
                env
            )
        )

    return {
        key:
            _stack_time(
                rows,
                key,
            )
        for key
        in rows[
            0
        ]
    }


def _require_numeric_close(run_config, field, expected, atol=1e-12):
    if field not in run_config:
        raise RuntimeError(f"Official run_config missing field: {field}")
    actual = float(run_config[field])
    if not np.isclose(actual, float(expected), rtol=0.0, atol=float(atol)):
        raise RuntimeError(
            f"Official CMA-ES config mismatch for {field}: "
            f"{actual} != {expected}"
        )


def validate_official_run_config(run_config, required):
    """Validate the full effective parameter set recorded by pinned CMA-ES."""
    integer_fields = {
        "n_envs": required["n_envs"],
        "n_steps": required["n_steps"],
        "n_steps_sub": required["n_steps_sub"],
        "act_dim": required["act_dim"],
        "popsize": required["popsize"],
        "max_iters": required["max_iter"],
        "seed": required["seed"],
    }
    for field, expected in integer_fields.items():
        if field not in run_config:
            raise RuntimeError(f"Official run_config missing field: {field}")
        actual = int(run_config[field])
        if actual != int(expected):
            raise RuntimeError(
                f"Official CMA-ES config mismatch for {field}: "
                f"{actual} != {expected}"
            )

    _require_numeric_close(run_config, "sigma0", required["sigma"])
    _require_numeric_close(run_config, "per_comp_bound", required["bound"])
    _require_numeric_close(run_config, "l2_bound", required["l2_bound"])
    _require_numeric_close(run_config, "angle_bound", required["angle_bound"])

    actual_angle_scale = np.asarray(
        run_config.get("angle_scale"), dtype=np.float64
    ).reshape(-1)
    expected_angle_scale = np.asarray(
        required["angle_scale"], dtype=np.float64
    ).reshape(-1)
    if (
        actual_angle_scale.shape != expected_angle_scale.shape
        or not np.allclose(
            actual_angle_scale,
            expected_angle_scale,
            rtol=0.0,
            atol=1e-12,
        )
    ):
        raise RuntimeError(
            "Official CMA-ES angle_scale mismatch: "
            f"{actual_angle_scale.tolist()} != "
            f"{expected_angle_scale.tolist()}"
        )

    for required_field in (
        "use_last_state_reward",
        "randomized_args",
    ):
        if required_field not in run_config:
            raise RuntimeError(
                f"Official run_config missing field: {required_field}"
            )

    if bool(run_config["use_last_state_reward"]) != bool(
        required["use_last_state_reward"]
    ):
        raise RuntimeError("Official CMA-ES use_last_state_reward mismatch")

    if run_config["randomized_args"] != required["randomized_args"]:
        raise RuntimeError(
            "Official CMA-ES randomized_args mismatch: "
            f"{run_config['randomized_args']} != "
            f"{required['randomized_args']}"
        )


def validate_official_artifacts(config):
    log_dir = official_log_dir()
    best_qpos_path = log_dir / "best_qpos.npy"
    best_traj_path = log_dir / "best_traj.npy"
    resume_meta_path = log_dir / "resume_meta.json"
    run_config_path = log_dir / "run_config.json"

    for path in (
        best_qpos_path,
        best_traj_path,
        resume_meta_path,
        run_config_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(str(path))

    qpos = normalize_qpos(np.load(best_qpos_path))
    traj = np.asarray(np.load(best_traj_path), dtype=np.float32)

    expected_qpos_shape = tuple(
        int(v) for v in config["replay_audit"]["expected_best_qpos_shape"]
    )
    expected_traj_shape = tuple(
        int(v) for v in config["replay_audit"]["expected_best_traj_shape"]
    )

    if qpos.shape != expected_qpos_shape:
        raise RuntimeError(
            f"Unexpected official best_qpos shape: "
            f"{qpos.shape} != {expected_qpos_shape}"
        )
    if traj.shape != expected_traj_shape:
        raise RuntimeError(
            f"Unexpected official best_traj shape: "
            f"{traj.shape} != {expected_traj_shape}"
        )
    if not (np.isfinite(qpos).all() and np.isfinite(traj).all()):
        raise RuntimeError("Official Wrapping optimizer output is non-finite")

    resume_meta = load_json(resume_meta_path)
    run_config = load_json(run_config_path)
    required = config["official_cmaes"]
    validate_official_run_config(run_config, required)

    last_iter = int(resume_meta["iter"])
    expected_last_iter = int(required["max_iter"]) - 1
    if last_iter != expected_last_iter:
        raise RuntimeError(
            f"Official Wrapping completion mismatch: "
            f"iter={last_iter}, expected={expected_last_iter}"
        )

    best_reward = float(resume_meta["best_reward"])
    if not np.isfinite(best_reward):
        raise RuntimeError("Official Wrapping best reward is non-finite")

    return {
        "log_dir": str(log_dir),
        "best_qpos_path": str(best_qpos_path),
        "best_traj_path": str(best_traj_path),
        "last_iter": last_iter,
        "best_reward": best_reward,
        "qpos": qpos,
        "traj": traj,
        "run_config": run_config,
    }

def _series_stats(
        values):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    return {
        "min":
            float(
                np.min(
                    values
                )
            ),

        "max":
            float(
                np.max(
                    values
                )
            ),

        "span":
            float(
                np.max(
                    values
                )
                - np.min(
                    values
                )
            ),

        "initial":
            float(
                values[
                    0
                ]
            ),

        "final":
            float(
                values[
                    -1
                ]
            ),
    }


def semantic_winding_transition_audit(
        magnitude_turns,
        *,
        unwrapped_max,
        wrapped_min,
        require_later):
    """Task-semantic transition audit tied to the published one-turn target."""
    magnitude_turns = np.asarray(magnitude_turns, dtype=np.float64)
    if magnitude_turns.ndim != 2 or magnitude_turns.shape[1] != 3:
        raise RuntimeError("Expected winding magnitude series [T,3]")

    per_post = {}
    for post_index in range(3):
        values = magnitude_turns[:, post_index]
        unwrapped = np.flatnonzero(values <= float(unwrapped_max))
        wrapped = np.flatnonzero(values >= float(wrapped_min))

        pair = None
        if unwrapped.size and wrapped.size:
            if require_later:
                for u in unwrapped:
                    later = wrapped[wrapped > u]
                    if later.size:
                        pair = (int(u), int(later[0]))
                        break
            else:
                pair = (int(unwrapped[0]), int(wrapped[0]))

        per_post[f"post{post_index + 1}"] = {
            "transition_observed": pair is not None,
            "first_unwrapped_like_sample": None if pair is None else pair[0],
            "first_later_wrapped_like_sample": None if pair is None else pair[1],
            "unwrapped_like_sample_count": int(unwrapped.size),
            "wrapped_like_sample_count": int(wrapped.size),
            "minimum_magnitude_turns": float(np.min(values)),
            "maximum_magnitude_turns": float(np.max(values)),
        }

    return {
        "unwrapped_max_magnitude_turns": float(unwrapped_max),
        "wrapped_min_magnitude_turns": float(wrapped_min),
        "require_wrapped_sample_after_unwrapped_sample": bool(require_later),
        "transition_post_count": int(
            sum(row["transition_observed"] for row in per_post.values())
        ),
        "per_post": per_post,
    }


def summarize_winding(replay, semantic_config):
    signed = np.asarray(
        replay["signed_winding_turns"][0], dtype=np.float64
    )
    magnitude = np.asarray(
        replay["winding_magnitude_turns"][0], dtype=np.float64
    )
    if signed.ndim != 2 or signed.shape[1] != 3:
        raise RuntimeError("Expected Wrapping winding series [T,3]")

    signed_by_post = {
        f"post{i + 1}": _series_stats(signed[:, i])
        for i in range(3)
    }
    magnitude_by_post = {
        f"post{i + 1}": _series_stats(magnitude[:, i])
        for i in range(3)
    }

    semantic_transition = semantic_winding_transition_audit(
        magnitude,
        unwrapped_max=semantic_config[
            "unwrapped_max_magnitude_turns"
        ],
        wrapped_min=semantic_config[
            "wrapped_min_magnitude_turns"
        ],
        require_later=semantic_config[
            "require_wrapped_sample_after_unwrapped_sample"
        ],
    )
    minimum_posts = int(semantic_config["minimum_transition_posts"])

    return {
        "signed_turns_by_post": signed_by_post,
        "magnitude_turns_by_post": magnitude_by_post,
        "maximum_signed_turn_span": float(
            max(row["span"] for row in signed_by_post.values())
        ),
        "task_semantic_excitation_observed": bool(
            semantic_transition["transition_post_count"] >= minimum_posts
        ),
        "semantic_transition": semantic_transition,
        "minimum_transition_posts_required": minimum_posts,
        "winding_loss": _series_stats(
            replay["winding_loss_official"][0]
        ),
        "reward": _series_stats(replay["reward"][0]),
    }

def build_report(
        config,
        official,
        replay):
    atol = float(
        config[
            "replay_audit"
        ][
            "winding_loss_match_atol"
        ]
    )

    loss_difference = np.abs(
        np.asarray(
            replay[
                "winding_loss_official"
            ],
            dtype=np.float64,
        )
        - np.asarray(
            replay[
                "winding_loss_reconstructed"
            ],
            dtype=np.float64,
        )
    )

    max_loss_difference = float(
        np.max(
            loss_difference
        )
    )

    if max_loss_difference > atol:
        raise RuntimeError(
            "Task-native winding reconstruction mismatch: {} > {}".format(
                max_loss_difference,
                atol,
            )
        )

    finite_replay = bool(
        all(
            np.isfinite(
                value
            ).all()
            for value
            in replay.values()
        )
    )

    if not finite_replay:
        raise RuntimeError(
            "Official Wrapping replay contains non-finite values"
        )

    winding = summarize_winding(
        replay,
        config["replay_audit"]["semantic_winding_transition"],
    )

    findings = [
        "PB2A_OFFICIAL_WRAPPING_REPRODUCED",
        "PB2B_TASK_NATIVE_WINDING_DEFINITION_REPRODUCED",
    ]

    if winding[
        "task_semantic_excitation_observed"
    ]:
        findings.append(
            "PB2B_TASK_SEMANTIC_WINDING_TRANSITION_OBSERVED"
        )

        verdict = (
            "PB2AB_WRAPPING_REPRODUCED_AND_WINDING_AUDITED"
        )

        next_action = (
            "Proceed to PB2-C natural pair discovery. Use the published "
            "Wrapping task unchanged, keep winding as privileged audit "
            "metadata, define deployable partial observation separately, "
            "and use repository-provided Wrapping position randomization "
            "for rollout diversity without changing task physics. "
            "Do not start StateDiff training yet."
        )
    else:
        findings.append(
            "PB2B_NO_TASK_SEMANTIC_WINDING_TRANSITION_OBSERVED"
        )

        verdict = (
            "PB2AB_WRAPPING_REPRODUCED_BUT_WINDING_NOT_SEMANTICALLY_EXCITED"
        )

        next_action = (
            "Do not start pair mining yet. Inspect whether the official "
            "optimized trajectory meaningfully executes the Wrapping task. "
            "Do not invent a new hidden proxy or modify task physics."
        )

    result = {
        "verdict":
            verdict,

        "findings":
            findings,

        "benchmark": {
            "task":
                "wrapping",

            "dlolab_revision":
                config[
                    "benchmark"
                ][
                    "revision"
                ],

            "physics_modified":
                False,

            "reward_modified":
                False,

            "geometry_modified":
                False,
        },

        "validated_effective_cmaes_config":
            dict(
                official[
                    "run_config"
                ]
            ),

        "pb2a_reproduction": {
            "official_last_iter":
                int(
                    official[
                        "last_iter"
                    ]
                ),

            "official_best_reward":
                float(
                    official[
                        "best_reward"
                    ]
                ),

            "best_traj_shape":
                list(
                    official[
                        "traj"
                    ].shape
                ),

            "best_qpos_shape":
                list(
                    official[
                        "qpos"
                    ].shape
                ),

            "finite_replay":
                finite_replay,

            "time_samples":
                int(
                    replay[
                        "rope_xyz"
                    ].shape[
                        1
                    ]
                ),
        },

        "pb2b_winding_audit": {
            "privileged_variable":
                (
                    "per-post signed winding turns and magnitude "
                    "from the published Wrapping winding-loss definition"
                ),

            "model_input":
                False,

            "official_loss_vs_reconstruction_max_abs":
                max_loss_difference,

            "loss_match_atol":
                atol,

            **winding,
        },

        "next_action":
            next_action,

        "non_claims": [
            "PB2-A/B does not establish a CCDA pair.",
            (
                "Winding is privileged audit state and is not deployable "
                "model input."
            ),
            "No pair-mining threshold is introduced in this phase.",
            (
                "No future-bifurcation or control-relevance gate is tested."
            ),
        ],
    }

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

    np.savez_compressed(
        raw_root
        / "official_best_replay.npz",
        **replay,
        common_qpos_replay=
            official[
                "qpos"
            ],
        best_traj=
            official[
                "traj"
            ],
    )

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

        "validated_effective_cmaes_config":
            dict(
                official[
                    "run_config"
                ]
            ),

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

    lines = [
        "# PB2-A/B DLO-Lab Wrapping Reproduction & Winding Audit",
        "",
        "Verdict: `{}`".format(
            verdict
        ),
        "",
        "## PB2-A — official reproduction",
        "",
        "- Official CMA-ES last iter: {}".format(
            official[
                "last_iter"
            ]
        ),
        "- Official best reward: {:.9g}".format(
            official[
                "best_reward"
            ]
        ),
        "- best_traj shape: `{}`".format(
            tuple(
                official[
                    "traj"
                ].shape
            )
        ),
        "- best_qpos shape: `{}`".format(
            tuple(
                official[
                    "qpos"
                ].shape
            )
        ),
        "- Replay finite: {}".format(
            finite_replay
        ),
        "",
        "## PB2-B — task-native privileged winding",
        "",
        "- Official/reconstructed winding-loss max abs diff: {:.3e}".format(
            max_loss_difference
        ),
        "- Dynamic variation observed: {}".format(
            winding[
                "task_semantic_excitation_observed"
            ]
        ),
        "- Maximum signed-turn span: {:.9g}".format(
            winding[
                "maximum_signed_turn_span"
            ]
        ),
        "",
        "Per-post signed and magnitude ranges are stored in `EVIDENCE.json`.",
        "",
        "## Findings",
        "",
    ]

    lines.extend(
        "- `{}`".format(
            item
        )
        for item
        in findings
    )

    lines.extend(
        [
            "",
            "## Next action",
            "",
            next_action,
            "",
            "## Boundary",
            "",
            (
                "This phase stops before natural pair mining, snapshot "
                "branching, StateDiff training, CFPM, IDM, or closed-loop "
                "evaluation."
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

    official = validate_official_artifacts(
        config
    )

    env = build_env(
        config[
            "replay_audit"
        ][
            "n_envs"
        ],
        config[
            "official_cmaes"
        ][
            "n_steps_sub"
        ],
        official_log_dir(),
    )

    try:
        replay = replay_best_qpos(
            env,
            official[
                "qpos"
            ],
        )
    finally:
        env.stop()

    return build_report(
        config,
        official,
        replay,
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

    for finding in result[
            "findings"]:
        print(
            "finding={}".format(
                finding
            )
        )


if __name__ == "__main__":
    main()
