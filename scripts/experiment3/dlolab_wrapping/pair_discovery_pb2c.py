"""PB2-C exact same-time natural pair discovery for DLO-Lab Wrapping.

Scientific selection uses only:
- an artificial post-local partial-state rope history,
- the published robot-state component for both arms,
- task-native privileged signed winding index,
- identical action history by construction (same time, common qpos replay).

The partial rope observation is NOT claimed to be a deployable perception stack.

Future divergence, robot force and sensor channels are NOT used to select pairs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from scripts.experiment3.dlolab_wrapping.paths import REPO_ROOT
from scripts.experiment3.dlolab_wrapping.replay_winding_audit import (
    load_json,
)


def git(*args):
    return subprocess.check_output(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
    ).strip()


def load_datasets(paths):
    datasets = []
    common_qpos = None

    for path in paths:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(str(path))

        with np.load(path) as handle:
            row = {
                key: np.asarray(handle[key])
                for key in handle.files
                if key != "common_qpos_replay"
            }
            qpos = np.asarray(handle["common_qpos_replay"])

        if common_qpos is None:
            common_qpos = qpos
        elif not np.array_equal(common_qpos, qpos):
            raise RuntimeError(
                "PB2-C datasets do not share the same qpos replay"
            )

        datasets.append(row)

    keys = set(datasets[0])
    if any(set(row) != keys for row in datasets[1:]):
        raise RuntimeError(
            "PB2-C dataset keys differ across collection files"
        )

    merged = {}
    for key in keys:
        if key in {
            "rollout_id",
            "batch_index",
            "env_index",
            "batch_seed",
        }:
            merged[key] = np.concatenate(
                [row[key] for row in datasets],
                axis=0,
            )
        else:
            merged[key] = np.concatenate(
                [row[key] for row in datasets],
                axis=0,
            )

    order = np.argsort(merged["rollout_id"])
    for key in merged:
        merged[key] = merged[key][order]

    ids = merged["rollout_id"]
    if len(np.unique(ids)) != len(ids):
        raise RuntimeError("Duplicate rollout_id across PB2-C datasets")

    merged["common_qpos_replay"] = common_qpos
    return merged


def occlusion_radius(config):
    obs = config["partial_state_observation"]["rope_component"]
    return (
        float(obs["local_occlusion_radius_multiplier"])
        * (
            float(obs["post_radius_m"])
            + float(obs["rope_radius_m"])
        )
    )


def compute_visibility_mask(
        rope_xyz,
        post_xyz,
        radius_m):
    rope_xy = np.asarray(
        rope_xyz,
        dtype=np.float32,
    )[..., :2]

    post_xy = np.asarray(
        post_xyz,
        dtype=np.float32,
    )[..., :2]

    delta = (
        rope_xy[:, :, :, None, :]
        - post_xy[:, :, None, :, :]
    )
    dist2 = np.sum(delta * delta, axis=-1)

    return np.min(
        dist2,
        axis=-1,
    ) > float(radius_m) ** 2


def signed_winding_index(signed_turns):
    return np.rint(
        np.asarray(signed_turns)
    ).astype(np.int8)


def batch_symmetric_chamfer(
        points_a,
        mask_a,
        points_b,
        mask_b):
    """Exact symmetric Chamfer for equally padded point arrays."""
    points_a = np.asarray(
        points_a,
        dtype=np.float32,
    )
    points_b = np.asarray(
        points_b,
        dtype=np.float32,
    )
    mask_a = np.asarray(mask_a, dtype=bool)
    mask_b = np.asarray(mask_b, dtype=bool)

    if (
        np.any(mask_a.sum(axis=1) == 0)
        or np.any(mask_b.sum(axis=1) == 0)
    ):
        raise ValueError(
            "Chamfer received an empty visible point set"
        )

    diff = (
        points_a[:, :, None, :]
        - points_b[:, None, :, :]
    )
    dist = np.sqrt(
        np.sum(diff * diff, axis=-1)
    )

    inf = np.float32(np.inf)

    a_to_b = np.min(
        np.where(
            mask_b[:, None, :],
            dist,
            inf,
        ),
        axis=2,
    )

    b_to_a = np.min(
        np.where(
            mask_a[:, :, None],
            dist,
            inf,
        ),
        axis=1,
    )

    mean_a = (
        np.sum(
            np.where(mask_a, a_to_b, 0.0),
            axis=1,
        )
        / mask_a.sum(axis=1)
    )

    mean_b = (
        np.sum(
            np.where(mask_b, b_to_a, 0.0),
            axis=1,
        )
        / mask_b.sum(axis=1)
    )

    return 0.5 * (mean_a + mean_b)


def chamfer_for_pairs_at_time(
        rope_xyz,
        visible,
        pair_a,
        pair_b,
        time_index,
        batch_size):
    result = np.empty(
        len(pair_a),
        dtype=np.float32,
    )

    for start in range(
        0,
        len(pair_a),
        int(batch_size),
    ):
        end = min(
            len(pair_a),
            start + int(batch_size),
        )
        ia = pair_a[start:end]
        ib = pair_b[start:end]

        result[start:end] = (
            batch_symmetric_chamfer(
                rope_xyz[ia, time_index],
                visible[ia, time_index],
                rope_xyz[ib, time_index],
                visible[ib, time_index],
            )
        )

    return result


def winding_integer_residual(signed_turns):
    signed = np.asarray(signed_turns, dtype=np.float64)
    return np.abs(signed - np.rint(signed))


def quaternion_geodesic_rad(quat_a, quat_b):
    """Sign-invariant quaternion geodesic distance in radians."""
    qa = np.asarray(quat_a, dtype=np.float64)
    qb = np.asarray(quat_b, dtype=np.float64)
    qa = qa / np.linalg.norm(qa, axis=-1, keepdims=True)
    qb = qb / np.linalg.norm(qb, axis=-1, keepdims=True)
    dot = np.abs(np.sum(qa * qb, axis=-1))
    return 2.0 * np.arccos(np.clip(dot, 0.0, 1.0))


def robot_history_metrics(data, pair_a, pair_b, time_index, history):
    """Mirror published robot observation semantics for both Frankas.

    Per arm: EE xyz + EE quaternion + 7 motor-joint qpos.
    """
    frames = slice(time_index - history + 1, time_index + 1)

    ee1_pos = np.linalg.norm(
        data["ee1_pos"][pair_a, frames] - data["ee1_pos"][pair_b, frames],
        axis=-1,
    )
    ee2_pos = np.linalg.norm(
        data["ee2_pos"][pair_a, frames] - data["ee2_pos"][pair_b, frames],
        axis=-1,
    )
    ee_pos_mean = 0.5 * (np.mean(ee1_pos, axis=1) + np.mean(ee2_pos, axis=1))

    quat1 = quaternion_geodesic_rad(
        data["ee1_quat"][pair_a, frames],
        data["ee1_quat"][pair_b, frames],
    )
    quat2 = quaternion_geodesic_rad(
        data["ee2_quat"][pair_a, frames],
        data["ee2_quat"][pair_b, frames],
    )
    ee_quat_mean = 0.5 * (np.mean(quat1, axis=1) + np.mean(quat2, axis=1))

    dq1 = data["motor_qpos_1"][pair_a, frames] - data["motor_qpos_1"][pair_b, frames]
    dq2 = data["motor_qpos_2"][pair_a, frames] - data["motor_qpos_2"][pair_b, frames]
    qpos1_rms = np.sqrt(np.mean(dq1 * dq1, axis=-1))
    qpos2_rms = np.sqrt(np.mean(dq2 * dq2, axis=-1))
    motor_qpos_rms = 0.5 * (np.mean(qpos1_rms, axis=1) + np.mean(qpos2_rms, axis=1))

    return {
        "dual_ee_position_history_mean_m": ee_pos_mean,
        "dual_ee_quaternion_geodesic_history_mean_rad": ee_quat_mean,
        "dual_motor_qpos_history_rms_rad": motor_qpos_rms,
    }


def class_histogram(index_at_time):
    values, counts = np.unique(index_at_time, axis=0, return_counts=True)
    return {
        ",".join(str(int(v)) for v in value): int(count)
        for value, count in zip(values, counts)
    }


def mine_pairs(data, config):
    official_valid = np.asarray(data["official_rollout_valid"], dtype=bool)
    valid_indices = np.flatnonzero(official_valid)

    rope_xyz = np.asarray(data["rope_xyz"], dtype=np.float32)
    post_xyz = np.asarray(data["post_xyz"], dtype=np.float32)
    signed_turns = np.asarray(data["signed_winding_turns"], dtype=np.float64)
    n_times = rope_xyz.shape[1]

    obs_cfg = config["partial_state_observation"]
    rope_cfg = obs_cfg["rope_component"]
    robot_cfg = obs_cfg["robot_component"]
    history = int(obs_cfg["history_samples"])

    chamfer_threshold = float(rope_cfg["max_visible_history_chamfer_m"])
    ee_pos_threshold = float(robot_cfg["max_dual_ee_position_history_mean_m"])
    ee_quat_threshold = float(
        robot_cfg["max_dual_ee_quaternion_geodesic_history_mean_rad"]
    )
    qpos_threshold = float(robot_cfg["max_dual_motor_qpos_history_rms_rad"])
    pair_batch_size = int(config["discovery"]["pair_batch_size"])

    radius = occlusion_radius(config)
    visible = compute_visibility_mask(rope_xyz, post_xyz, radius)
    visible_counts = visible.sum(axis=2)

    winding_index = signed_winding_index(signed_turns)
    residual = winding_integer_residual(signed_turns)

    local_a, local_b = np.triu_indices(len(valid_indices), k=1)
    upper_a = valid_indices[local_a]
    upper_b = valid_indices[local_b]

    funnel = {
        "total_rollouts": int(len(official_valid)),
        "official_valid_rollouts": int(official_valid.sum()),
        "official_invalid_rollouts": int((~official_valid).sum()),
        "total_valid_same_time_pair_comparisons": 0,
        "hidden_winding_index_different": 0,
        "nonempty_partial_rope_history": 0,
        "robot_ee_position_pass": 0,
        "robot_ee_quaternion_pass": 0,
        "robot_motor_qpos_pass": 0,
        "robot_observation_pass": 0,
        "visible_history_chamfer_pass": 0,
        "candidate_count": 0,
    }

    mixed_time_histograms = {}
    candidates = []
    safe_single_frame_bound = history * chamfer_threshold

    for time_index in range(history - 1, n_times):
        funnel["total_valid_same_time_pair_comparisons"] += int(len(upper_a))

        current_valid_index = winding_index[valid_indices, time_index]
        hist = class_histogram(current_valid_index)
        if len(hist) > 1:
            mixed_time_histograms[str(time_index)] = hist

        hidden_diff = np.any(
            winding_index[upper_a, time_index] != winding_index[upper_b, time_index],
            axis=1,
        )
        ia = upper_a[hidden_diff]
        ib = upper_b[hidden_diff]
        funnel["hidden_winding_index_different"] += int(len(ia))
        if len(ia) == 0:
            continue

        hist_ok = (
            np.all(
                visible_counts[ia, time_index-history+1:time_index+1] > 0,
                axis=1,
            )
            & np.all(
                visible_counts[ib, time_index-history+1:time_index+1] > 0,
                axis=1,
            )
        )
        ia = ia[hist_ok]
        ib = ib[hist_ok]
        funnel["nonempty_partial_rope_history"] += int(len(ia))
        if len(ia) == 0:
            continue

        robot = robot_history_metrics(data, ia, ib, time_index, history)

        pass_mask = robot["dual_ee_position_history_mean_m"] <= ee_pos_threshold
        funnel["robot_ee_position_pass"] += int(pass_mask.sum())
        ia, ib = ia[pass_mask], ib[pass_mask]
        robot = {k: v[pass_mask] for k, v in robot.items()}
        if len(ia) == 0:
            continue

        pass_mask = (
            robot["dual_ee_quaternion_geodesic_history_mean_rad"] <= ee_quat_threshold
        )
        funnel["robot_ee_quaternion_pass"] += int(pass_mask.sum())
        ia, ib = ia[pass_mask], ib[pass_mask]
        robot = {k: v[pass_mask] for k, v in robot.items()}
        if len(ia) == 0:
            continue

        pass_mask = robot["dual_motor_qpos_history_rms_rad"] <= qpos_threshold
        funnel["robot_motor_qpos_pass"] += int(pass_mask.sum())
        ia, ib = ia[pass_mask], ib[pass_mask]
        robot = {k: v[pass_mask] for k, v in robot.items()}
        funnel["robot_observation_pass"] += int(len(ia))
        if len(ia) == 0:
            continue

        current_chamfer = chamfer_for_pairs_at_time(
            rope_xyz, visible, ia, ib, time_index, pair_batch_size
        )
        safe = current_chamfer <= safe_single_frame_bound
        ia, ib = ia[safe], ib[safe]
        current_chamfer = current_chamfer[safe]
        robot = {k: v[safe] for k, v in robot.items()}
        if len(ia) == 0:
            continue

        history_sum = current_chamfer.astype(np.float64)
        for past_time in range(time_index - history + 1, time_index):
            history_sum += chamfer_for_pairs_at_time(
                rope_xyz, visible, ia, ib, past_time, pair_batch_size
            )
        history_chamfer = history_sum / history
        chamfer_pass = history_chamfer <= chamfer_threshold

        ia, ib = ia[chamfer_pass], ib[chamfer_pass]
        history_chamfer = history_chamfer[chamfer_pass]
        robot = {k: v[chamfer_pass] for k, v in robot.items()}
        funnel["visible_history_chamfer_pass"] += int(len(ia))

        for local_index in range(len(ia)):
            a = int(ia[local_index])
            b = int(ib[local_index])
            index_a = winding_index[a, time_index]
            index_b = winding_index[b, time_index]
            signed_a = signed_turns[a, time_index]
            signed_b = signed_turns[b, time_index]
            residual_a = residual[a, time_index]
            residual_b = residual[b, time_index]
            differing_posts = np.flatnonzero(index_a != index_b)

            candidates.append(
                {
                    "rollout_a": int(data["rollout_id"][a]),
                    "rollout_b": int(data["rollout_id"][b]),
                    "time_index": int(time_index),
                    "official_rollout_valid_a": True,
                    "official_rollout_valid_b": True,
                    "partial_state_observation_type": (
                        "artificial_post_local_occlusion_surrogate"
                    ),
                    "claim_deployable_observation": False,
                    "visible_rope_history_chamfer_m": float(
                        history_chamfer[local_index]
                    ),
                    "robot_observation": {
                        "dual_ee_position_history_mean_m": float(
                            robot["dual_ee_position_history_mean_m"][local_index]
                        ),
                        "dual_ee_quaternion_geodesic_history_mean_rad": float(
                            robot[
                                "dual_ee_quaternion_geodesic_history_mean_rad"
                            ][local_index]
                        ),
                        "dual_motor_qpos_history_rms_rad": float(
                            robot["dual_motor_qpos_history_rms_rad"][local_index]
                        ),
                    },
                    "signed_winding_turns_a": [float(v) for v in signed_a],
                    "signed_winding_turns_b": [float(v) for v in signed_b],
                    "winding_index_a": [int(v) for v in index_a],
                    "winding_index_b": [int(v) for v in index_b],
                    "winding_integer_residual_a": [float(v) for v in residual_a],
                    "winding_integer_residual_b": [float(v) for v in residual_b],
                    "winding_integer_residual_linf_a": float(np.max(residual_a)),
                    "winding_integer_residual_linf_b": float(np.max(residual_b)),
                    "pair_max_winding_integer_residual": float(
                        max(np.max(residual_a), np.max(residual_b))
                    ),
                    "differing_post_indices": [int(v) for v in differing_posts],
                    "differing_post_count": int(len(differing_posts)),
                    "winding_delta_linf_turns": float(
                        np.max(np.abs(signed_a - signed_b))
                    ),
                    "visible_vertex_counts_a": [
                        int(v)
                        for v in visible_counts[
                            a, time_index-history+1:time_index+1
                        ]
                    ],
                    "visible_vertex_counts_b": [
                        int(v)
                        for v in visible_counts[
                            b, time_index-history+1:time_index+1
                        ]
                    ],
                    "replay_a": {
                        "batch_index": int(data["batch_index"][a]),
                        "env_index": int(data["env_index"][a]),
                        "seed": int(data["batch_seed"][a]),
                    },
                    "replay_b": {
                        "batch_index": int(data["batch_index"][b]),
                        "env_index": int(data["env_index"][b]),
                        "seed": int(data["batch_seed"][b]),
                    },
                    "action_history_equal_by_construction": True,
                    "selection_used_future_divergence": False,
                    "selection_used_force_or_sensor": False,
                }
            )

    funnel["candidate_count"] = int(len(candidates))

    candidates.sort(
        key=lambda row: (
            row["visible_rope_history_chamfer_m"],
            row["robot_observation"]["dual_ee_position_history_mean_m"],
            row["robot_observation"][
                "dual_ee_quaternion_geodesic_history_mean_rad"
            ],
            row["robot_observation"]["dual_motor_qpos_history_rms_rad"],
            -row["differing_post_count"],
        )
    )

    valid_residual = residual[official_valid]
    if valid_residual.size:
        residual_summary = {
            "max_abs_turns": float(np.max(valid_residual)),
            "p99_abs_turns": float(np.quantile(valid_residual, 0.99)),
        }
    else:
        residual_summary = {
            "max_abs_turns": None,
            "p99_abs_turns": None,
        }

    top_k = int(config["discovery"]["top_k"])
    return {
        "funnel": funnel,
        "candidate_count": int(len(candidates)),
        "top_candidates": candidates[:top_k],
        "mixed_winding_time_histograms": mixed_time_histograms,
        "winding_index_residual": residual_summary,
        "partial_state_observation": {
            "name": obs_cfg["name"],
            "claim_deployable": bool(obs_cfg["claim_deployable"]),
            "history_samples": history,
            "local_occlusion_radius_m": float(radius),
            "max_visible_history_chamfer_m": chamfer_threshold,
            "robot_component": robot_cfg,
            "post_component": obs_cfg["post_component"],
        },
    }


def build_report(config, data, mining, label):
    batch_count = int(len(np.unique(data["batch_index"])))
    rollout_count = int(len(data["rollout_id"]))
    official_valid_count = int(
        np.asarray(data["official_rollout_valid"], dtype=bool).sum()
    )
    min_candidates = int(config["discovery"]["min_candidates_to_proceed"])
    max_batches = int(config["collection"]["max_batches"])

    if mining["candidate_count"] >= min_candidates:
        verdict = "PB2C_NATURAL_WINDING_PAIRS_FOUND"
        next_action = (
            "Proceed to PB3 snapshot same-action multi-horizon causal audit. "
            "Compare pair divergence against deterministic repeat/uncertainty. "
            "Do not reuse PB2-C discovery thresholds as Gate 4."
        )
    elif batch_count < max_batches:
        verdict = "PB2C_INITIAL_COLLECTION_INSUFFICIENT_SCALEUP_REQUIRED"
        next_action = (
            "Collect only the precommitted remaining Wrapping position-"
            "randomized batches up to 16 total, then rerun the frozen miner."
        )
    else:
        verdict = "PB2C_INSUFFICIENT_NATURAL_PAIRS_UNDER_FROZEN_PROTOCOL"
        next_action = (
            "Do not lower discovery thresholds or modify Wrapping physics. "
            "This bounded protocol did not yield enough candidates; it does "
            "not prove CCDA is absent."
        )

    result = {
        "verdict": verdict,
        "label": str(label),
        "rollout_count": rollout_count,
        "official_valid_rollout_count": official_valid_count,
        "official_invalid_rollout_count": rollout_count - official_valid_count,
        "official_validity_summary": {
            "failed_stretch_count": int(
                np.asarray(data["official_failed_stretch"], dtype=bool).sum()
            ),
            "failed_rope_nan_count": int(
                np.asarray(data["official_failed_rope_nan"], dtype=bool).sum()
            ),
            "final_reward_nan_count": int(
                np.asarray(data["official_final_reward_nan"], dtype=bool).sum()
            ),
        },
        "batch_count": batch_count,
        "candidate_count": int(mining["candidate_count"]),
        "minimum_candidates_to_proceed": min_candidates,
        "funnel": mining["funnel"],
        "mixed_winding_time_histograms": mining["mixed_winding_time_histograms"],
        "winding_index_residual": mining["winding_index_residual"],
        "partial_state_observation": mining["partial_state_observation"],
        "hidden_state": {
            "descriptor": config["hidden_state"]["descriptor"],
            "task_native": True,
            "model_input": False,
            "candidate_residual_saved": True,
        },
        "selection_contract": {
            "official_valid_rollouts_only": True,
            "same_time_only": True,
            "common_action_history": True,
            "future_divergence_used": False,
            "force_or_sensor_used": False,
        },
        "top_candidates": mining["top_candidates"],
        "next_action": next_action,
        "non_claims": [
            (
                "The rope observation is an artificial partial-state "
                "surrogate, not a demonstrated deployable perception stack."
            ),
            (
                "PB2-C discovery candidates are not yet formal same-action "
                "future-bifurcation evidence."
            ),
            (
                "The PB2-B 0.25/0.75 semantic bands are not used for PB2-C "
                "pair admission."
            ),
            (
                "A low candidate count does not prove CCDA is absent from "
                "Wrapping."
            ),
        ],
    }

    raw_root = Path(config["outputs"]["raw_root"])
    raw_root.mkdir(parents=True, exist_ok=True)
    raw_path = raw_root / f"MINING_{str(label).upper()}.json"
    raw_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    committed = REPO_ROOT / config["outputs"]["committed_report_dir"]
    committed.mkdir(parents=True, exist_ok=True)

    evidence = {
        "phase_name": config["phase_name"],
        "verdict": verdict,
        "repository": {
            "starting_main_sha": config["provenance"]["starting_main_sha"],
            "ending_main_sha": git("rev-parse", "HEAD"),
            "dlolab_gitlink": git("rev-parse", "HEAD:external/dlo-lab"),
        },
        "scientific": result,
    }
    (committed / "EVIDENCE.json").write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
    )
    (committed / "CANDIDATES.json").write_text(
        json.dumps(
            {
                "candidate_count": int(mining["candidate_count"]),
                "top_candidates": mining["top_candidates"],
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    obs = mining["partial_state_observation"]
    robot = obs["robot_component"]
    lines = [
        "# PB2-C REV1 DLO-Lab Wrapping Natural Pair Discovery",
        "",
        f"Verdict: `{verdict}`",
        "",
        "## Rollout validity",
        "",
        f"- Total rollouts: {rollout_count}",
        f"- Official-valid rollouts: {official_valid_count}",
        f"- Official-invalid rollouts: {rollout_count - official_valid_count}",
        (
            "- Stretch failures: "
            f"{result['official_validity_summary']['failed_stretch_count']}"
        ),
        (
            "- Rope-NaN failures: "
            f"{result['official_validity_summary']['failed_rope_nan_count']}"
        ),
        (
            "- Final-reward-NaN invalid: "
            f"{result['official_validity_summary']['final_reward_nan_count']}"
        ),
        "- Pair mining uses official-valid full rollouts only: True",
        "",
        "## Observation semantics",
        "",
        "- Rope observation: artificial post-local partial-state XYZ history",
        "- Claimed deployable: False",
        "- Rope velocity used for selection: False",
        "- Robot observation per arm: EE xyz + EE quaternion + 7 motor-joint qpos",
        "- Post state: fixed shared context",
        f"- Local occlusion radius: {obs['local_occlusion_radius_m']:.6f} m",
        f"- Visible-history Chamfer threshold: {obs['max_visible_history_chamfer_m']:.6f} m",
        (
            "- Dual-EE position threshold: "
            f"{robot['max_dual_ee_position_history_mean_m']:.6f} m"
        ),
        (
            "- Dual-EE quaternion threshold: "
            f"{robot['max_dual_ee_quaternion_geodesic_history_mean_rad']:.9f} rad"
        ),
        (
            "- Dual motor-qpos RMS threshold: "
            f"{robot['max_dual_motor_qpos_history_rms_rad']:.6f} rad"
        ),
        "",
        "## Funnel",
        "",
    ]
    for key, value in mining["funnel"].items():
        lines.append(f"- {key}: {value}")

    lines.extend(
        [
            "",
            "## Hidden state",
            "",
            "- Descriptor: rounded signed task-native winding index `[w1,w2,w3]`",
            (
                "- Candidate records include per-post winding integer residuals "
                "for both states and pair max residual."
            ),
            "",
            "## Next action",
            "",
            next_action,
            "",
        ]
    )
    (committed / "RESULT.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"verdict={verdict}")
    print(f"candidate_count={mining['candidate_count']}")
    print(f"raw={raw_path}")
    return result


def run(config_path, datasets, label):
    config = load_json(config_path)
    data = load_datasets(datasets)

    mining = mine_pairs(
        data,
        config,
    )

    return build_report(
        config,
        data,
        mining,
        label,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--label",
        required=True,
    )
    args = parser.parse_args()

    run(
        args.config,
        args.datasets,
        args.label,
    )


if __name__ == "__main__":
    main()
