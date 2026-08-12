"""PB3-R1: sequential acquisition-history replay diagnosis.

This diagnostic does NOT resume PB3, run future suffixes, or modify the frozen
50 um PB3 alignment criterion. Each repeat is launched in a fresh Python
process.

M0: exact PB2-C batch0 seed123 -> batch1 seed124 prefix to t13.
M1: isolated batch1 seed124 prefix to t13.
M2: build_state reset -> isolated batch1 seed124 prefix to t13.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import torch

from scripts.experiment3.dlolab_wrapping.collect_natural_rollouts_pb2c import (
    _initial_validity,
    _update_official_validity,
    replay_batch,
)
from scripts.experiment3.dlolab_wrapping.pair_discovery_pb2c import (
    signed_winding_index,
)
from scripts.experiment3.dlolab_wrapping.paths import REPO_ROOT, official_log_dir
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
    "replay_alignment_root_cause_pb3r1"
)


def git(*args):
    return subprocess.check_output(
        ["git", *args], cwd=REPO_ROOT, text=True
    ).strip()


def seed_everything(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def _jsonable(value):
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    return value


def spatial_error_metrics(live_rope, frozen_rope):
    live = np.asarray(live_rope, dtype=np.float64)
    frozen = np.asarray(frozen_rope, dtype=np.float64)
    if live.shape != frozen.shape or live.ndim != 2 or live.shape[1] != 3:
        raise ValueError(
            f"Expected matching [N,3] ropes, got {live.shape} / {frozen.shape}"
        )

    diff = live - frozen
    vertex_l2 = np.linalg.norm(diff, axis=-1)
    argmax = int(np.argmax(vertex_l2))
    translation = np.mean(diff, axis=0)
    centered = diff - translation[None, :]

    coord_rmse = float(np.sqrt(np.mean(diff * diff)))
    centered_rmse = float(np.sqrt(np.mean(centered * centered)))

    return {
        "coordinate_rmse_m": coord_rmse,
        "rope_max_abs_coordinate_m": float(np.max(np.abs(diff))),
        "max_vertex_l2_m": float(vertex_l2[argmax]),
        "argmax_vertex": argmax,
        "argmax_signed_xyz_difference_m": [float(v) for v in diff[argmax]],
        "median_vertex_l2_m": float(np.median(vertex_l2)),
        "min_vertex_l2_m": float(np.min(vertex_l2)),
        "mean_translation_xyz_m": [float(v) for v in translation],
        "mean_translation_norm_m": float(np.linalg.norm(translation)),
        "translation_removed_coordinate_rmse_m": centered_rmse,
        "translation_removed_fraction": (
            0.0 if coord_rmse == 0.0 else float(centered_rmse / coord_rmse)
        ),
    }


def robot_error_metrics(live, frozen, frozen_index, time_index):
    ee_max = float(
        max(
            np.max(
                np.abs(
                    live["ee1_pos"]
                    - frozen["ee1_pos"][frozen_index, time_index]
                )
            ),
            np.max(
                np.abs(
                    live["ee2_pos"]
                    - frozen["ee2_pos"][frozen_index, time_index]
                )
            ),
        )
    )
    qpos_max = float(
        max(
            np.max(
                np.abs(
                    live["motor_qpos_1"]
                    - frozen["motor_qpos_1"][frozen_index, time_index]
                )
            ),
            np.max(
                np.abs(
                    live["motor_qpos_2"]
                    - frozen["motor_qpos_2"][frozen_index, time_index]
                )
            ),
        )
    )

    live_index = signed_winding_index(live["signed_winding_turns"]).tolist()
    frozen_winding_index = signed_winding_index(
        frozen["signed_winding_turns"][frozen_index, time_index]
    ).tolist()

    return {
        "ee_max_abs_m": ee_max,
        "motor_qpos_max_abs_rad": qpos_max,
        "live_winding_index": live_index,
        "frozen_winding_index": frozen_winding_index,
        "winding_index_exact": bool(live_index == frozen_winding_index),
    }


def validate_sources(config):
    pb3_evidence = REPO_ROOT / config["source"]["pb3_evidence"]
    if not pb3_evidence.is_file():
        raise FileNotFoundError(str(pb3_evidence))

    evidence = load_json(pb3_evidence)
    expected = config["source"]["expected_pb3_verdict"]
    if evidence.get("verdict") != expected:
        raise RuntimeError(
            f"PB3 source verdict mismatch: {evidence.get('verdict')} != {expected}"
        )

    scientific = evidence["scientific"]
    if scientific.get("audit") is not None:
        raise RuntimeError("PB3-R1 source unexpectedly contains Gate-4 audit")
    if scientific.get("trajectories_path") is not None:
        raise RuntimeError("PB3-R1 source unexpectedly contains PB3 trajectories")

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
        data = {key: np.asarray(handle[key]) for key in handle.files}

    rollout_id = int(config["target"]["rollout_id"])
    matches = np.flatnonzero(data["rollout_id"] == rollout_id)
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one frozen rollout {rollout_id}, got {len(matches)}"
        )
    index = int(matches[0])

    target = config["target"]
    actual_meta = {
        "batch_index": int(data["batch_index"][index]),
        "env_index": int(data["env_index"][index]),
        "batch_seed": int(data["batch_seed"][index]),
    }
    expected_meta = {
        "batch_index": int(target["batch_index"]),
        "env_index": int(target["env_index"]),
        "batch_seed": int(target["batch_seed"]),
    }
    if actual_meta != expected_meta:
        raise RuntimeError(
            f"Frozen target metadata mismatch: {actual_meta} != {expected_meta}"
        )

    if not bool(data["official_rollout_valid"][index]):
        raise RuntimeError("Target rollout is not official-valid in PB2-C")

    return data, index


def load_qpos_and_validate_frozen(frozen):
    qpos_path = official_log_dir() / "best_qpos.npy"
    if not qpos_path.is_file():
        raise FileNotFoundError(str(qpos_path))

    qpos = normalize_qpos(np.load(qpos_path))
    if not np.array_equal(qpos, frozen["common_qpos_replay"]):
        raise RuntimeError(
            "Official best_qpos differs from frozen PB2-C common replay"
        )
    return qpos


def replay_prefix_exact(
        env,
        qpos,
        *,
        seed,
        target_time,
        stretch_ratio_limit):
    """Mirror the PB2-C collector call pattern through target_time."""
    seed_everything(seed)
    env.use_qpos = True
    env.reset()

    total_micro_steps = int(qpos.shape[0] - 1)
    validity = _initial_validity(env, total_micro_steps)

    first = sample_env(env)
    target_row = first if int(target_time) == 0 else None

    n_intervals = env.steps_interval // env._cmaes_n_steps_sub
    if n_intervals <= 0:
        raise RuntimeError("Invalid Wrapping replay interval count")

    for global_step in range(1, int(target_time) + 1):
        _dual_arm_command(env, qpos[global_step])
        for _ in range(n_intervals):
            env.scene.step()

        _update_official_validity(
            env,
            validity,
            global_step=global_step,
            stretch_ratio_limit=stretch_ratio_limit,
        )
        target_row = sample_env(env)

    return {
        "t0": first,
        "target": target_row,
        "target_alive": validity["alive"].copy(),
        "target_failed_stretch": validity["failed_stretch"].copy(),
        "target_failed_rope_nan": validity["failed_rope_nan"].copy(),
    }


def extract_env(row, env_index):
    keys = (
        "rope_xyz",
        "ee1_pos",
        "ee2_pos",
        "motor_qpos_1",
        "motor_qpos_2",
        "signed_winding_turns",
    )
    return {key: np.asarray(row[key][env_index]).copy() for key in keys}


def alignment_at_time(live, frozen, frozen_index, time_index, config):
    spatial = spatial_error_metrics(
        live["rope_xyz"],
        frozen["rope_xyz"][frozen_index, time_index],
    )
    robot = robot_error_metrics(
        live, frozen, frozen_index, time_index
    )

    align = config["frozen_pb3_alignment"]
    valid = (
        spatial["rope_max_abs_coordinate_m"] <= float(align["rope_max_abs_m"])
        and robot["ee_max_abs_m"] <= float(align["ee_max_abs_m"])
        and robot["motor_qpos_max_abs_rad"]
        <= float(align["motor_qpos_max_abs_rad"])
    )
    if align["require_winding_index_exact"]:
        valid = valid and robot["winding_index_exact"]

    return {
        "frozen_pb3_alignment_pass": bool(valid),
        **spatial,
        **robot,
    }


def worker_run(config_path, mode, repeat_index, output_root):
    config = load_json(config_path)
    validate_sources(config)
    frozen, frozen_index = load_frozen(config)
    qpos = load_qpos_and_validate_frozen(frozen)
    target = config["target"]

    env = build_env(
        n_envs=int(config["replay"]["n_envs"]),
        n_steps_sub=int(config["replay"]["n_steps_sub"]),
        log_dir=official_log_dir(),
    )
    env.init_domain_randomization(**wrapping_args)

    try:
        if mode == "m0_sequential_history":
            # Exact original PB2-C full preceding batch. Using the original
            # collector function preserves sample_env/reward/validity call order.
            replay_batch(
                env,
                qpos,
                int(target["preceding_batch_seed"]),
                float(config["replay"]["official_stretch_ratio_limit"]),
            )
            prefix = replay_prefix_exact(
                env,
                qpos,
                seed=int(target["batch_seed"]),
                target_time=int(target["time_index"]),
                stretch_ratio_limit=float(
                    config["replay"]["official_stretch_ratio_limit"]
                ),
            )

        elif mode == "m1_isolated_collector":
            prefix = replay_prefix_exact(
                env,
                qpos,
                seed=int(target["batch_seed"]),
                target_time=int(target["time_index"]),
                stretch_ratio_limit=float(
                    config["replay"]["official_stretch_ratio_limit"]
                ),
            )

        elif mode == "m2_build_state_restore":
            build_state = env.scene.get_state()
            env.scene.reset(state=build_state)
            prefix = replay_prefix_exact(
                env,
                qpos,
                seed=int(target["batch_seed"]),
                target_time=int(target["time_index"]),
                stretch_ratio_limit=float(
                    config["replay"]["official_stretch_ratio_limit"]
                ),
            )
        else:
            raise ValueError(f"Unknown PB3-R1 mode: {mode}")

        env_index = int(target["env_index"])
        t0_live = extract_env(prefix["t0"], env_index)
        target_live = extract_env(prefix["target"], env_index)

        if not bool(prefix["target_alive"][env_index]):
            raise RuntimeError("Target became invalid before t=13")

        t0_alignment = alignment_at_time(
            t0_live, frozen, frozen_index, 0, config
        )
        target_alignment = alignment_at_time(
            target_live,
            frozen,
            frozen_index,
            int(target["time_index"]),
            config,
        )

        output_root = Path(output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        stem = f"{mode}_repeat{int(repeat_index)}"

        npz_path = output_root / f"{stem}.npz"
        np.savez_compressed(
            npz_path,
            t0_rope_xyz=t0_live["rope_xyz"],
            target_rope_xyz=target_live["rope_xyz"],
            target_ee1_pos=target_live["ee1_pos"],
            target_ee2_pos=target_live["ee2_pos"],
            target_signed_winding_turns=target_live["signed_winding_turns"],
        )

        result = {
            "mode": mode,
            "repeat_index": int(repeat_index),
            "fresh_process": True,
            "target": target,
            "t0_alignment": t0_alignment,
            "target_alignment": target_alignment,
            "target_official_alive": True,
            "target_failed_stretch": bool(
                prefix["target_failed_stretch"][env_index]
            ),
            "target_failed_rope_nan": bool(
                prefix["target_failed_rope_nan"][env_index]
            ),
            "npz_path": str(npz_path),
        }

        (output_root / f"{stem}.json").write_text(
            json.dumps(result, indent=2) + "\n",
            encoding="utf-8",
        )

        print(
            json.dumps(
                {
                    "mode": mode,
                    "repeat_index": int(repeat_index),
                    "rope_max_abs_m": target_alignment[
                        "rope_max_abs_coordinate_m"
                    ],
                    "coordinate_rmse_m": target_alignment[
                        "coordinate_rmse_m"
                    ],
                    "old_alignment_pass": target_alignment[
                        "frozen_pb3_alignment_pass"
                    ],
                }
            )
        )
        return result
    finally:
        env.stop()


def pairwise_repeat_metrics(records):
    rows = []
    for left, right in itertools.combinations(records, 2):
        with np.load(left["npz_path"]) as a:
            rope_a = np.asarray(a["target_rope_xyz"])
        with np.load(right["npz_path"]) as b:
            rope_b = np.asarray(b["target_rope_xyz"])
        rows.append(spatial_error_metrics(rope_a, rope_b))

    if not rows:
        return {
            "pair_count": 0,
            "max_coordinate_rmse_m": None,
            "max_rope_max_abs_coordinate_m": None,
            "max_max_vertex_l2_m": None,
        }

    return {
        "pair_count": len(rows),
        "max_coordinate_rmse_m": float(
            max(row["coordinate_rmse_m"] for row in rows)
        ),
        "max_rope_max_abs_coordinate_m": float(
            max(row["rope_max_abs_coordinate_m"] for row in rows)
        ),
        "max_max_vertex_l2_m": float(
            max(row["max_vertex_l2_m"] for row in rows)
        ),
    }


def summarize_mode(records):
    alignments = [row["target_alignment"] for row in records]
    t0_alignments = [
        row["t0_alignment"]
        for row in records
    ]

    return {
        "repeat_count": len(records),
        "all_t0_alignment_pass": bool(
            all(
                row["frozen_pb3_alignment_pass"]
                for row in t0_alignments
            )
        ),
        "all_original_pb3_alignment_pass": bool(
            all(row["frozen_pb3_alignment_pass"] for row in alignments)
        ),
        "original_pb3_alignment_pass_count": int(
            sum(row["frozen_pb3_alignment_pass"] for row in alignments)
        ),
        "median_coordinate_rmse_m": float(
            np.median([row["coordinate_rmse_m"] for row in alignments])
        ),
        "median_rope_max_abs_coordinate_m": float(
            np.median(
                [row["rope_max_abs_coordinate_m"] for row in alignments]
            )
        ),
        "median_max_vertex_l2_m": float(
            np.median([row["max_vertex_l2_m"] for row in alignments])
        ),
        "median_translation_removed_coordinate_rmse_m": float(
            np.median(
                [
                    row["translation_removed_coordinate_rmse_m"]
                    for row in alignments
                ]
            )
        ),
        "target_alignments": alignments,
        "fresh_process_repeat_spread": pairwise_repeat_metrics(records),
    }


def classify_history_dependence(config, summaries):
    m0 = summaries["m0_sequential_history"]
    m1 = summaries["m1_isolated_collector"]
    limit = float(
        config["diagnostic_decision"][
            "history_dependence_material_reduction_ratio"
        ]
    )

    coord_ratio = (
        m0["median_coordinate_rmse_m"]
        / max(m1["median_coordinate_rmse_m"], np.finfo(np.float64).tiny)
    )
    max_abs_ratio = (
        m0["median_rope_max_abs_coordinate_m"]
        / max(
            m1["median_rope_max_abs_coordinate_m"],
            np.finfo(np.float64).tiny,
        )
    )

    t0_valid = bool(
        m0["all_t0_alignment_pass"]
        and m1["all_t0_alignment_pass"]
    )

    confirmed = bool(
        t0_valid
        and m0["all_original_pb3_alignment_pass"]
        and coord_ratio <= limit
        and max_abs_ratio <= limit
    )

    return {
        "t0_reconstruction_valid_for_m0_m1":
            t0_valid,
        "history_dependence_confirmed": confirmed,
        "m0_over_m1_coordinate_rmse_ratio": float(coord_ratio),
        "m0_over_m1_rope_max_abs_ratio": float(max_abs_ratio),
        "material_reduction_ratio_limit": limit,
        "m0_all_original_pb3_alignment_pass": bool(
            m0["all_original_pb3_alignment_pass"]
        ),
    }


def worker_plan(config):
    plan = []
    for mode, row in config["replay"]["modes"].items():
        for repeat_index in range(int(row["repeats"])):
            plan.append((mode, repeat_index))
    return plan


def aggregate(config_path):
    config = load_json(config_path)
    validate_sources(config)

    frozen, _ = load_frozen(config)
    load_qpos_and_validate_frozen(frozen)

    raw_root = Path(config["outputs"]["raw_root"])
    raw_root.mkdir(parents=True, exist_ok=True)

    records_by_mode = {
        mode: [] for mode in config["replay"]["modes"]
    }

    for mode, repeat_index in worker_plan(config):
        subprocess.run(
            [
                sys.executable,
                "-m",
                MODULE,
                "--config",
                str(config_path),
                "--worker",
                "--mode",
                mode,
                "--repeat-index",
                str(repeat_index),
                "--output-root",
                str(raw_root),
            ],
            cwd=REPO_ROOT,
            check=True,
        )

        record_path = raw_root / f"{mode}_repeat{repeat_index}.json"
        records_by_mode[mode].append(load_json(record_path))

    summaries = {
        mode: summarize_mode(records)
        for mode, records in records_by_mode.items()
    }
    decision = classify_history_dependence(config, summaries)

    if not decision["t0_reconstruction_valid_for_m0_m1"]:
        verdict = "PB3R1_T0_RECONSTRUCTION_FAILED"
        next_action = (
            "Fix t0 replay equivalence only. Do not interpret acquisition-"
            "history dependence and do not modify PB3 thresholds."
        )
    elif decision["history_dependence_confirmed"]:
        verdict = "PB3R1_ACQUISITION_HISTORY_DEPENDENCE_CONFIRMED"
        next_action = (
            "Do not resume frozen PB3 snapshot Gate 4 yet. Determine whether "
            "Genesis snapshot state captures the history-dependent dynamics; "
            "if not, replace snapshot repeats with full acquisition-history "
            "causal replays while preserving the frozen shortlist and Gate 4."
        )
    else:
        verdict = (
            "PB3R1_ACQUISITION_HISTORY_NOT_CONFIRMED_"
            "REPLAY_FLOOR_CALIBRATION_REQUIRED"
        )
        next_action = (
            "Proceed to PB3-R2 independent replay reconstruction-floor "
            "calibration using official-valid non-shortlist PB2-C states. "
            "Do not change the frozen PB3 50 um alignment threshold before "
            "that calibration."
        )

    result = {
        "verdict": verdict,
        "target": config["target"],
        "scope": {
            "future_suffix_executed": False,
            "gate4_executed": False,
            "pb3_threshold_modified": False,
            "pb3_shortlist_modified": False,
            "pb2c_frozen_data_modified": False,
        },
        "mode_summaries": summaries,
        "decision": decision,
        "next_action": next_action,
    }

    (raw_root / "AUDIT.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )

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
        json.dumps(evidence, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# PB3-R1 Wrapping Replay Alignment Root-Cause Audit",
        "",
        f"Verdict: `{verdict}`",
        "",
        "## Scope",
        "",
        "- Formal PB3 remains blocked: Yes",
        "- Future suffix executed: No",
        "- Gate 4 executed: No",
        "- Frozen PB3 threshold changed: No",
        "- Frozen PB3 shortlist changed: No",
        "",
        "## Target",
        "",
        f"- Rollout: {config['target']['rollout_id']}",
        (
            "- Batch/env/seed: "
            f"{config['target']['batch_index']} / "
            f"{config['target']['env_index']} / "
            f"{config['target']['batch_seed']}"
        ),
        f"- Time index: {config['target']['time_index']}",
        "",
        "## Fresh-process modes",
        "",
    ]

    for mode, summary in summaries.items():
        lines.extend(
            [
                f"### {mode}",
                "",
                f"- Repeats: {summary['repeat_count']}",
                (
                    "- t0 alignment valid for all repeats: "
                    f"{summary['all_t0_alignment_pass']}"
                ),
                (
                    "- Original 50 um alignment passes: "
                    f"{summary['original_pb3_alignment_pass_count']} / "
                    f"{summary['repeat_count']}"
                ),
                (
                    "- Median coordinate RMSE: "
                    f"{summary['median_coordinate_rmse_m']:.9e} m"
                ),
                (
                    "- Median rope max-abs: "
                    f"{summary['median_rope_max_abs_coordinate_m']:.9e} m"
                ),
                (
                    "- Median max vertex L2: "
                    f"{summary['median_max_vertex_l2_m']:.9e} m"
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Decision",
            "",
            (
                "- M0/M1 coordinate-RMSE ratio: "
                f"{decision['m0_over_m1_coordinate_rmse_ratio']:.6f}"
            ),
            (
                "- M0/M1 rope-max-abs ratio: "
                f"{decision['m0_over_m1_rope_max_abs_ratio']:.6f}"
            ),
            (
                "- Material reduction criterion: "
                f"<= {decision['material_reduction_ratio_limit']:.3f}"
            ),
            "",
            "## Next action",
            "",
            next_action,
            "",
        ]
    )

    (committed / "RESULT.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(f"verdict={verdict}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--mode")
    parser.add_argument("--repeat-index", type=int)
    parser.add_argument("--output-root")
    args = parser.parse_args()

    if args.worker:
        if (
            args.mode is None
            or args.repeat_index is None
            or args.output_root is None
        ):
            raise ValueError(
                "Worker mode requires --mode --repeat-index --output-root"
            )
        worker_run(
            args.config,
            args.mode,
            args.repeat_index,
            args.output_root,
        )
    else:
        aggregate(args.config)


if __name__ == "__main__":
    main()
