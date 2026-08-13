"""PB3-R3 — independent-calibration alignment preregistration.

CPU-only. This phase derives and freezes the future PB3 targeted-replay
alignment rule from completed PB3-R2 calibration artifacts. It does not run
Genesis, PB3 future suffixes, Gate 4, PB4, or training.

Formal rope threshold:
    for each of 20 independent PB3-R2 calibration states:
        median of its 3 fresh-process frozen-reconstruction coordinate RMSEs
    threshold = maximum of those 20 state-level medians

No formal PB3 future outcome is used. No arbitrary multiplier is used.

This module also contains pure helpers that the later PB3 Resume implementation
must reuse for:
- targeted replay alignment;
- live PB2-C pair revalidation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from scripts.experiment3.dlolab_wrapping.paths import REPO_ROOT
from scripts.experiment3.dlolab_wrapping.replay_winding_audit import load_json
from scripts.experiment3.dlolab_wrapping.pair_discovery_pb2c import (
    batch_symmetric_chamfer,
    compute_visibility_mask,
    occlusion_radius,
    quaternion_geodesic_rad,
    signed_winding_index,
)


def git(*args):
    return subprocess.check_output(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
    ).strip()


def coordinate_rmse_m(live_rope_xyz, frozen_rope_xyz):
    """Coordinate-wise RMSE over all material-vertex XYZ coordinates."""
    live = np.asarray(live_rope_xyz, dtype=np.float64)
    frozen = np.asarray(frozen_rope_xyz, dtype=np.float64)
    if live.shape != frozen.shape or live.ndim != 2 or live.shape[1] != 3:
        raise ValueError(
            f"Expected matching [N,3] rope arrays, got {live.shape}/{frozen.shape}"
        )
    delta = live - frozen
    if not np.isfinite(delta).all():
        raise ValueError("Non-finite rope alignment input")
    return float(np.sqrt(np.mean(delta * delta)))


def rope_max_abs_coordinate_m(live_rope_xyz, frozen_rope_xyz):
    live = np.asarray(live_rope_xyz, dtype=np.float64)
    frozen = np.asarray(frozen_rope_xyz, dtype=np.float64)
    return float(np.max(np.abs(live - frozen)))


def _load_sources(config):
    paths = {
        "pb3r2_evidence": REPO_ROOT / config["source"]["pb3r2_evidence"],
        "pb3r2_summary": REPO_ROOT / config["source"]["pb3r2_summary"],
        "pb2c_config": REPO_ROOT / config["source"]["pb2c_config"],
        "pb3_config": REPO_ROOT / config["source"]["pb3_config"],
        "pb3_shortlist": REPO_ROOT / config["source"]["pb3_shortlist"],
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(str(path))

    evidence = load_json(paths["pb3r2_evidence"])
    summary = load_json(paths["pb3r2_summary"])
    pb2c = load_json(paths["pb2c_config"])
    pb3 = load_json(paths["pb3_config"])
    shortlist = load_json(paths["pb3_shortlist"])

    expected = config["source"]["expected_pb3r2_verdict"]
    if evidence.get("verdict") != expected or summary.get("verdict") != expected:
        raise RuntimeError("PB3-R2 verdict mismatch")

    scientific = evidence["scientific"]
    if scientific["scope"].get("new_alignment_threshold_selected"):
        raise RuntimeError("PB3-R2 must not already have selected a new threshold")
    if scientific["scope"].get("future_suffix_executed"):
        raise RuntimeError("PB3-R2 unexpectedly executed future suffix")
    if scientific["scope"].get("gate4_executed"):
        raise RuntimeError("PB3-R2 unexpectedly executed Gate 4")
    if not scientific["scope"].get(
        "calibration_independent_of_formal_pb3_rollouts"
    ):
        raise RuntimeError("PB3-R2 calibration independence is not confirmed")

    if scientific["target_finiteness"] != {
        "expected_target_sample_count": 60,
        "actual_target_sample_count": 60,
        "all_60_target_samples_finite": True,
    }:
        raise RuntimeError("PB3-R2 target finiteness contract mismatch")

    if len(shortlist.get("pairs", [])) != 10:
        raise RuntimeError("Formal PB3 shortlist must remain 10 pairs")
    formal_ids = {
        int(pair[key])
        for pair in shortlist["pairs"]
        for key in ("rollout_a", "rollout_b")
    }
    if len(formal_ids) != 20:
        raise RuntimeError("Formal PB3 shortlist must remain 20 unique rollouts")

    excluded = set(
        int(v)
        for v in scientific["calibration_set"]["formal_pb3_rollout_ids_excluded"]
    )
    if excluded != formal_ids:
        raise RuntimeError("PB3-R2 exclusion set differs from formal PB3 shortlist")

    gate = pb3["gate4"]
    expected_gate = config["formal_gate4"]
    checks = {
        "absolute_effect_min_m": float(gate["absolute_effect_min_m"]),
        "repeat_floor_multiplier": float(gate["repeat_floor_multiplier"]),
        "minimum_passing_pairs": int(gate["minimum_passing_pairs"]),
        "minimum_passing_winding_strata": int(
            gate["minimum_passing_winding_strata"]
        ),
    }
    for key, value in checks.items():
        if value != expected_gate[key]:
            raise RuntimeError(f"Formal Gate 4 changed at {key}: {value}")

    return {
        "evidence": evidence,
        "summary": summary,
        "pb2c": pb2c,
        "pb3": pb3,
        "shortlist": shortlist,
        "formal_rollout_ids": sorted(formal_ids),
    }


def _expected_worker_ids():
    return [
        f"batch{batch}_repeat{repeat}"
        for batch in range(4)
        for repeat in range(3)
    ]


def load_pb3r2_worker_rows(config, sources):
    """Load canonical logical-worker artifacts only, never attempt logs."""
    raw_root = Path(config["source"]["pb3r2_raw_root"])
    calibration_states = sources["evidence"]["scientific"]["calibration_set"]["states"]
    state_by_id = {
        row["calibration_id"]: row
        for row in calibration_states
    }
    if len(state_by_id) != 20:
        raise RuntimeError("Expected 20 independent PB3-R2 calibration states")

    all_rows = []
    worker_records = []

    for worker_id in _expected_worker_ids():
        path = raw_root / f"{worker_id}.json"
        if not path.is_file():
            raise FileNotFoundError(str(path))
        worker = load_json(path)
        worker_records.append(worker)

        batch = int(worker_id.split("_")[0].replace("batch", ""))
        repeat = int(worker_id.split("_")[1].replace("repeat", ""))

        if worker.get("worker_verdict") != "PB3R2_WORKER_COMPLETE":
            raise RuntimeError(f"{worker_id} is not WORKER_COMPLETE")
        if int(worker["batch_index"]) != batch or int(worker["repeat_index"]) != repeat:
            raise RuntimeError(f"{worker_id} metadata mismatch")
        if int(worker.get("future_steps_executed", -1)) != 20:
            raise RuntimeError(f"{worker_id} did not reach t20")
        if not bool(worker.get("all_target_samples_finite")):
            raise RuntimeError(f"{worker_id} target finiteness missing")
        if len(worker.get("target_metrics", [])) != 5:
            raise RuntimeError(f"{worker_id} must contain five target rows")

        for row in worker["target_metrics"]:
            cid = row["calibration_id"]
            if cid not in state_by_id:
                raise RuntimeError(f"Unknown calibration_id: {cid}")
            expected_state = state_by_id[cid]
            if int(row["batch_index"]) != int(expected_state["batch_index"]):
                raise RuntimeError(f"{cid} batch mismatch")
            if int(row["rollout_id"]) != int(expected_state["rollout_id"]):
                raise RuntimeError(f"{cid} rollout mismatch")
            if int(row["time_index"]) != int(expected_state["time_index"]):
                raise RuntimeError(f"{cid} time mismatch")
            if int(row["repeat_index"]) != repeat:
                raise RuntimeError(f"{cid} repeat mismatch")
            if not bool(row.get("metrics_all_finite")):
                raise RuntimeError(f"{cid} has non-finite derived metrics")
            if not bool(row.get("finiteness", {}).get("all_finite")):
                raise RuntimeError(f"{cid} has non-finite state data")

            value = float(row["metrics"]["coordinate_rmse_m"])
            if not np.isfinite(value) or value < 0.0:
                raise RuntimeError(f"{cid} coordinate RMSE invalid")

            all_rows.append(
                {
                    "calibration_id": cid,
                    "rollout_id": int(row["rollout_id"]),
                    "batch_index": int(row["batch_index"]),
                    "time_index": int(row["time_index"]),
                    "repeat_index": repeat,
                    "coordinate_rmse_m": value,
                }
            )

    if len(all_rows) != 60:
        raise RuntimeError(f"Expected 60 reconstruction rows, got {len(all_rows)}")

    keys = {
        (row["calibration_id"], row["repeat_index"])
        for row in all_rows
    }
    if len(keys) != 60:
        raise RuntimeError("Duplicate PB3-R2 calibration state/repeat rows")

    return all_rows, worker_records


def _quantile_summary(values):
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "median": float(np.quantile(arr, 0.50)),
        "p95": float(np.quantile(arr, 0.95)),
        "p99": float(np.quantile(arr, 0.99)),
        "max": float(np.max(arr)),
    }


def validate_raw_against_committed_summary(rows, summary):
    actual = _quantile_summary(
        [row["coordinate_rmse_m"] for row in rows]
    )
    expected = summary["frozen_reconstruction_floor"]["coordinate_rmse_m"]

    for key in ("count", "min", "median", "p95", "p99", "max"):
        if key == "count":
            if int(actual[key]) != int(expected[key]):
                raise RuntimeError("PB3-R2 raw/summary count mismatch")
        elif not np.isclose(
            float(actual[key]),
            float(expected[key]),
            rtol=1e-12,
            atol=1e-15,
        ):
            raise RuntimeError(
                f"PB3-R2 raw/summary mismatch for coordinate_rmse_m.{key}: "
                f"{actual[key]} != {expected[key]}"
            )
    return actual


def derive_state_level_envelope(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["calibration_id"], []).append(row)

    if len(grouped) != 20:
        raise RuntimeError("Expected 20 state-level calibration groups")

    state_rows = []
    for calibration_id in sorted(grouped):
        group = sorted(
            grouped[calibration_id],
            key=lambda row: row["repeat_index"],
        )
        repeats = [row["repeat_index"] for row in group]
        if repeats != [0, 1, 2]:
            raise RuntimeError(
                f"{calibration_id} must contain repeats [0,1,2], got {repeats}"
            )
        values = np.asarray(
            [row["coordinate_rmse_m"] for row in group],
            dtype=np.float64,
        )
        state_rows.append(
            {
                "calibration_id": calibration_id,
                "rollout_id": int(group[0]["rollout_id"]),
                "batch_index": int(group[0]["batch_index"]),
                "time_index": int(group[0]["time_index"]),
                "repeat_coordinate_rmse_m": [float(v) for v in values],
                "state_median_coordinate_rmse_m": float(np.median(values)),
                "state_min_coordinate_rmse_m": float(np.min(values)),
                "state_max_coordinate_rmse_m": float(np.max(values)),
            }
        )

    governing = max(
        state_rows,
        key=lambda row: row["state_median_coordinate_rmse_m"],
    )
    threshold = float(governing["state_median_coordinate_rmse_m"])

    return state_rows, governing, threshold


def build_alignment_rule(config, sources, state_rows, governing, threshold):
    pb2c = sources["pb2c"]
    pb3 = sources["pb3"]

    obs = pb2c["partial_state_observation"]
    rope_cfg = obs["rope_component"]
    robot_cfg = obs["robot_component"]

    radius = occlusion_radius(pb2c)
    if not np.isclose(radius, 0.05, rtol=0.0, atol=1e-12):
        raise RuntimeError(f"Unexpected PB2-C occlusion radius: {radius}")

    rule = {
        "phase": "PB3-R3",
        "verdict": "PB3R3_ALIGNMENT_RULE_PREREGISTERED",
        "source": {
            "pb3r2_verdict": sources["evidence"]["verdict"],
            "independent_calibration_states": 20,
            "fresh_repeats_per_state": 3,
            "formal_pb3_outcomes_used_for_threshold": False,
        },
        "targeted_replay_engineering_alignment": {
            "rope": {
                "formal_metric": "coordinate_rmse_m",
                "definition": config["derivation"]["coordinate_rmse_definition"],
                "threshold_m": threshold,
                "threshold_derivation": (
                    "max over 20 independent PB3-R2 calibration states of "
                    "median over 3 fresh-process frozen-reconstruction "
                    "coordinate_rmse_m values"
                ),
                "arbitrary_multiplier": None,
                "rope_max_abs_coordinate_m_role": "diagnostic_only",
                "state_level_governing_calibration_id": governing["calibration_id"],
                "state_level_governing_rollout_id": governing["rollout_id"],
                "state_level_governing_time_index": governing["time_index"],
            },
            "ee_max_abs_m": float(pb3["alignment"]["ee_max_abs_m"]),
            "motor_qpos_max_abs_rad": float(
                pb3["alignment"]["motor_qpos_max_abs_rad"]
            ),
            "require_live_winding_index_equal_frozen": bool(
                pb3["alignment"]["require_winding_index_exact"]
            ),
            "all_20_formal_branch_states_required": True,
        },
        "snapshot_restore_alignment": {
            "metric": "rope_max_abs_coordinate_m",
            "threshold_m": float(
                pb3["alignment"]["snapshot_restore_rope_max_abs_m"]
            ),
            "status": "unchanged_from_frozen_pb3",
        },
        "live_pair_revalidation": {
            "history_samples": int(obs["history_samples"]),
            "partial_rope": {
                "occlusion_radius_m": float(radius),
                "max_visible_history_chamfer_m": float(
                    rope_cfg["max_visible_history_chamfer_m"]
                ),
                "require_nonempty_visible_rope_each_frame": True,
            },
            "robot": {
                "max_dual_ee_position_history_mean_m": float(
                    robot_cfg["max_dual_ee_position_history_mean_m"]
                ),
                "max_dual_ee_quaternion_geodesic_history_mean_rad": float(
                    robot_cfg[
                        "max_dual_ee_quaternion_geodesic_history_mean_rad"
                    ]
                ),
                "max_dual_motor_qpos_history_rms_rad": float(
                    robot_cfg["max_dual_motor_qpos_history_rms_rad"]
                ),
            },
            "hidden_state": {
                "descriptor": "rounded_signed_task_native_winding_index",
                "require_live_index_vector_different": True,
            },
            "past_action": {
                "require_same_time": True,
                "require_common_action_history_equal": True,
            },
            "all_10_formal_pairs_required": True,
            "allow_pair_drop": False,
            "allow_pair_replacement": False,
            "blocked_verdict": "PB3_R3_LIVE_BRANCH_ALIGNMENT_FAILED",
        },
        "gate4": {
            "status": "unchanged",
            "absolute_effect_min_m": float(
                pb3["gate4"]["absolute_effect_min_m"]
            ),
            "repeat_floor_multiplier": float(
                pb3["gate4"]["repeat_floor_multiplier"]
            ),
            "minimum_passing_pairs": int(
                pb3["gate4"]["minimum_passing_pairs"]
            ),
            "minimum_passing_winding_strata": int(
                pb3["gate4"]["minimum_passing_winding_strata"]
            ),
        },
        "execution_order_for_future_pb3_resume": [
            "replay all required formal branch histories to branch time",
            "validate all 20 targeted-replay engineering alignments",
            "revalidate all 10 live PB2-C pair semantics using 3-frame histories",
            "if any branch/pair fails, stop with PB3_R3_LIVE_BRANCH_ALIGNMENT_FAILED",
            "only if all pass, execute snapshot restore and future suffix",
            "apply the unchanged formal Gate 4",
        ],
        "state_level_calibration_count": len(state_rows),
    }
    return rule


def targeted_replay_alignment(live, frozen, rule):
    """Pure helper for future PB3 Resume."""
    rope_rmse = coordinate_rmse_m(live["rope_xyz"], frozen["rope_xyz"])
    rope_max_abs = rope_max_abs_coordinate_m(
        live["rope_xyz"], frozen["rope_xyz"]
    )

    ee_error = float(
        max(
            np.max(np.abs(np.asarray(live["ee1_pos"]) - frozen["ee1_pos"])),
            np.max(np.abs(np.asarray(live["ee2_pos"]) - frozen["ee2_pos"])),
        )
    )
    qpos_error = float(
        max(
            np.max(
                np.abs(
                    np.asarray(live["motor_qpos_1"])
                    - frozen["motor_qpos_1"]
                )
            ),
            np.max(
                np.abs(
                    np.asarray(live["motor_qpos_2"])
                    - frozen["motor_qpos_2"]
                )
            ),
        )
    )
    live_index = signed_winding_index(live["signed_winding_turns"]).tolist()
    frozen_index = signed_winding_index(
        frozen["signed_winding_turns"]
    ).tolist()

    cfg = rule["targeted_replay_engineering_alignment"]
    valid = bool(
        rope_rmse <= float(cfg["rope"]["threshold_m"])
        and ee_error <= float(cfg["ee_max_abs_m"])
        and qpos_error <= float(cfg["motor_qpos_max_abs_rad"])
        and (
            not cfg["require_live_winding_index_equal_frozen"]
            or live_index == frozen_index
        )
    )
    return {
        "valid": valid,
        "rope_coordinate_rmse_m": rope_rmse,
        "rope_max_abs_coordinate_m_diagnostic": rope_max_abs,
        "ee_max_abs_m": ee_error,
        "motor_qpos_max_abs_rad": qpos_error,
        "live_winding_index": live_index,
        "frozen_winding_index": frozen_index,
    }


def live_pair_revalidation(
        history_a,
        history_b,
        rule,
        *,
        same_time,
        common_action_history_equal):
    """Re-evaluate original PB2-C pair semantics on live reconstructed history."""
    cfg = rule["live_pair_revalidation"]
    history = int(cfg["history_samples"])
    if len(history_a) != history or len(history_b) != history:
        raise ValueError("Live pair history length mismatch")

    required = (
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
    for sample in list(history_a) + list(history_b):
        for key in required:
            value = np.asarray(sample[key])
            if not np.isfinite(value).all():
                return {
                    "valid": False,
                    "reason": f"nonfinite_{key}",
                }

    radius = float(cfg["partial_rope"]["occlusion_radius_m"])
    chamfers = []
    nonempty = True
    for a, b in zip(history_a, history_b):
        mask_a = compute_visibility_mask(
            np.asarray(a["rope_xyz"])[None, None],
            np.asarray(a["post_xyz"])[None, None],
            radius,
        )[0, 0]
        mask_b = compute_visibility_mask(
            np.asarray(b["rope_xyz"])[None, None],
            np.asarray(b["post_xyz"])[None, None],
            radius,
        )[0, 0]
        if not mask_a.any() or not mask_b.any():
            nonempty = False
            break
        value = batch_symmetric_chamfer(
            np.asarray(a["rope_xyz"])[None],
            mask_a[None],
            np.asarray(b["rope_xyz"])[None],
            mask_b[None],
        )[0]
        chamfers.append(float(value))

    visible_history_chamfer = (
        float("inf")
        if not nonempty
        else float(np.mean(chamfers))
    )

    ee1 = [
        np.linalg.norm(np.asarray(a["ee1_pos"]) - np.asarray(b["ee1_pos"]))
        for a, b in zip(history_a, history_b)
    ]
    ee2 = [
        np.linalg.norm(np.asarray(a["ee2_pos"]) - np.asarray(b["ee2_pos"]))
        for a, b in zip(history_a, history_b)
    ]
    ee_pos_mean = 0.5 * (float(np.mean(ee1)) + float(np.mean(ee2)))

    qa1 = np.stack([np.asarray(v["ee1_quat"]) for v in history_a])
    qb1 = np.stack([np.asarray(v["ee1_quat"]) for v in history_b])
    qa2 = np.stack([np.asarray(v["ee2_quat"]) for v in history_a])
    qb2 = np.stack([np.asarray(v["ee2_quat"]) for v in history_b])
    quat_mean = 0.5 * (
        float(np.mean(quaternion_geodesic_rad(qa1, qb1)))
        + float(np.mean(quaternion_geodesic_rad(qa2, qb2)))
    )

    q1 = []
    q2 = []
    for a, b in zip(history_a, history_b):
        d1 = np.asarray(a["motor_qpos_1"]) - np.asarray(b["motor_qpos_1"])
        d2 = np.asarray(a["motor_qpos_2"]) - np.asarray(b["motor_qpos_2"])
        q1.append(float(np.sqrt(np.mean(d1 * d1))))
        q2.append(float(np.sqrt(np.mean(d2 * d2))))
    qpos_rms = 0.5 * (float(np.mean(q1)) + float(np.mean(q2)))

    index_a = signed_winding_index(
        history_a[-1]["signed_winding_turns"]
    ).tolist()
    index_b = signed_winding_index(
        history_b[-1]["signed_winding_turns"]
    ).tolist()
    winding_different = bool(index_a != index_b)

    rope_cfg = cfg["partial_rope"]
    robot_cfg = cfg["robot"]

    valid = bool(
        same_time
        and common_action_history_equal
        and nonempty
        and visible_history_chamfer
        <= float(rope_cfg["max_visible_history_chamfer_m"])
        and ee_pos_mean
        <= float(robot_cfg["max_dual_ee_position_history_mean_m"])
        and quat_mean
        <= float(
            robot_cfg["max_dual_ee_quaternion_geodesic_history_mean_rad"]
        )
        and qpos_rms
        <= float(robot_cfg["max_dual_motor_qpos_history_rms_rad"])
        and winding_different
    )

    return {
        "valid": valid,
        "same_time": bool(same_time),
        "common_action_history_equal": bool(common_action_history_equal),
        "nonempty_visible_rope_each_frame": bool(nonempty),
        "visible_rope_history_chamfer_m": visible_history_chamfer,
        "dual_ee_position_history_mean_m": ee_pos_mean,
        "dual_ee_quaternion_geodesic_history_mean_rad": quat_mean,
        "dual_motor_qpos_history_rms_rad": qpos_rms,
        "live_winding_index_a": index_a,
        "live_winding_index_b": index_b,
        "live_winding_index_different": winding_different,
    }


def write_outputs(config, sources, raw_summary, state_rows, governing, rule):
    rule_path = REPO_ROOT / config["outputs"]["alignment_rule"]
    report_dir = REPO_ROOT / config["outputs"]["committed_report_dir"]
    rule_path.parent.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    rule_path.write_text(
        json.dumps(rule, indent=2) + "\n",
        encoding="utf-8",
    )
    (report_dir / "STATE_LEVEL_CALIBRATION.json").write_text(
        json.dumps(
            {
                "metric": "coordinate_rmse_m",
                "state_level_statistic": "median_of_three",
                "state_count": 20,
                "states": state_rows,
                "governing_state": governing,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    result = {
        "verdict": "PB3R3_ALIGNMENT_RULE_PREREGISTERED",
        "scope": {
            "cpu_only": True,
            "genesis_worker_executed": False,
            "future_suffix_executed": False,
            "gate4_executed": False,
            "pb4_started": False,
            "training_started": False,
            "formal_pb3_pair_outcomes_used_for_threshold": False,
        },
        "raw_pb3r2_coordinate_rmse_reproduced": raw_summary,
        "derived_rope_coordinate_rmse_threshold_m": float(
            rule["targeted_replay_engineering_alignment"]["rope"]["threshold_m"]
        ),
        "governing_calibration_state": governing,
        "snapshot_restore_alignment_changed": False,
        "gate4_changed": False,
        "pair_drop_allowed": False,
        "pair_replacement_allowed": False,
        "next_action": (
            "Commit this rule before any formal PB3 Resume. The later Resume "
            "must first validate all 20 branch alignments and all 10 live "
            "PB2-C pair semantics, then and only then run future suffix/Gate 4."
        ),
    }

    evidence = {
        "phase_name": config["phase_name"],
        "verdict": result["verdict"],
        "repository": {
            "starting_main_sha": config["provenance"]["starting_main_sha"],
            "ending_main_sha": git("rev-parse", "HEAD"),
            "dlolab_gitlink": git("rev-parse", "HEAD:external/dlo-lab"),
        },
        "scientific": result,
    }

    (report_dir / "EVIDENCE.json").write_text(
        json.dumps(evidence, indent=2) + "\n",
        encoding="utf-8",
    )

    threshold_um = (
        1e6
        * float(
            rule["targeted_replay_engineering_alignment"]["rope"]["threshold_m"]
        )
    )
    lines = [
        "# PB3-R3 Alignment Re-Preregistration",
        "",
        "Verdict: `PB3R3_ALIGNMENT_RULE_PREREGISTERED`",
        "",
        "## Scope",
        "",
        "- CPU-only: Yes",
        "- Genesis/GPU worker executed: No",
        "- Future suffix executed: No",
        "- Gate 4 executed: No",
        "- Formal PB3 outcomes used for threshold: No",
        "",
        "## New targeted-replay rope alignment rule",
        "",
        "- Formal metric: coordinate-wise rope XYZ RMSE",
        "- Calibration states: 20 independent states",
        "- Repeats/state: 3",
        "- State statistic: median of 3 reconstruction RMSEs",
        "- Envelope: maximum of 20 state medians",
        f"- Frozen threshold: {threshold_um:.9f} um",
        "- Rope max-abs: diagnostic only",
        "",
        "## Unchanged",
        "",
        f"- EE max abs: {rule['targeted_replay_engineering_alignment']['ee_max_abs_m']:.9e} m",
        f"- Motor qpos max abs: {rule['targeted_replay_engineering_alignment']['motor_qpos_max_abs_rad']:.9e} rad",
        f"- Snapshot restore rope max-abs: {rule['snapshot_restore_alignment']['threshold_m']:.9e} m",
        "- Frozen winding-index equality: required",
        "- Gate 4: 1 mm absolute effect and 5x repeat floor",
        "",
        "## Future live pair revalidation",
        "",
        "- All 10 formal pairs must pass original PB2-C 3-frame semantics live",
        "- Pair dropping: forbidden",
        "- Pair replacement: forbidden",
        "- Any failure verdict: `PB3_R3_LIVE_BRANCH_ALIGNMENT_FAILED`",
        "",
        "## Stop",
        "",
        "Commit this rule before any PB3 Resume. Do not run formal PB3 in this phase.",
        "",
    ]
    (report_dir / "RESULT.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    return result


def preregister(config_path):
    config = load_json(config_path)
    sources = _load_sources(config)
    rows, _ = load_pb3r2_worker_rows(config, sources)
    raw_summary = validate_raw_against_committed_summary(
        rows,
        sources["summary"],
    )
    state_rows, governing, threshold = derive_state_level_envelope(rows)
    rule = build_alignment_rule(
        config,
        sources,
        state_rows,
        governing,
        threshold,
    )
    result = write_outputs(
        config,
        sources,
        raw_summary,
        state_rows,
        governing,
        rule,
    )
    print(f"verdict={result['verdict']}")
    print(f"rope_coordinate_rmse_threshold_m={threshold:.17g}")
    return result


def validate_generated_rule(config_path):
    config = load_json(config_path)
    sources = _load_sources(config)
    rows, _ = load_pb3r2_worker_rows(config, sources)
    validate_raw_against_committed_summary(rows, sources["summary"])
    state_rows, governing, threshold = derive_state_level_envelope(rows)
    expected = build_alignment_rule(
        config,
        sources,
        state_rows,
        governing,
        threshold,
    )
    rule_path = REPO_ROOT / config["outputs"]["alignment_rule"]
    if not rule_path.is_file():
        raise FileNotFoundError(str(rule_path))
    actual = load_json(rule_path)
    if actual != expected:
        raise RuntimeError(
            "Committed/generated PB3-R3 alignment rule differs from independent "
            "PB3-R2 derivation"
        )
    print("PB3-R3 alignment rule validation: PASS")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--validate-only",
        action="store_true",
    )
    args = parser.parse_args()

    if args.validate_only:
        validate_generated_rule(args.config)
    else:
        preregister(args.config)


if __name__ == "__main__":
    main()
