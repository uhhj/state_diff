#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

import phase3_12d_r2_common as common


CANDIDATE_FIELDS = [
    "query_id",
    "query_status",
    "condition",
    "visible_seed",
    "query_step",
    "candidate_id",
    "candidate_family",
    "candidate_index",
    "uses_condition_label",
    "is_oracle",
    "valid",
    "failure_reason",
    "snapshot_restore_pass",
    "restore_bead_xy_max_abs",
    "restore_bead_xy_mae",
    "restore_bead_velocity_max_abs",
    "restore_bead_velocity_mae",
    "restore_robot_q_max_abs",
    "restore_fraction_diff",
    "restore_curve_diff",
    "constraint_restored_by_fallback",
    "action_ood",
    "clip_frac",
    "pose0_x",
    "pose0_y",
    "pose1_x",
    "pose1_y",
    "pull_len",
    "action_vec_json",
    "selected_train_index",
    "selected_condition",
    "future_nearest_condition",
    "future_condition_match",
    "before_ordered_distance",
    "after_ordered_distance",
    "ordered_gain",
    "before_chamfer_distance",
    "after_chamfer_distance",
    "chamfer_gain",
    "before_soft_coverage",
    "after_soft_coverage",
    "soft_coverage_gain",
    "before_endpoint_distance",
    "after_endpoint_distance",
    "endpoint_gain",
    "before_curve",
    "after_curve",
    "curve_gain",
    "before_fraction",
    "after_fraction",
    "fraction_gain",
    "dense_gain",
    "reward",
    "done",
    "breakaway_released_before",
    "breakaway_released_after",
    "breakaway_release_action_step",
    "breakaway_release_physics_step",
    "breakaway_max_disp_seen",
    "physics_hook_error",
    "scope",
]

def assert_gate() -> None:
    if os.environ.get("PHASE3_ALLOW_QUERY_LOCAL_SNAPSHOT", "0") != "1":
        raise SystemExit(
            "[Phase3.12d-r1][BLOCKED] Set PHASE3_ALLOW_QUERY_LOCAL_SNAPSHOT=1"
        )
    if os.environ.get("PHASE3_QUERY_LOCAL_SNAPSHOT_CONFIRMED", "0") != "1":
        raise SystemExit(
            "[Phase3.12d-r1][BLOCKED] Set PHASE3_QUERY_LOCAL_SNAPSHOT_CONFIRMED=1"
        )


QUERY_FIELDS = [
    "query_id",
    "status",
    "condition",
    "visible_seed",
    "query_step",
    "prefix_actions_requested",
    "prefix_actions_completed",
    "expected_candidate_count",
    "candidate_count",
    "valid_candidate_count",
    "restore_failure_count",
    "execution_failure_count",
    "query_bead_xy_sha256",
    "query_robot_q_sha256",
    "query_fraction",
    "query_ordered_distance",
    "query_chamfer_distance",
    "query_breakaway_released",
    "snapshot_constraint_id",
    "failure_reason",
    "duration_sec",
    "scope",
]


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(fields),
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            clean = {}
            for key in fields:
                value = row.get(key, "")
                if isinstance(value, str):
                    value = value.replace("\n", "\\n").replace("\r", "\\r")
                clean[key] = value
            writer.writerow(clean)


def append_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(fields),
            extrasaction="ignore",
            lineterminator="\n",
        )
        if write_header:
            writer.writeheader()
        for row in rows:
            clean = {}
            for key in fields:
                value = row.get(key, "")
                if isinstance(value, str):
                    value = value.replace("\n", "\\n").replace("\r", "\\r")
                clean[key] = value
            writer.writerow(clean)
        file.flush()
        os.fsync(file.fileno())


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def first_state_checkpoint(root: Path, checkpoint_root: str) -> Path:
    base = root / checkpoint_root / "state_action"
    for candidate in sorted(path for path in base.glob("fold_*_seed_*") if path.is_dir()):
        if (candidate / "state_model.pt").exists():
            return candidate / "state_model.pt"
    raise FileNotFoundError("No state_model.pt below {}".format(base))


def repaired_idm_checkpoint(root: Path, summary_path: str, ablation: str) -> Path:
    payload = load_json(root / summary_path)
    value = payload.get("results", {}).get(ablation, {}).get("checkpoint")
    if not value:
        raise FileNotFoundError(
            "No checkpoint for {} in {}".format(ablation, summary_path)
        )
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    if not path.exists():
        raise FileNotFoundError(str(path))
    return path


def query_id(condition: str, seed: int, step: int) -> str:
    return "{}__seed{}__step{}".format(condition, int(seed), int(step))


def build_model_input(
    info: Dict[str, Any],
    *,
    state_hist: List[np.ndarray],
    action_hist: List[np.ndarray],
    previous_xy: Optional[np.ndarray],
    th: int,
    action_dim: int,
    n_beads: int,
):
    from ccda_phase3.rollout import pad_history, state_from_live_info

    state = state_from_live_info(info, prev_xy=previous_xy)
    new_previous_xy = state[: n_beads * 2].reshape(n_beads, 2)
    state_hist.append(np.asarray(state, dtype=np.float32))
    history_states = pad_history(state_hist, th).reshape(-1)
    if action_hist:
        history_actions = pad_history(action_hist, th).reshape(-1)
    else:
        history_actions = np.zeros((th * action_dim,), dtype=np.float32)
    model_x = np.concatenate(
        [history_states, history_actions], axis=0
    ).reshape(1, -1)
    return model_x, history_states, new_previous_xy


def execute_action(env: Any, action: Mapping[str, Any]):
    env.start()
    try:
        result = env.step(copy.deepcopy(dict(action)))
    finally:
        common.pause_and_drain(env)
    hook_error = getattr(env, "_ccda_physics_hook_error", None)
    if hook_error not in (None, "", False):
        raise RuntimeError("Physics hook error: {!r}".format(hook_error))
    return result


def predict_action(codec: Any, idm: Any, history_states: np.ndarray, future: np.ndarray, clip_std: float):
    import phase3_12b_proxy_score_rollout as p12b

    return p12b.action_predict(
        codec,
        idm,
        history_states,
        np.asarray(future, dtype=np.float32).reshape(1, -1),
        float(clip_std),
    )


def future_nn(future: np.ndarray, bank: Mapping[str, Any]) -> Dict[str, Any]:
    import phase3_12b_proxy_score_rollout as p12b

    return p12b.future_nn_stats(
        np.asarray(future, dtype=np.float32).reshape(1, -1),
        bank["y"],
        bank["labels"],
    )


def candidate_from_pred(
    *,
    candidate_id: str,
    family: str,
    index: int,
    pred: Mapping[str, Any],
    future: Optional[np.ndarray],
    bank: Mapping[str, Any],
    condition: str,
    uses_condition_label: bool,
    is_oracle: bool,
    selected_train_index: int = -1,
    selected_condition: str = "",
) -> Dict[str, Any]:
    action = copy.deepcopy(pred["action"])
    future_info = (
        future_nn(np.asarray(future), bank)
        if future is not None
        else {
            "future_nearest_condition": "",
            "future_nn_l2": float("nan"),
        }
    )
    return {
        "candidate_id": candidate_id,
        "candidate_family": family,
        "candidate_index": int(index),
        "uses_condition_label": int(bool(uses_condition_label)),
        "is_oracle": int(bool(is_oracle)),
        "action": action,
        "action_vec": [
            float(x) for x in np.asarray(pred.get("vec", [])).reshape(-1)
        ],
        "action_ood": float(pred.get("action_ood", float("nan"))),
        "clip_frac": float(pred.get("clip_frac", float("nan"))),
        "selected_train_index": int(selected_train_index),
        "selected_condition": str(selected_condition),
        "future_nearest_condition": str(
            future_info.get("future_nearest_condition", "")
        ),
        "future_condition_match": int(
            str(future_info.get("future_nearest_condition", "")) == condition
        )
        if future is not None
        else 0,
    }


def build_candidates(
    *,
    selector_spec: Mapping[str, Any],
    condition: str,
    visible_seed: int,
    model_x: np.ndarray,
    history_states: np.ndarray,
    samples: np.ndarray,
    bank: Mapping[str, Any],
    codec: Any,
    idm: Any,
    task: Any,
    th: int,
    action_dim: int,
    n_beads: int,
) -> List[Dict[str, Any]]:
    import phase3_12b_proxy_score_rollout as p12b

    candidates: List[Dict[str, Any]] = []
    clip_std = float(selector_spec["action_clip_std"])

    mean_future = np.mean(samples, axis=0)
    mean_pred = predict_action(codec, idm, history_states, mean_future, clip_std)
    candidates.append(
        candidate_from_pred(
            candidate_id="ddpm_mean",
            family="ddpm_mean",
            index=0,
            pred=mean_pred,
            future=mean_future,
            bank=bank,
            condition=condition,
            uses_condition_label=False,
            is_oracle=False,
        )
    )

    for index, sample in enumerate(samples):
        pred = predict_action(codec, idm, history_states, sample, clip_std)
        candidates.append(
            candidate_from_pred(
                candidate_id="ddpm_raw_{:02d}".format(index),
                family="ddpm_raw",
                index=index,
                pred=pred,
                future=sample,
                bank=bank,
                condition=condition,
                uses_condition_label=False,
                is_oracle=True,
            )
        )

    selected_cache: Dict[str, Dict[str, Any]] = {}
    for selector in (
        "condition_nearest_upper",
        "proxy_state_motion_nn",
        "proxy_combined_topk_action_geom",
    ):
        selected = p12b.select_future(
            selector,
            condition,
            model_x,
            history_states,
            samples,
            bank,
            codec,
            idm,
            dict(selector_spec),
            th,
            action_dim,
            n_beads,
        )
        selected_cache[selector] = selected
        future = np.asarray(selected["future"], dtype=np.float32).reshape(-1)
        pred = predict_action(codec, idm, history_states, future, clip_std)
        candidates.append(
            candidate_from_pred(
                candidate_id=selector,
                family=selector,
                index=0,
                pred=pred,
                future=future,
                bank=bank,
                condition=condition,
                uses_condition_label=(selector == "condition_nearest_upper"),
                is_oracle=(selector == "condition_nearest_upper"),
                selected_train_index=int(selected["selected_train_index"]),
                selected_condition=str(selected["selected_condition"]),
            )
        )

    rng = np.random.default_rng(int(visible_seed) + 7919)
    random_index = int(rng.integers(0, len(bank["y"])))
    random_future = np.asarray(bank["y"][random_index], dtype=np.float32)
    random_pred = predict_action(
        codec, idm, history_states, random_future, clip_std
    )
    candidates.append(
        candidate_from_pred(
            candidate_id="random_train_future",
            family="random_train_future",
            index=random_index,
            pred=random_pred,
            future=random_future,
            bank=bank,
            condition=condition,
            uses_condition_label=False,
            is_oracle=False,
            selected_train_index=random_index,
            selected_condition=str(bank["labels"][random_index]),
        )
    )

    source_selected = selected_cache["condition_nearest_upper"]
    source_index = int(source_selected["selected_train_index"])
    source_vec = np.asarray(bank["ya"][source_index], dtype=np.float32).reshape(-1)
    source_action = codec.decode(source_vec)
    source_pred = {
        "action": source_action,
        "vec": source_vec,
        "action_ood": float(idm.ood_score(source_vec.reshape(1, -1))[0]),
        "clip_frac": 0.0,
    }
    candidates.append(
        candidate_from_pred(
            candidate_id="retrieved_source_action",
            family="retrieved_source_action",
            index=source_index,
            pred=source_pred,
            future=None,
            bank=bank,
            condition=condition,
            uses_condition_label=True,
            is_oracle=True,
            selected_train_index=source_index,
            selected_condition=str(bank["labels"][source_index]),
        )
    )

    goal_action = common.goal_geometry_action(task)
    goal_vec = common.serialize_action_vector(codec, goal_action)
    goal_ood = (
        float(idm.ood_score(np.asarray(goal_vec).reshape(1, -1))[0])
        if len(goal_vec) == action_dim
        else float("nan")
    )
    candidates.append(
        {
            "candidate_id": "goal_geometry_oracle",
            "candidate_family": "goal_geometry_oracle",
            "candidate_index": 0,
            "uses_condition_label": 0,
            "is_oracle": 1,
            "action": goal_action,
            "action_vec": goal_vec,
            "action_ood": goal_ood,
            "clip_frac": 0.0,
            "selected_train_index": -1,
            "selected_condition": "",
            "future_nearest_condition": "",
            "future_condition_match": 0,
        }
    )

    expected = int(selector_spec["raw_samples"]) + 7
    if len(candidates) != expected:
        raise RuntimeError(
            "Candidate count mismatch: expected {}, got {}".format(
                expected, len(candidates)
            )
        )
    return candidates


def setup_runtime(spec: Mapping[str, Any]):
    root = Path(spec["root"]).resolve()
    common.add_repo_paths(root)

    import phase3_12b_proxy_score_rollout as p12b
    import phase3_12c_matched_reset_common as p12c
    import phase3_policy_rollout as p34
    from ccda_phase3.data_io import load_action_codec_from_template
    from ccda_phase3.train_utils import load_future_model, load_inverse_model

    p34.set_selected_recoverable_env_defaults()
    tasks, Environment = p34.require_runtime(root)
    common.assert_no_forbidden_modules("query_worker_runtime")

    data = np.load(root / spec["windows"], allow_pickle=True)
    th = int(common.scalar_from_npz(data, "th"))
    action_dim = int(common.scalar_from_npz(data, "action_dim"))
    n_beads = int(common.scalar_from_npz(data, "n_beads"))
    if action_dim != 14:
        raise RuntimeError("Expected 14D action codec, got {}".format(action_dim))

    template_value = common.scalar_from_npz(
        data, "action_template_json_or_pickle_path"
    )
    template_path = Path(str(template_value))
    if not template_path.is_absolute():
        template_path = root / template_path
    codec = load_action_codec_from_template(template_path)
    if codec.dim() != 14:
        raise RuntimeError("Codec dimension is not 14.")
    if int(codec.summary().get("num_camera_config_paths", 0) or 0) != 0:
        raise RuntimeError("Codec contains camera_config paths.")

    bank = p12b.build_bank(data, th, action_dim, n_beads)
    state_model = load_future_model(Path(spec["state_model_path"]))
    idm = load_inverse_model(Path(spec["idm_path"]))

    # Load models before reset; then reset all RNGs immediately before task
    # construction so checkpoint loading cannot perturb scene initialization.
    common.seed_everything(int(spec["visible_seed"]))
    pair_group = common.set_ccda_environment(
        str(spec["condition"]), int(spec["visible_seed"])
    )

    task = tasks.names["hidden-contact-cable-line"]()
    task.mode = "train"
    env = Environment(disp=False, hz=240)
    env.t_lim = float(spec["motion_timeout"])
    info, reset_meta = common.deterministic_reset_deferred_arm(
        env,
        task,
        min_settle_steps=int(spec["min_settle_steps"]),
        max_settle_steps=int(spec["max_settle_steps"]),
        static_checks_required=int(spec["static_checks_required"]),
        static_check_interval=int(spec["static_check_interval"]),
    )
    common.pause_and_drain(env)

    return {
        "root": root,
        "p34": p34,
        "p12b": p12b,
        "p12c": p12c,
        "task": task,
        "env": env,
        "info": info,
        "reset_meta": reset_meta,
        "data": data,
        "th": th,
        "action_dim": action_dim,
        "n_beads": n_beads,
        "codec": codec,
        "bank": bank,
        "state_model": state_model,
        "idm": idm,
        "pair_group": pair_group,
    }


def close_runtime(runtime: Mapping[str, Any]) -> None:
    env = runtime.get("env")
    p34 = runtime.get("p34")
    if env is None:
        return
    try:
        common.pause_and_drain(env)
    except Exception:
        pass
    try:
        p34.close_env_safely(env)
    except Exception:
        try:
            env.stop()
        except Exception:
            pass


def run_query(spec: Mapping[str, Any]) -> Dict[str, Any]:
    started = time.time()
    query = query_id(
        str(spec["condition"]),
        int(spec["visible_seed"]),
        int(spec["query_step"]),
    )
    query_row: Dict[str, Any] = {
        "query_id": query,
        "status": "running",
        "condition": str(spec["condition"]),
        "visible_seed": int(spec["visible_seed"]),
        "query_step": int(spec["query_step"]),
        "prefix_actions_requested": int(spec["query_step"]),
        "prefix_actions_completed": 0,
        "expected_candidate_count": int(spec["raw_samples"]) + 7,
        "candidate_count": 0,
        "valid_candidate_count": 0,
        "restore_failure_count": 0,
        "execution_failure_count": 0,
        "failure_reason": "",
        "scope": "phase3_12d_r2_query_local_snapshot_no_phase4_no_cps",
    }
    rows: List[Dict[str, Any]] = []
    runtime: Dict[str, Any] = {}
    snapshot: Optional[common.QuerySnapshot] = None

    try:
        runtime = setup_runtime(spec)
        task = runtime["task"]
        env = runtime["env"]
        info = runtime["info"]
        th = runtime["th"]
        action_dim = runtime["action_dim"]
        n_beads = runtime["n_beads"]
        state_model = runtime["state_model"]
        idm = runtime["idm"]
        codec = runtime["codec"]
        bank = runtime["bank"]

        state_hist: List[np.ndarray] = []
        action_hist: List[np.ndarray] = []
        previous_xy: Optional[np.ndarray] = None

        # Execute the DDPM-mean prefix once. No candidate repeats this prefix.
        for prefix_step in range(int(spec["query_step"])):
            model_x, history_states, previous_xy = build_model_input(
                info,
                state_hist=state_hist,
                action_hist=action_hist,
                previous_xy=previous_xy,
                th=th,
                action_dim=action_dim,
                n_beads=n_beads,
            )
            samples = state_model.sample(
                model_x,
                n_samples=int(spec["raw_samples"]),
                seed=int(spec["visible_seed"]) * 100 + prefix_step,
            )[:, 0, :].astype(np.float32)
            future = np.mean(samples, axis=0)
            predicted = predict_action(
                codec,
                idm,
                history_states,
                future,
                float(spec["action_clip_std"]),
            )
            _, _, done, info = execute_action(env, predicted["action"])
            action_hist.append(
                np.asarray(predicted["vec"], dtype=np.float32)
            )
            query_row["prefix_actions_completed"] = prefix_step + 1
            if done and prefix_step + 1 < int(spec["query_step"]):
                raise RuntimeError(
                    "Episode ended before requested query step."
                )

        model_x, history_states, previous_xy = build_model_input(
            info,
            state_hist=state_hist,
            action_hist=action_hist,
            previous_xy=previous_xy,
            th=th,
            action_dim=action_dim,
            n_beads=n_beads,
        )
        samples = state_model.sample(
            model_x,
            n_samples=int(spec["raw_samples"]),
            seed=int(spec["visible_seed"]) * 100
            + int(spec["query_step"]),
        )[:, 0, :].astype(np.float32)

        candidates = build_candidates(
            selector_spec=spec,
            condition=str(spec["condition"]),
            visible_seed=int(spec["visible_seed"]),
            model_x=model_x,
            history_states=history_states,
            samples=samples,
            bank=bank,
            codec=codec,
            idm=idm,
            task=task,
            th=th,
            action_dim=action_dim,
            n_beads=n_beads,
        )

        snapshot = common.QuerySnapshot.create(task, env)
        reference = snapshot.reference
        query_row.update(
            {
                "query_bead_xy_sha256": reference["bead_xy_sha256"],
                "query_robot_q_sha256": reference["robot_q_sha256"],
                "query_fraction": reference["metrics"]["fraction"],
                "query_ordered_distance": reference["metrics"][
                    "ordered_distance"
                ],
                "query_chamfer_distance": reference["metrics"][
                    "chamfer_distance"
                ],
                "query_breakaway_released": int(
                    reference["breakaway_released"]
                ),
                "snapshot_constraint_id": reference[
                    "breakaway_constraint_id"
                ],
            }
        )

        for candidate in candidates:
            base_row: Dict[str, Any] = {
                "query_id": query,
                "query_status": "running",
                "condition": str(spec["condition"]),
                "visible_seed": int(spec["visible_seed"]),
                "query_step": int(spec["query_step"]),
                "candidate_id": candidate["candidate_id"],
                "candidate_family": candidate["candidate_family"],
                "candidate_index": candidate["candidate_index"],
                "uses_condition_label": candidate["uses_condition_label"],
                "is_oracle": candidate["is_oracle"],
                "valid": 0,
                "failure_reason": "",
                "action_ood": candidate["action_ood"],
                "clip_frac": candidate["clip_frac"],
                "action_vec_json": json.dumps(candidate["action_vec"]),
                "selected_train_index": candidate[
                    "selected_train_index"
                ],
                "selected_condition": candidate["selected_condition"],
                "future_nearest_condition": candidate[
                    "future_nearest_condition"
                ],
                "future_condition_match": candidate[
                    "future_condition_match"
                ],
                "scope": "phase3_12d_r2_query_local_snapshot_no_phase4_no_cps",
            }
            base_row.update(common.action_pose_metrics(candidate["action"]))

            try:
                integrity = snapshot.restore(
                    task,
                    env,
                    bead_xy_max_abs_threshold=float(
                        spec["restore_bead_xy_max_abs"]
                    ),
                    bead_xy_mae_threshold=float(
                        spec["restore_bead_xy_mae"]
                    ),
                    robot_q_max_abs_threshold=float(
                        spec["restore_robot_q_max_abs"]
                    ),
                    fraction_threshold=float(
                        spec["restore_fraction_threshold"]
                    ),
                    curve_threshold=float(
                        spec["restore_curve_threshold"]
                    ),
                )
                base_row.update(
                    {
                        "snapshot_restore_pass": int(
                            integrity["pass"]
                        ),
                        "restore_bead_xy_max_abs": integrity[
                            "bead_xy_max_abs"
                        ],
                        "restore_bead_xy_mae": integrity["bead_xy_mae"],
                        "restore_bead_velocity_max_abs": integrity[
                            "bead_velocity_max_abs"
                        ],
                        "restore_bead_velocity_mae": integrity[
                            "bead_velocity_mae"
                        ],
                        "restore_robot_q_max_abs": integrity[
                            "robot_q_max_abs"
                        ],
                        "restore_fraction_diff": integrity[
                            "fraction_diff"
                        ],
                        "restore_curve_diff": integrity["curve_diff"],
                        "constraint_restored_by_fallback": int(
                            integrity["constraint_restore"].get(
                                "recreated", False
                            )
                        ),
                    }
                )
                if not integrity["pass"]:
                    query_row["restore_failure_count"] += 1
                    base_row["failure_reason"] = (
                        "snapshot_restore_integrity_failed"
                    )
                    rows.append(base_row)
                    continue

                before = common.geometry_metrics(task)
                released_before = bool(
                    getattr(task, "_breakaway_released", False)
                )
                _, reward, done, after_info = execute_action(
                    env, candidate["action"]
                )
                after = common.geometry_metrics(task)
                effect = common.dense_effect(before, after)
                base_row.update(
                    {
                        "valid": 1,
                        "before_ordered_distance": before[
                            "ordered_distance"
                        ],
                        "after_ordered_distance": after[
                            "ordered_distance"
                        ],
                        "ordered_gain": effect["ordered_gain"],
                        "before_chamfer_distance": before[
                            "chamfer_distance"
                        ],
                        "after_chamfer_distance": after[
                            "chamfer_distance"
                        ],
                        "chamfer_gain": effect["chamfer_gain"],
                        "before_soft_coverage": before[
                            "soft_coverage"
                        ],
                        "after_soft_coverage": after["soft_coverage"],
                        "soft_coverage_gain": effect[
                            "soft_coverage_gain"
                        ],
                        "before_endpoint_distance": before[
                            "endpoint_distance"
                        ],
                        "after_endpoint_distance": after[
                            "endpoint_distance"
                        ],
                        "endpoint_gain": effect["endpoint_gain"],
                        "before_curve": before["curve"],
                        "after_curve": after["curve"],
                        "curve_gain": effect["curve_gain"],
                        "before_fraction": before["fraction"],
                        "after_fraction": after["fraction"],
                        "fraction_gain": effect["fraction_gain"],
                        "dense_gain": effect["dense_gain"],
                        "reward": float(reward),
                        "done": int(bool(done)),
                        "breakaway_released_before": int(
                            released_before
                        ),
                        "breakaway_released_after": int(
                            bool(
                                getattr(
                                    task, "_breakaway_released", False
                                )
                            )
                        ),
                        "breakaway_release_action_step": getattr(
                            task, "_breakaway_release_step", None
                        ),
                        "breakaway_release_physics_step": getattr(
                            task,
                            "_breakaway_release_physics_step",
                            None,
                        ),
                        "breakaway_max_disp_seen": float(
                            getattr(
                                task, "_breakaway_max_disp_seen", 0.0
                            )
                        ),
                        "physics_hook_error": repr(
                            getattr(
                                env, "_ccda_physics_hook_error", None
                            )
                        ),
                    }
                )
                query_row["valid_candidate_count"] += 1
            except Exception as exc:
                query_row["execution_failure_count"] += 1
                base_row["failure_reason"] = repr(exc)
            rows.append(base_row)

        query_row["candidate_count"] = len(rows)
        if (
            query_row["candidate_count"]
            == query_row["expected_candidate_count"]
            and query_row["valid_candidate_count"]
            == query_row["expected_candidate_count"]
            and query_row["restore_failure_count"] == 0
            and query_row["execution_failure_count"] == 0
        ):
            query_row["status"] = "complete"
        else:
            query_row["status"] = "incomplete"
            query_row["failure_reason"] = (
                "candidate_set_incomplete_or_invalid"
            )

    except Exception as exc:
        query_row["status"] = "failed"
        query_row["failure_reason"] = repr(exc)
    finally:
        if snapshot is not None:
            snapshot.remove()
        if runtime:
            close_runtime(runtime)
        query_row["duration_sec"] = float(time.time() - started)
        for row in rows:
            row["query_status"] = query_row["status"]

    return {"query": query_row, "candidates": rows}


def run_worker(args: argparse.Namespace) -> None:
    assert_gate()
    spec = json.loads(Path(args.worker_json).read_text())
    result = run_query(spec)
    output = Path(args.worker_out_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    common.write_json_atomic(output, result)
    print(json.dumps(result["query"], indent=2, sort_keys=True))
    if result["query"]["status"] != "complete":
        raise SystemExit(2)


def load_worker_results(worker_dir: Path) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    if not worker_dir.exists():
        return results
    for path in sorted(worker_dir.glob("*.result.json")):
        try:
            payload = json.loads(path.read_text())
            query = payload.get("query", {})
            qid = str(query.get("query_id", ""))
            if qid:
                results[qid] = payload
        except Exception:
            continue
    return results


def aggregate_worker_results(
    worker_dir: Path, query_csv: Path, candidate_csv: Path
) -> Dict[str, Any]:
    results = load_worker_results(worker_dir)
    query_rows: List[Dict[str, Any]] = []
    candidate_rows: List[Dict[str, Any]] = []
    for qid in sorted(results):
        payload = results[qid]
        query_rows.append(payload.get("query", {}))
        candidate_rows.extend(payload.get("candidates", []))
    write_csv(query_csv, query_rows, QUERY_FIELDS)
    write_csv(candidate_csv, candidate_rows, CANDIDATE_FIELDS)
    return {
        "results": results,
        "query_rows": len(query_rows),
        "candidate_rows": len(candidate_rows),
        "complete": sum(
            row.get("status") == "complete" for row in query_rows
        ),
    }


def build_specs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    root = Path(args.root).resolve()
    state_checkpoint = first_state_checkpoint(
        root, args.old_checkpoint_root
    )
    idm_checkpoint = repaired_idm_checkpoint(
        root, args.phase39b_raw, args.best_ablation
    )

    specs: List[Dict[str, Any]] = []
    for condition in args.conditions:
        for seed in args.visible_seeds:
            for step in args.query_steps:
                specs.append(
                    {
                        "root": str(root),
                        "condition": condition,
                        "visible_seed": int(seed),
                        "query_step": int(step),
                        "windows": args.windows,
                        "state_model_path": str(state_checkpoint),
                        "idm_path": str(idm_checkpoint),
                        "raw_samples": int(args.raw_samples),
                        "top_k": int(args.top_k),
                        "motion_timeout": float(args.motion_timeout),
                        "action_clip_std": float(args.action_clip_std),
                        "score_proxy_weight": float(
                            args.score_proxy_weight
                        ),
                        "score_ood_weight": float(args.score_ood_weight),
                        "score_clip_weight": float(
                            args.score_clip_weight
                        ),
                        "score_action_mae_weight": float(
                            args.score_action_mae_weight
                        ),
                        "score_pull_diff_weight": float(
                            args.score_pull_diff_weight
                        ),
                        "score_small_pull_weight": float(
                            args.score_small_pull_weight
                        ),
                        "min_pull": float(args.min_pull),
                        "min_settle_steps": int(args.min_settle_steps),
                        "max_settle_steps": int(args.max_settle_steps),
                        "static_checks_required": int(
                            args.static_checks_required
                        ),
                        "static_check_interval": int(
                            args.static_check_interval
                        ),
                        "restore_bead_xy_max_abs": float(
                            args.restore_bead_xy_max_abs
                        ),
                        "restore_bead_xy_mae": float(
                            args.restore_bead_xy_mae
                        ),
                        "restore_robot_q_max_abs": float(
                            args.restore_robot_q_max_abs
                        ),
                        "restore_fraction_threshold": float(
                            args.restore_fraction_threshold
                        ),
                        "restore_curve_threshold": float(
                            args.restore_curve_threshold
                        ),
                    }
                )
    return specs


def run_parent(args: argparse.Namespace) -> None:
    assert_gate()
    root = Path(args.root).resolve()
    worker_dir = root / args.worker_dir
    query_csv = root / args.out_query_csv
    candidate_csv = root / args.out_candidate_csv
    progress_path = root / args.progress_json
    raw_path = root / args.out_json

    if worker_dir.exists() and not args.resume:
        import shutil
        shutil.rmtree(worker_dir)
    worker_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        for path in (query_csv, candidate_csv):
            if path.exists():
                path.unlink()

    existing_results = load_worker_results(worker_dir)
    completed = {
        qid for qid, payload in existing_results.items()
        if payload.get("query", {}).get("status") == "complete"
    }

    specs = [
        spec
        for spec in build_specs(args)
        if query_id(
            spec["condition"], spec["visible_seed"], spec["query_step"]
        )
        not in completed
    ]
    progress = {
        "status": "running",
        "total_queries": len(completed) + len(specs),
        "completed_queries": len(completed),
        "complete": len(completed),
        "failed": 0,
        "timeout": 0,
        "candidate_rows": sum(
            len(payload.get("candidates", []))
            for payload in existing_results.values()
        ),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    common.write_json_atomic(progress_path, progress)

    script = Path(__file__).resolve()
    start_all = time.time()

    for index, spec in enumerate(specs):
        if (
            args.total_timeout_sec > 0
            and time.time() - start_all > args.total_timeout_sec
        ):
            progress["status"] = "stopped_total_timeout"
            break

        qid = query_id(
            spec["condition"], spec["visible_seed"], spec["query_step"]
        )
        spec_path = worker_dir / "{}.spec.json".format(qid)
        out_path = worker_dir / "{}.result.json".format(qid)
        err_path = worker_dir / "{}.stderr.txt".format(qid)
        common.write_json_atomic(spec_path, spec)

        child_env = os.environ.copy()
        child_env["PYTHONHASHSEED"] = str(spec["visible_seed"])
        command = [
            sys.executable,
            str(script),
            "--worker-json",
            str(spec_path),
            "--worker-out-json",
            str(out_path),
        ]

        try:
            process = subprocess.run(
                command,
                cwd=str(root),
                env=child_env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=float(args.query_timeout_sec),
            )
            err_path.write_text(
                ((process.stderr or "") + "\n" + (process.stdout or ""))[
                    -20000:
                ]
            )
            if not out_path.exists():
                raise RuntimeError(
                    "Worker produced no result: returncode={}".format(
                        process.returncode
                    )
                )
            result = json.loads(out_path.read_text())
        except subprocess.TimeoutExpired as exc:
            progress["timeout"] += 1
            progress["failed"] += 1
            progress["completed_queries"] += 1
            err_path.write_text(str(exc))
            progress["last_query"] = {
                "query_id": qid,
                "status": "timeout",
            }
            common.write_json_atomic(progress_path, progress)
            continue
        except Exception as exc:
            progress["failed"] += 1
            progress["completed_queries"] += 1
            progress["last_query"] = {
                "query_id": qid,
                "status": "parent_failure",
                "error": repr(exc),
            }
            common.write_json_atomic(progress_path, progress)
            continue

        aggregate = aggregate_worker_results(
            worker_dir, query_csv, candidate_csv
        )
        progress["completed_queries"] += 1
        progress["candidate_rows"] = aggregate["candidate_rows"]
        if result["query"]["status"] == "complete":
            progress["complete"] += 1
        else:
            progress["failed"] += 1
        progress["last_query"] = {
            "query_id": qid,
            "status": result["query"]["status"],
            "candidate_count": result["query"]["candidate_count"],
            "valid_candidate_count": result["query"][
                "valid_candidate_count"
            ],
        }
        common.write_json_atomic(progress_path, progress)

        if (
            args.stop_on_query_failure
            and result["query"]["status"] != "complete"
        ):
            progress["status"] = "stopped_query_failure"
            break

    if progress["status"] == "running":
        progress["status"] = "completed"
    progress["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    common.write_json_atomic(progress_path, progress)

    aggregate = aggregate_worker_results(
        worker_dir, query_csv, candidate_csv
    )
    progress["candidate_rows"] = aggregate["candidate_rows"]
    common.write_json_atomic(progress_path, progress)

    raw = {
        "status": progress["status"],
        "progress": progress,
        "matrix": {
            "conditions": args.conditions,
            "visible_seeds": args.visible_seeds,
            "query_steps": args.query_steps,
            "raw_samples": args.raw_samples,
            "candidates_per_query": args.raw_samples + 7,
            "expected_queries": (
                len(args.conditions)
                * len(args.visible_seeds)
                * len(args.query_steps)
            ),
            "expected_candidate_rows": (
                len(args.conditions)
                * len(args.visible_seeds)
                * len(args.query_steps)
                * (args.raw_samples + 7)
            ),
        },
        "scope": "phase3_12d_r2_query_local_snapshot_no_phase4_no_cps",
    }
    common.write_json_atomic(raw_path, raw)
    print(json.dumps(raw, indent=2, sort_keys=True))
    if progress["status"] != "completed" or progress["failed"] > 0:
        raise SystemExit(2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument(
        "--windows",
        default="data/phase3_state_diff_windows/phase3_windows.npz",
    )
    parser.add_argument(
        "--old-checkpoint-root", default="checkpoints/phase3"
    )
    parser.add_argument(
        "--phase39b-raw",
        default="reports/phase3_9b_ablation_raw_summary.json",
    )
    parser.add_argument(
        "--best-ablation", default="xy_only_high_weight"
    )
    parser.add_argument(
        "--conditions", nargs="+", default=list(common.DEFAULT_CONDITIONS)
    )
    parser.add_argument(
        "--visible-seeds",
        nargs="+",
        type=int,
        default=list(common.DEFAULT_VISIBLE_SEEDS),
    )
    parser.add_argument(
        "--query-steps", nargs="+", type=int, default=[0, 4]
    )
    parser.add_argument("--raw-samples", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--motion-timeout", type=float, default=15.0)
    parser.add_argument("--action-clip-std", type=float, default=3.0)
    parser.add_argument("--score-proxy-weight", type=float, default=1.0)
    parser.add_argument("--score-ood-weight", type=float, default=0.15)
    parser.add_argument("--score-clip-weight", type=float, default=2.0)
    parser.add_argument(
        "--score-action-mae-weight", type=float, default=4.0
    )
    parser.add_argument(
        "--score-pull-diff-weight", type=float, default=2.0
    )
    parser.add_argument(
        "--score-small-pull-weight", type=float, default=2.0
    )
    parser.add_argument("--min-pull", type=float, default=0.08)
    parser.add_argument("--min-settle-steps", type=int, default=540)
    parser.add_argument("--max-settle-steps", type=int, default=2400)
    parser.add_argument(
        "--static-checks-required", type=int, default=8
    )
    parser.add_argument(
        "--static-check-interval", type=int, default=10
    )
    parser.add_argument(
        "--restore-bead-xy-max-abs", type=float, default=1e-7
    )
    parser.add_argument(
        "--restore-bead-xy-mae", type=float, default=1e-8
    )
    parser.add_argument(
        "--restore-robot-q-max-abs", type=float, default=1e-7
    )
    parser.add_argument(
        "--restore-fraction-threshold", type=float, default=1e-12
    )
    parser.add_argument(
        "--restore-curve-threshold", type=float, default=1e-9
    )
    parser.add_argument(
        "--query-timeout-sec", type=float, default=2400.0
    )
    parser.add_argument(
        "--total-timeout-sec", type=float, default=86400.0
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--stop-on-query-failure",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--out-query-csv",
        default="reports/phase3_12d_r2_queries.csv",
    )
    parser.add_argument(
        "--out-candidate-csv",
        default="reports/phase3_12d_r2_candidate_effects.csv",
    )
    parser.add_argument(
        "--progress-json",
        default="reports/phase3_12d_r2_progress.json",
    )
    parser.add_argument(
        "--out-json",
        default="reports/phase3_12d_r2_raw_summary.json",
    )
    parser.add_argument(
        "--worker-dir", default="reports/phase3_12d_r2_workers"
    )
    parser.add_argument("--worker-json", default="")
    parser.add_argument("--worker-out-json", default="")
    args = parser.parse_args()

    if args.worker_json:
        run_worker(args)
    else:
        run_parent(args)


if __name__ == "__main__":
    main()
