#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

import numpy as np

import phase3_12c_matched_reset_common as common


RESET_FIELDS = [
    "status",
    "started_at",
    "finished_at",
    "duration_sec",
    "worker_returncode",
    "worker_timeout",
    "worker_stderr_tail",
    "selector",
    "uses_condition_label",
    "condition",
    "visible_seed",
    "seed_cohort",
    "episode_idx",
    "pair_group",
    "pythonhashseed",
    "settle_steps_used",
    "settle_static",
    "static_checks_observed",
    "initial_fraction",
    "initial_curve",
    "initial_state_dim",
    "initial_state_sha256",
    "initial_state_round_sha256",
    "initial_state_json",
    "initial_bead_xy_json",
    "initial_bead_velocity_json",
    "initial_robot_proxy_json",
    "initial_condition_logged",
    "initial_pair_group_logged",
    "initial_visible_seed_logged",
    "initial_breakaway_released",
    "failure_reason",
    "scope",
]

EPISODE_FIELDS = RESET_FIELDS + [
    "success",
    "final_fraction",
    "delta_fraction",
    "num_steps",
    "mean_step_delta",
    "num_positive_steps",
    "mean_future_match",
    "mean_future_nn_l2",
    "mean_proxy_distance",
    "mean_selected_context_l2",
    "mean_selected_score",
    "mean_action_ood",
    "max_action_ood",
    "mean_clip_frac",
    "mean_pull_len",
    "mean_source_pull_len",
    "mean_action_mae_to_source",
    "mean_pull_diff_to_source",
    "mean_future_std",
]

STEP_FIELDS = [
    "selector",
    "uses_condition_label",
    "condition",
    "visible_seed",
    "seed_cohort",
    "episode_idx",
    "step_idx",
    "fraction_before",
    "fraction_after",
    "fraction_delta",
    "done_after",
    "reward",
    "future_std_mean",
    "future_nn_l2",
    "future_nn_mae",
    "future_nearest_condition",
    "future_condition_match",
    "selected_train_index",
    "selected_window_idx",
    "selected_condition",
    "selected_context_l2",
    "selected_proxy_distance",
    "selected_score",
    "action_ood",
    "clip_frac",
    "pull_len",
    "source_pull_len",
    "action_mae_to_source",
    "action_l2_to_source",
    "scope",
]


def assert_stage_gate(stage: str) -> None:
    if (
        os.environ.get(
            "PHASE3_ALLOW_MATCHED_RESET_AUDIT",
            "0",
        )
        != "1"
    ):
        raise SystemExit(
            "[Phase3.12c][BLOCKED] "
            "Set PHASE3_ALLOW_MATCHED_RESET_AUDIT=1"
        )

    if (
        os.environ.get(
            "PHASE3_MATCHED_RESET_AUDIT_CONFIRMED",
            "0",
        )
        != "1"
    ):
        raise SystemExit(
            "[Phase3.12c][BLOCKED] "
            "Set PHASE3_MATCHED_RESET_AUDIT_CONFIRMED=1"
        )

    if stage == "rollout":
        if (
            os.environ.get(
                "PHASE3_ALLOW_PAIRED_SELECTOR_REEVAL",
                "0",
            )
            != "1"
        ):
            raise SystemExit(
                "[Phase3.12c][BLOCKED] "
                "Set PHASE3_ALLOW_PAIRED_SELECTOR_REEVAL=1"
            )

        if (
            os.environ.get(
                "PHASE3_PAIRED_SELECTOR_REEVAL_CONFIRMED",
                "0",
            )
            != "1"
        ):
            raise SystemExit(
                "[Phase3.12c][BLOCKED] "
                "Set PHASE3_PAIRED_SELECTOR_REEVAL_CONFIRMED=1"
            )


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def first_old_checkpoint(
    root: Path,
    checkpoint_root: str,
) -> Path:
    base = root / checkpoint_root / "state_action"

    for candidate in sorted(
        path
        for path in base.glob("fold_*_seed_*")
        if path.is_dir()
    ):
        if (
            (candidate / "state_model.pt").exists()
            and (candidate / "inverse_dynamics.pt").exists()
        ):
            return candidate

    raise FileNotFoundError(
        f"No complete checkpoint under {base}"
    )


def repaired_checkpoint(
    root: Path,
    raw_path: str,
    ablation: str,
) -> Path:
    payload = load_json(root / raw_path)
    value = (
        payload
        .get("results", {})
        .get(ablation, {})
        .get("checkpoint")
    )

    if not value:
        raise FileNotFoundError(
            f"No checkpoint for {ablation} in {raw_path}"
        )

    path = Path(value)
    if not path.is_absolute():
        path = root / path

    if not path.exists():
        raise FileNotFoundError(str(path))

    return path


def build_specs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    root = Path(args.root).resolve()

    old_checkpoint = first_old_checkpoint(
        root,
        args.old_checkpoint_root,
    )
    state_model_path = old_checkpoint / "state_model.pt"
    idm_path = repaired_checkpoint(
        root,
        args.phase39b_raw,
        args.best_ablation,
    )

    specs: List[Dict[str, Any]] = []

    for selector in args.selectors:
        if selector not in common.SELECTORS:
            raise ValueError(
                f"Unknown selector={selector}; "
                f"valid={common.SELECTORS}"
            )

        for condition in args.conditions:
            if condition not in common.CONDITIONS:
                raise ValueError(
                    f"Unknown condition={condition}; "
                    f"valid={common.CONDITIONS}"
                )

            for episode_idx, visible_seed in enumerate(
                args.visible_seeds
            ):
                specs.append({
                    "stage": args.stage,
                    "root": str(root),
                    "selector": selector,
                    "condition": condition,
                    "visible_seed": int(visible_seed),
                    "seed_cohort": common.seed_cohort(
                        int(visible_seed)
                    ),
                    "episode_idx": int(episode_idx),
                    "windows": args.windows,
                    "action_template": args.action_template,
                    "state_model_path": str(state_model_path),
                    "idm_path": str(idm_path),
                    "max_steps": int(args.max_steps),
                    "samples_per_step": int(
                        args.samples_per_step
                    ),
                    "top_k": int(args.top_k),
                    "motion_timeout": float(
                        args.motion_timeout
                    ),
                    "action_clip_std": float(
                        args.action_clip_std
                    ),
                    "score_proxy_weight": float(
                        args.score_proxy_weight
                    ),
                    "score_ood_weight": float(
                        args.score_ood_weight
                    ),
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
                    "min_settle_steps": int(
                        args.min_settle_steps
                    ),
                    "max_settle_steps": int(
                        args.max_settle_steps
                    ),
                    "static_checks_required": int(
                        args.static_checks_required
                    ),
                    "static_check_interval": int(
                        args.static_check_interval
                    ),
                    "initial_round_decimals": int(
                        args.initial_round_decimals
                    ),
                })

    if args.max_rows > 0:
        specs = specs[: int(args.max_rows)]

    return specs


def read_completed_keys(
    path: Path,
    stage: str,
) -> Set[Tuple[str, str, int]]:
    if not path.exists():
        return set()

    completed: Set[Tuple[str, str, int]] = set()

    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            if row.get("status") != "ok":
                continue
            try:
                key = (
                    str(row["selector"]),
                    str(row["condition"]),
                    int(float(row["visible_seed"])),
                )
            except Exception:
                continue
            completed.add(key)

    return completed


def error_row(
    spec: Dict[str, Any],
    *,
    started: float,
    finished: float,
    status: str,
    stderr: str,
    returncode: Any,
) -> Dict[str, Any]:
    row = {
        key: ""
        for key in EPISODE_FIELDS
    }

    row.update({
        "status": status,
        "started_at": time.strftime(
            "%Y-%m-%dT%H:%M:%S%z",
            time.localtime(started),
        ),
        "finished_at": time.strftime(
            "%Y-%m-%dT%H:%M:%S%z",
            time.localtime(finished),
        ),
        "duration_sec": f"{finished - started:.3f}",
        "worker_returncode": returncode,
        "worker_timeout": int(
            "timeout" in status.lower()
        ),
        "worker_stderr_tail": stderr[-2000:],
        "selector": spec.get("selector", ""),
        "uses_condition_label": int(
            spec.get("selector")
            == "condition_nearest_upper"
        ),
        "condition": spec.get("condition", ""),
        "visible_seed": spec.get("visible_seed", ""),
        "seed_cohort": spec.get("seed_cohort", ""),
        "episode_idx": spec.get("episode_idx", ""),
        "pair_group": (
            f"phase3_12c_seed_"
            f"{spec.get('visible_seed', '')}"
        ),
        "failure_reason": status,
        "scope": (
            "phase3_12c_matched_reset_no_phase4_no_cps"
        ),
    })
    return row


def run_parent(args: argparse.Namespace) -> None:
    assert_stage_gate(args.stage)

    root = Path(args.root).resolve()
    output_csv = root / args.out_csv
    step_csv = root / args.out_step_csv
    progress_json = root / args.progress_json
    raw_json = root / args.out_json
    worker_dir = root / args.worker_dir

    if worker_dir.exists() and not args.resume:
        import shutil

        shutil.rmtree(worker_dir)
    worker_dir.mkdir(parents=True, exist_ok=True)

    if not args.resume:
        for path in [output_csv, step_csv]:
            if path.exists():
                path.unlink()

    completed_keys = (
        read_completed_keys(output_csv, args.stage)
        if args.resume
        else set()
    )

    specs = [
        spec
        for spec in build_specs(args)
        if (
            spec["selector"],
            spec["condition"],
            int(spec["visible_seed"]),
        )
        not in completed_keys
    ]

    fields = (
        RESET_FIELDS
        if args.stage == "reset"
        else EPISODE_FIELDS
    )

    output_header = not output_csv.exists()
    step_header = not step_csv.exists()

    progress = {
        "status": "running",
        "stage": args.stage,
        "started_at": time.strftime(
            "%Y-%m-%dT%H:%M:%S%z"
        ),
        "total_specs": len(completed_keys) + len(specs),
        "completed": len(completed_keys),
        "ok": len(completed_keys),
        "timeout": 0,
        "failed": 0,
        "last_row": None,
        "scope": (
            "phase3_12c_matched_reset_no_phase4_no_cps"
        ),
    }
    common.write_json_atomic(progress_json, progress)

    script = Path(__file__).resolve()
    start_all = time.time()

    for index, spec in enumerate(
        specs,
        start=len(completed_keys),
    ):
        if (
            args.total_timeout_sec > 0
            and time.time() - start_all
            > float(args.total_timeout_sec)
        ):
            progress["status"] = "stopped_total_timeout"
            common.write_json_atomic(
                progress_json,
                progress,
            )
            break

        spec_path = worker_dir / f"worker_{index:04d}.json"
        out_path = (
            worker_dir / f"worker_{index:04d}_out.json"
        )
        err_path = (
            worker_dir / f"worker_{index:04d}.stderr.txt"
        )

        common.write_json_atomic(spec_path, spec)

        child_env = os.environ.copy()
        child_env["PYTHONHASHSEED"] = str(
            spec["visible_seed"]
        )

        command = [
            sys.executable,
            str(script),
            "--worker-json",
            str(spec_path),
            "--worker-out-json",
            str(out_path),
        ]

        started = time.time()
        stderr_text = ""

        try:
            process = subprocess.run(
                command,
                cwd=str(root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=float(args.row_timeout_sec),
                env=child_env,
            )

            finished = time.time()
            stderr_text = (
                (process.stderr or "")
                + "\n"
                + (process.stdout or "")
            )
            err_path.write_text(stderr_text[-16000:])

            if out_path.exists():
                payload = json.loads(out_path.read_text())
                row = payload.get("row", {})
                steps = payload.get("steps", [])

                row["worker_returncode"] = (
                    process.returncode
                )
                row["worker_timeout"] = 0
                row["worker_stderr_tail"] = (
                    stderr_text[-2000:]
                )
                row["started_at"] = time.strftime(
                    "%Y-%m-%dT%H:%M:%S%z",
                    time.localtime(started),
                )
                row["finished_at"] = time.strftime(
                    "%Y-%m-%dT%H:%M:%S%z",
                    time.localtime(finished),
                )
                row["duration_sec"] = (
                    f"{finished - started:.3f}"
                )

                if (
                    process.returncode != 0
                    and row.get("status") == "ok"
                ):
                    row["status"] = (
                        "worker_returned_nonzero"
                    )
            else:
                row = error_row(
                    spec,
                    started=started,
                    finished=finished,
                    status="worker_failed_no_output",
                    stderr=stderr_text,
                    returncode=process.returncode,
                )
                steps = []

        except subprocess.TimeoutExpired as exc:
            finished = time.time()

            if isinstance(exc.stderr, str):
                stderr_text = exc.stderr
            else:
                stderr_text = str(exc.stderr)

            stderr_text = stderr_text[-16000:]
            err_path.write_text(stderr_text)

            row = error_row(
                spec,
                started=started,
                finished=finished,
                status="timeout",
                stderr=stderr_text,
                returncode="timeout",
            )
            steps = []

        common.write_csv_row(
            output_csv,
            row,
            fields,
            output_header,
        )
        output_header = False

        if args.stage == "rollout":
            for step in steps:
                common.write_csv_row(
                    step_csv,
                    step,
                    STEP_FIELDS,
                    step_header,
                )
                step_header = False

        status = str(row.get("status", ""))

        progress["completed"] += 1
        if (
            status == "ok"
            and not row.get("failure_reason")
        ):
            progress["ok"] += 1
        elif "timeout" in status.lower():
            progress["timeout"] += 1
        else:
            progress["failed"] += 1

        progress["last_row"] = {
            "status": status,
            "selector": row.get("selector", ""),
            "condition": row.get("condition", ""),
            "visible_seed": row.get(
                "visible_seed",
                "",
            ),
            "initial_fraction": row.get(
                "initial_fraction",
                "",
            ),
            "delta_fraction": row.get(
                "delta_fraction",
                "",
            ),
            "failure_reason": row.get(
                "failure_reason",
                "",
            ),
        }

        common.write_json_atomic(
            progress_json,
            progress,
        )

    if progress["status"] == "running":
        progress["status"] = "completed"
        progress["finished_at"] = time.strftime(
            "%Y-%m-%dT%H:%M:%S%z"
        )
        common.write_json_atomic(
            progress_json,
            progress,
        )

    raw = {
        "status": progress["status"],
        "stage": args.stage,
        "num_rows_written": progress["completed"],
        "ok": progress["ok"],
        "timeout": progress["timeout"],
        "failed": progress["failed"],
        "matrix": {
            "selectors": args.selectors,
            "conditions": args.conditions,
            "visible_seeds": args.visible_seeds,
            "expected_rows": (
                len(args.selectors)
                * len(args.conditions)
                * len(args.visible_seeds)
            ),
            "max_steps": args.max_steps,
            "samples_per_step": args.samples_per_step,
            "top_k": args.top_k,
        },
        "deterministic_reset": {
            "min_settle_steps": args.min_settle_steps,
            "max_settle_steps": args.max_settle_steps,
            "static_checks_required": (
                args.static_checks_required
            ),
            "static_check_interval": (
                args.static_check_interval
            ),
            "initial_round_decimals": (
                args.initial_round_decimals
            ),
            "selector_independent_pair_group": True,
            "explicit_rng_seeding": True,
            "wall_clock_settle_disabled": True,
        },
        "out_csv": args.out_csv,
        "out_step_csv": args.out_step_csv,
        "progress_json": args.progress_json,
        "scope": (
            "phase3_12c_matched_reset_no_phase4_no_cps"
        ),
    }
    common.write_json_atomic(raw_json, raw)
    print(json.dumps(raw, indent=2, sort_keys=True))


def worker_row_base(
    spec: Dict[str, Any],
    pair_group: str,
) -> Dict[str, Any]:
    selector = str(spec["selector"])
    return {
        "status": "ok",
        "selector": selector,
        "uses_condition_label": int(
            selector == "condition_nearest_upper"
        ),
        "condition": str(spec["condition"]),
        "visible_seed": int(spec["visible_seed"]),
        "seed_cohort": str(spec["seed_cohort"]),
        "episode_idx": int(spec["episode_idx"]),
        "pair_group": pair_group,
        "pythonhashseed": os.environ.get(
            "PYTHONHASHSEED",
            "",
        ),
        "failure_reason": "",
        "scope": (
            "phase3_12c_matched_reset_no_phase4_no_cps"
        ),
    }


def run_worker(args: argparse.Namespace) -> None:
    spec = json.loads(Path(args.worker_json).read_text())
    stage = str(spec["stage"])
    assert_stage_gate(stage)

    root = Path(spec["root"]).resolve()
    common.add_repo_paths(root)

    common.seed_everything(int(spec["visible_seed"]))
    pair_group = common.set_ccda_environment(
        str(spec["condition"]),
        int(spec["visible_seed"]),
    )

    import phase3_12b_proxy_score_rollout as p12b
    import phase3_policy_rollout as p34
    from ccda_phase3.data_io import (
        load_action_codec_from_template,
    )
    from ccda_phase3.rollout import (
        final_fraction_from_info,
        pad_history,
        state_from_live_info,
    )
    from ccda_phase3.train_utils import (
        load_future_model,
        load_inverse_model,
    )

    p34.set_selected_recoverable_env_defaults()
    tasks, Environment = p34.require_runtime(root)
    common.assert_no_forbidden_modules(
        "after_runtime_import"
    )

    data = np.load(
        root / spec["windows"],
        allow_pickle=True,
    )

    th = int(common.scalar_from_npz(data, "th"))
    action_dim = int(
        common.scalar_from_npz(data, "action_dim")
    )
    n_beads = int(
        common.scalar_from_npz(data, "n_beads")
    )

    if action_dim != 14:
        raise RuntimeError(
            f"Expected action_dim=14, got {action_dim}"
        )

    selector = str(spec["selector"])
    condition = str(spec["condition"])
    visible_seed = int(spec["visible_seed"])

    task = tasks.names["hidden-contact-cable-line"]()
    task.mode = "train"

    env = None
    row = worker_row_base(spec, pair_group)
    steps: List[Dict[str, Any]] = []

    try:
        env = Environment(disp=False, hz=240)
        env.t_lim = float(spec["motion_timeout"])

        info, reset_meta = common.deterministic_reset(
            env,
            task,
            min_settle_steps=int(
                spec["min_settle_steps"]
            ),
            max_settle_steps=int(
                spec["max_settle_steps"]
            ),
            static_checks_required=int(
                spec["static_checks_required"]
            ),
            static_check_interval=int(
                spec["static_check_interval"]
            ),
        )

        snapshot = common.initial_snapshot(
            info,
            n_beads=n_beads,
            round_decimals=int(
                spec["initial_round_decimals"]
            ),
        )

        row.update(reset_meta)
        row.update(snapshot)

        if stage == "reset":
            Path(args.worker_out_json).parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            Path(args.worker_out_json).write_text(
                json.dumps(
                    {
                        "row": row,
                        "steps": [],
                    },
                    indent=2,
                    sort_keys=True,
                    allow_nan=True,
                )
            )
            print(json.dumps({
                "status": row["status"],
                "stage": stage,
                "selector": selector,
                "condition": condition,
                "visible_seed": visible_seed,
                "state_hash": row[
                    "initial_state_round_sha256"
                ],
            }))
            return

        template_value = common.scalar_from_npz(
            data,
            "action_template_json_or_pickle_path",
        )
        template_path = Path(str(template_value))
        if not template_path.is_absolute():
            template_path = root / template_path

        codec = load_action_codec_from_template(
            template_path
        )

        if (
            codec.dim() != 14
            or int(
                codec.summary().get(
                    "num_camera_config_paths",
                    0,
                )
                or 0
            )
            != 0
        ):
            raise RuntimeError(
                f"Invalid action codec: {codec.summary()}"
            )

        bank = p12b.build_bank(
            data,
            th,
            action_dim,
            n_beads,
        )
        state_model = load_future_model(
            Path(spec["state_model_path"])
        )
        idm = load_inverse_model(
            Path(spec["idm_path"])
        )
        common.assert_no_forbidden_modules(
            "after_model_load"
        )

        state_hist: List[np.ndarray] = []
        action_hist: List[np.ndarray] = []
        previous_xy = None

        initial_fraction = float(
            row["initial_fraction"]
        )
        previous_fraction = initial_fraction

        step_deltas: List[float] = []
        future_matches: List[float] = []
        future_nns: List[float] = []
        proxy_distances: List[float] = []
        context_distances: List[float] = []
        selected_scores: List[float] = []
        action_oods: List[float] = []
        clip_fractions: List[float] = []
        pulls: List[float] = []
        source_pulls: List[float] = []
        action_maes: List[float] = []
        pull_differences: List[float] = []
        future_stds: List[float] = []

        success = False
        done = False

        for step_idx in range(int(spec["max_steps"])):
            # Model input is observable history only.
            # condition/y_state/y_action are used solely inside the
            # explicitly diagnostic selector implementation.
            state = state_from_live_info(
                info,
                prev_xy=previous_xy,
            )

            previous_xy = state[
                : n_beads * 2
            ].reshape(n_beads, 2)

            state_hist.append(state)

            history_states = pad_history(
                state_hist,
                th,
            ).reshape(-1)

            if action_hist:
                history_actions = pad_history(
                    action_hist,
                    th,
                ).reshape(-1)
            else:
                history_actions = np.zeros(
                    (th * action_dim,),
                    dtype=np.float32,
                )

            model_x = np.concatenate(
                [
                    history_states,
                    history_actions,
                ],
                axis=0,
            ).reshape(1, -1)

            samples = state_model.sample(
                model_x,
                n_samples=int(
                    spec["samples_per_step"]
                ),
                seed=visible_seed + step_idx,
            )[:, 0, :].astype(np.float32)

            future_std = float(
                np.mean(
                    np.std(samples, axis=0)
                )
            )
            future_stds.append(future_std)

            selected = p12b.select_future(
                selector,
                condition,
                model_x,
                history_states,
                samples,
                bank,
                codec,
                idm,
                spec,
                th,
                action_dim,
                n_beads,
            )

            future = np.asarray(
                selected["future"],
                dtype=np.float32,
            ).reshape(1, -1)

            future_nn = p12b.future_nn_stats(
                future,
                bank["y"],
                bank["labels"],
            )

            future_match = int(
                str(
                    future_nn[
                        "future_nearest_condition"
                    ]
                )
                == condition
            )

            predicted = p12b.action_predict(
                codec,
                idm,
                history_states,
                future,
                float(spec["action_clip_std"]),
            )

            action_hist.append(
                np.asarray(
                    predicted["vec"],
                    dtype=np.float32,
                )
            )

            fraction_before = previous_fraction

            try:
                # Keep physics paused during model computation.
                # Run physics only while Environment.step executes.
                env.start()
                _, reward, done, info = env.step(
                    predicted["action"]
                )
            finally:
                env.pause()

            fraction_after = final_fraction_from_info(
                info
            )
            delta = (
                fraction_after - fraction_before
                if (
                    math.isfinite(fraction_after)
                    and math.isfinite(fraction_before)
                )
                else float("nan")
            )
            previous_fraction = fraction_after

            success = bool(
                info.get("extras", {}).get(
                    "task.done",
                    done,
                )
            )

            proxy_distance = float(
                selected[
                    "selected_proxy_distance"
                ]
            )
            context_distance = float(
                selected[
                    "selected_context_l2"
                ]
            )
            selected_score = float(
                selected["selected_score"]
            )
            source_pull = float(
                selected["source_pull_len"]
            )
            predicted_pull = float(
                predicted["pull_len"]
            )
            action_mae = float(
                selected["action_mae_to_source"]
            )

            pull_difference = (
                abs(predicted_pull - source_pull)
                if (
                    math.isfinite(predicted_pull)
                    and math.isfinite(source_pull)
                )
                else float("nan")
            )

            step_deltas.append(delta)
            future_matches.append(
                float(future_match)
            )
            future_nns.append(
                float(future_nn["future_nn_l2"])
            )
            proxy_distances.append(proxy_distance)
            context_distances.append(
                context_distance
            )
            selected_scores.append(selected_score)
            action_oods.append(
                float(predicted["action_ood"])
            )
            clip_fractions.append(
                float(predicted["clip_frac"])
            )
            pulls.append(predicted_pull)
            source_pulls.append(source_pull)
            action_maes.append(action_mae)
            pull_differences.append(
                pull_difference
            )

            steps.append({
                "selector": selector,
                "uses_condition_label": int(
                    selector
                    == "condition_nearest_upper"
                ),
                "condition": condition,
                "visible_seed": visible_seed,
                "seed_cohort": common.seed_cohort(
                    visible_seed
                ),
                "episode_idx": int(
                    spec["episode_idx"]
                ),
                "step_idx": int(step_idx),
                "fraction_before": fraction_before,
                "fraction_after": fraction_after,
                "fraction_delta": delta,
                "done_after": int(bool(done)),
                "reward": reward,
                "future_std_mean": future_std,
                "future_nn_l2": future_nn[
                    "future_nn_l2"
                ],
                "future_nn_mae": future_nn[
                    "future_nn_mae"
                ],
                "future_nearest_condition": (
                    future_nn[
                        "future_nearest_condition"
                    ]
                ),
                "future_condition_match": (
                    future_match
                ),
                "selected_train_index": selected[
                    "selected_train_index"
                ],
                "selected_window_idx": selected[
                    "selected_window_idx"
                ],
                "selected_condition": selected[
                    "selected_condition"
                ],
                "selected_context_l2": (
                    context_distance
                ),
                "selected_proxy_distance": (
                    proxy_distance
                ),
                "selected_score": selected_score,
                "action_ood": predicted[
                    "action_ood"
                ],
                "clip_frac": predicted[
                    "clip_frac"
                ],
                "pull_len": predicted_pull,
                "source_pull_len": source_pull,
                "action_mae_to_source": action_mae,
                "action_l2_to_source": selected[
                    "action_l2_to_source"
                ],
                "scope": (
                    "phase3_12c_matched_reset_"
                    "no_phase4_no_cps"
                ),
            })

            if done:
                break

        final_fraction = final_fraction_from_info(
            info
        )

        row.update({
            "success": int(bool(success)),
            "final_fraction": final_fraction,
            "delta_fraction": (
                final_fraction - initial_fraction
                if (
                    math.isfinite(final_fraction)
                    and math.isfinite(
                        initial_fraction
                    )
                )
                else float("nan")
            ),
            "num_steps": len(action_hist),
            "mean_step_delta": common.finite_mean(
                step_deltas
            ),
            "num_positive_steps": sum(
                1
                for value in step_deltas
                if (
                    math.isfinite(value)
                    and value > 1e-9
                )
            ),
            "mean_future_match": (
                common.finite_mean(
                    future_matches
                )
            ),
            "mean_future_nn_l2": (
                common.finite_mean(future_nns)
            ),
            "mean_proxy_distance": (
                common.finite_mean(
                    proxy_distances
                )
            ),
            "mean_selected_context_l2": (
                common.finite_mean(
                    context_distances
                )
            ),
            "mean_selected_score": (
                common.finite_mean(
                    selected_scores
                )
            ),
            "mean_action_ood": (
                common.finite_mean(action_oods)
            ),
            "max_action_ood": (
                float(np.max(action_oods))
                if action_oods
                else float("nan")
            ),
            "mean_clip_frac": (
                common.finite_mean(
                    clip_fractions
                )
            ),
            "mean_pull_len": (
                common.finite_mean(pulls)
            ),
            "mean_source_pull_len": (
                common.finite_mean(source_pulls)
            ),
            "mean_action_mae_to_source": (
                common.finite_mean(action_maes)
            ),
            "mean_pull_diff_to_source": (
                common.finite_mean(
                    pull_differences
                )
            ),
            "mean_future_std": (
                common.finite_mean(future_stds)
            ),
        })

    except Exception as exc:
        row["status"] = "worker_exception"
        row["failure_reason"] = repr(exc)

    finally:
        if env is not None:
            try:
                env.pause()
            except Exception:
                pass

            try:
                p34.close_env_safely(env)
            except Exception:
                try:
                    env.stop()
                except Exception:
                    pass

    output = {
        "row": row,
        "steps": steps,
    }

    out_path = Path(args.worker_out_json)
    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    out_path.write_text(
        json.dumps(
            output,
            indent=2,
            sort_keys=True,
            allow_nan=True,
        )
    )

    print(json.dumps({
        "status": row.get("status"),
        "stage": stage,
        "selector": selector,
        "condition": condition,
        "visible_seed": visible_seed,
        "initial_fraction": row.get(
            "initial_fraction"
        ),
        "delta_fraction": row.get(
            "delta_fraction"
        ),
        "failure_reason": row.get(
            "failure_reason"
        ),
    }))


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        default="/data/state_diff2",
    )
    parser.add_argument(
        "--stage",
        choices=["reset", "rollout"],
        default="reset",
    )
    parser.add_argument(
        "--windows",
        default=(
            "data/phase3_state_diff_windows/"
            "phase3_windows.npz"
        ),
    )
    parser.add_argument(
        "--action_template",
        default=(
            "data/phase3_state_diff_windows/"
            "phase3_action_template.pkl"
        ),
    )
    parser.add_argument(
        "--old_checkpoint_root",
        default="checkpoints/phase3",
    )
    parser.add_argument(
        "--phase39b_raw",
        default=(
            "reports/"
            "phase3_9b_ablation_raw_summary.json"
        ),
    )
    parser.add_argument(
        "--best_ablation",
        default="xy_only_high_weight",
    )
    parser.add_argument(
        "--selectors",
        nargs="+",
        default=common.SELECTORS,
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=common.CONDITIONS,
    )
    parser.add_argument(
        "--visible_seeds",
        nargs="+",
        type=int,
        default=common.VISIBLE_SEEDS,
    )

    parser.add_argument(
        "--max_steps",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--samples_per_step",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--motion_timeout",
        type=float,
        default=15.0,
    )
    parser.add_argument(
        "--action_clip_std",
        type=float,
        default=3.0,
    )

    parser.add_argument(
        "--score_proxy_weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--score_ood_weight",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--score_clip_weight",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--score_action_mae_weight",
        type=float,
        default=4.0,
    )
    parser.add_argument(
        "--score_pull_diff_weight",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--score_small_pull_weight",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--min_pull",
        type=float,
        default=0.08,
    )

    parser.add_argument(
        "--min_settle_steps",
        type=int,
        default=540,
    )
    parser.add_argument(
        "--max_settle_steps",
        type=int,
        default=2400,
    )
    parser.add_argument(
        "--static_checks_required",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--static_check_interval",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--initial_round_decimals",
        type=int,
        default=7,
    )

    parser.add_argument(
        "--row_timeout_sec",
        type=float,
        default=900.0,
    )
    parser.add_argument(
        "--total_timeout_sec",
        type=float,
        default=43200.0,
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=96,
    )
    parser.add_argument(
        "--resume",
        action="store_true",
    )

    parser.add_argument(
        "--out_csv",
        default=(
            "reports/"
            "phase3_12c_reset_integrity_rows.csv"
        ),
    )
    parser.add_argument(
        "--out_step_csv",
        default=(
            "reports/"
            "phase3_12c_paired_selector_steps.csv"
        ),
    )
    parser.add_argument(
        "--out_json",
        default=(
            "reports/"
            "phase3_12c_reset_integrity_raw.json"
        ),
    )
    parser.add_argument(
        "--progress_json",
        default=(
            "reports/"
            "phase3_12c_reset_integrity_progress.json"
        ),
    )
    parser.add_argument(
        "--worker_dir",
        default="reports/phase3_12c_reset_workers",
    )

    parser.add_argument(
        "--worker-json",
        default="",
    )
    parser.add_argument(
        "--worker-out-json",
        default="",
    )

    args = parser.parse_args()

    if args.worker_json:
        run_worker(args)
    else:
        run_parent(args)


if __name__ == "__main__":
    main()
