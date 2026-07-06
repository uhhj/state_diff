#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROLLOUT_CONDITIONS = ["free", "hidden_breakaway_pin", "hidden_high_friction"]
POLICIES = ["old_state_action", "phase39_default_geometry", "phase39b_xy_only_high_weight"]
FORBIDDEN_PREFIXES = ["tensorflow", "ravens.agents", "ravens.models", "ravens.datasets"]

FIELDNAMES = [
    "status",
    "started_at",
    "finished_at",
    "duration_sec",
    "worker_returncode",
    "worker_timeout",
    "worker_stderr_tail",
    "policy",
    "condition",
    "visible_seed",
    "episode_idx",
    "success",
    "final_fraction",
    "final_curve",
    "num_steps",
    "mean_action_ood_score",
    "max_action_ood_score",
    "mean_future_sample_std",
    "mean_pose0_xy_step_delta",
    "mean_pose1_xy_step_delta",
    "mean_pull_xy_len",
    "failure_reason",
    "checkpoint_state_model",
    "checkpoint_inverse_dynamics",
    "primary_hidden_condition",
    "diagnostic_hidden_condition",
    "scope",
]


def assert_gate() -> None:
    if os.environ.get("PHASE3_ALLOW_CONTROLLED_LEARNED_ROLLOUT", "0") != "1":
        raise SystemExit("[Phase3.10][BLOCKED] Set PHASE3_ALLOW_CONTROLLED_LEARNED_ROLLOUT=1")
    if os.environ.get("PHASE3_CONTROLLED_LEARNED_ROLLOUT_CONFIRMED", "0") != "1":
        raise SystemExit("[Phase3.10][BLOCKED] Set PHASE3_CONTROLLED_LEARNED_ROLLOUT_CONFIRMED=1")


def assert_no_forbidden(stage: str) -> None:
    bad: List[str] = []
    for name in sys.modules:
        for prefix in FORBIDDEN_PREFIXES:
            if name == prefix or name.startswith(prefix + "."):
                bad.append(name)
    if bad:
        raise RuntimeError(f"[Phase3.10][FAIL] forbidden modules loaded at {stage}: {bad[:30]}")


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(path)


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def write_row(csv_path: Path, row: Dict[str, Any], write_header: bool) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in FIELDNAMES})
        f.flush()
        os.fsync(f.fileno())


def error_row(spec: Dict[str, Any], started: float, finished: float, status: str, stderr: str, returncode: Any = "") -> Dict[str, Any]:
    row = {k: "" for k in FIELDNAMES}
    row.update({
        "status": status,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(started)),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(finished)),
        "duration_sec": f"{finished - started:.3f}",
        "worker_returncode": returncode,
        "worker_timeout": 1 if "timeout" in status else 0,
        "worker_stderr_tail": stderr[-2000:],
        "policy": spec.get("policy", ""),
        "condition": spec.get("condition", ""),
        "visible_seed": spec.get("visible_seed", ""),
        "episode_idx": spec.get("episode_idx", ""),
        "failure_reason": status,
        "checkpoint_state_model": spec.get("state_model_path", ""),
        "checkpoint_inverse_dynamics": spec.get("idm_path", ""),
        "primary_hidden_condition": "hidden_breakaway_pin",
        "diagnostic_hidden_condition": "hidden_pin",
        "scope": "phase3_10_controlled_learned_rollout_no_phase4_no_cps",
    })
    return row


def first_old_checkpoint(root: Path, checkpoint_root: str, baseline: str = "state_action") -> Path:
    base = root / checkpoint_root / baseline
    for cand in sorted(p for p in base.glob("fold_*_seed_*") if p.is_dir()):
        if (cand / "state_model.pt").exists() and (cand / "inverse_dynamics.pt").exists():
            return cand
    raise FileNotFoundError(f"No complete old checkpoint under {base}")


def checkpoint_from_phase39_train(root: Path, path: str) -> Path:
    summary = load_json(root / path)
    p = summary.get("inverse_dynamics_path")
    if not p:
        raise FileNotFoundError(f"No inverse_dynamics_path in {path}")
    pp = Path(p)
    if not pp.is_absolute():
        pp = root / pp
    if not pp.exists():
        raise FileNotFoundError(str(pp))
    return pp


def checkpoint_from_phase39b_raw(root: Path, raw_path: str, ablation: str) -> Path:
    raw = load_json(root / raw_path)
    p = raw.get("results", {}).get(ablation, {}).get("checkpoint")
    if not p:
        raise FileNotFoundError(f"No checkpoint for ablation={ablation} in {raw_path}")
    pp = Path(p)
    if not pp.is_absolute():
        pp = root / pp
    if not pp.exists():
        raise FileNotFoundError(str(pp))
    return pp


def build_policy_table(args: argparse.Namespace) -> Dict[str, Dict[str, str]]:
    root = Path(args.root).resolve()
    old_ckpt = first_old_checkpoint(root, args.old_checkpoint_root, "state_action")
    default_idm = checkpoint_from_phase39_train(root, args.phase39_train_summary)
    best_idm = checkpoint_from_phase39b_raw(root, args.phase39b_raw, args.best_ablation)

    state_model = old_ckpt / "state_model.pt"
    old_idm = old_ckpt / "inverse_dynamics.pt"

    return {
        "old_state_action": {
            "state_model_path": str(state_model),
            "idm_path": str(old_idm),
        },
        "phase39_default_geometry": {
            "state_model_path": str(state_model),
            "idm_path": str(default_idm),
        },
        f"phase39b_{args.best_ablation}": {
            "state_model_path": str(state_model),
            "idm_path": str(best_idm),
        },
    }


def build_specs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    policies = build_policy_table(args)
    wanted = args.policies
    missing = [p for p in wanted if p not in policies]
    if missing:
        raise RuntimeError(f"unknown policies={missing}; available={sorted(policies)}")

    specs: List[Dict[str, Any]] = []
    for policy in wanted:
        for condition in args.conditions:
            for ep in range(int(args.episodes_per_condition)):
                visible_seed = int(args.seed_start) + ep
                specs.append({
                    "root": str(Path(args.root).resolve()),
                    "policy": policy,
                    "condition": condition,
                    "episode_idx": ep,
                    "visible_seed": visible_seed,
                    "state_model_path": policies[policy]["state_model_path"],
                    "idm_path": policies[policy]["idm_path"],
                    "windows": args.windows,
                    "action_template": args.action_template,
                    "max_steps": args.max_steps,
                    "samples_per_step": args.samples_per_step,
                    "motion_timeout": args.motion_timeout,
                    "action_clip_std": args.action_clip_std,
                })
    if args.max_rows > 0:
        specs = specs[: args.max_rows]
    return specs


def run_parent(args: argparse.Namespace) -> None:
    assert_gate()
    root = Path(args.root).resolve()
    csv_path = root / args.out_csv
    progress_path = root / args.progress_json
    raw_path = root / args.out_json
    worker_dir = root / args.worker_dir

    if worker_dir.exists() and not args.resume:
        import shutil
        shutil.rmtree(worker_dir)
    worker_dir.mkdir(parents=True, exist_ok=True)

    if csv_path.exists() and not args.resume:
        csv_path.unlink()

    write_header = not csv_path.exists()
    specs = build_specs(args)

    existing_rows = 0
    if args.resume and csv_path.exists():
        existing_rows = max(0, sum(1 for _ in csv_path.open()) - 1)
        specs = specs[existing_rows:]
        write_header = False

    progress = {
        "status": "running",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "total_specs": existing_rows + len(specs),
        "completed": existing_rows,
        "ok": 0,
        "timeout": 0,
        "failed": 0,
        "last_row": None,
        "out_csv": args.out_csv,
        "scope": "phase3_10_controlled_learned_rollout_no_phase4_no_cps",
    }
    write_json_atomic(progress_path, progress)

    start_all = time.time()
    script_path = Path(__file__).resolve()

    for i, spec in enumerate(specs, start=existing_rows):
        if args.total_timeout_sec > 0 and time.time() - start_all > args.total_timeout_sec:
            progress["status"] = "stopped_total_timeout"
            write_json_atomic(progress_path, progress)
            break

        spec_path = worker_dir / f"worker_{i:04d}.json"
        out_path = worker_dir / f"worker_{i:04d}_out.json"
        err_path = worker_dir / f"worker_{i:04d}.stderr.txt"
        write_json_atomic(spec_path, spec)

        started = time.time()
        cmd = [sys.executable, str(script_path), "--worker_json", str(spec_path), "--worker_out_json", str(out_path)]
        stderr_text = ""
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=float(args.row_timeout_sec),
                env=os.environ.copy(),
            )
            finished = time.time()
            stderr_text = (proc.stderr or "") + "\n" + (proc.stdout or "")
            err_path.write_text(stderr_text[-8000:])

            if out_path.exists():
                row = json.loads(out_path.read_text())
                row["status"] = row.get("status", "ok" if proc.returncode == 0 else "worker_returned_nonzero")
                row["worker_returncode"] = proc.returncode
                row["worker_timeout"] = 0
                row["worker_stderr_tail"] = stderr_text[-2000:]
                row["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(started))
                row["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(finished))
                row["duration_sec"] = f"{finished - started:.3f}"
            else:
                row = error_row(spec, started, finished, "worker_failed_no_output", stderr_text, proc.returncode)
        except subprocess.TimeoutExpired as exc:
            finished = time.time()
            stderr_text = ((exc.stderr or "") if isinstance(exc.stderr, str) else str(exc.stderr))[-8000:]
            err_path.write_text(stderr_text)
            row = error_row(spec, started, finished, "timeout", stderr_text, "timeout")

        write_row(csv_path, row, write_header)
        write_header = False

        status = str(row.get("status", ""))
        progress["completed"] += 1
        if status == "ok":
            progress["ok"] += 1
        elif "timeout" in status:
            progress["timeout"] += 1
        else:
            progress["failed"] += 1
        progress["last_row"] = {
            "status": row.get("status", ""),
            "policy": row.get("policy", ""),
            "condition": row.get("condition", ""),
            "final_fraction": row.get("final_fraction", ""),
            "success": row.get("success", ""),
            "failure_reason": row.get("failure_reason", ""),
        }
        write_json_atomic(progress_path, progress)

    if progress["status"] == "running":
        progress["status"] = "completed"
        progress["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        write_json_atomic(progress_path, progress)

    raw = {
        "status": progress["status"],
        "num_rows_written": progress["completed"],
        "ok": progress["ok"],
        "timeout": progress["timeout"],
        "failed": progress["failed"],
        "matrix": {
            "policies": args.policies,
            "conditions": args.conditions,
            "episodes_per_condition": args.episodes_per_condition,
            "max_steps": args.max_steps,
            "samples_per_step": args.samples_per_step,
        },
        "out_csv": args.out_csv,
        "progress_json": args.progress_json,
        "worker_dir": args.worker_dir,
        "scope": "phase3_10_controlled_learned_rollout_no_phase4_no_cps",
    }
    write_json_atomic(raw_path, raw)
    print(json.dumps(raw, indent=2, sort_keys=True))


def run_worker(args: argparse.Namespace) -> None:
    assert_gate()
    spec = json.loads(Path(args.worker_json).read_text())
    root = Path(spec["root"]).resolve()
    for p in [root, root / "scripts", root / "external/deformable-ravens"]:
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

    from ccda_phase3.data_io import load_action_codec_from_template
    from ccda_phase3.metrics import curve_metric_from_state
    from ccda_phase3.rollout import final_fraction_from_info, pad_history, state_from_live_info
    from ccda_phase3.train_utils import load_future_model, load_inverse_model
    import phase3_policy_rollout as p34

    p34.set_selected_recoverable_env_defaults()
    tasks, Environment = p34.require_runtime(root)
    assert_no_forbidden("after_runtime_import")

    windows_path = root / spec["windows"]
    data = np.load(windows_path, allow_pickle=True)
    template_path = Path(str(data["action_template_json_or_pickle_path"]))
    codec = load_action_codec_from_template(template_path)
    if codec.dim() != 14 or codec.summary().get("num_camera_config_paths") != 0:
        raise RuntimeError(f"bad action codec summary: {codec.summary()}")

    th = int(data["th"])
    action_dim = int(data["action_dim"])
    n_beads = int(data["n_beads"])
    if action_dim != 14:
        raise RuntimeError(f"action_dim must be 14, got {action_dim}")

    state_model = load_future_model(Path(spec["state_model_path"]))
    idm = load_inverse_model(Path(spec["idm_path"]))
    assert_no_forbidden("after_model_load")

    condition = str(spec["condition"])
    visible_seed = int(spec["visible_seed"])
    episode_idx = int(spec["episode_idx"])
    policy = str(spec["policy"])

    os.environ["CCDA_HIDDEN_CONDITION"] = condition
    os.environ["CCDA_VISIBLE_SEED"] = str(visible_seed)
    os.environ["CCDA_PAIR_GROUP"] = f"phase3_10_{policy}_{visible_seed}"

    task = tasks.names["hidden-contact-cable-line"]()
    task.mode = "train"

    env = None
    row: Dict[str, Any] = {
        "policy": policy,
        "condition": condition,
        "visible_seed": visible_seed,
        "episode_idx": episode_idx,
        "success": 0,
        "final_fraction": float("nan"),
        "final_curve": float("nan"),
        "num_steps": 0,
        "mean_action_ood_score": float("nan"),
        "max_action_ood_score": float("nan"),
        "mean_future_sample_std": float("nan"),
        "mean_pose0_xy_step_delta": float("nan"),
        "mean_pose1_xy_step_delta": float("nan"),
        "mean_pull_xy_len": float("nan"),
        "failure_reason": "",
        "checkpoint_state_model": spec["state_model_path"],
        "checkpoint_inverse_dynamics": spec["idm_path"],
        "primary_hidden_condition": "hidden_breakaway_pin",
        "diagnostic_hidden_condition": "hidden_pin",
        "scope": "phase3_10_controlled_learned_rollout_no_phase4_no_cps",
    }

    try:
        env = Environment(disp=False, hz=240)
        env.t_lim = float(spec["motion_timeout"])

        env.reset(task)
        reward_extras = task.reward()[1]
        info = env.info
        reward_extras["task.done"] = task.done()
        info["extras"] = reward_extras

        state_hist: List[np.ndarray] = []
        action_hist: List[np.ndarray] = []
        ood_scores: List[float] = []
        future_stds: List[float] = []
        pose0_step_deltas: List[float] = []
        pose1_step_deltas: List[float] = []
        pull_lens: List[float] = []

        prev_xy = None
        prev_pose0 = None
        prev_pose1 = None
        done = False

        for step in range(int(spec["max_steps"])):
            # Observable policy input only:
            # live state history + past action history.
            # Do NOT concatenate condition labels, hidden metadata, breakaway fields,
            # success/final_fraction, branch labels, or future labels.
            state = state_from_live_info(info, prev_xy=prev_xy)
            prev_xy = state[: n_beads * 2].reshape(n_beads, 2)
            state_hist.append(state)

            hist_states = pad_history(state_hist, th).reshape(-1)
            if action_hist:
                hist_actions = pad_history(action_hist, th).reshape(-1)
            else:
                hist_actions = np.zeros((th * action_dim,), dtype=np.float32)

            model_x = np.concatenate([hist_states, hist_actions], axis=0).reshape(1, -1)

            samples = state_model.sample(
                model_x,
                n_samples=int(spec["samples_per_step"]),
                seed=visible_seed + step,
            )[:, 0, :]
            mean_future = np.mean(samples, axis=0, keepdims=True)
            future_stds.append(float(np.mean(np.std(samples, axis=0))))

            idm_x = np.concatenate([hist_states.reshape(1, -1), mean_future], axis=1)
            raw_action_vec = idm.predict(idm_x)[0]

            ood_scores.append(float(idm.ood_score(raw_action_vec.reshape(1, -1))[0]))

            lo = idm.train_action_mean - float(spec["action_clip_std"]) * idm.train_action_std
            hi = idm.train_action_mean + float(spec["action_clip_std"]) * idm.train_action_std
            action_vec = np.clip(raw_action_vec, lo, hi).astype(np.float32)

            action = codec.decode(action_vec)

            try:
                p0 = np.asarray(action["params"]["pose0"][0], dtype=np.float32)[:2]
                p1 = np.asarray(action["params"]["pose1"][0], dtype=np.float32)[:2]
                if prev_pose0 is not None:
                    pose0_step_deltas.append(float(np.linalg.norm(p0 - prev_pose0)))
                if prev_pose1 is not None:
                    pose1_step_deltas.append(float(np.linalg.norm(p1 - prev_pose1)))
                pull_lens.append(float(np.linalg.norm(p1 - p0)))
                prev_pose0 = p0
                prev_pose1 = p1
            except Exception:
                pass

            obs, reward, done, info = env.step(action)
            action_hist.append(action_vec.astype(np.float32))
            if done:
                break

        success = bool(info.get("extras", {}).get("task.done", done))
        final_fraction = final_fraction_from_info(info)

        final_curve = float("nan")
        try:
            final_state = state_from_live_info(info)
            final_curve = curve_metric_from_state(final_state, n_beads)
        except Exception:
            pass

        row.update({
            "status": "ok",
            "success": int(success),
            "final_fraction": float(final_fraction),
            "final_curve": float(final_curve),
            "num_steps": len(action_hist),
            "mean_action_ood_score": float(np.mean(ood_scores)) if ood_scores else float("nan"),
            "max_action_ood_score": float(np.max(ood_scores)) if ood_scores else float("nan"),
            "mean_future_sample_std": float(np.mean(future_stds)) if future_stds else float("nan"),
            "mean_pose0_xy_step_delta": float(np.mean(pose0_step_deltas)) if pose0_step_deltas else float("nan"),
            "mean_pose1_xy_step_delta": float(np.mean(pose1_step_deltas)) if pose1_step_deltas else float("nan"),
            "mean_pull_xy_len": float(np.mean(pull_lens)) if pull_lens else float("nan"),
            "failure_reason": "",
        })

    except Exception as exc:
        row["status"] = "ok"
        row["failure_reason"] = repr(exc)
    finally:
        if env is not None:
            try:
                p34.close_env_safely(env)
            except Exception:
                try:
                    env.stop()
                except Exception:
                    pass

    Path(args.worker_out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.worker_out_json).write_text(json.dumps(row, indent=2, sort_keys=True))
    print(json.dumps({
        "status": row.get("status"),
        "policy": policy,
        "condition": condition,
        "final_fraction": row.get("final_fraction"),
        "success": row.get("success"),
        "failure": row.get("failure_reason"),
    }, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--windows", default="data/phase3_state_diff_windows/phase3_windows.npz")
    parser.add_argument("--action_template", default="data/phase3_state_diff_windows/phase3_action_template.pkl")
    parser.add_argument("--old_checkpoint_root", default="checkpoints/phase3")
    parser.add_argument("--phase39_train_summary", default="reports/phase3_9_geometry_idm_train_summary.json")
    parser.add_argument("--phase39b_raw", default="reports/phase3_9b_ablation_raw_summary.json")
    parser.add_argument("--best_ablation", default="xy_only_high_weight")
    parser.add_argument("--policies", nargs="+", default=POLICIES)
    parser.add_argument("--conditions", nargs="+", default=ROLLOUT_CONDITIONS)
    parser.add_argument("--episodes_per_condition", type=int, default=4)
    parser.add_argument("--seed_start", type=int, default=310000)
    parser.add_argument("--max_steps", type=int, default=16)
    parser.add_argument("--samples_per_step", type=int, default=16)
    parser.add_argument("--motion_timeout", type=float, default=15.0)
    parser.add_argument("--action_clip_std", type=float, default=3.0)
    parser.add_argument("--row_timeout_sec", type=float, default=900.0)
    parser.add_argument("--total_timeout_sec", type=float, default=14400.0)
    parser.add_argument("--max_rows", type=int, default=36)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--out_csv", default="reports/phase3_10_controlled_learned_rollout_trials.csv")
    parser.add_argument("--out_json", default="reports/phase3_10_controlled_learned_rollout_raw_summary.json")
    parser.add_argument("--progress_json", default="reports/phase3_10_controlled_learned_rollout_progress.json")
    parser.add_argument("--worker_dir", default="reports/phase3_10_workers")
    parser.add_argument("--worker_json", default="")
    parser.add_argument("--worker_out_json", default="")
    args = parser.parse_args()

    if args.worker_json:
        run_worker(args)
    else:
        run_parent(args)


if __name__ == "__main__":
    main()