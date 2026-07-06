#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ABLATIONS: Dict[str, Dict[str, float]] = {
    "default_geometry": {
        "action_z_weight": 1.0,
        "pose0_xy_weight": 8.0,
        "pose1_xy_weight": 8.0,
        "coupled_xy_weight": 8.0,
        "pull_xy_weight": 4.0,
        "z_weight": 0.25,
        "quat_weight": 0.02,
    },
    "no_pull_loss": {
        "action_z_weight": 1.0,
        "pose0_xy_weight": 8.0,
        "pose1_xy_weight": 8.0,
        "coupled_xy_weight": 8.0,
        "pull_xy_weight": 0.0,
        "z_weight": 0.25,
        "quat_weight": 0.02,
    },
    "no_coupled_xy_loss": {
        "action_z_weight": 1.0,
        "pose0_xy_weight": 8.0,
        "pose1_xy_weight": 8.0,
        "coupled_xy_weight": 0.0,
        "pull_xy_weight": 4.0,
        "z_weight": 0.25,
        "quat_weight": 0.02,
    },
    "xy_only_high_weight": {
        "action_z_weight": 0.5,
        "pose0_xy_weight": 12.0,
        "pose1_xy_weight": 12.0,
        "coupled_xy_weight": 12.0,
        "pull_xy_weight": 6.0,
        "z_weight": 0.05,
        "quat_weight": 0.0,
    },
    "no_quat_loss": {
        "action_z_weight": 1.0,
        "pose0_xy_weight": 8.0,
        "pose1_xy_weight": 8.0,
        "coupled_xy_weight": 8.0,
        "pull_xy_weight": 4.0,
        "z_weight": 0.25,
        "quat_weight": 0.0,
    },
}


def require_gate() -> None:
    gates = [
        "PHASE3_ALLOW_GEOMETRY_IDM_ABLATION",
        "PHASE3_GEOMETRY_IDM_ABLATION_CONFIRMED",
        "PHASE3_ALLOW_GEOMETRY_IDM_TRAIN",
        "PHASE3_GEOMETRY_IDM_TRAIN_CONFIRMED",
        "PHASE3_ALLOW_GEOMETRY_IDM_RETRY",
        "PHASE3_GEOMETRY_IDM_RETRY_CONFIRMED",
    ]
    missing = [g for g in gates if os.environ.get(g, "0") != "1"]
    if missing:
        raise SystemExit(f"[Phase3.9b][BLOCKED] missing gates: {missing}")


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(path)


def run_cmd(cmd: List[str], root: Path, env: Dict[str, str], log_path: Path, timeout_sec: int | None = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=str(root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        start = time.time()
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
            log.flush()
            if timeout_sec and time.time() - start > timeout_sec:
                proc.kill()
                log.write(f"\n[Phase3.9b][TIMEOUT] killed after {timeout_sec}s\n")
                return 124
        return int(proc.wait())


def parse_list(text: str) -> List[str]:
    return [x for x in text.replace(",", " ").split() if x]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--ablations", default="default_geometry no_pull_loss no_coupled_xy_loss xy_only_high_weight no_quat_loss")
    parser.add_argument("--baseline", default="state_action")
    parser.add_argument("--windows", default="data/phase3_state_diff_windows/phase3_windows.npz")
    parser.add_argument("--action_template", default="data/phase3_state_diff_windows/phase3_action_template.pkl")
    parser.add_argument("--old_checkpoint_root", default="checkpoints/phase3")
    parser.add_argument("--phase37_raw", default="reports/phase3_7_learned_action_alignment_raw_summary.json")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--seed_base", type=int, default=392000)
    parser.add_argument("--samples_per_condition", type=int, default=4)
    parser.add_argument("--conditions", default="free hidden_breakaway_pin hidden_high_friction")
    parser.add_argument("--actions", default="gt_reference old_idm_gt_future phase39_idm_gt_future")
    parser.add_argument("--pred_samples", type=int, default=16)
    parser.add_argument("--motion_timeout", type=float, default=15.0)
    parser.add_argument("--max_prefix_actions", type=int, default=20)
    parser.add_argument("--row_timeout_sec", type=float, default=240.0)
    parser.add_argument("--total_timeout_sec", type=float, default=7200.0)
    parser.add_argument("--max_rows", type=int, default=36)
    parser.add_argument("--improve_threshold", type=float, default=0.05)
    parser.add_argument("--min_ok_rows_per_condition_action", type=int, default=2)
    parser.add_argument("--out_dir", default="reports/phase3_9b_ablation")
    parser.add_argument("--checkpoint_root", default="checkpoints/phase3_9b_geometry_idm")
    parser.add_argument("--progress_json", default="reports/phase3_9b_ablation_progress.json")
    parser.add_argument("--summary_json", default="reports/phase3_9b_ablation_raw_summary.json")
    args = parser.parse_args()

    require_gate()
    root = Path(args.root).resolve()
    ablations = parse_list(args.ablations)
    bad = [a for a in ablations if a not in ABLATIONS]
    if bad:
        raise SystemExit(f"[Phase3.9b][FAIL] unknown ablations: {bad}; available={sorted(ABLATIONS)}")

    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_root = root / args.checkpoint_root
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{root}:{root / 'scripts'}:{root / 'external/deformable-ravens'}:{env.get('PYTHONPATH', '')}"

    progress: Dict[str, Any] = {
        "status": "running",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "ablations": ablations,
        "completed": [],
        "failed": [],
        "results": {},
        "scope": "phase3_9b_geometry_idm_ablation_no_phase4_no_cps",
    }
    write_json_atomic(root / args.progress_json, progress)

    for i, ablation in enumerate(ablations):
        seed = args.seed_base + i
        weights = ABLATIONS[ablation]
        ab_dir = out_dir / ablation
        ab_dir.mkdir(parents=True, exist_ok=True)

        ckpt_rel = f"{args.checkpoint_root}/{args.baseline}/{ablation}/fold_phase3_9b_seed_{seed}"
        ckpt_abs = root / ckpt_rel
        train_summary = f"{args.out_dir}/{ablation}/train_summary.json"
        train_report = f"{args.out_dir}/{ablation}/train_report.md"
        retry_csv = f"{args.out_dir}/{ablation}/controlled_retry_trials.csv"
        retry_raw = f"{args.out_dir}/{ablation}/controlled_retry_raw_summary.json"
        retry_progress = f"{args.out_dir}/{ablation}/controlled_retry_progress.json"
        retry_workers = f"{args.out_dir}/{ablation}/workers"
        retry_summary = f"{args.out_dir}/{ablation}/controlled_retry_summary.json"
        retry_report = f"{args.out_dir}/{ablation}/controlled_retry_report.md"

        train_cmd = [
            sys.executable, "scripts/phase3_9_train_geometry_idm.py",
            "--root", str(root),
            "--windows", args.windows,
            "--action_template", args.action_template,
            "--old_checkpoint_root", args.old_checkpoint_root,
            "--baseline", args.baseline,
            "--out_dir", ckpt_rel,
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--idm_hidden_dim", str(args.hidden_dim),
            "--idm_lr", str(args.lr),
            "--idm_weight_decay", str(args.weight_decay),
            "--action_z_weight", str(weights["action_z_weight"]),
            "--pose0_xy_weight", str(weights["pose0_xy_weight"]),
            "--pose1_xy_weight", str(weights["pose1_xy_weight"]),
            "--coupled_xy_weight", str(weights["coupled_xy_weight"]),
            "--pull_xy_weight", str(weights["pull_xy_weight"]),
            "--z_weight", str(weights["z_weight"]),
            "--quat_weight", str(weights["quat_weight"]),
            "--seed", str(seed),
            "--summary_json", train_summary,
            "--summary_md", train_report,
        ]

        train_rc = run_cmd(train_cmd, root, env, ab_dir / "train.log", timeout_sec=None)

        new_idm = ckpt_abs / "inverse_dynamics.pt"
        retry_rc = 999
        analyze_rc = 999

        if train_rc == 0 and new_idm.exists():
            retry_cmd = [
                sys.executable, "scripts/phase3_9_controlled_retry.py",
                "--root", str(root),
                "--windows", args.windows,
                "--action_template", args.action_template,
                "--old_checkpoint_root", args.old_checkpoint_root,
                "--baseline", args.baseline,
                "--new_inverse_path", str(new_idm),
                "--phase37_raw", args.phase37_raw,
                "--conditions", *parse_list(args.conditions),
                "--actions", *parse_list(args.actions),
                "--samples_per_condition", str(args.samples_per_condition),
                "--pred_samples", str(args.pred_samples),
                "--seed_base", str(args.seed_base + 1000 + i),
                "--motion_timeout", str(args.motion_timeout),
                "--max_prefix_actions", str(args.max_prefix_actions),
                "--row_timeout_sec", str(args.row_timeout_sec),
                "--total_timeout_sec", str(args.total_timeout_sec),
                "--max_rows", str(args.max_rows),
                "--out_csv", retry_csv,
                "--out_json", retry_raw,
                "--progress_json", retry_progress,
                "--worker_dir", retry_workers,
            ]
            retry_rc = run_cmd(retry_cmd, root, env, ab_dir / "controlled_retry.log", timeout_sec=int(args.total_timeout_sec) + 600)

            analyze_cmd = [
                sys.executable, "scripts/phase3_9_analyze_controlled_retry.py",
                "--root", str(root),
                "--trials_csv", retry_csv,
                "--progress_json", retry_progress,
                "--raw_summary", retry_raw,
                "--train_summary", train_summary,
                "--out_json", retry_summary,
                "--out_md", retry_report,
                "--improve_threshold", str(args.improve_threshold),
                "--min_ok_rows_per_condition_action", str(args.min_ok_rows_per_condition_action),
            ]
            analyze_rc = run_cmd(analyze_cmd, root, env, ab_dir / "analyze.log", timeout_sec=600)
        else:
            (ab_dir / "failed_before_retry.txt").write_text(
                f"train_rc={train_rc}, checkpoint_exists={new_idm.exists()}, checkpoint={new_idm}\n"
            )

        result = {
            "ablation": ablation,
            "weights": weights,
            "seed": seed,
            "checkpoint": str(new_idm),
            "train_rc": train_rc,
            "retry_rc": retry_rc,
            "analyze_rc": analyze_rc,
            "train_summary": train_summary,
            "retry_summary": retry_summary,
            "retry_csv": retry_csv,
        }
        progress["results"][ablation] = result

        if train_rc == 0 and retry_rc == 0 and analyze_rc in {0, 1}:
            progress["completed"].append(ablation)
        else:
            progress["failed"].append(ablation)

        write_json_atomic(root / args.progress_json, progress)

    progress["status"] = "completed" if not progress["failed"] else "completed_with_failures"
    progress["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    write_json_atomic(root / args.progress_json, progress)

    raw_summary = {
        "status": progress["status"],
        "ablations": ablations,
        "completed": progress["completed"],
        "failed": progress["failed"],
        "results": progress["results"],
        "scope": "phase3_9b_geometry_idm_ablation_no_phase4_no_cps",
        "important_note": "Local checkpoints are diagnostic artifacts and must not be committed.",
    }
    write_json_atomic(root / args.summary_json, raw_summary)
    print(json.dumps(raw_summary, indent=2, sort_keys=True))

    if progress["failed"]:
        raise SystemExit("[Phase3.9b][WARN] some ablations failed; aggregate analyzer should still run")


if __name__ == "__main__":
    main()