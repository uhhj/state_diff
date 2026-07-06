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

CONDITIONS = ["free", "hidden_breakaway_pin", "hidden_high_friction"]
ACTIONS = ["gt_reference", "old_idm_gt_future", "phase39_idm_gt_future"]
FORBIDDEN_PREFIXES = ["tensorflow", "ravens.agents", "ravens.models", "ravens.datasets"]

FIELDNAMES = [
    "status",
    "started_at",
    "finished_at",
    "duration_sec",
    "worker_returncode",
    "worker_timeout",
    "worker_stderr_tail",
    "window_idx",
    "condition",
    "visible_seed",
    "window_t",
    "source_file",
    "baseline",
    "action_source",
    "future_key",
    "future_mae",
    "primitive",
    "has_params",
    "pose0_x",
    "pose0_y",
    "pose0_z",
    "pose1_x",
    "pose1_y",
    "pose1_z",
    "pose0_quat_norm",
    "pose1_quat_norm",
    "pull_xy_len",
    "action_valid",
    "action_l2_to_gt",
    "action_mae_to_gt",
    "action_max_abs_to_gt",
    "action_cosine_to_gt",
    "action_norm",
    "gt_action_norm",
    "pose0_xy_dist_to_gt",
    "pose0_z_abs_diff_to_gt",
    "pose1_xy_dist_to_gt",
    "pose1_z_abs_diff_to_gt",
    "pull_angle_deg_to_gt",
    "pull_len_ratio_to_gt",
    "exec_success",
    "exec_done",
    "exec_reward",
    "exec_final_fraction_before",
    "exec_final_fraction_after",
    "exec_delta_final_fraction",
    "exec_prefix_state_mae",
    "exec_prefix_state_l2",
    "exec_prefix_source",
    "exec_failure_reason",
    "scope",
    "action_repr",
]


def assert_gate() -> None:
    if os.environ.get("PHASE3_ALLOW_GEOMETRY_IDM_RETRY", "0") != "1":
        raise SystemExit("[Phase3.9][BLOCKED] Set PHASE3_ALLOW_GEOMETRY_IDM_RETRY=1")
    if os.environ.get("PHASE3_GEOMETRY_IDM_RETRY_CONFIRMED", "0") != "1":
        raise SystemExit("[Phase3.9][BLOCKED] Set PHASE3_GEOMETRY_IDM_RETRY_CONFIRMED=1")


def assert_no_forbidden(stage: str) -> None:
    bad: List[str] = []
    for name in sys.modules:
        for prefix in FORBIDDEN_PREFIXES:
            if name == prefix or name.startswith(prefix + "."):
                bad.append(name)
    if bad:
        raise RuntimeError(f"[Phase3.9][FAIL] forbidden modules loaded at {stage}: {bad[:20]}")


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(path)


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def make_error_row(spec: Dict[str, Any], started: float, finished: float, status: str, stderr: str, returncode: Any = "") -> Dict[str, Any]:
    row = {k: "" for k in FIELDNAMES}
    item = spec.get("item", {})
    row.update({
        "status": status,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(started)),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(finished)),
        "duration_sec": f"{finished - started:.3f}",
        "worker_returncode": returncode,
        "worker_timeout": 1 if "timeout" in status else 0,
        "worker_stderr_tail": stderr[-2000:],
        "window_idx": item.get("idx", ""),
        "condition": item.get("condition", ""),
        "visible_seed": item.get("visible_seed", ""),
        "window_t": item.get("window_t", ""),
        "source_file": item.get("source_file", ""),
        "baseline": spec.get("baseline", ""),
        "action_source": spec.get("action_source", ""),
        "scope": "phase3_9_geometry_idm_controlled_retry_no_phase4_no_cps",
        "exec_failure_reason": status,
    })
    return row


def write_row(csv_path: Path, row: Dict[str, Any], write_header: bool) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in FIELDNAMES})
        f.flush()
        os.fsync(f.fileno())


def select_items(root: Path, data: Any, p38: Any, phase37_raw: Dict[str, Any], conditions: List[str], samples_per_condition: int) -> List[Dict[str, Any]]:
    selected = p38.select_indices(data, phase37_raw, max(samples_per_condition, 1))
    grouped: Dict[str, List[Dict[str, Any]]] = {c: [] for c in conditions}
    for item in selected:
        cond = str(item.get("condition", ""))
        if cond in grouped:
            grouped[cond].append({
                "idx": int(item["idx"]),
                "condition": cond,
                "visible_seed": str(item["visible_seed"]),
                "window_t": int(item["window_t"]),
                "source_file": str(item["source_file"]),
            })

    out: List[Dict[str, Any]] = []
    if all(grouped[c] for c in conditions):
        for cond in conditions:
            out.extend(grouped[cond][:samples_per_condition])
        return out

    conds = np.asarray([str(x) for x in data["condition_name"]])
    seeds = np.asarray([str(x) for x in data["visible_seed"]])
    ts = np.asarray(data["window_t"]).astype(int)
    src = np.asarray([str(x) for x in data["source_file"]])
    for cond in conditions:
        idxs = np.where(conds == cond)[0]
        for idx in idxs[:samples_per_condition]:
            out.append({
                "idx": int(idx),
                "condition": cond,
                "visible_seed": str(seeds[idx]),
                "window_t": int(ts[idx]),
                "source_file": str(src[idx]),
            })
    return out


def build_specs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    root = Path(args.root).resolve()
    for p in [root, root / "scripts"]:
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

    import phase3_8_idm_geometry_repair_probe as p38

    data, meta = p38.load_windows(root, args.windows)
    phase37_raw = load_json(root / args.phase37_raw)
    items = select_items(root, data, p38, phase37_raw, args.conditions, args.samples_per_condition)

    specs: List[Dict[str, Any]] = []
    for item in items:
        for action_source in args.actions:
            specs.append({
                "root": str(root),
                "windows": args.windows,
                "action_template": args.action_template,
                "old_checkpoint_root": args.old_checkpoint_root,
                "baseline": args.baseline,
                "new_inverse_path": args.new_inverse_path,
                "action_source": action_source,
                "item": item,
                "pred_samples": args.pred_samples,
                "seed_base": args.seed_base,
                "motion_timeout": args.motion_timeout,
                "max_prefix_actions": args.max_prefix_actions,
            })
    if args.max_rows > 0:
        specs = specs[: args.max_rows]
    return specs


def run_parent(args: argparse.Namespace) -> None:
    assert_gate()
    root = Path(args.root).resolve()
    csv_path = root / args.out_csv
    progress_path = root / args.progress_json
    raw_summary_path = root / args.out_json
    worker_dir = root / args.worker_dir

    if worker_dir.exists():
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

    progress = {
        "status": "running",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "total_specs": existing_rows + len(specs),
        "completed": existing_rows,
        "ok": 0,
        "timeout": 0,
        "failed": 0,
        "out_csv": args.out_csv,
        "scope": "phase3_9_geometry_idm_controlled_retry_no_phase4_no_cps",
        "last_row": None,
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
                row = make_error_row(spec, started, finished, "worker_failed_no_output", stderr_text, proc.returncode)
        except subprocess.TimeoutExpired as exc:
            finished = time.time()
            stderr_text = ((exc.stderr or "") if isinstance(exc.stderr, str) else str(exc.stderr))[-8000:]
            err_path.write_text(stderr_text)
            row = make_error_row(spec, started, finished, "timeout", stderr_text, "timeout")

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
        progress["last_row"] = {k: row.get(k, "") for k in ["status", "condition", "action_source", "exec_delta_final_fraction", "exec_failure_reason"]}
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
        "out_csv": args.out_csv,
        "progress_json": args.progress_json,
        "worker_dir": args.worker_dir,
        "matrix": {
            "baseline": args.baseline,
            "actions": args.actions,
            "conditions": args.conditions,
            "samples_per_condition": args.samples_per_condition,
            "max_rows": args.max_rows,
        },
        "scope": "phase3_9_geometry_idm_controlled_retry_no_phase4_no_cps",
    }
    write_json_atomic(raw_summary_path, raw)
    print(json.dumps(raw, indent=2, sort_keys=True))


def run_worker(args: argparse.Namespace) -> None:
    assert_gate()
    spec = json.loads(Path(args.worker_json).read_text())
    root = Path(spec["root"]).resolve()
    for p in [root, root / "scripts"]:
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

    import phase3_8_idm_geometry_repair_probe as p38
    from ccda_phase3.train_utils import load_inverse_model

    data, meta = p38.load_windows(root, spec["windows"])
    codec = p38.load_codec(root, data, spec["action_template"])

    old_ckpt, state_model, old_idm = p38.load_models(root, spec["old_checkpoint_root"], spec["baseline"])
    new_idm = load_inverse_model(Path(spec["new_inverse_path"]))

    item = spec["item"]
    idx = int(item["idx"])

    gt_vec, old_vec, _old_pred_vec, future_key, future_mae = p38.build_idm_vectors(
        data=data,
        idx=idx,
        baseline=spec["baseline"],
        state_model=state_model,
        idm=old_idm,
        pred_samples=int(spec["pred_samples"]),
        seed=int(spec["seed_base"]) + idx,
        clip_std=3.0,
    )

    gt_vec2, new_vec, _new_pred_vec, future_key2, future_mae2 = p38.build_idm_vectors(
        data=data,
        idx=idx,
        baseline=spec["baseline"],
        state_model=state_model,
        idm=new_idm,
        pred_samples=int(spec["pred_samples"]),
        seed=int(spec["seed_base"]) + idx,
        clip_std=3.0,
    )

    if future_key2 != future_key:
        raise RuntimeError(f"future_key mismatch: old={future_key}, new={future_key2}")

    source = spec["action_source"]
    if source == "gt_reference":
        vec = gt_vec
    elif source == "old_idm_gt_future":
        vec = old_vec
    elif source == "phase39_idm_gt_future":
        vec = new_vec
    else:
        raise RuntimeError(f"unknown action_source={source}")

    action = p38.decode(codec, vec)
    ai = p38.action_info(action)
    gt_action = p38.decode(codec, gt_vec)
    gt_info = p38.action_info(gt_action)

    p36 = p38.import_phase36(root)
    tasks, Environment, close_helper = p36.import_tf_free_ravens(root)
    assert_no_forbidden("after_runtime_import")

    exec_info = p38.execute_after_matched_prefix(
        root=root,
        p36=p36,
        tasks=tasks,
        Environment=Environment,
        close_helper=close_helper,
        data=data,
        codec=codec,
        item=item,
        action=action,
        timeout=float(spec["motion_timeout"]),
        max_prefix_actions=int(spec["max_prefix_actions"]),
    )

    row: Dict[str, Any] = {
        "status": "ok",
        "window_idx": idx,
        "condition": item["condition"],
        "visible_seed": item["visible_seed"],
        "window_t": int(item["window_t"]),
        "source_file": item["source_file"],
        "baseline": spec["baseline"],
        "action_source": source,
        "future_key": future_key,
        "future_mae": future_mae,
        "scope": "phase3_9_geometry_idm_controlled_retry_no_phase4_no_cps",
    }
    row.update(ai)
    row.update(p38.vector_compare(vec, gt_vec))
    row.update(p38.geometry_compare(ai, gt_info))
    row.update(exec_info)

    Path(args.worker_out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.worker_out_json).write_text(json.dumps(row, indent=2, sort_keys=True))
    print(json.dumps({"status": "ok", "condition": item["condition"], "source": source, "delta": row.get("exec_delta_final_fraction")}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--windows", default="data/phase3_state_diff_windows/phase3_windows.npz")
    parser.add_argument("--action_template", default="data/phase3_state_diff_windows/phase3_action_template.pkl")
    parser.add_argument("--old_checkpoint_root", default="checkpoints/phase3")
    parser.add_argument("--baseline", default="state_action")
    parser.add_argument("--new_inverse_path", required=False, default="")
    parser.add_argument("--phase37_raw", default="reports/phase3_7_learned_action_alignment_raw_summary.json")
    parser.add_argument("--conditions", nargs="+", default=CONDITIONS)
    parser.add_argument("--actions", nargs="+", default=ACTIONS)
    parser.add_argument("--samples_per_condition", type=int, default=2)
    parser.add_argument("--pred_samples", type=int, default=16)
    parser.add_argument("--seed_base", type=int, default=391000)
    parser.add_argument("--motion_timeout", type=float, default=15.0)
    parser.add_argument("--max_prefix_actions", type=int, default=20)
    parser.add_argument("--row_timeout_sec", type=float, default=240.0)
    parser.add_argument("--total_timeout_sec", type=float, default=3600.0)
    parser.add_argument("--max_rows", type=int, default=18)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--out_csv", default="reports/phase3_9_geometry_idm_controlled_retry_trials.csv")
    parser.add_argument("--out_json", default="reports/phase3_9_geometry_idm_controlled_retry_raw_summary.json")
    parser.add_argument("--progress_json", default="reports/phase3_9_geometry_idm_controlled_retry_progress.json")
    parser.add_argument("--worker_dir", default="reports/phase3_9_workers")
    parser.add_argument("--worker_json", default="")
    parser.add_argument("--worker_out_json", default="")
    args = parser.parse_args()

    if args.worker_json:
        run_worker(args)
    else:
        if not args.new_inverse_path:
            raise SystemExit("--new_inverse_path is required in parent mode")
        run_parent(args)


if __name__ == "__main__":
    main()
