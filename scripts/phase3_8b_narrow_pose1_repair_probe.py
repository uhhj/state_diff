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
from typing import Any, Dict, List, Optional

import numpy as np

REQUIRED_CONDITIONS = ["free", "hidden_breakaway_pin", "hidden_high_friction"]
DEFAULT_VARIANTS = [
    "gt_reference",
    "idm_original",
    "idm_gt_pose1_xy",
    "idm_gt_pose0_xy",
    "idm_gt_pose0_pose1_xy",
    "idm_gt_pull_dir_gt_len",
]
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
    "source_action",
    "repair_variant",
    "future_key",
    "future_mae",
    "primary_hidden_condition",
    "diagnostic_hidden_condition",
    "primary_pair",
    "diagnostic_pair",
    "scope",
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
    "is_gt_blended_diagnostic",
    "is_deployable_policy_action",
    "action_repr",
]


def assert_gate() -> None:
    if os.environ.get("PHASE3_ALLOW_IDM_GEOMETRY_REPAIR", "0") != "1":
        raise SystemExit("[Phase3.8b][BLOCKED] Set PHASE3_ALLOW_IDM_GEOMETRY_REPAIR=1")
    if os.environ.get("PHASE3_IDM_GEOMETRY_REPAIR_CONFIRMED", "0") != "1":
        raise SystemExit("[Phase3.8b][BLOCKED] Set PHASE3_IDM_GEOMETRY_REPAIR_CONFIRMED=1")


def assert_no_forbidden(stage: str) -> None:
    bad: List[str] = []
    for name in sys.modules:
        for prefix in FORBIDDEN_PREFIXES:
            if name == prefix or name.startswith(prefix + "."):
                bad.append(name)
    if bad:
        raise RuntimeError(f"[Phase3.8b][FAIL] forbidden modules loaded at {stage}: {bad[:30]}")


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(path)


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def norm_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(v) for v in x]
    return [str(x)]


def make_timeout_row(spec: Dict[str, Any], started: float, finished: float, stderr_tail: str, returncode: Any = "") -> Dict[str, Any]:
    row = {k: "" for k in FIELDNAMES}
    row.update({
        "status": "timeout",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(started)),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(finished)),
        "duration_sec": f"{finished - started:.3f}",
        "worker_returncode": returncode,
        "worker_timeout": 1,
        "worker_stderr_tail": stderr_tail[-2000:],
        "window_idx": spec.get("item", {}).get("idx", ""),
        "condition": spec.get("item", {}).get("condition", ""),
        "visible_seed": spec.get("item", {}).get("visible_seed", ""),
        "window_t": spec.get("item", {}).get("window_t", ""),
        "source_file": spec.get("item", {}).get("source_file", ""),
        "baseline": spec.get("baseline", ""),
        "source_action": spec.get("source_action", ""),
        "repair_variant": spec.get("repair_variant", ""),
        "scope": "phase3_8b_narrow_pose1_repair_no_phase4_no_cps",
        "exec_failure_reason": "worker_timeout_or_failed_before_output",
    })
    return row


def select_items(root: Path, data: Any, p38: Any, phase37_raw: Dict[str, Any], conditions: List[str], samples_per_condition: int) -> List[Dict[str, Any]]:
    selected = p38.select_indices(data, phase37_raw, max(samples_per_condition, 1))
    by_cond: Dict[str, List[Dict[str, Any]]] = {c: [] for c in conditions}
    for item in selected:
        cond = str(item.get("condition", ""))
        if cond in by_cond:
            by_cond[cond].append({
                "idx": int(item["idx"]),
                "condition": cond,
                "visible_seed": str(item["visible_seed"]),
                "window_t": int(item["window_t"]),
                "source_file": str(item["source_file"]),
            })

    if all(by_cond[c] for c in conditions):
        out: List[Dict[str, Any]] = []
        for cond in conditions:
            out.extend(by_cond[cond][:samples_per_condition])
        return out

    conds = np.asarray([str(x) for x in data["condition_name"]])
    seeds = np.asarray([str(x) for x in data["visible_seed"]])
    ts = np.asarray(data["window_t"]).astype(int)
    src = np.asarray([str(x) for x in data["source_file"]])
    out = []
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
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    scripts_dir = root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    import phase3_8_idm_geometry_repair_probe as p38

    data, meta = p38.load_windows(root, args.windows)
    if meta.get("primary_hidden_condition") != "hidden_breakaway_pin":
        raise RuntimeError(f"bad primary hidden condition: {meta.get('primary_hidden_condition')}")
    phase37_raw = load_json(root / args.phase37_raw)

    items = select_items(root, data, p38, phase37_raw, args.conditions, args.samples_per_condition)
    specs: List[Dict[str, Any]] = []
    for baseline in args.baselines:
        for source_action in args.sources:
            for item in items:
                for variant in args.variants:
                    specs.append({
                        "root": str(root),
                        "windows": args.windows,
                        "action_template": args.action_template,
                        "checkpoint_root": args.checkpoint_root,
                        "baseline": baseline,
                        "source_action": source_action,
                        "repair_variant": variant,
                        "item": item,
                        "pred_samples": args.pred_samples,
                        "seed_base": args.seed_base,
                        "action_clip_std": args.action_clip_std,
                        "motion_timeout": args.motion_timeout,
                        "max_prefix_actions": args.max_prefix_actions,
                    })
    if args.max_rows > 0:
        specs = specs[: args.max_rows]
    return specs


def write_row(csv_path: Path, row: Dict[str, Any], write_header: bool) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in FIELDNAMES})
        f.flush()
        os.fsync(f.fileno())


def parse_effective_so_far(csv_path: Path, min_delta: float) -> bool:
    if not csv_path.exists():
        return False
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    orig = []
    repairs = []
    for row in rows:
        if row.get("source_action") != "idm_gt_future":
            continue
        try:
            delta = float(row.get("exec_delta_final_fraction", "nan"))
        except Exception:
            continue
        if row.get("repair_variant") == "idm_original":
            orig.append(delta)
        elif row.get("repair_variant") not in {"gt_reference", "idm_original"}:
            repairs.append(delta)
    if not repairs:
        return False
    orig_mean = float(np.nanmean(orig)) if orig else 0.0
    return any(np.isfinite(x) and x > max(orig_mean + min_delta, min_delta) for x in repairs)


def run_parent(args: argparse.Namespace) -> None:
    assert_gate()
    root = Path(args.root).resolve()
    csv_path = root / args.out_csv
    progress_path = root / args.progress_json
    raw_summary_path = root / args.out_json
    worker_dir = root / args.worker_dir
    worker_dir.mkdir(parents=True, exist_ok=True)

    if args.resume and csv_path.exists():
        existing_rows = sum(1 for _ in csv_path.open()) - 1
        write_header = False
    else:
        if csv_path.exists():
            csv_path.unlink()
        existing_rows = 0
        write_header = True

    specs = build_specs(args)
    if existing_rows > 0:
        specs = specs[existing_rows:]

    progress: Dict[str, Any] = {
        "status": "running",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "total_specs": existing_rows + len(specs),
        "already_completed_on_resume": existing_rows,
        "completed": existing_rows,
        "ok": 0,
        "timeout": 0,
        "failed": 0,
        "last_row": None,
        "out_csv": args.out_csv,
        "scope": "phase3_8b_narrow_pose1_repair_no_phase4_no_cps",
    }
    write_json_atomic(progress_path, progress)

    script_path = Path(__file__).resolve()
    start_all = time.time()

    for global_i, spec in enumerate(specs, start=existing_rows):
        if args.total_timeout_sec > 0 and time.time() - start_all > args.total_timeout_sec:
            progress["status"] = "stopped_total_timeout"
            progress["stopped_at_global_i"] = global_i
            write_json_atomic(progress_path, progress)
            break

        worker_json = worker_dir / f"worker_{global_i:04d}.json"
        worker_out = worker_dir / f"worker_{global_i:04d}_out.json"
        worker_err = worker_dir / f"worker_{global_i:04d}.stderr.txt"
        write_json_atomic(worker_json, spec)

        started = time.time()
        cmd = [
            sys.executable,
            str(script_path),
            "--worker_json",
            str(worker_json),
            "--worker_out_json",
            str(worker_out),
        ]

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
            stderr_text = (proc.stderr or "") + "\n" + (proc.stdout or "")
            worker_err.write_text(stderr_text[-8000:])
            finished = time.time()

            if worker_out.exists():
                row = json.loads(worker_out.read_text())
                row["status"] = row.get("status", "ok" if proc.returncode == 0 else "worker_returned_nonzero")
                row["worker_returncode"] = proc.returncode
                row["worker_timeout"] = 0
                row["worker_stderr_tail"] = stderr_text[-2000:]
                row["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(started))
                row["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(finished))
                row["duration_sec"] = f"{finished - started:.3f}"
            else:
                row = make_timeout_row(spec, started, finished, stderr_text, returncode=proc.returncode)
                row["status"] = "worker_failed_no_output"
        except subprocess.TimeoutExpired as exc:
            finished = time.time()
            stderr_text = ((exc.stderr or "") if isinstance(exc.stderr, str) else str(exc.stderr))[-8000:]
            worker_err.write_text(stderr_text)
            row = make_timeout_row(spec, started, finished, stderr_text, returncode="timeout")

        write_row(csv_path, row, write_header)
        write_header = False

        status = str(row.get("status", ""))
        progress["completed"] = int(progress.get("completed", 0)) + 1
        if status == "ok":
            progress["ok"] = int(progress.get("ok", 0)) + 1
        elif "timeout" in status:
            progress["timeout"] = int(progress.get("timeout", 0)) + 1
        else:
            progress["failed"] = int(progress.get("failed", 0)) + 1
        progress["last_row"] = {k: row.get(k, "") for k in ["status", "baseline", "source_action", "condition", "repair_variant", "exec_delta_final_fraction", "exec_failure_reason"]}
        write_json_atomic(progress_path, progress)

        if args.stop_after_first_effective and parse_effective_so_far(csv_path, args.effective_delta_threshold):
            progress["status"] = "stopped_after_first_effective"
            progress["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            write_json_atomic(progress_path, progress)
            break

    if progress.get("status") == "running":
        progress["status"] = "completed"
        progress["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        write_json_atomic(progress_path, progress)

    raw_summary = {
        "status": progress["status"],
        "num_rows_written": progress["completed"],
        "ok": progress["ok"],
        "timeout": progress["timeout"],
        "failed": progress["failed"],
        "out_csv": args.out_csv,
        "progress_json": args.progress_json,
        "worker_dir": args.worker_dir,
        "narrowed_matrix": {
            "baselines": args.baselines,
            "sources": args.sources,
            "conditions": args.conditions,
            "variants": args.variants,
            "samples_per_condition": args.samples_per_condition,
            "max_rows": args.max_rows,
        },
        "important_note": "This is a narrowed diagnostic. GT-blended variants are diagnostic upper bounds only, not deployable policy actions.",
        "scope": "phase3_8b_narrow_pose1_repair_no_phase4_no_cps",
    }
    write_json_atomic(raw_summary_path, raw_summary)
    print(json.dumps(raw_summary, indent=2, sort_keys=True))


def run_worker(args: argparse.Namespace) -> None:
    assert_gate()
    spec = json.loads(Path(args.worker_json).read_text())
    root = Path(spec["root"]).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    scripts_dir = root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    import phase3_8_idm_geometry_repair_probe as p38

    data, meta = p38.load_windows(root, spec["windows"])
    codec = p38.load_codec(root, data, spec["action_template"])
    paths = p38.codec_paths(codec)
    idxs = p38.make_path_indices(paths)

    ckpt, state_model, idm = p38.load_models(root, spec["checkpoint_root"], spec["baseline"])
    item = spec["item"]
    idx = int(item["idx"])

    gt_vec, idm_gt_vec, idm_pred_vec, future_key, future_mae = p38.build_idm_vectors(
        data=data,
        idx=idx,
        baseline=spec["baseline"],
        state_model=state_model,
        idm=idm,
        pred_samples=int(spec["pred_samples"]),
        seed=int(spec["seed_base"]) + idx,
        clip_std=float(spec["action_clip_std"]),
    )

    source_vectors = {
        "idm_gt_future": idm_gt_vec,
        "idm_pred_future": idm_pred_vec,
    }
    if spec["source_action"] not in source_vectors:
        raise RuntimeError(f"unknown source_action={spec['source_action']}")

    variants = p38.repair_variants(source_vectors[spec["source_action"]], gt_vec, idxs)
    if spec["repair_variant"] not in variants:
        raise RuntimeError(f"repair_variant={spec['repair_variant']} not available; got {sorted(variants.keys())}")

    vec = variants[spec["repair_variant"]]
    action = p38.decode(codec, vec)
    ai = p38.action_info(action)
    gt_action = p38.decode(codec, gt_vec)
    gt_info = p38.action_info(gt_action)

    p36 = p38.import_phase36(root)
    tasks, Environment, close_helper = p36.import_tf_free_ravens(root)
    assert_no_forbidden("after_import_runtime")

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
        "source_action": spec["source_action"],
        "repair_variant": spec["repair_variant"],
        "future_key": future_key,
        "future_mae": future_mae,
        "primary_hidden_condition": "hidden_breakaway_pin",
        "diagnostic_hidden_condition": "hidden_pin",
        "primary_pair": "free_vs_hidden_breakaway_pin",
        "diagnostic_pair": "free_vs_hidden_pin",
        "scope": "phase3_8b_narrow_pose1_repair_no_phase4_no_cps",
        "is_gt_blended_diagnostic": int(spec["repair_variant"] != "idm_original"),
        "is_deployable_policy_action": int(spec["repair_variant"] == "idm_original"),
    }
    row.update(ai)
    row.update(p38.vector_compare(vec, gt_vec))
    row.update(p38.geometry_compare(ai, gt_info))
    row.update(exec_info)

    Path(args.worker_out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.worker_out_json).write_text(json.dumps(row, indent=2, sort_keys=True))
    print(json.dumps({"status": "ok", "variant": spec["repair_variant"], "delta": row.get("exec_delta_final_fraction")}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--windows", default="data/phase3_state_diff_windows/phase3_windows.npz")
    parser.add_argument("--action_template", default="data/phase3_state_diff_windows/phase3_action_template.pkl")
    parser.add_argument("--checkpoint_root", default="checkpoints/phase3")
    parser.add_argument("--phase37_raw", default="reports/phase3_7_learned_action_alignment_raw_summary.json")
    parser.add_argument("--baselines", nargs="+", default=["state_action"])
    parser.add_argument("--sources", nargs="+", default=["idm_gt_future"])
    parser.add_argument("--conditions", nargs="+", default=REQUIRED_CONDITIONS)
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    parser.add_argument("--samples_per_condition", type=int, default=1)
    parser.add_argument("--pred_samples", type=int, default=16)
    parser.add_argument("--seed_base", type=int, default=385000)
    parser.add_argument("--action_clip_std", type=float, default=3.0)
    parser.add_argument("--motion_timeout", type=float, default=15.0)
    parser.add_argument("--max_prefix_actions", type=int, default=20)
    parser.add_argument("--row_timeout_sec", type=float, default=240.0)
    parser.add_argument("--total_timeout_sec", type=float, default=1800.0)
    parser.add_argument("--max_rows", type=int, default=18)
    parser.add_argument("--stop_after_first_effective", action="store_true")
    parser.add_argument("--effective_delta_threshold", type=float, default=0.05)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--out_csv", default="reports/phase3_8b_narrow_pose1_repair_trials.csv")
    parser.add_argument("--out_json", default="reports/phase3_8b_narrow_pose1_repair_raw_summary.json")
    parser.add_argument("--progress_json", default="reports/phase3_8b_narrow_pose1_repair_progress.json")
    parser.add_argument("--worker_dir", default="reports/phase3_8b_workers")
    parser.add_argument("--worker_json", default="")
    parser.add_argument("--worker_out_json", default="")
    args = parser.parse_args()

    if args.worker_json:
        run_worker(args)
    else:
        run_parent(args)


if __name__ == "__main__":
    main()
