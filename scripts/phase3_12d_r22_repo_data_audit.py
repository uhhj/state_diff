#!/usr/bin/env python3
"""Repository, windows, action-codec, raw-source, and checkpoint integrity audit."""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from phase3_12d_r22_integrity import (
    decode_text,
    sha256_file,
    stable_episode_key,
    stable_window_hash,
    strict_json_dump,
)


def command(args: List[str], cwd: Path) -> str:
    return subprocess.check_output(args, cwd=str(cwd), text=True).strip()


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if isinstance(row.get(key), float) and not math.isfinite(row[key]) else row.get(key, "") for key in fields})


def numeric_health(name: str, array: np.ndarray) -> Dict[str, Any]:
    value = np.asarray(array)
    finite = np.isfinite(value)
    flat = value[finite].astype(np.float64)
    return {
        "block": name,
        "shape": "x".join(str(x) for x in value.shape),
        "min": float(np.min(flat)) if flat.size else 0.0,
        "max": float(np.max(flat)) if flat.size else 0.0,
        "mean": float(np.mean(flat)) if flat.size else 0.0,
        "std": float(np.std(flat)) if flat.size else 0.0,
        "p01": float(np.quantile(flat, 0.01)) if flat.size else 0.0,
        "p50": float(np.quantile(flat, 0.50)) if flat.size else 0.0,
        "p99": float(np.quantile(flat, 0.99)) if flat.size else 0.0,
        "num_nan": int(np.isnan(value).sum()),
        "num_inf": int(np.isinf(value).sum()),
        "num_zero_variance": int(np.sum(np.std(value.astype(np.float64), axis=0) == 0)) if value.ndim >= 2 and np.all(finite) else 0,
        "num_near_zero_variance": int(np.sum(np.std(value.astype(np.float64), axis=0) < 1e-8)) if value.ndim >= 2 and np.all(finite) else 0,
        "num_abs_gt_1e3": int(np.sum(np.abs(value[finite]) > 1e3)),
    }


def checkpoint_path_from_phase39b(root: Path, report: Path, ablation: str) -> Path:
    payload = json.loads(report.read_text())
    value = payload["results"][ablation]["checkpoint"]
    path = Path(value)
    return path if path.is_absolute() else root / path


def first_state_checkpoint(root: Path, checkpoint_root: Path) -> Path:
    candidates = sorted((root / checkpoint_root / "state_action").glob("fold_*_seed_*/state_model.pt"))
    if not candidates:
        raise FileNotFoundError("state model checkpoint")
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--windows", default="data/phase3_state_diff_windows/phase3_windows.npz")
    parser.add_argument("--action-template", default="data/phase3_state_diff_windows/phase3_action_template.pkl")
    parser.add_argument("--old-checkpoint-root", default="checkpoints/phase3")
    parser.add_argument("--phase39b-raw", default="reports/phase3_9b_ablation_raw_summary.json")
    parser.add_argument("--best-ablation", default="xy_only_high_weight")
    parser.add_argument("--out-prefix", default="reports/phase3_12d_r22")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    for path in (root, root / "scripts", root / "external/deformable-ravens"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from ccda_phase3.data_io import (
        ROBOT_PROXY_DIM,
        build_windows_from_dataset,
        load_action_codec_from_template,
        state_from_info,
        visible_seed_from_extras,
    )
    from ccda_phase3.rollout import state_from_live_info
    from ccda_phase3.train_utils import load_future_model, load_inverse_model

    prefix = root / args.out_prefix
    windows_path = root / args.windows
    template_path = root / args.action_template
    state_checkpoint = first_state_checkpoint(root, Path(args.old_checkpoint_root))
    idm_checkpoint = checkpoint_path_from_phase39b(root, root / args.phase39b_raw, args.best_ablation)
    issues: List[Dict[str, Any]] = []

    def issue(level: str, name: str, detail: Any) -> None:
        issues.append({"level": level, "name": name, "detail": detail})

    data = np.load(windows_path, allow_pickle=False)
    required = ["paper_x", "state_action_x", "y_state", "y_action", "condition_name", "visible_seed", "split_name", "source_file", "window_t", "success", "final_fraction", "th", "action_dim", "n_beads"]
    for key in required:
        if key not in data.files:
            issue("FAIL", "missing_window_key", key)
    n = int(data["paper_x"].shape[0])
    scalar_keys = {"state_dim", "robot_pose_dim", "action_dim", "n_beads", "th", "tf", "action_template_json_or_pickle_path", "meta_json"}
    alignment = {}
    for key in data.files:
        value = data[key]
        if key not in scalar_keys and value.ndim > 0:
            alignment[key] = int(value.shape[0])
            if value.shape[0] != n:
                issue("FAIL", "row_alignment_failure", {"key": key, "length": int(value.shape[0]), "N": n})

    th = int(np.asarray(data["th"]).reshape(-1)[0])
    tf = int(np.asarray(data["tf"]).reshape(-1)[0]) if "tf" in data.files else int(data["y_state"].shape[1])
    action_dim = int(np.asarray(data["action_dim"]).reshape(-1)[0])
    n_beads = int(np.asarray(data["n_beads"]).reshape(-1)[0])
    state_dim = n_beads * 4 + ROBOT_PROXY_DIM
    paper_dim = th * state_dim
    state_action_dim = paper_dim + th * 14
    dimensions = {
        "N": n, "th": th, "tf": tf, "action_dim": action_dim, "n_beads": n_beads,
        "state_dim": state_dim, "expected_paper_x_dim": paper_dim,
        "expected_state_action_x_dim": state_action_dim,
        "paper_x_shape": list(data["paper_x"].shape), "state_action_x_shape": list(data["state_action_x"].shape),
        "y_state_shape": list(data["y_state"].shape), "y_action_shape": list(data["y_action"].shape),
    }
    if data["paper_x"].shape != (n, paper_dim) or data["state_action_x"].shape != (n, state_action_dim) or data["y_action"].shape != (n, 14):
        issue("FAIL", "dimension_reconstruction_failed", dimensions)
    prefix_diff = float(np.max(np.abs(data["state_action_x"][:, :paper_dim] - data["paper_x"])))
    if prefix_diff > 1e-7:
        issue("FAIL", "paper_state_action_prefix_mismatch", prefix_diff)

    health_rows = [numeric_health(key, data[key]) for key in ("paper_x", "state_action_x", "y_state", "y_action")]
    for row in health_rows:
        if row["num_nan"] or row["num_inf"]:
            issue("FAIL", "nonfinite_model_array", row)
    final_fraction_nonfinite = int(np.sum(~np.isfinite(data["final_fraction"])))

    codec = load_action_codec_from_template(template_path)
    codec_summary = codec.summary()
    codec_roundtrip = 0.0
    for row in data["y_action"][: min(n, 256)]:
        encoded = codec.encode(codec.decode(row))
        codec_roundtrip = max(codec_roundtrip, float(np.max(np.abs(encoded - row))))
    if codec.dim() != 14 or codec_summary.get("num_camera_config_paths", -1) != 0 or codec_roundtrip > 1e-6:
        issue("FAIL", "action_codec_integrity_failed", {"summary": codec_summary, "roundtrip": codec_roundtrip})

    conditions = np.asarray([decode_text(x) for x in data["condition_name"]])
    splits = np.asarray([decode_text(x) for x in data["split_name"]])
    sources = np.asarray([decode_text(x) for x in data["source_file"]])
    seeds = np.asarray(data["visible_seed"], dtype=np.int64)
    window_t = np.asarray(data["window_t"], dtype=np.int64)
    if np.any(seeds < 0) or np.any(window_t < 0):
        issue("FAIL", "invalid_seed_or_window_t", {"negative_seeds": int(np.sum(seeds < 0)), "negative_t": int(np.sum(window_t < 0))})

    episode_splits: Dict[str, set] = defaultdict(set)
    seed_splits: Dict[int, set] = defaultdict(set)
    source_splits: Dict[str, set] = defaultdict(set)
    tuple_counts: Counter = Counter()
    hash_splits: Dict[str, set] = defaultdict(set)
    for index in range(n):
        episode_splits[stable_episode_key(conditions[index], seeds[index], sources[index])].add(splits[index])
        seed_splits[int(seeds[index])].add(splits[index])
        source_splits[sources[index]].add(splits[index])
        tuple_counts[(conditions[index], int(seeds[index]), int(window_t[index]), splits[index])] += 1
        digest = stable_window_hash(data["paper_x"][index], data["state_action_x"][index], data["y_state"][index], data["y_action"][index])
        hash_splits[digest].add(splits[index])
    overlap_rows = []
    for kind, mapping in (("episode", episode_splits), ("visible_seed", seed_splits), ("source_file", source_splits), ("window_hash", hash_splits)):
        for key, value in mapping.items():
            if len(value) > 1:
                overlap_rows.append({"kind": kind, "key": str(key), "splits": " ".join(sorted(value))})
    duplicate_rows = [{"condition": key[0], "visible_seed": key[1], "window_t": key[2], "split": key[3], "count": count} for key, count in tuple_counts.items() if count > 1]
    hard_overlap = [row for row in overlap_rows if row["kind"] != "source_file"]
    source_basename_overlap = [row for row in overlap_rows if row["kind"] == "source_file"]
    if any(row["kind"] == "visible_seed" for row in hard_overlap):
        issue("FAIL", "phase312d_r22_visible_seed_split_leakage", len(hard_overlap))
    elif hard_overlap:
        issue("FAIL", "split_leakage_detected", len(hard_overlap))
    if source_basename_overlap:
        issue(
            "WARN",
            "source_file_basename_reused_across_split_roots",
            {
                "count": len(source_basename_overlap),
                "interpretation": (
                    "source_file stores an episode basename, not a globally unique path. "
                    "The train/heldout roots and visible seeds are disjoint; basename reuse "
                    "is reported but is not itself source leakage."
                ),
            },
        )
    if duplicate_rows:
        issue("FAIL", "duplicate_windows_detected", len(duplicate_rows))

    distribution_rows = []
    for split in sorted(set(splits)):
        mask_split = splits == split
        for condition in sorted(set(conditions)):
            mask = mask_split & (conditions == condition)
            distribution_rows.append({"split": split, "condition": condition, "windows": int(mask.sum()), "visible_seeds": int(np.unique(seeds[mask]).size), "source_files": int(np.unique(sources[mask]).size)})

    meta = json.loads(str(np.asarray(data["meta_json"]).item())) if "meta_json" in data.files else {}
    legacy = not bool(meta.get("environment_semantics_version")) or "deferred" not in str(meta.get("environment_semantics_version", ""))
    if legacy:
        issue("WARN", "legacy_environment_semantics", "windows lack deferred-arming semantics manifest")

    source_rows = []
    raw_roots = {
        "train": root / "external/deformable-ravens/data/phase3_ccda_large/train/hidden-contact-cable-line",
        "heldout": root / "external/deformable-ravens/data/phase3_ccda_large/heldout/hidden-contact-cable-line",
    }
    reconstruction_available = all(path.exists() for path in raw_roots.values())
    if reconstruction_available:
        rebuilt = {}
        active_conditions = sorted(set(conditions))
        for split, raw_root in raw_roots.items():
            rows, _, _ = build_windows_from_dataset(split, raw_root, th, tf, conditions=active_conditions)
            for row in rows:
                rebuilt[(split, row["condition_name"], int(row["visible_seed"]), row["source_file"], int(row["window_t"]))] = row
        free_npz = {}
        for index in range(n):
            if conditions[index] == "free":
                free_npz[(str(splits[index]), int(seeds[index]), int(window_t[index]))] = index
        canonicalization_enabled = bool(meta.get("canonicalized_paired_input_windows", 0))
        canonicalization_rule = str(meta.get("canonicalization_rule", ""))
        rng = np.random.default_rng(312220)
        for index in rng.choice(n, size=min(n, 128), replace=False):
            split = str(splits[index])
            raw_split = "heldout" if split == "heldout" else "train"
            key = (raw_split, conditions[index], int(seeds[index]), sources[index], int(window_t[index]))
            row = rebuilt.get(key)
            if row is None:
                source_rows.append({"index": int(index), "condition": conditions[index], "mode": "missing", "status": "missing", "paper_x": "", "state_action_x": "", "y_state": "", "y_action": ""})
                continue
            diffs = {name: float(np.max(np.abs(np.asarray(row[name]) - data[name][index]))) for name in ("paper_x", "state_action_x", "y_state", "y_action")}
            targets_ok = max(diffs["y_state"], diffs["y_action"]) <= 1e-6
            raw_inputs_ok = max(diffs["paper_x"], diffs["state_action_x"]) <= 1e-6
            mode = "raw_exact"
            canonical_diffs = {"canonical_paper_x": "", "canonical_state_action_x": ""}
            canonical_inputs_ok = False
            if conditions[index] != "free" and canonicalization_enabled:
                free_index = free_npz.get((str(splits[index]), int(seeds[index]), int(window_t[index])))
                if free_index is not None:
                    canonical_diffs = {
                        "canonical_paper_x": float(np.max(np.abs(data["paper_x"][free_index] - data["paper_x"][index]))),
                        "canonical_state_action_x": float(np.max(np.abs(data["state_action_x"][free_index] - data["state_action_x"][index]))),
                    }
                    canonical_inputs_ok = max(canonical_diffs.values()) <= 1e-6
            if raw_inputs_ok and targets_ok:
                status = "ok"
            elif targets_ok and canonical_inputs_ok:
                status = "ok"
                mode = "canonicalized_free_input_exact_targets"
            else:
                status = "mismatch"
                mode = "unexpected_mismatch"
            source_rows.append({"index": int(index), "condition": conditions[index], "mode": mode, "status": status, **diffs, **canonical_diffs})
        if any(row["status"] != "ok" for row in source_rows):
            issue("FAIL", "phase312d_r22_source_reconstruction_failed", Counter(row["status"] for row in source_rows))
        elif canonicalization_enabled:
            issue(
                "WARN",
                "paired_input_canonicalization_verified",
                {
                    "rule": canonicalization_rule,
                    "sample_modes": dict(Counter(row["mode"] for row in source_rows)),
                    "targets_reconstructed_exactly": True,
                },
            )
    else:
        issue("WARN", "source_reconstruction_unavailable", {key: str(path) for key, path in raw_roots.items()})

    checkpoint = {"state_path": str(state_checkpoint), "idm_path": str(idm_checkpoint), "state_sha256": sha256_file(state_checkpoint), "idm_sha256": sha256_file(idm_checkpoint)}
    try:
        future = load_future_model(state_checkpoint)
        inverse = load_inverse_model(idm_checkpoint)
        state_input = np.asarray(data["state_action_x"][:1], dtype=np.float32)
        future_sample = future.predict_mean(state_input, n_samples=1, seed=312220)
        idm_input = np.concatenate([data["paper_x"][:1], data["y_state"][:1].reshape(1, -1)], axis=1).astype(np.float32)
        action_sample = inverse.predict(idm_input)
        checkpoint.update({"future_x_dim": int(future.model.x_dim), "future_y_dim": int(future.model.y_dim), "future_sample_shape": list(future_sample.shape), "idm_input_shape": list(idm_input.shape), "idm_output_shape": list(action_sample.shape), "finite": bool(np.all(np.isfinite(future_sample)) and np.all(np.isfinite(action_sample))), "inverse_backend": type(inverse).__name__})
        if future.model.x_dim != state_action_dim or action_sample.shape != (1, 14) or not checkpoint["finite"] or "Numpy" in checkpoint["inverse_backend"]:
            issue("FAIL", "phase312d_r22_checkpoint_dimension_mismatch", checkpoint)
    except Exception as exc:
        issue("FAIL", "checkpoint_smoke_failed", repr(exc))

    files = {"windows": {"path": str(windows_path), "bytes": windows_path.stat().st_size, "sha256": sha256_file(windows_path)}, "action_template": {"path": str(template_path), "bytes": template_path.stat().st_size, "sha256": sha256_file(template_path)}, "state_checkpoint": checkpoint["state_sha256"], "idm_checkpoint": checkpoint["idm_sha256"]}
    functions = {name: str(inspect.signature(obj)) for name, obj in {"state_from_info": state_from_info, "state_from_live_info": state_from_live_info, "visible_seed_from_extras": visible_seed_from_extras, "build_windows_from_dataset": build_windows_from_dataset, "load_future_model": load_future_model, "load_inverse_model": load_inverse_model}.items()}
    discovery = {"main_branch": command(["git", "branch", "--show-current"], root), "main_head": command(["git", "rev-parse", "HEAD"], root), "origin_head": command(["git", "ls-remote", "origin", "Experiment1"], root).split()[0], "main_status": command(["git", "status", "--short"], root), "submodule_head": command(["git", "-C", "external/deformable-ravens", "rev-parse", "HEAD"], root), "submodule_status": command(["git", "-C", "external/deformable-ravens", "status", "--short"], root), "python": sys.version, "executable": sys.executable, "dependencies": {name: getattr(__import__(name), "__version__", "unknown") for name in ("numpy", "torch", "scipy", "pybullet")}, "files": files, "function_signatures": functions, "code_hashes": {str(path.relative_to(root)): sha256_file(path) for path in (root / "ccda_phase3/data_io.py", root / "ccda_phase3/rollout.py", root / "scripts/phase3_12d_r2_common.py", root / "scripts/phase3_12d_r21_paired_horizon_audit.py")}, "r21_root_cause": json.loads((root / "reports/phase3_12d_r21_environment_audit_summary.json").read_text()).get("root_cause")}
    try:
        import sklearn
        discovery["dependencies"]["sklearn"] = sklearn.__version__
    except Exception:
        discovery["dependencies"]["sklearn"] = "unavailable_torch_linear_fallback"
    strict_json_dump(root / "reports/phase3_12d_r22_repo_discovery.json", discovery)

    fatal = any(item["level"] == "FAIL" for item in issues)
    verdict = "FAIL" if fatal else "WARN" if issues else "PASS"
    if fatal:
        names = {item["name"] for item in issues if item["level"] == "FAIL"}
        if "phase312d_r22_visible_seed_split_leakage" in names or "split_leakage_detected" in names:
            root_cause = "phase312d_r22_split_leakage_detected"
        elif "phase312d_r22_source_reconstruction_failed" in names:
            root_cause = "phase312d_r22_source_reconstruction_failed"
        elif "phase312d_r22_checkpoint_dimension_mismatch" in names:
            root_cause = "phase312d_r22_checkpoint_dimension_mismatch"
        else:
            root_cause = "phase312d_r22_repo_or_dataset_integrity_failed"
    elif legacy:
        root_cause = "phase312d_r22_repo_data_integrity_passed_with_legacy_warning"
    else:
        root_cause = "phase312d_r22_repo_data_integrity_passed"

    overlap_counts = dict(Counter(row["kind"] for row in overlap_rows))
    reconstruction_counts = dict(Counter(row["mode"] for row in source_rows))
    payload = {
        "verdict": verdict,
        "root_cause": root_cause,
        "collection_allowed": not fatal,
        "legacy_environment_semantics": legacy,
        "formal_r2_training_allowed": False,
        "legacy_checkpoint_bridge_only": True,
        "dimensions": dimensions,
        "num_unique_episodes": len(episode_splits),
        "num_unique_visible_seeds": int(np.unique(seeds).size),
        "split_overlap_counts": overlap_counts,
        "hard_split_overlap_count": len(hard_overlap),
        "source_basename_overlap_count": len(source_basename_overlap),
        "duplicate_window_key_count": len(duplicate_rows),
        "source_reconstruction_counts": reconstruction_counts,
        "source_reconstruction_all_valid": bool(source_rows) and all(row["status"] == "ok" for row in source_rows),
        "row_alignment": alignment,
        "paper_prefix_max_abs": prefix_diff,
        "final_fraction_nonfinite": final_fraction_nonfinite,
        "feature_health": health_rows,
        "codec": {"summary": codec_summary, "roundtrip_max_abs": codec_roundtrip},
        "checkpoint": checkpoint,
        "distribution": distribution_rows,
        "issues": issues,
    }
    strict_json_dump(Path(str(prefix) + "_repo_data_audit_summary.json"), payload)
    write_csv(Path(str(prefix) + "_split_overlap.csv"), overlap_rows, ["kind", "key", "splits"])
    write_csv(Path(str(prefix) + "_duplicate_windows.csv"), duplicate_rows, ["condition", "visible_seed", "window_t", "split", "count"])
    write_csv(Path(str(prefix) + "_feature_health.csv"), health_rows, list(health_rows[0]))
    write_csv(Path(str(prefix) + "_dataset_distribution.csv"), distribution_rows, list(distribution_rows[0]))
    write_csv(Path(str(prefix) + "_source_reconstruction.csv"), source_rows, ["index", "condition", "mode", "status", "paper_x", "state_action_x", "y_state", "y_action", "canonical_paper_x", "canonical_state_action_x"])
    lines = ["# Phase3.12d-r2.2 Repository and Data Integrity Audit", "", f"- Verdict: `{verdict}`", f"- Root cause: `{root_cause}`", f"- Collection allowed: `{not fatal}`", f"- Windows: `{n}` rows", f"- Legacy checkpoint bridge only: `True`", "", "## Dimensions", "", "```json", json.dumps(dimensions, indent=2, sort_keys=True, allow_nan=False), "```", "", "## Issues", "", "| Level | Name | Detail |", "|---|---|---|"]
    if issues:
        lines.extend(f"| `{item['level']}` | `{item['name']}` | {str(item['detail']).replace('|', '/')} |" for item in issues)
    else:
        lines.append("| `PASS` | `none` | No issues. |")
    Path(str(prefix) + "_repo_data_audit_report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    if fatal:
        raise SystemExit("[Phase3.12d-r2.2] repository/data integrity audit failed")


if __name__ == "__main__":
    main()
