#!/usr/bin/env python3
"""Audit formal state-v2 paired data, windows, splits, and provenance."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from ccda_phase3.data_io import load_action_codec_from_template, load_episode
from ccda_phase3.schema_v2 import ACTION_DIM, DEFAULT_TF, DEFAULT_TH, FORMAL_CONDITIONS, FORMAL_TASK_NAME, PAPER_X_DIM, STATE_ACTION_X_DIM, STATE_DIM, build_window


def strict_dump(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive, negative = scores[labels == 1], scores[labels == 0]
    difference = positive[:, None] - negative[None, :]
    return float(np.mean(difference > 0) + 0.5 * np.mean(difference == 0))


def grouped_hiddenness(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> Dict[str, float]:
    unique = np.unique(groups)
    scores = np.zeros(y.size, dtype=np.float64)
    folds = min(5, unique.size)
    for fold in range(folds):
        test_groups = unique[np.arange(unique.size) % folds == fold]
        test = np.isin(groups, test_groups); train = ~test
        mean = x[train].mean(axis=0); scale = x[train].std(axis=0); scale[scale < 1e-8] = 1.0
        standardized = (x - mean) / scale
        direction = standardized[train & (y == 1)].mean(axis=0) - standardized[train & (y == 0)].mean(axis=0)
        scores[test] = standardized[test] @ direction
    point = auc(y, scores)
    threshold = float(np.median(scores))
    prediction = scores >= threshold
    balanced = 0.5 * (np.mean(prediction[y == 1]) + np.mean(~prediction[y == 0]))
    paired = []
    for group in unique:
        indices = np.flatnonzero(groups == group)
        neg = indices[y[indices] == 0][0]; pos = indices[y[indices] == 1][0]
        paired.append(1.0 if scores[pos] > scores[neg] else 0.5 if scores[pos] == scores[neg] else 0.0)
    rng = np.random.default_rng(313013)
    boot = []
    for _ in range(2000):
        sampled = rng.choice(unique, size=unique.size, replace=True)
        indices = np.concatenate([np.flatnonzero(groups == group) for group in sampled])
        boot.append(auc(y[indices], scores[indices]))
    return {"roc_auc": point, "roc_auc_ci_low": float(np.quantile(boot, 0.025)), "roc_auc_ci_high": float(np.quantile(boot, 0.975)), "balanced_accuracy": float(balanced), "paired_accuracy": float(np.mean(paired))}


def chamfer(left: np.ndarray, right: np.ndarray) -> float:
    distance = np.linalg.norm(left[:, None, :] - right[None, :, :], axis=2)
    return float(0.5 * (np.min(distance, axis=1).mean() + np.min(distance, axis=0).mean()))


def hash_row(arrays: Dict[str, np.ndarray], index: int) -> str:
    digest = hashlib.sha256()
    for key in ("paper_x", "state_action_x", "y_state", "y_action"):
        digest.update(np.ascontiguousarray(arrays[key][index]).tobytes())
    return digest.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/state_diff2")
    ap.add_argument("--raw-root", default="data/phase3_state_v2_slack/raw")
    ap.add_argument("--windows", default="data/phase3_state_v2_slack/windows/phase3_13_windows.npz")
    ap.add_argument("--template", default="data/phase3_state_v2_slack/windows/action_template.pkl")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out-prefix", default="reports/phase3_13")
    args = ap.parse_args()
    root = Path(args.root).resolve(); raw_root = root / args.raw_root
    with np.load(root / args.windows, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    schema_checks = {
        "paper_x_261": arrays["paper_x"].shape[1:] == (PAPER_X_DIM,),
        "state_action_x_303": arrays["state_action_x"].shape[1:] == (STATE_ACTION_X_DIM,),
        "future_4x87": arrays["y_state"].shape[1:] == (DEFAULT_TF, STATE_DIM),
        "final_state_87": arrays["y_final_state"].shape[1:] == (STATE_DIM,),
        "action_14": arrays["y_action"].shape[1:] == (ACTION_DIM,),
        "all_finite": all(np.all(np.isfinite(value)) for value in arrays.values() if value.dtype.kind in "fiu"),
        "no_object_arrays": all(value.dtype.kind != "O" for value in arrays.values()),
    }
    codec = load_action_codec_from_template(root / args.template)
    codec_summary = codec.summary()
    codec_roundtrip = codec.roundtrip_error(codec.template)
    codec_pass = codec.dim() == 14 and codec_summary["num_camera_config_paths"] == 0 and codec_roundtrip <= 1e-6
    splits = arrays["split_name"].astype(str); seeds = arrays["visible_seed"].astype(np.int64)
    groups = arrays["pair_group"].astype(str); sources = arrays["source_file"].astype(str)
    split_rows = []
    split_sets = {}
    for split in ("train", "val", "test"):
        mask = splits == split
        split_sets[split] = {
            "seeds": set(seeds[mask].tolist()), "groups": set(groups[mask].tolist()),
            "sources": set(sources[mask].tolist()), "hashes": {hash_row(arrays, index) for index in np.flatnonzero(mask)},
        }
        split_rows.append({"split": split, "visible_seeds": len(split_sets[split]["seeds"]), "episodes": len(split_sets[split]["sources"]), "windows": int(np.sum(mask))})
    overlap = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        for key in ("seeds", "groups", "sources", "hashes"):
            overlap[f"{left}_{right}_{key}"] = len(split_sets[left][key].intersection(split_sets[right][key]))
    split_pass = all(value == 0 for value in overlap.values())

    index = {(str(splits[i]), str(arrays["condition_name"][i]), int(seeds[i]), int(arrays["window_t"][i])): i for i in range(len(seeds))}
    reconstruction_rows = []
    reconstruction_max = 0.0
    for split in ("train", "val", "test"):
        for condition in FORMAL_CONDITIONS:
            paths = sorted((raw_root / split / condition).glob("*.pkl"))
            for path in paths[: min(32, len(paths))]:
                episode = load_episode(condition, path, codec=codec)
                for current in range(min(len(episode["actions"]), len(episode["states"]) - 1)):
                    key = (split, condition, episode["visible_seed"], current)
                    row = index[key]
                    rebuilt = build_window(states=episode["states"], action_vectors=episode["actions"], current_index=current)
                    error = max(float(np.max(np.abs(rebuilt[name] - arrays[name][row]))) for name in ("paper_x", "state_action_x", "y_state", "y_action"))
                    reconstruction_max = max(reconstruction_max, error)
                reconstruction_rows.append({"split": split, "condition": condition, "source_file": str(path), "max_abs": error})
    reconstruction_pass = bool(reconstruction_rows) and reconstruction_max <= 1e-6
    source_text = (root / "scripts/phase3_13_build_windows.py").read_text().lower() + (root / "ccda_phase3/data_io.py").read_text().lower()
    canonicalization_pass = "copy_free" not in source_text and "copy free" not in source_text and reconstruction_pass

    hiddenness_rows = []
    pre = arrays["pre_engagement"].astype(bool)
    for feature in ("paper_x", "state_action_x"):
        pair_x, pair_y, pair_groups = [], [], []
        conditions = arrays["condition_name"].astype(str)
        for split in ("train", "val", "test"):
            for seed in sorted(split_sets[split]["seeds"]):
                times = sorted(set(arrays["window_t"][(splits == split) & (seeds == seed)].tolist()))
                for current in times:
                    left = index.get((split, "free", seed, current)); right = index.get((split, "hidden_slack_breakaway_pin_v2", seed, current))
                    if left is None or right is None or not pre[right]:
                        continue
                    pair_x.extend([arrays[feature][left], arrays[feature][right]])
                    pair_y.extend([0, 1]); pair_groups.extend([seed, seed])
        result = grouped_hiddenness(np.asarray(pair_x, dtype=np.float32), np.asarray(pair_y), np.asarray(pair_groups))
        hiddenness_rows.append({"feature": feature, "paired_rows": len(pair_y), **result})
    hiddenness_pass = all(row["roc_auc"] <= 0.60 and row["roc_auc_ci_high"] < 0.65 and row["balanced_accuracy"] <= 0.60 for row in hiddenness_rows)

    pair_rows = []
    eligible_by_split = defaultdict(int); diverged_by_split = defaultdict(int)
    for split in ("train", "val", "test"):
        for seed in sorted(split_sets[split]["seeds"]):
            times = sorted(set(arrays["window_t"][(splits == split) & (seeds == seed)].tolist()))
            for current in times:
                left = index.get((split, "free", seed, current)); right = index.get((split, "hidden_slack_breakaway_pin_v2", seed, current))
                if left is None or right is None:
                    continue
                scale = float(np.std(np.concatenate([arrays["paper_x"][left], arrays["paper_x"][right]])))
                prefix = float(np.mean(np.abs(arrays["paper_x"][left] - arrays["paper_x"][right])) / max(scale, 1e-8))
                final_distance = chamfer(arrays["y_final_state"][left][:48].reshape(24, 2), arrays["y_final_state"][right][:48].reshape(24, 2))
                progress = abs(float(arrays["final_fraction"][left] - arrays["final_fraction"][right]))
                success_difference = bool(arrays["success"][left] != arrays["success"][right])
                eligible = prefix <= 0.05
                diverged = eligible and (final_distance >= 0.003 or progress >= 0.03 or success_difference)
                eligible_by_split[split] += int(eligible); diverged_by_split[split] += int(diverged)
                pair_rows.append({"split": split, "visible_seed": seed, "window_t": current, "normalized_prefix_mae": prefix, "future_final_xy_chamfer": final_distance, "final_fraction_abs_diff": progress, "success_diff": success_difference, "eligible": eligible, "diverged": diverged})
    divergence_rates = {split: diverged_by_split[split] / max(1, eligible_by_split[split]) for split in ("train", "val", "test")}
    divergence_pass = all(value >= 0.30 for value in divergence_rates.values())

    episode_rows = []
    for split in ("train", "val", "test"):
        for condition in FORMAL_CONDITIONS:
            for path in sorted((raw_root / split / condition).glob("*.pkl")):
                episode = load_episode(condition, path, codec=codec)
                episode_rows.append({"split": split, "condition": condition, "visible_seed": episode["visible_seed"], "engaged": episode["engagement_step"] >= 0, "released": episode["release_step"] >= 0})
    hidden_eps = [row for row in episode_rows if row["condition"] != "free"]
    free_eps = [row for row in episode_rows if row["condition"] == "free"]
    engagement_rate = sum(row["engaged"] for row in hidden_eps) / max(1, len(hidden_eps))
    release_rate = sum(row["released"] for row in hidden_eps) / max(1, len(hidden_eps))
    engagement_pass = engagement_rate >= 0.70 and release_rate >= 0.50 and not any(row["engaged"] for row in free_eps)
    expected_scale = {"train": 8, "val": 4, "test": 4} if args.smoke else {"train": 256, "val": 64, "test": 128}
    scale_pass = all(len(split_sets[split]["seeds"]) == expected for split, expected in expected_scale.items())

    if not all(schema_checks.values()): root_cause = "phase313_state_schema_mismatch"
    elif not codec_pass: root_cause = "phase313_state_schema_mismatch"
    elif not split_pass: root_cause = "phase313_split_leakage_detected"
    elif not reconstruction_pass: root_cause = "phase313_source_reconstruction_failed"
    elif not canonicalization_pass: root_cause = "phase313_input_canonicalization_detected"
    elif not hiddenness_pass: root_cause = "phase313_prefix_observably_leaked"
    elif not divergence_pass: root_cause = "phase313_future_branch_divergence_insufficient"
    elif not engagement_pass: root_cause = "phase313_action_policy_insufficient_engagement"
    elif not scale_pass: root_cause = "phase313_dataset_scale_insufficient"
    else: root_cause = "phase313_state_v2_dataset_supported"
    verdict = "PASS" if root_cause == "phase313_state_v2_dataset_supported" else "FAIL"
    payload = {
        "verdict": verdict, "root_cause": root_cause, "smoke": bool(args.smoke),
        "schema_checks": schema_checks, "codec": {"summary": codec_summary, "roundtrip_max_error": codec_roundtrip, "pass": codec_pass},
        "split_overlap": overlap, "split_pass": split_pass, "source_reconstruction_max_abs": reconstruction_max,
        "source_reconstruction_pass": reconstruction_pass, "canonicalization_pass": canonicalization_pass,
        "hiddenness": hiddenness_rows, "hiddenness_pass": hiddenness_pass,
        "future_divergence_rates": divergence_rates, "future_divergence_pass": divergence_pass,
        "hidden_engagement_rate": engagement_rate, "hidden_release_rate": release_rate, "engagement_coverage_pass": engagement_pass,
        "scale_expected": expected_scale, "scale_pass": scale_pass, "split_distribution": split_rows,
        "formal_task": FORMAL_TASK_NAME, "formal_conditions": list(FORMAL_CONDITIONS),
        "model_training": False, "candidate_matrix": False, "phase4": False, "cps": False,
    }
    prefix = root / args.out_prefix
    strict_dump(Path(str(prefix) + "_dataset_audit_summary.json"), payload)
    write_csv(Path(str(prefix) + "_split_distribution.csv"), split_rows)
    write_csv(Path(str(prefix) + "_pair_metrics.csv"), pair_rows)
    write_csv(Path(str(prefix) + "_source_reconstruction.csv"), reconstruction_rows)
    write_csv(Path(str(prefix) + "_hiddenness_metrics.csv"), hiddenness_rows)
    write_csv(Path(str(prefix) + "_future_divergence.csv"), pair_rows)
    lines = ["# Phase3.13 Formal Dataset Audit", "", f"- Verdict: `{verdict}`", f"- Root cause: `{root_cause}`", f"- Smoke: `{args.smoke}`", "", "| Gate | Result |", "|---|---:|", f"| Schema | `{all(schema_checks.values())}` |", f"| Codec | `{codec_pass}` |", f"| Split isolation | `{split_pass}` |", f"| Source reconstruction | `{reconstruction_pass}` |", f"| No canonicalization | `{canonicalization_pass}` |", f"| Pre-engagement hiddenness | `{hiddenness_pass}` |", f"| Future divergence | `{divergence_pass}` |", f"| Engagement/release coverage | `{engagement_pass}` |", f"| Scale | `{scale_pass}` |", "", "No future-model or IDM training, candidate matrix, Phase4, or CPS was run."]
    Path(str(prefix) + "_dataset_audit_report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    if verdict != "PASS":
        raise SystemExit(f"[Phase3.13] audit failed: {root_cause}")


if __name__ == "__main__":
    main()
