#!/usr/bin/env python3
import sys
from pathlib import Path as _Phase3Path
ROOT = _Phase3Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np

from ccda_phase3.action_codec import ExecutableActionCodec
from ccda_phase3.data_io import (
    FORBIDDEN_METADATA_NOT_IN_X,
    build_windows_from_dataset,
    normalize_conditions,
    save_action_template,
)


def hidden_conditions(conditions):
    return [c for c in conditions if c != "free"]


def paired_input_diff_stats(windows, conditions):
    groups = {}
    for w in windows:
        key = (w["split_name"], int(w["visible_seed"]), int(w["window_t"]))
        groups.setdefault(key, []).append(w)

    by_hidden = {}
    all_paper = []
    all_state_action = []
    for cond in hidden_conditions(conditions):
        paper_diffs = []
        state_action_diffs = []
        num_pairs = 0
        for rows in groups.values():
            by_cond = {r["condition_name"]: r for r in rows}
            if "free" not in by_cond or cond not in by_cond:
                continue
            num_pairs += 1
            paper = float(np.max(np.abs(by_cond["free"]["paper_x"] - by_cond[cond]["paper_x"])))
            state_action = float(np.max(np.abs(by_cond["free"]["state_action_x"] - by_cond[cond]["state_action_x"])))
            paper_diffs.append(paper)
            state_action_diffs.append(state_action)
            all_paper.append(paper)
            all_state_action.append(state_action)

        by_hidden[cond] = {
            "num_pairs": int(num_pairs),
            "paper_x_max_abs": stat(paper_diffs),
            "state_action_x_max_abs": stat(state_action_diffs),
        }

    return {
        "num_pairs": int(sum(v["num_pairs"] for v in by_hidden.values())),
        "paper_x_max_abs": stat(all_paper),
        "state_action_x_max_abs": stat(all_state_action),
        "by_hidden_condition": by_hidden,
    }


def stat(xs):
    if not xs:
        return {"mean": None, "max": None}
    return {"mean": float(np.mean(xs)), "max": float(np.max(xs))}


def canonicalize_paired_inputs(windows):
    """Force paired CCDA branches to share exactly the same model input.

    Targets remain condition-specific. Model inputs are copied from the free
    branch within each split/visible_seed/window_t group so Phase3 tests the
    contact-blind branch ambiguity rather than hidden/contact label leakage.
    """
    groups = {}
    for w in windows:
        key = (w["split_name"], int(w["visible_seed"]), int(w["window_t"]))
        groups.setdefault(key, []).append(w)
    changed = 0
    for rows in groups.values():
        ref = next((w for w in rows if w["condition_name"] == "free"), rows[0])
        paper = ref["paper_x"].copy()
        state_action = ref["state_action_x"].copy()
        source = ref.get("robot_pose_proxy_source", "canonical_free")
        for w in rows:
            if not np.array_equal(w["paper_x"], paper) or not np.array_equal(w["state_action_x"], state_action):
                changed += 1
            w["paper_x"] = paper.copy()
            w["state_action_x"] = state_action.copy()
            w["robot_pose_proxy_source"] = source
    return changed


def filter_complete_condition_groups(windows, conditions):
    """Keep only split/seed/window_t groups with every requested condition present."""
    groups = {}
    for w in windows:
        key = (w["split_name"], int(w["visible_seed"]), int(w["window_t"]))
        groups.setdefault(key, []).append(w)
    keep = []
    required = set(conditions)
    for rows in groups.values():
        present = {r["condition_name"] for r in rows}
        if required.issubset(present):
            by_condition = {}
            for r in rows:
                by_condition.setdefault(r["condition_name"], r)
            keep.extend(by_condition[c] for c in conditions)
    return keep, len(windows) - len(keep)


def max_pair_diff(pair_stats):
    vals = []
    for payload in pair_stats.get("by_hidden_condition", {}).values():
        for field in ["paper_x_max_abs", "state_action_x_max_abs"]:
            v = payload.get(field, {}).get("max")
            if v is not None:
                vals.append(float(v))
    return max(vals) if vals else None


def arr(values, dtype=None):
    return np.asarray(values, dtype=dtype) if dtype is not None else np.asarray(values)


def write_canonicalization_report(report_md: Path, summary: dict) -> None:
    lines = [
        "# Phase3 Canonicalization Report",
        "",
        "Canonicalization is an audit control for the matched-input CCDA premise. It changes only model inputs within paired conditions and never changes future-state targets, action targets, success labels, condition labels, hidden-contact metadata, or recoverability parameters.",
        "",
        "## Summary",
        "",
        f"- Enabled: `{summary['canonicalization_enabled']}`",
        f"- Rule: `{summary['rule']}`",
        f"- Conditions: `{', '.join(summary['conditions'])}`",
        f"- Primary pair: `{summary['primary_branch_pair']}`",
        f"- Diagnostic pair: `{summary['diagnostic_branch_pair']}`",
        f"- Canonicalized windows: `{summary['num_canonicalized_windows']}`",
        f"- Raw pair count: `{summary['raw_pair_stats']['num_pairs']}`",
        f"- Post pair count: `{summary['post_pair_stats']['num_pairs']}`",
        "",
        "## Input Differences By Hidden Condition",
        "",
        "| Hidden condition | Raw paper max | Raw state_action max | Post paper max | Post state_action max |",
        "|---|---:|---:|---:|---:|",
    ]
    raw = summary["raw_pair_stats"].get("by_hidden_condition", {})
    post = summary["post_pair_stats"].get("by_hidden_condition", {})
    for cond in hidden_conditions(summary["conditions"]):
        r = raw.get(cond, {})
        q = post.get(cond, {})
        lines.append(
            "| `{}` | `{}` | `{}` | `{}` | `{}` |".format(
                cond,
                r.get("paper_x_max_abs", {}).get("max"),
                r.get("state_action_x_max_abs", {}).get("max"),
                q.get("paper_x_max_abs", {}).get("max"),
                q.get("state_action_x_max_abs", {}).get("max"),
            )
        )
    lines.append("")
    report_md.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/state_diff2")
    ap.add_argument("--train_data_root", required=True)
    ap.add_argument("--heldout_data_root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--th", type=int, default=3)
    ap.add_argument("--tf", type=int, default=4)
    ap.add_argument("--max_windows_per_episode", type=int, default=0, help="<=0 uses all possible windows; >0 caps windows per episode")
    ap.add_argument("--conditions", nargs="+", default=None)
    ap.add_argument("--primary_hidden_condition", default=None)
    ap.add_argument("--diagnostic_hidden_condition", default="hidden_pin")
    ap.add_argument("--no_canonicalize_paired_inputs", action="store_true")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    conditions = normalize_conditions(args.conditions or os.environ.get("PHASE3_CONDITIONS", "").split() or None)
    primary_hidden = args.primary_hidden_condition or os.environ.get("PHASE3_PRIMARY_HIDDEN_CONDITION", "hidden_breakaway_pin")
    diagnostic_hidden = args.diagnostic_hidden_condition or os.environ.get("PHASE3_DIAGNOSTIC_HIDDEN_CONDITION", "hidden_pin")
    if primary_hidden not in conditions:
        raise SystemExit(f"[Phase3][FAIL] primary_hidden_condition={primary_hidden} not in conditions={conditions}")
    if diagnostic_hidden not in conditions:
        raise SystemExit(f"[Phase3][FAIL] diagnostic_hidden_condition={diagnostic_hidden} not in conditions={conditions}")

    out = Path(args.out)
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)

    windows = []
    source_counts = Counter()
    train_windows, codec, train_sources = build_windows_from_dataset(
        "train",
        Path(args.train_data_root),
        args.th,
        args.tf,
        codec=None,
        max_windows_per_episode=args.max_windows_per_episode,
        conditions=conditions,
    )
    heldout_windows, codec, heldout_sources = build_windows_from_dataset(
        "heldout",
        Path(args.heldout_data_root),
        args.th,
        args.tf,
        codec=codec,
        max_windows_per_episode=args.max_windows_per_episode,
        conditions=conditions,
    )
    windows.extend(train_windows)
    windows.extend(heldout_windows)
    source_counts.update(train_sources)
    source_counts.update(heldout_sources)

    raw_num_windows = len(windows)
    windows, dropped_incomplete_condition_groups = filter_complete_condition_groups(windows, conditions)
    raw_pair_stats = paired_input_diff_stats(windows, conditions)

    canonicalized_inputs = 0
    if not args.no_canonicalize_paired_inputs:
        canonicalized_inputs = canonicalize_paired_inputs(windows)
    post_pair_stats = paired_input_diff_stats(windows, conditions)
    post_max = max_pair_diff(post_pair_stats)
    if post_max is None or post_max > 1e-8:
        raise SystemExit(f"[Phase3][FAIL] paired inputs still differ after canonicalization: post_max={post_max}")

    if not windows:
        raise SystemExit("[Phase3][FAIL] no windows built")
    if codec is None:
        raise SystemExit("[Phase3][FAIL] action codec was not initialized")
    if not isinstance(codec, ExecutableActionCodec):
        raise SystemExit(f"[Phase3][FAIL] expected ExecutableActionCodec, got {type(codec).__name__}")
    if codec.dim() != 14:
        raise SystemExit(f"[Phase3][FAIL] expected executable action dim 14, got {codec.dim()}: {codec.summary()}")

    train_visible_seed_set = {int(w["visible_seed"]) for w in windows if w["split_name"] == "train"}
    heldout_visible_seed_set = {int(w["visible_seed"]) for w in windows if w["split_name"] == "heldout"}
    train_heldout_seed_overlap = sorted(train_visible_seed_set & heldout_visible_seed_set)
    if train_heldout_seed_overlap:
        raise SystemExit(f"[Phase3][FAIL] train/heldout visible_seed overlap: {train_heldout_seed_overlap[:20]}")

    feature_schema = {
        "paper_x": ["bead_xy_history", "bead_velocity_history", "robot_pose_proxy_history"],
        "state_action_x": ["paper_x", "past_action_history_excluding_current_action"],
        "y_state": ["future_state_trajectory"],
        "y_action": ["current_executable_pick_place_action"],
        "forbidden_not_in_x": list(FORBIDDEN_METADATA_NOT_IN_X),
    }

    state_dim = int(windows[0]["y_final_state"].shape[0])
    action_dim = int(windows[0]["y_action"].shape[0])
    n_beads = int(windows[0]["n_beads"])
    for w in windows:
        if int(w["y_final_state"].shape[0]) != state_dim:
            raise SystemExit("[Phase3][FAIL] inconsistent state_dim")
        if int(w["y_action"].shape[0]) != action_dim:
            raise SystemExit("[Phase3][FAIL] inconsistent action_dim/action template")
        if int(w["n_beads"]) != n_beads:
            raise SystemExit("[Phase3][FAIL] inconsistent n_beads")

    template_path = out.parent / "phase3_action_template.pkl"
    save_action_template(template_path, codec)

    paper_x = arr([w["paper_x"] for w in windows], np.float32)
    state_action_x = arr([w["state_action_x"] for w in windows], np.float32)
    y_state = arr([w["y_state"] for w in windows], np.float32)
    y_final_state = arr([w["y_final_state"] for w in windows], np.float32)
    y_action = arr([w["y_action"] for w in windows], np.float32)
    condition_name = arr([w["condition_name"] for w in windows])
    split_name = arr([w["split_name"] for w in windows])
    episode_action_len = arr([w.get("episode_action_len", -1) for w in windows], np.int64)

    assert y_action.shape[1] == 14, y_action.shape
    assert state_action_x.shape[1] == paper_x.shape[1] + args.th * 14, (state_action_x.shape, paper_x.shape, args.th)
    extra = state_action_x[:, paper_x.shape[1] :]
    extra_std = float(np.std(extra))

    action_lens = [int(x) for x in episode_action_len.tolist() if int(x) >= 0]
    windows_per_condition = Counter(str(x) for x in condition_name.tolist())
    num_episodes_loaded = len({(w["split_name"], w["condition_name"], w["source_file"]) for w in windows})

    canonical_summary = {
        "canonicalization_enabled": not args.no_canonicalize_paired_inputs,
        "conditions": conditions,
        "primary_hidden_condition": primary_hidden,
        "diagnostic_hidden_condition": diagnostic_hidden,
        "primary_branch_pair": f"free_vs_{primary_hidden}",
        "diagnostic_branch_pair": f"free_vs_{diagnostic_hidden}",
        "num_canonicalized_windows": int(canonicalized_inputs),
        "rule": "group by split_name, visible_seed, window_t; copy free input to paired hidden branches; targets remain condition-specific",
        "raw_pair_stats": raw_pair_stats,
        "post_pair_stats": post_pair_stats,
        "post_max_pair_input_diff": post_max,
    }

    meta = {
        "conditions": conditions,
        "primary_hidden_condition": primary_hidden,
        "diagnostic_hidden_condition": diagnostic_hidden,
        "primary_branch_pair": f"free_vs_{primary_hidden}",
        "diagnostic_branch_pair": f"free_vs_{diagnostic_hidden}",
        "phase2_5_selected_recoverable_condition": primary_hidden,
        "phase2_5_selected_config": os.environ.get("PHASE2_5_SELECTED_CONFIG", "breakaway_force_2p6_disp_0p045_pull_0p36"),
        "ccda_breakaway_force": os.environ.get("CCDA_BREAKAWAY_FORCE", "2.6"),
        "ccda_breakaway_disp": os.environ.get("CCDA_BREAKAWAY_DISP", "0.045"),
        "ccda_breakaway_bead_ratio": os.environ.get("CCDA_BREAKAWAY_BEAD_RATIO", "0.45"),
        "ccda_oracle_breakaway_pull_dist": os.environ.get("CCDA_ORACLE_BREAKAWAY_PULL_DIST", "0.36"),
        "num_windows": int(len(windows)),
        "raw_num_windows_before_complete_condition_filter": int(raw_num_windows),
        "dropped_windows_incomplete_condition_groups": int(dropped_incomplete_condition_groups),
        "complete_condition_filter": "keep only split/visible_seed/window_t groups containing: " + ", ".join(conditions),
        "num_episodes_loaded": int(num_episodes_loaded),
        "num_train_windows": int(np.sum(split_name == "train")),
        "num_heldout_windows": int(np.sum(split_name == "heldout")),
        "train_visible_seeds": sorted(train_visible_seed_set),
        "heldout_visible_seeds": sorted(heldout_visible_seed_set),
        "train_visible_seed_count": int(len(train_visible_seed_set)),
        "heldout_visible_seed_count": int(len(heldout_visible_seed_set)),
        "train_heldout_seed_overlap": train_heldout_seed_overlap,
        "feature_schema": feature_schema,
        "forbidden_metadata_not_in_x": list(FORBIDDEN_METADATA_NOT_IN_X),
        "episode_action_len_min": int(min(action_lens)) if action_lens else None,
        "episode_action_len_mean": float(np.mean(action_lens)) if action_lens else None,
        "episode_action_len_max": int(max(action_lens)) if action_lens else None,
        "windows_per_condition": {k: int(v) for k, v in sorted(windows_per_condition.items())},
        "state_dim": state_dim,
        "robot_pose_dim": 39,
        "action_dim": int(action_dim),
        "n_beads": n_beads,
        "th": int(args.th),
        "tf": int(args.tf),
        "max_windows_per_episode": int(args.max_windows_per_episode),
        "paper_x_shape": list(paper_x.shape),
        "state_action_x_shape": list(state_action_x.shape),
        "y_action_shape": list(y_action.shape),
        "action_codec": codec.summary(),
        "action_codec_summary": codec.summary(),
        "action_template_path": str(template_path),
        "action_template_json_or_pickle_path": str(template_path),
        "state_action_extra_std": extra_std,
        "state_action_extra_all_zero": bool(extra_std < 1e-8),
        "robot_pose_proxy_source_counts": dict(source_counts),
        "canonicalized_paired_input_windows": int(canonicalized_inputs),
        "canonicalization_rule": canonical_summary["rule"],
        "canonicalization_summary_path": str(root / "reports/phase3_canonicalization_summary.json"),
    }

    np.savez_compressed(
        out,
        paper_x=paper_x,
        state_action_x=state_action_x,
        y_state=y_state,
        y_final_state=y_final_state,
        y_action=y_action,
        condition_id=arr([w["condition_id"] for w in windows], np.int64),
        condition_name=condition_name,
        visible_seed=arr([w["visible_seed"] for w in windows], np.int64),
        split_name=split_name,
        source_file=arr([w["source_file"] for w in windows]),
        window_t=arr([w["window_t"] for w in windows], np.int64),
        episode_action_len=episode_action_len,
        success=arr([w["success"] for w in windows], np.int64),
        final_fraction=arr([w["final_fraction"] for w in windows], np.float32),
        state_dim=np.asarray(state_dim, dtype=np.int64),
        robot_pose_dim=np.asarray(39, dtype=np.int64),
        action_dim=np.asarray(action_dim, dtype=np.int64),
        n_beads=np.asarray(n_beads, dtype=np.int64),
        th=np.asarray(args.th, dtype=np.int64),
        tf=np.asarray(args.tf, dtype=np.int64),
        action_template_json_or_pickle_path=np.asarray(str(template_path)),
        robot_pose_proxy_source=arr([w["robot_pose_proxy_source"] for w in windows]),
        meta_json=np.asarray(json.dumps(meta, sort_keys=True)),
    )

    splits = {
        "train_visible_seeds": meta["train_visible_seeds"],
        "heldout_visible_seeds": meta["heldout_visible_seeds"],
        "split_rule": "separate generated seed ranges; folds are grouped by visible_seed inside train only",
    }
    (out.parent / "phase3_splits.json").write_text(json.dumps(splits, indent=2, sort_keys=True))

    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "phase3_prepare_windows_summary.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
    (reports / "phase3_canonicalization_summary.json").write_text(json.dumps(canonical_summary, indent=2, sort_keys=True))
    write_canonicalization_report(reports / "phase3_canonicalization_report.md", canonical_summary)
    print(json.dumps(meta, indent=2, sort_keys=True))
    print("[Phase3] wrote", out)
    print("[Phase3] wrote", out.parent / "phase3_splits.json")
    print("[Phase3] wrote", reports / "phase3_canonicalization_summary.json")
    print("[Phase3] wrote", reports / "phase3_canonicalization_report.md")


if __name__ == "__main__":
    main()
