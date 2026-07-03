#!/usr/bin/env python3
import sys
from pathlib import Path as _Phase3Path
ROOT = _Phase3Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np

from ccda_phase3.data_io import CONDITIONS, build_windows_from_dataset, save_action_template


def canonicalize_paired_inputs(windows):
    """Force paired CCDA branches to share exactly the same model input.

    DeformableRavens hidden contact is injected after reset and settling can
    introduce tiny visible/proprio numeric differences. Phase3 evaluates the
    contact-blind branch ambiguity, so all conditions for the same generated
    visible seed and prefix get the same canonical input, preferring the
    `free` branch. Targets remain condition-specific futures/actions.
    """
    groups = {}
    for w in windows:
        key = (w["split_name"], int(w["visible_seed"]), int(w["window_t"]))
        groups.setdefault(key, []).append(w)
    changed = 0
    for rows in groups.values():
        ref = None
        for w in rows:
            if w["condition_name"] == "free":
                ref = w
                break
        if ref is None:
            ref = rows[0]
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


def arr(values, dtype=None):
    return np.asarray(values, dtype=dtype) if dtype is not None else np.asarray(values)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/state_diff2")
    ap.add_argument("--train_data_root", required=True)
    ap.add_argument("--heldout_data_root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--th", type=int, default=3)
    ap.add_argument("--tf", type=int, default=4)
    ap.add_argument("--max_windows_per_episode", type=int, default=1, help="1 keeps the CCDA pre-branch prefix; <=0 uses all possible windows")
    ap.add_argument("--no_canonicalize_paired_inputs", action="store_true")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out = Path(args.out)
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)

    windows = []
    source_counts = Counter()
    train_windows, codec, train_sources = build_windows_from_dataset("train", Path(args.train_data_root), args.th, args.tf, codec=None, max_windows_per_episode=args.max_windows_per_episode)
    heldout_windows, codec, heldout_sources = build_windows_from_dataset("heldout", Path(args.heldout_data_root), args.th, args.tf, codec=codec, max_windows_per_episode=args.max_windows_per_episode)
    windows.extend(train_windows)
    windows.extend(heldout_windows)
    source_counts.update(train_sources)
    source_counts.update(heldout_sources)
    canonicalized_inputs = 0
    if not args.no_canonicalize_paired_inputs:
        canonicalized_inputs = canonicalize_paired_inputs(windows)

    if not windows:
        raise SystemExit("[Phase3][FAIL] no windows built")
    if codec is None:
        raise SystemExit("[Phase3][FAIL] action codec was not initialized")

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

    meta = {
        "conditions": CONDITIONS,
        "num_windows": len(windows),
        "num_train_windows": sum(1 for w in windows if w["split_name"] == "train"),
        "num_heldout_windows": sum(1 for w in windows if w["split_name"] == "heldout"),
        "train_visible_seeds": sorted({int(w["visible_seed"]) for w in windows if w["split_name"] == "train"}),
        "heldout_visible_seeds": sorted({int(w["visible_seed"]) for w in windows if w["split_name"] == "heldout"}),
        "state_dim": state_dim,
        "robot_pose_dim": 39,
        "action_dim": action_dim,
        "n_beads": n_beads,
        "th": args.th,
        "tf": args.tf,
        "max_windows_per_episode": args.max_windows_per_episode,
        "robot_pose_proxy_source_counts": dict(source_counts),
        "canonicalized_paired_input_windows": int(canonicalized_inputs),
        "canonicalization_rule": "group by split, visible_seed, window_t; copy free-branch paper_x and state_action_x to all conditions",
        "action_template_json_or_pickle_path": str(template_path),
    }

    np.savez_compressed(
        out,
        paper_x=arr([w["paper_x"] for w in windows], np.float32),
        state_action_x=arr([w["state_action_x"] for w in windows], np.float32),
        y_state=arr([w["y_state"] for w in windows], np.float32),
        y_final_state=arr([w["y_final_state"] for w in windows], np.float32),
        y_action=arr([w["y_action"] for w in windows], np.float32),
        condition_id=arr([w["condition_id"] for w in windows], np.int64),
        condition_name=arr([w["condition_name"] for w in windows]),
        visible_seed=arr([w["visible_seed"] for w in windows], np.int64),
        split_name=arr([w["split_name"] for w in windows]),
        source_file=arr([w["source_file"] for w in windows]),
        window_t=arr([w["window_t"] for w in windows], np.int64),
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
    print(json.dumps(meta, indent=2, sort_keys=True))
    print("[Phase3] wrote", out)
    print("[Phase3] wrote", out.parent / "phase3_splits.json")


if __name__ == "__main__":
    main()
