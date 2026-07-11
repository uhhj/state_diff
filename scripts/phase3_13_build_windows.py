#!/usr/bin/env python3
"""Build pickle-free formal state-v2 windows from raw paired episodes."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from ccda_phase3.data_io import build_windows_from_dataset, save_action_template
from ccda_phase3.schema_v2 import ACTION_DIM, DEFAULT_TF, DEFAULT_TH, FORMAL_CONDITIONS, FORMAL_TASK_NAME, PAPER_X_DIM, SCHEMA_VERSION, ENVIRONMENT_VERSION, STATE_ACTION_X_DIM, STATE_DIM


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_dump(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def strings(values: List[str]) -> np.ndarray:
    width = max(1, *(len(str(value)) for value in values))
    return np.asarray([str(value) for value in values], dtype=f"<U{width}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/state_diff2")
    ap.add_argument("--raw-root", default="data/phase3_state_v2_slack/raw")
    ap.add_argument("--output-dir", default="data/phase3_state_v2_slack/windows")
    ap.add_argument("--manifest", default="data/phase3_state_v2_slack/manifest.json")
    ap.add_argument("--max-windows-per-episode", type=int, default=0)
    args = ap.parse_args()
    root = Path(args.root).resolve()
    raw_root = root / args.raw_root
    output = root / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    codec = None
    sources = {}
    for split in ("train", "val", "test"):
        split_rows, codec, split_sources = build_windows_from_dataset(
            split,
            raw_root / split,
            DEFAULT_TH,
            DEFAULT_TF,
            codec=codec,
            max_windows_per_episode=args.max_windows_per_episode,
            conditions=FORMAL_CONDITIONS,
        )
        if not split_rows:
            raise SystemExit(f"no windows for split {split}")
        rows.extend(split_rows)
        sources[split] = dict(split_sources)
    if codec is None:
        raise SystemExit("no action codec")
    template = output / "action_template.pkl"
    save_action_template(template, codec)
    arrays = {
        "paper_x": np.stack([row["paper_x"] for row in rows]).astype(np.float32),
        "state_action_x": np.stack([row["state_action_x"] for row in rows]).astype(np.float32),
        "y_state": np.stack([row["y_state"] for row in rows]).astype(np.float32),
        "y_final_state": np.stack([row["y_final_state"] for row in rows]).astype(np.float32),
        "y_action": np.stack([row["y_action"] for row in rows]).astype(np.float32),
        "condition_name": strings([row["condition_name"] for row in rows]),
        "visible_seed": np.asarray([row["visible_seed"] for row in rows], dtype=np.int64),
        "split_name": strings([row["split_name"] for row in rows]),
        "source_file": strings([row["source_file"] for row in rows]),
        "pair_group": strings([row["pair_group"] for row in rows]),
        "window_t": np.asarray([row["window_t"] for row in rows], dtype=np.int64),
        "success": np.asarray([row["success"] for row in rows], dtype=np.bool_),
        "final_fraction": np.asarray([row["final_fraction"] for row in rows], dtype=np.float32),
        "engagement_step": np.asarray([row["engagement_step"] for row in rows], dtype=np.int64),
        "release_step": np.asarray([row["release_step"] for row in rows], dtype=np.int64),
        "pre_engagement": np.asarray([row["pre_engagement"] for row in rows], dtype=np.bool_),
    }
    expected = {
        "paper_x": (PAPER_X_DIM,), "state_action_x": (STATE_ACTION_X_DIM,),
        "y_state": (DEFAULT_TF, STATE_DIM), "y_final_state": (STATE_DIM,), "y_action": (ACTION_DIM,),
    }
    for name, tail in expected.items():
        if arrays[name].shape[1:] != tail or arrays[name].dtype != np.float32 or not np.all(np.isfinite(arrays[name])):
            raise RuntimeError(f"invalid formal array {name}: {arrays[name].shape} {arrays[name].dtype}")
    if any(value.dtype.kind == "O" for value in arrays.values()):
        raise RuntimeError("object array detected")
    npz = output / "phase3_13_windows.npz"
    np.savez_compressed(npz, **arrays)
    with np.load(npz, allow_pickle=False) as checked:
        if set(checked.files) != set(arrays):
            raise RuntimeError("saved NPZ key mismatch")
    split_counts = {split: int(np.sum(arrays["split_name"] == split)) for split in ("train", "val", "test")}
    seed_ranges = {split: sorted(set(arrays["visible_seed"][arrays["split_name"] == split].tolist())) for split in split_counts}
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "environment_version": ENVIRONMENT_VERSION,
        "task": FORMAL_TASK_NAME,
        "conditions": list(FORMAL_CONDITIONS),
        "dimensions": {"state": STATE_DIM, "paper_x": PAPER_X_DIM, "state_action_x": STATE_ACTION_X_DIM, "future": [DEFAULT_TF, STATE_DIM], "action": ACTION_DIM},
        "split_visible_seeds": seed_ranges,
        "split_window_counts": split_counts,
        "main_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "submodule_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root / "external/deformable-ravens", text=True).strip(),
        "window_npz": str(npz), "window_npz_sha256": sha(npz),
        "action_template": str(template), "action_template_sha256": sha(template),
        "source_code_sha256": {str(path.relative_to(root)): sha(path) for path in [root / "ccda_phase3/schema_v2.py", root / "ccda_phase3/data_io.py", Path(__file__).resolve()]},
        "robot_proxy_sources": sources,
        "input_canonicalization": False,
        "contains_simulator_bead_velocity": False,
        "object_dtype_count": 0,
    }
    strict_dump(root / args.manifest, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
