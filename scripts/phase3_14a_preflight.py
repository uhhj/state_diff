#!/usr/bin/env python3
"""Preflight the immutable cache before deterministic training."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

from ccda_phase3.phase314a_contract import (
    CACHE_ROWS,
    CACHE_VERSION,
    load_npz_no_pickle,
    sha256_array,
    sha256_file,
    strict_json_dump,
    strict_json_load,
    validate_formal_windows,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument(
        "--skip-provenance-refresh",
        action="store_true",
        help=(
            "Use the already-refreshed provenance report. The runner uses "
            "this after refreshing provenance while the worktree is clean."
        ),
    )
    parser.add_argument(
        "--cache",
        default=(
            "data/phase3_14_cache/"
            "phase3_14a_training_cache.npz"
        ),
    )
    parser.add_argument(
        "--cache-manifest",
        default=(
            "data/phase3_14_cache/"
            "phase3_14a_training_cache_manifest.json"
        ),
    )
    parser.add_argument(
        "--summary",
        default="reports/phase3_14a_preflight_summary.json",
    )
    parser.add_argument(
        "--report",
        default="reports/phase3_14a_preflight_report.md",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not args.skip_provenance_refresh:
        subprocess.run(
            [
                sys.executable,
                "scripts/phase3_14_provenance_gate.py",
                "--root",
                str(root),
            ],
            cwd=str(root),
            check=True,
        )
    provenance = strict_json_load(
        root / "reports/phase3_14_provenance_gate_summary.json"
    )
    if provenance.get("verdict") != "PASS":
        raise RuntimeError("formal provenance gate is not PASS")

    cache_path = root / args.cache
    manifest_path = root / args.cache_manifest
    manifest = strict_json_load(manifest_path)
    if manifest.get("cache_version") != CACHE_VERSION:
        raise RuntimeError("unsupported training cache version")
    if manifest.get("training_cache_immutable") is not True:
        raise RuntimeError("training cache is not marked immutable")
    if manifest.get("test_split_selection_allowed") is not False:
        raise RuntimeError("test split selection must be forbidden")
    if sha256_file(cache_path) != manifest["cache_sha256"]:
        raise RuntimeError("training cache hash mismatch")

    arrays = load_npz_no_pickle(cache_path)
    validation = validate_formal_windows(arrays)
    if validation["rows"] != CACHE_ROWS:
        raise RuntimeError("cache row count changed")

    required_derived = {
        "future_valid_mask": (CACHE_ROWS, 4),
        "state_history_valid_mask": (CACHE_ROWS, 3),
        "action_history_valid_mask": (CACHE_ROWS, 3),
        "raw_action_count": (CACHE_ROWS,),
        "canonical_source_file": (CACHE_ROWS,),
        "pair_key": (CACHE_ROWS,),
        "current_state": (CACHE_ROWS, 87),
        "next_state": (CACHE_ROWS, 87),
        "idm_pair_x": (CACHE_ROWS, 174),
        "idm_trajectory_x": (CACHE_ROWS, 609),
        "direct_action_x": (CACHE_ROWS, 303),
    }
    for key, shape in required_derived.items():
        if key not in arrays:
            raise RuntimeError(f"cache is missing {key}")
        if arrays[key].shape != shape:
            raise RuntimeError(
                f"{key} shape {arrays[key].shape} != {shape}"
            )

    for key, expected_hash in manifest["array_sha256"].items():
        if key not in arrays:
            raise RuntimeError(f"manifest array missing from cache: {key}")
        if sha256_array(arrays[key]) != expected_hash:
            raise RuntimeError(f"array hash mismatch: {key}")

    future_mask = arrays["future_valid_mask"].astype(np.bool_)
    state_mask = arrays["state_history_valid_mask"].astype(np.bool_)
    action_mask = arrays["action_history_valid_mask"].astype(np.bool_)
    if not np.all(future_mask[:, 0]):
        raise RuntimeError("every row must have one valid future step")
    if np.any(np.diff(future_mask.astype(np.int8), axis=1) > 0):
        raise RuntimeError("future mask contains false→true transitions")
    if np.any(np.diff(state_mask.astype(np.int8), axis=1) < 0):
        raise RuntimeError("state history mask is not right aligned")
    if np.any(np.diff(action_mask.astype(np.int8), axis=1) < 0):
        raise RuntimeError("action history mask is not right aligned")

    pair_key = arrays["pair_key"].astype(str)
    condition = arrays["condition_name"].astype(str)
    pair_counts: Dict[str, set] = {}
    for key, name in zip(pair_key, condition):
        pair_counts.setdefault(key, set()).add(name)
    incomplete = {
        key: sorted(values)
        for key, values in pair_counts.items()
        if values != {
            "free",
            "hidden_slack_breakaway_pin_v2",
        }
    }
    if incomplete:
        raise RuntimeError(
            "paired cache keys do not contain both conditions: "
            f"{list(incomplete.items())[:10]}"
        )

    goals = (root / "GOALS.md").read_text(encoding="utf-8")
    if "4256 windows" not in goals:
        raise RuntimeError(
            "GOALS.md must report the regenerated 4256 windows"
        )
    if "Phase3.14a" not in goals:
        raise RuntimeError("GOALS.md must identify Phase3.14a")

    submodule_status = subprocess.check_output(
        [
            "git",
            "-C",
            "external/deformable-ravens",
            "status",
            "--short",
        ],
        cwd=str(root),
        text=True,
    ).strip()
    if submodule_status:
        raise RuntimeError("submodule worktree is not clean")

    payload: Dict[str, Any] = {
        "verdict": "PASS",
        "root_cause": "phase314a_training_cache_preflight_supported",
        "formal_provenance": provenance["root_cause"],
        "cache_sha256": manifest["cache_sha256"],
        "rows": CACHE_ROWS,
        "split_rows": manifest["split_rows"],
        "split_visible_seeds": manifest["split_visible_seeds"],
        "future_padding_fraction": manifest[
            "future_padding_fraction"
        ],
        "source_path_relink": manifest["source_path_relink"],
        "pair_keys": len(pair_counts),
        "state_quaternion": manifest["state_quaternion"],
        "action_quaternion": manifest["action_quaternion"],
        "deterministic_training_allowed": True,
        "ddpm_training": False,
        "idm_training": False,
        "candidate_execution": False,
        "phase4": False,
        "cps": False,
    }
    strict_json_dump(root / args.summary, payload)
    lines = [
        "# Phase3.14a Training-Cache Preflight",
        "",
        "- Verdict: `PASS`",
        (
            "- Root cause: "
            "`phase314a_training_cache_preflight_supported`"
        ),
        f"- Rows: `{CACHE_ROWS}`",
        f"- Cache SHA256: `{manifest['cache_sha256']}`",
        "- Deterministic training allowed: `True`",
        "- DDPM/IDM/candidate execution: `False`",
        "- Phase4/CPS: `False`",
    ]
    (root / args.report).write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
