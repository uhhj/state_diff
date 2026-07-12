#!/usr/bin/env python3
"""Preflight Phase3.14b-r1 without modifying checkpoint-bound sources."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

from ccda_phase3.phase314a_contract import (
    sha256_file,
    strict_json_dump,
    strict_json_load,
)
from ccda_phase3.phase314b_contract import (
    CACHE_SHA256,
    load_locked_cache,
)
from ccda_phase3.phase314b_r1_diagnostics import (
    LOCKED_PHASE314B_PATHS,
    SELECTED_CHECKPOINT_SHA256,
    load_selected_model,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
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
        default="reports/phase3_14b_r1_preflight_summary.json",
    )
    parser.add_argument(
        "--report",
        default="reports/phase3_14b_r1_preflight_report.md",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()

    phase314b = strict_json_load(
        root / "reports/phase3_14b_summary.json"
    )
    if phase314b.get("verdict") != "FAIL":
        raise RuntimeError("Phase3.14b must be closed as FAIL")
    if (
        phase314b.get("root_cause")
        != "phase314b_candidate_physical_validity_failed"
    ):
        raise RuntimeError("unexpected Phase3.14b failure root cause")

    arrays, cache_manifest = load_locked_cache(
        root / args.cache,
        root / args.cache_manifest,
    )
    selected = strict_json_load(
        root / "reports/phase3_14b_selected_model.json"
    )
    checkpoint = root / str(selected["checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    actual_checkpoint_hash = sha256_file(checkpoint)
    if actual_checkpoint_hash != SELECTED_CHECKPOINT_SHA256:
        raise RuntimeError("selected checkpoint SHA256 changed")
    if selected["checkpoint_sha256"] != actual_checkpoint_hash:
        raise RuntimeError("selection/checkpoint hash mismatch")

    bound_hashes = {}
    checkpoint_payload = torch.load(checkpoint, map_location="cpu")
    for relative in LOCKED_PHASE314B_PATHS:
        expected = checkpoint_payload["source_sha256"].get(relative)
        if expected is None:
            raise RuntimeError(
                f"checkpoint does not bind required source: {relative}"
            )
        actual = sha256_file(root / relative)
        if actual != expected:
            raise RuntimeError(
                f"checkpoint-bound source changed: {relative}"
            )
        bound_hashes[relative] = actual

    submodule_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=str(root / "external/deformable-ravens"),
        text=True,
    ).strip()
    submodule_status = subprocess.check_output(
        ["git", "status", "--short"],
        cwd=str(root / "external/deformable-ravens"),
        text=True,
    ).strip()
    if submodule_head != (
        "633a88752445cf5d6776ed374fdbbdb35f93050c"
    ):
        raise RuntimeError("formal submodule commit changed")
    if submodule_status:
        raise RuntimeError("submodule worktree is not clean")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    ema_model, condition_z, ema_meta = load_selected_model(
        root=root,
        arrays=arrays,
        device=device,
        use_ema=True,
    )
    raw_model, _, raw_meta = load_selected_model(
        root=root,
        arrays=arrays,
        device=device,
        use_ema=False,
    )

    payload = {
        "verdict": "PASS",
        "root_cause": "phase314b_r1_preflight_supported",
        "cache_sha256": CACHE_SHA256,
        "checkpoint_sha256": actual_checkpoint_hash,
        "selected_family": selected["model_family"],
        "selected_input": selected["input_variant"],
        "selected_seed": selected["training_seed"],
        "checkpoint_bound_source_sha256": bound_hashes,
        "condition_shape": list(condition_z.shape),
        "ema_loaded": True,
        "raw_loaded": True,
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "submodule_commit": submodule_head,
        "submodule_clean": True,
        "training": False,
        "idm": False,
        "candidate_execution": False,
        "phase4": False,
        "cps": False,
    }
    strict_json_dump(root / args.summary, payload)

    lines = [
        "# Phase3.14b-r1 Preflight",
        "",
        "- Verdict: `PASS`",
        "- Root cause: `phase314b_r1_preflight_supported`",
        f"- Cache SHA256: `{CACHE_SHA256}`",
        f"- Checkpoint SHA256: `{actual_checkpoint_hash}`",
        f"- Device: `{device}`",
        "- Checkpoint-bound Phase3.14b sources unchanged: `True`",
        "- EMA and raw weights loaded: `True`",
        "- No training or candidate execution was run.",
        "- Phase4/CPS remain blocked.",
    ]
    (root / args.report).write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
