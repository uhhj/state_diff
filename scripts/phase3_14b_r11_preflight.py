#!/usr/bin/env python3
"""Preflight Phase3.14b-r1.1 exact scheduler equivalence repair."""
from __future__ import annotations

import argparse
import json
import subprocess
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
    SELECTED_CHECKPOINT_SHA256,
)
from ccda_phase3.phase314b_r11_exact_scheduler import (
    LOCKED_PHASE314B_PATHS,
    exact_step_equivalence,
    scheduler_source_contract,
    summarize_equivalence,
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
        default="reports/phase3_14b_r11_preflight_summary.json",
    )
    parser.add_argument(
        "--report",
        default="reports/phase3_14b_r11_preflight_report.md",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    current = strict_json_load(
        root / "reports/phase3_14b_r1_summary.json"
    )
    if current.get("verdict") != "FAIL":
        raise RuntimeError("Phase3.14b-r1 must currently be FAIL")
    if (
        current.get("root_cause")
        != "phase314b_r1_scheduler_implementation_failed"
    ):
        raise RuntimeError("unexpected Phase3.14b-r1 root cause")

    arrays, cache_manifest = load_locked_cache(
        root / args.cache,
        root / args.cache_manifest,
    )
    if cache_manifest["cache_sha256"] != CACHE_SHA256:
        raise RuntimeError("cache manifest SHA256 changed")

    selected = strict_json_load(
        root / "reports/phase3_14b_selected_model.json"
    )
    checkpoint_path = root / str(selected["checkpoint"])
    actual_checkpoint_hash = sha256_file(checkpoint_path)
    if actual_checkpoint_hash != SELECTED_CHECKPOINT_SHA256:
        raise RuntimeError("selected checkpoint SHA256 changed")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("cache_sha256") != CACHE_SHA256:
        raise RuntimeError("checkpoint used another cache")

    bound_hashes = {}
    for relative in LOCKED_PHASE314B_PATHS:
        expected = checkpoint["source_sha256"].get(relative)
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

    submodule_root = root / "external/deformable-ravens"
    submodule_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=str(submodule_root),
        text=True,
    ).strip()
    submodule_status = subprocess.check_output(
        ["git", "status", "--short"],
        cwd=str(submodule_root),
        text=True,
    ).strip()
    if submodule_head != (
        "633a88752445cf5d6776ed374fdbbdb35f93050c"
    ):
        raise RuntimeError("submodule commit changed")
    if submodule_status:
        raise RuntimeError("submodule worktree is not clean")

    source_contract = scheduler_source_contract()
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    smoke = {}
    for device in devices:
        rows = exact_step_equivalence(
            device=device,
            shape=(2, 4, 87),
            timesteps=(0, 50, 99),
            seed=971000,
        )
        smoke[str(device)] = summarize_equivalence(rows)
        if not smoke[str(device)]["all_exact_prev_close"]:
            raise RuntimeError(
                f"exact scheduler smoke failed on {device}"
            )
        if not smoke[str(device)]["all_pred_x0_close"]:
            raise RuntimeError(
                f"exact pred-x0 smoke failed on {device}"
            )

    payload = {
        "verdict": "PASS",
        "root_cause": "phase314b_r11_preflight_supported",
        "cache_sha256": CACHE_SHA256,
        "checkpoint_sha256": actual_checkpoint_hash,
        "scheduler_source_contract": source_contract,
        "exact_equivalence_smoke": smoke,
        "checkpoint_bound_source_sha256": bound_hashes,
        "submodule_commit": submodule_head,
        "submodule_clean": True,
        "cuda_available": bool(torch.cuda.is_available()),
        "training": False,
        "idm": False,
        "candidate_execution": False,
        "phase4": False,
        "cps": False,
    }
    strict_json_dump(root / args.summary, payload)

    lines = [
        "# Phase3.14b-r1.1 Preflight",
        "",
        "- Verdict: `PASS`",
        "- Root cause: `phase314b_r11_preflight_supported`",
        f"- Cache SHA256: `{CACHE_SHA256}`",
        f"- Checkpoint SHA256: `{actual_checkpoint_hash}`",
        (
            "- Diffusers: "
            f"`{source_contract['diffusers_version']}`"
        ),
        "- Checkpoint-bound sources unchanged: `True`",
        "- Exact scheduler smoke: `PASS`",
        "- No retraining, IDM, execution, Phase4, or CPS was run.",
    ]
    (root / args.report).write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
