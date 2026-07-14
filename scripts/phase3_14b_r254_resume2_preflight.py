#!/usr/bin/env python3
"""Preflight for r2.5.4 Resume2 functional-prior attribution."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np
import torch

from ccda_phase3.phase314b_r23_diagnostics import require_repository_state, write_json_once
from ccda_phase3.phase314b_r254_resume2_functional_prior import (
    EXPECTED_BASELINE_COMMIT,
    EXPECTED_CACHE_SHA256,
    EXPECTED_CONTRACT_SHA256,
    EXPECTED_CURRENT_GPU,
    EXPECTED_R253_PILOT_SHA256,
    EXPECTED_SUBMODULE_COMMIT,
    IMMUTABLE_PATHS_AT_BASELINE,
    ORIGINAL_R254_BLOCKED_REPORT_PATH,
    ORIGINAL_R254_BLOCKED_SUMMARY_PATH,
    RESUME1_AUDIT_PATH,
    RESUME1_EVIDENCE_PATH,
    RESUME1_SUMMARY_PATH,
    RESUME2_BLOCKED_REPORT_PATH,
    RESUME2_BLOCKED_SUMMARY_PATH,
    RESUME2_PILOT_PATH,
    RESUME2_PREFLIGHT_PATH,
    R253_PILOT_PATH,
    assert_only_allowed_worktree_paths,
    assert_paths_match_commit,
    load_json,
    require_ancestor,
    sha256_file,
    source_sha256,
    verify_resume1_functional_contract,
)

EXPECTED_PYTHON_REQUEST = "/miniforge3/envs/coord_bimanual/bin/python"
EXPECTED_PYTHON_RESOLVED = "/miniforge3/envs/coord_bimanual/bin/python3.9"
EXPECTED_PYTHON_VERSION = "3.9.15"
EXPECTED_NUMPY_VERSION = "1.23.3"
EXPECTED_TORCH_VERSION = "1.12.1.post200"
EXPECTED_CUDA_RUNTIME = "11.2"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--output", default=RESUME2_PREFLIGHT_PATH)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    output = output.resolve()
    if output.exists():
        raise RuntimeError("refusing to overwrite Resume2 preflight")

    require_ancestor(root, EXPECTED_BASELINE_COMMIT)
    repository = require_repository_state(root, require_clean=False)
    if repository["submodule_commit"] != EXPECTED_SUBMODULE_COMMIT:
        raise RuntimeError("submodule commit mismatch")
    if repository["cache_sha256"] != EXPECTED_CACHE_SHA256:
        raise RuntimeError("cache SHA mismatch")
    if repository["frozen_contract_sha256"] != EXPECTED_CONTRACT_SHA256:
        raise RuntimeError("frozen contract SHA mismatch")

    allowed = (output.relative_to(root).as_posix(),)
    assert_only_allowed_worktree_paths(root, allowed)
    historical_hashes = assert_paths_match_commit(
        root, commit=EXPECTED_BASELINE_COMMIT, paths=IMMUTABLE_PATHS_AT_BASELINE
    )

    for forbidden in (
        RESUME2_PILOT_PATH,
        RESUME2_BLOCKED_SUMMARY_PATH,
        RESUME2_BLOCKED_REPORT_PATH,
        "reports/phase3_14b_r254_summary.json",
        "reports/phase3_14b_r254_report.md",
    ):
        if (root / forbidden).exists():
            raise RuntimeError(f"Resume2/final artifact already exists: {forbidden}")

    if not (root / ORIGINAL_R254_BLOCKED_SUMMARY_PATH).is_file():
        raise RuntimeError("original r2.5.4 blocked summary is missing")
    if not (root / ORIGINAL_R254_BLOCKED_REPORT_PATH).is_file():
        raise RuntimeError("original r2.5.4 blocked report is missing")

    if sha256_file(root / R253_PILOT_PATH) != EXPECTED_R253_PILOT_SHA256:
        raise RuntimeError("committed r2.5.3 pilot SHA mismatch")

    summary = load_json(root / RESUME1_SUMMARY_PATH)
    audit = load_json(root / RESUME1_AUDIT_PATH)
    evidence = load_json(root / RESUME1_EVIDENCE_PATH)
    functional_contract = verify_resume1_functional_contract(summary, audit, evidence)

    requested = os.path.realpath(EXPECTED_PYTHON_REQUEST)
    observed = os.path.realpath(sys.executable)
    expected_resolved = os.path.realpath(EXPECTED_PYTHON_RESOLVED)
    if requested != expected_resolved or observed != expected_resolved:
        raise RuntimeError(
            f"interpreter mismatch: requested={requested}, observed={observed}, "
            f"expected={expected_resolved}"
        )
    if sys.version.split()[0] != EXPECTED_PYTHON_VERSION:
        raise RuntimeError("Python version mismatch")
    if np.__version__ != EXPECTED_NUMPY_VERSION:
        raise RuntimeError("NumPy version mismatch")
    if torch.__version__ != EXPECTED_TORCH_VERSION:
        raise RuntimeError("PyTorch version mismatch")
    if str(torch.version.cuda) != EXPECTED_CUDA_RUNTIME:
        raise RuntimeError("PyTorch CUDA runtime mismatch")
    if not torch.cuda.is_available():
        raise RuntimeError("Resume2 attribution requires CUDA")
    gpu_name = torch.cuda.get_device_name(0)
    if gpu_name != EXPECTED_CURRENT_GPU:
        raise RuntimeError(f"Resume2 same-device contract requires {EXPECTED_CURRENT_GPU}; got {gpu_name}")
    if os.environ.get("PYTHONNOUSERSITE") != "1" or not bool(sys.flags.no_user_site):
        raise RuntimeError("PYTHONNOUSERSITE contract failed")

    payload: Dict[str, Any] = {
        "phase": "Phase3.14b-r2.5.4 Resume2",
        "verdict": "PASS",
        "meaning": "functional-prior gate verified before robot-proxy attribution",
        "repository": repository,
        "historical_path_sha256": historical_hashes,
        "r253_pilot_sha256": EXPECTED_R253_PILOT_SHA256,
        "functional_prior_contract": functional_contract,
        "environment": {
            "requested_python": EXPECTED_PYTHON_REQUEST,
            "resolved_python": observed,
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": gpu_name,
            "python_no_user_site": os.environ.get("PYTHONNOUSERSITE"),
            "sys_flags_no_user_site": bool(sys.flags.no_user_site),
        },
        "source_sha256": source_sha256(root),
        "robot_proxy_attribution_run": False,
        "reverse_sampling_rerun": False,
        "validation_targets_used": False,
        "formal_test_read": False,
        "formal_training": False,
        "idm": False,
        "candidate_execution": False,
        "phase4": False,
        "cps": False,
        "train_only_recommendation": None,
        "selected_configuration": None,
    }
    write_json_once(output, payload)
    print(json.dumps({"verdict": "PASS", "gpu_name": gpu_name}, sort_keys=True))


if __name__ == "__main__":
    main()
