#!/usr/bin/env python3
"""Preflight for r2.5.4 Resume3 in-memory prior-prediction reconstruction."""
from __future__ import annotations

import argparse
import inspect
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
from ccda_phase3.phase314b_r251_gradient_calibration import fit_shared_prior_snapshot
from ccda_phase3.phase314b_r254_resume2_functional_prior import (
    verify_resume1_functional_contract,
)
from ccda_phase3.phase314b_r254_resume3_prediction_adapter import (
    EXPECTED_BASELINE_COMMIT,
    EXPECTED_CACHE_SHA256,
    EXPECTED_CONTRACT_SHA256,
    EXPECTED_CURRENT_GPU,
    EXPECTED_R253_PILOT_SHA256,
    EXPECTED_SUBMODULE_COMMIT,
    FINAL_REPORT_PATH,
    FINAL_SUMMARY_PATH,
    IMMUTABLE_PATHS_AT_BASELINE,
    ORIGINAL_R254_STANDARD_PILOT_PATH,
    RESUME1_AUDIT_PATH,
    RESUME1_EVIDENCE_PATH,
    RESUME1_SUMMARY_PATH,
    RESUME2_BLOCKED_REPORT_PATH,
    RESUME2_BLOCKED_SUMMARY_PATH,
    RESUME2_PILOT_PATH,
    RESUME3_BLOCKED_REPORT_PATH,
    RESUME3_BLOCKED_SUMMARY_PATH,
    RESUME3_PILOT_PATH,
    RESUME3_PREFLIGHT_PATH,
    R253_PILOT_PATH,
    assert_only_allowed_worktree_paths,
    assert_paths_match_commit,
    load_json,
    require_ancestor,
    sha256_file,
    source_sha256,
    verify_resume2_blocked_contract,
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
    parser.add_argument("--output", default=RESUME3_PREFLIGHT_PATH)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    output = output.resolve()
    if output.exists():
        raise RuntimeError("refusing to overwrite Resume3 preflight")

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
        root,
        commit=EXPECTED_BASELINE_COMMIT,
        paths=IMMUTABLE_PATHS_AT_BASELINE,
    )

    forbidden = (
        RESUME3_PILOT_PATH,
        RESUME3_BLOCKED_SUMMARY_PATH,
        RESUME3_BLOCKED_REPORT_PATH,
        FINAL_SUMMARY_PATH,
        FINAL_REPORT_PATH,
        ORIGINAL_R254_STANDARD_PILOT_PATH,
        RESUME2_PILOT_PATH,
    )
    for relative in forbidden:
        if (root / relative).exists():
            raise RuntimeError(f"Resume3/final artifact already exists: {relative}")

    for required in (RESUME2_BLOCKED_SUMMARY_PATH, RESUME2_BLOCKED_REPORT_PATH):
        if not (root / required).is_file():
            raise RuntimeError(f"Resume2 blocked evidence missing: {required}")
    resume2_blocked = verify_resume2_blocked_contract(
        load_json(root / RESUME2_BLOCKED_SUMMARY_PATH)
    )

    if sha256_file(root / R253_PILOT_PATH) != EXPECTED_R253_PILOT_SHA256:
        raise RuntimeError("committed r2.5.3 pilot SHA mismatch")
    functional_contract = verify_resume1_functional_contract(
        load_json(root / RESUME1_SUMMARY_PATH),
        load_json(root / RESUME1_AUDIT_PATH),
        load_json(root / RESUME1_EVIDENCE_PATH),
    )

    fit_source = inspect.getsource(fit_shared_prior_snapshot)
    if '"_prior_state": state' not in fit_source:
        raise RuntimeError("historical fit no longer retains _prior_state")
    if "_prior_prediction_z" in fit_source:
        raise RuntimeError("historical fit API unexpectedly exposes prediction tensor")
    resume2_adapter_source = (
        root / "ccda_phase3/phase314b_r254_resume2_functional_prior.py"
    ).read_text(encoding="utf-8")
    if 'snapshot.get("_prior_prediction_z")' not in resume2_adapter_source:
        raise RuntimeError("Resume2 missing-field failure signature changed")

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
        raise RuntimeError("Resume3 attribution requires CUDA")
    gpu_name = torch.cuda.get_device_name(0)
    if gpu_name != EXPECTED_CURRENT_GPU:
        raise RuntimeError(f"Resume3 same-device contract requires {EXPECTED_CURRENT_GPU}")
    if os.environ.get("PYTHONNOUSERSITE") != "1" or not bool(sys.flags.no_user_site):
        raise RuntimeError("PYTHONNOUSERSITE contract failed")

    payload: Dict[str, Any] = {
        "phase": "Phase3.14b-r2.5.4 Resume3",
        "resume_generation": 3,
        "verdict": "PASS",
        "meaning": (
            "Resume2 missing prediction-field failure verified; pure-memory "
            "prediction reconstruction authorized"
        ),
        "repository": repository,
        "historical_path_sha256": historical_hashes,
        "r253_pilot_sha256": EXPECTED_R253_PILOT_SHA256,
        "functional_prior_contract": functional_contract,
        "resume2_blocked_contract": resume2_blocked,
        "snapshot_api_contract": {
            "retained_prior_state": True,
            "historical_prediction_tensor_field": False,
            "resume2_faulty_field": "_prior_prediction_z",
            "resume3_repairs_snapshot_schema": False,
            "resume3_prediction_reconstruction_in_memory": True,
        },
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
        "residual_training_run": False,
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
