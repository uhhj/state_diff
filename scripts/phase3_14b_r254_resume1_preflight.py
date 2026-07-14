#!/usr/bin/env python3
"""Preflight for the r2.5.4 shared-prior determinism audit."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Dict, Mapping

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np
import torch

from ccda_phase3.phase314b_r23_diagnostics import (
    load_verified_inputs,
    require_repository_state,
    select_balanced_paired_rows,
    write_json_once,
)
from ccda_phase3.phase314b_r251_gradient_calibration import (
    assert_canonical_paired_row_contract,
)
from ccda_phase3.phase314b_r254_resume1_prior_determinism import (
    BASE_BLOCKED_REPORT_COMMIT,
    BASE_IMPLEMENTATION_COMMIT,
    BASE_R253_REPORT_COMMIT,
    COMMON_PAIRED_PRIOR_SEED,
    EXPECTED_CACHE_SHA256,
    EXPECTED_CONTRACT_SHA256,
    EXPECTED_HISTORICAL_PRIOR_SHA256,
    EXPECTED_PAIRED_ROWS,
    EXPECTED_PYTHON,
    EXPECTED_R253_PILOT_SHA256,
    EXPECTED_SUBMODULE_COMMIT,
    IMMUTABLE_R254_PATHS,
    PHASE,
    PriorEquivalenceSpec,
    assert_only_allowed_worktree_paths,
    assert_paths_match_commit,
    git_output,
    sha256_file,
    source_sha256,
)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be object: {path}")
    return value


def require_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{name} must be a mapping")
    return value


def interpreter_report() -> Dict[str, Any]:
    expected = os.path.realpath(EXPECTED_PYTHON)
    observed = os.path.realpath(sys.executable)
    if observed != expected:
        raise RuntimeError(
            f"interpreter mismatch: expected={expected}, observed={observed}"
        )
    if sys.version_info[:3] != (3, 9, 15):
        raise RuntimeError(f"Python version mismatch: {sys.version}")
    if np.__version__ != "1.23.3":
        raise RuntimeError(f"NumPy version mismatch: {np.__version__}")
    if torch.__version__ != "1.12.1.post200":
        raise RuntimeError(f"PyTorch version mismatch: {torch.__version__}")
    if str(torch.version.cuda) != "11.2":
        raise RuntimeError(f"CUDA runtime mismatch: {torch.version.cuda}")
    if not torch.cuda.is_available():
        raise RuntimeError("shared-prior determinism audit requires CUDA")
    if os.environ.get("PYTHONNOUSERSITE") != "1" or not sys.flags.no_user_site:
        raise RuntimeError("user-site isolation contract failed")
    capability = torch.cuda.get_device_capability(0)
    return {
        "requested_python": EXPECTED_PYTHON,
        "resolved_python": observed,
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "cuda_runtime": str(torch.version.cuda),
        "cuda_available": True,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_capability": [int(capability[0]), int(capability[1])],
        "python_no_user_site": os.environ.get("PYTHONNOUSERSITE"),
        "sys_no_user_site": bool(sys.flags.no_user_site),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def historical_prior_contract(root: Path) -> Dict[str, Any]:
    pilot_path = root / "reports/phase3_14b_r253_pilot_summary.json"
    if sha256_file(pilot_path) != EXPECTED_R253_PILOT_SHA256:
        raise RuntimeError("r2.5.3 pilot SHA mismatch")
    pilot = load_json(pilot_path)
    summary = load_json(root / "reports/phase3_14b_r253_summary.json")
    if pilot.get("verdict") != "PASS" or summary.get("verdict") != "PASS":
        raise RuntimeError("r2.5.3 final evidence did not pass")
    if summary.get("root_cause") != (
        "phase314b_r253_robot_proxy_reconstruction_conflation_supported"
    ):
        raise RuntimeError("r2.5.3 root cause changed")
    shared = require_mapping(pilot.get("shared_prior"), name="r2.5.3 shared prior")
    if shared.get("prior_state_sha256") != EXPECTED_HISTORICAL_PRIOR_SHA256:
        raise RuntimeError("historical paired-prior SHA changed")
    if int(shared.get("seed", -1)) != COMMON_PAIRED_PRIOR_SEED:
        raise RuntimeError("historical paired-prior seed changed")
    prior_z_mse = float(shared.get("prior_z_mse", float("nan")))
    if not math.isfinite(prior_z_mse):
        raise RuntimeError("historical prior z-MSE is missing or non-finite")
    history = shared.get("prior_history")
    if not isinstance(history, list) or not history:
        raise RuntimeError("historical prior history is missing")
    for row in history:
        if not isinstance(row, Mapping):
            raise RuntimeError("historical prior history row is invalid")
        if not all(key in row for key in ("step", "loss", "gradient_norm")):
            raise RuntimeError("historical prior history schema changed")
        if not math.isfinite(float(row["loss"])) or not math.isfinite(
            float(row["gradient_norm"])
        ):
            raise RuntimeError("historical prior history contains non-finite values")
    return {
        "pilot_sha256": EXPECTED_R253_PILOT_SHA256,
        "gpu_name": pilot.get("gpu_name"),
        "device": pilot.get("device"),
        "prior_state_sha256": str(shared["prior_state_sha256"]),
        "prior_z_mse": prior_z_mse,
        "prior_history": history,
        "state_tensor_count": int(shared.get("state_tensor_count", -1)),
        "seed": int(shared["seed"]),
    }


def blocked_contract(root: Path) -> Dict[str, Any]:
    payload = load_json(root / "reports/phase3_14b_r254_blocked_summary.json")
    if payload.get("verdict") != "BLOCKED":
        raise RuntimeError("original r2.5.4 blocked verdict changed")
    if payload.get("root_cause") != "phase314b_r254_execution_failed":
        raise RuntimeError("original r2.5.4 blocked root cause changed")
    if payload.get("failed_stage") != "pilot":
        raise RuntimeError("original r2.5.4 did not block in pilot")
    if bool(payload.get("pilot_summary_exists")):
        raise RuntimeError("original r2.5.4 unexpectedly persisted a pilot")
    traceback = str(payload.get("traceback_tail", ""))
    if "shared paired-prior SHA did not reproduce" not in traceback:
        raise RuntimeError("original shared-prior mismatch signature changed")
    return {
        "verdict": payload["verdict"],
        "root_cause": payload["root_cause"],
        "failed_stage": payload["failed_stage"],
        "pilot_summary_exists": False,
        "mismatch_signature_verified": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument(
        "--output",
        default="reports/phase3_14b_r254_resume1_preflight_summary.json",
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    output = output.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite preflight: {output}")

    repository = require_repository_state(root, require_clean=False)
    assert_only_allowed_worktree_paths(root, ())
    if git_output(root, "branch", "--show-current") != "Experiment1":
        raise RuntimeError("r2.5.4 Resume1 requires Experiment1 branch")
    try:
        git_output(root, "merge-base", "--is-ancestor", BASE_BLOCKED_REPORT_COMMIT, "HEAD")
    except Exception as exc:
        raise RuntimeError("r2.5.4 blocked-report commit is not an ancestor") from exc
    if repository["submodule_commit"] != EXPECTED_SUBMODULE_COMMIT:
        raise RuntimeError("submodule commit mismatch")
    if repository["cache_sha256"] != EXPECTED_CACHE_SHA256:
        raise RuntimeError("cache SHA mismatch")
    if repository["frozen_contract_sha256"] != EXPECTED_CONTRACT_SHA256:
        raise RuntimeError("frozen contract SHA mismatch")
    immutable_hashes = assert_paths_match_commit(
        root,
        commit=BASE_BLOCKED_REPORT_COMMIT,
        paths=IMMUTABLE_R254_PATHS,
    )
    for path in (
        "reports/phase3_14b_r254_pilot_summary.json",
        "reports/phase3_14b_r254_summary.json",
        "reports/phase3_14b_r254_report.md",
    ):
        if (root / path).exists():
            raise RuntimeError(f"unexpected original r2.5.4 artifact exists: {path}")

    arrays, _, _, _, train, _, _, _ = load_verified_inputs(root)
    paired_rows = select_balanced_paired_rows(arrays, train, 16)
    paired_contract = assert_canonical_paired_row_contract(arrays, paired_rows)
    if paired_contract.get("observed_rows") != list(EXPECTED_PAIRED_ROWS):
        raise RuntimeError("paired-row identity changed")

    spec = PriorEquivalenceSpec()
    spec.validate()
    report = {
        "phase": PHASE,
        "verdict": "PASS",
        "base_commits": {
            "r254_blocked_report": BASE_BLOCKED_REPORT_COMMIT,
            "r254_implementation": BASE_IMPLEMENTATION_COMMIT,
            "r253_final_report": BASE_R253_REPORT_COMMIT,
        },
        "repository": repository,
        "environment": interpreter_report(),
        "source_sha256": source_sha256(root),
        "immutable_r254_sha256": immutable_hashes,
        "blocked_contract": blocked_contract(root),
        "historical_prior": historical_prior_contract(root),
        "dataset": {
            "paired_rows": paired_rows.tolist(),
            "paired_row_contract": paired_contract,
        },
        "audit_spec": asdict(spec),
        "repeat_run_ids": list(("repeat_a", "repeat_b", "repeat_c")),
        "training_changed": False,
        "robot_proxy_attribution_run": False,
        "reverse_sampling_rerun": False,
        "validation_targets_used": False,
        "formal_test_read": False,
        "formal_training": False,
        "checkpoint_saved": False,
        "idm": False,
        "candidate_execution": False,
        "phase4": False,
        "cps": False,
    }
    write_json_once(output, report)
    print(json.dumps({"verdict": "PASS", "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
