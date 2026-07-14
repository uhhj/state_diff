"""Resume3 provenance and interpreter helpers for r2.5.3 finalization.

Resume2 corrected the repository import path but invoked the process-wide
``python`` command.  On the target host that resolved to ``/usr/bin/python``
without NumPy.  Resume3 binds every Python process to the immutable
``coord_bimanual`` interpreter and remains finalizer-only.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple

from ccda_phase3.phase314b_r253_resume1_finalizer import (
    PILOT_SUMMARY_RELATIVE,
    PILOT_SUMMARY_SHA256,
    assert_path_matches_commit,
    load_json,
)
from ccda_phase3.phase314b_r253_resume2_finalizer import (
    FINAL_REPORT_RELATIVE,
    FINAL_SUMMARY_RELATIVE,
    pythonpath_contract,
    source_sha256,
)

PHASE = "Phase3.14b-r2.5.3-Resume3"
BASE_RESUME2_BLOCKED_COMMIT = "de9f23cc79cb2468291922a3101c7979f57a0889"
BASE_RESUME2_IMPLEMENTATION_COMMIT = "b1a2d9c657bd88d9fdf54e046785a4d840a28d5a"
BASE_RESUME1_BLOCKED_COMMIT = "5a74197c13f4e675ee9ca63d8a6efa5d9c0b3944"

EXPECTED_PYTHON_EXECUTABLE = "/miniforge3/envs/coord_bimanual/bin/python"
EXPECTED_PYTHON_VERSION = "3.9.15"
EXPECTED_NUMPY_VERSION = "1.23.3"
EXPECTED_TORCH_VERSION = "1.12.1.post200"
EXPECTED_TORCH_CUDA_VERSION = "11.2"

RESUME3_PREFLIGHT_RELATIVE = (
    "reports/phase3_14b_r253_resume3_preflight_summary.json"
)
RESUME3_BLOCKED_SUMMARY_RELATIVE = (
    "reports/phase3_14b_r253_resume3_blocked_summary.json"
)
RESUME3_BLOCKED_REPORT_RELATIVE = (
    "reports/phase3_14b_r253_resume3_blocked_report.md"
)

RESUME2_BLOCKED_SUMMARY_RELATIVE = (
    "reports/phase3_14b_r253_resume2_blocked_summary.json"
)
RESUME2_BLOCKED_REPORT_RELATIVE = (
    "reports/phase3_14b_r253_resume2_blocked_report.md"
)

RESUME3_PROVENANCE_SCHEMA = "phase314b_r253_finalizer_only_resume_v3"
INTERPRETER_CONTRACT_SCHEMA = "phase314b_r253_coord_bimanual_python_v1"

RESUME2_EVIDENCE_PATHS: Tuple[str, ...] = (
    "ccda_phase3/phase314b_r253_resume2_finalizer.py",
    "scripts/phase3_14b_r253_resume2_preflight.py",
    "scripts/phase3_14b_r253_resume2_finalize.py",
    "scripts/phase3_14b_r253_resume2_run.sh",
    "tests/test_phase314b_r253_resume2_finalizer.py",
    RESUME2_BLOCKED_SUMMARY_RELATIVE,
    RESUME2_BLOCKED_REPORT_RELATIVE,
)

RESUME3_SOURCE_PATHS: Tuple[str, ...] = (
    "ccda_phase3/phase314b_r253_resume3_finalizer.py",
    "scripts/phase3_14b_r253_resume3_preflight.py",
    "scripts/phase3_14b_r253_resume3_finalize.py",
    "scripts/phase3_14b_r253_resume3_run.sh",
    "tests/test_phase314b_r253_resume3_finalizer.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resume3_source_sha256(root: Path) -> str:
    return source_sha256(root, RESUME3_SOURCE_PATHS)


def resume2_evidence_sha256(root: Path) -> Dict[str, str]:
    return {
        relative: assert_path_matches_commit(
            root,
            BASE_RESUME2_BLOCKED_COMMIT,
            relative,
        )
        for relative in RESUME2_EVIDENCE_PATHS
    }


def _require_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{path} must be a mapping")
    return value


def validate_resume2_blocked_evidence(root: Path) -> Dict[str, Any]:
    summary_path = root / RESUME2_BLOCKED_SUMMARY_RELATIVE
    report_path = root / RESUME2_BLOCKED_REPORT_RELATIVE
    if not summary_path.is_file() or not report_path.is_file():
        raise RuntimeError("Resume2 blocked evidence is incomplete")

    summary = load_json(summary_path)
    required = {
        "phase": "Phase3.14b-r2.5.3-Resume2",
        "verdict": "BLOCKED",
        "root_cause": "phase314b_r253_resume2_finalization_failed",
        "exit_code": 1,
        "failed_line": "0",
        "failed_command": (
            "python -c 'import ccda_phase3; from "
            "ccda_phase3.phase314b_r253_resume1_finalizer import "
            "corrected_pilot_view; print(\"repository import contract: PASS\")'"
        ),
        "finalizer_only": True,
        "gpu_pilot_rerun": False,
        "training_rerun": False,
        "reverse_rerun": False,
        "pythonpath_repository_root_first": True,
        "pilot_summary": PILOT_SUMMARY_RELATIVE,
        "pilot_summary_sha256": PILOT_SUMMARY_SHA256,
        "expected_pilot_summary_sha256": PILOT_SUMMARY_SHA256,
        "pilot_summary_unchanged": True,
        "preflight_report": None,
        "final_summary_sha256": None,
        "final_report_sha256": None,
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
    mismatches = {
        key: {"expected": expected, "observed": summary.get(key)}
        for key, expected in required.items()
        if summary.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"Resume2 blocked evidence mismatch: {mismatches}")

    pythonpath = summary.get("pythonpath")
    if not isinstance(pythonpath, str) or not pythonpath:
        raise RuntimeError("Resume2 blocked evidence is missing PYTHONPATH")

    return {
        "blocked_report_commit": BASE_RESUME2_BLOCKED_COMMIT,
        "implementation_commit": BASE_RESUME2_IMPLEMENTATION_COMMIT,
        "summary": RESUME2_BLOCKED_SUMMARY_RELATIVE,
        "summary_sha256": sha256_file(summary_path),
        "report": RESUME2_BLOCKED_REPORT_RELATIVE,
        "report_sha256": sha256_file(report_path),
        "failure_class": "default_python_missing_required_dependencies",
        "failed_command_used_bare_python": summary["failed_command"].startswith(
            "python "
        ),
        "preflight_executed": False,
        "pilot_summary_sha256": PILOT_SUMMARY_SHA256,
    }


def collect_interpreter_observation() -> Dict[str, Any]:
    """Collect the runtime observation without acquiring a GPU context."""

    import numpy
    import torch

    return {
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "numpy_version": str(numpy.__version__),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "python_no_user_site": os.environ.get("PYTHONNOUSERSITE"),
        "no_user_site_flag": bool(sys.flags.no_user_site),
    }


def validate_interpreter_observation(
    observation: Mapping[str, Any],
    *,
    expected_executable: str = EXPECTED_PYTHON_EXECUTABLE,
) -> Dict[str, Any]:
    expected = {
        "python_executable": str(Path(expected_executable).resolve()),
        "python_version": EXPECTED_PYTHON_VERSION,
        "numpy_version": EXPECTED_NUMPY_VERSION,
        "torch_version": EXPECTED_TORCH_VERSION,
        "torch_cuda_version": EXPECTED_TORCH_CUDA_VERSION,
        "cuda_available": False,
        "cuda_visible_devices": "",
        "python_no_user_site": "1",
        "no_user_site_flag": True,
    }
    mismatches = {
        key: {"expected": value, "observed": observation.get(key)}
        for key, value in expected.items()
        if observation.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"interpreter contract mismatch: {mismatches}")

    return {
        "schema": INTERPRETER_CONTRACT_SCHEMA,
        "expected": expected,
        "observed": dict(observation),
        "pass": True,
        "finalizer_only": True,
        "gpu_context_forbidden": True,
    }


def interpreter_contract() -> Dict[str, Any]:
    return validate_interpreter_observation(collect_interpreter_observation())


def validate_no_final_artifacts(root: Path) -> None:
    existing = [
        relative
        for relative in (FINAL_SUMMARY_RELATIVE, FINAL_REPORT_RELATIVE)
        if (root / relative).exists()
    ]
    if existing:
        raise RuntimeError(f"final artifacts already exist: {existing}")


def validate_resume3_outputs_absent(root: Path) -> None:
    outputs = (
        RESUME3_PREFLIGHT_RELATIVE,
        RESUME3_BLOCKED_SUMMARY_RELATIVE,
        RESUME3_BLOCKED_REPORT_RELATIVE,
    )
    existing = [relative for relative in outputs if (root / relative).exists()]
    if existing:
        raise RuntimeError(f"Resume3 write-once output already exists: {existing}")


def finite_json_roundtrip(value: Mapping[str, Any]) -> Dict[str, Any]:
    encoded = json.dumps(value, allow_nan=False, sort_keys=True)
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):
        raise RuntimeError("JSON payload root changed type")
    return decoded


def write_text_once(path: Path, text: str) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
