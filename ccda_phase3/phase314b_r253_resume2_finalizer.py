"""Resume2 provenance helpers for Phase3.14b-r2.5.3 finalizer-only recovery."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple

from ccda_phase3.phase314b_r253_resume1_finalizer import (
    PILOT_SUMMARY_RELATIVE,
    PILOT_SUMMARY_SHA256,
    assert_path_matches_commit,
    load_json,
)

PHASE = "Phase3.14b-r2.5.3-Resume2"
BASE_RESUME1_BLOCKED_COMMIT = "5a74197c13f4e675ee9ca63d8a6efa5d9c0b3944"
BASE_RESUME1_CORRECTION_COMMIT = "fbb953cbe0dce78d1b8bc0c0b897ed935c72f790"

RESUME2_PREFLIGHT_RELATIVE = (
    "reports/phase3_14b_r253_resume2_preflight_summary.json"
)
RESUME2_BLOCKED_SUMMARY_RELATIVE = (
    "reports/phase3_14b_r253_resume2_blocked_summary.json"
)
RESUME2_BLOCKED_REPORT_RELATIVE = (
    "reports/phase3_14b_r253_resume2_blocked_report.md"
)

RESUME1_BLOCKED_SUMMARY_RELATIVE = (
    "reports/phase3_14b_r253_resume_blocked_summary.json"
)
RESUME1_BLOCKED_REPORT_RELATIVE = (
    "reports/phase3_14b_r253_resume_blocked_report.md"
)

FINAL_SUMMARY_RELATIVE = "reports/phase3_14b_r253_summary.json"
FINAL_REPORT_RELATIVE = "reports/phase3_14b_r253_report.md"

RESUME2_PROVENANCE_SCHEMA = "phase314b_r253_finalizer_only_resume_v2"
PYTHONPATH_CONTRACT_SCHEMA = "phase314b_r253_repo_root_pythonpath_v1"

RESUME1_EVIDENCE_PATHS: Tuple[str, ...] = (
    "ccda_phase3/phase314b_r253_resume1_finalizer.py",
    "scripts/phase3_14b_r253_resume1_preflight.py",
    "scripts/phase3_14b_r253_resume1_finalize.py",
    "scripts/phase3_14b_r253_resume1_run.sh",
    "tests/test_phase314b_r253_resume1_finalizer.py",
    RESUME1_BLOCKED_SUMMARY_RELATIVE,
    RESUME1_BLOCKED_REPORT_RELATIVE,
)

RESUME2_SOURCE_PATHS: Tuple[str, ...] = (
    "ccda_phase3/phase314b_r253_resume2_finalizer.py",
    "scripts/phase3_14b_r253_resume2_preflight.py",
    "scripts/phase3_14b_r253_resume2_finalize.py",
    "scripts/phase3_14b_r253_resume2_run.sh",
    "tests/test_phase314b_r253_resume2_finalizer.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_sha256(root: Path, paths: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for relative in sorted(paths):
        path = root / relative
        if not path.is_file():
            raise RuntimeError(f"required source path is missing: {relative}")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def resume2_source_sha256(root: Path) -> str:
    return source_sha256(root, RESUME2_SOURCE_PATHS)


def resume1_evidence_sha256(root: Path) -> Dict[str, str]:
    return {
        relative: assert_path_matches_commit(
            root,
            BASE_RESUME1_BLOCKED_COMMIT,
            relative,
        )
        for relative in RESUME1_EVIDENCE_PATHS
    }


def validate_resume1_blocked_evidence(root: Path) -> Dict[str, Any]:
    summary_path = root / RESUME1_BLOCKED_SUMMARY_RELATIVE
    report_path = root / RESUME1_BLOCKED_REPORT_RELATIVE
    if not summary_path.is_file() or not report_path.is_file():
        raise RuntimeError("Resume1 blocked evidence is incomplete")

    summary = load_json(summary_path)
    required = {
        "phase": "Phase3.14b-r2.5.3-Resume1",
        "verdict": "BLOCKED",
        "root_cause": "phase314b_r253_resume1_finalization_failed",
        "exit_code": 1,
        "failed_line": "0",
        "finalizer_only": True,
        "gpu_pilot_rerun": False,
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
        raise RuntimeError(f"Resume1 blocked evidence mismatch: {mismatches}")

    return {
        "blocked_report_commit": BASE_RESUME1_BLOCKED_COMMIT,
        "correction_commit": BASE_RESUME1_CORRECTION_COMMIT,
        "summary": RESUME1_BLOCKED_SUMMARY_RELATIVE,
        "summary_sha256": sha256_file(summary_path),
        "report": RESUME1_BLOCKED_REPORT_RELATIVE,
        "report_sha256": sha256_file(report_path),
        "failure_class": "repository_root_missing_from_pythonpath",
        "pilot_summary_sha256": PILOT_SUMMARY_SHA256,
    }


def pythonpath_contract(root: Path, value: str | None = None) -> Dict[str, Any]:
    raw = os.environ.get("PYTHONPATH", "") if value is None else value
    entries = [item for item in raw.split(os.pathsep) if item]
    resolved = [str(Path(item).resolve()) for item in entries]
    expected = str(root.resolve())
    if not resolved:
        raise RuntimeError("PYTHONPATH is empty")
    if resolved[0] != expected:
        raise RuntimeError(
            "repository root must be the first PYTHONPATH entry: "
            f"expected={expected} observed={resolved[0]}"
        )
    return {
        "schema": PYTHONPATH_CONTRACT_SCHEMA,
        "expected_repository_root": expected,
        "raw": raw,
        "resolved_entries": resolved,
        "repository_root_is_first": True,
    }


def validate_no_final_artifacts(root: Path) -> None:
    unexpected = [
        relative
        for relative in (FINAL_SUMMARY_RELATIVE, FINAL_REPORT_RELATIVE)
        if (root / relative).exists()
    ]
    if unexpected:
        raise RuntimeError(f"final artifacts already exist: {unexpected}")


def validate_resume2_outputs_absent(root: Path) -> None:
    outputs = (
        RESUME2_PREFLIGHT_RELATIVE,
        RESUME2_BLOCKED_SUMMARY_RELATIVE,
        RESUME2_BLOCKED_REPORT_RELATIVE,
    )
    existing = [relative for relative in outputs if (root / relative).exists()]
    if existing:
        raise RuntimeError(f"Resume2 write-once output already exists: {existing}")


def write_text_once(path: Path, text: str) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def finite_json_roundtrip(value: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a JSON-safe deep copy while rejecting NaN and infinities."""
    encoded = json.dumps(value, allow_nan=False, sort_keys=True)
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):
        raise RuntimeError("JSON payload root changed type")
    return decoded
