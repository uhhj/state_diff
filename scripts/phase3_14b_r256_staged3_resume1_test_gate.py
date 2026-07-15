#!/usr/bin/env python3
"""Manifest-driven test gate for Stage D.3 Resume1."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ccda_phase3.phase314b_r256_staged3_resume1_upper_gate_freeze import (
    BLOCKED_IMPLEMENTATION_COMMIT,
    BLOCKED_SUMMARY,
    BLOCKED_TEST_GATE,
    EXPECTED_BLOCKED_SUMMARY_SHA256,
    EXPECTED_BLOCKED_TEST_GATE_SHA256,
    atomic_write_once,
    load_json,
    sha256_file,
    stable_json_bytes,
    validate_blocked_attempt,
)

NEW_TEST = (
    "tests/"
    "test_phase3_14b_r256_staged3_resume1_source_predicate.py"
)

CURRENT_ADDITIVE_TESTS = (
    "tests/test_phase3_14b_r255_robot_proxy_provenance.py",
    "tests/test_phase3_14b_r255_stageb_dataset.py",
    "tests/test_phase3_14b_r255_stagec_cache_attribution.py",
    "tests/test_phase3_14b_r256_stagea_contract.py",
    "tests/test_phase3_14b_r256_stageb_cable_diffusion.py",
    "tests/test_phase3_14b_r256_stagec_reverse_attribution.py",
    "tests/test_phase3_14b_r256_staged_segment_recalibration.py",
    "tests/test_phase3_14b_r256_staged1_asymmetric_gate_audit.py",
    "tests/test_phase3_14b_r256_staged2_collapse_provenance.py",
    "tests/test_phase3_14b_r256_staged3_upper_gate_freeze.py",
    NEW_TEST,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument(
        "--output",
        default=(
            "reports/"
            "phase3_14b_r256_staged3_resume1_test_gate_summary.json"
        ),
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output = (root / args.output).resolve()
    if output.exists():
        raise FileExistsError(
            f"refusing to overwrite Resume1 test gate: {output}"
        )

    completed = subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            BLOCKED_IMPLEMENTATION_COMMIT,
            "HEAD",
        ],
        cwd=str(root),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "blocked D.3 implementation is not an ancestor"
        )

    blocked_attempt = validate_blocked_attempt(root)
    previous_path = root / BLOCKED_TEST_GATE
    previous = load_json(previous_path)
    if sha256_file(previous_path) != EXPECTED_BLOCKED_TEST_GATE_SHA256:
        raise RuntimeError("blocked D.3 test-gate SHA changed")
    if sha256_file(root / BLOCKED_SUMMARY) != EXPECTED_BLOCKED_SUMMARY_SHA256:
        raise RuntimeError("blocked D.3 summary SHA changed")

    manifest = previous.get("test_manifest_sha256")
    if not isinstance(manifest, dict) or len(manifest) != 37:
        raise RuntimeError("blocked D.3 test manifest changed")
    for relative, expected in manifest.items():
        if sha256_file(root / relative) != expected:
            raise RuntimeError(f"frozen test changed: {relative}")

    compile_targets = [
        root
        / (
            "ccda_phase3/"
            "phase314b_r256_staged3_resume1_upper_gate_freeze.py"
        ),
        root
        / "scripts/phase3_14b_r256_staged3_resume1_worker.py",
        root
        / "scripts/phase3_14b_r256_staged3_resume1_run_freeze.py",
        root
        / "scripts/phase3_14b_r256_staged3_resume1_test_gate.py",
        root
        / "scripts/phase3_14b_r256_staged3_resume1_blocked.py",
    ]
    subprocess.run(
        [sys.executable, "-m", "py_compile"]
        + [str(path) for path in compile_targets],
        cwd=str(root),
        check=True,
    )

    completed_tests = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"]
        + list(CURRENT_ADDITIVE_TESTS),
        cwd=str(root),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(completed_tests.stdout, end="")
    if completed_tests.returncode != 0:
        raise RuntimeError("additive Resume1 tests failed")
    matches = re.findall(
        r"(\d+)\s+passed",
        completed_tests.stdout,
    )
    if not matches:
        raise RuntimeError("could not parse pytest pass count")
    additive_passed = int(matches[-1])
    historical_additive = (
        6 + 14 + 18 + 9 + 11 + 17 + 22 + 23 + 28 + 24
    )
    new_passed = additive_passed - historical_additive
    if new_passed != 6:
        raise RuntimeError(
            f"Resume1 new-test count changed: {new_passed}"
        )

    total_passed = 425 + additive_passed
    if total_passed != 603:
        raise RuntimeError(
            f"Resume1 total pass count changed: {total_passed}"
        )

    test_manifest: Dict[str, str] = dict(manifest)
    test_manifest[NEW_TEST] = sha256_file(root / NEW_TEST)
    report: Dict[str, Any] = {
        "phase": "Phase3.14b-r2.5.6 Stage D.3 Resume1",
        "schema":
            "phase314b_r256_staged3_resume1_test_gate_v1",
        "verdict": "PASS",
        "historical_detached_passed": 425,
        "historical_additive_passed": historical_additive,
        "staged3_resume1_new_passed": new_passed,
        "passed_test_count": total_passed,
        "test_file_count": 38,
        "test_manifest_sha256": test_manifest,
        "blocked_implementation_commit":
            BLOCKED_IMPLEMENTATION_COMMIT,
        "blocked_evidence_commit":
            blocked_attempt["blocked_evidence_commit"],
        "blocked_test_gate_path": BLOCKED_TEST_GATE,
        "blocked_test_gate_sha256":
            EXPECTED_BLOCKED_TEST_GATE_SHA256,
        "blocked_summary_path": BLOCKED_SUMMARY,
        "blocked_summary_sha256":
            EXPECTED_BLOCKED_SUMMARY_SHA256,
        "selection_uses_glob": False,
        "selection_uses_ignore": False,
        "selection_uses_k_expression": False,
        "selection_uses_deselection": False,
        "selection_temporarily_hides_files": False,
    }
    atomic_write_once(output, stable_json_bytes(report))
    print(
        json.dumps(
            {
                "verdict": "PASS",
                "passed": total_passed,
                "new_passed": new_passed,
                "blocked_evidence_commit":
                    blocked_attempt["blocked_evidence_commit"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
