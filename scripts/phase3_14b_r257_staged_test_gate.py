#!/usr/bin/env python3
"""Manifest-driven test gate for r2.5.7 Stage D."""
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

from ccda_phase3.phase314b_r257_staged_timestep_gate import (
    BASE_EVIDENCE_COMMIT,
    atomic_write_once,
    load_json,
    sha256_file,
    stable_json_bytes,
)

PREVIOUS_GATE = "reports/phase3_14b_r257_stagec_test_gate_summary.json"
NEW_TEST = "tests/test_phase3_14b_r257_staged_timestep_gate.py"

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
    "tests/test_phase3_14b_r256_staged3_resume1_source_predicate.py",
    "tests/test_phase3_14b_r257_stagea_upper_objective.py",
    "tests/test_phase3_14b_r257_stageb_mechanism_audit.py",
    "tests/test_phase3_14b_r257_stagec_topk_quadratic.py",
    NEW_TEST,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument(
        "--output",
        default="reports/phase3_14b_r257_staged_test_gate_summary.json",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output = (root / args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite Stage-D test gate: {output}")

    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", BASE_EVIDENCE_COMMIT, "HEAD"],
        cwd=str(root),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if completed.returncode != 0:
        raise RuntimeError("r2.5.7 Stage-C evidence is not an ancestor")

    previous_path = root / PREVIOUS_GATE
    previous = load_json(previous_path)
    if previous.get("verdict") != "PASS":
        raise RuntimeError("previous test gate is not PASS")
    if int(previous.get("passed_test_count", -1)) != 706:
        raise RuntimeError("previous pass count changed")
    if int(previous.get("test_file_count", -1)) != 41:
        raise RuntimeError("previous test-file count changed")

    committed = subprocess.check_output(
        ["git", "show", f"{BASE_EVIDENCE_COMMIT}:{PREVIOUS_GATE}"],
        cwd=str(root),
    )
    if committed != previous_path.read_bytes():
        raise RuntimeError("previous gate differs from committed Stage-C evidence")

    manifest = previous.get("test_manifest_sha256")
    if not isinstance(manifest, dict) or len(manifest) != 41:
        raise RuntimeError("previous test manifest changed")
    for relative, expected in manifest.items():
        if sha256_file(root / relative) != expected:
            raise RuntimeError(f"frozen test changed: {relative}")

    compile_targets = [
        root / "ccda_phase3/phase314b_r257_staged_timestep_gate.py",
        root / "scripts/phase3_14b_r257_staged_worker.py",
        root / "scripts/phase3_14b_r257_staged_run_calibration.py",
        root / "scripts/phase3_14b_r257_staged_test_gate.py",
        root / "scripts/phase3_14b_r257_staged_blocked.py",
    ]
    subprocess.run(
        [sys.executable, "-m", "py_compile"] + [str(path) for path in compile_targets],
        cwd=str(root),
        check=True,
    )

    completed_tests = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"] + list(CURRENT_ADDITIVE_TESTS),
        cwd=str(root),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(completed_tests.stdout, end="")
    if completed_tests.returncode != 0:
        raise RuntimeError("additive r2.5.7 Stage-D tests failed")
    matches = re.findall(r"(\d+)\s+passed", completed_tests.stdout)
    if not matches:
        raise RuntimeError("could not parse pytest pass count")
    additive_passed = int(matches[-1])
    historical_additive = 706 - 425
    new_passed = additive_passed - historical_additive
    if new_passed != 43:
        raise RuntimeError(f"r2.5.7 Stage-D new-test count changed: {new_passed}")

    total_passed = 425 + additive_passed
    if total_passed != 749:
        raise RuntimeError(f"r2.5.7 Stage-D total pass count changed: {total_passed}")

    test_manifest: Dict[str, str] = dict(manifest)
    test_manifest[NEW_TEST] = sha256_file(root / NEW_TEST)
    report: Dict[str, Any] = {
        "phase": "Phase3.14b-r2.5.7 Stage D",
        "schema": "phase314b_r257_staged_test_gate_v1",
        "verdict": "PASS",
        "historical_detached_passed": 425,
        "historical_additive_passed": historical_additive,
        "r257_staged_new_passed": new_passed,
        "passed_test_count": total_passed,
        "test_file_count": 42,
        "test_manifest_sha256": test_manifest,
        "previous_gate_path": PREVIOUS_GATE,
        "previous_gate_blob_bound_to_commit": BASE_EVIDENCE_COMMIT,
        "selection_uses_glob": False,
        "selection_uses_ignore": False,
        "selection_uses_k_expression": False,
        "selection_uses_deselection": False,
        "selection_temporarily_hides_files": False,
    }
    atomic_write_once(output, stable_json_bytes(report))
    print(
        json.dumps(
            {"verdict": "PASS", "passed": total_passed, "new_passed": new_passed},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
