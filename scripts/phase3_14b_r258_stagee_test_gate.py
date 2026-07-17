#!/usr/bin/env python3
"""Temporal frozen-test gate for Phase3.14b-r2.5.8 Stage E."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ccda_phase3.phase314b_r258_staged_resume2_temporal_views import (
    BASE_EVIDENCE_COMMIT as STAGEC_RESUME1_EVIDENCE_COMMIT,
    CURRENT_RESUME1_TEST,
    CURRENT_RESUME2_TEST,
    CURRENT_STAGE_D_TEST,
    EXPECTED_BASE_PASSED,
    EXPECTED_BASE_REGULAR_FILES,
    EXPECTED_BASE_REGULAR_PASSED,
    EXPECTED_BASE_TEST_FILES,
    EXPECTED_HISTORICAL_TEST_PASSED,
    EXPECTED_RESUME1_PASSED,
    EXPECTED_RESUME2_PASSED,
    EXPECTED_STAGE_D_PASSED,
    HISTORICAL_CLOSED_WORLD_COMMIT,
    HISTORICAL_CLOSED_WORLD_TEST,
    create_temporal_clone,
    historical_test_blob_identity,
    inspect_historical_scope,
    parse_pytest_pass_count,
    split_base_manifest,
    validate_base_test_gate,
    validate_manifest_files,
)
from ccda_phase3.phase314b_r258_stagee_constrained_integrator import (
    BASE_EVIDENCE_COMMIT,
    BASE_TEST_GATE,
    STAGEE_IMPLEMENTATION_PATHS,
    atomic_write_once,
    load_json,
    sha256_bytes,
    sha256_file,
    stable_json_bytes,
    validate_base_evidence,
    validate_implementation_commit,
)

STAGEE_TEST = "tests/test_phase3_14b_r258_stagee_constrained_integrator.py"
OUTPUT = "reports/phase3_14b_r258_stagee_test_gate_summary.json"
EXPECTED_STAGEE_PASSED = 81
EXPECTED_TOTAL_FILES = 55
EXPECTED_TOTAL_PASSED = 1324


def run_pytest(
    *,
    python_bin: str,
    cwd: Path,
    paths: Sequence[Path],
    pycache_prefix: Path,
) -> Dict[str, Any]:
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONNOUSERSITE": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONPYCACHEPREFIX": str(pycache_prefix),
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
        }
    )
    completed = subprocess.run(
        [str(python_bin), "-m", "pytest", "-q", *[str(path) for path in paths]],
        cwd=str(cwd),
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(completed.stdout, end="")
    if completed.returncode != 0:
        raise RuntimeError(
            "pytest population failed with status {}".format(completed.returncode)
        )
    return {
        "passed": parse_pytest_pass_count(completed.stdout),
        "stdout_sha256": sha256_bytes(completed.stdout.encode("utf-8")),
        "return_code": int(completed.returncode),
        "path_count": len(paths),
        "paths": [path.relative_to(cwd).as_posix() for path in paths],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument(
        "--python-bin",
        default="/miniforge3/envs/coord_bimanual/bin/python",
    )
    parser.add_argument("--output", default=OUTPUT)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output = (
        Path(args.output).resolve()
        if Path(args.output).is_absolute()
        else (root / args.output).resolve()
    )
    if output.exists():
        raise FileExistsError("refusing to overwrite Stage-E test gate: {}".format(output))

    implementation = validate_implementation_commit(root)
    base_evidence = validate_base_evidence(root)
    current_base_gate = load_json(root / BASE_TEST_GATE)
    if current_base_gate.get("verdict") != "PASS":
        raise RuntimeError("Stage-D Resume2 base test gate is not PASS")
    if int(current_base_gate.get("test_file_count", -1)) != 54:
        raise RuntimeError("Stage-D Resume2 base file count changed")
    if int(current_base_gate.get("passed_test_count", -1)) != 1243:
        raise RuntimeError("Stage-D Resume2 base pass count changed")

    stagec_gate = validate_base_test_gate(root)
    stagec_manifest = {
        str(relative): str(expected)
        for relative, expected in stagec_gate["test_manifest_sha256"].items()
    }
    split = split_base_manifest(stagec_manifest)
    blob_identity = historical_test_blob_identity(root)

    base54_manifest = {
        str(relative): str(expected)
        for relative, expected in current_base_gate["test_manifest_sha256"].items()
    }
    expected_current = (CURRENT_STAGE_D_TEST, CURRENT_RESUME1_TEST, CURRENT_RESUME2_TEST)
    if set(base54_manifest) != set(stagec_manifest) | set(expected_current):
        raise RuntimeError("Stage-D Resume2 54-file manifest partition changed")

    compile_targets = [
        root / "ccda_phase3/phase314b_r258_stagee_constrained_integrator.py",
        root / "scripts/phase3_14b_r258_stagee_worker.py",
        root / "scripts/phase3_14b_r258_stagee_run_calibration.py",
        root / "scripts/phase3_14b_r258_stagee_test_gate.py",
        root / "scripts/phase3_14b_r258_stagee_blocked.py",
    ]
    subprocess.run(
        [str(args.python_bin), "-m", "py_compile", *[str(path) for path in compile_targets]],
        cwd=str(root),
        check=True,
    )
    subprocess.run(
        ["bash", "-n", str(root / "scripts/phase3_14b_r258_stagee_run.sh")],
        cwd=str(root),
        check=True,
    )

    temporary_root = Path(
        tempfile.mkdtemp(prefix="phase314b_r258_stagee_gate_", dir="/tmp")
    )
    stagec_clone = temporary_root / "stagec_resume1_clone"
    historical_clone = temporary_root / "historical_closed_world_clone"
    base54_clone = temporary_root / "staged_resume2_clone"
    try:
        stagec_clone_record = create_temporal_clone(
            source_root=root,
            destination=stagec_clone,
            commit=STAGEC_RESUME1_EVIDENCE_COMMIT,
        )
        historical_clone_record = create_temporal_clone(
            source_root=root,
            destination=historical_clone,
            commit=HISTORICAL_CLOSED_WORLD_COMMIT,
        )
        base54_clone_record = create_temporal_clone(
            source_root=root,
            destination=base54_clone,
            commit=BASE_EVIDENCE_COMMIT,
        )

        stagec_manifest_observed = validate_manifest_files(
            root=stagec_clone,
            manifest=stagec_manifest,
        )
        base54_manifest_observed = validate_manifest_files(
            root=base54_clone,
            manifest=base54_manifest,
        )
        historical_scope = inspect_historical_scope(
            python_bin=args.python_bin,
            clone_root=historical_clone,
        )

        regular_result = run_pytest(
            python_bin=args.python_bin,
            cwd=stagec_clone,
            paths=[stagec_clone / relative for relative in sorted(split["regular_manifest"])],
            pycache_prefix=temporary_root / "regular_pycache",
        )
        if regular_result["path_count"] != EXPECTED_BASE_REGULAR_FILES:
            raise RuntimeError("regular frozen test-file count changed")
        if regular_result["passed"] != EXPECTED_BASE_REGULAR_PASSED:
            raise RuntimeError("regular frozen pass count changed")

        historical_result = run_pytest(
            python_bin=args.python_bin,
            cwd=historical_clone,
            paths=[historical_clone / HISTORICAL_CLOSED_WORLD_TEST],
            pycache_prefix=temporary_root / "historical_pycache",
        )
        if historical_result["passed"] != EXPECTED_HISTORICAL_TEST_PASSED:
            raise RuntimeError("historical closed-world pass count changed")

        reconstructed_files = regular_result["path_count"] + historical_result["path_count"]
        reconstructed_passed = regular_result["passed"] + historical_result["passed"]
        if reconstructed_files != EXPECTED_BASE_TEST_FILES:
            raise RuntimeError("reconstructed Stage-C file count changed")
        if reconstructed_passed != EXPECTED_BASE_PASSED:
            raise RuntimeError("reconstructed Stage-C pass count changed")

        current_base_results = {}
        expected_counts = {
            CURRENT_STAGE_D_TEST: EXPECTED_STAGE_D_PASSED,
            CURRENT_RESUME1_TEST: EXPECTED_RESUME1_PASSED,
            CURRENT_RESUME2_TEST: EXPECTED_RESUME2_PASSED,
        }
        for index, relative in enumerate(expected_current):
            result = run_pytest(
                python_bin=args.python_bin,
                cwd=base54_clone,
                paths=[base54_clone / relative],
                pycache_prefix=temporary_root / "base_current_{}_pycache".format(index),
            )
            if result["passed"] != expected_counts[relative]:
                raise RuntimeError("base current test count changed: {}".format(relative))
            current_base_results[relative] = result

        stagee_result = run_pytest(
            python_bin=args.python_bin,
            cwd=root,
            paths=[root / STAGEE_TEST],
            pycache_prefix=temporary_root / "stagee_pycache",
        )
        if stagee_result["passed"] != EXPECTED_STAGEE_PASSED:
            raise RuntimeError("Stage-E test count changed")
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)

    total_passed = reconstructed_passed + sum(
        result["passed"] for result in current_base_results.values()
    ) + stagee_result["passed"]
    if total_passed != EXPECTED_TOTAL_PASSED:
        raise RuntimeError("Stage-E total pass count changed: {}".format(total_passed))

    combined_manifest = dict(base54_manifest)
    combined_manifest[STAGEE_TEST] = sha256_file(root / STAGEE_TEST)
    if len(combined_manifest) != EXPECTED_TOTAL_FILES:
        raise RuntimeError("Stage-E combined test manifest changed")

    report: Dict[str, Any] = {
        "phase": "Phase3.14b-r2.5.8 Stage E",
        "schema": "phase314b_r258_stagee_test_gate_v1",
        "verdict": "PASS",
        "base_evidence_commit": BASE_EVIDENCE_COMMIT,
        "implementation": implementation,
        "base_evidence_file_sha256": base_evidence["file_sha256"],
        "temporal_test_views": {
            "stagec_resume1_view": stagec_clone_record,
            "historical_closed_world_view": historical_clone_record,
            "stage_d_resume2_view": base54_clone_record,
            "historical_blob_identity": blob_identity,
            "historical_scope": historical_scope,
            "stagec_manifest_sha256": stagec_manifest_observed,
            "stage_d_resume2_manifest_sha256": base54_manifest_observed,
            "all_base_tests_executed": True,
            "test_implementation_modified": False,
            "test_source_copied_or_patched": False,
            "pytest_ignore_used": False,
            "pytest_k_used": False,
            "pytest_deselection_used": False,
            "current_test_files_renamed": False,
        },
        "frozen_regular_population": regular_result,
        "historical_closed_world_population": historical_result,
        "reconstructed_stagec_population": {
            "test_file_count": reconstructed_files,
            "passed": reconstructed_passed,
            "matches_stagec_evidence": True,
        },
        "stage_d_resume2_current_populations": current_base_results,
        "reconstructed_stage_d_resume2_population": {
            "test_file_count": 54,
            "passed": 1243,
            "matches_base_evidence": True,
        },
        "stagee_population": stagee_result,
        "test_file_count": EXPECTED_TOTAL_FILES,
        "passed_test_count": total_passed,
        "stagee_new_passed": EXPECTED_STAGEE_PASSED,
        "test_manifest_sha256": combined_manifest,
        "scientific_calibration_run": False,
        "selection_holdout_accessed": False,
        "frozen_probe_accessed": False,
    }
    atomic_write_once(output, stable_json_bytes(report))
    print(
        json.dumps(
            {
                "verdict": "PASS",
                "test_file_count": EXPECTED_TOTAL_FILES,
                "passed": total_passed,
                "stagee_new_passed": EXPECTED_STAGEE_PASSED,
                "output": str(output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
