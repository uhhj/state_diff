#!/usr/bin/env python3
"""Temporal frozen-test gate for Stage-F Resume2."""
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
    EXPECTED_RESUME2_PASSED as EXPECTED_STAGED_RESUME2_PASSED,
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
from ccda_phase3.phase314b_r258_stagef_constraint_aware_surrogate import (
    BASE_EVIDENCE_COMMIT,
    BASE_TEST_GATE,
    load_json,
    sha256_bytes,
    sha256_file,
    validate_base_evidence,
)
from ccda_phase3.phase314b_r258_stagef_resume1_translation_fix import (
    ORIGINAL_STAGEF_TEST,
    RESUME1_TEST,
)
from ccda_phase3.phase314b_r258_stagef_resume2_porcelain_recovery import (
    RESUME2_TEST,
    RESUME2_TEST_GATE,
    atomic_write_once,
    stable_json_bytes,
    validate_failure_reports,
    validate_resume2_implementation_commit,
)

STAGED_RESUME2_EVIDENCE_COMMIT = "61be1c377eeda6da4c25e21399e160beb9a33249"
STAGEE_TEST = "tests/test_phase3_14b_r258_stagee_constrained_integrator.py"
EXPECTED_STAGEE_PASSED = 81
EXPECTED_STAGEF_PASSED = 84
EXPECTED_RESUME1_TRANSLATION_PASSED = 47
EXPECTED_STAGEF_RESUME2_PASSED = 48
EXPECTED_TOTAL_FILES = 58
EXPECTED_TOTAL_PASSED = 1503


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
    parser.add_argument("--output", default=RESUME2_TEST_GATE)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    output = (
        Path(args.output).resolve()
        if Path(args.output).is_absolute()
        else (root / args.output).resolve()
    )
    if output.exists():
        raise FileExistsError(
            "refusing to overwrite Stage-F Resume2 test gate: {}".format(output)
        )

    implementation = validate_resume2_implementation_commit(root)
    failure_evidence = validate_failure_reports(root)
    base_evidence = validate_base_evidence(root)
    current_base_gate = load_json(root / BASE_TEST_GATE)
    if current_base_gate.get("verdict") != "PASS":
        raise RuntimeError("Stage-E base test gate is not PASS")
    if int(current_base_gate.get("test_file_count", -1)) != 55:
        raise RuntimeError("Stage-E base file count changed")
    if int(current_base_gate.get("passed_test_count", -1)) != 1324:
        raise RuntimeError("Stage-E base pass count changed")

    stagec_gate = validate_base_test_gate(root)
    stagec_manifest = {
        str(relative): str(expected)
        for relative, expected in stagec_gate["test_manifest_sha256"].items()
    }
    split = split_base_manifest(stagec_manifest)
    blob_identity = historical_test_blob_identity(root)

    base55_manifest = {
        str(relative): str(expected)
        for relative, expected in current_base_gate["test_manifest_sha256"].items()
    }
    current_base_tests = (CURRENT_STAGE_D_TEST, CURRENT_RESUME1_TEST, CURRENT_RESUME2_TEST)
    expected_base55 = set(stagec_manifest) | set(current_base_tests) | {STAGEE_TEST}
    if set(base55_manifest) != expected_base55:
        raise RuntimeError("Stage-E 55-file manifest partition changed")

    compile_targets = [
        root / "ccda_phase3/phase314b_r258_stagef_constraint_aware_surrogate.py",
        root / "ccda_phase3/phase314b_r258_stagef_resume1_translation_fix.py",
        root / "ccda_phase3/phase314b_r258_stagef_resume2_porcelain_recovery.py",
        root / "scripts/phase3_14b_r258_stagef_resume1_worker.py",
        root / "scripts/phase3_14b_r258_stagef_resume1_run_calibration.py",
        root / "scripts/phase3_14b_r258_stagef_resume1_test_gate.py",
        root / "scripts/phase3_14b_r258_stagef_resume1_blocked.py",
        root / "scripts/phase3_14b_r258_stagef_resume4_worker.py",
        root / "scripts/phase3_14b_r258_stagef_resume4_run_calibration.py",
        root / "scripts/phase3_14b_r258_stagef_resume4_test_gate.py",
        root / "scripts/phase3_14b_r258_stagef_resume4_blocked.py",
    ]
    subprocess.run(
        [str(args.python_bin), "-m", "py_compile", *[str(path) for path in compile_targets]],
        cwd=str(root),
        check=True,
    )
    subprocess.run(
        ["bash", "-n", str(root / "scripts/phase3_14b_r258_stagef_resume4_run.sh")],
        cwd=str(root),
        check=True,
    )

    temporary_root = Path(
        tempfile.mkdtemp(prefix="phase314b_r258_stagef_resume2_gate_", dir="/tmp")
    )
    stagec_clone = temporary_root / "stagec_resume1_clone"
    historical_clone = temporary_root / "historical_closed_world_clone"
    base54_clone = temporary_root / "staged_resume2_clone"
    stagee_clone = temporary_root / "stagee_clone"
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
            commit=STAGED_RESUME2_EVIDENCE_COMMIT,
        )
        stagee_clone_record = create_temporal_clone(
            source_root=root,
            destination=stagee_clone,
            commit=BASE_EVIDENCE_COMMIT,
        )

        stagec_manifest_observed = validate_manifest_files(
            root=stagec_clone,
            manifest=stagec_manifest,
        )
        base55_manifest_observed = validate_manifest_files(
            root=stagee_clone,
            manifest=base55_manifest,
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

        base_current_results = {}
        expected_counts = {
            CURRENT_STAGE_D_TEST: EXPECTED_STAGE_D_PASSED,
            CURRENT_RESUME1_TEST: EXPECTED_RESUME1_PASSED,
            CURRENT_RESUME2_TEST: EXPECTED_STAGED_RESUME2_PASSED,
        }
        for index, relative in enumerate(current_base_tests):
            result = run_pytest(
                python_bin=args.python_bin,
                cwd=base54_clone,
                paths=[base54_clone / relative],
                pycache_prefix=temporary_root / "base_current_{}_pycache".format(index),
            )
            if result["passed"] != expected_counts[relative]:
                raise RuntimeError("base current test count changed: {}".format(relative))
            base_current_results[relative] = result

        stagee_result = run_pytest(
            python_bin=args.python_bin,
            cwd=stagee_clone,
            paths=[stagee_clone / STAGEE_TEST],
            pycache_prefix=temporary_root / "stagee_pycache",
        )
        if stagee_result["passed"] != EXPECTED_STAGEE_PASSED:
            raise RuntimeError("Stage-E test count changed")

        stagef_result = run_pytest(
            python_bin=args.python_bin,
            cwd=root,
            paths=[root / ORIGINAL_STAGEF_TEST],
            pycache_prefix=temporary_root / "stagef_pycache",
        )
        if stagef_result["passed"] != EXPECTED_STAGEF_PASSED:
            raise RuntimeError("corrected Stage-F test count changed")

        resume1_result = run_pytest(
            python_bin=args.python_bin,
            cwd=root,
            paths=[root / RESUME1_TEST],
            pycache_prefix=temporary_root / "resume1_pycache",
        )
        if resume1_result["passed"] != EXPECTED_RESUME1_TRANSLATION_PASSED:
            raise RuntimeError("Stage-F Resume1 test count changed")

        resume2_result = run_pytest(
            python_bin=args.python_bin,
            cwd=root,
            paths=[root / RESUME2_TEST],
            pycache_prefix=temporary_root / "resume2_pycache",
        )
        if resume2_result["passed"] != EXPECTED_STAGEF_RESUME2_PASSED:
            raise RuntimeError("Stage-F Resume2 test count changed")
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)

    total_passed = (
        reconstructed_passed
        + sum(result["passed"] for result in base_current_results.values())
        + stagee_result["passed"]
        + stagef_result["passed"]
        + resume1_result["passed"]
        + resume2_result["passed"]
    )
    if total_passed != EXPECTED_TOTAL_PASSED:
        raise RuntimeError("Stage-F Resume2 total pass count changed: {}".format(total_passed))
    combined_manifest = dict(base55_manifest)
    combined_manifest[ORIGINAL_STAGEF_TEST] = sha256_file(root / ORIGINAL_STAGEF_TEST)
    combined_manifest[RESUME1_TEST] = sha256_file(root / RESUME1_TEST)
    combined_manifest[RESUME2_TEST] = sha256_file(root / RESUME2_TEST)
    if len(combined_manifest) != EXPECTED_TOTAL_FILES:
        raise RuntimeError("Stage-F Resume2 combined test manifest changed")

    report = {
        "phase": "Phase3.14b-r2.5.8 Stage F Resume4",
        "schema": "phase314b_r258_stagef_resume2_test_gate_v1",
        "verdict": "PASS",
        "base_evidence_commit": BASE_EVIDENCE_COMMIT,
        "implementation": implementation,
        "blocked_provenance": {
            "sha256": failure_evidence["sha256"],
            "original_scientific_calibration_completed": False,
            "resume1_scientific_calibration_started": False,
        },
        "base_evidence": {
            "worker_sha256": base_evidence["worker"]["comparison"]["left_sha256"],
            "contract_file_sha256": sha256_file(
                root / "reports/phase3_14b_r258_stagee_contract.json"
            ),
            "selection_sha256": base_evidence["contract"]["selection_sha256"],
        },
        "temporal_test_views": {
            "stagec_resume1_view": stagec_clone_record,
            "historical_closed_world_view": historical_clone_record,
            "staged_resume2_view": base54_clone_record,
            "stagee_view": stagee_clone_record,
            "historical_test_blob_identity": blob_identity,
            "historical_scope": historical_scope,
            "stagec_manifest_sha256": stagec_manifest_observed,
            "base55_manifest_sha256": base55_manifest_observed,
            "all_frozen_tests_executed": True,
            "pytest_ignore_used": False,
            "pytest_k_used": False,
            "pytest_deselection_used": False,
            "test_sources_renamed": False,
        },
        "frozen_regular_population": regular_result,
        "historical_closed_world_population": historical_result,
        "reconstructed_stagec_population": {
            "test_file_count": reconstructed_files,
            "passed": reconstructed_passed,
        },
        "stage_d_resume_populations": base_current_results,
        "stagee_population": stagee_result,
        "corrected_stagef_population": stagef_result,
        "resume1_population": resume1_result,
        "resume2_population": resume2_result,
        "porcelain_recovery": {
            "format": "v1-z",
            "stdout_trimmed": False,
            "nul_delimited": True,
            "rename_copy_supported": True,
        },
        "test_file_count": EXPECTED_TOTAL_FILES,
        "passed_test_count": total_passed,
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
                "passed_test_count": total_passed,
                "stagef_passed": stagef_result["passed"],
                "resume1_passed": resume1_result["passed"],
                "resume2_passed": resume2_result["passed"],
                "output": str(output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
