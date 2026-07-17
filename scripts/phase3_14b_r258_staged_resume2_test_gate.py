#!/usr/bin/env python3
"""Temporal-split frozen test gate for Stage-D Resume2."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ccda_phase3.phase314b_r258_staged_resume2_temporal_views import (
    BASE_EVIDENCE_COMMIT,
    BASE_TEST_GATE,
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
    RESUME2_TEST_GATE,
    atomic_write_once,
    create_temporal_clone,
    historical_test_blob_identity,
    inspect_historical_scope,
    load_json,
    parse_pytest_pass_count,
    sha256_bytes,
    sha256_file,
    split_base_manifest,
    stable_json_bytes,
    validate_base_test_gate,
    validate_manifest_files,
    validate_resume2_implementation_commit,
)


EXPECTED_TOTAL_FILES = 54
EXPECTED_TOTAL_PASSED = (
    EXPECTED_BASE_PASSED
    + EXPECTED_STAGE_D_PASSED
    + EXPECTED_RESUME1_PASSED
    + EXPECTED_RESUME2_PASSED
)


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
            "PYTHONPYCACHEPREFIX":
                str(pycache_prefix),
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
        }
    )
    completed = subprocess.run(
        [
            str(python_bin),
            "-m",
            "pytest",
            "-q",
            *[
                str(path)
                for path in paths
            ],
        ],
        cwd=str(cwd),
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(
        completed.stdout,
        end="",
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "pytest population failed with status {}".format(
                completed.returncode
            )
        )
    return {
        "passed":
            parse_pytest_pass_count(
                completed.stdout
            ),
        "stdout_sha256":
            sha256_bytes(
                completed.stdout.encode(
                    "utf-8"
                )
            ),
        "return_code":
            int(completed.returncode),
        "path_count":
            len(paths),
        "paths": [
            path.relative_to(cwd).as_posix()
            for path in paths
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="/data/state_diff2",
    )
    parser.add_argument(
        "--python-bin",
        default=(
            "/miniforge3/envs/"
            "coord_bimanual/bin/python"
        ),
    )
    parser.add_argument(
        "--output",
        default=RESUME2_TEST_GATE,
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output = (
        Path(args.output).resolve()
        if Path(args.output).is_absolute()
        else (root / args.output).resolve()
    )
    if output.exists():
        raise FileExistsError(
            "refusing to overwrite Resume2 test gate: {}".format(
                output
            )
        )

    commits = (
        validate_resume2_implementation_commit(
            root
        )
    )
    base = validate_base_test_gate(
        root
    )
    manifest = {
        str(relative): str(expected)
        for relative, expected
        in base[
            "test_manifest_sha256"
        ].items()
    }
    split = split_base_manifest(
        manifest
    )
    blob_identity = (
        historical_test_blob_identity(
            root
        )
    )

    compile_targets = [
        root
        / (
            "ccda_phase3/"
            "phase314b_r258_staged_direction_surrogate.py"
        ),
        root
        / (
            "ccda_phase3/"
            "phase314b_r258_staged_resume1_test_view.py"
        ),
        root
        / (
            "ccda_phase3/"
            "phase314b_r258_staged_resume2_temporal_views.py"
        ),
        root
        / "scripts/phase3_14b_r258_staged_worker.py",
        root
        / "scripts/phase3_14b_r258_staged_run_calibration.py",
        root
        / "scripts/phase3_14b_r258_staged_resume1_run_calibration.py",
        root
        / "scripts/phase3_14b_r258_staged_resume2_test_gate.py",
        root
        / "scripts/phase3_14b_r258_staged_resume2_run_calibration.py",
        root
        / "scripts/phase3_14b_r258_staged_resume2_blocked.py",
    ]
    subprocess.run(
        [
            str(args.python_bin),
            "-m",
            "py_compile",
            *[
                str(path)
                for path in compile_targets
            ],
        ],
        cwd=str(root),
        check=True,
    )
    subprocess.run(
        [
            "bash",
            "-n",
            str(
                root
                / "scripts/"
                "phase3_14b_r258_staged_resume2_run.sh"
            ),
        ],
        cwd=str(root),
        check=True,
    )

    temporary_root = Path(
        tempfile.mkdtemp(
            prefix=(
                "phase314b_r258_"
                "staged_resume2_gate_"
            ),
            dir="/tmp",
        )
    )
    base_clone = (
        temporary_root
        / "base_evidence_clone"
    )
    historical_clone = (
        temporary_root
        / "historical_closed_world_clone"
    )
    try:
        base_clone_record = create_temporal_clone(
            source_root=root,
            destination=base_clone,
            commit=BASE_EVIDENCE_COMMIT,
        )
        historical_clone_record = (
            create_temporal_clone(
                source_root=root,
                destination=historical_clone,
                commit=(
                    HISTORICAL_CLOSED_WORLD_COMMIT
                ),
            )
        )

        base_manifest_observed = (
            validate_manifest_files(
                root=base_clone,
                manifest=manifest,
            )
        )
        historical_scope = (
            inspect_historical_scope(
                python_bin=args.python_bin,
                clone_root=historical_clone,
            )
        )

        regular_paths = [
            base_clone / relative
            for relative in sorted(
                split[
                    "regular_manifest"
                ]
            )
        ]
        regular_result = run_pytest(
            python_bin=args.python_bin,
            cwd=base_clone,
            paths=regular_paths,
            pycache_prefix=(
                temporary_root
                / "base_regular_pycache"
            ),
        )
        if regular_result[
            "path_count"
        ] != EXPECTED_BASE_REGULAR_FILES:
            raise RuntimeError(
                "regular base file count changed"
            )
        if regular_result[
            "passed"
        ] != EXPECTED_BASE_REGULAR_PASSED:
            raise RuntimeError(
                "regular base pass count changed: {}".format(
                    regular_result[
                        "passed"
                    ]
                )
            )

        historical_result = run_pytest(
            python_bin=args.python_bin,
            cwd=historical_clone,
            paths=[
                historical_clone
                / HISTORICAL_CLOSED_WORLD_TEST
            ],
            pycache_prefix=(
                temporary_root
                / "historical_pycache"
            ),
        )
        if historical_result[
            "path_count"
        ] != 1:
            raise RuntimeError(
                "historical file count changed"
            )
        if historical_result[
            "passed"
        ] != EXPECTED_HISTORICAL_TEST_PASSED:
            raise RuntimeError(
                "historical pass count changed: {}".format(
                    historical_result[
                        "passed"
                    ]
                )
            )

        frozen_total = (
            regular_result["passed"]
            + historical_result["passed"]
        )
        frozen_files = (
            regular_result["path_count"]
            + historical_result["path_count"]
        )
        if frozen_files != EXPECTED_BASE_TEST_FILES:
            raise RuntimeError(
                "reconstructed frozen file count changed"
            )
        if frozen_total != EXPECTED_BASE_PASSED:
            raise RuntimeError(
                "reconstructed frozen pass count changed"
            )

        stage_d_result = run_pytest(
            python_bin=args.python_bin,
            cwd=root,
            paths=[
                root / CURRENT_STAGE_D_TEST
            ],
            pycache_prefix=(
                temporary_root
                / "stage_d_pycache"
            ),
        )
        if stage_d_result[
            "passed"
        ] != EXPECTED_STAGE_D_PASSED:
            raise RuntimeError(
                "Stage-D test count changed: {}".format(
                    stage_d_result[
                        "passed"
                    ]
                )
            )

        resume1_result = run_pytest(
            python_bin=args.python_bin,
            cwd=root,
            paths=[
                root / CURRENT_RESUME1_TEST
            ],
            pycache_prefix=(
                temporary_root
                / "resume1_pycache"
            ),
        )
        if resume1_result[
            "passed"
        ] != EXPECTED_RESUME1_PASSED:
            raise RuntimeError(
                "Resume1 test count changed: {}".format(
                    resume1_result[
                        "passed"
                    ]
                )
            )

        resume2_result = run_pytest(
            python_bin=args.python_bin,
            cwd=root,
            paths=[
                root / CURRENT_RESUME2_TEST
            ],
            pycache_prefix=(
                temporary_root
                / "resume2_pycache"
            ),
        )
        if resume2_result[
            "passed"
        ] != EXPECTED_RESUME2_PASSED:
            raise RuntimeError(
                "Resume2 test count changed: {}".format(
                    resume2_result[
                        "passed"
                    ]
                )
            )
    finally:
        shutil.rmtree(
            temporary_root,
            ignore_errors=True,
        )

    total_passed = (
        frozen_total
        + stage_d_result["passed"]
        + resume1_result["passed"]
        + resume2_result["passed"]
    )
    if total_passed != EXPECTED_TOTAL_PASSED:
        raise RuntimeError(
            "Resume2 total pass count changed: {}".format(
                total_passed
            )
        )

    combined_manifest: Dict[str, str] = dict(
        manifest
    )
    combined_manifest[
        CURRENT_STAGE_D_TEST
    ] = sha256_file(
        root / CURRENT_STAGE_D_TEST
    )
    combined_manifest[
        CURRENT_RESUME1_TEST
    ] = sha256_file(
        root / CURRENT_RESUME1_TEST
    )
    combined_manifest[
        CURRENT_RESUME2_TEST
    ] = sha256_file(
        root / CURRENT_RESUME2_TEST
    )
    if len(
        combined_manifest
    ) != EXPECTED_TOTAL_FILES:
        raise RuntimeError(
            "combined test manifest changed"
        )

    report: Dict[str, Any] = {
        "phase":
            "Phase3.14b-r2.5.8 Stage D Resume2",
        "schema":
            "phase314b_r258_staged_resume2_test_gate_v1",
        "verdict":
            "PASS",
        "commit_chain":
            commits,
        "base_evidence_commit":
            BASE_EVIDENCE_COMMIT,
        "base_test_gate_path":
            BASE_TEST_GATE,
        "base_test_gate_sha256":
            sha256_file(
                root / BASE_TEST_GATE
            ),
        "failure_class": (
            "one_frozen_test_requires_its_"
            "historical_closed_world_repository_view"
        ),
        "temporal_test_views": {
            "base_evidence_view":
                base_clone_record,
            "historical_closed_world_view":
                historical_clone_record,
            "historical_test_blob_identity":
                blob_identity,
            "historical_scope":
                historical_scope,
            "base_manifest_sha256":
                base_manifest_observed,
            "manifest_partition": {
                "regular_files":
                    sorted(
                        split[
                            "regular_manifest"
                        ]
                    ),
                "historical_files":
                    sorted(
                        split[
                            "historical_manifest"
                        ]
                    ),
                "disjoint":
                    True,
                "union_equals_base_manifest":
                    True,
            },
            "all_frozen_tests_executed":
                True,
            "test_implementation_modified":
                False,
            "test_source_copied_or_patched":
                False,
            "pytest_ignore_used":
                False,
            "pytest_k_used":
                False,
            "pytest_deselection_used":
                False,
            "current_test_files_renamed":
                False,
            "original_stage_d_wrapper_rerun":
                False,
            "resume1_wrapper_rerun":
                False,
        },
        "frozen_regular_population": {
            "commit":
                BASE_EVIDENCE_COMMIT,
            "test_file_count":
                regular_result[
                    "path_count"
                ],
            "passed":
                regular_result[
                    "passed"
                ],
            "stdout_sha256":
                regular_result[
                    "stdout_sha256"
                ],
        },
        "historical_closed_world_population": {
            "commit":
                HISTORICAL_CLOSED_WORLD_COMMIT,
            "test_file_count":
                historical_result[
                    "path_count"
                ],
            "passed":
                historical_result[
                    "passed"
                ],
            "stdout_sha256":
                historical_result[
                    "stdout_sha256"
                ],
        },
        "reconstructed_frozen_population": {
            "test_file_count":
                frozen_files,
            "passed":
                frozen_total,
            "matches_base_evidence":
                True,
        },
        "current_stage_d_population": {
            "test_file_count": 1,
            "passed":
                stage_d_result[
                    "passed"
                ],
            "stdout_sha256":
                stage_d_result[
                    "stdout_sha256"
                ],
        },
        "current_resume1_population": {
            "test_file_count": 1,
            "passed":
                resume1_result[
                    "passed"
                ],
            "stdout_sha256":
                resume1_result[
                    "stdout_sha256"
                ],
        },
        "current_resume2_population": {
            "test_file_count": 1,
            "passed":
                resume2_result[
                    "passed"
                ],
            "stdout_sha256":
                resume2_result[
                    "stdout_sha256"
                ],
        },
        "test_file_count":
            EXPECTED_TOTAL_FILES,
        "passed_test_count":
            total_passed,
        "test_manifest_sha256":
            combined_manifest,
        "scientific_calibration_run":
            False,
        "selection_holdout_accessed":
            False,
        "frozen_probe_accessed":
            False,
    }
    atomic_write_once(
        output,
        stable_json_bytes(report),
    )
    print(
        json.dumps(
            {
                "verdict": "PASS",
                "frozen_regular_passed":
                    regular_result[
                        "passed"
                    ],
                "historical_passed":
                    historical_result[
                        "passed"
                    ],
                "reconstructed_frozen_passed":
                    frozen_total,
                "stage_d_passed":
                    stage_d_result[
                        "passed"
                    ],
                "resume1_passed":
                    resume1_result[
                        "passed"
                    ],
                "resume2_passed":
                    resume2_result[
                        "passed"
                    ],
                "total_passed":
                    total_passed,
                "test_file_count":
                    EXPECTED_TOTAL_FILES,
                "output":
                    str(output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
