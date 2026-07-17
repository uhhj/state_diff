#!/usr/bin/env python3
"""Isolated frozen-test gate for Stage-D Resume1."""
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

from ccda_phase3.phase314b_r258_staged_resume1_test_view import (
    BASE_EVIDENCE_COMMIT,
    BASE_TEST_GATE,
    EXPECTED_BASE_PASSED,
    EXPECTED_BASE_TEST_FILES,
    EXPECTED_RESUME1_PASSED,
    EXPECTED_STAGE_D_PASSED,
    EXPECTED_TOTAL_FILES,
    EXPECTED_TOTAL_PASSED,
    ORIGINAL_STAGE_D_TEST,
    RESUME1_TEST,
    RESUME1_TEST_GATE,
    atomic_write_once,
    create_frozen_clone,
    discover_phase3_r2_tests,
    load_json,
    parse_pytest_pass_count,
    sha256_bytes,
    sha256_file,
    stable_json_bytes,
    validate_base_test_gate,
    validate_frozen_manifest,
    validate_original_blocked,
    validate_original_files,
    validate_resume1_implementation_commit,
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
        default=RESUME1_TEST_GATE,
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
            "refusing to overwrite Resume1 test gate: {}".format(
                output
            )
        )

    commits = (
        validate_resume1_implementation_commit(
            root
        )
    )
    original_sha = validate_original_files(
        root
    )
    blocked = validate_original_blocked(
        root
    )
    base = validate_base_test_gate(
        root
    )
    manifest = base[
        "test_manifest_sha256"
    ]

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
        / "scripts/phase3_14b_r258_staged_worker.py",
        root
        / "scripts/phase3_14b_r258_staged_run_calibration.py",
        root
        / "scripts/phase3_14b_r258_staged_test_gate.py",
        root
        / "scripts/phase3_14b_r258_staged_blocked.py",
        root
        / "scripts/phase3_14b_r258_staged_resume1_test_gate.py",
        root
        / "scripts/phase3_14b_r258_staged_resume1_run_calibration.py",
        root
        / "scripts/phase3_14b_r258_staged_resume1_blocked.py",
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
                "phase3_14b_r258_staged_resume1_run.sh"
            ),
        ],
        cwd=str(root),
        check=True,
    )

    current_discovered = (
        discover_phase3_r2_tests(
            root
        )
    )
    current_extra = tuple(
        sorted(
            set(current_discovered)
            - set(
                # The frozen closed-world population.
                load_json(
                    root / BASE_TEST_GATE
                )[
                    "test_manifest_sha256"
                ]
            )
        )
    )

    temporary_root = Path(
        tempfile.mkdtemp(
            prefix=(
                "phase314b_r258_"
                "staged_resume1_gate_"
            ),
            dir="/tmp",
        )
    )
    clone_root = (
        temporary_root / "frozen_base_clone"
    )
    try:
        clone_record = create_frozen_clone(
            source_root=root,
            destination=clone_root,
        )
        frozen_manifest = (
            validate_frozen_manifest(
                clone_root=clone_root,
                manifest=manifest,
            )
        )
        frozen_paths = [
            clone_root / relative
            for relative in sorted(
                manifest
            )
        ]
        frozen_result = run_pytest(
            python_bin=args.python_bin,
            cwd=clone_root,
            paths=frozen_paths,
            pycache_prefix=(
                temporary_root
                / "frozen_pycache"
            ),
        )
        if frozen_result[
            "path_count"
        ] != EXPECTED_BASE_TEST_FILES:
            raise RuntimeError(
                "frozen test-file count changed"
            )
        if frozen_result[
            "passed"
        ] != EXPECTED_BASE_PASSED:
            raise RuntimeError(
                "frozen pass count changed: {}".format(
                    frozen_result[
                        "passed"
                    ]
                )
            )

        stage_d_result = run_pytest(
            python_bin=args.python_bin,
            cwd=root,
            paths=[
                root / ORIGINAL_STAGE_D_TEST
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
                "original Stage-D test count changed: {}".format(
                    stage_d_result[
                        "passed"
                    ]
                )
            )

        resume1_result = run_pytest(
            python_bin=args.python_bin,
            cwd=root,
            paths=[
                root / RESUME1_TEST
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
    finally:
        shutil.rmtree(
            temporary_root,
            ignore_errors=True,
        )

    total = (
        frozen_result["passed"]
        + stage_d_result["passed"]
        + resume1_result["passed"]
    )
    if total != EXPECTED_TOTAL_PASSED:
        raise RuntimeError(
            "Resume1 total pass count changed: {}".format(
                total
            )
        )

    test_manifest: Dict[str, str] = {
        str(relative): str(expected)
        for relative, expected
        in manifest.items()
    }
    test_manifest[
        ORIGINAL_STAGE_D_TEST
    ] = sha256_file(
        root / ORIGINAL_STAGE_D_TEST
    )
    test_manifest[
        RESUME1_TEST
    ] = sha256_file(
        root / RESUME1_TEST
    )
    if len(
        test_manifest
    ) != EXPECTED_TOTAL_FILES:
        raise RuntimeError(
            "Resume1 combined test manifest changed"
        )

    report: Dict[str, Any] = {
        "phase":
            "Phase3.14b-r2.5.8 Stage D Resume1",
        "schema":
            "phase314b_r258_staged_resume1_test_gate_v1",
        "verdict":
            "PASS",
        "base_evidence_commit":
            BASE_EVIDENCE_COMMIT,
        "commit_chain":
            commits,
        "original_stage_d_file_sha256":
            original_sha,
        "original_blocked_sha256":
            sha256_file(
                root
                / (
                    "reports/"
                    "phase3_14b_r258_staged_"
                    "blocked_summary.json"
                )
            ),
        "original_blocked_exit_code":
            blocked.get(
                "exit_code"
            ),
        "failure_class":
            (
                "historical_closed_world_test_glob_"
                "saw_new_stage_d_test"
            ),
        "base_test_gate_path":
            BASE_TEST_GATE,
        "base_test_gate_sha256":
            sha256_file(
                root / BASE_TEST_GATE
            ),
        "test_view_isolation": {
            **clone_record,
            "frozen_manifest":
                frozen_manifest,
            "current_phase3_r2_discovered":
                list(
                    current_discovered
                ),
            "current_extra_relative_to_"
            "frozen_51_file_manifest":
                list(current_extra),
            "original_faulty_gate_rerun":
                False,
            "frozen_tests_run_from_current_root":
                False,
            "frozen_tests_run_from_base_clone":
                True,
            "current_test_files_renamed":
                False,
            "current_test_files_hidden":
                False,
            "pytest_ignore_used":
                False,
            "pytest_k_used":
                False,
            "pytest_deselection_used":
                False,
        },
        "frozen_population": {
            "test_file_count":
                frozen_result[
                    "path_count"
                ],
            "passed":
                frozen_result[
                    "passed"
                ],
            "stdout_sha256":
                frozen_result[
                    "stdout_sha256"
                ],
        },
        "original_stage_d_population": {
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
        "resume1_population": {
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
        "test_file_count":
            EXPECTED_TOTAL_FILES,
        "passed_test_count":
            total,
        "test_manifest_sha256":
            test_manifest,
        "scientific_calibration_run":
            False,
        "frozen_probe_accessed":
            False,
        "selection_holdout_accessed":
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
                "frozen_passed":
                    frozen_result[
                        "passed"
                    ],
                "stage_d_passed":
                    stage_d_result[
                        "passed"
                    ],
                "resume1_passed":
                    resume1_result[
                        "passed"
                    ],
                "total_passed":
                    total,
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
