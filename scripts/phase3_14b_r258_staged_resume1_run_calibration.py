#!/usr/bin/env python3
"""Run unchanged Stage-D science under Resume1 evidence namespace."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ccda_phase3.phase314b_r258_staged_direction_surrogate import (
    EXPECTED_SUBMODULE_COMMIT,
    compare_worker_results,
)
from ccda_phase3.phase314b_r258_staged_resume1_test_view import (
    BASE_EVIDENCE_COMMIT,
    ORIGINAL_BLOCKED_PATH,
    ORIGINAL_BLOCKED_SHA256,
    PHASE,
    RESUME1_CONTRACT,
    RESUME1_REPORT,
    RESUME1_SUMMARY,
    RESUME1_TEST_GATE,
    RESUME1_WORKER,
    atomic_write_once,
    git_output,
    load_json,
    sha256_file,
    stable_json_bytes,
    status_paths,
    validate_original_blocked,
    validate_original_files,
    validate_resume1_implementation_commit,
)


def resolve(
    root: Path,
    value: str,
) -> Path:
    candidate = Path(value)
    return (
        candidate.resolve()
        if candidate.is_absolute()
        else (root / candidate).resolve()
    )


def load_original_renderer(
    root: Path,
) -> Any:
    path = (
        root
        / "scripts/"
        "phase3_14b_r258_staged_run_calibration.py"
    )
    spec = importlib.util.spec_from_file_location(
        "phase314b_r258_staged_original_runner",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            "could not load original Stage-D renderer"
        )
    module = importlib.util.module_from_spec(
        spec
    )
    spec.loader.exec_module(module)
    return module


def assert_repository(
    root: Path,
    test_gate: Path,
) -> Dict[str, Any]:
    if git_output(
        root,
        "branch",
        "--show-current",
    ) != "Experiment1":
        raise RuntimeError(
            "Stage-D Resume1 requires Experiment1"
        )
    commits = (
        validate_resume1_implementation_commit(
            root
        )
    )
    remote = git_output(
        root,
        "rev-parse",
        "origin/Experiment1",
    )
    if remote != BASE_EVIDENCE_COMMIT:
        raise RuntimeError(
            "origin/Experiment1 changed before Resume1 push"
        )

    submodule = (
        root / "external/deformable-ravens"
    )
    if git_output(
        submodule,
        "rev-parse",
        "HEAD",
    ) != EXPECTED_SUBMODULE_COMMIT:
        raise RuntimeError(
            "DeformableRavens commit changed"
        )
    if status_paths(
        submodule
    ):
        raise RuntimeError(
            "DeformableRavens worktree is dirty"
        )

    allowed = {
        test_gate.relative_to(root).as_posix()
    }
    observed = status_paths(root)
    if observed != tuple(
        sorted(allowed)
    ):
        raise RuntimeError(
            "unexpected worktree paths before scientific run: {}".format(
                observed
            )
        )
    return {
        "branch":
            "Experiment1",
        "head":
            git_output(
                root,
                "rev-parse",
                "HEAD",
            ),
        "origin_experiment1":
            remote,
        "base_evidence_commit":
            BASE_EVIDENCE_COMMIT,
        "commit_chain":
            commits,
        "submodule_commit":
            EXPECTED_SUBMODULE_COMMIT,
    }


def render_resume1_report(
    *,
    summary: Dict[str, Any],
    original_renderer: Any,
) -> str:
    recovery = summary[
        "test_view_recovery"
    ]
    prefix = [
        "# Phase3.14b-r2.5.8 Stage D Resume1",
        "",
        "## Frozen test-view recovery",
        "",
        "- Original Stage-D implementation commit: `{}`".format(
            summary[
                "repository"
            ][
                "commit_chain"
            ][
                "original_implementation_commit"
            ]
        ),
        "- Original blocked provenance commit: `{}`".format(
            summary[
                "repository"
            ][
                "commit_chain"
            ][
                "blocked_provenance_commit"
            ]
        ),
        "- Resume1 implementation commit: `{}`".format(
            summary[
                "repository"
            ][
                "commit_chain"
            ][
                "resume1_implementation_commit"
            ]
        ),
        "- Original blocked SHA256: `{}`".format(
            ORIGINAL_BLOCKED_SHA256
        ),
        "- Failure class: `{}`".format(
            recovery[
                "failure_class"
            ]
        ),
        "- Frozen base view: local clone reset to `{}`".format(
            BASE_EVIDENCE_COMMIT
        ),
        "- Frozen tests run from current root: `false`",
        "- `--ignore`, `-k`, deselection, or test renaming used: `false`",
        "- Original Stage-D scientific calibration started before Resume1: `false`",
        "",
        "## Resume1 scientific result",
        "",
    ]
    original_summary = copy.deepcopy(
        summary
    )
    original = original_renderer.render_report(
        original_summary
    )
    return (
        "\n".join(prefix)
        + original
    )


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
        "--contract",
        default=RESUME1_CONTRACT,
    )
    parser.add_argument(
        "--worker-evidence",
        default=RESUME1_WORKER,
    )
    parser.add_argument(
        "--summary",
        default=RESUME1_SUMMARY,
    )
    parser.add_argument(
        "--report",
        default=RESUME1_REPORT,
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    contract_path = resolve(
        root,
        args.contract,
    )
    worker_path = resolve(
        root,
        args.worker_evidence,
    )
    summary_path = resolve(
        root,
        args.summary,
    )
    report_path = resolve(
        root,
        args.report,
    )
    test_gate_path = (
        root / RESUME1_TEST_GATE
    )
    for path in (
        contract_path,
        worker_path,
        summary_path,
        report_path,
    ):
        if path.exists():
            raise FileExistsError(
                "Resume1 write-once output exists: {}".format(
                    path
                )
            )

    repository = assert_repository(
        root,
        test_gate_path,
    )
    test_gate = load_json(
        test_gate_path
    )
    if test_gate.get("verdict") != "PASS":
        raise RuntimeError(
            "Resume1 test gate is not PASS"
        )
    if int(
        test_gate.get(
            "test_file_count",
            -1,
        )
    ) != 53:
        raise RuntimeError(
            "Resume1 test-file count changed"
        )
    if int(
        test_gate.get(
            "passed_test_count",
            -1,
        )
    ) != 1188:
        raise RuntimeError(
            "Resume1 pass count changed"
        )

    original_sha = validate_original_files(
        root
    )
    original_blocked = (
        validate_original_blocked(
            root
        )
    )
    if sha256_file(
        root / ORIGINAL_BLOCKED_PATH
    ) != ORIGINAL_BLOCKED_SHA256:
        raise RuntimeError(
            "original blocked report changed"
        )

    original_renderer = (
        load_original_renderer(root)
    )
    temporary_root = Path(
        tempfile.mkdtemp(
            prefix=(
                "phase314b_r258_"
                "staged_resume1_"
            ),
            dir="/tmp",
        )
    )
    worker_files = [
        temporary_root / "worker_1.json",
        temporary_root / "worker_2.json",
    ]
    command = [
        str(args.python_bin),
        str(
            root
            / "scripts/"
            "phase3_14b_r258_staged_worker.py"
        ),
        "--root",
        str(root),
        "--mode",
        "run",
    ]
    try:
        for worker_file in worker_files:
            subprocess.run(
                command
                + [
                    "--output",
                    str(worker_file),
                ],
                cwd=str(root),
                check=True,
            )
        worker_results = [
            load_json(path)
            for path in worker_files
        ]
        comparison = compare_worker_results(
            worker_results[0],
            worker_results[1],
        )
        if not comparison["exact"]:
            raise RuntimeError(
                "isolated Stage-D Resume1 workers differ"
            )
        result = worker_results[0]

        recovery = {
            "schema":
                "phase314b_r258_staged_resume1_test_view_recovery_v1",
            "failure_class":
                "historical_closed_world_test_glob_saw_new_stage_d_test",
            "original_blocked_path":
                ORIGINAL_BLOCKED_PATH,
            "original_blocked_sha256":
                ORIGINAL_BLOCKED_SHA256,
            "original_blocked_exit_code":
                original_blocked.get(
                    "exit_code"
                ),
            "original_scientific_result_sealed":
                original_blocked.get(
                    "scientific_result_sealed"
                ),
            "original_stage_d_file_sha256":
                original_sha,
            "test_gate_path":
                RESUME1_TEST_GATE,
            "test_gate_sha256":
                sha256_file(
                    test_gate_path
                ),
            "frozen_population":
                test_gate[
                    "frozen_population"
                ],
            "original_stage_d_population":
                test_gate[
                    "original_stage_d_population"
                ],
            "resume1_population":
                test_gate[
                    "resume1_population"
                ],
            "test_view_isolation":
                test_gate[
                    "test_view_isolation"
                ],
            "original_faulty_wrapper_rerun":
                False,
            "scientific_code_changed":
                False,
            "candidate_contract_changed":
                False,
            "threshold_changed":
                False,
        }

        contract_record = {
            **result[
                "direction_surrogate_contract"
            ],
            "phase": PHASE,
            "schema":
                "phase314b_r258_staged_resume1_contract_record_v1",
            "test_view_recovery":
                recovery,
            "root_cause":
                result["root_cause"],
            "required_next_path":
                result["required_next_path"],
            "scientific_status":
                result["scientific_status"],
            "selection_sha256":
                result[
                    "selection"
                ]["selection_sha256"],
            "selected_configuration":
                result[
                    "selected_configuration"
                ],
        }
        atomic_write_once(
            contract_path,
            stable_json_bytes(
                contract_record
            ),
        )

        worker_evidence = {
            "phase": PHASE,
            "schema":
                "phase314b_r258_staged_resume1_worker_evidence_v1",
            "worker_count": 2,
            "workers_exact": True,
            "comparison":
                comparison,
            "test_view_recovery":
                recovery,
            "worker_result":
                result,
        }
        atomic_write_once(
            worker_path,
            stable_json_bytes(
                worker_evidence
            ),
        )

        summary = {
            "phase": PHASE,
            "schema":
                "phase314b_r258_staged_resume1_summary_v1",
            "verdict":
                "PASS",
            "scientific_status":
                result[
                    "scientific_status"
                ],
            "root_cause":
                result["root_cause"],
            "required_next_path":
                result[
                    "required_next_path"
                ],
            "repository":
                repository,
            "test_view_recovery":
                recovery,
            "test_gate_path":
                RESUME1_TEST_GATE,
            "test_gate_sha256":
                sha256_file(
                    test_gate_path
                ),
            "test_gate":
                test_gate,
            "contract_path":
                contract_path.relative_to(
                    root
                ).as_posix(),
            "contract_file_sha256":
                sha256_file(
                    contract_path
                ),
            "worker_evidence_path":
                worker_path.relative_to(
                    root
                ).as_posix(),
            "worker_evidence_sha256":
                sha256_file(
                    worker_path
                ),
            "workers_exact":
                comparison["exact"],
            "worker_result":
                result,
            "selected_configuration":
                result[
                    "selected_configuration"
                ],
            "train_only_recommendation":
                result[
                    "train_only_recommendation"
                ],
            "test_view_isolation_completed":
                True,
            "scientific_result_sealed":
                True,
            "direction_surrogate_trained":
                True,
            "new_diffusion_model_candidate_trained":
                False,
            "selection_holdout_evaluated":
                result[
                    "selection_holdout_evaluated"
                ],
            "selection_holdout_used_for_fit":
                False,
            "selection_holdout_used_for_selection":
                False,
            "frozen_probe_accessed":
                False,
            "reverse_sampling_run":
                False,
            "full_stageb_repaired_model_trained":
                False,
            "formal_pilot_run":
                False,
            "checkpoint_saved":
                False,
            "weights_persisted":
                False,
            "surrogate_weights_persisted":
                False,
            "prediction_tensor_persisted":
                False,
            "oracle_tensor_persisted":
                False,
            "candidate_tensor_persisted":
                False,
            "npz_saved":
                False,
            "cache_saved":
                False,
            "formal_diffusion_training":
                False,
            "formal_reverse_sampling":
                False,
            "formal_idm_training":
                False,
            "action_diverse_data_collection":
                False,
            "candidate_execution":
                False,
            "deformable_ravens_executed":
                False,
            "phase4":
                False,
            "cps":
                False,
        }
        atomic_write_once(
            report_path,
            render_resume1_report(
                summary=summary,
                original_renderer=
                    original_renderer,
            ).encode("utf-8"),
        )
        atomic_write_once(
            summary_path,
            stable_json_bytes(
                summary
            ),
        )
    finally:
        shutil.rmtree(
            temporary_root,
            ignore_errors=True,
        )

    print(
        json.dumps(
            {
                "verdict":
                    "PASS",
                "scientific_status":
                    summary[
                        "scientific_status"
                    ],
                "root_cause":
                    summary[
                        "root_cause"
                    ],
                "required_next_path":
                    summary[
                        "required_next_path"
                    ],
                "workers_exact":
                    summary[
                        "workers_exact"
                    ],
                "worker_sha256":
                    comparison[
                        "left_sha256"
                    ],
                "contract_file_sha256":
                    summary[
                        "contract_file_sha256"
                    ],
                "selected_configuration":
                    summary[
                        "selected_configuration"
                    ],
                "summary":
                    str(summary_path),
                "report":
                    str(report_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
