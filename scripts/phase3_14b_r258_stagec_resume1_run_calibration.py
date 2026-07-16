#!/usr/bin/env python3
"""Run two isolated Stage-C Resume1 workers and seal new evidence."""
from __future__ import annotations

import argparse
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

from ccda_phase3.phase314b_r258_stagec_resume1_commit_recovery import (
    EVIDENCE_MESSAGE,
    EXPECTED_SUBMODULE_COMMIT,
    PHASE,
    REMOTE_BASE_COMMIT,
    RESUME_CONTRACT,
    RESUME_REPORT,
    RESUME_SUMMARY,
    RESUME_TEST_GATE,
    RESUME_WORKER,
    atomic_write_once,
    compare_worker_results,
    git_output,
    load_json,
    sha256_file,
    stable_json_bytes,
    validate_first_attempt_reports,
    validate_remote_before_resume,
    validate_resume_implementation_commit,
)

DEFAULT_CONTRACT = RESUME_CONTRACT
DEFAULT_WORKER = RESUME_WORKER
DEFAULT_SUMMARY = RESUME_SUMMARY
DEFAULT_REPORT = RESUME_REPORT


def resolve(root: Path, value: str) -> Path:
    candidate = Path(value)
    return (
        candidate.resolve()
        if candidate.is_absolute()
        else (root / candidate).resolve()
    )


def load_original_renderer(root: Path) -> Any:
    path = (
        root
        / "scripts/"
        "phase3_14b_r258_stagec_run_calibration.py"
    )
    spec = importlib.util.spec_from_file_location(
        "phase314b_r258_stagec_original_runner",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            "could not load original Stage-C renderer"
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
            "Stage-C Resume1 requires Experiment1"
        )
    resume = (
        validate_resume_implementation_commit(
            root
        )
    )
    remote = validate_remote_before_resume(
        root
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
    if git_output(
        submodule,
        "status",
        "--porcelain",
        "--untracked-files=all",
    ):
        raise RuntimeError(
            "DeformableRavens worktree is dirty"
        )

    allowed = {
        test_gate.relative_to(root).as_posix()
    }
    unexpected: List[str] = []
    status = git_output(
        root,
        "status",
        "--porcelain",
        "--untracked-files=all",
    )
    for line in status.splitlines():
        if not line.strip():
            continue
        path = line[3:].strip()
        if " -> " in path:
            path = path.split(
                " -> ",
                1,
            )[1]
        if path not in allowed:
            unexpected.append(line)
    if unexpected:
        raise RuntimeError(
            "unexpected worktree paths before Resume1 calibration: "
            + repr(unexpected)
        )
    return {
        "branch": "Experiment1",
        "head": git_output(
            root,
            "rev-parse",
            "HEAD",
        ),
        "origin_experiment1":
            remote,
        "remote_base_commit":
            REMOTE_BASE_COMMIT,
        "resume_implementation":
            resume,
        "submodule_commit":
            EXPECTED_SUBMODULE_COMMIT,
    }


def render_resume_report(
    *,
    summary: Dict[str, Any],
    original_renderer: Any,
) -> str:
    provenance = summary[
        "first_attempt_provenance"
    ]
    prefix = [
        "# Phase3.14b-r2.5.8 Stage C Resume1",
        "",
        "## Commit-outcome recovery",
        "",
        "- Original implementation commit: `{}`".format(
            provenance[
                "implementation_commit"
            ]
        ),
        "- Original `git commit` process status: `141`",
        "- Git transaction completed: `true`",
        "- First test-gate SHA256: `{}`".format(
            provenance[
                "reports"
            ]["test_gate_sha256"]
        ),
        "- First blocked SHA256: `{}`".format(
            provenance[
                "reports"
            ]["blocked_sha256"]
        ),
        "- First calibration started: `false`",
        "- Original reports modified: `false`",
        "- Recovery namespace: `phase3_14b_r258_stagec_resume1`",
        "",
        "## Scientific calibration result",
        "",
    ]
    original = original_renderer.render_report(
        summary
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
        default=DEFAULT_CONTRACT,
    )
    parser.add_argument(
        "--worker-evidence",
        default=DEFAULT_WORKER,
    )
    parser.add_argument(
        "--summary",
        default=DEFAULT_SUMMARY,
    )
    parser.add_argument(
        "--report",
        default=DEFAULT_REPORT,
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
    test_gate_path = root / RESUME_TEST_GATE
    for path in (
        contract_path,
        worker_path,
        summary_path,
        report_path,
    ):
        if path.exists():
            raise FileExistsError(
                "Resume1 write-once output exists: {}".format(path)
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
    first_reports = (
        validate_first_attempt_reports(
            root
        )
    )
    original_renderer = (
        load_original_renderer(root)
    )

    temporary_root = Path(
        tempfile.mkdtemp(
            prefix="phase314b_r258_stagec_resume1_",
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
            "phase3_14b_r258_stagec_resume1_worker.py"
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
        comparison = (
            compare_worker_results(
                worker_results[0],
                worker_results[1],
            )
        )
        if not comparison["exact"]:
            raise RuntimeError(
                "isolated Resume1 workers differ"
            )
        result = worker_results[0]

        recovery = {
            "schema":
                "phase314b_r258_stagec_resume1_recovery_v1",
            "implementation_commit":
                "0692a5ba794e1b7477d56874f7d003ec887eafc1",
            "implementation_transaction_completed":
                True,
            "implementation_command_exit_code":
                141,
            "implementation_command_failure_class":
                "ssh_output_channel_sigpipe",
            "scientific_calibration_started_before_resume":
                False,
            "reports":
                first_reports,
            "original_reports_modified":
                False,
            "old_wrapper_rerun":
                False,
            "new_write_once_namespace":
                "phase3_14b_r258_stagec_resume1",
        }

        contract_record = {
            "phase": PHASE,
            "schema":
                "phase314b_r258_stagec_resume1_contract_record_v1",
            "recovery": recovery,
            **result[
                "balanced_geometry_contract"
            ],
            "root_cause":
                result["root_cause"],
            "required_next_path":
                result["required_next_path"],
            "scientific_status":
                result["scientific_status"],
            "selection_sha256":
                result["selection"][
                    "selection_sha256"
                ],
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
                "phase314b_r258_stagec_resume1_worker_evidence_v1",
            "worker_count": 2,
            "workers_exact": True,
            "comparison": comparison,
            "recovery": recovery,
            "worker_result": result,
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
                "phase314b_r258_stagec_resume1_summary_v1",
            "verdict": "PASS",
            "scientific_status":
                result["scientific_status"],
            "root_cause":
                result["root_cause"],
            "required_next_path":
                result["required_next_path"],
            "repository": repository,
            "first_attempt_provenance":
                recovery,
            "test_gate_path":
                RESUME_TEST_GATE,
            "test_gate_sha256":
                sha256_file(
                    test_gate_path
                ),
            "test_gate": test_gate,
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
            "commit_outcome_recovery":
                True,
            "new_model_candidate_trained":
                False,
            "direct_x0_tensor_optimization_run":
                True,
            "balanced_objective_calibration_run":
                True,
            "control_replay_exact":
                result[
                    "control_replay_exact"
                ],
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
            render_resume_report(
                summary=summary,
                original_renderer=
                    original_renderer,
            ).encode("utf-8"),
        )
        atomic_write_once(
            summary_path,
            stable_json_bytes(summary),
        )
    finally:
        shutil.rmtree(
            temporary_root,
            ignore_errors=True,
        )

    print(
        json.dumps(
            {
                "verdict": "PASS",
                "scientific_status":
                    summary[
                        "scientific_status"
                    ],
                "root_cause":
                    summary["root_cause"],
                "required_next_path":
                    summary[
                        "required_next_path"
                    ],
                "workers_exact":
                    summary["workers_exact"],
                "contract_file_sha256":
                    summary[
                        "contract_file_sha256"
                    ],
                "selected_configuration":
                    summary[
                        "selected_configuration"
                    ],
                "evidence_commit_message":
                    EVIDENCE_MESSAGE,
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
