#!/usr/bin/env python3
"""Run two isolated Stage-D Resume4 RTX-3090 workers."""
from __future__ import annotations

import argparse
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

from ccda_phase3.phase314b_r257_staged_resume4_3090 import (
    EXPECTED_SUBMODULE_COMMIT,
    PHASE,
    RESUME3_IMPLEMENTATION_COMMIT,
    atomic_write_once,
    compare_worker_results,
    git_output,
    load_json,
    sha256_file,
    stable_json_bytes,
)

TEST_GATE = (
    "reports/"
    "phase3_14b_r257_staged_resume4_3090_"
    "test_gate_summary.json"
)
DEFAULT_CONTRACT = (
    "reports/"
    "phase3_14b_r257_staged_resume4_3090_"
    "contract.json"
)
DEFAULT_WORKER = (
    "reports/"
    "phase3_14b_r257_staged_resume4_3090_"
    "worker_evidence.json"
)
DEFAULT_SUMMARY = (
    "reports/"
    "phase3_14b_r257_staged_resume4_3090_"
    "summary.json"
)
DEFAULT_REPORT = (
    "reports/"
    "phase3_14b_r257_staged_resume4_3090_"
    "report.md"
)


def resolve(root: Path, value: str) -> Path:
    candidate = Path(value)
    return (
        candidate.resolve()
        if candidate.is_absolute()
        else (root / candidate).resolve()
    )


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
            "Resume4 RTX-3090 requires Experiment1"
        )
    completed = subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            RESUME3_IMPLEMENTATION_COMMIT,
            "HEAD",
        ],
        cwd=str(root),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Resume3 implementation is not an ancestor"
        )

    submodule = root / "external/deformable-ravens"
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
            path = path.split(" -> ", 1)[1]
        if path not in allowed:
            unexpected.append(line)
    if unexpected:
        raise RuntimeError(
            "unexpected worktree paths before Resume4 run: "
            + repr(unexpected)
        )
    return {
        "branch": "Experiment1",
        "head": git_output(
            root,
            "rev-parse",
            "HEAD",
        ),
        "resume3_implementation_commit":
            RESUME3_IMPLEMENTATION_COMMIT,
        "submodule_commit":
            EXPECTED_SUBMODULE_COMMIT,
    }


def render_report(summary: Dict[str, Any]) -> str:
    result = summary["worker_result"]
    environment = result[
        "resume4_3090_environment"
    ]["environment_probe"]
    cold = result[
        "resume4_3090_environment"
    ]["cold_main_worker_context"]
    attribution = result[
        "resume3_calibration_attribution"
    ]
    manual = attribution["manual_calibration_diff"]
    capture = attribution["stagec_entrypoint_capture"]
    selection = result["selection"]
    selected = result["selected_configuration"]
    classification = result["classification"]

    lines = [
        "# Phase3.14b-r2.5.7 Stage D Resume4 RTX 3090",
        "",
        "- Audit verdict: `{}`".format(summary["verdict"]),
        "- Scientific status: `{}`".format(
            summary["scientific_status"]
        ),
        "- Root cause: `{}`".format(
            summary["root_cause"]
        ),
        "- Required next path: `{}`".format(
            summary["required_next_path"]
        ),
        "",
        "## Resume3 failure provenance",
        "",
        "- Resume3 implementation commit: `{}`".format(
            result["resume4_3090_environment"][
                "resume3_implementation_commit"
            ]
        ),
        "- Resume3 blocked SHA256: `{}`".format(
            result["resume4_3090_environment"][
                "resume3_blocked_sha256"
            ]
        ),
        "- Resume3 test-gate SHA256: `{}`".format(
            result["resume4_3090_environment"][
                "resume3_test_gate_sha256"
            ]
        ),
        "- Resume3 files/reports modified: `false`",
        "",
        "## RTX-3090 replay environment",
        "",
        "- GPU: `{}`".format(
            environment["torch_device_name"]
        ),
        "- Compute capability: `{}`".format(
            ".".join(
                str(value)
                for value in environment[
                    "compute_capability"
                ]
            )
        ),
        "- GPU UUID: `{}`".format(
            environment["nvidia_smi"]["uuid"]
        ),
        "- PCI bus: `{}`".format(
            environment["nvidia_smi"][
                "pci_bus_id"
            ]
        ),
        "- Driver: `{}`".format(
            environment["nvidia_smi"][
                "driver_version"
            ]
        ),
        "- Python: `{}`".format(
            ".".join(
                str(value)
                for value in environment[
                    "python_version"
                ]
            )
        ),
        "- NumPy: `{}`".format(
            environment["numpy_version"]
        ),
        "- PyTorch: `{}`".format(
            environment["torch_version"]
        ),
        "- Torch CUDA: `{}`".format(
            environment["torch_cuda_version"]
        ),
        "- Environment SHA256: `{}`".format(
            environment["environment_sha256"]
        ),
        "- Main worker CUDA context cold before Stage-C: `{}`".format(
            str(
                cold["torch_cuda_is_initialized"]
                is False
            ).lower()
        ),
        "",
        "## Manual calibration diagnostic",
        "",
        "- Expected calibration SHA256: `{}`".format(
            manual["expected_calibration_sha256"]
        ),
        "- Observed calibration SHA256: `{}`".format(
            manual["observed_calibration_sha256"]
        ),
        "- Total differences: `{}`".format(
            manual["difference_summary"][
                "total_difference_count"
            ]
        ),
        "- Primary difference locus: `{}`".format(
            manual["difference_summary"][
                "primary_difference_locus"
            ]
        ),
        "",
        "## Frozen Stage-C entrypoint",
        "",
        "- Calibration exact: `{}`".format(
            str(capture["calibration_exact"]).lower()
        ),
        "- Calibration SHA256: `{}`".format(
            capture["calibration_sha256"]
        ),
        "- Control record exact: `{}`".format(
            str(
                capture["control_record_exact"]
            ).lower()
        ),
        "- Control record SHA256: `{}`".format(
            capture["control_record_sha256"]
        ),
        "- Stage-C nonzero candidates trained: `{}`".format(
            capture[
                "stagec_nonzero_candidate_training_count"
            ]
        ),
        "",
        "## Resume4 contract",
        "",
        "- Contract SHA256: `{}`".format(
            result["calibration_contract"][
                "contract_sha256"
            ]
        ),
        "- Selection SHA256: `{}`".format(
            selection["selection_sha256"]
        ),
        "- Selected configuration: `{}`".format(
            "none"
            if selected is None
            else selected["candidate_id"]
        ),
        "",
        "## Gated candidates",
        "",
        (
            "| Candidate | Cutoff | Target ratio | Lambda | "
            "t10 reduction | Train ratio | t10 upper | Eligible |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in selection["candidate_records"]:
        candidate = record["candidate"]
        lines.append(
            "| {} | {} | {:.6g} | {:.12g} | {:.9g} | "
            "{:.9g} | {:.9g} | {} |".format(
                record["candidate_id"],
                candidate["timestep_cutoff"],
                candidate["target_gradient_ratio"],
                candidate["lambda_upper"],
                record["continuous_response"][
                    "t10_mean_excess_reduction"
                ],
                record["relative_to_control"][
                    "train_control_nmse_ratio"
                ],
                record["binary_rate"]["10"],
                str(
                    record["eligible_for_selection"]
                ).lower(),
            )
        )

    lines.extend(
        [
            "",
            "## Classification",
            "",
            "- Primary failure locus: `{}`".format(
                classification[
                    "primary_failure_locus"
                ]
            ),
            "",
            "## Boundary",
            "",
            "- Resume4 used a new write-once RTX-3090 namespace; no Resume3 artifact was deleted, overwritten, or rerun.",
            "- Environment and manual-diff probes ran in disposable child processes.",
            "- The main worker had no initialized CUDA context before the frozen Stage-C entry point.",
            "- No calibration/control SHA or scientific gate was relaxed.",
            "- The frozen probe was not accessed.",
            "- No full-874-row repaired model, reverse sampling, formal pilot, IDM, data collection, candidate execution, DeformableRavens, Phase4, or CPS was run.",
            "- No checkpoint, weights, tensor, NPZ, cache, image, or video was persisted.",
        ]
    )
    return "\n".join(lines) + "\n"


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
    contract_path = resolve(root, args.contract)
    worker_path = resolve(root, args.worker_evidence)
    summary_path = resolve(root, args.summary)
    report_path = resolve(root, args.report)
    test_gate_path = root / TEST_GATE
    for path in (
        contract_path,
        worker_path,
        summary_path,
        report_path,
    ):
        if path.exists():
            raise FileExistsError(
                "Resume4 RTX-3090 write-once output exists: "
                "{}".format(path)
            )

    repository = assert_repository(
        root,
        test_gate_path,
    )
    test_gate = load_json(test_gate_path)
    if test_gate.get("verdict") != "PASS":
        raise RuntimeError(
            "Resume4 test gate is not PASS"
        )

    temporary_root = Path(
        tempfile.mkdtemp(
            prefix="phase314b_r257_resume4_3090_",
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
            "phase3_14b_r257_staged_resume4_3090_worker.py"
        ),
        "--root",
        str(root),
        "--mode",
        "run",
    ]
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
            "isolated Resume4 RTX-3090 workers differ"
        )
    result = worker_results[0]

    contract_record = {
        "phase": PHASE,
        **result["calibration_contract"],
        "root_cause": result["root_cause"],
        "required_next_path":
            result["required_next_path"],
        "selection_sha256":
            result["selection"]["selection_sha256"],
        "selected_configuration":
            result["selected_configuration"],
    }
    atomic_write_once(
        contract_path,
        stable_json_bytes(contract_record),
    )

    worker_evidence = {
        "phase": PHASE,
        "schema":
            "phase314b_r257_staged_resume4_3090_worker_evidence_v1",
        "worker_count": 2,
        "workers_exact": True,
        "comparison": comparison,
        "worker_result": result,
    }
    atomic_write_once(
        worker_path,
        stable_json_bytes(worker_evidence),
    )

    summary = {
        "phase": PHASE,
        "schema":
            "phase314b_r257_staged_resume4_3090_summary_v1",
        "verdict": "PASS",
        "scientific_status": "BLOCKED",
        "root_cause": result["root_cause"],
        "required_next_path":
            result["required_next_path"],
        "repository": repository,
        "test_gate_path": TEST_GATE,
        "test_gate_sha256":
            sha256_file(test_gate_path),
        "test_gate": test_gate,
        "contract_path":
            contract_path.relative_to(
                root
            ).as_posix(),
        "contract_file_sha256":
            sha256_file(contract_path),
        "worker_evidence_path":
            worker_path.relative_to(
                root
            ).as_posix(),
        "worker_evidence_sha256":
            sha256_file(worker_path),
        "workers_exact": comparison["exact"],
        "worker_result": result,
        "selected_configuration":
            result["selected_configuration"],
        "train_only_recommendation":
            result["train_only_recommendation"],
        "resume4_3090_correction_applied": True,
        "resume3_files_modified": False,
        "resume3_blocked_modified": False,
        "resume3_test_gate_modified": False,
        "control_replay_exact": True,
        "new_hyperparameter_candidate_run":
            result[
                "new_hyperparameter_candidate_run"
            ],
        "new_objective_variant_run":
            result["new_objective_variant_run"],
        "frozen_probe_accessed": False,
        "reverse_sampling_run": False,
        "full_stageb_repaired_model_trained": False,
        "formal_pilot_run": False,
        "checkpoint_saved": False,
        "weights_persisted": False,
        "prediction_tensor_persisted": False,
        "candidate_tensor_persisted": False,
        "npz_saved": False,
        "cache_saved": False,
        "formal_diffusion_training": False,
        "formal_reverse_sampling": False,
        "formal_idm_training": False,
        "action_diverse_data_collection": False,
        "candidate_execution": False,
        "deformable_ravens_executed": False,
        "phase4": False,
        "cps": False,
    }
    atomic_write_once(
        report_path,
        render_report(summary).encode("utf-8"),
    )
    atomic_write_once(
        summary_path,
        stable_json_bytes(summary),
    )
    shutil.rmtree(
        temporary_root,
        ignore_errors=True,
    )

    print(
        json.dumps(
            {
                "verdict": "PASS",
                "scientific_status": "BLOCKED",
                "root_cause":
                    summary["root_cause"],
                "required_next_path":
                    summary["required_next_path"],
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
                "summary": str(summary_path),
                "report": str(report_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
