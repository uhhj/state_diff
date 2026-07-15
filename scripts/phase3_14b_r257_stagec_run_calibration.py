#!/usr/bin/env python3
"""Run two isolated r2.5.7 Stage-C workers and seal evidence."""
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

from ccda_phase3.phase314b_r257_stagec_topk_quadratic import (
    BASE_EVIDENCE_COMMIT,
    EXPECTED_SUBMODULE_COMMIT,
    PHASE,
    atomic_write_once,
    compare_worker_results,
    git_output,
    load_json,
    sha256_file,
    stable_json_bytes,
)

TEST_GATE = (
    "reports/phase3_14b_r257_stagec_test_gate_summary.json"
)
DEFAULT_CONTRACT = (
    "reports/"
    "phase3_14b_r257_stagec_topk_quadratic_contract.json"
)
DEFAULT_WORKER = (
    "reports/phase3_14b_r257_stagec_worker_evidence.json"
)
DEFAULT_SUMMARY = (
    "reports/phase3_14b_r257_stagec_summary.json"
)
DEFAULT_REPORT = (
    "reports/phase3_14b_r257_stagec_report.md"
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
            "r2.5.7 Stage C requires Experiment1"
        )
    completed = subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            BASE_EVIDENCE_COMMIT,
            "HEAD",
        ],
        cwd=str(root),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "r2.5.7 Stage-B evidence is not an ancestor"
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
            "unexpected worktree paths before Stage-C run: "
            + repr(unexpected)
        )
    return {
        "branch": "Experiment1",
        "head": git_output(
            root,
            "rev-parse",
            "HEAD",
        ),
        "base_evidence_commit":
            BASE_EVIDENCE_COMMIT,
        "submodule_commit":
            EXPECTED_SUBMODULE_COMMIT,
    }


def render_report(
    summary: Dict[str, Any],
) -> str:
    result = summary["worker_result"]
    selection = result["selection"]
    selected = result["selected_configuration"]
    classification = result["classification"]
    calibration = result["calibration_contract"][
        "calibration"
    ]

    lines = [
        "# Phase3.14b-r2.5.7 Stage C Top-K Quadratic Calibration",
        "",
        f"- Audit verdict: `{summary['verdict']}`",
        f"- Scientific status: `{summary['scientific_status']}`",
        f"- Root cause: `{summary['root_cause']}`",
        (
            "- Required next path: `"
            f"{summary['required_next_path']}`"
        ),
        "",
        "## Immutable Stage-B binding",
        "",
        (
            "- Stage-B worker identity SHA256: `"
            + result["immutable_inputs"][
                "stageb_worker_identity_sha256"
            ]
            + "`"
        ),
        (
            "- Stage-B mechanism contract SHA256: `"
            + result["immutable_inputs"][
                "stageb_mechanism_contract_sha256"
            ]
            + "`"
        ),
        (
            "- Stage-B mechanism contract file SHA256: `"
            + result["immutable_inputs"][
                "stageb_mechanism_contract_file_sha256"
            ]
            + "`"
        ),
        "",
        "## Gradient-balanced contract",
        "",
        (
            "- Contract SHA256: `"
            + result["calibration_contract"][
                "contract_sha256"
            ]
            + "`"
        ),
        (
            "- Initial model SHA256: `"
            + calibration["initial_model_sha256"]
            + "`"
        ),
        (
            "- Lambda calibration SHA256: `"
            + calibration["calibration_sha256"]
            + "`"
        ),
        (
            "- Top-k variants: `"
            + ", ".join(
                str(value)
                for value in result[
                    "calibration_spec"
                ]["top_k_values"]
            )
            + "`"
        ),
        (
            "- Target gradient ratios: `"
            + ", ".join(
                f"{value:.6g}"
                for value in result[
                    "calibration_spec"
                ]["target_gradient_ratios"]
            )
            + "`"
        ),
        "- Frozen probe accessed: `false`",
        "- Reverse sampling run: `false`",
        "",
        "## Train-only candidates",
        "",
        (
            "| Candidate | K | Target ratio | Lambda | "
            "t10 excess reduction | t10 upper | "
            "Train-control ratio | Eligible |"
        ),
        (
            "|---|---:|---:|---:|---:|---:|---:|---:|"
        ),
    ]
    for record in selection["candidate_records"]:
        if record["candidate"] is None:
            lines.append(
                "| control | — | 0 | 0 | 0 | "
                f"{record['holdout_profiles']['10']['row_any_rate']:.9g} | "
                "1 | false |"
            )
            continue
        candidate = record["candidate"]
        lines.append(
            "| "
            f"{record['candidate_id']} | "
            f"{candidate['top_k']} | "
            f"{candidate['target_gradient_ratio']:.6g} | "
            f"{candidate['lambda_upper']:.12g} | "
            f"{record['continuous_response']['t10_mean_excess_reduction']:.9g} | "
            f"{record['binary_rate']['10']:.9g} | "
            f"{record['relative_to_control']['train_control_nmse_ratio']:.9g} | "
            f"{str(record['eligible_for_selection']).lower()} |"
        )

    lines.extend(
        [
            "",
            "## Selection",
            "",
            (
                "- Selection SHA256: `"
                + selection["selection_sha256"]
                + "`"
            ),
            (
                "- Selected configuration: `"
                + (
                    "none"
                    if selected is None
                    else selected["candidate_id"]
                )
                + "`"
            ),
            (
                "- Best continuous-response candidate: `"
                + classification[
                    "best_candidate_by_t10_continuous_excess"
                ]
                + "`"
            ),
            (
                "- Best t10 mean-excess reduction: `"
                f"{classification['best_t10_mean_excess_reduction']:.9g}`"
            ),
            (
                "- Best t10 binary upper row-any: `"
                f"{classification['best_t10_binary_upper_row_any']:.9g}`"
            ),
            "",
            "## Classification",
            "",
            (
                "- Primary failure locus: `"
                f"{classification['primary_failure_locus']}`"
            ),
            "",
            "## Boundary",
            "",
            "- The Stage-B evidence and mechanism contract were immutable.",
            "- One exact diffusion-only control and six gradient-balanced top-k quadratic candidates were trained on the Stage-A objective-training split.",
            "- Candidate lambda values were derived from the common initial-model gradient norms; Stage-A Huber lambda values were not reused.",
            "- Selection used only the 236-row grouped train-only holdout.",
            "- The frozen 126-row probe was not accessed.",
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
    test_gate_path = root / TEST_GATE
    for path in (
        contract_path,
        worker_path,
        summary_path,
        report_path,
    ):
        if path.exists():
            raise FileExistsError(
                "r2.5.7 Stage-C write-once "
                f"output exists: {path}"
            )

    repository = assert_repository(
        root,
        test_gate_path,
    )
    test_gate = load_json(test_gate_path)
    if test_gate.get("verdict") != "PASS":
        raise RuntimeError(
            "r2.5.7 Stage-C test gate is not PASS"
        )

    temporary_root = Path(
        tempfile.mkdtemp(
            prefix="phase314b_r257_stagec_",
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
            "phase3_14b_r257_stagec_worker.py"
        ),
        "--root",
        str(root),
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
            "isolated r2.5.7 Stage-C workers differ"
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
            "phase314b_r257_stagec_worker_evidence_v1",
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
            "phase314b_r257_stagec_summary_v1",
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
        "new_hyperparameter_candidate_run": True,
        "new_objective_variant_run": True,
        "control_replay_exact": True,
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
        render_report(summary).encode(
            "utf-8"
        ),
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
                "summary": str(summary_path),
                "report": str(report_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
