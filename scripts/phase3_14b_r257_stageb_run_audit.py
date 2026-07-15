#!/usr/bin/env python3
"""Run two isolated r2.5.7 Stage-B workers and seal evidence."""
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

from ccda_phase3.phase314b_r257_stageb_mechanism_audit import (
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
    "reports/phase3_14b_r257_stageb_test_gate_summary.json"
)
DEFAULT_CONTRACT = (
    "reports/phase3_14b_r257_stageb_mechanism_contract.json"
)
DEFAULT_WORKER = (
    "reports/phase3_14b_r257_stageb_worker_evidence.json"
)
DEFAULT_SUMMARY = (
    "reports/phase3_14b_r257_stageb_summary.json"
)
DEFAULT_REPORT = (
    "reports/phase3_14b_r257_stageb_report.md"
)


def resolve(root: Path, value: str) -> Path:
    candidate = Path(value)
    return (
        candidate.resolve()
        if candidate.is_absolute()
        else (root / candidate).resolve()
    )


def assert_repository(root: Path, test_gate: Path) -> Dict[str, Any]:
    if git_output(root, "branch", "--show-current") != "Experiment1":
        raise RuntimeError("r2.5.7 Stage B requires Experiment1")
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
            "r2.5.7 Stage-A evidence is not an ancestor"
        )

    submodule = root / "external/deformable-ravens"
    if git_output(submodule, "rev-parse", "HEAD") != EXPECTED_SUBMODULE_COMMIT:
        raise RuntimeError("DeformableRavens commit changed")
    if git_output(
        submodule,
        "status",
        "--porcelain",
        "--untracked-files=all",
    ):
        raise RuntimeError("DeformableRavens worktree is dirty")

    allowed = {test_gate.relative_to(root).as_posix()}
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
            "unexpected worktree paths before Stage-B run: "
            + repr(unexpected)
        )
    return {
        "branch": "Experiment1",
        "head": git_output(root, "rev-parse", "HEAD"),
        "base_evidence_commit": BASE_EVIDENCE_COMMIT,
        "submodule_commit": EXPECTED_SUBMODULE_COMMIT,
    }


def render_report(summary: Dict[str, Any]) -> str:
    result = summary["worker_result"]
    mechanism = result["mechanism_summary"]
    classification = result["classification"]

    lines = [
        "# Phase3.14b-r2.5.7 Stage B Upper-Objective Mechanism Audit",
        "",
        f"- Audit verdict: `{summary['verdict']}`",
        f"- Scientific status: `{summary['scientific_status']}`",
        f"- Root cause: `{summary['root_cause']}`",
        f"- Required next path: `{summary['required_next_path']}`",
        "",
        "## Immutable Stage-A binding",
        "",
        (
            "- Stage-A worker identity SHA256: `"
            + result["immutable_inputs"][
                "stagea_worker_identity_sha256"
            ]
            + "`"
        ),
        (
            "- Stage-A objective contract SHA256: `"
            + result["immutable_inputs"][
                "stagea_objective_contract_sha256"
            ]
            + "`"
        ),
        (
            "- Stage-A selection SHA256: `"
            + result["immutable_inputs"][
                "stagea_selection_sha256"
            ]
            + "`"
        ),
        "",
        "## Mechanism contract",
        "",
        (
            "- Contract SHA256: `"
            + result["mechanism_contract"]["contract_sha256"]
            + "`"
        ),
        (
            "- Training checkpoints: `"
            + ", ".join(
                str(value)
                for value in result["mechanism_contract"][
                    "completed_step_checkpoints"
                ]
            )
            + "`"
        ),
        (
            "- Fixed diagnostic timesteps: `"
            + ", ".join(
                str(value)
                for value in result["mechanism_contract"][
                    "diagnostic_timesteps"
                ]
            )
            + "`"
        ),
        "- Frozen probe accessed: `false`",
        "- Reverse sampling run: `false`",
        "",
        "## Candidate response",
        "",
        "| Lambda | Exact replay | t10 mean row-best excess | t10 p95 excess | Positive segments | Top position |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for record in result["candidate_audits"]:
        profile = record["holdout_profiles"]["10"]
        lines.append(
            "| "
            f"{record['lambda_upper']:.6g} | "
            f"{str(record['training_replay_exact']).lower()} | "
            f"{profile['row_best_max_log_excess']['mean']:.9g} | "
            f"{profile['row_best_max_log_excess']['p95']:.9g} | "
            f"{profile['candidate_positive_segment_count']['mean']:.9g} | "
            f"{profile['top_position_fraction']:.9g} |"
        )

    lines.extend(
        [
            "",
            "## Gradient mechanism",
            "",
            (
                "- Best lambda by continuous t10 response: `"
                f"{mechanism['best_lambda_by_t10_mean_excess']:.6g}`"
            ),
            (
                "- Best t10 mean-excess reduction: `"
                f"{mechanism['best_t10_mean_excess_reduction']:.9g}`"
            ),
            (
                "- Strongest-lambda scaled geometry/diffusion gradient ratio: `"
                f"{mechanism['strongest_scaled_gradient_ratio_median']:.9g}`"
            ),
            (
                "- Strongest-lambda gradient cosine: `"
                f"{mechanism['strongest_gradient_cosine_median']:.9g}`"
            ),
            (
                "- Strongest-lambda active-element conflict fraction: `"
                f"{mechanism['strongest_gradient_conflict_fraction_median']:.9g}`"
            ),
            (
                "- Strongest-lambda zero geometry-gradient fraction: `"
                f"{mechanism['strongest_zero_geometry_fraction_median']:.9g}`"
            ),
            (
                "- Strongest-lambda Huber linear fraction: `"
                f"{mechanism['strongest_huber_linear_fraction_median']:.9g}`"
            ),
            (
                "- Strongest-lambda row-max top-position fraction: `"
                f"{mechanism['strongest_rowmax_top_position_fraction_median']:.9g}`"
            ),
            "",
            "## Classification",
            "",
            (
                "- Primary failure locus: `"
                f"{classification['primary_failure_locus']}`"
            ),
            (
                "- Mechanism recommendation: `"
                f"{summary['mechanism_recommendation']}`"
            ),
            "",
            "## Boundary",
            "",
            "- The four Stage-A pilots were replayed exactly; no new lambda or objective variant was run.",
            "- Gradient diagnostics were read-only and the final model/optimizer/loss/gradient/exposure identities remained exact.",
            "- Selection holdout only was used for prediction-response attribution.",
            "- The frozen 126-row probe was not accessed.",
            "- Reverse sampling, formal pilot, IDM, data collection, candidate execution, DeformableRavens, Phase4 and CPS were not run.",
            "- No checkpoint, weights, prediction tensor, candidate tensor, NPZ, cache, image or video was persisted.",
            "- `selected_configuration=None` and `train_only_recommendation=None`.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument(
        "--python-bin",
        default="/miniforge3/envs/coord_bimanual/bin/python",
    )
    parser.add_argument("--contract", default=DEFAULT_CONTRACT)
    parser.add_argument("--worker-evidence", default=DEFAULT_WORKER)
    parser.add_argument("--summary", default=DEFAULT_SUMMARY)
    parser.add_argument("--report", default=DEFAULT_REPORT)
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
                f"r2.5.7 Stage-B write-once output exists: {path}"
            )

    repository = assert_repository(root, test_gate_path)
    test_gate = load_json(test_gate_path)
    if test_gate.get("verdict") != "PASS":
        raise RuntimeError("r2.5.7 Stage-B test gate is not PASS")

    temporary_root = Path(
        tempfile.mkdtemp(
            prefix="phase314b_r257_stageb_",
            dir="/tmp",
        )
    )
    worker_files = [
        temporary_root / "worker_1.json",
        temporary_root / "worker_2.json",
    ]
    command = [
        str(args.python_bin),
        str(root / "scripts/phase3_14b_r257_stageb_worker.py"),
        "--root",
        str(root),
    ]
    for worker_file in worker_files:
        subprocess.run(
            command + ["--output", str(worker_file)],
            cwd=str(root),
            check=True,
        )

    worker_results = [load_json(path) for path in worker_files]
    comparison = compare_worker_results(
        worker_results[0],
        worker_results[1],
    )
    if not comparison["exact"]:
        raise RuntimeError(
            "isolated r2.5.7 Stage-B workers differ"
        )
    result = worker_results[0]

    contract_record = {
        "phase": PHASE,
        **result["mechanism_contract"],
        "base_evidence_commit": BASE_EVIDENCE_COMMIT,
        "root_cause": result["root_cause"],
        "required_next_path":
            result["required_next_path"],
    }
    atomic_write_once(
        contract_path,
        stable_json_bytes(contract_record),
    )

    worker_evidence = {
        "phase": PHASE,
        "schema": "phase314b_r257_stageb_worker_evidence_v1",
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
        "schema": "phase314b_r257_stageb_summary_v1",
        "verdict": "PASS",
        "scientific_status": "BLOCKED",
        "root_cause": result["root_cause"],
        "required_next_path": result["required_next_path"],
        "repository": repository,
        "test_gate_path": TEST_GATE,
        "test_gate_sha256": sha256_file(test_gate_path),
        "test_gate": test_gate,
        "contract_path":
            contract_path.relative_to(root).as_posix(),
        "contract_file_sha256": sha256_file(contract_path),
        "worker_evidence_path":
            worker_path.relative_to(root).as_posix(),
        "worker_evidence_sha256": sha256_file(worker_path),
        "workers_exact": comparison["exact"],
        "worker_result": result,
        "selected_configuration": None,
        "train_only_recommendation": None,
        "mechanism_recommendation":
            result["mechanism_recommendation"],
        "training_performed": True,
        "training_is_exact_stagea_replay": True,
        "new_hyperparameter_candidate_run": False,
        "new_objective_variant_run": False,
        "frozen_probe_accessed": False,
        "reverse_sampling_run": False,
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
    shutil.rmtree(temporary_root, ignore_errors=True)

    print(
        json.dumps(
            {
                "verdict": "PASS",
                "scientific_status": "BLOCKED",
                "root_cause": summary["root_cause"],
                "required_next_path":
                    summary["required_next_path"],
                "workers_exact": summary["workers_exact"],
                "contract_file_sha256":
                    summary["contract_file_sha256"],
                "mechanism_recommendation":
                    summary["mechanism_recommendation"],
                "summary": str(summary_path),
                "report": str(report_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
