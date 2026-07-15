#!/usr/bin/env python3
"""Run two isolated Stage-D recalibration workers and seal evidence."""
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

from ccda_phase3.phase314b_r256_staged_segment_recalibration import (
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

TEST_GATE = "reports/phase3_14b_r256_staged_test_gate_summary.json"
DEFAULT_GATE = "reports/phase3_14b_r256_staged_segment_gate_contract.json"
DEFAULT_WORKER = "reports/phase3_14b_r256_staged_worker_evidence.json"
DEFAULT_SUMMARY = "reports/phase3_14b_r256_staged_summary.json"
DEFAULT_REPORT = "reports/phase3_14b_r256_staged_report.md"


def resolve(root: Path, value: str) -> Path:
    candidate = Path(value)
    return (
        candidate.resolve()
        if candidate.is_absolute()
        else (root / candidate).resolve()
    )


def assert_repository(root: Path, test_gate: Path) -> Dict[str, Any]:
    if git_output(root, "branch", "--show-current") != "Experiment1":
        raise RuntimeError("Stage D requires Experiment1")
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
        raise RuntimeError("Stage-C evidence is not an ancestor")
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
            "unexpected worktree paths before Stage-D run: "
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
    calibration = result["ground_truth_calibration"]
    predictions = result["frozen_prediction_evaluation"]
    classification = result["classification"]
    branch = result["physical_branch_attribution"]["final_k8"]
    lines = [
        "# Phase3.14b-r2.5.6 Stage D Segment-Gate Recalibration",
        "",
        f"- Audit verdict: `{summary['verdict']}`",
        f"- Scientific status: `{summary['scientific_status']}`",
        f"- Root cause: `{summary['root_cause']}`",
        f"- Required next path: `{summary['required_next_path']}`",
        "",
        "## Frozen replay",
        "",
        f"- Isolated workers exact: `{str(summary['workers_exact']).lower()}`",
        (
            "- Model SHA256: `"
            + result["frozen_replay"]["identity"]["final_model_sha256"]
            + "`"
        ),
        (
            "- Reverse candidate SHA256: `"
            + result["frozen_replay"]["identity"][
                "reverse_candidate_sha256"
            ]
            + "`"
        ),
        "",
        "## Recalibrated gate",
        "",
        (
            "- Gate contract SHA256: `"
            + result["gate_contract"]["contract_sha256"]
            + "`"
        ),
        (
            "- Joint robust-z threshold: `"
            f"{result['gate_contract']['joint_threshold']:.9g}`"
        ),
        (
            "- Calibration group acceptance: `"
            f"{classification['calibration_group_acceptance_rate']:.9g}`"
        ),
        (
            "- Probe ground-truth row acceptance: `"
            f"{classification['probe_segment_row_acceptance_rate']:.9g}`"
        ),
        (
            "- Probe ground-truth group acceptance: `"
            f"{classification['probe_segment_group_acceptance_rate']:.9g}`"
        ),
        (
            "- Probe minimum condition acceptance: `"
            f"{classification['probe_minimum_condition_acceptance_rate']:.9g}`"
        ),
        "",
        "## Frozen predictions under recalibrated gate",
        "",
        (
            "- One-step t=10 segment acceptance: `"
            f"{classification['one_step_t10_segment_acceptance_rate']:.9g}`"
        ),
        (
            "- Reverse segment row-any rate: `"
            f"{classification['reverse_segment_row_any_rate']:.9g}`"
        ),
        (
            "- Reverse combined row-any rate: `"
            f"{classification['reverse_combined_row_any_rate']:.9g}`"
        ),
        (
            "- Physical-valid branch eligible-row rate: `"
            f"{branch['eligible_row_rate']:.9g}`"
        ),
        (
            "- Physical-valid branch support among eligible: `"
            + (
                "unavailable"
                if branch["support_rate_among_eligible"] is None
                else f"{branch['support_rate_among_eligible']:.9g}"
            )
            + "`"
        ),
        "",
        "## Boundary",
        "",
        "- The Stage-B model, scheduler, split, normalization, objective, seed and K were unchanged.",
        "- Gate fitting used only grouped diagnostic-training ground truth.",
        "- Probe targets and model candidates were not used to fit or select the gate.",
        "- The condition label was not used to fit the gate.",
        "- No checkpoint, tensor, candidate pool, NPZ, image or video was persisted.",
        "- No formal diffusion/reverse, IDM, candidate execution, Phase4 or CPS was run.",
        "- No action-diverse data was collected.",
        "- `train_only_recommendation=None` and `selected_configuration=None`.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument(
        "--python-bin",
        default="/miniforge3/envs/coord_bimanual/bin/python",
    )
    parser.add_argument("--gate-contract", default=DEFAULT_GATE)
    parser.add_argument("--worker-evidence", default=DEFAULT_WORKER)
    parser.add_argument("--summary", default=DEFAULT_SUMMARY)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    gate_path = resolve(root, args.gate_contract)
    worker_path = resolve(root, args.worker_evidence)
    summary_path = resolve(root, args.summary)
    report_path = resolve(root, args.report)
    test_gate_path = root / TEST_GATE
    for path in (gate_path, worker_path, summary_path, report_path):
        if path.exists():
            raise FileExistsError(
                f"Stage-D write-once output exists: {path}"
            )

    repository = assert_repository(root, test_gate_path)
    test_gate = load_json(test_gate_path)
    if test_gate.get("verdict") != "PASS":
        raise RuntimeError("Stage-D test gate is not PASS")

    temporary_root = Path(
        tempfile.mkdtemp(prefix="phase314b_r256_staged_", dir="/tmp")
    )
    worker_files = [
        temporary_root / "worker_1.json",
        temporary_root / "worker_2.json",
    ]
    command = [
        str(args.python_bin),
        str(root / "scripts/phase3_14b_r256_staged_worker.py"),
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
            "isolated Stage-D recalibration workers differ"
        )
    result = worker_results[0]

    gate_contract = {
        "phase": PHASE,
        **result["gate_contract"],
        "base_evidence_commit": BASE_EVIDENCE_COMMIT,
        "frozen_model_sha256":
            result["frozen_replay"]["identity"]["final_model_sha256"],
        "frozen_reverse_candidate_sha256":
            result["frozen_replay"]["identity"][
                "reverse_candidate_sha256"
            ],
    }
    atomic_write_once(gate_path, stable_json_bytes(gate_contract))

    worker_evidence = {
        "phase": PHASE,
        "schema": "phase314b_r256_staged_worker_evidence_v1",
        "worker_count": 2,
        "workers_exact": True,
        "comparison": comparison,
        "worker_result": result,
    }
    atomic_write_once(worker_path, stable_json_bytes(worker_evidence))

    summary = {
        "phase": PHASE,
        "schema": "phase314b_r256_staged_summary_v1",
        "verdict": "PASS",
        "scientific_status": "BLOCKED",
        "root_cause": result["root_cause"],
        "required_next_path": result["required_next_path"],
        "repository": repository,
        "test_gate_path": TEST_GATE,
        "test_gate_sha256": sha256_file(test_gate_path),
        "test_gate": test_gate,
        "gate_contract_path": gate_path.relative_to(root).as_posix(),
        "gate_contract_sha256": sha256_file(gate_path),
        "worker_evidence_path": worker_path.relative_to(root).as_posix(),
        "worker_evidence_sha256": sha256_file(worker_path),
        "workers_exact": comparison["exact"],
        "worker_result": result,
        "gate_recalibrated": True,
        "model_or_branch_repaired": False,
        "formal_pilot_run": False,
        "action_diverse_idm_data_still_required": True,
        "checkpoint_saved": False,
        "prediction_tensor_persisted": False,
        "candidate_tensor_persisted": False,
        "npz_saved": False,
        "formal_diffusion_training": False,
        "formal_reverse_sampling": False,
        "formal_idm_training": False,
        "action_diverse_data_collection": False,
        "candidate_execution": False,
        "phase4": False,
        "cps": False,
        "train_only_recommendation": None,
        "selected_configuration": None,
    }
    atomic_write_once(report_path, render_report(summary).encode("utf-8"))
    atomic_write_once(summary_path, stable_json_bytes(summary))
    shutil.rmtree(temporary_root, ignore_errors=True)

    print(
        json.dumps(
            {
                "verdict": "PASS",
                "scientific_status": "BLOCKED",
                "root_cause": summary["root_cause"],
                "required_next_path": summary["required_next_path"],
                "workers_exact": summary["workers_exact"],
                "gate_contract_sha256":
                    summary["gate_contract_sha256"],
                "summary": str(summary_path),
                "report": str(report_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
