#!/usr/bin/env python3
"""Run isolated Stage-B workers and seal write-once diagnostic evidence."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ccda_phase3.phase314b_r256_stageb_cable_diffusion import (
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

TEST_GATE = "reports/phase3_14b_r256_stageb_test_gate_summary.json"
DEFAULT_WORKER_EVIDENCE = (
    "reports/phase3_14b_r256_stageb_worker_evidence.json"
)
DEFAULT_SUMMARY = "reports/phase3_14b_r256_stageb_summary.json"
DEFAULT_REPORT = "reports/phase3_14b_r256_stageb_report.md"


def resolve(root: Path, value: str) -> Path:
    candidate = Path(value)
    return (
        candidate.resolve()
        if candidate.is_absolute()
        else (root / candidate).resolve()
    )


def assert_repository(root: Path, test_gate: Path) -> Dict[str, Any]:
    if git_output(root, "branch", "--show-current") != "Experiment1":
        raise RuntimeError("Stage B requires Experiment1")
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
        raise RuntimeError("r2.5.6 Stage-A evidence is not an ancestor")
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
    evaluation = summary["worker_result"]["evaluation"]
    reverse = evaluation["probe_reverse"]
    branch = evaluation["branch"]
    physical = evaluation["physical"]
    lines = [
        "# Phase3.14b-r2.5.6 Stage B Cable-only Diffusion Diagnostic",
        "",
        f"- Audit verdict: `{summary['verdict']}`",
        f"- Scientific status: `{summary['scientific_status']}`",
        f"- Root cause: `{summary['root_cause']}`",
        f"- Required next path: `{summary['required_next_path']}`",
        "",
        "## Determinism",
        "",
        f"- Isolated workers exact: `{str(summary['workers_exact']).lower()}`",
        (
            "- Final model SHA256: "
            f"`{summary['worker_result']['identity']['final_model_sha256']}`"
        ),
        "",
        "## Train-only diagnostic",
        "",
        (
            "- Train fixed-noise NMSE: "
            f"`{evaluation['train_control']['normalized_mse']:.9g}`"
        ),
        (
            "- Reverse best-of-K NMSE: "
            f"`{reverse['best_of_k_normalized_mse']:.9g}`"
        ),
        (
            "- Best simple baseline NMSE: "
            f"`{reverse['best_baseline_normalized_mse']:.9g}`"
        ),
        (
            "- Relative improvement: "
            f"`{reverse['relative_improvement']:.9g}`"
        ),
        (
            "- Own-branch support rate: "
            f"`{branch['row_own_branch_support_rate']:.9g}`"
        ),
        (
            "- Physical candidate rate: "
            f"`{physical['candidate_rate']:.9g}`"
        ),
        "",
        "## Boundary",
        "",
        "- Only the train/full-horizon state-v3 view was used.",
        "- The target was `[4,48]` ordered cable XY; robot future was absent.",
        "- No validation/formal-test target, checkpoint, weights, or prediction tensor was persisted.",
        "- Reverse sampling was diagnostic only.",
        "- No formal diffusion, IDM, candidate execution, Phase4, or CPS was run.",
        "- Action-diverse IDM data was not collected in this stage.",
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
    parser.add_argument(
        "--worker-evidence",
        default=DEFAULT_WORKER_EVIDENCE,
    )
    parser.add_argument("--summary", default=DEFAULT_SUMMARY)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    worker_evidence_path = resolve(root, args.worker_evidence)
    summary_path = resolve(root, args.summary)
    report_path = resolve(root, args.report)
    test_gate_path = root / TEST_GATE
    for path in (worker_evidence_path, summary_path, report_path):
        if path.exists():
            raise FileExistsError(
                f"Stage-B write-once output exists: {path}"
            )

    repository = assert_repository(root, test_gate_path)
    test_gate = load_json(test_gate_path)
    if test_gate.get("verdict") != "PASS":
        raise RuntimeError("Stage-B test gate is not PASS")

    temporary_root = Path(
        tempfile.mkdtemp(prefix="phase314b_r256_stageb_", dir="/tmp")
    )
    worker_paths = [
        temporary_root / "worker_1.json",
        temporary_root / "worker_2.json",
    ]
    command = [
        str(args.python_bin),
        str(root / "scripts/phase3_14b_r256_stageb_worker.py"),
        "--root",
        str(root),
    ]
    for path in worker_paths:
        subprocess.run(
            command + ["--output", str(path)],
            cwd=str(root),
            check=True,
        )
    worker_results = [load_json(path) for path in worker_paths]
    comparison = compare_worker_results(
        worker_results[0],
        worker_results[1],
    )
    if not comparison["exact"]:
        raise RuntimeError(
            "isolated cable-only diffusion workers differ"
        )
    worker_result = worker_results[0]
    worker_evidence = {
        "phase": PHASE,
        "schema": "phase314b_r256_stageb_worker_evidence_v1",
        "worker_count": 2,
        "workers_exact": True,
        "comparison": comparison,
        "worker_result": worker_result,
    }
    atomic_write_once(
        worker_evidence_path,
        stable_json_bytes(worker_evidence),
    )
    summary = {
        "phase": PHASE,
        "schema": "phase314b_r256_stageb_summary_v1",
        "verdict": "PASS",
        "scientific_status": "BLOCKED",
        "root_cause": worker_result["root_cause"],
        "required_next_path": worker_result["required_next_path"],
        "repository": repository,
        "test_gate_path": TEST_GATE,
        "test_gate_sha256": sha256_file(test_gate_path),
        "test_gate": test_gate,
        "worker_evidence_path":
            worker_evidence_path.relative_to(root).as_posix(),
        "worker_evidence_sha256":
            sha256_file(worker_evidence_path),
        "workers_exact": comparison["exact"],
        "worker_result": worker_result,
        "cable_only_diffusion_diagnostic_supported":
            worker_result["evaluation"]["diagnostic_pass"],
        "action_diverse_idm_data_still_required": True,
        "validation_targets_used": False,
        "formal_test_targets_used": False,
        "checkpoint_saved": False,
        "weights_persisted": False,
        "prediction_tensor_persisted": False,
        "diffusion_training": True,
        "formal_diffusion_training": False,
        "reverse_sampling": True,
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
                "summary": str(summary_path),
                "report": str(report_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
