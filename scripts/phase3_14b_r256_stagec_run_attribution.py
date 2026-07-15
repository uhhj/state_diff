#!/usr/bin/env python3
"""Run two isolated Stage-C workers and seal write-once evidence."""
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

from ccda_phase3.phase314b_r256_stageb_cable_diffusion import (
    atomic_write_once,
    git_output,
    load_json,
    sha256_file,
    stable_json_bytes,
)
from ccda_phase3.phase314b_r256_stagec_reverse_attribution import (
    BASE_EVIDENCE_COMMIT,
    EXPECTED_SUBMODULE_COMMIT,
    PHASE,
    compare_worker_results,
)

TEST_GATE = "reports/phase3_14b_r256_stagec_test_gate_summary.json"
DEFAULT_WORKER_EVIDENCE = (
    "reports/phase3_14b_r256_stagec_worker_evidence.json"
)
DEFAULT_SUMMARY = "reports/phase3_14b_r256_stagec_summary.json"
DEFAULT_REPORT = "reports/phase3_14b_r256_stagec_report.md"


def resolve(root: Path, value: str) -> Path:
    candidate = Path(value)
    return (
        candidate.resolve()
        if candidate.is_absolute()
        else (root / candidate).resolve()
    )


def assert_repository(root: Path, test_gate: Path) -> Dict[str, Any]:
    if git_output(root, "branch", "--show-current") != "Experiment1":
        raise RuntimeError("Stage C requires Experiment1")
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
        raise RuntimeError("Stage-B evidence is not an ancestor")
    submodule = root / "external/deformable-ravens"
    if git_output(submodule, "rev-parse", "HEAD") != (
        EXPECTED_SUBMODULE_COMMIT
    ):
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
            "unexpected worktree paths before Stage-C run: "
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
    physical = result["physical_attribution"]
    reverse = physical["reverse_final"]
    segment = reverse["segment"]
    branch = result["branch_attribution"]
    classification = result["classification"]
    trace = physical["reverse_trace"]

    lines = [
        "# Phase3.14b-r2.5.6 Stage C Reverse Physical-Gate Attribution",
        "",
        f"- Audit verdict: `{summary['verdict']}`",
        f"- Scientific status: `{summary['scientific_status']}`",
        f"- Root cause: `{summary['root_cause']}`",
        f"- Required next path: `{summary['required_next_path']}`",
        "",
        "## Frozen Stage-B replay",
        "",
        f"- Isolated workers exact: `{str(summary['workers_exact']).lower()}`",
        (
            "- Model SHA256: `"
            f"{result['frozen_replay']['identity']['final_model_sha256']}`"
        ),
        (
            "- Reverse candidate SHA256: `"
            f"{result['frozen_replay']['identity']['reverse_candidate_sha256']}`"
        ),
        "- Stage-B model, scheduler, split, thresholds and K were unchanged.",
        "",
        "## Physical attribution",
        "",
        (
            "- Training-ground-truth segment candidate rate: `"
            f"{physical['training_ground_truth']['segment']['candidate_all_pass_rate']:.9g}`"
        ),
        (
            "- Probe-ground-truth segment candidate rate: `"
            f"{physical['probe_ground_truth']['segment']['candidate_all_pass_rate']:.9g}`"
        ),
        (
            "- Reverse segment element pass rate: `"
            f"{segment['element_pass_rate']:.9g}`"
        ),
        (
            "- Reverse segment candidate-all pass rate: `"
            f"{segment['candidate_all_pass_rate']:.9g}`"
        ),
        (
            "- Reverse lower-bound violations: `"
            f"{segment['lower_violation_count']}`"
        ),
        (
            "- Reverse upper-bound violations: `"
            f"{segment['upper_violation_count']}`"
        ),
        (
            "- Reverse combined physical candidate rate: `"
            f"{reverse['combined_candidate_rate']:.9g}`"
        ),
        "",
        "## Reverse trace: predicted-x0 segment candidate rate",
        "",
    ]
    for timestep in ("99", "75", "50", "25", "10", "0"):
        item = trace[timestep]["predicted_x0"]["segment"]
        lines.append(
            f"- `t={timestep}`: "
            f"`{item['candidate_all_pass_rate']:.9g}` "
            f"(element `{item['element_pass_rate']:.9g}`)"
        )
    lines.extend(
        [
            "",
            "## Branch attribution",
            "",
            (
                "- Historical K=8 support: `"
                f"{branch['historical_k8']:.9g}`"
            ),
            (
                "- Pair-bootstrap 95% CI: `"
                f"{branch['pair_bootstrap_ci']['ci95']}`"
            ),
            (
                "- Physical-valid-only branch audit available: `"
                f"{str(branch['physical_valid_only']['available']).lower()}`"
            ),
            (
                "- Historical classifier masked simultaneous physical failure: `"
                f"{str(classification['historical_branch_failure_masked_physical_failure']).lower()}`"
            ),
            "",
            "## Boundary",
            "",
            "- Only the train/full-horizon view was used.",
            "- No validation or formal-test target was opened.",
            "- The Stage-B model and candidates were reproduced exactly.",
            "- No model, scheduler, objective, threshold or candidate count was changed.",
            "- No checkpoint, weights, candidate tensor, NPZ or video was persisted.",
            "- No formal diffusion/reverse, IDM, candidate execution, Phase4 or CPS was run.",
            "- No action-diverse data was collected.",
            "- `train_only_recommendation=None` and `selected_configuration=None`.",
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
            raise FileExistsError(f"write-once output exists: {path}")

    repository = assert_repository(root, test_gate_path)
    test_gate = load_json(test_gate_path)
    if test_gate.get("verdict") != "PASS":
        raise RuntimeError("Stage-C test gate is not PASS")

    temporary_root = Path(
        tempfile.mkdtemp(prefix="phase314b_r256_stagec_", dir="/tmp")
    )
    worker_paths = [
        temporary_root / "worker_1.json",
        temporary_root / "worker_2.json",
    ]
    command = [
        str(args.python_bin),
        str(root / "scripts/phase3_14b_r256_stagec_worker.py"),
        "--root",
        str(root),
    ]
    for path in worker_paths:
        subprocess.run(
            command + ["--output", str(path)],
            cwd=str(root),
            check=True,
        )
    workers = [load_json(path) for path in worker_paths]
    comparison = compare_worker_results(workers[0], workers[1])
    if not comparison["exact"]:
        raise RuntimeError("isolated Stage-C workers differ")

    result = workers[0]
    worker_evidence = {
        "phase": PHASE,
        "schema": "phase314b_r256_stagec_worker_evidence_v1",
        "worker_count": 2,
        "workers_exact": True,
        "comparison": comparison,
        "worker_result": result,
    }
    atomic_write_once(
        worker_evidence_path,
        stable_json_bytes(worker_evidence),
    )
    summary = {
        "phase": PHASE,
        "schema": "phase314b_r256_stagec_summary_v1",
        "verdict": "PASS",
        "scientific_status": "BLOCKED",
        "root_cause": result["root_cause"],
        "required_next_path": result["required_next_path"],
        "repository": repository,
        "test_gate_path": TEST_GATE,
        "test_gate_sha256": sha256_file(test_gate_path),
        "test_gate": test_gate,
        "worker_evidence_path":
            worker_evidence_path.relative_to(root).as_posix(),
        "worker_evidence_sha256":
            sha256_file(worker_evidence_path),
        "workers_exact": comparison["exact"],
        "worker_result": result,
        "historical_stageb_evidence_modified": False,
        "historical_root_cause_rewritten": False,
        "frozen_stageb_reproduced_exactly": True,
        "primary_failure_locus":
            result["classification"]["primary_failure_locus"],
        "simultaneous_historical_failures":
            result["classification"]["simultaneous_historical_failures"],
        "validation_targets_used": False,
        "formal_test_targets_used": False,
        "checkpoint_saved": False,
        "weights_persisted": False,
        "prediction_tensor_persisted": False,
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
                "summary": str(summary_path),
                "report": str(report_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
