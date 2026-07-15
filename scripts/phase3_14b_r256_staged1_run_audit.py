#!/usr/bin/env python3
"""Run two isolated Stage-D.1 workers and seal write-once evidence."""
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

from ccda_phase3.phase314b_r256_staged1_asymmetric_gate_audit import (
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

TEST_GATE = "reports/phase3_14b_r256_staged1_test_gate_summary.json"
DEFAULT_AUDIT = "reports/phase3_14b_r256_staged1_gate_audit.json"
DEFAULT_WORKER = "reports/phase3_14b_r256_staged1_worker_evidence.json"
DEFAULT_SUMMARY = "reports/phase3_14b_r256_staged1_summary.json"
DEFAULT_REPORT = "reports/phase3_14b_r256_staged1_report.md"


def resolve(root: Path, value: str) -> Path:
    candidate = Path(value)
    return (
        candidate.resolve()
        if candidate.is_absolute()
        else (root / candidate).resolve()
    )


def assert_repository(root: Path, test_gate: Path) -> Dict[str, Any]:
    if git_output(root, "branch", "--show-current") != "Experiment1":
        raise RuntimeError("Stage D.1 requires Experiment1")
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
        raise RuntimeError("Stage-D evidence is not an ancestor")

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
            "unexpected worktree paths before Stage-D.1 run: "
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
    driver = result["threshold_driver_attribution"]
    classification = result["classification"]
    reverse = result["gate_evaluation"]["reverse_candidates"]
    branch = result["physical_branch_attribution"]
    bonf = result["audit_contract"]["candidate_gate_definitions"][
        "bonferroni_asymmetric"
    ]
    lines = [
        "# Phase3.14b-r2.5.6 Stage D.1 Asymmetric Gate Audit",
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
        "## Directional threshold attribution",
        "",
        (
            "- Stage-D joint threshold: `"
            f"{driver['directional_thresholds']['joint']:.9g}`"
        ),
        (
            "- Stage-D lower threshold: `"
            f"{driver['directional_thresholds']['lower']:.9g}`"
        ),
        (
            "- Stage-D upper threshold: `"
            f"{driver['directional_thresholds']['upper']:.9g}`"
        ),
        (
            "- Joint-to-upper ratio: `"
            f"{driver['directional_thresholds']['joint_to_upper_ratio']:.9g}`"
        ),
        (
            "- Scale-floor saturation rate: `"
            f"{driver['reference_scale']['floor_saturation_rate']:.9g}`"
        ),
        (
            "- Lowest lower-tail driver ratio: `"
            f"{driver['lowest_driver_ratio']:.9g}`"
        ),
        "",
        "## Candidate directional gate",
        "",
        (
            "- Bonferroni lower threshold: `"
            f"{bonf['lower_threshold']:.9g}`"
        ),
        (
            "- Bonferroni upper threshold: `"
            f"{bonf['upper_threshold']:.9g}`"
        ),
        (
            "- Bonferroni calibration group acceptance: `"
            f"{classification['bonferroni_calibration_group_acceptance']:.9g}`"
        ),
        (
            "- Bonferroni probe row acceptance: `"
            f"{classification['bonferroni_probe_row_acceptance']:.9g}`"
        ),
        (
            "- Bonferroni reverse combined row-any: `"
            f"{classification['bonferroni_reverse_combined_row_any']:.9g}`"
        ),
        "",
        "## Frozen physical-valid branch",
        "",
        (
            "- Stage-D joint K=8 support: `"
            + str(
                branch["stage_d_joint"]["k8"][
                    "support_rate_among_eligible"
                ]
            )
            + "`"
        ),
        (
            "- Naive asymmetric K=8 support: `"
            + str(
                branch["stage_d_naive_asymmetric"]["k8"][
                    "support_rate_among_eligible"
                ]
            )
            + "`"
        ),
        (
            "- Bonferroni asymmetric K=8 support: `"
            + str(
                branch["bonferroni_asymmetric"]["k8"][
                    "support_rate_among_eligible"
                ]
            )
            + "`"
        ),
        "",
        "## Boundary",
        "",
        "- The Stage-B model, scheduler, split, normalization, seed, objective and K were unchanged.",
        "- The Stage-D gate file and all historical evidence were not modified.",
        "- No candidate gate was selected and no threshold was changed.",
        "- Probe targets and model candidates were not used to fit the candidate thresholds.",
        "- DeformableRavens was not modified or executed.",
        "- No formal pilot, formal diffusion/reverse, IDM, action-diverse collection, candidate execution, Phase4 or CPS was run.",
        "- No checkpoint, tensor, NPZ, image or video was persisted.",
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
    parser.add_argument("--audit", default=DEFAULT_AUDIT)
    parser.add_argument("--worker-evidence", default=DEFAULT_WORKER)
    parser.add_argument("--summary", default=DEFAULT_SUMMARY)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    audit_path = resolve(root, args.audit)
    worker_path = resolve(root, args.worker_evidence)
    summary_path = resolve(root, args.summary)
    report_path = resolve(root, args.report)
    test_gate_path = root / TEST_GATE
    for path in (audit_path, worker_path, summary_path, report_path):
        if path.exists():
            raise FileExistsError(
                f"Stage-D.1 write-once output exists: {path}"
            )

    repository = assert_repository(root, test_gate_path)
    test_gate = load_json(test_gate_path)
    if test_gate.get("verdict") != "PASS":
        raise RuntimeError("Stage-D.1 test gate is not PASS")

    temporary_root = Path(
        tempfile.mkdtemp(prefix="phase314b_r256_staged1_", dir="/tmp")
    )
    worker_files = [
        temporary_root / "worker_1.json",
        temporary_root / "worker_2.json",
    ]
    command = [
        str(args.python_bin),
        str(root / "scripts/phase3_14b_r256_staged1_worker.py"),
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
            "isolated Stage-D.1 workers differ"
        )
    result = worker_results[0]

    audit_record = {
        "phase": PHASE,
        **result["audit_contract"],
        "classification": result["classification"],
        "threshold_driver_attribution":
            result["threshold_driver_attribution"],
        "ground_truth_ratio_audit":
            result["ground_truth_ratio_audit"],
        "gate_evaluation": result["gate_evaluation"],
        "physical_branch_attribution":
            result["physical_branch_attribution"],
    }
    atomic_write_once(audit_path, stable_json_bytes(audit_record))

    worker_evidence = {
        "phase": PHASE,
        "schema": "phase314b_r256_staged1_worker_evidence_v1",
        "worker_count": 2,
        "workers_exact": True,
        "comparison": comparison,
        "worker_result": result,
    }
    atomic_write_once(worker_path, stable_json_bytes(worker_evidence))

    summary = {
        "phase": PHASE,
        "schema": "phase314b_r256_staged1_summary_v1",
        "verdict": "PASS",
        "scientific_status": "BLOCKED",
        "root_cause": result["root_cause"],
        "required_next_path": result["required_next_path"],
        "repository": repository,
        "test_gate_path": TEST_GATE,
        "test_gate_sha256": sha256_file(test_gate_path),
        "test_gate": test_gate,
        "audit_path": audit_path.relative_to(root).as_posix(),
        "audit_sha256": sha256_file(audit_path),
        "worker_evidence_path": worker_path.relative_to(root).as_posix(),
        "worker_evidence_sha256": sha256_file(worker_path),
        "workers_exact": comparison["exact"],
        "worker_result": result,
        "gate_selected": False,
        "threshold_changed": False,
        "model_or_branch_repaired": False,
        "deformable_ravens_modified": False,
        "formal_pilot_run": False,
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
                "audit_sha256": summary["audit_sha256"],
                "summary": str(summary_path),
                "report": str(report_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
