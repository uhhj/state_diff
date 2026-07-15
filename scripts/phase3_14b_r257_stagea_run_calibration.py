#!/usr/bin/env python3
"""Run two isolated r2.5.7 Stage-A workers and seal evidence."""
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

from ccda_phase3.phase314b_r257_stagea_upper_objective import (
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
    "reports/phase3_14b_r257_stagea_test_gate_summary.json"
)
DEFAULT_CONTRACT = (
    "reports/phase3_14b_r257_stagea_upper_objective_contract.json"
)
DEFAULT_WORKER = (
    "reports/phase3_14b_r257_stagea_worker_evidence.json"
)
DEFAULT_SUMMARY = (
    "reports/phase3_14b_r257_stagea_summary.json"
)
DEFAULT_REPORT = (
    "reports/phase3_14b_r257_stagea_report.md"
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
        raise RuntimeError("r2.5.7 Stage A requires Experiment1")
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
            "Stage-D.3 Resume1 evidence is not an ancestor"
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
            "unexpected worktree paths before r2.5.7 Stage-A run: "
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
    selection = result["train_only_selection"]
    selected = result["selected_configuration"]
    classification = result["classification"]

    lines = [
        "# Phase3.14b-r2.5.7 Stage A Upper-Expansion Objective Calibration",
        "",
        f"- Audit verdict: `{summary['verdict']}`",
        f"- Scientific status: `{summary['scientific_status']}`",
        f"- Root cause: `{summary['root_cause']}`",
        f"- Required next path: `{summary['required_next_path']}`",
        "",
        "## Objective contract",
        "",
        (
            "- Contract SHA256: `"
            + result["objective_contract"]["contract_sha256"]
            + "`"
        ),
        (
            "- Frozen upper-gate SHA256: `"
            + result["objective_contract"][
                "upper_gate_contract_sha256"
            ]
            + "`"
        ),
        (
            "- Frozen upper threshold: `"
            f"{result['objective_contract']['upper_threshold']:.12g}`"
        ),
        (
            "- Candidate lambdas: `"
            + ", ".join(
                f"{value:.6g}"
                for value in result["objective_spec"][
                    "candidate_lambdas"
                ]
            )
            + "`"
        ),
        "- Lower XY score used for validity: `false`",
        "",
        "## Leakage-controlled selection",
        "",
        (
            "- Objective-train rows / groups: `"
            f"{result['split']['objective_train_rows']} / "
            f"{result['split']['objective_train_groups']}`"
        ),
        (
            "- Selection-holdout rows / groups: `"
            f"{result['split']['selection_holdout_rows']} / "
            f"{result['split']['selection_holdout_groups']}`"
        ),
        (
            "- Selection SHA256: `"
            + selection["selection_sha256"]
            + "`"
        ),
        (
            "- Frozen probe accessed during selection: `"
            f"{str(selection['probe_accessed_during_selection']).lower()}`"
        ),
        (
            "- Selected lambda: `"
            + (
                "none"
                if selected is None
                else f"{selected['lambda_upper']:.6g}"
            )
            + "`"
        ),
        "",
        "## Pilot candidates",
        "",
        "| Lambda | Feasible | Train-control NMSE | t10 upper | t25 upper | t50 upper |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for record in selection["candidate_records"]:
        one = record["one_step"]["timesteps"]
        lines.append(
            "| "
            f"{record['lambda_upper']:.6g} | "
            f"{str(record['feasible_nonzero_configuration']).lower()} | "
            f"{record['train_control']['normalized_mse']:.9g} | "
            f"{one['10']['upper']['upper_segment']['row_any_rate']:.9g} | "
            f"{one['25']['upper']['upper_segment']['row_any_rate']:.9g} | "
            f"{one['50']['upper']['upper_segment']['row_any_rate']:.9g} |"
        )

    final = result["final_frozen_probe_audit"]
    lines.extend(
        [
            "",
            "## Final frozen-probe audit",
            "",
            (
                "- Performed: `"
                f"{str(final['performed']).lower()}`"
            ),
        ]
    )
    if final["performed"]:
        repaired = final["repaired"]
        baseline = final["baseline"]
        lines.extend(
            [
                (
                    "- Repaired model SHA256: `"
                    + repaired["training"]["final_model_sha256"]
                    + "`"
                ),
                (
                    "- Baseline / repaired train-control NMSE: `"
                    f"{baseline['train_control']['normalized_mse']:.9g} / "
                    f"{repaired['train_control']['normalized_mse']:.9g}`"
                ),
                (
                    "- Repaired t=10 upper row-any: `"
                    f"{repaired['one_step']['timesteps']['10']['upper']['upper_segment']['row_any_rate']:.9g}`"
                ),
                (
                    "- Repaired reverse upper row-any: `"
                    f"{repaired['reverse']['final']['upper_segment']['row_any_rate']:.9g}`"
                ),
                (
                    "- Repaired reverse combined row-any: `"
                    f"{repaired['reverse']['final']['combined']['row_any_rate']:.9g}`"
                ),
                (
                    "- Repaired K=8 eligible/support: `"
                    f"{repaired['reverse']['physical_branch']['k8']['eligible_row_rate']:.9g} / "
                    + (
                        "unavailable"
                        if repaired["reverse"]["physical_branch"]["k8"][
                            "support_rate_among_eligible"
                        ]
                        is None
                        else f"{repaired['reverse']['physical_branch']['k8']['support_rate_among_eligible']:.9g}"
                    )
                    + "`"
                ),
            ]
        )

    lines.extend(
        [
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
            "- r2.5.6 Stage D.3 Resume1 evidence was immutable.",
            "- The model continues to predict direct normalized x0.",
            "- Hyperparameter selection used only a grouped holdout inside the frozen Stage-B training partition.",
            "- Frozen probe targets were not accessed unless a nonzero configuration passed train-only selection.",
            "- The upper gate, scheduler, K, formal split and DeformableRavens were unchanged.",
            "- No checkpoint, tensor, NPZ, cache, image or video was persisted.",
            "- No formal pilot, formal diffusion/reverse, IDM, data collection, candidate execution, Phase4 or CPS was run.",
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
                f"r2.5.7 Stage-A write-once output exists: {path}"
            )

    repository = assert_repository(root, test_gate_path)
    test_gate = load_json(test_gate_path)
    if test_gate.get("verdict") != "PASS":
        raise RuntimeError("r2.5.7 Stage-A test gate is not PASS")

    temporary_root = Path(
        tempfile.mkdtemp(
            prefix="phase314b_r257_stagea_",
            dir="/tmp",
        )
    )
    worker_files = [
        temporary_root / "worker_1.json",
        temporary_root / "worker_2.json",
    ]
    command = [
        str(args.python_bin),
        str(root / "scripts/phase3_14b_r257_stagea_worker.py"),
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
            "isolated r2.5.7 Stage-A workers differ"
        )
    result = worker_results[0]

    contract_record = {
        "phase": PHASE,
        **result["objective_contract"],
        "base_evidence_commit": BASE_EVIDENCE_COMMIT,
        "selection_sha256":
            result["train_only_selection"]["selection_sha256"],
        "selected_configuration":
            result["selected_configuration"],
    }
    atomic_write_once(
        contract_path,
        stable_json_bytes(contract_record),
    )

    worker_evidence = {
        "phase": PHASE,
        "schema": "phase314b_r257_stagea_worker_evidence_v1",
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
        "schema": "phase314b_r257_stagea_summary_v1",
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
        "selected_configuration":
            result["selected_configuration"],
        "train_only_recommendation":
            result["train_only_recommendation"],
        "model_objective_changed": True,
        "scheduler_changed": False,
        "split_leakage_detected": False,
        "upper_gate_changed": False,
        "lower_score_used_for_validity": False,
        "formal_pilot_run": False,
        "checkpoint_saved": False,
        "prediction_tensor_persisted": False,
        "candidate_tensor_persisted": False,
        "npz_saved": False,
        "cache_saved": False,
        "formal_diffusion_training": False,
        "formal_reverse_sampling": False,
        "formal_idm_training": False,
        "action_diverse_data_collection": False,
        "candidate_execution": False,
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
                "selected_configuration":
                    summary["selected_configuration"],
                "summary": str(summary_path),
                "report": str(report_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
