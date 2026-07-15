#!/usr/bin/env python3
"""Run two isolated Stage-D.2 workers and seal write-once evidence."""
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

from ccda_phase3.phase314b_r256_staged2_collapse_provenance import (
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

TEST_GATE = "reports/phase3_14b_r256_staged2_test_gate_summary.json"
DEFAULT_AUDIT = (
    "reports/phase3_14b_r256_staged2_source_provenance.json"
)
DEFAULT_WORKER = (
    "reports/phase3_14b_r256_staged2_worker_evidence.json"
)
DEFAULT_SUMMARY = "reports/phase3_14b_r256_staged2_summary.json"
DEFAULT_REPORT = "reports/phase3_14b_r256_staged2_report.md"


def resolve(root: Path, value: str) -> Path:
    candidate = Path(value)
    return (
        candidate.resolve()
        if candidate.is_absolute()
        else (root / candidate).resolve()
    )


def assert_repository(root: Path, test_gate: Path) -> Dict[str, Any]:
    if git_output(root, "branch", "--show-current") != "Experiment1":
        raise RuntimeError("Stage D.2 requires Experiment1")
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
        raise RuntimeError("Stage-D.1 evidence is not an ancestor")

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
            "unexpected worktree paths before Stage-D.2 run: "
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
    classification = result["classification"]
    reconstruction = result["raw_reconstruction_audit"]
    calibration = result["population_geometry_audit"][
        "gate_calibration"
    ]
    driver = result["stage_d1_driver_verification"][
        "reconstructed_event"
    ]
    paired = result["paired_counterpart_audit"]

    lines = [
        "# Phase3.14b-r2.5.6 Stage D.2 Cable-XY Collapse Provenance Audit",
        "",
        f"- Audit verdict: `{summary['verdict']}`",
        f"- Scientific status: `{summary['scientific_status']}`",
        f"- Root cause: `{summary['root_cause']}`",
        f"- Required next path: `{summary['required_next_path']}`",
        "",
        "## Exact source reconstruction",
        "",
        (
            "- Isolated workers exact: `"
            f"{str(summary['workers_exact']).lower()}`"
        ),
        (
            "- Raw source episodes opened: `"
            f"{reconstruction['source_episode_count']}`"
        ),
        (
            "- Raw XY to cached target exact rate: `"
            f"{reconstruction['raw_xy_float32_exact_rate']:.9g}`"
        ),
        (
            "- Raw XYZ audit SHA256: `"
            + reconstruction["raw_target_xyz_sha256"]
            + "`"
        ),
        (
            "- Bead order stable: `"
            f"{str(reconstruction['bead_id_order_stable_across_target_horizons']).lower()}`"
        ),
        "",
        "## Stage-D.1 driver",
        "",
        (
            "- Window-contract row index: `"
            f"{driver['window_contract_row_index']}`"
        ),
        (
            "- Raw pickle frame index: `"
            f"{driver['raw_pickle_frame_index']}`"
        ),
        (
            "- Source: `"
            f"{driver['source_file']}`"
        ),
        (
            "- Horizon / segment: `"
            f"{driver['future_horizon']} / {driver['segment_index']}`"
        ),
        (
            "- XY reference ratio: `"
            f"{driver['xy_reference_ratio']:.9g}`"
        ),
        (
            "- XYZ reference ratio: `"
            f"{driver['xyz_reference_ratio']:.9g}`"
        ),
        (
            "- XY / XYZ projection ratio: `"
            f"{driver['xy_over_xyz']:.9g}`"
        ),
        (
            "- Driver category: `"
            f"{driver['category']}`"
        ),
        "",
        "## Calibration lower-tail attribution",
        "",
        (
            "- Severe XY events: `"
            f"{calibration['threshold_counts']['xy_ratio_below_0p25']}`"
        ),
        (
            "- Projection-artifact fraction: `"
            f"{calibration['severe_event_fractions']['projection_artifact']:.9g}`"
        ),
        (
            "- True-XYZ-collapse fraction: `"
            f"{calibration['severe_event_fractions']['true_xyz_collapse']:.9g}`"
        ),
        (
            "- Partial-XYZ-collapse fraction: `"
            f"{calibration['severe_event_fractions']['partial_xyz_collapse']:.9g}`"
        ),
        "",
        "## Paired counterpart",
        "",
        (
            "- Severe event instances: `"
            f"{paired['severe_event_count']}`"
        ),
        (
            "- Both-condition severe fraction: `"
            f"{paired['both_conditions_severe_fraction']:.9g}`"
        ),
        (
            "- Counterpart-not-severe fraction: `"
            f"{paired['counterpart_not_severe_fraction']:.9g}`"
        ),
        "",
        "## Boundary",
        "",
        "- The raw pickles were SHA-verified and opened read-only.",
        "- `row_index` was treated as the window-contract row, not a pickle frame.",
        "- Raw frame mapping used `window_t + 1 + future_horizon`.",
        "- Raw ordered XYZ was used for audit only; state-v3 and caches were not modified.",
        "- DeformableRavens source was statically audited but not executed or modified.",
        "- No gate was selected and no threshold was changed.",
        "- No model training, reverse sampling, formal pilot, IDM, data collection, candidate execution, Phase4, or CPS was run.",
        "- No checkpoint, tensor, NPZ, cache, image, or video was written.",
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
                f"Stage-D.2 write-once output exists: {path}"
            )

    repository = assert_repository(root, test_gate_path)
    test_gate = load_json(test_gate_path)
    if test_gate.get("verdict") != "PASS":
        raise RuntimeError("Stage-D.2 test gate is not PASS")

    temporary_root = Path(
        tempfile.mkdtemp(prefix="phase314b_r256_staged2_", dir="/tmp")
    )
    worker_files = [
        temporary_root / "worker_1.json",
        temporary_root / "worker_2.json",
    ]
    command = [
        str(args.python_bin),
        str(root / "scripts/phase3_14b_r256_staged2_worker.py"),
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
            "isolated Stage-D.2 provenance workers differ"
        )
    result = worker_results[0]

    audit_record = {
        "phase": PHASE,
        "schema": "phase314b_r256_staged2_source_provenance_v1",
        "root_cause": result["root_cause"],
        "required_next_path": result["required_next_path"],
        "source_logic_audit": result["source_logic_audit"],
        "contract_mapping_audit":
            result["contract_mapping_audit"],
        "raw_reconstruction_audit":
            result["raw_reconstruction_audit"],
        "geometry_references": result["geometry_references"],
        "population_geometry_audit":
            result["population_geometry_audit"],
        "stage_d1_driver_verification":
            result["stage_d1_driver_verification"],
        "stage_d1_driver_source_timeline":
            result["stage_d1_driver_source_timeline"],
        "paired_counterpart_audit":
            result["paired_counterpart_audit"],
        "classification": result["classification"],
    }
    atomic_write_once(audit_path, stable_json_bytes(audit_record))

    worker_evidence = {
        "phase": PHASE,
        "schema": "phase314b_r256_staged2_worker_evidence_v1",
        "worker_count": 2,
        "workers_exact": True,
        "comparison": comparison,
        "worker_result": result,
    }
    atomic_write_once(worker_path, stable_json_bytes(worker_evidence))

    summary = {
        "phase": PHASE,
        "schema": "phase314b_r256_staged2_summary_v1",
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
        "worker_evidence_path":
            worker_path.relative_to(root).as_posix(),
        "worker_evidence_sha256": sha256_file(worker_path),
        "workers_exact": comparison["exact"],
        "worker_result": result,
        "source_pickle_loaded_read_only": True,
        "provenance_row_semantics_corrected": True,
        "deformable_ravens_executed": False,
        "deformable_ravens_modified": False,
        "gate_selected": False,
        "threshold_changed": False,
        "model_or_branch_repaired": False,
        "diffusion_training": False,
        "reverse_sampling": False,
        "formal_pilot_run": False,
        "checkpoint_saved": False,
        "prediction_tensor_persisted": False,
        "candidate_tensor_persisted": False,
        "npz_saved": False,
        "formal_idm_training": False,
        "action_diverse_data_collection": False,
        "candidate_execution": False,
        "phase4": False,
        "cps": False,
        "train_only_recommendation": None,
        "selected_configuration": None,
    }
    atomic_write_once(
        report_path,
        render_report(summary).encode("utf-8"),
    )
    atomic_write_once(summary_path, stable_json_bytes(summary))
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
                "audit_sha256": summary["audit_sha256"],
                "summary": str(summary_path),
                "report": str(report_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
