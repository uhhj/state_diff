#!/usr/bin/env python3
"""Run two isolated Stage-F workers and seal constraint-aware evidence."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ccda_phase3.phase314b_r258_stagef_constraint_aware_surrogate import (
    BASE_EVIDENCE_COMMIT,
    EXPECTED_SUBMODULE_COMMIT,
    PHASE,
    atomic_write_once,
    compare_worker_results,
    git_output,
    load_json,
    sha256_file,
    stable_json_bytes,
    status_paths,
    validate_implementation_commit,
)

TEST_GATE = "reports/phase3_14b_r258_stagef_test_gate_summary.json"
DEFAULT_CONTRACT = "reports/phase3_14b_r258_stagef_contract.json"
DEFAULT_WORKER = "reports/phase3_14b_r258_stagef_worker_evidence.json"
DEFAULT_SUMMARY = "reports/phase3_14b_r258_stagef_summary.json"
DEFAULT_REPORT = "reports/phase3_14b_r258_stagef_report.md"


def resolve(root: Path, value: str) -> Path:
    candidate = Path(value)
    return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()


def assert_repository(root: Path, test_gate: Path) -> Dict[str, Any]:
    if git_output(root, "branch", "--show-current") != "Experiment1":
        raise RuntimeError("Stage F requires Experiment1")
    implementation = validate_implementation_commit(root)
    submodule = root / "external/deformable-ravens"
    if git_output(submodule, "rev-parse", "HEAD") != EXPECTED_SUBMODULE_COMMIT:
        raise RuntimeError("DeformableRavens commit changed")
    if status_paths(submodule):
        raise RuntimeError("DeformableRavens worktree is dirty")
    expected = (test_gate.relative_to(root).as_posix(),)
    if status_paths(root) != expected:
        raise RuntimeError(
            "unexpected worktree paths before Stage-F scientific run: {}".format(
                status_paths(root)
            )
        )
    return {"branch": "Experiment1", **implementation}


def _cell(record: Mapping[str, Any]) -> str:
    metrics = record["projected_direction_metrics"]
    evaluation = record["evaluation"]
    integration = record["integration"]
    return (
        "pcos={:.4f}; feasible={:.4f}; retention={:.4f}; "
        "physical={:.4f}; nmse={:.4f}; reduction={:.4f}; pass={}"
    ).format(
        float(metrics["cosine"]["mean"]),
        float(integration["observable_feasible_rate"]),
        float(integration["direction_retention"]["mean"]),
        float(evaluation["historical_physical_row_any_rate"]),
        float(evaluation["normalized_mse_ratio"]),
        float(evaluation["target_distance_reduction_fraction"]["mean"]),
        str(bool(record["scientific_pass"])).lower(),
    )


def render_report(summary: Dict[str, Any]) -> str:
    result = summary["worker_result"]
    environment = result["environment"]
    hardware = environment["hardware_observation"]
    selected = result["objective_train_selected_configuration"]
    validated = result["selected_configuration"]
    classification = result["classification"]
    lines = [
        "# Phase3.14b-r2.5.8 Stage F",
        "",
        "- Audit verdict: `{}`".format(summary["verdict"]),
        "- Scientific status: `{}`".format(summary["scientific_status"]),
        "- Root cause: `{}`".format(summary["root_cause"]),
        "- Required next path: `{}`".format(summary["required_next_path"]),
        "",
        "## Immutable Stage-E binding",
        "",
        "- Base evidence commit: `{}`".format(
            result["immutable_inputs"]["base_evidence_commit"]
        ),
        "- Base worker SHA256: `{}`".format(
            result["immutable_inputs"]["base_worker_sha256"]
        ),
        "- Base internal contract SHA256: `{}`".format(
            result["immutable_inputs"]["base_internal_contract_sha256"]
        ),
        "- Base contract file SHA256: `{}`".format(
            result["immutable_inputs"]["base_contract_file_sha256"]
        ),
        "- Base selection SHA256: `{}`".format(
            result["immutable_inputs"]["base_selection_sha256"]
        ),
        "",
        "## Portable environment and frozen control",
        "",
        "- Compatibility pass: `{}`".format(
            str(environment["compatibility_pass"]).lower()
        ),
        "- Compatibility SHA256: `{}`".format(environment["compatibility_sha256"]),
        "- Observation SHA256: `{}`".format(environment["observation_sha256"]),
        "- GPU: `{}`".format(hardware["torch_device_name"]),
        "- Compute capability: `{}`".format(
            ".".join(str(value) for value in hardware["compute_capability"])
        ),
        "- Required-operation dry run: `{}`".format(
            str(environment["required_operation_dry_run"]["pass"]).lower()
        ),
        "- Cold CUDA context before control replay: `{}`".format(
            str(
                result["cold_main_worker_context"]["torch_cuda_is_initialized"]
                is False
            ).lower()
        ),
        "",
        "## Split and leakage boundary",
        "",
        "- Objective-train rows / groups: `{}` / `{}`".format(
            result["split"]["objective_train_rows"],
            result["split"]["objective_train_groups"],
        ),
        "- Selection-holdout rows: `{}`".format(
            result["split"]["selection_holdout_rows"]
        ),
        "- Frozen-probe rows: `{}`".format(result["split"]["frozen_probe_rows"]),
        "- Constraint-aware target: `frozen Stage-E projected-oracle displacement`",
        "- Test-fold target used for fitting: `false`",
        "- Holdout evaluated: `{}`".format(
            str(result["selection_holdout_evaluated"]).lower()
        ),
        "- Frozen probe accessed: `false`",
        "- Selectable features translation invariant: `{}`".format(
            str(result["translation_invariance"]["all_selectable_modes_invariant"]).lower()
        ),
        "",
        "## Projected-oracle supervision",
        "",
        "| t | Feasible rate | Physical | NMSE | Reduction |",
        "|---:|---:|---:|---:|---:|",
    ]
    for timestep in (10, 25, 50):
        record = result["projected_oracle_targets"][str(timestep)]
        evaluation = record["evaluation"]
        lines.append(
            "| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |".format(
                timestep,
                float(record["feasible_rate"]),
                float(evaluation["historical_physical_row_any_rate"]),
                float(evaluation["normalized_mse_ratio"]),
                float(evaluation["target_distance_reduction_fraction"]["mean"]),
            )
        )
    lines.extend(
        [
            "",
            "## Projected-direction baselines",
            "",
            "| Baseline | t10 cosine | t25 cosine | t50 cosine | Selectable |",
            "|---|---:|---:|---:|---:|",
            "| projected_global_mean | {:.6f} | {:.6f} | {:.6f} | false |".format(
                float(result["global_projected_baselines"]["10"]),
                float(result["global_projected_baselines"]["25"]),
                float(result["global_projected_baselines"]["50"]),
            ),
            "| projected_condition_mean | {:.6f} | {:.6f} | {:.6f} | false |".format(
                float(result["condition_projected_baselines"]["10"]),
                float(result["condition_projected_baselines"]["25"]),
                float(result["condition_projected_baselines"]["50"]),
            ),
            "",
            "## Objective-train grouped OOF candidates",
            "",
            "| Candidate | Role | Target | Model | t10 | t25 | t50 | Eligible |",
            "|---|---|---|---|---|---|---|---:|",
        ]
    )
    for candidate in result["candidate_records"]:
        records = candidate["timestep_records"]
        definition = candidate["definition"]
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                candidate["candidate_id"],
                candidate["role"],
                definition["target_mode"],
                definition["model_mode"],
                _cell(records["10"]),
                _cell(records["25"]),
                _cell(records["50"]),
                str(bool(candidate["eligible"])).lower(),
            )
        )
    lines.extend(
        [
            "",
            "## Selection and controls",
            "",
            "- Objective-train selected configuration: `{}`".format(
                "none" if selected is None else selected["candidate_id"]
            ),
            "- Validated train-only recommendation: `{}`".format(
                "none" if validated is None else validated["candidate_id"]
            ),
            "- Permutation control: `{}`".format(
                "not-run"
                if result["permutation_control"] is None
                else str(result["permutation_control"]["all_pass"]).lower()
            ),
            "- Projection-only control: `{}`".format(
                "not-run"
                if result["projection_only_control"] is None
                else str(result["projection_only_control"]["pass"]).lower()
            ),
            "- Locked holdout: `{}`".format(
                "not-run"
                if result["locked_holdout_evaluation"] is None
                else str(result["locked_holdout_evaluation"]["scientific_pass"]).lower()
            ),
            "- Projected oracle all timesteps: `{}`".format(
                str(classification["projected_oracle_all_timesteps"]).lower()
            ),
            "- Direction-all candidates: `{}`".format(
                ", ".join(classification["direction_all_candidate_ids"]) or "none"
            ),
            "- State-all candidates: `{}`".format(
                ", ".join(classification["state_all_candidate_ids"]) or "none"
            ),
            "- Eligible candidates: `{}`".format(
                ", ".join(classification["eligible_candidate_ids"]) or "none"
            ),
            "- Primary failure locus: `{}`".format(
                classification["primary_failure_locus"]
            ),
            "- Contract SHA256: `{}`".format(
                result["constraint_aware_contract"]["contract_sha256"]
            ),
            "- Selection SHA256: `{}`".format(
                result["selection"]["selection_sha256"]
            ),
            "",
            "## Boundary",
            "",
            "- Projected-oracle supervision used ground truth only inside objective-train training targets and evaluation targets.",
            "- Every OOF feature standardizer and output basis was fitted on the outer training fold only.",
            "- No condition label, group, pair, seed, or window metadata entered selectable features.",
            "- The hidden-label condition-mean model was diagnostic-only; every selectable candidate had to beat both global-mean and condition-mean projected-direction baselines.",
            "- The Stage-E constrained integrator and all of its thresholds were frozen by deriving the Stage-F definitions from the immutable Stage-E oracle definition.",
            "- The selection holdout target was not used for fitting, prediction, candidate generation, or selection; it was used only after the locked prediction for evaluation.",
            "- The selection holdout could not change candidate, feature mode, target mode, model, rank, RFF map, or thresholds.",
            "- No diffusion-model candidate was trained and no architecture was changed.",
            "- The frozen probe was not accessed.",
            "- No reverse sampling, formal training, IDM, candidate execution, DeformableRavens, Phase4, or CPS was run.",
            "- No checkpoint, weights, surrogate weights, prediction tensor, candidate tensor, NPZ, cache, image, or video was persisted.",
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
    for path in (contract_path, worker_path, summary_path, report_path):
        if path.exists():
            raise FileExistsError("Stage-F write-once output exists: {}".format(path))
    repository = assert_repository(root, test_gate_path)
    test_gate = load_json(test_gate_path)
    if test_gate.get("verdict") != "PASS":
        raise RuntimeError("Stage-F test gate is not PASS")
    if int(test_gate.get("test_file_count", -1)) != 56:
        raise RuntimeError("Stage-F test-file count changed")
    if int(test_gate.get("passed_test_count", -1)) != 1408:
        raise RuntimeError("Stage-F pass count changed")

    temporary_root = Path(
        tempfile.mkdtemp(prefix="phase314b_r258_stagef_workers_", dir="/tmp")
    )
    worker_files = [temporary_root / "worker_1.json", temporary_root / "worker_2.json"]
    command = [
        str(args.python_bin),
        str(root / "scripts/phase3_14b_r258_stagef_worker.py"),
        "--root",
        str(root),
        "--mode",
        "run",
    ]
    try:
        for worker_file in worker_files:
            subprocess.run(
                command + ["--output", str(worker_file)],
                cwd=str(root),
                check=True,
            )
        worker_results = [load_json(path) for path in worker_files]
        comparison = compare_worker_results(worker_results[0], worker_results[1])
        if not comparison["exact"]:
            raise RuntimeError("isolated Stage-F workers differ")
        result = worker_results[0]
        contract_record = {
            **result["constraint_aware_contract"],
            "root_cause": result["root_cause"],
            "required_next_path": result["required_next_path"],
            "scientific_status": result["scientific_status"],
            "selection_sha256": result["selection"]["selection_sha256"],
            "selected_configuration": result["selected_configuration"],
        }
        atomic_write_once(contract_path, stable_json_bytes(contract_record))
        worker_evidence = {
            "phase": PHASE,
            "schema": "phase314b_r258_stagef_worker_evidence_v1",
            "worker_count": 2,
            "workers_exact": True,
            "comparison": comparison,
            "worker_result": result,
        }
        atomic_write_once(worker_path, stable_json_bytes(worker_evidence))
        summary = {
            "phase": PHASE,
            "schema": "phase314b_r258_stagef_summary_v1",
            "verdict": "PASS",
            "scientific_status": result["scientific_status"],
            "root_cause": result["root_cause"],
            "required_next_path": result["required_next_path"],
            "repository": repository,
            "test_gate_path": TEST_GATE,
            "test_gate_sha256": sha256_file(test_gate_path),
            "test_gate": test_gate,
            "contract_path": contract_path.relative_to(root).as_posix(),
            "contract_file_sha256": sha256_file(contract_path),
            "worker_evidence_path": worker_path.relative_to(root).as_posix(),
            "worker_evidence_sha256": sha256_file(worker_path),
            "workers_exact": comparison["exact"],
            "worker_result": result,
            "selected_configuration": result["selected_configuration"],
            "train_only_recommendation": result["train_only_recommendation"],
            "scientific_result_sealed": True,
            "frozen_probe_accessed": False,
            "new_diffusion_model_candidate_trained": False,
            "reverse_sampling_run": False,
            "formal_training_run": False,
            "idm_run": False,
            "candidate_execution": False,
            "deformable_ravens_executed": False,
            "phase4": False,
            "cps": False,
            "checkpoint_saved": False,
            "weights_persisted": False,
            "surrogate_weights_persisted": False,
            "prediction_tensor_persisted": False,
            "candidate_tensor_persisted": False,
            "npz_saved": False,
            "cache_saved": False,
            "image_saved": False,
            "video_saved": False,
        }
        atomic_write_once(report_path, render_report(summary).encode("utf-8"))
        atomic_write_once(summary_path, stable_json_bytes(summary))
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)
    print(
        json.dumps(
            {
                "verdict": "PASS",
                "scientific_status": summary["scientific_status"],
                "root_cause": summary["root_cause"],
                "required_next_path": summary["required_next_path"],
                "workers_exact": summary["workers_exact"],
                "worker_sha256": comparison["left_sha256"],
                "contract_file_sha256": summary["contract_file_sha256"],
                "selected_configuration": summary["selected_configuration"],
                "summary": str(summary_path),
                "report": str(report_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
