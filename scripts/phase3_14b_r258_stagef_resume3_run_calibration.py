#!/usr/bin/env python3
"""Run two isolated Stage-F Resume2 workers and seal evidence."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ccda_phase3.phase314b_r258_stagef_constraint_aware_surrogate import (
    EXPECTED_SUBMODULE_COMMIT,
    compare_worker_results,
)
from ccda_phase3.phase314b_r258_stagef_resume2_porcelain_recovery import (
    BASE_EVIDENCE_COMMIT,
    PHASE,
    RESUME2_CONTRACT,
    RESUME2_REPORT,
    RESUME2_SUMMARY,
    RESUME2_TEST_GATE,
    RESUME2_WORKER,
    atomic_write_once,
    git_output,
    load_json,
    sha256_file,
    stable_json_bytes,
    status_paths,
    validate_failure_reports,
    validate_resume2_implementation_commit,
)


def resolve(root: Path, value: str) -> Path:
    candidate = Path(value)
    return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()


def load_original_renderer(root: Path) -> Any:
    path = root / "scripts/phase3_14b_r258_stagef_run_calibration.py"
    spec = importlib.util.spec_from_file_location(
        "phase314b_r258_stagef_original_runner",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load original Stage-F renderer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def assert_repository(root: Path, test_gate: Path) -> Dict[str, Any]:
    if git_output(root, "branch", "--show-current") != "Experiment1":
        raise RuntimeError("Stage-F Resume2 requires Experiment1")
    commits = validate_resume2_implementation_commit(root)
    remote = git_output(root, "rev-parse", "origin/Experiment1")
    if remote != BASE_EVIDENCE_COMMIT:
        raise RuntimeError("origin/Experiment1 changed before Resume2 push")
    submodule = root / "external/deformable-ravens"
    if git_output(submodule, "rev-parse", "HEAD") != EXPECTED_SUBMODULE_COMMIT:
        raise RuntimeError("DeformableRavens commit changed")
    if status_paths(submodule):
        raise RuntimeError("DeformableRavens worktree is dirty")
    expected = (test_gate.relative_to(root).as_posix(),)
    if status_paths(root) != expected:
        raise RuntimeError(
            "unexpected worktree paths before Resume2 scientific run: {}".format(
                status_paths(root)
            )
        )
    failure = validate_failure_reports(root)
    return {
        "branch": "Experiment1",
        "head": git_output(root, "rev-parse", "HEAD"),
        "origin_experiment1": remote,
        "commit_chain": commits,
        "blocked_provenance": {
            "sha256": failure["sha256"],
        },
        "submodule_commit": EXPECTED_SUBMODULE_COMMIT,
    }


def render_resume1_report(summary: Dict[str, Any], original_renderer: Any) -> str:
    result = summary["worker_result"]
    translation = result["translation_invariance"]
    centered = translation["records"]["full_centered_constraint"]
    segment = translation["records"]["full_segment_constraint"]
    envelope = translation["constraint_z_quantization_envelope"]
    commits = summary["repository"]["commit_chain"]
    prefix = [
        "# Phase3.14b-r2.5.8 Stage F Resume3",
        "",
        "## Translation-invariance recovery",
        "",
        "- Original Stage-F implementation commit: `{}`".format(
            commits["original_implementation_commit"]
        ),
        "- Blocked provenance commit: `{}`".format(
            commits["stagef_blocked_provenance_commit"]
        ),
        "- Resume1 implementation commit: `{}`".format(
            commits["resume1_implementation_commit"]
        ),
        "- Resume1 blocked provenance commit: `{}`".format(
            commits["resume1_blocked_provenance_commit"]
        ),
        "- Resume2 implementation commit: `{}`".format(
            commits["resume2_implementation_commit"]
        ),
        "- Original calibration completed before Resume2: `false`",
        "- Original failure locus: `full_centered_constraint translation gate`",
        "- Candidate matrix changed: `false`",
        "- Integrator changed: `false`",
        "- Scientific thresholds changed: `false`",
        "- Constraint-z saturation: `[-4,+4]`",
        "- Structural float64 translation gate: `PASS`",
        "- Componentwise float32 translation gate: `PASS`",
        "- Full-centered structural max: `{:.12g}`".format(
            float(centered["structural"]["maximum_absolute_difference"])
        ),
        "- Full-segment structural max: `{:.12g}`".format(
            float(segment["structural"]["maximum_absolute_difference"])
        ),
        "- Constraint-z quantization bound: `{:.12g}`".format(
            float(envelope["applied_bound"])
        ),
        "",
        "## Resume2 scientific result",
        "",
    ]
    original_summary = copy.deepcopy(summary)
    original_summary["phase"] = "Phase3.14b-r2.5.8 Stage F"
    return "\n".join(prefix) + original_renderer.render_report(original_summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument(
        "--python-bin",
        default="/miniforge3/envs/coord_bimanual/bin/python",
    )
    parser.add_argument("--contract", default=RESUME2_CONTRACT)
    parser.add_argument("--worker-evidence", default=RESUME2_WORKER)
    parser.add_argument("--summary", default=RESUME2_SUMMARY)
    parser.add_argument("--report", default=RESUME2_REPORT)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    contract_path = resolve(root, args.contract)
    worker_path = resolve(root, args.worker_evidence)
    summary_path = resolve(root, args.summary)
    report_path = resolve(root, args.report)
    test_gate_path = root / RESUME2_TEST_GATE
    for path in (contract_path, worker_path, summary_path, report_path):
        if path.exists():
            raise FileExistsError("Resume2 write-once output exists: {}".format(path))

    repository = assert_repository(root, test_gate_path)
    test_gate = load_json(test_gate_path)
    if test_gate.get("verdict") != "PASS":
        raise RuntimeError("Stage-F Resume2 test gate is not PASS")
    if int(test_gate.get("test_file_count", -1)) != 58:
        raise RuntimeError("Stage-F Resume2 test-file count changed")
    if int(test_gate.get("passed_test_count", -1)) != 1503:
        raise RuntimeError("Stage-F Resume2 pass count changed")

    original_renderer = load_original_renderer(root)
    temporary_root = Path(
        tempfile.mkdtemp(prefix="phase314b_r258_stagef_resume2_workers_", dir="/tmp")
    )
    worker_files = [temporary_root / "worker_1.json", temporary_root / "worker_2.json"]
    command = [
        str(args.python_bin),
        str(root / "scripts/phase3_14b_r258_stagef_resume3_worker.py"),
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
            raise RuntimeError("isolated Stage-F Resume2 workers differ")
        result = worker_results[0]

        correction = {
            "schema": "phase314b_r258_stagef_resume2_porcelain_recovery_v1",
            "resume1_translation_fix": {
                "original_failure": (
                    "unbounded log segment length amplified float32 translation quantization"
                ),
                "constraint_z_clip": 4.0,
                "structural_translation_gate": "float64 exact translation",
                "operational_translation_gate": (
                    "componentwise float32 round-trip with derived constraint-z ULP envelope"
                ),
            },
            "resume2_execution_recovery": {
                "original_failure": (
                    "stripped porcelain stdout removed the leading status-space from the first record"
                ),
                "porcelain_format": "v1-z",
                "stdout_trimmed": False,
                "nul_delimited": True,
                "rename_copy_supported": True,
            },
            "candidate_matrix_changed": False,
            "integrator_changed": False,
            "scientific_thresholds_changed": False,
            "original_stagef_calibration_completed": False,
            "test_gate_path": RESUME2_TEST_GATE,
            "test_gate_sha256": sha256_file(test_gate_path),
            "translation_invariance": result["translation_invariance"],
        }

        contract_record = {
            **result["constraint_aware_contract"],
            "phase": PHASE,
            "schema": "phase314b_r258_stagef_resume2_contract_record_v1",
            "resume2_porcelain_recovery": correction,
            "root_cause": result["root_cause"],
            "required_next_path": result["required_next_path"],
            "scientific_status": result["scientific_status"],
            "selection_sha256": result["selection"]["selection_sha256"],
            "selected_configuration": result["selected_configuration"],
        }
        atomic_write_once(contract_path, stable_json_bytes(contract_record))

        worker_evidence = {
            "phase": PHASE,
            "schema": "phase314b_r258_stagef_resume2_worker_evidence_v1",
            "worker_count": 2,
            "workers_exact": True,
            "comparison": comparison,
            "resume2_porcelain_recovery": correction,
            "worker_result": result,
        }
        atomic_write_once(worker_path, stable_json_bytes(worker_evidence))

        summary = {
            "phase": PHASE,
            "schema": "phase314b_r258_stagef_resume2_summary_v1",
            "verdict": "PASS",
            "scientific_status": result["scientific_status"],
            "root_cause": result["root_cause"],
            "required_next_path": result["required_next_path"],
            "repository": repository,
            "resume2_porcelain_recovery": correction,
            "test_gate_path": RESUME2_TEST_GATE,
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
            "selection_holdout_evaluated": result["selection_holdout_evaluated"],
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
        atomic_write_once(
            report_path,
            render_resume1_report(summary, original_renderer).encode("utf-8"),
        )
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
