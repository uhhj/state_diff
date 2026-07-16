#!/usr/bin/env python3
"""Run two isolated Stage-A Portable Resume1 workers and seal evidence."""
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

from ccda_phase3.phase314b_r258_stagea_resume1_output_lifecycle import (
    correction_record,
    validate_failed_files,
)

from ccda_phase3.phase314b_r258_stagea_conflict_projected_k16 import (
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
    "reports/"
    "phase3_14b_r258_stagea_resume1_"
    "test_gate_summary.json"
)
DEFAULT_CONTRACT = (
    "reports/"
    "phase3_14b_r258_stagea_resume1_"
    "contract.json"
)
DEFAULT_WORKER = (
    "reports/"
    "phase3_14b_r258_stagea_resume1_"
    "worker_evidence.json"
)
DEFAULT_SUMMARY = (
    "reports/"
    "phase3_14b_r258_stagea_resume1_"
    "summary.json"
)
DEFAULT_REPORT = (
    "reports/"
    "phase3_14b_r258_stagea_resume1_"
    "report.md"
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
            "r2.5.8 Stage A requires Experiment1"
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
            "Resume4 evidence is not an ancestor"
        )

    submodule = root / "external/deformable-ravens"
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
            "unexpected worktree paths before "
            "r2.5.8 Stage-A run: "
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
    environment = result["environment"]
    hardware = environment["hardware_observation"]
    compatibility = environment["compatibility"]
    dry_run = environment["required_operation_dry_run"]
    control = result["control_capture"]
    calibration = result[
        "calibration_contract"
    ]["calibration"]
    selection = result["selection"]
    classification = result[
        "classification"
    ]
    selected = result[
        "selected_configuration"
    ]

    lines = [
        "# Phase3.14b-r2.5.8 Stage A Portable Resume1 — Hardware-Portable Conflict Projection",
        "",
        "- Audit verdict: `{}`".format(
            summary["verdict"]
        ),
        "- Scientific status: `{}`".format(
            summary["scientific_status"]
        ),
        "- Root cause: `{}`".format(
            summary["root_cause"]
        ),
        "- Required next path: `{}`".format(
            summary["required_next_path"]
        ),
        "",
        "## Immutable Resume4 binding",
        "",
        "- Base evidence commit: `{}`".format(
            result["immutable_inputs"][
                "base_evidence_commit"
            ]
        ),
        "- Resume4 worker SHA256: `{}`".format(
            result["immutable_inputs"][
                "resume4_worker_sha256"
            ]
        ),
        "- Resume4 environment SHA256: `{}`".format(
            result["immutable_inputs"][
                "resume4_environment_sha256"
            ]
        ),
        "- Resume4 selection SHA256: `{}`".format(
            result["immutable_inputs"][
                "resume4_selection_sha256"
            ]
        ),
        "- Resume4 contract SHA256: `{}`".format(
            result["immutable_inputs"][
                "resume4_contract_sha256"
            ]
        ),
        "",
        "## Portable compute environment",
        "",
        "- Compatibility pass: `{}`".format(
            str(environment["compatibility_pass"]).lower()
        ),
        "- Compatibility SHA256: `{}`".format(
            environment["compatibility_sha256"]
        ),
        "- Hardware observation SHA256: `{}`".format(
            environment["observation_sha256"]
        ),
        "- GPU: `{}`".format(hardware["torch_device_name"]),
        "- Compute capability: `{}`".format(
            ".".join(str(value) for value in hardware["compute_capability"])
        ),
        "- Total memory bytes: `{}`".format(
            hardware["total_memory_bytes"]
        ),
        "- Driver observation: `{}`".format(
            hardware.get("nvidia_smi", {}).get("driver_version", "unavailable")
        ),
        "- Python/NumPy/PyTorch/Torch-CUDA: `{}` / `{}` / `{}` / `{}`".format(
            ".".join(str(value) for value in compatibility["python_version"]),
            compatibility["numpy_version"],
            compatibility["torch_version"],
            compatibility["torch_cuda_version"],
        ),
        "- Required-operation dry run: `{}`".format(
            str(dry_run["pass"]).lower()
        ),
        "- Dry-run peak reserved bytes: `{}`".format(
            dry_run["peak_memory_reserved_bytes"]
        ),
        "- Cold CUDA context before control replay: `{}`".format(
            str(
                result["cold_main_worker_context"][
                    "torch_cuda_is_initialized"
                ] is False
            ).lower()
        ),
        "",
        "## Frozen Stage-C reference replay",
        "",
        "- Reference equivalence pass: `{}`".format(
            str(control["reference_equivalence_pass"]).lower()
        ),
        "- Byte-exact to reference: `{}`".format(
            str(control["byte_exact_to_reference"]).lower()
        ),
        "- Reference calibration SHA256: `{}`".format(
            control["reference_calibration_sha256"]
        ),
        "- Observed calibration SHA256: `{}`".format(
            control["observed_calibration_sha256"]
        ),
        "- Reference control SHA256: `{}`".format(
            control["reference_control_sha256"]
        ),
        "- Observed control SHA256: `{}`".format(
            control["observed_control_sha256"]
        ),
        "- Calibration numeric differences: `{}`".format(
            control["calibration_numerical_equivalence"]["difference_count"]
        ),
        "- Control numeric differences: `{}`".format(
            control["control_numerical_equivalence"]["difference_count"]
        ),
        "- Stage-C nonzero candidates trained: `{}`".format(
            control["stagec_nonzero_candidate_training_count"]
        ),
        "",
        "## Conflict-projection calibration",
        "",
        "- Initial model SHA256: `{}`".format(
            calibration[
                "initial_model_sha256"
            ]
        ),
        "- Calibration SHA256: `{}`".format(
            calibration[
                "calibration_sha256"
            ]
        ),
        "- Block order: `{}`".format(
            ", ".join(
                calibration[
                    "block_order"
                ]
            )
        ),
        "",
        (
            "| Candidate | Mode | Cutoff | Target ratio | "
            "Lambda | Initial pre cosine | Initial post cosine | "
            "Initial removed norm |"
        ),
        (
            "|---|---|---:|---:|---:|---:|---:|---:|"
        ),
    ]
    for candidate in calibration["candidates"]:
        lines.append(
            "| {} | {} | {} | {:.6g} | {:.12g} | "
            "{:.9g} | {:.9g} | {:.9g} |".format(
                candidate["candidate_id"],
                candidate[
                    "projection_mode"
                ],
                candidate[
                    "timestep_cutoff"
                ],
                candidate[
                    "target_gradient_ratio"
                ],
                candidate["lambda_upper"],
                candidate[
                    "pre_projection_cosine"
                ],
                candidate[
                    "post_projection_cosine"
                ],
                candidate[
                    "removed_geometry_norm_fraction"
                ],
            )
        )

    lines.extend(
        [
            "",
            "## Train-only candidates",
            "",
            (
                "| Candidate | t10 reduction | t25 reduction | "
                "t50 reduction | Train ratio | Projection trigger | "
                "Removed norm | t10 upper | Eligible |"
            ),
            (
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
            ),
        ]
    )
    for record in selection[
        "candidate_records"
    ]:
        diagnostics = record[
            "training_diagnostics"
        ]
        lines.append(
            "| {} | {:.9g} | {:.9g} | {:.9g} | "
            "{:.9g} | {:.9g} | {:.9g} | {:.9g} | {} |".format(
                record["candidate_id"],
                record[
                    "continuous_response"
                ][
                    "t10_mean_excess_reduction"
                ],
                record[
                    "continuous_response"
                ][
                    "t25_mean_excess_reduction"
                ],
                record[
                    "continuous_response"
                ][
                    "t50_mean_excess_reduction"
                ],
                record[
                    "relative_to_control"
                ][
                    "train_control_nmse_ratio"
                ],
                diagnostics[
                    "projection_trigger_frequency"
                ],
                diagnostics[
                    "removed_norm_fraction_mean"
                ],
                record["binary_rate"]["10"],
                str(
                    record[
                        "eligible_for_selection"
                    ]
                ).lower(),
            )
        )

    lines.extend(
        [
            "",
            "## Selection and mechanism attribution",
            "",
            "- Selection SHA256: `{}`".format(
                selection[
                    "selection_sha256"
                ]
            ),
            "- Selected configuration: `{}`".format(
                "none"
                if selected is None
                else selected[
                    "candidate_id"
                ]
            ),
            "- Global negative control: `{}`".format(
                classification[
                    "global_negative_control"
                ]
            ),
            "- Best blockwise candidate: `{}`".format(
                classification[
                    "best_blockwise_candidate"
                ]
            ),
            "- Blockwise t10 advantage over global: `{}`".format(
                classification[
                    "blockwise_advantage_over_global_t10"
                ]
            ),
            "- Primary failure locus: `{}`".format(
                classification[
                    "primary_failure_locus"
                ]
            ),
            "",
            "## Boundary",
            "",
            "- Resume1 corrected only the preflight output lifecycle: a temporary directory was created and environment.json did not exist before atomic_write_once.",
            "- The original seven portable files and first blocked summary were committed unchanged as failure provenance.",
            "- The model architecture and upper-geometry target were unchanged.",
            "- Only the optimizer gradient-combination rule changed.",
            "- Global projection was a negative control and could not be selected.",
            "- Diffusion gradients were immutable under projection.",
            "- GPU model, UUID, PCI bus, compute capability, driver, and memory were audit observations rather than Resume4 equality gates.",
            "- Frozen Stage-C reference admission used exact discrete semantics and pre-registered numerical tolerances; candidate thresholds were not relaxed.",
            "- Two formal workers on the current rented instance were byte-exact.",
            "- The frozen probe was not accessed.",
            "- No full-874-row repaired model, reverse sampling, formal training, IDM, candidate execution, DeformableRavens, Phase4, or CPS was run.",
            "- No checkpoint, weights, prediction tensor, candidate tensor, NPZ, cache, image, or video was persisted.",
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
                "r2.5.8 Stage-A write-once "
                "output exists: {}".format(
                    path
                )
            )

    provenance = validate_failed_files(
        root,
        require_tracked=True,
    )
    repository = assert_repository(
        root,
        test_gate_path,
    )
    test_gate = load_json(
        test_gate_path
    )
    if test_gate.get("verdict") != "PASS":
        raise RuntimeError(
            "r2.5.8 Stage-A test gate "
            "is not PASS"
        )

    temporary_root = Path(
        tempfile.mkdtemp(
            prefix="phase314b_r258_stagea_",
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
            "phase3_14b_r258_stagea_worker.py"
        ),
        "--root",
        str(root),
        "--mode",
        "run",
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
    comparison = (
        compare_worker_results(
            worker_results[0],
            worker_results[1],
        )
    )
    if not comparison["exact"]:
        raise RuntimeError(
            "isolated r2.5.8 Stage-A "
            "workers differ"
        )
    result = worker_results[0]
    result["resume1_output_lifecycle"] = (
        correction_record(provenance)
    )

    contract_record = {
        "phase": PHASE,
        **result[
            "calibration_contract"
        ],
        "root_cause":
            result["root_cause"],
        "required_next_path":
            result[
                "required_next_path"
            ],
        "selection_sha256":
            result["selection"][
                "selection_sha256"
            ],
        "selected_configuration":
            result[
                "selected_configuration"
            ],
        "resume1_output_lifecycle":
            result["resume1_output_lifecycle"],
    }
    atomic_write_once(
        contract_path,
        stable_json_bytes(
            contract_record
        ),
    )

    worker_evidence = {
        "phase": PHASE,
        "schema":
            "phase314b_r258_stagea_resume1_worker_evidence_v1",
        "worker_count": 2,
        "workers_exact": True,
        "comparison": comparison,
        "worker_result": result,
    }
    atomic_write_once(
        worker_path,
        stable_json_bytes(
            worker_evidence
        ),
    )

    summary = {
        "phase": PHASE,
        "schema":
            "phase314b_r258_stagea_resume1_summary_v1",
        "verdict": "PASS",
        "scientific_status": "BLOCKED",
        "root_cause":
            result["root_cause"],
        "required_next_path":
            result[
                "required_next_path"
            ],
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
        "workers_exact":
            comparison["exact"],
        "resume1_output_lifecycle":
            result["resume1_output_lifecycle"],
        "worker_result": result,
        "selected_configuration":
            result[
                "selected_configuration"
            ],
        "train_only_recommendation":
            result[
                "train_only_recommendation"
            ],
        "control_replay_exact":
            result["control_replay_exact"],
        "control_reference_equivalent":
            result["control_reference_equivalent"],
        "environment_compatibility_pass":
            result["environment"]["compatibility_pass"],
        "required_operation_dry_run_pass":
            result["environment"][
                "required_operation_dry_run"
            ]["pass"],
        "hardware_model_restricted": False,
        "new_hyperparameter_candidate_run":
            True,
        "new_objective_variant_run":
            True,
        "frozen_probe_accessed":
            False,
        "reverse_sampling_run":
            False,
        "full_stageb_repaired_model_trained":
            False,
        "formal_pilot_run": False,
        "checkpoint_saved": False,
        "weights_persisted": False,
        "prediction_tensor_persisted":
            False,
        "candidate_tensor_persisted":
            False,
        "npz_saved": False,
        "cache_saved": False,
        "formal_diffusion_training":
            False,
        "formal_reverse_sampling":
            False,
        "formal_idm_training":
            False,
        "action_diverse_data_collection":
            False,
        "candidate_execution": False,
        "deformable_ravens_executed":
            False,
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
                "scientific_status":
                    "BLOCKED",
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
                "summary":
                    str(summary_path),
                "report":
                    str(report_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
