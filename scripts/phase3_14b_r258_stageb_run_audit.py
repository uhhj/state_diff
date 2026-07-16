#!/usr/bin/env python3
"""Run two isolated Stage-B reachability workers and seal evidence."""
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

from ccda_phase3.phase314b_r258_stageb_direct_x0_reachability import (
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
    "phase3_14b_r258_stageb_"
    "test_gate_summary.json"
)
DEFAULT_CONTRACT = (
    "reports/"
    "phase3_14b_r258_stageb_"
    "contract.json"
)
DEFAULT_WORKER = (
    "reports/"
    "phase3_14b_r258_stageb_"
    "worker_evidence.json"
)
DEFAULT_SUMMARY = (
    "reports/"
    "phase3_14b_r258_stageb_"
    "summary.json"
)
DEFAULT_REPORT = (
    "reports/"
    "phase3_14b_r258_stageb_"
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
            "r2.5.8 Stage B requires Experiment1"
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
            "Stage-A Resume1 evidence "
            "is not an ancestor"
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
            "Stage-B run: "
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


def _oracle_cell(value: Any) -> str:
    if value is None:
        return "none"
    final = value["final"]
    return (
        "r={:.3f}; upper={:.4f}; "
        "nmse-ratio={:.4f}; lower={:.4f}; "
        "physical={:.4f}; collapse={:.4f}"
    ).format(
        float(value["radius"]),
        float(final["upper_row_pass_rate"]),
        float(final["normalized_mse_ratio"]),
        float(final["lower_row_pass_rate"]),
        float(
            final[
                "historical_physical_row_any_rate"
            ]
        ),
        float(
            final[
                "segment_length"
            ]["collapse_fraction"]
        ),
    )


def render_report(
    summary: Dict[str, Any],
) -> str:
    result = summary["worker_result"]
    environment = result["environment"]
    observation = environment[
        "hardware_observation"
    ]
    control = result["control_capture"]
    classification = result[
        "classification"
    ]

    lines = [
        "# Phase3.14b-r2.5.8 Stage B",
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
        "## Immutable Stage-A Resume1 binding",
        "",
        "- Base evidence commit: `{}`".format(
            result["immutable_inputs"][
                "base_evidence_commit"
            ]
        ),
        "- Base worker SHA256: `{}`".format(
            result["immutable_inputs"][
                "base_worker_sha256"
            ]
        ),
        "- Base contract SHA256: `{}`".format(
            result["immutable_inputs"][
                "base_contract_sha256"
            ]
        ),
        "- Base selection SHA256: `{}`".format(
            result["immutable_inputs"][
                "base_selection_sha256"
            ]
        ),
        "",
        "## Portable compute environment",
        "",
        "- Compatibility pass: `{}`".format(
            str(
                environment[
                    "compatibility_pass"
                ]
            ).lower()
        ),
        "- Compatibility SHA256: `{}`".format(
            environment[
                "compatibility_sha256"
            ]
        ),
        "- Observation SHA256: `{}`".format(
            environment[
                "observation_sha256"
            ]
        ),
        "- GPU: `{}`".format(
            observation[
                "torch_device_name"
            ]
        ),
        "- Compute capability: `{}`".format(
            ".".join(
                str(value)
                for value
                in observation[
                    "compute_capability"
                ]
            )
        ),
        "- Driver: `{}`".format(
            observation[
                "nvidia_smi"
            ]["driver_version"]
        ),
        "- Required-operation dry run: `{}`".format(
            str(
                environment[
                    "required_operation_dry_run"
                ]["pass"]
            ).lower()
        ),
        "- Cold CUDA context before control replay: `{}`".format(
            str(
                result[
                    "cold_main_worker_context"
                ][
                    "torch_cuda_is_initialized"
                ]
                is False
            ).lower()
        ),
        "",
        "## Frozen Stage-C control",
        "",
        "- Reference equivalence pass: `{}`".format(
            str(
                control[
                    "reference_equivalence_pass"
                ]
            ).lower()
        ),
        "- Calibration reference/observed SHA256: `{}` / `{}`".format(
            control[
                "reference_calibration_sha256"
            ],
            control[
                "observed_calibration_sha256"
            ],
        ),
        "- Control reference/observed SHA256: `{}` / `{}`".format(
            control[
                "reference_control_sha256"
            ],
            control[
                "observed_control_sha256"
            ],
        ),
        "- Stage-C nonzero candidates trained: `{}`".format(
            control[
                "stagec_nonzero_candidate_training_count"
            ]
        ),
        "- Retained-control wrapper restored: `{}`".format(
            str(
                control[
                    "capture_wrapper_restored"
                ]
            ).lower()
        ),
        "",
        "## Source audit",
        "",
        "- Direct-x0 residual parameterization: `{}`".format(
            str(
                result[
                    "source_logic_audit"
                ]["checks"][
                    "direct_x0_residual_parameterization"
                ]
            ).lower()
        ),
        "- Upper objective one-sided: `{}`".format(
            str(
                result[
                    "source_logic_audit"
                ]["checks"][
                    "upper_objective_is_one_sided"
                ]
            ).lower()
        ),
        "- Explicit lower constraint present: `false`",
        "- Identified risk: `{}`".format(
            result[
                "source_logic_audit"
            ]["identified_risk"]
        ),
        "",
        "## Target-line reachability",
        "",
        (
            "| t | Control upper | Target upper | "
            "Line reachable | First valid alpha | "
            "Target physical | Monotonic violation |"
        ),
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for timestep in ("10", "25", "50"):
        record = result[
            "timestep_records"
        ][timestep]["target_line"]
        first = record[
            "first_aggregate_valid"
        ]
        lines.append(
            "| {} | {:.6g} | {:.6g} | {} | {} | "
            "{:.6g} | {:.6g} |".format(
                timestep,
                float(
                    record[
                        "control_anchor"
                    ][
                        "upper_row_pass_rate"
                    ]
                ),
                float(
                    record[
                        "target_anchor"
                    ][
                        "upper_row_pass_rate"
                    ]
                ),
                str(
                    record[
                        "aggregate_line_reachable"
                    ]
                ).lower(),
                (
                    "{:.4f}".format(
                        float(first["alpha"])
                    )
                    if first is not None
                    else "none"
                ),
                float(
                    record[
                        "target_anchor"
                    ][
                        "historical_physical_row_any_rate"
                    ]
                ),
                float(
                    record[
                        "monotonic_pass_to_fail_row_rate"
                    ]
                ),
            )
        )

    lines.extend(
        [
            "",
            "## Direct-x0 upper-only oracle",
            "",
            (
                "| t | Geometry-descent target cosine | "
                "Upper+fidelity reachable | Cable-valid reachable | "
                "Best upper+fidelity | Best cable-valid |"
            ),
            "|---:|---:|---:|---:|---|---|",
        ]
    )
    for timestep in ("10", "25", "50"):
        item = result[
            "timestep_records"
        ][timestep]
        oracle = item[
            "direct_x0_oracle"
        ]
        cosine = item[
            "geometry_gradient_alignment"
        ][
            "descent_target_cosine"
        ]["mean"]
        lines.append(
            "| {} | {:.6g} | {} | {} | {} | {} |".format(
                timestep,
                float(cosine),
                str(
                    oracle[
                        "upper_fidelity_reachable"
                    ]
                ).lower(),
                str(
                    oracle[
                        "cable_valid_reachable"
                    ]
                ).lower(),
                _oracle_cell(
                    oracle[
                        "best_upper_fidelity_record"
                    ]
                ),
                _oracle_cell(
                    oracle[
                        "best_cable_valid_record"
                    ]
                ),
            )
        )

    lines.extend(
        [
            "",
            "## Direct-x0 model tangent",
            "",
            (
                "| t | Rows | 8-step Krylov explained lower bound | "
                "Global response stability | Output-head explained | "
                "Output-head relative update |"
            ),
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for timestep in ("10", "25", "50"):
        tangent = result[
            "timestep_records"
        ][timestep][
            "tangent_reachability"
        ]
        global_record = tangent[
            "parameter_groups"
        ]["global"]
        iterative = tangent[
            "iterative_global_tangent_lower_bound"
        ]
        head = tangent[
            "output_head_linear"
        ]
        lines.append(
            "| {} | {} | {:.6g} | {:.6g} | "
            "{:.6g} | {:.6g} |".format(
                timestep,
                int(
                    tangent[
                        "selected_row_count"
                    ]
                ),
                float(
                    iterative[
                        "final_explained_ratio"
                    ]
                ),
                float(
                    global_record[
                        "response_stability_min_cosine"
                    ]
                ),
                float(
                    head[
                        "explained_ratio"
                    ]
                ),
                float(
                    head[
                        "relative_parameter_update_norm"
                    ]
                ),
            )
        )

    lines.extend(
        [
            "",
            "## Classification",
            "",
            "- Primary failure locus: `{}`".format(
                classification[
                    "primary_failure_locus"
                ]
            ),
            "- Target anchor all pass: `{}`".format(
                str(
                    classification[
                        "target_anchor_all_pass"
                    ]
                ).lower()
            ),
            "- Target line all reachable: `{}`".format(
                str(
                    classification[
                        "target_line_all_reachable"
                    ]
                ).lower()
            ),
            "- Oracle upper+fidelity count: `{}`".format(
                classification[
                    "oracle_upper_fidelity_count"
                ]
            ),
            "- Oracle cable-valid count: `{}`".format(
                classification[
                    "oracle_valid_count"
                ]
            ),
            "- One-sided nonphysical shortcut count: `{}`".format(
                classification[
                    "nonphysical_shortcut_count"
                ]
            ),
            "- Geometry-descent mean target cosine: `{}`".format(
                classification[
                    "geometry_descent_target_cosine_mean"
                ]
            ),
            "- Contract SHA256: `{}`".format(
                result[
                    "reachability_contract"
                ]["contract_sha256"]
            ),
            "- Selection SHA256: `{}`".format(
                result["selection"][
                    "selection_sha256"
                ]
            ),
            "",
            "## Boundary",
            "",
            "- No new model candidate was trained.",
            "- Direct-x0 oracle optimization modified only temporary output tensors.",
            "- Central finite differences restored the model byte-exact after every parameter perturbation.",
            "- Ground truth was used only for train-only diagnostic reachability, never for candidate selection.",
            "- The frozen probe was not accessed.",
            "- No full repaired model, reverse sampling, formal training, IDM, candidate execution, DeformableRavens, Phase4, or CPS was run.",
            "- No checkpoint, weights, prediction tensor, oracle tensor, candidate tensor, NPZ, cache, image, or video was persisted.",
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
                "Stage-B write-once output exists: "
                "{}".format(path)
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
            "Stage-B test gate is not PASS"
        )

    temporary_root = Path(
        tempfile.mkdtemp(
            prefix="phase314b_r258_stageb_",
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
            "phase3_14b_r258_stageb_worker.py"
        ),
        "--root",
        str(root),
        "--mode",
        "run",
    ]
    try:
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
                "isolated Stage-B workers differ"
            )
        result = worker_results[0]

        contract_record = {
            "phase": PHASE,
            **result[
                "reachability_contract"
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
            "selected_configuration": None,
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
                "phase314b_r258_stageb_worker_evidence_v1",
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
                "phase314b_r258_stageb_summary_v1",
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
                sha256_file(
                    test_gate_path
                ),
            "test_gate": test_gate,
            "contract_path":
                contract_path.relative_to(
                    root
                ).as_posix(),
            "contract_file_sha256":
                sha256_file(
                    contract_path
                ),
            "worker_evidence_path":
                worker_path.relative_to(
                    root
                ).as_posix(),
            "worker_evidence_sha256":
                sha256_file(
                    worker_path
                ),
            "workers_exact":
                comparison["exact"],
            "worker_result": result,
            "selected_configuration": None,
            "train_only_recommendation": None,
            "new_model_candidate_trained":
                False,
            "direct_x0_tensor_optimization_run":
                True,
            "tangent_audit_run": True,
            "control_replay_exact":
                result[
                    "control_replay_exact"
                ],
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
            "oracle_tensor_persisted":
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
            "candidate_execution":
                False,
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
    finally:
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
                    summary[
                        "workers_exact"
                    ],
                "contract_file_sha256":
                    summary[
                        "contract_file_sha256"
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
