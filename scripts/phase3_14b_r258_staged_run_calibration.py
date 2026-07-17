#!/usr/bin/env python3
"""Run two isolated Stage-D direction-surrogate workers and seal evidence."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ccda_phase3.phase314b_r258_staged_direction_surrogate import (
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
    "phase3_14b_r258_staged_"
    "test_gate_summary.json"
)
DEFAULT_CONTRACT = (
    "reports/"
    "phase3_14b_r258_staged_"
    "contract.json"
)
DEFAULT_WORKER = (
    "reports/"
    "phase3_14b_r258_staged_"
    "worker_evidence.json"
)
DEFAULT_SUMMARY = (
    "reports/"
    "phase3_14b_r258_staged_"
    "summary.json"
)
DEFAULT_REPORT = (
    "reports/"
    "phase3_14b_r258_staged_"
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
            "Stage D requires Experiment1"
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
            "Stage-C Resume1 evidence is not an ancestor"
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
        value = line[3:].strip()
        if " -> " in value:
            value = value.split(
                " -> ",
                1,
            )[1]
        if value not in allowed:
            unexpected.append(line)
    if unexpected:
        raise RuntimeError(
            "unexpected worktree paths before Stage-D run: "
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


def _witness_text(
    value: Mapping[str, Any],
) -> str:
    if value is None:
        return "none"
    evaluation = value["evaluation"]
    return (
        "s={:.2f}; upper={:.4f}; lower={:.4f}; "
        "physical={:.4f}; nmse={:.4f}; "
        "collapse={:.4f}; move-cos={:.4f}"
    ).format(
        float(value["scale"]),
        float(
            evaluation[
                "upper_row_pass_rate"
            ]
        ),
        float(
            evaluation[
                "lower_row_pass_rate"
            ]
        ),
        float(
            evaluation[
                "historical_physical_row_any_rate"
            ]
        ),
        float(
            evaluation[
                "normalized_mse_ratio"
            ]
        ),
        float(
            evaluation[
                "segment_length"
            ]["collapse_fraction"]
        ),
        float(
            evaluation[
                "target_direction_cosine"
            ]["mean"]
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
    classification = result[
        "classification"
    ]
    selected_oof = result[
        "objective_train_selected_configuration"
    ]
    validated = result[
        "selected_configuration"
    ]
    permutation = result[
        "permutation_control"
    ]
    holdout = result[
        "locked_holdout_evaluation"
    ]

    lines = [
        "# Phase3.14b-r2.5.8 Stage D",
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
        "## Immutable Stage-C Resume1 binding",
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
        "## Portable environment and frozen control",
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
        "- Stage-C reference equivalence: `{}`".format(
            str(
                result[
                    "control_capture"
                ][
                    "reference_equivalence_pass"
                ]
            ).lower()
        ),
        "",
        "## Split and leakage boundary",
        "",
        "- Objective-train rows / groups: `{}` / `{}`".format(
            result["split"][
                "objective_train_rows"
            ],
            result["split"][
                "objective_train_groups"
            ],
        ),
        "- Selection-holdout rows: `{}`".format(
            result["split"][
                "selection_holdout_rows"
            ]
        ),
        "- Frozen-probe rows: `{}`".format(
            result["split"][
                "frozen_probe_rows"
            ]
        ),
        "- Candidate selection uses holdout: `false`",
        "- Holdout evaluated: `{}`".format(
            str(
                result[
                    "selection_holdout_evaluated"
                ]
            ).lower()
        ),
        "- Frozen probe accessed: `false`",
        "- Selectable features translation invariant: `true`",
        "- Absolute-coordinate diagnostic selectable: `false`",
        "",
        "## Baselines",
        "",
        "| Baseline | t10 cosine | t25 cosine | t50 cosine |",
        "|---|---:|---:|---:|",
    ]
    for candidate_id in (
        "global_mean",
        "condition_mean",
    ):
        means = result[
            "baseline_means"
        ][candidate_id]
        lines.append(
            "| {} | {:.6g} | {:.6g} | {:.6g} |".format(
                candidate_id,
                float(means["10"]),
                float(means["25"]),
                float(means["50"]),
            )
        )

    lines.extend(
        [
            "",
            "## Objective-train grouped OOF candidates",
            "",
            (
                "| Candidate | Role | t10 cosine | t25 cosine | "
                "t50 cosine | Direction all | State all | Eligible |"
            ),
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for record in result[
        "candidate_records"
    ]:
        time = record[
            "timestep_records"
        ]
        direction_all = all(
            item[
                "direction_gates"
            ].get("all", False)
            for item in time.values()
        )
        state_all = all(
            item[
                "scale_sweep"
            ][
                "scientific_reachable"
            ]
            for item in time.values()
        )
        lines.append(
            "| {} | {} | {:.6g} | {:.6g} | {:.6g} | {} | {} | {} |".format(
                record[
                    "candidate_id"
                ],
                record["role"],
                float(
                    time["10"][
                        "direction_metrics"
                    ]["cosine"]["mean"]
                ),
                float(
                    time["25"][
                        "direction_metrics"
                    ]["cosine"]["mean"]
                ),
                float(
                    time["50"][
                        "direction_metrics"
                    ]["cosine"]["mean"]
                ),
                str(direction_all).lower(),
                str(state_all).lower(),
                str(
                    record["eligible"]
                ).lower(),
            )
        )

    lines.extend(
        [
            "",
            "## OOF scientific witnesses",
            "",
        ]
    )
    for record in result[
        "candidate_records"
    ]:
        if record["role"] != "selectable":
            continue
        lines.extend(
            [
                "### `{}`".format(
                    record[
                        "candidate_id"
                    ]
                ),
                "",
                "| t | Direction gate | Scientific witness |",
                "|---:|---:|---|",
            ]
        )
        for timestep in ("10", "25", "50"):
            item = record[
                "timestep_records"
            ][timestep]
            lines.append(
                "| {} | {} | {} |".format(
                    timestep,
                    str(
                        item[
                            "direction_gates"
                        ]["all"]
                    ).lower(),
                    _witness_text(
                        item[
                            "scale_sweep"
                        ][
                            "scientific_witness"
                        ]
                    ),
                )
            )
        lines.append("")

    lines.extend(
        [
            "## Locked selection",
            "",
            "- Objective-train selected configuration: `{}`".format(
                (
                    "none"
                    if selected_oof is None
                    else selected_oof[
                        "candidate_id"
                    ]
                )
            ),
            "- Validated train-only recommendation: `{}`".format(
                (
                    "none"
                    if validated is None
                    else validated[
                        "candidate_id"
                    ]
                )
            ),
            "- Permutation control pass: `{}`".format(
                (
                    "not-run"
                    if permutation is None
                    else str(
                        permutation[
                            "all_pass"
                        ]
                    ).lower()
                )
            ),
            "- Locked holdout scientific pass: `{}`".format(
                (
                    "not-run"
                    if holdout is None
                    else str(
                        holdout[
                            "scientific_pass"
                        ]
                    ).lower()
                )
            ),
            "- Eligible candidates: `{}`".format(
                ", ".join(
                    classification[
                        "eligible_candidate_ids"
                    ]
                )
                or "none"
            ),
            "- Primary failure locus: `{}`".format(
                classification[
                    "primary_failure_locus"
                ]
            ),
            "- Contract SHA256: `{}`".format(
                result[
                    "direction_surrogate_contract"
                ][
                    "contract_sha256"
                ]
            ),
            "- Selection SHA256: `{}`".format(
                result[
                    "selection"
                ][
                    "selection_sha256"
                ]
            ),
            "",
            "## Boundary",
            "",
            "- Surrogate candidates were fitted only on the 638-row objective-training population.",
            "- Candidate and scale selection used only six-fold grouped OOF objective-train predictions.",
            "- The 236-row selection holdout could not change candidate, features, regularization, output rank, scale, or thresholds.",
            "- Group IDs, pair keys, visible seeds, and window indices were excluded from all features.",
            "- Absolute-coordinate features were diagnostic-only and could not be selected.",
            "- No diffusion-model candidate was trained and no model architecture was changed.",
            "- The frozen probe was not accessed.",
            "- No reverse sampling, formal training, IDM, candidate execution, DeformableRavens, Phase4, or CPS was run.",
            "- No checkpoint, weights, surrogate weights, prediction tensor, oracle tensor, candidate tensor, NPZ, cache, image, or video was persisted.",
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
                "Stage-D write-once output exists: {}".format(path)
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
            "Stage-D test gate is not PASS"
        )

    temporary_root = Path(
        tempfile.mkdtemp(
            prefix="phase314b_r258_staged_",
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
            "phase3_14b_r258_staged_worker.py"
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
                "isolated Stage-D workers differ"
            )
        result = worker_results[0]

        contract_record = {
            "phase": PHASE,
            **result[
                "direction_surrogate_contract"
            ],
            "root_cause":
                result["root_cause"],
            "required_next_path":
                result["required_next_path"],
            "scientific_status":
                result["scientific_status"],
            "selection_sha256":
                result["selection"][
                    "selection_sha256"
                ],
            "selected_configuration":
                result[
                    "selected_configuration"
                ],
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
                "phase314b_r258_staged_worker_evidence_v1",
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
                "phase314b_r258_staged_summary_v1",
            "verdict": "PASS",
            "scientific_status":
                result["scientific_status"],
            "root_cause":
                result["root_cause"],
            "required_next_path":
                result["required_next_path"],
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
            "worker_result":
                result,
            "selected_configuration":
                result[
                    "selected_configuration"
                ],
            "train_only_recommendation":
                result[
                    "train_only_recommendation"
                ],
            "direction_surrogate_trained":
                True,
            "new_diffusion_model_candidate_trained":
                False,
            "selection_holdout_evaluated":
                result[
                    "selection_holdout_evaluated"
                ],
            "selection_holdout_used_for_fit":
                False,
            "selection_holdout_used_for_selection":
                False,
            "frozen_probe_accessed":
                False,
            "reverse_sampling_run":
                False,
            "full_stageb_repaired_model_trained":
                False,
            "formal_pilot_run":
                False,
            "checkpoint_saved":
                False,
            "weights_persisted":
                False,
            "surrogate_weights_persisted":
                False,
            "prediction_tensor_persisted":
                False,
            "oracle_tensor_persisted":
                False,
            "candidate_tensor_persisted":
                False,
            "npz_saved":
                False,
            "cache_saved":
                False,
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
            "phase4":
                False,
            "cps":
                False,
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
                    summary[
                        "scientific_status"
                    ],
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
