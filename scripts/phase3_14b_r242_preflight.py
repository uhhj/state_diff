#!/usr/bin/env python3
"""Preflight for Phase3.14b-r2.4.2 train-only frozen-prior pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ccda_phase3.phase314b_r23_diagnostics import (
    require_repository_state,
    write_json_once,
)
from ccda_phase3.phase314b_r241_multirow import dependency_sha256 as r241_dependency_sha256
from ccda_phase3.phase314b_r242_frozen_prior import (
    BASE_COMMIT,
    EXPECTED_CACHE_SHA256,
    EXPECTED_CONTRACT_SHA256,
    EXPECTED_R23_TINY_SHA256,
    EXPECTED_R241_MODULE_SHA256,
    EXPECTED_R241_ROOT_CAUSE,
    EXPECTED_R24_MODULE_SHA256,
    EXPECTED_SUBMODULE_COMMIT,
    PHASE,
    EVALUATION_RESULT_SCHEMA_VERSION,
    PRIOR_RESULT_SCHEMA_VERSION,
    PRIOR_SEED_RUN_SCHEMA_VERSION,
    PRIOR_SEED_STABILITY_SCHEMA_VERSION,
    RECONSTRUCTION_METRICS_SCHEMA_VERSION,
    EXPECTED_R242_BLOCKED_ROOT_CAUSE,
    R242_BLOCKED_REPORT_COMMIT,
    R242_SCHEMA_CORRECTION_COMMIT,
    R242_RESUME_BLOCKED_REPORT_COMMIT,
    R242_NESTED_SCHEMA_CORRECTION_COMMIT,
    R242_RESUME2_BLOCKED_REPORT_COMMIT,
    corrected_r241_interpretation,
    git_output,
    sha256_file,
    source_sha256,
)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument(
        "--output",
        default="reports/phase3_14b_r242_preflight_summary.json",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    repository = require_repository_state(root, require_clean=True)
    if repository["branch"] != "Experiment1":
        raise RuntimeError("r2.4.2 requires Experiment1")
    if repository["submodule_commit"] != EXPECTED_SUBMODULE_COMMIT:
        raise RuntimeError("submodule commit mismatch")
    if repository["cache_sha256"] != EXPECTED_CACHE_SHA256:
        raise RuntimeError("cache SHA mismatch")
    if repository["frozen_contract_sha256"] != EXPECTED_CONTRACT_SHA256:
        raise RuntimeError("frozen contract SHA mismatch")

    subprocess_result = git_output(
        root,
        "merge-base",
        "--is-ancestor",
        BASE_COMMIT,
        "HEAD",
    )
    if subprocess_result:
        # git merge-base --is-ancestor emits no stdout on success.
        raise RuntimeError("unexpected merge-base output")

    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    output = output.resolve()
    try:
        output.relative_to(root)
    except ValueError as exc:
        raise RuntimeError("preflight output must remain inside repository") from exc

    resume_specs = {
        "phase3_14b_r242_resume_preflight_summary.json": {
            "generation": 1,
            "blocked_report_commit": R242_BLOCKED_REPORT_COMMIT,
            "blocked_report": "reports/phase3_14b_r242_blocked_summary.json",
            "correction_commit": R242_SCHEMA_CORRECTION_COMMIT,
            "correction": "direct_prior_top_level_result_schema",
        },
        "phase3_14b_r242_resume2_preflight_summary.json": {
            "generation": 2,
            "blocked_report_commit": R242_RESUME_BLOCKED_REPORT_COMMIT,
            "blocked_report": (
                "reports/phase3_14b_r242_resume_blocked_summary.json"
            ),
            "correction_commit": R242_NESTED_SCHEMA_CORRECTION_COMMIT,
            "correction": "nested_reconstruction_metric_consumer_schema",
        },
        "phase3_14b_r242_resume3_preflight_summary.json": {
            "generation": 3,
            "blocked_report_commit": R242_RESUME2_BLOCKED_REPORT_COMMIT,
            "blocked_report": (
                "reports/phase3_14b_r242_resume2_blocked_summary.json"
            ),
            "correction_commit": None,
            "correction": "prior_result_aggregate_wrapper_removal",
        },
    }
    resume_spec = resume_specs.get(output.name)
    resume_mode = resume_spec is not None
    if output.name not in {
        "phase3_14b_r242_preflight_summary.json",
        *resume_specs.keys(),
    }:
        raise RuntimeError("unsupported r2.4.2 preflight output name")

    if resume_mode:
        blocked_ancestor = git_output(
            root,
            "merge-base",
            "--is-ancestor",
            resume_spec["blocked_report_commit"],
            "HEAD",
        )
        if blocked_ancestor:
            raise RuntimeError("unexpected blocked-commit merge-base output")
        correction_commit = resume_spec.get("correction_commit")
        if correction_commit is not None:
            correction_ancestor = git_output(
                root,
                "merge-base",
                "--is-ancestor",
                correction_commit,
                "HEAD",
            )
            if correction_ancestor:
                raise RuntimeError(
                    "unexpected correction-commit merge-base output"
                )
        blocked_path = root / resume_spec["blocked_report"]
        blocked = load_json(blocked_path)
        if blocked.get("verdict") != "BLOCKED":
            raise RuntimeError("historical r2.4.2 blocked verdict mismatch")
        if blocked.get("root_cause") != EXPECTED_R242_BLOCKED_ROOT_CAUSE:
            raise RuntimeError("historical r2.4.2 blocked root cause mismatch")
        if resume_spec["generation"] == 2:
            mismatch = blocked.get("schema_mismatch")
            expected_mismatch = {
                "consumer_expected": "metrics.ordered_rmse_p95",
                "producer_emitted": "metrics.ordered_rmse.p95",
            }
            if mismatch != expected_mismatch:
                raise RuntimeError(
                    "resume2 blocked report schema mismatch evidence changed"
                )
            if blocked.get("failure_stage") != (
                "direct_prior_width_512_seed_stability_summary"
            ):
                raise RuntimeError("resume2 failure stage mismatch")
        if resume_spec["generation"] == 3:
            if blocked.get("exception_type") != "KeyError":
                raise RuntimeError("resume3 exception type mismatch")
            if blocked.get("exception_message") != "'aggregate'":
                raise RuntimeError("resume3 exception message mismatch")
            if blocked.get("failure_stage") != (
                "unique_free_first_factorized_variant_prior_drift"
            ):
                raise RuntimeError("resume3 failure stage mismatch")
        for key in (
            "validation_targets_used",
            "formal_test_read",
            "formal_training",
            "candidate_eligible",
            "checkpoint_saved",
            "idm",
            "candidate_execution",
            "phase4",
            "cps",
        ):
            if bool(blocked.get(key)):
                raise RuntimeError(
                    f"historical blocked report violates boundary: {key}"
                )

    r241_summary_path = root / "reports/phase3_14b_r241_summary.json"
    r241_summary = load_json(r241_summary_path)
    if r241_summary.get("root_cause") != EXPECTED_R241_ROOT_CAUSE:
        raise RuntimeError("unexpected r2.4.1 root cause")
    if r241_summary.get("selected_configuration") is not None:
        raise RuntimeError("r2.4.1 unexpectedly selected a configuration")
    if bool(r241_summary.get("validation_targets_used")):
        raise RuntimeError("r2.4.1 used validation targets")
    if bool(r241_summary.get("formal_test_read")):
        raise RuntimeError("r2.4.1 read formal test")

    module_hash = sha256_file(
        root / "ccda_phase3/phase314b_r241_multirow.py"
    )
    if module_hash != EXPECTED_R241_MODULE_SHA256:
        raise RuntimeError(
            "r2.4.1 module SHA mismatch: "
            f"{module_hash} != {EXPECTED_R241_MODULE_SHA256}"
        )
    r24_hash = sha256_file(
        root / "ccda_phase3/phase314b_r24_noisy_skip.py"
    )
    if r24_hash != EXPECTED_R24_MODULE_SHA256:
        raise RuntimeError("r2.4 module SHA mismatch")

    historical_tiny = sha256_file(
        root / "reports/phase3_14b_r23_tiny_overfit_summary.json"
    )
    if historical_tiny != EXPECTED_R23_TINY_SHA256:
        raise RuntimeError("historical r2.3 tiny report changed")

    corrected = corrected_r241_interpretation(r241_summary)
    if not corrected["classifier_precedence_bug_supported"]:
        raise RuntimeError(
            "r2.4.1 evidence no longer supports classifier-precedence audit"
        )

    report = {
        "phase": PHASE,
        "verdict": "PASS",
        "meaning": (
            "immutable evidence and corrected r2.4.1 interpretation verified"
        ),
        "repository": repository,
        "base_commit": BASE_COMMIT,
        "main_head": git_output(root, "rev-parse", "HEAD"),
        "submodule_commit": EXPECTED_SUBMODULE_COMMIT,
        "cache_sha256": EXPECTED_CACHE_SHA256,
        "frozen_contract_sha256": EXPECTED_CONTRACT_SHA256,
        "historical_r23_tiny_sha256": historical_tiny,
        "r24_module_sha256": r24_hash,
        "r241_module_sha256": module_hash,
        "r241_corrected_interpretation": corrected,
        "r241_dependency_sha256": r241_dependency_sha256(root),
        "r242_source_sha256": source_sha256(root),
        "result_schema_contract": {
            "prior_seed_run": PRIOR_SEED_RUN_SCHEMA_VERSION,
            "prior_seed_stability": PRIOR_SEED_STABILITY_SCHEMA_VERSION,
            "reconstruction_metrics": RECONSTRUCTION_METRICS_SCHEMA_VERSION,
            "prior_result": PRIOR_RESULT_SCHEMA_VERSION,
            "evaluation_result": EVALUATION_RESULT_SCHEMA_VERSION,
            "prior_metrics_path": "metrics",
            "evaluation_metrics_path": "aggregate.metrics",
            "obsolete_prior_aggregate_wrapper_forbidden": True,
            "ordered_rmse_p95_path": "metrics.ordered_rmse.p95",
            "flat_ordered_rmse_alias_forbidden": True,
            "three_distinct_seeds_required": True,
            "single_hidden_dim_required": True,
        },
        "resume": {
            "enabled": resume_mode,
            "generation": (
                int(resume_spec["generation"]) if resume_mode else 0
            ),
            "blocked_report_commit": (
                resume_spec["blocked_report_commit"] if resume_mode else None
            ),
            "blocked_report": (
                resume_spec["blocked_report"] if resume_mode else None
            ),
            "blocked_root_cause": (
                EXPECTED_R242_BLOCKED_ROOT_CAUSE if resume_mode else None
            ),
            "correction_commit": (
                resume_spec.get("correction_commit") if resume_mode else None
            ),
            "correction": (
                resume_spec["correction"] if resume_mode else None
            ),
        },
        "validation_targets_used": False,
        "formal_test_read": False,
        "formal_training": False,
        "candidate_eligible": False,
        "selected_configuration": None,
        "checkpoint_saved": False,
        "idm": False,
        "candidate_execution": False,
        "phase4": False,
        "cps": False,
    }
    write_json_once(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
