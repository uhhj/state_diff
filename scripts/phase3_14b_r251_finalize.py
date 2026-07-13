#!/usr/bin/env python3
"""Finalize Phase3.14b-r2.5.1 gradient-calibration audit."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping

from ccda_phase3.phase314b_r23_diagnostics import require_repository_state, write_json_once
from ccda_phase3.phase314b_r251_gradient_calibration import (
    CALIBRATED_GEOMETRY_OBJECTIVES,
    EXPECTED_CACHE_SHA256,
    EXPECTED_CONTRACT_SHA256,
    EXPECTED_PAIRED_ROWS,
    EXPECTED_SUBMODULE_COMMIT,
    OBJECTIVE_MATRIX_SCHEMA,
    PAIRED_INVERSION_SCHEMA,
    PHASE,
    assert_only_allowed_worktree_paths,
    calibrated_objective_names,
    classify_pilot,
    source_sha256,
    validate_advancing_objective_order,
    validate_and_order_objective_mapping,
    validate_explicit_objective_order,
)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be object: {path}")
    return value


def assert_finite_json(value: Any, path: str = "root") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite numeric value at {path}")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            assert_finite_json(item, f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            assert_finite_json(item, f"{path}[{index}]")
        return
    raise TypeError(f"unsupported JSON-like value at {path}: {type(value).__name__}")


def compact_unique(value: Mapping[str, Any]) -> Dict[str, Any]:
    calibration = value.get("calibration", {})
    tracking = value.get("gradient_tracking", {})
    base = {
        "pass": bool(value.get("pass", False)),
        "failure": value.get("failure"),
        "family": str(value.get("geometry_objective", {}).get("family", "unknown")),
        "target_gradient_ratio": float(
            value.get("geometry_objective", {}).get("target_gradient_ratio", 0.0)
        ),
        "calibration_pass": bool(calibration.get("pass", False)),
        "gradient_multiplier": float(calibration.get("multiplier", 0.0)),
        "calibration_raw_ratio_median": float(
            calibration.get("raw_ratio_median", 0.0)
        ),
        "gradient_tracking_pass": bool(tracking.get("pass", False)),
        "observed_gradient_ratio_median": float(
            tracking.get("observed_ratio_median", 0.0)
        ),
        "observed_gradient_ratio_p95": float(
            tracking.get("observed_ratio_p95", 0.0)
        ),
        "prior_drift_ratio": float(value.get("prior_drift_ratio", 1.0)),
        "condition_effect": bool(
            value.get("condition_effect", {}).get(
                "condition_effect_supported", False
            )
        ),
        "z_mse": None,
        "ordered_rmse_p95": None,
        "segment_relative_p95": None,
        "sample_validity_rate": None,
    }
    evaluations = value.get("evaluations", {})
    true_value = evaluations.get("true") if isinstance(evaluations, Mapping) else None
    if isinstance(true_value, Mapping):
        true_metrics = true_value["aggregate"]["metrics"]
        base.update(
            {
                "z_mse": float(true_metrics["z_mse"]),
                "ordered_rmse_p95": float(true_metrics["ordered_rmse"]["p95"]),
                "segment_relative_p95": float(
                    true_metrics["segment_relative_error"]["p95"]
                ),
                "sample_validity_rate": float(
                    value["one_step_physical"]["sample_validity_rate"]
                ),
            }
        )
    return base


def compact_paired(value: Mapping[str, Any]) -> Dict[str, Any]:
    base = {
        "one_step_pass": bool(value.get("one_step_pass", False)),
        "one_step_branch_pass": bool(
            value.get("one_step_branch_audit", {}).get("pass", False)
        ),
        "failure": value.get("failure"),
        "target_gradient_ratio": float(
            value.get("geometry_objective", {}).get("target_gradient_ratio", 0.0)
        ),
        "gradient_multiplier": float(
            value.get("calibration", {}).get("multiplier", 0.0)
        ),
        "observed_gradient_ratio_median": float(
            value.get("gradient_tracking", {}).get("observed_ratio_median", 0.0)
        ),
        "prior_drift_ratio": float(value.get("prior_drift_ratio", 1.0)),
        "reverse_sample_validity_rate": None,
        "reverse_query_has_valid_candidate_rate": None,
        "reverse_best_ordered_rmse_mean": None,
        "reverse_k1_ordered_rmse_mean": None,
        "reverse_segment_score_p95": None,
        "reverse_nearest_inversion_mean": None,
        "reverse_nearest_inversion_p95": None,
        "reverse_nearest_inversion_max": None,
        "reverse_nearest_unique_fraction_mean": None,
        "reverse_nearest_unique_fraction_p05": None,
        "reverse_pool_diversity": None,
        "both_branch_support_rate": None,
        "two_branch_occupancy_rate": None,
        "branch_support_pass": False,
        "comparison_to_control_pass": False,
        "comparison_segment_score_ratio": None,
        "comparison_validity_delta": None,
        "comparison_ordered_rmse_ratio": None,
        "comparison_inversion_ratio": None,
    }
    reverse = value.get("reverse_metrics", {})
    calibrated = reverse.get("calibrated") if isinstance(reverse, Mapping) else None
    branch = value.get("full_reverse_branch_support", {})
    comparison = value.get("comparison_to_control", {})
    if isinstance(calibrated, Mapping):
        base.update(
            {
                "reverse_sample_validity_rate": float(
                    calibrated["sample_validity_rate"]
                ),
                "reverse_query_has_valid_candidate_rate": float(
                    calibrated["query_has_valid_candidate_rate"]
                ),
                "reverse_best_ordered_rmse_mean": float(
                    reverse["best_ordered_rmse_mean"]
                ),
                "reverse_k1_ordered_rmse_mean": float(
                    reverse["k1_ordered_rmse_mean"]
                ),
                "reverse_segment_score_p95": float(
                    calibrated["segment_score_p95"]
                ),
                "reverse_nearest_inversion_mean": float(
                    reverse["nearest_inversion_mean"]
                ),
                "reverse_nearest_inversion_p95": float(
                    reverse["nearest_inversion_p95"]
                ),
                "reverse_nearest_inversion_max": float(
                    reverse["nearest_inversion_max"]
                ),
                "reverse_nearest_unique_fraction_mean": float(
                    reverse["nearest_unique_fraction_mean"]
                ),
                "reverse_nearest_unique_fraction_p05": float(
                    reverse["nearest_unique_fraction_p05"]
                ),
                "reverse_pool_diversity": float(reverse["pool_diversity"]),
                "both_branch_support_rate": float(
                    branch["both_branch_support_rate"]
                ),
                "two_branch_occupancy_rate": float(
                    branch["two_branch_occupancy_rate"]
                ),
                "branch_support_pass": bool(branch["pass"]),
                "comparison_to_control_pass": bool(comparison["pass"]),
                "comparison_segment_score_ratio": float(
                    comparison["segment_score_p95_ratio"]
                ),
                "comparison_validity_delta": float(
                    comparison["sample_validity_delta"]
                ),
                "comparison_ordered_rmse_ratio": float(
                    comparison["best_ordered_rmse_ratio"]
                ),
                "comparison_inversion_ratio": float(
                    comparison["nearest_inversion_p95_ratio"]
                ),
            }
        )
    return base


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}g}"


def markdown_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Phase3.14b-r2.5.1 Geometry-Gradient Calibration Audit",
        "",
        "## Verdict",
        "",
        f"- Verdict: `{summary['verdict']}` (train-only diagnostic)",
        f"- Root cause: `{summary['root_cause']}`",
        f"- Train-only recommendation: `{summary['train_only_recommendation']}`",
        "- Selected configuration: `None`",
        "",
        "## Fixed contract",
        "",
        "- Model: `frozen_p512_r512`",
        "- Gradient calibration: actual residual-gradient norms",
        "- Target ratios: `0.10 / 0.50 / 1.00`",
        "- Reverse audit: `100 steps`, `K=16`",
        "",
        "## Unique-free calibration",
        "",
        "| Variant | PASS | Target | Multiplier | Observed median | Observed p95 | Prior drift | Ordered p95 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, value in summary["unique_free_variants"].items():
        lines.append(
            "| `{}` | {} | {} | {} | {} | {} | {} | {} |".format(
                name,
                str(value["pass"]).lower(),
                _fmt(value["target_gradient_ratio"], 4),
                _fmt(value["gradient_multiplier"]),
                _fmt(value["observed_gradient_ratio_median"]),
                _fmt(value["observed_gradient_ratio_p95"]),
                _fmt(value["prior_drift_ratio"]),
                _fmt(value["ordered_rmse_p95"]),
            )
        )
    lines.extend(
        [
            "",
            "## Paired and full reverse",
            "",
            "| Variant | One-step | Validity | Valid-query | Best K | Inversion p95 | Both branches | Compare control |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, value in summary["paired_variants"].items():
        lines.append(
            "| `{}` | {} | {} | {} | {} | {} | {} | {} |".format(
                name,
                str(value["one_step_pass"]).lower(),
                _fmt(value["reverse_sample_validity_rate"]),
                _fmt(value["reverse_query_has_valid_candidate_rate"]),
                _fmt(value["reverse_best_ordered_rmse_mean"]),
                "{}/{}/{}".format(
                    _fmt(value["reverse_nearest_inversion_mean"]),
                    _fmt(value["reverse_nearest_inversion_p95"]),
                    _fmt(value["reverse_nearest_inversion_max"]),
                ),
                _fmt(value["both_branch_support_rate"]),
                str(value["comparison_to_control_pass"]).lower(),
            )
        )
    lines.extend(
        [
            "",
            "## Boundaries",
            "",
            "- Validation targets used: `False`",
            "- Formal test read: `False`",
            "- Formal training: `False`",
            "- Candidate selected: `False`",
            "- Checkpoint saved: `False`",
            "- IDM / candidate execution: `False`",
            "- Phase4 / CPS: `False`",
            "",
            "A PASS verdict means only that the train-only diagnostic chain completed.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument(
        "--preflight-report",
        default="reports/phase3_14b_r251_resume2_preflight_summary.json",
    )
    parser.add_argument(
        "--pilot-report",
        default="reports/phase3_14b_r251_pilot_summary.json",
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()
    preflight_path = (root / args.preflight_report).resolve()
    pilot_path = (root / args.pilot_report).resolve()
    preflight_relative = preflight_path.relative_to(root).as_posix()
    pilot_relative = pilot_path.relative_to(root).as_posix()

    repository = require_repository_state(root, require_clean=False)
    assert_only_allowed_worktree_paths(root, (preflight_relative, pilot_relative))
    if repository["submodule_commit"] != EXPECTED_SUBMODULE_COMMIT:
        raise RuntimeError("submodule commit mismatch")
    if repository["cache_sha256"] != EXPECTED_CACHE_SHA256:
        raise RuntimeError("cache SHA mismatch")
    if repository["frozen_contract_sha256"] != EXPECTED_CONTRACT_SHA256:
        raise RuntimeError("frozen-contract SHA mismatch")

    preflight = load_json(preflight_path)
    pilot = load_json(pilot_path)
    current_source = source_sha256(root)
    if preflight.get("r251_source_sha256") != current_source:
        raise RuntimeError("source changed after preflight")
    if pilot.get("source_sha256") != current_source:
        raise RuntimeError("pilot source provenance mismatch")
    resume = preflight.get("resume")
    if not isinstance(resume, Mapping) or resume.get("generation") != 2:
        raise RuntimeError("Resume2 preflight provenance missing")
    if resume.get("correction") != "json_object_order_not_semantic_objective_contract":
        raise RuntimeError("Resume2 correction provenance mismatch")
    if pilot.get("resume") != resume:
        raise RuntimeError("pilot Resume2 provenance mismatch")
    if pilot.get("phase") != PHASE or pilot.get("verdict") != "PASS":
        raise RuntimeError("pilot did not complete")
    if pilot.get("selected_configuration") is not None:
        raise RuntimeError("train-only pilot selected a formal configuration")
    if pilot.get("paired_nearest_inversion_instrumented") is not True:
        raise RuntimeError("paired nearest-index inversion was not instrumented")
    if pilot.get("paired_nearest_inversion_schema") != PAIRED_INVERSION_SCHEMA:
        raise RuntimeError("paired nearest-index inversion schema mismatch")

    paired_contract = pilot.get("dataset", {}).get("paired_row_contract", {})
    if paired_contract.get("pass") is not True:
        raise RuntimeError("paired-row contract did not pass")
    if paired_contract.get("observed_rows") != list(EXPECTED_PAIRED_ROWS):
        raise RuntimeError("paired-row identity changed")

    boundary = {
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
    for key, expected in boundary.items():
        if pilot.get(key) != expected:
            raise RuntimeError(f"pilot boundary mismatch for {key}")

    expected_names = calibrated_objective_names()
    if pilot.get("objective_matrix_schema") != OBJECTIVE_MATRIX_SCHEMA:
        raise RuntimeError("pilot objective-matrix schema mismatch")
    preflight_contract = preflight.get("objective_matrix_contract", {})
    if preflight_contract.get("schema") != OBJECTIVE_MATRIX_SCHEMA:
        raise RuntimeError("preflight objective-matrix schema mismatch")
    validate_explicit_objective_order(
        preflight_contract.get("objective_order"),
        expected_names=expected_names,
        name="preflight objective_order",
    )
    validate_explicit_objective_order(
        pilot.get("objective_order"),
        expected_names=expected_names,
        name="pilot objective_order",
    )

    unique_values = validate_and_order_objective_mapping(
        pilot.get("unique_free_variants", {}),
        expected_names=expected_names,
        name="unique_free_variants",
    )
    advancing_names = validate_advancing_objective_order(
        pilot.get("unique_advancing_variants"),
        expected_names=expected_names,
        unique_variants=unique_values,
    )
    paired_expected_names = tuple(
        name for name in expected_names if name in set(advancing_names)
    )
    if list(pilot.get("paired_objective_order", [])) != list(paired_expected_names):
        raise RuntimeError(
            "paired_objective_order changed: "
            f"expected={list(paired_expected_names)}, "
            f"observed={pilot.get('paired_objective_order')}"
        )
    paired_values = (
        validate_and_order_objective_mapping(
            pilot.get("paired_variants", {}),
            expected_names=paired_expected_names,
            name="paired_variants",
        )
        if paired_expected_names
        else {}
    )

    objective_by_name = {
        objective.name: objective for objective in CALIBRATED_GEOMETRY_OBJECTIVES
    }
    for name, value in unique_values.items():
        objective = value.get("geometry_objective", {})
        expected = objective_by_name[name]
        if objective.get("name") != name:
            raise RuntimeError(f"unique variant internal name mismatch for {name}")
        if objective.get("family") != expected.family:
            raise RuntimeError(f"unique variant family mismatch for {name}")
        if float(objective.get("target_gradient_ratio", -1.0)) != float(
            expected.target_gradient_ratio
        ):
            raise RuntimeError(f"unique variant target-ratio mismatch for {name}")

    required_reverse_fields = (
        "schema",
        "batched_item_count",
        "nearest_inversion_mean",
        "nearest_inversion_p95",
        "nearest_inversion_max",
        "nearest_unique_fraction_mean",
        "nearest_unique_fraction_p05",
        "nearest_unique_fraction_min",
    )
    for name, value in paired_values.items():
        reverse = value.get("reverse_metrics", {}) if isinstance(value, Mapping) else {}
        if not reverse:
            continue
        missing = [field for field in required_reverse_fields if field not in reverse]
        if missing:
            raise RuntimeError(
                f"paired reverse metrics missing for {name}: {', '.join(missing)}"
            )
        if reverse.get("schema") != PAIRED_INVERSION_SCHEMA:
            raise RuntimeError(f"paired inversion schema mismatch for {name}")
        if int(reverse.get("batched_item_count", -1)) != 128:
            raise RuntimeError(f"paired inversion batch count mismatch for {name}")

    recomputed = classify_pilot(
        {
            "unique_free_variants": unique_values,
            "paired_variants": paired_values,
        }
    )
    for key in ("root_cause", "next_stage", "train_only_recommendation"):
        if pilot.get(key) != recomputed.get(key):
            raise RuntimeError(
                f"pilot classifier output mismatch for {key}: "
                f"{pilot.get(key)!r} != {recomputed.get(key)!r}"
            )

    unique_compact = {
        name: compact_unique(unique_values[name]) for name in expected_names
    }
    paired_compact = {
        name: compact_paired(paired_values[name]) for name in paired_expected_names
    }
    recommendation = pilot.get("train_only_recommendation")
    if recommendation == "v_only_frozen_control":
        raise RuntimeError("control cannot be recommended")
    if recommendation is not None and recommendation not in paired_compact:
        raise RuntimeError("recommendation lacks paired evidence")

    summary: Dict[str, Any] = {
        "phase": PHASE,
        "verdict": "PASS",
        "meaning": "train-only geometry-gradient calibration completed; no formal selection",
        "repository": repository,
        "preflight_report": preflight_relative,
        "resume": dict(resume),
        "pilot_report": pilot_relative,
        "source_sha256": current_source,
        "cache_sha256": EXPECTED_CACHE_SHA256,
        "frozen_contract_sha256": EXPECTED_CONTRACT_SHA256,
        "fixed_model_contract": pilot["fixed_model_contract"],
        "geometry_scales": pilot["geometry_scales"],
        "objective_matrix_schema": OBJECTIVE_MATRIX_SCHEMA,
        "objective_order": list(expected_names),
        "paired_objective_order": list(paired_expected_names),
        "gradient_calibration_contract": pilot["gradient_calibration_contract"],
        "unique_free_variants": unique_compact,
        "unique_advancing_variants": list(advancing_names),
        "paired_variants": paired_compact,
        "paired_nearest_inversion_instrumented": True,
        "paired_nearest_inversion_schema": PAIRED_INVERSION_SCHEMA,
        "paired_row_contract": paired_contract,
        "root_cause": pilot["root_cause"],
        "next_stage": pilot["next_stage"],
        "train_only_recommendation": recommendation,
        **boundary,
    }
    assert_finite_json(summary)
    write_json_once(root / "reports/phase3_14b_r251_summary.json", summary)
    report_path = root / "reports/phase3_14b_r251_report.md"
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite {report_path}")
    report_path.write_text(markdown_report(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
