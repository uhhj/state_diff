#!/usr/bin/env python3
"""Finalize Phase3.14b-r2.5 train-only geometry repair audit."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping

from ccda_phase3.phase314b_r23_diagnostics import (
    require_repository_state,
    write_json_once,
)
from ccda_phase3.phase314b_r25_ordered_geometry import (
    EXPECTED_CACHE_SHA256,
    EXPECTED_CONTRACT_SHA256,
    EXPECTED_SUBMODULE_COMMIT,
    PHASE,
    assert_only_allowed_worktree_paths,
    source_sha256,
)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
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
    true_metrics = value["evaluations"]["true"]["aggregate"]["metrics"]
    ordered = true_metrics["ordered_rmse"]
    segment = true_metrics["segment_relative_error"]
    return {
        "pass": bool(value["pass"]),
        "geometry_gradient_gate": bool(value["geometry_gradient_gate"]),
        "geometry_to_v_gradient_ratio_median": float(
            value["geometry_to_v_gradient_ratio_median"]
        ),
        "prior_drift_ratio": float(value["prior_drift_ratio"]),
        "condition_effect": bool(
            value["condition_effect"]["condition_effect_supported"]
        ),
        "z_mse": float(true_metrics["z_mse"]),
        "ordered_rmse_p95": float(ordered["p95"]),
        "segment_relative_p95": float(segment["p95"]),
        "sample_validity_rate": float(
            value["one_step_physical"]["sample_validity_rate"]
        ),
    }


def compact_paired(value: Mapping[str, Any]) -> Dict[str, Any]:
    reverse = value["reverse_metrics"]
    calibrated = reverse["calibrated"]
    branch = value["full_reverse_branch_support"]
    comparison = value["comparison_to_control"]
    return {
        "one_step_pass": bool(value["one_step_pass"]),
        "one_step_branch_pass": bool(value["one_step_branch_audit"]["pass"]),
        "reverse_sample_validity_rate": float(calibrated["sample_validity_rate"]),
        "reverse_query_has_valid_candidate_rate": float(
            calibrated["query_has_valid_candidate_rate"]
        ),
        "reverse_best_ordered_rmse_mean": float(
            reverse["best_ordered_rmse_mean"]
        ),
        "reverse_segment_score_p95": float(calibrated["segment_score_p95"]),
        "both_branch_support_rate": float(branch["both_branch_support_rate"]),
        "two_branch_occupancy_rate": float(branch["two_branch_occupancy_rate"]),
        "branch_support_pass": bool(branch["pass"]),
        "comparison_to_control_pass": bool(comparison["pass"]),
        "comparison_segment_score_ratio": float(
            comparison["segment_score_p95_ratio"]
        ),
        "comparison_validity_delta": float(comparison["sample_validity_delta"]),
        "comparison_ordered_rmse_ratio": float(
            comparison["best_ordered_rmse_ratio"]
        ),
    }


def markdown_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Phase3.14b-r2.5 Frozen-Prior Ordered-Geometry Pilot",
        "",
        "## Verdict",
        "",
        f"- Verdict: `{summary['verdict']}` (train-only diagnostic; not formal model repair)",
        f"- Root cause: `{summary['root_cause']}`",
        f"- Train-only recommendation: `{summary['train_only_recommendation']}`",
        "- Selected configuration: `None`",
        "",
        "## Fixed model contract",
        "",
        "- Architecture: `factorized_analytic_x0_skip_mlp`",
        "- Prior/residual widths: `512 / 512`",
        "- Prior policy: `strict frozen after warmup`",
        "- Objective: `v_prediction` with optional train-only ordered-geometry loss",
        "- Reverse audit: `100 official scheduler steps`, `K=16`",
        "",
        "## Unique-free controls",
        "",
        "| Variant | PASS | Gradient ratio | Prior drift | z MSE | Ordered p95 | Validity |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, value in summary["unique_free_variants"].items():
        lines.append(
            "| `{}` | {} | {:.6g} | {:.6g} | {:.6g} | {:.6g} | {:.6g} |".format(
                name,
                str(value["pass"]).lower(),
                value["geometry_to_v_gradient_ratio_median"],
                value["prior_drift_ratio"],
                value["z_mse"],
                value["ordered_rmse_p95"],
                value["sample_validity_rate"],
            )
        )
    lines.extend(
        [
            "",
            "## Paired low/mid and full-reverse controls",
            "",
            "| Variant | One-step | Reverse validity | Valid-query | Both branches | Compare control |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, value in summary["paired_variants"].items():
        lines.append(
            "| `{}` | {} | {:.6g} | {:.6g} | {:.6g} | {} |".format(
                name,
                str(value["one_step_pass"]).lower(),
                value["reverse_sample_validity_rate"],
                value["reverse_query_has_valid_candidate_rate"],
                value["both_branch_support_rate"],
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
            "A PASS verdict means only that the train-only diagnostic evidence chain completed.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument(
        "--preflight-report",
        default="reports/phase3_14b_r25_preflight_summary.json",
    )
    parser.add_argument(
        "--pilot-report",
        default="reports/phase3_14b_r25_pilot_summary.json",
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
    if preflight.get("r25_source_sha256") != current_source:
        raise RuntimeError("source changed after preflight")
    if pilot.get("source_sha256") != current_source:
        raise RuntimeError("pilot source provenance mismatch")
    if pilot.get("phase") != PHASE or pilot.get("verdict") != "PASS":
        raise RuntimeError("pilot did not complete")
    if pilot.get("selected_configuration") is not None:
        raise RuntimeError("train-only pilot selected a formal configuration")

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

    unique_compact = {
        name: compact_unique(value)
        for name, value in pilot["unique_free_variants"].items()
    }
    paired_compact = {
        name: compact_paired(value)
        for name, value in pilot["paired_variants"].items()
    }
    recommendation = pilot.get("train_only_recommendation")
    if recommendation == "v_only_frozen_control":
        raise RuntimeError("control cannot be recommended")
    if recommendation is not None and recommendation not in paired_compact:
        raise RuntimeError("recommendation lacks paired evidence")

    summary: Dict[str, Any] = {
        "phase": PHASE,
        "verdict": "PASS",
        "meaning": "train-only frozen-prior ordered-geometry diagnostic completed; no formal selection",
        "repository": repository,
        "preflight_report": preflight_relative,
        "pilot_report": pilot_relative,
        "source_sha256": current_source,
        "cache_sha256": EXPECTED_CACHE_SHA256,
        "frozen_contract_sha256": EXPECTED_CONTRACT_SHA256,
        "fixed_model_contract": pilot["fixed_model_contract"],
        "geometry_scales": pilot["geometry_scales"],
        "unique_free_variants": unique_compact,
        "unique_advancing_variants": pilot["unique_advancing_variants"],
        "paired_variants": paired_compact,
        "root_cause": pilot["root_cause"],
        "next_stage": pilot["next_stage"],
        "train_only_recommendation": recommendation,
        **boundary,
    }
    assert_finite_json(summary)
    write_json_once(root / "reports/phase3_14b_r25_summary.json", summary)
    report_path = root / "reports/phase3_14b_r25_report.md"
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite {report_path}")
    report_path.write_text(markdown_report(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
