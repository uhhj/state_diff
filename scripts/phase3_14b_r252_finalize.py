#!/usr/bin/env python3
"""Finalize Phase3.14b-r2.5.2 paired transport attribution."""

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
from ccda_phase3.phase314b_r252_transport_attribution import (
    ATTRIBUTION_SCHEMA,
    BASE_REPORT_COMMIT,
    DIAGNOSTIC_OBJECTIVE_NAMES,
    EXPECTED_CACHE_SHA256,
    EXPECTED_CONTRACT_SHA256,
    EXPECTED_PAIRED_ROWS,
    EXPECTED_SUBMODULE_COMMIT,
    PER_TIMESTEP_GRADIENT_SCHEMA,
    PHASE,
    TRAJECTORY_SCHEMA,
    assert_only_allowed_worktree_paths,
    classify_attribution,
    source_sha256,
)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be object: {path}")
    return value


def assert_finite_json(value: Any, path: str = "root") -> None:
    if value is None or isinstance(value, (bool, str, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite number at {path}")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            assert_finite_json(item, f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            assert_finite_json(item, f"{path}[{index}]")
        return
    raise TypeError(f"unsupported JSON value at {path}: {type(value).__name__}")


def compact_variant(value: Mapping[str, Any]) -> Dict[str, Any]:
    gate = value.get("one_step_gate_decomposition", {})
    attribution = value.get("one_step_attribution", {})
    legacy = attribution.get("legacy_model_audit", {})
    per_timestep = attribution.get("by_timestep", {})
    gradients = value.get("per_timestep_gradient", {})
    trajectory = value.get("trajectory", {})
    endpoint = trajectory.get("endpoint", {})
    compact_timestep: Dict[str, Any] = {}
    for timestep in (10, 25, 50):
        key = str(timestep)
        item = per_timestep.get(key, {})
        final_gradient = gradients.get("final", {}).get(key, {})
        compact_timestep[key] = {
            "own_target_closer_fraction": float(
                item.get("own_target_closer_fraction", 0.0)
            ),
            "model_separation_ratio_p50": float(
                item.get("model_separation_ratio", {}).get("p50", 0.0)
            ),
            "model_delta_cosine_p50": float(
                item.get("model_delta_cosine", {}).get("p50", 0.0)
            ),
            "noisy_separation_ratio_p50": float(
                item.get("noisy_separation_ratio", {}).get("p50", 0.0)
            ),
            "expected_forward_alpha": float(
                item.get("expected_forward_alpha", 0.0)
            ),
            "forward_alpha_max_abs_error": float(
                item.get("forward_alpha_max_abs_error", 0.0)
            ),
            "residual_delta_gain_p50": float(
                item.get("residual_delta_gain", {}).get("p50", 0.0)
            ),
            "residual_delta_cosine_p50": float(
                item.get("residual_delta_cosine", {}).get("p50", 0.0)
            ),
            "required_residual_delta_rms_p50": float(
                item.get("required_residual_delta_rms", {}).get("p50", 0.0)
            ),
            "legacy_gate_pass": bool(item.get("legacy_gate_pass", False)),
            "gradient_ratio_median": float(
                final_gradient.get("tracking", {}).get(
                    "observed_ratio_median", 0.0
                )
            ),
            "gradient_ratio_p95": float(
                final_gradient.get("tracking", {}).get("observed_ratio_p95", 0.0)
            ),
            "gradient_tracking_pass": bool(
                final_gradient.get("tracking", {}).get("pass", False)
            ),
        }
    compact_checkpoints: Dict[str, Any] = {}
    for timestep in (99, 90, 75, 50, 25, 10, 0):
        item = trajectory.get("checkpoints", {}).get(str(timestep), {})
        metrics = item.get("metrics", {})
        branch = item.get("branch_support", {})
        compact_checkpoints[str(timestep)] = {
            "sample_validity_rate": float(metrics.get("sample_validity_rate", 0.0)),
            "valid_query_rate": float(
                metrics.get("query_has_valid_candidate_rate", 0.0)
            ),
            "best_ordered_rmse_mean": float(
                metrics.get("best_ordered_rmse_mean", 0.0)
            ),
            "nearest_inversion_p95": float(
                metrics.get("nearest_inversion_p95", 0.0)
            ),
            "both_branch_support_rate": float(
                branch.get("both_branch_support_rate", 0.0)
            ),
            "two_branch_occupancy_rate": float(
                branch.get("two_branch_occupancy_rate", 0.0)
            ),
            "branch_support_pass": bool(branch.get("pass", False)),
        }
    endpoint_metrics = endpoint.get("metrics", {})
    endpoint_branch = endpoint.get("branch_support", {})
    return {
        "training_completed": bool(value.get("training_completed", False)),
        "family": str(value.get("geometry_objective", {}).get("family", "unknown")),
        "target_gradient_ratio": float(
            value.get("geometry_objective", {}).get("target_gradient_ratio", 0.0)
        ),
        "calibration_pass": bool(gate.get("calibration_pass", False)),
        "global_gradient_tracking_pass": bool(
            gate.get("gradient_tracking_pass", False)
        ),
        "reconstruction_aggregate_gate_pass": bool(
            gate.get("reconstruction_aggregate_gate_pass", False)
        ),
        "reconstruction_all_source_gate_pass": bool(
            gate.get("reconstruction_all_source_gate_pass", False)
        ),
        "condition_effect_pass": bool(gate.get("condition_effect_pass", False)),
        "denoising_result_pass": bool(gate.get("denoising_result_pass", False)),
        "branch_audit_pass": bool(gate.get("branch_audit_pass", False)),
        "composite_one_step_pass": bool(
            gate.get("composite_one_step_pass", False)
        ),
        "legacy_own_target_closer_fraction": float(
            legacy.get("own_target_closer_fraction", 0.0)
        ),
        "legacy_separation_ratio_p50": float(
            legacy.get("separation_ratio", {}).get("p50", 0.0)
        ),
        "legacy_delta_cosine_p50": float(
            legacy.get("branch_delta_cosine", {}).get("p50", 0.0)
        ),
        "pipeline_controls_pass": bool(
            attribution.get("pipeline_controls", {}).get("pass", False)
        ),
        "per_timestep": compact_timestep,
        "per_timestep_gradient_initial_pass": bool(
            gradients.get("initial_all_timesteps_pass", False)
        ),
        "per_timestep_gradient_final_pass": bool(
            gradients.get("final_all_timesteps_pass", False)
        ),
        "trajectory_checkpoints": compact_checkpoints,
        "trajectory_attribution": trajectory.get("attribution", {}),
        "endpoint": {
            "sample_validity_rate": float(
                endpoint_metrics.get("sample_validity_rate", 0.0)
            ),
            "valid_query_rate": float(
                endpoint_metrics.get("query_has_valid_candidate_rate", 0.0)
            ),
            "best_ordered_rmse_mean": float(
                endpoint_metrics.get("best_ordered_rmse_mean", 0.0)
            ),
            "k1_ordered_rmse_mean": float(
                endpoint_metrics.get("k1_ordered_rmse_mean", 0.0)
            ),
            "nearest_inversion_mean": float(
                endpoint_metrics.get("nearest_inversion_mean", 0.0)
            ),
            "nearest_inversion_p95": float(
                endpoint_metrics.get("nearest_inversion_p95", 0.0)
            ),
            "nearest_inversion_max": float(
                endpoint_metrics.get("nearest_inversion_max", 0.0)
            ),
            "both_branch_support_rate": float(
                endpoint_branch.get("both_branch_support_rate", 0.0)
            ),
            "two_branch_occupancy_rate": float(
                endpoint_branch.get("two_branch_occupancy_rate", 0.0)
            ),
            "branch_support_pass": bool(endpoint_branch.get("pass", False)),
        },
        "r251_endpoint_reproduction_pass": bool(
            value.get("r251_endpoint_reproduction", {}).get("pass", False)
        ),
        "endpoint_comparison_to_control": value.get(
            "endpoint_comparison_to_control", {}
        ),
    }


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}g}"


def markdown_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Phase3.14b-r2.5.2 Paired One-Step / Reverse-Trajectory Attribution",
        "",
        "## Verdict",
        "",
        "- Verdict: `PASS` (train-only diagnostic chain completed)",
        f"- Root cause: `{summary['root_cause']}`",
        f"- Supported mechanisms: `{summary['supported_mechanisms']}`",
        "- Train-only recommendation: `None`",
        "- Selected configuration: `None`",
        "",
        "## One-step gate decomposition",
        "",
        "| Variant | Denoising gate | Branch gate | Composite | Own closer | Separation p50 | Cosine p50 | Pipeline |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in DIAGNOSTIC_OBJECTIVE_NAMES:
        value = summary["variants"][name]
        lines.append(
            "| `{}` | {} | {} | {} | {} | {} | {} | {} |".format(
                name,
                str(value["denoising_result_pass"]).lower(),
                str(value["branch_audit_pass"]).lower(),
                str(value["composite_one_step_pass"]).lower(),
                _fmt(value["legacy_own_target_closer_fraction"]),
                _fmt(value["legacy_separation_ratio_p50"]),
                _fmt(value["legacy_delta_cosine_p50"]),
                str(value["pipeline_controls_pass"]).lower(),
            )
        )
    lines.extend(
        [
            "",
            "## Per-timestep branch transport and gradients",
            "",
            "| Variant | t | Own closer | Separation | Cosine | Residual gain | Residual cosine | Grad med/p95 | Gate |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name in DIAGNOSTIC_OBJECTIVE_NAMES:
        for timestep in (10, 25, 50):
            value = summary["variants"][name]["per_timestep"][str(timestep)]
            lines.append(
                "| `{}` | {} | {} | {} | {} | {} | {} | {}/{} | {} |".format(
                    name,
                    timestep,
                    _fmt(value["own_target_closer_fraction"]),
                    _fmt(value["model_separation_ratio_p50"]),
                    _fmt(value["model_delta_cosine_p50"]),
                    _fmt(value["residual_delta_gain_p50"]),
                    _fmt(value["residual_delta_cosine_p50"]),
                    _fmt(value["gradient_ratio_median"]),
                    _fmt(value["gradient_ratio_p95"]),
                    str(value["legacy_gate_pass"]).lower(),
                )
            )
    lines.extend(
        [
            "",
            "## Predicted-x0 reverse trajectory",
            "",
            "| Variant | Scheduler t | Validity | Valid-query | Best K | Inversion p95 | Both branches | Occupancy |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name in DIAGNOSTIC_OBJECTIVE_NAMES:
        for timestep in (99, 90, 75, 50, 25, 10, 0):
            value = summary["variants"][name]["trajectory_checkpoints"][str(timestep)]
            lines.append(
                "| `{}` | {} | {} | {} | {} | {} | {} | {} |".format(
                    name,
                    timestep,
                    _fmt(value["sample_validity_rate"]),
                    _fmt(value["valid_query_rate"]),
                    _fmt(value["best_ordered_rmse_mean"]),
                    _fmt(value["nearest_inversion_p95"]),
                    _fmt(value["both_branch_support_rate"]),
                    _fmt(value["two_branch_occupancy_rate"]),
                )
            )
    lines.extend(
        [
            "",
            "## Endpoint reproduction",
            "",
            "| Variant | Validity | Valid-query | Best K | Inversion mean/p95/max | Both branches | Reproduces r2.5.1 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name in DIAGNOSTIC_OBJECTIVE_NAMES:
        value = summary["variants"][name]
        endpoint = value["endpoint"]
        lines.append(
            "| `{}` | {} | {} | {} | {}/{}/{} | {} | {} |".format(
                name,
                _fmt(endpoint["sample_validity_rate"]),
                _fmt(endpoint["valid_query_rate"]),
                _fmt(endpoint["best_ordered_rmse_mean"]),
                _fmt(endpoint["nearest_inversion_mean"]),
                _fmt(endpoint["nearest_inversion_p95"]),
                _fmt(endpoint["nearest_inversion_max"]),
                _fmt(endpoint["both_branch_support_rate"]),
                str(value["r251_endpoint_reproduction_pass"]).lower(),
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
            "- Checkpoint/model weights saved: `False`",
            "- IDM / candidate execution: `False`",
            "- Phase4 / CPS: `False`",
            "",
            "This audit attributes the one-step/reverse mismatch only. It does not repair ordered geometry.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument(
        "--preflight-report",
        default="reports/phase3_14b_r252_preflight_summary.json",
    )
    parser.add_argument(
        "--pilot-report",
        default="reports/phase3_14b_r252_pilot_summary.json",
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
        raise RuntimeError("frozen contract SHA mismatch")

    preflight = load_json(preflight_path)
    pilot = load_json(pilot_path)
    current_source = source_sha256(root)
    if preflight.get("source_sha256") != current_source:
        raise RuntimeError("source changed after preflight")
    if pilot.get("source_sha256") != current_source:
        raise RuntimeError("pilot source provenance mismatch")
    if pilot.get("phase") != PHASE or pilot.get("verdict") != "PASS":
        raise RuntimeError("pilot did not complete")
    if pilot.get("preflight_report") != preflight_relative:
        raise RuntimeError("pilot preflight provenance mismatch")
    if pilot.get("schemas") != preflight.get("schemas"):
        raise RuntimeError("pilot schema provenance mismatch")
    if pilot.get("schemas", {}).get("attribution") != ATTRIBUTION_SCHEMA:
        raise RuntimeError("attribution schema mismatch")
    if pilot.get("schemas", {}).get("trajectory") != TRAJECTORY_SCHEMA:
        raise RuntimeError("trajectory schema mismatch")
    if pilot.get("schemas", {}).get("per_timestep_gradient") != PER_TIMESTEP_GRADIENT_SCHEMA:
        raise RuntimeError("per-timestep gradient schema mismatch")
    if pilot.get("fixed_contract", {}).get("diagnostic_objective_order") != list(
        DIAGNOSTIC_OBJECTIVE_NAMES
    ):
        raise RuntimeError("diagnostic objective order changed")
    if pilot.get("dataset", {}).get("paired_rows") != list(EXPECTED_PAIRED_ROWS):
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

    variants = pilot.get("variants")
    if not isinstance(variants, Mapping) or set(variants) != set(
        DIAGNOSTIC_OBJECTIVE_NAMES
    ):
        raise RuntimeError("diagnostic variant mapping changed")
    for name in DIAGNOSTIC_OBJECTIVE_NAMES:
        value = variants[name]
        if value.get("geometry_objective", {}).get("name") != name:
            raise RuntimeError(f"variant internal identity mismatch: {name}")
        if not bool(value.get("training_completed")):
            raise RuntimeError(f"diagnostic training did not complete: {name}")
        if value.get("one_step_attribution", {}).get("schema") != ATTRIBUTION_SCHEMA:
            raise RuntimeError(f"one-step attribution schema mismatch: {name}")
        if value.get("per_timestep_gradient", {}).get("schema") != PER_TIMESTEP_GRADIENT_SCHEMA:
            raise RuntimeError(f"per-timestep gradient schema mismatch: {name}")
        if value.get("trajectory", {}).get("schema") != TRAJECTORY_SCHEMA:
            raise RuntimeError(f"trajectory schema mismatch: {name}")
        if not bool(value.get("r251_endpoint_reproduction", {}).get("pass")):
            raise RuntimeError(f"r2.5.1 endpoint did not reproduce: {name}")

    recomputed = classify_attribution({"variants": variants})
    for key in (
        "root_cause",
        "supported_mechanisms",
        "next_stage",
        "train_only_recommendation",
    ):
        if pilot.get(key) != recomputed.get(key):
            raise RuntimeError(f"classifier output mismatch for {key}")
    if pilot.get("train_only_recommendation") is not None:
        raise RuntimeError("attribution stage cannot recommend a configuration")

    compact = {
        name: compact_variant(variants[name])
        for name in DIAGNOSTIC_OBJECTIVE_NAMES
    }
    summary: Dict[str, Any] = {
        "phase": PHASE,
        "verdict": "PASS",
        "meaning": "train-only paired one-step and reverse-trajectory attribution completed",
        "repository": repository,
        "base_report_commit": BASE_REPORT_COMMIT,
        "preflight_report": preflight_relative,
        "pilot_report": pilot_relative,
        "source_sha256": current_source,
        "cache_sha256": EXPECTED_CACHE_SHA256,
        "frozen_contract_sha256": EXPECTED_CONTRACT_SHA256,
        "fixed_contract": pilot["fixed_contract"],
        "geometry_scales": pilot["geometry_scales"],
        "shared_prior": pilot["shared_prior"],
        "variants": compact,
        "root_cause": pilot["root_cause"],
        "supported_mechanisms": pilot["supported_mechanisms"],
        "next_stage": pilot["next_stage"],
        "train_only_recommendation": None,
        **boundary,
    }
    assert_finite_json(summary)
    write_json_once(root / "reports/phase3_14b_r252_summary.json", summary)
    report_path = root / "reports/phase3_14b_r252_report.md"
    if report_path.exists():
        raise RuntimeError(f"refusing to overwrite report: {report_path}")
    report_path.write_text(markdown_report(summary), encoding="utf-8")
    print(
        json.dumps(
            {
                "verdict": "PASS",
                "root_cause": summary["root_cause"],
                "supported_mechanisms": summary["supported_mechanisms"],
                "train_only_recommendation": None,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
