#!/usr/bin/env python3
"""Finalize Phase3.14b-r2.5.3 gate-separation audit."""

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
from ccda_phase3.phase314b_r253_gate_separation import (
    BASE_REPORT_COMMIT,
    DIAGNOSTIC_OBJECTIVE_NAMES,
    EXPECTED_CACHE_SHA256,
    EXPECTED_CONTRACT_SHA256,
    EXPECTED_PAIRED_ROWS,
    EXPECTED_SUBMODULE_COMMIT,
    GATE_SEPARATION_SCHEMA,
    HISTORICAL_REPRODUCTION_SCHEMA,
    PHASE,
    RECONSTRUCTION_DECOMPOSITION_SCHEMA,
    SYNTHETIC_CONTROL_SCHEMA,
    assert_only_allowed_worktree_paths,
    classify_gate_separation,
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


def _failing_groups(grouped: Mapping[str, Any]) -> list[str]:
    groups = grouped.get("groups", {})
    return [
        str(name)
        for name, value in groups.items()
        if not bool(value.get("gate_pass", False))
    ]


def compact_variant(value: Mapping[str, Any]) -> Dict[str, Any]:
    separation = value.get("gate_separation", {})
    components = separation.get("full_reconstruction_components", {})
    branch = separation.get("branch_transport", {})
    physical = separation.get("one_step_physical", {})
    topology = separation.get("ordered_topology", {})
    group_z = separation.get("state_group_z_metrics", {})
    contracts = separation.get("contracts", {})
    horizons = separation.get("by_horizon", {})
    compact_horizons: Dict[str, Any] = {}
    for horizon in range(4):
        item = horizons.get(str(horizon), {})
        metrics = item.get("metrics", {})
        compact_horizons[str(horizon)] = {
            "full_gate_pass": bool(item.get("full_gate_pass", False)),
            "cable_geometry_gate_pass": bool(
                item.get("cable_geometry_gate_pass", False)
            ),
            "z_mse": float(metrics.get("z_mse", 0.0)),
            "cable_z_mse": float(metrics.get("cable_z_mse", 0.0)),
            "robot_proxy_z_mse": float(
                metrics.get("robot_proxy_z_mse", 0.0)
            ),
            "ordered_rmse_p95": float(
                metrics.get("ordered_rmse", {}).get("p95", 0.0)
            ),
            "segment_relative_error_p95": float(
                metrics.get("segment_relative_error", {}).get("p95", 0.0)
            ),
            "chain_relative_error_p95": float(
                metrics.get("chain_relative_error", {}).get("p95", 0.0)
            ),
        }
    return {
        "training_completed": bool(value.get("training_completed", False)),
        "family": str(value.get("geometry_objective", {}).get("family", "unknown")),
        "target_gradient_ratio": float(
            value.get("geometry_objective", {}).get("target_gradient_ratio", 0.0)
        ),
        "training_contract": value.get("training_contract", {}),
        "historical_reconstruction_pass": bool(
            contracts.get("historical_exact_reconstruction_pass", False)
        ),
        "branch_transport_pass": bool(
            contracts.get("branch_transport_pass", False)
        ),
        "one_step_physical_pass": bool(
            contracts.get("one_step_physical_pass", False)
        ),
        "ordered_topology_pass": bool(
            contracts.get("ordered_topology_pass", False)
        ),
        "ordered_topology": topology,
        "separated_branch_physical_pass": bool(
            contracts.get("separated_branch_physical_pass", False)
        ),
        "historical_composite_pass": bool(
            contracts.get("historical_composite_pass", False)
        ),
        "full_reconstruction_components": components,
        "state_group_z_metrics": group_z,
        "branch_own_target_closer_fraction": float(
            branch.get("own_target_closer_fraction", 0.0)
        ),
        "branch_separation_ratio_p50": float(
            branch.get("separation_ratio", {}).get("p50", 0.0)
        ),
        "branch_delta_cosine_p50": float(
            branch.get("branch_delta_cosine", {}).get("p50", 0.0)
        ),
        "one_step_sample_validity_rate": float(
            physical.get("sample_validity_rate", 0.0)
        ),
        "pair_relative_error": separation.get("pair_relative_error", {}),
        "by_horizon": compact_horizons,
        "failing_source_ids": _failing_groups(separation.get("by_source", {})),
        "failing_timesteps": _failing_groups(separation.get("by_timestep", {})),
        "failing_noise_ids": _failing_groups(separation.get("by_noise", {})),
        "r252_one_step_reproduction_pass": bool(
            value.get("r252_one_step_reproduction", {}).get("pass", False)
        ),
        "immutable_reverse_contract": value.get("immutable_reverse_contract", {}),
    }


def compact_controls(value: Mapping[str, Any]) -> Dict[str, Any]:
    output: Dict[str, Any] = {"contract": value.get("contract", {})}
    controls = value.get("controls", {})
    output["controls"] = {}
    for name, item in controls.items():
        contracts = item.get("contracts", {})
        components = item.get("full_reconstruction_components", {})
        branch = item.get("branch_transport", {})
        output["controls"][name] = {
            "historical_reconstruction_pass": bool(
                contracts.get("historical_exact_reconstruction_pass", False)
            ),
            "branch_transport_pass": bool(
                contracts.get("branch_transport_pass", False)
            ),
            "one_step_physical_pass": bool(
                contracts.get("one_step_physical_pass", False)
            ),
            "ordered_topology_pass": bool(
                contracts.get("ordered_topology_pass", False)
            ),
            "separated_branch_physical_pass": bool(
                contracts.get("separated_branch_physical_pass", False)
            ),
            "cable_geometry_gate_pass": bool(
                components.get("cable_geometry_gate_pass", False)
            ),
            "failed_components": list(components.get("failed_components", [])),
            "own_target_closer_fraction": float(
                branch.get("own_target_closer_fraction", 0.0)
            ),
            "separation_ratio_p50": float(
                branch.get("separation_ratio", {}).get("p50", 0.0)
            ),
            "delta_cosine_p50": float(
                branch.get("branch_delta_cosine", {}).get("p50", 0.0)
            ),
        }
    return output


def fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}g}"


def markdown_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Phase3.14b-r2.5.3 Reconstruction / Branch-Transport Gate Separation",
        "",
        "## Verdict",
        "",
        "- Verdict: `PASS` (train-only diagnostic chain completed)",
        f"- Root cause: `{summary['root_cause']}`",
        f"- Supported mechanisms: `{summary['supported_mechanisms']}`",
        f"- Next: `{summary['next_stage']}`",
        "- Train-only recommendation: `None`",
        "- Selected configuration: `None`",
        "",
        "## Model gate separation",
        "",
        "| Model | Exact reconstruction | Branch transport | One-step physical | Ordered topology | Cable geometry | Cable/robot error fraction | Failing sources |",
        "|---|---:|---:|---:|---:|---:|---|---:|",
    ]
    for name in DIAGNOSTIC_OBJECTIVE_NAMES:
        value = summary["variants"][name]
        components = value["full_reconstruction_components"]
        group = value["state_group_z_metrics"]
        lines.append(
            "| `{}` | {} | {} | {} | {} | {} | {}/{} | {} |".format(
                name,
                str(value["historical_reconstruction_pass"]).lower(),
                str(value["branch_transport_pass"]).lower(),
                str(value["one_step_physical_pass"]).lower(),
                str(value["ordered_topology_pass"]).lower(),
                str(components["cable_geometry_gate_pass"]).lower(),
                fmt(group.get("cable_error_fraction"), 4),
                fmt(group.get("robot_proxy_error_fraction"), 4),
                len(value["failing_source_ids"]),
            )
        )
    lines.extend(
        [
            "",
            "## Reconstruction components",
            "",
            "| Model | z MSE | Ordered p95 | Segment p95 | Chain p95 | Failed components |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for name in DIAGNOSTIC_OBJECTIVE_NAMES:
        component = summary["variants"][name]["full_reconstruction_components"]
        values = component["values"]
        lines.append(
            "| `{}` | {} | {} | {} | {} | `{}` |".format(
                name,
                fmt(values["z_mse"]),
                fmt(values["ordered_rmse_p95"]),
                fmt(values["segment_relative_error_p95"]),
                fmt(values["chain_relative_error_p95"]),
                component["failed_components"],
            )
        )
    lines.extend(
        [
            "",
            "## Per-horizon cable fidelity",
            "",
            "| Model | h0 | h1 | h2 | h3 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name in DIAGNOSTIC_OBJECTIVE_NAMES:
        horizons = summary["variants"][name]["by_horizon"]
        lines.append(
            "| `{}` | {} | {} | {} | {} |".format(
                name,
                *[
                    str(horizons[str(index)]["cable_geometry_gate_pass"]).lower()
                    for index in range(4)
                ],
            )
        )
    lines.extend(
        [
            "",
            "## Synthetic controls",
            "",
            "| Control | Exact reconstruction | Branch transport | Physical | Ordered topology | Cable geometry |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, value in summary["synthetic_controls"]["controls"].items():
        lines.append(
            "| `{}` | {} | {} | {} | {} | {} |".format(
                name,
                str(value["historical_reconstruction_pass"]).lower(),
                str(value["branch_transport_pass"]).lower(),
                str(value["one_step_physical_pass"]).lower(),
                str(value["ordered_topology_pass"]).lower(),
                str(value["cable_geometry_gate_pass"]).lower(),
            )
        )
    lines.extend(
        [
            "",
            "## Immutable reverse evidence",
            "",
            "| Model | Candidate-quality gate | Validity | Valid-query | Both branches | Best-K RMSE |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name in DIAGNOSTIC_OBJECTIVE_NAMES:
        reverse = summary["variants"][name]["immutable_reverse_contract"]
        lines.append(
            "| `{}` | {} | {} | {} | {} | {} |".format(
                name,
                str(reverse["candidate_quality_gate_pass"]).lower(),
                fmt(reverse["sample_validity_rate"]),
                fmt(reverse["valid_query_rate"]),
                fmt(reverse["both_branch_support_rate"]),
                fmt(reverse["best_ordered_rmse_mean"]),
            )
        )
    lines.extend(
        [
            "",
            "## Boundaries",
            "",
            "- Historical reconstruction threshold changed: `False`",
            "- Branch threshold changed: `False`",
            "- Training changed: `False`",
            "- Validation targets used: `False`",
            "- Formal test read: `False`",
            "- Formal training: `False`",
            "- Candidate selected: `False`",
            "- Checkpoint saved: `False`",
            "- IDM / candidate execution: `False`",
            "- Phase4 / CPS: `False`",
            "",
            "A PASS verdict means only that the gate-separation diagnostic chain completed.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument(
        "--preflight-report",
        default="reports/phase3_14b_r253_preflight_summary.json",
    )
    parser.add_argument(
        "--pilot-report",
        default="reports/phase3_14b_r253_pilot_summary.json",
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()
    preflight_path = Path(args.preflight_report)
    pilot_path = Path(args.pilot_report)
    if not preflight_path.is_absolute():
        preflight_path = root / preflight_path
    if not pilot_path.is_absolute():
        pilot_path = root / pilot_path
    preflight_path = preflight_path.resolve()
    pilot_path = pilot_path.resolve()
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
        raise RuntimeError("pilot source hash mismatch")
    if pilot.get("preflight_report") != preflight_relative:
        raise RuntimeError("pilot/preflight provenance mismatch")
    if pilot.get("verdict") != "PASS":
        raise RuntimeError("pilot verdict is not PASS")
    if pilot.get("fixed_contract", {}).get("diagnostic_objective_order") != list(
        DIAGNOSTIC_OBJECTIVE_NAMES
    ):
        raise RuntimeError("diagnostic objective order changed")
    if pilot.get("dataset", {}).get("paired_rows") != list(EXPECTED_PAIRED_ROWS):
        raise RuntimeError("paired-row identity changed")
    schemas = pilot.get("schemas", {})
    if schemas.get("gate_separation") != GATE_SEPARATION_SCHEMA:
        raise RuntimeError("gate-separation schema mismatch")
    if schemas.get("reconstruction_decomposition") != RECONSTRUCTION_DECOMPOSITION_SCHEMA:
        raise RuntimeError("reconstruction-decomposition schema mismatch")
    if schemas.get("synthetic_controls") != SYNTHETIC_CONTROL_SCHEMA:
        raise RuntimeError("synthetic-control schema mismatch")
    if schemas.get("historical_reproduction") != HISTORICAL_REPRODUCTION_SCHEMA:
        raise RuntimeError("historical-reproduction schema mismatch")

    variants = pilot.get("variants", {})
    if set(variants) != set(DIAGNOSTIC_OBJECTIVE_NAMES):
        raise RuntimeError("diagnostic variant matrix changed")
    for name in DIAGNOSTIC_OBJECTIVE_NAMES:
        value = variants[name]
        if not bool(value.get("training_completed")):
            raise RuntimeError(f"training incomplete: {name}")
        if value.get("gate_separation", {}).get("schema") != GATE_SEPARATION_SCHEMA:
            raise RuntimeError(f"gate-separation payload mismatch: {name}")
        if not bool(value.get("r252_one_step_reproduction", {}).get("pass")):
            raise RuntimeError(f"r2.5.2 one-step reproduction failed: {name}")
    if not bool(pilot.get("synthetic_controls", {}).get("contract", {}).get("pass")):
        raise RuntimeError("synthetic gate controls did not pass")

    recomputed = classify_gate_separation(
        {
            "variants": variants,
            "synthetic_controls": pilot["synthetic_controls"],
        }
    )
    for key in (
        "root_cause",
        "supported_mechanisms",
        "next_stage",
        "train_only_recommendation",
    ):
        if pilot.get(key) != recomputed.get(key):
            raise RuntimeError(f"classifier output mismatch for {key}")
    if pilot.get("train_only_recommendation") is not None:
        raise RuntimeError("gate-separation audit cannot recommend a configuration")

    compact = {
        name: compact_variant(variants[name])
        for name in DIAGNOSTIC_OBJECTIVE_NAMES
    }
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
            raise RuntimeError(f"boundary mismatch: {key}")

    summary: Dict[str, Any] = {
        "phase": PHASE,
        "verdict": "PASS",
        "meaning": (
            "train-only exact-reconstruction / branch-transport gate-separation "
            "audit completed"
        ),
        "repository": repository,
        "base_report_commit": BASE_REPORT_COMMIT,
        "preflight_report": preflight_relative,
        "pilot_report": pilot_relative,
        "source_sha256": current_source,
        "cache_sha256": EXPECTED_CACHE_SHA256,
        "frozen_contract_sha256": EXPECTED_CONTRACT_SHA256,
        "fixed_contract": pilot["fixed_contract"],
        "dataset": pilot["dataset"],
        "geometry_scales": pilot["geometry_scales"],
        "shared_prior": pilot["shared_prior"],
        "synthetic_controls": compact_controls(pilot["synthetic_controls"]),
        "variants": compact,
        "root_cause": pilot["root_cause"],
        "supported_mechanisms": pilot["supported_mechanisms"],
        "mechanisms_by_variant": pilot.get("mechanisms_by_variant", {}),
        "next_stage": pilot["next_stage"],
        "train_only_recommendation": None,
        **boundary,
    }
    assert_finite_json(summary)
    write_json_once(root / "reports/phase3_14b_r253_summary.json", summary)
    report_path = root / "reports/phase3_14b_r253_report.md"
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
