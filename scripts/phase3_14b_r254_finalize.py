#!/usr/bin/env python3
"""Finalize Phase3.14b-r2.5.4 robot-proxy attribution audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, Mapping

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ccda_phase3.phase314b_r23_diagnostics import (
    require_repository_state,
    write_json_once,
)
from ccda_phase3.phase314b_r254_robot_proxy_attribution import (
    DIAGNOSTIC_OBJECTIVE_NAMES,
    EXPECTED_CACHE_SHA256,
    EXPECTED_CONTRACT_SHA256,
    EXPECTED_SUBMODULE_COMMIT,
    PHASE,
    ROBOT_ATTRIBUTION_SCHEMA,
    assert_only_allowed_worktree_paths,
    classify_robot_proxy_attribution,
    source_sha256,
)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be object: {path}")
    return value


def write_text_once(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(content)


def compact_variant(value: Mapping[str, Any]) -> Dict[str, Any]:
    fixed = value.get("fixed_cable_contract", {})
    robot = value.get("robot_error", {})
    source = value.get("source_mean_robot_error", {})
    hybrid = value.get("hybrid_counterfactual", {})
    action = value.get("action_sensitivity", {})
    mechanisms = value.get("supported_mechanisms", [])
    return {
        "training_completed": bool(value.get("training_completed")),
        "r253_reproduction_pass": bool(value.get("r253_reproduction_pass")),
        "fixed_cable_contract": dict(fixed),
        "robot_bank_z_mse": float(robot.get("z_mse", float("inf"))),
        "robot_bank_nmse": float(robot.get("nmse", float("inf"))),
        "robot_source_mean_z_mse": float(source.get("z_mse", float("inf"))),
        "robot_source_mean_nmse": float(source.get("nmse", float("inf"))),
        "robot_group_metrics": robot.get("by_group", {}),
        "quaternion": robot.get("quaternion", {}),
        "hybrid_counterfactual": hybrid,
        "action_sensitivity": action,
        "supported_mechanisms": list(mechanisms),
    }


def render_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Phase3.14b-r2.5.4 Robot-Proxy Fidelity Attribution",
        "",
        "## Verdict",
        "",
        f"- Verdict: `{summary['verdict']}` (train-only diagnostic chain completed)",
        f"- Root cause: `{summary['root_cause']}`",
        f"- Supported mechanisms: `{summary['supported_mechanisms']}`",
        f"- Next: `{summary['next_stage']}`",
        "- Train-only recommendation: `None`",
        "- Selected configuration: `None`",
        "",
        "## Robot-proxy schema",
        "",
        f"- Layout: `{summary['schema_audit']['layout']}`",
        f"- Structural-padding dimensions: `{summary['schema_audit']['structural_padding_dimensions']}`",
        f"- Intermittent-zero dimensions: `{summary['schema_audit']['intermittent_zero_dimensions']}`",
        f"- Representation warnings: `{summary['schema_audit']['representation_warnings']}`",
        "",
        "## Predictability baselines",
        "",
        "| Baseline | Robot z MSE | Robot NMSE |",
        "|---|---:|---:|",
    ]
    for name, value in summary["predictability_baselines"]["baselines"].items():
        lines.append(f"| `{name}` | {value['z_mse']:.7g} | {value['nmse']:.7g} |")
    lines.extend([
        "",
        f"- Best deployable baseline: `{summary['predictability_baselines']['best_deployable_baseline']}`",
        f"- Action-conditioned gain: `{summary['predictability_baselines']['action_conditioned_gain']:.7g}`",
        "",
        "## Model attribution",
        "",
        "| Model | Cable branch | Physical | Topology | Robot z MSE | Robot NMSE | Oracle-robot fixes z gate | Action robot value |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for name in DIAGNOSTIC_OBJECTIVE_NAMES:
        value = summary["variants"][name]
        fixed = value["fixed_cable_contract"]
        hybrid = value["hybrid_counterfactual"]
        action = value["action_sensitivity"]
        lines.append(
            f"| `{name}` | {str(fixed['branch_transport_pass']).lower()} | "
            f"{str(fixed['one_step_physical_pass']).lower()} | "
            f"{str(fixed['ordered_topology_pass']).lower()} | "
            f"{value['robot_source_mean_z_mse']:.7g} | "
            f"{value['robot_source_mean_nmse']:.7g} | "
            f"{str(hybrid['robot_proxy_drives_full_z_failure']).lower()} | "
            f"{str(action['robot_proxy_material_for_action_probe']).lower()} |"
        )
    lines.extend([
        "",
        "## Boundaries",
        "",
        "- Cable branch contract changed: `False`",
        "- Ordered-topology contract changed: `False`",
        "- Reverse sampling rerun: `False`",
        "- Validation targets used: `False`",
        "- Formal test read: `False`",
        "- Formal training: `False`",
        "- Candidate selected: `False`",
        "- Checkpoint saved: `False`",
        "- IDM / candidate execution: `False`",
        "- Phase4 / CPS: `False`",
        "",
        "A PASS verdict means only that the train-only attribution chain completed.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument(
        "--preflight-report", default="reports/phase3_14b_r254_preflight_summary.json"
    )
    parser.add_argument(
        "--pilot-report", default="reports/phase3_14b_r254_pilot_summary.json"
    )
    parser.add_argument(
        "--summary-output", default="reports/phase3_14b_r254_summary.json"
    )
    parser.add_argument(
        "--markdown-output", default="reports/phase3_14b_r254_report.md"
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()

    def resolve(value: str) -> Path:
        path = Path(value)
        return (path if path.is_absolute() else root / path).resolve()

    preflight_path = resolve(args.preflight_report)
    pilot_path = resolve(args.pilot_report)
    summary_path = resolve(args.summary_output)
    markdown_path = resolve(args.markdown_output)
    if summary_path.exists() or markdown_path.exists():
        raise RuntimeError("refusing to overwrite final r2.5.4 artifacts")

    allowed = (
        preflight_path.relative_to(root).as_posix(),
        pilot_path.relative_to(root).as_posix(),
    )
    repository = require_repository_state(root, require_clean=False)
    assert_only_allowed_worktree_paths(root, allowed)
    if repository["submodule_commit"] != EXPECTED_SUBMODULE_COMMIT:
        raise RuntimeError("submodule commit mismatch")
    if repository["cache_sha256"] != EXPECTED_CACHE_SHA256:
        raise RuntimeError("cache SHA mismatch")
    if repository["frozen_contract_sha256"] != EXPECTED_CONTRACT_SHA256:
        raise RuntimeError("frozen contract SHA mismatch")

    preflight = load_json(preflight_path)
    pilot = load_json(pilot_path)
    if preflight.get("verdict") != "PASS" or pilot.get("verdict") != "PASS":
        raise RuntimeError("preflight or pilot did not pass")
    if preflight.get("source_sha256") != source_sha256(root):
        raise RuntimeError("source changed after preflight")
    if pilot.get("source_sha256") != source_sha256(root):
        raise RuntimeError("pilot source hash mismatch")
    if pilot.get("schema") != ROBOT_ATTRIBUTION_SCHEMA:
        raise RuntimeError("pilot schema mismatch")
    variants = pilot.get("variants")
    if not isinstance(variants, Mapping) or set(variants) != set(DIAGNOSTIC_OBJECTIVE_NAMES):
        raise RuntimeError("diagnostic matrix changed")
    for name, value in variants.items():
        if not bool(value.get("training_completed")):
            raise RuntimeError(f"training incomplete: {name}")
        if not bool(value.get("r253_reproduction_pass")):
            raise RuntimeError(f"r2.5.3 reproduction failed: {name}")
        fixed = value.get("fixed_cable_contract", {})
        if not bool(fixed.get("branch_transport_pass")):
            raise RuntimeError(f"cable branch contract failed: {name}")
        if not bool(fixed.get("ordered_topology_pass")):
            raise RuntimeError(f"ordered topology contract failed: {name}")
    if not bool(pilot.get("synthetic_controls", {}).get("pass")):
        raise RuntimeError("robot-proxy synthetic controls failed")

    classification = classify_robot_proxy_attribution(pilot)
    for key in ("root_cause", "supported_mechanisms", "next_stage"):
        if pilot.get(key) != classification.get(key):
            raise RuntimeError(f"classifier mismatch: {key}")
    if classification.get("train_only_recommendation") is not None:
        raise RuntimeError("r2.5.4 must not recommend a configuration")
    if classification.get("selected_configuration") is not None:
        raise RuntimeError("r2.5.4 must not select a configuration")
    for key in (
        "validation_targets_used", "formal_test_read", "formal_training",
        "candidate_eligible", "checkpoint_saved", "idm",
        "candidate_execution", "phase4", "cps",
    ):
        if bool(pilot.get(key)):
            raise RuntimeError(f"forbidden boundary crossed: {key}")

    compact_variants: Dict[str, Any] = {}
    mechanisms_by_variant = classification.get("variant_supported_mechanisms", {})
    for name in DIAGNOSTIC_OBJECTIVE_NAMES:
        value = dict(variants[name])
        value["supported_mechanisms"] = mechanisms_by_variant.get(name, [])
        compact_variants[name] = compact_variant(value)

    summary: Dict[str, Any] = {
        "phase": PHASE,
        "verdict": "PASS",
        "meaning": "train-only robot-proxy attribution diagnostic chain completed",
        "root_cause": classification["root_cause"],
        "supported_mechanisms": classification["supported_mechanisms"],
        "next_stage": classification["next_stage"],
        "train_only_recommendation": None,
        "selected_configuration": None,
        "repository": repository,
        "preflight_report": preflight_path.relative_to(root).as_posix(),
        "pilot_report": pilot_path.relative_to(root).as_posix(),
        "source_sha256": source_sha256(root),
        "fixed_contract": pilot["fixed_contract"],
        "dataset": pilot["dataset"],
        "shared_prior": pilot["shared_prior"],
        "schema_audit": pilot["schema_audit"],
        "synthetic_controls": pilot["synthetic_controls"],
        "predictability_baselines": pilot["predictability_baselines"],
        "variants": compact_variants,
        "mechanism_counts": classification.get("mechanism_counts", {}),
        "validation_targets_used": False,
        "formal_test_read": False,
        "formal_training": False,
        "candidate_eligible": False,
        "checkpoint_saved": False,
        "idm": False,
        "candidate_execution": False,
        "phase4": False,
        "cps": False,
    }
    write_json_once(summary_path, summary)
    write_text_once(markdown_path, render_report(summary))
    print(json.dumps({
        "verdict": "PASS",
        "root_cause": summary["root_cause"],
        "supported_mechanisms": summary["supported_mechanisms"],
        "train_only_recommendation": None,
        "selected_configuration": None,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
