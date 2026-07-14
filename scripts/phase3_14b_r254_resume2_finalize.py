#!/usr/bin/env python3
"""Finalize r2.5.4 Resume2 robot-proxy attribution."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Dict, Mapping

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ccda_phase3.phase314b_r23_diagnostics import require_repository_state, write_json_once
from ccda_phase3.phase314b_r254_robot_proxy_attribution import (
    DIAGNOSTIC_OBJECTIVE_NAMES,
    EXPECTED_CACHE_SHA256,
    EXPECTED_CONTRACT_SHA256,
    EXPECTED_SUBMODULE_COMMIT,
    PHASE as ORIGINAL_PHASE,
    ROBOT_ATTRIBUTION_SCHEMA,
    classify_robot_proxy_attribution,
)
from ccda_phase3.phase314b_r254_resume2_functional_prior import (
    FINAL_REPORT_PATH,
    FINAL_SUMMARY_PATH,
    RESUME2_PILOT_PATH,
    RESUME2_PREFLIGHT_PATH,
    RESUME2_SCHEMA,
    assert_only_allowed_worktree_paths,
    compact_functional_prior_contract,
    load_json,
    source_sha256,
    validate_resume2_pilot_contract,
)


def load_original_finalizer(root: Path) -> ModuleType:
    path = root / "scripts/phase3_14b_r254_finalize.py"
    spec = importlib.util.spec_from_file_location("phase314b_r254_original_finalizer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load original r2.5.4 finalizer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_text_once(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(content)


def render_report(original_module: ModuleType, summary: Mapping[str, Any]) -> str:
    base = original_module.render_report(summary)
    contract = summary["functional_prior_contract"]
    fresh = summary["fresh_prior_validation"]
    appendix = [
        "## Resume2 functional-prior contract",
        "",
        f"- Resume generation: `{summary['resume_generation']}`",
        f"- Resume1 root cause: `{contract['resume1_root_cause']}`",
        f"- Historical RTX-4080 prior SHA: `{contract['historical_prior_state_sha256']}`",
        f"- RTX-4090 prior SHA: `{contract['current_device_prior_state_sha256']}`",
        f"- RTX-4090 prediction SHA: `{contract['current_device_prior_prediction_sha256']}`",
        f"- Fresh state SHA exact: `{str(fresh['state_sha_exact']).lower()}`",
        f"- Fresh prediction SHA exact: `{str(fresh['prediction_sha_exact']).lower()}`",
        f"- Historical functional fingerprint: `{str(contract['historical_functional_fingerprint']).lower()}`",
        "- Historical cross-device parameter-byte SHA required: `False`",
        "- Same-device exact SHA required: `True`",
        "",
    ]
    return base.rstrip() + "\n\n" + "\n".join(appendix)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--preflight-report", default=RESUME2_PREFLIGHT_PATH)
    parser.add_argument("--pilot-report", default=RESUME2_PILOT_PATH)
    parser.add_argument("--summary-output", default=FINAL_SUMMARY_PATH)
    parser.add_argument("--markdown-output", default=FINAL_REPORT_PATH)
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
    assert_only_allowed_worktree_paths(root, allowed)
    repository = require_repository_state(root, require_clean=False)
    if repository["submodule_commit"] != EXPECTED_SUBMODULE_COMMIT:
        raise RuntimeError("submodule commit mismatch")
    if repository["cache_sha256"] != EXPECTED_CACHE_SHA256:
        raise RuntimeError("cache SHA mismatch")
    if repository["frozen_contract_sha256"] != EXPECTED_CONTRACT_SHA256:
        raise RuntimeError("frozen contract SHA mismatch")

    preflight = load_json(preflight_path)
    pilot = load_json(pilot_path)
    if preflight.get("verdict") != "PASS" or pilot.get("verdict") != "PASS":
        raise RuntimeError("Resume2 preflight or pilot did not pass")
    if preflight.get("source_sha256") != source_sha256(root):
        raise RuntimeError("Resume2 source changed after preflight")
    if pilot.get("resume2_source_sha256") != source_sha256(root):
        raise RuntimeError("Resume2 pilot source hash mismatch")
    if pilot.get("schema") != ROBOT_ATTRIBUTION_SCHEMA:
        raise RuntimeError("original r2.5.4 pilot schema mismatch")
    if pilot.get("resume2_schema") != RESUME2_SCHEMA:
        raise RuntimeError("Resume2 pilot schema mismatch")
    validate_resume2_pilot_contract(pilot)

    variants = pilot.get("variants")
    if not isinstance(variants, Mapping) or set(variants) != set(DIAGNOSTIC_OBJECTIVE_NAMES):
        raise RuntimeError("diagnostic variant matrix changed")
    for name, value in variants.items():
        if not bool(value.get("training_completed")):
            raise RuntimeError(f"training incomplete: {name}")
        if not bool(value.get("r253_reproduction_pass")):
            raise RuntimeError(f"r2.5.3 contract reproduction failed: {name}")
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
        if bool(pilot.get(key)):
            raise RuntimeError(f"forbidden boundary crossed: {key}")

    original = load_original_finalizer(root)
    compact_variants: Dict[str, Any] = {}
    mechanisms_by_variant = classification.get("variant_supported_mechanisms", {})
    for name in DIAGNOSTIC_OBJECTIVE_NAMES:
        value = dict(variants[name])
        value["supported_mechanisms"] = mechanisms_by_variant.get(name, [])
        compact_variants[name] = original.compact_variant(value)

    functional = compact_functional_prior_contract(pilot["functional_prior_contract"])
    summary: Dict[str, Any] = {
        "phase": ORIGINAL_PHASE,
        "resume_generation": 2,
        "resume2_schema": RESUME2_SCHEMA,
        "verdict": "PASS",
        "meaning": "train-only robot-proxy attribution completed under the audited functional-prior contract",
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
        "functional_prior_contract": functional,
        "fresh_prior_validation": pilot["fresh_prior_validation"],
        "schema_audit": pilot["schema_audit"],
        "synthetic_controls": pilot["synthetic_controls"],
        "predictability_baselines": pilot["predictability_baselines"],
        "variants": compact_variants,
        "mechanism_counts": classification.get("mechanism_counts", {}),
        "historical_exact_prior_sha_required_for_cross_device": False,
        "same_device_exact_prior_sha_required": True,
        "robot_proxy_attribution_run": True,
        "reverse_sampling_rerun": False,
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
    write_text_once(markdown_path, render_report(original, summary))
    print(json.dumps({
        "verdict": "PASS",
        "root_cause": summary["root_cause"],
        "supported_mechanisms": summary["supported_mechanisms"],
        "train_only_recommendation": None,
        "selected_configuration": None,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
