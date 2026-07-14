#!/usr/bin/env python3
"""Finalize the committed r2.5.3 pilot without rerunning training."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, Mapping

from ccda_phase3.phase314b_r23_diagnostics import require_repository_state
from ccda_phase3.phase314b_r253_gate_separation import (
    BASE_REPORT_COMMIT,
    DIAGNOSTIC_OBJECTIVE_NAMES,
    EXPECTED_CACHE_SHA256,
    EXPECTED_CONTRACT_SHA256,
    EXPECTED_PAIRED_ROWS,
    EXPECTED_SUBMODULE_COMMIT,
    GATE_SEPARATION_SCHEMA,
    RECONSTRUCTION_DECOMPOSITION_SCHEMA,
    SYNTHETIC_CONTROL_SCHEMA,
    source_sha256 as original_source_sha256,
)
from ccda_phase3.phase314b_r253_resume1_finalizer import (
    BASE_BLOCKED_COMMIT,
    CORRECTION_SCHEMA,
    FINAL_REPORT_RELATIVE,
    FINAL_SUMMARY_RELATIVE,
    PHASE,
    PILOT_SUMMARY_RELATIVE,
    PILOT_SUMMARY_SHA256,
    RESUME_PREFLIGHT_RELATIVE,
    RESUME_PROVENANCE_SCHEMA,
    assert_only_allowed_paths,
    corrected_pilot_view,
    historical_evidence_sha256,
    load_json,
    resume_source_sha256,
    validate_committed_pilot,
    write_json_once,
)


def load_original_finalizer(root: Path):
    path = root / "scripts/phase3_14b_r253_finalize.py"
    spec = importlib.util.spec_from_file_location("phase3_14b_r253_original_finalize", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load original r2.5.3 finalizer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def assert_boundary(pilot: Mapping[str, Any]) -> Dict[str, Any]:
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
    return boundary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--preflight-report", default=RESUME_PREFLIGHT_RELATIVE)
    parser.add_argument("--pilot-report", default=PILOT_SUMMARY_RELATIVE)
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
    if preflight_relative != RESUME_PREFLIGHT_RELATIVE:
        raise RuntimeError("Resume1 preflight path changed")
    if pilot_relative != PILOT_SUMMARY_RELATIVE:
        raise RuntimeError("committed pilot path changed")

    assert_only_allowed_paths(root, (preflight_relative,))
    repository = require_repository_state(root, require_clean=False)
    if repository["submodule_commit"] != EXPECTED_SUBMODULE_COMMIT:
        raise RuntimeError("submodule commit mismatch")
    if repository["cache_sha256"] != EXPECTED_CACHE_SHA256:
        raise RuntimeError("cache SHA mismatch")
    if repository["frozen_contract_sha256"] != EXPECTED_CONTRACT_SHA256:
        raise RuntimeError("frozen contract SHA mismatch")

    preflight = load_json(preflight_path)
    if preflight.get("verdict") != "PASS":
        raise RuntimeError("Resume1 preflight did not pass")
    resume = preflight.get("resume", {})
    if resume.get("schema") != RESUME_PROVENANCE_SCHEMA:
        raise RuntimeError("Resume1 provenance schema mismatch")
    if resume.get("gpu_pilot_rerun") is not False:
        raise RuntimeError("Resume1 must be finalizer-only")
    if preflight.get("resume1_source_sha256") != resume_source_sha256(root):
        raise RuntimeError("Resume1 source changed after preflight")
    if preflight.get("historical_evidence_sha256") != historical_evidence_sha256(root):
        raise RuntimeError("historical evidence changed after preflight")

    pilot = load_json(pilot_path)
    pilot_contract = validate_committed_pilot(root, pilot)
    if pilot_contract["pilot_summary_sha256"] != PILOT_SUMMARY_SHA256:
        raise RuntimeError("pilot SHA contract mismatch")
    if pilot.get("source_sha256") != original_source_sha256(root):
        raise RuntimeError("pilot/original r2.5.3 source mismatch")
    if pilot.get("fixed_contract", {}).get("diagnostic_objective_order") != list(
        DIAGNOSTIC_OBJECTIVE_NAMES
    ):
        raise RuntimeError("diagnostic objective order changed")
    if pilot.get("dataset", {}).get("paired_rows") != list(EXPECTED_PAIRED_ROWS):
        raise RuntimeError("paired rows changed")
    schemas = pilot.get("schemas", {})
    if schemas.get("gate_separation") != GATE_SEPARATION_SCHEMA:
        raise RuntimeError("gate-separation schema mismatch")
    if schemas.get("reconstruction_decomposition") != RECONSTRUCTION_DECOMPOSITION_SCHEMA:
        raise RuntimeError("reconstruction-decomposition schema mismatch")
    if schemas.get("synthetic_controls") != SYNTHETIC_CONTROL_SCHEMA:
        raise RuntimeError("synthetic-control schema mismatch")

    corrected = corrected_pilot_view(pilot)
    variants = corrected["variants"]
    if set(variants) != set(DIAGNOSTIC_OBJECTIVE_NAMES):
        raise RuntimeError("corrected variant matrix changed")
    for name in DIAGNOSTIC_OBJECTIVE_NAMES:
        value = variants[name]
        if not bool(value.get("training_completed")):
            raise RuntimeError(f"training incomplete: {name}")
        if value.get("gate_separation", {}).get("schema") != GATE_SEPARATION_SCHEMA:
            raise RuntimeError(f"gate-separation payload mismatch: {name}")
        reproduction = value.get("r252_one_step_reproduction", {})
        if reproduction.get("schema") != CORRECTION_SCHEMA:
            raise RuntimeError(f"corrected reproduction schema mismatch: {name}")
        if reproduction.get("pass") is not True:
            raise RuntimeError(f"corrected reproduction failed: {name}")
    if not bool(corrected.get("synthetic_controls", {}).get("contract", {}).get("pass")):
        raise RuntimeError("synthetic controls failed")
    if corrected.get("train_only_recommendation") is not None:
        raise RuntimeError("Resume1 cannot recommend a configuration")

    boundary = assert_boundary(corrected)
    original = load_original_finalizer(root)
    compact = {
        name: original.compact_variant(variants[name])
        for name in DIAGNOSTIC_OBJECTIVE_NAMES
    }
    summary: Dict[str, Any] = {
        "phase": "Phase3.14b-r2.5.3",
        "resume_phase": PHASE,
        "verdict": "PASS",
        "meaning": (
            "train-only exact-reconstruction / branch-transport gate-separation "
            "audit finalized from the committed pilot after correcting the "
            "oracle adapter key"
        ),
        "repository": repository,
        "base_report_commit": BASE_REPORT_COMMIT,
        "blocked_report_commit": BASE_BLOCKED_COMMIT,
        "resume_preflight_report": preflight_relative,
        "pilot_report": pilot_relative,
        "pilot_summary_sha256": PILOT_SUMMARY_SHA256,
        "pilot_source_commit": pilot_contract["pilot_implementation_commit"],
        "original_r253_source_sha256": original_source_sha256(root),
        "resume1_source_sha256": resume_source_sha256(root),
        "cache_sha256": EXPECTED_CACHE_SHA256,
        "frozen_contract_sha256": EXPECTED_CONTRACT_SHA256,
        "fixed_contract": corrected["fixed_contract"],
        "dataset": corrected["dataset"],
        "geometry_scales": corrected["geometry_scales"],
        "shared_prior": corrected["shared_prior"],
        "synthetic_controls": original.compact_controls(corrected["synthetic_controls"]),
        "variants": compact,
        "reproduction_correction": {
            "schema": CORRECTION_SCHEMA,
            "producer_key": "legacy_oracle_audit",
            "faulty_consumer_key": "exact_v_oracle_audit",
            "persisted_proof_path": (
                "one_step_attribution_controls.pipeline_controls."
                "oracle_legacy_gate_pass"
            ),
            "gpu_pilot_rerun": False,
            "pilot_summary_mutated": False,
        },
        "root_cause": corrected["root_cause"],
        "supported_mechanisms": corrected["supported_mechanisms"],
        "mechanisms_by_variant": corrected.get("mechanisms_by_variant", {}),
        "next_stage": corrected["next_stage"],
        "train_only_recommendation": None,
        **boundary,
    }
    original.assert_finite_json(summary)
    write_json_once(root / FINAL_SUMMARY_RELATIVE, summary)
    report_path = root / FINAL_REPORT_RELATIVE
    if report_path.exists():
        raise RuntimeError(f"refusing to overwrite report: {report_path}")
    report = original.markdown_report(summary)
    report += (
        "\n## Resume1 finalizer correction\n\n"
        "- GPU pilot rerun: `False`\n"
        f"- Committed pilot SHA256: `{PILOT_SUMMARY_SHA256}`\n"
        "- Producer key: `legacy_oracle_audit`\n"
        "- Faulty consumer key: `exact_v_oracle_audit`\n"
        "- Persisted oracle proof: "
        "`pipeline_controls.oracle_legacy_gate_pass`\n"
    )
    report_path.write_text(report, encoding="utf-8")
    print(
        json.dumps(
            {
                "verdict": "PASS",
                "root_cause": summary["root_cause"],
                "supported_mechanisms": summary["supported_mechanisms"],
                "train_only_recommendation": None,
                "gpu_pilot_rerun": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
