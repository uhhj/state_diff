#!/usr/bin/env python3
"""Preflight for r2.5.3 Resume1 finalizer-only correction."""
from __future__ import annotations

import argparse
from pathlib import Path

from ccda_phase3.phase314b_r23_diagnostics import require_repository_state
from ccda_phase3.phase314b_r253_gate_separation import (
    EXPECTED_CACHE_SHA256,
    EXPECTED_CONTRACT_SHA256,
    EXPECTED_SUBMODULE_COMMIT,
    source_sha256 as original_source_sha256,
)
from ccda_phase3.phase314b_r253_resume1_finalizer import (
    BASE_BLOCKED_COMMIT,
    FINAL_REPORT_RELATIVE,
    FINAL_SUMMARY_RELATIVE,
    PHASE,
    PILOT_SUMMARY_RELATIVE,
    RESUME_PREFLIGHT_RELATIVE,
    RESUME_PROVENANCE_SCHEMA,
    assert_commit_is_ancestor,
    assert_only_allowed_paths,
    corrected_pilot_view,
    historical_evidence_sha256,
    load_json,
    resume_source_sha256,
    validate_committed_pilot,
    write_json_once,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--output", default=RESUME_PREFLIGHT_RELATIVE)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    output = output.resolve()
    output_relative = output.relative_to(root).as_posix()
    if output_relative != RESUME_PREFLIGHT_RELATIVE:
        raise RuntimeError("Resume1 preflight output path changed")

    assert_commit_is_ancestor(root, BASE_BLOCKED_COMMIT)
    assert_only_allowed_paths(root, ())
    repository = require_repository_state(root, require_clean=True)
    if repository["submodule_commit"] != EXPECTED_SUBMODULE_COMMIT:
        raise RuntimeError("submodule commit mismatch")
    if repository["cache_sha256"] != EXPECTED_CACHE_SHA256:
        raise RuntimeError("cache SHA mismatch")
    if repository["frozen_contract_sha256"] != EXPECTED_CONTRACT_SHA256:
        raise RuntimeError("frozen contract SHA mismatch")
    if (root / FINAL_SUMMARY_RELATIVE).exists() or (root / FINAL_REPORT_RELATIVE).exists():
        raise RuntimeError("r2.5.3 final artifacts already exist")

    evidence_sha = historical_evidence_sha256(root)
    pilot = load_json(root / PILOT_SUMMARY_RELATIVE)
    pilot_contract = validate_committed_pilot(root, pilot)
    original_hash = original_source_sha256(root)
    if pilot.get("source_sha256") != original_hash:
        raise RuntimeError("committed pilot/original r2.5.3 source hash mismatch")

    corrected = corrected_pilot_view(pilot)
    classification_preview = {
        "root_cause": corrected["root_cause"],
        "supported_mechanisms": corrected["supported_mechanisms"],
        "next_stage": corrected["next_stage"],
        "train_only_recommendation": corrected["train_only_recommendation"],
    }
    payload = {
        "phase": PHASE,
        "verdict": "PASS",
        "meaning": "finalizer-only correction preflight completed",
        "resume": {
            "enabled": True,
            "generation": 1,
            "schema": RESUME_PROVENANCE_SCHEMA,
            "blocked_report_commit": BASE_BLOCKED_COMMIT,
            "restart_from": "finalization",
            "gpu_pilot_rerun": False,
            "training_rerun": False,
            "pilot_summary_reused": True,
            "pilot_summary_mutated": False,
            "correction": "legacy_oracle_audit_consumer_key_mismatch",
        },
        "repository": repository,
        "output": output_relative,
        "pilot_contract": pilot_contract,
        "historical_evidence_sha256": evidence_sha,
        "original_r253_source_sha256": original_hash,
        "resume1_source_sha256": resume_source_sha256(root),
        "correction_contract": {
            "producer_key": "legacy_oracle_audit",
            "faulty_consumer_key": "exact_v_oracle_audit",
            "persisted_proof_path": (
                "one_step_attribution_controls.pipeline_controls."
                "oracle_legacy_gate_pass"
            ),
            "missing_key_defaults_to_false_forbidden": True,
            "original_pilot_write_forbidden": True,
        },
        "classification_preview": classification_preview,
        "boundaries": {
            "validation_targets_used": False,
            "formal_test_read": False,
            "formal_training": False,
            "candidate_selected": False,
            "checkpoint_saved": False,
            "model_weights_saved": False,
            "gpu_pilot_rerun": False,
            "idm": False,
            "candidate_execution": False,
            "phase4": False,
            "cps": False,
        },
    }
    write_json_once(output, payload)
    print(f"PASS: {output_relative}")


if __name__ == "__main__":
    main()
