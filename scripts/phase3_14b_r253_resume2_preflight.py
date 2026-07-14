#!/usr/bin/env python3
"""Preflight for r2.5.3 Resume2 finalizer-only recovery."""
from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import argparse

from ccda_phase3.phase314b_r23_diagnostics import require_repository_state
from ccda_phase3.phase314b_r253_gate_separation import (
    EXPECTED_CACHE_SHA256,
    EXPECTED_CONTRACT_SHA256,
    EXPECTED_SUBMODULE_COMMIT,
    source_sha256 as original_source_sha256,
)
from ccda_phase3.phase314b_r253_resume1_finalizer import (
    PILOT_SUMMARY_RELATIVE,
    assert_commit_is_ancestor,
    assert_only_allowed_paths,
    corrected_pilot_view,
    historical_evidence_sha256,
    load_json,
    validate_committed_pilot,
    write_json_once,
)
from ccda_phase3.phase314b_r253_resume2_finalizer import (
    BASE_RESUME1_BLOCKED_COMMIT,
    PHASE,
    RESUME2_PREFLIGHT_RELATIVE,
    RESUME2_PROVENANCE_SCHEMA,
    pythonpath_contract,
    resume1_evidence_sha256,
    resume2_source_sha256,
    validate_no_final_artifacts,
    validate_resume1_blocked_evidence,
    validate_resume2_outputs_absent,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--output", default=RESUME2_PREFLIGHT_RELATIVE)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    output = output.resolve()
    output_relative = output.relative_to(root).as_posix()
    if output_relative != RESUME2_PREFLIGHT_RELATIVE:
        raise RuntimeError("Resume2 preflight output path changed")

    assert_commit_is_ancestor(root, BASE_RESUME1_BLOCKED_COMMIT)
    assert_only_allowed_paths(root, ())
    validate_resume2_outputs_absent(root)
    validate_no_final_artifacts(root)

    repository = require_repository_state(root, require_clean=True)
    if repository["submodule_commit"] != EXPECTED_SUBMODULE_COMMIT:
        raise RuntimeError("submodule commit mismatch")
    if repository["cache_sha256"] != EXPECTED_CACHE_SHA256:
        raise RuntimeError("cache SHA mismatch")
    if repository["frozen_contract_sha256"] != EXPECTED_CONTRACT_SHA256:
        raise RuntimeError("frozen contract SHA mismatch")

    import_contract = pythonpath_contract(root)
    resume1_contract = validate_resume1_blocked_evidence(root)
    resume1_hashes = resume1_evidence_sha256(root)
    historical_hashes = historical_evidence_sha256(root)

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
        "meaning": "Resume2 finalizer-only import-path preflight completed",
        "resume": {
            "enabled": True,
            "generation": 2,
            "schema": RESUME2_PROVENANCE_SCHEMA,
            "resume1_blocked_report_commit": BASE_RESUME1_BLOCKED_COMMIT,
            "restart_from": "finalization",
            "gpu_pilot_rerun": False,
            "training_rerun": False,
            "reverse_rerun": False,
            "pilot_summary_reused": True,
            "pilot_summary_mutated": False,
            "correction": "repository_root_missing_from_parent_pythonpath",
        },
        "repository": repository,
        "output": output_relative,
        "pythonpath_contract": import_contract,
        "resume1_blocked_contract": resume1_contract,
        "resume1_evidence_sha256": resume1_hashes,
        "historical_r253_evidence_sha256": historical_hashes,
        "pilot_contract": pilot_contract,
        "original_r253_source_sha256": original_hash,
        "resume2_source_sha256": resume2_source_sha256(root),
        "classification_preview": classification_preview,
        "boundaries": {
            "validation_targets_used": False,
            "formal_test_read": False,
            "formal_training": False,
            "candidate_selected": False,
            "checkpoint_saved": False,
            "model_weights_saved": False,
            "gpu_pilot_rerun": False,
            "training_rerun": False,
            "reverse_rerun": False,
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
