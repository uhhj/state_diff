#!/usr/bin/env python3
"""Verify a promoted Phase3.13-r1 dataset and authorize Phase3.14 data use."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from ccda_phase3.provenance_v2 import (
    ATTESTATION_VERSION,
    resolve_manifest_artifact,
    sha256_file,
    strict_json_dump,
    strict_json_load,
    tree_merkle_root,
    verify_source_lock,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument(
        "--formal-root",
        default="data/phase3_state_v2_slack",
    )
    parser.add_argument(
        "--source-lock",
        default="reports/phase3_13_r1_source_lock.json",
    )
    parser.add_argument(
        "--audit-summary",
        default=(
            "reports/"
            "phase3_13_r1_dataset_audit_summary.json"
        ),
    )
    parser.add_argument(
        "--output",
        default=(
            "reports/"
            "phase3_13_r1_provenance_gate_summary.json"
        ),
    )
    parser.add_argument(
        "--report",
        default=(
            "reports/"
            "phase3_13_r1_provenance_gate_report.md"
        ),
    )
    parser.add_argument(
        "--require-exact-heads",
        action="store_true",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    formal = root / args.formal_root
    lock_path = root / args.source_lock
    lock = strict_json_load(lock_path)
    source_result = verify_source_lock(
        root,
        lock,
        require_exact_heads=bool(args.require_exact_heads),
        require_clean_tracked_worktree=True,
    )

    manifest_path = formal / "manifest.json"
    attestation_path = formal / "provenance_attestation.json"
    manifest = strict_json_load(manifest_path)
    attestation = strict_json_load(attestation_path)
    audit_path = root / args.audit_summary
    audit = strict_json_load(audit_path)

    if (
        attestation.get("attestation_version")
        != ATTESTATION_VERSION
    ):
        raise RuntimeError("unsupported dataset attestation")
    if attestation["source_lock_sha256"] != sha256_file(lock_path):
        raise RuntimeError("source-lock file hash mismatch")
    if attestation["generator_main_commit"] != lock["main_commit"]:
        raise RuntimeError("attestation main commit mismatch")
    if (
        attestation["generator_submodule_commit"]
        != lock["submodule_commit"]
    ):
        raise RuntimeError("attestation submodule mismatch")

    if manifest["main_commit"] != lock["main_commit"]:
        raise RuntimeError("manifest main commit mismatch")
    if manifest["submodule_commit"] != lock["submodule_commit"]:
        raise RuntimeError("manifest submodule commit mismatch")
    if manifest.get("input_canonicalization") is not False:
        raise RuntimeError("input canonicalization flag is not false")
    if (
        manifest.get("contains_simulator_bead_velocity")
        is not False
    ):
        raise RuntimeError("simulator bead velocity is present")

    for relative, expected in manifest[
        "source_code_sha256"
    ].items():
        locked = lock["main_source_sha256"].get(relative)
        if locked is None:
            raise RuntimeError(
                "manifest references unlocked source %s" % relative
            )
        if expected != locked:
            raise RuntimeError(
                "manifest source hash mismatch for %s" % relative
            )

    windows_path = resolve_manifest_artifact(
        manifest_path,
        manifest["window_npz"],
    )
    template_path = resolve_manifest_artifact(
        manifest_path,
        manifest["action_template"],
    )
    if sha256_file(windows_path) != manifest["window_npz_sha256"]:
        raise RuntimeError("windows hash does not match manifest")
    if (
        sha256_file(template_path)
        != manifest["action_template_sha256"]
    ):
        raise RuntimeError("action template hash mismatch")
    if (
        sha256_file(windows_path)
        != attestation["windows_sha256"]
    ):
        raise RuntimeError("windows hash does not match attestation")
    if (
        sha256_file(template_path)
        != attestation["action_template_sha256"]
    ):
        raise RuntimeError(
            "action template hash does not match attestation"
        )
    if (
        sha256_file(manifest_path)
        != attestation["manifest_sha256"]
    ):
        raise RuntimeError("manifest hash does not match attestation")

    raw_merkle, raw_files = tree_merkle_root(
        formal / "raw",
        suffixes=(".pkl", ".json"),
    )
    if raw_merkle != attestation["raw_merkle_root"]:
        raise RuntimeError("raw Merkle root mismatch")
    if len(raw_files) != int(attestation["raw_file_count"]):
        raise RuntimeError("raw file count mismatch")

    if audit.get("verdict") != "PASS":
        raise RuntimeError("Phase3.13-r1 audit is not PASS")
    if (
        audit.get("root_cause")
        != "phase313_state_v2_dataset_supported"
    ):
        raise RuntimeError("unexpected Phase3.13-r1 audit root cause")
    if (
        sha256_file(audit_path)
        != attestation["audit_summary_sha256"]
    ):
        raise RuntimeError("audit summary hash mismatch")

    payload: Dict[str, Any] = {
        "verdict": "PASS",
        "root_cause": (
            "phase313_r1_provenance_locked_dataset_supported"
        ),
        "source_lock": source_result,
        "generator_main_commit": lock["main_commit"],
        "generator_submodule_commit": lock["submodule_commit"],
        "windows_hash_matches_manifest": True,
        "action_template_hash_matches_manifest": True,
        "all_source_hashes_match": True,
        "raw_merkle_root": raw_merkle,
        "raw_file_count": len(raw_files),
        "phase313_audit_pass": True,
        "formal_training_allowed": True,
        "phase4": False,
        "cps": False,
    }
    strict_json_dump(root / args.output, payload)

    lines = [
        "# Phase3.13-r1 Provenance Gate",
        "",
        "- Verdict: `PASS`",
        (
            "- Root cause: "
            "`phase313_r1_provenance_locked_dataset_supported`"
        ),
        f"- Generator main commit: `{lock['main_commit']}`",
        (
            "- Generator submodule commit: "
            f"`{lock['submodule_commit']}`"
        ),
        f"- Raw Merkle root: `{raw_merkle}`",
        "",
        "| Check | Result |",
        "|---|---:|",
        "| Source lock | `True` |",
        "| Current source hashes | `True` |",
        "| Windows hash | `True` |",
        "| Action template hash | `True` |",
        "| Raw Merkle root | `True` |",
        "| Phase3.13 audit | `True` |",
        "",
        (
            "Formal Phase3.14 training may begin. "
            "Phase4 and CPS remain blocked."
        ),
    ]
    (root / args.report).write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
