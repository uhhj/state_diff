#!/usr/bin/env python3
"""Write-once correction preflight for Stage-C Resume1.

This preflight does not rerun pytest.  It verifies the already completed
30-file/463-test gate, preserves the original blocked evidence, proves that the
only correction to the scientific implementation is the Stage-B summary SHA,
and verifies the new Resume1 output namespace is unused.
"""
from __future__ import annotations

import argparse
import json
import os
import py_compile
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ccda_phase3.phase314b_r255_stagec_cache import (
    EXPECTED_STAGE_B_SUMMARY_SHA256,
    EXPECTED_SUBMODULE_COMMIT,
    atomic_write_once,
    sha256_file,
    stable_json_bytes,
)

PHASE = "Phase3.14b-r2.5.5 Stage C Resume1"
IMPLEMENTATION_COMMIT = "e806f78c02938ebe08fba0be75f6ffc205d00d51"
REMOTE_STAGE_B_HEAD = "83d8aa4bf6895a91c264c8350d95cb5878cebc9b"
CORRECT_STAGE_B_SUMMARY_SHA256 = (
    "af7dec74a81f65f8b351df3f4333063cc25b7723ff65fc25256e40768389447e"
)
WRONG_STAGE_A_SUMMARY_SHA256 = (
    "1306eaf32acda52fcab247ee4caf0517c89e8d7a3cf6313a09f661aafff1ebc5"
)
TEST_GATE_SHA256 = (
    "fb5115e4c368727a474e80230962f0cefa08ddab525806f0413a13f2c011bfcd"
)
BLOCKED_SUMMARY_SHA256 = (
    "b9cd513336e3c9c242c807560e2325e60e5b5f382eecc6b73ae4782d9a61d9a5"
)
TEST_GATE_PATH = "reports/phase3_14b_r255_stagec_test_gate_summary.json"
BLOCKED_PATH = "reports/phase3_14b_r255_stagec_blocked_summary.json"
STAGE_B_SUMMARY_PATH = "reports/phase3_14b_r255_stageb_summary.json"
DEFAULT_OUTPUT = (
    "reports/phase3_14b_r255_stagec_resume1_correction_preflight.json"
)
RESUME1_OUTPUTS = (
    "reports/phase3_14b_r255_stagec_resume1_correction_preflight.json",
    "reports/phase3_14b_r255_stagec_resume1_attribution.json",
    "reports/phase3_14b_r255_stagec_resume1_summary.json",
    "reports/phase3_14b_r255_stagec_resume1_report.md",
    "reports/phase3_14b_r255_stagec_resume1_blocked_summary.json",
    "data/phase3_14_cache_v3_r255_stagec_resume1",
)
ORIGINAL_UNFINISHED_OUTPUTS = (
    "reports/phase3_14b_r255_stagec_attribution.json",
    "reports/phase3_14b_r255_stagec_summary.json",
    "reports/phase3_14b_r255_stagec_report.md",
    "data/phase3_14_cache_v3",
)
ALLOWED_DIFF_PATHS = {
    TEST_GATE_PATH,
    BLOCKED_PATH,
    "ccda_phase3/phase314b_r255_stagec_cache.py",
    "scripts/phase3_14b_r255_stagec_build_and_audit.py",
    "scripts/phase3_14b_r255_stagec_resume1_preflight.py",
    "scripts/phase3_14b_r255_stagec_resume1_run.sh",
}


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=str(root),
        text=True,
    ).strip()


def load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return value


def assert_ancestor(root: Path, ancestor: str) -> None:
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, "HEAD"],
        cwd=str(root),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"required commit is not an ancestor: {ancestor}")


def validate_test_gate(root: Path) -> Dict[str, Any]:
    path = root / TEST_GATE_PATH
    if sha256_file(path) != TEST_GATE_SHA256:
        raise RuntimeError("existing Stage-C test-gate SHA changed")
    report = load_json(path)
    if report.get("verdict") != "PASS":
        raise RuntimeError("existing Stage-C test gate is not PASS")
    if int(report.get("passed_test_count", -1)) != 463:
        raise RuntimeError("existing Stage-C pass count is not 463")
    manifest = report.get("test_manifest_sha256")
    if not isinstance(manifest, Mapping) or len(manifest) != 30:
        raise RuntimeError("existing Stage-C test manifest is not 30 files")
    for relative, expected in manifest.items():
        observed = sha256_file(root / str(relative))
        if observed != str(expected):
            raise RuntimeError(f"test changed after completed gate: {relative}")
    return report


def validate_blocked_evidence(root: Path) -> Dict[str, Any]:
    path = root / BLOCKED_PATH
    if sha256_file(path) != BLOCKED_SUMMARY_SHA256:
        raise RuntimeError("original Stage-C blocked evidence SHA changed")
    report = load_json(path)
    if report.get("verdict") != "BLOCKED":
        raise RuntimeError("original Stage-C blocked evidence is not BLOCKED")
    if report.get("scientific_status") != "BLOCKED":
        raise RuntimeError("blocked scientific status changed")
    if report.get("train_only_recommendation") is not None:
        raise RuntimeError("blocked evidence selected a recommendation")
    if report.get("selected_configuration") is not None:
        raise RuntimeError("blocked evidence selected a configuration")
    return report


def validate_correction_scope(root: Path) -> List[str]:
    changed = [
        value
        for value in git(
            root,
            "diff",
            "--name-only",
            f"{IMPLEMENTATION_COMMIT}..HEAD",
        ).splitlines()
        if value
    ]
    unexpected = sorted(set(changed) - ALLOWED_DIFF_PATHS)
    if unexpected:
        raise RuntimeError(
            "correction history contains unexpected paths: "
            + repr(unexpected)
        )
    required = {
        TEST_GATE_PATH,
        BLOCKED_PATH,
        "ccda_phase3/phase314b_r255_stagec_cache.py",
        "scripts/phase3_14b_r255_stagec_build_and_audit.py",
        "scripts/phase3_14b_r255_stagec_resume1_preflight.py",
        "scripts/phase3_14b_r255_stagec_resume1_run.sh",
    }
    missing = sorted(required - set(changed))
    if missing:
        raise RuntimeError(
            "correction history is missing required paths: " + repr(missing)
        )
    return changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    output = Path(args.output)
    output = (
        output.resolve()
        if output.is_absolute()
        else (root / output).resolve()
    )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite preflight: {output}")

    if git(root, "branch", "--show-current") != "Experiment1":
        raise RuntimeError("Resume1 requires Experiment1")
    assert_ancestor(root, IMPLEMENTATION_COMMIT)
    if git(root, "rev-parse", "origin/Experiment1") != REMOTE_STAGE_B_HEAD:
        raise RuntimeError(
            "origin/Experiment1 changed before Resume1 evidence was completed"
        )
    if git(root, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError(
            "worktree must be clean after blocked-evidence and correction commits"
        )

    submodule = root / "external/deformable-ravens"
    if git(submodule, "rev-parse", "HEAD") != EXPECTED_SUBMODULE_COMMIT:
        raise RuntimeError("DeformableRavens commit changed")
    if git(submodule, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("DeformableRavens worktree is dirty")

    if EXPECTED_STAGE_B_SUMMARY_SHA256 != CORRECT_STAGE_B_SUMMARY_SHA256:
        raise RuntimeError("runtime Stage-B summary binding is still wrong")
    stage_b_summary = root / STAGE_B_SUMMARY_PATH
    if sha256_file(stage_b_summary) != CORRECT_STAGE_B_SUMMARY_SHA256:
        raise RuntimeError("committed Stage-B summary SHA is not the corrected value")

    cache_source = (
        root / "ccda_phase3/phase314b_r255_stagec_cache.py"
    ).read_text(encoding="utf-8")
    if WRONG_STAGE_A_SUMMARY_SHA256 in cache_source:
        raise RuntimeError("wrong Stage-A summary SHA remains in Stage-C source")
    if CORRECT_STAGE_B_SUMMARY_SHA256 not in cache_source:
        raise RuntimeError("correct Stage-B summary SHA is absent from source")

    for relative in RESUME1_OUTPUTS:
        if (root / relative).exists():
            raise FileExistsError(
                f"Resume1 write-once namespace is already occupied: {relative}"
            )
    for relative in ORIGINAL_UNFINISHED_OUTPUTS:
        if (root / relative).exists():
            raise RuntimeError(
                f"original unfinished Stage-C output unexpectedly exists: {relative}"
            )

    for relative in (
        "ccda_phase3/phase314b_r255_stagec_cache.py",
        "scripts/phase3_14b_r255_stagec_build_and_audit.py",
        "scripts/phase3_14b_r255_stagec_resume1_preflight.py",
    ):
        py_compile.compile(str(root / relative), doraise=True)
    subprocess.run(
        ["bash", "-n", str(root / "scripts/phase3_14b_r255_stagec_resume1_run.sh")],
        cwd=str(root),
        check=True,
    )

    test_gate = validate_test_gate(root)
    blocked = validate_blocked_evidence(root)
    changed = validate_correction_scope(root)

    payload = {
        "phase": PHASE,
        "schema": "phase314b_r255_stagec_resume1_correction_preflight_v1",
        "verdict": "PASS",
        "scientific_status": "BLOCKED",
        "root_cause": (
            "phase314b_r255_stagec_resume1_stageb_summary_sha_binding_corrected"
        ),
        "repository": {
            "branch": "Experiment1",
            "head": git(root, "rev-parse", "HEAD"),
            "implementation_commit": IMPLEMENTATION_COMMIT,
            "remote_experiment1": REMOTE_STAGE_B_HEAD,
            "submodule_commit": EXPECTED_SUBMODULE_COMMIT,
            "worktree_clean_before_preflight": True,
        },
        "correction": {
            "wrong_stage_a_summary_sha256": WRONG_STAGE_A_SUMMARY_SHA256,
            "correct_stage_b_summary_sha256":
                CORRECT_STAGE_B_SUMMARY_SHA256,
            "changed_paths_since_implementation": changed,
            "scientific_contract_changed": False,
            "test_gate_rerun": False,
            "test_files_modified": False,
            "thresholds_modified": False,
        },
        "retained_evidence": {
            "test_gate_path": TEST_GATE_PATH,
            "test_gate_sha256": TEST_GATE_SHA256,
            "test_files": 30,
            "tests_passed": 463,
            "blocked_path": BLOCKED_PATH,
            "blocked_sha256": BLOCKED_SUMMARY_SHA256,
            "blocked_verdict": blocked["verdict"],
        },
        "resume1_namespace": {
            "correction_preflight": DEFAULT_OUTPUT,
            "cache_root": "data/phase3_14_cache_v3_r255_stagec_resume1",
            "attribution":
                "reports/phase3_14b_r255_stagec_resume1_attribution.json",
            "summary":
                "reports/phase3_14b_r255_stagec_resume1_summary.json",
            "markdown":
                "reports/phase3_14b_r255_stagec_resume1_report.md",
            "blocked":
                "reports/phase3_14b_r255_stagec_resume1_blocked_summary.json",
        },
        "validation_targets_used": False,
        "formal_test_targets_used": False,
        "diffusion_training": False,
        "reverse_sampling": False,
        "idm_training": False,
        "candidate_execution": False,
        "phase4": False,
        "cps": False,
        "train_only_recommendation": None,
        "selected_configuration": None,
    }
    atomic_write_once(output, stable_json_bytes(payload))
    print(
        json.dumps(
            {
                "verdict": "PASS",
                "test_gate_reused": True,
                "test_gate_rerun": False,
                "output": str(output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
