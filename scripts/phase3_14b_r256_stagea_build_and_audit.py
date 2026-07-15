#!/usr/bin/env python3
"""Run two isolated Stage-A builds, compare, promote, and seal evidence."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ccda_phase3.phase314b_r256_stagea_contract import (
    BASE_EVIDENCE_COMMIT,
    CONTRACT_FILE,
    EXPECTED_SUBMODULE_COMMIT,
    MANIFEST_FILE,
    PHASE,
    TRAIN_VIEW_FILE,
    atomic_write_once,
    compare_directories,
    git_output,
    load_json,
    sha256_file,
    stable_json_bytes,
)

TEST_GATE = "reports/phase3_14b_r256_stagea_test_gate_summary.json"
DEFAULT_OUTPUT_ROOT = "data/phase3_14_cache_v3_r256_stagea"
DEFAULT_SUMMARY = "reports/phase3_14b_r256_stagea_summary.json"
DEFAULT_REPORT = "reports/phase3_14b_r256_stagea_report.md"


def resolve(root: Path, value: str) -> Path:
    candidate = Path(value)
    return (
        candidate.resolve()
        if candidate.is_absolute()
        else (root / candidate).resolve()
    )


def assert_repository(root: Path, test_gate: Path) -> Dict[str, Any]:
    if git_output(root, "branch", "--show-current") != "Experiment1":
        raise RuntimeError("Stage A requires Experiment1")
    completed = subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            BASE_EVIDENCE_COMMIT,
            "HEAD",
        ],
        cwd=str(root),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if completed.returncode != 0:
        raise RuntimeError("Stage C Resume1 evidence is not an ancestor")
    submodule = root / "external/deformable-ravens"
    if git_output(submodule, "rev-parse", "HEAD") != EXPECTED_SUBMODULE_COMMIT:
        raise RuntimeError("DeformableRavens commit changed")
    if git_output(
        submodule,
        "status",
        "--porcelain",
        "--untracked-files=all",
    ):
        raise RuntimeError("DeformableRavens worktree is dirty")

    allowed = {test_gate.relative_to(root).as_posix()}
    unexpected: List[str] = []
    status = git_output(
        root,
        "status",
        "--porcelain",
        "--untracked-files=all",
    )
    for line in status.splitlines():
        if not line.strip():
            continue
        path = line[3:].strip()
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        if path not in allowed:
            unexpected.append(line)
    if unexpected:
        raise RuntimeError(
            "unexpected worktree paths before Stage A: "
            + repr(unexpected)
        )
    return {
        "branch": "Experiment1",
        "head": git_output(root, "rev-parse", "HEAD"),
        "base_evidence_commit": BASE_EVIDENCE_COMMIT,
        "submodule_commit": EXPECTED_SUBMODULE_COMMIT,
    }


def promote_directory(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"promotion destination exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.promote.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    shutil.copytree(source, temporary)
    os.replace(temporary, destination)
    directory_fd = os.open(str(destination.parent), os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def render_report(summary: Dict[str, Any]) -> str:
    audit = summary["idm_identifiability_audit"]
    paired = audit["paired_action_intervention"]
    lines = [
        "# Phase3.14b-r2.5.6 Stage A Cable-only / IDM Contract",
        "",
        f"- Audit verdict: `{summary['verdict']}`",
        f"- Scientific status: `{summary['scientific_status']}`",
        f"- Root cause: `{summary['root_cause']}`",
        f"- Required next path: `{summary['required_next_path']}`",
        "",
        "## Cable-only diffusion contract",
        "",
        "- Condition: `state-v3 history + past action history`",
        "- Condition dimension: `243`",
        "- Target: `ordered cable XY future only`",
        "- Target shape: `[4,48]`",
        "- Robot history retained as condition: `true`",
        "- Robot future in target: `false`",
        "",
        "## Deployable IDM audit",
        "",
        f"- Train full-horizon rows: `{audit['train_full_horizon_rows']}`",
        f"- Paired episode groups: `{audit['paired_episode_groups']}`",
        f"- Best cable feature: `{audit['best_cable_feature']}`",
        f"- Cable incremental gain: `{audit['cable_incremental_gain']:.9g}`",
        f"- Maximum permuted gain: `{audit['maximum_permuted_gain']:.9g}`",
        f"- Gain over permuted: `{audit['gain_over_permuted']:.9g}`",
        f"- Active action dimensions: `{audit['active_action_dimensions']}`",
        f"- Unique action vectors: `{audit['unique_action_vectors_rounded_1e6']}`",
        (
            "- Paired action-diverse fraction: "
            f"`{paired['paired_action_diverse_fraction']:.9g}`"
        ),
        f"- Formal IDM data ready: `{str(audit['formal_idm_data_ready']).lower()}`",
        "",
        "## Boundary",
        "",
        "- Historical state-v2/v3 cache and evidence were not modified.",
        "- Historical full-state diffusion and IDM fields were not reused.",
        "- No diffusion training, reverse sampling, formal IDM training, candidate execution, Phase4, or CPS was run.",
        "- `train_only_recommendation=None` and `selected_configuration=None`.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument(
        "--python-bin",
        default="/miniforge3/envs/coord_bimanual/bin/python",
    )
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--summary", default=DEFAULT_SUMMARY)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output_root = resolve(root, args.output_root)
    summary_path = resolve(root, args.summary)
    report_path = resolve(root, args.report)
    test_gate_path = root / TEST_GATE
    if output_root.exists() or summary_path.exists() or report_path.exists():
        raise FileExistsError("Stage-A write-once namespace is occupied")
    repository = assert_repository(root, test_gate_path)
    test_gate = load_json(test_gate_path)
    if test_gate.get("verdict") != "PASS":
        raise RuntimeError("Stage-A test gate is not PASS")

    temporary_parent = Path(
        tempfile.mkdtemp(prefix="phase314b_r256_stagea_", dir="/tmp")
    )
    worker_a = temporary_parent / "worker_a"
    worker_b = temporary_parent / "worker_b"
    command_base = [
        str(args.python_bin),
        str(root / "scripts/phase3_14b_r256_stagea_worker.py"),
        "--root",
        str(root),
    ]
    subprocess.run(
        command_base + ["--output-root", str(worker_a)],
        cwd=str(root),
        check=True,
    )
    subprocess.run(
        command_base + ["--output-root", str(worker_b)],
        cwd=str(root),
        check=True,
    )
    comparison = compare_directories(worker_a, worker_b)
    if not comparison["exact"]:
        raise RuntimeError(
            "independent Stage-A workers differ: "
            + repr(comparison["different"])
        )
    promote_directory(worker_a, output_root)
    promotion = compare_directories(output_root, worker_b)
    if not promotion["exact"]:
        raise RuntimeError("promoted Stage-A directory differs")

    manifest_path = output_root / MANIFEST_FILE
    manifest = load_json(manifest_path)
    audit = manifest["idm_identifiability_audit"]
    summary = {
        "phase": PHASE,
        "schema": "phase314b_r256_stagea_summary_v1",
        "verdict": "PASS",
        "scientific_status": "BLOCKED",
        "root_cause": audit["root_cause"],
        "required_next_path": audit["required_next_path"],
        "repository": repository,
        "test_gate_path": TEST_GATE,
        "test_gate_sha256": sha256_file(test_gate_path),
        "test_gate": test_gate,
        "output_root": output_root.relative_to(root).as_posix(),
        "contract_sha256": sha256_file(output_root / CONTRACT_FILE),
        "train_view_sha256": sha256_file(output_root / TRAIN_VIEW_FILE),
        "manifest_sha256": sha256_file(manifest_path),
        "independent_workers_exact": comparison["exact"],
        "promotion_exact": promotion["exact"],
        "validation": manifest["validation"],
        "source_contract_audit": manifest["source_contract_audit"],
        "diffusion_contract": manifest["diffusion_contract"],
        "idm_identifiability_audit": audit,
        "cable_only_diffusion_contract_ready": True,
        "formal_idm_data_ready": audit["formal_idm_data_ready"],
        "robot_future_in_diffusion_target": False,
        "robot_future_in_idm_inputs": False,
        "legacy_state_v2_attribution_reinterpreted": False,
        "historical_stagec_root_cause_modified": False,
        "diffusion_training": False,
        "reverse_sampling": False,
        "formal_idm_training": False,
        "candidate_execution": False,
        "phase4": False,
        "cps": False,
        "train_only_recommendation": None,
        "selected_configuration": None,
    }
    atomic_write_once(report_path, render_report(summary).encode("utf-8"))
    atomic_write_once(summary_path, stable_json_bytes(summary))
    shutil.rmtree(temporary_parent, ignore_errors=True)
    print(
        json.dumps(
            {
                "verdict": "PASS",
                "scientific_status": "BLOCKED",
                "root_cause": summary["root_cause"],
                "required_next_path": summary["required_next_path"],
                "summary": str(summary_path),
                "report": str(report_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
