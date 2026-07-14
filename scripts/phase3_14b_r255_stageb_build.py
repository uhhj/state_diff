#!/usr/bin/env python3
"""Build, independently reproduce, audit, and atomically promote state-v3."""
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ccda_phase3.phase314b_r255_robot_proxy_provenance import (
    EXPECTED_CACHE_SHA256,
    EXPECTED_SUBMODULE_COMMIT,
    git_output,
    sha256_file,
    write_json_once,
    write_text_once,
)
from ccda_phase3.phase314b_r255_stageb_dataset import (
    BASE_EVIDENCE_COMMIT,
    DATASET_FILE,
    EXPECTED_ACTIONS,
    EXPECTED_EPISODES,
    EXPECTED_PAIRS,
    EXPECTED_RECORDS,
    MANIFEST_FILE,
    WINDOW_FILE,
    compare_build_directories,
    load_npz_strict,
)
from ccda_phase3.schema_v3 import STATE_DIM

TEST_GATE_RELATIVE = "reports/phase3_14b_r255_stageb_test_gate_summary.json"
SUMMARY_RELATIVE = "reports/phase3_14b_r255_stageb_summary.json"
REPORT_RELATIVE = "reports/phase3_14b_r255_stageb_report.md"
FINAL_DATA_RELATIVE = "data/phase3_state_v3_slack"


def assert_repository_contract(root: Path, test_gate: Path) -> Dict[str, Any]:
    if git_output(root, "branch", "--show-current") != "Experiment1":
        raise RuntimeError("Stage B requires Experiment1")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", BASE_EVIDENCE_COMMIT, "HEAD"],
        cwd=str(root),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if ancestor.returncode != 0:
        raise RuntimeError("Stage-A evidence commit is not an ancestor")
    submodule = root / "external/deformable-ravens"
    if git_output(submodule, "rev-parse", "HEAD") != EXPECTED_SUBMODULE_COMMIT:
        raise RuntimeError("pinned DeformableRavens commit changed")
    if git_output(submodule, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("DeformableRavens worktree is dirty")
    legacy_cache = root / "data/phase3_14_cache/phase3_14a_training_cache.npz"
    if sha256_file(legacy_cache) != EXPECTED_CACHE_SHA256:
        raise RuntimeError("legacy immutable cache changed")
    if not test_gate.is_file():
        raise RuntimeError("Stage-B test-gate evidence is missing")
    gate = json.loads(test_gate.read_text(encoding="utf-8"))
    if gate.get("verdict") != "PASS":
        raise RuntimeError("Stage-B test gate is not PASS")
    for relative, expected in gate["test_manifest_sha256"].items():
        if sha256_file(root / relative) != expected:
            raise RuntimeError(f"test changed after gate: {relative}")

    allowed = {test_gate.relative_to(root).as_posix()}
    unexpected: List[str] = []
    status = git_output(root, "status", "--porcelain", "--untracked-files=all")
    for line in status.splitlines():
        path = line[3:].strip()
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        if path not in allowed:
            unexpected.append(line)
    if unexpected:
        raise RuntimeError(f"unexpected worktree paths before Stage B: {unexpected}")
    return {
        "branch": "Experiment1",
        "implementation_head": git_output(root, "rev-parse", "HEAD"),
        "base_evidence_commit": BASE_EVIDENCE_COMMIT,
        "submodule_commit": EXPECTED_SUBMODULE_COMMIT,
        "legacy_cache_sha256": EXPECTED_CACHE_SHA256,
        "test_gate_sha256": sha256_file(test_gate),
    }


def run_worker(root: Path, output_root: Path) -> Dict[str, Any]:
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONHASHSEED": "0",
            "PYTHONNOUSERSITE": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    command = [
        sys.executable,
        str(root / "scripts/phase3_14b_r255_stageb_worker.py"),
        "--root",
        str(root),
        "--output-root",
        str(output_root),
    ]
    completed = subprocess.run(
        command,
        cwd=str(root),
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(completed.stdout, end="")
    if completed.returncode != 0:
        raise RuntimeError(f"isolated Stage-B worker failed: {completed.returncode}")
    lines = [line for line in completed.stdout.splitlines() if line.strip().startswith("{")]
    if not lines:
        raise RuntimeError("worker returned no JSON summary")
    value = json.loads(lines[-1])
    if value.get("verdict") != "PASS":
        raise RuntimeError("worker verdict is not PASS")
    return value


def fsync_directory(path: Path) -> None:
    descriptor = os.open(str(path), os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def render_markdown(summary: Dict[str, Any]) -> str:
    artifacts = summary["artifacts"]
    equivalence = summary["legacy_equivalence"]
    return "\n".join(
        [
            "# Phase3.14b-r2.5.5 Stage B State-v3 Materialization",
            "",
            f"- Audit verdict: `{summary['verdict']}`",
            f"- Scientific status: `{summary['scientific_status']}`",
            f"- Root cause: `{summary['root_cause']}`",
            f"- Required next path: `{summary['required_next_path']}`",
            "",
            "## Materialized dataset",
            "",
            f"- Episodes: `{summary['counts']['episodes']}`",
            f"- State records: `{summary['counts']['state_records']}`",
            f"- Action records/windows: `{summary['counts']['action_records']}`",
            f"- State dimension: `{summary['counts']['state_dim']}`",
            f"- Independent builds exact: `{str(summary['determinism']['exact']).lower()}`",
            f"- Dataset SHA256: `{artifacts['dataset_sha256']}`",
            f"- Windows SHA256: `{artifacts['windows_sha256']}`",
            "",
            "## Legacy invariants",
            "",
            f"- Cable history exact: `{equivalence['cable_history_max_abs'] == 0.0}`",
            f"- Cable future exact: `{equivalence['cable_future_max_abs'] == 0.0}`",
            f"- Action history exact: `{equivalence['action_history_max_abs'] == 0.0}`",
            f"- Target action exact: `{equivalence['target_action_max_abs'] == 0.0}`",
            "",
            "## Boundary",
            "",
            "- Legacy raw, windows, cache, reports, and submodule were not modified.",
            "- No state-v3 training cache was built.",
            "- No diffusion, reverse sampling, IDM, candidate execution, Phase4, or CPS was run.",
            "- `robot_proxy_attribution_interpretable=false`.",
            "- `train_only_recommendation=None` and `selected_configuration=None`.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--final-data-root", default=FINAL_DATA_RELATIVE)
    parser.add_argument("--test-gate", default=TEST_GATE_RELATIVE)
    parser.add_argument("--summary", default=SUMMARY_RELATIVE)
    parser.add_argument("--report", default=REPORT_RELATIVE)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    final_data_root = (root / args.final_data_root).resolve()
    test_gate = (root / args.test_gate).resolve()
    summary_path = (root / args.summary).resolve()
    report_path = (root / args.report).resolve()
    for path in (final_data_root, summary_path, report_path):
        if path.exists():
            raise RuntimeError(f"write-once Stage-B output already exists: {path}")
    repository = assert_repository_contract(root, test_gate)

    temporary_parent = Path(
        tempfile.mkdtemp(prefix=".phase3_14b_r255_stageb.", dir=str(root / "data"))
    )
    run_a = temporary_parent / "run_a" / "phase3_state_v3_slack"
    run_b = temporary_parent / "run_b" / "phase3_state_v3_slack"
    promoted = False
    try:
        worker_a = run_worker(root, run_a)
        worker_b = run_worker(root, run_b)
        if worker_a != worker_b:
            raise RuntimeError("isolated worker scalar summaries differ")
        determinism = compare_build_directories(run_a, run_b)
        manifest = json.loads((run_a / MANIFEST_FILE).read_text(encoding="utf-8"))
        final_data_root.parent.mkdir(parents=True, exist_ok=True)
        os.replace(run_a, final_data_root)
        fsync_directory(final_data_root.parent)
        promoted = True
        promotion_verification = compare_build_directories(
            final_data_root,
            run_b,
        )

        dataset_path = final_data_root / DATASET_FILE
        windows_path = final_data_root / WINDOW_FILE
        dataset = load_npz_strict(dataset_path)
        windows = load_npz_strict(windows_path)
        counts = {
            "episodes": int(dataset["split_name"].shape[0]),
            "episode_pairs": int(manifest["episode_pair_count"]),
            "state_records": int(dataset["states"].shape[0]),
            "action_records": int(dataset["actions"].shape[0]),
            "windows": int(windows["paper_x"].shape[0]),
            "state_dim": int(dataset["states"].shape[1]),
        }
        required_counts = {
            "episodes": EXPECTED_EPISODES,
            "episode_pairs": EXPECTED_PAIRS,
            "state_records": EXPECTED_RECORDS,
            "action_records": EXPECTED_ACTIONS,
            "windows": EXPECTED_ACTIONS,
            "state_dim": STATE_DIM,
        }
        for key, expected in required_counts.items():
            if counts[key] != expected:
                raise RuntimeError(
                    f"promoted state-v3 {key}={counts[key]} != {expected}"
                )
        if counts["action_records"] != counts["windows"]:
            raise RuntimeError(f"promoted state-v3 counts invalid: {counts}")
        if sha256_file(dataset_path) != worker_a["dataset_sha256"]:
            raise RuntimeError("promoted dataset SHA differs from worker summary")
        if sha256_file(windows_path) != worker_a["windows_sha256"]:
            raise RuntimeError("promoted windows SHA differs from worker summary")

        summary: Dict[str, Any] = {
            "phase": "Phase3.14b-r2.5.5 Stage B",
            "schema": "phase314b_r255_stageb_materialization_summary_v1",
            "verdict": "PASS",
            "scientific_status": "BLOCKED",
            "root_cause": "phase314b_r255_stageb_state_v3_dataset_materialized_and_byte_reproducible",
            "required_next_path": "BUILD_NEW_STATE_V3_CACHE_AND_RERUN_ROBOT_PROXY_ATTRIBUTION",
            "repository": repository,
            "counts": counts,
            "determinism": determinism,
            "promotion_verification": promotion_verification,
            "artifacts": {
                "data_root": final_data_root.relative_to(root).as_posix(),
                "manifest": (final_data_root / MANIFEST_FILE).relative_to(root).as_posix(),
                "manifest_sha256": sha256_file(final_data_root / MANIFEST_FILE),
                "dataset": dataset_path.relative_to(root).as_posix(),
                "dataset_sha256": sha256_file(dataset_path),
                "windows": windows_path.relative_to(root).as_posix(),
                "windows_sha256": sha256_file(windows_path),
                "action_template": (final_data_root / "action_template.pkl").relative_to(root).as_posix(),
                "action_template_sha256": sha256_file(final_data_root / "action_template.pkl"),
            },
            "legacy_equivalence": manifest["legacy_equivalence"],
            "split_isolation": manifest["split_isolation"],
            "window_pair_audit": manifest["window_pair_audit"],
            "controlled_joint_indices": manifest["controlled_joint_indices"],
            "ee_tip_link": manifest["ee_tip_link"],
            "ee_tip_link_name": manifest["ee_tip_link_name"],
            "legacy_raw_modified": False,
            "legacy_windows_modified": False,
            "legacy_cache_modified": False,
            "submodule_modified": False,
            "new_state_v3_dataset_written": True,
            "new_state_v3_windows_written": True,
            "new_cache_written": False,
            "diffusion_training": False,
            "reverse_sampling": False,
            "checkpoint_saved": False,
            "weights_persisted": False,
            "prediction_tensor_persisted": False,
            "idm": False,
            "candidate_execution": False,
            "phase4": False,
            "cps": False,
            "robot_proxy_attribution_interpretable": False,
            "train_only_recommendation": None,
            "selected_configuration": None,
        }
        write_text_once(report_path, render_markdown(summary))
        write_json_once(summary_path, summary)
        print(
            json.dumps(
                {
                    "verdict": "PASS",
                    "scientific_status": "BLOCKED",
                    "root_cause": summary["root_cause"],
                    "dataset_sha256": summary["artifacts"]["dataset_sha256"],
                    "windows_sha256": summary["artifacts"]["windows_sha256"],
                    "summary": str(summary_path),
                    "report": str(report_path),
                },
                sort_keys=True,
            )
        )
    finally:
        if run_b.exists():
            shutil.rmtree(run_b)
        if not promoted and run_a.exists():
            shutil.rmtree(run_a)
        if temporary_parent.exists():
            shutil.rmtree(temporary_parent)


if __name__ == "__main__":
    main()
