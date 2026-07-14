#!/usr/bin/env python3
"""Preflight for Phase3.14b-r2.5.4 robot-proxy attribution audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, Mapping, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np

from ccda_phase3.phase314b_r22_contract import load_self_hashed_json
from ccda_phase3.phase314b_r23_diagnostics import (
    load_verified_inputs,
    require_repository_state,
    select_balanced_paired_rows,
    write_json_once,
)
from ccda_phase3.phase314b_r251_gradient_calibration import (
    assert_canonical_paired_row_contract,
)
from ccda_phase3.phase314b_r254_robot_proxy_attribution import (
    BASE_REPORT_COMMIT,
    DEPENDENCY_PATHS,
    DIAGNOSTIC_OBJECTIVE_NAMES,
    EXPECTED_CACHE_SHA256,
    EXPECTED_CONTRACT_SHA256,
    EXPECTED_PAIRED_ROWS,
    EXPECTED_R253_MECHANISMS,
    EXPECTED_R253_PILOT_SHA256,
    EXPECTED_R253_RECOMMENDATION,
    EXPECTED_R253_ROOT_CAUSE,
    EXPECTED_SHARED_PAIRED_PRIOR_SHA256,
    EXPECTED_SUBMODULE_COMMIT,
    PHASE,
    ROBOT_ACTION_SCHEMA,
    ROBOT_ATTRIBUTION_SCHEMA,
    ROBOT_BASELINE_SCHEMA,
    ROBOT_HYBRID_SCHEMA,
    ROBOT_MODEL_ERROR_SCHEMA,
    ROBOT_PROXY_SCHEMA,
    RobotProxyAuditSpec,
    assert_only_allowed_worktree_paths,
    dependency_sha256,
    git_output,
    robot_proxy_groups,
    sha256_file,
    source_sha256,
)
from ccda_phase3.schema_v2 import N_BEADS, ROBOT_PROXY_DIM, STATE_DIM


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be object: {path}")
    return value


def historical_paths(root: Path) -> Sequence[str]:
    names = git_output(
        root, "ls-tree", "-r", "--name-only", BASE_REPORT_COMMIT
    ).splitlines()
    selected = []
    for name in names:
        if (
            "phase314b_r253" in name
            or "phase3_14b_r253" in name
            or name == "GOALS.md"
        ):
            selected.append(name)
    if not selected:
        raise RuntimeError("no r2.5.3 historical paths found in base commit")
    return tuple(sorted(selected))


def assert_historical_paths_unchanged(root: Path, paths: Sequence[str]) -> None:
    for path in paths:
        try:
            git_output(root, "diff", "--quiet", BASE_REPORT_COMMIT, "HEAD", "--", path)
        except Exception as exc:
            raise RuntimeError(f"historical r2.5.3 path changed: {path}") from exc


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{name} must be a mapping")
    return value


def historical_r253_contract(
    summary: Mapping[str, Any],
    pilot: Mapping[str, Any],
) -> Dict[str, Any]:
    if summary.get("verdict") != "PASS":
        raise RuntimeError("r2.5.3 final verdict is not PASS")
    if summary.get("root_cause") != EXPECTED_R253_ROOT_CAUSE:
        raise RuntimeError("r2.5.3 root cause changed")
    mechanisms = summary.get("supported_mechanisms")
    if not isinstance(mechanisms, list) or set(mechanisms) != set(EXPECTED_R253_MECHANISMS):
        raise RuntimeError("r2.5.3 supported mechanisms changed")
    if summary.get("train_only_recommendation") is not EXPECTED_R253_RECOMMENDATION:
        raise RuntimeError("r2.5.3 recommendation changed")
    if summary.get("selected_configuration") is not None:
        raise RuntimeError("r2.5.3 selected a configuration")

    variants = _mapping(pilot.get("variants"), name="r2.5.3 pilot variants")
    if set(variants) != set(DIAGNOSTIC_OBJECTIVE_NAMES):
        raise RuntimeError("r2.5.3 diagnostic matrix changed")
    expected: Dict[str, Any] = {}
    for name in DIAGNOSTIC_OBJECTIVE_NAMES:
        value = _mapping(variants[name], name=f"r2.5.3 variant {name}")
        separation = _mapping(value.get("gate_separation"), name=f"gate separation {name}")
        contracts = _mapping(separation.get("contracts"), name=f"contracts {name}")
        group_z = _mapping(separation.get("state_group_z_metrics"), name=f"group z {name}")
        components = _mapping(
            separation.get("full_reconstruction_components"),
            name=f"reconstruction components {name}",
        )
        expected[name] = {
            "contracts": dict(contracts),
            "state_group_z_metrics": dict(group_z),
            "full_reconstruction_components": dict(components),
            "training_completed": bool(value.get("training_completed")),
        }
        if not expected[name]["training_completed"]:
            raise RuntimeError(f"historical r2.5.3 training incomplete: {name}")
        if not bool(contracts.get("branch_transport_pass")):
            raise RuntimeError(f"historical cable branch contract failed: {name}")
        if not bool(contracts.get("ordered_topology_pass")):
            raise RuntimeError(f"historical topology contract failed: {name}")

    shared_prior = _mapping(pilot.get("shared_prior"), name="r2.5.3 shared prior")
    prior_sha = shared_prior.get("prior_state_sha256")
    if prior_sha != EXPECTED_SHARED_PAIRED_PRIOR_SHA256:
        raise RuntimeError("r2.5.3 paired-prior SHA changed")
    return {
        "root_cause": summary["root_cause"],
        "supported_mechanisms": list(mechanisms),
        "next_stage": summary.get("next_stage"),
        "train_only_recommendation": None,
        "selected_configuration": None,
        "shared_prior_sha256": prior_sha,
        "variants": expected,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument(
        "--output", default="reports/phase3_14b_r254_preflight_summary.json"
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    output = output.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite preflight: {output}")

    repository = require_repository_state(root, require_clean=False)
    assert_only_allowed_worktree_paths(root, ())
    if git_output(root, "branch", "--show-current") != "Experiment1":
        raise RuntimeError("r2.5.4 requires Experiment1 branch")
    try:
        git_output(root, "merge-base", "--is-ancestor", BASE_REPORT_COMMIT, "HEAD")
    except Exception as exc:
        raise RuntimeError("r2.5.3 report commit is not an ancestor") from exc
    if repository["submodule_commit"] != EXPECTED_SUBMODULE_COMMIT:
        raise RuntimeError("submodule commit mismatch")
    if repository["cache_sha256"] != EXPECTED_CACHE_SHA256:
        raise RuntimeError("cache SHA mismatch")
    if repository["frozen_contract_sha256"] != EXPECTED_CONTRACT_SHA256:
        raise RuntimeError("frozen contract SHA mismatch")

    immutable_paths = historical_paths(root)
    assert_historical_paths_unchanged(root, immutable_paths)
    pilot_path = root / "reports/phase3_14b_r253_pilot_summary.json"
    if sha256_file(pilot_path) != EXPECTED_R253_PILOT_SHA256:
        raise RuntimeError("r2.5.3 pilot SHA mismatch")
    summary = load_json(root / "reports/phase3_14b_r253_summary.json")
    pilot = load_json(pilot_path)
    historical = historical_r253_contract(summary, pilot)

    frozen = load_self_hashed_json(root / "reports/phase3_14b_r22_frozen_contract.json")
    if frozen.get("artifact_sha256") != EXPECTED_CONTRACT_SHA256:
        raise RuntimeError("frozen contract payload changed")

    arrays, manifest, x_raw, _, train, fit, _, validation = load_verified_inputs(root)
    split = np.asarray(arrays["split_name"]).astype(str)
    if np.any(split[train] != "train"):
        raise RuntimeError("train indices contain non-train rows")
    paired_rows = select_balanced_paired_rows(arrays, train, 16)
    paired_contract = assert_canonical_paired_row_contract(arrays, paired_rows)
    if paired_contract.get("observed_rows") != list(EXPECTED_PAIRED_ROWS):
        raise RuntimeError("paired-row identity changed")

    if STATE_DIM != 87 or ROBOT_PROXY_DIM != 39 or N_BEADS != 24:
        raise RuntimeError("state-v2 dimensions changed")
    future = np.asarray(arrays["y_state"])
    active = np.asarray(arrays["future_active"], dtype=bool)
    if future.ndim != 3 or future.shape[-1] != STATE_DIM:
        raise RuntimeError("future state shape changed")
    if active.shape != future.shape[1:]:
        raise RuntimeError("future active-mask shape changed")
    if np.asarray(x_raw).shape[1] % STATE_DIM != 0:
        raise RuntimeError("history state width is not divisible by state dimension")
    history_steps = int(np.asarray(x_raw).shape[1] // STATE_DIM)
    if history_steps < 2:
        raise RuntimeError("robot baseline audit requires two history states")

    report: Dict[str, Any] = {
        "phase": PHASE,
        "verdict": "PASS",
        "meaning": "train-only robot-proxy schema/predictability/action attribution preflight",
        "repository": repository,
        "base_report_commit": BASE_REPORT_COMMIT,
        "source_sha256": source_sha256(root),
        "dependency_sha256": dependency_sha256(root),
        "historical_paths_unchanged": list(immutable_paths),
        "r253_contract": historical,
        "diagnostic_objective_order": list(DIAGNOSTIC_OBJECTIVE_NAMES),
        "schemas": {
            "robot_proxy": ROBOT_PROXY_SCHEMA,
            "baselines": ROBOT_BASELINE_SCHEMA,
            "model_error": ROBOT_MODEL_ERROR_SCHEMA,
            "action_sensitivity": ROBOT_ACTION_SCHEMA,
            "hybrid": ROBOT_HYBRID_SCHEMA,
            "attribution": ROBOT_ATTRIBUTION_SCHEMA,
        },
        "spec": RobotProxyAuditSpec().__dict__,
        "dataset": {
            "manifest_schema": manifest.get("schema_version"),
            "train_row_count": int(len(train)),
            "fit_row_count": int(len(fit)),
            "validation_rows_verified_but_not_used": int(len(validation)),
            "history_steps": history_steps,
            "future_steps": int(future.shape[1]),
            "state_dim": int(STATE_DIM),
            "cable_dim": int(N_BEADS * 2),
            "robot_proxy_dim": int(ROBOT_PROXY_DIM),
            "robot_layout": robot_proxy_groups(),
            "paired_rows": paired_rows.tolist(),
            "paired_row_contract": paired_contract,
        },
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
    write_json_once(output, report)
    print(json.dumps({"verdict": "PASS", "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
