#!/usr/bin/env python3
"""Run the single fixed wide-stop Z-latch Phase 0J smoke audit."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
SUBMODULE_ROOT = REPO_ROOT / "external" / "deformable-ravens"
for path in (REPO_ROOT, SUBMODULE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.experiment2.phase0.common import canonical_json_sha256
from scripts.experiment2.phase0.observation_common import grouped_ridge_accuracy
from scripts.experiment2.phase0.phase0j_outcome_metrics import (
    hidden_latch_outcome_decomposition,
)
from scripts.experiment2.phase0.run_hidden_hook_bootstrap import (
    git_sha,
    summarize_hidden_hook,
    write_json,
)
from scripts.experiment2.phase0.run_hidden_hook_observability import (
    contact_alignment,
    motion_debug_rows,
)
from scripts.experiment2.phase0.run_hidden_latch_phase0i import run_one_pair


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/experiment2/phase0/hidden_latch_phase0j.json"
    )
    parser.add_argument(
        "--output", default="reports/experiment2/phase0_hidden_hook/phase0j"
    )
    parser.add_argument(
        "--baseline",
        default=(
            "reports/experiment2/phase0_hidden_hook/phase0i/"
            "candidate_delayed_z_latch_v1.json"
        ),
    )
    args = parser.parse_args()
    config_path = (REPO_ROOT / args.config).resolve()
    output_root = (REPO_ROOT / args.output).resolve()
    baseline_path = (REPO_ROOT / args.baseline).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    raw_root = output_root / "raw"
    observation = dict(config["observation"])
    execution = dict(config["execution"])
    targets = dict(config["selection_targets"])
    topology_id = str(config["topology_id"])
    fixed_change = dict(config["fixed_change"])
    candidate = {"id": topology_id, "latch": dict(config["latch"])}
    pair_rows, samples, pair_runs, mechanism_rows = [], [], [], []
    sensor_rows, alignment_rows, outcome_rows = [], [], []

    for seed_value in config["seeds"]:
        seed = int(seed_value)
        result, row, pair_samples, mechanism = run_one_pair(
            config, seed, execution, observation, raw_root
        )
        free_trace, hidden_trace, free_meta, hidden_meta, actions = result[:5]
        outcome = hidden_latch_outcome_decomposition(
            free_trace, hidden_trace, actions
        )
        # Diagnostic only: never rewrite row["mean_cable_progress_gap"].
        outcome_rows.append({
            "seed": seed,
            "group_id": row["group_id"],
            **outcome,
        })
        pair_rows.append(row)
        samples.extend(pair_samples)
        mechanism_rows.append({
            "seed": seed, "group_id": row["group_id"], **mechanism
        })
        pair_runs.append({
            "seed": seed,
            "group_id": row["group_id"],
            "free_metadata": free_meta,
            "hidden_hook_metadata": hidden_meta,
        })
        for condition, trace, sample in (
            ("free", free_trace, pair_samples[0]),
            ("hidden_hook", hidden_trace, pair_samples[1]),
        ):
            sensor_rows.append({
                "seed": seed,
                "group_id": row["group_id"],
                "condition": condition,
                "feature_norm": float(np.linalg.norm(sample["formal_sensor"])),
                "feature_min": float(np.min(sample["formal_sensor"])),
                "feature_max": float(np.max(sample["formal_sensor"])),
                "feature_finite": bool(np.all(np.isfinite(sample["formal_sensor"]))),
            })
            alignment_rows.append({
                "seed": seed,
                "group_id": row["group_id"],
                "condition": condition,
                **contact_alignment(trace),
            })
        print(
            f"{topology_id} seed={seed} "
            f"preload={row['preload_end_max_abs_xy']:.6f} "
            f"fde={row['main_branch_fde']:.6f} "
            f"progress_gap={row['mean_cable_progress_gap']:.6f}",
            flush=True,
        )

    schemas = {tuple(sample["formal_schema"]) for sample in samples}
    if len(schemas) != 1:
        raise RuntimeError("formal sensor schema changed across Phase 0J samples")
    formal_schema = list(next(iter(schemas)))
    l2 = float(observation["ridge_l2"])

    def classifier(key):
        return grouped_ridge_accuracy(samples, key, l2)

    topology = summarize_hidden_hook(
        topology_id,
        candidate,
        pair_rows,
        targets,
        classifier("vision_raw"),
        classifier("vision_delta"),
        classifier("formal_sensor"),
        classifier("oracle_contact"),
    )
    motion_debug = motion_debug_rows(pair_runs)
    verdict = (
        "HIDDEN_WIDE_STOP_SMOKE_ELIGIBLE"
        if (
            topology["eligible"]
            and motion_debug["all_cartesian_stages_successful"]
        )
        else "HIDDEN_WIDE_STOP_SMOKE_BLOCKED"
    )
    failed_checks = [
        key for key, passed in topology["checks"].items() if not passed
    ]
    summary = {
        "stage": "Experiment2 Phase 0J",
        "gate": "Stage M0 fixed wide-stop Z-latch smoke",
        "runtime_main_code_sha": git_sha(REPO_ROOT),
        "runtime_submodule_sha": git_sha(SUBMODULE_ROOT),
        "config": str(config_path.relative_to(REPO_ROOT)),
        "config_hash": canonical_json_sha256(config),
        "seeds": [int(value) for value in config["seeds"]],
        "topology_policy": "single_fixed_topology_no_grid_search",
        "fixed_change": fixed_change,
        "official_outcome_gate": "mean_cable_progress_gap >= 0.01 m",
        "outcome_diagnostics_are_gates": False,
        "oracle_used_as_formal_feature": False,
        "training_performed": False,
        "geometry_search_performed": False,
        "action_search_performed": False,
        "phase0i_evidence_preserved": True,
        "phase0i_baseline": {
            "candidate_id": baseline.get("candidate_id"),
            "median": baseline.get("median"),
            "classifiers": baseline.get("classifiers"),
        },
        "topology": topology,
        "targets": targets,
        "failed_checks": failed_checks,
        "verdict": verdict,
        "interpretation": (
            "Fixed 3-seed wide-stop smoke only; even ELIGIBLE permits only "
            "expanded validation of this fixed topology, not training or "
            "Scientific PASS."
        ),
    }
    write_json(output_root / "candidate_wide_stop_z_latch_v2.json", topology)
    write_json(output_root / "motion_debug.json", motion_debug)
    write_json(output_root / "sensor_audit.json", {
        "formal_sensor_feature_version": "stage_aligned_formal_sensor_v1",
        "oracle_used_as_formal_feature": False,
        "rows": sensor_rows,
    })
    write_json(output_root / "contact_alignment.json", {
        "oracle_is_privileged_audit_only": True,
        "rows": alignment_rows,
    })
    write_json(output_root / "feature_schema.json", {
        "version": "stage_aligned_formal_sensor_v1",
        "dimension": len(formal_schema),
        "schema": formal_schema,
    })
    write_json(output_root / "mechanism_audit.json", {
        "topology": topology_id,
        "metrics_are_additional_gates": False,
        "oracle_is_privileged_audit_only": True,
        "rows": mechanism_rows,
    })
    write_json(output_root / "outcome_audit.json", {
        "metric_version": "hidden_latch_outcome_decomposition_v1",
        "diagnostic_only": True,
        "changes_official_progress_gate": False,
        "rows": outcome_rows,
    })
    write_json(output_root / "summary.json", summary)
    (output_root / "summary.md").write_text("\n".join([
        "# Phase 0J — Outcome Localization and Fixed Wide-Stop Z-Latch Smoke",
        "",
        f"- Topology: `{topology_id}` (single fixed topology; no grid search)",
        "- Only physical change: `latch.wall_width 0.032 m -> 0.080 m`",
        f"- Seeds: `{summary['seeds']}`",
        "- Official outcome gate: `mean_cable_progress_gap >= 0.01 m`",
        "- Outcome decomposition is diagnostic only: `True`",
        f"- Motion stages successful: `{motion_debug['all_cartesian_stages_successful']}`",
        f"- Failed checks: `{failed_checks}`",
        f"- Verdict: `{verdict}`",
        "- Training performed: `False`",
        "- Geometry search performed: `False`",
        "- Action search performed: `False`",
        "- Status: fixed 3-seed Stage M0 smoke, not Scientific PASS.",
    ]) + "\n", encoding="utf-8")
    print(output_root / "summary.json")


if __name__ == "__main__":
    main()
