#!/usr/bin/env python3
"""Run fixed Phase 0K same-end tension extension smoke."""
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
from scripts.experiment2.phase0.hidden_hook_common import (
    generate_hidden_latch_tension_action_script,
)
from scripts.experiment2.phase0.observation_common import grouped_ridge_accuracy
from scripts.experiment2.phase0.phase0k_tension_metrics import (
    tension_extension_metrics,
    tension_motion_valid,
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


def _median(rows, getter):
    values = np.asarray([float(getter(row)) for row in rows], dtype=np.float64)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("invalid median input")
    return float(np.median(values))


def _update_pair_json(raw_root, topology_id, group_id, row, tension):
    pair_path = raw_root / topology_id / group_id / "pair.json"
    payload = json.loads(pair_path.read_text(encoding="utf-8"))
    payload["metrics"] = row
    payload["phase0k_tension_audit"] = tension
    write_json(pair_path, payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/experiment2/phase0/hidden_latch_phase0k.json"
    )
    parser.add_argument(
        "--output", default="reports/experiment2/phase0_hidden_hook/phase0k"
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

    if config["topology_id"] != "delayed_z_latch_v1":
        raise ValueError("Phase 0K geometry must be Phase 0I")
    if config["intervention_id"] != "same_end_tension_extension_v1":
        raise ValueError("unexpected Phase 0K intervention")
    if config["seeds"] != [71001, 71002, 71003]:
        raise ValueError("Phase 0K seed list changed")
    forbidden = {"candidates", "action_candidates", "extension_candidates"}
    if forbidden.intersection(config):
        raise ValueError("Phase 0K must not contain searches")

    raw_root = output_root / "raw"
    observation = dict(config["observation"])
    execution = dict(config["execution"])
    targets = dict(config["selection_targets"])
    topology_id = str(config["topology_id"])
    intervention_id = str(config["intervention_id"])
    candidate_id = str(config["candidate_id"])
    candidate = {
        "id": candidate_id,
        "topology_id": topology_id,
        "intervention_id": intervention_id,
        "latch": dict(config["latch"]),
        "action": {
            "stage1_distance": float(config["action"]["main_pull_distance"]),
            "extension_distance": float(
                config["action"]["tension_extension_distance"]
            ),
            "final_distance": float(
                config["action"]["main_pull_distance"]
                + config["action"]["tension_extension_distance"]
            ),
        },
    }
    pair_rows, samples, pair_runs = [], [], []
    mechanism_rows, tension_rows, sensor_rows, alignment_rows = [], [], [], []
    minimum_fraction = float(config["action"]["min_achieved_fraction"])

    for seed_value in config["seeds"]:
        seed = int(seed_value)
        result, row, pair_samples, mechanism = run_one_pair(
            config,
            seed,
            execution,
            observation,
            raw_root,
            action_generator_fn=generate_hidden_latch_tension_action_script,
        )
        free_trace, hidden_trace, free_meta, hidden_meta, actions = result[:5]
        tension = tension_extension_metrics(
            free_trace, hidden_trace, free_meta, hidden_meta, actions
        )
        free_tension_valid = tension_motion_valid(free_meta, minimum_fraction)
        hidden_tension_valid = tension_motion_valid(hidden_meta, minimum_fraction)
        row["candidate_id"] = candidate_id
        row["intervention_id"] = intervention_id
        row["free_tension_motion_valid"] = free_tension_valid
        row["hidden_tension_motion_valid"] = hidden_tension_valid
        row["tension_motion_valid"] = bool(
            free_tension_valid and hidden_tension_valid
        )
        diagnostic_final = float(tension["final"]["mean_progress_gap"])
        official_final = float(row["mean_cable_progress_gap"])
        if abs(diagnostic_final - official_final) > 1e-9:
            raise RuntimeError(
                "Phase 0K diagnostic final progress does not match official progress"
            )
        _update_pair_json(
            raw_root, topology_id, row["group_id"], row, tension
        )
        pair_rows.append(row)
        samples.extend(pair_samples)
        mechanism_rows.append({
            "seed": seed, "group_id": row["group_id"], **mechanism
        })
        tension_rows.append({
            "seed": seed,
            "group_id": row["group_id"],
            "free_tension_motion_valid": free_tension_valid,
            "hidden_tension_motion_valid": hidden_tension_valid,
            **tension,
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
            f"{candidate_id} seed={seed} "
            f"stage1_gap={tension['stage1']['mean_progress_gap']:.6f} "
            f"stage2_gap={tension['stage2']['mean_progress_gap']:.6f} "
            f"final_gap={row['mean_cable_progress_gap']:.6f}",
            flush=True,
        )

    schemas = {tuple(sample["formal_schema"]) for sample in samples}
    if len(schemas) != 1:
        raise RuntimeError("formal sensor schema changed")
    formal_schema = list(next(iter(schemas)))
    l2 = float(observation["ridge_l2"])

    def classifier(key):
        return grouped_ridge_accuracy(samples, key, l2)

    topology = summarize_hidden_hook(
        candidate_id,
        candidate,
        pair_rows,
        targets,
        classifier("vision_raw"),
        classifier("vision_delta"),
        classifier("formal_sensor"),
        classifier("oracle_contact"),
    )
    motion_debug = motion_debug_rows(pair_runs)
    tension_execution_valid = bool(
        all(row["tension_motion_valid"] for row in pair_rows)
    )
    failed_checks = [
        key for key, passed in topology["checks"].items() if not passed
    ]
    if not tension_execution_valid:
        failed_checks.append("tension_motion_execution")
    if not motion_debug["all_cartesian_stages_successful"]:
        failed_checks.append("cartesian_motion_execution")
    failed_checks = sorted(set(failed_checks))
    verdict = (
        "HIDDEN_TENSION_EXTENSION_SMOKE_ELIGIBLE"
        if (
            topology["eligible"]
            and tension_execution_valid
            and motion_debug["all_cartesian_stages_successful"]
        )
        else "HIDDEN_TENSION_EXTENSION_SMOKE_BLOCKED"
    )
    tension_median = {
        "stage1_progress_gap": _median(
            tension_rows, lambda row: row["stage1"]["mean_progress_gap"]
        ),
        "stage2_progress_gap": _median(
            tension_rows, lambda row: row["stage2"]["mean_progress_gap"]
        ),
        "final_progress_gap": _median(
            tension_rows, lambda row: row["final"]["mean_progress_gap"]
        ),
        "stage1_branch_distance": _median(
            tension_rows, lambda row: row["stage1"]["mean_bead_branch_distance"]
        ),
        "stage2_branch_distance": _median(
            tension_rows, lambda row: row["stage2"]["mean_bead_branch_distance"]
        ),
        "stage2_minus_stage1_gap": _median(
            tension_rows, lambda row: row["stage2_minus_stage1_progress_gap"]
        ),
        "final_minus_stage1_gap": _median(
            tension_rows, lambda row: row["final_minus_stage1_progress_gap"]
        ),
    }
    summary = {
        "stage": "Experiment2 Phase 0K",
        "gate": "Stage M0 fixed same-end tension-extension smoke",
        "runtime_main_code_sha": git_sha(REPO_ROOT),
        "runtime_submodule_sha": git_sha(SUBMODULE_ROOT),
        "config": str(config_path.relative_to(REPO_ROOT)),
        "config_hash": canonical_json_sha256(config),
        "baseline_report": str(baseline_path.relative_to(REPO_ROOT)),
        "seeds": [int(value) for value in config["seeds"]],
        "topology_id": topology_id,
        "intervention_id": intervention_id,
        "candidate_id": candidate_id,
        "intervention_policy": "single_fixed_future_intervention_no_action_search",
        "fixed_intervention": dict(config["fixed_intervention"]),
        "formal_sensor_feature_version": "stage_aligned_formal_sensor_v1",
        "official_outcome_gate": "mean_cable_progress_gap >= 0.01 m",
        "tension_diagnostics_are_gates": False,
        "oracle_used_as_formal_feature": False,
        "training_performed": False,
        "geometry_search_performed": False,
        "action_search_performed": False,
        "extension_search_performed": False,
        "phase0i_evidence_preserved": True,
        "phase0j_evidence_preserved": True,
        "tension_motion_execution_valid": tension_execution_valid,
        "topology": topology,
        "targets": targets,
        "tension_median": tension_median,
        "phase0i_baseline": {
            "candidate_id": baseline.get("candidate_id"),
            "median": baseline.get("median"),
            "classifiers": baseline.get("classifiers"),
        },
        "failed_checks": failed_checks,
        "verdict": verdict,
        "interpretation": (
            "One fixed same-grasp collinear 0.08 m + 0.04 m intervention. "
            "Even ELIGIBLE permits only expanded validation of this exact "
            "intervention, not training or Scientific PASS."
        ),
    }
    write_json(
        output_root / "candidate_delayed_z_latch_v1_same_end_tension_v1.json",
        topology,
    )
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
        "intervention_id": intervention_id,
        "metrics_are_additional_gates": False,
        "oracle_is_privileged_audit_only": True,
        "rows": mechanism_rows,
    })
    write_json(output_root / "tension_outcome_audit.json", {
        "metric_version": "same_end_tension_extension_metrics_v1",
        "diagnostic_only": True,
        "changes_official_progress_gate": False,
        "median": tension_median,
        "rows": tension_rows,
    })
    write_json(output_root / "summary.json", summary)
    (output_root / "summary.md").write_text("\n".join([
        "# Phase 0K — Fixed Same-End Tension Extension",
        "",
        f"- Topology: `{topology_id}`",
        f"- Intervention: `{intervention_id}`",
        "- Fixed path: `0.080 m + 0.040 m`, same grasp, same direction",
        f"- Seeds: `{summary['seeds']}`",
        "- Official outcome gate: `mean_cable_progress_gap >= 0.01 m`",
        "- Tension diagnostics are gates: `False`",
        f"- Tension motion valid: `{tension_execution_valid}`",
        f"- All Cartesian stages successful: `{motion_debug['all_cartesian_stages_successful']}`",
        f"- Stage-1 progress gap median: `{tension_median['stage1_progress_gap']:.9f}`",
        f"- Stage-2 progress gap median: `{tension_median['stage2_progress_gap']:.9f}`",
        f"- Final progress gap median: `{tension_median['final_progress_gap']:.9f}`",
        f"- Failed checks: `{failed_checks}`",
        f"- Verdict: `{verdict}`",
        "- Training performed: `False`",
        "- Geometry search performed: `False`",
        "- Action search performed: `False`",
        "- Extension search performed: `False`",
        "- Status: fixed 3-seed Stage M0 smoke, not Scientific PASS.",
    ]) + "\n", encoding="utf-8")
    print(output_root / "summary.json")


if __name__ == "__main__":
    main()
