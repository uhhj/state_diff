#!/usr/bin/env python3
"""Run the single-topology Phase 0I hidden Z-latch smoke audit."""
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

from scripts.experiment2.phase0.common import (
    canonical_json_sha256,
    compute_pair_metrics,
    phase_slice,
)
from scripts.experiment2.phase0.exact_counterfactual import run_exact_counterfactual_pair
from scripts.experiment2.phase0.hidden_hook_common import (
    configure_hidden_latch_environment,
    generate_hidden_latch_action_script,
)
from scripts.experiment2.phase0.hidden_hook_metrics import hidden_hook_outcome_metrics
from scripts.experiment2.phase0.observation_common import (
    delta_image_feature,
    grouped_ridge_accuracy,
    image_feature,
    load_rgb,
)
from scripts.experiment2.phase0.phase0f_common import oracle_contact_feature
from scripts.experiment2.phase0.phase0i_sensor_features import (
    LOAD_STAGES,
    UNLOAD_STAGES,
    motion_stage_mask,
    stage_aligned_formal_sensor_feature,
)
from scripts.experiment2.phase0.run_hidden_hook_bootstrap import (
    git_sha,
    save_trace,
    summarize_hidden_hook,
    write_json,
)
from scripts.experiment2.phase0.run_hidden_hook_observability import (
    contact_alignment,
    motion_debug_rows,
)


def phase0i_classifier_sample(
    group_id, condition, trace, metadata, observation, config
):
    no_action_rgb = load_rgb(Path(metadata["observations"]["no_action_end"]["path"]))
    pre_main_rgb = load_rgb(Path(metadata["observations"]["pre_main"]["path"]))
    formal, schema = stage_aligned_formal_sensor_feature(
        trace,
        metadata["motion_events"],
        hz=float(config["hz"]),
        trace_stride=int(config["trace_stride"]),
    )
    return {
        "group_id": group_id,
        "condition": condition,
        "label": -1 if condition == "free" else 1,
        "vision_raw": image_feature(
            pre_main_rgb, observation["feature_width"], observation["feature_height"]
        ),
        "vision_delta": delta_image_feature(
            no_action_rgb, pre_main_rgb,
            observation["feature_width"], observation["feature_height"],
        ),
        "formal_sensor": formal,
        "formal_schema": schema,
        "oracle_contact": oracle_contact_feature(
            trace, float(config["hz"]), int(config["trace_stride"])
        ),
    }


def _fraction(trace, phase):
    values = phase_slice(trace, phase)
    return float(np.mean(np.asarray(values["contact_active_beads"]) > 0))


def latch_mechanism_metrics(
    free_trace, hidden_trace, free_metadata, hidden_metadata
):
    free_steps = np.asarray(free_trace["physics_step"])
    hidden_steps = np.asarray(hidden_trace["physics_step"])
    free_load = motion_stage_mask(
        free_steps, free_metadata["motion_events"], LOAD_STAGES
    )
    hidden_load = motion_stage_mask(
        hidden_steps, hidden_metadata["motion_events"], LOAD_STAGES
    )
    free_unload = motion_stage_mask(
        free_steps, free_metadata["motion_events"], UNLOAD_STAGES
    )
    hidden_unload = motion_stage_mask(
        hidden_steps, hidden_metadata["motion_events"], UNLOAD_STAGES
    )
    hidden_post = motion_stage_mask(
        hidden_steps,
        hidden_metadata["motion_events"],
        ["latch_probe_post_release"],
    )
    free_post = motion_stage_mask(
        free_steps,
        free_metadata["motion_events"],
        ["latch_probe_post_release"],
    )
    return {
        "hidden_preload_contact_fraction_privileged": _fraction(
            hidden_trace, "preload"
        ),
        "hidden_main_contact_fraction_privileged": _fraction(
            hidden_trace, "main_pull"
        ),
        "hidden_post_release_contact_fraction_privileged": float(np.mean(
            np.asarray(hidden_trace["contact_active_beads"])[hidden_post] > 0
        )),
        "free_preload_contact_fraction_privileged": _fraction(free_trace, "preload"),
        "free_post_release_contact_fraction_privileged": float(np.mean(
            np.asarray(free_trace["contact_active_beads"])[free_post] > 0
        )),
        "formal_load_window_sample_count": {
            "free": int(np.sum(free_load)), "hidden_hook": int(np.sum(hidden_load))
        },
        "formal_unload_window_sample_count": {
            "free": int(np.sum(free_unload)), "hidden_hook": int(np.sum(hidden_unload))
        },
    }


def _probe_motion_valid(metadata, minimum_fraction):
    events = [
        row for row in metadata.get("motion_events", [])
        if str(row.get("stage", "")).startswith("latch_probe_")
    ]
    required = {
        "latch_probe_lift", "latch_probe_hold", "latch_probe_lower_release",
        "latch_probe_return_hold", "latch_probe_post_release",
    }
    stages = {row["stage"] for row in events}
    return bool(
        required == stages
        and all(row["success"] for row in events)
        and all(float(row["achieved_fraction"]) >= minimum_fraction for row in events)
    )


def run_one_pair(config, seed, execution, observation, raw_root):
    group_id = f"zl_{seed:06d}"
    group_dir = raw_root / config["topology_id"] / group_id
    result = run_exact_counterfactual_pair(
        config,
        seed=seed,
        group_id=group_id,
        execution=execution,
        observation_output_dir=group_dir / "observations",
        observation_config=observation,
        hidden_condition="hidden_hook",
        configure_environment_fn=configure_hidden_latch_environment,
        action_generator_fn=generate_hidden_latch_action_script,
    )
    free_trace, hidden_trace, free_meta, hidden_meta, actions, pair_meta = result
    metrics = compute_pair_metrics(
        free_trace, hidden_trace, free_meta, hidden_meta,
        hz=float(config["hz"]), trace_stride=int(config["trace_stride"]),
    )
    outcome = hidden_hook_outcome_metrics(free_trace, hidden_trace, actions)
    mechanism = latch_mechanism_metrics(
        free_trace, hidden_trace, free_meta, hidden_meta
    )
    minimum_fraction = float(config["action"]["min_achieved_fraction"])
    row = {
        "candidate_id": config["topology_id"],
        "group_id": group_id,
        "seed": int(seed),
        **metrics,
        **outcome,
        **pair_meta,
        **mechanism,
        "free_probe_motion_valid": _probe_motion_valid(free_meta, minimum_fraction),
        "hidden_probe_motion_valid": _probe_motion_valid(hidden_meta, minimum_fraction),
    }
    row["probe_motion_valid"] = bool(
        row["free_probe_motion_valid"] and row["hidden_probe_motion_valid"]
    )
    row["branch_amplification"] = float(
        metrics["main_branch_fde"]
        / max(float(metrics["preload_end_max_abs_xy"]), 1e-9)
    )
    save_trace(group_dir / "free.npz", free_trace)
    save_trace(group_dir / "hidden_hook.npz", hidden_trace)
    write_json(group_dir / "pair.json", {
        "config": config,
        "action_script": actions,
        "free_metadata": free_meta,
        "hidden_metadata": hidden_meta,
        "pair_metadata": pair_meta,
        "metrics": row,
    })
    samples = [
        phase0i_classifier_sample(
            group_id, "free", free_trace, free_meta, observation, config
        ),
        phase0i_classifier_sample(
            group_id, "hidden_hook", hidden_trace, hidden_meta, observation, config
        ),
    ]
    return result, row, samples, mechanism


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/experiment2/phase0/hidden_latch_phase0i.json"
    )
    parser.add_argument(
        "--output", default="reports/experiment2/phase0_hidden_hook/phase0i"
    )
    args = parser.parse_args()
    config_path = (REPO_ROOT / args.config).resolve()
    output_root = (REPO_ROOT / args.output).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    raw_root = output_root / "raw"
    observation = dict(config["observation"])
    execution = dict(config["execution"])
    targets = dict(config["selection_targets"])
    topology_id = str(config["topology_id"])
    candidate = {"id": topology_id, "latch": dict(config["latch"])}
    pair_rows, samples, pair_runs, mechanism_rows = [], [], [], []
    sensor_rows, alignment_rows = [], []

    for seed_value in config["seeds"]:
        seed = int(seed_value)
        result, row, pair_samples, mechanism = run_one_pair(
            config, seed, execution, observation, raw_root
        )
        free_trace, hidden_trace, free_meta, hidden_meta = result[:4]
        pair_rows.append(row)
        samples.extend(pair_samples)
        mechanism_rows.append({"seed": seed, "group_id": row["group_id"], **mechanism})
        pair_runs.append({
            "seed": seed,
            "group_id": row["group_id"],
            "free_metadata": free_meta,
            "hidden_hook_metadata": hidden_meta,
        })
        for condition, trace, metadata, sample in (
            ("free", free_trace, free_meta, pair_samples[0]),
            ("hidden_hook", hidden_trace, hidden_meta, pair_samples[1]),
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
                "seed": seed, "group_id": row["group_id"], "condition": condition,
                **contact_alignment(trace),
            })
        print(
            f"{topology_id} seed={seed} preload={row['preload_end_max_abs_xy']:.6f} "
            f"fde={row['main_branch_fde']:.6f} progress_gap={row['mean_cable_progress_gap']:.6f}",
            flush=True,
        )

    schemas = {tuple(sample["formal_schema"]) for sample in samples}
    if len(schemas) != 1:
        raise RuntimeError("formal sensor schema changed across Phase 0I samples")
    formal_schema = list(next(iter(schemas)))
    l2 = float(observation["ridge_l2"])
    classifier = lambda key: grouped_ridge_accuracy(samples, key, l2)
    topology = summarize_hidden_hook(
        topology_id, candidate, pair_rows, targets,
        classifier("vision_raw"), classifier("vision_delta"),
        classifier("formal_sensor"), classifier("oracle_contact"),
    )
    motion_debug = motion_debug_rows(pair_runs)
    verdict = (
        "HIDDEN_Z_LATCH_SMOKE_ELIGIBLE"
        if topology["eligible"] and motion_debug["all_cartesian_stages_successful"]
        else "HIDDEN_Z_LATCH_SMOKE_BLOCKED"
    )
    failed_checks = [key for key, passed in topology["checks"].items() if not passed]
    summary = {
        "stage": "Experiment2 Phase 0I",
        "gate": "Stage M0 hidden Z-latch smoke",
        "runtime_main_code_sha": git_sha(REPO_ROOT),
        "runtime_submodule_sha": git_sha(SUBMODULE_ROOT),
        "config": str(config_path.relative_to(REPO_ROOT)),
        "config_hash": canonical_json_sha256(config),
        "seeds": [int(value) for value in config["seeds"]],
        "topology_policy": "single_fixed_topology_no_grid_search",
        "formal_sensor_feature_version": "stage_aligned_formal_sensor_v1",
        "oracle_used_as_formal_feature": False,
        "training_performed": False,
        "geometry_search_performed": False,
        "phase0g_evidence_preserved": True,
        "phase0h_evidence_preserved": True,
        "topology": topology,
        "targets": targets,
        "failed_checks": failed_checks,
        "verdict": verdict,
        "interpretation": (
            "Fixed 3-seed Stage M0 smoke only; even ELIGIBLE would permit only "
            "expanded fixed-topology validation, not training or Scientific PASS."
        ),
    }
    write_json(output_root / "candidate_delayed_z_latch_v1.json", topology)
    write_json(output_root / "motion_debug.json", motion_debug)
    write_json(output_root / "sensor_audit.json", {
        "formal_sensor_feature_version": "stage_aligned_formal_sensor_v1",
        "oracle_used_as_formal_feature": False,
        "allowed_inputs": [
            "physics_step", "joint motor torque", "joint reaction wrench",
            "EE constraint reaction", "grasp state", "constraint availability",
            "robot motion-event intervals",
        ],
        "rows": sensor_rows,
    })
    write_json(output_root / "contact_alignment.json", {
        "oracle_is_privileged_audit_only": True, "rows": alignment_rows
    })
    write_json(output_root / "feature_schema.json", {
        "version": "stage_aligned_formal_sensor_v1",
        "dimension": len(formal_schema), "schema": formal_schema,
    })
    write_json(output_root / "mechanism_audit.json", {
        "topology": "delayed_z_latch_v1",
        "metrics_are_additional_gates": False,
        "oracle_is_privileged_audit_only": True,
        "rows": mechanism_rows,
    })
    write_json(output_root / "summary.json", summary)
    (output_root / "summary.md").write_text("\n".join([
        "# Phase 0I — Formal Sensor Reanalysis and Hidden Z-Latch Smoke",
        "",
        f"- Topology: `{topology_id}` (single fixed topology; no grid search)",
        f"- Seeds: `{summary['seeds']}`",
        f"- Formal sensor feature: `stage_aligned_formal_sensor_v1` ({len(formal_schema)}D)",
        f"- Motion stages successful: `{motion_debug['all_cartesian_stages_successful']}`",
        f"- Failed checks: `{failed_checks}`",
        f"- Verdict: `{verdict}`",
        "- Oracle used as formal feature: `False`",
        "- Training performed: `False`",
        "- Geometry search performed: `False`",
        "- Status: fixed 3-seed Stage M0 smoke, not Scientific PASS.",
    ]) + "\n", encoding="utf-8")
    print(output_root / "summary.json")


if __name__ == "__main__":
    main()
