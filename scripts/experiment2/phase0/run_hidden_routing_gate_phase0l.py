#!/usr/bin/env python3
"""Run the fixed outcome-aligned hidden routing-gate smoke."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
SUBMODULE_ROOT = (
    REPO_ROOT / "external" / "deformable-ravens"
)
for path in (REPO_ROOT, SUBMODULE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ravens.tasks.ccda_hidden_routing_gate_geometry import (
    HiddenRoutingGateGeometryError,
)

from scripts.experiment2.phase0.common import (
    canonical_json_sha256,
    compute_pair_metrics,
    phase_slice,
)
from scripts.experiment2.phase0.exact_counterfactual import (
    run_exact_counterfactual_pair,
)
from scripts.experiment2.phase0.hidden_hook_metrics import (
    hidden_hook_outcome_metrics,
)
from scripts.experiment2.phase0.hidden_routing_gate_common import (
    configure_hidden_routing_gate_environment,
    generate_hidden_routing_gate_action_script,
)
from scripts.experiment2.phase0.hidden_routing_gate_metrics import (
    hidden_routing_gate_outcome_metrics,
    summarize_hidden_routing_gate,
)
from scripts.experiment2.phase0.observation_common import (
    grouped_ridge_accuracy,
)
from scripts.experiment2.phase0.phase0k_tension_metrics import (
    tension_motion_valid,
)
from scripts.experiment2.phase0.run_hidden_hook_bootstrap import (
    git_sha,
    save_trace,
    write_json,
)
from scripts.experiment2.phase0.run_hidden_hook_observability import (
    contact_alignment,
)
from scripts.experiment2.phase0.run_hidden_latch_phase0i import (
    phase0i_classifier_sample,
)


def _fraction(trace, phase):
    block = phase_slice(trace, phase)
    return float(np.mean(
        np.asarray(
            block["contact_active_beads"]
        ) > 0
    ))


def _probe_motion_valid(
    metadata,
    minimum_fraction,
):
    events = [
        row
        for row in metadata.get(
            "motion_events",
            [],
        )
        if str(
            row.get("stage", "")
        ).startswith("latch_probe_")
    ]
    required = {
        "latch_probe_lift",
        "latch_probe_hold",
        "latch_probe_lower_release",
        "latch_probe_return_hold",
        "latch_probe_post_release",
    }
    stages = {
        row["stage"]
        for row in events
    }
    return bool(
        stages == required
        and all(
            bool(row["success"])
            for row in events
        )
        and all(
            float(
                row["achieved_fraction"]
            ) >= minimum_fraction
            for row in events
        )
    )


def _public_layout_match(
    action_script,
    free_metadata,
    hidden_metadata,
):
    main = [
        action
        for action in action_script
        if action["phase"] == "main_pull"
    ]
    if len(main) != 1:
        raise ValueError(
            "expected one main action"
        )
    action_public = main[0][
        "public_task_layout"
    ]
    free_public = free_metadata[
        "privileged_state"
    ]["public_task"]
    hidden_public = hidden_metadata[
        "privileged_state"
    ]["public_task"]
    hashes = {
        canonical_json_sha256(
            action_public
        ),
        canonical_json_sha256(
            free_public
        ),
        canonical_json_sha256(
            hidden_public
        ),
    }
    if len(hashes) != 1:
        raise RuntimeError(
            "action/task public routing layouts "
            "do not match"
        )
    return next(iter(hashes))


def _motion_debug(pair_runs):
    rows: List[Dict[str, Any]] = []
    for pair in pair_runs:
        for condition, metadata_key in (
            ("free", "free_metadata"),
            ("hidden_hook", "hidden_metadata"),
        ):
            for event in pair[
                metadata_key
            ]["motion_events"]:
                rows.append({
                    "seed": int(pair["seed"]),
                    "group_id": (
                        pair["group_id"]
                    ),
                    "condition": condition,
                    **event,
                })
    failures = [
        row
        for row in rows
        if not bool(row.get("success"))
    ]
    recovered = [
        row
        for row in rows
        if row.get("timeout_reason") == (
            "joint_timeout_recovered_by_"
            "cartesian_endpoint"
        )
    ]
    return {
        "event_count": len(rows),
        "all_cartesian_stages_successful": (
            not failures
        ),
        "failure_count": len(failures),
        "recovered_joint_timeout_count": (
            len(recovered)
        ),
        "failures": failures,
        "rows": rows,
    }


def _routing_mechanism_metrics(
    free_trace,
    hidden_trace,
):
    return {
        "free_preload_contact_fraction_privileged": (
            _fraction(free_trace, "preload")
        ),
        "hidden_preload_contact_fraction_privileged": (
            _fraction(hidden_trace, "preload")
        ),
        "free_main_contact_fraction_privileged": (
            _fraction(free_trace, "main_pull")
        ),
        "hidden_main_contact_fraction_privileged": (
            _fraction(hidden_trace, "main_pull")
        ),
    }


def _geometry_audit_from_metadata(metadata):
    privileged = metadata["privileged_state"]
    layout = privileged["routing_gate_layout_privileged"]
    audit = layout.get("geometry_audit")
    if not isinstance(audit, dict):
        raise RuntimeError("routing geometry audit missing")
    selected = [row for row in audit["candidates"] if row["accepted"]]
    if not selected:
        raise RuntimeError(
            "successful reset has no accepted geometry candidate"
        )
    return audit


def _write_geometry_failure(
    output_root,
    config,
    config_path,
    seed,
    error,
    stage_name,
    blocked_verdict,
):
    diagnostics = error.diagnostics
    failure = {
        "stage": stage_name,
        "failure_class": "engineering_blocker",
        "failed_check": "legal_workspace_geometry",
        "first_requested_seed": int(seed),
        "exception_type": type(error).__name__,
        "exception_message": str(error),
        "completed_pair_count": 0,
        "training_performed": False,
        "geometry_search_performed": False,
        "action_search_performed": False,
        "outcome_search_performed": False,
        "diagnostics": diagnostics,
    }
    write_json(output_root / "geometry_failure_diagnostics.json", failure)
    summary = {
        "stage": stage_name,
        "runtime_main_code_sha": git_sha(REPO_ROOT),
        "runtime_submodule_sha": git_sha(SUBMODULE_ROOT),
        "config": str(config_path.relative_to(REPO_ROOT)),
        "config_hash": canonical_json_sha256(config),
        "candidate_id": config["candidate_id"],
        "topology_id": config["topology_id"],
        "official_outcome": config["official_outcome"],
        "engineering_blocked_before_pairs": True,
        "completed_pair_count": 0,
        "failed_checks": ["legal_workspace_geometry"],
        "failure": failure,
        "training_performed": False,
        "geometry_search_performed": False,
        "action_search_performed": False,
        "outcome_search_performed": False,
        "verdict": blocked_verdict,
    }
    write_json(output_root / "summary.json", summary)
    return summary


def run_one_pair(
    config,
    seed,
    execution,
    observation,
    raw_root,
):
    candidate_id = str(
        config["candidate_id"]
    )
    group_id = f"rg_{seed:06d}"
    group_dir = (
        raw_root
        / candidate_id
        / group_id
    )
    result = run_exact_counterfactual_pair(
        config,
        seed=seed,
        group_id=group_id,
        execution=execution,
        observation_output_dir=(
            group_dir / "observations"
        ),
        observation_config=observation,
        hidden_condition="hidden_hook",
        configure_environment_fn=(
            configure_hidden_routing_gate_environment
        ),
        action_generator_fn=(
            generate_hidden_routing_gate_action_script
        ),
    )
    (
        free_trace,
        hidden_trace,
        free_meta,
        hidden_meta,
        actions,
        pair_meta,
    ) = result

    public_layout_sha = (
        _public_layout_match(
            actions,
            free_meta,
            hidden_meta,
        )
    )
    metrics = compute_pair_metrics(
        free_trace,
        hidden_trace,
        free_meta,
        hidden_meta,
        hz=float(config["hz"]),
        trace_stride=int(
            config["trace_stride"]
        ),
    )
    routing = (
        hidden_routing_gate_outcome_metrics(
            free_trace,
            hidden_trace,
            actions,
        )
    )
    legacy_progress = (
        hidden_hook_outcome_metrics(
            free_trace,
            hidden_trace,
            actions,
        )
    )
    mechanism = _routing_mechanism_metrics(
        free_trace,
        hidden_trace,
    )
    minimum_fraction = float(
        config["action"][
            "min_achieved_fraction"
        ]
    )
    free_probe_valid = (
        _probe_motion_valid(
            free_meta,
            minimum_fraction,
        )
    )
    hidden_probe_valid = (
        _probe_motion_valid(
            hidden_meta,
            minimum_fraction,
        )
    )
    free_routing_valid = (
        tension_motion_valid(
            free_meta,
            minimum_fraction,
        )
    )
    hidden_routing_valid = (
        tension_motion_valid(
            hidden_meta,
            minimum_fraction,
        )
    )

    row = {
        "candidate_id": (
            config["candidate_id"]
        ),
        "topology_id": (
            config["topology_id"]
        ),
        "group_id": group_id,
        "seed": int(seed),
        **metrics,
        **routing,
        **legacy_progress,
        **pair_meta,
        **mechanism,
        "public_task_layout_sha256": (
            public_layout_sha
        ),
        "free_probe_motion_valid": (
            free_probe_valid
        ),
        "hidden_probe_motion_valid": (
            hidden_probe_valid
        ),
        "probe_motion_valid": bool(
            free_probe_valid
            and hidden_probe_valid
        ),
        "free_routing_motion_valid": (
            free_routing_valid
        ),
        "hidden_routing_motion_valid": (
            hidden_routing_valid
        ),
        "routing_motion_valid": bool(
            free_routing_valid
            and hidden_routing_valid
        ),
    }
    row["hidden_gate_engagement_fraction"] = (
        float(
            mechanism[
                "hidden_main_contact_fraction_"
                "privileged"
            ]
        )
    )
    row["branch_amplification"] = float(
        metrics["main_branch_fde"]
        / max(
            float(
                metrics[
                    "preload_end_max_abs_xy"
                ]
            ),
            1e-9,
        )
    )

    save_trace(
        group_dir / "free.npz",
        free_trace,
    )
    save_trace(
        group_dir / "hidden_hook.npz",
        hidden_trace,
    )
    write_json(
        group_dir / "pair.json",
        {
            "config": config,
            "action_script": actions,
            "free_metadata": free_meta,
            "hidden_metadata": hidden_meta,
            "pair_metadata": pair_meta,
            "metrics": row,
            "official_outcome": (
                "pulled_endpoint_target_"
                "success_gap"
            ),
            "legacy_progress_is_diagnostic": (
                True
            ),
        },
    )

    samples = [
        phase0i_classifier_sample(
            group_id,
            "free",
            free_trace,
            free_meta,
            observation,
            config,
        ),
        phase0i_classifier_sample(
            group_id,
            "hidden_hook",
            hidden_trace,
            hidden_meta,
            observation,
            config,
        ),
    ]
    return result, row, samples, mechanism


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=(
            "configs/experiment2/phase0/"
            "hidden_routing_gate_phase0l.json"
        ),
    )
    parser.add_argument(
        "--output",
        default=(
            "reports/experiment2/"
            "phase0_hidden_routing_gate/"
            "phase0l"
        ),
    )
    args = parser.parse_args()

    config_path = (
        REPO_ROOT / args.config
    ).resolve()
    output_root = (
        REPO_ROOT / args.output
    ).resolve()
    config = json.loads(
        config_path.read_text(
            encoding="utf-8"
        )
    )
    stage_name = str(config.get("stage_name", "Experiment2 Phase 0L"))
    eligible_verdict = str(config.get(
        "eligible_verdict", "HIDDEN_ROUTING_GATE_SMOKE_ELIGIBLE"
    ))
    blocked_verdict = str(config.get(
        "blocked_verdict", "HIDDEN_ROUTING_GATE_SMOKE_BLOCKED"
    ))

    if config["topology_policy"] != (
        "single_fixed_topology_no_grid_search"
    ):
        raise ValueError(
            "Phase 0L must use one topology"
        )
    if config["action_policy"] != (
        "single_fixed_probe_and_"
        "routing_pull_no_search"
    ):
        raise ValueError(
            "Phase 0L must use one action"
        )
    if config["official_outcome"] != (
        "pulled_endpoint_target_success_gap"
    ):
        raise ValueError(
            "Phase 0L outcome changed"
        )
    if config["seeds"] != [
        71001,
        71002,
        71003,
    ]:
        raise ValueError(
            "Phase 0L seed list changed"
        )
    forbidden = {
        "candidates",
        "barrier_candidates",
        "action_candidates",
        "target_candidates",
    }
    if forbidden.intersection(config):
        raise ValueError(
            "Phase 0L contains a search"
        )

    raw_root = output_root / "raw"
    observation = dict(
        config["observation"]
    )
    execution = dict(
        config["execution"]
    )
    targets = dict(
        config["selection_targets"]
    )

    pair_rows = []
    samples = []
    pair_runs = []
    mechanism_rows = []
    sensor_rows = []
    alignment_rows = []
    geometry_rows = []

    for seed_value in config["seeds"]:
        seed = int(seed_value)
        try:
            (
                result,
                row,
                pair_samples,
                mechanism,
            ) = run_one_pair(
                config,
                seed,
                execution,
                observation,
                raw_root,
            )
        except HiddenRoutingGateGeometryError as error:
            _write_geometry_failure(
                output_root,
                config,
                config_path,
                seed,
                error,
                stage_name,
                blocked_verdict,
            )
            print(output_root / "summary.json")
            return
        (
            free_trace,
            hidden_trace,
            free_meta,
            hidden_meta,
        ) = result[:4]

        free_geometry = _geometry_audit_from_metadata(free_meta)
        hidden_geometry = _geometry_audit_from_metadata(hidden_meta)
        free_geometry_hash = canonical_json_sha256(free_geometry)
        hidden_geometry_hash = canonical_json_sha256(hidden_geometry)
        if free_geometry_hash != hidden_geometry_hash:
            raise RuntimeError(
                "free/hidden geometry provenance does not match"
            )
        geometry_rows.append({
            "seed": seed,
            "group_id": row["group_id"],
            "geometry_audit_sha256": free_geometry_hash,
            "audit": free_geometry,
        })

        pair_rows.append(row)
        samples.extend(pair_samples)
        mechanism_rows.append({
            "seed": seed,
            "group_id": row["group_id"],
            **mechanism,
        })
        pair_runs.append({
            "seed": seed,
            "group_id": row["group_id"],
            "free_metadata": free_meta,
            "hidden_metadata": hidden_meta,
        })

        for condition, trace, sample in (
            (
                "free",
                free_trace,
                pair_samples[0],
            ),
            (
                "hidden_hook",
                hidden_trace,
                pair_samples[1],
            ),
        ):
            sensor_rows.append({
                "seed": seed,
                "group_id": (
                    row["group_id"]
                ),
                "condition": condition,
                "feature_norm": float(
                    np.linalg.norm(
                        sample["formal_sensor"]
                    )
                ),
                "feature_finite": bool(
                    np.all(np.isfinite(
                        sample["formal_sensor"]
                    ))
                ),
            })
            alignment_rows.append({
                "seed": seed,
                "group_id": (
                    row["group_id"]
                ),
                "condition": condition,
                **contact_alignment(trace),
            })

        print(
            f"{config['candidate_id']} "
            f"seed={seed} "
            f"free_success="
            f"{row['free_endpoint_target_success']} "
            f"hidden_success="
            f"{row['hidden_endpoint_target_success']} "
            f"routing_gap="
            f"{row['routing_success_gap']} "
            f"margin_gap="
            f"{row['endpoint_target_margin_gap']:.6f}",
            flush=True,
        )

    schemas = {
        tuple(sample["formal_schema"])
        for sample in samples
    }
    if len(schemas) != 1:
        raise RuntimeError(
            "formal sensor schema changed"
        )
    formal_schema = list(
        next(iter(schemas))
    )
    l2 = float(
        observation["ridge_l2"]
    )

    def classifier(key):
        return grouped_ridge_accuracy(
            samples,
            key,
            l2,
        )

    candidate = {
        "id": config["candidate_id"],
        "topology_id": (
            config["topology_id"]
        ),
        "routing_gate": dict(
            config["routing_gate"]
        ),
        "action": {
            "stage1_pull_distance": (
                config["action"][
                    "stage1_pull_distance"
                ]
            ),
            "final_pull_distance": (
                config["action"][
                    "final_pull_distance"
                ]
            ),
        },
        "official_outcome": (
            config["official_outcome"]
        ),
    }
    topology = (
        summarize_hidden_routing_gate(
            config["candidate_id"],
            candidate,
            pair_rows,
            targets,
            classifier("vision_raw"),
            classifier("vision_delta"),
            classifier("formal_sensor"),
            classifier("oracle_contact"),
        )
    )
    candidate_report_filename = str(config.get(
        "candidate_report_filename",
        "candidate_hidden_routing_gate_v1.json",
    ))
    if Path(candidate_report_filename).name != candidate_report_filename:
        raise ValueError(
            "candidate_report_filename must be a plain filename"
        )
    if not candidate_report_filename.endswith(".json"):
        raise ValueError("candidate report must be JSON")
    motion_debug = _motion_debug(
        pair_runs
    )
    failed_checks = [
        key
        for key, passed
        in topology["checks"].items()
        if not passed
    ]
    if not motion_debug[
        "all_cartesian_stages_successful"
    ]:
        failed_checks.append(
            "cartesian_motion_execution"
        )
    failed_checks = sorted(
        set(failed_checks)
    )

    verdict = (
        eligible_verdict
        if (
            topology["eligible"]
            and motion_debug[
                "all_cartesian_stages_successful"
            ]
        )
        else blocked_verdict
    )

    summary = {
        "stage": stage_name,
        "resume_of": config.get("resume_of"),
        "geometry_repair": config.get("geometry_repair"),
        "required_provenance_report": config.get("required_provenance_report"),
        "required_preflight_report": config.get("required_preflight_report"),
        "candidate_report": candidate_report_filename,
        "probe_selector_mode": (
            config["routing_gate"].get(
                "probe_selector_mode",
                "fixed_center_ratio",
            )
        ),
        "gate": (
            "Stage M0 fixed outcome-aligned "
            "hidden routing-gate smoke"
        ),
        "runtime_main_code_sha": git_sha(
            REPO_ROOT
        ),
        "runtime_submodule_sha": git_sha(
            SUBMODULE_ROOT
        ),
        "config": str(
            config_path.relative_to(
                REPO_ROOT
            )
        ),
        "config_hash": (
            canonical_json_sha256(
                config
            )
        ),
        "candidate_id": (
            config["candidate_id"]
        ),
        "topology_id": (
            config["topology_id"]
        ),
        "official_outcome": (
            config["official_outcome"]
        ),
        "official_outcome_gate": (
            "median routing_success_gap "
            ">= 1.0"
        ),
        "legacy_mean_progress_is_gate": (
            False
        ),
        "oracle_used_as_formal_feature": (
            False
        ),
        "training_performed": False,
        "geometry_search_performed": False,
        "action_search_performed": False,
        "outcome_search_performed": False,
        "phase0i_phase0j_phase0k_"
        "verdicts_changed": False,
        "seeds": [
            int(value)
            for value in config["seeds"]
        ],
        "topology": topology,
        "targets": targets,
        "failed_checks": failed_checks,
        "verdict": verdict,
        "interpretation": (
            "One fixed outcome-aligned routing "
            "task. ELIGIBLE permits only expanded "
            "validation of this exact task; it is "
            "not Scientific PASS and does not "
            "permit training."
        ),
    }

    write_json(
        output_root
        / candidate_report_filename,
        topology,
    )
    write_json(
        output_root / "motion_debug.json",
        motion_debug,
    )
    write_json(
        output_root / "sensor_audit.json",
        {
            "formal_sensor_feature_version": (
                "stage_aligned_formal_sensor_v1"
            ),
            "oracle_used_as_formal_feature": (
                False
            ),
            "rows": sensor_rows,
        },
    )
    write_json(
        output_root / "contact_alignment.json",
        {
            "oracle_is_privileged_audit_only": (
                True
            ),
            "rows": alignment_rows,
        },
    )
    write_json(
        output_root / "feature_schema.json",
        {
            "version": (
                "stage_aligned_formal_sensor_v1"
            ),
            "dimension": len(formal_schema),
            "schema": formal_schema,
        },
    )
    write_json(
        output_root / "mechanism_audit.json",
        {
            "topology": (
                config["topology_id"]
            ),
            "oracle_is_privileged_audit_only": (
                True
            ),
            "rows": mechanism_rows,
        },
    )
    write_json(
        output_root / "geometry_audit.json",
        {
            "engineering_provenance": True,
            "official_model_feature": False,
            "rows": geometry_rows,
        },
    )
    write_json(
        output_root / "routing_outcome_audit.json",
        {
            "metric_version": (
                "pulled_endpoint_target_"
                "success_v1"
            ),
            "official_outcome": (
                True
            ),
            "legacy_mean_progress_is_gate": (
                False
            ),
            "rows": [
                {
                    "seed": row["seed"],
                    "group_id": (
                        row["group_id"]
                    ),
                    "free_endpoint_target_success": (
                        row[
                            "free_endpoint_target_success"
                        ]
                    ),
                    "hidden_endpoint_target_success": (
                        row[
                            "hidden_endpoint_target_success"
                        ]
                    ),
                    "routing_success_gap": (
                        row[
                            "routing_success_gap"
                        ]
                    ),
                    "free_endpoint_target_margin": (
                        row[
                            "free_endpoint_target_margin"
                        ]
                    ),
                    "hidden_endpoint_target_margin": (
                        row[
                            "hidden_endpoint_target_margin"
                        ]
                    ),
                    "endpoint_target_margin_gap": (
                        row[
                            "endpoint_target_margin_gap"
                        ]
                    ),
                    "leading_crossing_fraction_gap": (
                        row[
                            "leading_crossing_fraction_gap"
                        ]
                    ),
                    "legacy_mean_cable_progress_gap": (
                        row[
                            "mean_cable_progress_gap"
                        ]
                    ),
                }
                for row in pair_rows
            ],
        },
    )
    write_json(
        output_root / "summary.json",
        summary,
    )

    lines = [
        f"# {stage_name} — Hidden Routing-Gate Smoke",
        "",
        f"- Candidate: "
        f"`{config['candidate_id']}`",
        f"- Candidate report: "
        f"`{candidate_report_filename}`",
        f"- Topology: "
        f"`{config['topology_id']}`",
        f"- Probe selector: "
        f"`{summary['probe_selector_mode']}`",
        "- Official outcome: "
        "`pulled_endpoint_target_success_gap`",
        "- Official gate: "
        "`median routing_success_gap >= 1.0`",
        "- Legacy mean progress is a gate: `False`",
        f"- Seeds: `{summary['seeds']}`",
        f"- Free target success fraction: "
        f"`{topology['success_fraction']['free_endpoint_target']:.6f}`",
        f"- Hidden target success fraction: "
        f"`{topology['success_fraction']['hidden_endpoint_target']:.6f}`",
        f"- Median routing success gap: "
        f"`{topology['median']['routing_success_gap']:.6f}`",
        f"- Median target margin gap: "
        f"`{topology['median']['endpoint_target_margin_gap']:.9f}`",
        f"- Failed checks: `{failed_checks}`",
        f"- Verdict: `{verdict}`",
        "- Training performed: `False`",
        "- Geometry search performed: `False`",
        "- Action search performed: `False`",
        "- Outcome search performed: `False`",
        "- Status: fixed 3-seed Stage M0 smoke, "
        "not Scientific PASS.",
    ]
    (
        output_root / "summary.md"
    ).write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(output_root / "summary.json")


if __name__ == "__main__":
    main()
