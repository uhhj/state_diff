#!/usr/bin/env python3
"""Run the bounded Phase 0H hidden-hook observability audit (no training)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
SUBMODULE_ROOT = REPO_ROOT / "external" / "deformable-ravens"
for path in (REPO_ROOT, SUBMODULE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.experiment2.phase0.common import canonical_json_sha256
from scripts.experiment2.phase0.observation_common import grouped_ridge_accuracy
from scripts.experiment2.phase0.run_hidden_hook_bootstrap import (
    git_sha,
    run_one_pair,
    summarize_hidden_hook,
    write_json,
)


SENSOR_KEYS = (
    "sensor_joint_motor_torque_norm",
    "sensor_joint_reaction_force_torque_norm",
    "sensor_suction_force_norm",
    "sensor_suction_torque_norm",
    "sensor_grasp_active",
    "sensor_constraint_available",
)


def probe_phase_index(
    physics_steps: np.ndarray,
    motion_events: Iterable[Dict[str, Any]],
) -> np.ndarray:
    """Map trace samples to precise probe stages without hidden-state access."""
    result = np.zeros(np.asarray(physics_steps).shape, dtype=np.int64)
    stages = {
        "hook_probe_lift": 1,
        "hook_probe_out": 2,
        "hook_probe_return": 3,
        "hook_probe_lower_release": 4,
    }
    for event in motion_events:
        stage = stages.get(str(event.get("stage", event.get("label", ""))))
        start = event.get("physics_step_start")
        end = event.get("physics_step_end")
        if stage is None or start is None or end is None:
            continue
        mask = (physics_steps >= int(start)) & (physics_steps <= int(end))
        result[mask] = stage
    return result


def export_sensor_timeline(
    path: Path,
    trace: Dict[str, np.ndarray],
    metadata: Dict[str, Any],
) -> None:
    """Export raw formal sensor streams plus separately labelled Oracle timing."""
    steps = np.asarray(trace["physics_step"], dtype=np.int64)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        physics_step=steps,
        task_phase=np.asarray(trace["phase"]),
        probe_phase_index=probe_phase_index(steps, metadata["motion_events"]),
        joint_motor_torque=np.asarray(trace["sensor_joint_motor_torque"]),
        joint_reaction_force_torque=np.asarray(
            trace["sensor_joint_reaction_force_torque"]
        ),
        ee_constraint_force_xyz=np.asarray(trace["sensor_suction_force_xyz"]),
        ee_constraint_torque_xyz=np.asarray(trace["sensor_suction_torque_xyz"]),
        grasp_state=np.asarray(trace["sensor_grasp_active"]),
        constraint_available=np.asarray(trace["sensor_constraint_available"]),
        oracle_contact_force_norm_privileged=np.asarray(
            trace["contact_force_norm"]
        ),
        oracle_contact_active_beads_privileged=np.asarray(
            trace["contact_active_beads"]
        ),
    )


def _signal_summary(values: np.ndarray) -> Dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "shape": list(array.shape),
        "finite": bool(np.all(np.isfinite(array))),
        "mean": float(np.mean(array)),
        "maximum": float(np.max(array)),
        "nonzero_fraction": float(np.mean(np.abs(array) > 1e-12)),
    }


def sensor_trace_summary(trace: Dict[str, np.ndarray]) -> Dict[str, Any]:
    phase = np.asarray(trace["phase"])
    return {
        "sample_count": int(phase.size),
        "phase_sample_count": {
            str(value): int(np.sum(phase == value)) for value in np.unique(phase)
        },
        "signals": {key: _signal_summary(trace[key]) for key in SENSOR_KEYS},
    }


def free_hidden_sensor_difference(
    free_trace: Dict[str, np.ndarray], hidden_trace: Dict[str, np.ndarray]
) -> Dict[str, Any]:
    result = {}
    for key in SENSOR_KEYS:
        free = np.asarray(free_trace[key], dtype=np.float64)
        hidden = np.asarray(hidden_trace[key], dtype=np.float64)
        count = min(free.shape[0], hidden.shape[0])
        difference = hidden[:count] - free[:count]
        result[key] = {
            "aligned_samples": int(count),
            "mean_abs_difference": float(np.mean(np.abs(difference))),
            "root_mean_square_difference": float(
                np.sqrt(np.mean(np.square(difference)))
            ),
            "max_abs_difference": float(np.max(np.abs(difference))),
        }
    return result


def contact_alignment(trace: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Align formal sensor peaks to privileged contact onset for audit only."""
    steps = np.asarray(trace["physics_step"], dtype=np.int64)
    oracle_active = (
        np.asarray(trace["contact_active_beads"], dtype=np.float64) > 0
    ) | (np.asarray(trace["contact_force_norm"], dtype=np.float64) > 1e-12)
    onset_indices = np.flatnonzero(oracle_active)
    onset_index = int(onset_indices[0]) if onset_indices.size else None
    onset_step = int(steps[onset_index]) if onset_index is not None else None
    formal = {
        "joint_motor_torque_norm": np.asarray(
            trace["sensor_joint_motor_torque_norm"], dtype=np.float64
        ),
        "joint_reaction_force_torque_norm": np.asarray(
            trace["sensor_joint_reaction_force_torque_norm"], dtype=np.float64
        ),
        "ee_constraint_force_norm": np.asarray(
            trace["sensor_suction_force_norm"], dtype=np.float64
        ),
        "ee_constraint_torque_norm": np.asarray(
            trace["sensor_suction_torque_norm"], dtype=np.float64
        ),
    }
    peaks = {}
    window_radius_samples = 30
    for name, values in formal.items():
        if onset_index is not None:
            search_start = max(0, onset_index - window_radius_samples)
            search_end = min(values.size, onset_index + window_radius_samples + 1)
        else:
            search_start, search_end = 0, values.size
        peak_index = search_start + int(
            np.argmax(values[search_start:search_end])
        )
        before = values[max(0, (onset_index or 0) - 10):(onset_index or 0)]
        after = values[(onset_index or 0):min(values.size, (onset_index or 0) + 10)]
        peaks[name] = {
            "peak_physics_step": int(steps[peak_index]),
            "peak_value": float(values[peak_index]),
            "peak_lag_from_oracle_contact_steps": (
                int(steps[peak_index] - onset_step) if onset_step is not None else None
            ),
            "pre_contact_window_mean": (
                float(np.mean(before)) if onset_step is not None and before.size else None
            ),
            "post_contact_window_mean": (
                float(np.mean(after)) if onset_step is not None and after.size else None
            ),
        }
    return {
        "oracle_is_privileged_audit_only": True,
        "oracle_contact_detected": bool(onset_indices.size),
        "oracle_first_contact_physics_step": onset_step,
        "oracle_contact_sample_fraction": float(np.mean(oracle_active)),
        "alignment_peak_window_radius_samples": window_radius_samples,
        "formal_sensor_peak_alignment": peaks,
    }


def motion_debug_rows(pair_runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    rows = []
    required = (
        "primitive", "stage", "physics_step_count", "timeout_reason",
        "final_joint_error_max_abs", "final_joint_error_norm",
        "cartesian_endpoint_error", "success",
    )
    for pair in pair_runs:
        for condition in ("free", "hidden_hook"):
            for event in pair[f"{condition}_metadata"]["motion_events"]:
                row = {
                    "seed": int(pair["seed"]),
                    "group_id": pair["group_id"],
                    "condition": condition,
                    **event,
                }
                missing = [key for key in required if key not in row]
                if missing:
                    raise RuntimeError(f"motion debug event missing {missing}")
                rows.append(row)
    failures = [row for row in rows if not row["success"]]
    recovered = [
        row for row in rows
        if row["timeout_reason"] == "joint_timeout_recovered_by_cartesian_endpoint"
    ]
    return {
        "event_count": len(rows),
        "all_cartesian_stages_successful": not failures,
        "failed_event_count": len(failures),
        "recovered_timeout_event_count": len(recovered),
        "recovered_joint_timeout_count": int(sum(
            row["joint_timeout_count"] for row in recovered
        )),
        "unchanged_tolerances": True,
        "events": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/experiment2/phase0/hidden_hook_phase0h.json"
    )
    parser.add_argument(
        "--output", default="reports/experiment2/phase0_hidden_hook/phase0h"
    )
    args = parser.parse_args()
    config_path = (REPO_ROOT / args.config).resolve()
    output_root = (REPO_ROOT / args.output).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    raw_root = output_root / "raw"
    observation = dict(config["observation"])
    execution = dict(config["execution"])
    targets = dict(config["selection_targets"])
    candidate_id = str(config["topology_id"])
    candidate = {"id": candidate_id, "hook": dict(config["hook"])}

    pair_rows, samples, pair_runs = [], [], []
    sensor_rows, alignment_rows = [], []
    for seed_value in config["seeds"]:
        seed = int(seed_value)
        result, row, pair_samples = run_one_pair(
            candidate_id, config, seed, execution, observation, raw_root
        )
        free_trace, hidden_trace, free_meta, hidden_meta = result[:4]
        pair_rows.append(row)
        samples.extend(pair_samples)
        pair_runs.append({
            "seed": seed,
            "group_id": row["group_id"],
            "free_metadata": free_meta,
            "hidden_hook_metadata": hidden_meta,
        })
        for condition, trace, metadata in (
            ("free", free_trace, free_meta),
            ("hidden_hook", hidden_trace, hidden_meta),
        ):
            timeline_path = (
                raw_root / candidate_id / row["group_id"]
                / f"sensor_timeline_{condition}.npz"
            )
            export_sensor_timeline(timeline_path, trace, metadata)
            sensor_rows.append({
                "seed": seed,
                "group_id": row["group_id"],
                "condition": condition,
                "timeline": str(timeline_path.relative_to(REPO_ROOT)),
                **sensor_trace_summary(trace),
            })
            alignment_rows.append({
                "seed": seed,
                "group_id": row["group_id"],
                "condition": condition,
                **contact_alignment(trace),
            })
        sensor_rows.append({
            "seed": seed,
            "group_id": row["group_id"],
            "condition": "free_hidden_difference",
            "signals": free_hidden_sensor_difference(free_trace, hidden_trace),
        })
        print(
            f"{candidate_id} seed={seed} preload={row['preload_end_max_abs_xy']:.6f} "
            f"fde={row['main_branch_fde']:.6f} progress_gap={row['mean_cable_progress_gap']:.6f}",
            flush=True,
        )

    l2 = float(observation["ridge_l2"])
    classifier = lambda key: grouped_ridge_accuracy(samples, key, l2)
    topology_summary = summarize_hidden_hook(
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
    sensor_audit = {
        "formal_sensor_contract": {
            "allowed": [
                "joint motor torque", "joint reaction force/torque",
                "end-effector constraint reaction", "grasp state",
                "constraint availability", "probe phase index",
            ],
            "forbidden": [
                "hidden condition label", "hidden object ID", "hook layout",
                "Oracle contact force",
            ],
            "oracle_contact_use": "contact-alignment audit only",
        },
        "rows": sensor_rows,
    }
    alignment = {
        "method": (
            "formal sensor local peaks within +/-30 trace samples aligned to "
            "privileged contact onset; Oracle is audit-only"
        ),
        "rows": alignment_rows,
    }
    verdict = (
        "HIDDEN_HOOK_OBSERVABILITY_PASS"
        if topology_summary["eligible"]
        and motion_debug["all_cartesian_stages_successful"]
        else "HIDDEN_HOOK_OBSERVABILITY_BLOCKED"
    )
    failed_checks = [
        key for key, passed in topology_summary["checks"].items() if not passed
    ]
    summary = {
        "stage": "Experiment2 Phase 0H",
        "gate": "Stage M0 hidden-hook observability redesign",
        "runtime_main_code_sha": git_sha(REPO_ROOT),
        "runtime_submodule_sha": git_sha(SUBMODULE_ROOT),
        "config": str(config_path.relative_to(REPO_ROOT)),
        "config_hash": canonical_json_sha256(config),
        "seeds": [int(value) for value in config["seeds"]],
        "topology_policy": config["topology_policy"],
        "topology": topology_summary,
        "targets": targets,
        "motion_debug": "motion_debug.json",
        "sensor_audit": "sensor_audit.json",
        "contact_alignment": "contact_alignment.json",
        "failed_checks": failed_checks,
        "verdict": verdict,
        "training_performed": False,
        "geometry_search_performed": False,
        "interpretation": (
            "Bounded Stage M0 environment audit only; no model, Gate A-H, "
            "Scientific PASS, or closed-loop claim."
        ),
    }
    write_json(output_root / "candidate_recessed_u_hook_shallow.json", topology_summary)
    write_json(output_root / "motion_debug.json", motion_debug)
    write_json(output_root / "sensor_audit.json", sensor_audit)
    write_json(output_root / "contact_alignment.json", alignment)
    write_json(output_root / "summary.json", summary)
    status_lines = [
        "# Phase 0H Hidden-Hook Observability Redesign",
        "",
        f"- Topology: `{candidate_id}` (single fixed topology; no grid search)",
        f"- Seeds: `{summary['seeds']}`",
        f"- Motion stages successful: `{motion_debug['all_cartesian_stages_successful']}`",
        f"- Recovered joint timeouts: `{motion_debug['recovered_joint_timeout_count']}`",
        f"- Failed checks: `{failed_checks}`",
        f"- Verdict: `{verdict}`",
        "- Training performed: `False`",
        "- Oracle contact is privileged and used only for alignment audit.",
        "- Status: bounded Stage M0 engineering evidence, not Scientific PASS.",
    ]
    (output_root / "summary.md").write_text(
        "\n".join(status_lines) + "\n", encoding="utf-8"
    )
    print(output_root / "summary.json")


if __name__ == "__main__":
    main()
