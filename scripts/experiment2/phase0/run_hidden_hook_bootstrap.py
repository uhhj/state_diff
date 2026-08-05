#!/usr/bin/env python3
"""Run the bounded Phase 0G Hidden-Hook bootstrap audit."""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
SUBMODULE_ROOT = REPO_ROOT / "external" / "deformable-ravens"
for path in (REPO_ROOT, SUBMODULE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.experiment2.phase0.calibration_common import summarize_candidate
from scripts.experiment2.phase0.common import canonical_json_sha256, compute_pair_metrics
from scripts.experiment2.phase0.determinism_common import (
    compare_repeat_metadata,
    compare_traces,
    max_numeric_error,
)
from scripts.experiment2.phase0.exact_counterfactual import run_exact_counterfactual_pair
from scripts.experiment2.phase0.hidden_hook_common import (
    configure_hidden_hook_environment,
    generate_hidden_hook_action_script,
)
from scripts.experiment2.phase0.hidden_hook_metrics import hidden_hook_outcome_metrics
from scripts.experiment2.phase0.observation_common import (
    delta_image_feature,
    grouped_ridge_accuracy,
    image_feature,
    load_rgb,
)
from scripts.experiment2.phase0.phase0f_common import (
    oracle_contact_feature,
    sensor_contact_feature,
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def git_sha(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def candidate_config(base: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    result["hook"] = copy.deepcopy(candidate["hook"])
    return result


def save_trace(path: Path, trace: Dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **trace)


def classifier_sample(group_id, condition, trace, metadata, observation, config):
    no_action_rgb = load_rgb(Path(metadata["observations"]["no_action_end"]["path"]))
    pre_main_rgb = load_rgb(Path(metadata["observations"]["pre_main"]["path"]))
    return {
        "group_id": group_id,
        "condition": condition,
        "label": -1 if condition == "free" else 1,
        "vision_raw": image_feature(
            pre_main_rgb, observation["feature_width"], observation["feature_height"]
        ),
        "vision_delta": delta_image_feature(
            no_action_rgb,
            pre_main_rgb,
            observation["feature_width"],
            observation["feature_height"],
        ),
        "formal_sensor": sensor_contact_feature(
            trace, float(config["hz"]), int(config["trace_stride"])
        ),
        "oracle_contact": oracle_contact_feature(
            trace, float(config["hz"]), int(config["trace_stride"])
        ),
    }


def probe_motion(metadata):
    rows = [
        row for row in metadata.get("motion_events", [])
        if row.get("label") == "hook_probe_out"
    ]
    if len(rows) != 1:
        raise RuntimeError(f"expected one hook_probe_out event, got {len(rows)}")
    return rows[0]


def run_one_pair(candidate_id, config, seed, execution, observation, raw_root):
    group_id = f"hh_{seed:06d}"
    group_dir = raw_root / candidate_id / group_id
    result = run_exact_counterfactual_pair(
        config,
        seed=seed,
        group_id=group_id,
        execution=execution,
        observation_output_dir=group_dir / "observations",
        observation_config=observation,
        hidden_condition="hidden_hook",
        configure_environment_fn=configure_hidden_hook_environment,
        action_generator_fn=generate_hidden_hook_action_script,
    )
    free_trace, hidden_trace, free_meta, hidden_meta, actions, pair_meta = result
    metrics = compute_pair_metrics(
        free_trace,
        hidden_trace,
        free_meta,
        hidden_meta,
        hz=float(config["hz"]),
        trace_stride=int(config["trace_stride"]),
    )
    outcome = hidden_hook_outcome_metrics(free_trace, hidden_trace, actions)
    free_probe = probe_motion(free_meta)
    hidden_probe = probe_motion(hidden_meta)
    minimum_fraction = float(config["action"]["min_achieved_fraction"])
    motion_valid = bool(
        free_probe["success"]
        and hidden_probe["success"]
        and float(free_probe["achieved_fraction"]) >= minimum_fraction
        and float(hidden_probe["achieved_fraction"]) >= minimum_fraction
    )
    row = {
        "candidate_id": candidate_id,
        "group_id": group_id,
        "seed": int(seed),
        **metrics,
        **outcome,
        **pair_meta,
        "free_probe_motion": free_probe,
        "hidden_probe_motion": hidden_probe,
        "probe_motion_valid": motion_valid,
    }
    row["branch_amplification"] = (
        float(metrics["main_branch_fde"])
        / max(float(metrics["preload_end_max_abs_xy"]), 1e-9)
    )
    save_trace(group_dir / "free.npz", free_trace)
    save_trace(group_dir / "hidden_hook.npz", hidden_trace)
    write_json(group_dir / "pair.json", {
        "candidate_config": config,
        "action_script": actions,
        "free_metadata": free_meta,
        "hidden_metadata": hidden_meta,
        "pair_metadata": pair_meta,
        "metrics": row,
    })
    samples = [
        classifier_sample(group_id, "free", free_trace, free_meta, observation, config),
        classifier_sample(
            group_id, "hidden_hook", hidden_trace, hidden_meta, observation, config
        ),
    ]
    return result, row, samples


def summarize_hidden_hook(
    candidate_id,
    candidate,
    pair_rows,
    targets,
    vision_raw,
    vision_delta,
    formal_sensor,
    oracle_contact,
):
    base_keys = (
        "max_initial_abs_xy", "max_arm_jump", "max_median_no_action_drift",
        "max_median_preload_visible_difference", "min_median_contact_impulse_gap",
        "min_median_main_branch_ade", "min_median_main_branch_fde",
        "min_median_branch_amplification",
    )
    summary = summarize_candidate(
        candidate_id,
        candidate,
        pair_rows,
        {key: targets[key] for key in base_keys},
    )
    vision_accuracy = max(float(vision_raw["accuracy"]), float(vision_delta["accuracy"]))
    sensor_accuracy = float(formal_sensor["accuracy"])
    oracle_accuracy = float(oracle_contact["accuracy"])
    sensor_margin = sensor_accuracy - vision_accuracy
    progress_gap = float(np.median([row["mean_cable_progress_gap"] for row in pair_rows]))
    opposite_gap = float(np.median([
        row["opposite_endpoint_progress_gap"] for row in pair_rows
    ]))
    engagement = float(np.median([
        row["hidden_hook_engagement_fraction"] for row in pair_rows
    ]))
    fde_fraction = float(np.mean([
        row["main_branch_fde"] >= targets["min_median_main_branch_fde"]
        for row in pair_rows
    ]))
    amplification_fraction = float(np.mean([
        row["branch_amplification"] >= targets["min_median_branch_amplification"]
        for row in pair_rows
    ]))
    summary["checks"].update({
        "vision_screen": vision_accuracy <= targets["max_grouped_vision_accuracy"],
        "formal_sensor_classifier": sensor_accuracy >= targets["min_grouped_sensor_accuracy"],
        "oracle_classifier": oracle_accuracy >= targets["min_grouped_oracle_accuracy"],
        "sensor_over_vision_margin": sensor_margin >= targets["min_sensor_over_vision_margin"],
        "outcome_progress_gap": progress_gap >= targets["min_median_progress_gap"],
        "hook_engagement": engagement >= targets["min_median_engagement_fraction"],
        "precise_probe_execution": all(row["probe_motion_valid"] for row in pair_rows),
        "fde_seed_fraction": fde_fraction >= targets["min_seed_pass_fraction"],
        "amplification_seed_fraction": amplification_fraction >= targets["min_seed_pass_fraction"],
    })
    summary["eligible"] = bool(all(summary["checks"].values()))
    summary["classifiers"] = {
        "vision_raw": vision_raw,
        "vision_delta": vision_delta,
        "formal_sensor": formal_sensor,
        "oracle_contact": oracle_contact,
        "maximum_vision_accuracy": vision_accuracy,
        "sensor_over_vision_margin": sensor_margin,
    }
    summary["median"].update({
        "mean_cable_progress_gap": progress_gap,
        "opposite_endpoint_progress_gap": opposite_gap,
        "hidden_hook_engagement_fraction": engagement,
    })
    summary["seed_pass_fraction"] = {
        "main_fde": fde_fraction,
        "branch_amplification": amplification_fraction,
    }
    summary["probe_motion_valid"] = bool(
        all(row["probe_motion_valid"] for row in pair_rows)
    )
    summary["score"] = float(
        summary["score"] + 2.0 * sensor_accuracy + oracle_accuracy
        + max(0.0, sensor_margin) + (1.0 - vision_accuracy)
        + min(max(progress_gap / max(targets["min_median_progress_gap"], 1e-12), 0.0), 2.0)
        + min(max(engagement / max(targets["min_median_engagement_fraction"], 1e-12), 0.0), 2.0)
        + fde_fraction + amplification_fraction
    )
    return summary


def rank_candidates(rows):
    return sorted(rows, key=lambda row: (
        not bool(row["eligible"]),
        -float(row["score"]),
        float(row["classifiers"]["maximum_vision_accuracy"]),
        -float(row["classifiers"]["formal_sensor"]["accuracy"]),
        -float(row["median"]["mean_cable_progress_gap"]),
        str(row["candidate_id"]),
    ))


def run_repeat_check(top, base_config, execution, repeat_config):
    rows = []
    numeric_atol = float(repeat_config["numeric_atol"])
    config = candidate_config(base_config, top["candidate"])
    for seed in repeat_config["seeds"]:
        traces, metadata = [], []
        for _ in range(int(repeat_config["repeats"])):
            result = run_exact_counterfactual_pair(
                config,
                seed=int(seed),
                group_id=f"phase0g_repeat_{top['candidate_id']}_{int(seed):06d}",
                execution=execution,
                hidden_condition="hidden_hook",
                configure_environment_fn=configure_hidden_hook_environment,
                action_generator_fn=generate_hidden_hook_action_script,
            )
            traces.append((result[0], result[1]))
            metadata.append(result[5])
        comparisons = []
        for index in range(1, len(traces)):
            comparisons.append({
                "repeat_a": 0,
                "repeat_b": index,
                "free": compare_traces(traces[0][0], traces[index][0], numeric_atol),
                "hidden": compare_traces(traces[0][1], traces[index][1], numeric_atol),
            })
        flat = [
            value for comparison in comparisons
            for value in (comparison["free"], comparison["hidden"])
        ]
        metadata_result = compare_repeat_metadata(metadata)
        rows.append({
            "candidate_id": top["candidate_id"],
            "seed": int(seed),
            "metadata": metadata_result,
            "comparisons": comparisons,
            "max_numeric_error": max_numeric_error(flat),
            "passed": bool(
                metadata_result["passed"] and all(value["passed"] for value in flat)
            ),
        })
    return {
        "numeric_atol": numeric_atol,
        "rows": rows,
        "passed": bool(rows and all(row["passed"] for row in rows)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/experiment2/phase0/hidden_hook_phase0g.json"
    )
    parser.add_argument(
        "--output", default="reports/experiment2/phase0_hidden_hook/phase0g"
    )
    args = parser.parse_args()

    config_path = (REPO_ROOT / args.config).resolve()
    output_root = (REPO_ROOT / args.output).resolve()
    phase0g = json.loads(config_path.read_text(encoding="utf-8"))
    base_config = {
        key: copy.deepcopy(value)
        for key, value in phase0g.items()
        if key not in {"candidates", "selection_targets", "repeat_check", "seeds"}
    }
    seeds = [int(value) for value in phase0g["seeds"]]
    candidates = list(phase0g["candidates"])
    execution = dict(phase0g["execution"])
    observation = dict(phase0g["observation"])
    targets = dict(phase0g["selection_targets"])
    raw_root = output_root / "raw"

    summaries = []
    for candidate in candidates:
        candidate_id = str(candidate["id"])
        config = candidate_config(base_config, candidate)
        pair_rows, samples = [], []
        for seed in seeds:
            _, row, pair_samples = run_one_pair(
                candidate_id, config, seed, execution, observation, raw_root
            )
            pair_rows.append(row)
            samples.extend(pair_samples)
            print(
                f"{candidate_id} seed={seed} preload={row['preload_end_max_abs_xy']:.6f} "
                f"fde={row['main_branch_fde']:.6f} progress_gap={row['mean_cable_progress_gap']:.6f}",
                flush=True,
            )

        l2 = float(observation["ridge_l2"])
        summary = summarize_hidden_hook(
            candidate_id,
            candidate,
            pair_rows,
            targets,
            grouped_ridge_accuracy(samples, "vision_raw", l2),
            grouped_ridge_accuracy(samples, "vision_delta", l2),
            grouped_ridge_accuracy(samples, "formal_sensor", l2),
            grouped_ridge_accuracy(samples, "oracle_contact", l2),
        )
        summary["candidate_config_hash"] = canonical_json_sha256(config)
        summaries.append(summary)
        write_json(output_root / f"candidate_{candidate_id}.json", summary)

    ranked = rank_candidates(summaries)
    top = ranked[0]
    repeats = run_repeat_check(
        top, base_config, execution, phase0g["repeat_check"]
    )
    verdict = (
        "HIDDEN_HOOK_BOOTSTRAP_PASS"
        if bool(top["eligible"] and repeats["passed"])
        else "HIDDEN_HOOK_BOOTSTRAP_BLOCKED"
    )
    selected_path = REPO_ROOT / "configs/experiment2/phase0/hidden_hook_phase0g_selected.json"
    selected_written = False
    if verdict == "HIDDEN_HOOK_BOOTSTRAP_PASS":
        selected = candidate_config(base_config, top["candidate"])
        selected["phase0g_selection"] = {
            "candidate_id": top["candidate_id"],
            "bootstrap_eligible": True,
            "source_config": str(config_path.relative_to(REPO_ROOT)),
            "source_config_hash": canonical_json_sha256(phase0g),
            "note": "Stage M0 bootstrap selection; not Scientific PASS.",
        }
        write_json(selected_path, selected)
        selected_written = True

    result = {
        "stage": "Experiment2 Phase 0G",
        "runtime_main_code_sha": git_sha(REPO_ROOT),
        "runtime_submodule_sha": git_sha(SUBMODULE_ROOT),
        "config": str(config_path.relative_to(REPO_ROOT)),
        "config_hash": canonical_json_sha256(phase0g),
        "seeds": seeds,
        "execution": execution,
        "targets": targets,
        "ranked_candidates": ranked,
        "top_candidate": top["candidate_id"],
        "top_candidate_eligible": bool(top["eligible"]),
        "repeat_check": repeats,
        "verdict": verdict,
        "selected_config_written": selected_written,
        "selected_config_path": (
            str(selected_path.relative_to(REPO_ROOT)) if selected_written else None
        ),
        "interpretation": (
            "Bounded Stage M0 Hidden-Hook bootstrap; no model training or formal Gate A-H claim."
        ),
    }
    write_json(output_root / "summary.json", result)
    write_json(output_root / "repeat_check.json", repeats)
    (output_root / "summary.md").write_text(
        "\n".join([
            "# Phase 0G Hidden-Hook Bootstrap",
            "",
            f"- Candidates: `{len(ranked)}`",
            f"- Seeds: `{seeds}`",
            f"- Top candidate: `{top['candidate_id']}`",
            f"- Top eligible: `{top['eligible']}`",
            f"- Repeat check: `{repeats['passed']}`",
            f"- Verdict: `{verdict}`",
            "- Status: Stage M0 bootstrap only.",
            "- Oracle hook contact is privileged and is not a formal sensor input.",
        ]) + "\n",
        encoding="utf-8",
    )
    print(output_root / "summary.json")


if __name__ == "__main__":
    main()
