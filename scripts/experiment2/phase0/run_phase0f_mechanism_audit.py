#!/usr/bin/env python3
"""Run bounded Phase 0F mechanism and sensor audit."""
from __future__ import annotations

import argparse
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

from scripts.experiment2.phase0.calibration_common import build_candidate_config
from scripts.experiment2.phase0.common import (
    canonical_json_sha256,
    compute_pair_metrics,
)
from scripts.experiment2.phase0.determinism_common import (
    compare_repeat_metadata,
    compare_traces,
    max_numeric_error,
)
from scripts.experiment2.phase0.exact_counterfactual import (
    run_exact_counterfactual_pair,
)
from scripts.experiment2.phase0.observation_common import (
    delta_image_feature,
    grouped_ridge_accuracy,
    image_feature,
    load_rgb,
)
from scripts.experiment2.phase0.phase0e_common import preload_return_residual
from scripts.experiment2.phase0.phase0f_common import (
    oracle_contact_feature,
    rank_phase0f,
    sensor_contact_feature,
    summarize_phase0f_candidate,
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )


def git_sha(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def save_trace(path: Path, trace: Dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **trace)


def classifier_sample(
    group_id,
    condition,
    trace,
    metadata,
    observation,
    config,
):
    no_action_rgb = load_rgb(
        Path(metadata["observations"]["no_action_end"]["path"])
    )
    pre_main_rgb = load_rgb(
        Path(metadata["observations"]["pre_main"]["path"])
    )
    return {
        "group_id": group_id,
        "condition": condition,
        "label": -1 if condition == "free" else 1,
        "vision_raw": image_feature(
            pre_main_rgb,
            observation["feature_width"],
            observation["feature_height"],
        ),
        "vision_delta": delta_image_feature(
            no_action_rgb,
            pre_main_rgb,
            observation["feature_width"],
            observation["feature_height"],
        ),
        "formal_sensor": sensor_contact_feature(
            trace,
            float(config["hz"]),
            int(config["trace_stride"]),
        ),
        "oracle_contact": oracle_contact_feature(
            trace,
            float(config["hz"]),
            int(config["trace_stride"]),
        ),
    }


def run_one_pair(
    candidate_id,
    candidate_config,
    seed,
    execution,
    observation,
    raw_root,
):
    group_id = f"hf_{seed:06d}"
    group_dir = raw_root / candidate_id / group_id
    result = run_exact_counterfactual_pair(
        candidate_config,
        seed=seed,
        group_id=group_id,
        execution=execution,
        observation_output_dir=group_dir / "observations",
        observation_config=observation,
    )
    free_trace, hidden_trace, free_meta, hidden_meta, action_script, pair_meta = result
    metrics = compute_pair_metrics(
        free_trace,
        hidden_trace,
        free_meta,
        hidden_meta,
        hz=float(candidate_config["hz"]),
        trace_stride=int(candidate_config["trace_stride"]),
    )
    row = {
        "candidate_id": candidate_id,
        "group_id": group_id,
        "seed": int(seed),
        **metrics,
        **pair_meta,
    }
    row["free_preload_return_residual"] = preload_return_residual(free_trace)
    row["hidden_preload_return_residual"] = preload_return_residual(hidden_trace)
    row["branch_amplification"] = (
        float(metrics["main_branch_fde"])
        / max(float(metrics["preload_end_max_abs_xy"]), 1e-9)
    )
    save_trace(group_dir / "free.npz", free_trace)
    save_trace(group_dir / "hidden_high_friction.npz", hidden_trace)
    write_json(group_dir / "pair.json", {
        "candidate_config": candidate_config,
        "action_script": action_script,
        "free_metadata": free_meta,
        "hidden_metadata": hidden_meta,
        "pair_metadata": pair_meta,
        "metrics": row,
    })
    samples = [
        classifier_sample(
            group_id,
            "free",
            free_trace,
            free_meta,
            observation,
            candidate_config,
        ),
        classifier_sample(
            group_id,
            "hidden_high_friction",
            hidden_trace,
            hidden_meta,
            observation,
            candidate_config,
        ),
    ]
    return result, row, samples


def run_repeat_check(ranked, base_config, execution, repeat_config):
    rows = []
    numeric_atol = float(repeat_config["numeric_atol"])
    for summary in ranked[:int(repeat_config["top_k"])]:
        candidate_id = summary["candidate_id"]
        candidate_config = build_candidate_config(
            base_config, summary["candidate"]
        )
        for seed in repeat_config["seeds"]:
            traces, metadata = [], []
            for _ in range(int(repeat_config["repeats"])):
                result = run_exact_counterfactual_pair(
                    candidate_config,
                    seed=int(seed),
                    group_id=(
                        f"phase0f_repeat_{candidate_id}_{int(seed):06d}"
                    ),
                    execution=execution,
                )
                traces.append((result[0], result[1]))
                metadata.append(result[5])
            comparisons = []
            for index in range(1, len(traces)):
                comparisons.append({
                    "repeat_a": 0,
                    "repeat_b": index,
                    "free": compare_traces(
                        traces[0][0], traces[index][0], numeric_atol
                    ),
                    "hidden": compare_traces(
                        traces[0][1], traces[index][1], numeric_atol
                    ),
                })
            flat = [
                value
                for item in comparisons
                for value in (item["free"], item["hidden"])
            ]
            metadata_result = compare_repeat_metadata(metadata)
            rows.append({
                "candidate_id": candidate_id,
                "seed": int(seed),
                "metadata": metadata_result,
                "comparisons": comparisons,
                "max_numeric_error": max_numeric_error(flat),
                "passed": bool(
                    metadata_result["passed"]
                    and all(value["passed"] for value in flat)
                ),
            })
    return {
        "numeric_atol": numeric_atol,
        "rows": rows,
        "passed": bool(rows and all(row["passed"] for row in rows)),
    }


def decide_mechanism(ranked):
    eligible = [row for row in ranked if row["eligible"]]
    native = [
        row for row in eligible
        if row["candidate"]["friction"].get("mechanism")
        == "native_segment"
    ]
    legacy = [
        row for row in eligible
        if row["candidate"]["friction"].get("mechanism")
        == "external_patch"
    ]
    if native:
        return {
            "verdict": "NATIVE_SEGMENT_PROVISIONAL_SCREEN_PASS",
            "selected_candidate": native[0]["candidate_id"],
            "next_allowed_action": (
                "Generate formal Phase 0 Gate A-H data with this native "
                "candidate."
            ),
        }
    if legacy:
        return {
            "verdict": "LEGACY_EXTERNAL_ONLY_PROVISIONAL_PASS",
            "selected_candidate": legacy[0]["candidate_id"],
            "next_allowed_action": (
                "Review physical defensibility of the external-force "
                "mechanism before formal Gate data."
            ),
        }
    return {
        "verdict": "HIDDEN_FRICTION_MECHANISMS_BLOCKED",
        "selected_candidate": None,
        "next_allowed_action": (
            "Stop friction parameter search and move to Hidden-Hook Cable "
            "Routing."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/experiment2/phase0/hidden_friction_phase0f.json",
    )
    parser.add_argument(
        "--output",
        default="reports/experiment2/phase0_hidden_friction/phase0f",
    )
    args = parser.parse_args()

    config_path = (REPO_ROOT / args.config).resolve()
    output_root = (REPO_ROOT / args.output).resolve()
    phase0f = json.loads(config_path.read_text(encoding="utf-8"))
    base_path = (REPO_ROOT / phase0f["base_config"]).resolve()
    base_config = json.loads(base_path.read_text(encoding="utf-8"))
    seeds = [int(value) for value in phase0f["seeds"]]
    execution = dict(phase0f["execution"])
    observation = dict(phase0f["observation"])
    targets = dict(phase0f["selection_targets"])
    raw_root = output_root / "raw"

    summaries = []
    for candidate in phase0f["candidates"]:
        candidate_id = str(candidate["id"])
        candidate_config = build_candidate_config(base_config, candidate)
        pair_rows, samples = [], []
        for seed in seeds:
            _, row, pair_samples = run_one_pair(
                candidate_id,
                candidate_config,
                seed,
                execution,
                observation,
                raw_root,
            )
            pair_rows.append(row)
            samples.extend(pair_samples)
            print(
                f"{candidate_id} seed={seed} "
                f"preload={row['preload_end_max_abs_xy']:.6f} "
                f"fde={row['main_branch_fde']:.6f}",
                flush=True,
            )

        l2 = float(observation["ridge_l2"])
        vision_raw = grouped_ridge_accuracy(samples, "vision_raw", l2)
        vision_delta = grouped_ridge_accuracy(samples, "vision_delta", l2)
        formal_sensor = grouped_ridge_accuracy(samples, "formal_sensor", l2)
        oracle_contact = grouped_ridge_accuracy(
            samples, "oracle_contact", l2
        )
        summary = summarize_phase0f_candidate(
            candidate_id,
            candidate,
            pair_rows,
            targets,
            vision_raw,
            vision_delta,
            formal_sensor,
            oracle_contact,
        )
        summary["candidate_config_hash"] = canonical_json_sha256(
            candidate_config
        )
        summaries.append(summary)
        write_json(output_root / f"candidate_{candidate_id}.json", summary)

    ranked = rank_phase0f(summaries)
    repeats = run_repeat_check(
        ranked, base_config, execution, phase0f["repeat_check"]
    )
    decision = decide_mechanism(ranked)

    selected_config_written = False
    selected_path = (
        REPO_ROOT
        / "configs/experiment2/phase0/hidden_friction_cable_phase0f_selected.json"
    )
    if decision["selected_candidate"] is not None:
        selected = next(
            row for row in ranked
            if row["candidate_id"] == decision["selected_candidate"]
        )
        selected_config = build_candidate_config(
            base_config, selected["candidate"]
        )
        selected_config["phase0f_selection"] = {
            "candidate_id": selected["candidate_id"],
            "engineering_screen_eligible": True,
            "source_config": str(config_path.relative_to(REPO_ROOT)),
            "source_config_hash": canonical_json_sha256(phase0f),
            "note": (
                "Provisional Stage M0 selection; not formal Gate A-H PASS."
            ),
        }
        write_json(selected_path, selected_config)
        selected_config_written = True

    top = ranked[0]
    result = {
        "stage": "Experiment2 Phase 0F",
        "runtime_main_code_sha": git_sha(REPO_ROOT),
        "runtime_submodule_sha": git_sha(SUBMODULE_ROOT),
        "config": str(config_path.relative_to(REPO_ROOT)),
        "config_hash": canonical_json_sha256(phase0f),
        "seeds": seeds,
        "execution": execution,
        "targets": targets,
        "ranked_candidates": ranked,
        "top_candidate": top["candidate_id"],
        "top_candidate_eligible": bool(top["eligible"]),
        "repeat_check": repeats,
        "mechanism_decision": decision,
        "selected_config_written": selected_config_written,
        "selected_config_path": (
            str(selected_path.relative_to(REPO_ROOT))
            if selected_config_written else None
        ),
        "interpretation": (
            "Bounded Stage M0 mechanism and sensor audit; no model training "
            "or formal Gate A-H claim."
        ),
    }
    write_json(output_root / "summary.json", result)
    write_json(output_root / "repeat_check.json", repeats)
    write_json(output_root / "mechanism_decision.json", decision)

    (output_root / "summary.md").write_text(
        "\n".join([
            "# Phase 0F Mechanism and Sensor Audit",
            "",
            f"- Candidates: `{len(ranked)}`",
            f"- Seeds: `{seeds}`",
            f"- Top candidate: `{top['candidate_id']}`",
            f"- Top eligible: `{top['eligible']}`",
            f"- Mechanism verdict: `{decision['verdict']}`",
            "- Status: Stage M0 only.",
        ]) + "\n",
        encoding="utf-8",
    )
    (output_root / "mechanism_decision.md").write_text(
        "\n".join([
            "# Phase 0F Mechanism Decision",
            "",
            f"- Verdict: `{decision['verdict']}`",
            f"- Selected candidate: `{decision['selected_candidate']}`",
            f"- Next allowed action: {decision['next_allowed_action']}",
            "",
            (
                "The Oracle hidden-force channel is not treated as a formal "
                "model input."
            ),
        ]) + "\n",
        encoding="utf-8",
    )
    print(output_root / "summary.json")


if __name__ == "__main__":
    main()
