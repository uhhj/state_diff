#!/usr/bin/env python3
"""Offline Phase 0H sensor reanalysis with the frozen Phase 0I feature."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiment2.phase0.observation_common import grouped_ridge_accuracy
from scripts.experiment2.phase0.phase0f_common import sensor_contact_feature
from scripts.experiment2.phase0.phase0i_sensor_features import (
    reaction_only_feature,
    stage_aligned_formal_sensor_feature,
)
from scripts.experiment2.phase0.run_hidden_hook_bootstrap import write_json


def _resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _load_trace(path: Path):
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", default="reports/experiment2/phase0_hidden_hook/phase0h"
    )
    parser.add_argument(
        "--output",
        default=(
            "reports/experiment2/phase0_hidden_hook/phase0i/"
            "phase0h_reanalysis"
        ),
    )
    args = parser.parse_args()
    input_root, output_root = _resolve(args.input), _resolve(args.output)
    source_summary = json.loads(
        (input_root / "summary.json").read_text(encoding="utf-8")
    )
    config_path = REPO_ROOT / source_summary["config"]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    raw_root = input_root / "raw" / "recessed_u_hook_shallow"
    groups = sorted(path for path in raw_root.iterdir() if path.is_dir())
    if not groups:
        raise RuntimeError(f"no Phase 0H raw groups under {raw_root}")

    samples, formal_schema, reaction_schema = [], None, None
    for group_dir in groups:
        pair = json.loads((group_dir / "pair.json").read_text(encoding="utf-8"))
        for condition, metadata_key in (
            ("free", "free_metadata"),
            ("hidden_hook", "hidden_metadata"),
        ):
            trace = _load_trace(group_dir / f"{condition}.npz")
            events = pair[metadata_key]["motion_events"]
            formal, schema = stage_aligned_formal_sensor_feature(
                trace, events, float(config["hz"]), int(config["trace_stride"])
            )
            reaction, current_reaction_schema = reaction_only_feature(
                trace, events, float(config["hz"]), int(config["trace_stride"])
            )
            if formal_schema is None:
                formal_schema = schema
                reaction_schema = current_reaction_schema
            if schema != formal_schema or current_reaction_schema != reaction_schema:
                raise RuntimeError("formal sensor schema changed across samples")
            samples.append({
                "group_id": group_dir.name,
                "condition": condition,
                "label": -1 if condition == "free" else 1,
                "legacy_formal_sensor": sensor_contact_feature(
                    trace, float(config["hz"]), int(config["trace_stride"])
                ),
                "formal_sensor_v1": formal,
                "reaction_only": reaction,
            })

    results = {
        key: grouped_ridge_accuracy(samples, key, l2=0.01)
        for key in (
            "legacy_formal_sensor", "formal_sensor_v1", "reaction_only"
        )
    }
    maximum_vision = float(
        source_summary["topology"]["classifiers"]["maximum_vision_accuracy"]
    )
    result = {
        "stage": "Experiment2 Phase 0I-A",
        "source_stage": "Phase 0H",
        "source_report": str(input_root.relative_to(REPO_ROOT)),
        "post_hoc_reanalysis": True,
        "phase0h_verdict_changed": False,
        "phase0h_source_verdict": source_summary["verdict"],
        "oracle_used_as_feature": False,
        "training_performed": False,
        "geometry_search_performed": False,
        "classifier": "grouped leave-one-seed-out ridge",
        "ridge_l2": 0.01,
        "group_count": len(groups),
        "legacy_formal_accuracy": float(results["legacy_formal_sensor"]["accuracy"]),
        "formal_sensor_v1_accuracy": float(results["formal_sensor_v1"]["accuracy"]),
        "reaction_only_accuracy": float(results["reaction_only"]["accuracy"]),
        "phase0h_maximum_vision_accuracy": maximum_vision,
        "formal_sensor_v1_over_vision_margin": float(
            results["formal_sensor_v1"]["accuracy"] - maximum_vision
        ),
        "classifiers": results,
        "interpretation": (
            "Post-hoc diagnostic only; Phase 0H remains BLOCKED regardless "
            "of corrected sensor accuracy."
        ),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "sensor_reanalysis.json", result)
    write_json(output_root / "feature_schema.json", {
        "formal_sensor_feature_version": "stage_aligned_formal_sensor_v1",
        "formal_sensor_v1": formal_schema,
        "reaction_only": reaction_schema,
    })
    with (output_root / "predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "feature", "group_id", "condition", "truth", "prediction", "score"
        ))
        writer.writeheader()
        for feature, classifier in results.items():
            for row in classifier["predictions"]:
                writer.writerow({"feature": feature, **row})
    (output_root / "summary.md").write_text("\n".join([
        "# Phase 0I-A — Phase 0H Formal Sensor Reanalysis",
        "",
        f"- Legacy formal accuracy: `{result['legacy_formal_accuracy']}`",
        f"- Formal sensor v1 accuracy: `{result['formal_sensor_v1_accuracy']}`",
        f"- Reaction-only accuracy: `{result['reaction_only_accuracy']}`",
        f"- Phase 0H maximum vision accuracy: `{maximum_vision}`",
        f"- Corrected sensor-over-vision margin: `{result['formal_sensor_v1_over_vision_margin']}`",
        "- Phase 0H verdict changed: `False`",
        "- Oracle used as feature: `False`",
        "- Training performed: `False`",
        "- Geometry search performed: `False`",
    ]) + "\n", encoding="utf-8")
    print(output_root / "sensor_reanalysis.json")


if __name__ == "__main__":
    main()
