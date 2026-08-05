#!/usr/bin/env python3
"""Offline diagnostic decomposition of saved Phase 0I outcomes."""
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

from scripts.experiment2.phase0.phase0j_outcome_metrics import (
    hidden_latch_outcome_decomposition,
)
from scripts.experiment2.phase0.run_hidden_hook_bootstrap import write_json


def load_trace(path: Path):
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def _median(rows, getter):
    return float(np.median([float(getter(row)) for row in rows]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", default="reports/experiment2/phase0_hidden_hook/phase0i"
    )
    parser.add_argument(
        "--output",
        default=(
            "reports/experiment2/phase0_hidden_hook/phase0j/"
            "phase0i_outcome_reanalysis"
        ),
    )
    args = parser.parse_args()
    input_root = (REPO_ROOT / args.input).resolve()
    output_root = (REPO_ROOT / args.output).resolve()
    pair_paths = sorted(input_root.glob("raw/*/*/pair.json"))
    if not pair_paths:
        raise FileNotFoundError(f"no Phase 0I pairs under {input_root / 'raw'}")

    rows = []
    for pair_path in pair_paths:
        pair = json.loads(pair_path.read_text(encoding="utf-8"))
        decomposition = hidden_latch_outcome_decomposition(
            load_trace(pair_path.parent / "free.npz"),
            load_trace(pair_path.parent / "hidden_hook.npz"),
            pair["action_script"],
        )
        metrics = pair.get("metrics", {})
        rows.append({
            "group_id": str(metrics.get("group_id", pair_path.parent.name)),
            "seed": int(metrics.get("seed", 0)),
            "source_pair": str(pair_path.relative_to(input_root)),
            "official_mean_progress_gap": float(
                metrics["mean_cable_progress_gap"]
            ),
            "decomposition": decomposition,
        })

    medians = {
        "all_progress_gap": _median(
            rows, lambda row: row["decomposition"]["segments"]["all"]["progress_gap"]
        ),
        "blocked_progress_gap": _median(
            rows, lambda row: row["decomposition"]["segments"]["blocked"]["progress_gap"]
        ),
        "pulled_side_progress_gap": _median(
            rows, lambda row: row["decomposition"]["segments"]["pulled_side"]["progress_gap"]
        ),
        "trailing_side_progress_gap": _median(
            rows, lambda row: row["decomposition"]["segments"]["trailing_side"]["progress_gap"]
        ),
        "halfway_crossing_fraction_gap": _median(
            rows, lambda row: row["decomposition"]["halfway_crossing"]["fraction_gap"]
        ),
        "motion_normal_rms": _median(
            rows, lambda row: row["decomposition"]["motion_difference_components"]["normal_rms"]
        ),
        "motion_tangent_rms": _median(
            rows, lambda row: row["decomposition"]["motion_difference_components"]["tangent_rms"]
        ),
        "motion_vertical_rms": _median(
            rows, lambda row: row["decomposition"]["motion_difference_components"]["vertical_rms"]
        ),
    }
    result = {
        "stage": "Experiment2 Phase 0J-A",
        "source_stage": "Phase 0I",
        "post_hoc_diagnostic": True,
        "phase0i_verdict_changed": False,
        "changes_official_progress_gate": False,
        "training_performed": False,
        "geometry_search_performed": False,
        "pair_count": len(rows),
        "medians": medians,
        "rows": rows,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "outcome_decomposition.json", result)
    fieldnames = [
        "group_id", "seed", "bead_index", "segments", "free_progress",
        "hidden_progress", "progress_gap", "free_tangent_motion",
        "hidden_tangent_motion", "free_vertical_motion",
        "hidden_vertical_motion", "free_crossed_halfway_plane",
        "hidden_crossed_halfway_plane",
    ]
    with (output_root / "per_bead_progress.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            for bead in row["decomposition"]["per_bead"]:
                writer.writerow({
                    "group_id": row["group_id"],
                    "seed": row["seed"],
                    **{
                        key: (
                            "|".join(bead[key])
                            if key == "segments" else bead[key]
                        )
                        for key in fieldnames[2:]
                    },
                })
    (output_root / "summary.md").write_text("\n".join([
        "# Phase 0J-A — Phase 0I Outcome Localization",
        "",
        "- Source: saved Phase 0I raw only; PyBullet was not rerun.",
        "- Post-hoc diagnostic: `True`",
        "- Changes official progress gate: `False`",
        "- Phase 0I verdict changed: `False`",
        f"- Pair count: `{len(rows)}`",
        *[f"- {key}: `{value}`" for key, value in medians.items()],
        "- Training performed: `False`",
        "- Geometry search performed: `False`",
    ]) + "\n", encoding="utf-8")
    print(output_root / "outcome_decomposition.json")


if __name__ == "__main__":
    main()
