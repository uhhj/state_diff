"""Classify OHJ control relevance from the fixed action matrix."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiment3.phase0_ohj_cable.common import (  # noqa: E402
    load_config, pair_dir, report_dir, write_json)


def classify_control(matrix, config):
    analysis = config["analysis"]
    commands_equal = all(
        matrix["free"][candidate][key]
        == matrix["jam_right"][candidate][key]
        for candidate in matrix["free"]
        for key in ("command_phase", "command_ee_target", "command_joint_target"))
    rows = {}
    for condition, candidates in matrix.items():
        rows[condition] = {}
        for name, value in candidates.items():
            success = bool(
                value["final_progress_m"] >= analysis["min_extraction_progress_m"]
                and value["grasp_retained"]
                and value["peak_force_n"] <=
                analysis["max_bruteforce_wrench_force_n"])
            rows[condition][name] = {
                "success": success,
                "final_progress_m": float(value["final_progress_m"]),
                "peak_force_n": float(value["peak_force_n"]),
                "path_length_m": float(value["path_length_m"]),
                "grasp_retained": bool(value["grasp_retained"]),
            }
    best = {}
    for condition in rows:
        best[condition] = max(rows[condition], key=lambda name: (
            rows[condition][name]["success"],
            -rows[condition][name]["peak_force_n"],
            -rows[condition][name]["path_length_m"]))
    free_straight = rows["free"]["straight"]
    jam_straight = rows["jam_right"]["straight"]
    jam_left = rows["jam_right"]["left_release"]
    relevant = bool(
        free_straight["success"]
        and (not jam_straight["success"]
             or jam_straight["final_progress_m"] <=
             jam_left["final_progress_m"] - 0.015)
        and (jam_left["success"]
             or jam_left["final_progress_m"] >=
             jam_straight["final_progress_m"] + 0.015))
    return {
        "verdict": ("PHASE0D_CONTROL_RELEVANCE_COMPLETE" if relevant
                    else "PHASE0D_CONTROL_NOT_RELEVANT"),
        "control_relevant": relevant,
        "same_candidate_commands_across_branches": commands_equal,
        "best_candidate": best,
        "matrix": rows,
        "jam_left_progress_advantage_m": (
            jam_left["final_progress_m"] - jam_straight["final_progress_m"]),
    }


def analyze(config_path):
    config = load_config(config_path)
    source = json.loads((pair_dir(config) / "control_relevance.json").read_text(
        encoding="utf-8"))
    metrics = classify_control(source["matrix"], config)
    write_json(report_dir(config) / "control_relevance_metrics.json", metrics)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print("verdict={}".format(analyze(args.config)["verdict"]))


if __name__ == "__main__":
    main()
