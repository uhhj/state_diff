"""Classify control relevance from the fixed condition-by-action matrix."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiment3.phase0c_hidden_dynamics.common import (  # noqa: E402
    load_config, report_pair_dir)
from scripts.experiment3.phase0c_hidden_dynamics.io_utils import write_json  # noqa: E402


def classify_control_relevance(matrix: dict, threshold_m: float) -> dict:
    """Apply the three predeclared control-relevance criteria."""
    low = matrix["uniform_low"]
    high = matrix["right_local_high"]
    candidates = list(low)
    best_low = max(candidates, key=lambda key: low[key]["progress_m"])
    best_high = max(candidates, key=lambda key: high[key]["progress_m"])
    regret_low = float(low[best_low]["progress_m"] - low[best_high]["progress_m"])
    regret_high = float(high[best_high]["progress_m"] - high[best_low]["progress_m"])
    success_flip = any(bool(low[key]["success"]) != bool(high[key]["success"])
                       for key in candidates)
    progress_gaps = {key: abs(float(low[key]["progress_m"])
                              - float(high[key]["progress_m"]))
                     for key in candidates}
    different_best_regret = (best_low != best_high
                             and max(regret_low, regret_high) >= threshold_m)
    candidate_gap = max(progress_gaps.values()) >= threshold_m
    relevant = different_best_regret or success_flip or candidate_gap
    return {
        "verdict": ("PHASE0C_CONTROL_RELEVANCE_COMPLETE" if relevant
                    else "PHASE0C_CONTROL_NOT_RELEVANT"),
        "control_relevant": relevant, "candidates": candidates,
        "best_candidate_uniform_low": best_low,
        "best_candidate_right_local_high": best_high,
        "progress_matrix_m": {
            condition: {key: float(value[key]["progress_m"]) for key in candidates}
            for condition, value in (("uniform_low", low),
                                     ("right_local_high", high))},
        "success_matrix": {
            condition: {key: bool(value[key]["success"]) for key in candidates}
            for condition, value in (("uniform_low", low),
                                     ("right_local_high", high))},
        "cross_condition_regret_low_m": regret_low,
        "cross_condition_regret_high_m": regret_high,
        "candidate_progress_gap_m": progress_gaps,
        "criteria": {"different_best_with_regret": different_best_regret,
                     "condition_specific_success_flip": success_flip,
                     "candidate_progress_gap": candidate_gap}}


def analyze(config_path: str) -> dict:
    config = load_config(config_path)
    source = Path(config["output_root"]) / config["pair_id"] / "control_relevance.json"
    matrix = json.loads(source.read_text(encoding="utf-8"))["matrix"]
    metrics = classify_control_relevance(
        matrix, float(config["analysis"]["control_progress_regret_min_m"]))
    write_json(report_pair_dir(config) / "control_relevance_metrics.json", metrics)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print("verdict={}".format(analyze(args.config)["verdict"]))


if __name__ == "__main__":
    main()
