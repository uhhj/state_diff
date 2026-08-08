from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from scripts.experiment3.dlolab_wiring_post.features import (
    hidden_difference,
    history_observable_distance,
    oracle_descriptor,
    symmetric_chamfer,
    visible_points,
)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def future_distance(rope_a, rope_b, post_xy, occlusion_radius_m):
    a = visible_points(rope_a, post_xy, occlusion_radius_m)
    b = visible_points(rope_b, post_xy, occlusion_radius_m)
    return symmetric_chamfer(a, b)


def mine(config_path, dataset_path):
    config = load_json(config_path)
    data = np.load(dataset_path)
    rope = data["rope_xyz"]
    ee_pos = data["ee_pos"]
    post_xyz = data["post_xyz"]
    n_rollouts = rope.shape[0]
    n_time = rope.shape[1]
    obs = config["observation_protocol"]
    oracle = config["oracle_pair_descriptor"]
    mining = config["pair_mining"]
    history = int(obs["history_samples"])
    future_horizon = int(mining["future_horizon_samples"])
    candidates = []

    for time_index in range(history - 1, n_time - future_horizon):
        for first, second in combinations(range(n_rollouts), 2):
            post_xy = 0.5 * (
                post_xyz[first, time_index, :, :2]
                + post_xyz[second, time_index, :, :2])
            start = time_index - history + 1
            visible_distance, ee_distance = history_observable_distance(
                rope[first, start:time_index + 1],
                rope[second, start:time_index + 1],
                ee_pos[first, start:time_index + 1],
                ee_pos[second, start:time_index + 1],
                post_xy,
                obs["post_occlusion_radius_m"],
            )
            if visible_distance > float(
                    mining["max_observable_chamfer_m"]):
                continue
            if ee_distance > float(
                    mining["max_ee_position_distance_m"]):
                continue

            descriptor_a = oracle_descriptor(
                rope[first, time_index],
                post_xy,
                rope_radius_m=oracle["rope_radius_m"],
                post_radius_m=oracle["post_radius_m"],
                roi_radius_m=oracle["post_roi_radius_m"],
                contact_proxy_margin_m=oracle[
                    "contact_proxy_margin_m"],
            )
            descriptor_b = oracle_descriptor(
                rope[second, time_index],
                post_xy,
                rope_radius_m=oracle["rope_radius_m"],
                post_radius_m=oracle["post_radius_m"],
                roi_radius_m=oracle["post_roi_radius_m"],
                contact_proxy_margin_m=oracle[
                    "contact_proxy_margin_m"],
            )
            hidden = hidden_difference(
                descriptor_a,
                descriptor_b,
                oracle["wrap_difference_min_rad"],
            )
            if not hidden["different"]:
                continue

            future = future_distance(
                rope[first, time_index + future_horizon],
                rope[second, time_index + future_horizon],
                post_xy,
                obs["post_occlusion_radius_m"],
            )
            denominator = max(visible_distance, 1e-5)
            ratio = float(future / denominator)
            if ratio < float(
                    mining["min_future_to_observable_ratio"]):
                continue
            wrap_strength = max(
                hidden["wrap_difference_rad"], default=0.0)
            score = float(
                (1.0 + wrap_strength
                 + (1.0 if hidden["contact_like_mismatch"] else 0.0))
                * future / denominator)
            candidates.append({
                "rollout_a": int(first),
                "rollout_b": int(second),
                "time_index": int(time_index),
                "future_index": int(time_index + future_horizon),
                "visible_history_chamfer_m": float(visible_distance),
                "ee_position_distance_m": float(ee_distance),
                "future_visible_chamfer_m": float(future),
                "future_to_observable_ratio": ratio,
                "hidden_difference": hidden,
                "descriptor_a": descriptor_a,
                "descriptor_b": descriptor_b,
                "score": score,
            })

    candidates.sort(key=lambda row: row["score"], reverse=True)
    result = {
        "dataset": str(dataset_path),
        "num_rollouts": int(n_rollouts),
        "time_samples": int(n_time),
        "candidate_count": int(len(candidates)),
        "top_candidates": candidates[:int(mining["top_k"])],
    }
    raw_root = Path(config["outputs"]["raw_root"])
    output = raw_root / (
        "PAIR_CANDIDATES.json"
        if "scaleup" not in str(dataset_path)
        else "PAIR_CANDIDATES_SCALEUP.json")
    output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    print("pairs={}".format(mine(args.config, args.dataset)))


if __name__ == "__main__":
    main()
