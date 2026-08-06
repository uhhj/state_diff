"""Run the R3 composite material gravity-and-floor audit."""
import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiment3.phase0_soft_blockpush_r3.common import (  # noqa: E402
    load_config, load_profile, load_selected_microsteps, write_json)
from state_diff.env.block_pushing.angle_elastic_coupon import (  # noqa: E402
    AngleElasticTableSettleCoupon)


def run_settle(config_path: str, profile_path: str, selection_path: str) -> Path:
    config = load_config(config_path); _, _, profile = load_profile(profile_path)
    selection = load_selected_microsteps(selection_path)
    output = Path(config["output_root"]) / "table_settle" / profile["profile_name"]
    output.mkdir(parents=True, exist_ok=True)
    coupon = AngleElasticTableSettleCoupon(
        config, profile_path, selection["microsteps_per_outer"])
    height, width = config["camera"]["image_size"]
    video = cv2.VideoWriter(str(output / "table_settle.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"), config["camera"]["fps"], (width, height))
    if not video.isOpened():
        coupon.close(); raise RuntimeError("unable to write table-settle video")
    def frame(step, phase):
        del phase
        if step % config["camera"]["frame_stride"] == 0:
            video.write(cv2.cvtColor(coupon.render(width, height), cv2.COLOR_RGB2BGR))
    try:
        trace = coupon.run(frame)
        np.savez_compressed(output / "trajectory.npz", **{
            field: np.asarray([row[field] for row in trace]) for field in trace[0]})
        block = coupon.block
        write_json(output / "metadata.json", {
            "config": config, "material_profile": profile,
            "material_profile_path": profile_path,
            "selected_microsteps_path": selection_path,
            "microstep_selection": selection, "node_count": len(block.body_ids),
            "all_nodes_dynamic": all(coupon.client.getDynamicsInfo(body, -1)[0] > 0
                                     for body in block.body_ids),
            "node_mass_kg": coupon.client.getDynamicsInfo(block.body_ids[0], -1)[0],
            "floor_body_count": 1, "floor_id": coupon.floor_id,
            "fixture_constraint_count": 0, "external_load": False,
            "robot_loaded": False, "target_loaded": False,
            "recenter_calls": 0, "velocity_reset_calls": 0,
            "structural_edges": block.edges["structural"].tolist(),
            "structural_rest_lengths": [edge["rest_length"] for edge in
                                        block.edge_metadata["structural"]],
            "internal_evaluations_per_microstep":
                block.internal_evaluations_per_microstep,
            "angle_constraint_count": len(block.angle_triplets),
            "angle_planes": block.angle_plane_counts})
    finally:
        video.release(); coupon.close()
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--material-profile", required=True)
    parser.add_argument("--selected-microsteps", required=True)
    args = parser.parse_args()
    print("settle_dir={}".format(run_settle(
        args.config, args.material_profile, args.selected_microsteps)))


if __name__ == "__main__": main()
