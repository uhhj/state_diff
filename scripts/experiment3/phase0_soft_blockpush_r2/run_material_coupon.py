"""Run one selected R2 profile in axial or shear free space."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiment3.phase0_soft_blockpush_r2.common import (  # noqa: E402
    load_config, load_profile, load_selected_microsteps, write_json)
from state_diff.env.block_pushing.material_coupon_r2 import (  # noqa: E402
    COUPON_MODES, MaterialCouponR2)


def run_coupon(config_path: str, profile_path: str, selection_path: str,
               mode: str) -> Path:
    """Run and save one outer-step-only R2 coupon trajectory."""
    config = load_config(config_path); _, profile = load_profile(profile_path)
    selection = load_selected_microsteps(selection_path)
    output = (Path(config["output_root"]) / "coupon" /
              profile["profile_name"] / mode)
    output.mkdir(parents=True, exist_ok=True)
    coupon = MaterialCouponR2(
        config, profile_path, selection["microsteps_per_outer"], mode)
    height, width = config["camera"]["image_size"]
    video = cv2.VideoWriter(str(output / "coupon.mp4"),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            config["camera"]["fps"], (width, height))
    if not video.isOpened():
        coupon.close(); raise RuntimeError("unable to write coupon video")

    def frame(step: int, phase: str) -> None:
        del phase
        if step % config["camera"]["frame_stride"] == 0:
            video.write(cv2.cvtColor(coupon.render(width, height),
                                     cv2.COLOR_RGB2BGR))
    try:
        trace = coupon.run(frame)
        fields = tuple(trace[0])
        arrays = {field: np.asarray([row[field] for row in trace]) for field in fields}
        np.savez_compressed(output / "trajectory.npz", **arrays)
        write_json(output / "metadata.json", {
            "mode": mode, "config": config, "material_profile": profile,
            "material_profile_path": profile_path,
            "selected_microsteps_path": selection_path,
            "microstep_selection": selection,
            "anchor_mode": "static_nodes", "fixture_constraint_count": 0,
            "gravity": [0, 0, 0], "floor_loaded": False,
            "node_count": len(coupon.block.body_ids),
            "anchor_indices": coupon.anchor_indices.tolist(),
            "load_indices": coupon.load_indices.tolist(),
            "static_anchor_masses": [coupon.client.getDynamicsInfo(
                coupon.block.body_ids[int(i)], -1)[0] for i in coupon.anchor_indices],
            "dynamic_node_mass": coupon.client.getDynamicsInfo(
                coupon.block.body_ids[int(coupon.load_indices[0])], -1)[0],
            "structural_edges": coupon.block.edges["structural"].tolist(),
            "structural_rest_lengths": [edge["rest_length"] for edge in
                                        coupon.block.edge_metadata["structural"]]})
    finally:
        video.release(); coupon.close()
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--material-profile", required=True)
    parser.add_argument("--selected-microsteps", required=True)
    parser.add_argument("--mode", choices=COUPON_MODES, required=True)
    args = parser.parse_args()
    print("coupon_dir={}".format(run_coupon(
        args.config, args.material_profile, args.selected_microsteps, args.mode)))


if __name__ == "__main__":
    main()
