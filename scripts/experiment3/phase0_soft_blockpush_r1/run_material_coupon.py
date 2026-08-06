"""Run one independently configured Kelvin-Voigt material coupon."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiment3.phase0_soft_blockpush_r1.common import (  # noqa: E402
    SPRING_FIELDS, load_config, trace_arrays, with_material_profile, write_json)
from state_diff.env.block_pushing.material_coupon import MaterialCoupon  # noqa: E402
from state_diff.env.block_pushing.kelvin_voigt_soft_block import (  # noqa: E402
    KELVIN_VOIGT_BULLET_SUBSTEPS)


FIELDS = ("physics_step", "phase", "node_positions", "node_velocities",
          "visible_keypoints", *SPRING_FIELDS, "external_load_scale",
          "external_load_force_n")


def run_coupon(config_path: str, material_profile_path: str) -> Path:
    """Execute a coupon and persist trajectory, metadata, trace, and video."""
    config = with_material_profile(load_config(config_path), material_profile_path)
    profile = config["material_profile"]
    output = Path(config["output_root"]) / "coupon" / profile["profile_name"]
    output.mkdir(parents=True, exist_ok=True)
    coupon = MaterialCoupon(config, material_profile_path)
    height, width = config["camera"]["image_size"]
    video_path = output / "coupon.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             float(config["camera"]["fps"]),
                             (int(width), int(height)))
    if not writer.isOpened():
        coupon.close()
        raise RuntimeError("unable to create coupon video")

    def record(step: int, phase: str) -> None:
        del phase
        if step % int(config["camera"]["frame_stride"]) == 0:
            frame = coupon.render(int(width), int(height))
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    try:
        trace = coupon.run(record)
        arrays = trace_arrays(trace, FIELDS)
        np.savez_compressed(output / "trajectory.npz", **arrays)
        write_json(output / "trace.json", trace)
        metadata = {
            "coupon_fixture_only": True,
            "material_model": "kelvin_voigt",
            "material_profile": profile,
            "material_profile_path": str(material_profile_path),
            "anchor_indices": coupon.anchor_indices.tolist(),
            "load_indices": coupon.load_indices.tolist(),
            "fixture_constraint_ids": coupon.fixture_constraint_ids,
            "internal_constraint_count": len(coupon.block.constraint_ids),
            "bullet_internal_substeps": KELVIN_VOIGT_BULLET_SUBSTEPS,
            "node_count": len(coupon.block.body_ids),
            "top_indices": coupon.block.top_indices.tolist(),
            "structural_edges": coupon.block.edges["structural"].tolist(),
            "structural_rest_lengths": [e["rest_length"] for e in
                                        coupon.block.edge_metadata["structural"]],
            "shear_edges": coupon.block.edges["shear"].tolist(),
            "shear_rest_lengths": [e["rest_length"] for e in
                                   coupon.block.edge_metadata["shear"]],
            "config": config,
            "video_path": str(video_path),
        }
        write_json(output / "metadata.json", metadata)
    finally:
        writer.release()
        coupon.close()
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--material-profile", required=True)
    args = parser.parse_args()
    print("coupon_dir={}".format(run_coupon(args.config, args.material_profile)))


if __name__ == "__main__":
    main()
