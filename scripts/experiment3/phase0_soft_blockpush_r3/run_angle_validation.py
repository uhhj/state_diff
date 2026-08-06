"""Run the A070 three-node angle confirmation at M8 and M16."""
import argparse
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiment3.phase0_soft_blockpush_r3.common import (  # noqa: E402
    load_config, load_profile, write_json)
from state_diff.env.block_pushing.angle_cell_validation import (  # noqa: E402
    run_angle_cell_validation)


def run(config_path: str, profile_path: str) -> Path:
    config = load_config(config_path)
    _, parameters, profile = load_profile(profile_path)
    if profile["profile_name"] != "kv_angle_r3_a070_z025":
        raise ValueError("angle validation requires A070")
    root = Path(config["output_root"]) / "angle_validation"
    root.mkdir(parents=True, exist_ok=True)
    for microsteps in config["physics"]["angle_confirmation_microsteps"]:
        output = root / "M{}".format(microsteps)
        output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output / "trajectory.npz",
            **run_angle_cell_validation(config, parameters, microsteps))
        write_json(output / "metadata.json", {
            "config": config, "material_profile": profile,
            "microsteps_per_outer": microsteps, "node_count": 3,
            "center_static": True, "gravity": [0, 0, 0],
            "collision_shapes": 0, "constraint_count": 1})
    return root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--material-profile", required=True)
    args = parser.parse_args()
    print("angle_validation_dir={}".format(run(args.config, args.material_profile)))


if __name__ == "__main__":
    main()
