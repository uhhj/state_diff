"""Run A8 mechanics validation for every frozen microstep candidate."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiment3.phase0_soft_blockpush_r2.common import (  # noqa: E402
    load_config, load_profile, soft_block_config, write_json)
from state_diff.env.block_pushing.mechanics_validation import (  # noqa: E402
    run_cube_validation, run_two_node_validation)


def run_validation(config_path: str, profile_path: str) -> Path:
    """Run two-node and cube scenes for M=4/8/16/32."""
    config = load_config(config_path); material, profile = load_profile(profile_path)
    if profile["profile_name"] != "kv_r2_a8_z025":
        raise ValueError("mechanics validation must use A8")
    output = Path(config["output_root"]) / "mechanics_validation"
    output.mkdir(parents=True, exist_ok=True)
    cube_config = soft_block_config(config, (2, 2, 2))
    for microsteps in config["physics"]["microstep_candidates"]:
        np.savez_compressed(output / "two_node_{}.npz".format(microsteps),
                            **run_two_node_validation(config, material, microsteps))
        np.savez_compressed(output / "cube_{}.npz".format(microsteps),
                            **run_cube_validation(
                                config, material, microsteps, cube_config))
    write_json(output / "metadata.json", {
        "config": config, "material_profile": profile,
        "material_profile_path": profile_path,
        "microstep_candidates": config["physics"]["microstep_candidates"]})
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--material-profile", required=True)
    args = parser.parse_args()
    print("validation_dir={}".format(
        run_validation(args.config, args.material_profile)))


if __name__ == "__main__":
    main()
