"""Run only the A1.5 M8/M16 low-stiffness family confirmation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiment3.phase0_soft_blockpush_r2.common import soft_block_config  # noqa: E402
from scripts.experiment3.phase0_soft_blockpush_r2r1.common import (  # noqa: E402
    STIFFER_PROFILE, load_config, load_profile, write_json)
from state_diff.env.block_pushing.mechanics_validation import (  # noqa: E402
    run_cube_validation, run_two_node_validation)


def run_confirmation(config_path: str, profile_path: str) -> Path:
    """Generate A1.5 mechanics traces for exactly M8 and M16."""
    config = load_config(config_path)
    material, profile = load_profile(profile_path, config)
    if profile["profile_name"] != STIFFER_PROFILE:
        raise ValueError("confirmation must use A1.5")
    prior_path = Path(config["prior_r2_selection_path"])
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    if (prior.get("verdict") != "PHASE0B_R2_MICROSTEP_VALIDATION_COMPLETE"
            or prior.get("microsteps_per_outer") != 8):
        raise ValueError("prior R2 M=8 selection is unavailable")
    output = Path(config["output_root"]) / "family_confirmation"
    output.mkdir(parents=True, exist_ok=True)
    cube_config = soft_block_config(config, (2, 2, 2))
    candidates = config["physics"]["family_confirmation_microsteps"]
    for microsteps in candidates:
        np.savez_compressed(output / "two_node_{}.npz".format(microsteps),
                            **run_two_node_validation(config, material, microsteps))
        np.savez_compressed(output / "cube_{}.npz".format(microsteps),
                            **run_cube_validation(config, material, microsteps,
                                                  cube_config))
    write_json(output / "metadata.json", {
        "config": config, "material_profile": profile,
        "material_profile_path": profile_path,
        "confirmation_microsteps": candidates,
        "prior_r2_selection": prior, "selection_is_reopened": False})
    return output


def main() -> None:
    """Run the family confirmation CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--material-profile", required=True)
    args = parser.parse_args()
    print("confirmation_dir={}".format(run_confirmation(
        args.config, args.material_profile)))


if __name__ == "__main__":
    main()
