import copy
import json
from pathlib import Path

import pytest

from scripts.experiment3.phase0_soft_blockpush_r2r2.common import (
    PROFILE_SHEAR, load_config, load_profile)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/experiment3/soft_blockpush_phase0b_r2r2.json"
PROFILES = ROOT / "configs/experiment3/material_profiles_r2r2"


def test_three_decoupled_profiles_and_m8_config_load():
    config = load_config(str(CONFIG))
    assert config["physics"]["frozen_microsteps_per_outer"] == 8
    for name, shear in PROFILE_SHEAR.items():
        material, payload = load_profile(
            str(PROFILES / ("soft_block_" + name + ".json")), config)
        assert material.structural_stiffness_n_per_m == 10
        assert material.shear_stiffness_n_per_m == shear
        assert material.bending_stiffness_n_per_m == 1
        assert payload["stiffness_multiplier"] == 1


def test_path_coefficient_and_old_family_are_rejected(tmp_path):
    config = load_config(str(CONFIG))
    source = json.loads((PROFILES / "soft_block_kv_r2r2_s12p5_z025.json").read_text())
    mismatch = copy.deepcopy(source); mismatch["shear_damping_ns_per_m"] += 2e-12
    path = tmp_path / "soft_block_kv_r2r2_s12p5_z025.json"
    path.write_text(json.dumps(mismatch))
    with pytest.raises(ValueError):
        load_profile(str(path), config)
    wrong_path = tmp_path / "wrong.json"; wrong_path.write_text(json.dumps(source))
    with pytest.raises(ValueError, match="path/name"):
        load_profile(str(wrong_path), config)
    old = ROOT / "configs/experiment3/material_profiles_r2r1/soft_block_kv_r2r1_a1p25_z025.json"
    with pytest.raises(ValueError, match="outside R2-R2"):
        load_profile(str(old), config)
