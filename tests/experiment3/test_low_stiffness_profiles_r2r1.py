import copy
import json
from pathlib import Path

import pytest

from scripts.experiment3.phase0_soft_blockpush_r2r1.common import (
    PROFILE_NAMES, load_config, load_profile)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/experiment3/soft_blockpush_phase0b_r2r1.json"
PROFILE_DIR = ROOT / "configs/experiment3/material_profiles_r2r1"


def test_all_low_stiffness_coefficients_recompute_exactly():
    config = load_config(str(CONFIG))
    for multiplier, name in PROFILE_NAMES.items():
        path = PROFILE_DIR / ("soft_block_" + name + ".json")
        material, payload = load_profile(str(path), config)
        assert payload["stiffness_multiplier"] == multiplier
        assert material.structural_stiffness_n_per_m == 8 * multiplier
        assert material.shear_stiffness_n_per_m == 3 * multiplier
        assert material.bending_stiffness_n_per_m == pytest.approx(.8 * multiplier)
        assert payload["damping_ratio"] == .25


def test_profile_mismatch_unknown_name_and_path_are_rejected(tmp_path):
    config = load_config(str(CONFIG))
    source = json.loads((PROFILE_DIR / "soft_block_kv_r2r1_a1p25_z025.json").read_text())
    cases = []
    mismatch = copy.deepcopy(source); mismatch["structural_damping_ns_per_m"] += 2e-12
    cases.append(("soft_block_kv_r2r1_a1p25_z025.json", mismatch))
    unknown = copy.deepcopy(source); unknown["stiffness_multiplier"] = 2.0
    cases.append(("soft_block_kv_r2r1_a2p00_z025.json", unknown))
    wrong_name = copy.deepcopy(source); wrong_name["profile_name"] = "wrong"
    cases.append(("soft_block_wrong.json", wrong_name))
    for filename, payload in cases:
        path = tmp_path / filename; path.write_text(json.dumps(payload))
        with pytest.raises(ValueError):
            load_profile(str(path), config)


def test_legacy_r2_profiles_keep_their_frozen_multipliers():
    directory = ROOT / "configs/experiment3/material_profiles_r2"
    expected = {"a4": 4.0, "a6": 6.0, "a8": 8.0}
    for suffix, multiplier in expected.items():
        path = directory / "soft_block_kv_r2_{}_z025.json".format(suffix)
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["stiffness_multiplier"] == multiplier
        assert payload["base_stiffness_n_per_m"] == {
            "structural": 8.0, "shear": 3.0, "bending": .8}
