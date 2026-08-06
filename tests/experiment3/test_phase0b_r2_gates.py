from pathlib import Path

from scripts.experiment3.phase0_soft_blockpush_r2.analyze_table_settle import (
    classify_table)
from scripts.experiment3.phase0_soft_blockpush_r2.common import (
    load_config, load_selected_microsteps, select_next_profile, write_json)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/experiment3/soft_blockpush_phase0b_r2.json"


def test_phase_config_is_frozen_and_selection_requires_reference(tmp_path):
    config = load_config(str(CONFIG))
    assert config["physics"]["microstep_candidates"] == [4, 8, 16, 32]
    assert config["soft_block"]["grid_shape"] == [6, 4, 3]
    path = tmp_path / "selection.json"
    write_json(path, {"verdict": "PHASE0B_R2_MICROSTEP_VALIDATION_COMPLETE",
                      "microsteps_per_outer": 16})
    assert load_selected_microsteps(str(path))["microsteps_per_outer"] == 16
    write_json(path, {"verdict": "PHASE0B_R2_MICROSTEP_VALIDATION_COMPLETE",
                      "microsteps_per_outer": 32})
    try:
        load_selected_microsteps(str(path))
    except ValueError:
        pass
    else:
        raise AssertionError("32 cannot be frozen without a 2M reference")


def test_table_and_profile_gates_are_closed_world():
    assert classify_table(False, True).endswith("ENGINEERING_BLOCKED")
    assert classify_table(True, False).endswith("UNSTABLE")
    assert classify_table(True, True).endswith("COMPLETE")
    assert select_next_profile("kv_r2_a4_z025", "PHASE0B_R2_AXIAL_TOO_SOFT", 1) is None
    assert select_next_profile("kv_r2_a6_z025", "PHASE0B_R2_AXIAL_COMPLETE", 1) is None
