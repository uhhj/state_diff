import copy
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.experiment3.phase0_soft_blockpush_r2.common import (
    load_selected_microsteps)
from scripts.experiment3.phase0_soft_blockpush_r2r1 import (
    analyze_family_confirmation as analyzer,
    run_family_confirmation as runner)
from scripts.experiment3.phase0_soft_blockpush_r2r1.common import load_config


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/experiment3/soft_blockpush_phase0b_r2r1.json"
PROFILES = ROOT / "configs/experiment3/material_profiles_r2r1"


def _summary(value=1.0, stable=True, zero_cap=True):
    return {"stable": stable, "gate": {"zero_cap": zero_cap},
            "peak_extension_m": value, "first_zero_crossing_time_s": value,
            "final_energy_ratio": value,
            "peak_primary_face_displacement_m": value,
            "peak_rigid_aligned_rmse_m": value, "recovery_ratio": value}


def test_confirmation_pass_failure_cap_and_no_m16_selection():
    config = load_config(str(CONFIG))
    two = {8: _summary(1.0), 16: _summary(1.01)}
    cube = copy.deepcopy(two)
    passed = analyzer.evaluate_confirmation(two, cube, config)
    assert passed["verdict"].endswith("COMPLETE")
    assert passed["retained_microsteps"] == 8 and not passed["selection_reopened"]
    two[8]["peak_extension_m"] = 1.2
    assert analyzer.evaluate_confirmation(two, cube, config)["verdict"].endswith("BLOCKED")
    two[8] = _summary(1.0, zero_cap=False)
    blocked = analyzer.evaluate_confirmation(two, cube, config)
    assert blocked["verdict"].endswith("BLOCKED") and blocked["retained_microsteps"] is None


def test_runner_requires_a1p5_prior_m8_and_runs_only_8_16(tmp_path, monkeypatch):
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["output_root"] = str(tmp_path / "data")
    prior = tmp_path / "prior.json"; config["prior_r2_selection_path"] = str(prior)
    config_path = tmp_path / "config.json"; config_path.write_text(json.dumps(config))
    prior.write_text(json.dumps({"verdict": "PHASE0B_R2_MICROSTEP_VALIDATION_COMPLETE",
                                 "microsteps_per_outer": 8}))
    calls = []
    monkeypatch.setattr(runner, "run_two_node_validation",
                        lambda config, material, m: calls.append(("two", m)) or {"x": np.array([m])})
    monkeypatch.setattr(runner, "run_cube_validation",
                        lambda config, material, m, cube: calls.append(("cube", m)) or {"x": np.array([m])})
    runner.run_confirmation(str(config_path), str(
        PROFILES / "soft_block_kv_r2r1_a1p50_z025.json"))
    assert calls == [("two", 8), ("cube", 8), ("two", 16), ("cube", 16)]
    with pytest.raises(ValueError, match="A1.5"):
        runner.run_confirmation(str(config_path), str(
            PROFILES / "soft_block_kv_r2r1_a1p25_z025.json"))
    prior.write_text(json.dumps({"verdict": "PHASE0B_R2_MICROSTEP_VALIDATION_COMPLETE",
                                 "microsteps_per_outer": 16}))
    with pytest.raises(ValueError, match="prior R2 M=8"):
        runner.run_confirmation(str(config_path), str(
            PROFILES / "soft_block_kv_r2r1_a1p50_z025.json"))


def test_analyzer_writes_r2_compatible_non_reopened_selection(tmp_path, monkeypatch):
    config = load_config(str(CONFIG)); config = copy.deepcopy(config)
    confirmation = tmp_path / "confirmation"; report = tmp_path / "report"
    confirmation.mkdir()
    profile = json.loads((PROFILES / "soft_block_kv_r2r1_a1p50_z025.json").read_text())
    (confirmation / "metadata.json").write_text(json.dumps({
        "config": config, "material_profile": profile,
        "confirmation_microsteps": [8, 16],
        "prior_r2_selection": {"microsteps_per_outer": 8}}))
    for mode in ("two_node", "cube"):
        for microsteps in (8, 16):
            np.savez_compressed(confirmation / "{}_{}.npz".format(mode, microsteps), x=[1])
    monkeypatch.setattr(analyzer, "summarize_two_node",
                        lambda data, config, m: _summary(1 + m / 1000))
    monkeypatch.setattr(analyzer, "summarize_cube",
                        lambda data, config, m: _summary(1 + m / 1000))
    result = analyzer.analyze(confirmation, report)
    assert result["verdict"].endswith("COMPLETE")
    selection = load_selected_microsteps(str(report / "selected_microsteps.json"))
    assert selection["microsteps_per_outer"] == 8
    assert selection["selection_reopened"] is False
