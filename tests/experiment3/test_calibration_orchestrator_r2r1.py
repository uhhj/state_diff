import copy
import json
from pathlib import Path

import pytest

from scripts.experiment3.phase0_soft_blockpush_r2r1 import run_calibration as orchestration


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/experiment3/soft_blockpush_phase0b_r2r1.json"


def _setup(tmp_path):
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["report_root"] = str(tmp_path / "reports")
    config["output_root"] = str(tmp_path / "data")
    config_path = tmp_path / "config.json"; config_path.write_text(json.dumps(config))
    mechanics = Path(config["report_root"]) / "mechanics_validation"
    mechanics.mkdir(parents=True)
    (mechanics / "summary.json").write_text(json.dumps({
        "verdict": "PHASE0B_R2R1_FAMILY_CONFIRMATION_COMPLETE"}))
    selection = mechanics / "selected_microsteps.json"
    selection.write_text(json.dumps({
        "verdict": "PHASE0B_R2_MICROSTEP_VALIDATION_COMPLETE",
        "confirmation_verdict": "PHASE0B_R2R1_FAMILY_CONFIRMATION_COMPLETE",
        "microsteps_per_outer": 8, "selection_reopened": False}))
    return config_path, selection


def _run(tmp_path, monkeypatch, verdicts, table_complete=True):
    config_path, selection = _setup(tmp_path)
    calls, queue = [], list(verdicts)
    def coupon(config, profile, selected, mode):
        del config, selected
        name = Path(profile).stem.replace("soft_block_", "")
        calls.append((name, mode))
        verdict = queue.pop(0)
        return {"verdict": verdict, "profile_name": name,
                "peak_primary_displacement_m": .003}
    monkeypatch.setattr(orchestration, "execute_coupon", coupon)
    monkeypatch.setattr(orchestration, "run_settle",
                        lambda *args: tmp_path / "settle")
    table_verdict = ("PHASE0B_R2_TABLE_SETTLE_COMPLETE" if table_complete
                     else "PHASE0B_R2_TABLE_SETTLE_UNSTABLE")
    monkeypatch.setattr(orchestration, "analyze_table",
                        lambda *args: {"verdict": table_verdict,
                                       "frozen_outputs": ({"material": "x"}
                                                          if table_complete else {})})
    result = orchestration.run_calibration(
        str(config_path), str(tmp_path / "profiles"), str(selection))
    return result, calls


@pytest.mark.parametrize("first_failure,replacement", [
    ("PHASE0B_R2_AXIAL_TOO_STIFF", "kv_r2r1_a1p00_z025"),
    ("PHASE0B_R2_AXIAL_TOO_SOFT", "kv_r2r1_a1p50_z025")])
def test_axial_neighbor_gets_full_axial_shear_rerun(tmp_path, monkeypatch,
                                                    first_failure, replacement):
    result, calls = _run(tmp_path, monkeypatch, [
        first_failure, "PHASE0B_R2_AXIAL_COMPLETE", "PHASE0B_R2_SHEAR_COMPLETE"])
    assert result["verdict"].endswith("MATERIAL_CALIBRATION_COMPLETE")
    assert calls == [("kv_r2r1_a1p25_z025", "axial"),
                     (replacement, "axial"), (replacement, "shear")]


@pytest.mark.parametrize("shear_failure,replacement", [
    ("PHASE0B_R2_SHEAR_TOO_STIFF", "kv_r2r1_a1p00_z025"),
    ("PHASE0B_R2_SHEAR_TOO_SOFT", "kv_r2r1_a1p50_z025")])
def test_shear_neighbor_restarts_at_axial(tmp_path, monkeypatch,
                                          shear_failure, replacement):
    result, calls = _run(tmp_path, monkeypatch, [
        "PHASE0B_R2_AXIAL_COMPLETE", shear_failure,
        "PHASE0B_R2_AXIAL_COMPLETE", "PHASE0B_R2_SHEAR_COMPLETE"])
    assert result["verdict"].endswith("MATERIAL_CALIBRATION_COMPLETE")
    assert calls[-2:] == [(replacement, "axial"), (replacement, "shear")]
    assert len(result["profiles_run"]) == 2


def test_start_profile_complete_and_complete_freeze_flags(tmp_path, monkeypatch):
    result, calls = _run(tmp_path, monkeypatch, [
        "PHASE0B_R2_AXIAL_COMPLETE", "PHASE0B_R2_SHEAR_COMPLETE"])
    assert calls == [("kv_r2r1_a1p25_z025", "axial"),
                     ("kv_r2r1_a1p25_z025", "shear")]
    assert result["table_settle"]["frozen_outputs"]
    assert result["pair_run"] is False and result["training_run"] is False


def test_unstable_stops_without_profile_switch_or_table(tmp_path, monkeypatch):
    result, calls = _run(tmp_path, monkeypatch, ["PHASE0B_R2_AXIAL_UNSTABLE"])
    assert result["verdict"] == "PHASE0B_R2R1_NO_PROFILE_PASSED"
    assert len(calls) == 1 and result["table_settle"] is None


def test_second_profile_failure_enforces_two_profile_limit(tmp_path, monkeypatch):
    result, calls = _run(tmp_path, monkeypatch, [
        "PHASE0B_R2_AXIAL_TOO_STIFF", "PHASE0B_R2_AXIAL_TOO_SOFT"])
    assert result["verdict"] == "PHASE0B_R2R1_NO_PROFILE_PASSED"
    assert len(result["profiles_run"]) == len(calls) == 2


def test_table_failure_does_not_report_frozen_outputs(tmp_path, monkeypatch):
    result, _ = _run(tmp_path, monkeypatch, [
        "PHASE0B_R2_AXIAL_COMPLETE", "PHASE0B_R2_SHEAR_COMPLETE"],
        table_complete=False)
    assert result["verdict"] == "PHASE0B_R2R1_TABLE_SETTLE_FAILED"
    assert result["table_settle"]["frozen_outputs"] == {}
    assert result["pair_run"] is False and result["training_run"] is False
