import json
from pathlib import Path

import pytest

from scripts.experiment3.phase0_soft_blockpush_r2r2 import run_calibration as orchestration


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/experiment3/soft_blockpush_phase0b_r2r2.json"


def _metric(verdict):
    return {"verdict": verdict, "peak_primary_displacement_m": .008}


def _setup(tmp_path):
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["report_root"] = str(tmp_path / "reports")
    config["output_root"] = str(tmp_path / "data")
    selection = tmp_path / "source_selection.json"
    selection.write_text(json.dumps({
        "microsteps_per_outer": 8,
        "confirmation_verdict": "PHASE0B_R2R1_FAMILY_CONFIRMATION_COMPLETE"}))
    config["selected_microsteps_path"] = str(selection)
    config_path = tmp_path / "config.json"; config_path.write_text(json.dumps(config))
    return config_path


def _run(tmp_path, monkeypatch, outcomes, table_complete=True):
    config_path = _setup(tmp_path); queue = list(outcomes); calls = []
    def profile(config, path, selection):
        del config, selection
        calls.append(Path(path).stem.replace("soft_block_", ""))
        axial_verdict, shear_verdict = queue.pop(0)
        axial = _metric(axial_verdict)
        shear = None if shear_verdict is None else _metric(shear_verdict)
        return axial, shear
    monkeypatch.setattr(orchestration, "run_profile", profile)
    monkeypatch.setattr(orchestration, "run_settle", lambda *args: tmp_path / "settle")
    verdict = ("PHASE0B_R2_TABLE_SETTLE_COMPLETE" if table_complete
               else "PHASE0B_R2_TABLE_SETTLE_UNSTABLE")
    monkeypatch.setattr(orchestration, "analyze_table", lambda *args: {
        "verdict": verdict,
        "frozen_outputs": {"material": "x"} if table_complete else {}})
    result = orchestration.run_calibration(str(config_path), str(tmp_path / "profiles"))
    return result, calls


def test_start_complete_runs_table_and_reports_no_pair_training(tmp_path, monkeypatch):
    result, calls = _run(tmp_path, monkeypatch, [
        ("PHASE0B_R2_AXIAL_COMPLETE", "PHASE0B_R2_SHEAR_COMPLETE")])
    assert result["verdict"].endswith("MATERIAL_CALIBRATION_COMPLETE")
    assert calls == ["kv_r2r2_s12p5_z025"]
    assert result["table_settle"]["frozen_outputs"]
    assert result["pair_run"] is False and result["training_run"] is False


@pytest.mark.parametrize("failure,replacement", [
    ("PHASE0B_R2_SHEAR_TOO_SOFT", "kv_r2r2_s15p0_z025"),
    ("PHASE0B_R2_SHEAR_TOO_STIFF", "kv_r2r2_s10p5_z025")])
def test_softness_direction_runs_one_full_neighbor(tmp_path, monkeypatch,
                                                   failure, replacement):
    result, calls = _run(tmp_path, monkeypatch, [
        ("PHASE0B_R2_AXIAL_COMPLETE", failure),
        ("PHASE0B_R2_AXIAL_COMPLETE", "PHASE0B_R2_SHEAR_COMPLETE")])
    assert result["verdict"].endswith("MATERIAL_CALIBRATION_COMPLETE")
    assert calls == ["kv_r2r2_s12p5_z025", replacement]
    assert len(result["profiles_run"]) == 2


@pytest.mark.parametrize("failure", [
    "PHASE0B_R2_SHEAR_UNSTABLE", "PHASE0B_R2_SHEAR_ENGINEERING_BLOCKED"])
def test_unstable_or_engineering_blocked_stops(tmp_path, monkeypatch, failure):
    result, calls = _run(tmp_path, monkeypatch, [
        ("PHASE0B_R2_AXIAL_COMPLETE", failure)])
    assert result["verdict"] == "PHASE0B_R2R2_NO_PROFILE_PASSED"
    assert len(calls) == 1 and result["table_settle"] is None


def test_second_failure_never_runs_third_profile(tmp_path, monkeypatch):
    result, calls = _run(tmp_path, monkeypatch, [
        ("PHASE0B_R2_AXIAL_COMPLETE", "PHASE0B_R2_SHEAR_TOO_SOFT"),
        ("PHASE0B_R2_AXIAL_COMPLETE", "PHASE0B_R2_SHEAR_TOO_SOFT")])
    assert result["verdict"] == "PHASE0B_R2R2_NO_PROFILE_PASSED"
    assert len(calls) == len(result["profiles_run"]) == 2


def test_table_failure_does_not_freeze(tmp_path, monkeypatch):
    result, _ = _run(tmp_path, monkeypatch, [
        ("PHASE0B_R2_AXIAL_COMPLETE", "PHASE0B_R2_SHEAR_COMPLETE")], False)
    assert result["verdict"] == "PHASE0B_R2R2_TABLE_SETTLE_FAILED"
    assert result["table_settle"]["frozen_outputs"] == {}
