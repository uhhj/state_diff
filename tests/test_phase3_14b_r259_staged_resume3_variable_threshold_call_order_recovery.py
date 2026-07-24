from __future__ import annotations

import ast
import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ccda_phase3 import (
    phase314b_r259_staged_resume3_variable_threshold_call_order_recovery as r,
)
from ccda_phase3 import (
    phase314b_r259_staged_t10_risk_descriptor_calibration_repair as staged,
)
from ccda_phase3 import (
    phase314b_r258_stagex_resume1_import_crossfit_recovery as kernel,
)
from ccda_phase3.phase314b_r258_stagex_tail_robust_nested_oof import StageXSpec


def sample(rows=8):
    rng = np.random.default_rng(11)
    control = rng.normal(size=(rows, 4, 48)).astype(np.float32)
    candidate = control + rng.normal(scale=0.01, size=control.shape).astype(np.float32)
    scale = np.where(np.arange(rows) % 3 == 0, 0.0, 0.25).astype(np.float64)
    risk = np.linspace(0.05, 0.85, rows, dtype=np.float64)
    threshold = np.linspace(0.25, 0.75, rows, dtype=np.float64)
    target = control + rng.normal(scale=0.02, size=control.shape).astype(np.float32)
    groups = np.asarray(["g{}".format(i // 2) for i in range(rows)], dtype=object)
    constant = np.full(rows, 0.4, dtype=np.float64)
    return control, candidate, scale, risk, target, groups, threshold, constant


@pytest.mark.parametrize(
    "value",
    [r.BASE_HEAD, r.BASE_RESUME2_IMPLEMENTATION, r.EXPECTED_SUBMODULE],
)
def test_commit_values_are_sha1(value):
    assert len(value) == 40
    int(value, 16)


def test_implementation_is_add_only_three_files():
    assert len(r.IMPLEMENTATION_PATHS) == 3
    assert {status for status, _ in r.IMPLEMENTATION_PATHS} == {"A"}


def test_resume2_report_sha_is_frozen():
    assert r.RESUME2_BLOCKED_REPORT_SHA256 == (
        "f13bde1a6e7adf3fb18d05901ab58ce36f811897c505f440214f6f5767017f58"
    )


def test_stable_json_is_order_independent():
    assert r.stable_json_bytes({"b": 1, "a": 2}) == r.stable_json_bytes({"a": 2, "b": 1})


def test_sha256_bytes_known_value():
    assert r.sha256_bytes(b"abc") == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_original_stage_d_call_is_misordered():
    path = Path(staged.__file__)
    module = ast.parse(path.read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "evaluate_variable_threshold_policy"
    ]
    assert len(calls) == 1
    call = calls[0]
    assert [ast.unparse(arg) for arg in call.args[4:8]] == [
        "target",
        "groups",
        "procedure_threshold",
        "procedure_constant",
    ]


def test_frozen_kernel_signature_order():
    source = Path(kernel.__file__) if getattr(kernel, "__file__", None) else (
        Path(__file__).resolve().parents[1]
        / "ccda_phase3/phase314b_r258_stagex_resume1_import_crossfit_recovery.py"
    )
    module = ast.parse(source.read_text(encoding="utf-8"))
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "evaluate_variable_threshold_policy"
    )
    names = [argument.arg for argument in function.args.args]
    assert names[4:8] == [
        "thresholds",
        "target",
        "groups",
        "baseline_probability",
    ]


def test_adapter_matches_direct_frozen_kernel_in_clean_subprocess():
    root = Path(__file__).resolve().parents[1]
    code = r"""
import numpy as np
from ccda_phase3 import phase314b_r259_staged_t10_risk_descriptor_calibration_repair as staged
from ccda_phase3 import phase314b_r258_stagex_resume1_import_crossfit_recovery as kernel
from ccda_phase3 import phase314b_r259_staged_resume3_variable_threshold_call_order_recovery as recovery
from ccda_phase3.phase314b_r258_stagex_tail_robust_nested_oof import StageXSpec
rng=np.random.default_rng(11)
rows=8
control=rng.normal(size=(rows,4,48)).astype(np.float32)
candidate=control+rng.normal(scale=0.01,size=control.shape).astype(np.float32)
scale=np.where(np.arange(rows)%3==0,0.0,0.25).astype(np.float64)
risk=np.linspace(0.05,0.85,rows,dtype=np.float64)
threshold=np.linspace(0.25,0.75,rows,dtype=np.float64)
target=control+rng.normal(scale=0.02,size=control.shape).astype(np.float32)
groups=np.asarray([f'g{i//2}' for i in range(rows)],dtype=object)
constant=np.full(rows,0.4,dtype=np.float64)
spec=StageXSpec()
direct=kernel.evaluate_variable_threshold_policy(control,candidate,scale,risk,threshold,target,groups,constant,spec)
with recovery.installed_resume3_adapters():
    recovered=staged.kernel.evaluate_variable_threshold_policy(control,candidate,scale,risk,target,groups,threshold,constant,spec)
assert recovered == direct
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root)
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(root),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_adapter_passes_threshold_vector_unchanged(monkeypatch):
    values = sample(5)
    captured = {}

    def fake_original(*args):
        captured["threshold"] = np.asarray(args[4]).copy()
        captured["target"] = np.asarray(args[5]).copy()
        captured["groups"] = np.asarray(args[6]).copy()
        return {"ok": True}

    original = staged.kernel.evaluate_variable_threshold_policy
    monkeypatch.setattr(staged.kernel, "evaluate_variable_threshold_policy", fake_original)
    try:
        with r.installed_resume3_adapters():
            result = staged.kernel.evaluate_variable_threshold_policy(*values, StageXSpec())
        assert result == {"ok": True}
        np.testing.assert_array_equal(captured["threshold"], values[6])
        np.testing.assert_array_equal(captured["target"], values[4])
        np.testing.assert_array_equal(captured["groups"], values[5])
    finally:
        monkeypatch.setattr(staged.kernel, "evaluate_variable_threshold_policy", original)


def test_adapter_installs_resume2_descriptor_and_restores():
    original_descriptor = staged.build_risk_descriptors
    original_policy = staged.kernel.evaluate_variable_threshold_policy
    with r.installed_resume3_adapters():
        assert staged.build_risk_descriptors is r.resume2.build_risk_descriptors_48
        assert staged.kernel.evaluate_variable_threshold_policy is r.evaluate_stage_d_variable_threshold_call
    assert staged.build_risk_descriptors is original_descriptor
    assert staged.kernel.evaluate_variable_threshold_policy is original_policy


def test_adapter_restores_after_exception():
    original_descriptor = staged.build_risk_descriptors
    original_policy = staged.kernel.evaluate_variable_threshold_policy
    with pytest.raises(RuntimeError):
        with r.installed_resume3_adapters():
            raise RuntimeError("boom")
    assert staged.build_risk_descriptors is original_descriptor
    assert staged.kernel.evaluate_variable_threshold_policy is original_policy


def test_nested_adapter_install_is_rejected():
    with r.installed_resume3_adapters():
        with pytest.raises(r.StageDResume3Error, match="already installed"):
            with r.installed_resume3_adapters():
                pass


@pytest.mark.parametrize("bad_threshold", [np.zeros((8, 1)), np.zeros(7), np.zeros((2, 4))])
def test_adapter_rejects_bad_threshold_population(bad_threshold):
    values = list(sample())
    values[6] = bad_threshold
    with r.installed_resume3_adapters():
        with pytest.raises(r.StageDResume3Error, match="threshold vector"):
            staged.kernel.evaluate_variable_threshold_policy(*values, StageXSpec())


def test_adapter_rejects_bad_target_shape():
    values = list(sample())
    values[4] = values[4][..., :-1]
    with r.installed_resume3_adapters():
        with pytest.raises(r.StageDResume3Error, match="target tensor"):
            staged.kernel.evaluate_variable_threshold_policy(*values, StageXSpec())


def test_adapter_rejects_bad_group_population():
    values = list(sample())
    values[5] = values[5][:-1]
    with r.installed_resume3_adapters():
        with pytest.raises(r.StageDResume3Error, match="group population"):
            staged.kernel.evaluate_variable_threshold_policy(*values, StageXSpec())


def test_resume2_descriptor_accepts_real_cable_shape():
    values = sample(3)
    direction = values[1] - values[0]
    result = r.resume2.build_risk_descriptors_48(
        "compact_geometry_v2", values[0], direction, values[1], values[2]
    )
    assert result.shape == (3, 66)
    assert np.all(np.isfinite(result))


def test_atomic_write_once(tmp_path):
    path = tmp_path / "evidence.json"
    r.atomic_write_once(path, b"abc")
    assert path.read_bytes() == b"abc"


def test_atomic_write_refuses_overwrite(tmp_path):
    path = tmp_path / "evidence.json"
    path.write_bytes(b"old")
    with pytest.raises(r.StageDResume3Error, match="write-once"):
        r.atomic_write_once(path, b"new")


def test_promote_write_ahead_is_exact(tmp_path):
    source = tmp_path / "source.json"
    destination = tmp_path / "repo" / "destination.json"
    source.write_text(json.dumps({"x": 1}), encoding="utf-8")
    digest = r.promote_write_ahead(source, destination, lambda value: value)
    assert source.read_bytes() == destination.read_bytes()
    assert digest == r.sha256_bytes(source.read_bytes())


def fake_base():
    false_boundaries = ("formal_training_run", "frozen_probe_accessed", "candidate_execution")

    def validate_worker(payload):
        assert payload["schema"] == "fake_worker"

    return SimpleNamespace(
        validate_worker_evidence=validate_worker,
        FALSE_BOUNDARIES=false_boundaries,
    )


def recovery_contract():
    return {
        "cable_tensor_contract": ["N", 4, 48],
        "descriptor_adapter": "resume2_48d_24_bead",
        "historical_stage_d_argument_order": [
            "control", "candidate", "scale", "risk", "target", "groups", "thresholds", "constant", "spec"
        ],
        "frozen_kernel_argument_order": [
            "control", "candidate", "scale", "risk", "thresholds", "target", "groups", "constant", "spec"
        ],
        "threshold_population_changed": False,
        "recipe_bank_changed": False,
        "threshold_values_changed": False,
        "science_kernel_copied": False,
        "single_cold_science_worker": True,
    }


def synthetic_worker(monkeypatch, status="BLOCKED"):
    monkeypatch.setattr(r, "_base", fake_base)
    payload = {
        "schema": "fake_worker",
        "process_id": 22,
        "scientific_status": status,
        "root_cause": "root",
        "required_next_path": "next",
        "primary_failure_locus": "locus",
        "population": {"row_count": 638},
        "legacy_stagec_control_contract": {},
        "recipe_population": [],
        "outer_fold_selections": [],
        "outer_fold_selected_recipe_metrics": [],
        "outer_selection_modal_recipe_diagnostic": {},
        "outer_crossfit_fold_selected_recipe_record": {},
        "outer_crossfit_no_abstention_baseline_record": {},
        "outer_crossfit_procedure_eligibility": status == "READY",
        "full_objective_oof_fixed_recipe_selection": {},
        "procedure_gate_totals": {},
        "execution_counts": {},
        "train_only_recommendation": {"recipe_id": "r"} if status == "READY" else None,
        "resume3_call_order_recovery": recovery_contract(),
        "formal_training_run": False,
        "frozen_probe_accessed": False,
        "candidate_execution": False,
    }
    payload["worker_result_sha256"] = r.sha256_bytes(r.stable_json_bytes(payload))
    return payload


def test_worker_validation_accepts(monkeypatch):
    r.validate_worker_evidence(synthetic_worker(monkeypatch))


@pytest.mark.parametrize(
    "field,value",
    [
        ("threshold_population_changed", True),
        ("recipe_bank_changed", True),
        ("threshold_values_changed", True),
        ("science_kernel_copied", True),
        ("single_cold_science_worker", False),
        ("descriptor_adapter", "wrong"),
    ],
)
def test_worker_validation_rejects_changed_contract(monkeypatch, field, value):
    payload = synthetic_worker(monkeypatch)
    payload["resume3_call_order_recovery"][field] = value
    payload["worker_result_sha256"] = r.sha256_bytes(
        r.stable_json_bytes({k: v for k, v in payload.items() if k != "worker_result_sha256"})
    )
    with pytest.raises(r.StageDResume3Error, match="recovery contract"):
        r.validate_worker_evidence(payload)


def test_worker_validation_rejects_bad_hash(monkeypatch):
    payload = synthetic_worker(monkeypatch)
    payload["worker_result_sha256"] = "0" * 64
    with pytest.raises(r.StageDResume3Error, match="self-hash"):
        r.validate_worker_evidence(payload)


def synthetic_summary(monkeypatch, status="BLOCKED"):
    monkeypatch.setattr(r, "_base", fake_base)
    payload = {
        "schema": r.SCHEMA,
        "execution_verdict": "PASS",
        "scientific_status": status,
        "call_order_recovery": recovery_contract(),
        "selected_configuration": None,
        "train_only_recommendation": {"recipe_id": "r"} if status == "READY" else None,
        "selection_holdout_evaluation_count_added": 0,
        "cumulative_selection_holdout_evaluation_count": 1,
        "frozen_probe_accessed": False,
        "rerun_authorized": False,
        "formal_training_run": False,
        "candidate_execution": False,
    }
    payload["summary_sha256"] = r.sha256_bytes(r.stable_json_bytes(payload))
    return payload


@pytest.mark.parametrize("status", ["READY", "BLOCKED"])
def test_summary_accepts_status(monkeypatch, status):
    r.validate_summary(synthetic_summary(monkeypatch, status))


def rehash(payload):
    payload["summary_sha256"] = r.sha256_bytes(
        r.stable_json_bytes({k: v for k, v in payload.items() if k != "summary_sha256"})
    )


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("selected_configuration", {"x": 1}, "selected"),
        ("selection_holdout_evaluation_count_added", 1, "holdout"),
        ("cumulative_selection_holdout_evaluation_count", 2, "holdout"),
        ("frozen_probe_accessed", True, "frozen probe"),
        ("rerun_authorized", True, "rerun"),
        ("formal_training_run", True, "forbidden"),
    ],
)
def test_summary_rejects_boundary_changes(monkeypatch, field, value, match):
    payload = synthetic_summary(monkeypatch)
    payload[field] = value
    rehash(payload)
    with pytest.raises(r.StageDResume3Error, match=match):
        r.validate_summary(payload)


def test_ready_requires_recommendation(monkeypatch):
    payload = synthetic_summary(monkeypatch, "READY")
    payload["train_only_recommendation"] = None
    rehash(payload)
    with pytest.raises(r.StageDResume3Error, match="recommendation"):
        r.validate_summary(payload)


def test_blocked_rejects_recommendation(monkeypatch):
    payload = synthetic_summary(monkeypatch, "BLOCKED")
    payload["train_only_recommendation"] = {"x": 1}
    rehash(payload)
    with pytest.raises(r.StageDResume3Error, match="recommendation"):
        r.validate_summary(payload)


def test_blocked_report_never_authorizes_retry(monkeypatch, tmp_path):
    monkeypatch.setattr(r, "_base", fake_base)
    payload = r.blocked_report(
        repository=None,
        error=RuntimeError("x"),
        probe_path=tmp_path / "probe",
        worker_path=tmp_path / "worker",
    )
    assert payload["science_reexecution_authorized"] is False
    assert payload["rerun_authorized"] is False
    assert payload["selected_configuration"] is None
    assert payload["train_only_recommendation"] is None
