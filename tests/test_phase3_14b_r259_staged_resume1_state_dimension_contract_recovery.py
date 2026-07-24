from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

import ccda_phase3
from ccda_phase3 import (
    phase314b_r259_staged_resume1_state_dimension_contract_recovery as r,
)


def fake_legacy(control, direction, candidate, scale):
    rows = control.shape[0]
    c = np.asarray(control, dtype=np.float64).reshape(rows, -1)
    d = np.asarray(direction, dtype=np.float64).reshape(rows, -1)
    m = (np.asarray(candidate, dtype=np.float64) - np.asarray(control, dtype=np.float64)).reshape(rows, -1)
    d_h = np.asarray(direction, dtype=np.float64).reshape(rows, 4, -1)
    m_h = (np.asarray(candidate, dtype=np.float64) - np.asarray(control, dtype=np.float64)).reshape(rows, 4, -1)
    dn = np.linalg.norm(d, axis=1)
    mn = np.linalg.norm(m, axis=1)
    columns = [
        np.asarray(scale, dtype=np.float64),
        (np.asarray(scale) > 0).astype(np.float64),
        dn,
        mn,
        np.linalg.norm(c, axis=1),
        np.max(np.abs(d), axis=1),
        np.max(np.abs(m), axis=1),
        np.sum(d * m, axis=1) / np.maximum(dn * mn, 1.0e-12),
        mn / np.maximum(dn, 1.0e-12),
    ]
    columns.extend(np.linalg.norm(d_h, axis=2)[:, index] for index in range(4))
    columns.extend(np.linalg.norm(m_h, axis=2)[:, index] for index in range(4))
    return np.stack(columns, axis=1)


def install_fake_base(monkeypatch):
    name = "ccda_phase3.phase314b_r259_staged_t10_risk_descriptor_calibration_repair"
    module = ModuleType(name)
    module.stagex = SimpleNamespace(compact_risk_descriptors=fake_legacy)
    module.FALSE_BOUNDARIES = (
        "selection_holdout_accessed",
        "frozen_probe_used",
        "formal_training_run",
        "reverse_sampling_run",
    )

    def validate_worker(payload):
        assert payload["schema"] == "base_worker"
        expected = r.sha256_bytes(
            r.stable_json_bytes(
                {key: value for key, value in payload.items() if key != "worker_result_sha256"}
            )
        )
        assert payload["worker_result_sha256"] == expected

    module.validate_worker_evidence = validate_worker
    module.validate_probe_evidence = lambda payload: payload["environment"]
    module.make_probe_evidence = lambda root: {
        "schema": "probe",
        "process_id": 101,
        "portable_environment_audit": {"compatibility_pass": True},
        "environment": {"compatibility_pass": True},
    }
    module.build_risk_descriptors = lambda *args: "original"
    monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(
        ccda_phase3,
        "phase314b_r259_staged_t10_risk_descriptor_calibration_repair",
        module,
        raising=False,
    )
    return module


def arrays(rows=3):
    rng = np.random.default_rng(7)
    control = rng.normal(size=(rows, 4, 87)).astype(np.float64)
    direction = rng.normal(scale=0.01, size=(rows, 4, 87)).astype(np.float64)
    candidate = control + 0.5 * direction
    scale = np.linspace(0.1, 0.3, rows)
    return control, direction, candidate, scale


@pytest.mark.parametrize(
    "name,expected",
    [
        ("EXPECTED_HORIZONS", 4),
        ("EXPECTED_STATE_DIM", 87),
        ("CABLE_XY_DIM", 46),
        ("CABLE_BEADS", 23),
        ("EXPECTED_LEGACY_DESCRIPTOR_DIM", 17),
        ("EXPECTED_GEOMETRY_DESCRIPTOR_DIM", 49),
        ("EXPECTED_COMBINED_DESCRIPTOR_DIM", 66),
    ],
)
def test_frozen_dimensions(name, expected):
    assert getattr(r, name) == expected


@pytest.mark.parametrize(
    "value",
    [
        r.BASE_STAGE_D_IMPLEMENTATION_COMMIT,
        r.BASE_STAGE_D_BLOCKED_EVIDENCE_COMMIT,
        r.BASE_STAGE_D_PARENT,
        r.EXPECTED_SUBMODULE,
    ],
)
def test_identity_values_are_sha1(value):
    assert len(value) == 40
    int(value, 16)


def test_remote_binding():
    assert r.EXPECTED_REMOTE == "591d1fd0814f7bd50b8e580274462908512a745f"


def test_implementation_is_add_only_three_files():
    assert len(r.IMPLEMENTATION_PATHS) == 3
    assert {status for status, _ in r.IMPLEMENTATION_PATHS} == {"A"}


def test_stable_json_is_order_independent():
    assert r.stable_json_bytes({"b": 1, "a": 2}) == r.stable_json_bytes({"a": 2, "b": 1})


def test_compact_json_has_no_whitespace():
    assert r.compact_json_bytes({"a": 1}) == b'{"a":1}'


def test_sha256_bytes_known_value():
    assert r.sha256_bytes(b"abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_compact_descriptor_accepts_87d(monkeypatch):
    install_fake_base(monkeypatch)
    result = r.build_risk_descriptors_87("compact_v1", *arrays())
    assert result.shape == (3, 17)


def test_geometry_descriptor_accepts_87d(monkeypatch):
    install_fake_base(monkeypatch)
    result = r.build_risk_descriptors_87("compact_geometry_v2", *arrays())
    assert result.shape == (3, 66)
    assert np.all(np.isfinite(result))


def test_legacy_prefix_is_exact(monkeypatch):
    base = install_fake_base(monkeypatch)
    values = arrays()
    result = r.build_risk_descriptors_87("compact_geometry_v2", *values)
    expected = base.stagex.compact_risk_descriptors(*values)
    np.testing.assert_array_equal(result[:, :17], expected)


def test_tail_state_affects_legacy_descriptor(monkeypatch):
    install_fake_base(monkeypatch)
    control, direction, candidate, scale = arrays()
    first = r.build_risk_descriptors_87("compact_v1", control, direction, candidate, scale)
    changed = control.copy()
    changed[..., 86] += 3.0
    second = r.build_risk_descriptors_87("compact_v1", changed, direction, candidate, scale)
    assert not np.array_equal(first, second)


def test_tail_state_does_not_change_geometry_suffix_when_movement_is_preserved(monkeypatch):
    install_fake_base(monkeypatch)
    control, direction, candidate, scale = arrays()
    first = r.build_risk_descriptors_87("compact_geometry_v2", control, direction, candidate, scale)
    control2 = control.copy()
    candidate2 = candidate.copy()
    control2[..., 46:] += 2.0
    candidate2[..., 46:] += 2.0
    second = r.build_risk_descriptors_87("compact_geometry_v2", control2, direction, candidate2, scale)
    np.testing.assert_allclose(first[:, 17:], second[:, 17:], rtol=0, atol=0)


@pytest.mark.parametrize(
    "shape",
    [
        (2, 4, 67),
        (2, 3, 87),
        (2, 4, 88),
        (2, 87),
    ],
)
def test_wrong_state_shapes_fail(monkeypatch, shape):
    install_fake_base(monkeypatch)
    value = np.zeros(shape)
    scale = np.zeros(shape[0])
    with pytest.raises(r.StageDResume1Error, match=r"\[N,4,87\]"):
        r.build_risk_descriptors_87("compact_v1", value, value, value, scale)


def test_different_tensor_shapes_fail(monkeypatch):
    install_fake_base(monkeypatch)
    control, direction, candidate, scale = arrays()
    with pytest.raises(r.StageDResume1Error, match="shapes differ"):
        r.build_risk_descriptors_87("compact_v1", control, direction[:, :, :-1], candidate, scale)


def test_wrong_scale_shape_fails(monkeypatch):
    install_fake_base(monkeypatch)
    control, direction, candidate, _ = arrays()
    with pytest.raises(r.StageDResume1Error, match="selected-scale"):
        r.build_risk_descriptors_87("compact_v1", control, direction, candidate, np.zeros((3, 1)))


def test_unknown_descriptor_fails(monkeypatch):
    install_fake_base(monkeypatch)
    with pytest.raises(r.StageDResume1Error, match="descriptor ID"):
        r.build_risk_descriptors_87("unknown", *arrays())


def test_nonfinite_descriptor_fails(monkeypatch):
    install_fake_base(monkeypatch)
    control, direction, candidate, scale = arrays()
    control[0, 0, 0] = np.nan
    with pytest.raises(r.StageDResume1Error, match="NaN or Inf"):
        r.build_risk_descriptors_87("compact_geometry_v2", control, direction, candidate, scale)


def test_adapter_installs_and_restores(monkeypatch):
    base = install_fake_base(monkeypatch)
    original = base.build_risk_descriptors
    with r.installed_state_dimension_adapter():
        assert base.build_risk_descriptors is r.build_risk_descriptors_87
    assert base.build_risk_descriptors is original


def test_adapter_restores_after_error(monkeypatch):
    base = install_fake_base(monkeypatch)
    original = base.build_risk_descriptors
    with pytest.raises(RuntimeError):
        with r.installed_state_dimension_adapter():
            raise RuntimeError("boom")
    assert base.build_risk_descriptors is original


def test_probe_delegates(monkeypatch, tmp_path):
    install_fake_base(monkeypatch)
    payload = r.make_probe_evidence(tmp_path)
    assert payload["process_id"] == 101
    assert r.validate_probe_evidence(payload)["compatibility_pass"] is True


def test_atomic_write_once(tmp_path):
    path = tmp_path / "evidence.json"
    r.atomic_write_once(path, b"abc")
    assert path.read_bytes() == b"abc"


def test_atomic_write_refuses_overwrite(tmp_path):
    path = tmp_path / "evidence.json"
    path.write_bytes(b"old")
    with pytest.raises(r.StageDResume1Error, match="write-once"):
        r.atomic_write_once(path, b"new")


def test_promote_write_ahead_is_byte_exact(tmp_path):
    source = tmp_path / "source.json"
    destination = tmp_path / "repo" / "destination.json"
    source.write_text(json.dumps({"x": 1}), encoding="utf-8")
    digest = r.promote_write_ahead(source, destination, lambda value: value)
    assert destination.read_bytes() == source.read_bytes()
    assert digest == r.sha256_bytes(source.read_bytes())


def test_promote_requires_source(tmp_path):
    with pytest.raises(r.StageDResume1Error, match="source is missing"):
        r.promote_write_ahead(tmp_path / "missing", tmp_path / "out", lambda value: value)


def synthetic_worker(monkeypatch, status="BLOCKED"):
    base = install_fake_base(monkeypatch)
    payload = {
        "schema": "base_worker",
        "process_id": 202,
        "execution_verdict": "PASS",
        "scientific_status": status,
        "root_cause": "root",
        "required_next_path": "next",
        "primary_failure_locus": "locus",
        "population": {"row_count": 638, "group_count": 126},
        "legacy_stagec_control_contract": {"t25": True, "t50": True},
        "recipe_population": [{"recipe_id": "r"}],
        "outer_fold_selections": [{"outer_fold": i} for i in range(6)],
        "outer_fold_selected_recipe_metrics": [{"outer_fold": i} for i in range(6)],
        "outer_selection_modal_recipe_diagnostic": {"support_count": 4},
        "outer_crossfit_fold_selected_recipe_record": {"acceptance_rate": 0.5},
        "outer_crossfit_no_abstention_baseline_record": {"acceptance_rate": 1.0},
        "outer_crossfit_procedure_eligibility": status == "READY",
        "all_outer_inner_selections_eligible": status == "READY",
        "outer_recipe_modal_support_stable": True,
        "full_objective_oof_fixed_recipe_selection": {"recipe_id": "r"},
        "full_objective_oof_selected_recipe_eligible": status == "READY",
        "procedure_gate_totals": {"aligned_upper_element_failure_count": 0},
        "execution_counts": {"direction_fit_count": 36},
        "train_only_recommendation": {"recipe_id": "r"} if status == "READY" else None,
        "selected_configuration": None,
        "selection_holdout_evaluation_count_added": 0,
        "cumulative_selection_holdout_evaluation_count": 1,
        "frozen_probe_accessed": False,
        "rerun_authorized": False,
        "resume1_state_dimension_recovery": {
            "original_stage_d_failure_before_durable_science_evidence": True,
            "original_required_state_dim": 67,
            "actual_frozen_state_dim": 87,
            "horizon_count": 4,
            "ordered_cable_xy_dim": 46,
            "ordered_cable_bead_count": 23,
            "legacy_descriptor_uses_full_87d_state": True,
            "geometry_descriptor_uses_first_46_ordered_xy_values": True,
            "recipe_bank_changed": False,
            "nested_group_oof_changed": False,
            "direction_or_candidate_mechanism_changed": False,
            "thresholds_changed": False,
            "single_cold_science_worker": True,
        },
    }
    for key in base.FALSE_BOUNDARIES:
        payload[key] = False
    payload["worker_result_sha256"] = r.sha256_bytes(r.stable_json_bytes(payload))
    return payload


def test_resume_worker_validation_accepts(monkeypatch):
    r.validate_worker_evidence(synthetic_worker(monkeypatch))


@pytest.mark.parametrize(
    "key",
    [
        "original_stage_d_failure_before_durable_science_evidence",
        "legacy_descriptor_uses_full_87d_state",
        "geometry_descriptor_uses_first_46_ordered_xy_values",
        "single_cold_science_worker",
    ],
)
def test_resume_worker_rejects_changed_true_contract(monkeypatch, key):
    payload = synthetic_worker(monkeypatch)
    payload["resume1_state_dimension_recovery"][key] = False
    payload["worker_result_sha256"] = r.sha256_bytes(
        r.stable_json_bytes({k: v for k, v in payload.items() if k != "worker_result_sha256"})
    )
    with pytest.raises(r.StageDResume1Error, match="recovery contract"):
        r.validate_worker_evidence(payload)


@pytest.mark.parametrize(
    "key",
    [
        "recipe_bank_changed",
        "nested_group_oof_changed",
        "direction_or_candidate_mechanism_changed",
        "thresholds_changed",
    ],
)
def test_resume_worker_rejects_changed_false_contract(monkeypatch, key):
    payload = synthetic_worker(monkeypatch)
    payload["resume1_state_dimension_recovery"][key] = True
    payload["worker_result_sha256"] = r.sha256_bytes(
        r.stable_json_bytes({k: v for k, v in payload.items() if k != "worker_result_sha256"})
    )
    with pytest.raises(r.StageDResume1Error, match="recovery contract"):
        r.validate_worker_evidence(payload)


def test_resume_worker_rejects_bad_hash(monkeypatch):
    payload = synthetic_worker(monkeypatch)
    payload["worker_result_sha256"] = "x" * 64
    with pytest.raises((AssertionError, r.StageDResume1Error)):
        r.validate_worker_evidence(payload)


def synthetic_summary(monkeypatch, status="BLOCKED"):
    base = install_fake_base(monkeypatch)
    payload = {
        "schema": r.SCHEMA,
        "execution_verdict": "PASS",
        "scientific_status": status,
        "recovery_contract": {"actual_frozen_state_dim": 87},
        "durable_evidence_protocol": {
            "worker_persisted_before_controller_decoration": True,
            "external_write_ahead_outside_git_worktree": True,
            "repository_evidence_promoted_byte_exact": True,
            "write_ahead_files_write_once": True,
            "controller_recomputed_worker_science": False,
        },
        "selected_configuration": None,
        "train_only_recommendation": {"recipe_id": "r"} if status == "READY" else None,
        "selection_holdout_evaluation_count_added": 0,
        "cumulative_selection_holdout_evaluation_count": 1,
        "frozen_probe_accessed": False,
        "rerun_authorized": False,
    }
    for key in base.FALSE_BOUNDARIES:
        payload[key] = False
    payload["summary_sha256"] = r.sha256_bytes(r.stable_json_bytes(payload))
    return payload


@pytest.mark.parametrize("status", ["READY", "BLOCKED"])
def test_summary_validation_accepts(monkeypatch, status):
    r.validate_summary(synthetic_summary(monkeypatch, status))


def rehash_summary(payload):
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
    ],
)
def test_summary_rejects_boundary_changes(monkeypatch, field, value, match):
    payload = synthetic_summary(monkeypatch)
    payload[field] = value
    rehash_summary(payload)
    with pytest.raises(r.StageDResume1Error, match=match):
        r.validate_summary(payload)


def test_ready_summary_requires_recommendation(monkeypatch):
    payload = synthetic_summary(monkeypatch, "READY")
    payload["train_only_recommendation"] = None
    rehash_summary(payload)
    with pytest.raises(r.StageDResume1Error, match="recommendation"):
        r.validate_summary(payload)


def test_blocked_summary_rejects_recommendation(monkeypatch):
    payload = synthetic_summary(monkeypatch, "BLOCKED")
    payload["train_only_recommendation"] = {"x": 1}
    rehash_summary(payload)
    with pytest.raises(r.StageDResume1Error, match="recommendation"):
        r.validate_summary(payload)


def test_summary_rejects_bad_hash(monkeypatch):
    payload = synthetic_summary(monkeypatch)
    payload["summary_sha256"] = "x" * 64
    with pytest.raises(r.StageDResume1Error, match="self-hash"):
        r.validate_summary(payload)


def test_blocked_report_never_authorizes_rerun(monkeypatch, tmp_path):
    install_fake_base(monkeypatch)
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
