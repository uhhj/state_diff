from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from ccda_phase3 import phase314b_r259_staged_resume2_cable_state_schema_recovery as r


def fake_legacy(control, direction, candidate, scale):
    rows = control.shape[0]
    control_flat = np.asarray(control, dtype=np.float64).reshape(rows, -1)
    direction_flat = np.asarray(direction, dtype=np.float64).reshape(rows, -1)
    movement = (
        np.asarray(candidate, dtype=np.float64) - np.asarray(control, dtype=np.float64)
    )
    movement_flat = movement.reshape(rows, -1)
    direction_h = np.asarray(direction, dtype=np.float64).reshape(rows, 4, -1)
    movement_h = movement.reshape(rows, 4, -1)
    direction_norm = np.linalg.norm(direction_flat, axis=1)
    movement_norm = np.linalg.norm(movement_flat, axis=1)
    columns = [
        np.asarray(scale, dtype=np.float64),
        (np.asarray(scale) > 0.0).astype(np.float64),
        direction_norm,
        movement_norm,
        np.linalg.norm(control_flat, axis=1),
        np.max(np.abs(direction_flat), axis=1),
        np.max(np.abs(movement_flat), axis=1),
        np.sum(direction_flat * movement_flat, axis=1)
        / np.maximum(direction_norm * movement_norm, 1.0e-12),
        movement_norm / np.maximum(direction_norm, 1.0e-12),
    ]
    columns.extend(np.linalg.norm(direction_h, axis=2)[:, i] for i in range(4))
    columns.extend(np.linalg.norm(movement_h, axis=2)[:, i] for i in range(4))
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
                {
                    key: value
                    for key, value in payload.items()
                    if key != "worker_result_sha256"
                }
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
    monkeypatch.setattr(r, "_base", lambda: module)
    return module


def arrays(rows=3):
    rng = np.random.default_rng(17)
    control = rng.normal(size=(rows, 4, 48)).astype(np.float64)
    direction = rng.normal(scale=0.01, size=(rows, 4, 48)).astype(np.float64)
    candidate = control + 0.5 * direction
    scale = np.linspace(0.1, 0.3, rows)
    return control, direction, candidate, scale


@pytest.mark.parametrize(
    "name,expected",
    [
        ("DEFAULT_TF", 4),
        ("CABLE_DIM", 48),
        ("N_BEADS", 24),
        ("SEGMENT_COUNT", 23),
        ("LEGACY_DESCRIPTOR_DIM", 17),
        ("GEOMETRY_DESCRIPTOR_DIM", 49),
        ("COMBINED_DESCRIPTOR_DIM", 66),
    ],
)
def test_frozen_schema_constants(name, expected):
    assert getattr(r, name) == expected


def test_schema_identity_is_cable_only():
    assert r.CABLE_DIM == r.N_BEADS * 2
    assert r.CABLE_DIM != 67
    assert r.CABLE_DIM != 87


@pytest.mark.parametrize(
    "value",
    [
        r.BASE_HEAD,
        r.BASE_RESUME1_IMPLEMENTATION,
        r.BASE_STAGE_D_BLOCKED_EVIDENCE,
        r.BASE_STAGE_D_IMPLEMENTATION,
        r.BASE_STAGE_C_EVIDENCE,
        r.EXPECTED_REMOTE,
        r.EXPECTED_SUBMODULE,
    ],
)
def test_commit_identities_are_sha1(value):
    assert len(value) == 40
    int(value, 16)


def test_expected_remote_matches_bundle_head():
    assert r.EXPECTED_REMOTE == r.BASE_HEAD


def test_add_only_three_files():
    assert len(r.IMPLEMENTATION_PATHS) == 3
    assert {status for status, _ in r.IMPLEMENTATION_PATHS} == {"A"}


def test_stable_json_is_order_independent():
    assert r.stable_json_bytes({"b": 1, "a": 2}) == r.stable_json_bytes(
        {"a": 2, "b": 1}
    )


def test_sha256_bytes_known_value():
    assert r.sha256_bytes(b"abc") == hashlib.sha256(b"abc").hexdigest()


def test_compact_descriptor_accepts_48d(monkeypatch):
    install_fake_base(monkeypatch)
    result = r.build_risk_descriptors_48("compact_v1", *arrays())
    assert result.shape == (3, 17)


def test_geometry_descriptor_accepts_48d(monkeypatch):
    install_fake_base(monkeypatch)
    result = r.build_risk_descriptors_48("compact_geometry_v2", *arrays())
    assert result.shape == (3, 66)
    assert np.all(np.isfinite(result))


def test_legacy_prefix_is_exact(monkeypatch):
    base = install_fake_base(monkeypatch)
    values = arrays()
    result = r.build_risk_descriptors_48("compact_geometry_v2", *values)
    expected = base.stagex.compact_risk_descriptors(*values)
    np.testing.assert_array_equal(result[:, :17], expected)


def test_last_bead_changes_geometry(monkeypatch):
    install_fake_base(monkeypatch)
    control, direction, candidate, scale = arrays(rows=1)
    first = r.build_risk_descriptors_48(
        "compact_geometry_v2", control, direction, candidate, scale
    )
    changed_candidate = candidate.copy()
    changed_candidate[:, :, -2:] += np.array([[[0.25, -0.1]]])
    second = r.build_risk_descriptors_48(
        "compact_geometry_v2", control, direction, changed_candidate, scale
    )
    assert not np.array_equal(first[:, 17:], second[:, 17:])


def test_no_cable_coordinate_is_discarded(monkeypatch):
    install_fake_base(monkeypatch)
    control, direction, candidate, scale = arrays(rows=1)
    baseline = r.build_risk_descriptors_48(
        "compact_geometry_v2", control, direction, candidate, scale
    )
    for coordinate in (0, 1, 46, 47):
        changed = candidate.copy()
        changed[..., coordinate] += 0.125
        observed = r.build_risk_descriptors_48(
            "compact_geometry_v2", control, direction, changed, scale
        )
        assert not np.array_equal(baseline, observed)


@pytest.mark.parametrize(
    "shape",
    [
        (2, 4, 46),
        (2, 4, 67),
        (2, 4, 87),
        (2, 3, 48),
        (2, 48),
    ],
)
def test_wrong_tensor_shape_fails(monkeypatch, shape):
    install_fake_base(monkeypatch)
    value = np.zeros(shape, dtype=np.float64)
    scale = np.zeros(shape[0], dtype=np.float64)
    with pytest.raises(r.StageDResume2Error, match=r"\[N,4,48\]"):
        r.build_risk_descriptors_48("compact_v1", value, value, value, scale)


def test_mismatched_shapes_fail(monkeypatch):
    install_fake_base(monkeypatch)
    control, direction, candidate, scale = arrays()
    with pytest.raises(r.StageDResume2Error, match="shapes differ"):
        r.build_risk_descriptors_48(
            "compact_v1", control, direction[..., :-2], candidate, scale
        )


def test_wrong_scale_shape_fails(monkeypatch):
    install_fake_base(monkeypatch)
    control, direction, candidate, _ = arrays()
    with pytest.raises(r.StageDResume2Error, match="selected-scale"):
        r.build_risk_descriptors_48(
            "compact_v1", control, direction, candidate, np.zeros((3, 1))
        )


def test_unknown_descriptor_fails(monkeypatch):
    install_fake_base(monkeypatch)
    with pytest.raises(r.StageDResume2Error, match="descriptor ID"):
        r.build_risk_descriptors_48("unknown", *arrays())


def test_nonfinite_values_fail(monkeypatch):
    install_fake_base(monkeypatch)
    control, direction, candidate, scale = arrays()
    control[0, 0, 0] = np.nan
    with pytest.raises(r.StageDResume2Error, match="NaN or Inf"):
        r.build_risk_descriptors_48(
            "compact_geometry_v2", control, direction, candidate, scale
        )


def test_adapter_installs_and_restores(monkeypatch):
    base = install_fake_base(monkeypatch)
    original = base.build_risk_descriptors
    with r.installed_cable_schema_adapter():
        assert base.build_risk_descriptors is r.build_risk_descriptors_48
    assert base.build_risk_descriptors is original


def test_adapter_restores_after_exception(monkeypatch):
    base = install_fake_base(monkeypatch)
    original = base.build_risk_descriptors
    with pytest.raises(RuntimeError):
        with r.installed_cable_schema_adapter():
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
    with pytest.raises(r.StageDResume2Error, match="write-once"):
        r.atomic_write_once(path, b"new")


def test_promote_write_ahead_is_exact(tmp_path):
    source = tmp_path / "source.json"
    destination = tmp_path / "repo" / "destination.json"
    source.write_text(json.dumps({"x": 1}), encoding="utf-8")
    digest = r.promote_write_ahead(source, destination, lambda value: value)
    assert destination.read_bytes() == source.read_bytes()
    assert digest == r.sha256_bytes(source.read_bytes())


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
        "legacy_stagec_control_contract": {"10": {}, "25": {}, "50": {}},
        "recipe_population": [{"recipe_id": "r"}],
        "outer_fold_selections": [{"outer_fold": i} for i in range(6)],
        "outer_fold_selected_recipe_metrics": [{"outer_fold": i} for i in range(6)],
        "outer_selection_modal_recipe_diagnostic": {"support_count": 4},
        "outer_crossfit_fold_selected_recipe_record": {"acceptance_rate": 0.5},
        "outer_crossfit_no_abstention_baseline_record": {"acceptance_rate": 1.0},
        "outer_crossfit_procedure_eligibility": status == "READY",
        "full_objective_oof_fixed_recipe_selection": {"recipe_id": "r"},
        "procedure_gate_totals": {"aligned_upper_element_failure_count": 0},
        "execution_counts": {"direction_fit_count": 36},
        "train_only_recommendation": {"recipe_id": "r"} if status == "READY" else None,
        "selected_configuration": None,
        "selection_holdout_evaluation_count_added": 0,
        "cumulative_selection_holdout_evaluation_count": 1,
        "frozen_probe_accessed": False,
        "rerun_authorized": False,
        "resume2_cable_schema_recovery": {
            "actual_tensor_contract": ["N", 4, 48],
            "state_v3_bead_count": 24,
            "ordered_segment_count": 23,
            "legacy_descriptor_uses_all_cable_coordinates": True,
            "geometry_descriptor_uses_all_ordered_beads": True,
            "discarded_cable_coordinate_count": 0,
            "recipe_bank_changed": False,
            "nested_group_oof_changed": False,
            "candidate_mechanism_changed": False,
            "thresholds_changed": False,
            "single_cold_science_worker": True,
        },
    }
    for key in base.FALSE_BOUNDARIES:
        payload[key] = False
    payload["worker_result_sha256"] = r.sha256_bytes(r.stable_json_bytes(payload))
    return payload


def test_worker_recovery_contract_accepts(monkeypatch):
    r.validate_worker_evidence(synthetic_worker(monkeypatch))


@pytest.mark.parametrize(
    "key,value",
    [
        ("actual_tensor_contract", ["N", 4, 87]),
        ("state_v3_bead_count", 23),
        ("ordered_segment_count", 22),
        ("discarded_cable_coordinate_count", 2),
        ("recipe_bank_changed", True),
        ("nested_group_oof_changed", True),
        ("candidate_mechanism_changed", True),
        ("thresholds_changed", True),
    ],
)
def test_worker_rejects_changed_recovery_contract(monkeypatch, key, value):
    payload = synthetic_worker(monkeypatch)
    payload["resume2_cable_schema_recovery"][key] = value
    payload["worker_result_sha256"] = r.sha256_bytes(
        r.stable_json_bytes(
            {k: v for k, v in payload.items() if k != "worker_result_sha256"}
        )
    )
    with pytest.raises(r.StageDResume2Error, match="recovery contract"):
        r.validate_worker_evidence(payload)


def synthetic_summary(monkeypatch, status="BLOCKED"):
    base = install_fake_base(monkeypatch)
    payload = {
        "schema": r.SCHEMA,
        "execution_verdict": "PASS",
        "scientific_status": status,
        "cable_schema_recovery": {"actual_tensor_contract": ["N", 4, 48]},
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
def test_summary_accepts_valid_status(monkeypatch, status):
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
    with pytest.raises(r.StageDResume2Error, match=match):
        r.validate_summary(payload)


def test_ready_requires_recommendation(monkeypatch):
    payload = synthetic_summary(monkeypatch, "READY")
    payload["train_only_recommendation"] = None
    rehash_summary(payload)
    with pytest.raises(r.StageDResume2Error, match="recommendation"):
        r.validate_summary(payload)


def test_blocked_rejects_recommendation(monkeypatch):
    payload = synthetic_summary(monkeypatch, "BLOCKED")
    payload["train_only_recommendation"] = {"x": 1}
    rehash_summary(payload)
    with pytest.raises(r.StageDResume2Error, match="recommendation"):
        r.validate_summary(payload)


def test_blocked_report_never_authorizes_retry(monkeypatch, tmp_path):
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
