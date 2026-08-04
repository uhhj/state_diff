from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from ccda_phase3 import phase314b_r260_staged_candidate_signal_benchmark as m


def synthetic(rows: int = 60, features: int = 12):
    rng = np.random.RandomState(7)
    x = rng.normal(size=(rows, features)).astype(np.float64)
    coefficient = rng.normal(scale=0.04, size=(features, m.OUTPUT_DIM))
    y = (x @ coefficient).reshape(rows, m.DEFAULT_TF, m.CABLE_DIM)
    folds = np.arange(rows, dtype=np.int64) % m.EXPECTED_FOLDS
    groups = np.asarray([f"g{i // 2:03d}" for i in range(rows)])
    conditions = np.asarray([m.FORMAL_CONDITIONS[i % 2] for i in range(rows)])
    return x, y, folds, groups, conditions


def valid_worker(ready: bool = True):
    gates = {"all": ready}
    result = {
        "phase": m.PHASE,
        "schema": m.WORKER_SCHEMA,
        "execution_verdict": "PASS",
        "scientific_status": "READY" if ready else "BLOCKED",
        "primary_failure_locus": "x",
        "root_cause": "x",
        "required_next_path": "x",
        "source": {},
        "stage_c_reference": {},
        "candidate_spec": {},
        "gate_spec": {},
        "fit_contract": {},
        "oof": {},
        "evaluations": {},
        "gates": gates,
        "execution_counts": {
            "objective_train_logical_open_count": 1,
            "objective_train_file_hash_count": 1,
            "objective_train_npz_parse_count": 1,
            "stage_c_contract_npz_parse_count": 1,
            "direction_fit_count": 12,
            "candidate_generation_count": 3,
            "candidate_evaluation_count": 3,
            "risk_fit_count": 0,
            "diffusion_model_fit_count": 0,
            "selection_holdout_npz_open_count": 0,
            "frozen_probe_npz_open_count": 0,
            "final_evaluation_npz_open_count": 0,
        },
        "selected_configuration": None,
        "train_only_recommendation": {"candidate_id": "x"} if ready else None,
        **{key: False for key in m.FALSE_BOUNDARIES},
        "direction_fit_run": True,
        "candidate_generation_run": True,
        "candidate_evaluation_run": True,
    }
    result["worker_result_sha256"] = m.sha256_bytes(m.stable_json_bytes(result))
    return result


def test_specs_validate():
    m.PRIMARY_SPEC.validate()
    m.GATE_SPEC.validate()


@pytest.mark.parametrize(
    "field,value",
    [
        ("candidate_id", "bad"),
        ("feature_source", "paper_x"),
        ("model", "rridge"),
        ("ridge_l2", 1.0),
        ("candidate_scale", 0.5),
        ("folds", 5),
        ("permutation_seed", 1),
        ("standardizer_epsilon", 0.0),
        ("ridge_jitter", 0.0),
    ],
)
def test_candidate_spec_rejects_change(field, value):
    data = m.asdict(m.PRIMARY_SPEC)
    data[field] = value
    with pytest.raises(m.StageDError):
        m.CandidateSpec(**data).validate()


@pytest.mark.parametrize(
    "field,value",
    [
        ("aggregate_mse_ratio_max", 0.96),
        ("group_relative_improvement_mean_min", 0.04),
        ("group_relative_improvement_ci_lower_min", 0.0),
        ("positive_group_rate_min", 0.5),
        ("positive_fold_support_min", 4),
        ("condition_mse_ratio_max", 0.99),
        ("physical_valid_rate_min", 0.9),
        ("primary_vs_permutation_relative_margin_min", 0.02),
        ("primary_vs_global_mean_mse_ratio_margin_min", 0.01),
        ("minimum_candidate_movement_rms", 0.0),
    ],
)
def test_gate_spec_rejects_change(field, value):
    data = m.asdict(m.GATE_SPEC)
    data[field] = value
    with pytest.raises(m.StageDError):
        m.SignalGateSpec(**data).validate()


def test_stable_json_has_trailing_newline():
    assert m.stable_json_bytes({"b": 1, "a": 2}).endswith(b"\n")


def test_sha_array_is_shape_sensitive():
    x = np.arange(6, dtype=np.float64)
    assert m.sha256_array(x) != m.sha256_array(x.reshape(2, 3))


def test_atomic_write_once(tmp_path: Path):
    path = tmp_path / "x.json"
    m.atomic_write_once(path, b"x")
    assert path.read_bytes() == b"x"
    with pytest.raises(m.StageDError):
        m.atomic_write_once(path, b"y")


def test_standardizer_constant_feature():
    x = np.asarray([[1.0, 2.0], [1.0, 4.0], [1.0, 6.0]])
    s = m.fit_standardizer(x, 1e-12)
    assert s["constant_mask"].tolist() == [True, False]
    z = m.apply_standardizer(x, s)
    assert np.allclose(z[:, 0], 0.0)


def test_standardizer_rejects_one_row():
    with pytest.raises(m.StageDError):
        m.fit_standardizer(np.zeros((1, 2)), 1e-12)


def test_ridge_recovers_linear_signal():
    x, y, *_ = synthetic(rows=72, features=8)
    model = m.fit_ridge(x, y, l2=1e-6, jitter=1e-10, epsilon=1e-12)
    prediction = m.predict_ridge(model, x, y.shape[1:])
    assert np.mean((prediction - y) ** 2) < 1e-8


def test_predict_shape():
    x, y, *_ = synthetic(rows=60, features=8)
    model = m.fit_ridge(x[:40], y[:40], l2=10.0, jitter=1e-10, epsilon=1e-12)
    prediction = m.predict_ridge(model, x[40:], y.shape[1:])
    assert prediction.shape == y[40:].shape


def test_oof_predictions_are_complete():
    x, y, folds, *_ = synthetic(rows=72, features=8)
    result = m.oof_predictions(features=x, residual=y, row_fold_id=folds, spec=m.PRIMARY_SPEC)
    assert result["primary"].shape == y.shape
    assert result["fit_count"] == 12
    assert len(result["records"]) == 6


def test_oof_is_deterministic():
    x, y, folds, *_ = synthetic(rows=72, features=8)
    a = m.oof_predictions(features=x, residual=y, row_fold_id=folds, spec=m.PRIMARY_SPEC)
    b = m.oof_predictions(features=x, residual=y, row_fold_id=folds, spec=m.PRIMARY_SPEC)
    assert m.sha256_array(a["primary"]) == m.sha256_array(b["primary"])
    assert m.sha256_array(a["permuted"]) == m.sha256_array(b["permuted"])


def test_oof_rejects_empty_fold():
    x, y, _, *_ = synthetic(rows=60, features=8)
    folds = np.zeros(60, dtype=np.int64)
    with pytest.raises(m.StageDError):
        m.oof_predictions(features=x, residual=y, row_fold_id=folds, spec=m.PRIMARY_SPEC)


def test_geometry_bounds_and_mask():
    rng = np.random.RandomState(1)
    target = rng.normal(size=(20, m.DEFAULT_TF, m.CABLE_DIM))
    bounds = m.fold_geometry_bounds(target)
    rows = [bounds for _ in range(20)]
    mask = m.physical_valid_mask(target, rows)
    assert np.all(mask)


def test_geometry_rejects_exploded_candidate():
    target = np.zeros((10, m.DEFAULT_TF, m.CABLE_DIM), dtype=np.float64)
    points = target.reshape(10, m.DEFAULT_TF, m.N_BEADS, 2)
    points[:, :, :, 0] = np.arange(m.N_BEADS)[None, None, :]
    bounds = m.fold_geometry_bounds(target)
    candidate = target.copy()
    candidate[:, :, -2:] *= 100.0
    mask = m.physical_valid_mask(candidate, [bounds] * 10)
    assert not np.all(mask)


def test_group_effect_perfect_candidate():
    groups = np.asarray(["a", "a", "b", "b"])
    result = m._group_effects(np.ones(4), np.zeros(4), groups)
    assert result["relative_improvement_mean"] == pytest.approx(1.0)
    assert result["positive_group_rate"] == 1.0


def test_evaluate_perfect_candidate():
    rows = 12
    target = np.ones((rows, m.DEFAULT_TF, m.CABLE_DIM))
    control = np.zeros_like(target)
    groups = np.asarray([f"g{i // 2}" for i in range(rows)])
    conditions = np.asarray([m.FORMAL_CONDITIONS[i % 2] for i in range(rows)])
    folds = np.arange(rows) % 6
    record = m.evaluate_candidate(
        name="x",
        control=control,
        candidate=target,
        target=target,
        groups=groups,
        conditions=conditions,
        row_fold_id=folds,
        physical_mask=np.ones(rows, dtype=bool),
    )
    assert record["candidate_mse_ratio"] == 0.0
    assert record["positive_fold_support"] == 6


@pytest.mark.parametrize("key", m.FALSE_BOUNDARIES)
def test_worker_rejects_forbidden_boundary(key):
    worker = valid_worker()
    worker[key] = True
    worker["worker_result_sha256"] = m.sha256_bytes(
        m.stable_json_bytes({k: v for k, v in worker.items() if k != "worker_result_sha256"})
    )
    with pytest.raises(m.StageDError):
        m.validate_worker_result(worker)


@pytest.mark.parametrize(
    "key,value",
    [
        ("objective_train_logical_open_count", 2),
        ("objective_train_file_hash_count", 0),
        ("objective_train_npz_parse_count", 0),
        ("stage_c_contract_npz_parse_count", 0),
        ("direction_fit_count", 11),
        ("candidate_generation_count", 2),
        ("candidate_evaluation_count", 2),
        ("risk_fit_count", 1),
        ("diffusion_model_fit_count", 1),
        ("selection_holdout_npz_open_count", 1),
        ("frozen_probe_npz_open_count", 1),
        ("final_evaluation_npz_open_count", 1),
    ],
)
def test_worker_rejects_count_change(key, value):
    worker = valid_worker()
    worker["execution_counts"][key] = value
    worker["worker_result_sha256"] = m.sha256_bytes(
        m.stable_json_bytes({k: v for k, v in worker.items() if k != "worker_result_sha256"})
    )
    with pytest.raises(m.StageDError):
        m.validate_worker_result(worker)


@pytest.mark.parametrize("ready", [True, False])
def test_worker_validation(ready):
    m.validate_worker_result(valid_worker(ready))


def test_worker_rejects_bad_self_hash():
    worker = valid_worker()
    worker["worker_result_sha256"] = "0" * 64
    with pytest.raises(m.StageDError):
        m.validate_worker_result(worker)


def test_worker_rejects_recommendation_mismatch():
    worker = valid_worker(False)
    worker["train_only_recommendation"] = {"x": 1}
    worker["worker_result_sha256"] = m.sha256_bytes(
        m.stable_json_bytes({k: v for k, v in worker.items() if k != "worker_result_sha256"})
    )
    with pytest.raises(m.StageDError):
        m.validate_worker_result(worker)


@pytest.mark.parametrize("ready", [True, False])
def test_summary_validation(ready):
    worker = valid_worker(ready)
    summary = m.build_summary(repository={"head": "x"}, worker=worker, worker_file_sha256="a" * 64)
    m.validate_summary(summary)
    assert summary["candidate_signal_ready"] is ready


def test_summary_rejects_bad_self_hash():
    summary = m.build_summary(repository={}, worker=valid_worker(), worker_file_sha256="a" * 64)
    summary["summary_sha256"] = "0" * 64
    with pytest.raises(m.StageDError):
        m.validate_summary(summary)


@pytest.mark.parametrize(
    "worker_started,expected",
    [(False, 0), (True, None)],
)
def test_blocked_report_access_count(worker_started, expected):
    report = m.blocked_report(repository=None, error=RuntimeError("x"), worker_started=worker_started)
    assert report["objective_train_access_count_added"] is expected
    assert report["rerun_authorized"] is False


@pytest.mark.parametrize("condition", m.FORMAL_CONDITIONS)
def test_condition_names_are_fixed(condition):
    assert condition in ("free", "hidden_slack_breakaway_pin_v2")


@pytest.mark.parametrize("fold", range(6))
def test_fold_population_constant(fold):
    assert fold < m.EXPECTED_FOLDS


@pytest.mark.parametrize(
    "field",
    [
        "candidate_id",
        "feature_source",
        "model",
        "ridge_l2",
        "candidate_scale",
        "folds",
        "permutation_seed",
        "standardizer_epsilon",
        "ridge_jitter",
    ],
)
def test_candidate_spec_serializable(field):
    assert field in m.asdict(m.PRIMARY_SPEC)


@pytest.mark.parametrize(
    "field",
    [
        "aggregate_mse_ratio_max",
        "group_relative_improvement_mean_min",
        "group_relative_improvement_ci_lower_min",
        "positive_group_rate_min",
        "positive_fold_support_min",
        "condition_mse_ratio_max",
        "physical_valid_rate_min",
        "primary_vs_permutation_relative_margin_min",
        "primary_vs_global_mean_mse_ratio_margin_min",
        "minimum_candidate_movement_rms",
    ],
)
def test_gate_spec_serializable(field):
    assert field in m.asdict(m.GATE_SPEC)
