from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ccda_phase3 import phase314b_r259_stagej_prevalence_anchored_risk_repair as j


@pytest.mark.parametrize(
    "name,expected",
    [
        ("BASE_HEAD", "ca5bbe74154e0ba239f5d499c8a96d15b860b902"),
        ("BASE_IMPLEMENTATION", "8def374d4398786b068992ec1830c9b2945d04e2"),
        ("EXPECTED_SUBMODULE", "633a88752445cf5d6776ed374fdbbdb35f93050c"),
        ("STAGEI_REPORT_SHA256", "5943a506ca5ccaf0045e8c2ec75df83aca3c434920cda16cbc4c21a4401de63f"),
        ("STAGEI_SELF_SHA256", "9a2ad6a69f2c6c68a1c50b3786979d69965da6a7fdf7cbe7bcd2c07cd3e09bf8"),
        ("STAGEI_SEAL_INTERNAL_SHA256", "31bc307dbe62431a9e4c602ce10a1cea384b66ef3cd12dba80a4d34692e2b083"),
        ("STAGEI_WINDOW_NPZ_SHA256", "b355febefee93de7065e71daedb4ed7f5c6c74948743b727c7675557560a586d"),
        ("STAGEI_WINDOW_ROWS", 1270),
        ("STAGEI_PAIR_GROUPS", 128),
        ("EXPECTED_OBJECTIVE_ROWS", 638),
        ("EXPECTED_OBJECTIVE_GROUPS", 126),
        ("LOCKED_DESCRIPTOR", "compact_v1"),
        ("LOCKED_SHRINKAGE", 1.0),
        ("LOCKED_THRESHOLD", 0.5),
        ("MINIMUM_BRIER_FOLD_SUPPORT", 5),
        ("MINIMUM_POLICY_FOLD_SUPPORT", 4),
    ],
)
def test_frozen_constants(name, expected):
    assert getattr(j, name) == expected


def test_locked_timestep_and_l2_contract():
    assert j.LOCKED_TIMESTEPS == (10, 25, 50)
    assert j.RISK_L2_BY_TIMESTEP == {10: 4.0, 25: 1.0, 50: 1.0}


@pytest.mark.parametrize("timestep", [10, 25, 50])
def test_expected_candidate_identity_has_all_axes(timestep):
    assert set(j.EXPECTED_STAGEG_IDENTITIES[timestep]) == {
        "candidate",
        "selected_scale",
        "descriptor",
    }
    assert all(len(value) == 64 for value in j.EXPECTED_STAGEG_IDENTITIES[timestep].values())


def test_repair_spec_validates():
    j.RepairSpec().validate()


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("minimum_brier_fold_support", 4, "Brier"),
        ("minimum_policy_fold_support", 3, "policy"),
        ("minimum_acceptance_rate", 0.4, "acceptance"),
        ("minimum_positive_reduction_rate", 0.4, "positive"),
        ("adverse_sse_reduction_fraction", 0.2, "adverse"),
        ("group_cvar_fraction", 0.3, "CVaR"),
    ],
)
def test_repair_spec_rejects_contract_changes(field, value, match):
    values = vars(j.RepairSpec()).copy()
    values[field] = value
    with pytest.raises(j.StageJError, match=match):
        j.RepairSpec(**values).validate()


def test_anchored_probability_identity_at_alpha_one():
    raw = np.asarray([0.1, 0.3, 0.8, 0.9])
    assert np.allclose(j.anchored_probability(raw, 0.4, 1.0), raw, rtol=0.0, atol=1e-15)


def test_anchored_probability_constant_at_alpha_zero():
    raw = np.asarray([0.1, 0.3, 0.8, 0.9])
    assert np.array_equal(j.anchored_probability(raw, 0.4, 0.0), np.full(4, 0.4))


def test_anchored_probability_linear_shrinkage():
    raw = np.asarray([0.0, 0.25, 0.75, 1.0])
    expected = np.asarray([0.25, 0.375, 0.625, 0.75])
    assert np.allclose(j.anchored_probability(raw, 0.5, 0.5), expected)


@pytest.mark.parametrize("anchor,alpha", [(-0.1, 0.5), (1.1, 0.5), (0.5, -0.1), (0.5, 1.1)])
def test_anchored_probability_rejects_invalid_parameters(anchor, alpha):
    with pytest.raises(j.StageJError, match="outside"):
        j.anchored_probability(np.asarray([0.5]), anchor, alpha)


def test_anchored_probability_rejects_nonfinite_input():
    with pytest.raises(j.StageJError, match="non-finite"):
        j.anchored_probability(np.asarray([np.nan]), 0.5, 0.5)


def test_reliability_slope_recovers_known_shrinkage():
    labels = np.asarray([0.0, 0.0, 1.0, 1.0])
    # raw deviations are twice the label deviations around q=.5 -> alpha=.5
    raw = np.asarray([0.0, 0.0, 1.0, 1.0])
    result = j.reliability_slope(raw, labels, 0.5, np.ones(4, dtype=bool))
    assert result["alpha"] == pytest.approx(1.0)
    assert result["row_count"] == 4
    assert result["anchor"] == 0.5


def test_reliability_slope_clips_negative_to_zero():
    labels = np.asarray([0.0, 0.0, 1.0, 1.0])
    raw = 1.0 - labels
    result = j.reliability_slope(raw, labels, 0.5, np.ones(4, dtype=bool))
    assert result["alpha"] == 0.0
    assert result["unconstrained_alpha"] < 0.0


def test_reliability_slope_degenerate_returns_zero():
    labels = np.asarray([0.0, 1.0, 0.0, 1.0])
    raw = np.full(4, 0.5)
    result = j.reliability_slope(raw, labels, 0.5, np.ones(4, dtype=bool))
    assert result["alpha"] == 0.0
    assert result["denominator"] == 0.0


@pytest.mark.parametrize(
    "raw,labels,mask,match",
    [
        (np.asarray([0.2]), np.asarray([0.0, 1.0]), np.asarray([True]), "population"),
        (np.asarray([0.2]), np.asarray([0.0]), np.asarray([False]), "no rows"),
    ],
)
def test_reliability_slope_rejects_bad_population(raw, labels, mask, match):
    with pytest.raises(j.StageJError, match=match):
        j.reliability_slope(raw, labels, 0.5, mask)


def test_brier_known_value():
    p = np.asarray([0.0, 0.5, 1.0])
    y = np.asarray([0.0, 1.0, 1.0])
    assert j.brier(p, y, np.ones(3, dtype=bool)) == pytest.approx(1.0 / 12.0)


@pytest.mark.parametrize(
    "p,y,mask",
    [
        (np.zeros(2), np.zeros(3), np.ones(2, dtype=bool)),
        (np.zeros(2), np.zeros(2), np.zeros(2, dtype=bool)),
    ],
)
def test_brier_rejects_invalid_population(p, y, mask):
    with pytest.raises(j.StageJError, match="population"):
        j.brier(p, y, mask)


def test_condition_brier_requires_both_conditions():
    p = np.asarray([0.2, 0.8])
    y = np.asarray([0.0, 1.0])
    with pytest.raises(j.StageJError, match="hidden_slack"):
        j._condition_brier(
            p,
            y,
            np.ones(2, dtype=bool),
            np.asarray(["free", "free"]),
            np.full(2, 0.5),
        )


def test_condition_brier_reports_nonworse():
    p = np.asarray([0.1, 0.9, 0.2, 0.8])
    y = np.asarray([0.0, 1.0, 0.0, 1.0])
    names = np.asarray(["free", "free", "hidden_slack_breakaway_pin_v2", "hidden_slack_breakaway_pin_v2"])
    result = j._condition_brier(p, y, np.ones(4, dtype=bool), names, np.full(4, 0.5))
    assert all(item["nonworse"] for item in result.values())


class FakeStageX:
    class StageXSpec:
        pass

    @staticmethod
    def predict_risk(model, descriptor):
        x = np.asarray(descriptor, dtype=np.float64)
        base = np.clip(x[:, 0], 0.0, 1.0)
        # Deliberately over-dispersed around prevalence so reliability shrinkage helps.
        q = float(model["prevalence"])
        return np.clip(q + 1.8 * (base - q), 0.0, 1.0)

    @staticmethod
    def evaluate_policy(control, candidate, selected_scale, risk_probability, target, groups, threshold, prevalence, spec):
        risk = np.asarray(risk_probability)
        accepted = (np.asarray(selected_scale) > 0) & (risk <= float(threshold))
        count = int(np.sum(accepted))
        rows = len(risk)
        return {
            "row_count": rows,
            "selected_row_count": count,
            "acceptance_rate": count / float(rows),
            "overall_mse_ratio": 0.9,
            "accepted_row_mse_ratio": 0.8,
            "positive_distance_reduction_rate": 0.75,
            "relative_distance_reduction_mean": 0.1,
            "adverse_sse_mass": 0.1,
            "beneficial_sse_mass": 0.3,
            "net_sse_reduction_mass": 0.2,
            "risk_brier_score": 0.1,
            "risk_constant_brier_score": 0.25,
            "group_tail": {"worst_fraction_cvar_mse_ratio": 0.95},
        }


class FakeStaged:
    stagex = FakeStageX()

    class StageDSpec:
        pass

    def __init__(self):
        self.fit_calls = []

    def fit_risk_model_checked(self, descriptor, labels, fit_mask, l2):
        use = np.asarray(fit_mask, dtype=bool)
        q = float(np.mean(np.asarray(labels)[use]))
        identity = j.sha256_bytes(j.stable_json_bytes({"rows": int(np.sum(use)), "q": q, "l2": l2}))
        self.fit_calls.append((int(np.sum(use)), float(l2)))
        return {"prevalence": q, "identity": identity}

    @staticmethod
    def recipe_is_eligible(record, baseline, spec):
        checks = {
            "acceptance": True,
            "accepted_mse": True,
            "adverse_sse_reduction": True,
            "group_cvar_improvement": True,
            "overall_mse": True,
            "positive_reduction": True,
            "relative_reduction": True,
            "risk_brier_nonworse": True,
        }
        return {"pass": True, "checks": checks}


def synthetic_nested_inputs():
    rows = j.EXPECTED_OBJECTIVE_ROWS
    folds = np.arange(rows) % 6
    # balanced, signal-bearing labels in every fold and condition
    labels = ((np.arange(rows) * 7 + folds) % 11 < 5).astype(np.float64)
    signal = np.where(labels > 0.5, 0.75, 0.25)
    # inject deterministic label noise so raw scores are not perfect
    noise = ((np.arange(rows) % 13) - 6) * 0.015
    descriptor = np.zeros((rows, 17), dtype=np.float64)
    descriptor[:, 0] = np.clip(signal + noise, 0.01, 0.99)
    risk_fit_mask = np.ones(rows, dtype=bool)
    control = np.zeros((rows, 4, 48), dtype=np.float32)
    candidate = np.ones((rows, 4, 48), dtype=np.float32) * 0.01
    selected_scale = np.ones(rows, dtype=np.float64)
    target = np.ones((rows, 4, 48), dtype=np.float32) * 0.02
    groups = np.asarray([f"group_{i // 5:03d}" for i in range(rows)])
    names = np.asarray(
        [j.EXPECTED_CONDITIONS[i % 2] for i in range(rows)]
    )
    return descriptor, labels, risk_fit_mask, folds, control, candidate, selected_scale, target, groups, names


def test_nested_repair_has_exact_fit_count_and_fold_population():
    staged = FakeStaged()
    result = j.nested_repair_for_timestep(
        timestep=10,
        staged=staged,
        descriptor=synthetic_nested_inputs()[0],
        labels=synthetic_nested_inputs()[1],
        risk_fit_mask=synthetic_nested_inputs()[2],
        folds=synthetic_nested_inputs()[3],
        control=synthetic_nested_inputs()[4],
        candidate=synthetic_nested_inputs()[5],
        selected_scale=synthetic_nested_inputs()[6],
        target=synthetic_nested_inputs()[7],
        groups=synthetic_nested_inputs()[8],
        condition_name=synthetic_nested_inputs()[9],
        spec=j.RepairSpec(),
    )
    assert result["fit_count"] == 22
    assert len(result["outer_records"]) == 6
    assert len(result["final_fold_slopes"]) == 6
    assert len(staged.fit_calls) == 22
    assert all(record["alpha"] >= 0.0 for record in result["outer_records"])
    assert result["final_alpha"] >= 0.0


@pytest.mark.parametrize("timestep,expected_l2", [(10, 4.0), (25, 1.0), (50, 1.0)])
def test_nested_repair_uses_locked_l2(timestep, expected_l2):
    inputs = synthetic_nested_inputs()
    staged = FakeStaged()
    result = j.nested_repair_for_timestep(
        timestep=timestep,
        staged=staged,
        descriptor=inputs[0], labels=inputs[1], risk_fit_mask=inputs[2], folds=inputs[3],
        control=inputs[4], candidate=inputs[5], selected_scale=inputs[6], target=inputs[7],
        groups=inputs[8], condition_name=inputs[9], spec=j.RepairSpec(),
    )
    assert result["risk_l2"] == expected_l2
    assert all(call[1] == expected_l2 for call in staged.fit_calls)


def test_nested_repair_rejects_wrong_row_population():
    inputs = list(synthetic_nested_inputs())
    inputs = [value[:-1] if isinstance(value, np.ndarray) else value for value in inputs]
    with pytest.raises(j.StageJError, match="row population"):
        j.nested_repair_for_timestep(
            timestep=10, staged=FakeStaged(), descriptor=inputs[0], labels=inputs[1],
            risk_fit_mask=inputs[2], folds=inputs[3], control=inputs[4], candidate=inputs[5],
            selected_scale=inputs[6], target=inputs[7], groups=inputs[8], condition_name=inputs[9],
            spec=j.RepairSpec(),
        )


def test_objective_only_adapter_source_forbids_historical_holdout_runtime_calls():
    source = Path(j.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_objective_only_control_and_context")
    calls = []
    for node in ast.walk(function):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                calls.append(node.func.attr)
            elif isinstance(node.func, ast.Name):
                calls.append(node.func.id)
    assert "build_audit_context" not in calls
    assert "replay_stagec_control_portable" not in calls
    assert "capture_portable_control_model" not in calls


def test_stagei_validation_source_does_not_open_fresh_npz_or_raw():
    source = Path(j.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "validate_stagei_contract")
    rendered = ast.unparse(function)
    assert "np.load" not in rendered
    assert "pickle" not in rendered
    assert "fresh_eval_windows.npz" not in rendered


def minimal_stagei_report():
    seal = {
        "dataset_relative_root": j.STAGEI_DATA_ROOT,
        "seal_sha256": j.STAGEI_SEAL_INTERNAL_SHA256,
        "window_manifest_sha256": j.STAGEI_WINDOW_MANIFEST_SHA256,
        "window_npz_sha256": j.STAGEI_WINDOW_NPZ_SHA256,
        "window_row_count": j.STAGEI_WINDOW_ROWS,
        "window_pair_group_count": j.STAGEI_PAIR_GROUPS,
        "governance": {
            "generated_before_new_risk_repair_search": True,
            "targets_not_used_for_fit_selection_or_evaluation": True,
            "future_evaluation_must_be_one_shot": True,
            "future_evaluation_requires_independently_locked_policy": True,
            "post_evaluation_retuning_forbidden": True,
            "fallback_after_evaluation_forbidden": True,
            "future_evaluation_count": 0,
        },
    }
    report = {
        "schema": "phase314b_r259_stagei_fresh_untouched_evaluation_seal_v1",
        "execution_verdict": "PASS",
        "scientific_status": "READY",
        "summary_sha256": j.STAGEI_SELF_SHA256,
        "selected_configuration": None,
        "train_only_recommendation": None,
        "fresh_evaluation_count_added": 0,
        "cumulative_fresh_evaluation_count": 0,
        "rerun_authorized": False,
        "fresh_evaluation_seal": seal,
    }
    return report, seal


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(j.stable_json_bytes(value))


def test_validate_stagei_contract_metadata_only(monkeypatch, tmp_path):
    report, seal_summary = minimal_stagei_report()
    # Replace identities with self-consistent temporary identities.
    seal = {
        "schema": "seal",
        "inventory_sha256": "i" * 64,
        "seal_sha256": "",
    }
    seal["seal_sha256"] = j._self_hash(seal, "seal_sha256")
    manifest = {
        "row_count": j.STAGEI_WINDOW_ROWS,
        "window_npz_sha256": j.STAGEI_WINDOW_NPZ_SHA256,
    }
    inventory = {"inventory_sha256": "i" * 64}
    write_json(tmp_path / j.STAGEI_SEAL, seal)
    write_json(tmp_path / j.STAGEI_WINDOW_MANIFEST, manifest)
    report["fresh_evaluation_seal"]["seal_sha256"] = seal["seal_sha256"]
    report["fresh_evaluation_seal"]["window_manifest_sha256"] = j.sha256_file(tmp_path / j.STAGEI_WINDOW_MANIFEST)
    report["summary_sha256"] = j._self_hash(report, "summary_sha256")
    write_json(tmp_path / j.STAGEI_REPORT, report)
    write_json(tmp_path / j.STAGEI_INVENTORY, inventory)
    monkeypatch.setattr(j, "STAGEI_REPORT_SHA256", j.sha256_file(tmp_path / j.STAGEI_REPORT))
    monkeypatch.setattr(j, "STAGEI_SELF_SHA256", report["summary_sha256"])
    monkeypatch.setattr(j, "STAGEI_SEAL_INTERNAL_SHA256", seal["seal_sha256"])
    monkeypatch.setattr(j, "STAGEI_WINDOW_MANIFEST_SHA256", j.sha256_file(tmp_path / j.STAGEI_WINDOW_MANIFEST))
    result = j.validate_stagei_contract(tmp_path)
    assert result["fresh_npz_opened"] is False
    assert result["fresh_raw_episode_opened"] is False


def test_validate_stagei_contract_rejects_evaluated_fresh_set(monkeypatch, tmp_path):
    report, _ = minimal_stagei_report()
    report["fresh_evaluation_seal"]["governance"]["future_evaluation_count"] = 1
    report["summary_sha256"] = j._self_hash(report, "summary_sha256")
    write_json(tmp_path / j.STAGEI_REPORT, report)
    monkeypatch.setattr(j, "STAGEI_REPORT_SHA256", j.sha256_file(tmp_path / j.STAGEI_REPORT))
    monkeypatch.setattr(j, "STAGEI_SELF_SHA256", report["summary_sha256"])
    with pytest.raises(j.StageJError, match="already evaluated"):
        j.validate_stagei_contract(tmp_path)


def minimal_timestep_record(all_pass=True):
    return {
        "gates": {"all": all_pass},
        "final_alpha": 0.5,
        "full_model_identity": "m" * 64,
    }


def minimal_worker(ready=True):
    records = {str(t): minimal_timestep_record(ready) for t in j.LOCKED_TIMESTEPS}
    payload = {
        "schema": j.WORKER_SCHEMA,
        "execution_verdict": "PASS",
        "scientific_status": "READY" if ready else "BLOCKED",
        "timestep_records": records,
        "train_only_recommendation": {"x": 1} if ready else None,
        "selected_configuration": None,
        "selection_holdout_evaluation_count_added": 0,
        "cumulative_selection_holdout_evaluation_count": 1,
        "frozen_probe_evaluation_count_added": 0,
        "cumulative_frozen_probe_evaluation_count": 1,
        "fresh_evaluation_count_added": 0,
        "cumulative_fresh_evaluation_count": 0,
        "rerun_authorized": False,
        "execution_counts": dict(j.EXPECTED_EXECUTION_COUNTS),
    }
    payload.update({key: False for key in j.FALSE_BOUNDARIES})
    payload["worker_result_sha256"] = j._self_hash(payload, "worker_result_sha256")
    return payload


@pytest.mark.parametrize("ready", [True, False])
def test_validate_worker_accepts_consistent_terminal_states(ready):
    j.validate_worker(minimal_worker(ready))


def rehash_worker(payload):
    payload["worker_result_sha256"] = j._self_hash(payload, "worker_result_sha256")


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("selected_configuration", {"bad": True}, "final configuration"),
        ("fresh_evaluation_count_added", 1, "evaluation count"),
        ("cumulative_fresh_evaluation_count", 1, "evaluation count"),
        ("cumulative_selection_holdout_evaluation_count", 2, "selection holdout"),
        ("cumulative_frozen_probe_evaluation_count", 2, "frozen probe"),
        ("rerun_authorized", True, "rerun"),
    ],
)
def test_validate_worker_rejects_governance_changes(field, value, match):
    payload = minimal_worker(True)
    payload[field] = value
    rehash_worker(payload)
    with pytest.raises(j.StageJError, match=match):
        j.validate_worker(payload)


def test_validate_worker_rejects_execution_count_change():
    payload = minimal_worker(True)
    payload["execution_counts"]["objective_risk_fit_count"] = 65
    rehash_worker(payload)
    with pytest.raises(j.StageJError, match="execution counts"):
        j.validate_worker(payload)


def test_validate_worker_rejects_forbidden_boundary():
    payload = minimal_worker(True)
    payload["fresh_window_npz_opened"] = True
    rehash_worker(payload)
    with pytest.raises(j.StageJError, match="fresh_window_npz_opened"):
        j.validate_worker(payload)


def minimal_summary():
    payload = {
        "schema": j.SCHEMA,
        "execution_verdict": "PASS",
        "scientific_status": "READY",
        "selected_configuration": None,
        "fresh_evaluation_count_added": 0,
        "cumulative_fresh_evaluation_count": 0,
    }
    payload.update({key: False for key in j.FALSE_BOUNDARIES})
    payload["summary_sha256"] = j._self_hash(payload, "summary_sha256")
    return payload


def test_validate_summary_accepts_minimal_ready():
    j.validate_summary(minimal_summary())


def test_validate_summary_rejects_fresh_access():
    payload = minimal_summary()
    payload["fresh_target_indexed"] = True
    payload["summary_sha256"] = j._self_hash(payload, "summary_sha256")
    with pytest.raises(j.StageJError, match="forbidden"):
        j.validate_summary(payload)


def test_blocked_report_never_authorizes_reexecution_or_data_access():
    report = j.blocked_report(None, RuntimeError("x"), worker_started=True)
    assert report["execution_verdict"] == "BLOCKED"
    assert report["science_reexecution_authorized"] is False
    assert report["rerun_authorized"] is False
    assert report["selected_configuration"] is None
    assert report["train_only_recommendation"] is None
    assert all(report[key] is False for key in j.FALSE_BOUNDARIES)


def test_latest_repository_bundle_has_expected_stagei_report_when_present():
    root = Path(__file__).resolve().parents[1]
    report = root / j.STAGEI_REPORT
    if not report.is_file():
        pytest.skip("package-only validation")
    assert j.sha256_file(report) == j.STAGEI_REPORT_SHA256
    value = json.loads(report.read_text())
    assert value["summary_sha256"] == j.STAGEI_SELF_SHA256
    assert value["fresh_evaluation_count_added"] == 0
    assert value["scientific_status"] == "READY"
