from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path

import pytest

from ccda_phase3 import phase314b_r259_stageb_outer_crossfit_tail_failure_audit as stageb


def metric(
    *,
    acceptance: float,
    overall: float,
    accepted: float,
    positive: float,
    relative: float,
    adverse: float,
    cvar: float,
    brier_nonworse: bool = True,
):
    return {
        "row_count": 638,
        "selected_row_count": int(round(638 * acceptance)),
        "acceptance_rate": acceptance,
        "overall_mse_ratio": overall,
        "accepted_row_mse_ratio": accepted,
        "positive_distance_reduction_rate": positive,
        "relative_distance_reduction_mean": relative,
        "adverse_sse_mass": adverse,
        "beneficial_sse_mass": 0.20,
        "net_sse_reduction_mass": 0.20 - adverse,
        "risk_brier_score": 0.12,
        "risk_constant_brier_score": 0.13,
        "risk_brier_nonworse": brier_nonworse,
        "group_tail": {
            "group_count": 126,
            "tail_count": 26,
            "ratio_stats": {"mean": cvar},
            "worst_group_mse_ratio": cvar + 0.2,
            "worst_fraction_cvar_mse_ratio": cvar,
            "group_ratio_sha256": "a" * 64,
        },
        "distance_reduction_stats": {"mean": relative},
        "adverse_row_count": 20,
        "raw_accepted_row_count": 500,
        "output_candidate_sha256": "b" * 64,
        "output_selected_scale_sha256": "c" * 64,
    }


def baseline(overall: float, cvar: float = 1.20):
    return metric(
        acceptance=0.80,
        overall=overall,
        accepted=overall,
        positive=0.65,
        relative=0.02,
        adverse=0.10,
        cvar=cvar,
    )


def checks(all_pass: bool):
    result = {
        "acceptance": all_pass,
        "overall_mse": True,
        "accepted_mse": True,
        "positive_reduction": True,
        "relative_reduction": True,
        "adverse_sse_reduction": True,
        "group_cvar_improvement": True,
        "risk_brier_nonworse": True,
    }
    if not all_pass:
        result["acceptance"] = False
    return result


def selection(fold: int, policy_id: str, eligible: bool):
    modal_eligible = fold not in (1, 4)
    selected_checks = {str(t): checks(eligible) for t in stageb.EXPECTED_TIMESTEPS}
    modal_checks = {str(t): checks(modal_eligible) for t in stageb.EXPECTED_TIMESTEPS}
    policy_eligibility = {
        policy_id: {
            "all_timesteps_pass": eligible,
            "checks": selected_checks,
        },
        stageb.EXPECTED_MODAL_POLICY: {
            "all_timesteps_pass": modal_eligible,
            "checks": modal_checks,
        },
    }
    return {
        "selected_policy": {"shrinkage": 1.0, "risk_threshold": 0.5},
        "selected_policy_id": policy_id,
        "selected_policy_inner_eligible": eligible,
        "eligible_policy_ids": [policy_id] if eligible else [],
        "policy_eligibility": policy_eligibility,
        "selection_score": [0 if eligible else 1],
        "diagnostic_fallback_used": not eligible,
        "outer_fold": fold,
        "inner_train_row_count": 530,
        "outer_test_row_count": 108,
    }


def worker_payload():
    policies = [
        stageb.EXPECTED_MODAL_POLICY,
        "shrink_0.75__risk_0.25",
        stageb.EXPECTED_MODAL_POLICY,
        stageb.EXPECTED_MODAL_POLICY,
        "shrink_0.50__risk_0.25",
        stageb.EXPECTED_MODAL_POLICY,
    ]
    eligibles = [True, False, True, True, False, True]
    payload = {
        "execution_verdict": "PASS",
        "scientific_status": "BLOCKED",
        "root_cause": stageb.EXPECTED_STAGEA_ROOT,
        "required_next_path": stageb.EXPECTED_STAGEA_NEXT,
        "primary_failure_locus": stageb.EXPECTED_STAGEA_LOCUS,
        "stagex_spec": {
            "minimum_acceptance_rate": 0.5,
            "minimum_positive_reduction_rate": 0.5,
            "adverse_sse_reduction_fraction": 0.10,
        },
        "outer_crossfit_fold_selected_policy_records": {
            "10": metric(
                acceptance=0.4310,
                overall=0.9414,
                accepted=0.8458,
                positive=0.7418,
                relative=0.03,
                adverse=0.08,
                cvar=1.0266,
            ),
            "25": metric(
                acceptance=0.6599,
                overall=0.8827,
                accepted=0.8069,
                positive=0.7672,
                relative=0.04,
                adverse=0.07,
                cvar=1.0837,
            ),
            "50": metric(
                acceptance=0.5846,
                overall=0.9395,
                accepted=0.8973,
                positive=0.6568,
                relative=0.02,
                adverse=0.08,
                cvar=1.1173,
            ),
        },
        "outer_crossfit_baseline_records": {
            "10": baseline(0.9641, 1.20),
            "25": baseline(0.8711, 1.20),
            "50": baseline(0.9065, 1.20),
        },
        "outer_fold_selections": [
            selection(fold, policy_id, eligible)
            for fold, (policy_id, eligible) in enumerate(zip(policies, eligibles))
        ],
        "inner_selection_modal_policy_diagnostic": {
            "policy": {"shrinkage": 1.0, "risk_threshold": 0.5},
            "policy_id": stageb.EXPECTED_MODAL_POLICY,
            "support_count": 4,
            "counts": {
                stageb.EXPECTED_MODAL_POLICY: 4,
                "shrink_0.75__risk_0.25": 1,
                "shrink_0.50__risk_0.25": 1,
            },
        },
        "full_objective_oof_fixed_policy_selection": {
            "selected_policy": {"shrinkage": 1.0, "risk_threshold": 0.5},
            "selected_policy_id": stageb.EXPECTED_MODAL_POLICY,
            "selected_policy_inner_eligible": True,
            "eligible_policy_ids": [stageb.EXPECTED_MODAL_POLICY],
            "policy_eligibility": {},
            "diagnostic_fallback_used": False,
        },
        "execution_counts": dict(stageb.EXPECTED_EXECUTION_COUNTS),
        "selected_configuration": None,
        "train_only_recommendation": None,
        "selection_holdout_evaluation_count_added": 0,
        "cumulative_selection_holdout_evaluation_count": 1,
        "rerun_authorized": False,
    }
    payload["worker_result_sha256"] = stageb.sha256_bytes(
        stageb.stable_json_bytes(payload)
    )
    return payload


def audit_payload():
    gates = stageb.reconstruct_outer_procedure_gates(worker_payload())
    selections = stageb.audit_outer_selections(worker_payload())
    surface = stageb.audit_attribution_surface(worker_payload())
    classification = stageb.classify_audit(gates, selections, surface)
    payload = {
        "phase": stageb.PHASE,
        "schema": stageb.SCHEMA,
        "execution_verdict": "PASS",
        **classification,
        "audit_scope": {
            "evidence_only": True,
            "new_fit_count": 0,
            "new_candidate_generation_count": 0,
            "new_risk_fit_count": 0,
            "new_internal_scale_attempt_count": 0,
            "new_policy_evaluation_count": 0,
        },
        "selected_configuration": None,
        "train_only_recommendation": None,
        "selection_holdout_evaluation_count_added": 0,
        "cumulative_selection_holdout_evaluation_count": 1,
        "rerun_authorized": False,
    }
    payload.update({key: False for key in stageb.FALSE_BOUNDARIES})
    payload["audit_sha256"] = stageb.sha256_bytes(stageb.stable_json_bytes(payload))
    return payload


def test_phase_and_schema():
    assert stageb.PHASE.endswith("Stage B")
    assert stageb.SCHEMA.endswith("_v1")


def test_base_commit_chain_constants():
    assert stageb.BASE_STAGEA_EVIDENCE_COMMIT.startswith("3100a877")
    assert stageb.BASE_STAGEA_IMPLEMENTATION_COMMIT.startswith("50746104")


def test_implementation_is_add_only_three_files():
    assert len(stageb.IMPLEMENTATION_PATHS) == 3
    assert all(status == "A" for status, _ in stageb.IMPLEMENTATION_PATHS)


def test_expected_execution_counts_exact():
    assert sum(stageb.EXPECTED_EXECUTION_COUNTS.values()) == 3276
    assert stageb.EXPECTED_EXECUTION_COUNTS["nonconverged_risk_fit_count"] == 0


def test_stable_json_is_order_independent():
    assert stageb.stable_json_bytes({"b": 2, "a": 1}) == stageb.stable_json_bytes(
        {"a": 1, "b": 2}
    )


def test_sha256_bytes():
    assert stageb.sha256_bytes(b"abc") == hashlib.sha256(b"abc").hexdigest()


def test_write_once(tmp_path: Path):
    path = tmp_path / "x.json"
    stageb.write_once(path, b"{}")
    assert path.read_bytes() == b"{}"
    with pytest.raises(stageb.StageBError, match="write-once"):
        stageb.write_once(path, b"again")


@pytest.mark.parametrize("value", [True, False, float("nan"), float("inf")])
def test_finite_rejects_invalid(value):
    with pytest.raises(stageb.StageBError):
        stageb._finite(value, "x")


@pytest.mark.parametrize(
    "timestep,failed",
    [(10, ["acceptance"]), (25, []), (50, [])],
)
def test_reconstructed_gate_pattern(timestep, failed):
    result = stageb.reconstruct_outer_procedure_gates(worker_payload())
    assert result["frozen_gate_pattern"][str(timestep)] == failed


@pytest.mark.parametrize(
    "gate",
    [
        "overall_mse",
        "accepted_mse",
        "positive_reduction",
        "relative_reduction",
        "adverse_sse_reduction",
        "group_cvar_improvement",
        "risk_brier_nonworse",
    ],
)
def test_t10_all_nonacceptance_gates_pass(gate):
    result = stageb.reconstruct_outer_procedure_gates(worker_payload())
    assert result["by_timestep"]["10"]["frozen_checks"][gate] is True


def test_t10_acceptance_deficit_exact():
    result = stageb.reconstruct_outer_procedure_gates(worker_payload())
    assert result["by_timestep"]["10"]["acceptance_deficit"] == pytest.approx(0.069)


def test_t10_accepted_subset_is_high_fidelity():
    result = stageb.reconstruct_outer_procedure_gates(worker_payload())
    assert result["by_timestep"]["10"]["accepted_subset_fidelity_gain"] == pytest.approx(
        0.1542
    )


def test_t25_and_t50_pass_all_checks():
    result = stageb.reconstruct_outer_procedure_gates(worker_payload())
    assert result["by_timestep"]["25"]["all_checks_pass"] is True
    assert result["by_timestep"]["50"]["all_checks_pass"] is True


def test_wrong_gate_pattern_is_rejected():
    value = worker_payload()
    value["outer_crossfit_fold_selected_policy_records"]["25"]["acceptance_rate"] = 0.4
    with pytest.raises(stageb.StageBError, match="gate pattern"):
        stageb.reconstruct_outer_procedure_gates(value)


def test_outer_selection_fallback_folds():
    result = stageb.audit_outer_selections(worker_payload())
    assert result["fallback_folds"] == [1, 4]


def test_outer_selection_modal_policy():
    result = stageb.audit_outer_selections(worker_payload())
    assert result["modal_policy_id"] == stageb.EXPECTED_MODAL_POLICY
    assert result["modal_policy_support_count"] == 4


def test_outer_selection_eligible_count():
    result = stageb.audit_outer_selections(worker_payload())
    assert result["eligible_selection_fold_count"] == 4
    assert result["fallback_fold_count"] == 2


def test_full_objective_surface_role_is_not_validation():
    result = stageb.audit_outer_selections(worker_payload())
    assert result["full_objective_oof_selected_policy_eligible"] is True
    assert result["full_objective_surface_role"].startswith("tuning_only")


def test_wrong_fallback_population_is_rejected():
    value = worker_payload()
    value["outer_fold_selections"][4] = selection(
        4, stageb.EXPECTED_MODAL_POLICY, True
    )
    with pytest.raises(stageb.StageBError, match="fallback-fold"):
        stageb.audit_outer_selections(value)


def test_attribution_surface_is_aggregate_only():
    result = stageb.audit_attribution_surface(worker_payload())
    assert result["aggregate_timestep_metrics_available"] is True
    assert result["outer_test_metrics_by_fold_available"] is False
    assert result["group_identity_or_ratio_table_available"] is False


def test_fallback_fold_causality_not_claimed():
    result = stageb.audit_attribution_surface(worker_payload())
    assert result["fallback_folds_can_be_identified_from_inner_selection"] is True
    assert result["fallback_folds_can_be_claimed_as_outer_test_failure_causes"] is False


def test_unexpected_fold_metrics_are_rejected():
    value = worker_payload()
    value["outer_fold_test_metrics"] = {}
    with pytest.raises(stageb.StageBError, match="fold-resolved"):
        stageb.audit_attribution_surface(value)


def test_classification_requires_new_stagec():
    worker = worker_payload()
    result = stageb.classify_audit(
        stageb.reconstruct_outer_procedure_gates(worker),
        stageb.audit_outer_selections(worker),
        stageb.audit_attribution_surface(worker),
    )
    assert result["scientific_status"] == "BLOCKED"
    assert result["required_next_path"].startswith("PREREGISTER_R259_STAGEC")


def test_classification_does_not_authorize_threshold_relaxation():
    worker = worker_payload()
    result = stageb.classify_audit(
        stageb.reconstruct_outer_procedure_gates(worker),
        stageb.audit_outer_selections(worker),
        stageb.audit_attribution_surface(worker),
    )
    assert "threshold" not in result["required_next_path"].lower()


def test_validate_audit_accepts_exact_payload():
    stageb.validate_audit(audit_payload())


def test_validate_audit_rejects_recommendation():
    value = audit_payload()
    value["train_only_recommendation"] = {"x": 1}
    value["audit_sha256"] = stageb.sha256_bytes(
        stageb.stable_json_bytes({k: v for k, v in value.items() if k != "audit_sha256"})
    )
    with pytest.raises(stageb.StageBError, match="recommendation"):
        stageb.validate_audit(value)


def test_validate_audit_rejects_new_fit():
    value = audit_payload()
    value["audit_scope"]["new_fit_count"] = 1
    value["audit_sha256"] = stageb.sha256_bytes(
        stageb.stable_json_bytes({k: v for k, v in value.items() if k != "audit_sha256"})
    )
    with pytest.raises(stageb.StageBError, match="performed science"):
        stageb.validate_audit(value)


def test_validate_audit_rejects_holdout_access():
    value = audit_payload()
    value["selection_holdout_evaluation_count_added"] = 1
    value["audit_sha256"] = stageb.sha256_bytes(
        stageb.stable_json_bytes({k: v for k, v in value.items() if k != "audit_sha256"})
    )
    with pytest.raises(stageb.StageBError, match="selection holdout"):
        stageb.validate_audit(value)


def test_validate_worker_rejects_wrong_counts(monkeypatch):
    value = worker_payload()
    monkeypatch.setattr(stageb, "EXPECTED_WORKER_RESULT_SHA256", value["worker_result_sha256"])
    value["execution_counts"]["direction_fit_count"] = 109
    value["worker_result_sha256"] = stageb.sha256_bytes(
        stageb.stable_json_bytes({k: v for k, v in value.items() if k != "worker_result_sha256"})
    )
    monkeypatch.setattr(stageb, "EXPECTED_WORKER_RESULT_SHA256", value["worker_result_sha256"])
    with pytest.raises(stageb.StageBError, match="execution counts"):
        stageb.validate_worker_evidence(value)


def test_blocked_report_preserves_boundaries():
    result = stageb.blocked_report(None, RuntimeError("x"))
    assert result["execution_verdict"] == "BLOCKED"
    assert result["rerun_authorized"] is False
    assert result["selection_holdout_evaluation_count_added"] == 0
    assert all(result[key] is False for key in stageb.FALSE_BOUNDARIES)


def test_false_boundaries_are_unique():
    assert len(stageb.FALSE_BOUNDARIES) == len(set(stageb.FALSE_BOUNDARIES))


def test_source_hash_population_matches_stagea_paths():
    assert set(stageb.BASE_STAGEA_SOURCE_SHA256) == {
        path for _, path in stageb.BASE_STAGEA_IMPLEMENTATION_PATHS
    }
