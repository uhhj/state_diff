from __future__ import annotations

from copy import deepcopy

import pytest

from ccda_phase3.phase314b_r253_gate_separation import DIAGNOSTIC_OBJECTIVE_NAMES
from ccda_phase3.phase314b_r253_resume1_finalizer import (
    CORRECTION_SCHEMA,
    corrected_pilot_view,
    repair_r252_reproduction,
)


def reproduction_fixture() -> dict:
    return {
        "schema": "phase314b_r253_r252_one_step_reproduction_v1",
        "checks": {
            "branch_audit_pass": {
                "observed": True,
                "expected": True,
                "pass": True,
            },
            "composite_one_step_pass": {
                "observed": False,
                "expected": False,
                "pass": True,
            },
            "own_target_closer_fraction": {
                "observed": 0.99,
                "expected": 0.99,
                "pass": True,
            },
            "separation_ratio_p50": {
                "observed": 1.0,
                "expected": 1.0,
                "pass": True,
            },
            "branch_delta_cosine_p50": {
                "observed": 0.99,
                "expected": 0.99,
                "pass": True,
            },
            "exact_v_oracle_pass": {
                "observed": False,
                "expected": True,
                "pass": False,
            },
            "pipeline_controls_pass": {
                "observed": True,
                "expected": True,
                "pass": True,
            },
        },
        "reconstruction_metrics": {"pass": True, "checks": {}},
        "pass": False,
    }


def gate_separation_fixture() -> dict:
    return {
        "schema": "phase314b_r253_gate_separation_v1",
        "contracts": {
            "branch_transport_pass": True,
            "historical_exact_reconstruction_pass": False,
            "one_step_physical_pass": True,
            "ordered_topology_pass": True,
        },
        "full_reconstruction_components": {
            "z_state_gate_pass": False,
            "cable_geometry_gate_pass": True,
        },
        "state_group_z_metrics": {
            "robot_proxy_error_fraction": 0.8,
            "cable_error_fraction": 0.2,
        },
        "by_horizon": {
            "0": {"cable_geometry_gate_pass": True},
            "1": {"cable_geometry_gate_pass": True},
            "2": {"cable_geometry_gate_pass": True},
            "3": {"cable_geometry_gate_pass": True},
        },
        "by_source": {"groups": {"0": {"gate_pass": False}}},
    }


def variant_fixture() -> dict:
    return {
        "training_completed": True,
        "r252_one_step_reproduction": reproduction_fixture(),
        "one_step_attribution_controls": {
            "exact_v_oracle_audit": {},
            "pipeline_controls": {
                "oracle_legacy_gate_pass": True,
                "pass": True,
            },
        },
        "gate_separation": gate_separation_fixture(),
    }


def pilot_fixture() -> dict:
    return {
        "variants": {
            name: variant_fixture() for name in DIAGNOSTIC_OBJECTIVE_NAMES
        },
        "synthetic_controls": {"contract": {"pass": True}},
        "root_cause": "phase314b_r253_r252_one_step_reproduction_failed",
        "supported_mechanisms": [],
        "next_stage": "repair one-step reproduction",
        "train_only_recommendation": None,
    }


def test_repairs_only_known_missing_key_signature() -> None:
    value = repair_r252_reproduction(
        variant_fixture(), objective_name="ordered_mean_raw_g100"
    )
    reproduction = value["r252_one_step_reproduction"]
    assert reproduction["schema"] == CORRECTION_SCHEMA
    assert reproduction["pass"] is True
    check = reproduction["checks"]["exact_v_oracle_pass"]
    assert check["observed"] is True
    assert check["source"].endswith("oracle_legacy_gate_pass")
    assert check["original_adapter_check"]["observed"] is False


def test_original_input_is_not_mutated() -> None:
    original = variant_fixture()
    snapshot = deepcopy(original)
    repair_r252_reproduction(original, objective_name="v_only_frozen_control")
    assert original == snapshot


def test_rejects_explicit_real_oracle_failure() -> None:
    value = variant_fixture()
    value["r252_one_step_reproduction"]["checks"]["exact_v_oracle_pass"] = {
        "observed": False,
        "expected": True,
        "pass": False,
        "source": "explicit_producer_value",
    }
    with pytest.raises(RuntimeError, match="unexpected exact-v mismatch signature"):
        repair_r252_reproduction(value, objective_name="x")


def test_rejects_pipeline_oracle_failure() -> None:
    value = variant_fixture()
    value["one_step_attribution_controls"]["pipeline_controls"][
        "oracle_legacy_gate_pass"
    ] = False
    with pytest.raises(RuntimeError, match="oracle legacy gate did not pass"):
        repair_r252_reproduction(value, objective_name="x")


def test_rejects_pipeline_contract_failure() -> None:
    value = variant_fixture()
    value["one_step_attribution_controls"]["pipeline_controls"]["pass"] = False
    with pytest.raises(RuntimeError, match="pipeline controls did not pass"):
        repair_r252_reproduction(value, objective_name="x")


def test_rejects_missing_pipeline_controls() -> None:
    value = variant_fixture()
    value["one_step_attribution_controls"].pop("pipeline_controls")
    with pytest.raises(RuntimeError, match="pipeline_controls must be a mapping"):
        repair_r252_reproduction(value, objective_name="x")


def test_rejects_non_oracle_reproduction_failure() -> None:
    value = variant_fixture()
    value["r252_one_step_reproduction"]["checks"]["branch_audit_pass"][
        "pass"
    ] = False
    with pytest.raises(RuntimeError, match="non-oracle.*checks failed"):
        repair_r252_reproduction(value, objective_name="x")


def test_rejects_reconstruction_reproduction_failure() -> None:
    value = variant_fixture()
    value["r252_one_step_reproduction"]["reconstruction_metrics"]["pass"] = False
    with pytest.raises(RuntimeError, match="reconstruction reproduction failed"):
        repair_r252_reproduction(value, objective_name="x")


def test_corrected_pilot_recomputes_scientific_classifier() -> None:
    corrected = corrected_pilot_view(pilot_fixture())
    assert corrected["root_cause"] == (
        "phase314b_r253_robot_proxy_reconstruction_conflation_supported"
    )
    assert "robot_proxy_reconstruction_conflation" in corrected["supported_mechanisms"]
    assert corrected["train_only_recommendation"] is None
    for value in corrected["variants"].values():
        assert value["r252_one_step_reproduction"]["pass"] is True


def test_corrected_pilot_requires_complete_matrix() -> None:
    value = pilot_fixture()
    value["variants"].pop(DIAGNOSTIC_OBJECTIVE_NAMES[0])
    with pytest.raises(RuntimeError):
        corrected_pilot_view(value)


def test_wrong_historical_schema_is_rejected() -> None:
    value = variant_fixture()
    value["r252_one_step_reproduction"]["schema"] = "wrong"
    with pytest.raises(RuntimeError, match="schema mismatch"):
        repair_r252_reproduction(value, objective_name="x")
