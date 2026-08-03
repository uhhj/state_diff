from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from ccda_phase3 import (
    phase314b_r259_stagef_operational_family_canonical_lock as f,
)


def recipe_population():
    values = []
    for descriptor in ("compact_v1", "compact_geometry_v2"):
        for l2 in (0.25, 1.0, 4.0):
            for temperature in (0.75, 1.0, 1.25, 1.5, 2.0):
                for shrinkage in (0.5, 0.75, 1.0):
                    for threshold in (0.25, 0.5, 0.75, 1.0):
                        values.append(
                            {
                                "descriptor_id": descriptor,
                                "risk_l2": l2,
                                "temperature": temperature,
                                "shrinkage": shrinkage,
                                "risk_threshold": threshold,
                            }
                        )
    return values


def all_checks(value=True):
    return {
        "pass": value,
        "checks": {
            "acceptance": value,
            "accepted_mse": value,
            "adverse_sse_reduction": value,
            "group_cvar_improvement": value,
            "overall_mse": value,
            "positive_reduction": value,
            "relative_reduction": value,
            "risk_brier_nonworse": value,
        },
    }


def canonical_record():
    return {
        "row_count": 638,
        "acceptance_rate": 0.6316614420062696,
        "overall_mse_ratio": 0.9211018929745592,
        "accepted_row_mse_ratio": 0.8605000982262854,
        "positive_distance_reduction_rate": 0.7146401985111662,
        "relative_distance_reduction_mean": 0.07480610639671968,
        "risk_brier_score": 0.2025145146622673,
        "risk_constant_brier_score": 0.24297551501533382,
        "output_candidate_sha256": "04dec9f03c82c4f4c97f3b7c9a8017f90741d001876ede872d53ec3a01d4ea05",
        "output_selected_scale_sha256": "967407ceab68cf55856cb9f94ff59abdd498afe8476e66470acf612d6b8e66bf",
    }


def stagee_fixture():
    return {
        "audit": {
            "selected_operational_family": {
                "family_id": f.LOCKED_FAMILY_ID,
                "recipe_population": 30,
                "eligible_recipe_count_by_fold": [30] * 6,
                "all_family_recipes_eligible_in_every_fold": True,
                "family_recipe_ids_sha256": f.EXPECTED_FAMILY_RECIPE_IDS_SHA256,
            }
        }
    }


def worker_fixture():
    population = recipe_population()
    ids = [f.recipe_from_mapping(item).recipe_id for item in population]
    eligibility = {rid: all_checks(True) for rid in ids}
    selections = [
        {
            "outer_fold": fold,
            "eligible_recipe_ids": list(ids),
            "recipe_eligibility": copy.deepcopy(eligibility),
        }
        for fold in range(6)
    ]
    records = {rid: {"row_count": 638} for rid in ids}
    records[f.CANONICAL_RECIPE_ID] = canonical_record()
    return {
        "recipe_population": population,
        "outer_fold_selections": selections,
        "full_objective_oof_fixed_recipe_records": records,
        "full_objective_oof_fixed_recipe_selection": {
            "recipe_eligibility": eligibility,
        },
        "legacy_stagec_control_contract": {
            "25": {"all_pass": True},
            "50": {"all_pass": True},
        },
    }


@pytest.mark.parametrize(
    "name,expected",
    [
        ("BASE_HEAD", "3457df7143f8585bb381fa1acb07452daf2567ef"),
        ("BASE_IMPLEMENTATION", "dbd38ee9a4389ecddd86bf673b1d6b1794fead99"),
        ("EXPECTED_SUBMODULE", "633a88752445cf5d6776ed374fdbbdb35f93050c"),
        ("LOCKED_SHRINKAGE", 1.0),
        ("LOCKED_RISK_THRESHOLD", 0.5),
        ("LOCKED_FAMILY_ID", "shrink_1.00__risk_0.50"),
        ("EXPECTED_FAMILY_RECIPE_COUNT", 30),
        ("EXPECTED_RECIPE_COUNT", 360),
        ("EXPECTED_OUTER_FOLDS", 6),
        ("CANONICAL_RECIPE_ID", "desc_compact_v1__l2_4.00__temp_1.00__shrink_1.00__risk_0.50"),
    ],
)
def test_frozen_constants(name, expected):
    assert getattr(f, name) == expected


def test_recipe_population_has_360_entries():
    assert len(recipe_population()) == 360


def test_recipe_population_has_unique_ids():
    ids = [f.recipe_from_mapping(item).recipe_id for item in recipe_population()]
    assert len(set(ids)) == 360


@pytest.mark.parametrize(
    "payload,expected",
    [
        ({"descriptor_id": "compact_v1", "risk_l2": 4, "temperature": 1, "shrinkage": 1, "risk_threshold": 0.5}, f.CANONICAL_RECIPE_ID),
        ({"descriptor_id": "compact_geometry_v2", "risk_l2": 1, "temperature": 0.75, "shrinkage": 1, "risk_threshold": 0.5}, "desc_compact_geometry_v2__l2_1.00__temp_0.75__shrink_1.00__risk_0.50"),
    ],
)
def test_recipe_id(payload, expected):
    assert f.recipe_from_mapping(payload).recipe_id == expected


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("descriptor_id", "bad", "descriptor"),
        ("risk_l2", 2.0, "L2"),
        ("temperature", 3.0, "temperature"),
        ("shrinkage", 0.6, "shrinkage"),
        ("risk_threshold", 0.6, "risk threshold"),
    ],
)
def test_recipe_rejects_outside_bank(field, value, match):
    payload = {
        "descriptor_id": "compact_v1",
        "risk_l2": 4.0,
        "temperature": 1.0,
        "shrinkage": 1.0,
        "risk_threshold": 0.5,
    }
    payload[field] = value
    with pytest.raises(f.StageFError, match=match):
        f.recipe_from_mapping(payload)


def test_operational_family_id():
    recipe = f.recipe_from_mapping(recipe_population()[10])
    assert f.operational_family_id(recipe).startswith("shrink_")


def test_canonical_order_rejects_outside_family():
    recipe = f.Recipe("compact_v1", 4.0, 1.0, 0.75, 0.5)
    with pytest.raises(f.StageFError, match="outside locked family"):
        f.canonical_order(recipe)


def test_descriptor_simplicity_precedes_geometry():
    compact = f.Recipe("compact_v1", 0.25, 2.0, 1.0, 0.5)
    geometry = f.Recipe("compact_geometry_v2", 4.0, 1.0, 1.0, 0.5)
    assert f.canonical_order(compact) < f.canonical_order(geometry)


def test_identity_temperature_precedes_scaled_temperature():
    identity = f.Recipe("compact_v1", 0.25, 1.0, 1.0, 0.5)
    scaled = f.Recipe("compact_v1", 4.0, 0.75, 1.0, 0.5)
    assert f.canonical_order(identity) < f.canonical_order(scaled)


def test_stronger_l2_breaks_equal_temperature_tie():
    strong = f.Recipe("compact_v1", 4.0, 1.0, 1.0, 0.5)
    weak = f.Recipe("compact_v1", 1.0, 1.0, 1.0, 0.5)
    assert f.canonical_order(strong) < f.canonical_order(weak)


def test_canonicalize_selects_expected_recipe():
    result = f.canonicalize_family(recipe_population())
    assert result["canonical_recipe_id"] == f.CANONICAL_RECIPE_ID
    assert result["performance_metrics_used"] is False
    assert result["family_recipe_count"] == 30


def test_canonicalize_rule_is_explicit():
    result = f.canonicalize_family(recipe_population())
    assert result["ordering_axes"] == [
        "descriptor_dimension_ascending",
        "absolute_log_temperature_ascending",
        "risk_l2_descending",
        "recipe_id_ascending",
    ]


def test_canonicalize_rejects_wrong_population():
    with pytest.raises(f.StageFError, match="exactly 360"):
        f.canonicalize_family(recipe_population()[:-1])


def test_canonicalize_rejects_duplicate_ids():
    population = recipe_population()
    population[-1] = copy.deepcopy(population[0])
    with pytest.raises(f.StageFError, match="not unique"):
        f.canonicalize_family(population)


def test_lock_canonical_recipe_passes():
    lock = f.lock_canonical_recipe(stagee_fixture(), worker_fixture())
    assert lock["canonical_lock_ready"] is True
    assert lock["outer_fold_nested_eligibility"]["all_six_folds_eligible"] is True
    assert lock["fixed_recipe_outer_oof"]["eligibility"]["pass"] is True


@pytest.mark.parametrize("fold", range(6))
def test_each_fold_is_recorded(fold):
    lock = f.lock_canonical_recipe(stagee_fixture(), worker_fixture())
    record = lock["outer_fold_nested_eligibility"]["fold_records"][fold]
    assert record == {
        "outer_fold": fold,
        "canonical_recipe_eligible": True,
        "eligible_recipe_count": 360,
    }


def test_lock_rejects_missing_canonical_fold_eligibility():
    worker = worker_fixture()
    worker["outer_fold_selections"][2]["eligible_recipe_ids"].remove(f.CANONICAL_RECIPE_ID)
    with pytest.raises(f.StageFError, match="missing from outer-fold"):
        f.lock_canonical_recipe(stagee_fixture(), worker)


def test_lock_rejects_failed_canonical_fold_check():
    worker = worker_fixture()
    worker["outer_fold_selections"][1]["recipe_eligibility"][f.CANONICAL_RECIPE_ID] = all_checks(False)
    with pytest.raises(f.StageFError, match="not eligible"):
        f.lock_canonical_recipe(stagee_fixture(), worker)


def test_lock_rejects_missing_full_record():
    worker = worker_fixture()
    del worker["full_objective_oof_fixed_recipe_records"][f.CANONICAL_RECIPE_ID]
    with pytest.raises(f.StageFError, match="record population"):
        f.lock_canonical_recipe(stagee_fixture(), worker)


def test_lock_rejects_failed_full_eligibility():
    worker = worker_fixture()
    worker["full_objective_oof_fixed_recipe_selection"]["recipe_eligibility"][f.CANONICAL_RECIPE_ID] = all_checks(False)
    with pytest.raises(f.StageFError, match="not eligible"):
        f.lock_canonical_recipe(stagee_fixture(), worker)


@pytest.mark.parametrize("timestep", [25, 50])
def test_lock_rejects_failed_frozen_control(timestep):
    worker = worker_fixture()
    worker["legacy_stagec_control_contract"][str(timestep)]["all_pass"] = False
    with pytest.raises(f.StageFError, match="frozen t{}".format(timestep)):
        f.lock_canonical_recipe(stagee_fixture(), worker)


@pytest.mark.parametrize(
    "key",
    [
        "row_count",
        "acceptance_rate",
        "overall_mse_ratio",
        "accepted_row_mse_ratio",
        "positive_distance_reduction_rate",
        "relative_distance_reduction_mean",
        "risk_brier_score",
        "risk_constant_brier_score",
        "output_candidate_sha256",
        "output_selected_scale_sha256",
    ],
)
def test_lock_rejects_canonical_record_drift(key):
    worker = worker_fixture()
    value = worker["full_objective_oof_fixed_recipe_records"][f.CANONICAL_RECIPE_ID][key]
    worker["full_objective_oof_fixed_recipe_records"][f.CANONICAL_RECIPE_ID][key] = (
        value + 1 if isinstance(value, (int, float)) else "changed"
    )
    with pytest.raises(f.StageFError, match="canonical"):
        f.lock_canonical_recipe(stagee_fixture(), worker)


def test_build_summary_is_ready():
    summary = f.build_summary({"root": "/tmp", "head": "h", "parent": "p"}, stagee_fixture(), worker_fixture())
    assert summary["scientific_status"] == "READY"
    assert summary["selected_configuration"] is None
    assert summary["train_only_recommendation"]["t10_descriptor_id"] == "compact_v1"


def test_recommendation_is_joint_timesteps():
    summary = f.build_summary({}, stagee_fixture(), worker_fixture())
    recommendation = summary["train_only_recommendation"]
    assert recommendation["timesteps"] == [10, 25, 50]
    assert recommendation["timestep_policy"] == "joint_all_timesteps_no_cherry_pick"


def test_recommendation_does_not_authorize_probe_access():
    summary = f.build_summary({}, stagee_fixture(), worker_fixture())
    assert summary["train_only_recommendation"]["frozen_probe_access_authorized_by_this_report"] is False
    assert summary["frozen_probe_accessed"] is False


def test_summary_has_zero_science_counts():
    summary = f.build_summary({}, stagee_fixture(), worker_fixture())
    assert summary["execution_counts"]["report_read_count"] == 2
    assert all(
        value == 0
        for key, value in summary["execution_counts"].items()
        if key != "report_read_count"
    )


def rehash(payload):
    payload["summary_sha256"] = f.sha256_bytes(
        f.stable_json_bytes(
            {key: value for key, value in payload.items() if key != "summary_sha256"}
        )
    )


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("scientific_status", "BLOCKED", "not READY"),
        ("selected_configuration", {"x": 1}, "final configuration"),
        ("selection_holdout_evaluation_count_added", 1, "selection holdout"),
        ("cumulative_selection_holdout_evaluation_count", 2, "cumulative"),
        ("frozen_probe_accessed", True, "frozen probe"),
        ("rerun_authorized", True, "rerun"),
    ],
)
def test_summary_validation_rejects_boundary_changes(field, value, match):
    summary = f.build_summary({}, stagee_fixture(), worker_fixture())
    summary[field] = value
    rehash(summary)
    with pytest.raises(f.StageFError, match=match):
        f.validate_summary(summary)


def test_summary_validation_rejects_bad_hash():
    summary = f.build_summary({}, stagee_fixture(), worker_fixture())
    summary["summary_sha256"] = "x" * 64
    with pytest.raises(f.StageFError, match="self-hash"):
        f.validate_summary(summary)


def test_blocked_report_preserves_boundaries():
    report = f.blocked_report(None, RuntimeError("x"))
    assert report["execution_verdict"] == "BLOCKED"
    assert report["selected_configuration"] is None
    assert report["train_only_recommendation"] is None
    assert report["frozen_probe_accessed"] is False
    assert report["rerun_authorized"] is False


def test_real_source_evidence_when_present():
    root = Path(__file__).resolve().parents[1]
    stagee_path = root / f.SOURCE_STAGEE
    worker_path = root / f.SOURCE_WORKER
    if not stagee_path.is_file() or not worker_path.is_file():
        # Package-only validation still exercises the complete synthetic schema.
        assert f.lock_canonical_recipe(stagee_fixture(), worker_fixture())["canonical_lock_ready"]
        return
    stagee, worker = f.validate_source_evidence(root)
    lock = f.lock_canonical_recipe(stagee, worker)
    assert lock["canonicalization"]["canonical_recipe_id"] == f.CANONICAL_RECIPE_ID
