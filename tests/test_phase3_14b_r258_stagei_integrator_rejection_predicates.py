from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ccda_phase3 import phase314b_r258_stagei_integrator_rejection_predicates as stagei


def test_stagei_population_is_frozen() -> None:
    assert len(stagei.BASE_DIRECTION_IDS) == 9
    assert stagei.SCALE_MULTIPLIERS == (0.25, 0.5, 1.0, 2.0)
    assert stagei.SOURCE_IDS == (
        "oof_surrogate",
        "raw_oracle",
        "projected_oracle",
    )
    assert stagei.EXPECTED_SELECTED_ULP_FACTOR == 1.0


def test_stagei_spec_validates() -> None:
    spec = stagei.IntegratorRejectionPredicateSpec()
    spec.validate()
    assert spec.grouped_cv_folds == 6
    assert spec.maximum_leaf_count == 128
    assert spec.oracle_acceptance_rate_min == 0.95


def test_predicate_family_maps_retention() -> None:
    assert stagei.predicate_family("$.direction_retention") == "direction_retention"


def test_predicate_family_maps_segment_sides() -> None:
    assert stagei.predicate_family("$.gates.lower_segment_pass") == (
        "lower_segment_geometry"
    )
    assert stagei.predicate_family("$.gates.upper_length_valid") == (
        "upper_segment_geometry"
    )


def test_predicate_family_maps_final_acceptance() -> None:
    assert stagei.predicate_family("$.gates.final_pass") == "final_acceptance"


def test_predicate_leaves_extract_boolean_pass_rate() -> None:
    integration = {
        "candidate": np.ones((2, 1, 2)),
        "gates": {"valid": np.asarray([False, True])},
    }
    result = stagei.predicate_leaves(
        integration,
        retention_minimum=0.25,
        spec=stagei.IntegratorRejectionPredicateSpec(),
    )
    leaf = next(item for item in result if item["path"] == "$.gates.valid")
    assert leaf["pass_rate"] == pytest.approx(0.5)
    assert leaf["pass_rate_rule"] == "boolean_true"
    assert not any(item["path"].startswith("$.candidate") for item in result)


def test_predicate_leaves_apply_retention_minimum() -> None:
    integration = {"direction_retention": np.asarray([0.1, 0.3, 0.5])}
    result = stagei.predicate_leaves(
        integration,
        retention_minimum=0.25,
        spec=stagei.IntegratorRejectionPredicateSpec(),
    )
    assert result[0]["pass_rate"] == pytest.approx(2.0 / 3.0)
    assert result[0]["pass_rate_rule"] == "retention_ge_minimum"


def test_predicate_leaves_apply_positive_scale_rule() -> None:
    integration = {"selected_scale": np.asarray([0.0, 0.5, 1.0])}
    result = stagei.predicate_leaves(
        integration,
        retention_minimum=0.25,
        spec=stagei.IntegratorRejectionPredicateSpec(),
    )
    assert result[0]["pass_rate"] == pytest.approx(2.0 / 3.0)
    assert result[0]["family"] == "scale_selection"


def test_family_summary_detects_failing_leaf() -> None:
    leaves = [
        {
            "path": "$.retention_pass",
            "family": "direction_retention",
            "pass_rate": 0.0,
            "pass_rate_rule": "boolean_true",
        }
    ]
    source_index = {
        "family_identifiers": {
            "direction_retention": [{"identifier": "retention_pass", "line": 10}]
        }
    }
    result = stagei.summarize_predicate_families(
        leaves,
        source_index=source_index,
        spec=stagei.IntegratorRejectionPredicateSpec(),
    )
    assert result["direction_retention"]["minimum_pass_rate"] == 0.0
    assert result["direction_retention"]["source_line_minimum"] == 10
    assert len(result["direction_retention"]["failing_leaves"]) == 1


def test_direction_comparison_identical_is_one() -> None:
    value = np.asarray([[[1.0, 2.0]], [[2.0, 1.0]]])
    result = stagei.direction_comparison(value, value, epsilon=1.0e-14)
    assert result["cosine"]["minimum"] == pytest.approx(1.0)
    assert result["norm_ratio"]["mean"] == pytest.approx(1.0)


def _internal(retention_rate: float, selected: float = 0.0) -> dict:
    return {
        "selected_bank": np.asarray([[selected]]),
        "motion_bank": np.asarray([[selected > 0.0]]),
        "family_rates": {
            family: ([retention_rate] if family == "direction_retention" else [1.0])
            for family in stagei.PREDICATE_FAMILY_ORDER
        },
    }


def _summary(acceptance: float) -> dict:
    return {
        "scale_bank_acceptance": {
            "rows_with_any_accepted_multiplier_rate": acceptance,
        }
    }


def test_compare_identifies_retention_discriminator() -> None:
    result = stagei.compare_rejection_predicates(
        oof_summary=_summary(0.0),
        oof_internal=_internal(0.0),
        raw_summary=_summary(1.0),
        raw_internal=_internal(1.0, selected=1.0),
        projected_summary=_summary(1.0),
        projected_internal=_internal(1.0, selected=1.0),
        spec=stagei.IntegratorRejectionPredicateSpec(),
    )
    assert result["locus"] == "oof_rejected_at_direction_retention"
    assert result["discriminator_family"] == "direction_retention"
    assert result["comparator_source"] == "projected_oracle"


def test_compare_blocks_when_oracle_controls_fail() -> None:
    result = stagei.compare_rejection_predicates(
        oof_summary=_summary(0.0),
        oof_internal=_internal(0.0),
        raw_summary=_summary(0.0),
        raw_internal=_internal(0.0),
        projected_summary=_summary(0.0),
        projected_internal=_internal(0.0),
        spec=stagei.IntegratorRejectionPredicateSpec(),
    )
    assert result["locus"] == "oracle_control_not_admitted"


def test_compare_detects_stageh_replay_change() -> None:
    result = stagei.compare_rejection_predicates(
        oof_summary=_summary(0.5),
        oof_internal=_internal(1.0, selected=1.0),
        raw_summary=_summary(1.0),
        raw_internal=_internal(1.0, selected=1.0),
        projected_summary=_summary(1.0),
        projected_internal=_internal(1.0, selected=1.0),
        spec=stagei.IntegratorRejectionPredicateSpec(),
    )
    assert result["locus"] == "stageh_rejection_not_reproduced"


def test_compare_detects_zero_selection_after_predicate_pass() -> None:
    result = stagei.compare_rejection_predicates(
        oof_summary=_summary(0.0),
        oof_internal=_internal(1.0),
        raw_summary=_summary(1.0),
        raw_internal=_internal(1.0, selected=1.0),
        projected_summary=_summary(1.0),
        projected_internal=_internal(1.0, selected=1.0),
        spec=stagei.IntegratorRejectionPredicateSpec(),
    )
    assert result["locus"] == "scale_selection_zero_despite_observed_predicate_pass"
    assert result["discriminator_family"] == "scale_selection"


def test_classify_unanimous_retention_rejection() -> None:
    records = [
        {
            "rejection_comparison": {
                "locus": "oof_rejected_at_direction_retention",
                "discriminator_family": "direction_retention",
            }
        }
        for _ in range(27)
    ]
    result = stagei.classify(records)
    assert result["root_cause"] == (
        "phase314b_r258_stagei_oof_direction_rejected_by_retention_predicate"
    )
    assert result["required_next_path"] == (
        "CALIBRATE_INTEGRATOR_COMPATIBLE_DIRECTION_BASIS_ON_OBJECTIVE_TRAIN_ONLY"
    )
    assert result["all_records_share_discriminator_family"] is True


def test_classify_heterogeneous_predicates() -> None:
    records = [
        {
            "rejection_comparison": {
                "locus": "oof_rejected_at_direction_retention",
                "discriminator_family": "direction_retention",
            }
        },
        {
            "rejection_comparison": {
                "locus": "oof_rejected_at_upper_segment_geometry",
                "discriminator_family": "upper_segment_geometry",
            }
        },
    ]
    result = stagei.classify(records)
    assert result["root_cause"] == (
        "phase314b_r258_stagei_rejection_predicates_are_heterogeneous"
    )


def test_predicate_source_index_records_lines(tmp_path: Path) -> None:
    path = tmp_path / "fake_integrator.py"
    path.write_text(
        "def integrate_rowwise():\n"
        "    retention_pass = True\n"
        "    return {'retention_pass': retention_pass}\n",
        encoding="utf-8",
    )
    namespace = {}
    exec(compile(path.read_text(), str(path), "exec"), namespace)
    module = SimpleNamespace(
        __file__=str(path),
        integrate_rowwise=namespace["integrate_rowwise"],
    )
    result = stagei.predicate_source_index(module)
    assert result["module_sha256"]
    assert "direction_retention" in result["family_identifiers"]


def test_audit_direction_source_uses_unchanged_scale_bank(monkeypatch: pytest.MonkeyPatch) -> None:
    control = np.zeros((2, 1, 2), dtype=np.float32)
    direction = np.ones_like(control, dtype=np.float64)

    def integrate_rowwise(**kwargs):
        proposed = np.asarray(kwargs["direction"])
        selected = np.ones(control.shape[0], dtype=np.float64)
        return {
            "candidate": (control + proposed).astype(np.float32),
            "selected_scale": selected,
            "gates": {"retention_pass": np.ones(control.shape[0], dtype=np.bool_)},
        }

    monkeypatch.setattr(stagei.stagee258, "integrate_rowwise", integrate_rowwise)
    monkeypatch.setattr(stagei.stagef, "fixed_integrator_definition", lambda: object())
    summary, internal = stagei.audit_direction_source(
        source_id="oof_surrogate",
        control=control,
        base_direction=direction,
        context={},
        integrator_spec=SimpleNamespace(retention_min=0.25),
        source_index={"family_identifiers": {}},
        spec=stagei.IntegratorRejectionPredicateSpec(),
    )
    assert len(summary["multiplier_records"]) == 4
    assert summary["scale_bank_acceptance"][
        "rows_with_any_accepted_multiplier_rate"
    ] == 1.0
    assert internal["selected_bank"].shape == (2, 4)


def test_audit_direction_source_rejects_scale_motion_inconsistency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control = np.zeros((1, 1, 2), dtype=np.float32)

    def integrate_rowwise(**kwargs):
        return {
            "candidate": control.copy(),
            "selected_scale": np.ones(1),
        }

    monkeypatch.setattr(stagei.stagee258, "integrate_rowwise", integrate_rowwise)
    monkeypatch.setattr(stagei.stagef, "fixed_integrator_definition", lambda: object())
    with pytest.raises(stagei.IntegratorRejectionPredicateError):
        stagei.audit_direction_source(
            source_id="oof_surrogate",
            control=control,
            base_direction=np.ones_like(control),
            context={},
            integrator_spec=SimpleNamespace(retention_min=0.25),
            source_index={"family_identifiers": {}},
            spec=stagei.IntegratorRejectionPredicateSpec(),
        )


def test_run_calibration_keeps_holdout_and_probe_closed() -> None:
    source = inspect.getsource(stagei.run_calibration)
    assert "holdout_target" not in source
    assert '"selection_holdout_evaluated": False' in source
    assert '"frozen_probe_accessed": False' in source
    assert "fit_feasibility_ranker" not in source


def test_result_contract_forbids_tensor_persistence() -> None:
    source = inspect.getsource(stagei.run_calibration)
    assert '"predicate_tensor_persisted": False' in source
    assert '"candidate_tensor_persisted": False' in source
    assert '"prediction_tensor_persisted": False' in source
