from __future__ import annotations

import inspect

import numpy as np
import pytest

from ccda_phase3 import phase314b_r258_stageh_candidate_descriptor_identifiability as stageh


def test_stageh_population_is_frozen() -> None:
    assert len(stageh.BASE_DIRECTION_IDS) == 9
    assert stageh.SCALE_MULTIPLIERS == (0.25, 0.5, 1.0, 2.0)
    assert stageh.EXPECTED_SELECTED_ULP_FACTOR == 1.0


def test_stageh_spec_validates() -> None:
    spec = stageh.CandidateDescriptorIdentifiabilitySpec()
    spec.validate()
    assert spec.grouped_cv_folds == 6
    assert spec.diagnostic_leaf_limit == 96


def test_descriptor_layout_is_exact_partition() -> None:
    stageh.validate_descriptor_layout()
    assert list(stageh.DESCRIPTOR_BLOCKS.values())[0] == (0, 1)
    assert list(stageh.DESCRIPTOR_BLOCKS.values())[-1] == (88, 92)
    assert set(stageh.PROPOSAL_DESCRIPTOR_BLOCKS).isdisjoint(
        stageh.INTEGRATED_DESCRIPTOR_BLOCKS
    )


def test_descriptor_block_dimensions() -> None:
    descriptor = np.zeros((2, 4, 92), dtype=np.float64)
    proposal = stageh.descriptor_block(
        descriptor, stageh.PROPOSAL_DESCRIPTOR_BLOCKS
    )
    integrated = stageh.descriptor_block(
        descriptor, stageh.INTEGRATED_DESCRIPTOR_BLOCKS
    )
    assert proposal.shape == (2, 4, 46)
    assert integrated.shape == (2, 4, 46)


def test_row_unique_counts_exact() -> None:
    value = np.asarray(
        [[[0.0], [1.0], [2.0], [3.0]], [[1.0], [1.0], [1.0], [1.0]]]
    )
    result = stageh.row_unique_counts(value, atol=1.0e-12)
    assert result.tolist() == [4, 1]


def test_row_unique_counts_uses_numeric_tolerance() -> None:
    value = np.asarray([[[0.0], [1.0e-13], [1.0], [1.0 + 1.0e-13]]])
    result = stageh.row_unique_counts(value, atol=1.0e-12)
    assert result.tolist() == [2]


def test_pairwise_distance_summary_detects_diversity() -> None:
    value = np.asarray([[[0.0], [1.0], [2.0], [3.0]]])
    result = stageh.pairwise_distance_summary(value, epsilon=1.0e-12)
    assert result["minimum"] == pytest.approx(1.0)
    assert result["maximum"] == pytest.approx(3.0)
    assert result["positive_pair_rate"] == 1.0


def test_maximum_feature_signal_finds_linear_dimension() -> None:
    descriptor = np.zeros((3, 4, 2), dtype=np.float64)
    target = np.arange(12, dtype=np.float64).reshape(3, 4)
    descriptor[:, :, 1] = target
    result = stageh.maximum_feature_signal(
        descriptor, target, epsilon=1.0e-14
    )
    assert result["best_dimension"] == 1
    assert result["maximum_absolute_correlation"] == pytest.approx(1.0)


def _control_and_bank() -> tuple[np.ndarray, dict]:
    control = np.zeros((2, 1, 2), dtype=np.float64)
    candidates = np.repeat(control[:, None], 4, axis=1)
    selected_scales = np.zeros((2, 4), dtype=np.float64)
    return control, {
        "candidates": candidates,
        "selected_scales": selected_scales,
    }


def test_proposal_audit_detects_nondegenerate_scale_bank() -> None:
    base = np.ones((2, 1, 2), dtype=np.float64)
    proposals = np.stack(
        [base * multiplier for multiplier in stageh.SCALE_MULTIPLIERS], axis=1
    )
    result = stageh.proposal_audit(
        base_direction=base,
        proposals=proposals,
        spec=stageh.CandidateDescriptorIdentifiabilitySpec(),
    )
    assert result["non_degenerate"] is True
    assert result["per_row_unique_count"]["mean"] == 4.0


def test_candidate_audit_detects_complete_rejection() -> None:
    control, bank = _control_and_bank()
    result = stageh.candidate_audit(
        control=control,
        bank=bank,
        spec=stageh.CandidateDescriptorIdentifiabilitySpec(),
    )
    assert result["selected_scale"]["positive_rate"] == 0.0
    assert result["candidate_motion"]["positive_rate"] == 0.0
    assert result["candidate_motion"]["exact_control_rate"] == 1.0
    assert result["observable_motion_consistency"]["consistent"] is True


def test_candidate_audit_detects_observable_motion_mismatch() -> None:
    control, bank = _control_and_bank()
    bank["selected_scales"][0, 0] = 1.0
    result = stageh.candidate_audit(
        control=control,
        bank=bank,
        spec=stageh.CandidateDescriptorIdentifiabilitySpec(),
    )
    assert (
        result["observable_motion_consistency"]
        ["positive_scale_without_motion_count"]
        == 1
    )
    assert result["observable_motion_consistency"]["consistent"] is False


def test_label_audit_detects_constant_zero_supervision() -> None:
    utility = np.zeros((3, 4), dtype=np.float64)
    beneficial = np.zeros((3, 4), dtype=np.bool_)
    result = stageh.label_audit(
        utility=utility,
        beneficial=beneficial,
        spec=stageh.CandidateDescriptorIdentifiabilitySpec(),
    )
    assert result["utility"]["unique_count"] == 1
    assert result["utility"]["all_scales_tied_rate"] == 1.0
    assert result["beneficial"]["entropy_bits"] == 0.0


def test_descriptor_audit_separates_proposal_and_integrated_variation() -> None:
    descriptor = np.zeros((2, 4, 92), dtype=np.float64)
    descriptor[:, :, 0] = np.asarray(stageh.SCALE_MULTIPLIERS)[None]
    utility = np.zeros((2, 4), dtype=np.float64)
    beneficial = np.zeros((2, 4), dtype=np.bool_)
    result = stageh.descriptor_audit(
        descriptor=descriptor,
        utility=utility,
        beneficial=beneficial,
        spec=stageh.CandidateDescriptorIdentifiabilitySpec(),
    )
    assert result["proposal_only"]["per_row_unique_count"]["mean"] == 4.0
    assert result["post_integrator"]["per_row_unique_count"]["mean"] == 1.0


def _proposal_record() -> dict:
    return {
        "non_degenerate": True,
    }


def _candidate_record(*, consistent: bool = True) -> dict:
    return {
        "observable_motion_consistency": {"consistent": consistent},
        "selected_scale": {"positive_rate": 0.0},
        "candidate_motion": {"positive_rate": 0.0},
        "per_row_unique_count": {"mean": 1.0},
    }


def _descriptor_record() -> dict:
    return {
        "post_integrator": {
            "per_row_unique_count": {"mean": 1.0},
            "utility_signal": {"maximum_absolute_correlation": 0.0},
        }
    }


def _label_record() -> dict:
    return {
        "utility": {"unique_count": 1},
        "beneficial": {"unique_count": 1},
    }


def test_identify_record_locus_prefers_integrator_rejection() -> None:
    locus = stageh.identify_record_locus(
        proposal=_proposal_record(),
        candidate=_candidate_record(),
        descriptor=_descriptor_record(),
        labels=_label_record(),
    )
    assert locus == "frozen_integrator_rejects_all_scale_bank_candidates"


def test_identify_record_locus_prioritizes_observable_inconsistency() -> None:
    locus = stageh.identify_record_locus(
        proposal=_proposal_record(),
        candidate=_candidate_record(consistent=False),
        descriptor=_descriptor_record(),
        labels=_label_record(),
    )
    assert locus == "observable_feasibility_indicator_inconsistent"


def test_classify_unanimous_integrator_rejection() -> None:
    records = [
        {"primary_failure_locus": "frozen_integrator_rejects_all_scale_bank_candidates"}
        for _ in range(27)
    ]
    result = stageh.classify(records)
    assert result["root_cause"] == (
        "phase314b_r258_stageh_frozen_integrator_rejects_entire_surrogate_scale_bank"
    )
    assert result["required_next_path"] == (
        "AUDIT_FROZEN_INTEGRATOR_REJECTION_PREDICATES_ON_OOF_DIRECTION_BANK"
    )
    assert result["all_records_share_primary_locus"] is True


def test_integration_diagnostics_are_bounded_and_skip_candidate() -> None:
    integration = {
        "candidate": np.ones((2, 4)),
        "selected_scale": np.asarray([0.0, 1.0]),
        "gates": {"valid": np.asarray([False, True])},
        "unrelated": np.arange(4),
    }
    result = stageh.integration_diagnostic_leaves(integration, limit=96)
    assert "$.selected_scale" in result
    assert "$.gates.valid" in result
    assert not any(path.startswith("$.candidate") for path in result)
    assert "$.unrelated" not in result


def test_run_calibration_keeps_holdout_and_ranker_closed() -> None:
    source = inspect.getsource(stageh.run_calibration)
    assert "holdout_target" not in source
    assert "fit_feasibility_ranker" not in source
    assert '"selection_holdout_evaluated": False' in source
    assert '"frozen_probe_accessed": False' in source
