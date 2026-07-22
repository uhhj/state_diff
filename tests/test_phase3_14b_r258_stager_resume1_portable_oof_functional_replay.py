from __future__ import annotations

import copy
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from ccda_phase3 import (
    phase314b_r258_stager_resume1_portable_oof_functional_replay as resume1,
)


def array_sha(value):
    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(
        str(array.dtype).encode() + repr(tuple(array.shape)).encode() + array.tobytes()
    ).hexdigest()


def fake_fit_factory(*, second_prediction=None, second_folds=None):
    calls = {"count": 0}

    def fit(**kwargs):
        calls["count"] += 1
        prediction = (
            "prediction-a"
            if calls["count"] % 2 == 1 or second_prediction is None
            else second_prediction
        )
        folds = [
            {
                "fold": index,
                "model_identity": {"model_sha256": f"m{index}"},
                "test_target_used_for_fit": False,
            }
            for index in range(6)
        ]
        if calls["count"] % 2 == 0 and second_folds is not None:
            folds = copy.deepcopy(second_folds)
        return {
            "prediction": np.zeros((2, 3), dtype=np.float64),
            "prediction_sha256": prediction,
            "fold_records": folds,
        }

    fit.calls = calls
    return fit


def make_registry_record(base="anchor_point_rr32_all"):
    fit = fake_fit_factory()
    registry = resume1.RepeatFitRegistry(fit)
    first = registry(base_direction_id=base)
    return registry, first


def make_validator_inputs(*, historical_prediction="prediction-a"):
    registry, first = make_registry_record()
    control = np.zeros((2, 3), dtype=np.float32)
    candidate = control.copy()
    selected = np.zeros(2, dtype=np.float64)
    proposed = np.ones((2, 3), dtype=np.float64)
    assembly = {"closed": True, "count": 2}
    event = {"event_type": "scale_attempt"}
    stagel = SimpleNamespace(
        sha256_array=array_sha,
        aggregate_sequential_attempts=lambda events: assembly,
    )
    stagek = SimpleNamespace(
        _row_norm=lambda value: np.sqrt(np.sum(np.asarray(value) ** 2, axis=1))
    )
    integration = {"selected_scale": selected, "candidate": candidate}
    capture = {
        "events": [event],
        "events_sha256": "capture",
        "returned_result_bit_exact": True,
    }
    expected = {
        "scale_multiplier": 0.25,
        "proposed_direction_sha256": array_sha(proposed),
        "selected_scale_positive_rate": 0.0,
        "selected_scale_sha256": array_sha(selected),
        "candidate_motion_positive_rate": 0.0,
        "candidate_sha256": array_sha(candidate),
        "callback_capture_sha256": "capture",
        "callback_result_bit_exact": True,
        "assembly": assembly,
    }
    persisted = {
        "base_direction_id": "anchor_point_rr32_all",
        "timestep": 10,
        "feature_sha256": "feature",
        "oof_prediction_sha256": historical_prediction,
        "oof_predicate_assembly": {"multiplier_records": [expected]},
    }
    spec = SimpleNamespace(binary_tolerance=1e-12, candidate_motion_epsilon=1e-12)
    kwargs = dict(
        registry=registry,
        base_direction_id="anchor_point_rr32_all",
        timestep=10,
        feature_sha256="feature",
        prediction_sha256=str(first["prediction_sha256"]),
        proposed_direction=proposed,
        integration=integration,
        capture=capture,
        persisted_record=persisted,
        stagel=stagel,
        stagek=stagek,
        spec=spec,
        control=control,
    )
    return kwargs


def test_phase_is_resume1():
    assert "Resume1" in resume1.PHASE


def test_schema_version():
    assert resume1.SCHEMA.endswith("_v1")


def test_commit_chain_frozen():
    assert resume1.BASE_STAGER_IMPLEMENTATION_COMMIT == (
        "3fe97091427b7a6bcef1c79805ebebece93746a5"
    )
    assert resume1.BASE_STAGER_BLOCKED_EVIDENCE_COMMIT == (
        "61985677dde41f48396331c8c446f97f1d626d30"
    )


def test_blocked_report_sha_frozen():
    assert resume1.EXPECTED_STAGER_BLOCKED_REPORT_SHA256 == (
        "d10376e855ceb89c453366488213524c98b04b28ac2050f1b3df46d9bc6549a6"
    )


def test_original_source_sha_frozen():
    assert resume1.EXPECTED_STAGER_SOURCE_SHA256 == (
        "9f8da64be049bb9cd9a61e229e4dc20e9a91cccc55261f43511a76f600fb4ed0"
    )


def test_add_only_three_paths():
    assert len(resume1.IMPLEMENTATION_PATHS) == 3
    assert all(status == "A" for status, _ in resume1.IMPLEMENTATION_PATHS)


def test_fit_counts():
    assert resume1.EXPECTED_SCIENCE_FIT_COUNT == 27
    assert resume1.EXPECTED_REPEAT_FIT_COUNT == 27
    assert resume1.EXPECTED_TOTAL_FIT_COUNT == 54
    assert resume1.EXPECTED_CALLBACK_PAIR_COUNT == 27


def test_repeat_fit_accepts_exact_result():
    registry, first = make_registry_record()
    assert first["prediction_sha256"] == "prediction-a"
    assert registry.fit_call_count == 2
    assert registry.records[0]["repeat_prediction_byte_exact"] is True


@pytest.mark.parametrize(
    "second_prediction",
    ["prediction-b", "", "different"],
)
def test_repeat_fit_rejects_prediction_change(second_prediction):
    registry = resume1.RepeatFitRegistry(
        fake_fit_factory(second_prediction=second_prediction)
    )
    with pytest.raises(resume1.StageRResume1Error, match="repeat fit"):
        registry(base_direction_id="anchor_point_rr32_all")


def test_repeat_fit_rejects_fold_change():
    changed = [
        {
            "fold": index,
            "model_identity": {"model_sha256": f"changed-{index}"},
            "test_target_used_for_fit": False,
        }
        for index in range(6)
    ]
    registry = resume1.RepeatFitRegistry(fake_fit_factory(second_folds=changed))
    with pytest.raises(resume1.StageRResume1Error, match="repeat fit"):
        registry(base_direction_id="anchor_point_rr32_all")


def test_registry_consumes_in_order():
    registry, _ = make_registry_record()
    value = registry.consume("anchor_point_rr32_all")
    assert value["base_direction_id"] == "anchor_point_rr32_all"
    assert registry.pending == []


def test_registry_rejects_wrong_order():
    registry, _ = make_registry_record()
    with pytest.raises(resume1.StageRResume1Error, match="order"):
        registry.consume("other")


def test_registry_rejects_empty_queue():
    registry = resume1.RepeatFitRegistry(fake_fit_factory())
    with pytest.raises(resume1.StageRResume1Error, match="empty"):
        registry.consume("anchor_point_rr32_all")


def test_validator_accepts_historical_prediction_exact():
    result = resume1.portable_validate_legacy_oof_identity(
        **make_validator_inputs(historical_prediction="prediction-a")
    )
    assert result["all_functional_exact"] is True
    assert result["historical_prediction_sha_exact"] is True


def test_validator_accepts_historical_prediction_mismatch_when_functional_exact():
    result = resume1.portable_validate_legacy_oof_identity(
        **make_validator_inputs(historical_prediction="historical-other")
    )
    assert result["all_functional_exact"] is True
    assert result["historical_prediction_sha_exact"] is False
    assert result["checks"]["prediction_sha256"] is False


def test_validator_records_proposed_direction_exact():
    result = resume1.portable_validate_legacy_oof_identity(
        **make_validator_inputs()
    )
    assert result["historical_proposed_direction_sha_exact"] is True


def test_validator_accepts_proposed_hash_mismatch_if_functional_exact():
    kwargs = make_validator_inputs()
    kwargs["persisted_record"]["oof_predicate_assembly"]["multiplier_records"][0][
        "proposed_direction_sha256"
    ] = "historical-other"
    result = resume1.portable_validate_legacy_oof_identity(**kwargs)
    assert result["historical_proposed_direction_sha_exact"] is False


@pytest.mark.parametrize(
    "field,bad",
    [
        ("selected_scale_positive_rate", 1.0),
        ("selected_scale_sha256", "bad"),
        ("candidate_motion_positive_rate", 1.0),
        ("candidate_sha256", "bad"),
        ("callback_capture_sha256", "bad"),
        ("callback_result_bit_exact", False),
        ("assembly", {"different": True}),
    ],
)
def test_validator_rejects_functional_mismatch(field, bad):
    kwargs = make_validator_inputs()
    record = kwargs["persisted_record"]["oof_predicate_assembly"][
        "multiplier_records"
    ][0]
    record[field] = bad
    with pytest.raises(
        resume1.StageRResume1Error,
        match="functional replay differs",
    ):
        resume1.portable_validate_legacy_oof_identity(**kwargs)


def test_validator_rejects_feature_change():
    kwargs = make_validator_inputs()
    kwargs["feature_sha256"] = "different"
    with pytest.raises(resume1.StageRResume1Error, match="feature"):
        resume1.portable_validate_legacy_oof_identity(**kwargs)


def test_validator_rejects_backbone_change():
    kwargs = make_validator_inputs()
    kwargs["base_direction_id"] = "other"
    with pytest.raises(resume1.StageRResume1Error, match="backbone"):
        resume1.portable_validate_legacy_oof_identity(**kwargs)


def test_validator_rejects_timestep_change():
    kwargs = make_validator_inputs()
    kwargs["timestep"] = 25
    with pytest.raises(resume1.StageRResume1Error, match="timestep"):
        resume1.portable_validate_legacy_oof_identity(**kwargs)


def test_validator_rejects_selected_motion_disagreement():
    kwargs = make_validator_inputs()
    kwargs["integration"]["selected_scale"] = np.ones(2, dtype=np.float64)
    with pytest.raises(resume1.StageRResume1Error, match="disagree"):
        resume1.portable_validate_legacy_oof_identity(**kwargs)


def test_validator_does_not_persist_prediction_tensor():
    result = resume1.portable_validate_legacy_oof_identity(
        **make_validator_inputs(historical_prediction="other")
    )
    repeat = result["portable_repeat_fit"]
    assert "prediction" not in repeat


def test_patched_context_restores_functions():
    original_validate = object()

    def original_fit(**kwargs):
        return fake_fit_factory()(**kwargs)

    fake_stageh = SimpleNamespace(fit_oof_base_direction=original_fit)
    fake_stagel = SimpleNamespace(stageh=fake_stageh)
    fake_stager = SimpleNamespace(validate_legacy_oof_identity=original_validate)
    with resume1.patched_portable_replay(fake_stager, fake_stagel):
        assert fake_stager.validate_legacy_oof_identity is not original_validate
        assert fake_stageh.fit_oof_base_direction is not original_fit
    assert fake_stager.validate_legacy_oof_identity is original_validate
    assert fake_stageh.fit_oof_base_direction is original_fit


def test_patched_context_restores_after_exception():
    original_validate = object()
    original_fit = fake_fit_factory()
    fake_stageh = SimpleNamespace(fit_oof_base_direction=original_fit)
    fake_stagel = SimpleNamespace(stageh=fake_stageh)
    fake_stager = SimpleNamespace(validate_legacy_oof_identity=original_validate)
    with pytest.raises(RuntimeError):
        with resume1.patched_portable_replay(fake_stager, fake_stagel):
            raise RuntimeError("synthetic")
    assert fake_stager.validate_legacy_oof_identity is original_validate
    assert fake_stageh.fit_oof_base_direction is original_fit


def make_correctable_result(*, exact_count=0):
    cells = []
    for index in range(27):
        exact = index < exact_count
        cells.append(
            {
                "legacy_identity": {
                    "all_functional_exact": True,
                    "historical_prediction_sha_exact": exact,
                    "historical_proposed_direction_sha_exact": exact,
                    "portable_repeat_fit": {
                        "repeat_prediction_byte_exact": True
                    },
                }
            }
        )
    return {
        "execution_verdict": "PASS",
        "scientific_status": "BLOCKED",
        "root_cause": "scientific-root",
        "required_next_path": "NEXT",
        "primary_failure_locus": "science",
        "selected_configuration": None,
        "train_only_recommendation": None,
        "tolerance_aligned_oof_post_upper_audit": {
            "cell_records": cells,
            "classification": {
                "meaningful_support_cell_count": 0,
                "dominant_dual_oracle_discriminator": None,
            },
        },
        "immutable_inputs": {},
        "mechanism_boundary": {},
        **{key: False for key in resume1.FALSE_BOUNDARIES},
    }


def complete_registry():
    registry = resume1.RepeatFitRegistry(fake_fit_factory())
    registry.records = [
        {"repeat_prediction_byte_exact": True} for _ in range(27)
    ]
    registry.pending = []
    registry.fit_call_count = 54
    return registry


def test_correct_result_counts_mismatches():
    result = resume1._correct_stage_r_result(
        make_correctable_result(exact_count=3), complete_registry()
    )
    audit = result["tolerance_aligned_oof_post_upper_audit"]
    assert audit["historical_prediction_sha_exact_cell_count"] == 3
    assert audit["historical_prediction_sha_mismatch_cell_count"] == 24
    assert audit["total_oof_fit_count"] == 54


def test_correct_result_marks_prediction_change_honestly():
    result = resume1._correct_stage_r_result(
        make_correctable_result(exact_count=0), complete_registry()
    )
    assert result["mechanism_boundary"]["oof_predictions_changed"] is True
    assert result["mechanism_boundary"]["historical_prediction_sha_replaced"] is False


def test_correct_result_marks_full_historical_exactness():
    result = resume1._correct_stage_r_result(
        make_correctable_result(exact_count=27), complete_registry()
    )
    assert result["mechanism_boundary"]["oof_predictions_changed"] is False
    assert result["mechanism_boundary"][
        "historical_prediction_byte_identity_preserved"
    ] is True


def test_correct_result_recomputes_scientific_sha():
    result = resume1._correct_stage_r_result(
        make_correctable_result(exact_count=1), complete_registry()
    )
    assert len(result["scientific_result_sha256"]) == 64


def test_correct_result_rejects_nonfunctional_cell():
    value = make_correctable_result()
    value["tolerance_aligned_oof_post_upper_audit"]["cell_records"][0][
        "legacy_identity"
    ]["all_functional_exact"] = False
    with pytest.raises(resume1.StageRResume1Error, match="functional"):
        resume1._correct_stage_r_result(value, complete_registry())


def test_correct_result_rejects_wrong_cell_count():
    value = make_correctable_result()
    value["tolerance_aligned_oof_post_upper_audit"]["cell_records"].pop()
    with pytest.raises(resume1.StageRResume1Error, match="cell count"):
        resume1._correct_stage_r_result(value, complete_registry())


def test_blocked_payload_preserves_provenance():
    payload = resume1.blocked_report(
        repository={"head": "x"}, error=RuntimeError("synthetic")
    )
    assert payload["execution_verdict"] == "BLOCKED"
    assert payload["original_stage_r_blocked_report_preserved"] is True
    assert payload["historical_prediction_sha_forged_or_overwritten"] is False
    assert payload["selected_configuration"] is None
    assert payload["train_only_recommendation"] is None


def test_false_boundaries_unique():
    assert len(resume1.FALSE_BOUNDARIES) == len(set(resume1.FALSE_BOUNDARIES))


def test_environment_contract_complete():
    assert resume1.EXPECTED_ENV["PYTHONHASHSEED"] == "0"
    assert resume1.EXPECTED_ENV["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert len(resume1.EXPECTED_ENV) == 9
