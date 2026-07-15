from __future__ import annotations

import contextlib
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from ccda_phase3 import phase314b_r257_staged_resume3_calibration_trace as resume3


def test_commit_constants_are_frozen():
    assert resume3.RESUME2_IMPLEMENTATION_COMMIT == (
        "e68d069a8ad5135a10d8e4ab3a07f46049d596aa"
    )
    assert resume3.RESUME1_BLOCKED_EVIDENCE_COMMIT == (
        "85ec59d45c6445e9cd61e91b7e6d6d94fe729ecf"
    )


def test_resume2_report_shas_are_frozen():
    assert resume3.EXPECTED_RESUME2_BLOCKED_SHA256 == (
        "f737cd10bcbe50b053df62e6e4840f357f0ea5257459846ad37b8d48014931d3"
    )
    assert resume3.EXPECTED_RESUME2_TEST_GATE_SHA256 == (
        "2cab7c00335472da24f1ac62894ac7b971dde6965037decd9d1b168b06cffe94"
    )


def test_stable_json_is_deterministic():
    assert resume3.stable_json_bytes(
        {"b": 2, "a": 1}
    ) == resume3.stable_json_bytes(
        {"a": 1, "b": 2}
    )


def test_stable_json_supports_numpy():
    payload = resume3.stable_json_bytes(
        {
            "array": np.asarray(
                [1.0, 2.0],
                dtype=np.float32,
            ),
            "scalar": np.float64(3.0),
        }
    )
    loaded = json.loads(payload)
    assert loaded["array"] == [1.0, 2.0]
    assert loaded["scalar"] == 3.0


def test_atomic_write_once(tmp_path):
    path = tmp_path / "value.json"
    resume3.atomic_write_once(path, b"{}\n")
    assert path.read_bytes() == b"{}\n"
    with pytest.raises(FileExistsError):
        resume3.atomic_write_once(path, b"{}\n")


def test_load_json_rejects_non_object(tmp_path):
    path = tmp_path / "value.json"
    path.write_text("[1]\n", encoding="utf-8")
    with pytest.raises(resume3.Resume3TraceError):
        resume3.load_json(path)


def test_recursive_diff_exact():
    records, total = resume3.recursive_diff(
        {"a": [1, 2]},
        {"a": [1, 2]},
    )
    assert records == []
    assert total == 0


def test_recursive_diff_value_mismatch():
    records, total = resume3.recursive_diff(
        {"a": 1.0},
        {"a": 1.5},
    )
    assert total == 1
    assert records[0]["path"] == "$.a"
    assert records[0]["kind"] == "value_mismatch"
    assert records[0]["absolute_difference"] == pytest.approx(0.5)


def test_recursive_diff_missing_key():
    records, total = resume3.recursive_diff(
        {"a": 1},
        {},
    )
    assert total == 1
    assert records[0]["kind"] == "missing_observed_key"


def test_recursive_diff_unexpected_key():
    records, total = resume3.recursive_diff(
        {},
        {"a": 1},
    )
    assert total == 1
    assert records[0]["kind"] == "unexpected_observed_key"


def test_recursive_diff_sequence_length():
    records, total = resume3.recursive_diff(
        [1, 2],
        [1],
    )
    assert total == 1
    assert records[0]["kind"] == "sequence_length"


def test_recursive_diff_type_mismatch():
    records, total = resume3.recursive_diff(
        1,
        1.0,
    )
    assert total == 1
    assert records[0]["kind"] == "type_mismatch"


def test_recursive_diff_limit():
    expected = {"a": 1, "b": 2, "c": 3}
    observed = {"a": 4, "b": 5, "c": 6}
    records, total = resume3.recursive_diff(
        expected,
        observed,
        limit=2,
    )
    assert total == 3
    assert len(records) == 2


@pytest.mark.parametrize(
    "path,kind,expected",
    [
        (
            "$.initial_model_sha256",
            "value_mismatch",
            "initial_model_identity",
        ),
        (
            "$.raw.gradient.cosine",
            "value_mismatch",
            "gradient_metric",
        ),
        (
            "$.raw.predicted_x0_sha256",
            "value_mismatch",
            "prediction_identity",
        ),
        (
            "$.candidate_order",
            "sequence_length",
            "structure",
        ),
    ],
)
def test_diff_category(path, kind, expected):
    assert resume3.diff_category(path, kind) == expected


def test_classify_diff_exact():
    result = resume3.classify_diff([], 0)
    assert result["primary_difference_locus"] == "none"
    assert result["total_difference_count"] == 0


def test_classify_diff_identity():
    records = [
        {
            "path": "$.initial_model_sha256",
            "kind": "value_mismatch",
        }
    ]
    result = resume3.classify_diff(records, 1)
    assert result["identity_or_structure_difference"]
    assert result["primary_difference_locus"] == (
        "calibration_identity_or_structure"
    )


def test_classify_diff_gradient_only():
    records = [
        {
            "path": "$.raw.gradient.cosine",
            "kind": "value_mismatch",
        }
    ]
    result = resume3.classify_diff(records, 1)
    assert not result["identity_or_structure_difference"]
    assert result["primary_difference_locus"] == (
        "floating_gradient_metrics"
    )


def test_capture_context_stops_before_nonzero(monkeypatch):
    original_calibrate = resume3.stagec_topk.calibrate_candidates
    original_train = resume3.stagec_topk.train_candidate
    original_selection = resume3.stagec_topk.selection_record

    candidate = type(
        "Candidate",
        (),
        {"candidate_id": "topk8_r0p25"},
    )()

    monkeypatch.setattr(
        resume3.stagec_topk,
        "calibrate_candidates",
        lambda *args, **kwargs: (
            [candidate],
            {
                "calibration_sha256": "a" * 64,
            },
        ),
    )
    monkeypatch.setattr(
        resume3.stagec_topk,
        "train_candidate",
        lambda *args, **kwargs: (
            object(),
            {"final_model_sha256": "m" * 64},
            {"x": 1},
        ),
    )
    monkeypatch.setattr(
        resume3.stagec_topk,
        "selection_record",
        lambda *args, **kwargs: {
            "candidate": None,
            "training": {"final_model_sha256": "m" * 64},
        },
    )

    with resume3.capture_stagec_control_entrypoint() as state:
        candidates, calibration = (
            resume3.stagec_topk.calibrate_candidates()
        )
        assert candidates[0].candidate_id == "topk8_r0p25"
        resume3.stagec_topk.train_candidate()
        resume3.stagec_topk.selection_record(
            candidate=None
        )
        with pytest.raises(
            resume3._StopAfterStageCControl
        ):
            resume3.stagec_topk.train_candidate()

    assert state["train_candidate_call_count"] == 1
    assert state[
        "stopped_before_first_nonzero_candidate"
    ]
    assert state["control_selection_record_count"] == 1
    assert (
        resume3.stagec_topk.calibrate_candidates
        is not original_calibrate
    )
    # monkeypatch owns restoration after the test; the Resume3 context
    # restored the functions to the monkeypatched base functions.
    assert (
        resume3.stagec_topk.train_candidate
        is not original_train
    )
    assert (
        resume3.stagec_topk.selection_record
        is not original_selection
    )


def test_install_captured_control_restores():
    original = resume3.resume1.exact_control_replay
    bundle = {"candidate_id": "control"}
    identity = {"training_exact": True}
    with resume3.install_captured_control(
        bundle,
        identity,
    ):
        observed_bundle, observed_identity = (
            resume3.resume1.exact_control_replay()
        )
        assert observed_bundle == bundle
        assert observed_identity == identity
        observed_bundle["candidate_id"] = "changed"
        assert bundle["candidate_id"] == "control"
    assert resume3.resume1.exact_control_replay is original


def test_run_manual_calibration_diff_exact(monkeypatch):
    context = {
        "stageb_spec": object(),
        "diagnostic_batch": {},
        "target_standardizer": object(),
        "objective_contract": object(),
        "expected_calibration": {
            "calibration_sha256": "a" * 64,
            "candidate_order": ["x"],
        },
        "split": {"frozen_probe_accessed": False},
    }
    monkeypatch.setattr(
        resume3,
        "build_manual_calibration_context",
        lambda root: context,
    )
    candidate = type(
        "Candidate",
        (),
        {"candidate_id": "x"},
    )()
    monkeypatch.setattr(
        resume3.stagec_topk,
        "calibrate_candidates",
        lambda **kwargs: (
            [candidate],
            copy.deepcopy(
                context["expected_calibration"]
            ),
        ),
    )
    result = resume3.run_manual_calibration_diff(
        root=Path("/tmp")
    )
    assert result["manual_replay_exact"]
    assert result["difference_summary"][
        "total_difference_count"
    ] == 0


def test_run_manual_calibration_diff_records_difference(
    monkeypatch,
):
    context = {
        "stageb_spec": object(),
        "diagnostic_batch": {},
        "target_standardizer": object(),
        "objective_contract": object(),
        "expected_calibration": {
            "calibration_sha256": "a" * 64,
            "value": 1.0,
        },
        "split": {"frozen_probe_accessed": False},
    }
    monkeypatch.setattr(
        resume3,
        "build_manual_calibration_context",
        lambda root: context,
    )
    monkeypatch.setattr(
        resume3.stagec_topk,
        "calibrate_candidates",
        lambda **kwargs: (
            [],
            {
                "calibration_sha256": "b" * 64,
                "value": 2.0,
            },
        ),
    )
    result = resume3.run_manual_calibration_diff(
        root=Path("/tmp")
    )
    assert not result["manual_replay_exact"]
    assert result["difference_summary"][
        "total_difference_count"
    ] == 2


def test_capture_exact_stagec_control_success(monkeypatch):
    calibration = {
        "calibration_sha256": "a" * 64,
    }
    control = {
        "candidate_id": "control_diffusion_only",
        "training": {
            "final_model_sha256": "m" * 64,
            "final_optimizer_sha256": "o" * 64,
            "total_loss_history_sha256": "l" * 64,
            "gradient_history_sha256": "g" * 64,
            "source_exposure_sha256": "e" * 64,
        },
        "train_control": {"normalized_mse": 1.0},
        "one_step": {"prediction_sha256": "p" * 64},
        "holdout_profiles": {"10": {"row_any_rate": 0.0}},
    }
    monkeypatch.setattr(
        resume3.failed_staged,
        "validate_immutable_inputs",
        lambda root: {
            "stagec_contract": {
                "calibration": calibration,
            },
            "control_record": control,
        },
    )

    def fake_run(*, root):
        candidates, observed_calibration = (
            resume3.stagec_topk.calibrate_candidates()
        )
        assert observed_calibration == calibration
        model, training, diagnostics = (
            resume3.stagec_topk.train_candidate()
        )
        resume3.stagec_topk.selection_record(
            candidate=None,
            training=training,
            train_control=control["train_control"],
            one_step=control["one_step"],
            profiles=control["holdout_profiles"],
        )
        resume3.stagec_topk.train_candidate(
            candidate=object()
        )

    candidate = type(
        "Candidate",
        (),
        {"candidate_id": "x"},
    )()
    monkeypatch.setattr(
        resume3.stagec_topk,
        "calibrate_candidates",
        lambda *args, **kwargs: (
            [candidate],
            calibration,
        ),
    )
    monkeypatch.setattr(
        resume3.stagec_topk,
        "train_candidate",
        lambda *args, **kwargs: (
            object(),
            control["training"],
            {"checkpoint": 1},
        ),
    )
    monkeypatch.setattr(
        resume3.stagec_topk,
        "selection_record",
        lambda *args, **kwargs: control,
    )
    monkeypatch.setattr(
        resume3.stagec_topk,
        "run_calibration",
        fake_run,
    )
    result = resume3.capture_exact_stagec_control(
        root=Path("/tmp")
    )
    assert result["control_identity"][
        "calibration_exact"
    ]
    assert result["control_identity"][
        "control_record_exact"
    ]
    assert result["control_bundle"] == control


def test_run_resume3_uses_captured_control(monkeypatch):
    manual = {
        "completed": True,
        "training_performed": False,
        "frozen_probe_accessed": False,
        "manual_replay_exact": False,
    }
    monkeypatch.setattr(
        resume3,
        "validate_resume2_failure",
        lambda root: {"blocked": True},
    )
    captured = {
        "control_bundle": {
            "candidate_id": "control_diffusion_only",
        },
        "control_identity": {
            "training_exact": True,
        },
    }
    monkeypatch.setattr(
        resume3,
        "capture_exact_stagec_control",
        lambda root: captured,
    )

    @contextlib.contextmanager
    def no_op_validator():
        yield

    monkeypatch.setattr(
        resume3.resume2,
        "adapted_stagec_validator",
        no_op_validator,
    )

    def fake_resume1(*, root, spec):
        bundle, identity = (
            resume3.resume1.exact_control_replay()
        )
        assert bundle == captured["control_bundle"]
        assert identity == captured["control_identity"]
        return {
            "phase": "old",
            "phase_id": "old",
            "schema": "old",
            "verdict": "PASS",
            "scientific_status": "BLOCKED",
            "root_cause": "r",
            "required_next_path": "n",
            "control_trace_correction": identity,
            "split": {"x": 1},
            "calibration_contract": {
                "contract_sha256": "x" * 64,
            },
            "selection": {"selection_sha256": "s" * 64},
            "classification": {"primary_failure_locus": "x"},
            "selected_configuration": None,
            "train_only_recommendation": None,
            "new_hyperparameter_candidate_run": True,
            "new_objective_variant_run": True,
        }

    monkeypatch.setattr(
        resume3.resume1,
        "run_resume1",
        fake_resume1,
    )
    result = resume3.run_resume3(
        root=Path("/tmp"),
        manual_diff=manual,
        spec=None,
    )
    assert result["phase"] == resume3.PHASE
    assert result["resume3_correction_applied"]
    assert result["control_replay_exact"]
    assert result[
        "resume3_calibration_attribution"
    ]["manual_trace_relaxed"] is False


def test_compare_worker_results_exact():
    base = {
        "root_cause": "r",
        "required_next_path": "n",
        "resume3_calibration_attribution": {
            "manual_calibration_diff": {"x": 1},
            "stagec_entrypoint_capture": {"x": 1},
        },
        "control_trace_correction": {"x": 1},
        "split": {"x": 1},
        "calibration_contract": {"x": 1},
        "selection": {"x": 1},
        "classification": {"x": 1},
        "selected_configuration": None,
    }
    result = resume3.compare_worker_results(
        base,
        dict(base),
    )
    assert result["exact"]
    assert result["manual_diff_exact"]


def test_compare_worker_results_detects_manual_diff():
    left = {
        "root_cause": "r",
        "required_next_path": "n",
        "resume3_calibration_attribution": {
            "manual_calibration_diff": {"x": 1},
            "stagec_entrypoint_capture": {"x": 1},
        },
        "control_trace_correction": {"x": 1},
        "split": {"x": 1},
        "calibration_contract": {"x": 1},
        "selection": {"x": 1},
        "classification": {"x": 1},
        "selected_configuration": None,
    }
    right = copy.deepcopy(left)
    right["resume3_calibration_attribution"][
        "manual_calibration_diff"
    ] = {"x": 2}
    result = resume3.compare_worker_results(
        left,
        right,
    )
    assert not result["exact"]
    assert not result["manual_diff_exact"]
