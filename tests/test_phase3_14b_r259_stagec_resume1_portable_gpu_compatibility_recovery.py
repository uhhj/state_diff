from __future__ import annotations

import copy
import hashlib
import importlib
import json
import sys
import types
from pathlib import Path

import pytest

BASE_NAME = "ccda_phase3.phase314b_r259_stagec_fold_resolved_tail_attribution"
try:
    importlib.import_module(BASE_NAME)
except Exception:
    fake = types.ModuleType(BASE_NAME)
    fake.PHASE = "Phase3.14b-r2.5.9 Stage C"
    fake.BLOCKED_SCHEMA = "phase314b_r259_stagec_fold_resolved_tail_attribution_blocked_v1"
    fake.FALSE_BOUNDARIES = (
        "selection_holdout_evaluated",
        "selection_holdout_used_for_fit_or_selection",
        "frozen_probe_accessed",
        "formal_training_run",
        "reverse_sampling_run",
        "idm_run",
        "candidate_execution",
        "deformable_ravens_executed",
        "phase4",
        "cps",
        "checkpoint_saved",
        "weights_persisted",
        "prediction_tensor_persisted",
        "candidate_tensor_persisted",
        "risk_probability_tensor_persisted",
        "npz_saved",
        "cache_saved",
        "image_saved",
        "video_saved",
    )
    fake.EXPECTED_EXECUTION_COUNTS = {
        "direction_fit_count": 108,
        "candidate_generation_count": 324,
        "risk_fit_count": 324,
        "internal_scale_attempt_count": 2268,
        "inner_policy_evaluation_count": 216,
        "full_objective_oof_policy_evaluation_count": 36,
        "nonconverged_risk_fit_count": 0,
    }
    fake.EXPECTED_STAGEA_PROBE_SHA256 = "a" * 64
    fake.EXPECTED_STAGEA_WORKER_RESULT_SHA256 = "b" * 64
    fake.STAGEA_PROBE = "stagea_probe.json"
    fake.STAGEA_WORKER = "stagea_worker.json"
    fake.STAGEB_RESUME1_REPORT = "stageb.json"
    fake.validate_stageb_resume1_report = lambda value: None
    fake.stagea = types.SimpleNamespace(
        resume2a=types.SimpleNamespace(
            stable_json_bytes=lambda value: json.dumps(
                value, sort_keys=True, separators=(",", ":")
            ).encode(),
            sha256_bytes=lambda value: hashlib.sha256(value).hexdigest(),
            validate_probe_payload=lambda value: value["environment"],
        ),
        validate_probe_evidence=lambda value: value["environment"],
        validate_worker_evidence=lambda value: None,
    )
    sys.modules[BASE_NAME] = fake

from ccda_phase3 import (  # noqa: E402
    phase314b_r259_stagec_resume1_portable_gpu_compatibility_recovery as r,
)


def environment(name="NVIDIA GeForce RTX 4080 SUPER", *, compatibility="c" * 64):
    return {
        "compatibility_pass": True,
        "compatibility": {
            "python_version": [3, 9, 15],
            "numpy_version": "1.23.3",
            "torch_version": "1.12.1.post200",
            "torch_cuda_version": "11.2",
            "required_operation_pass": True,
        },
        "compatibility_sha256": compatibility,
        "hardware_observation": {
            "cuda_device_count": 1,
            "torch_device_name": name,
            "compute_capability": [8, 9],
            "total_memory_bytes": 16 * 1024**3,
            "multi_processor_count": 80,
            "cudnn_version": 8302,
        },
        "required_operation_dry_run": {"pass": True},
    }


def reproduction_pair():
    exact = {
        "locked_scientific_base": {"backbone": "segment_target_rr64_feasible"},
        "stagex_spec": {"outer_folds": 6},
        "policy_population": [{"policy_id": "p"}],
        "population": {"rows": 638, "groups": 126},
        "procedure_gate_totals": {"mismatch": 0},
        "execution_counts": dict(r.base.EXPECTED_EXECUTION_COUNTS),
    }
    selections = [
        {
            "outer_fold": fold,
            "selected_policy": {"shrinkage": 1.0, "risk_threshold": 0.5},
            "selected_policy_id": "shrink_1.00__risk_0.50",
            "selected_policy_inner_eligible": fold not in (1, 4),
            "diagnostic_fallback_used": fold in (1, 4),
            "diagnostic_score": 0.1 + fold,
        }
        for fold in range(6)
    ]
    records = {
        "10": {
            "acceptance_count": 275,
            "acceptance_rate": 0.43103448275862066,
            "overall_mse": 0.9414,
            "candidate_sha256": "x" * 64,
            "selected_scale_sha256": "s" * 64,
            "eligible": False,
        },
        "25": {
            "acceptance_count": 421,
            "acceptance_rate": 0.6598746,
            "overall_mse": 0.8827,
            "candidate_sha256": "y" * 64,
            "selected_scale_sha256": "t" * 64,
            "eligible": True,
        },
    }
    current = {
        **copy.deepcopy(exact),
        "outer_fold_selections": copy.deepcopy(selections),
        "inner_selection_modal_policy_diagnostic": {
            "selected_policy_id": "shrink_1.00__risk_0.50",
            "support_count": 4,
            "stable": True,
            "mean_score": 0.2,
        },
        "outer_crossfit_fold_selected_policy_records": copy.deepcopy(records),
        "outer_crossfit_baseline_records": copy.deepcopy(records),
        "full_objective_oof_fixed_policy_selection": {
            "selected_policy_id": "shrink_1.00__risk_0.50",
            "eligible": True,
            "score": 0.3,
        },
        "full_objective_oof_fixed_policy_records": {"p": copy.deepcopy(records)},
        "scientific_status": "BLOCKED",
        "primary_failure_locus": "outer_crossfit_tail_fidelity_failure",
    }
    reference = copy.deepcopy(current)
    reference["scientific_kernel"] = {
        "kernel_primary_failure_locus": "outer_crossfit_tail_fidelity_failure"
    }
    return current, reference


def worker(slot="A"):
    payload = {
        "schema": r.WORKER_SCHEMA,
        "execution_verdict": "PASS",
        "scientific_status": "BLOCKED",
        "root_cause": "root",
        "required_next_path": "next",
        "primary_failure_locus": "locus",
        "process_id": 100 if slot == "A" else 101,
        "worker_slot": slot,
        "execution_counts": dict(r.base.EXPECTED_EXECUTION_COUNTS),
        "portable_stagea_reproduction_identity": {
            "portable_contract_passed": True,
            "hardware_identity_required": False,
        },
        "instrumentation_identity": {
            "stitch_call_count": 1,
            "outer_outputs_unchanged": True,
        },
        "fold_resolved_attribution": {
            "fold_timestep_record_count": 18,
            "fold_timestep_records": [],
        },
        "outer_fold_selections": [],
        "inner_selection_modal_policy_diagnostic": {},
        "global_outer_crossfit_fold_selected_policy_records": {},
        "global_outer_crossfit_baseline_records": {},
        "full_objective_oof_fixed_policy_selection": {},
        "procedure_gate_totals": {},
        "selected_configuration": None,
        "train_only_recommendation": None,
        "selection_holdout_evaluation_count_added": 0,
        "cumulative_selection_holdout_evaluation_count": 1,
        "frozen_probe_accessed": False,
        "rerun_authorized": False,
        "durable_write_ahead_evidence": True,
        "write_ahead_location_outside_git_worktree": True,
    }
    payload.update({key: False for key in r.FALSE_BOUNDARIES})
    payload["worker_result_sha256"] = r.sha256_bytes(r.stable_json_bytes(payload))
    return payload


def test_phase_name():
    assert r.PHASE.endswith("Stage C Resume1")


def test_starting_commit_is_blocked_evidence():
    assert r.BASE_STAGEC_BLOCKED_EVIDENCE_COMMIT.startswith("042ab9b4")


def test_original_implementation_is_frozen():
    assert r.BASE_STAGEC_IMPLEMENTATION_COMMIT.startswith("c8e99921")


def test_remote_is_latest_pushed_stageb_head():
    assert r.EXPECTED_REMOTE == "4f8a73fb5cc9b60c92c440b6e61bc1765704f3e5"


def test_submodule_is_frozen():
    assert r.EXPECTED_SUBMODULE.startswith("633a8875")


def test_original_blocked_report_hash_is_frozen():
    assert r.BASE_STAGEC_BLOCKED_REPORT_SHA256.startswith("6d0d455c")


def test_add_only_four_files():
    assert len(r.IMPLEMENTATION_PATHS) == 4
    assert {status for status, _ in r.IMPLEMENTATION_PATHS} == {"A"}


def test_portable_spec_validates():
    r.PortableAggregateSpec().validate()


def test_portable_spec_rejects_wider_rtol():
    with pytest.raises(ValueError):
        r.PortableAggregateSpec(rtol=0.01).validate()


def test_portable_spec_rejects_wider_atol():
    with pytest.raises(ValueError):
        r.PortableAggregateSpec(atol=2e-6).validate()


def test_stable_json_is_ordered():
    assert r.stable_json_bytes({"b": 1, "a": 2}).startswith(b'{\n  "a"')


def test_compact_json_is_compact():
    assert r.compact_json_bytes({"b": 1, "a": 2}) == b'{"a":2,"b":1}'


def test_sha256_bytes():
    assert r.sha256_bytes(b"abc") == hashlib.sha256(b"abc").hexdigest()


def test_atomic_write_once(tmp_path: Path):
    path = tmp_path / "x.json"
    r.atomic_write_once(path, b"{}\n")
    assert path.read_bytes() == b"{}\n"
    with pytest.raises(r.StageCResume1Error):
        r.atomic_write_once(path, b"again")


def test_extract_process_id():
    assert r.extract_process_id({"process_id": 1}, "x") == 1


def test_extract_process_id_rejects_bool():
    with pytest.raises(r.StageCResume1Error):
        r.extract_process_id({"process_id": True}, "x")


def test_hardware_observation():
    assert r._hardware_observation(environment())["cuda_device_count"] == 1


def test_hardware_observation_rejects_missing():
    with pytest.raises(r.StageCResume1Error):
        r._hardware_observation({})


def test_environment_audit_accepts_different_gpu(monkeypatch):
    monkeypatch.setattr(r.base.stagea.resume2a, "validate_probe_payload", lambda value: value["environment"])
    current = environment("NVIDIA GeForce RTX 4080 SUPER")
    reference = environment("NVIDIA GeForce RTX 3090")
    result = r.portable_environment_audit(current, reference)
    assert result["portable_contract_passed"] is True
    assert result["hardware_identity_exact"] is False
    assert result["hardware_identity_required"] is False


def test_environment_audit_rejects_compatibility_difference(monkeypatch):
    monkeypatch.setattr(r.base.stagea.resume2a, "validate_probe_payload", lambda value: value["environment"])
    with pytest.raises(r.StageCResume1Error, match="compatibility"):
        r.portable_environment_audit(environment(compatibility="a" * 64), environment(compatibility="b" * 64))


def test_environment_audit_rejects_small_memory(monkeypatch):
    monkeypatch.setattr(r.base.stagea.resume2a, "validate_probe_payload", lambda value: value["environment"])
    current = environment()
    current["hardware_observation"]["total_memory_bytes"] = 8 * 1024**3
    with pytest.raises(r.StageCResume1Error, match="memory"):
        r.portable_environment_audit(current, environment("RTX 3090"))


def test_environment_audit_rejects_old_compute(monkeypatch):
    monkeypatch.setattr(r.base.stagea.resume2a, "validate_probe_payload", lambda value: value["environment"])
    current = environment()
    current["hardware_observation"]["compute_capability"] = [7, 5]
    with pytest.raises(r.StageCResume1Error, match="capability"):
        r.portable_environment_audit(current, environment("RTX 3090"))


def test_environment_audit_rejects_multiple_devices(monkeypatch):
    monkeypatch.setattr(r.base.stagea.resume2a, "validate_probe_payload", lambda value: value["environment"])
    current = environment()
    current["hardware_observation"]["cuda_device_count"] = 2
    with pytest.raises(r.StageCResume1Error, match="exactly one"):
        r.portable_environment_audit(current, environment("RTX 3090"))


def test_semanticize_removes_candidate_sha():
    result = r._semanticize_portable({"candidate_sha256": "x", "count": 1})
    assert "candidate_sha256" not in result


def test_semanticize_keeps_selected_scale_sha():
    result = r._semanticize_portable({"selected_scale_sha256": "x"})
    assert result["selected_scale_sha256"] == "x"


def test_numeric_compare_exact():
    result = r._numeric_tree_compare({"x": 1.0}, {"x": 1.0}, spec=r.PortableAggregateSpec())
    assert result["pass"] is True


def test_numeric_compare_within_tolerance():
    result = r._numeric_tree_compare({"x": 1.0001}, {"x": 1.0}, spec=r.PortableAggregateSpec())
    assert result["pass"] is True


def test_numeric_compare_outside_tolerance():
    result = r._numeric_tree_compare({"x": 1.01}, {"x": 1.0}, spec=r.PortableAggregateSpec())
    assert result["pass"] is False


def test_numeric_compare_integer_is_exact():
    result = r._numeric_tree_compare({"x": 2}, {"x": 1}, spec=r.PortableAggregateSpec())
    assert result["pass"] is False


def test_numeric_compare_boolean_is_exact():
    result = r._numeric_tree_compare({"x": True}, {"x": False}, spec=r.PortableAggregateSpec())
    assert result["pass"] is False


def test_numeric_compare_keys_are_exact():
    result = r._numeric_tree_compare({"x": 1}, {"y": 1}, spec=r.PortableAggregateSpec())
    assert result["pass"] is False


def test_selection_projection_ignores_diagnostic_float():
    values = [{
        "outer_fold": 0,
        "selected_policy": {"shrinkage": 1.0},
        "selected_policy_id": "p",
        "selected_policy_inner_eligible": True,
        "diagnostic_fallback_used": False,
        "score": 0.2,
    }]
    assert "score" not in r._selection_projection(values)[0]


def test_decision_projection_keeps_booleans_and_counts():
    value = r._decision_projection({"eligible": True, "support_count": 4, "score": 0.1})
    assert value == {"eligible": True, "support_count": 4}


def test_portable_reproduction_accepts_small_float_difference():
    current, reference = reproduction_pair()
    current["outer_crossfit_fold_selected_policy_records"]["10"]["overall_mse"] += 1e-4
    result = r.portable_compare_stagea_reproduction(current, reference)
    assert result["portable_contract_passed"] is True
    assert result["historical_all_bitwise_exact"] is False


def test_portable_reproduction_rejects_policy_change():
    current, reference = reproduction_pair()
    current["outer_fold_selections"][0]["selected_policy_id"] = "other"
    with pytest.raises(r.StageCResume1Error, match="decisions"):
        r.portable_compare_stagea_reproduction(current, reference)


def test_portable_reproduction_rejects_count_change():
    current, reference = reproduction_pair()
    current["execution_counts"]["direction_fit_count"] += 1
    with pytest.raises(r.StageCResume1Error, match="exact"):
        r.portable_compare_stagea_reproduction(current, reference)


def test_portable_reproduction_rejects_acceptance_count_change():
    current, reference = reproduction_pair()
    current["outer_crossfit_fold_selected_policy_records"]["10"]["acceptance_count"] += 1
    with pytest.raises(r.StageCResume1Error, match="numeric"):
        r.portable_compare_stagea_reproduction(current, reference)


def test_portable_reproduction_rejects_metric_large_difference():
    current, reference = reproduction_pair()
    current["outer_crossfit_fold_selected_policy_records"]["10"]["overall_mse"] += 0.1
    with pytest.raises(r.StageCResume1Error, match="numeric"):
        r.portable_compare_stagea_reproduction(current, reference)


def test_portable_reproduction_rejects_status_change():
    current, reference = reproduction_pair()
    current["scientific_status"] = "READY"
    with pytest.raises(r.StageCResume1Error, match="decisions"):
        r.portable_compare_stagea_reproduction(current, reference)


def test_validate_worker_accepts_synthetic():
    r.validate_worker_evidence(worker())


def test_validate_worker_rejects_ready():
    value = worker()
    value["scientific_status"] = "READY"
    value["worker_result_sha256"] = r.sha256_bytes(r.stable_json_bytes({k: v for k, v in value.items() if k != "worker_result_sha256"}))
    with pytest.raises(r.StageCResume1Error):
        r.validate_worker_evidence(value)


def test_validate_worker_rejects_holdout():
    value = worker()
    value["selection_holdout_evaluation_count_added"] = 1
    value["worker_result_sha256"] = r.sha256_bytes(r.stable_json_bytes({k: v for k, v in value.items() if k != "worker_result_sha256"}))
    with pytest.raises(r.StageCResume1Error):
        r.validate_worker_evidence(value)


def test_validate_worker_rejects_execution_count():
    value = worker()
    value["execution_counts"]["risk_fit_count"] += 1
    value["worker_result_sha256"] = r.sha256_bytes(r.stable_json_bytes({k: v for k, v in value.items() if k != "worker_result_sha256"}))
    with pytest.raises(r.StageCResume1Error):
        r.validate_worker_evidence(value)


def test_worker_projection_removes_pid_slot_and_hash():
    projected = r.worker_scientific_projection(worker())
    assert "process_id" not in projected
    assert "worker_slot" not in projected
    assert "worker_result_sha256" not in projected


def test_same_device_workers_accept_exact_projection():
    result = r.compare_same_device_workers(worker("A"), worker("B"))
    assert result["same_device_scientific_projection_byte_exact"] is True


def test_same_device_workers_reject_science_difference():
    a = worker("A")
    b = worker("B")
    b["root_cause"] = "different"
    b["worker_result_sha256"] = r.sha256_bytes(r.stable_json_bytes({k: v for k, v in b.items() if k != "worker_result_sha256"}))
    with pytest.raises(r.StageCResume1Error, match="byte-exact"):
        r.compare_same_device_workers(a, b)


def test_promote_write_ahead(tmp_path: Path):
    source = tmp_path / "source.json"
    destination = tmp_path / "destination.json"
    source.write_text("{}\n")
    r.promote_write_ahead(source, destination, lambda value: None)
    assert source.read_bytes() == destination.read_bytes()


def test_promote_write_ahead_is_write_once(tmp_path: Path):
    source = tmp_path / "source.json"
    destination = tmp_path / "destination.json"
    source.write_text("{}\n")
    destination.write_text("existing")
    with pytest.raises(r.StageCResume1Error):
        r.promote_write_ahead(source, destination, lambda value: None)


def test_blocked_before_worker():
    payload = r.blocked_report(repository=None, error=RuntimeError("x"))
    assert payload["durable_worker_a_evidence_available"] is False
    assert payload["science_reexecution_authorized"] is False


def test_blocked_after_worker(tmp_path: Path, monkeypatch):
    path = tmp_path / "worker.json"
    path.write_text(json.dumps(worker()))
    payload = r.blocked_report(repository={}, error=RuntimeError("x"), worker_a_path=path)
    assert payload["durable_worker_a_evidence_available"] is True
    assert payload["primary_failure_locus"] == "controller_after_durable_worker_evidence"


def test_blocked_never_selects_configuration():
    payload = r.blocked_report(repository=None, error=RuntimeError("x"))
    assert payload["selected_configuration"] is None
    assert payload["train_only_recommendation"] is None


def test_false_boundaries_are_false_in_worker():
    value = worker()
    assert all(value[key] is False for key in r.FALSE_BOUNDARIES)


def test_minimum_memory_allows_4080_super_capacity():
    assert 16 * 1024**3 >= r.MIN_TOTAL_MEMORY_BYTES


def test_compute_capability_allows_ampere_and_ada():
    assert r.MIN_COMPUTE_CAPABILITY_MAJOR == 8


def test_output_paths_are_resume1_namespaced():
    assert "resume1" in r.PROBE_EVIDENCE
    assert "resume1" in r.WORKER_A_EVIDENCE
    assert "resume1" in r.WORKER_B_EVIDENCE
    assert "resume1" in r.SUCCESS_REPORT


def test_workers_are_distinct_evidence_paths():
    assert r.WORKER_A_EVIDENCE != r.WORKER_B_EVIDENCE


def test_hardware_output_hashes_are_not_required():
    assert "candidate" in r.OUTPUT_SHA_TOKENS
    assert "selected_scale" in r.EXACT_SHA_TOKENS


def test_make_worker_adapter_transforms_original(monkeypatch):
    original = worker("A")
    original["schema"] = "original"
    original.pop("worker_slot")
    original["stagea_reproduction_identity"] = {
        "all_exact": True,
        "portable_contract_passed": True,
        "hardware_identity_required": False,
    }
    original.pop("portable_stagea_reproduction_identity")
    original.pop("worker_result_sha256")
    original["preregistration_contract"] = {}
    if not hasattr(r.base, "validate_probe_evidence"):
        monkeypatch.setattr(r.base, "validate_probe_evidence", lambda value: value, raising=False)
    if not hasattr(r.base, "compare_stagea_reproduction"):
        monkeypatch.setattr(r.base, "compare_stagea_reproduction", lambda *args, **kwargs: {}, raising=False)
    old_probe = r.base.validate_probe_evidence
    old_compare = r.base.compare_stagea_reproduction
    monkeypatch.setattr(r, "validate_probe_evidence", lambda value: value)
    monkeypatch.setattr(r.base, "make_worker_evidence", lambda **kwargs: copy.deepcopy(original), raising=False)
    result = r.make_worker_evidence(
        root=Path("/tmp"),
        probe_payload={"process_id": 1},
        repository_head="h" * 40,
        stageu_contract={},
        worker_slot="A",
    )
    assert result["schema"] == r.WORKER_SCHEMA
    assert result["portable_stagea_reproduction_identity"]["portable_contract_passed"] is True
    assert "stagea_reproduction_identity" not in result
    assert r.base.validate_probe_evidence is old_probe
    assert r.base.compare_stagea_reproduction is old_compare


def test_make_worker_adapter_rejects_unknown_slot(monkeypatch):
    monkeypatch.setattr(r, "validate_probe_evidence", lambda value: value)
    with pytest.raises(r.StageCResume1Error, match="slot"):
        r.make_worker_evidence(
            root=Path("/tmp"),
            probe_payload={},
            repository_head="h" * 40,
            stageu_contract={},
            worker_slot="C",
        )
