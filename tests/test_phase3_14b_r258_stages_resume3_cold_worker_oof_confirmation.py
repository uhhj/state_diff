from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from ccda_phase3 import (
    phase314b_r258_stages_resume3_cold_worker_oof_confirmation as resume3,
)


def fake_summary():
    return {
        "cell_count": 27,
        "nonzero_support_cell_count": 27,
        "zero_acceptance_cell_count": 0,
        "acceptance_min": 0.3166,
        "acceptance_mean": 0.8083,
        "acceptance_max": 0.9812,
        "oracle_like_cell_count": 7,
        "dominant_discriminator": "direction_retention",
        "dominant_support_count": 2,
        "dominant_backbone_coverage": 1,
        "dominant_timestep_coverage": 2,
    }


def fake_functional():
    return {
        "cell_count": 27,
        "classification": {"stable": True},
        "cell_records": [{"index": index} for index in range(27)],
    }


def fake_fit_identity():
    return {
        "cell_count": 27,
        "records": [{"index": index, "sha": f"p{index}"} for index in range(27)],
    }


def fake_probe(pid: int = 101):
    environment = {
        "compatibility_pass": True,
        "compatibility": {"required_operation_pass": True},
        "required_operation_dry_run": {"pass": True},
    }
    return {
        "mode": "environment_probe",
        "execution_verdict": "PASS",
        "process_id": pid,
        "environment": environment,
        "environment_sha256": resume3.sha256_bytes(
            resume3.stable_json_bytes(environment)
        ),
        "cuda_initialization_allowed_in_this_process": True,
        "process_disposable": True,
    }


def fake_worker(worker_id: str = "worker-1", pid: int = 201):
    functional = fake_functional()
    fit = fake_fit_identity()
    environment = fake_probe()["environment"]
    environment_sha = resume3.sha256_bytes(resume3.stable_json_bytes(environment))
    base_sha = resume3.sha256_bytes(resume3.stable_json_bytes(functional))
    value = {
        "phase": resume3.PHASE,
        "schema": resume3.WORKER_SCHEMA,
        "worker_id": worker_id,
        "process_id": pid,
        "execution_verdict": "PASS",
        "repository_head": "implementation-head",
        "environment": environment,
        "environment_sha256": environment_sha,
        "cold_cuda_precheck": {
            "checked": True,
            "torch_cuda_is_initialized": False,
        },
        "environment_probe_rerun_in_science_process": False,
        "base_functional_projection_sha256": base_sha,
        "functional_projection_sha256": base_sha,
        "current_fit_projection_sha256": resume3.sha256_bytes(
            resume3.stable_json_bytes(fit)
        ),
        "functional_projection_matches_base": True,
        "functional_projection": functional,
        "current_fit_projection": fit,
        "confirmation_summary": fake_summary(),
        "scientific_oof_fit_count": 27,
        "repeat_identity_fit_count": 27,
        "total_oof_fit_count": 54,
        "callback_pair_count": 27,
        "oracle_callback_rerun_count": 0,
        "internal_scale_attempt_count": 189,
        **{key: False for key in resume3.FALSE_BOUNDARIES},
    }
    return value


def install_fake_runtime_modules(monkeypatch):
    stages_name = "ccda_phase3.phase314b_r258_stages_oof_support_confirmation"
    stages_resume1_name = (
        "ccda_phase3.phase314b_r258_stages_resume1_oracle_rerun_schema_recovery"
    )
    resume2a_name = (
        "ccda_phase3.phase314b_r258_stages_resume2a_"
        "cold_cuda_process_boundary_smoke"
    )

    stages = ModuleType(stages_name)
    stages.EXPECTED_BASE_REPORT_SHA256 = "stage-r-report"
    stages.EXPECTED_BASE_STAGE_R_PAYLOAD_SHA256 = "stage-r-payload"
    stages.EXPECTED_BASE_SCIENTIFIC_SHA256 = "stage-r-science"
    stages.stable_json_bytes = resume3.stable_json_bytes
    stages.sha256_bytes = resume3.sha256_bytes

    def compare_workers(first, second):
        keys = (
            "functional_projection_sha256",
            "current_fit_projection_sha256",
            "environment_sha256",
            "confirmation_summary",
            "functional_projection",
            "current_fit_projection",
        )
        if any(
            resume3.stable_json_bytes(first.get(key))
            != resume3.stable_json_bytes(second.get(key))
            for key in keys
        ):
            raise RuntimeError("workers differ")
        return {"all_exact": True, "checks": {key: True for key in keys}}

    stages.compare_workers = compare_workers
    stages.classify_confirmation = lambda summary: {
        "root_cause": "phase314b_r258_stages_tolerance_aligned_oof_support_confirmed",
        "required_next_path": (
            "FREEZE_TOLERANCE_ALIGNED_GATE_AND_CONFIRM_OOF_CANDIDATE_MATRIX_"
            "ON_OBJECTIVE_TRAIN_ONLY"
        ),
        "primary_failure_locus": "confirmed_broad_oof_support",
    }
    stages.validate_base_report = lambda root: {
        "functional_projection": fake_functional(),
        "repository": {"head": "base"},
    }
    stages.current_fit_projection = lambda replay: copy.deepcopy(
        replay["current_fit_projection"]
    )
    stages.validate_confirmation_projection = lambda projection: fake_summary()

    stages_resume1 = ModuleType(stages_resume1_name)
    stages_resume1.corrected_functional_projection = (
        lambda wrapper, stages: copy.deepcopy(wrapper["functional_projection"])
    )

    resume2a = ModuleType(resume2a_name)

    def validate_probe(payload):
        if payload.get("mode") != "environment_probe":
            raise RuntimeError("bad probe")
        return payload["environment"]

    resume2a.validate_probe_payload = validate_probe

    monkeypatch.setitem(sys.modules, stages_name, stages)
    monkeypatch.setitem(sys.modules, stages_resume1_name, stages_resume1)
    monkeypatch.setitem(sys.modules, resume2a_name, resume2a)
    return stages, stages_resume1, resume2a


def test_phase_and_schema():
    assert resume3.PHASE.endswith("Stage S Resume3")
    assert resume3.SCHEMA.endswith("_v1")
    assert resume3.WORKER_SCHEMA.endswith("_v1")


def test_frozen_resume2a_commits():
    assert resume3.BASE_RESUME2A_IMPLEMENTATION_COMMIT == (
        "edf85f19228bec61e500e7b9aa922f136e9d1d8d"
    )
    assert resume3.BASE_RESUME2A_EVIDENCE_COMMIT == (
        "38b096bf5820cdb2f1cadde78149ce389c464352"
    )


def test_frozen_resume2a_report_hashes():
    assert resume3.EXPECTED_RESUME2A_REPORT_SHA256.startswith("c06c594f")
    assert resume3.EXPECTED_RESUME2A_SCIENTIFIC_SHA256.startswith("9c635e3e")


def test_add_only_four_paths():
    assert len(resume3.IMPLEMENTATION_PATHS) == 4
    assert all(status == "A" for status, _ in resume3.IMPLEMENTATION_PATHS)


def test_expected_execution_counts():
    assert resume3.EXPECTED_TOTAL_FITS == 108
    assert resume3.EXPECTED_TOTAL_CALLBACK_PAIRS == 54
    assert resume3.EXPECTED_TOTAL_INTERNAL_ATTEMPTS == 378
    assert resume3.EXPECTED_PROBE_PROCESS_COUNT == 1


def test_false_boundaries_unique():
    assert len(resume3.FALSE_BOUNDARIES) == len(set(resume3.FALSE_BOUNDARIES))


def test_hash_is_order_stable():
    left = resume3.sha256_bytes(resume3.stable_json_bytes({"b": 2, "a": 1}))
    right = resume3.sha256_bytes(resume3.stable_json_bytes({"a": 1, "b": 2}))
    assert left == right


def test_environment_contract_complete():
    assert resume3.EXPECTED_ENV["PYTHONHASHSEED"] == "0"
    assert resume3.EXPECTED_ENV["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_validate_environment_variables_passes(monkeypatch):
    for key, value in resume3.EXPECTED_ENV.items():
        monkeypatch.setenv(key, value)
    assert resume3.validate_environment_variables() == dict(resume3.EXPECTED_ENV)


@pytest.mark.parametrize("key", list(resume3.EXPECTED_ENV)[:4])
def test_validate_environment_variables_rejects_mismatch(monkeypatch, key):
    for name, value in resume3.EXPECTED_ENV.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv(key, "wrong")
    with pytest.raises(resume3.StageSResume3Error):
        resume3.validate_environment_variables()


def test_validate_probe_payload_for_science_accepts():
    probe = fake_probe()
    module = SimpleNamespace(validate_probe_payload=lambda payload: payload["environment"])
    value = resume3.validate_probe_payload_for_science(probe, resume2a=module)
    assert value["compatibility_pass"] is True


@pytest.mark.parametrize(
    "field,bad",
    [
        ("process_id", None),
        ("process_disposable", False),
        ("cuda_initialization_allowed_in_this_process", False),
        ("environment_sha256", "bad"),
    ],
)
def test_validate_probe_payload_for_science_rejects(field, bad):
    probe = fake_probe()
    probe[field] = bad
    module = SimpleNamespace(validate_probe_payload=lambda payload: payload["environment"])
    with pytest.raises(resume3.StageSResume3Error):
        resume3.validate_probe_payload_for_science(probe, resume2a=module)


def test_validate_cold_worker_accepts():
    worker = fake_worker()
    resume3.validate_cold_science_worker_payload(
        worker,
        expected_environment_sha256=worker["environment_sha256"],
        expected_base_projection_sha256=worker[
            "base_functional_projection_sha256"
        ],
    )


@pytest.mark.parametrize(
    "field,bad",
    [
        ("schema", "bad"),
        ("execution_verdict", "BLOCKED"),
        ("environment_sha256", "bad"),
        ("environment_probe_rerun_in_science_process", True),
        ("functional_projection_matches_base", False),
        ("scientific_oof_fit_count", 26),
        ("repeat_identity_fit_count", 26),
        ("total_oof_fit_count", 53),
        ("callback_pair_count", 26),
        ("oracle_callback_rerun_count", 1),
        ("internal_scale_attempt_count", 188),
    ],
)
def test_validate_cold_worker_rejects(field, bad):
    worker = fake_worker()
    worker[field] = bad
    with pytest.raises(resume3.StageSResume3Error):
        resume3.validate_cold_science_worker_payload(
            worker,
            expected_environment_sha256=fake_worker()["environment_sha256"],
            expected_base_projection_sha256=fake_worker()[
                "base_functional_projection_sha256"
            ],
        )


def test_validate_cold_worker_rejects_warm_cuda():
    worker = fake_worker()
    worker["cold_cuda_precheck"]["torch_cuda_is_initialized"] = True
    with pytest.raises(resume3.StageSResume3Error):
        resume3.validate_cold_science_worker_payload(
            worker,
            expected_environment_sha256=worker["environment_sha256"],
        )


def test_validate_cold_worker_rejects_boundary_change():
    worker = fake_worker()
    worker[resume3.FALSE_BOUNDARIES[0]] = True
    with pytest.raises(resume3.StageSResume3Error):
        resume3.validate_cold_science_worker_payload(
            worker,
            expected_environment_sha256=worker["environment_sha256"],
        )


def test_validate_confirmation_summary_accepts():
    assert resume3.validate_confirmation_summary(fake_summary())[
        "nonzero_support_cell_count"
    ] == 27


@pytest.mark.parametrize(
    "field,bad",
    [
        ("cell_count", 26),
        ("nonzero_support_cell_count", 26),
        ("zero_acceptance_cell_count", 1),
        ("oracle_like_cell_count", 6),
        ("dominant_discriminator", "topology"),
        ("dominant_support_count", 3),
        ("dominant_backbone_coverage", 2),
        ("dominant_timestep_coverage", 3),
    ],
)
def test_validate_confirmation_summary_rejects(field, bad):
    summary = fake_summary()
    summary[field] = bad
    with pytest.raises(resume3.StageSResume3Error):
        resume3.validate_confirmation_summary(summary)


def test_worker_summary_is_compact():
    summary = resume3._worker_summary(fake_worker())
    assert summary["cold_cuda_before_replay"] is True
    assert "functional_projection" not in summary


def test_blocked_report_does_not_claim_totals():
    payload = resume3.blocked_report(
        repository={"head": "x", "base_report": {}, "resume2a_report": {}},
        error=RuntimeError("x"),
    )
    assert payload["execution_verdict"] == "BLOCKED"
    assert payload["complete_worker_population_claimed"] is False
    assert payload["total_fit_count_claimed"] is False


def test_science_worker_source_does_not_probe_environment():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts/phase3_14b_r258_stages_resume3_worker.py"
    )
    text = path.read_text(encoding="utf-8")
    assert "probe_environment(" not in text
    assert "--probe" in text


def test_controller_reuses_resume2a_probe_worker():
    path = Path(resume3.__file__).resolve()
    text = path.read_text(encoding="utf-8")
    assert "phase3_14b_r258_stages_resume2a_worker.py" in text
    assert '"environment-probe"' in text


def test_cold_science_worker_payload_with_fake_runtime(monkeypatch):
    stages = SimpleNamespace(
        validate_base_report=lambda root: {
            "functional_projection": fake_functional(),
            "repository": {"head": "base"},
        },
        stable_json_bytes=resume3.stable_json_bytes,
        sha256_bytes=resume3.sha256_bytes,
        current_fit_projection=lambda replay: replay["current_fit_projection"],
        validate_confirmation_projection=lambda functional: fake_summary(),
    )
    stages_resume1 = SimpleNamespace(
        corrected_functional_projection=lambda wrapper, stages: copy.deepcopy(
            wrapper["functional_projection"]
        )
    )
    resume2a = SimpleNamespace(
        validate_probe_payload=lambda payload: payload["environment"]
    )
    stagea = SimpleNamespace(
        validate_environment_payload=lambda environment: None,
        assert_cold_cuda_context_portable=lambda: {
            "checked": True,
            "torch_cuda_is_initialized": False,
        },
    )
    stageo = SimpleNamespace(
        _runtime_modules=lambda: {
            "stagel": SimpleNamespace(
                stagef=SimpleNamespace(
                    stagec258=SimpleNamespace(stagea258=stagea)
                )
            )
        }
    )
    stager_resume1 = SimpleNamespace(
        execute_recovery=lambda **kwargs: {
            "functional_projection": fake_functional(),
            "current_fit_projection": fake_fit_identity(),
        }
    )
    monkeypatch.setattr(
        resume3,
        "_load_science_modules",
        lambda: {
            "stageo": stageo,
            "stages": stages,
            "stages_resume1": stages_resume1,
            "resume2a": resume2a,
            "stager_resume1": stager_resume1,
        },
    )
    for key, value in resume3.EXPECTED_ENV.items():
        monkeypatch.setenv(key, value)
    result = resume3.cold_science_worker_payload(
        worker_id="worker-1",
        root=Path("/tmp"),
        probe_payload=fake_probe(),
        repository_head="implementation-head",
    )
    assert result["execution_verdict"] == "PASS"
    assert result["cold_cuda_precheck"]["torch_cuda_is_initialized"] is False
    assert result["total_oof_fit_count"] == 54


def test_run_confirmation_process_order_and_totals(monkeypatch, tmp_path):
    stages, _, _ = install_fake_runtime_modules(monkeypatch)
    for key, value in resume3.EXPECTED_ENV.items():
        monkeypatch.setenv(key, value)

    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "phase3_14b_r258_stages_resume2a_worker.py").write_text("", encoding="utf-8")
    (scripts / "phase3_14b_r258_stages_resume3_worker.py").write_text("", encoding="utf-8")

    calls = []

    def fake_run(command, *, root, label):
        calls.append(label)
        output = Path(command[command.index("--output") + 1])
        if "probe" in label:
            payload = fake_probe(101)
        else:
            index = 1 if label.endswith("1") else 2
            payload = fake_worker(f"worker-{index}", 200 + index)
        output.write_text(json.dumps(payload), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(resume3, "_run_child", fake_run)
    repository = {
        "head": "implementation-head",
        "base_report": {"functional_projection": fake_functional()},
        "resume2a_report": {},
    }
    result = resume3.run_confirmation(
        root=tmp_path,
        repository=repository,
        python_bin=sys.executable,
    )
    assert calls == [
        "disposable environment probe",
        "cold science worker 1",
        "cold science worker 2",
    ]
    assert result["confirmation_execution"]["total_oof_fit_count"] == 108
    assert result["confirmation_execution"]["total_callback_pair_count"] == 54
    assert result["confirmation_execution"][
        "total_internal_scale_attempt_count"
    ] == 378
    assert result["confirmation_summary"]["nonzero_support_cell_count"] == 27


def test_run_confirmation_probe_failure_starts_no_science(monkeypatch, tmp_path):
    install_fake_runtime_modules(monkeypatch)
    for key, value in resume3.EXPECTED_ENV.items():
        monkeypatch.setenv(key, value)
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "phase3_14b_r258_stages_resume2a_worker.py").write_text("", encoding="utf-8")
    (scripts / "phase3_14b_r258_stages_resume3_worker.py").write_text("", encoding="utf-8")
    calls = []

    def fail_probe(command, *, root, label):
        calls.append(label)
        raise resume3.StageSResume3Error("probe failed")

    monkeypatch.setattr(resume3, "_run_child", fail_probe)
    with pytest.raises(resume3.StageSResume3Error):
        resume3.run_confirmation(
            root=tmp_path,
            repository={
                "head": "implementation-head",
                "base_report": {"functional_projection": fake_functional()},
            },
            python_bin=sys.executable,
        )
    assert calls == ["disposable environment probe"]


def test_run_confirmation_worker1_failure_starts_no_worker2(monkeypatch, tmp_path):
    install_fake_runtime_modules(monkeypatch)
    for key, value in resume3.EXPECTED_ENV.items():
        monkeypatch.setenv(key, value)
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "phase3_14b_r258_stages_resume2a_worker.py").write_text("", encoding="utf-8")
    (scripts / "phase3_14b_r258_stages_resume3_worker.py").write_text("", encoding="utf-8")
    calls = []

    def fail_worker1(command, *, root, label):
        calls.append(label)
        output = Path(command[command.index("--output") + 1])
        if label == "disposable environment probe":
            output.write_text(json.dumps(fake_probe()), encoding="utf-8")
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        raise resume3.StageSResume3Error("worker1 failed")

    monkeypatch.setattr(resume3, "_run_child", fail_worker1)
    with pytest.raises(resume3.StageSResume3Error):
        resume3.run_confirmation(
            root=tmp_path,
            repository={
                "head": "implementation-head",
                "base_report": {"functional_projection": fake_functional()},
            },
            python_bin=sys.executable,
        )
    assert calls == [
        "disposable environment probe",
        "cold science worker 1",
    ]
