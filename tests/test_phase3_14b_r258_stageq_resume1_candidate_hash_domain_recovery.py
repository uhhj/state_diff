from __future__ import annotations

import copy
import math
from types import SimpleNamespace

import numpy as np
import pytest

from ccda_phase3 import (
    phase314b_r258_stageq_resume1_candidate_hash_domain_recovery as resume1,
)


class FakeDefinition:
    maximum_scale = 1.0

    def validate(self):
        return None


class FakeStageE:
    @staticmethod
    def sha256_array(value):
        return resume1.stagee_array_sha_contract(value)

    @staticmethod
    def segment_bounds(*, definition, context):
        return {
            "lower": np.full((4, 23), 0.01, dtype=np.float64),
            "upper": np.full((4, 23), 10.0, dtype=np.float64),
        }

    @staticmethod
    def reconstruct_segment_vectors(
        *, proposed, control, lower, upper, coordinate_abs_max, epsilon
    ):
        return {
            "candidate": np.asarray(proposed, dtype=np.float32).copy(),
            "bound_pass": np.ones(proposed.shape[0], dtype=np.bool_),
            "coordinate_possible": np.ones(proposed.shape[0], dtype=np.bool_),
        }


class FakeStageP:
    @staticmethod
    def sha256_array(value):
        return resume1.stageop_array_sha_contract(value)


class FakeStageQ:
    @staticmethod
    def sha256_array(value):
        return resume1.stageop_array_sha_contract(value)

    @staticmethod
    def _expected_internal_order(runtime):
        return (1.0, 0.5, 0.25)


class FakeStageF:
    @staticmethod
    def fixed_integrator_definition():
        return FakeDefinition()


class FakeCallbackModule:
    def __init__(self, integration, capture):
        self.integration = integration
        self.capture = capture
        self.calls = 0

    def callback_integrate_rowwise(self, **kwargs):
        self.calls += 1
        return self.integration, self.capture


def make_fixture(rows=3):
    control = np.zeros((rows, 4, 48), dtype=np.float32)
    direction = np.full((rows, 4, 48), 0.125, dtype=np.float64)
    external = 0.25
    proposed = direction * external
    scales = (1.0, 0.5, 0.25)
    scale_candidates = []
    base_attempts = []
    events = []
    for scale in scales:
        raw = (control.astype(np.float64) + scale * proposed).astype(np.float32)
        candidate = raw.copy()
        scale_candidates.append(
            {
                "scale": scale,
                "candidate_sha256": resume1.stagee_array_sha_contract(candidate),
            }
        )
        base_attempts.append(
            {
                "internal_scale": scale,
                "raw_proposal_sha256": resume1.stageop_array_sha_contract(raw),
                "reconstructed_candidate_sha256": (
                    resume1.stageop_array_sha_contract(candidate)
                ),
            }
        )
        events.append(
            {
                "event_type": "scale_attempt",
                "attempted_scale": scale,
            }
        )
    selected = np.zeros(rows, dtype=np.float64)
    final_candidate = control.copy()
    integration = {
        "selected_scale": selected,
        "candidate": final_candidate,
        "candidate_sha256": resume1.stagee_array_sha_contract(final_candidate),
        "scale_candidates": scale_candidates,
    }
    capture = {
        "events": events,
        "events_sha256": "capture-sha",
        "returned_result_bit_exact": True,
    }
    base_cell = {
        "source_id": "raw_oracle",
        "timestep": 10,
        "final_selected_scale_sha256": resume1.stageop_array_sha_contract(selected),
        "final_candidate_sha256": resume1.stagee_array_sha_contract(final_candidate),
        "callback_capture_sha256": "capture-sha",
        "attempt_records": base_attempts,
    }
    stagek = FakeCallbackModule(integration, capture)
    runtime = {
        "stageq": FakeStageQ,
        "stageo": FakeStageQ,
        "stagep": FakeStageP,
        "stagel": SimpleNamespace(stagee258=FakeStageE),
        "stagef": FakeStageF,
        "stagek": stagek,
        "integrator_spec": SimpleNamespace(segment_tolerance=1.0e-6),
        "callback_spec": SimpleNamespace(name="callback"),
    }
    context = {
        "historical_geometry": SimpleNamespace(coordinate_abs_max=10.0),
    }
    spec = SimpleNamespace(external_multiplier=external)
    return {
        "control": control,
        "direction": direction,
        "runtime": runtime,
        "context": context,
        "spec": spec,
        "integration": integration,
        "capture": capture,
        "base_cell": base_cell,
        "stagek": stagek,
    }


def preflight(fixture):
    return resume1.prepare_hash_domain_recovery(
        source_id="raw_oracle",
        timestep=10,
        control=fixture["control"],
        base_direction=fixture["direction"],
        context=fixture["context"],
        runtime=fixture["runtime"],
        base_cell=fixture["base_cell"],
        spec=fixture["spec"],
        integration=fixture["integration"],
        capture=fixture["capture"],
    )


def test_phase_and_schema_are_resume1():
    assert "Resume1" in resume1.PHASE
    assert resume1.SCHEMA.endswith("_v1")


def test_commit_chain_constants():
    assert resume1.BASE_STAGEQ_IMPLEMENTATION_COMMIT == (
        "8597b2da8c5a1270887d4bd5c7bd0816770ec759"
    )
    assert resume1.BASE_STAGEQ_BLOCKED_EVIDENCE_COMMIT == (
        "d328122fb299167f9bc0c453ff7d45405d6467e7"
    )


def test_blocked_report_sha_constant():
    assert resume1.EXPECTED_STAGEQ_BLOCKED_REPORT_SHA256 == (
        "378f97da7e46aad3cb35d3eef404314974c06199c4a43ed25dfc0fab5838584c"
    )


def test_stageq_source_sha_constant():
    assert resume1.EXPECTED_STAGEQ_SOURCE_SHA256 == (
        "42f7577ec15f38e5222221595fa424c3276336ac383796c67e88db382df78884"
    )


def test_add_only_population_has_three_paths():
    assert len(resume1.IMPLEMENTATION_PATHS) == 3
    assert all(status == "A" for status, _ in resume1.IMPLEMENTATION_PATHS)


def test_stagee_and_stageop_hash_domains_are_distinct():
    value = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    assert resume1.stagee_array_sha_contract(value) != (
        resume1.stageop_array_sha_contract(value)
    )


@pytest.mark.parametrize(
    "dtype",
    [np.float32, np.float64, np.int32, np.bool_],
)
def test_hash_domains_are_deterministic(dtype):
    value = np.arange(24).reshape(2, 3, 4).astype(dtype)
    assert resume1.stagee_array_sha_contract(value) == (
        resume1.stagee_array_sha_contract(value.copy())
    )
    assert resume1.stageop_array_sha_contract(value) == (
        resume1.stageop_array_sha_contract(value.copy())
    )


def test_hash_domains_change_with_dtype():
    value = np.arange(12).reshape(1, 3, 4)
    assert resume1.stagee_array_sha_contract(value.astype(np.float32)) != (
        resume1.stagee_array_sha_contract(value.astype(np.float64))
    )


def test_hash_domains_change_with_shape():
    value = np.arange(12, dtype=np.float32)
    assert resume1.stageop_array_sha_contract(value.reshape(1, 3, 4)) != (
        resume1.stageop_array_sha_contract(value.reshape(1, 4, 3))
    )


def test_preflight_passes_valid_fixture():
    result = preflight(make_fixture())
    assert result["all_hash_domains_validated"] is True
    assert result["candidate_bytes_modified"] is False
    assert len(result["per_scale_records"]) == 3


def test_preflight_keeps_original_base_cell_unchanged():
    fixture = make_fixture()
    before = copy.deepcopy(fixture["base_cell"])
    result = preflight(fixture)
    assert fixture["base_cell"] == before
    assert result["corrected_base_cell"] != before


def test_preflight_adapts_only_per_scale_candidate_hash_labels():
    fixture = make_fixture()
    result = preflight(fixture)
    corrected = result["corrected_base_cell"]
    for original, adapted in zip(
        fixture["base_cell"]["attempt_records"],
        corrected["attempt_records"],
    ):
        assert original["raw_proposal_sha256"] == adapted["raw_proposal_sha256"]
        assert original["reconstructed_candidate_sha256"] != (
            adapted["reconstructed_candidate_sha256"]
        )


def test_preflight_final_stagee_and_stagep_hashes_are_distinct():
    result = preflight(make_fixture())
    assert result["final_candidate_stagee_sha256"] != (
        result["final_candidate_stagep_sha256"]
    )


def test_preflight_rejects_source_change():
    fixture = make_fixture()
    fixture["base_cell"]["source_id"] = "projected_oracle"
    with pytest.raises(resume1.StageQResume1Error, match="final hash-domain"):
        preflight(fixture)


def test_preflight_rejects_timestep_change():
    fixture = make_fixture()
    fixture["base_cell"]["timestep"] = 25
    with pytest.raises(resume1.StageQResume1Error, match="final hash-domain"):
        preflight(fixture)


def test_preflight_rejects_selected_scale_sha_change():
    fixture = make_fixture()
    fixture["base_cell"]["final_selected_scale_sha256"] = "bad"
    with pytest.raises(resume1.StageQResume1Error, match="final hash-domain"):
        preflight(fixture)


def test_preflight_rejects_declared_final_candidate_sha_change():
    fixture = make_fixture()
    fixture["integration"]["candidate_sha256"] = "bad"
    with pytest.raises(resume1.StageQResume1Error, match="final hash-domain"):
        preflight(fixture)


def test_preflight_rejects_stagep_final_candidate_field_change():
    fixture = make_fixture()
    fixture["base_cell"]["final_candidate_sha256"] = "bad"
    with pytest.raises(resume1.StageQResume1Error, match="final hash-domain"):
        preflight(fixture)


def test_preflight_rejects_callback_capture_change():
    fixture = make_fixture()
    fixture["capture"]["events_sha256"] = "changed"
    with pytest.raises(resume1.StageQResume1Error, match="final hash-domain"):
        preflight(fixture)


def test_preflight_rejects_non_bit_exact_callback():
    fixture = make_fixture()
    fixture["capture"]["returned_result_bit_exact"] = False
    with pytest.raises(resume1.StageQResume1Error, match="final hash-domain"):
        preflight(fixture)


def test_preflight_rejects_callback_order_change():
    fixture = make_fixture()
    fixture["capture"]["events"][0], fixture["capture"]["events"][1] = (
        fixture["capture"]["events"][1],
        fixture["capture"]["events"][0],
    )
    with pytest.raises(resume1.StageQResume1Error, match="attempt order"):
        preflight(fixture)


def test_preflight_rejects_attempt_population_change():
    fixture = make_fixture()
    fixture["base_cell"]["attempt_records"].pop()
    with pytest.raises(resume1.StageQResume1Error, match="population"):
        preflight(fixture)


def test_preflight_rejects_attempt_scale_change():
    fixture = make_fixture()
    fixture["base_cell"]["attempt_records"][0]["internal_scale"] = 0.75
    with pytest.raises(resume1.StageQResume1Error, match="attempt order"):
        preflight(fixture)


def test_preflight_rejects_legacy_stagee_scale_sha_change():
    fixture = make_fixture()
    fixture["integration"]["scale_candidates"][0]["candidate_sha256"] = "bad"
    with pytest.raises(resume1.StageQResume1Error, match="per-scale"):
        preflight(fixture)


def test_preflight_rejects_stagep_candidate_sha_change():
    fixture = make_fixture()
    fixture["base_cell"]["attempt_records"][0][
        "reconstructed_candidate_sha256"
    ] = "bad"
    with pytest.raises(resume1.StageQResume1Error, match="per-scale"):
        preflight(fixture)


def test_preflight_rejects_stagep_raw_proposal_sha_change():
    fixture = make_fixture()
    fixture["base_cell"]["attempt_records"][0]["raw_proposal_sha256"] = "bad"
    with pytest.raises(resume1.StageQResume1Error, match="per-scale"):
        preflight(fixture)


def test_preflight_rejects_external_multiplier_change():
    fixture = make_fixture()
    fixture["spec"].external_multiplier = 0.5
    with pytest.raises(resume1.StageQResume1Error, match="multiplier"):
        preflight(fixture)


def test_candidate_tensor_rejects_wrong_dtype():
    with pytest.raises(resume1.StageQResume1Error, match="float32"):
        resume1._candidate_tensor(np.zeros((2, 3, 4), dtype=np.float64), "x")


def test_candidate_tensor_rejects_wrong_rank():
    with pytest.raises(resume1.StageQResume1Error, match="float32"):
        resume1._candidate_tensor(np.zeros((2, 3), dtype=np.float32), "x")


def test_candidate_tensor_rejects_nonfinite():
    value = np.zeros((2, 3, 4), dtype=np.float32)
    value[0, 0, 0] = np.nan
    with pytest.raises(resume1.StageQResume1Error, match="non-finite"):
        resume1._candidate_tensor(value, "x")


def test_selected_vector_rejects_wrong_rows():
    with pytest.raises(resume1.StageQResume1Error, match="selected-scale"):
        resume1._selected_scale_vector(np.zeros(2), 3)


def test_selected_vector_rejects_nonfinite():
    with pytest.raises(resume1.StageQResume1Error, match="selected-scale"):
        resume1._selected_scale_vector(np.array([0.0, np.inf]), 2)


def test_cached_callback_is_single_use_and_exact():
    fixture = make_fixture()
    proposed = fixture["direction"] * fixture["spec"].external_multiplier
    cached = resume1._SingleUseCachedCallback(
        integration=fixture["integration"],
        capture=fixture["capture"],
        expected_control=fixture["control"],
        expected_direction=proposed,
        expected_context=fixture["context"],
        expected_integrator_spec=fixture["runtime"]["integrator_spec"],
        expected_callback_spec=fixture["runtime"]["callback_spec"],
    )
    result = cached(
        control=fixture["control"],
        direction=proposed,
        context=fixture["context"],
        integrator_spec=fixture["runtime"]["integrator_spec"],
        callback_spec=fixture["runtime"]["callback_spec"],
    )
    assert result == (fixture["integration"], fixture["capture"])
    with pytest.raises(resume1.StageQResume1Error, match="more than once"):
        cached(
            control=fixture["control"],
            direction=proposed,
            context=fixture["context"],
            integrator_spec=fixture["runtime"]["integrator_spec"],
            callback_spec=fixture["runtime"]["callback_spec"],
        )


@pytest.mark.parametrize("field", ["control", "direction"])
def test_cached_callback_rejects_array_change(field):
    fixture = make_fixture()
    proposed = fixture["direction"] * fixture["spec"].external_multiplier
    cached = resume1._SingleUseCachedCallback(
        integration=fixture["integration"],
        capture=fixture["capture"],
        expected_control=fixture["control"],
        expected_direction=proposed,
        expected_context=fixture["context"],
        expected_integrator_spec=fixture["runtime"]["integrator_spec"],
        expected_callback_spec=fixture["runtime"]["callback_spec"],
    )
    kwargs = {
        "control": fixture["control"],
        "direction": proposed,
        "context": fixture["context"],
        "integrator_spec": fixture["runtime"]["integrator_spec"],
        "callback_spec": fixture["runtime"]["callback_spec"],
    }
    kwargs[field] = np.asarray(kwargs[field]).copy()
    kwargs[field].flat[0] += 1.0
    with pytest.raises(resume1.StageQResume1Error, match=field):
        cached(**kwargs)


def test_runtime_patch_dispatches_candidate_and_selected_domains():
    fixture = make_fixture()
    proposed = fixture["direction"] * fixture["spec"].external_multiplier
    cached = resume1._SingleUseCachedCallback(
        integration=fixture["integration"],
        capture=fixture["capture"],
        expected_control=fixture["control"],
        expected_direction=proposed,
        expected_context=fixture["context"],
        expected_integrator_spec=fixture["runtime"]["integrator_spec"],
        expected_callback_spec=fixture["runtime"]["callback_spec"],
    )
    stageo = fixture["runtime"]["stageo"]
    stagek = fixture["runtime"]["stagek"]
    original_sha = stageo.sha256_array
    original_callback = stagek.callback_integrate_rowwise
    candidate_sha = resume1.stagee_array_sha_contract(
        fixture["integration"]["candidate"]
    )
    with resume1._patched_runtime_hash_and_callback(
        runtime=fixture["runtime"],
        candidate_stagee_sha=candidate_sha,
        cached_callback=cached,
    ):
        assert stageo.sha256_array(fixture["integration"]["candidate"]) == candidate_sha
        assert stageo.sha256_array(fixture["integration"]["selected_scale"]) == (
            resume1.stageop_array_sha_contract(
                fixture["integration"]["selected_scale"]
            )
        )
        assert stagek.callback_integrate_rowwise is cached
    assert stageo.sha256_array is original_sha
    assert stagek.callback_integrate_rowwise == original_callback


def test_runtime_patch_restores_after_exception():
    fixture = make_fixture()
    proposed = fixture["direction"] * fixture["spec"].external_multiplier
    cached = resume1._SingleUseCachedCallback(
        integration=fixture["integration"],
        capture=fixture["capture"],
        expected_control=fixture["control"],
        expected_direction=proposed,
        expected_context=fixture["context"],
        expected_integrator_spec=fixture["runtime"]["integrator_spec"],
        expected_callback_spec=fixture["runtime"]["callback_spec"],
    )
    stageo = fixture["runtime"]["stageo"]
    stagek = fixture["runtime"]["stagek"]
    original_sha = stageo.sha256_array
    original_callback = stagek.callback_integrate_rowwise
    with pytest.raises(RuntimeError):
        with resume1._patched_runtime_hash_and_callback(
            runtime=fixture["runtime"],
            candidate_stagee_sha=resume1.stagee_array_sha_contract(
                fixture["integration"]["candidate"]
            ),
            cached_callback=cached,
        ):
            raise RuntimeError("synthetic")
    assert stageo.sha256_array is original_sha
    assert stagek.callback_integrate_rowwise == original_callback


def test_recovered_audit_calls_original_logic_with_one_cached_pair():
    fixture = make_fixture()
    observed = {"calls": 0}

    def original_audit(**kwargs):
        observed["calls"] += 1
        runtime = kwargs["runtime"]
        integration, capture = runtime["stagek"].callback_integrate_rowwise(
            control=kwargs["control"],
            direction=kwargs["base_direction"] * kwargs["spec"].external_multiplier,
            context=kwargs["context"],
            integrator_spec=runtime["integrator_spec"],
            callback_spec=runtime["callback_spec"],
        )
        selected = np.asarray(integration["selected_scale"], dtype=np.float64)
        candidate = np.asarray(integration["candidate"], dtype=np.float32)
        assert runtime["stageo"].sha256_array(selected) == kwargs["base_cell"][
            "final_selected_scale_sha256"
        ]
        assert runtime["stageo"].sha256_array(candidate) == kwargs["base_cell"][
            "final_candidate_sha256"
        ]
        for record, legacy in zip(
            kwargs["base_cell"]["attempt_records"],
            integration["scale_candidates"],
        ):
            assert record["reconstructed_candidate_sha256"] == legacy[
                "candidate_sha256"
            ]
        return {"legacy_identity": {"all_exact": True}}

    result = resume1.recovered_audit_shadow_cell(
        original_audit=original_audit,
        source_id="raw_oracle",
        timestep=10,
        control=fixture["control"],
        base_direction=fixture["direction"],
        context=fixture["context"],
        runtime=fixture["runtime"],
        base_cell=fixture["base_cell"],
        spec=fixture["spec"],
    )
    assert observed["calls"] == 1
    assert fixture["stagek"].calls == 1
    assert result["hash_domain_recovery"]["callback_pair_run_count"] == 1
    assert result["hash_domain_recovery"]["original_stageq_logic_called"] is True


def test_recovered_audit_restores_runtime_after_original_failure():
    fixture = make_fixture()
    stageo = fixture["runtime"]["stageo"]
    stagek = fixture["runtime"]["stagek"]
    original_sha = stageo.sha256_array
    original_callback = stagek.callback_integrate_rowwise

    def original_audit(**kwargs):
        kwargs["runtime"]["stagek"].callback_integrate_rowwise(
            control=kwargs["control"],
            direction=kwargs["base_direction"] * kwargs["spec"].external_multiplier,
            context=kwargs["context"],
            integrator_spec=kwargs["runtime"]["integrator_spec"],
            callback_spec=kwargs["runtime"]["callback_spec"],
        )
        raise RuntimeError("synthetic")

    with pytest.raises(RuntimeError):
        resume1.recovered_audit_shadow_cell(
            original_audit=original_audit,
            source_id="raw_oracle",
            timestep=10,
            control=fixture["control"],
            base_direction=fixture["direction"],
            context=fixture["context"],
            runtime=fixture["runtime"],
            base_cell=fixture["base_cell"],
            spec=fixture["spec"],
        )
    assert stageo.sha256_array is original_sha
    assert stagek.callback_integrate_rowwise == original_callback


def test_patched_stageq_audit_restores_original():
    original = lambda **kwargs: {"ok": True}
    fake = SimpleNamespace(audit_shadow_cell=original)
    with resume1.patched_stageq_cell_audit(fake):
        assert fake.audit_shadow_cell is not original
    assert fake.audit_shadow_cell is original


def test_patched_stageq_audit_restores_after_exception():
    original = lambda **kwargs: {"ok": True}
    fake = SimpleNamespace(audit_shadow_cell=original)
    with pytest.raises(RuntimeError):
        with resume1.patched_stageq_cell_audit(fake):
            raise RuntimeError("synthetic")
    assert fake.audit_shadow_cell is original


def test_blocked_report_preserves_original_provenance():
    payload = resume1.blocked_report(
        repository={"head": "x"}, error=RuntimeError("synthetic")
    )
    assert payload["execution_verdict"] == "BLOCKED"
    assert payload["original_stage_q_blocked_report_preserved"] is True
    assert payload["original_stage_q_source_modified"] is False
    assert payload["candidate_bytes_modified"] is False
    assert payload["callback_pairs_duplicated"] is False
    assert payload["selected_configuration"] is None
    assert payload["train_only_recommendation"] is None


def test_false_boundaries_are_unique():
    assert len(resume1.FALSE_BOUNDARIES) == len(set(resume1.FALSE_BOUNDARIES))


def test_expected_environment_contract_is_complete():
    assert resume1.EXPECTED_ENV == {
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
    }


def test_replay_population_is_frozen():
    assert resume1.EXPECTED_PAIR_COUNT == 6
    assert resume1.EXPECTED_TIMESTEPS == (10, 25, 50)
    assert resume1.ORACLE_SOURCES == ("raw_oracle", "projected_oracle")
    assert resume1.EXTERNAL_MULTIPLIER == 0.25
