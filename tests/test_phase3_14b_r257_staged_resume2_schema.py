from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ccda_phase3 import phase314b_r257_staged_resume2_schema as resume2


def canonical_payload():
    staged3_contract = {
        "contract_sha256": "3" * 64,
        "upper_threshold": 3.937946393999212,
    }
    stagea_contract = {
        "contract_sha256": "a" * 64,
        "selected_configuration": None,
    }
    return {
        "base_commit": "c" * 40,
        "upstream_immutable": {
            "stagea_contract": stagea_contract,
            "upstream_immutable": {
                "staged3_contract": staged3_contract,
                "other": {"value": 1},
            },
            "stageb_value": 2,
        },
        "stagec_value": 3,
    }


def test_commit_constants_are_frozen():
    assert resume2.RESUME1_IMPLEMENTATION_COMMIT == (
        "d34838bfebbde2376d128dff398dadcc97877191"
    )
    assert resume2.STAGED_BLOCKED_EVIDENCE_COMMIT == (
        "630cff532d811c889702bb236766b659bdfd83d4"
    )


def test_resume1_report_shas_are_frozen():
    assert resume2.EXPECTED_RESUME1_BLOCKED_SHA256 == (
        "ab1ec6cd75c7164b29da29e296925fb9ed9897bcc552fd835060e0b1e1c4eb24"
    )
    assert resume2.EXPECTED_RESUME1_TEST_GATE_SHA256 == (
        "d31a882153134304b37f3fd2b16b853fdec8c177ef49c3ee879813a43ea06298"
    )


def test_stable_json_is_deterministic():
    assert resume2.stable_json_bytes(
        {"b": 2, "a": 1}
    ) == resume2.stable_json_bytes(
        {"a": 1, "b": 2}
    )


def test_stable_json_supports_numpy():
    payload = resume2.stable_json_bytes(
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
    resume2.atomic_write_once(path, b"{}\n")
    assert path.read_bytes() == b"{}\n"
    with pytest.raises(FileExistsError):
        resume2.atomic_write_once(path, b"{}\n")


def test_load_json_rejects_non_object(tmp_path):
    path = tmp_path / "value.json"
    path.write_text("[1]\n", encoding="utf-8")
    with pytest.raises(resume2.Resume2SchemaError):
        resume2.load_json(path)


def test_require_mapping_accepts_mapping():
    value = {"x": 1}
    assert resume2.require_mapping(
        value,
        path="x",
    ) is value


def test_require_mapping_rejects_scalar():
    with pytest.raises(
        resume2.Resume2SchemaError,
        match="schema path",
    ):
        resume2.require_mapping(1, path="x")


def test_resolve_canonical_schema_paths():
    resolved = resume2.resolve_canonical_stagec_schema(
        canonical_payload()
    )
    assert resolved["stagea_contract"][
        "contract_sha256"
    ] == "a" * 64
    assert resolved["staged3_contract"][
        "contract_sha256"
    ] == "3" * 64
    assert resolved[
        "canonical_stagea_contract_path"
    ] == "upstream_immutable.stagea_contract"
    assert resolved[
        "canonical_staged3_contract_path"
    ] == (
        "upstream_immutable."
        "upstream_immutable.staged3_contract"
    )


def test_resolve_rejects_missing_stageb_layer():
    with pytest.raises(
        resume2.Resume2SchemaError,
        match="stagec.upstream_immutable",
    ):
        resume2.resolve_canonical_stagec_schema({})


def test_resolve_rejects_missing_stagea_contract():
    payload = canonical_payload()
    del payload["upstream_immutable"]["stagea_contract"]
    with pytest.raises(
        resume2.Resume2SchemaError,
        match="stagea_contract",
    ):
        resume2.resolve_canonical_stagec_schema(
            payload
        )


def test_resolve_rejects_missing_staged3_contract():
    payload = canonical_payload()
    del payload["upstream_immutable"][
        "upstream_immutable"
    ]["staged3_contract"]
    with pytest.raises(
        resume2.Resume2SchemaError,
        match="staged3_contract",
    ):
        resume2.resolve_canonical_stagec_schema(
            payload
        )


def test_adapter_adds_exact_aliases_without_mutation():
    payload = canonical_payload()
    original = json.loads(
        json.dumps(payload)
    )
    adapted = resume2.adapt_stagec_schema_for_resume1(
        payload
    )
    assert payload == original
    assert adapted["stagea_contract"] == (
        payload["upstream_immutable"][
            "stagea_contract"
        ]
    )
    assert adapted["upstream_immutable"][
        "staged3_contract"
    ] == (
        payload["upstream_immutable"][
            "upstream_immutable"
        ]["staged3_contract"]
    )


def test_adapter_rejects_conflicting_top_alias():
    payload = canonical_payload()
    payload["stagea_contract"] = {
        "contract_sha256": "x" * 64,
    }
    with pytest.raises(
        resume2.Resume2SchemaError,
        match="top-level",
    ):
        resume2.adapt_stagec_schema_for_resume1(
            payload
        )


def test_adapter_rejects_conflicting_upstream_alias():
    payload = canonical_payload()
    payload["upstream_immutable"][
        "staged3_contract"
    ] = {
        "contract_sha256": "x" * 64,
    }
    with pytest.raises(
        resume2.Resume2SchemaError,
        match="staged3_contract alias",
    ):
        resume2.adapt_stagec_schema_for_resume1(
            payload
        )


def test_schema_audit_reports_exact_aliases():
    result = resume2.schema_audit(
        canonical_payload()
    )
    assert result["stagea_contract_alias_exact"]
    assert result["staged3_contract_alias_exact"]
    assert result["source_payload_unchanged"]
    assert result["correction_scope"] == (
        "schema compatibility aliases only"
    )


def test_context_manager_restores_validator_success(
    monkeypatch,
):
    original = lambda root: canonical_payload()
    monkeypatch.setattr(
        resume2.resume1.stagec_topk,
        "validate_immutable_inputs",
        original,
    )
    with resume2.adapted_stagec_validator():
        active = (
            resume2.resume1.stagec_topk
            .validate_immutable_inputs
        )
        assert active is not original
        result = active(Path("/tmp"))
        assert "stagea_contract" in result
        assert "staged3_contract" in (
            result["upstream_immutable"]
        )
    assert (
        resume2.resume1.stagec_topk
        .validate_immutable_inputs
        is original
    )


def test_context_manager_restores_validator_failure(
    monkeypatch,
):
    original = lambda root: canonical_payload()
    monkeypatch.setattr(
        resume2.resume1.stagec_topk,
        "validate_immutable_inputs",
        original,
    )
    with pytest.raises(RuntimeError):
        with resume2.adapted_stagec_validator():
            raise RuntimeError("stop")
    assert (
        resume2.resume1.stagec_topk
        .validate_immutable_inputs
        is original
    )


def test_run_resume2_applies_schema_and_restores(
    monkeypatch,
):
    canonical = canonical_payload()
    monkeypatch.setattr(
        resume2,
        "validate_resume1_failure",
        lambda root: {"blocked": True},
    )
    monkeypatch.setattr(
        resume2.stagec_topk,
        "validate_immutable_inputs",
        lambda root: canonical,
    )

    observed = {}

    def fake_resume1(*, root, spec):
        validator = (
            resume2.resume1.stagec_topk
            .validate_immutable_inputs
        )
        adapted = validator(root)
        observed["stagea"] = adapted[
            "stagea_contract"
        ]
        observed["staged3"] = adapted[
            "upstream_immutable"
        ]["staged3_contract"]
        return {
            "phase": "old",
            "phase_id": "old",
            "schema": "old",
            "verdict": "PASS",
            "scientific_status": "BLOCKED",
            "root_cause": "r",
            "required_next_path": "n",
            "control_trace_correction": {"exact": True},
            "split": {"x": 1},
            "calibration_contract": {
                "contract_sha256": "x" * 64,
                "value": 1,
            },
            "selection": {"x": 1},
            "classification": {"x": 1},
            "selected_configuration": None,
        }

    monkeypatch.setattr(
        resume2.resume1,
        "run_resume1",
        fake_resume1,
    )
    before = (
        resume2.resume1.stagec_topk
        .validate_immutable_inputs
    )
    result = resume2.run_resume2(
        root=Path("/tmp"),
        spec=None,
    )
    after = (
        resume2.resume1.stagec_topk
        .validate_immutable_inputs
    )
    assert before is after
    assert observed["stagea"] == (
        canonical["upstream_immutable"][
            "stagea_contract"
        ]
    )
    assert observed["staged3"] == (
        canonical["upstream_immutable"][
            "upstream_immutable"
        ]["staged3_contract"]
    )
    assert result["phase"] == resume2.PHASE
    assert result["resume2_correction_applied"]
    assert result["resume1_files_modified"] is False
    assert result["calibration_contract"][
        "resume2_schema_correction"
    ]["validator_restored"]


def test_compare_worker_results_exact():
    base = {
        "root_cause": "r",
        "required_next_path": "n",
        "resume2_schema_correction": {"x": 1},
        "control_trace_correction": {"x": 1},
        "split": {"x": 1},
        "calibration_contract": {"x": 1},
        "selection": {"x": 1},
        "classification": {"x": 1},
        "selected_configuration": None,
    }
    result = resume2.compare_worker_results(
        base,
        dict(base),
    )
    assert result["exact"]
    assert result["schema_correction_exact"]


def test_compare_worker_results_detects_schema_difference():
    left = {
        "root_cause": "r",
        "required_next_path": "n",
        "resume2_schema_correction": {"x": 1},
        "control_trace_correction": {"x": 1},
        "split": {"x": 1},
        "calibration_contract": {"x": 1},
        "selection": {"x": 1},
        "classification": {"x": 1},
        "selected_configuration": None,
    }
    right = dict(left)
    right["resume2_schema_correction"] = {"x": 2}
    result = resume2.compare_worker_results(
        left,
        right,
    )
    assert not result["exact"]
    assert not result["schema_correction_exact"]
