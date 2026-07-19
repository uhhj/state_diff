from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from ccda_phase3 import phase314b_r258_stagej_readonly_integrator_telemetry as stagej


def _load_fake_module(tmp_path: Path, body: str) -> ModuleType:
    path = tmp_path / "fake_integrator.py"
    path.write_text(body, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("fake_integrator_stagej", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fake_source() -> str:
    return (
        "import numpy as np\n"
        "def integrate_rowwise(*, control, direction, definition, context, spec):\n"
        "    retention = np.linalg.norm(direction.reshape(direction.shape[0], -1), axis=1)\n"
        "    retention_pass = retention >= spec.retention_min\n"
        "    finite_pass = np.all(np.isfinite(direction), axis=(1, 2))\n"
        "    upper_segment_pass = np.asarray(context['upper_pass'], dtype=bool)\n"
        "    final_pass = retention_pass & finite_pass & upper_segment_pass\n"
        "    selected_scale = np.where(final_pass, 1.0, 0.0)\n"
        "    candidate = control + direction.astype(control.dtype) * selected_scale[:, None, None]\n"
        "    return {'selected_scale': selected_scale, 'candidate': candidate}\n"
    )


def _args(rows: int = 4) -> dict:
    return {
        "control": np.zeros((rows, 2, 2), dtype=np.float32),
        "direction": np.ones((rows, 2, 2), dtype=np.float64),
        "definition": object(),
        "context": {"upper_pass": np.ones(rows, dtype=bool)},
        "integrator_spec": SimpleNamespace(retention_min=0.25),
        "telemetry_spec": stagej.ReadOnlyIntegratorTelemetrySpec(),
    }


def test_stagej_population_is_frozen() -> None:
    assert len(stagej.BASE_DIRECTION_IDS) == 9
    assert stagej.SCALE_MULTIPLIERS == (0.25, 0.5, 1.0, 2.0)
    assert stagej.SOURCE_IDS == (
        "oof_surrogate",
        "raw_oracle",
        "projected_oracle",
    )
    assert stagej.EXPECTED_SELECTED_ULP_FACTOR == 1.0


def test_stagej_spec_validates() -> None:
    spec = stagej.ReadOnlyIntegratorTelemetrySpec()
    spec.validate()
    assert spec.grouped_cv_folds == 6
    assert spec.maximum_trace_events == 4096
    assert spec.maximum_values_per_event == 32


def test_stagej_spec_rejects_scale_change() -> None:
    with pytest.raises(ValueError, match="scale bank"):
        stagej.ReadOnlyIntegratorTelemetrySpec(
            scale_multipliers=(0.5, 1.0),
        ).validate()


def test_predicate_expression_detects_compare() -> None:
    node = stagej.ast.parse("x = a > 0").body[0]
    assert isinstance(node, stagej.ast.Assign)
    assert stagej._predicate_expression(node.value)


def test_predicate_expression_detects_bitwise_gate() -> None:
    node = stagej.ast.parse("x = a & b").body[0]
    assert isinstance(node, stagej.ast.Assign)
    assert stagej._predicate_expression(node.value)


def test_source_index_finds_real_assignment_names(tmp_path: Path) -> None:
    module = _load_fake_module(tmp_path, _fake_source())
    result = stagej.predicate_source_index(module)
    names = {
        name
        for values in result["indexed_names_by_line"].values()
        for name in values
    }
    assert "retention_pass" in names
    assert "finite_pass" in names
    assert "upper_segment_pass" in names
    assert "final_pass" in names


def test_source_index_records_source_identity(tmp_path: Path) -> None:
    module = _load_fake_module(tmp_path, _fake_source())
    result = stagej.predicate_source_index(module)
    assert len(result["source_sha256"]) == 64
    assert len(result["integrate_rowwise_source_sha256"]) == 64
    assert result["indexed_name_count"] >= 4


def test_summarize_boolean_array() -> None:
    result = stagej.summarize_trace_value(
        "finite_pass",
        np.asarray([True, False, True]),
        rows=3,
        retention_minimum=0.25,
    )
    assert result is not None
    assert result["pass_rate"] == pytest.approx(2.0 / 3.0)
    assert result["false_count"] == 1


def test_summarize_retention_array() -> None:
    result = stagej.summarize_trace_value(
        "direction_retention",
        np.asarray([0.1, 0.3, 0.5]),
        rows=3,
        retention_minimum=0.25,
    )
    assert result is not None
    assert result["pass_rate"] == pytest.approx(2.0 / 3.0)
    assert result["pass_rule"] == "retention_ge_minimum"


def test_summarize_scale_array() -> None:
    result = stagej.summarize_trace_value(
        "selected_scale",
        np.asarray([0.0, 0.5, 1.0]),
        rows=3,
        retention_minimum=0.25,
    )
    assert result is not None
    assert result["pass_rate"] == pytest.approx(2.0 / 3.0)


def test_exact_identity_accepts_equal_arrays() -> None:
    left = {"x": np.asarray([1.0, 2.0], dtype=np.float32)}
    right = {"x": np.asarray([1.0, 2.0], dtype=np.float32)}
    stagej._exact_identity(left, right)


def test_exact_identity_rejects_dtype_change() -> None:
    with pytest.raises(stagej.ReadOnlyIntegratorTelemetryError, match="array contract"):
        stagej._exact_identity(
            np.asarray([1.0], dtype=np.float32),
            np.asarray([1.0], dtype=np.float64),
        )


def test_exact_identity_rejects_byte_change() -> None:
    with pytest.raises(stagej.ReadOnlyIntegratorTelemetryError, match="array bytes"):
        stagej._exact_identity(
            np.asarray([1.0], dtype=np.float32),
            np.asarray([2.0], dtype=np.float32),
        )


def test_traced_call_is_bit_exact(tmp_path: Path) -> None:
    module = _load_fake_module(tmp_path, _fake_source())
    source_index = stagej.predicate_source_index(module)
    result, telemetry = stagej.traced_integrate_rowwise(
        integrate=module.integrate_rowwise,
        source_index=source_index,
        **_args(),
    )
    assert np.all(result["selected_scale"] == 1.0)
    assert telemetry["returned_result_bit_exact_with_trace_disabled"] is True
    assert telemetry["event_count"] > 0
    assert telemetry["truncated"] is False


def test_traced_call_captures_predicate_values(tmp_path: Path) -> None:
    module = _load_fake_module(tmp_path, _fake_source())
    source_index = stagej.predicate_source_index(module)
    _, telemetry = stagej.traced_integrate_rowwise(
        integrate=module.integrate_rowwise,
        source_index=source_index,
        **_args(),
    )
    names = {
        value["name"]
        for event in telemetry["events"]
        for value in event["values"]
    }
    assert "retention_pass" in names
    assert "finite_pass" in names
    assert "final_pass" in names


def test_traced_call_restores_previous_trace(tmp_path: Path) -> None:
    module = _load_fake_module(tmp_path, _fake_source())
    source_index = stagej.predicate_source_index(module)
    assert sys.gettrace() is None
    stagej.traced_integrate_rowwise(
        integrate=module.integrate_rowwise,
        source_index=source_index,
        **_args(),
    )
    assert sys.gettrace() is None


def test_trace_leaf_aggregation_uses_minimum() -> None:
    telemetry = {
        "events": [
            {
                "sequence": 0,
                "function": "integrate_rowwise",
                "line": 10,
                "values": [
                    {
                        "name": "retention_pass",
                        "family": "direction_retention",
                        "pass_rate": 1.0,
                        "pass_rule": "boolean_true",
                        "kind": "boolean_array",
                    }
                ],
            },
            {
                "sequence": 1,
                "function": "integrate_rowwise",
                "line": 10,
                "values": [
                    {
                        "name": "retention_pass",
                        "family": "direction_retention",
                        "pass_rate": 0.0,
                        "pass_rule": "boolean_true",
                        "kind": "boolean_array",
                    }
                ],
            },
        ]
    }
    result = stagej._aggregate_trace_leaves([telemetry])
    key = "integrate_rowwise:10:retention_pass"
    assert result[key]["minimum_pass_rate"] == 0.0
    assert result[key]["maximum_pass_rate"] == 1.0


def _audit(acceptance: float, rate: float, family: str = "direction_retention") -> dict:
    key = "integrate_rowwise:10:retention_pass"
    return {
        "scale_bank_acceptance": {
            "rows_with_any_accepted_multiplier_rate": acceptance,
        },
        "trace_leaf_index": {
            key: {
                "key": key,
                "function": "integrate_rowwise",
                "line": 10,
                "name": "retention_pass",
                "family": family,
                "minimum_pass_rate": rate,
                "mean_pass_rate": rate,
                "maximum_pass_rate": rate,
                "observation_count": 4,
                "first_sequence": 2,
            }
        },
    }


def test_compare_identifies_first_retention_divergence() -> None:
    result = stagej.compare_trace_surfaces(
        oof=_audit(0.0, 0.0),
        raw=_audit(1.0, 1.0),
        projected=_audit(1.0, 1.0),
        spec=stagej.ReadOnlyIntegratorTelemetrySpec(),
    )
    assert result["locus"] == "oof_first_diverges_at_direction_retention"
    assert result["discriminator"]["family"] == "direction_retention"


def test_compare_blocks_when_oracle_not_admitted() -> None:
    result = stagej.compare_trace_surfaces(
        oof=_audit(0.0, 0.0),
        raw=_audit(0.0, 0.0),
        projected=_audit(0.0, 0.0),
        spec=stagej.ReadOnlyIntegratorTelemetrySpec(),
    )
    assert result["locus"] == "oracle_control_not_admitted"


def test_compare_reports_insufficient_trace() -> None:
    oof = _audit(0.0, 0.5)
    raw = _audit(1.0, 0.5)
    projected = _audit(1.0, 0.5)
    result = stagej.compare_trace_surfaces(
        oof=oof,
        raw=raw,
        projected=projected,
        spec=stagej.ReadOnlyIntegratorTelemetrySpec(),
    )
    assert result["locus"] == "readonly_trace_does_not_identify_rejection"


def test_classify_unanimous_retention() -> None:
    records = [
        {
            "trace_comparison": {
                "locus": "oof_first_diverges_at_direction_retention",
                "discriminator": {
                    "family": "direction_retention",
                    "key": "integrate_rowwise:10:retention_pass",
                },
            }
        }
        for _ in range(27)
    ]
    result = stagej.classify(records)
    assert result["root_cause"] == (
        "phase314b_r258_stagej_oof_rejected_by_direction_retention"
    )
    assert result["all_records_share_discriminator_family"] is True
    assert result["all_records_share_discriminator_key"] is True


def test_classify_trace_insufficient() -> None:
    records = [
        {
            "trace_comparison": {
                "locus": "readonly_trace_does_not_identify_rejection",
                "discriminator": None,
            }
        }
    ]
    result = stagej.classify(records)
    assert result["root_cause"] == (
        "phase314b_r258_stagej_readonly_trace_surface_insufficient"
    )


def test_stagej_does_not_modify_stagee_source() -> None:
    source = inspect.getsource(stagej)
    assert "stagee_source_modified\": False" in source
    assert "sys.settrace" in source
    assert "every_traced_result_compared_to_untraced_result" in source


def test_stagej_forbidden_boundaries_are_false_in_source() -> None:
    source = inspect.getsource(stagej.run_calibration)
    for field in (
        "frozen_probe_accessed",
        "formal_training_run",
        "reverse_sampling_run",
        "idm_run",
        "candidate_execution",
        "deformable_ravens_executed",
        "phase4",
        "cps",
    ):
        assert '"{}": False'.format(field) in source
