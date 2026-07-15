from __future__ import annotations

from pathlib import Path

import pytest

from ccda_phase3.phase314b_r256_staged3_resume1_upper_gate_freeze import (
    UpperGateFreezeError,
    d2_exactness_signature,
    source_logic_audit,
    validate_blocked_summary_payload,
    validate_blocked_test_gate_payload,
)


def valid_blocked_test_gate():
    return {
        "verdict": "PASS",
        "passed_test_count": 597,
        "test_file_count": 37,
        "staged3_new_passed": 24,
    }


def valid_blocked_summary():
    return {
        "verdict": "BLOCKED",
        "scientific_status": "BLOCKED",
        "root_cause":
            "phase314b_r256_staged3_execution_failed_before_completion",
        "gate_selected": False,
        "threshold_recomputed": False,
        "train_only_recommendation": None,
        "selected_configuration": None,
    }


def test_d2_exactness_signature_accepts_frozen_contract():
    source = """
@dataclass(frozen=True)
class ProvenanceAuditSpec:
    cache_float32_exact_required: bool = True

exact = bool(np.array_equal(raw_xy32, cached_xy32))
if spec.cache_float32_exact_required and not np.all(xy_exact):
    raise CollapseProvenanceError("mismatch")
"""
    assert d2_exactness_signature(source)


def test_d2_exactness_signature_rejects_blocked_alias():
    source = """
raw_xy_float32_exact_required: bool = True
exact = bool(np.array_equal(raw_xy32, cached_xy32))
if spec.raw_xy_float32_exact_required and not np.all(xy_exact):
    raise RuntimeError("mismatch")
"""
    assert not d2_exactness_signature(source)


def test_blocked_test_gate_payload_is_bound():
    validate_blocked_test_gate_payload(valid_blocked_test_gate())


def test_blocked_test_gate_payload_rejects_count_change():
    payload = valid_blocked_test_gate()
    payload["passed_test_count"] = 596
    with pytest.raises(UpperGateFreezeError):
        validate_blocked_test_gate_payload(payload)


def test_blocked_summary_payload_is_bound():
    validate_blocked_summary_payload(valid_blocked_summary())


def test_source_logic_audit_accepts_actual_frozen_sources():
    repository_root = Path(__file__).resolve().parents[1]
    result = source_logic_audit(repository_root)
    assert result["checks"][
        "stage_d2_requires_raw_xy_cache_exact"
    ]
    assert result["all_confirmed"]
