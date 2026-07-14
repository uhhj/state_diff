import copy
import json
from pathlib import Path

import pytest

from ccda_phase3.phase314b_r254_resume4_reproduction_audit import (
    DIAGNOSTIC_OBJECTIVE_NAMES,
    EXPECTED_PHASE3_R2_PASS_COUNT,
    PHASE3_R2_TEST_PATHS,
    R253_PILOT_PATH,
    R253_SUMMARY_PATH,
    RESUME3_BLOCKED_PATH,
    RESUME3_PILOT_PATH,
    ReproductionAuditSpec,
    audit_robot_proxy_sources,
    build_reproduction_audit,
    parse_pytest_pass_count,
    phase3_r2_test_manifest,
    render_markdown,
)


ROOT = Path(__file__).resolve().parents[1]


def load(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def source_audit():
    task = """
states = [p.getJointState(body_id, index) for index in range(count)]
link = p.getLinkState(body_id, count - 1)
except Exception:
    source = 'missing_zero_proxy'
"""
    environment = """
self.joints = [j[0] for j in joints if j[2] == p.JOINT_REVOLUTE]
self.ee_tip_link = 12
"""
    return audit_robot_proxy_sources(task, environment)


def committed_audit():
    return build_reproduction_audit(
        r253_pilot=load(R253_PILOT_PATH),
        r253_summary=load(R253_SUMMARY_PATH),
        resume3_pilot=load(RESUME3_PILOT_PATH),
        resume3_blocked=load(RESUME3_BLOCKED_PATH),
        robot_proxy_source_audit=source_audit(),
    )


def test_spec_rejects_negative_tolerance():
    with pytest.raises(ValueError):
        ReproductionAuditSpec(historical_absolute_tolerance=-1.0).validate()
    manifest = phase3_r2_test_manifest(ROOT)
    assert tuple(manifest) == tuple(sorted(PHASE3_R2_TEST_PATHS))
    assert len(manifest) == 27
    assert parse_pytest_pass_count("425 passed in 9.1s") == (
        EXPECTED_PHASE3_R2_PASS_COUNT
    )
    with pytest.raises(ValueError, match="not all-pass"):
        parse_pytest_pass_count("424 passed, 1 skipped in 9.1s")


def test_static_source_audit_finds_three_proxy_defects():
    result = source_audit()
    assert result["source_patterns_recognized"] is True
    assert {item["id"] for item in result["findings"]} == {
        "robot_proxy_joint_selection_includes_fixed_joints",
        "robot_proxy_end_effector_link_not_environment_tip_link",
        "robot_proxy_collection_errors_silently_zero_filled",
    }


def test_committed_evidence_classifies_identity_as_unobservable():
    report = committed_audit()
    assert report["verdict"] == "PASS"
    assert report["scientific_status"] == "BLOCKED"
    assert report["root_cause"] == (
        "phase314b_r254_resume4_residual_reproduction_identity_unobservable"
    )
    assert report["adapter_schema_pass"] is True
    assert report["pipeline_identity"]["fixed_pipeline_identity_pass"] is True
    assert report["current_run_internal_metric_reproduction_pass"] is True
    assert report["residual_training_identity_complete"] is False
    assert report["aggregate"] == {
        "model_count": 3,
        "historical_reproduction_pass_count": 0,
        "historical_reproduction_fail_count": 3,
        "contract_boolean_change_count": 3,
        "continuous_metric_failure_count": 15,
    }


def test_every_model_has_complete_scalar_comparison_table():
    report = committed_audit()
    assert tuple(report["models"]) == DIAGNOSTIC_OBJECTIVE_NAMES
    for model in report["models"].values():
        assert len(model["checks"]) == 11
        assert len(model["training_metric_reproduction"]["checks"]) == 4
        assert model["training_metric_reproduction"]["all_pass"] is True


def test_historical_expected_binding_change_is_detected():
    current = load(RESUME3_PILOT_PATH)
    current = copy.deepcopy(current)
    model = DIAGNOSTIC_OBJECTIVE_NAMES[0]
    current["variants"][model]["r253_reproduction"]["checks"][
        "state_group_z_metrics.full_z_mse"
    ]["expected"] += 1.0
    report = build_reproduction_audit(
        r253_pilot=load(R253_PILOT_PATH),
        r253_summary=load(R253_SUMMARY_PATH),
        resume3_pilot=current,
        resume3_blocked=load(RESUME3_BLOCKED_PATH),
    )
    assert report["adapter_schema_pass"] is False
    assert report["root_cause"] == (
        "phase314b_r254_resume4_reproduction_adapter_defect_supported"
    )


def test_missing_check_is_detected_without_key_error():
    current = copy.deepcopy(load(RESUME3_PILOT_PATH))
    model = DIAGNOSTIC_OBJECTIVE_NAMES[0]
    del current["variants"][model]["r253_reproduction"]["checks"][
        "contract.branch_transport_pass"
    ]
    report = build_reproduction_audit(
        r253_pilot=load(R253_PILOT_PATH),
        r253_summary=load(R253_SUMMARY_PATH),
        resume3_pilot=current,
        resume3_blocked=load(RESUME3_BLOCKED_PATH),
    )
    assert report["adapter_schema_pass"] is False


def test_nonfinite_metric_is_rejected():
    current = copy.deepcopy(load(RESUME3_PILOT_PATH))
    model = DIAGNOSTIC_OBJECTIVE_NAMES[0]
    current["variants"][model]["r253_reproduction"]["checks"][
        "state_group_z_metrics.full_z_mse"
    ]["observed"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        build_reproduction_audit(
            r253_pilot=load(R253_PILOT_PATH),
            r253_summary=load(R253_SUMMARY_PATH),
            resume3_pilot=current,
            resume3_blocked=load(RESUME3_BLOCKED_PATH),
        )


def test_markdown_preserves_blocking_boundary():
    report = committed_audit()
    markdown = render_markdown(report, 405)
    assert "Scientific status: `BLOCKED`" in markdown
    assert "Static tests: `405 passed`" in markdown
    assert "robot_proxy_joint_selection_includes_fixed_joints" in markdown
    assert report["next_stage"] in markdown
