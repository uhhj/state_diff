from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ccda_phase3 import phase314b_r258_stagex_resume2_lost_worker_result_adjudication as r2


def controller_source() -> str:
    return '''
import json, tempfile

def main():
    with tempfile.TemporaryDirectory(prefix="x") as directory:
        _child([], root, "cold Stage-X Resume1 worker", env)
        worker = json.loads(worker_path.read_text(encoding="utf-8"))
        validate_worker_payload(worker)
        probe_pid = int(probe["probe"]["process_id"])
'''


def probe_source() -> str:
    return '''
def main():
    print({"process_id": result["process_id"]})
'''


def blocked_payload() -> dict:
    return {
        "phase": "Phase3.14b-r2.5.8 Stage X Resume1",
        "execution_verdict": "BLOCKED",
        "scientific_status": "BLOCKED",
        "root_cause": "phase314b_r258_stagex_resume1_execution_contract_failed",
        "required_next_path": "DESIGN_ADD_ONLY_STAGEX_RESUME1_EXECUTION_RECOVERY_WITHOUT_HOLDOUT_OR_PROBE_REACCESS_IF_SCIENCE_DID_NOT_START",
        "primary_failure_locus": "execution_contract",
        "selected_configuration": None,
        "train_only_recommendation": None,
        "error_type": "KeyError",
        "error_message": "'probe'",
        "selection_holdout_evaluation_count_added": 0,
        "cumulative_selection_holdout_evaluation_count": 1,
        "rerun_authorized": False,
        "complete_nested_oof_population_claimed": False,
    }


def adjudication_payload() -> dict:
    value = {
        "schema": r2.SCHEMA,
        "execution_verdict": "PASS",
        "scientific_status": "BLOCKED",
        "selected_configuration": None,
        "train_only_recommendation": None,
        "rerun_authorized": False,
        "replay_governance": {
            "stagex_resume1_science_replay_authorized": False,
            "environment_probe_reaccess_authorized": False,
            "science_worker_reexecution_authorized": False,
            "conditional_recovery_clause_satisfied": False,
            "probe_pid_schema_correction_changes_science_authorization": False,
        },
        "execution_loss_adjudication": {
            "environment_probe_started": True,
            "science_worker_started": True,
            "science_worker_completed": True,
            "worker_payload_read": True,
            "worker_payload_validated": True,
        },
    }
    value.update({key: False for key in r2.FALSE_BOUNDARIES})
    value["adjudication_sha256"] = r2.sha256_bytes(r2.stable_json_bytes(value))
    return value


def test_phase_and_schema():
    assert r2.PHASE.endswith("Stage X Resume2")
    assert r2.SCHEMA.endswith("_v1")


def test_starting_commits_are_frozen():
    assert r2.BASE_RESUME1_IMPLEMENTATION_COMMIT.startswith("703aa399")
    assert r2.BASE_RESUME1_BLOCKED_EVIDENCE_COMMIT.startswith("be66026a")


def test_remote_and_submodule_are_frozen():
    assert r2.EXPECTED_REMOTE.startswith("6758ea7")
    assert r2.EXPECTED_SUBMODULE.startswith("633a887")


def test_add_only_three_paths():
    assert len(r2.IMPLEMENTATION_PATHS) == 3
    assert all(status == "A" for status, _ in r2.IMPLEMENTATION_PATHS)


def test_resume1_source_population_is_bound():
    assert set(r2.RESUME1_SOURCE_SHA256) == {path for _, path in r2.RESUME1_IMPLEMENTATION_PATHS}


def test_blocked_report_hash_is_frozen():
    assert r2.EXPECTED_RESUME1_BLOCKED_REPORT_SHA256 == "d6b7e02e71fb3041c228559a9bb089333e540db8cc49e0cffb0dc3b3c5c4f877"


def test_probe_worker_hash_is_frozen():
    assert r2.EXPECTED_PROBE_WORKER_SHA256.startswith("096ffbed")


def test_stable_json_is_sorted():
    assert r2.stable_json_bytes({"b": 2, "a": 1}) == b'{\n  "a": 1,\n  "b": 2\n}'


def test_sha256_bytes():
    assert r2.sha256_bytes(b"abc") == hashlib.sha256(b"abc").hexdigest()


def test_write_once(tmp_path: Path):
    path = tmp_path / "x.json"
    r2.write_once(path, b"{}\n")
    assert path.read_bytes() == b"{}\n"
    with pytest.raises(r2.StageXResume2Error):
        r2.write_once(path, b"again")


def test_extract_probe_process_id():
    assert r2.extract_probe_process_id({"process_id": 123}) == 123


def test_extract_probe_process_id_rejects_nested_wrapper():
    with pytest.raises(r2.StageXResume2Error, match="nested"):
        r2.extract_probe_process_id({"probe": {"process_id": 1}, "process_id": 2})


def test_extract_probe_process_id_rejects_missing():
    with pytest.raises(r2.StageXResume2Error):
        r2.extract_probe_process_id({})


def test_extract_probe_process_id_rejects_bool():
    with pytest.raises(r2.StageXResume2Error):
        r2.extract_probe_process_id({"process_id": True})


def test_controller_audit_detects_order():
    result = r2.controller_control_flow_audit(controller_source())
    assert result["worker_completed_before_failure"] is True
    assert result["worker_payload_validated_before_failure"] is True


def test_controller_audit_detects_invalid_path():
    result = r2.controller_control_flow_audit(controller_source())
    assert result["invalid_probe_pid_path"] == ["probe", "probe", "process_id"]


def test_controller_audit_records_correct_path():
    result = r2.controller_control_flow_audit(controller_source())
    assert result["correct_probe_pid_path"] == ["probe", "process_id"]


def test_controller_audit_rejects_corrected_source():
    source = controller_source().replace('probe["probe"]["process_id"]', 'probe["process_id"]')
    with pytest.raises(r2.StageXResume2Error):
        r2.controller_control_flow_audit(source)


def test_controller_audit_rejects_missing_worker_validation():
    source = controller_source().replace("        validate_worker_payload(worker)\n", "")
    with pytest.raises(r2.StageXResume2Error):
        r2.controller_control_flow_audit(source)


def test_probe_worker_schema_audit():
    result = r2.probe_worker_schema_audit(probe_source())
    assert result["probe_pid_schema"] == "top_level_process_id"


def test_probe_worker_schema_rejects_missing_output():
    with pytest.raises(r2.StageXResume2Error):
        r2.probe_worker_schema_audit("def main():\n    return None\n")


def test_validate_blocked_report():
    result = r2.validate_blocked_report(blocked_payload())
    assert result["conditional_recovery_clause_satisfied"] is False


def test_validate_blocked_report_accepts_plain_probe_message():
    value = blocked_payload()
    value["error_message"] = "probe"
    r2.validate_blocked_report(value)


def test_validate_blocked_report_rejects_rerun_true():
    value = blocked_payload()
    value["rerun_authorized"] = True
    with pytest.raises(r2.StageXResume2Error):
        r2.validate_blocked_report(value)


def test_validate_blocked_report_rejects_wrong_next_path():
    value = blocked_payload()
    value["required_next_path"] = "RETRY"
    with pytest.raises(r2.StageXResume2Error):
        r2.validate_blocked_report(value)


def test_validate_blocked_report_rejects_selected_configuration():
    value = blocked_payload()
    value["selected_configuration"] = {"x": 1}
    with pytest.raises(r2.StageXResume2Error):
        r2.validate_blocked_report(value)


def test_validate_adjudication_accepts_bound_payload():
    r2.validate_adjudication(adjudication_payload())


def test_validate_adjudication_rejects_ready():
    value = adjudication_payload()
    value["scientific_status"] = "READY"
    value["adjudication_sha256"] = r2.sha256_bytes(r2.stable_json_bytes({k: v for k, v in value.items() if k != "adjudication_sha256"}))
    with pytest.raises(r2.StageXResume2Error):
        r2.validate_adjudication(value)


def test_validate_adjudication_rejects_replay():
    value = adjudication_payload()
    value["replay_governance"]["science_worker_reexecution_authorized"] = True
    value["adjudication_sha256"] = r2.sha256_bytes(r2.stable_json_bytes({k: v for k, v in value.items() if k != "adjudication_sha256"}))
    with pytest.raises(r2.StageXResume2Error):
        r2.validate_adjudication(value)


def test_validate_adjudication_rejects_missing_completion():
    value = adjudication_payload()
    value["execution_loss_adjudication"]["science_worker_completed"] = False
    value["adjudication_sha256"] = r2.sha256_bytes(r2.stable_json_bytes({k: v for k, v in value.items() if k != "adjudication_sha256"}))
    with pytest.raises(r2.StageXResume2Error):
        r2.validate_adjudication(value)


def test_validate_adjudication_rejects_forbidden_boundary():
    value = adjudication_payload()
    value["frozen_probe_accessed"] = True
    value["adjudication_sha256"] = r2.sha256_bytes(r2.stable_json_bytes({k: v for k, v in value.items() if k != "adjudication_sha256"}))
    with pytest.raises(r2.StageXResume2Error):
        r2.validate_adjudication(value)


def test_blocked_report_preserves_no_rerun():
    value = r2.blocked_report(None, RuntimeError("x"))
    assert value["rerun_authorized"] is False
    assert value["environment_probe_run"] is False
    assert value["science_worker_run"] is False
