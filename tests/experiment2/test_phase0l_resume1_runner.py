import inspect
import json

from ravens.tasks.ccda_hidden_routing_gate_geometry import (
    HiddenRoutingGateGeometryError,
)
from scripts.experiment2.phase0 import run_hidden_routing_gate_phase0l as base
from scripts.experiment2.phase0 import run_hidden_routing_gate_phase0l_resume1 as resume


def test_resume1_requires_provenance_report():
    source = inspect.getsource(resume)
    assert "required_provenance_report" in source
    assert "report is missing" in source
    assert 'provenance.get("passed")' in source


def test_resume1_verifies_config_hash():
    source = inspect.getsource(resume)
    assert "canonical_json_sha256(config)" in source
    assert "different Resume1 config" in source


def test_original_runner_defaults_are_unchanged():
    source = inspect.getsource(base)
    assert 'config.get("stage_name", "Experiment2 Phase 0L")' in source
    assert '"HIDDEN_ROUTING_GATE_SMOKE_ELIGIBLE"' in source
    assert '"HIDDEN_ROUTING_GATE_SMOKE_BLOCKED"' in source


def test_resume1_verdict_names_are_configured():
    source = inspect.getsource(base)
    assert "eligible_verdict" in source and "blocked_verdict" in source
    assert "if (" in source


def test_geometry_failure_writes_structured_diagnostics(tmp_path):
    config_path = base.REPO_ROOT / "configs/experiment2/phase0/hidden_routing_gate_phase0l_resume1.json"
    config = json.loads(config_path.read_text())
    error = HiddenRoutingGateGeometryError("blocked", {"candidate_count": 4})
    result = base._write_geometry_failure(
        tmp_path, config, config_path, 71001, error,
        "Experiment2 Phase 0L Resume1",
        "HIDDEN_ROUTING_GATE_RESUME1_SMOKE_BLOCKED",
    )
    assert result["completed_pair_count"] == 0
    assert result["failure"]["diagnostics"]["candidate_count"] == 4
    assert (tmp_path / "geometry_failure_diagnostics.json").is_file()


def test_successful_pairs_export_geometry_audit():
    source = inspect.getsource(base)
    assert "_geometry_audit_from_metadata" in source
    assert 'output_root / "geometry_audit.json"' in source
    assert "free/hidden geometry provenance does not match" in source


def test_geometry_audit_is_not_formal_feature():
    source = inspect.getsource(base)
    assert '"official_model_feature": False' in source
    classifier_source = inspect.getsource(base.phase0i_classifier_sample)
    assert "geometry_audit" not in classifier_source
