import inspect

from scripts.experiment2.phase0 import run_hidden_routing_gate_phase0l_resume2 as resume2
from scripts.experiment2.phase0 import run_hidden_routing_gate_phase0l as base


def test_resume2_runner_requires_preflight():
    source = inspect.getsource(resume2)
    assert "required_preflight_report" in source
    assert "report is missing" in source
    assert 'preflight.get("passed")' in source


def test_resume2_runner_checks_config_hash():
    source = inspect.getsource(resume2)
    assert "canonical_json_sha256" in source
    assert "another config" in source


def test_resume2_uses_existing_phase0l_runner():
    source = inspect.getsource(resume2)
    assert "phase0l_main" in source
    assert "run_exact_counterfactual_pair" not in source


def test_resume2_bootstraps_submodule_before_runner_import():
    source = inspect.getsource(resume2)
    assert "SUBMODULE_ROOT" in source
    assert source.index("SUBMODULE_ROOT") < source.index(
        "from scripts.experiment2.phase0.run_hidden_routing_gate_phase0l import"
    )


def test_resume2_verdict_names_are_configured():
    source = inspect.getsource(base)
    assert "eligible_verdict" in source and "blocked_verdict" in source


def test_candidate_report_is_config_driven():
    source = inspect.getsource(base)
    assert "candidate_report_filename" in source
    assert "must be a plain filename" in source


def test_old_candidate_report_default_is_unchanged():
    assert "candidate_hidden_routing_gate_v1.json" in inspect.getsource(base)


def test_geometry_audit_remains_privileged():
    source = inspect.getsource(base)
    assert '"official_model_feature": False' in source
    assert "geometry_audit" not in inspect.getsource(base.phase0i_classifier_sample)


def test_training_and_search_flags_remain_false():
    source = inspect.getsource(base)
    for field in (
        "training_performed", "geometry_search_performed",
        "action_search_performed", "outcome_search_performed",
    ):
        assert f'"{field}": False' in source
