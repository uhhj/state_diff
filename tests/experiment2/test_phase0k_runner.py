import inspect

from scripts.experiment2.phase0 import run_hidden_latch_phase0k as runner


def test_runner_reuses_pair_execution_and_official_summary():
    source = inspect.getsource(runner)
    assert "run_one_pair(" in source
    assert "action_generator_fn=generate_hidden_latch_tension_action_script" in source
    assert "summarize_hidden_hook(" in source
    assert "exact_counterfactual" not in source
    assert "diagnostic final progress does not match official progress" in source


def test_runner_has_only_allowed_verdicts_and_no_search_or_training():
    source = inspect.getsource(runner)
    assert source.count("HIDDEN_TENSION_EXTENSION_SMOKE_ELIGIBLE") == 1
    assert source.count("HIDDEN_TENSION_EXTENSION_SMOKE_BLOCKED") == 1
    assert '"tension_diagnostics_are_gates": False' in source
    for field in (
        "training_performed", "geometry_search_performed",
        "action_search_performed", "extension_search_performed",
    ):
        assert f'"{field}": False' in source
    assert "tension_execution_valid" in source
    assert "all_cartesian_stages_successful" in source

