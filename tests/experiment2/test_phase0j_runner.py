import inspect

from scripts.experiment2.phase0 import run_hidden_latch_phase0j as runner


def test_phase0j_reuses_frozen_pair_runner_and_official_summary():
    source = inspect.getsource(runner)
    assert "from scripts.experiment2.phase0.run_hidden_latch_phase0i import run_one_pair" in source
    assert "summarize_hidden_hook(" in source
    assert 'row["mean_cable_progress_gap"] =' not in source
    assert "Diagnostic only: never rewrite" in source


def test_phase0j_outcome_is_diagnostic_and_scope_is_frozen():
    source = inspect.getsource(runner)
    assert '"diagnostic_only": True' in source
    assert '"changes_official_progress_gate": False' in source
    assert '"outcome_diagnostics_are_gates": False' in source
    assert '"training_performed": False' in source
    assert '"geometry_search_performed": False' in source
    assert '"action_search_performed": False' in source
    assert source.count("HIDDEN_WIDE_STOP_SMOKE_ELIGIBLE") == 1
    assert source.count("HIDDEN_WIDE_STOP_SMOKE_BLOCKED") == 1

