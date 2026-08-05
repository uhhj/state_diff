import inspect

from scripts.experiment2.phase0 import run_hidden_routing_gate_phase0l as runner


def test_runner_reuses_pair_classifier_and_fixed_tension_execution():
    source = inspect.getsource(runner)
    assert "run_exact_counterfactual_pair" in source
    assert "phase0i_classifier_sample" in source
    assert "generate_hidden_routing_gate_action_script" in source
    assert "pick_precise_tension_extension" in inspect.getsource(
        runner.generate_hidden_routing_gate_action_script
    )


def test_runner_has_only_routing_gate_verdicts_and_no_search_or_training():
    source = inspect.getsource(runner)
    assert source.count("HIDDEN_ROUTING_GATE_SMOKE_ELIGIBLE") == 1
    assert source.count("HIDDEN_ROUTING_GATE_SMOKE_BLOCKED") == 1
    assert '"legacy_mean_progress_is_gate": (' in source
    for field in (
        "training_performed", "geometry_search_performed",
        "action_search_performed", "outcome_search_performed",
    ):
        assert f'"{field}": False' in source
    assert "min_median_routing_success_gap" not in source
    assert "routing layouts " in source and "do not match" in source


def test_runner_enforces_fixed_seed_topology_action_and_outcome():
    source = inspect.getsource(runner.main)
    assert "single_fixed_topology_no_grid_search" in source
    assert "single_fixed_probe_and_" in source
    assert "pulled_endpoint_target_success_gap" in source
    for seed in (71001, 71002, 71003):
        assert str(seed) in source
