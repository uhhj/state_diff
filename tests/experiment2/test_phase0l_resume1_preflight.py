import inspect

from scripts.experiment2.phase0 import capture_phase0l_geometry_provenance as preflight


def source():
    return inspect.getsource(preflight)


def test_preflight_uses_only_seed_71001():
    text = source()
    assert "default=71001" in text
    assert "args.seed != 71001" in text


def test_preflight_compares_only_0340_and_0350():
    text = source()
    assert "original_width != 0.340" in text
    assert "fixed_width != 0.350" in text
    assert "for width" not in text


def test_preflight_requires_original_coverage_only_failure():
    text = source()
    assert '"barrier_coverage": 4' in text
    assert "accepted_candidate_count" in text


def test_preflight_requires_fixed_legal_candidate():
    text = source()
    assert "fixed_has_legal_candidate" in text
    assert ">= 1" in text


def test_preflight_does_not_write_scientific_metrics():
    text = source()
    assert '"scientific_pair_completed": False' in text
    assert '"scientific_metrics_computed": False' in text
    assert '"geometry_search_performed": False' in text


def test_preflight_uses_temporary_observation_directory():
    text = source()
    assert "TemporaryDirectory" in text
    assert "Path(temporary)" in text
