import ast
import inspect

from scripts.experiment2.phase0 import preflight_phase0l_resume2_corridor_geometry as preflight


def source():
    return inspect.getsource(preflight)


def test_preflight_reads_resume1_evidence():
    assert "geometry_provenance_original.json" in source()
    assert 'provenance["original_audit"]' in source()


def test_preflight_verifies_source_sha():
    text = source()
    assert "source_resume1_provenance_" in text
    assert "actual_sha != expected_sha" in text


def test_preflight_does_not_import_environment_or_run_pair():
    tree = ast.parse(source())
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "pybullet" not in imported
    assert not any("environment" in name.lower() for name in imported)
    assert "run_exact_counterfactual_pair" not in source()


def test_preflight_has_no_parameter_loop():
    text = source()
    assert "barrier_candidates" not in text
    assert "for width" not in text
    assert '"geometry_search_performed": (' in text


def test_preflight_requires_workspace_only_resume1_failure():
    assert '"workspace": 4' in source()


def test_preflight_requires_legal_corridor_candidate():
    text = source()
    assert "accepted_count_ok" in text
    assert "accepted_geometry_ok" in text


def test_preflight_records_no_scientific_pair():
    text = source()
    assert '"scientific_pair_completed": (' in text
    assert '"scientific_metrics_computed": (' in text
    assert "False" in text
