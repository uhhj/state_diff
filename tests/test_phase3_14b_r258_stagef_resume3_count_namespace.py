from __future__ import annotations

import ast
import importlib.util
import sys
from collections import OrderedDict
from pathlib import Path

import pytest

from ccda_phase3.phase314b_r258_stagef_resume3_test_count_namespace import (
    CURRENT_STAGE_F_RESUME2_NAME,
    EXPECTED_STAGEF_RESUME2_PASSED,
    EXPECTED_STAGED_RESUME2_PASSED,
    EXPECTED_TEMPORAL_FILES,
    EXPECTED_TEMPORAL_PASSED,
    HISTORICAL_STAGE_D_RESUME2_NAME,
    TemporalPopulation,
    validate_observed_counts,
)


GENERATOR_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts/phase3_14b_r258_stagef_resume3_materialize.py"
)


def _load_generator():
    spec = importlib.util.spec_from_file_location("resume3_materialize", GENERATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _minimal_gate() -> str:
    return '''from ccda_phase3.phase314b_r258_staged_resume2_temporal_views import (
    CURRENT_RESUME2_TEST,
    EXPECTED_RESUME2_PASSED,
)

EXPECTED_RESUME2_PASSED = 48
EXPECTED_TOTAL_PASSED = 1503
expected_counts = {
    CURRENT_RESUME2_TEST: EXPECTED_RESUME2_PASSED,
}
resume2_result = {"passed": 48}
if resume2_result["passed"] != EXPECTED_RESUME2_PASSED:
    raise RuntimeError("Stage-F Resume2 test count changed")
REPORT = "phase3_14b_r258_stagef_resume2_test_gate_summary.json"
TITLE = "Phase3.14b-r2.5.8 Stage F Resume2"
'''


def test_historical_and_current_names_are_distinct() -> None:
    assert HISTORICAL_STAGE_D_RESUME2_NAME != CURRENT_STAGE_F_RESUME2_NAME


def test_historical_stage_d_resume2_count_is_frozen_to_55() -> None:
    assert EXPECTED_STAGED_RESUME2_PASSED == 55


def test_current_stage_f_resume2_count_is_frozen_to_48() -> None:
    assert EXPECTED_STAGEF_RESUME2_PASSED == 48


def test_frozen_temporal_arithmetic_is_exact() -> None:
    population = TemporalPopulation()
    population.validate()
    assert population.total_passed() == EXPECTED_TEMPORAL_PASSED == 1503
    assert EXPECTED_TEMPORAL_FILES == 58


def test_observed_mapping_is_order_sensitive() -> None:
    population = TemporalPopulation()
    observed = OrderedDict(population.ordered_items())
    validate_observed_counts(observed)
    reversed_observed = OrderedDict(reversed(tuple(observed.items())))
    with pytest.raises(RuntimeError, match="temporal population changed"):
        validate_observed_counts(reversed_observed)


def test_generator_aliases_import_and_separates_current_constant() -> None:
    generator = _load_generator()
    transformed = generator.transform_source(
        "scripts/phase3_14b_r258_stagef_resume2_test_gate.py",
        _minimal_gate(),
    )
    assert (
        "EXPECTED_RESUME2_PASSED as EXPECTED_STAGED_RESUME2_PASSED," in transformed
    )
    assert "EXPECTED_STAGEF_RESUME2_PASSED = 48" in transformed
    assert "CURRENT_RESUME2_TEST: EXPECTED_STAGED_RESUME2_PASSED," in transformed
    assert '!= EXPECTED_STAGEF_RESUME2_PASSED' in transformed
    assert "phase3_14b_r258_stagef_resume3_test_gate_summary.json" in transformed
    generator.validate_test_gate_ast(transformed)


def test_generator_does_not_rewrite_porcelain_semantic_reference() -> None:
    generator = _load_generator()
    source = _minimal_gate() + (
        "\nPORCELAIN = "
        '"ccda_phase3.phase314b_r258_stagef_resume2_porcelain_recovery"\n'
    )
    transformed = generator.transform_source(
        "scripts/phase3_14b_r258_stagef_resume2_test_gate.py",
        source,
    )
    assert "phase314b_r258_stagef_resume2_porcelain_recovery" in transformed
    assert "phase314b_r258_stagef_resume3_porcelain_recovery" not in transformed


def test_generator_rejects_duplicate_current_assignment() -> None:
    generator = _load_generator()
    source = _minimal_gate() + "\nEXPECTED_RESUME2_PASSED = 48\n"
    with pytest.raises(generator.MaterializationError, match="exactly one occurrence"):
        generator.transform_source(
            "scripts/phase3_14b_r258_stagef_resume2_test_gate.py",
            source,
        )


def test_generated_gate_has_no_assignment_to_imported_historical_name() -> None:
    generator = _load_generator()
    transformed = generator.transform_source(
        "scripts/phase3_14b_r258_stagef_resume2_test_gate.py",
        _minimal_gate(),
    )
    tree = ast.parse(transformed)
    assigned = {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert "EXPECTED_RESUME2_PASSED" not in assigned


def test_stage_e_and_stage_f_science_names_are_not_transformed() -> None:
    generator = _load_generator()
    source = _minimal_gate() + (
        '\nSCIENCE = "phase314b_r258_stagef_constraint_aware_surrogate"\n'
        'INTEGRATOR = "phase314b_r258_stagee_constrained_integrator"\n'
    )
    transformed = generator.transform_source(
        "scripts/phase3_14b_r258_stagef_resume2_test_gate.py",
        source,
    )
    assert "phase314b_r258_stagef_constraint_aware_surrogate" in transformed
    assert "phase314b_r258_stagee_constrained_integrator" in transformed
