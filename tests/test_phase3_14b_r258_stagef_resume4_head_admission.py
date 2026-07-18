from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from ccda_phase3 import phase314b_r258_stagef_resume4_head_admission as admission
from ccda_phase3.phase314b_r258_stagef_resume4_preflight_evidence import (
    Resume4BlockedReportError,
    write_resume4_preflight_blocked_report,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
MATERIALIZER_PATH = (
    PACKAGE_ROOT / "scripts/phase3_14b_r258_stagef_resume4_materialize.py"
)
PREFLIGHT_PATH = PACKAGE_ROOT / "scripts/phase3_14b_r258_stagef_resume4_preflight.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _materializer():
    return _load_module("resume4_materializer_test", MATERIALIZER_PATH)


def _preflight():
    return _load_module("resume4_preflight_test", PREFLIGHT_PATH)


def _exact_changes():
    return tuple(("A", path) for path in sorted(admission.ALLOWED_IMPLEMENTATION_PATHS))


def _valid_manifest(repo: Path):
    hashes = {}
    for relpath in sorted(
        admission.ALLOWED_IMPLEMENTATION_PATHS - {admission.MATERIALIZATION_MANIFEST}
    ):
        path = repo / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture:{relpath}\n", encoding="utf-8")
        hashes[relpath] = admission._sha256_path(path)
    return {
        "phase": "Phase3.14b-r2.5.8 Stage F Resume4",
        "operation": "add_only_versioned_head_admission_recovery",
        "input_head": admission.EXPECTED_IMPLEMENTATION_PARENT,
        "scientific_logic_changed": False,
        "frozen_temporal_files": 58,
        "frozen_temporal_passed": 1503,
        "generated_sha256": hashes,
    }


def test_allowed_population_contains_only_resume4_additions() -> None:
    assert admission.MATERIALIZATION_MANIFEST in admission.ALLOWED_IMPLEMENTATION_PATHS
    assert all("resume4" in path for path in admission.ALLOWED_IMPLEMENTATION_PATHS)
    assert not any("resume3" in path for path in admission.ALLOWED_IMPLEMENTATION_PATHS)


def test_exact_add_only_implementation_population_is_accepted() -> None:
    observed = admission._validate_implementation_changes(_exact_changes())
    assert observed == tuple(sorted(admission.ALLOWED_IMPLEMENTATION_PATHS))


def test_non_addition_is_rejected() -> None:
    changes = list(_exact_changes())
    changes[0] = ("M", changes[0][1])
    with pytest.raises(admission.Resume4AdmissionError, match="not add-only"):
        admission._validate_implementation_changes(tuple(changes))


def test_missing_implementation_path_is_rejected() -> None:
    changes = _exact_changes()[:-1]
    with pytest.raises(admission.Resume4AdmissionError, match="path population changed"):
        admission._validate_implementation_changes(changes)


def test_nul_name_status_parser_preserves_spaces() -> None:
    raw = b"A\0scripts/a.py\0A\0tests/name with space.py\0"
    assert admission._parse_nul_name_status(raw) == (
        ("A", "scripts/a.py"),
        ("A", "tests/name with space.py"),
    )


def test_nul_name_status_parser_rejects_odd_fields() -> None:
    with pytest.raises(admission.Resume4AdmissionError, match="field count"):
        admission._parse_nul_name_status(b"A\0path\0orphan")


def test_valid_materialization_manifest_is_accepted(tmp_path: Path) -> None:
    manifest = _valid_manifest(tmp_path)
    verified = admission._validate_manifest(tmp_path, manifest)
    assert set(verified) == (
        admission.ALLOWED_IMPLEMENTATION_PATHS - {admission.MATERIALIZATION_MANIFEST}
    )


def test_manifest_cannot_claim_scientific_change(tmp_path: Path) -> None:
    manifest = dict(_valid_manifest(tmp_path))
    manifest["scientific_logic_changed"] = True
    with pytest.raises(admission.Resume4AdmissionError, match="scientific logic"):
        admission._validate_manifest(tmp_path, manifest)


def test_generated_wrapper_rejects_legacy_validator(tmp_path: Path) -> None:
    for relpath in admission.GENERATED_WRAPPER_PATHS:
        path = tmp_path / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("validate_resume4_initial_state()\n", encoding="utf-8")
    bad = tmp_path / admission.GENERATED_WRAPPER_PATHS[0]
    bad.write_text("validate_initial_worktree()\n", encoding="utf-8")
    with pytest.raises(admission.Resume4AdmissionError, match="legacy"):
        admission._validate_generated_wrappers(tmp_path)


def test_generated_wrapper_requires_new_validator_call(tmp_path: Path) -> None:
    for relpath in admission.GENERATED_WRAPPER_PATHS:
        path = tmp_path / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("VALUE = 1\n", encoding="utf-8")
    with pytest.raises(admission.Resume4AdmissionError, match="no generated"):
        admission._validate_generated_wrappers(tmp_path)


def test_materializer_replaces_bare_legacy_validator() -> None:
    materializer = _materializer()
    source = '''from ccda_phase3.legacy import (\n    keep_me,\n    validate_initial_worktree,\n)\n\ndef main():\n    validate_initial_worktree()\n    return keep_me\n'''
    transformed, count = materializer.patch_legacy_validator(source)
    assert count == 1
    assert "validate_initial_worktree" not in transformed
    assert materializer.NEW_IMPORT in transformed
    assert "validate_resume4_initial_state()" in transformed
    assert "keep_me" in transformed


def test_materializer_preserves_legacy_import_execution_position() -> None:
    materializer = _materializer()
    source = "import sys\nsys.path.insert(0, \"/tmp/repo\")\nfrom legacy import validate_initial_worktree\nvalidate_initial_worktree()\n"
    transformed, count = materializer.patch_legacy_validator(source)
    assert count == 1
    assert transformed.index('sys.path.insert') < transformed.index(materializer.NEW_IMPORT)
    assert transformed.index(materializer.NEW_IMPORT) < transformed.index(
        'validate_resume4_initial_state()'
    )


def test_materializer_replaces_qualified_legacy_validator() -> None:
    materializer = _materializer()
    source = '''from __future__ import annotations\nimport legacy_module\n\ndef main():\n    legacy_module.validate_initial_worktree()\n'''
    transformed, count = materializer.patch_legacy_validator(source)
    assert count == 1
    assert "legacy_module.validate_initial_worktree" not in transformed
    assert "validate_resume4_initial_state()" in transformed


def test_materializer_rejects_orphan_legacy_import() -> None:
    materializer = _materializer()
    source = "from legacy import validate_initial_worktree\nVALUE = 1\n"
    with pytest.raises(materializer.MaterializationError, match="without a corresponding call"):
        materializer.patch_legacy_validator(source)


def test_materializer_preserves_scientific_module_identifiers() -> None:
    materializer = _materializer()
    source = '''from legacy import validate_initial_worktree\nSCIENCE = "phase314b_r258_stagef_constraint_aware_surrogate"\nINTEGRATOR = "phase314b_r258_stagee_constrained_integrator"\n\ndef main():\n    validate_initial_worktree()\n'''
    transformed, _ = materializer.transform_source(
        "scripts/phase3_14b_r258_stagef_resume3_worker.py",
        source,
    )
    assert "phase314b_r258_stagef_constraint_aware_surrogate" in transformed
    assert "phase314b_r258_stagee_constrained_integrator" in transformed


def test_shell_transform_changes_only_wrapper_report_and_phase_names() -> None:
    materializer = _materializer()
    source = '''#!/usr/bin/env bash
python scripts/phase3_14b_r258_stagef_resume3_worker.py
REPORT=phase3_14b_r258_stagef_resume3_summary.json
TITLE="Phase3.14b-r2.5.8 Stage F Resume3"
MODULE=scripts.phase3_14b_r258_stagef_resume3_worker
SEMANTIC=phase314b_r258_stagef_resume3_test_count_namespace
'''
    transformed, calls = materializer.transform_source(
        "scripts/phase3_14b_r258_stagef_resume3_run.sh",
        source,
    )
    assert calls == 0
    assert "phase3_14b_r258_stagef_resume4_worker.py" in transformed
    assert "phase3_14b_r258_stagef_resume4_summary.json" in transformed
    assert "Phase3.14b-r2.5.8 Stage F Resume4" in transformed
    assert "scripts.phase3_14b_r258_stagef_resume4_worker" in transformed
    assert "phase314b_r258_stagef_resume3_test_count_namespace" in transformed


def test_shell_python_heredoc_legacy_validator_is_patched() -> None:
    materializer = _materializer()
    source = '''#!/usr/bin/env bash
set -euo pipefail
"${PYTHON:-python}" - <<'PY'
from ccda_phase3.legacy_gate import (
    keep_me,
    validate_initial_worktree,
)

validate_initial_worktree()
print(keep_me)
PY
echo done
'''
    transformed, calls = materializer.transform_source(
        "scripts/phase3_14b_r258_stagef_resume3_run.sh",
        source,
    )
    assert calls == 1
    assert "validate_initial_worktree" not in transformed
    assert materializer.NEW_IMPORT in transformed
    assert "validate_resume4_initial_state()" in transformed
    assert "print(keep_me)" in transformed
    assert transformed.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
    assert transformed.endswith("PY\necho done\n")


def test_transformed_shell_python_heredoc_passes_bash_syntax(
    tmp_path: Path,
) -> None:
    materializer = _materializer()
    source = """#!/usr/bin/env bash
set -euo pipefail
python - <<'PY'
from legacy import validate_initial_worktree
validate_initial_worktree()
PY
echo done
"""
    transformed, calls = materializer.transform_source(
        "scripts/phase3_14b_r258_stagef_resume3_run.sh",
        source,
    )
    assert calls == 1
    path = tmp_path / "run.sh"
    path.write_text(transformed, encoding="utf-8")
    result = subprocess.run(
        ["bash", "-n", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_shell_tab_stripped_python_heredoc_is_patched() -> None:
    materializer = _materializer()
    source = """#!/usr/bin/env bash
python - <<-'PY'
\tfrom legacy import validate_initial_worktree
\tvalidate_initial_worktree()
\tPY
"""
    transformed, calls = materializer.transform_source(
        "scripts/phase3_14b_r258_stagef_resume3_run.sh",
        source,
    )
    assert calls == 1
    assert "validate_initial_worktree" not in transformed
    assert "validate_resume4_initial_state()" in transformed
    assert "\tPY\n" in transformed


def test_shell_multiple_heredocs_patch_only_legacy_python_body() -> None:
    materializer = _materializer()
    source = '''#!/usr/bin/env bash
cat <<'TEXT'
scientific payload remains byte-stable
TEXT
python - <<'PY'
from legacy import validate_initial_worktree
validate_initial_worktree()
PY
'''
    transformed, calls = materializer.transform_source(
        "scripts/phase3_14b_r258_stagef_resume3_run.sh",
        source,
    )
    assert calls == 1
    assert "scientific payload remains byte-stable" in transformed
    assert "validate_initial_worktree" not in transformed
    assert transformed.count("validate_resume4_initial_state") >= 2


def test_shell_legacy_token_outside_python_heredoc_is_rejected() -> None:
    materializer = _materializer()
    source = '''#!/usr/bin/env bash
validate_initial_worktree
'''
    with pytest.raises(
        materializer.MaterializationError,
        match="outside a safely patchable Python heredoc",
    ):
        materializer.transform_source(
            "scripts/phase3_14b_r258_stagef_resume3_run.sh",
            source,
        )


def test_shell_invalid_legacy_heredoc_is_rejected() -> None:
    materializer = _materializer()
    source = '''#!/usr/bin/env bash
python - <<'PY'
from legacy import validate_initial_worktree
this is not valid python
validate_initial_worktree()
PY
'''
    with pytest.raises(
        materializer.MaterializationError,
        match="not safely patchable Python",
    ):
        materializer.transform_source(
            "scripts/phase3_14b_r258_stagef_resume3_run.sh",
            source,
        )

def test_preflight_pytest_count_parser_uses_last_summary() -> None:
    preflight = _preflight()
    output = "1 passed in 0.01s\n================ 17 passed in 0.20s ================\n"
    assert preflight._parse_pytest_passed(output) == 17
    assert preflight._parse_pytest_passed("no tests ran") is None


def test_preflight_blocked_report_is_write_once(tmp_path: Path) -> None:
    result = write_resume4_preflight_blocked_report(
        failure_stage="fixture",
        root_cause="fixture_root",
        detail="fixture detail",
        repo=tmp_path,
    )
    assert result["scientific_status"] == "BLOCKED"
    assert result["scientific_calibration_run"] is False
    assert result["frozen_probe_accessed"] is False
    with pytest.raises(Resume4BlockedReportError, match="refusing to overwrite"):
        write_resume4_preflight_blocked_report(
            failure_stage="fixture2",
            root_cause="fixture_root2",
            detail="fixture detail2",
            repo=tmp_path,
        )
