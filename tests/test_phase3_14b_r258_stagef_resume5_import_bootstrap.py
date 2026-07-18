from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ccda_phase3 import phase314b_r258_stagef_resume5_head_admission as admission
from ccda_phase3.phase314b_r258_stagef_resume5_preflight_evidence import (
    Resume5BlockedReportError,
    write_resume5_preflight_blocked_report,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
MATERIALIZER_PATH = (
    PACKAGE_ROOT / "scripts/phase3_14b_r258_stagef_resume5_materialize.py"
)
PREFLIGHT_PATH = PACKAGE_ROOT / "scripts/phase3_14b_r258_stagef_resume5_preflight.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _materializer():
    return _load_module("resume5_materializer_test", MATERIALIZER_PATH)


def _preflight():
    return _load_module("resume5_preflight_test", PREFLIGHT_PATH)


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
        "phase": "Phase3.14b-r2.5.8 Stage F Resume5",
        "operation": "add_only_script_import_bootstrap_recovery",
        "input_head": admission.EXPECTED_IMPLEMENTATION_PARENT,
        "scientific_logic_changed": False,
        "frozen_temporal_files": 58,
        "frozen_temporal_passed": 1503,
        "python_entrypoints_bootstrapped": len(admission.PYTHON_ENTRYPOINT_PATHS),
        "shell_wrappers_bootstrapped": 1,
        "validator_calls_replaced": 1,
        "generated_sha256": hashes,
    }


def _write_valid_generated_wrappers(repo: Path) -> None:
    for relpath in admission.GENERATED_WRAPPER_PATHS:
        path = repo / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        if relpath.endswith(".py"):
            path.write_text(
                "_RESUME5_REPO_ROOT = '/data/state_diff2'\n"
                "from ccda_phase3.phase314b_r258_stagef_resume5_head_admission "
                "import validate_resume5_initial_state\n"
                "validate_resume5_initial_state()\n",
                encoding="utf-8",
            )
        else:
            path.write_text(
                "#!/usr/bin/env bash\n"
                "export CCDA_REPO_ROOT=/data/state_diff2\n"
                "export PYTHONPATH=/data/state_diff2\n"
                "python - <<'PY'\n"
                "_RESUME5_REPO_ROOT = '/data/state_diff2'\n"
                "from ccda_phase3.phase314b_r258_stagef_resume5_head_admission "
                "import validate_resume5_initial_state\n"
                "validate_resume5_initial_state()\n"
                "PY\n",
                encoding="utf-8",
            )
    preflight = repo / "scripts/phase3_14b_r258_stagef_resume5_preflight.py"
    preflight.write_text(
        "_RESUME5_REPO_ROOT = '/data/state_diff2'\n"
        "from ccda_phase3.phase314b_r258_stagef_resume5_head_admission import "
        "validate_resume5_initial_state\n",
        encoding="utf-8",
    )


# 1
def test_allowed_population_contains_only_resume5_additions() -> None:
    assert admission.MATERIALIZATION_MANIFEST in admission.ALLOWED_IMPLEMENTATION_PATHS
    assert all("resume5" in path for path in admission.ALLOWED_IMPLEMENTATION_PATHS)
    assert not any("resume4" in path for path in admission.ALLOWED_IMPLEMENTATION_PATHS)


# 2
def test_exact_add_only_implementation_population_is_accepted() -> None:
    observed = admission._validate_implementation_changes(_exact_changes())
    assert observed == tuple(sorted(admission.ALLOWED_IMPLEMENTATION_PATHS))


# 3
def test_non_addition_is_rejected() -> None:
    changes = list(_exact_changes())
    changes[0] = ("M", changes[0][1])
    with pytest.raises(admission.Resume5AdmissionError, match="not add-only"):
        admission._validate_implementation_changes(tuple(changes))


# 4
def test_missing_implementation_path_is_rejected() -> None:
    with pytest.raises(admission.Resume5AdmissionError, match="path population changed"):
        admission._validate_implementation_changes(_exact_changes()[:-1])


# 5
def test_nul_name_status_parser_preserves_spaces() -> None:
    raw = b"A\0scripts/a.py\0A\0tests/name with space.py\0"
    assert admission._parse_nul_name_status(raw) == (
        ("A", "scripts/a.py"),
        ("A", "tests/name with space.py"),
    )


# 6
def test_nul_name_status_parser_rejects_odd_fields() -> None:
    with pytest.raises(admission.Resume5AdmissionError, match="field count"):
        admission._parse_nul_name_status(b"A\0path\0orphan")


# 7
def test_valid_materialization_manifest_is_accepted(tmp_path: Path) -> None:
    manifest = _valid_manifest(tmp_path)
    verified = admission._validate_manifest(tmp_path, manifest)
    assert set(verified) == (
        admission.ALLOWED_IMPLEMENTATION_PATHS - {admission.MATERIALIZATION_MANIFEST}
    )


# 8
def test_manifest_cannot_claim_scientific_change(tmp_path: Path) -> None:
    manifest = dict(_valid_manifest(tmp_path))
    manifest["scientific_logic_changed"] = True
    with pytest.raises(admission.Resume5AdmissionError, match="scientific logic"):
        admission._validate_manifest(tmp_path, manifest)


# 9
def test_manifest_temporal_file_count_is_frozen(tmp_path: Path) -> None:
    manifest = dict(_valid_manifest(tmp_path))
    manifest["frozen_temporal_files"] = 59
    with pytest.raises(admission.Resume5AdmissionError, match="file count"):
        admission._validate_manifest(tmp_path, manifest)


# 10
def test_manifest_temporal_pass_count_is_frozen(tmp_path: Path) -> None:
    manifest = dict(_valid_manifest(tmp_path))
    manifest["frozen_temporal_passed"] = 1504
    with pytest.raises(admission.Resume5AdmissionError, match="passed count"):
        admission._validate_manifest(tmp_path, manifest)


# 11
def test_manifest_python_bootstrap_count_is_frozen(tmp_path: Path) -> None:
    manifest = dict(_valid_manifest(tmp_path))
    manifest["python_entrypoints_bootstrapped"] = 5
    with pytest.raises(admission.Resume5AdmissionError, match="Python entrypoint"):
        admission._validate_manifest(tmp_path, manifest)


# 12
def test_manifest_shell_bootstrap_count_is_frozen(tmp_path: Path) -> None:
    manifest = dict(_valid_manifest(tmp_path))
    manifest["shell_wrappers_bootstrapped"] = 0
    with pytest.raises(admission.Resume5AdmissionError, match="shell wrapper"):
        admission._validate_manifest(tmp_path, manifest)


# 13
def test_manifest_requires_validator_replacement(tmp_path: Path) -> None:
    manifest = dict(_valid_manifest(tmp_path))
    manifest["validator_calls_replaced"] = 0
    with pytest.raises(admission.Resume5AdmissionError, match="validator call"):
        admission._validate_manifest(tmp_path, manifest)


# 14
def test_generated_wrapper_rejects_resume4_validator(tmp_path: Path) -> None:
    _write_valid_generated_wrappers(tmp_path)
    bad = tmp_path / admission.GENERATED_WRAPPER_PATHS[0]
    bad.write_text("validate_resume4_initial_state()\n", encoding="utf-8")
    with pytest.raises(admission.Resume5AdmissionError, match="legacy Resume4"):
        admission._validate_generated_wrappers(tmp_path)


# 15
def test_generated_wrapper_requires_resume5_validator(tmp_path: Path) -> None:
    _write_valid_generated_wrappers(tmp_path)
    for relpath in admission.GENERATED_WRAPPER_PATHS:
        path = tmp_path / relpath
        text = path.read_text(encoding="utf-8")
        path.write_text(
            text.replace("validate_resume5_initial_state", "other_call"),
            encoding="utf-8",
        )
    with pytest.raises(admission.Resume5AdmissionError, match="no generated"):
        admission._validate_generated_wrappers(tmp_path)


# 16
def test_generated_python_wrapper_requires_bootstrap(tmp_path: Path) -> None:
    _write_valid_generated_wrappers(tmp_path)
    bad = tmp_path / admission.GENERATED_WRAPPER_PATHS[0]
    bad.write_text(
        "from ccda_phase3.phase314b_r258_stagef_resume5_head_admission import "
        "validate_resume5_initial_state\nvalidate_resume5_initial_state()\n",
        encoding="utf-8",
    )
    with pytest.raises(admission.Resume5AdmissionError, match="lacks Resume5 repo bootstrap"):
        admission._validate_generated_wrappers(tmp_path)


# 17
def test_generated_shell_wrapper_requires_environment_bootstrap(tmp_path: Path) -> None:
    _write_valid_generated_wrappers(tmp_path)
    bad = tmp_path / admission.GENERATED_WRAPPER_PATHS[-1]
    bad.write_text("#!/usr/bin/env bash\necho no-bootstrap\n", encoding="utf-8")
    with pytest.raises(admission.Resume5AdmissionError, match="shell wrapper lacks"):
        admission._validate_generated_wrappers(tmp_path)


# 18
def test_preflight_bootstrap_must_precede_ccda_import(tmp_path: Path) -> None:
    _write_valid_generated_wrappers(tmp_path)
    preflight = tmp_path / "scripts/phase3_14b_r258_stagef_resume5_preflight.py"
    preflight.write_text(
        "from ccda_phase3 import value\n_RESUME5_REPO_ROOT = '/data/state_diff2'\n",
        encoding="utf-8",
    )
    with pytest.raises(admission.Resume5AdmissionError, match="before ccda_phase3"):
        admission._validate_generated_wrappers(tmp_path)


# 19
def test_materializer_inserts_file_bootstrap_before_ccda_import() -> None:
    materializer = _materializer()
    source = (
        "from __future__ import annotations\n"
        "from ccda_phase3.phase314b_r258_stagef_resume4_head_admission import "
        "validate_resume4_initial_state\n"
        "validate_resume4_initial_state()\n"
    )
    transformed, calls, inserted = materializer.transform_python_source("fixture.py", source)
    assert calls >= 1
    assert inserted is True
    assert transformed.index(materializer.PYTHON_BOOTSTRAP_MARKER) < transformed.index("from ccda_phase3")


# 20
def test_materializer_file_bootstrap_is_idempotent() -> None:
    materializer = _materializer()
    source = (
        "_RESUME5_REPO_ROOT = '/data/state_diff2'\n"
        "from ccda_phase3 import value\n"
    )
    transformed, inserted = materializer.ensure_python_bootstrap(source)
    assert transformed == source
    assert inserted is False


# 21
def test_materializer_replaces_resume4_validator_and_module() -> None:
    materializer = _materializer()
    source = (
        "from ccda_phase3.phase314b_r258_stagef_resume4_head_admission import "
        "validate_resume4_initial_state\n"
        "validate_resume4_initial_state()\n"
    )
    transformed, count = materializer.replace_validator(source)
    assert count >= 1
    assert "resume4_head_admission" not in transformed
    assert "validate_resume4_initial_state" not in transformed
    assert "validate_resume5_initial_state" in transformed


# 22
def test_materializer_replaces_only_stage_execution_literals() -> None:
    materializer = _materializer()
    source = (
        "scripts/phase3_14b_r258_stagef_resume4_worker.py\n"
        "phase314b_r258_stagef_resume4_head_admission\n"
        "Phase3.14b-r2.5.8 Stage F Resume4\n"
        "phase314b_r258_stagef_constraint_aware_surrogate\n"
    )
    transformed = materializer.replace_stage_literals(source)
    assert "resume5_worker.py" in transformed
    assert "resume5_head_admission" in transformed
    assert "Stage F Resume5" in transformed
    assert "phase314b_r258_stagef_constraint_aware_surrogate" in transformed


# 23
def test_shell_python_heredoc_is_replaced_and_bootstrapped() -> None:
    materializer = _materializer()
    source = """#!/usr/bin/env bash
python - <<'INNERPY'
from ccda_phase3.phase314b_r258_stagef_resume4_head_admission import validate_resume4_initial_state
validate_resume4_initial_state()
INNERPY
"""
    transformed, calls, py_bootstraps, shell_bootstraps = materializer.transform_source(
        "scripts/phase3_14b_r258_stagef_resume4_run.sh", source
    )
    assert calls >= 1
    assert py_bootstraps == 1
    assert shell_bootstraps == 1
    assert "validate_resume4_initial_state" not in transformed
    assert "_RESUME5_REPO_ROOT" in transformed


# 24
def test_shell_environment_bootstrap_exports_pythonpath() -> None:
    materializer = _materializer()
    transformed, inserted = materializer.ensure_shell_environment_bootstrap(
        "#!/usr/bin/env bash\necho ok\n"
    )
    assert inserted is True
    assert "CCDA_REPO_ROOT" in transformed
    assert "PYTHONPATH" in transformed


# 25
def test_shell_invalid_ccda_heredoc_is_rejected() -> None:
    materializer = _materializer()
    source = """#!/usr/bin/env bash
python - <<'INNERPY'
from ccda_phase3 import thing
this is not valid python
INNERPY
"""
    with pytest.raises(materializer.MaterializationError, match="not Python"):
        materializer.patch_shell_python_heredocs(source)


# 26
def test_shell_resume4_validator_outside_heredoc_is_rejected() -> None:
    materializer = _materializer()
    source = "#!/usr/bin/env bash\nvalidate_resume4_initial_state\n"
    transformed = materializer.replace_stage_literals(source)
    with pytest.raises(materializer.MaterializationError, match="outside"):
        materializer.patch_shell_python_heredocs(transformed)


# 27
def test_preflight_pytest_count_parser_uses_last_summary() -> None:
    preflight = _preflight()
    output = "1 passed in 0.01s\n================ 30 passed in 0.20s ================\n"
    assert preflight._parse_pytest_passed(output) == 30
    assert preflight._parse_pytest_passed("no tests ran") is None


# 28
def test_preflight_direct_bootstrap_probe_works_outside_repo(tmp_path: Path) -> None:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("CCDA_REPO_ROOT", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, str(PREFLIGHT_PATH), "--bootstrap-probe"],
        cwd=str(tmp_path),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "PASS"
    assert payload["repo_root_in_sys_path"] is True
    assert payload["ccda_admission_imported"] is True


# 29
def test_preflight_execution_env_prepends_repo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preflight = _preflight()
    monkeypatch.setenv("PYTHONPATH", "/existing")
    env = preflight._execution_env(tmp_path)
    assert env["PYTHONPATH"].split(os.pathsep)[0] == str(tmp_path.resolve())
    assert env["CCDA_REPO_ROOT"] == str(tmp_path.resolve())
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"


# 30
def test_preflight_blocked_report_is_write_once(tmp_path: Path) -> None:
    result = write_resume5_preflight_blocked_report(
        failure_stage="fixture",
        root_cause="fixture_root",
        detail="fixture detail",
        repo=tmp_path,
    )
    assert result["scientific_status"] == "BLOCKED"
    assert result["scientific_calibration_run"] is False
    assert result["frozen_probe_accessed"] is False
    with pytest.raises(Resume5BlockedReportError, match="refusing to overwrite"):
        write_resume5_preflight_blocked_report(
            failure_stage="fixture2",
            root_cause="fixture_root2",
            detail="fixture detail2",
            repo=tmp_path,
        )
