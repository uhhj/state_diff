from __future__ import annotations

import json
from pathlib import Path

import pytest

from ccda_phase3.phase314b_r253_resume3_finalizer import (
    BASE_RESUME2_BLOCKED_COMMIT,
    EXPECTED_NUMPY_VERSION,
    EXPECTED_PYTHON_EXECUTABLE,
    EXPECTED_PYTHON_VERSION,
    EXPECTED_TORCH_CUDA_VERSION,
    EXPECTED_TORCH_VERSION,
    INTERPRETER_CONTRACT_SCHEMA,
    PHASE,
    RESUME3_BLOCKED_REPORT_RELATIVE,
    RESUME3_BLOCKED_SUMMARY_RELATIVE,
    RESUME3_PREFLIGHT_RELATIVE,
    collect_interpreter_observation,
    finite_json_roundtrip,
    validate_interpreter_observation,
    validate_no_final_artifacts,
    validate_resume2_blocked_evidence,
    validate_resume3_outputs_absent,
    write_text_once,
)


def valid_observation() -> dict:
    return {
        "python_executable": str(Path(EXPECTED_PYTHON_EXECUTABLE).resolve()),
        "python_version": EXPECTED_PYTHON_VERSION,
        "numpy_version": EXPECTED_NUMPY_VERSION,
        "torch_version": EXPECTED_TORCH_VERSION,
        "torch_cuda_version": EXPECTED_TORCH_CUDA_VERSION,
        "cuda_available": False,
        "cuda_visible_devices": "",
        "python_no_user_site": "1",
        "no_user_site_flag": True,
    }


def test_resume3_constants_are_additive() -> None:
    assert PHASE.endswith("Resume3")
    assert BASE_RESUME2_BLOCKED_COMMIT == (
        "de9f23cc79cb2468291922a3101c7979f57a0889"
    )
    assert "resume3" in RESUME3_PREFLIGHT_RELATIVE
    assert "resume3" in RESUME3_BLOCKED_SUMMARY_RELATIVE
    assert "resume3" in RESUME3_BLOCKED_REPORT_RELATIVE


def test_valid_interpreter_observation_passes() -> None:
    result = validate_interpreter_observation(valid_observation())
    assert result["pass"] is True
    assert result["schema"] == INTERPRETER_CONTRACT_SCHEMA


@pytest.mark.parametrize(
    "key,bad_value",
    [
        ("python_executable", "/usr/bin/python"),
        ("python_version", "3.10.12"),
        ("numpy_version", "missing"),
        ("torch_version", "2.0.0"),
        ("torch_cuda_version", "11.8"),
        ("cuda_available", True),
        ("cuda_visible_devices", "0"),
        ("python_no_user_site", None),
        ("no_user_site_flag", False),
    ],
)
def test_interpreter_contract_rejects_mismatch(key: str, bad_value) -> None:
    observation = valid_observation()
    observation[key] = bad_value
    with pytest.raises(RuntimeError, match="interpreter contract mismatch"):
        validate_interpreter_observation(observation)


def test_runtime_observation_has_required_keys() -> None:
    observed = collect_interpreter_observation()
    assert set(observed) == set(valid_observation())


def test_validate_resume3_outputs_absent(tmp_path: Path) -> None:
    validate_resume3_outputs_absent(tmp_path)
    path = tmp_path / RESUME3_PREFLIGHT_RELATIVE
    path.parent.mkdir(parents=True)
    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="write-once"):
        validate_resume3_outputs_absent(tmp_path)


def test_validate_no_final_artifacts(tmp_path: Path) -> None:
    validate_no_final_artifacts(tmp_path)
    path = tmp_path / "reports/phase3_14b_r253_summary.json"
    path.parent.mkdir(parents=True)
    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="final artifacts"):
        validate_no_final_artifacts(tmp_path)


def test_write_text_once(tmp_path: Path) -> None:
    path = tmp_path / "report.md"
    write_text_once(path, "first")
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        write_text_once(path, "second")


def test_finite_json_roundtrip_rejects_nan() -> None:
    with pytest.raises(ValueError):
        finite_json_roundtrip({"value": float("nan")})


def test_resume2_blocked_contract(tmp_path: Path) -> None:
    summary_path = tmp_path / "reports/phase3_14b_r253_resume2_blocked_summary.json"
    report_path = tmp_path / "reports/phase3_14b_r253_resume2_blocked_report.md"
    summary_path.parent.mkdir(parents=True)
    payload = {
        "phase": "Phase3.14b-r2.5.3-Resume2",
        "verdict": "BLOCKED",
        "root_cause": "phase314b_r253_resume2_finalization_failed",
        "exit_code": 1,
        "failed_line": "0",
        "failed_command": (
            "python -c 'import ccda_phase3; from "
            "ccda_phase3.phase314b_r253_resume1_finalizer import "
            "corrected_pilot_view; print(\"repository import contract: PASS\")'"
        ),
        "finalizer_only": True,
        "gpu_pilot_rerun": False,
        "training_rerun": False,
        "reverse_rerun": False,
        "pythonpath": "/data/state_diff2:/data/state_diff2",
        "pythonpath_repository_root_first": True,
        "pilot_summary": "reports/phase3_14b_r253_pilot_summary.json",
        "pilot_summary_sha256": (
            "a6a8aa2faf739e80159af7e70cb57b0e2a4c852ab5af5c2d7ecb29ff31116c86"
        ),
        "expected_pilot_summary_sha256": (
            "a6a8aa2faf739e80159af7e70cb57b0e2a4c852ab5af5c2d7ecb29ff31116c86"
        ),
        "pilot_summary_unchanged": True,
        "preflight_report": None,
        "final_summary_sha256": None,
        "final_report_sha256": None,
        "validation_targets_used": False,
        "formal_test_read": False,
        "formal_training": False,
        "idm": False,
        "candidate_execution": False,
        "phase4": False,
        "cps": False,
        "train_only_recommendation": None,
        "selected_configuration": None,
    }
    summary_path.write_text(json.dumps(payload), encoding="utf-8")
    report_path.write_text("blocked\n", encoding="utf-8")
    result = validate_resume2_blocked_evidence(tmp_path)
    assert result["failed_command_used_bare_python"] is True
    assert result["preflight_executed"] is False


def test_resume2_blocked_contract_rejects_non_bare_command(tmp_path: Path) -> None:
    summary_path = tmp_path / "reports/phase3_14b_r253_resume2_blocked_summary.json"
    report_path = tmp_path / "reports/phase3_14b_r253_resume2_blocked_report.md"
    summary_path.parent.mkdir(parents=True)
    summary_path.write_text(
        json.dumps(
            {
                "phase": "Phase3.14b-r2.5.3-Resume2",
                "verdict": "BLOCKED",
            }
        ),
        encoding="utf-8",
    )
    report_path.write_text("blocked\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="evidence mismatch"):
        validate_resume2_blocked_evidence(tmp_path)


def test_runner_binds_every_python_command_to_python_bin() -> None:
    root = Path(__file__).resolve().parents[1]
    text = (root / "scripts/phase3_14b_r253_resume3_run.sh").read_text(
        encoding="utf-8"
    )
    assert 'EXPECTED_PYTHON="/miniforge3/envs/coord_bimanual/bin/python"' in text
    assert 'export PYTHONPATH="${ROOT}"' in text
    assert 'export PYTHONNOUSERSITE="1"' in text
    assert 'export CUDA_VISIBLE_DEVICES=""' in text
    assert '"${PYTHON_BIN}" scripts/phase3_14b_r253_resume3_preflight.py' in text
    assert '"${PYTHON_BIN}" scripts/phase3_14b_r253_resume3_finalize.py' in text
    command_lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip().startswith(("python ", "python3 "))
    ]
    assert command_lines == []


def test_entrypoints_bootstrap_repository_before_project_import() -> None:
    root = Path(__file__).resolve().parents[1]
    for relative in (
        "scripts/phase3_14b_r253_resume3_preflight.py",
        "scripts/phase3_14b_r253_resume3_finalize.py",
    ):
        text = (root / relative).read_text(encoding="utf-8")
        bootstrap = text.index("sys.path.insert(0, str(REPOSITORY_ROOT))")
        project_import = text.index("from ccda_phase3")
        assert bootstrap < project_import


def test_resume3_sources_have_no_training_or_reverse_calls() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = (
        "ccda_phase3/phase314b_r253_resume3_finalizer.py",
        "scripts/phase3_14b_r253_resume3_preflight.py",
        "scripts/phase3_14b_r253_resume3_finalize.py",
    )
    forbidden = (
        "train_factorized_residual_model(",
        "fit_factorized_prior(",
        "paired_reverse_pool_metrics(",
        "make_repair_scheduler(",
        "optimizer.step(",
    )
    for relative in paths:
        text = (root / relative).read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text
