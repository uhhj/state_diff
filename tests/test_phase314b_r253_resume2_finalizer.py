from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ccda_phase3 import phase314b_r253_resume2_finalizer as r2


def test_pythonpath_contract_accepts_repository_root_first(tmp_path: Path) -> None:
    result = r2.pythonpath_contract(tmp_path, f"{tmp_path}:/other/path")
    assert result["repository_root_is_first"] is True
    assert result["resolved_entries"][0] == str(tmp_path.resolve())


def test_pythonpath_contract_rejects_empty(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="PYTHONPATH is empty"):
        r2.pythonpath_contract(tmp_path, "")


def test_pythonpath_contract_rejects_repository_root_not_first(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="first PYTHONPATH entry"):
        r2.pythonpath_contract(tmp_path, f"/other/path:{tmp_path}")


def test_source_sha256_is_path_order_independent(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("a\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("b\n", encoding="utf-8")
    assert r2.source_sha256(tmp_path, ["a.py", "b.py"]) == r2.source_sha256(
        tmp_path,
        ["b.py", "a.py"],
    )


def test_source_sha256_binds_relative_path(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("same\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("same\n", encoding="utf-8")
    assert r2.source_sha256(tmp_path, ["a.py"]) != r2.source_sha256(
        tmp_path,
        ["b.py"],
    )


def test_validate_resume2_outputs_absent_rejects_existing(tmp_path: Path) -> None:
    target = tmp_path / r2.RESUME2_BLOCKED_REPORT_RELATIVE
    target.parent.mkdir(parents=True)
    target.write_text("blocked\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="write-once output already exists"):
        r2.validate_resume2_outputs_absent(tmp_path)


def test_validate_no_final_artifacts_rejects_summary(tmp_path: Path) -> None:
    target = tmp_path / r2.FINAL_SUMMARY_RELATIVE
    target.parent.mkdir(parents=True)
    target.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="final artifacts already exist"):
        r2.validate_no_final_artifacts(tmp_path)


def test_write_text_once_is_write_once(tmp_path: Path) -> None:
    path = tmp_path / "report.md"
    r2.write_text_once(path, "first\n")
    assert path.read_text(encoding="utf-8") == "first\n"
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        r2.write_text_once(path, "second\n")


def test_finite_json_roundtrip_rejects_nan() -> None:
    with pytest.raises(ValueError):
        r2.finite_json_roundtrip({"value": float("nan")})


def test_finite_json_roundtrip_returns_deep_json_copy() -> None:
    source = {"a": {"b": [1, 2, 3]}}
    copied = r2.finite_json_roundtrip(source)
    copied["a"]["b"].append(4)
    assert source == {"a": {"b": [1, 2, 3]}}


def test_validate_resume1_blocked_evidence_accepts_expected_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary_path = tmp_path / r2.RESUME1_BLOCKED_SUMMARY_RELATIVE
    report_path = tmp_path / r2.RESUME1_BLOCKED_REPORT_RELATIVE
    summary_path.parent.mkdir(parents=True)
    payload = {
        "phase": "Phase3.14b-r2.5.3-Resume1",
        "verdict": "BLOCKED",
        "root_cause": "phase314b_r253_resume1_finalization_failed",
        "exit_code": 1,
        "failed_line": "0",
        "finalizer_only": True,
        "gpu_pilot_rerun": False,
        "pilot_summary": r2.PILOT_SUMMARY_RELATIVE,
        "pilot_summary_sha256": r2.PILOT_SUMMARY_SHA256,
        "expected_pilot_summary_sha256": r2.PILOT_SUMMARY_SHA256,
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
    result = r2.validate_resume1_blocked_evidence(tmp_path)
    assert result["failure_class"] == "repository_root_missing_from_pythonpath"
    assert result["pilot_summary_sha256"] == r2.PILOT_SUMMARY_SHA256


def test_validate_resume1_blocked_evidence_rejects_mutated_contract(
    tmp_path: Path,
) -> None:
    summary_path = tmp_path / r2.RESUME1_BLOCKED_SUMMARY_RELATIVE
    report_path = tmp_path / r2.RESUME1_BLOCKED_REPORT_RELATIVE
    summary_path.parent.mkdir(parents=True)
    summary_path.write_text(
        json.dumps({"phase": "wrong", "verdict": "BLOCKED"}),
        encoding="utf-8",
    )
    report_path.write_text("blocked\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="blocked evidence mismatch"):
        r2.validate_resume1_blocked_evidence(tmp_path)


def test_run_wrapper_exports_pythonpath_before_import() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts/phase3_14b_r253_resume2_run.sh"
    text = script.read_text(encoding="utf-8")
    export_index = text.index('export PYTHONPATH="${ROOT}')
    import_index = text.index("python -c 'import ccda_phase3")
    assert export_index < import_index
    assert "phase3_14b_r253_resume2_blocked_summary.json" in text
    assert "phase3_14b_r253_resume_blocked_summary.json" not in text


def test_python_entries_bootstrap_root_before_project_import() -> None:
    root = Path(__file__).resolve().parents[1]
    for relative in (
        "scripts/phase3_14b_r253_resume2_preflight.py",
        "scripts/phase3_14b_r253_resume2_finalize.py",
    ):
        text = (root / relative).read_text(encoding="utf-8")
        bootstrap = text.index("sys.path.insert(0, str(REPOSITORY_ROOT))")
        project_import = text.index("from ccda_phase3")
        assert bootstrap < project_import


def test_resume2_scripts_do_not_call_training_or_reverse() -> None:
    root = Path(__file__).resolve().parents[1]
    joined = "\n".join(
        (root / relative).read_text(encoding="utf-8")
        for relative in (
            "ccda_phase3/phase314b_r253_resume2_finalizer.py",
            "scripts/phase3_14b_r253_resume2_preflight.py",
            "scripts/phase3_14b_r253_resume2_finalize.py",
            "scripts/phase3_14b_r253_resume2_run.sh",
        )
    )
    forbidden = (
        "train_factorized_residual_model(",
        "fit_factorized_prior(",
        "paired_reverse_pool_metrics",
        "make_repair_scheduler(",
        "optimizer.step(",
    )
    for token in forbidden:
        assert token not in joined
