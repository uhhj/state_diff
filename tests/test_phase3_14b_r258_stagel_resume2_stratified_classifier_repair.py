from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from ccda_phase3 import phase314b_r258_stagel_resume2_stratified_classifier_repair as resume2


def strata(counts):
    return {
        "discriminator_counts": counts,
        "shared_discriminator_predicate": None,
        "records": [],
        "stratum_count": 4,
    }


def record(external=None, internal=None, locus=resume2.EXPECTED_RECORD_LOCUS):
    return {
        "assembly_comparison": {
            "locus": locus,
            "corrected_discriminator_predicate": None,
            "external_multiplier_strata": strata(external or {"direction_retention": 2}),
            "internal_scale_strata": strata(internal or {"segment_geometry": 3}),
        }
    }


def scientific(records=None):
    records = [record() for _ in range(27)] if records is None else records
    return {
        "root_cause": resume2.EXPECTED_BASE_ROOT_CAUSE,
        "required_next_path": resume2.EXPECTED_BASE_NEXT_PATH,
        "selected_configuration": None,
        "train_only_recommendation": None,
        "predicate_assembly_audit": {
            "callback_off_on_pair_count": 132,
            "all_callback_results_bit_exact": True,
            "oof_records": records,
            "classification": {
                "root_cause": resume2.EXPECTED_BASE_ROOT_CAUSE,
                "required_next_path": resume2.EXPECTED_BASE_NEXT_PATH,
            },
        },
        **{key: False for key in resume2.FALSE_BOUNDARIES},
    }


def report(records=None):
    return {
        "execution_verdict": "PASS",
        "scientific_status": "BLOCKED",
        "root_cause": resume2.EXPECTED_BASE_ROOT_CAUSE,
        "required_next_path": resume2.EXPECTED_BASE_NEXT_PATH,
        "selected_configuration": None,
        "train_only_recommendation": None,
        "recovery_contract": {
            "underlying_stage_l_single_run_result_sha256": resume2.EXPECTED_BASE_SINGLE_RUN_SHA256,
            "underlying_callback_pair_count": 132,
            "stage_l_science_modified": False,
            "existing_test_gates_rerun": False,
        },
        "stage_l_result": {
            "execution_verdict": "PASS",
            "scientific_result": scientific(records),
        },
    }


def source_with_bug():
    return '''\ndef classify(records):\n    for record in records:\n        value = record["external_multiplier_strata"]\n        shared = value.get("shared_discriminator_predicate")\n    if False:\n        pass\n    elif external_predicates or internal_predicates:\n        pass\n    else:\n        root = "phase314b_r258_stagel_callback_predicate_masks_not_discriminative"\n\ndef run_calibration():\n    pass\n'''


def test_phase_constants():
    assert resume2.PHASE.endswith("Stage L Resume2")
    assert resume2.BASE_EVIDENCE_COMMIT == "c8c9bd0139ab7249689855c55be1b491d759a3f4"


def test_scope_is_add_only_three_paths():
    assert len(resume2.IMPLEMENTATION_PATHS) == 3
    assert all(status == "A" for status, _ in resume2.IMPLEMENTATION_PATHS)


def test_stable_json_is_deterministic():
    assert resume2.stable_json_bytes({"b": 1, "a": 2}) == resume2.stable_json_bytes({"a": 2, "b": 1})


def test_write_once(tmp_path):
    path = tmp_path / "x.json"
    resume2.write_once(path, b"{}\n")
    with pytest.raises(resume2.StageLResume2Error):
        resume2.write_once(path, b"{}\n")


def test_classifier_bug_is_detected():
    value = resume2.validate_classifier_source_bug(source_with_bug())
    assert value["old_classifier_ignores_heterogeneous_discriminator_counts"] is True


def test_classifier_bug_rejects_already_fixed_source():
    source = source_with_bug().replace(
        'shared = value.get("shared_discriminator_predicate")',
        'shared = value.get("shared_discriminator_predicate")\n        counts = value.get("discriminator_counts")',
    )
    with pytest.raises(resume2.StageLResume2Error):
        resume2.validate_classifier_source_bug(source)


def test_extract_scientific_result():
    value = resume2.extract_stage_l_scientific_result(report())
    assert value["predicate_assembly_audit"]["callback_off_on_pair_count"] == 132


def test_extract_rejects_single_run_change():
    value = report()
    value["recovery_contract"]["underlying_stage_l_single_run_result_sha256"] = "bad"
    with pytest.raises(resume2.StageLResume2Error):
        resume2.extract_stage_l_scientific_result(value)


def test_extract_rejects_boundary_violation():
    value = report()
    value["stage_l_result"]["scientific_result"]["frozen_probe_accessed"] = True
    with pytest.raises(resume2.StageLResume2Error):
        resume2.extract_stage_l_scientific_result(value)


def test_collect_stratified_signal():
    signal = resume2.collect_stratified_signal([record() for _ in range(27)])
    assert signal.records_with_any_stratified_signal == 27
    assert signal.external_predicate_counts["direction_retention"] == 54
    assert signal.internal_predicate_counts["segment_geometry"] == 81


def test_collect_accepts_one_side_only():
    value = record(external={}, internal={"topology": 4})
    # helper uses fallback for empty dict; construct exact one-sided record
    value["assembly_comparison"]["external_multiplier_strata"] = strata({})
    signal = resume2.collect_stratified_signal([value])
    assert signal.external_record_count == 0
    assert signal.internal_record_count == 1


def test_collect_rejects_locus_change():
    with pytest.raises(resume2.StageLResume2Error):
        resume2.collect_stratified_signal([record(locus="other")])


def test_collect_rejects_missing_counts():
    value = record()
    value["assembly_comparison"]["external_multiplier_strata"] = strata({})
    value["assembly_comparison"]["internal_scale_strata"] = strata({})
    with pytest.raises(resume2.StageLResume2Error):
        resume2.collect_stratified_signal([value])


def test_collect_rejects_shared_predicate():
    value = record()
    value["assembly_comparison"]["external_multiplier_strata"]["shared_discriminator_predicate"] = "topology"
    with pytest.raises(resume2.StageLResume2Error):
        resume2.collect_stratified_signal([value])


def test_corrected_classification():
    result = resume2.corrected_classification([record() for _ in range(27)])
    assert result["root_cause"] == resume2.CORRECTED_ROOT_CAUSE
    assert result["required_next_path"] == resume2.CORRECTED_NEXT_PATH
    assert result["all_records_scale_stratified_and_heterogeneous"] is True


def test_corrected_classification_requires_27_records():
    with pytest.raises(resume2.StageLResume2Error):
        resume2.corrected_classification([record()])


def test_build_report(tmp_path):
    source = tmp_path / "ccda_phase3"
    source.mkdir()
    (source / "phase314b_r258_stagel_predicate_assembly_audit.py").write_text(source_with_bug(), encoding="utf-8")
    result = resume2.build_corrected_report(
        root=tmp_path,
        repository={"head": resume2.BASE_EVIDENCE_COMMIT},
        base_report=report(),
    )
    assert result["execution_verdict"] == "PASS"
    assert result["scientific_status"] == "BLOCKED"
    assert result["correction_contract"]["stage_l_science_rerun"] is False
    assert result["corrected_classification"]["record_count"] == 27


def test_build_report_preserves_null_selection(tmp_path):
    source = tmp_path / "ccda_phase3"
    source.mkdir()
    (source / "phase314b_r258_stagel_predicate_assembly_audit.py").write_text(source_with_bug(), encoding="utf-8")
    result = resume2.build_corrected_report(root=tmp_path, repository={}, base_report=report())
    assert result["selected_configuration"] is None
    assert result["train_only_recommendation"] is None


def test_blocked_payload_boundaries_false():
    payload = resume2.blocked_payload(RuntimeError("x"), None)
    assert payload["execution_verdict"] == "BLOCKED"
    assert all(value is False for value in payload["boundaries"].values())


def init_repo(path: Path):
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True)


def test_commit_name_status_non_root(tmp_path):
    init_repo(tmp_path)
    (tmp_path / "base").write_text("x", encoding="utf-8")
    subprocess.run(["git", "add", "base"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=tmp_path, check=True)
    (tmp_path / "added").write_text("y", encoding="utf-8")
    subprocess.run(["git", "add", "added"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "add"], cwd=tmp_path, check=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True).strip()
    assert resume2.commit_name_status(tmp_path, commit) == (("A", "added"),)


def test_require_false_rejects_missing():
    with pytest.raises(resume2.StageLResume2Error):
        resume2.require_false({}, ["x"], "value")


def test_sha256_path(tmp_path):
    path = tmp_path / "x"
    path.write_bytes(b"abc")
    assert resume2.sha256_path(path) == resume2.sha256_bytes(b"abc")


def test_output_paths_are_distinct():
    assert resume2.SUCCESS_REPORT != resume2.BASE_REPORT
    assert resume2.SUCCESS_REPORT != resume2.BLOCKED_REPORT


def test_corrected_next_path_does_not_modify_masks():
    assert resume2.CORRECTED_NEXT_PATH == "STRATIFY_OOF_REJECTION_BY_CALLBACK_SCALE_AND_PREDICATE"
