from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ccda_phase3 import phase314b_r258_stagef_constraint_aware_surrogate as stagef
from ccda_phase3 import phase314b_r258_stagef_resume1_translation_fix as resume1


def make_names(rows: int) -> np.ndarray:
    values = np.asarray(stagef.CONDITION_VALUES)
    return values[np.arange(rows) % len(values)]


class FakeReference:
    def __init__(self) -> None:
        self.center_log = np.full((4, 23), np.log(0.02), dtype=np.float64)
        self.scale_log = np.full((4, 23), 0.25, dtype=np.float64)

    def validate(self) -> None:
        assert self.center_log.shape == (4, 23)
        assert self.scale_log.shape == (4, 23)


def make_context() -> dict:
    return {"stage_d_contract": SimpleNamespace(reference=FakeReference())}


def make_condition(rows: int = 12, base: float = 0.8) -> np.ndarray:
    condition = np.zeros((rows, 243), dtype=np.float32)
    history = condition[:, : 3 * 67].reshape(rows, 3, 67)
    for row in range(rows):
        for step in range(3):
            points = np.zeros((24, 2), dtype=np.float32)
            points[0] = np.asarray([base, base + 0.01 * step], dtype=np.float32)
            for index in range(1, 24):
                length = 0.02
                if index == 3:
                    length = 4.8e-5
                points[index] = points[index - 1] + np.asarray(
                    [length, 2.0e-6 * ((index % 3) - 1)], dtype=np.float32
                )
            points[:, 1] += np.float32(0.001 * row)
            history[row, step, :48] = points.reshape(-1)
            robot = history[row, step, 48:]
            robot[:6] = np.linspace(-0.2, 0.2, 6, dtype=np.float32)
            robot[6:12] = np.linspace(0.1, -0.1, 6, dtype=np.float32)
            robot[12:15] = np.asarray(
                [points[:, 0].mean() + 0.03, points[:, 1].mean() - 0.02, 0.15],
                dtype=np.float32,
            )
            robot[15:19] = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    condition[:, 201:] = np.linspace(-0.5, 0.5, 42, dtype=np.float32)[None]
    return condition


def make_control(rows: int = 12, base: float = 0.8) -> np.ndarray:
    output = np.zeros((rows, 4, 48), dtype=np.float32)
    for row in range(rows):
        for horizon in range(4):
            points = np.zeros((24, 2), dtype=np.float32)
            points[0] = np.asarray(
                [base + 0.004 * horizon, base + 0.01 * horizon], dtype=np.float32
            )
            for index in range(1, 24):
                length = 0.02
                if index == 3:
                    length = 4.8e-5
                points[index] = points[index - 1] + np.asarray(
                    [length, 2.0e-6 * ((index % 3) - 1)], dtype=np.float32
                )
            points[:, 1] += np.float32(0.001 * row)
            output[row, horizon] = points.reshape(-1)
    return output


def raw_unclipped_z(control: np.ndarray) -> np.ndarray:
    points = np.asarray(control, dtype=np.float32).reshape(-1, 4, 24, 2)
    vectors = points[:, :, 1:, :] - points[:, :, :-1, :]
    lengths = np.sqrt(np.sum(vectors * vectors, axis=-1)).astype(np.float64)
    return (np.log(np.maximum(lengths, 1.0e-12)) - np.log(0.02)) / 0.25


def init_repo(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", "-b", "Experiment1"], cwd=str(root), check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],
        cwd=str(root),
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"], cwd=str(root), check=True
    )


def commit_all(root: Path, message: str) -> str:
    subprocess.run(["git", "add", "-A"], cwd=str(root), check=True)
    subprocess.run(["git", "commit", "-q", "-m", message], cwd=str(root), check=True)
    return resume1.git_output(root, "rev-parse", "HEAD")


def test_phase_constant():
    assert resume1.PHASE == "Phase3.14b-r2.5.8 Stage F Resume1"


def test_initial_commit_constant():
    assert resume1.EXPECTED_INITIAL_HEAD == (
        "519531f411c43b17a668c3c6c1a43b46a94f40a7"
    )


def test_remote_commit_constant():
    assert resume1.EXPECTED_INITIAL_REMOTE == (
        "6758ea7ad800667a436b0243d3b1f6c63256d854"
    )


def test_submodule_constant():
    assert resume1.EXPECTED_SUBMODULE_COMMIT.startswith("633a887")


def test_original_path_population():
    assert len(resume1.ORIGINAL_IMPLEMENTATION_PATHS) == 7


def test_resume1_path_population():
    assert len(resume1.RESUME1_IMPLEMENTATION_PATHS) == 8


def test_blocked_provenance_population():
    assert len(resume1.BLOCKED_PROVENANCE_PATHS) == 2


def test_success_population():
    assert len(resume1.RESUME1_SUCCESS_PATHS) == 5


def test_original_module_sha_constant():
    assert resume1.ORIGINAL_IMPLEMENTATION_PATH_SHA256[
        "ccda_phase3/phase314b_r258_stagef_constraint_aware_surrogate.py"
    ] == "3443ae27d7688a5646c9c5c95e33e916ae9dbf3c123c8ef41d097671be078d1b"


def test_constraint_z_clip_constant():
    assert stagef.CONSTRAINT_Z_CLIP == 4.0
    assert stagef.ConstraintAwareSpec().constraint_z_clip == 4.0


def test_spec_structural_tolerance():
    assert stagef.ConstraintAwareSpec().translation_structural_tolerance == 1.0e-10


def test_spec_quantization_envelope():
    spec = stagef.ConstraintAwareSpec()
    assert spec.translation_float32_z_ulp_factor == 8.0
    assert spec.translation_float32_z_bound_max == 1.0e-2


def test_spec_rejects_z_clip_change():
    with pytest.raises(ValueError):
        stagef.ConstraintAwareSpec(constraint_z_clip=3.0).validate()


def test_stable_center_points_dimension():
    points = make_control(3).reshape(3, 4, 24, 2)
    centered = stagef.stable_center_points(points)
    assert centered.shape == points.shape
    assert np.max(np.abs(np.mean(centered, axis=2))) < 1.0e-12


def test_stable_center_points_exact_translation():
    points = make_control(3).reshape(3, 4, 24, 2).astype(np.float64)
    translated = points + np.asarray([0.375, -0.625])[None, None, None]
    assert np.allclose(
        stagef.stable_center_points(points),
        stagef.stable_center_points(translated),
        atol=1.0e-12,
        rtol=0.0,
    )


def test_stable_segment_vectors_exact_translation():
    points = make_control(3).reshape(3, 4, 24, 2).astype(np.float64)
    translated = points + np.asarray([0.375, -0.625])[None, None, None]
    assert np.allclose(
        stagef.stable_segment_vectors(points),
        stagef.stable_segment_vectors(translated),
        atol=1.0e-12,
        rtol=0.0,
    )


def test_exact_translate_preserves_float64():
    condition, control = stagef.exact_translate_cable_inputs(
        condition=make_condition(2),
        control=make_control(2),
        offset_xy=(0.375, -0.625),
    )
    assert condition.dtype == np.float64
    assert control.dtype == np.float64


def test_exact_translate_moves_cable_and_ee_together():
    original = make_condition(2)
    translated, _ = stagef.exact_translate_cable_inputs(
        condition=original,
        control=make_control(2),
        offset_xy=(0.375, -0.625),
    )
    before = original[:, :201].reshape(2, 3, 67)
    after = translated[:, :201].reshape(2, 3, 67)
    before_cable = before[:, :, :48].reshape(2, 3, 24, 2)
    after_cable = after[:, :, :48].reshape(2, 3, 24, 2)
    assert np.allclose(after_cable - before_cable, [0.375, -0.625])
    before_ee = before[:, :, 60:63]
    after_ee = after[:, :, 60:63]
    assert np.allclose(after_ee[:, :, :2] - before_ee[:, :, :2], [0.375, -0.625])
    assert np.array_equal(after_ee[:, :, 2], before_ee[:, :, 2])


def test_unclipped_log_z_reproduces_quantization_amplification():
    control = make_control(12)
    _, translated = stagef.staged258.translate_cable_inputs(
        condition=make_condition(12),
        control=control,
        offset_xy=(0.375, -0.625),
    )
    difference = np.abs(raw_unclipped_z(control) - raw_unclipped_z(translated))
    assert float(np.max(difference)) > 2.0e-5


def test_clipped_constraint_z_is_bounded():
    z = stagef.constraint_z_features(make_control(12), make_context())
    assert z.shape == (12, 4, 23)
    assert float(np.max(z)) <= 4.0
    assert float(np.min(z)) >= -4.0


def test_collapsed_segment_saturates_at_lower_constraint():
    z = stagef.constraint_z_features(make_control(2), make_context())
    assert np.allclose(z[:, :, 2], -4.0)


def test_constraint_z_exact_translation_invariant():
    condition = make_condition(12)
    control = make_control(12)
    _, translated = stagef.exact_translate_cable_inputs(
        condition=condition,
        control=control,
        offset_xy=(0.375, -0.625),
    )
    original_z = stagef.constraint_z_features(control, make_context())
    translated_z = stagef.constraint_z_features(translated, make_context())
    assert np.allclose(original_z, translated_z, atol=1.0e-10, rtol=0.0)


def test_constraint_z_float32_drift_within_derived_envelope():
    condition = make_condition(12)
    control = make_control(12)
    _, translated = stagef.staged258.translate_cable_inputs(
        condition=condition,
        control=control,
        offset_xy=(0.375, -0.625),
    )
    difference = np.abs(
        stagef.constraint_z_features(control, make_context())
        - stagef.constraint_z_features(translated, make_context())
    )
    envelope = stagef._float32_constraint_z_bound(
        control=control,
        offset_xy=(0.375, -0.625),
        context=make_context(),
        spec=stagef.ConstraintAwareSpec(),
    )
    assert float(np.max(difference)) <= envelope["applied_bound"]
    assert envelope["applied_bound"] <= envelope["maximum_allowed_bound"]


@pytest.mark.parametrize(
    "mode,dimension",
    [
        ("full_centered_constraint", 527),
        ("full_segment_constraint", 525),
    ],
)
def test_corrected_full_feature_dimensions(mode, dimension):
    features = stagef.build_constraint_features(
        condition=make_condition(8),
        control=make_control(8),
        condition_name=make_names(8),
        feature_mode=mode,
        context=make_context(),
    )
    assert features.shape == (8, dimension)


@pytest.mark.parametrize(
    "mode",
    ("anchor", "full_centered_constraint", "full_segment_constraint"),
)
def test_corrected_selectable_features_ignore_condition_labels(mode):
    condition = make_condition(8)
    control = make_control(8)
    first = stagef.build_constraint_features(
        condition=condition,
        control=control,
        condition_name=make_names(8),
        feature_mode=mode,
        context=make_context(),
    )
    second = stagef.build_constraint_features(
        condition=condition,
        control=control,
        condition_name=make_names(8)[::-1],
        feature_mode=mode,
        context=make_context(),
    )
    assert np.array_equal(first, second)


def test_feature_block_layout_centered():
    blocks = stagef._feature_block_slices("full_centered_constraint")
    assert blocks[-1][0] == "constraint_z"
    assert blocks[-1][1] == slice(435, 527)


def test_feature_block_layout_segment():
    blocks = stagef._feature_block_slices("full_segment_constraint")
    assert blocks[4][0] == "constraint_z"
    assert blocks[4][1] == slice(421, 513)


def test_translation_audit_passes_pathological_full_resolution_inputs():
    audit = stagef.translation_invariance_audit(
        condition=make_condition(12),
        control=make_control(12),
        condition_name=make_names(12),
        context=make_context(),
        spec=stagef.ConstraintAwareSpec(),
    )
    assert audit["all_selectable_modes_invariant"]
    assert audit["records"]["full_centered_constraint"]["structural"]["pass"]
    assert audit["records"]["full_segment_constraint"]["structural"]["pass"]


def test_translation_audit_records_constraint_z_separately():
    audit = stagef.translation_invariance_audit(
        condition=make_condition(6),
        control=make_control(6),
        condition_name=make_names(6),
        context=make_context(),
        spec=stagef.ConstraintAwareSpec(),
    )
    centered = audit["records"]["full_centered_constraint"]["float32_roundtrip"]
    assert centered["blocks"]["constraint_z"]["kind"] == "constraint_z"
    assert centered["blocks"]["constraint_z"]["pass"]


def test_translation_audit_keeps_ordinary_tolerance_strict():
    audit = stagef.translation_invariance_audit(
        condition=make_condition(6),
        control=make_control(6),
        condition_name=make_names(6),
        context=make_context(),
        spec=stagef.ConstraintAwareSpec(),
    )
    blocks = audit["records"]["full_centered_constraint"]["float32_roundtrip"]["blocks"]
    assert blocks["history_centered"]["tolerance"] == 2.0e-5
    assert blocks["future_centered"]["tolerance"] == 2.0e-5


def test_translation_audit_records_ulp_envelope():
    audit = stagef.translation_invariance_audit(
        condition=make_condition(6),
        control=make_control(6),
        condition_name=make_names(6),
        context=make_context(),
        spec=stagef.ConstraintAwareSpec(),
    )
    envelope = audit["constraint_z_quantization_envelope"]
    assert envelope["maximum_coordinate_spacing"] > 0.0
    assert envelope["minimum_clipped_segment_length"] > 0.0
    assert envelope["applied_bound"] <= 1.0e-2


def test_translation_audit_rejects_insufficient_coordinate_resolution():
    condition = make_condition(2, base=1000.0)
    control = make_control(2, base=1000.0)
    with pytest.raises(stagef.ConstraintAwareSurrogateError):
        stagef.translation_invariance_audit(
            condition=condition,
            control=control,
            condition_name=make_names(2),
            context=make_context(),
            spec=stagef.ConstraintAwareSpec(translation_float32_z_bound_max=1.0e-4),
        )


def test_candidate_matrix_unchanged():
    assert len(stagef.CANDIDATE_DEFINITIONS) == 13
    assert tuple(item.candidate_id for item in stagef.CANDIDATE_DEFINITIONS)[-1] == (
        "projected_oracle_control"
    )


def test_integrator_contract_unchanged():
    value = stagef.fixed_integrator_definition()
    oracle = stagef.oracle_integrator_definition()
    assert value.integration_mode == oracle.integration_mode
    assert value.bound_mode == oracle.bound_mode
    assert value.lower_z == oracle.lower_z
    assert value.upper_z == oracle.upper_z
    assert value.retention_min == oracle.retention_min
    assert value.maximum_scale == oracle.maximum_scale


def test_resume1_stable_json_deterministic():
    assert resume1.stable_json_bytes({"b": 2, "a": 1}) == resume1.stable_json_bytes(
        {"a": 1, "b": 2}
    )


def test_resume1_atomic_write_once(tmp_path):
    path = tmp_path / "value.json"
    resume1.atomic_write_once(path, b"{}\n")
    assert path.read_bytes() == b"{}\n"
    with pytest.raises(FileExistsError):
        resume1.atomic_write_once(path, b"{}\n")


def test_status_paths_sorted(tmp_path):
    init_repo(tmp_path)
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    assert resume1.status_paths(tmp_path) == ("a.txt", "b.txt")


def test_commit_helpers(tmp_path):
    init_repo(tmp_path)
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    first = commit_all(tmp_path, "first")
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")
    second = commit_all(tmp_path, "second")
    assert resume1.commit_parent(tmp_path, second) == first
    assert resume1.commit_subject(tmp_path, second) == "second"
    assert resume1.commit_paths(tmp_path, second) == ("b.txt",)


def test_assert_commit_shape_accepts(tmp_path):
    init_repo(tmp_path)
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    first = commit_all(tmp_path, "first")
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")
    second = commit_all(tmp_path, "second")
    resume1.assert_commit_shape(
        tmp_path,
        commit=second,
        parent=first,
        subject="second",
        paths=("b.txt",),
    )


@pytest.mark.parametrize("field", ("parent", "subject", "paths"))
def test_assert_commit_shape_rejects(tmp_path, field):
    init_repo(tmp_path)
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    first = commit_all(tmp_path, "first")
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")
    second = commit_all(tmp_path, "second")
    values = {
        "commit": second,
        "parent": first,
        "subject": "second",
        "paths": ("b.txt",),
    }
    if field == "parent":
        values["parent"] = second
    elif field == "subject":
        values["subject"] = "wrong"
    else:
        values["paths"] = ("a.txt",)
    with pytest.raises(resume1.StageFResume1Error):
        resume1.assert_commit_shape(tmp_path, **values)


def test_validate_original_blocked_reports(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    gate = {
        "phase": "Phase3.14b-r2.5.8 Stage F",
        "verdict": "PASS",
        "test_file_count": 56,
        "passed_test_count": 1408,
        "scientific_calibration_run": False,
    }
    blocked = {
        "phase": "Phase3.14b-r2.5.8 Stage F",
        "verdict": "BLOCKED",
        "scientific_status": "BLOCKED",
        "root_cause": "phase314b_r258_stagef_execution_failed_before_completion",
        "base_evidence_commit": resume1.BASE_EVIDENCE_COMMIT,
        "head": resume1.ORIGINAL_IMPLEMENTATION_COMMIT,
        "origin_experiment1": resume1.BASE_EVIDENCE_COMMIT,
        "scientific_result_sealed": False,
        "success_evidence_commit_created": False,
        "push_completed": False,
        "selected_configuration": None,
        "train_only_recommendation": None,
        "selection_holdout_evaluated": False,
        "frozen_probe_accessed": False,
        "new_diffusion_model_candidate_trained": False,
        "reverse_sampling_run": False,
        "formal_training_run": False,
        "idm_run": False,
        "candidate_execution": False,
        "deformable_ravens_executed": False,
        "phase4": False,
        "cps": False,
        "failed_command": "python stagef worker",
    }
    (tmp_path / resume1.BLOCKED_PROVENANCE_PATHS[0]).write_text(
        json.dumps(gate), encoding="utf-8"
    )
    (tmp_path / resume1.BLOCKED_PROVENANCE_PATHS[1]).write_text(
        json.dumps(blocked), encoding="utf-8"
    )
    result = resume1.validate_original_blocked_reports(tmp_path)
    assert result["test_gate_sha256"]
    assert result["blocked_sha256"]


def test_validate_original_blocked_reports_rejects_scientific_seal(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    gate = {
        "phase": "Phase3.14b-r2.5.8 Stage F",
        "verdict": "PASS",
        "test_file_count": 56,
        "passed_test_count": 1408,
        "scientific_calibration_run": False,
    }
    blocked = {
        "phase": "Phase3.14b-r2.5.8 Stage F",
        "verdict": "BLOCKED",
        "scientific_status": "BLOCKED",
        "root_cause": "phase314b_r258_stagef_execution_failed_before_completion",
        "base_evidence_commit": resume1.BASE_EVIDENCE_COMMIT,
        "head": resume1.ORIGINAL_IMPLEMENTATION_COMMIT,
        "origin_experiment1": resume1.BASE_EVIDENCE_COMMIT,
        "scientific_result_sealed": True,
        "success_evidence_commit_created": False,
        "push_completed": False,
        "selected_configuration": None,
        "train_only_recommendation": None,
        "selection_holdout_evaluated": False,
        "frozen_probe_accessed": False,
        "new_diffusion_model_candidate_trained": False,
        "reverse_sampling_run": False,
        "formal_training_run": False,
        "idm_run": False,
        "candidate_execution": False,
        "deformable_ravens_executed": False,
        "phase4": False,
        "cps": False,
        "failed_command": "python stagef worker",
    }
    (tmp_path / resume1.BLOCKED_PROVENANCE_PATHS[0]).write_text(
        json.dumps(gate), encoding="utf-8"
    )
    (tmp_path / resume1.BLOCKED_PROVENANCE_PATHS[1]).write_text(
        json.dumps(blocked), encoding="utf-8"
    )
    with pytest.raises(resume1.StageFResume1Error):
        resume1.validate_original_blocked_reports(tmp_path)
