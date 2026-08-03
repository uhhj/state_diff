from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pytest

from ccda_phase3 import phase314b_r260_stagec_objective_train_baseline as stagec


def _synthetic_arrays(rows: int = stagec.EXPECTED_OBJECTIVE_ROWS) -> Dict[str, np.ndarray]:
    groups = [
        "phase314b_r260_objective_train_seed_{}".format(900000 + index)
        for index in range(stagec.EXPECTED_OBJECTIVE_GROUPS)
    ]
    # Ensure every group and both conditions are represented, then fill the rest.
    group_rows = []
    condition_rows = []
    seed_rows = []
    time_rows = []
    for index, group in enumerate(groups):
        for condition_id, condition in enumerate(stagec.EXPECTED_CONDITIONS):
            group_rows.append(group)
            condition_rows.append(condition)
            seed_rows.append(900000 + index)
            time_rows.append(condition_id)
    cursor = 0
    while len(group_rows) < rows:
        index = cursor % len(groups)
        condition_id = (cursor // len(groups)) % 2
        group_rows.append(groups[index])
        condition_rows.append(stagec.EXPECTED_CONDITIONS[condition_id])
        seed_rows.append(900000 + index)
        time_rows.append(2 + cursor // (2 * len(groups)))
        cursor += 1
    rng = np.random.RandomState(260)
    paper = rng.normal(size=(rows, stagec.PAPER_X_DIM)).astype(np.float32)
    current = paper.reshape(rows, stagec.DEFAULT_TH, stagec.STATE_DIM)[:, -1]
    future = np.stack(
        [current + np.float32(0.01 * (step + 1)) for step in range(stagec.DEFAULT_TF)],
        axis=1,
    ).astype(np.float32)
    state_action = np.concatenate(
        [paper, rng.normal(size=(rows, stagec.DEFAULT_TH * stagec.ACTION_DIM)).astype(np.float32)],
        axis=1,
    ).astype(np.float32)
    action = rng.normal(size=(rows, stagec.ACTION_DIM)).astype(np.float32)
    conditions = np.asarray(condition_rows)
    success = conditions == "free"
    final_fraction = np.where(success, 1.0, 0.8).astype(np.float32)
    engagement = np.where(conditions == "free", -1, 3).astype(np.int64)
    release = np.where(conditions == "free", -1, 8).astype(np.int64)
    pre = np.asarray(time_rows) < engagement
    return {
        "paper_x": paper,
        "state_action_x": state_action,
        "y_state": future,
        "y_final_state": future[:, -1].copy(),
        "y_action": action,
        "condition_name": conditions,
        "visible_seed": np.asarray(seed_rows, dtype=np.int64),
        "split_name": np.asarray([stagec.OBJECTIVE_ROLE] * rows),
        "source_file": np.asarray(["synthetic.pkl"] * rows),
        "pair_group": np.asarray(group_rows),
        "window_t": np.asarray(time_rows, dtype=np.int64),
        "success": success.astype(np.bool_),
        "final_fraction": final_fraction,
        "engagement_step": engagement,
        "release_step": release,
        "pre_engagement": pre.astype(np.bool_),
    }


@pytest.fixture(scope="module")
def arrays() -> Dict[str, np.ndarray]:
    return _synthetic_arrays()


@pytest.fixture(scope="module")
def folds(arrays):
    return stagec.deterministic_group_folds(arrays["pair_group"])


def test_phase_and_schema():
    assert stagec.PHASE == "Phase3.14b-r2.6.0 Stage C"
    assert stagec.SCHEMA.endswith("objective_train_baseline_v1")


@pytest.mark.parametrize(
    "name,value",
    [
        ("BASE_HEAD", "5296ecb948b91058ce2f0b502bf50baffac133d8"),
        ("BASE_IMPLEMENTATION", "4b19caa1de2b89ddf700dcad4a4b72928c2a4818"),
        ("STAGEA_EVIDENCE", "636a2853db1a972073e2dd254e75a9724a5a1d17"),
        ("STAGEA_IMPLEMENTATION", "b7ca7bc3396c0879f13662fd953881a9f42e40f2"),
        ("EXPECTED_SUBMODULE", "633a88752445cf5d6776ed374fdbbdb35f93050c"),
    ],
)
def test_frozen_identities(name, value):
    assert getattr(stagec, name) == value


@pytest.mark.parametrize(
    "name,value",
    [
        ("EXPECTED_OBJECTIVE_ROWS", 3006),
        ("EXPECTED_OBJECTIVE_GROUPS", 320),
        ("EXPECTED_OBJECTIVE_SEEDS", 320),
        ("EXPECTED_FOLDS", 6),
        ("CABLE_DIM", 48),
        ("STATE_DIM", 87),
        ("PAPER_X_DIM", 261),
        ("STATE_ACTION_X_DIM", 303),
        ("ACTION_DIM", 14),
    ],
)
def test_dimension_contract(name, value):
    assert getattr(stagec, name) == value


def test_stable_json_bytes_has_terminal_newline():
    payload = stagec.stable_json_bytes({"b": 2, "a": 1})
    assert payload.endswith(b"\n")
    assert payload.index(b'"a"') < payload.index(b'"b"')


def test_compact_json_bytes_no_terminal_newline():
    payload = stagec.compact_json_bytes({"b": 2, "a": 1})
    assert not payload.endswith(b"\n")
    assert payload == b'{"a":1,"b":2}'


def test_sha256_bytes_known():
    assert stagec.sha256_bytes(b"abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_sha256_array_identity():
    value = np.arange(6, dtype=np.float32).reshape(2, 3)
    assert stagec.sha256_array(value) == stagec.sha256_array(value.copy())
    assert stagec.sha256_array(value) != stagec.sha256_array(value.astype(np.float64))


def test_atomic_write_once(tmp_path):
    path = tmp_path / "value.json"
    stagec.atomic_write_once(path, b"x")
    assert path.read_bytes() == b"x"
    with pytest.raises(stagec.StageCError):
        stagec.atomic_write_once(path, b"y")


def test_write_ahead_root():
    assert stagec.write_ahead_root(Path("/data/state_diff2")) == Path(
        "/data/state_diff2.phase314b_r260_stagec_write_ahead"
    )


def test_folds_are_deterministic(arrays):
    left = stagec.deterministic_group_folds(arrays["pair_group"])
    right = stagec.deterministic_group_folds(arrays["pair_group"])
    assert left["assignment_sha256"] == right["assignment_sha256"]
    np.testing.assert_array_equal(left["group_fold_id"], right["group_fold_id"])
    np.testing.assert_array_equal(left["row_fold_id"], right["row_fold_id"])


def test_folds_are_balanced(folds):
    assert sum(folds["fold_group_counts"]) == 320
    assert max(folds["fold_group_counts"]) - min(folds["fold_group_counts"]) == 1


def test_folds_cover_all_ids(folds):
    assert set(folds["group_fold_id"].astype(int).tolist()) == set(range(6))
    assert set(folds["row_fold_id"].astype(int).tolist()) == set(range(6))


def test_pair_group_never_crosses_fold(arrays, folds):
    groups = arrays["pair_group"].astype(str)
    for group in set(groups.tolist()):
        assert len(set(folds["row_fold_id"][groups == group].tolist())) == 1


def test_wrong_group_population_blocks():
    with pytest.raises(stagec.StageCError):
        stagec.deterministic_group_folds(np.asarray(["a", "b"]))


@pytest.mark.parametrize("count", [1, 2, 3, 5, 10, 100])
def test_quantiles_finite(count):
    result = stagec._quantiles(np.arange(count, dtype=np.float64))
    assert all(np.isfinite(value) for value in result.values())
    assert result["min"] == 0.0
    assert result["max"] == float(count - 1)


def test_quantiles_reject_empty():
    with pytest.raises(stagec.StageCError):
        stagec._quantiles(np.asarray([], dtype=np.float64))


def test_quantiles_reject_nonfinite():
    with pytest.raises(stagec.StageCError):
        stagec._quantiles(np.asarray([0.0, np.nan]))


@pytest.mark.parametrize(
    "numerator,denominator,expected",
    [(1.0, 2.0, 0.5), (0.0, 0.0, 0.0), (2.0, -4.0, 0.5)],
)
def test_safe_relative(numerator, denominator, expected):
    assert stagec._safe_relative(numerator, denominator) == pytest.approx(expected)


def test_feature_normalization_population(arrays):
    result = stagec.feature_normalization(arrays)
    assert len(result) == 20
    for name in ("paper_x", "state_action_x", "y_state", "y_final_state", "y_action"):
        assert name + "_mean" in result
        assert name + "_std" in result
        assert name + "_scale" in result
        assert name + "_constant_mask" in result


@pytest.mark.parametrize(
    "name,shape",
    [
        ("paper_x", (261,)),
        ("state_action_x", (303,)),
        ("y_state", (4, 87)),
        ("y_final_state", (87,)),
        ("y_action", (14,)),
    ],
)
def test_feature_normalization_shapes(arrays, name, shape):
    result = stagec.feature_normalization(arrays)
    assert result[name + "_mean"].shape == shape
    assert result[name + "_std"].shape == shape
    assert result[name + "_scale"].shape == shape
    assert result[name + "_constant_mask"].shape == shape


def test_feature_scale_replaces_zero_std():
    arrays = {
        "paper_x": np.ones((2, 3), dtype=np.float32),
        "state_action_x": np.ones((2, 4), dtype=np.float32),
        "y_state": np.ones((2, 2, 3), dtype=np.float32),
        "y_final_state": np.ones((2, 3), dtype=np.float32),
        "y_action": np.ones((2, 1), dtype=np.float32),
    }
    result = stagec.feature_normalization(arrays)
    assert np.all(result["paper_x_scale"] == 1.0)
    assert np.all(result["paper_x_constant_mask"])


def test_group_means():
    names, values = stagec._group_means(
        np.asarray([1.0, 3.0, 2.0, 4.0]), np.asarray(["a", "a", "b", "b"])
    )
    assert names.tolist() == ["a", "b"]
    np.testing.assert_allclose(values, [2.0, 3.0])


def test_detectability_reference():
    result = stagec.detectability_reference(np.arange(1, 321, dtype=np.float64))
    assert result["group_count"] == 320
    assert result["mean_loss"] > 0
    assert result["normal_80_power_two_sided_mde_relative"] > 0
    assert "baseline-scale" in result["interpretation"]


def test_detectability_rejects_singleton():
    with pytest.raises(stagec.StageCError):
        stagec.detectability_reference(np.asarray([1.0]))


def test_semantic_statistics(arrays, folds):
    result = stagec.semantic_statistics(arrays, folds["row_fold_id"])
    assert result["row_count"] == 3006
    assert result["pair_group_count"] == 320
    assert result["visible_seed_count"] == 320
    assert result["fold_count"] == 6
    assert set(result["conditions"]) == set(stagec.EXPECTED_CONDITIONS)


@pytest.mark.parametrize("condition", stagec.EXPECTED_CONDITIONS)
def test_semantic_condition_stats(arrays, folds, condition):
    result = stagec.semantic_statistics(arrays, folds["row_fold_id"])
    row = result["conditions"][condition]
    assert row["row_count"] > 0
    assert row["pair_group_count"] == 320
    assert row["visible_seed_count"] == 320


def test_paired_branch_statistics(arrays):
    result = stagec.paired_branch_statistics(arrays)
    assert result["matched_pair_window_count"] >= 320
    assert result["paper_x_pair_mae"]["mean"] >= 0
    assert result["final_ordered_cable_mse"]["mean"] >= 0


def test_paired_branch_rejects_no_matches(arrays):
    broken = dict(arrays)
    broken["condition_name"] = np.asarray(["free"] * len(arrays["condition_name"]))
    with pytest.raises(stagec.StageCError):
        stagec.paired_branch_statistics(broken)


def test_analytic_baselines(arrays, folds):
    result = stagec.analytic_baselines(arrays, folds["row_fold_id"])
    assert set(result) == {"persistence", "constant_velocity"}
    for baseline in result.values():
        assert baseline["cable_mse"] > 0
        assert len(baseline["horizon_cable_mse"]) == 4
        assert len(baseline["fold_cable_mse"]) == 6
        assert baseline["detectability_reference"]["group_count"] == 320


@pytest.mark.parametrize("baseline", ["persistence", "constant_velocity"])
def test_baseline_condition_population(arrays, folds, baseline):
    result = stagec.analytic_baselines(arrays, folds["row_fold_id"])[baseline]
    assert set(result["condition_cable_mse"]) == set(stagec.EXPECTED_CONDITIONS)


def test_float_precision_audit(arrays):
    result = stagec.float_precision_audit(arrays)
    assert result["absolute_difference"] >= 0
    assert result["relative_difference"] >= 0
    assert result["recommended_future_candidate_numerical_margin"] >= 1.0e-8


def test_contract_arrays(arrays, folds):
    contract = stagec.build_contract_arrays(arrays, folds)
    stagec.validate_contract_arrays(contract)
    assert contract["row_fold_id"].shape == (3006,)
    assert contract["group_fold_id"].shape == (320,)


def test_contract_array_wrong_population_blocks(arrays, folds):
    contract = dict(stagec.build_contract_arrays(arrays, folds))
    contract.pop("paper_x_mean")
    with pytest.raises(stagec.StageCError):
        stagec.validate_contract_arrays(contract)


def test_contract_array_object_blocks(arrays, folds):
    contract = dict(stagec.build_contract_arrays(arrays, folds))
    contract["group_names"] = np.asarray([object()] * 320, dtype=object)
    with pytest.raises(stagec.StageCError):
        stagec.validate_contract_arrays(contract)


def test_deterministic_npz_payload(arrays, folds):
    contract = stagec.build_contract_arrays(arrays, folds)
    left = stagec._deterministic_npz_payload(contract)
    right = stagec._deterministic_npz_payload(contract)
    assert left == right
    assert stagec.sha256_bytes(left) == stagec.sha256_bytes(right)


def test_write_contract_once(tmp_path, arrays, folds):
    path = tmp_path / "contract.npz"
    contract = stagec.build_contract_arrays(arrays, folds)
    digest = stagec.write_contract_once(path, contract)
    assert digest == stagec.sha256_file(path)
    with np.load(str(path), allow_pickle=False) as loaded:
        assert set(loaded.files) == set(contract)


def test_contract_manifest(tmp_path, arrays, folds):
    contract = stagec.build_contract_arrays(arrays, folds)
    path = tmp_path / "contract.npz"
    digest = stagec.write_contract_once(path, contract)
    source = {
        "objective_npz_sha256": "a" * 64,
        "objective_manifest_self_sha256": "b" * 64,
    }
    manifest = stagec.build_contract_manifest(
        source=source,
        contract_arrays=contract,
        contract_file_sha256=digest,
        folds=folds,
    )
    stagec.validate_contract_manifest(manifest)
    assert manifest["contract_npz_sha256"] == digest


@pytest.mark.parametrize(
    "key,bad_value",
    [
        ("schema", "bad"),
        ("contract_npz", "bad.npz"),
        ("fold_count", 5),
        ("objective_train_open_count", 2),
        ("model_fit_count", 1),
        ("model_evaluation_count", 1),
    ],
)
def test_contract_manifest_rejects_changed_field(tmp_path, arrays, folds, key, bad_value):
    contract = stagec.build_contract_arrays(arrays, folds)
    path = tmp_path / "contract.npz"
    digest = stagec.write_contract_once(path, contract)
    manifest = stagec.build_contract_manifest(
        source={"objective_npz_sha256": "a" * 64, "objective_manifest_self_sha256": "b" * 64},
        contract_arrays=contract,
        contract_file_sha256=digest,
        folds=folds,
    )
    manifest[key] = bad_value
    with pytest.raises(stagec.StageCError):
        stagec.validate_contract_manifest(manifest)


def _minimal_worker(tmp_path, arrays, folds):
    contract = stagec.build_contract_arrays(arrays, folds)
    contract_path = tmp_path / stagec.CONTRACT_NPZ_RELATIVE
    contract_path.parent.mkdir(parents=True)
    contract_sha = stagec.write_contract_once(contract_path, contract)
    manifest = stagec.build_contract_manifest(
        source={"objective_npz_sha256": "a" * 64, "objective_manifest_self_sha256": "b" * 64},
        contract_arrays=contract,
        contract_file_sha256=contract_sha,
        folds=folds,
    )
    manifest_path = tmp_path / stagec.CONTRACT_MANIFEST_RELATIVE
    stagec.atomic_write_once(manifest_path, stagec.stable_json_bytes(manifest))
    worker = {
        "phase": stagec.PHASE,
        "schema": stagec.WORKER_SCHEMA,
        "execution_verdict": "PASS",
        "scientific_status": "READY",
        "source": {},
        "fold_contract": {
            "fold_count": 6,
            "fold_group_counts": folds["fold_group_counts"],
            "fold_assignment_sha256": folds["assignment_sha256"],
            "group_fold_population_sha256": stagec.sha256_array(folds["group_fold_id"]),
            "row_fold_population_sha256": stagec.sha256_array(folds["row_fold_id"]),
            "group_assignment": [
                {"pair_group": group, "fold": int(folds["group_to_fold"][group])}
                for group in folds["group_names"].astype(str).tolist()
            ],
            "paired_conditions_never_cross_folds": True,
        },
        "semantic_statistics": {},
        "paired_branch_statistics": {},
        "analytic_baselines": {},
        "float_precision_audit": {},
        "contract_artifact": {
            "contract_npz": stagec.CONTRACT_NPZ_RELATIVE,
            "contract_npz_sha256": contract_sha,
            "contract_manifest": stagec.CONTRACT_MANIFEST_RELATIVE,
            "contract_manifest_file_sha256": stagec.sha256_file(manifest_path),
            "contract_manifest_self_sha256": manifest["manifest_sha256"],
        },
        "checks": {"all": True},
        "execution_counts": {
            "objective_train_logical_open_count": 1,
            "objective_train_file_hash_count": 1,
            "objective_train_npz_parse_count": 1,
            "objective_train_role_manifest_read_count": 1,
            "objective_train_analytic_baseline_count": 2,
            "objective_train_model_fit_count": 0,
            "objective_train_model_evaluation_count": 0,
            "selection_holdout_npz_open_count": 0,
            "frozen_probe_npz_open_count": 0,
            "final_evaluation_npz_open_count": 0,
        },
        "selected_configuration": None,
        "train_only_recommendation": None,
        **{key: False for key in stagec.FALSE_BOUNDARIES},
    }
    worker["worker_result_sha256"] = stagec.sha256_bytes(stagec.stable_json_bytes(worker))
    return worker


def test_validate_worker_result(tmp_path, arrays, folds):
    worker = _minimal_worker(tmp_path, arrays, folds)
    stagec.validate_worker_result(worker, tmp_path)


@pytest.mark.parametrize("boundary", stagec.FALSE_BOUNDARIES)
def test_worker_rejects_forbidden_boundary(tmp_path, arrays, folds, boundary):
    worker = _minimal_worker(tmp_path, arrays, folds)
    worker[boundary] = True
    worker["worker_result_sha256"] = stagec.sha256_bytes(
        stagec.stable_json_bytes({k: v for k, v in worker.items() if k != "worker_result_sha256"})
    )
    with pytest.raises(stagec.StageCError):
        stagec.validate_worker_result(worker, tmp_path)


def test_build_summary(tmp_path, arrays, folds):
    worker = _minimal_worker(tmp_path, arrays, folds)
    summary = stagec.build_summary(
        repository={"head": "c" * 40},
        worker=worker,
        worker_file_sha256="d" * 64,
    )
    stagec.validate_summary(summary)
    assert summary["objective_train_opened"] is True
    assert summary["selection_holdout_opened"] is False


@pytest.mark.parametrize(
    "key,bad_value",
    [
        ("execution_verdict", "BLOCKED"),
        ("scientific_status", "BLOCKED"),
        ("objective_train_opened", False),
        ("objective_train_access_count_added", 2),
        ("selection_holdout_access_count_added", 1),
        ("frozen_probe_access_count_added", 1),
        ("final_evaluation_access_count_added", 1),
        ("selected_configuration", {}),
        ("train_only_recommendation", {}),
        ("rerun_authorized", True),
        ("resume_authorized", True),
    ],
)
def test_summary_rejects_changed_field(tmp_path, arrays, folds, key, bad_value):
    worker = _minimal_worker(tmp_path, arrays, folds)
    summary = stagec.build_summary(
        repository={"head": "c" * 40}, worker=worker, worker_file_sha256="d" * 64
    )
    summary[key] = bad_value
    with pytest.raises(stagec.StageCError):
        stagec.validate_summary(summary)


def test_effect_gate_policy_does_not_lock_threshold(tmp_path, arrays, folds):
    worker = _minimal_worker(tmp_path, arrays, folds)
    summary = stagec.build_summary(
        repository={"head": "c" * 40}, worker=worker, worker_file_sha256="d" * 64
    )
    assert summary["stage_d_effect_gate_policy"]["minimum_effect_threshold_locked_here"] is False
    assert summary["stage_d_effect_gate_policy"]["must_report_95_percent_group_confidence_interval"] is True


def test_blocked_report_before_worker():
    report = stagec.blocked_report(
        repository={"head": "x"}, error=RuntimeError("failed"), worker_started=False
    )
    assert report["execution_verdict"] == "BLOCKED"
    assert report["objective_train_access_count_added"] == 0
    assert report["rerun_authorized"] is False


def test_blocked_report_after_worker():
    report = stagec.blocked_report(
        repository={"head": "x"}, error=RuntimeError("failed"), worker_started=True
    )
    assert report["objective_train_access_count_added"] is None
    assert report["objective_train_open_attempt_started"] is True
    assert report["selection_holdout_access_count_added"] == 0


@pytest.mark.parametrize("relative", stagec.FORBIDDEN_ROLE_NPZ_RELATIVES)
def test_forbidden_role_paths_do_not_include_objective(relative):
    assert "objective_train" not in relative
    assert relative.endswith(".npz")


def test_source_text_does_not_load_forbidden_npz():
    source = Path(stagec.__file__).read_text(encoding="utf-8")
    assert "stagea.load_npz_strict(npz_path)" in source
    assert "stagea.validate_inventory" not in source
    assert "stagea.load_npz_strict(repo / FORBIDDEN" not in source
    assert "selection_holdout.npz\")" not in source.replace("FORBIDDEN_ROLE_NPZ_RELATIVES", "")


def test_implementation_paths_are_add_only():
    assert len(stagec.IMPLEMENTATION_PATHS) == 4
    assert all(status == "A" for status, _ in stagec.IMPLEMENTATION_PATHS)


@pytest.mark.parametrize("subject", [stagec.IMPLEMENTATION_SUBJECT, stagec.EVIDENCE_SUBJECT, stagec.BLOCKED_SUBJECT])
def test_commit_subjects_are_stagec(subject):
    assert "Stage C" in subject
    assert "r2.6.0" in subject
