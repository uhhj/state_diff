from __future__ import annotations

import json
import os
import tarfile
from pathlib import Path

import numpy as np
import pytest

from ccda_phase3 import phase314b_r260_stagea_data_universe_genesis as stagea


def _arrays(role: str, rows: int = 8):
    spec = stagea.ROLE_BY_NAME[role]
    seeds = np.asarray([spec.seed_start + index % spec.seed_count for index in range(rows)], dtype=np.int64)
    conditions = np.asarray([
        stagea.EXPECTED_CONDITIONS[index % 2] for index in range(rows)
    ], dtype="<U40")
    groups = np.asarray([
        "phase314b_r260_{}_seed_{}".format(role, seed) for seed in seeds
    ], dtype="<U80")
    return {
        "paper_x": np.zeros((rows, 261), dtype=np.float32),
        "state_action_x": np.zeros((rows, 303), dtype=np.float32),
        "y_state": np.zeros((rows, 4, 87), dtype=np.float32),
        "y_final_state": np.zeros((rows, 87), dtype=np.float32),
        "y_action": np.zeros((rows, 14), dtype=np.float32),
        "condition_name": conditions,
        "visible_seed": seeds,
        "split_name": np.asarray([role] * rows, dtype="<U40"),
        "source_file": np.asarray(["x"] * rows, dtype="<U1"),
        "pair_group": groups,
        "window_t": np.arange(rows, dtype=np.int64),
        "success": np.zeros(rows, dtype=np.bool_),
        "final_fraction": np.zeros(rows, dtype=np.float32),
        "engagement_step": np.full(rows, -1, dtype=np.int64),
        "release_step": np.full(rows, -1, dtype=np.int64),
        "pre_engagement": np.ones(rows, dtype=np.bool_),
    }


def _valid_seal():
    payload = {
        "schema": "phase314b_r260_stagea_data_universe_seal_v1",
        "dataset_schema": stagea.DATASET_SCHEMA,
        "dataset_relative_root": stagea.DATASET_RELATIVE_ROOT,
        "repository": {"root": "/x"},
        "source_lock_sha256": "a" * 64,
        "role_contract": stagea.role_contract(),
        "role_manifests": {},
        "audit": {},
        "inventory_sha256": "b" * 64,
        "inventory_file_count": 1,
        "inventory_total_size_bytes": 1,
        "exports": {"records": []},
        "governance": {
            "legacy_r255_r259_data_universe_abandoned": True,
            "legacy_cache_reconstruction_terminated": True,
            "new_role_seed_contract_frozen_before_generation": True,
            "objective_train_model_fit_count": 0,
            "selection_holdout_model_evaluation_count": 0,
            "frozen_probe_model_evaluation_count": 0,
            "final_evaluation_model_evaluation_count": 0,
            "selection_holdout_may_not_be_opened_before_policy_lock": True,
            "frozen_probe_may_not_be_opened_before_selection_lock": True,
            "final_evaluation_may_not_be_opened_before_final_preregistration": True,
            "role_regeneration_forbidden_after_terminal_report": True,
            "external_redundant_backup_required": True,
            "external_redundant_backup_attested": False,
        },
    }
    payload["seal_sha256"] = stagea.sha256_bytes(stagea.stable_json_bytes(payload))
    return payload


def test_phase_and_schema():
    assert stagea.PHASE == "Phase3.14b-r2.6.0 Stage A"
    assert stagea.SCHEMA.endswith("_v1")
    assert stagea.DATASET_SCHEMA.endswith("_v1")


def test_base_identities():
    assert len(stagea.BASE_HEAD) == 40
    assert len(stagea.BASE_IMPLEMENTATION) == 40
    assert len(stagea.EXPECTED_SUBMODULE) == 40
    assert len(stagea.STAGEJ_BLOCKED_REPORT_SHA256) == 64


def test_implementation_paths_are_add_only():
    assert len(stagea.IMPLEMENTATION_PATHS) == 4
    assert {status for status, _ in stagea.IMPLEMENTATION_PATHS} == {"A"}


def test_role_contract_totals():
    contract = stagea.role_contract()
    assert contract["total_visible_seeds"] == 640
    assert contract["total_pair_groups"] == 640
    assert contract["total_episodes"] == 1280
    assert contract["legacy_seed_overlap"] is False


@pytest.mark.parametrize(
    "role,start,count,governance",
    [
        ("objective_train", 900000, 320, "fit_and_nested_group_oof_only"),
        ("selection_holdout", 910000, 96, "one_shot_model_selection_only"),
        ("frozen_probe", 920000, 96, "one_shot_pre_final_probe_only"),
        ("final_evaluation", 930000, 128, "one_shot_final_evaluation_only"),
    ],
)
def test_role_specs(role, start, count, governance):
    spec = stagea.ROLE_BY_NAME[role]
    assert spec.seed_start == start
    assert spec.seed_count == count
    assert spec.seed_stop == start + count
    assert spec.governance_role == governance
    assert len(spec.seeds()) == count


@pytest.mark.parametrize("left", [spec.name for spec in stagea.ROLE_SPECS])
@pytest.mark.parametrize("right", [spec.name for spec in stagea.ROLE_SPECS])
def test_role_seed_disjointness(left, right):
    intersection = set(stagea.ROLE_BY_NAME[left].seeds()) & set(stagea.ROLE_BY_NAME[right].seeds())
    if left == right:
        assert len(intersection) == stagea.ROLE_BY_NAME[left].seed_count
    else:
        assert not intersection


@pytest.mark.parametrize("role", [spec.name for spec in stagea.ROLE_SPECS])
def test_roles_do_not_overlap_legacy_ranges(role):
    seeds = set(stagea.ROLE_BY_NAME[role].seeds())
    for legacy in (
        range(400000, 400256),
        range(410000, 410064),
        range(420000, 420128),
        range(430000, 430128),
    ):
        assert not seeds.intersection(legacy)


def test_stable_json_is_deterministic():
    left = stagea.stable_json_bytes({"b": 1, "a": 2})
    right = stagea.stable_json_bytes({"a": 2, "b": 1})
    assert left == right
    assert stagea.sha256_bytes(left) == stagea.sha256_bytes(right)


def test_sha256_array_includes_shape_and_dtype():
    base = np.arange(4, dtype=np.float32)
    assert stagea.sha256_array(base) != stagea.sha256_array(base.reshape(2, 2))
    assert stagea.sha256_array(base) != stagea.sha256_array(base.astype(np.float64))


def test_atomic_write_once(tmp_path: Path):
    path = tmp_path / "x.json"
    stagea.atomic_write_once(path, b"abc")
    assert path.read_bytes() == b"abc"
    with pytest.raises(stagea.StageAError, match="write-once"):
        stagea.atomic_write_once(path, b"def")


def test_deterministic_npz_byte_exact(tmp_path: Path):
    arrays = {
        "b": np.arange(8, dtype=np.int64),
        "a": np.arange(4, dtype=np.float32),
    }
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    assert stagea.write_npz_once(first, arrays) == stagea.write_npz_once(second, arrays)
    assert first.read_bytes() == second.read_bytes()
    loaded = stagea.load_npz_strict(first)
    assert set(loaded) == set(arrays)
    assert np.array_equal(loaded["a"], arrays["a"])


def test_deterministic_npz_rejects_object():
    with pytest.raises(stagea.StageAError, match="object dtype"):
        stagea._deterministic_npz_payload({"x": np.asarray([object()], dtype=object)})


@pytest.mark.parametrize("role", [spec.name for spec in stagea.ROLE_SPECS])
def test_validate_role_arrays_rejects_small_population(role):
    arrays = _arrays(role, rows=8)
    with pytest.raises(stagea.StageAError, match="implausibly small"):
        stagea.validate_role_arrays(role, arrays)


@pytest.mark.parametrize(
    "key,shape",
    [
        ("paper_x", (10, 260)),
        ("state_action_x", (10, 302)),
        ("y_state", (10, 4, 86)),
        ("y_final_state", (10, 86)),
        ("y_action", (10, 13)),
    ],
)
def test_validate_role_arrays_shape_failures(key, shape):
    role = "objective_train"
    rows = 640
    arrays = _arrays(role, rows=rows)
    arrays[key] = np.zeros(shape, dtype=np.float32)
    with pytest.raises(stagea.StageAError, match="shape/dtype"):
        stagea.validate_role_arrays(role, arrays)


def test_validate_role_arrays_rejects_nonfinite():
    role = "objective_train"
    arrays = _arrays(role, rows=640)
    arrays["paper_x"][0, 0] = np.nan
    with pytest.raises(stagea.StageAError, match="non-finite"):
        stagea.validate_role_arrays(role, arrays)


def test_validate_role_arrays_rejects_wrong_split():
    role = "objective_train"
    arrays = _arrays(role, rows=640)
    arrays["split_name"][:] = "wrong"
    with pytest.raises(stagea.StageAError, match="split label"):
        stagea.validate_role_arrays(role, arrays)


def test_validate_role_arrays_rejects_wrong_conditions():
    role = "objective_train"
    arrays = _arrays(role, rows=640)
    arrays["condition_name"][:] = "free"
    with pytest.raises(stagea.StageAError, match="condition coverage"):
        stagea.validate_role_arrays(role, arrays)


def test_validate_role_arrays_rejects_wrong_seed_coverage():
    role = "objective_train"
    arrays = _arrays(role, rows=640)
    arrays["visible_seed"][:] = 900000
    with pytest.raises(stagea.StageAError, match="seed coverage"):
        stagea.validate_role_arrays(role, arrays)


def test_pair_manifest_contract():
    manifest = {
        "dataset_schema": stagea.DATASET_SCHEMA,
        "environment_semantics_version": stagea.ENVIRONMENT_VERSION,
        "observation_schema_version": stagea.SCHEMA_VERSION,
        "task": stagea.FORMAL_TASK_NAME,
        "condition": "free",
        "visible_seed": 900000,
        "pair_group": "phase314b_r260_objective_train_seed_900000",
        "role": "objective_train",
        "action_source_condition": "free",
        "main_commit": "x" * 40,
        "submodule_commit": stagea.EXPECTED_SUBMODULE,
        "contains_simulator_bead_velocity": False,
        "input_canonicalization": False,
        "num_actions": 4,
    }
    stagea._strict_manifest(
        manifest,
        role="objective_train",
        seed=900000,
        implementation_commit="x" * 40,
    )
    manifest["input_canonicalization"] = True
    with pytest.raises(stagea.StageAError, match="input_canonicalization"):
        stagea._strict_manifest(
            manifest,
            role="objective_train",
            seed=900000,
            implementation_commit="x" * 40,
        )


def test_inventory_roundtrip(tmp_path: Path):
    root = tmp_path / "data"
    (root / "raw").mkdir(parents=True)
    (root / "raw/a").write_bytes(b"a")
    (root / "raw/b").write_bytes(b"b")
    inventory = stagea.build_inventory(root)
    stagea.validate_inventory(root, inventory)
    assert inventory["file_count"] == 2


def test_inventory_detects_mutation(tmp_path: Path):
    root = tmp_path / "data"
    root.mkdir()
    path = root / "a"
    path.write_bytes(b"a")
    inventory = stagea.build_inventory(root)
    path.write_bytes(b"b")
    with pytest.raises(stagea.StageAError, match="SHA"):
        stagea.validate_inventory(root, inventory)


def test_deterministic_tar_gz(tmp_path: Path):
    base = tmp_path / "base"
    (base / "x").mkdir(parents=True)
    (base / "x/a.txt").write_text("a", encoding="utf-8")
    (base / "x/b.txt").write_text("b", encoding="utf-8")
    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"
    first_sha = stagea.deterministic_tar_gz(
        output=first, base_root=base, relative_paths=[Path("x")]
    )
    second_sha = stagea.deterministic_tar_gz(
        output=second, base_root=base, relative_paths=[Path("x")]
    )
    assert first_sha == second_sha
    assert first.read_bytes() == second.read_bytes()
    with tarfile.open(first, "r:gz") as archive:
        assert archive.getnames() == ["x/a.txt", "x/b.txt"]
        assert all(member.mtime == 0 for member in archive.getmembers())


def test_deterministic_tar_rejects_missing_input(tmp_path: Path):
    with pytest.raises(stagea.StageAError, match="missing"):
        stagea.deterministic_tar_gz(
            output=tmp_path / "x.tar.gz",
            base_root=tmp_path,
            relative_paths=[Path("missing")],
        )


def test_validate_seal_accepts_valid_payload():
    stagea.validate_seal(_valid_seal())


@pytest.mark.parametrize(
    "key",
    [
        "legacy_r255_r259_data_universe_abandoned",
        "legacy_cache_reconstruction_terminated",
        "new_role_seed_contract_frozen_before_generation",
        "selection_holdout_may_not_be_opened_before_policy_lock",
        "frozen_probe_may_not_be_opened_before_selection_lock",
        "final_evaluation_may_not_be_opened_before_final_preregistration",
        "role_regeneration_forbidden_after_terminal_report",
        "external_redundant_backup_required",
    ],
)
def test_validate_seal_rejects_false_governance(key):
    payload = _valid_seal()
    payload["governance"][key] = False
    payload["seal_sha256"] = stagea.sha256_bytes(
        stagea.stable_json_bytes({k: v for k, v in payload.items() if k != "seal_sha256"})
    )
    with pytest.raises(stagea.StageAError, match="governance"):
        stagea.validate_seal(payload)


@pytest.mark.parametrize(
    "key",
    [
        "objective_train_model_fit_count",
        "selection_holdout_model_evaluation_count",
        "frozen_probe_model_evaluation_count",
        "final_evaluation_model_evaluation_count",
    ],
)
def test_validate_seal_rejects_model_access(key):
    payload = _valid_seal()
    payload["governance"][key] = 1
    payload["seal_sha256"] = stagea.sha256_bytes(
        stagea.stable_json_bytes({k: v for k, v in payload.items() if k != "seal_sha256"})
    )
    with pytest.raises(stagea.StageAError, match="access count"):
        stagea.validate_seal(payload)


def test_validate_seal_rejects_backup_attested():
    payload = _valid_seal()
    payload["governance"]["external_redundant_backup_attested"] = True
    payload["seal_sha256"] = stagea.sha256_bytes(
        stagea.stable_json_bytes({k: v for k, v in payload.items() if k != "seal_sha256"})
    )
    with pytest.raises(stagea.StageAError, match="incorrectly attested"):
        stagea.validate_seal(payload)


def test_build_summary_is_blocked_until_backup():
    summary = stagea.build_summary(
        repository={"root": "/x"},
        seal=_valid_seal(),
        legacy_write_ahead_before_sha256="a" * 64,
        legacy_write_ahead_after_sha256="a" * 64,
    )
    assert summary["execution_verdict"] == "PASS"
    assert summary["scientific_status"] == "BLOCKED"
    assert summary["external_backup_attested"] is False
    assert summary["selected_configuration"] is None
    assert summary["train_only_recommendation"] is None
    stagea.validate_summary(summary)


def test_build_summary_rejects_legacy_write_ahead_mutation():
    with pytest.raises(stagea.StageAError, match="write-ahead"):
        stagea.build_summary(
            repository={"root": "/x"},
            seal=_valid_seal(),
            legacy_write_ahead_before_sha256="a" * 64,
            legacy_write_ahead_after_sha256="b" * 64,
        )


@pytest.mark.parametrize("boundary", stagea.FALSE_BOUNDARIES)
def test_summary_false_boundaries(boundary):
    summary = stagea.build_summary(
        repository={"root": "/x"},
        seal=_valid_seal(),
        legacy_write_ahead_before_sha256="a" * 64,
        legacy_write_ahead_after_sha256="a" * 64,
    )
    assert summary[boundary] is False


def test_validate_summary_detects_boundary_crossing():
    summary = stagea.build_summary(
        repository={"root": "/x"},
        seal=_valid_seal(),
        legacy_write_ahead_before_sha256="a" * 64,
        legacy_write_ahead_after_sha256="a" * 64,
    )
    summary[stagea.FALSE_BOUNDARIES[0]] = True
    summary["summary_sha256"] = stagea.sha256_bytes(
        stagea.stable_json_bytes({k: v for k, v in summary.items() if k != "summary_sha256"})
    )
    with pytest.raises(stagea.StageAError, match="forbidden"):
        stagea.validate_summary(summary)


def test_blocked_report_is_fail_closed(tmp_path: Path):
    payload = stagea.blocked_report(
        repository=None,
        error=RuntimeError("x"),
        dataset_root=tmp_path / "data",
        write_ahead=tmp_path / "wa",
    )
    assert payload["execution_verdict"] == "BLOCKED"
    assert payload["scientific_status"] == "BLOCKED"
    assert payload["rerun_authorized"] is False
    assert payload["resume_authorized"] is False
    assert payload["selected_configuration"] is None


def test_source_lock_paths_include_simulator_semantics():
    assert "external/deformable-ravens/ravens/tasks/ccda_slack_cable_v2.py" in stagea.SOURCE_LOCK_PATHS
    assert "external/deformable-ravens/ravens/environment.py" in stagea.SOURCE_LOCK_PATHS
    assert "scripts/phase3_13_runtime.py" in stagea.SOURCE_LOCK_PATHS


def test_data_paths_are_new_epoch():
    assert "r260" in stagea.DATASET_RELATIVE_ROOT
    assert "r259" not in stagea.DATASET_RELATIVE_ROOT
    assert stagea.DATASET_RELATIVE_ROOT.startswith("data/")


def test_counts_and_dimensions():
    assert stagea.EXPECTED_STATE_DIM == 87
    assert stagea.EXPECTED_PAPER_X_DIM == 261
    assert stagea.EXPECTED_STATE_ACTION_X_DIM == 303
    assert stagea.EXPECTED_FUTURE_SHAPE == (4, 87)
    assert stagea.EXPECTED_ACTION_DIM == 14
    assert stagea.EXPECTED_HZ == 480


def _write_synthetic_pair(tmp_path: Path):
    directory = tmp_path / "seed_900000"
    directory.mkdir()
    implementation = "x" * 40
    vector = np.zeros((2, 14), dtype=np.float32)
    action_sha = __import__("hashlib").sha256(np.ascontiguousarray(vector).tobytes()).hexdigest()
    records = []
    for condition in stagea.EXPECTED_CONDITIONS:
        manifest = {
            "dataset_schema": stagea.DATASET_SCHEMA,
            "environment_semantics_version": stagea.ENVIRONMENT_VERSION,
            "observation_schema_version": stagea.SCHEMA_VERSION,
            "task": stagea.FORMAL_TASK_NAME,
            "condition": condition,
            "visible_seed": 900000,
            "pair_group": "phase314b_r260_objective_train_seed_900000",
            "role": "objective_train",
            "governance_role": "fit_and_nested_group_oof_only",
            "action_source_condition": "free",
            "action_sequence_sha256": action_sha,
            "main_commit": implementation,
            "submodule_commit": stagea.EXPECTED_SUBMODULE,
            "contains_simulator_bead_velocity": False,
            "input_canonicalization": False,
            "num_actions": 2,
            "settle_steps": 1,
        }
        manifest_path = directory / f"{condition}.manifest.json"
        manifest_path.write_bytes(stagea.stable_json_bytes(manifest))
        episode_path = directory / f"{condition}.pkl"
        import pickle
        with episode_path.open("wb") as handle:
            pickle.dump({"manifest": manifest, "action_vectors": vector}, handle)
        for path in (episode_path, manifest_path):
            records.append({
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": stagea.sha256_file(path),
            })
    receipt = {
        "schema": "phase314b_r260_stagea_pair_receipt_v1",
        "role": "objective_train",
        "visible_seed": 900000,
        "pair_group": "phase314b_r260_objective_train_seed_900000",
        "action_sequence_sha256": action_sha,
        "main_commit": implementation,
        "submodule_commit": stagea.EXPECTED_SUBMODULE,
        "files": sorted(records, key=lambda item: item["name"]),
    }
    receipt["receipt_sha256"] = stagea.sha256_bytes(stagea.stable_json_bytes(receipt))
    (directory / "pair_receipt.json").write_bytes(stagea.stable_json_bytes(receipt))
    return directory, implementation


def test_validate_pair_directory_roundtrip(tmp_path: Path):
    directory, implementation = _write_synthetic_pair(tmp_path)
    result = stagea.validate_pair_directory(
        directory,
        role="objective_train",
        seed=900000,
        implementation_commit=implementation,
    )
    assert result["visible_seed"] == 900000
    assert result["num_actions"] == 2


def test_validate_pair_directory_detects_episode_mutation(tmp_path: Path):
    directory, implementation = _write_synthetic_pair(tmp_path)
    path = directory / "free.pkl"
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(stagea.StageAError, match="receipt file size"):
        stagea.validate_pair_directory(
            directory,
            role="objective_train",
            seed=900000,
            implementation_commit=implementation,
        )
