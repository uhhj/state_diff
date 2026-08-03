from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from ccda_phase3 import phase314b_r259_stagei_fresh_eval_seal as s


def stageh_payload():
    payload = {
        "schema": "phase314b_r259_stageh_frozen_probe_transfer_failure_audit_v1",
        "execution_verdict": "PASS",
        "scientific_status": "BLOCKED",
        "root_cause": (
            "phase314b_r259_stageh_risk_probability_transfer_fails_beyond_"
            "prevalence_shift_while_raw_candidates_preserve_mean_improvement"
        ),
        "required_next_path": (
            "DESIGN_R259_RISK_TRANSFER_REPAIR_ON_OBJECTIVE_TRAIN_WITH_FRESH_"
            "UNTOUCHED_EVALUATION_SET"
        ),
        "selected_configuration": None,
        "train_only_recommendation": None,
        "cumulative_frozen_probe_evaluation_count": 1,
        "frozen_probe_evaluation_count_added": 0,
        "frozen_probe_reaccessed": False,
        "cumulative_selection_holdout_evaluation_count": 1,
        "selection_holdout_evaluation_count_added": 0,
        "selection_holdout_reaccessed": False,
        "rerun_authorized": False,
        "audit": {
            "classification": (
                "risk_probability_transfer_failure_beyond_base_rate_shift_with_"
                "raw_candidate_mean_improvement_preserved"
            ),
            "governance_conclusion": {
                "frozen_probe_access_consumed": True,
                "frozen_probe_reaccess_authorized": False,
                "future_final_evaluation_requires_fresh_untouched_data": True,
                "recipe_fallback_authorized": False,
                "retuning_against_frozen_probe_authorized": False,
                "timestep_cherry_pick_authorized": False,
            },
        },
    }
    payload["summary_sha256"] = s.sha256_bytes(s.stable_json_bytes(payload))
    return payload


def write_stageh(tmp_path, monkeypatch):
    payload = stageh_payload()
    path = tmp_path / "stageh.json"
    path.write_bytes(s.stable_json_bytes(payload))
    monkeypatch.setattr(s, "STAGE_H_REPORT_SHA256", s.sha256_file(path))
    monkeypatch.setattr(s, "STAGE_H_SELF_SHA256", payload["summary_sha256"])
    return path, payload


@pytest.mark.parametrize(
    "name,expected",
    [
        ("FRESH_VISIBLE_SEED_START", 430000),
        ("FRESH_VISIBLE_SEED_COUNT", 128),
        ("FRESH_VISIBLE_SEED_STOP", 430128),
        ("EXPECTED_EPISODE_COUNT", 256),
        ("EXPECTED_PAIR_GROUP_COUNT", 128),
        ("EXPECTED_STATE_DIM", 87),
        ("EXPECTED_PAPER_X_DIM", 261),
        ("EXPECTED_STATE_ACTION_X_DIM", 303),
        ("EXPECTED_FUTURE_HORIZON", 4),
        ("EXPECTED_ACTION_DIM", 14),
        ("EXPECTED_HZ", 480),
    ],
)
def test_frozen_constants(name, expected):
    assert getattr(s, name) == expected


def test_paths_are_under_unique_data_root():
    for value in (
        s.RAW_RELATIVE_ROOT,
        s.WINDOWS_RELATIVE_ROOT,
        s.WINDOW_NPZ_RELATIVE,
        s.ACTION_TEMPLATE_RELATIVE,
        s.WINDOW_MANIFEST_RELATIVE,
        s.INVENTORY_RELATIVE,
        s.SEAL_RELATIVE,
        s.ATTEMPT_MARKER_RELATIVE,
    ):
        assert value.startswith(s.DATASET_RELATIVE_ROOT + "/")


def test_implementation_is_three_add_only_files():
    assert len(s.IMPLEMENTATION_PATHS) == 3
    assert {status for status, _ in s.IMPLEMENTATION_PATHS} == {"A"}


def test_seed_contract_exact():
    value = s.validate_seed_contract()
    assert value["visible_seed_start"] == 430000
    assert value["visible_seed_stop_exclusive"] == 430128
    assert value["visible_seed_count"] == 128
    assert value["historical_seed_overlap"] is False


def test_expected_visible_seeds_are_contiguous():
    seeds = s.expected_visible_seeds()
    assert seeds[0] == 430000
    assert seeds[-1] == 430127
    assert all(right == left + 1 for left, right in zip(seeds, seeds[1:]))

@pytest.mark.parametrize(
    "start,stop",
    [(400000, 400256), (410000, 410064), (420000, 420128)],
)
def test_seed_range_does_not_overlap_historical(start, stop):
    assert not set(s.expected_visible_seeds()).intersection(range(start, stop))


def test_stable_json_order_and_format():
    left = s.stable_json_bytes({"b": 1, "a": 2})
    right = s.stable_json_bytes({"a": 2, "b": 1})
    assert left == right
    assert not left.endswith(b"\n")


def test_compact_json_has_no_spaces():
    assert s.compact_json_bytes({"a": 1}) == b'{"a":1}'


def test_sha256_known_value():
    assert s.sha256_bytes(b"abc") == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_array_hash_includes_shape():
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    b = a.reshape(3, 2)
    assert s.sha256_array(a) != s.sha256_array(b)


def test_array_hash_is_exact():
    a = np.arange(6, dtype=np.float32)
    assert s.sha256_array(a) == s.sha256_array(a.copy())


def test_atomic_write_once(tmp_path):
    path = tmp_path / "x.json"
    s.atomic_write_once(path, b"abc")
    assert path.read_bytes() == b"abc"


def test_atomic_write_once_refuses_existing(tmp_path):
    path = tmp_path / "x.json"
    path.write_bytes(b"old")
    with pytest.raises(s.StageIError, match="write-once"):
        s.atomic_write_once(path, b"new")


def test_load_json_requires_mapping(tmp_path):
    path = tmp_path / "x.json"
    path.write_text("[]")
    with pytest.raises(s.StageIError, match="not a mapping"):
        s.load_json(path)


def test_validate_stageh_report_accepts(tmp_path, monkeypatch):
    path, payload = write_stageh(tmp_path, monkeypatch)
    assert s.validate_stageh_report(path)["summary_sha256"] == payload["summary_sha256"]


@pytest.mark.parametrize(
    "key,value",
    [
        ("execution_verdict", "BLOCKED"),
        ("scientific_status", "READY"),
        ("selected_configuration", {"x": 1}),
        ("train_only_recommendation", {"x": 1}),
        ("frozen_probe_reaccessed", True),
        ("selection_holdout_reaccessed", True),
        ("rerun_authorized", True),
    ],
)
def test_validate_stageh_rejects_changed_fields(tmp_path, monkeypatch, key, value):
    path, payload = write_stageh(tmp_path, monkeypatch)
    payload[key] = value
    payload["summary_sha256"] = s.sha256_bytes(
        s.stable_json_bytes({k: v for k, v in payload.items() if k != "summary_sha256"})
    )
    path.write_bytes(s.stable_json_bytes(payload))
    monkeypatch.setattr(s, "STAGE_H_REPORT_SHA256", s.sha256_file(path))
    monkeypatch.setattr(s, "STAGE_H_SELF_SHA256", payload["summary_sha256"])
    with pytest.raises(s.StageIError, match="Stage-H report field"):
        s.validate_stageh_report(path)


@pytest.mark.parametrize(
    "key",
    [
        "frozen_probe_access_consumed",
        "future_final_evaluation_requires_fresh_untouched_data",
    ],
)
def test_validate_stageh_rejects_governance_true_to_false(tmp_path, monkeypatch, key):
    path, payload = write_stageh(tmp_path, monkeypatch)
    payload["audit"]["governance_conclusion"][key] = False
    payload["summary_sha256"] = s.sha256_bytes(
        s.stable_json_bytes({k: v for k, v in payload.items() if k != "summary_sha256"})
    )
    path.write_bytes(s.stable_json_bytes(payload))
    monkeypatch.setattr(s, "STAGE_H_REPORT_SHA256", s.sha256_file(path))
    monkeypatch.setattr(s, "STAGE_H_SELF_SHA256", payload["summary_sha256"])
    with pytest.raises(s.StageIError, match="governance"):
        s.validate_stageh_report(path)


@pytest.mark.parametrize(
    "key",
    [
        "frozen_probe_reaccess_authorized",
        "recipe_fallback_authorized",
        "retuning_against_frozen_probe_authorized",
        "timestep_cherry_pick_authorized",
    ],
)
def test_validate_stageh_rejects_governance_false_to_true(tmp_path, monkeypatch, key):
    path, payload = write_stageh(tmp_path, monkeypatch)
    payload["audit"]["governance_conclusion"][key] = True
    payload["summary_sha256"] = s.sha256_bytes(
        s.stable_json_bytes({k: v for k, v in payload.items() if k != "summary_sha256"})
    )
    path.write_bytes(s.stable_json_bytes(payload))
    monkeypatch.setattr(s, "STAGE_H_REPORT_SHA256", s.sha256_file(path))
    monkeypatch.setattr(s, "STAGE_H_SELF_SHA256", payload["summary_sha256"])
    with pytest.raises(s.StageIError, match="governance"):
        s.validate_stageh_report(path)


def raw_manifest(seed, condition, commit="impl", submodule="sub"):
    return {
        "environment_semantics_version": "ccda_hidden_slack_breakaway_v2",
        "observation_schema_version": "ccda_state_v2_position_proprio",
        "task": s.EXPECTED_TASK,
        "condition": condition,
        "visible_seed": seed,
        "pair_group": "phase313_test_seed_{}".format(seed),
        "split": "test",
        "action_source_condition": "free",
        "action_sequence_sha256": "action-{}".format(seed),
        "main_commit": commit,
        "submodule_commit": submodule,
        "contains_simulator_bead_velocity": False,
        "input_canonicalization": False,
        "num_actions": 3,
    }


def make_raw_tree(tmp_path, monkeypatch, seed_count=2):
    monkeypatch.setattr(s, "FRESH_VISIBLE_SEED_COUNT", seed_count)
    monkeypatch.setattr(s, "FRESH_VISIBLE_SEED_STOP", s.FRESH_VISIBLE_SEED_START + seed_count)
    monkeypatch.setattr(s, "EXPECTED_EPISODE_COUNT", 2 * seed_count)
    monkeypatch.setattr(s, "EXPECTED_PAIR_GROUP_COUNT", seed_count)
    root = tmp_path / "raw"
    for seed in s.expected_visible_seeds():
        for condition in s.EXPECTED_CONDITIONS:
            directory = root / "test" / condition
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "seed_{}.manifest.json".format(seed)).write_bytes(
                s.stable_json_bytes(raw_manifest(seed, condition))
            )
            with (directory / "seed_{}.pkl".format(seed)).open("wb") as handle:
                pickle.dump({"seed": seed, "condition": condition}, handle)
    return root


def test_audit_raw_dataset_accepts(tmp_path, monkeypatch):
    root = make_raw_tree(tmp_path, monkeypatch)
    audit = s.audit_raw_dataset(root, implementation_commit="impl", submodule_commit="sub")
    assert audit["episode_count"] == 4
    assert audit["paired_action_hashes_match"] is True


def test_audit_raw_rejects_action_mismatch(tmp_path, monkeypatch):
    root = make_raw_tree(tmp_path, monkeypatch)
    path = next((root / "test" / "hidden_slack_breakaway_pin_v2").glob("*.manifest.json"))
    payload = json.loads(path.read_text())
    payload["action_sequence_sha256"] = "different"
    path.write_bytes(s.stable_json_bytes(payload))
    with pytest.raises(s.StageIError, match="action sequence"):
        s.audit_raw_dataset(root, implementation_commit="impl", submodule_commit="sub")


def test_audit_raw_rejects_wrong_commit(tmp_path, monkeypatch):
    root = make_raw_tree(tmp_path, monkeypatch)
    path = next(root.glob("*/*/*.manifest.json"))
    payload = json.loads(path.read_text())
    payload["main_commit"] = "wrong"
    path.write_bytes(s.stable_json_bytes(payload))
    with pytest.raises(s.StageIError, match="main commit"):
        s.audit_raw_dataset(root, implementation_commit="impl", submodule_commit="sub")


def test_audit_raw_rejects_missing_episode(tmp_path, monkeypatch):
    root = make_raw_tree(tmp_path, monkeypatch)
    next(root.glob("*/*/*.pkl")).unlink()
    with pytest.raises(s.StageIError, match="episode count"):
        s.audit_raw_dataset(root, implementation_commit="impl", submodule_commit="sub")


def test_audit_raw_rejects_orphan_temp(tmp_path, monkeypatch):
    root = make_raw_tree(tmp_path, monkeypatch)
    (root / "orphan.tmp").write_bytes(b"x")
    with pytest.raises(s.StageIError, match="orphan"):
        s.audit_raw_dataset(root, implementation_commit="impl", submodule_commit="sub")


def fake_rows(seed_count=2):
    rows = []
    for seed in range(s.FRESH_VISIBLE_SEED_START, s.FRESH_VISIBLE_SEED_START + seed_count):
        for condition in s.EXPECTED_CONDITIONS:
            rows.append(
                {
                    "paper_x": np.zeros((261,), dtype=np.float32),
                    "state_action_x": np.zeros((303,), dtype=np.float32),
                    "y_state": np.zeros((4, 87), dtype=np.float32),
                    "y_final_state": np.zeros((87,), dtype=np.float32),
                    "y_action": np.zeros((14,), dtype=np.float32),
                    "condition_name": condition,
                    "visible_seed": seed,
                    "split_name": "test",
                    "source_file": "seed.pkl",
                    "pair_group": "phase313_test_seed_{}".format(seed),
                    "window_t": 0,
                    "success": False,
                    "final_fraction": 0.0,
                    "engagement_step": 1,
                    "release_step": 2,
                    "pre_engagement": True,
                }
            )
    return rows


def install_fake_window_modules(monkeypatch, rows):
    data_io = ModuleType("ccda_phase3.data_io")
    data_io.build_windows_from_dataset = lambda *args, **kwargs: (
        rows,
        SimpleNamespace(name="codec"),
        {"source": "fake"},
    )
    data_io.save_action_template = lambda path, codec: Path(path).write_bytes(b"template")
    schema = ModuleType("ccda_phase3.schema_v2")
    schema.ACTION_DIM = 14
    schema.DEFAULT_TF = 4
    schema.DEFAULT_TH = 3
    schema.FORMAL_CONDITIONS = s.EXPECTED_CONDITIONS
    schema.PAPER_X_DIM = 261
    schema.STATE_ACTION_X_DIM = 303
    schema.STATE_DIM = 87
    monkeypatch.setitem(sys.modules, "ccda_phase3.data_io", data_io)
    monkeypatch.setitem(sys.modules, "ccda_phase3.schema_v2", schema)


def test_build_fresh_windows(tmp_path, monkeypatch):
    monkeypatch.setattr(s, "DATASET_RELATIVE_ROOT", "data/fresh")
    monkeypatch.setattr(s, "RAW_RELATIVE_ROOT", "data/fresh/raw")
    monkeypatch.setattr(s, "WINDOWS_RELATIVE_ROOT", "data/fresh/windows")
    monkeypatch.setattr(s, "WINDOW_NPZ_RELATIVE", "data/fresh/windows/fresh_eval_windows.npz")
    monkeypatch.setattr(s, "ACTION_TEMPLATE_RELATIVE", "data/fresh/windows/action_template.pkl")
    monkeypatch.setattr(s, "WINDOW_MANIFEST_RELATIVE", "data/fresh/window_manifest.json")
    monkeypatch.setattr(s, "FRESH_VISIBLE_SEED_COUNT", 2)
    monkeypatch.setattr(s, "FRESH_VISIBLE_SEED_STOP", s.FRESH_VISIBLE_SEED_START + 2)
    monkeypatch.setattr(s, "EXPECTED_PAIR_GROUP_COUNT", 2)
    rows = fake_rows(2)
    install_fake_window_modules(monkeypatch, rows)
    manifest = s.build_fresh_windows(tmp_path)
    assert manifest["row_count"] == 4
    assert manifest["pair_group_count"] == 2
    assert (tmp_path / s.WINDOW_NPZ_RELATIVE).is_file()
    assert (tmp_path / s.WINDOW_MANIFEST_RELATIVE).is_file()


def test_build_windows_rejects_nan(tmp_path, monkeypatch):
    monkeypatch.setattr(s, "DATASET_RELATIVE_ROOT", "data/fresh")
    monkeypatch.setattr(s, "RAW_RELATIVE_ROOT", "data/fresh/raw")
    monkeypatch.setattr(s, "WINDOWS_RELATIVE_ROOT", "data/fresh/windows")
    monkeypatch.setattr(s, "WINDOW_NPZ_RELATIVE", "data/fresh/windows/fresh_eval_windows.npz")
    monkeypatch.setattr(s, "ACTION_TEMPLATE_RELATIVE", "data/fresh/windows/action_template.pkl")
    monkeypatch.setattr(s, "WINDOW_MANIFEST_RELATIVE", "data/fresh/window_manifest.json")
    monkeypatch.setattr(s, "FRESH_VISIBLE_SEED_COUNT", 2)
    monkeypatch.setattr(s, "FRESH_VISIBLE_SEED_STOP", s.FRESH_VISIBLE_SEED_START + 2)
    monkeypatch.setattr(s, "EXPECTED_PAIR_GROUP_COUNT", 2)
    rows = fake_rows(2)
    rows[0]["paper_x"][0] = np.nan
    install_fake_window_modules(monkeypatch, rows)
    with pytest.raises(s.StageIError, match="invalid fresh window"):
        s.build_fresh_windows(tmp_path)


def test_inventory_roundtrip(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    (root / "a.bin").write_bytes(b"a")
    (root / "b.bin").write_bytes(b"bb")
    inventory = s.build_inventory(root)
    s.validate_inventory(root, inventory)
    assert inventory["file_count"] == 2
    assert inventory["total_size_bytes"] == 3


def test_inventory_detects_mutation(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    path = root / "a.bin"
    path.write_bytes(b"a")
    inventory = s.build_inventory(root)
    path.write_bytes(b"changed")
    with pytest.raises(s.StageIError, match="size changed|SHA changed"):
        s.validate_inventory(root, inventory)


def test_inventory_excludes_seal_and_inventory_files(tmp_path, monkeypatch):
    root = tmp_path / "data"
    root.mkdir()
    (root / "a.bin").write_bytes(b"a")
    (root / "artifact_inventory.json").write_bytes(b"inventory")
    (root / "seal_manifest.json").write_bytes(b"seal")
    monkeypatch.setattr(s, "INVENTORY_RELATIVE", "data/artifact_inventory.json")
    monkeypatch.setattr(s, "SEAL_RELATIVE", "data/seal_manifest.json")
    inventory = s.build_inventory(root)
    assert [record["path"] for record in inventory["records"]] == ["a.bin"]


def minimal_seal():
    seal = {
        "schema": "phase314b_r259_stagei_fresh_evaluation_seal_v1",
        "dataset_role": "fresh_untouched_final_risk_transfer_evaluation",
        "dataset_relative_root": s.DATASET_RELATIVE_ROOT,
        "seed_contract": dict(s.validate_seed_contract()),
        "governance": {
            "generated_before_new_risk_repair_search": True,
            "targets_not_used_for_fit_selection_or_evaluation": True,
            "historical_selection_holdout_reaccessed": False,
            "historical_frozen_probe_reaccessed": False,
            "future_evaluation_count": 0,
            "future_evaluation_must_be_one_shot": True,
            "future_evaluation_requires_independently_locked_policy": True,
            "post_evaluation_retuning_forbidden": True,
            "fallback_after_evaluation_forbidden": True,
        },
    }
    seal["seal_sha256"] = s.sha256_bytes(s.stable_json_bytes(seal))
    return seal


def test_validate_seal_accepts():
    s.validate_seal(minimal_seal())


@pytest.mark.parametrize(
    "key,value",
    [
        ("generated_before_new_risk_repair_search", False),
        ("targets_not_used_for_fit_selection_or_evaluation", False),
        ("historical_selection_holdout_reaccessed", True),
        ("historical_frozen_probe_reaccessed", True),
        ("future_evaluation_count", 1),
        ("future_evaluation_must_be_one_shot", False),
        ("fallback_after_evaluation_forbidden", False),
    ],
)
def test_validate_seal_rejects_governance_changes(key, value):
    seal = minimal_seal()
    seal["governance"][key] = value
    seal["seal_sha256"] = s.sha256_bytes(
        s.stable_json_bytes({k: v for k, v in seal.items() if k != "seal_sha256"})
    )
    with pytest.raises(s.StageIError):
        s.validate_seal(seal)


def summary_payload():
    summary = {
        "schema": s.SCHEMA,
        "execution_verdict": "PASS",
        "scientific_status": "READY",
        "fresh_evaluation_seal": minimal_seal(),
        "selected_configuration": None,
        "train_only_recommendation": None,
        "selection_holdout_evaluation_count_added": 0,
        "selection_holdout_reaccessed": False,
        "frozen_probe_evaluation_count_added": 0,
        "frozen_probe_reaccessed": False,
        "fresh_evaluation_count_added": 0,
        "cumulative_fresh_evaluation_count": 0,
        "rerun_authorized": False,
    }
    for key in s.FALSE_BOUNDARIES:
        summary[key] = False
    summary["summary_sha256"] = s.sha256_bytes(s.stable_json_bytes(summary))
    return summary


def test_validate_summary_accepts():
    s.validate_summary(summary_payload())


@pytest.mark.parametrize(
    "field,value",
    [
        ("scientific_status", "BLOCKED"),
        ("selected_configuration", {"x": 1}),
        ("train_only_recommendation", {"x": 1}),
        ("selection_holdout_reaccessed", True),
        ("frozen_probe_reaccessed", True),
        ("fresh_evaluation_count_added", 1),
        ("rerun_authorized", True),
    ],
)
def test_validate_summary_rejects_changes(field, value):
    payload = summary_payload()
    payload[field] = value
    payload["summary_sha256"] = s.sha256_bytes(
        s.stable_json_bytes({k: v for k, v in payload.items() if k != "summary_sha256"})
    )
    with pytest.raises(s.StageIError):
        s.validate_summary(payload)


def test_validate_summary_rejects_forbidden_boundary():
    payload = summary_payload()
    payload["risk_fit_run"] = True
    payload["summary_sha256"] = s.sha256_bytes(
        s.stable_json_bytes({k: v for k, v in payload.items() if k != "summary_sha256"})
    )
    with pytest.raises(s.StageIError, match="forbidden"):
        s.validate_summary(payload)


def test_blocked_report_never_authorizes_retry(tmp_path):
    payload = s.blocked_report(
        repository=None,
        error=RuntimeError("boom"),
        dataset_root=tmp_path / "fresh",
    )
    assert payload["execution_verdict"] == "BLOCKED"
    assert payload["rerun_authorized"] is False
    assert payload["selected_configuration"] is None
    assert payload["train_only_recommendation"] is None
    assert payload["fresh_evaluation_count_added"] == 0
