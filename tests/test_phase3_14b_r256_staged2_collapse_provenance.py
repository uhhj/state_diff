from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import numpy as np
import pytest

from ccda_phase3.phase314b_r256_staged2_collapse_provenance import (
    CABLE_DIM,
    EXPECTED_TRAIN_FULL_ROWS,
    FUTURE_STEPS,
    N_BEADS,
    NOMINAL_SEGMENT_LENGTH,
    CollapseProvenanceError,
    EpisodeStore,
    ProvenanceAuditSpec,
    RawFrame,
    _event_category,
    classify_audit,
    compare_worker_results,
    extract_raw_frame,
    fit_xyz_reference,
    load_source_episode,
    paired_counterpart_audit,
    population_audit,
    segment_components,
    source_episode_timeline,
    stable_json_bytes,
    validate_contract_mapping,
)


def ordered_xyz(
    *,
    spacing: float = 0.014,
    vertical_segment: int = -1,
) -> np.ndarray:
    value = np.zeros((N_BEADS, 3), dtype=np.float64)
    for index in range(1, N_BEADS):
        delta = np.asarray([spacing, 0.0, 0.0])
        if index - 1 == vertical_segment:
            delta = np.asarray([0.0, 0.0, spacing])
        value[index] = value[index - 1] + delta
    return value


def raw_info(
    *,
    seed: int = 7,
    pair_group: str = "pair_seed_7",
    condition: str = "free",
    positions: np.ndarray = None,
):
    xyz = ordered_xyz() if positions is None else positions
    return {
        "extras": {
            "ccda_task": "ccda-slack-cable-v2",
            "hidden_condition": condition,
            "ccda_visible_seed": seed,
            "ccda_pair_group": pair_group,
            "bead_ids": list(range(100, 100 + N_BEADS)),
            "bead_positions": xyz.tolist(),
            "bead_orientations": [
                [0.0, 0.0, 0.0, 1.0]
                for _ in range(N_BEADS)
            ],
        }
    }


def write_episode(
    path: Path,
    *,
    condition: str = "free",
    frame_count: int = 6,
) -> str:
    frames = [
        raw_info(
            condition=condition,
            positions=ordered_xyz(
                vertical_segment=11 if index == 3 else -1
            ),
        )
        for index in range(frame_count)
    ]
    payload = {
        "infos": frames[:-1],
        "last_info": frames[-1],
        "actions": [{"primitive": "x"}] * (frame_count - 1),
        "manifest": {
            "visible_seed": 7,
            "pair_group": "pair_seed_7",
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=4)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def synthetic_raw(rows: int = 4):
    positions = np.zeros(
        (rows, FUTURE_STEPS, N_BEADS, 3),
        dtype=np.float64,
    )
    for row in range(rows):
        for horizon in range(FUTURE_STEPS):
            positions[row, horizon] = ordered_xyz()
    bead_ids = np.broadcast_to(
        np.arange(N_BEADS, dtype=np.int64),
        (rows, FUTURE_STEPS, N_BEADS),
    ).copy()
    return {
        "positions_xyz": positions,
        "bead_ids": bead_ids,
        "raw_info_index": np.broadcast_to(
            np.arange(1, FUTURE_STEPS + 1),
            (rows, FUTURE_STEPS),
        ).copy(),
        "source_file": np.asarray(
            [f"raw/train/free/seed_{row}.pkl" for row in range(rows)]
        ),
        "source_pickle_sha256": np.asarray(["a" * 64] * rows),
        "episode_index": np.arange(rows, dtype=np.int64),
        "window_contract_row_index": np.arange(
            rows,
            dtype=np.int64,
        ),
        "window_t": np.zeros(rows, dtype=np.int64),
        "condition_name": np.asarray(
            ["free", "hidden", "free", "hidden"][:rows]
        ),
        "visible_seed": np.arange(rows, dtype=np.int64),
        "pair_group": np.asarray(
            [f"g{row // 2}" for row in range(rows)]
        ),
        "pair_key": np.asarray(
            [f"p{row // 2}" for row in range(rows)]
        ),
        "episode_group_key": np.asarray(
            [f"eg{row // 2}" for row in range(rows)]
        ),
    }


def population_stub(
    *,
    projection_fraction: float,
    true_fraction: float,
    partial_fraction: float = 0.0,
    severe_count: int = 10,
):
    return {
        "threshold_counts": {
            "xy_ratio_below_0p25": severe_count,
        },
        "severe_event_fractions": {
            "projection_artifact": projection_fraction,
            "true_xyz_collapse": true_fraction,
            "partial_xyz_collapse": partial_fraction,
        },
    }


def mapping_stub(exact: bool = True):
    return {"mapping_exact": exact}


def reconstruction_stub(
    *,
    exact_rate: float = 1.0,
    order_stable: bool = True,
):
    return {
        "raw_xy_float32_exact_rate": exact_rate,
        "bead_id_order_stable_across_target_horizons":
            order_stable,
    }


def driver_stub(category: str):
    return {"reconstructed_event": {"category": category}}


def test_spec_validates():
    ProvenanceAuditSpec().validate()


def test_spec_rejects_invalid_xy_thresholds():
    with pytest.raises(ValueError):
        ProvenanceAuditSpec(
            severe_xy_ratio=0.8,
            moderate_xy_ratio=0.5,
        ).validate()


def test_raw_frame_validates():
    frame = extract_raw_frame(
        raw_info(),
        spec=ProvenanceAuditSpec(),
    )
    frame.validate(ProvenanceAuditSpec())
    assert frame.positions_xyz.shape == (24, 3)


def test_raw_frame_rejects_duplicate_ids():
    info = raw_info()
    info["extras"]["bead_ids"][1] = info["extras"]["bead_ids"][0]
    with pytest.raises(CollapseProvenanceError):
        extract_raw_frame(info, spec=ProvenanceAuditSpec())


def test_extract_raw_frame_preserves_xyz_and_metadata():
    frame = extract_raw_frame(
        raw_info(seed=12, pair_group="p", condition="hidden"),
        spec=ProvenanceAuditSpec(),
    )
    assert frame.visible_seed == 12
    assert frame.pair_group == "p"
    assert frame.condition == "hidden"
    assert frame.positions_xyz.shape[1] == 3


def test_load_source_episode_verifies_sha(tmp_path: Path):
    path = tmp_path / "episode.pkl"
    digest = write_episode(path)
    episode = load_source_episode(
        path,
        source_file="raw/train/free/episode.pkl",
        expected_sha256=digest,
        spec=ProvenanceAuditSpec(),
    )
    assert len(episode.frames) == 6
    assert episode.action_count == 5


def test_load_source_episode_rejects_sha_mismatch(tmp_path: Path):
    path = tmp_path / "episode.pkl"
    write_episode(path)
    with pytest.raises(CollapseProvenanceError):
        load_source_episode(
            path,
            source_file="episode.pkl",
            expected_sha256="0" * 64,
            spec=ProvenanceAuditSpec(),
        )


def test_episode_store_rejects_path_escape(tmp_path: Path):
    store = EpisodeStore(
        repository_root=tmp_path,
        formal_root=tmp_path / "formal",
        spec=ProvenanceAuditSpec(),
    )
    with pytest.raises(CollapseProvenanceError):
        store.get("../outside.pkl", "0" * 64)


def make_contract_arrays():
    rows = EXPECTED_TRAIN_FULL_ROWS
    target = np.zeros(
        (rows, FUTURE_STEPS, CABLE_DIM),
        dtype=np.float32,
    )
    condition = np.asarray(
        ["free" if row % 2 == 0 else "hidden" for row in range(rows)]
    )
    split = np.asarray(["train"] * rows)
    seed = np.arange(rows, dtype=np.int64)
    pair_group = np.asarray([f"g{row // 2}" for row in range(rows)])
    pair_key = np.asarray([f"p{row // 2}" for row in range(rows)])
    episode_group = np.asarray([f"e{row // 2}" for row in range(rows)])
    window_t = np.arange(rows, dtype=np.int64) % 5
    train = {
        "diffusion_target_cable": target,
        "future_valid_mask": np.ones(
            (rows, FUTURE_STEPS),
            dtype=np.bool_,
        ),
        "condition_name": condition,
        "split_name": split,
        "visible_seed": seed,
        "pair_group": pair_group,
        "pair_key": pair_key,
        "episode_group_key": episode_group,
        "window_t": window_t,
        "source_row_index": np.arange(rows, dtype=np.int64),
    }
    full = {
        "row_index": np.arange(rows, dtype=np.int64),
        "diffusion_target_cable": target.copy(),
        "future_valid_mask": train["future_valid_mask"].copy(),
        "condition_name": condition.copy(),
        "split_name": split.copy(),
        "visible_seed": seed.copy(),
        "pair_group": pair_group.copy(),
        "pair_key": pair_key.copy(),
        "episode_group_key": episode_group.copy(),
        "source_file": np.asarray(["x.pkl"] * rows),
        "source_pickle_sha256": np.asarray(["a" * 64] * rows),
        "episode_index": np.arange(rows, dtype=np.int64),
        "window_t": window_t.copy(),
    }
    return train, full


def test_contract_mapping_is_exact_and_labels_row_semantics():
    train, full = make_contract_arrays()
    result = validate_contract_mapping(train, full)
    assert result["mapping_exact"]
    assert "window contract" in result["source_row_index_semantics"]


def test_contract_mapping_rejects_target_mismatch():
    train, full = make_contract_arrays()
    full["diffusion_target_cable"][0, 0, 0] = 1.0
    with pytest.raises(CollapseProvenanceError):
        validate_contract_mapping(train, full)


def test_segment_components_horizontal_segment():
    raw = synthetic_raw(1)
    result = segment_components(raw["positions_xyz"])
    np.testing.assert_allclose(
        result["xy_over_xyz"],
        1.0,
        rtol=2.0e-6,
        atol=2.0e-6,
    )
    np.testing.assert_allclose(
        result["abs_dz_over_xyz"],
        0.0,
        atol=1.0e-12,
    )


def test_segment_components_vertical_projection_collapse():
    raw = synthetic_raw(1)
    raw["positions_xyz"][0, 0] = ordered_xyz(
        vertical_segment=11
    )
    result = segment_components(raw["positions_xyz"])
    assert result["xy_length"][0, 0, 11] == 0.0
    assert result["xyz_length"][0, 0, 11] > 0.0
    assert result["xy_over_xyz"][0, 0, 11] == 0.0


def test_segment_components_handles_zero_xyz_without_nan():
    value = np.zeros((1, 4, 24, 3), dtype=np.float64)
    result = segment_components(value)
    assert np.all(np.isfinite(result["xy_over_xyz"]))
    assert np.all(result["xy_over_xyz"] == 0.0)


def test_fit_xyz_reference_uses_fit_rows_only():
    raw = synthetic_raw(4)
    components = segment_components(raw["positions_xyz"])
    fit = np.asarray([True, True, False, False])
    reference = fit_xyz_reference(
        components["xyz_length"],
        fit,
    )
    assert reference["median"].shape == (4, 23)
    np.testing.assert_allclose(
        reference["median"],
        0.014,
        atol=1.0e-12,
    )


def test_event_category_projection_artifact():
    category = _event_category(
        0.01,
        1.0,
        ProvenanceAuditSpec(),
    )
    assert category == "xy_projection_artifact"


def test_event_category_true_xyz_collapse():
    category = _event_category(
        0.01,
        0.01,
        ProvenanceAuditSpec(),
    )
    assert category == "true_xyz_collapse"


def test_population_audit_counts_projection_event():
    raw = synthetic_raw(2)
    raw["positions_xyz"][0, 0] = ordered_xyz(
        vertical_segment=11
    )
    components = segment_components(raw["positions_xyz"])
    xy_reference = np.full((4, 23), 0.014)
    xyz_reference = np.full((4, 23), 0.014)
    result = population_audit(
        name="x",
        population_mask=np.asarray([True, True]),
        raw=raw,
        components=components,
        xy_reference=xy_reference,
        xyz_reference=xyz_reference,
        spec=ProvenanceAuditSpec(),
    )
    assert (
        result["threshold_counts"][
            "xy_projection_artifact_events"
        ]
        >= 1
    )
    assert result["threshold_counts"]["true_xyz_collapse_events"] == 0


def test_paired_counterpart_audit_detects_one_condition_only():
    raw = synthetic_raw(4)
    raw["positions_xyz"][0, 0] = ordered_xyz(
        vertical_segment=11
    )
    components = segment_components(raw["positions_xyz"])
    result = paired_counterpart_audit(
        raw=raw,
        components=components,
        xy_reference=np.full((4, 23), 0.014),
        xyz_reference=np.full((4, 23), 0.014),
        spec=ProvenanceAuditSpec(),
    )
    assert result["counterpart_not_severe_event_instances"] >= 1


def test_source_episode_timeline_marks_driver_frame(tmp_path: Path):
    path = tmp_path / "episode.pkl"
    digest = write_episode(path)
    episode = load_source_episode(
        path,
        source_file="episode.pkl",
        expected_sha256=digest,
        spec=ProvenanceAuditSpec(),
    )
    timeline = source_episode_timeline(
        episode=episode,
        segment_index=11,
        driver_info_index=3,
        xyz_reference=np.full((4, 23), 0.014),
        xy_reference=np.full((4, 23), 0.014),
        neighbor_radius=1,
    )
    assert timeline["frames"][3]["is_driver_frame"]
    assert timeline["neighbor_segments"] == [10, 11, 12]


def test_classification_detects_mapping_failure():
    result = classify_audit(
        mapping_audit=mapping_stub(False),
        reconstruction_audit=reconstruction_stub(),
        calibration_population=population_stub(
            projection_fraction=1.0,
            true_fraction=0.0,
        ),
        driver_verification=driver_stub(
            "xy_projection_artifact"
        ),
        spec=ProvenanceAuditSpec(),
    )
    assert result["primary_failure_locus"] == (
        "window_provenance_mapping"
    )


def test_classification_detects_cache_mismatch():
    result = classify_audit(
        mapping_audit=mapping_stub(),
        reconstruction_audit=reconstruction_stub(
            exact_rate=0.99
        ),
        calibration_population=population_stub(
            projection_fraction=1.0,
            true_fraction=0.0,
        ),
        driver_verification=driver_stub(
            "xy_projection_artifact"
        ),
        spec=ProvenanceAuditSpec(),
    )
    assert result["primary_failure_locus"] == "cache_extraction"


def test_classification_detects_bead_order_failure():
    result = classify_audit(
        mapping_audit=mapping_stub(),
        reconstruction_audit=reconstruction_stub(
            order_stable=False
        ),
        calibration_population=population_stub(
            projection_fraction=1.0,
            true_fraction=0.0,
        ),
        driver_verification=driver_stub(
            "xy_projection_artifact"
        ),
        spec=ProvenanceAuditSpec(),
    )
    assert result["primary_failure_locus"] == "bead_order"


def test_classification_confirms_projection_contract():
    result = classify_audit(
        mapping_audit=mapping_stub(),
        reconstruction_audit=reconstruction_stub(),
        calibration_population=population_stub(
            projection_fraction=0.95,
            true_fraction=0.0,
            partial_fraction=0.05,
        ),
        driver_verification=driver_stub(
            "xy_projection_artifact"
        ),
        spec=ProvenanceAuditSpec(),
    )
    assert result["primary_failure_locus"] == (
        "xy_projection_contract"
    )


def test_classification_detects_true_xyz_collapse():
    result = classify_audit(
        mapping_audit=mapping_stub(),
        reconstruction_audit=reconstruction_stub(),
        calibration_population=population_stub(
            projection_fraction=0.5,
            true_fraction=0.2,
            partial_fraction=0.3,
        ),
        driver_verification=driver_stub(
            "true_xyz_collapse"
        ),
        spec=ProvenanceAuditSpec(),
    )
    assert result["primary_failure_locus"] == (
        "source_xyz_collapse"
    )


def test_classification_detects_mixed_geometry():
    result = classify_audit(
        mapping_audit=mapping_stub(),
        reconstruction_audit=reconstruction_stub(),
        calibration_population=population_stub(
            projection_fraction=0.8,
            true_fraction=0.05,
            partial_fraction=0.15,
        ),
        driver_verification=driver_stub(
            "xy_projection_artifact"
        ),
        spec=ProvenanceAuditSpec(),
    )
    assert result["primary_failure_locus"] == (
        "mixed_source_geometry"
    )


def test_compare_worker_results_is_exact():
    base = {
        "root_cause": "r",
        "required_next_path": "n",
        "source_logic_audit": {"x": 1},
        "contract_mapping_audit": {"x": 1},
        "raw_reconstruction_audit": {"x": 1},
        "split": {"x": 1},
        "geometry_references": {"x": 1},
        "population_geometry_audit": {"x": 1},
        "stage_d1_driver_verification": {"x": 1},
        "stage_d1_driver_source_timeline": {"x": 1},
        "paired_counterpart_audit": {"x": 1},
        "classification": {"x": 1},
    }
    assert compare_worker_results(base, dict(base))["exact"]


def test_stable_json_bytes_is_deterministic():
    left = stable_json_bytes({"b": 2, "a": 1})
    right = stable_json_bytes({"a": 1, "b": 2})
    assert left == right


def test_nominal_segment_length_matches_source_formula():
    expected = 2.0 * 0.005 * np.sqrt(2.0)
    assert NOMINAL_SEGMENT_LENGTH == pytest.approx(expected)
