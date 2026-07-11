from pathlib import Path

import numpy as np
import pytest

from phase3_12d_r22_integrity import (
    assert_disjoint_groups,
    binary_roc_auc,
    decode_text,
    group_bootstrap_indices,
    grouped_fold_indices,
    sha256_array,
    stable_episode_key,
    stable_window_hash,
    strict_json_dump,
)


def test_decode_text_handles_bytes_and_numpy_scalars():
    assert decode_text(b"free") == "free"
    assert decode_text(np.bytes_(b"hidden")) == "hidden"
    assert decode_text(np.asarray("free")) == "free"


def test_strict_json_rejects_nonfinite(tmp_path: Path):
    with pytest.raises(ValueError):
        strict_json_dump(tmp_path / "bad.json", {"x": np.nan})
    strict_json_dump(tmp_path / "ok.json", {"x": np.float32(1.0), "p": tmp_path})


def test_hashes_include_dtype_shape_and_all_blocks():
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    assert sha256_array(a) != sha256_array(a.astype(np.float64))
    assert sha256_array(a) != sha256_array(a.reshape(3, 2))
    assert stable_window_hash(a, a, a, a) != stable_window_hash(a + 1, a, a, a)


def test_stable_episode_key_is_explicit():
    assert stable_episode_key("free", 12, "x.pkl") == "free|12|x.pkl"


def test_group_bootstrap_resamples_whole_groups():
    groups = np.asarray([1, 1, 2, 2, 3, 3])
    for indices in group_bootstrap_indices(groups, 8, 7):
        for group in np.unique(groups[indices]):
            assert np.sum(groups[indices] == group) % np.sum(groups == group) == 0


def test_grouped_folds_never_overlap_seeds():
    groups = np.repeat(np.arange(10), 4)
    for train, test in grouped_fold_indices(groups, 5, 0):
        assert_disjoint_groups(groups[train], groups[test])


def test_horizon_zero_identical_controls_have_chance_auc():
    features = np.tile(np.arange(8, dtype=np.float32), (2, 1))
    labels = np.asarray([0, 1])
    scores = np.asarray([features[0].sum(), features[1].sum()])
    assert binary_roc_auc(labels, scores) == 0.5


def test_disjoint_group_guard_detects_leakage():
    with pytest.raises(AssertionError):
        assert_disjoint_groups([1, 2], [2, 3])
