from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ccda_phase3.provenance_v2 import (
    compare_npz,
    merkle_root,
    resolve_manifest_artifact,
    sha256_array,
    strict_json_dump,
)


def test_merkle_root_is_deterministic_and_path_sensitive(tmp_path):
    (tmp_path / "a.txt").write_text("same")
    (tmp_path / "b.txt").write_text("same")
    first = merkle_root(tmp_path, ["a.txt", "b.txt"])
    second = merkle_root(tmp_path, ["b.txt", "a.txt"])
    assert first == second

    (tmp_path / "c.txt").write_text("same")
    third = merkle_root(tmp_path, ["a.txt", "c.txt"])
    assert third != first


def test_strict_json_rejects_nonfinite(tmp_path):
    with pytest.raises(ValueError):
        strict_json_dump(tmp_path / "bad.json", {"x": np.inf})


def test_strict_json_writes_valid_json(tmp_path):
    path = tmp_path / "ok.json"
    strict_json_dump(
        path,
        {"a": np.int64(3), "b": np.array([1.0, 2.0])},
    )
    assert json.loads(path.read_text()) == {
        "a": 3,
        "b": [1.0, 2.0],
    }


def test_resolve_manifest_artifact_is_relocatable(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    manifest = root / "manifest.json"
    manifest.write_text("{}")
    expected = root / "windows" / "x.npz"
    expected.parent.mkdir()
    expected.write_bytes(b"x")
    assert (
        resolve_manifest_artifact(manifest, "windows/x.npz")
        == expected.resolve()
    )


def test_resolve_manifest_artifact_rejects_absolute(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    with pytest.raises(ValueError):
        resolve_manifest_artifact(manifest, "/tmp/x.npz")


def test_resolve_manifest_artifact_rejects_escape(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    manifest = root / "manifest.json"
    manifest.write_text("{}")
    with pytest.raises(ValueError):
        resolve_manifest_artifact(manifest, "../outside.npz")


def test_compare_npz_reports_exact_and_changed_arrays(tmp_path):
    old = tmp_path / "old.npz"
    new = tmp_path / "new.npz"
    np.savez_compressed(
        old,
        x=np.array([[1.0, 2.0]], dtype=np.float32),
        name=np.array(["a"], dtype="<U1"),
    )
    np.savez_compressed(
        new,
        x=np.array([[1.0, 3.0]], dtype=np.float32),
        name=np.array(["a"], dtype="<U1"),
    )
    result = compare_npz(old, new)
    assert result["all_shapes_match"]
    assert result["all_dtypes_match"]
    assert not result["all_arrays_exact"]
    rows = {row["key"]: row for row in result["rows"]}
    assert rows["x"]["max_abs"] == 1.0
    assert rows["name"]["exact_equal"]


def test_sha256_array_includes_shape_and_dtype():
    a = np.array([1, 2], dtype=np.int32)
    b = np.array([[1, 2]], dtype=np.int32)
    c = np.array([1, 2], dtype=np.int64)
    assert sha256_array(a) != sha256_array(b)
    assert sha256_array(a) != sha256_array(c)
