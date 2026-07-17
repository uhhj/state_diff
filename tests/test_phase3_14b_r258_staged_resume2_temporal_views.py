from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from ccda_phase3 import (
    phase314b_r258_staged_resume2_temporal_views as resume2,
)


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=str(root),
        text=True,
    ).strip()


def init_repo(
    root: Path,
    *,
    branch: str = "Experiment1",
) -> None:
    root.mkdir(
        parents=True,
        exist_ok=True,
    )
    subprocess.run(
        [
            "git",
            "init",
            "-q",
            "-b",
            branch,
        ],
        cwd=str(root),
        check=True,
    )
    subprocess.run(
        [
            "git",
            "config",
            "user.email",
            "test@example.invalid",
        ],
        cwd=str(root),
        check=True,
    )
    subprocess.run(
        [
            "git",
            "config",
            "user.name",
            "Test",
        ],
        cwd=str(root),
        check=True,
    )


def commit_all(
    root: Path,
    message: str,
) -> str:
    subprocess.run(
        ["git", "add", "-A"],
        cwd=str(root),
        check=True,
    )
    subprocess.run(
        [
            "git",
            "commit",
            "-q",
            "-m",
            message,
        ],
        cwd=str(root),
        check=True,
    )
    return git(
        root,
        "rev-parse",
        "HEAD",
    )


def make_manifest() -> dict:
    result = {
        "tests/test_{:02d}.py".format(index):
            "{:064x}".format(index + 1)
        for index in range(50)
    }
    result[
        resume2.HISTORICAL_CLOSED_WORLD_TEST
    ] = "f" * 64
    return result


def build_temporal_fixture(
    tmp_path: Path,
):
    submodule = tmp_path / "submodule"
    init_repo(
        submodule,
        branch="ccda-cable",
    )
    (
        submodule / "environment.py"
    ).write_text(
        "VALUE = 1\n",
        encoding="utf-8",
    )
    submodule_commit = commit_all(
        submodule,
        "submodule",
    )

    main = tmp_path / "main"
    init_repo(main)
    (
        main / "tests"
    ).mkdir()
    (
        main / "tests/test_base.py"
    ).write_text(
        "def test_base():\n"
        "    assert True\n",
        encoding="utf-8",
    )
    subprocess.run(
        [
            "git",
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "add",
            "-q",
            str(submodule),
            "external/deformable-ravens",
        ],
        cwd=str(main),
        check=True,
    )
    first = commit_all(
        main,
        "first",
    )
    (
        main / "tests/test_later.py"
    ).write_text(
        "def test_later():\n"
        "    assert True\n",
        encoding="utf-8",
    )
    second = commit_all(
        main,
        "second",
    )
    return (
        main,
        first,
        second,
        submodule_commit,
    )


def build_scope_fixture(
    tmp_path: Path,
):
    root = tmp_path / "scope"
    package = root / "ccda_phase3"
    package.mkdir(parents=True)
    (
        package / "__init__.py"
    ).write_text(
        "",
        encoding="utf-8",
    )
    expected = (
        "tests/test_phase3_14b_r21_a.py",
        "tests/test_phase314b_r22_b.py",
    )
    module = (
        "from pathlib import Path\n"
        "PHASE3_R2_TEST_PATHS = {!r}\n"
        "def phase3_r2_test_manifest(root):\n"
        "    paths = set()\n"
        "    for pattern in (\n"
        "        'tests/test_phase3_14b_r2*.py',\n"
        "        'tests/test_phase314b_r2*.py',\n"
        "    ):\n"
        "        for path in Path(root).glob(pattern):\n"
        "            paths.add(path.relative_to(root).as_posix())\n"
        "    return tuple(sorted(paths))\n"
    ).format(expected)
    (
        package
        / "phase314b_r254_resume4_reproduction_audit.py"
    ).write_text(
        module,
        encoding="utf-8",
    )
    for relative in expected:
        path = root / relative
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        path.write_text(
            "",
            encoding="utf-8",
        )
    return root, expected


def test_phase_constant():
    assert resume2.PHASE == (
        "Phase3.14b-r2.5.8 Stage D Resume2"
    )


def test_phase_id_constant():
    assert resume2.PHASE_ID == (
        "phase314b_r258_staged_resume2"
    )


def test_initial_head_constant():
    assert resume2.EXPECTED_INITIAL_HEAD == (
        "8b692966d9f332ad2557c755c1b3657fc4a94943"
    )


def test_initial_remote_constant():
    assert resume2.EXPECTED_INITIAL_REMOTE == (
        "6866507c42b9bc9d2d271becd1a9423f61710405"
    )


def test_historical_commit_constant():
    assert (
        resume2.HISTORICAL_CLOSED_WORLD_COMMIT
        == (
            "3b30ad610bf35244839c2f78d5fed1921d98c6bc"
        )
    )


def test_historical_test_constant():
    assert (
        resume2.HISTORICAL_CLOSED_WORLD_TEST
        == (
            "tests/"
            "test_phase314b_r254_resume4_reproduction_audit.py"
        )
    )


def test_historical_pass_count_constant():
    assert (
        resume2.EXPECTED_HISTORICAL_TEST_PASSED
        == 8
    )


def test_regular_base_pass_count():
    assert (
        resume2.EXPECTED_BASE_REGULAR_PASSED
        == 1056
    )


def test_original_file_population():
    assert len(
        resume2.ORIGINAL_IMPLEMENTATION_PATHS
    ) == 7


def test_resume1_file_population():
    assert len(
        resume2.RESUME1_IMPLEMENTATION_PATHS
    ) == 6


def test_resume2_file_population():
    assert len(
        resume2.RESUME2_IMPLEMENTATION_PATHS
    ) == 6


def test_resume2_success_population():
    assert len(
        resume2.RESUME2_SUCCESS_PATHS
    ) == 5


def test_blocked_sha_constants():
    assert resume2.ORIGINAL_BLOCKED_SHA256 == (
        "c6176abe55cdbf63506765b5ec67745a6"
        "828a582d6eda70fd95b122e2e012bdc"
    )
    assert resume2.RESUME1_BLOCKED_SHA256 == (
        "f14373370a56a269722dbdb6a3474ef5b"
        "e4c9983d0cd44b61f56e1dd4d46ddeb"
    )


def test_commit_messages_distinct():
    assert len(
        {
            resume2.ORIGINAL_IMPLEMENTATION_MESSAGE,
            resume2.ORIGINAL_BLOCKED_MESSAGE,
            resume2.RESUME1_IMPLEMENTATION_MESSAGE,
            resume2.RESUME1_BLOCKED_MESSAGE,
            resume2.RESUME2_IMPLEMENTATION_MESSAGE,
            resume2.RESUME2_EVIDENCE_MESSAGE,
        }
    ) == 6


def test_stable_json_is_deterministic():
    assert resume2.stable_json_bytes(
        {"b": 2, "a": 1}
    ) == resume2.stable_json_bytes(
        {"a": 1, "b": 2}
    )


def test_stable_json_rejects_nan():
    with pytest.raises(ValueError):
        resume2.stable_json_bytes(
            {"x": float("nan")}
        )


def test_atomic_write_once(tmp_path):
    path = tmp_path / "value.json"
    resume2.atomic_write_once(
        path,
        b"{}\n",
    )
    assert path.read_bytes() == b"{}\n"
    with pytest.raises(FileExistsError):
        resume2.atomic_write_once(
            path,
            b"{}\n",
        )


@pytest.mark.parametrize(
    "output,count",
    [
        ("1056 passed in 1.0s", 1056),
        ("8 passed in 0.1s", 8),
        ("84 passed in 0.2s", 84),
        ("40 passed in 0.1s", 40),
    ],
)
def test_parse_pytest_pass_count(
    output,
    count,
):
    assert resume2.parse_pytest_pass_count(
        output
    ) == count


@pytest.mark.parametrize(
    "output",
    [
        "1 failed, 1055 passed in 1.0s",
        "8 passed, 1 skipped in 0.1s",
        "8 passed\n40 passed",
        "no summary",
    ],
)
def test_parse_pytest_rejects_invalid(
    output,
):
    with pytest.raises(ValueError):
        resume2.parse_pytest_pass_count(
            output
        )


def test_split_base_manifest():
    result = resume2.split_base_manifest(
        make_manifest()
    )
    assert result["regular_count"] == 50
    assert result["historical_count"] == 1
    assert (
        resume2.HISTORICAL_CLOSED_WORLD_TEST
        not in result["regular_manifest"]
    )
    assert (
        resume2.HISTORICAL_CLOSED_WORLD_TEST
        in result["historical_manifest"]
    )


def test_split_base_manifest_rejects_missing():
    manifest = make_manifest()
    del manifest[
        resume2.HISTORICAL_CLOSED_WORLD_TEST
    ]
    with pytest.raises(
        resume2.TemporalTestViewError
    ):
        resume2.split_base_manifest(
            manifest
        )


def test_split_base_manifest_rejects_count():
    manifest = make_manifest()
    manifest["tests/extra.py"] = "0" * 64
    with pytest.raises(
        resume2.TemporalTestViewError
    ):
        resume2.split_base_manifest(
            manifest
        )


def test_status_paths_sorted(tmp_path):
    init_repo(tmp_path)
    (
        tmp_path / "b.txt"
    ).write_text(
        "b",
        encoding="utf-8",
    )
    (
        tmp_path / "a.txt"
    ).write_text(
        "a",
        encoding="utf-8",
    )
    assert resume2.status_paths(
        tmp_path
    ) == (
        "a.txt",
        "b.txt",
    )


def test_commit_helpers(tmp_path):
    init_repo(tmp_path)
    (
        tmp_path / "a.txt"
    ).write_text(
        "a",
        encoding="utf-8",
    )
    first = commit_all(
        tmp_path,
        "first",
    )
    (
        tmp_path / "b.txt"
    ).write_text(
        "b",
        encoding="utf-8",
    )
    second = commit_all(
        tmp_path,
        "second",
    )
    assert resume2.commit_subject(
        tmp_path,
        second,
    ) == "second"
    assert resume2.commit_parent(
        tmp_path,
        second,
    ) == first
    assert resume2.commit_paths(
        tmp_path,
        second,
    ) == ("b.txt",)


def test_assert_commit_shape_accepts(tmp_path):
    init_repo(tmp_path)
    (
        tmp_path / "a.txt"
    ).write_text(
        "a",
        encoding="utf-8",
    )
    first = commit_all(
        tmp_path,
        "first",
    )
    (
        tmp_path / "b.txt"
    ).write_text(
        "b",
        encoding="utf-8",
    )
    second = commit_all(
        tmp_path,
        "second",
    )
    resume2.assert_commit_shape(
        tmp_path,
        commit=second,
        parent=first,
        subject="second",
        paths=("b.txt",),
    )


@pytest.mark.parametrize(
    "field",
    ("parent", "subject", "paths"),
)
def test_assert_commit_shape_rejects(
    tmp_path,
    field,
):
    init_repo(tmp_path)
    (
        tmp_path / "a.txt"
    ).write_text(
        "a",
        encoding="utf-8",
    )
    first = commit_all(
        tmp_path,
        "first",
    )
    (
        tmp_path / "b.txt"
    ).write_text(
        "b",
        encoding="utf-8",
    )
    second = commit_all(
        tmp_path,
        "second",
    )
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
    with pytest.raises(
        resume2.TemporalTestViewError
    ):
        resume2.assert_commit_shape(
            tmp_path,
            **values,
        )


def test_assert_file_sha_map(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text(
        "a",
        encoding="utf-8",
    )
    result = resume2.assert_file_sha_map(
        tmp_path,
        {
            "a.txt":
                resume2.sha256_file(path)
        },
    )
    assert result[
        "a.txt"
    ] == resume2.sha256_file(path)


def test_assert_file_sha_map_rejects(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text(
        "a",
        encoding="utf-8",
    )
    with pytest.raises(
        resume2.TemporalTestViewError
    ):
        resume2.assert_file_sha_map(
            tmp_path,
            {"a.txt": "0" * 64},
        )


def test_validate_blocked_report(tmp_path):
    path = tmp_path / "blocked.json"
    payload = {
        "phase": "X",
        "verdict": "BLOCKED",
        "scientific_status": "BLOCKED",
        "scientific_result_sealed": False,
        "push_completed": False,
        "selected_configuration": None,
        "train_only_recommendation": None,
    }
    path.write_text(
        json.dumps(
            payload,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    result = resume2.validate_blocked_report(
        path,
        expected_sha256=
            resume2.sha256_file(path),
        expected_phase="X",
    )
    assert result["verdict"] == "BLOCKED"


@pytest.mark.parametrize(
    "field,value",
    [
        ("verdict", "PASS"),
        ("scientific_status", "READY"),
        ("scientific_result_sealed", True),
        ("push_completed", True),
        ("selected_configuration", {}),
        ("train_only_recommendation", {}),
    ],
)
def test_validate_blocked_report_rejects(
    tmp_path,
    field,
    value,
):
    path = tmp_path / "blocked.json"
    payload = {
        "phase": "X",
        "verdict": "BLOCKED",
        "scientific_status": "BLOCKED",
        "scientific_result_sealed": False,
        "push_completed": False,
        "selected_configuration": None,
        "train_only_recommendation": None,
    }
    payload[field] = value
    path.write_text(
        json.dumps(
            payload,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    with pytest.raises(
        resume2.TemporalTestViewError
    ):
        resume2.validate_blocked_report(
            path,
            expected_sha256=
                resume2.sha256_file(path),
            expected_phase="X",
        )


def test_gitlink_commit(tmp_path):
    (
        main,
        first,
        _,
        submodule_commit,
    ) = build_temporal_fixture(
        tmp_path
    )
    assert resume2.gitlink_commit(
        main,
        first,
    ) == submodule_commit


def test_create_temporal_clone_first_commit(
    tmp_path,
):
    (
        main,
        first,
        second,
        submodule_commit,
    ) = build_temporal_fixture(
        tmp_path
    )
    assert first != second
    destination = tmp_path / "clone"
    record = resume2.create_temporal_clone(
        source_root=main,
        destination=destination,
        commit=first,
    )
    assert record["clone_head"] == first
    assert record[
        "clone_origin_experiment1"
    ] == first
    assert record[
        "clone_submodule_commit"
    ] == submodule_commit
    assert not (
        destination
        / "tests/test_later.py"
    ).exists()


def test_create_temporal_clone_second_commit(
    tmp_path,
):
    (
        main,
        _,
        second,
        _,
    ) = build_temporal_fixture(
        tmp_path
    )
    destination = tmp_path / "clone"
    record = resume2.create_temporal_clone(
        source_root=main,
        destination=destination,
        commit=second,
    )
    assert record["clone_head"] == second
    assert (
        destination
        / "tests/test_later.py"
    ).is_file()


def test_temporal_clone_does_not_mutate_source(
    tmp_path,
):
    (
        main,
        first,
        second,
        _,
    ) = build_temporal_fixture(
        tmp_path
    )
    before = git(
        main,
        "status",
        "--porcelain",
    )
    resume2.create_temporal_clone(
        source_root=main,
        destination=tmp_path / "clone",
        commit=first,
    )
    assert git(
        main,
        "rev-parse",
        "HEAD",
    ) == second
    assert git(
        main,
        "status",
        "--porcelain",
    ) == before


def test_temporal_clone_rejects_existing(
    tmp_path,
):
    (
        main,
        first,
        _,
        _,
    ) = build_temporal_fixture(
        tmp_path
    )
    destination = tmp_path / "clone"
    destination.mkdir()
    with pytest.raises(FileExistsError):
        resume2.create_temporal_clone(
            source_root=main,
            destination=destination,
            commit=first,
        )


def test_temporal_clone_declares_no_filters(
    tmp_path,
):
    (
        main,
        first,
        _,
        _,
    ) = build_temporal_fixture(
        tmp_path
    )
    record = resume2.create_temporal_clone(
        source_root=main,
        destination=tmp_path / "clone",
        commit=first,
    )
    for key in (
        "uses_network",
        "uses_ignore",
        "uses_k_expression",
        "uses_deselection",
        "renames_current_tests",
        "copies_or_patches_test_sources",
        "mutates_current_worktree",
    ):
        assert record[key] is False


def test_validate_manifest_files(tmp_path):
    manifest = {}
    for index in range(3):
        relative = (
            "tests/test_{}.py".format(index)
        )
        path = tmp_path / relative
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        path.write_text(
            str(index),
            encoding="utf-8",
        )
        manifest[relative] = (
            resume2.sha256_file(path)
        )
    result = resume2.validate_manifest_files(
        root=tmp_path,
        manifest=manifest,
    )
    assert result == manifest


def test_validate_manifest_files_rejects(tmp_path):
    path = tmp_path / "tests/test_a.py"
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    path.write_text(
        "a",
        encoding="utf-8",
    )
    with pytest.raises(
        resume2.TemporalTestViewError
    ):
        resume2.validate_manifest_files(
            root=tmp_path,
            manifest={
                "tests/test_a.py": "0" * 64
            },
        )


def test_inspect_historical_scope(tmp_path):
    root, expected = build_scope_fixture(
        tmp_path
    )
    original_count = (
        resume2.EXPECTED_HISTORICAL_SCOPE_COUNT
    )
    try:
        resume2.EXPECTED_HISTORICAL_SCOPE_COUNT = 2
        result = resume2.inspect_historical_scope(
            python_bin="python",
            clone_root=root,
        )
        assert result["exact"]
        assert tuple(
            result["observed"]
        ) == tuple(sorted(expected))
    finally:
        resume2.EXPECTED_HISTORICAL_SCOPE_COUNT = (
            original_count
        )


def test_inspect_historical_scope_rejects_extra(
    tmp_path,
):
    root, _ = build_scope_fixture(
        tmp_path
    )
    extra = (
        root
        / "tests/test_phase3_14b_r23_extra.py"
    )
    extra.write_text(
        "",
        encoding="utf-8",
    )
    original_count = (
        resume2.EXPECTED_HISTORICAL_SCOPE_COUNT
    )
    try:
        resume2.EXPECTED_HISTORICAL_SCOPE_COUNT = 2
        with pytest.raises(
            resume2.TemporalTestViewError
        ):
            resume2.inspect_historical_scope(
                python_bin="python",
                clone_root=root,
            )
    finally:
        resume2.EXPECTED_HISTORICAL_SCOPE_COUNT = (
            original_count
        )


def test_historical_blob_identity(tmp_path):
    init_repo(tmp_path)
    relative = (
        resume2.HISTORICAL_CLOSED_WORLD_TEST
    )
    path = tmp_path / relative
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    path.write_text(
        "same",
        encoding="utf-8",
    )
    module_path = (
        tmp_path
        / resume2.HISTORICAL_CLOSED_WORLD_MODULE
    )
    module_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    module_path.write_text(
        "MODULE = 1\n",
        encoding="utf-8",
    )
    first = commit_all(
        tmp_path,
        "first",
    )
    (
        tmp_path / "other.txt"
    ).write_text(
        "later",
        encoding="utf-8",
    )
    second = commit_all(
        tmp_path,
        "second",
    )
    old_base = resume2.BASE_EVIDENCE_COMMIT
    old_historical = (
        resume2.HISTORICAL_CLOSED_WORLD_COMMIT
    )
    try:
        resume2.BASE_EVIDENCE_COMMIT = second
        resume2.HISTORICAL_CLOSED_WORLD_COMMIT = first
        result = resume2.historical_test_blob_identity(
            tmp_path
        )
        assert result["byte_exact"]
    finally:
        resume2.BASE_EVIDENCE_COMMIT = old_base
        resume2.HISTORICAL_CLOSED_WORLD_COMMIT = (
            old_historical
        )


def test_historical_blob_identity_rejects_change(
    tmp_path,
):
    init_repo(tmp_path)
    relative = (
        resume2.HISTORICAL_CLOSED_WORLD_TEST
    )
    path = tmp_path / relative
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    path.write_text(
        "first",
        encoding="utf-8",
    )
    module_path = (
        tmp_path
        / resume2.HISTORICAL_CLOSED_WORLD_MODULE
    )
    module_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    module_path.write_text(
        "MODULE = 1\n",
        encoding="utf-8",
    )
    first = commit_all(
        tmp_path,
        "first",
    )
    path.write_text(
        "second",
        encoding="utf-8",
    )
    second = commit_all(
        tmp_path,
        "second",
    )
    old_base = resume2.BASE_EVIDENCE_COMMIT
    old_historical = (
        resume2.HISTORICAL_CLOSED_WORLD_COMMIT
    )
    try:
        resume2.BASE_EVIDENCE_COMMIT = second
        resume2.HISTORICAL_CLOSED_WORLD_COMMIT = first
        with pytest.raises(
            resume2.TemporalTestViewError
        ):
            resume2.historical_test_blob_identity(
                tmp_path
            )
    finally:
        resume2.BASE_EVIDENCE_COMMIT = old_base
        resume2.HISTORICAL_CLOSED_WORLD_COMMIT = (
            old_historical
        )
