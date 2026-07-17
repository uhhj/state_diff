from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from ccda_phase3 import phase314b_r258_staged_resume1_test_view as resume


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=str(root),
        text=True,
    ).strip()


def init_repo(
    root: Path,
    *,
    branch: str,
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
    return git(root, "rev-parse", "HEAD")


def build_clone_fixture(
    tmp_path: Path,
):
    submodule_source = (
        tmp_path / "submodule-source"
    )
    init_repo(
        submodule_source,
        branch="ccda-cable",
    )
    (
        submodule_source / "environment.py"
    ).write_text(
        "VALUE = 1\n",
        encoding="utf-8",
    )
    submodule_commit = commit_all(
        submodule_source,
        "submodule base",
    )

    main = tmp_path / "main"
    init_repo(
        main,
        branch="Experiment1",
    )
    for relative in (
        resume.EXPECTED_PHASE3_R2_TEST_PATHS
    ):
        path = main / relative
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        path.write_text(
            "def test_placeholder():\n"
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
            str(submodule_source),
            "external/deformable-ravens",
        ],
        cwd=str(main),
        check=True,
    )
    base_commit = commit_all(
        main,
        "base",
    )

    stage_d = main / resume.ORIGINAL_STAGE_D_TEST
    stage_d.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    stage_d.write_text(
        "def test_new():\n"
        "    assert True\n",
        encoding="utf-8",
    )
    current_commit = commit_all(
        main,
        "current",
    )
    return (
        main,
        base_commit,
        current_commit,
        submodule_commit,
    )


def test_phase_constant():
    assert resume.PHASE == (
        "Phase3.14b-r2.5.8 Stage D Resume1"
    )


def test_phase_id_constant():
    assert resume.PHASE_ID == (
        "phase314b_r258_staged_resume1"
    )


def test_base_commit_constant():
    assert resume.BASE_EVIDENCE_COMMIT == (
        "6866507c42b9bc9d2d271becd1a9423f61710405"
    )


def test_submodule_constant():
    assert resume.EXPECTED_SUBMODULE_COMMIT == (
        "633a88752445cf5d6776ed374fdbbdb35f93050c"
    )


def test_original_file_population():
    assert len(
        resume.ORIGINAL_IMPLEMENTATION_PATH_SHA256
    ) == 7


def test_resume1_file_population():
    assert len(
        resume.RESUME1_IMPLEMENTATION_PATHS
    ) == 6


def test_success_file_population():
    assert len(
        resume.RESUME1_SUCCESS_PATHS
    ) == 5


def test_original_blocked_sha_constant():
    assert resume.ORIGINAL_BLOCKED_SHA256 == (
        "c6176abe55cdbf63506765b5ec67745a6"
        "828a582d6eda70fd95b122e2e012bdc"
    )


def test_expected_counts():
    assert resume.EXPECTED_TOTAL_FILES == 53
    assert resume.EXPECTED_TOTAL_PASSED == (
        resume.EXPECTED_BASE_PASSED
        + resume.EXPECTED_STAGE_D_PASSED
        + resume.EXPECTED_RESUME1_PASSED
    )


def test_commit_messages_distinct():
    assert len(
        {
            resume.ORIGINAL_IMPLEMENTATION_MESSAGE,
            resume.ORIGINAL_BLOCKED_MESSAGE,
            resume.RESUME1_IMPLEMENTATION_MESSAGE,
            resume.RESUME1_EVIDENCE_MESSAGE,
        }
    ) == 4


def test_stable_json_deterministic():
    assert resume.stable_json_bytes(
        {"b": 2, "a": 1}
    ) == resume.stable_json_bytes(
        {"a": 1, "b": 2}
    )


def test_stable_json_rejects_nonfinite():
    with pytest.raises(ValueError):
        resume.stable_json_bytes(
            {"value": float("nan")}
        )


def test_atomic_write_once(tmp_path):
    path = tmp_path / "value.json"
    resume.atomic_write_once(
        path,
        b"{}\n",
    )
    assert path.read_bytes() == b"{}\n"
    with pytest.raises(FileExistsError):
        resume.atomic_write_once(
            path,
            b"{}\n",
        )


@pytest.mark.parametrize(
    "output,count",
    [
        ("1064 passed in 1.0s", 1064),
        ("84 passed in 0.2s", 84),
        ("45 passed in 0.1s", 45),
    ],
)
def test_parse_pytest_pass_count(
    output,
    count,
):
    assert resume.parse_pytest_pass_count(
        output
    ) == count


@pytest.mark.parametrize(
    "output",
    [
        "1 failed, 1063 passed in 1.0s",
        "1064 passed, 1 skipped in 1.0s",
        "1064 passed\n84 passed",
        "no summary",
    ],
)
def test_parse_pytest_rejects_invalid(
    output,
):
    with pytest.raises(ValueError):
        resume.parse_pytest_pass_count(
            output
        )


def test_discover_closed_world(tmp_path):
    for relative in (
        resume.EXPECTED_PHASE3_R2_TEST_PATHS
    ):
        path = tmp_path / relative
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        path.write_text(
            "",
            encoding="utf-8",
        )
    discovered = (
        resume.discover_phase3_r2_tests(
            tmp_path
        )
    )
    assert discovered == tuple(
        sorted(
            resume.EXPECTED_PHASE3_R2_TEST_PATHS
        )
    )


def test_closed_world_rejects_extra(tmp_path):
    for relative in (
        resume.EXPECTED_PHASE3_R2_TEST_PATHS
    ):
        path = tmp_path / relative
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        path.write_text(
            "",
            encoding="utf-8",
        )
    extra = (
        tmp_path
        / resume.ORIGINAL_STAGE_D_TEST
    )
    extra.write_text(
        "",
        encoding="utf-8",
    )
    with pytest.raises(
        resume.TestViewIsolationError
    ):
        resume.validate_closed_world_test_scope(
            tmp_path
        )


def test_closed_world_rejects_missing(tmp_path):
    for relative in (
        resume.EXPECTED_PHASE3_R2_TEST_PATHS[1:]
    ):
        path = tmp_path / relative
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        path.write_text(
            "",
            encoding="utf-8",
        )
    with pytest.raises(
        resume.TestViewIsolationError
    ):
        resume.validate_closed_world_test_scope(
            tmp_path
        )


def test_closed_world_record(tmp_path):
    for relative in (
        resume.EXPECTED_PHASE3_R2_TEST_PATHS
    ):
        path = tmp_path / relative
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        path.write_text(
            "",
            encoding="utf-8",
        )
    record = (
        resume.validate_closed_world_test_scope(
            tmp_path
        )
    )
    assert record["count"] == 27
    assert not record[
        "stage_d_test_visible"
    ]
    assert not record[
        "resume1_test_visible"
    ]


def test_status_paths_empty(tmp_path):
    init_repo(
        tmp_path,
        branch="Experiment1",
    )
    (
        tmp_path / "tracked.txt"
    ).write_text(
        "x\n",
        encoding="utf-8",
    )
    commit_all(
        tmp_path,
        "base",
    )
    assert resume.status_paths(
        tmp_path
    ) == ()


def test_status_paths_sorted(tmp_path):
    init_repo(
        tmp_path,
        branch="Experiment1",
    )
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
    assert resume.status_paths(
        tmp_path
    ) == (
        "a.txt",
        "b.txt",
    )


def test_commit_helpers(tmp_path):
    init_repo(
        tmp_path,
        branch="Experiment1",
    )
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
    assert resume.commit_subject(
        tmp_path,
        second,
    ) == "second"
    assert resume.commit_parent(
        tmp_path,
        second,
    ) == first
    assert resume.commit_paths(
        tmp_path,
        second,
    ) == ("b.txt",)


def test_assert_commit_shape_accepts(tmp_path):
    init_repo(
        tmp_path,
        branch="Experiment1",
    )
    (
        tmp_path / "a.txt"
    ).write_text(
        "a",
        encoding="utf-8",
    )
    parent = commit_all(
        tmp_path,
        "first",
    )
    (
        tmp_path / "b.txt"
    ).write_text(
        "b",
        encoding="utf-8",
    )
    child = commit_all(
        tmp_path,
        "second",
    )
    resume.assert_commit_shape(
        tmp_path,
        commit=child,
        parent=parent,
        subject="second",
        paths=("b.txt",),
    )


@pytest.mark.parametrize(
    "field",
    [
        "parent",
        "subject",
        "paths",
    ],
)
def test_assert_commit_shape_rejects(
    tmp_path,
    field,
):
    init_repo(
        tmp_path,
        branch="Experiment1",
    )
    (
        tmp_path / "a.txt"
    ).write_text(
        "a",
        encoding="utf-8",
    )
    parent = commit_all(
        tmp_path,
        "first",
    )
    (
        tmp_path / "b.txt"
    ).write_text(
        "b",
        encoding="utf-8",
    )
    child = commit_all(
        tmp_path,
        "second",
    )
    kwargs = {
        "commit": child,
        "parent": parent,
        "subject": "second",
        "paths": ("b.txt",),
    }
    if field == "parent":
        kwargs["parent"] = child
    elif field == "subject":
        kwargs["subject"] = "wrong"
    else:
        kwargs["paths"] = ("a.txt",)
    with pytest.raises(
        resume.TestViewIsolationError
    ):
        resume.assert_commit_shape(
            tmp_path,
            **kwargs,
        )


def test_validate_original_files(tmp_path):
    for relative, expected in (
        resume.ORIGINAL_IMPLEMENTATION_PATH_SHA256.items()
    ):
        path = tmp_path / relative
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        path.write_bytes(
            b"x"
        )
    original = dict(
        resume.ORIGINAL_IMPLEMENTATION_PATH_SHA256
    )
    try:
        for relative in original:
            resume.ORIGINAL_IMPLEMENTATION_PATH_SHA256[
                relative
            ] = resume.sha256_bytes(b"x")
        result = resume.validate_original_files(
            tmp_path
        )
        assert len(result) == 7
    finally:
        resume.ORIGINAL_IMPLEMENTATION_PATH_SHA256.clear()
        resume.ORIGINAL_IMPLEMENTATION_PATH_SHA256.update(
            original
        )


def test_validate_original_files_rejects_change(
    tmp_path,
):
    relative = next(
        iter(
            resume.ORIGINAL_IMPLEMENTATION_PATH_SHA256
        )
    )
    path = tmp_path / relative
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    path.write_bytes(b"wrong")
    with pytest.raises(
        resume.TestViewIsolationError
    ):
        resume.validate_original_files(
            tmp_path
        )


def test_validate_frozen_manifest(tmp_path):
    manifest = {}
    for index in range(
        resume.EXPECTED_BASE_TEST_FILES
    ):
        relative = (
            "tests/test_{:02d}.py".format(
                index
            )
        )
        path = tmp_path / relative
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        path.write_text(
            "x",
            encoding="utf-8",
        )
        manifest[relative] = (
            resume.sha256_file(path)
        )
    original_scope = (
        resume.validate_closed_world_test_scope
    )
    try:
        resume.validate_closed_world_test_scope = (
            lambda root: {
                "count": 27,
                "stage_d_test_visible": False,
                "resume1_test_visible": False,
            }
        )
        result = resume.validate_frozen_manifest(
            clone_root=tmp_path,
            manifest=manifest,
        )
        assert result[
            "manifest_count"
        ] == 51
    finally:
        resume.validate_closed_world_test_scope = (
            original_scope
        )


def test_validate_frozen_manifest_rejects_sha(
    tmp_path,
):
    manifest = {}
    for index in range(
        resume.EXPECTED_BASE_TEST_FILES
    ):
        relative = (
            "tests/test_{:02d}.py".format(
                index
            )
        )
        path = tmp_path / relative
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        path.write_text(
            "x",
            encoding="utf-8",
        )
        manifest[relative] = (
            resume.sha256_file(path)
        )
    first = next(iter(manifest))
    manifest[first] = "0" * 64
    with pytest.raises(
        resume.TestViewIsolationError
    ):
        resume.validate_frozen_manifest(
            clone_root=tmp_path,
            manifest=manifest,
        )


def test_create_frozen_clone_isolates_new_test(
    tmp_path,
):
    (
        source,
        base_commit,
        current_commit,
        submodule_commit,
    ) = build_clone_fixture(
        tmp_path
    )
    assert current_commit != base_commit
    destination = (
        tmp_path / "frozen-clone"
    )
    record = resume.create_frozen_clone(
        source_root=source,
        destination=destination,
        base_commit=base_commit,
        submodule_commit=submodule_commit,
    )
    assert record[
        "clone_head"
    ] == base_commit
    assert record[
        "clone_branch"
    ] == "Experiment1"
    assert record[
        "clone_origin_experiment1"
    ] == base_commit
    assert git(
        destination,
        "rev-parse",
        "origin/Experiment1",
    ) == base_commit
    assert not (
        destination
        / resume.ORIGINAL_STAGE_D_TEST
    ).exists()
    assert (
        destination
        / "external/deformable-ravens/environment.py"
    ).is_file()


def test_create_frozen_clone_does_not_mutate_source(
    tmp_path,
):
    (
        source,
        base_commit,
        current_commit,
        submodule_commit,
    ) = build_clone_fixture(
        tmp_path
    )
    before = git(
        source,
        "status",
        "--porcelain",
    )
    destination = (
        tmp_path / "frozen-clone"
    )
    resume.create_frozen_clone(
        source_root=source,
        destination=destination,
        base_commit=base_commit,
        submodule_commit=submodule_commit,
    )
    assert git(
        source,
        "rev-parse",
        "HEAD",
    ) == current_commit
    assert git(
        source,
        "status",
        "--porcelain",
    ) == before


def test_create_frozen_clone_rejects_existing(
    tmp_path,
):
    (
        source,
        base_commit,
        _,
        submodule_commit,
    ) = build_clone_fixture(
        tmp_path
    )
    destination = (
        tmp_path / "frozen-clone"
    )
    destination.mkdir()
    with pytest.raises(
        FileExistsError
    ):
        resume.create_frozen_clone(
            source_root=source,
            destination=destination,
            base_commit=base_commit,
            submodule_commit=submodule_commit,
        )


def test_clone_scope_exact(tmp_path):
    (
        source,
        base_commit,
        _,
        submodule_commit,
    ) = build_clone_fixture(
        tmp_path
    )
    destination = (
        tmp_path / "frozen-clone"
    )
    record = resume.create_frozen_clone(
        source_root=source,
        destination=destination,
        base_commit=base_commit,
        submodule_commit=submodule_commit,
    )
    scope = record[
        "closed_world_scope"
    ]
    assert scope["count"] == 27
    assert tuple(
        scope["discovered"]
    ) == tuple(
        sorted(
            resume.EXPECTED_PHASE3_R2_TEST_PATHS
        )
    )


def test_clone_declares_no_filtering(tmp_path):
    (
        source,
        base_commit,
        _,
        submodule_commit,
    ) = build_clone_fixture(
        tmp_path
    )
    destination = (
        tmp_path / "frozen-clone"
    )
    record = resume.create_frozen_clone(
        source_root=source,
        destination=destination,
        base_commit=base_commit,
        submodule_commit=submodule_commit,
    )
    assert not record["uses_network"]
    assert not record["uses_ignore"]
    assert not record[
        "uses_k_expression"
    ]
    assert not record[
        "uses_deselection"
    ]
    assert not record[
        "renames_current_tests"
    ]
    assert not record[
        "mutates_current_worktree"
    ]
