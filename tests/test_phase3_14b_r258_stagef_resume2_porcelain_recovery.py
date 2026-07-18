from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from ccda_phase3 import phase314b_r258_stagef_resume2_porcelain_recovery as r2


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=str(root), text=True).strip()


def init_repo(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", "-b", "Experiment1"], cwd=str(root), check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=str(root), check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=str(root), check=True)


def commit_all(root: Path, message: str) -> str:
    subprocess.run(["git", "add", "-A"], cwd=str(root), check=True)
    subprocess.run(["git", "commit", "-q", "-m", message], cwd=str(root), check=True)
    return git(root, "rev-parse", "HEAD")


def test_phase_constants():
    assert r2.PHASE == "Phase3.14b-r2.5.8 Stage F Resume2"
    assert r2.PHASE_ID == "phase314b_r258_stagef_resume2"


def test_commit_constants():
    assert r2.BASE_EVIDENCE_COMMIT == "6758ea7ad800667a436b0243d3b1f6c63256d854"
    assert r2.ORIGINAL_IMPLEMENTATION_COMMIT == "519531f411c43b17a668c3c6c1a43b46a94f40a7"


def test_report_sha_constants():
    assert len(r2.STAGEF_TEST_GATE_SHA256) == 64
    assert len(r2.STAGEF_BLOCKED_SHA256) == 64
    assert len(r2.RESUME1_BLOCKED_SHA256) == 64


def test_resume1_file_population():
    assert len(r2.RESUME1_IMPLEMENTATION_PATHS) == 8


def test_resume2_file_population():
    assert len(r2.RESUME2_IMPLEMENTATION_PATHS) == 7


def test_success_population():
    assert len(r2.RESUME2_SUCCESS_PATHS) == 5


def test_porcelain_entry_status():
    entry = r2.PorcelainEntry(" ", "M", "a.py")
    assert entry.status == " M"


def test_parse_empty():
    assert r2.parse_porcelain_v1_z(b"") == ()


def test_parse_modified_preserves_first_character():
    result = r2.parse_porcelain_v1_z(b" M ccda_phase3/file.py\0")
    assert result[0].path == "ccda_phase3/file.py"
    assert result[0].status == " M"


def test_parse_staged_path():
    result = r2.parse_porcelain_v1_z(b"M  ccda_phase3/file.py\0")
    assert result[0].path == "ccda_phase3/file.py"
    assert result[0].status == "M "


def test_parse_untracked_path():
    result = r2.parse_porcelain_v1_z(b"?? reports/new.json\0")
    assert result[0].path == "reports/new.json"


def test_parse_multiple_records():
    result = r2.parse_porcelain_v1_z(b" M a.py\0?? b.py\0A  c.py\0")
    assert [entry.path for entry in result] == ["a.py", "b.py", "c.py"]


def test_parse_rename_record():
    result = r2.parse_porcelain_v1_z(b"R  new.py\0old.py\0")
    assert result[0].path == "new.py"
    assert result[0].original_path == "old.py"


def test_parse_copy_record():
    result = r2.parse_porcelain_v1_z(b"C  copy.py\0source.py\0")
    assert result[0].path == "copy.py"
    assert result[0].original_path == "source.py"


def test_parse_path_with_spaces():
    result = r2.parse_porcelain_v1_z(b"?? path with spaces.py\0")
    assert result[0].path == "path with spaces.py"


def test_parse_rejects_nonterminated():
    with pytest.raises(r2.StageFResume2Error):
        r2.parse_porcelain_v1_z(b"?? a.py")


def test_parse_rejects_short_record():
    with pytest.raises(r2.StageFResume2Error):
        r2.parse_porcelain_v1_z(b"??\0")


def test_parse_rejects_missing_separator():
    with pytest.raises(r2.StageFResume2Error):
        r2.parse_porcelain_v1_z(b"??xa.py\0")


def test_parse_rejects_empty_path():
    with pytest.raises(r2.StageFResume2Error):
        r2.parse_porcelain_v1_z(b"?? \0")


def test_parse_rejects_rename_without_source():
    with pytest.raises(r2.StageFResume2Error):
        r2.parse_porcelain_v1_z(b"R  new.py\0")


def test_status_paths_tracks_modified_file(tmp_path):
    init_repo(tmp_path)
    path = tmp_path / "ccda_phase3/file.py"
    path.parent.mkdir()
    path.write_text("a\n", encoding="utf-8")
    commit_all(tmp_path, "base")
    path.write_text("b\n", encoding="utf-8")
    assert r2.status_paths(tmp_path) == ("ccda_phase3/file.py",)


def test_status_paths_untracked_sorted(tmp_path):
    init_repo(tmp_path)
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    assert r2.status_paths(tmp_path) == ("a.txt", "b.txt")


def test_status_entries_preserve_status(tmp_path):
    init_repo(tmp_path)
    path = tmp_path / "a.txt"
    path.write_text("a", encoding="utf-8")
    commit_all(tmp_path, "base")
    path.write_text("b", encoding="utf-8")
    entry = r2.status_entries(tmp_path)[0]
    assert entry.worktree_status == "M"
    assert entry.path == "a.txt"


def test_status_paths_clean_repo(tmp_path):
    init_repo(tmp_path)
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    commit_all(tmp_path, "base")
    assert r2.status_paths(tmp_path) == ()


def test_expected_initial_status_contains_correct_ccda_prefix():
    paths = r2.expected_initial_status_paths()
    assert "ccda_phase3/phase314b_r258_stagef_constraint_aware_surrogate.py" in paths
    assert all(not path.startswith("cda_phase3/") for path in paths)


def test_expected_initial_status_count():
    assert len(r2.expected_initial_status_paths()) == 18


def test_stable_json_deterministic():
    assert r2.stable_json_bytes({"b": 2, "a": 1}) == r2.stable_json_bytes({"a": 1, "b": 2})


def test_atomic_write_once(tmp_path):
    path = tmp_path / "x.json"
    r2.atomic_write_once(path, b"{}\n")
    assert path.read_bytes() == b"{}\n"
    with pytest.raises(FileExistsError):
        r2.atomic_write_once(path, b"{}\n")


def test_assert_file_sha_map(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("a", encoding="utf-8")
    result = r2.assert_file_sha_map(tmp_path, {"a.txt": r2.sha256_file(path)})
    assert result["a.txt"] == r2.sha256_file(path)


def test_assert_file_sha_map_rejects(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("a", encoding="utf-8")
    with pytest.raises(r2.StageFResume2Error):
        r2.assert_file_sha_map(tmp_path, {"a.txt": "0" * 64})


def test_validate_report_sha(tmp_path):
    path = tmp_path / "x.json"
    path.write_text(json.dumps({"x": 1}), encoding="utf-8")
    assert r2.validate_report_sha(path, r2.sha256_file(path))["x"] == 1


def test_validate_report_sha_rejects(tmp_path):
    path = tmp_path / "x.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(r2.StageFResume2Error):
        r2.validate_report_sha(path, "0" * 64)


def test_commit_helpers(tmp_path):
    init_repo(tmp_path)
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    first = commit_all(tmp_path, "first")
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")
    second = commit_all(tmp_path, "second")
    assert r2.commit_parent(tmp_path, second) == first
    assert r2.commit_subject(tmp_path, second) == "second"
    assert r2.commit_paths(tmp_path, second) == ("b.txt",)


def test_assert_commit_shape_accepts(tmp_path):
    init_repo(tmp_path)
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    first = commit_all(tmp_path, "first")
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")
    second = commit_all(tmp_path, "second")
    r2.assert_commit_shape(tmp_path, commit=second, parent=first, subject="second", paths=("b.txt",))


@pytest.mark.parametrize("field", ("parent", "subject", "paths"))
def test_assert_commit_shape_rejects(tmp_path, field):
    init_repo(tmp_path)
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    first = commit_all(tmp_path, "first")
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")
    second = commit_all(tmp_path, "second")
    values = {"commit": second, "parent": first, "subject": "second", "paths": ("b.txt",)}
    if field == "parent":
        values["parent"] = second
    elif field == "subject":
        values["subject"] = "wrong"
    else:
        values["paths"] = ("a.txt",)
    with pytest.raises(r2.StageFResume2Error):
        r2.assert_commit_shape(tmp_path, **values)


def test_assert_commit_file_sha_map(tmp_path):
    init_repo(tmp_path)
    path = tmp_path / "a.txt"
    path.write_text("a", encoding="utf-8")
    commit = commit_all(tmp_path, "base")
    expected = {"a.txt": r2.sha256_bytes(b"a")}
    assert r2.assert_commit_file_sha_map(tmp_path, commit=commit, expected=expected) == expected


def test_assert_commit_file_sha_map_rejects(tmp_path):
    init_repo(tmp_path)
    path = tmp_path / "a.txt"
    path.write_text("a", encoding="utf-8")
    commit = commit_all(tmp_path, "base")
    with pytest.raises(r2.StageFResume2Error):
        r2.assert_commit_file_sha_map(tmp_path, commit=commit, expected={"a.txt": "0" * 64})


def test_git_output_strips_regular_scalar(tmp_path):
    init_repo(tmp_path)
    assert r2.git_output(tmp_path, "branch", "--show-current") == "Experiment1"


def test_resume2_messages_distinct():
    assert len({
        r2.STAGEF_BLOCKED_PROVENANCE_MESSAGE,
        r2.RESUME1_IMPLEMENTATION_MESSAGE,
        r2.RESUME1_BLOCKED_PROVENANCE_MESSAGE,
        r2.RESUME2_IMPLEMENTATION_MESSAGE,
        r2.RESUME2_EVIDENCE_MESSAGE,
    }) == 5


def test_parser_contract_metadata():
    entry = r2.PorcelainEntry("?", "?", "x")
    assert entry.status == "??"


def test_path_with_leading_dash(tmp_path):
    init_repo(tmp_path)
    (tmp_path / "-file.txt").write_text("x", encoding="utf-8")
    assert r2.status_paths(tmp_path) == ("-file.txt",)


def test_path_with_unicode(tmp_path):
    init_repo(tmp_path)
    (tmp_path / "测试.txt").write_text("x", encoding="utf-8")
    assert r2.status_paths(tmp_path) == ("测试.txt",)


def test_path_with_newline_is_preserved(tmp_path):
    init_repo(tmp_path)
    name = "line\nbreak.txt"
    (tmp_path / name).write_text("x", encoding="utf-8")
    assert r2.status_paths(tmp_path) == (name,)


def test_parse_rejects_nonbytes():
    with pytest.raises(TypeError):
        r2.parse_porcelain_v1_z("?? a\0")  # type: ignore[arg-type]


def test_resume2_blocked_path_is_distinct():
    assert r2.RESUME2_BLOCKED not in r2.RESUME2_SUCCESS_PATHS


def test_resume2_test_path_is_in_implementation_population():
    assert r2.RESUME2_TEST in r2.RESUME2_IMPLEMENTATION_PATHS
