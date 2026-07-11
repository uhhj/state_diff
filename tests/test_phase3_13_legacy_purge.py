import os
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/phase3_13_legacy_purge.py"


def run(cwd, *args, check=True, env=None):
    return subprocess.run(list(args), cwd=cwd, check=check, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)


def write(root, relative, text="x"):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def repository(tmp_path):
    root = tmp_path / "main"; root.mkdir()
    run(root, "git", "init", "-b", "Experiment1")
    run(root, "git", "config", "user.email", "test@example.com")
    run(root, "git", "config", "user.name", "test")
    required_main = [
        "ccda_phase3/schema_v2.py", "scripts/phase3_13_generate_raw.py",
        "scripts/phase3_13_build_windows.py", "scripts/phase3_13_audit_dataset.py",
        "scripts/phase3_13_runtime.py", "scripts/phase3_13_run.sh",
        "tests/test_ccda_slack_breakaway.py", "tests/test_ccda_state_v2.py",
        "tests/test_ccda_dataset_windows.py", "tests/test_phase3_13_legacy_purge.py",
    ]
    for relative in required_main: write(root, relative)
    write(root, "scripts/phase3_old.py")
    write(root, "reports/phase3_old.md")
    write(root, "GOALS.md")
    sub = root / "external/deformable-ravens"; sub.mkdir(parents=True)
    run(sub, "git", "init", "-b", "ccda-cable")
    run(sub, "git", "config", "user.email", "test@example.com")
    run(sub, "git", "config", "user.name", "test")
    write(sub, "ravens/tasks/ccda_slack_cable_v2.py")
    write(sub, "ravens/tasks/ccda_slack_breakaway.py")
    write(sub, "ccda_generate_hidden_contact.py")
    run(sub, "git", "add", "."); run(sub, "git", "commit", "-m", "sub")
    run(root, "git", "add", "."); run(root, "git", "commit", "-m", "main")
    write(root, "checkpoints/phase3_old/model.pt")
    return root, sub


def invoke(root, apply=False, confirmed=False):
    command = [sys.executable, str(SCRIPT), "--root", str(root)]
    if apply: command.append("--apply")
    env = os.environ.copy()
    if confirmed: env["PHASE313_LEGACY_PURGE_CONFIRM"] = "DELETE_PHASE3_LEGACY_ASSETS"
    return run(root, *command, check=False, env=env)


def test_dry_run_does_not_delete(tmp_path):
    root, sub = repository(tmp_path)
    result = invoke(root)
    assert result.returncode == 0
    assert (root / "scripts/phase3_old.py").exists()
    assert (root / "checkpoints/phase3_old/model.pt").exists()
    assert (sub / "ccda_generate_hidden_contact.py").exists()


def test_apply_requires_confirmation(tmp_path):
    root, _ = repository(tmp_path)
    result = invoke(root, apply=True)
    assert result.returncode != 0
    assert (root / "scripts/phase3_old.py").exists()


def test_confirmed_apply_uses_git_rm_and_preserves_history(tmp_path):
    root, sub = repository(tmp_path)
    before = run(root, "git", "rev-list", "--count", "HEAD").stdout.strip()
    result = invoke(root, apply=True, confirmed=True)
    assert result.returncode == 0, result.stdout
    assert not (root / "scripts/phase3_old.py").exists()
    assert not (root / "checkpoints/phase3_old").exists()
    assert not (sub / "ccda_generate_hidden_contact.py").exists()
    assert (root / ".git").is_dir()
    assert (root / "scripts/phase3_13_run.sh").exists()
    assert run(root, "git", "rev-list", "--count", "HEAD").stdout.strip() == before
    status = run(root, "git", "status", "--short").stdout
    assert "D  scripts/phase3_old.py" in status


def test_unknown_legacy_path_blocks(tmp_path):
    root, _ = repository(tmp_path)
    write(root, "reports/candidate_unknown.txt")
    result = invoke(root)
    assert result.returncode != 0
    assert "Unknown legacy-looking assets" in result.stdout
