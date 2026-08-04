#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"
out_dir="reports/experiment2/bootstrap"
mkdir -p "$out_dir"

sanitize_remote() {
  sed -E 's#(https?://)[^/@]+@#\1#; s#(ssh://)[^/@]+@#\1#'
}

{
  printf 'timestamp_utc=%s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  printf 'repository_root=%s\n' "$repo_root"
  printf 'branch=%s\n' "$(git branch --show-current)"
  printf 'head=%s\n' "$(git rev-parse HEAD)"
  git remote -v | sanitize_remote
  printf '%s\n' 'status_porcelain_v1:'
  git status --porcelain=v1
  printf '%s\n' 'recent_commits:'
  git log -5 --oneline --decorate
} > "$out_dir/git_state.txt"

{
  git submodule status --recursive
  git -C external/deformable-ravens remote -v | sanitize_remote
  git -C external/deformable-ravens rev-parse HEAD
  git -C external/deformable-ravens status --short
  git config -f .gitmodules --get-regexp '^submodule\..*\.(path|url|branch)$'
} > "$out_dir/submodule_state.txt"

{
  printf 'timestamp_utc=%s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  git --version
  python3 --version 2>&1 || python --version 2>&1 || true
  uname -a 2>&1 || true
  nvidia-smi 2>&1 || true
} > "$out_dir/environment_local.txt"

{
  sha256sum docs/CCDA_RESEARCH_ROADMAP.md
  wc -l docs/CCDA_RESEARCH_ROADMAP.md
} > "$out_dir/roadmap_state.txt"
