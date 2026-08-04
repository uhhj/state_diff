#!/usr/bin/env bash
set -euo pipefail

hostname
whoami
pwd
uname -a
git --version
python3 --version || true
nvidia-smi || true
df -h
git status --short --branch
git rev-parse HEAD
git submodule status --recursive
