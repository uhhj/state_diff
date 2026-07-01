#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ENV_NAME:-defravens37}"
DEFRAVENS_URL="${DEFRAVENS_URL:-https://github.com/DanielTakeshi/deformable-ravens.git}"
DEFRAVENS_ROOT="${DEFRAVENS_ROOT:-$ROOT/external/deformable-ravens}"
INSTALL_SYSTEM_DEPS="${INSTALL_SYSTEM_DEPS:-0}"

echo "[Phase0] repo root: $ROOT"
cd "$ROOT"

mkdir -p "$ROOT/external" "$ROOT/reports"

echo "[Phase0] freezing current StateDiff Python environment if possible..."
{
  echo "timestamp: $(date -Iseconds)"
  echo "pwd: $(pwd)"
  echo "git_branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
  echo "git_commit: $(git rev-parse HEAD 2>/dev/null || true)"
  echo ""
  echo "python:"
  command -v python || true
  python --version 2>&1 || true
  echo ""
  echo "pip_freeze:"
  python -m pip freeze 2>/dev/null || true
} > "$ROOT/reports/phase0_state_diff_env_freeze.txt"

if [[ "$INSTALL_SYSTEM_DEPS" == "1" ]]; then
  echo "[Phase0] installing Ubuntu system dependencies..."
  sudo apt-get update
  sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    xvfb
else
  echo "[Phase0] skipping apt packages. To install them, rerun:"
  echo "INSTALL_SYSTEM_DEPS=1 bash scripts/phase0_defravens_setup.sh"
fi

if [[ ! -d "$DEFRAVENS_ROOT/.git" ]]; then
  echo "[Phase0] cloning DeformableRavens into $DEFRAVENS_ROOT"
  git clone "$DEFRAVENS_URL" "$DEFRAVENS_ROOT"
else
  echo "[Phase0] DeformableRavens already exists: $DEFRAVENS_ROOT"
fi

cd "$DEFRAVENS_ROOT"
echo "[Phase0] DeformableRavens commit: $(git rev-parse HEAD)"

detect_env_tool() {
  if command -v mamba >/dev/null 2>&1; then
    echo "mamba"
  elif command -v conda >/dev/null 2>&1; then
    echo "conda"
  elif command -v micromamba >/dev/null 2>&1; then
    echo "micromamba"
  else
    echo ""
  fi
}

ENV_TOOL="$(detect_env_tool)"
if [[ -z "$ENV_TOOL" ]]; then
  echo "[Phase0][ERROR] No conda/mamba/micromamba found in PATH."
  echo "Install Miniconda/Mambaforge first, then rerun."
  exit 1
fi

echo "[Phase0] env tool: $ENV_TOOL"

env_exists() {
  case "$ENV_TOOL" in
    mamba|conda)
      "$ENV_TOOL" env list | awk '{print $1}' | grep -qx "$ENV_NAME"
      ;;
    micromamba)
      micromamba env list | awk '{print $1}' | grep -qx "$ENV_NAME"
      ;;
  esac
}

if env_exists; then
  echo "[Phase0] env exists: $ENV_NAME"
else
  echo "[Phase0] creating env: $ENV_NAME with Python 3.7"
  case "$ENV_TOOL" in
    mamba)
      mamba create -n "$ENV_NAME" python=3.7 -y
      ;;
    conda)
      conda create -n "$ENV_NAME" python=3.7 -y
      ;;
    micromamba)
      micromamba create -n "$ENV_NAME" python=3.7 -y -c conda-forge
      ;;
  esac
fi

run_in_env() {
  case "$ENV_TOOL" in
    mamba|conda)
      eval "$("$ENV_TOOL" shell.bash hook)"
      "$ENV_TOOL" activate "$ENV_NAME"
      "$@"
      ;;
    micromamba)
      eval "$(micromamba shell hook -s bash)"
      micromamba activate "$ENV_NAME"
      "$@"
      ;;
  esac
}

echo "[Phase0] installing Python dependencies inside $ENV_NAME"
run_in_env bash -lc "
  set -euo pipefail
  cd '$DEFRAVENS_ROOT'
  python --version

  python -m pip install --upgrade 'pip<24' setuptools wheel

  # Official script uses conda tensorflow-gpu==2.4.1. This fallback keeps Phase0 CPU-compatible.
  if command -v conda >/dev/null 2>&1; then
    conda install -y ipython || true
    conda install -y tensorflow-gpu==2.4.1 || python -m pip install tensorflow==2.4.1
  else
    python -m pip install ipython tensorflow==2.4.1
  fi

  python -m pip install \
    pybullet==3.0.4 \
    packaging==19.2 \
    matplotlib==3.1.1 \
    opencv-python==4.1.2.30 \
    meshcat==0.0.18 \
    transformations==2020.1.1 \
    scikit-image==0.17.2 \
    gputil==1.4.0 \
    circle-fit==0.1.3 \
    tensorflow-addons==0.13.0 \
    tensorflow_hub==0.8.0

  python -m pip install -e .
"

echo "[Phase0] setup complete."
echo "Next:"
echo "  conda activate $ENV_NAME"
echo "  bash scripts/phase0_defravens_run.sh"
