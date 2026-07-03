#!/usr/bin/env bash
set -euo pipefail

echo "[Phase3] This installs CPU PyTorch into defravens37 for rollout only."
echo "[Phase3] It should not be used for StateDiff training."

if ! command -v conda >/dev/null 2>&1; then
  source /root/miniforge3/etc/profile.d/conda.sh
fi
conda activate defravens37

python - <<'PY'
import sys
print("python:", sys.executable)
print("version:", sys.version)
PY

python -m pip install "torch==1.13.1" --extra-index-url https://download.pytorch.org/whl/cpu

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
PY
