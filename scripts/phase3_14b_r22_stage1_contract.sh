#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"; export PYTHONPATH="$ROOT:$ROOT/scripts:$ROOT/external/deformable-ravens:${PYTHONPATH:-}"
[[ "${PHASE314B_R22_ALLOW_CONTRACT_FREEZE:-0}" == 1 && "${PHASE314B_R22_CONTRACT_FREEZE_CONFIRMED:-0}" == 1 ]] || { echo blocked; exit 1; }
python -m py_compile ccda_phase3/phase314b_r22_contract.py ccda_phase3/phase314b_r22_geometry.py ccda_phase3/phase314b_r22_loss.py scripts/phase3_14b_r22_*.py
pytest -q tests/test_phase314b_r22_contract.py tests/test_phase314b_r22_geometry.py tests/test_phase314b_r22_loss.py tests/test_phase314b_r22_gates.py tests/test_phase3_14b_r21_contract.py tests/test_phase3_14b_r21_geometry.py
python scripts/phase3_14b_r22_preflight.py --root "$ROOT"
python scripts/phase3_14b_r22_freeze_contract.py --root "$ROOT"
python -c "from ccda_phase3.phase314b_r22_contract import load_self_hashed_json; load_self_hashed_json('reports/phase3_14b_r22_frozen_contract.json')"
python scripts/phase3_14b_r22_reference_audit.py --root "$ROOT"
python scripts/phase3_14b_r22_analyze.py --root "$ROOT"
