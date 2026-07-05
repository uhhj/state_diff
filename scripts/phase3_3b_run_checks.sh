#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p reports
export PYTHONPATH="$ROOT/external/deformable-ravens:${PYTHONPATH:-}"

echo "[Phase3.3b] Python: $(which python)"
echo "[Phase3.3b] Conda env: ${CONDA_DEFAULT_ENV:-}"
echo "[Phase3.3b] HEAD: $(git rev-parse HEAD)"

python scripts/phase3_3b_tf_free_import_probe.py \
  --root "$ROOT" \
  --out_json reports/phase3_3b_tf_free_import_probe_summary.json \
  --out_md reports/phase3_3b_tf_free_import_probe_report.md

python scripts/phase3_3b_rollout_import_hazard_audit.py \
  --root "$ROOT" \
  --out_json reports/phase3_3b_rollout_import_hazard_summary.json \
  --out_md reports/phase3_3b_rollout_import_hazard_report.md

python scripts/phase3_3b_task_env_dryrun.py \
  --root "$ROOT" \
  --out_json reports/phase3_3b_task_env_dryrun_summary.json \
  --out_md reports/phase3_3b_task_env_dryrun_report.md

python scripts/phase3_3b_runtime_verdict.py \
  --root "$ROOT" \
  --import_probe reports/phase3_3b_tf_free_import_probe_summary.json \
  --hazard reports/phase3_3b_rollout_import_hazard_summary.json \
  --dryrun reports/phase3_3b_task_env_dryrun_summary.json \
  --out_json reports/phase3_3b_runtime_verdict_summary.json \
  --out_md reports/phase3_3b_runtime_verdict_report.md

TS="$(date -Is)"
cat > reports/phase3_3b_no_rollout_confirmation.md <<EOF
# Phase3.3b No Rollout Confirmation

- Timestamp: $TS
- No learned rollout was run.
- PHASE3_ALLOW_ROLLOUT was not set.
- PHASE3_ROLLOUT_CONFIRMED was not set.
- No Phase4 was run.
- No CPS was run.
- No TensorFlow was installed by this script.
EOF

echo "[Phase3.3b] done"
