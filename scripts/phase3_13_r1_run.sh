#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="$ROOT:$ROOT/scripts:$ROOT/external/deformable-ravens:${PYTHONPATH:-}"

for gate in \
  PHASE313_R1_ALLOW_REGENERATION \
  PHASE313_R1_REGENERATION_CONFIRMED \
  PHASE313_R1_ALLOW_PROMOTION \
  PHASE313_R1_PROMOTION_CONFIRMED
do
  if [[ "${!gate:-0}" != "1" ]]; then
    echo "[Phase3.13-r1][BLOCKED] $gate must be 1"
    exit 1
  fi
done

mkdir -p reports

python -m py_compile \
  ccda_phase3/provenance_v2.py \
  scripts/phase3_13_r1_create_source_lock.py \
  scripts/phase3_13_r1_regenerate.py \
  scripts/phase3_13_r1_compare.py \
  scripts/phase3_13_r1_promote.py \
  scripts/phase3_13_r1_verify.py

pytest -q \
  tests/test_phase3_13_r1_provenance.py \
  tests/test_ccda_state_v2.py \
  tests/test_phase3_13_legacy_purge.py

git diff --check
git -C external/deformable-ravens diff --check

python scripts/phase3_13_r1_create_source_lock.py \
  --root "$ROOT"

python scripts/phase3_13_r1_regenerate.py \
  --root "$ROOT" \
  --source-lock reports/phase3_13_r1_source_lock.json \
  --staging-root data/phase3_state_v2_slack_r1_staging \
  --audit-prefix reports/phase3_13_r1 \
  --workers "${PHASE313_R1_WORKERS:-4}" \
  --fresh

python scripts/phase3_13_r1_compare.py \
  --root "$ROOT" \
  --old-root data/phase3_state_v2_slack \
  --new-root data/phase3_state_v2_slack_r1_staging

python scripts/phase3_13_r1_verify.py \
  --root "$ROOT" \
  --formal-root data/phase3_state_v2_slack_r1_staging \
  --audit-summary reports/phase3_13_r1_dataset_audit_summary.json \
  --require-exact-heads \
  --output reports/phase3_13_r1_staging_gate_summary.json \
  --report reports/phase3_13_r1_staging_gate_report.md

PROMOTION_ARGS=()
if [[ "${PHASE313_R1_DELETE_OLD_DATASET:-0}" == "1" ]]; then
  PROMOTION_ARGS+=(--delete-quarantine)
fi

python scripts/phase3_13_r1_promote.py \
  --root "$ROOT" \
  --staging-root data/phase3_state_v2_slack_r1_staging \
  --formal-root data/phase3_state_v2_slack \
  --audit-summary reports/phase3_13_r1_dataset_audit_summary.json \
  "${PROMOTION_ARGS[@]}"

python scripts/phase3_13_r1_verify.py \
  --root "$ROOT" \
  --formal-root data/phase3_state_v2_slack \
  --audit-summary reports/phase3_13_r1_dataset_audit_summary.json \
  --require-exact-heads

cat > reports/phase3_13_r1_no_phase4_confirmation.md <<EOF
# Phase3.13-r1 No Phase4 Confirmation

- Future-model training: \`False\`
- IDM training: \`False\`
- Candidate support: \`False\`
- Query-local execution: \`False\`
- Phase4: \`False\`
- CPS: \`False\`

The only authorized work was provenance-locked formal dataset regeneration,
comparison, audit, and promotion.
EOF

echo "[Phase3.13-r1] completed"
