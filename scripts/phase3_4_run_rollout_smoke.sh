#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p reports

export PYTHONPATH="$ROOT/external/deformable-ravens:${PYTHONPATH:-}"
export PHASE3_CONDITIONS="${PHASE3_CONDITIONS:-free hidden_pin hidden_high_friction hidden_breakaway_pin}"
export PHASE3_PRIMARY_HIDDEN_CONDITION="${PHASE3_PRIMARY_HIDDEN_CONDITION:-hidden_breakaway_pin}"
export PHASE3_DIAGNOSTIC_HIDDEN_CONDITION="${PHASE3_DIAGNOSTIC_HIDDEN_CONDITION:-hidden_pin}"

export CCDA_BREAKAWAY_FORCE="${CCDA_BREAKAWAY_FORCE:-2.6}"
export CCDA_BREAKAWAY_DISP="${CCDA_BREAKAWAY_DISP:-0.045}"
export CCDA_BREAKAWAY_BEAD_RATIO="${CCDA_BREAKAWAY_BEAD_RATIO:-0.45}"
export CCDA_ORACLE_BREAKAWAY_PULL_DIST="${CCDA_ORACLE_BREAKAWAY_PULL_DIST:-0.36}"

export MAX_EPISODES_PER_CONDITION="${MAX_EPISODES_PER_CONDITION:-4}"
export MAX_STEPS="${MAX_STEPS:-16}"
export ROLLOUT_MODELS="${ROLLOUT_MODELS:-paper_state state_action}"

echo "[Phase3.4] root=$ROOT"
echo "[Phase3.4] python=$(which python)"
echo "[Phase3.4] conda_env=${CONDA_DEFAULT_ENV:-}"
echo "[Phase3.4] head=$(git rev-parse HEAD)"
echo "[Phase3.4] conditions=$PHASE3_CONDITIONS"
echo "[Phase3.4] primary=$PHASE3_PRIMARY_HIDDEN_CONDITION diagnostic=$PHASE3_DIAGNOSTIC_HIDDEN_CONDITION"
echo "[Phase3.4] max_episodes_per_condition=$MAX_EPISODES_PER_CONDITION max_steps=$MAX_STEPS"
echo "[Phase3.4] rollout_models=$ROLLOUT_MODELS"

python scripts/phase3_4_rollout_preflight.py \
  --root "$ROOT" \
  --windows data/phase3_state_diff_windows/phase3_windows.npz \
  --action_template data/phase3_state_diff_windows/phase3_action_template.pkl \
  --checkpoint_root checkpoints/phase3 \
  --out_json reports/phase3_4_rollout_preflight_summary.json \
  --out_md reports/phase3_4_rollout_preflight_report.md

if [[ "${PHASE3_ALLOW_ROLLOUT:-0}" != "1" ]]; then
  echo "[Phase3.4][BLOCKED] Preflight passed, but rollout not run because PHASE3_ALLOW_ROLLOUT != 1."
  exit 1
fi

if [[ "${PHASE3_ROLLOUT_CONFIRMED:-0}" != "1" ]]; then
  echo "[Phase3.4][BLOCKED] Preflight passed, but rollout not run because PHASE3_ROLLOUT_CONFIRMED != 1."
  exit 1
fi

if [[ "$PHASE3_PRIMARY_HIDDEN_CONDITION" != "hidden_breakaway_pin" ]]; then
  echo "[Phase3.4][FAIL] primary hidden condition must be hidden_breakaway_pin"
  exit 1
fi

if [[ "$PHASE3_DIAGNOSTIC_HIDDEN_CONDITION" != "hidden_pin" ]]; then
  echo "[Phase3.4][FAIL] diagnostic hidden condition must be hidden_pin"
  exit 1
fi

TS="$(date -Is)"
cat > reports/phase3_4_rollout_smoke_runtime_config.md <<EOF
# Phase3.4 Rollout Smoke Runtime Config

- Timestamp: $TS
- Python: $(which python)
- Conda env: ${CONDA_DEFAULT_ENV:-}
- Main HEAD: $(git rev-parse HEAD)
- Conditions: $PHASE3_CONDITIONS
- Primary hidden condition: $PHASE3_PRIMARY_HIDDEN_CONDITION
- Diagnostic hidden condition: $PHASE3_DIAGNOSTIC_HIDDEN_CONDITION
- Max episodes per condition: $MAX_EPISODES_PER_CONDITION
- Max steps: $MAX_STEPS
- Rollout models: $ROLLOUT_MODELS
- CCDA_BREAKAWAY_FORCE: $CCDA_BREAKAWAY_FORCE
- CCDA_BREAKAWAY_DISP: $CCDA_BREAKAWAY_DISP
- CCDA_BREAKAWAY_BEAD_RATIO: $CCDA_BREAKAWAY_BEAD_RATIO
- CCDA_ORACLE_BREAKAWAY_PULL_DIST: $CCDA_ORACLE_BREAKAWAY_PULL_DIST

Scope:
- learned rollout smoke only
- no Phase4
- no CPS
- no paper-level claims
EOF

python scripts/phase3_policy_rollout.py \
  --conditions $PHASE3_CONDITIONS \
  --primary_hidden_condition "$PHASE3_PRIMARY_HIDDEN_CONDITION" \
  --diagnostic_hidden_condition "$PHASE3_DIAGNOSTIC_HIDDEN_CONDITION" \
  --selected_recoverable_config "breakaway_force_2p6_disp_0p045_pull_0p36" \
  --max_episodes_per_condition "$MAX_EPISODES_PER_CONDITION" \
  --max_steps "$MAX_STEPS" \
  --rollout_models $ROLLOUT_MODELS \
  --checkpoint_root checkpoints/phase3 \
  --windows data/phase3_state_diff_windows/phase3_windows.npz \
  --out_csv reports/phase3_4_rollout_smoke_trials.csv \
  --out_json reports/phase3_4_rollout_smoke_summary.json \
  --out_md reports/phase3_4_rollout_smoke_report.md

python scripts/phase3_4_rollout_result_audit.py \
  --root "$ROOT" \
  --trials_csv reports/phase3_4_rollout_smoke_trials.csv \
  --summary_json reports/phase3_4_rollout_smoke_summary.json \
  --out_json reports/phase3_4_rollout_result_audit_summary.json \
  --out_md reports/phase3_4_rollout_result_audit_report.md

TS="$(date -Is)"
cat > reports/phase3_4_no_phase4_confirmation.md <<EOF
# Phase3.4 No Phase4 / No CPS Confirmation

- Timestamp: $TS
- Phase3.4 ran rollout smoke only.
- No Phase4 was run.
- No CPS was run.
- No paper-level claim should be made from this smoke alone.
EOF

echo "[Phase3.4] rollout smoke complete"
