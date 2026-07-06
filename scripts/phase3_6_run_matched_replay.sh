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

echo "[Phase3.6] root=$ROOT"
echo "[Phase3.6] python=$(which python)"
echo "[Phase3.6] conda_env=${CONDA_DEFAULT_ENV:-}"
echo "[Phase3.6] head=$(git rev-parse HEAD)"

python scripts/phase3_6_window_schema_audit.py \
  --root "$ROOT" \
  --windows data/phase3_state_diff_windows/phase3_windows.npz \
  --out_json reports/phase3_6_window_schema_audit_summary.json \
  --out_md reports/phase3_6_window_schema_audit_report.md

python scripts/phase3_6_matched_replay_preflight.py \
  --root "$ROOT" \
  --windows data/phase3_state_diff_windows/phase3_windows.npz \
  --action_template data/phase3_state_diff_windows/phase3_action_template.pkl \
  --schema_json reports/phase3_6_window_schema_audit_summary.json \
  --out_json reports/phase3_6_matched_replay_preflight_summary.json \
  --out_md reports/phase3_6_matched_replay_preflight_report.md

if [[ "${PHASE3_ALLOW_MATCHED_REPLAY:-0}" != "1" ]]; then
  echo "[Phase3.6][BLOCKED] Schema/preflight completed. Matched replay not run because PHASE3_ALLOW_MATCHED_REPLAY != 1."
  exit 1
fi

if [[ "${PHASE3_MATCHED_REPLAY_CONFIRMED:-0}" != "1" ]]; then
  echo "[Phase3.6][BLOCKED] Schema/preflight completed. Matched replay not run because PHASE3_MATCHED_REPLAY_CONFIRMED != 1."
  exit 1
fi

cat > reports/phase3_6_matched_replay_runtime_config.md <<EOF
# Phase3.6 Matched Replay Runtime Config

- Timestamp: \`$(date -Is)\`
- Python: \`$(which python)\`
- Conda env: \`${CONDA_DEFAULT_ENV:-}\`
- Main HEAD: \`$(git rev-parse HEAD)\`
- Conditions: \`$PHASE3_CONDITIONS\`
- Primary hidden condition: \`$PHASE3_PRIMARY_HIDDEN_CONDITION\`
- Diagnostic hidden condition: \`$PHASE3_DIAGNOSTIC_HIDDEN_CONDITION\`
- Samples per condition: \`${PHASE3_6_SAMPLES_PER_CONDITION:-3}\`
- Motion timeouts: \`${PHASE3_6_MOTION_TIMEOUTS:-15.0}\`
- Max prefix actions: \`${PHASE3_6_MAX_PREFIX_ACTIONS:-20}\`
- Allow low-confidence prefix: \`${PHASE3_6_ALLOW_LOW_CONF_PREFIX:-0}\`

Scope:
- matched-state y_action replay diagnostic only
- no Phase4
- no CPS
- no paper-level claims
EOF

extra_args=()
if [[ "${PHASE3_6_ALLOW_LOW_CONF_PREFIX:-0}" == "1" ]]; then
  extra_args+=(--allow_low_conf_prefix)
fi

python scripts/phase3_6_matched_action_replay_diag.py \
  --root "$ROOT" \
  --windows data/phase3_state_diff_windows/phase3_windows.npz \
  --action_template data/phase3_state_diff_windows/phase3_action_template.pkl \
  --schema_json reports/phase3_6_window_schema_audit_summary.json \
  --samples_per_condition "${PHASE3_6_SAMPLES_PER_CONDITION:-3}" \
  --motion_timeouts ${PHASE3_6_MOTION_TIMEOUTS:-15.0} \
  --max_prefix_actions "${PHASE3_6_MAX_PREFIX_ACTIONS:-20}" \
  "${extra_args[@]}" \
  --out_csv reports/phase3_6_matched_action_replay_trials.csv \
  --out_json reports/phase3_6_matched_action_replay_raw_summary.json

python scripts/phase3_6_analyze_matched_replay.py \
  --root "$ROOT" \
  --trials_csv reports/phase3_6_matched_action_replay_trials.csv \
  --out_json reports/phase3_6_matched_replay_summary.json \
  --out_md reports/phase3_6_matched_replay_report.md

cat > reports/phase3_6_no_phase4_confirmation.md <<EOF
# Phase3.6 No Phase4 / No CPS Confirmation

- Timestamp: \`$(date -Is)\`
- Phase3.6 ran matched-state action replay diagnostics only.
- No Phase4 was run.
- No CPS was run.
- No paper-level claim should be made from this diagnostic alone.
EOF

echo "[Phase3.6] done"
