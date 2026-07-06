#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p reports

export PYTHONPATH="$ROOT:$ROOT/scripts:$ROOT/external/deformable-ravens:${PYTHONPATH:-}"

export PHASE3_CONDITIONS="${PHASE3_CONDITIONS:-free hidden_pin hidden_high_friction hidden_breakaway_pin}"
export PHASE3_PRIMARY_HIDDEN_CONDITION="${PHASE3_PRIMARY_HIDDEN_CONDITION:-hidden_breakaway_pin}"
export PHASE3_DIAGNOSTIC_HIDDEN_CONDITION="${PHASE3_DIAGNOSTIC_HIDDEN_CONDITION:-hidden_pin}"

export CCDA_BREAKAWAY_FORCE="${CCDA_BREAKAWAY_FORCE:-2.6}"
export CCDA_BREAKAWAY_DISP="${CCDA_BREAKAWAY_DISP:-0.045}"
export CCDA_BREAKAWAY_BEAD_RATIO="${CCDA_BREAKAWAY_BEAD_RATIO:-0.45}"
export CCDA_ORACLE_BREAKAWAY_PULL_DIST="${CCDA_ORACLE_BREAKAWAY_PULL_DIST:-0.36}"

echo "[Phase3.7] root=$ROOT"
echo "[Phase3.7] python=$(which python)"
echo "[Phase3.7] conda_env=${CONDA_DEFAULT_ENV:-}"
echo "[Phase3.7] head=$(git rev-parse HEAD)"

python scripts/phase3_7_learned_action_alignment_preflight.py \
  --root "$ROOT" \
  --windows data/phase3_state_diff_windows/phase3_windows.npz \
  --action_template data/phase3_state_diff_windows/phase3_action_template.pkl \
  --checkpoint_root checkpoints/phase3 \
  --phase36_summary reports/phase3_6_matched_replay_summary.json \
  --out_json reports/phase3_7_learned_action_alignment_preflight_summary.json \
  --out_md reports/phase3_7_learned_action_alignment_preflight_report.md

if [[ "${PHASE3_ALLOW_ACTION_ALIGNMENT:-0}" != "1" ]]; then
  echo "[Phase3.7][BLOCKED] Preflight completed. Alignment diagnostic not run because PHASE3_ALLOW_ACTION_ALIGNMENT != 1."
  exit 1
fi

if [[ "${PHASE3_ACTION_ALIGNMENT_CONFIRMED:-0}" != "1" ]]; then
  echo "[Phase3.7][BLOCKED] Preflight completed. Alignment diagnostic not run because PHASE3_ACTION_ALIGNMENT_CONFIRMED != 1."
  exit 1
fi

cat > reports/phase3_7_learned_action_alignment_runtime_config.md <<EOF
# Phase3.7 Learned Action Alignment Runtime Config

- Timestamp: \`$(date -Is)\`
- Python: \`$(which python)\`
- Conda env: \`${CONDA_DEFAULT_ENV:-}\`
- Main HEAD: \`$(git rev-parse HEAD)\`
- Conditions: \`$PHASE3_CONDITIONS\`
- Primary hidden condition: \`$PHASE3_PRIMARY_HIDDEN_CONDITION\`
- Diagnostic hidden condition: \`$PHASE3_DIAGNOSTIC_HIDDEN_CONDITION\`
- Samples per condition: \`${PHASE3_7_SAMPLES_PER_CONDITION:-3}\`
- Baselines: \`${PHASE3_7_BASELINES:-paper_state state_action}\`
- Pred samples: \`${PHASE3_7_PRED_SAMPLES:-16}\`
- Execute actions: \`${PHASE3_7_EXECUTE_ACTIONS:-1}\`
- Motion timeout: \`${PHASE3_7_MOTION_TIMEOUT:-15.0}\`
- Max prefix actions: \`${PHASE3_7_MAX_PREFIX_ACTIONS:-20}\`
- Action clip std: \`${PHASE3_7_ACTION_CLIP_STD:-3.0}\`

Scope:
- learned action alignment diagnostic only
- no Phase4
- no CPS
- no paper-level claims
EOF

extra_args=()
if [[ "${PHASE3_7_EXECUTE_ACTIONS:-1}" == "1" ]]; then
  extra_args+=(--execute_actions)
fi

python scripts/phase3_7_learned_action_alignment_diag.py \
  --root "$ROOT" \
  --windows data/phase3_state_diff_windows/phase3_windows.npz \
  --action_template data/phase3_state_diff_windows/phase3_action_template.pkl \
  --checkpoint_root checkpoints/phase3 \
  --phase36_raw reports/phase3_6_matched_action_replay_raw_summary.json \
  --baselines ${PHASE3_7_BASELINES:-paper_state state_action} \
  --samples_per_condition "${PHASE3_7_SAMPLES_PER_CONDITION:-3}" \
  --pred_samples "${PHASE3_7_PRED_SAMPLES:-16}" \
  --seed_base "${PHASE3_7_SEED_BASE:-370000}" \
  --action_clip_std "${PHASE3_7_ACTION_CLIP_STD:-3.0}" \
  --motion_timeout "${PHASE3_7_MOTION_TIMEOUT:-15.0}" \
  --max_prefix_actions "${PHASE3_7_MAX_PREFIX_ACTIONS:-20}" \
  "${extra_args[@]}" \
  --out_csv reports/phase3_7_learned_action_alignment_trials.csv \
  --out_json reports/phase3_7_learned_action_alignment_raw_summary.json

python scripts/phase3_7_analyze_learned_action_alignment.py \
  --root "$ROOT" \
  --trials_csv reports/phase3_7_learned_action_alignment_trials.csv \
  --out_json reports/phase3_7_learned_action_alignment_summary.json \
  --out_md reports/phase3_7_learned_action_alignment_report.md

cat > reports/phase3_7_no_phase4_confirmation.md <<EOF
# Phase3.7 No Phase4 / No CPS Confirmation

- Timestamp: \`$(date -Is)\`
- Phase3.7 ran learned action alignment diagnostics only.
- No Phase4 was run.
- No CPS was run.
- No paper-level claim should be made from this diagnostic alone.
EOF

echo "[Phase3.7] done"
