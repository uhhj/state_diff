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

echo "[Phase3.8] root=$ROOT"
echo "[Phase3.8] python=$(which python)"
echo "[Phase3.8] conda_env=${CONDA_DEFAULT_ENV:-}"
echo "[Phase3.8] head=$(git rev-parse HEAD)"

python scripts/phase3_8_idm_geometry_preflight.py \
  --root "$ROOT" \
  --windows data/phase3_state_diff_windows/phase3_windows.npz \
  --action_template data/phase3_state_diff_windows/phase3_action_template.pkl \
  --checkpoint_root checkpoints/phase3 \
  --phase37_summary reports/phase3_7_learned_action_alignment_summary.json \
  --phase36_summary reports/phase3_6_matched_replay_summary.json \
  --out_json reports/phase3_8_idm_geometry_preflight_summary.json \
  --out_md reports/phase3_8_idm_geometry_preflight_report.md

python scripts/phase3_8_action_geometry_sensitivity.py \
  --root "$ROOT" \
  --windows data/phase3_state_diff_windows/phase3_windows.npz \
  --action_template data/phase3_state_diff_windows/phase3_action_template.pkl \
  --checkpoint_root checkpoints/phase3 \
  --phase37_raw reports/phase3_7_learned_action_alignment_raw_summary.json \
  --baselines ${PHASE3_8_BASELINES:-paper_state state_action} \
  --samples_per_condition "${PHASE3_8_SAMPLES_PER_CONDITION:-6}" \
  --pred_samples "${PHASE3_8_PRED_SAMPLES:-16}" \
  --seed_base "${PHASE3_8_SEED_BASE:-380000}" \
  --out_csv reports/phase3_8_action_geometry_sensitivity_dims.csv \
  --out_json reports/phase3_8_action_geometry_sensitivity_summary.json \
  --out_md reports/phase3_8_action_geometry_sensitivity_report.md

if [[ "${PHASE3_ALLOW_IDM_GEOMETRY_REPAIR:-0}" != "1" ]]; then
  echo "[Phase3.8][BLOCKED] Preflight/sensitivity completed. Repair probe not run because PHASE3_ALLOW_IDM_GEOMETRY_REPAIR != 1."
  exit 1
fi

if [[ "${PHASE3_IDM_GEOMETRY_REPAIR_CONFIRMED:-0}" != "1" ]]; then
  echo "[Phase3.8][BLOCKED] Preflight/sensitivity completed. Repair probe not run because PHASE3_IDM_GEOMETRY_REPAIR_CONFIRMED != 1."
  exit 1
fi

cat > reports/phase3_8_idm_geometry_runtime_config.md <<EOF
# Phase3.8 IDM Geometry Runtime Config

- Timestamp: \`$(date -Is)\`
- Python: \`$(which python)\`
- Conda env: \`${CONDA_DEFAULT_ENV:-}\`
- Main HEAD: \`$(git rev-parse HEAD)\`
- Conditions: \`$PHASE3_CONDITIONS\`
- Primary hidden condition: \`$PHASE3_PRIMARY_HIDDEN_CONDITION\`
- Diagnostic hidden condition: \`$PHASE3_DIAGNOSTIC_HIDDEN_CONDITION\`
- Baselines: \`${PHASE3_8_BASELINES:-paper_state state_action}\`
- Sources: \`${PHASE3_8_SOURCES:-idm_gt_future idm_pred_future}\`
- Samples per condition: \`${PHASE3_8_SAMPLES_PER_CONDITION:-3}\`
- Pred samples: \`${PHASE3_8_PRED_SAMPLES:-16}\`
- Motion timeout: \`${PHASE3_8_MOTION_TIMEOUT:-15.0}\`
- Max prefix actions: \`${PHASE3_8_MAX_PREFIX_ACTIONS:-20}\`
- Action clip std: \`${PHASE3_8_ACTION_CLIP_STD:-3.0}\`

Scope:
- IDM action geometry repair diagnostic only
- GT-blended repair variants are diagnostic upper bounds only
- no Phase4
- no CPS
- no retraining
EOF

python scripts/phase3_8_idm_geometry_repair_probe.py \
  --root "$ROOT" \
  --windows data/phase3_state_diff_windows/phase3_windows.npz \
  --action_template data/phase3_state_diff_windows/phase3_action_template.pkl \
  --checkpoint_root checkpoints/phase3 \
  --phase37_raw reports/phase3_7_learned_action_alignment_raw_summary.json \
  --baselines ${PHASE3_8_BASELINES:-paper_state state_action} \
  --sources ${PHASE3_8_SOURCES:-idm_gt_future idm_pred_future} \
  --samples_per_condition "${PHASE3_8_SAMPLES_PER_CONDITION:-3}" \
  --pred_samples "${PHASE3_8_PRED_SAMPLES:-16}" \
  --seed_base "${PHASE3_8_SEED_BASE:-380000}" \
  --action_clip_std "${PHASE3_8_ACTION_CLIP_STD:-3.0}" \
  --motion_timeout "${PHASE3_8_MOTION_TIMEOUT:-15.0}" \
  --max_prefix_actions "${PHASE3_8_MAX_PREFIX_ACTIONS:-20}" \
  --out_csv reports/phase3_8_idm_geometry_repair_trials.csv \
  --out_json reports/phase3_8_idm_geometry_repair_raw_summary.json

python scripts/phase3_8_analyze_geometry_repair.py \
  --root "$ROOT" \
  --trials_csv reports/phase3_8_idm_geometry_repair_trials.csv \
  --sensitivity_json reports/phase3_8_action_geometry_sensitivity_summary.json \
  --out_json reports/phase3_8_idm_geometry_repair_summary.json \
  --out_md reports/phase3_8_idm_geometry_repair_report.md

cat > reports/phase3_8_no_phase4_confirmation.md <<EOF
# Phase3.8 No Phase4 / No CPS Confirmation

- Timestamp: \`$(date -Is)\`
- Phase3.8 ran IDM action geometry repair diagnostics only.
- No Phase4 was run.
- No CPS was run.
- No retraining was run.
- GT-blended repair variants are diagnostic upper bounds only.
- No paper-level claim should be made from this diagnostic alone.
EOF

echo "[Phase3.8] done"
