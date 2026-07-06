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

echo "[Phase3.8c] root=$ROOT"
echo "[Phase3.8c] python=$(which python)"
echo "[Phase3.8c] conda_env=${CONDA_DEFAULT_ENV:-}"
echo "[Phase3.8c] head=$(git rev-parse HEAD)"

python scripts/phase3_8c_pose0_confirmation_preflight.py \
  --root "$ROOT" \
  --windows data/phase3_state_diff_windows/phase3_windows.npz \
  --phase36_summary reports/phase3_6_matched_replay_summary.json \
  --phase37_summary reports/phase3_7_learned_action_alignment_summary.json \
  --phase38_sensitivity reports/phase3_8_action_geometry_sensitivity_summary.json \
  --phase38b_summary reports/phase3_8b_narrow_pose1_repair_summary.json \
  --out_json reports/phase3_8c_pose0_confirmation_preflight_summary.json \
  --out_md reports/phase3_8c_pose0_confirmation_preflight_report.md

if [[ "${PHASE3_ALLOW_POSE0_CONFIRMATION:-0}" != "1" ]]; then
  echo "[Phase3.8c][BLOCKED] Preflight completed. Confirmation not run because PHASE3_ALLOW_POSE0_CONFIRMATION != 1."
  exit 1
fi

if [[ "${PHASE3_POSE0_CONFIRMATION_CONFIRMED:-0}" != "1" ]]; then
  echo "[Phase3.8c][BLOCKED] Preflight completed. Confirmation not run because PHASE3_POSE0_CONFIRMATION_CONFIRMED != 1."
  exit 1
fi

# Phase3.8b worker code still uses these geometry-repair gates.
if [[ "${PHASE3_ALLOW_IDM_GEOMETRY_REPAIR:-0}" != "1" ]]; then
  echo "[Phase3.8c][BLOCKED] Phase3.8b worker requires PHASE3_ALLOW_IDM_GEOMETRY_REPAIR=1."
  exit 1
fi

if [[ "${PHASE3_IDM_GEOMETRY_REPAIR_CONFIRMED:-0}" != "1" ]]; then
  echo "[Phase3.8c][BLOCKED] Phase3.8b worker requires PHASE3_IDM_GEOMETRY_REPAIR_CONFIRMED=1."
  exit 1
fi

cat > reports/phase3_8c_pose0_confirmation_runtime_config.md <<EOF
# Phase3.8c Pose0-XY Repair Confirmation Runtime Config

- Timestamp: \`$(date -Is)\`
- Python: \`$(which python)\`
- Conda env: \`${CONDA_DEFAULT_ENV:-}\`
- Main HEAD: \`$(git rev-parse HEAD)\`
- Conditions: \`${PHASE3_8C_CONDITIONS:-free hidden_breakaway_pin hidden_high_friction}\`
- Baselines: \`${PHASE3_8C_BASELINES:-state_action}\`
- Sources: \`${PHASE3_8C_SOURCES:-idm_gt_future}\`
- Variants: \`${PHASE3_8C_VARIANTS:-gt_reference idm_original idm_gt_pose0_xy idm_gt_pose1_xy idm_gt_pose0_pose1_xy}\`
- Samples per condition: \`${PHASE3_8C_SAMPLES_PER_CONDITION:-2}\`
- Max rows: \`${PHASE3_8C_MAX_ROWS:-30}\`
- Row timeout sec: \`${PHASE3_8C_ROW_TIMEOUT_SEC:-240}\`
- Total timeout sec: \`${PHASE3_8C_TOTAL_TIMEOUT_SEC:-5400}\`
- Early stop: \`disabled\`

Scope:
- pose0-XY confirmation diagnostic only
- reuses Phase3.8b row-by-row CSV flush and per-row worker timeout
- GT-blended variants are diagnostic upper bounds only
- no Phase4
- no CPS
- no retraining
EOF

python scripts/phase3_8b_narrow_pose1_repair_probe.py \
  --root "$ROOT" \
  --windows data/phase3_state_diff_windows/phase3_windows.npz \
  --action_template data/phase3_state_diff_windows/phase3_action_template.pkl \
  --checkpoint_root checkpoints/phase3 \
  --phase37_raw reports/phase3_7_learned_action_alignment_raw_summary.json \
  --baselines ${PHASE3_8C_BASELINES:-state_action} \
  --sources ${PHASE3_8C_SOURCES:-idm_gt_future} \
  --conditions ${PHASE3_8C_CONDITIONS:-free hidden_breakaway_pin hidden_high_friction} \
  --variants ${PHASE3_8C_VARIANTS:-gt_reference idm_original idm_gt_pose0_xy idm_gt_pose1_xy idm_gt_pose0_pose1_xy} \
  --samples_per_condition "${PHASE3_8C_SAMPLES_PER_CONDITION:-2}" \
  --pred_samples "${PHASE3_8C_PRED_SAMPLES:-16}" \
  --seed_base "${PHASE3_8C_SEED_BASE:-386000}" \
  --action_clip_std "${PHASE3_8C_ACTION_CLIP_STD:-3.0}" \
  --motion_timeout "${PHASE3_8C_MOTION_TIMEOUT:-15.0}" \
  --max_prefix_actions "${PHASE3_8C_MAX_PREFIX_ACTIONS:-20}" \
  --row_timeout_sec "${PHASE3_8C_ROW_TIMEOUT_SEC:-240}" \
  --total_timeout_sec "${PHASE3_8C_TOTAL_TIMEOUT_SEC:-5400}" \
  --max_rows "${PHASE3_8C_MAX_ROWS:-30}" \
  --out_csv reports/phase3_8c_pose0_confirmation_trials.csv \
  --out_json reports/phase3_8c_pose0_confirmation_raw_summary.json \
  --progress_json reports/phase3_8c_pose0_confirmation_progress.json \
  --worker_dir reports/phase3_8c_workers

set +e
python scripts/phase3_8c_analyze_pose0_confirmation.py \
  --root "$ROOT" \
  --trials_csv reports/phase3_8c_pose0_confirmation_trials.csv \
  --progress_json reports/phase3_8c_pose0_confirmation_progress.json \
  --raw_summary reports/phase3_8c_pose0_confirmation_raw_summary.json \
  --out_json reports/phase3_8c_pose0_confirmation_summary.json \
  --out_md reports/phase3_8c_pose0_confirmation_report.md \
  --effective_delta_threshold "${PHASE3_8C_EFFECTIVE_DELTA_THRESHOLD:-0.05}"
ANALYZE_RC=$?
set -e

cat > reports/phase3_8c_no_phase4_confirmation.md <<EOF
# Phase3.8c No Phase4 / No CPS Confirmation

- Timestamp: \`$(date -Is)\`
- Phase3.8c ran pose0-XY repair confirmation diagnostics only.
- No Phase4 was run.
- No CPS was run.
- No retraining was run.
- No medium/full rollout was run.
- GT-blended repair variants are diagnostic upper bounds only.
- No paper-level claim should be made from this diagnostic alone.
EOF

exit "$ANALYZE_RC"
