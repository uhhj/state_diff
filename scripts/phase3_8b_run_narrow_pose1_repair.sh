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

echo "[Phase3.8b] root=$ROOT"
echo "[Phase3.8b] python=$(which python)"
echo "[Phase3.8b] conda_env=${CONDA_DEFAULT_ENV:-}"
echo "[Phase3.8b] head=$(git rev-parse HEAD)"

python scripts/phase3_8b_narrow_repair_preflight.py \
  --root "$ROOT" \
  --windows data/phase3_state_diff_windows/phase3_windows.npz \
  --phase38_sensitivity reports/phase3_8_action_geometry_sensitivity_summary.json \
  --phase38_repair_report reports/phase3_8_idm_geometry_repair_report.md \
  --phase37_summary reports/phase3_7_learned_action_alignment_summary.json \
  --phase36_summary reports/phase3_6_matched_replay_summary.json \
  --out_json reports/phase3_8b_narrow_repair_preflight_summary.json \
  --out_md reports/phase3_8b_narrow_repair_preflight_report.md

if [[ "${PHASE3_ALLOW_IDM_GEOMETRY_REPAIR:-0}" != "1" ]]; then
  echo "[Phase3.8b][BLOCKED] Preflight completed. Narrow repair not run because PHASE3_ALLOW_IDM_GEOMETRY_REPAIR != 1."
  exit 1
fi

if [[ "${PHASE3_IDM_GEOMETRY_REPAIR_CONFIRMED:-0}" != "1" ]]; then
  echo "[Phase3.8b][BLOCKED] Preflight completed. Narrow repair not run because PHASE3_IDM_GEOMETRY_REPAIR_CONFIRMED != 1."
  exit 1
fi

cat > reports/phase3_8b_narrow_pose1_repair_runtime_config.md <<EOF
# Phase3.8b Narrowed Pose1-XY Repair Runtime Config

- Timestamp: \`$(date -Is)\`
- Python: \`$(which python)\`
- Conda env: \`${CONDA_DEFAULT_ENV:-}\`
- Main HEAD: \`$(git rev-parse HEAD)\`
- Conditions: \`${PHASE3_8B_CONDITIONS:-free hidden_breakaway_pin hidden_high_friction}\`
- Baselines: \`${PHASE3_8B_BASELINES:-state_action}\`
- Sources: \`${PHASE3_8B_SOURCES:-idm_gt_future}\`
- Variants: \`${PHASE3_8B_VARIANTS:-gt_reference idm_original idm_gt_pose1_xy idm_gt_pose0_xy idm_gt_pose0_pose1_xy idm_gt_pull_dir_gt_len}\`
- Samples per condition: \`${PHASE3_8B_SAMPLES_PER_CONDITION:-1}\`
- Max rows: \`${PHASE3_8B_MAX_ROWS:-18}\`
- Row timeout sec: \`${PHASE3_8B_ROW_TIMEOUT_SEC:-240}\`
- Total timeout sec: \`${PHASE3_8B_TOTAL_TIMEOUT_SEC:-1800}\`
- Stop after first effective: \`${PHASE3_8B_STOP_AFTER_FIRST_EFFECTIVE:-1}\`

Scope:
- narrowed pose1-XY repair diagnostic only
- row-by-row CSV flush
- per-row worker timeout
- GT-blended variants are diagnostic upper bounds only
- no Phase4
- no CPS
- no retraining
EOF

extra_args=()
if [[ "${PHASE3_8B_STOP_AFTER_FIRST_EFFECTIVE:-1}" == "1" ]]; then
  extra_args+=(--stop_after_first_effective)
fi

python scripts/phase3_8b_narrow_pose1_repair_probe.py \
  --root "$ROOT" \
  --windows data/phase3_state_diff_windows/phase3_windows.npz \
  --action_template data/phase3_state_diff_windows/phase3_action_template.pkl \
  --checkpoint_root checkpoints/phase3 \
  --phase37_raw reports/phase3_7_learned_action_alignment_raw_summary.json \
  --baselines ${PHASE3_8B_BASELINES:-state_action} \
  --sources ${PHASE3_8B_SOURCES:-idm_gt_future} \
  --conditions ${PHASE3_8B_CONDITIONS:-free hidden_breakaway_pin hidden_high_friction} \
  --variants ${PHASE3_8B_VARIANTS:-gt_reference idm_original idm_gt_pose1_xy idm_gt_pose0_xy idm_gt_pose0_pose1_xy idm_gt_pull_dir_gt_len} \
  --samples_per_condition "${PHASE3_8B_SAMPLES_PER_CONDITION:-1}" \
  --pred_samples "${PHASE3_8B_PRED_SAMPLES:-16}" \
  --seed_base "${PHASE3_8B_SEED_BASE:-385000}" \
  --action_clip_std "${PHASE3_8B_ACTION_CLIP_STD:-3.0}" \
  --motion_timeout "${PHASE3_8B_MOTION_TIMEOUT:-15.0}" \
  --max_prefix_actions "${PHASE3_8B_MAX_PREFIX_ACTIONS:-20}" \
  --row_timeout_sec "${PHASE3_8B_ROW_TIMEOUT_SEC:-240}" \
  --total_timeout_sec "${PHASE3_8B_TOTAL_TIMEOUT_SEC:-1800}" \
  --max_rows "${PHASE3_8B_MAX_ROWS:-18}" \
  "${extra_args[@]}" \
  --out_csv reports/phase3_8b_narrow_pose1_repair_trials.csv \
  --out_json reports/phase3_8b_narrow_pose1_repair_raw_summary.json \
  --progress_json reports/phase3_8b_narrow_pose1_repair_progress.json \
  --worker_dir reports/phase3_8b_workers

python scripts/phase3_8b_analyze_narrow_pose1_repair.py \
  --root "$ROOT" \
  --trials_csv reports/phase3_8b_narrow_pose1_repair_trials.csv \
  --progress_json reports/phase3_8b_narrow_pose1_repair_progress.json \
  --raw_summary reports/phase3_8b_narrow_pose1_repair_raw_summary.json \
  --out_json reports/phase3_8b_narrow_pose1_repair_summary.json \
  --out_md reports/phase3_8b_narrow_pose1_repair_report.md

cat > reports/phase3_8b_no_phase4_confirmation.md <<EOF
# Phase3.8b No Phase4 / No CPS Confirmation

- Timestamp: \`$(date -Is)\`
- Phase3.8b ran narrowed pose1-XY repair diagnostics only.
- No Phase4 was run.
- No CPS was run.
- No retraining was run.
- No medium/full rollout was run.
- GT-blended repair variants are diagnostic upper bounds only.
- No paper-level claim should be made from this diagnostic alone.
EOF

echo "[Phase3.8b] done"
