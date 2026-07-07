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

echo "[Phase3.11b] root=$ROOT"
echo "[Phase3.11b] python=$(which python)"
echo "[Phase3.11b] conda_env=${CONDA_DEFAULT_ENV:-}"
echo "[Phase3.11b] head=$(git rev-parse HEAD)"

python scripts/phase3_11b_retrieval_feasibility_preflight.py \
  --root "$ROOT" \
  --windows data/phase3_state_diff_windows/phase3_windows.npz \
  --action_template data/phase3_state_diff_windows/phase3_action_template.pkl \
  --old_checkpoint_root checkpoints/phase3 \
  --phase39b_raw reports/phase3_9b_ablation_raw_summary.json \
  --phase311_summary reports/phase3_11_future_source_swap_summary.json \
  --phase311_steps reports/phase3_11_future_source_swap_steps.csv \
  --best_ablation "${PHASE3_11B_BEST_ABLATION:-xy_only_high_weight}" \
  --out_json reports/phase3_11b_retrieval_feasibility_preflight_summary.json \
  --out_md reports/phase3_11b_retrieval_feasibility_preflight_report.md

for gate in \
  PHASE3_ALLOW_RETRIEVAL_FEASIBILITY \
  PHASE3_RETRIEVAL_FEASIBILITY_CONFIRMED
do
  if [[ "${!gate:-0}" != "1" ]]; then
    echo "[Phase3.11b][BLOCKED] $gate must be 1"
    exit 1
  fi
done

cat > reports/phase3_11b_retrieval_feasibility_runtime_config.md <<EOF
# Phase3.11b Retrieval Feasibility Runtime Config

- Timestamp: \`$(date -Is)\`
- Python: \`$(which python)\`
- Conda env: \`${CONDA_DEFAULT_ENV:-}\`
- Main HEAD: \`$(git rev-parse HEAD)\`
- Conditions: \`${PHASE3_11B_CONDITIONS:-free hidden_breakaway_pin hidden_high_friction}\`
- Candidates per condition: \`${PHASE3_11B_CANDIDATES_PER_CONDITION:-3}\`
- Variants: \`${PHASE3_11B_VARIANTS:-source_gt_action source_old_idm_future source_repaired_idm_future live_retrieved_gt_action live_repaired_idm_retrieved_future live_ddpm_mean_repaired_idm live_condition_retrieval_repaired_idm}\`
- Samples per step: \`${PHASE3_11B_SAMPLES_PER_STEP:-32}\`
- Motion timeout: \`${PHASE3_11B_MOTION_TIMEOUT:-15.0}\`
- Action clip std: \`${PHASE3_11B_ACTION_CLIP_STD:-3.0}\`
- Row timeout sec: \`${PHASE3_11B_ROW_TIMEOUT_SEC:-360}\`
- Total timeout sec: \`${PHASE3_11B_TOTAL_TIMEOUT_SEC:-14400}\`
- Max rows: \`${PHASE3_11B_MAX_ROWS:-63}\`
- Best ablation: \`${PHASE3_11B_BEST_ABLATION:-xy_only_high_weight}\`

Scope:
- retrieval feasibility audit only
- no model training
- no future DDPM training
- no Phase4
- no CPS
- no medium/full rollout
EOF

rm -rf reports/phase3_11b_workers
mkdir -p reports/phase3_11b_workers

set +e
python scripts/phase3_11b_retrieval_feasibility_probe.py \
  --root "$ROOT" \
  --windows data/phase3_state_diff_windows/phase3_windows.npz \
  --action_template data/phase3_state_diff_windows/phase3_action_template.pkl \
  --old_checkpoint_root checkpoints/phase3 \
  --phase39b_raw reports/phase3_9b_ablation_raw_summary.json \
  --phase311_step_csv reports/phase3_11_future_source_swap_steps.csv \
  --best_ablation "${PHASE3_11B_BEST_ABLATION:-xy_only_high_weight}" \
  --conditions ${PHASE3_11B_CONDITIONS:-free hidden_breakaway_pin hidden_high_friction} \
  --variants ${PHASE3_11B_VARIANTS:-source_gt_action source_old_idm_future source_repaired_idm_future live_retrieved_gt_action live_repaired_idm_retrieved_future live_ddpm_mean_repaired_idm live_condition_retrieval_repaired_idm} \
  --candidates_per_condition "${PHASE3_11B_CANDIDATES_PER_CONDITION:-3}" \
  --samples_per_step "${PHASE3_11B_SAMPLES_PER_STEP:-32}" \
  --motion_timeout "${PHASE3_11B_MOTION_TIMEOUT:-15.0}" \
  --action_clip_std "${PHASE3_11B_ACTION_CLIP_STD:-3.0}" \
  --max_steps_replay "${PHASE3_11B_MAX_STEPS_REPLAY:-16}" \
  --max_prefix_actions "${PHASE3_11B_MAX_PREFIX_ACTIONS:-20}" \
  --seed_base "${PHASE3_11B_SEED_BASE:-311500}" \
  --row_timeout_sec "${PHASE3_11B_ROW_TIMEOUT_SEC:-360}" \
  --total_timeout_sec "${PHASE3_11B_TOTAL_TIMEOUT_SEC:-14400}" \
  --max_rows "${PHASE3_11B_MAX_ROWS:-63}" \
  --out_csv reports/phase3_11b_retrieval_feasibility_trials.csv \
  --out_json reports/phase3_11b_retrieval_feasibility_raw_summary.json \
  --progress_json reports/phase3_11b_retrieval_feasibility_progress.json \
  --worker_dir reports/phase3_11b_workers
PROBE_RC=$?
set -e

set +e
python scripts/phase3_11b_analyze_retrieval_feasibility.py \
  --root "$ROOT" \
  --trials_csv reports/phase3_11b_retrieval_feasibility_trials.csv \
  --progress_json reports/phase3_11b_retrieval_feasibility_progress.json \
  --raw_summary reports/phase3_11b_retrieval_feasibility_raw_summary.json \
  --out_json reports/phase3_11b_retrieval_feasibility_summary.json \
  --out_md reports/phase3_11b_retrieval_feasibility_report.md \
  --primary_condition hidden_breakaway_pin \
  --progress_threshold "${PHASE3_11B_PROGRESS_THRESHOLD:-0.05}" \
  --min_ok_rows_per_variant_condition "${PHASE3_11B_MIN_OK_ROWS_PER_VARIANT_CONDITION:-2}"
ANALYZE_RC=$?
set -e

cat > reports/phase3_11b_no_phase4_confirmation.md <<EOF
# Phase3.11b No Phase4 / No CPS Confirmation

- Timestamp: \`$(date -Is)\`
- Phase3.11b ran retrieved future/action feasibility diagnostics only.
- No model training was run.
- No future DDPM was trained.
- No Phase4 was run.
- No CPS was run.
- No medium/full rollout was run.
- Retrieved y_state / y_action / condition labels were used only for diagnostic feasibility audit, not deployable policy input.
- No paper-level claim should be made from this diagnostic alone.
EOF

if [[ "$PROBE_RC" -ne 0 || "$ANALYZE_RC" -ne 0 ]]; then
  exit 1
fi

echo "[Phase3.11b] done"
