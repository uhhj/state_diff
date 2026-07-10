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
export CCDA_BREAKAWAY_MIN_PHYSICS_STEPS="${CCDA_BREAKAWAY_MIN_PHYSICS_STEPS:-1}"

CONDITIONS="${PHASE3_12D_R1_CONDITIONS:-free hidden_breakaway_pin hidden_high_friction}"
SEEDS="${PHASE3_12D_R1_VISIBLE_SEEDS:-312000 312001 312002 312003 312500 312501 312502 312503}"
STEPS="${PHASE3_12D_R1_QUERY_STEPS:-0 4}"
RAW_SAMPLES="${PHASE3_12D_R1_RAW_SAMPLES:-8}"

write_scope_confirmation() {
  cat > reports/phase3_12d_r1_no_phase4_confirmation.md <<EOF
# Phase3.12d-r1 No Phase4 / No CPS Confirmation

- Timestamp: \`$(date -Is)\`
- Phase3.12d-r1 performed environment semantics audit and query-local candidate execution only.
- No model training was run.
- No future DDPM was trained.
- No Phase4 was run.
- No CPS was run.
- Best realized DDPM sample and goal-geometry action are diagnostic oracles only.
EOF
}

for gate in \
  PHASE3_ALLOW_ENVIRONMENT_SEMANTICS_REPAIR \
  PHASE3_ENVIRONMENT_SEMANTICS_REPAIR_CONFIRMED \
  PHASE3_ALLOW_QUERY_LOCAL_SNAPSHOT \
  PHASE3_QUERY_LOCAL_SNAPSHOT_CONFIRMED
do
  if [[ "${!gate:-0}" != "1" ]]; then
    echo "[Phase3.12d-r1][BLOCKED] $gate must be 1"
    write_scope_confirmation
    exit 1
  fi
done

cat > reports/phase3_12d_r1_runtime_config.md <<EOF
# Phase3.12d-r1 Environment Semantics + Query-Local Snapshot Runtime Config

- Timestamp: \`$(date -Is)\`
- Python: \`$(which python)\`
- Conda env: \`${CONDA_DEFAULT_ENV:-}\`
- Main HEAD before run: \`$(git rev-parse HEAD)\`
- Submodule before run: \`$(git submodule status external/deformable-ravens)\`
- Conditions: \`$CONDITIONS\`
- Visible seeds: \`$SEEDS\`
- Query steps: \`$STEPS\`
- Raw DDPM samples/query: \`$RAW_SAMPLES\`
- Candidate count/query: \`$((RAW_SAMPLES + 7))\`
- Full query count: \`48\`
- Full candidate rows: \`720\`

Breakaway:
- force: \`$CCDA_BREAKAWAY_FORCE\`
- displacement: \`$CCDA_BREAKAWAY_DISP\`
- bead ratio: \`$CCDA_BREAKAWAY_BEAD_RATIO\`
- minimum physics steps: \`$CCDA_BREAKAWAY_MIN_PHYSICS_STEPS\`

Snapshot integrity:
- bead XY max abs: \`${PHASE3_12D_R1_RESTORE_BEAD_XY_MAX_ABS:-1e-7}\`
- bead XY MAE: \`${PHASE3_12D_R1_RESTORE_BEAD_XY_MAE:-1e-8}\`
- robot q max abs: \`${PHASE3_12D_R1_RESTORE_ROBOT_Q_MAX_ABS:-1e-7}\`
- fraction diff: \`${PHASE3_12D_R1_RESTORE_FRACTION:-1e-12}\`
- curve diff: \`${PHASE3_12D_R1_RESTORE_CURVE:-1e-9}\`

Scope:
- minimal DeformableRavens environment/task semantics repair
- one prefix execution per query
- PyBullet saveState/restoreState per candidate
- no training
- no Phase4
- no CPS
EOF

# Apply the minimal, idempotent submodule patch before importing Ravens.
python scripts/phase3_12d_r1_patch_environment.py --root "$ROOT"
python scripts/phase3_12d_r1_patch_environment.py --root "$ROOT" --check-only

python -m py_compile \
  scripts/phase3_12d_r1_common.py \
  scripts/phase3_12d_r1_patch_environment.py \
  scripts/phase3_12d_r1_preflight.py \
  scripts/phase3_12d_r1_environment_audit.py \
  scripts/phase3_12d_r1_query_local_snapshot.py \
  scripts/phase3_12d_r1_analyze.py \
  external/deformable-ravens/ravens/environment.py \
  external/deformable-ravens/ravens/tasks/ccda_hidden_contact_cable.py

python scripts/phase3_12d_r1_preflight.py \
  --root "$ROOT" \
  --windows data/phase3_state_diff_windows/phase3_windows.npz \
  --action-template data/phase3_state_diff_windows/phase3_action_template.pkl \
  --checkpoint-root checkpoints/phase3 \
  --phase39b-summary reports/phase3_9b_ablation_raw_summary.json \
  --best-ablation "${PHASE3_12D_R1_BEST_ABLATION:-xy_only_high_weight}" \
  --phase312c-summary reports/phase3_12c_paired_selector_summary.json \
  --phase312d-summary "${PHASE3_12D_R1_PRIOR_SUMMARY:-reports/phase3_12d_candidate_oracle_summary.json}" \
  --out-json reports/phase3_12d_r1_preflight_summary.json \
  --out-md reports/phase3_12d_r1_preflight_report.md

python scripts/phase3_12d_r1_environment_audit.py \
  --root "$ROOT" \
  --visible-seeds 312000 312500 \
  --breakaway-seed 312000 \
  --breakaway-force "${PHASE3_12D_R1_BREAKAWAY_PROBE_FORCE:-15.0}" \
  --breakaway-max-steps "${PHASE3_12D_R1_BREAKAWAY_PROBE_STEPS:-2400}" \
  --visible-max-abs-threshold "${PHASE3_12D_R1_VISIBLE_MAX_ABS:-0.002}" \
  --goal-actionable-threshold "${PHASE3_12D_R1_ACTIONABLE_THRESHOLD:-0.003}" \
  --out-json reports/phase3_12d_r1_environment_audit_summary.json \
  --out-md reports/phase3_12d_r1_environment_audit_report.md

run_matrix() {
  local label="$1"
  local conditions="$2"
  local seeds="$3"
  local steps="$4"
  local stop_on_failure="$5"
  local query_csv="reports/phase3_12d_r1_${label}_queries.csv"
  local candidate_csv="reports/phase3_12d_r1_${label}_candidate_effects.csv"
  local progress_json="reports/phase3_12d_r1_${label}_progress.json"
  local raw_json="reports/phase3_12d_r1_${label}_raw_summary.json"
  local workers="reports/phase3_12d_r1_${label}_workers"
  local stop_arg=()
  if [[ "$stop_on_failure" == "1" ]]; then
    stop_arg+=(--stop-on-query-failure)
  fi
  rm -rf "$workers"
  rm -f "$query_csv" "$candidate_csv" "$progress_json" "$raw_json"

  python scripts/phase3_12d_r1_query_local_snapshot.py \
    --root "$ROOT" \
    --windows data/phase3_state_diff_windows/phase3_windows.npz \
    --old-checkpoint-root checkpoints/phase3 \
    --phase39b-raw reports/phase3_9b_ablation_raw_summary.json \
    --best-ablation "${PHASE3_12D_R1_BEST_ABLATION:-xy_only_high_weight}" \
    --conditions $conditions \
    --visible-seeds $seeds \
    --query-steps $steps \
    --raw-samples "$RAW_SAMPLES" \
    --top-k "${PHASE3_12D_R1_TOP_K:-64}" \
    --motion-timeout "${PHASE3_12D_R1_MOTION_TIMEOUT:-15.0}" \
    --action-clip-std "${PHASE3_12D_R1_ACTION_CLIP_STD:-3.0}" \
    --score-proxy-weight "${PHASE3_12D_R1_SCORE_PROXY_WEIGHT:-1.0}" \
    --score-ood-weight "${PHASE3_12D_R1_SCORE_OOD_WEIGHT:-0.15}" \
    --score-clip-weight "${PHASE3_12D_R1_SCORE_CLIP_WEIGHT:-2.0}" \
    --score-action-mae-weight "${PHASE3_12D_R1_SCORE_ACTION_MAE_WEIGHT:-4.0}" \
    --score-pull-diff-weight "${PHASE3_12D_R1_SCORE_PULL_DIFF_WEIGHT:-2.0}" \
    --score-small-pull-weight "${PHASE3_12D_R1_SCORE_SMALL_PULL_WEIGHT:-2.0}" \
    --min-pull "${PHASE3_12D_R1_MIN_PULL:-0.08}" \
    --min-settle-steps "${PHASE3_12D_R1_MIN_SETTLE_STEPS:-540}" \
    --max-settle-steps "${PHASE3_12D_R1_MAX_SETTLE_STEPS:-2400}" \
    --static-checks-required "${PHASE3_12D_R1_STATIC_CHECKS_REQUIRED:-8}" \
    --static-check-interval "${PHASE3_12D_R1_STATIC_CHECK_INTERVAL:-10}" \
    --restore-bead-xy-max-abs "${PHASE3_12D_R1_RESTORE_BEAD_XY_MAX_ABS:-1e-7}" \
    --restore-bead-xy-mae "${PHASE3_12D_R1_RESTORE_BEAD_XY_MAE:-1e-8}" \
    --restore-robot-q-max-abs "${PHASE3_12D_R1_RESTORE_ROBOT_Q_MAX_ABS:-1e-7}" \
    --restore-fraction-threshold "${PHASE3_12D_R1_RESTORE_FRACTION:-1e-12}" \
    --restore-curve-threshold "${PHASE3_12D_R1_RESTORE_CURVE:-1e-9}" \
    --query-timeout-sec "${PHASE3_12D_R1_QUERY_TIMEOUT_SEC:-2400}" \
    --total-timeout-sec "${PHASE3_12D_R1_TOTAL_TIMEOUT_SEC:-86400}" \
    "${stop_arg[@]}" \
    --out-query-csv "$query_csv" \
    --out-candidate-csv "$candidate_csv" \
    --progress-json "$progress_json" \
    --out-json "$raw_json" \
    --worker-dir "$workers"

  python scripts/phase3_12d_r1_analyze.py \
    --root "$ROOT" \
    --query-csv "$query_csv" \
    --candidate-csv "$candidate_csv" \
    --progress-json "$progress_json" \
    --raw-json "$raw_json" \
    --environment-audit reports/phase3_12d_r1_environment_audit_summary.json \
    --conditions $conditions \
    --visible-seeds $seeds \
    --query-steps $steps \
    --raw-samples "$RAW_SAMPLES" \
    --primary-condition "${PHASE3_12D_R1_PRIMARY_CONDITION:-hidden_breakaway_pin}" \
    --actionable-threshold "${PHASE3_12D_R1_ACTIONABLE_THRESHOLD:-0.003}" \
    --headroom-threshold "${PHASE3_12D_R1_HEADROOM_THRESHOLD:-0.003}" \
    --min-positive-query-fraction "${PHASE3_12D_R1_MIN_POSITIVE_QUERY_FRACTION:-0.5}" \
    --goal-actionable-rate-threshold "${PHASE3_12D_R1_GOAL_ACTIONABLE_RATE:-0.5}" \
    --bootstrap-samples "${PHASE3_12D_R1_BOOTSTRAP_SAMPLES:-10000}" \
    --out-json "reports/phase3_12d_r1_${label}_summary.json" \
    --out-md "reports/phase3_12d_r1_${label}_report.md" \
    --out-query-summary-csv "reports/phase3_12d_r1_${label}_query_headroom.csv" \
    --out-family-summary-csv "reports/phase3_12d_r1_${label}_family_summary.csv"
}

# Stage 1: the exact kind of query that failed in the old cross-process design.
run_matrix "single_query_smoke" "free" "312000" "4" "1"

# Stage 2: primary-condition smoke across both historical seed cohorts.
run_matrix "primary_smoke" "hidden_breakaway_pin" "312000 312500" "0 4" "1"

# Stage 3: full matrix. Only reached when both integrity smoke stages pass.
run_matrix "full" "$CONDITIONS" "$SEEDS" "$STEPS" "0"

# Copy the full report to stable canonical names.
cp reports/phase3_12d_r1_full_summary.json reports/phase3_12d_r1_summary.json
cp reports/phase3_12d_r1_full_report.md reports/phase3_12d_r1_report.md
cp reports/phase3_12d_r1_full_query_headroom.csv reports/phase3_12d_r1_query_headroom.csv
cp reports/phase3_12d_r1_full_family_summary.csv reports/phase3_12d_r1_family_summary.csv

write_scope_confirmation

echo "[Phase3.12d-r1] completed"
