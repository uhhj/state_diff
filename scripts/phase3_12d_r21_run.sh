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
export CCDA_BREAKAWAY_DAMPING="0.0"
export CCDA_SETTLE_SECONDS="0"
export CCDA_POST_ARM_SETTLE_SECONDS="0"
export CCDA_DEFER_HIDDEN_CONTACT_ARMING="1"

CONDITIONS="${PHASE3_12D_R21_CONDITIONS:-free hidden_breakaway_pin hidden_high_friction}"
SEEDS="${PHASE3_12D_R21_VISIBLE_SEEDS:-312000 312001 312002 312003 312500 312501 312502 312503}"
STEPS="${PHASE3_12D_R21_QUERY_STEPS:-0 4}"
RAW_SAMPLES="${PHASE3_12D_R21_RAW_SAMPLES:-8}"
AUDIT_JSON="reports/phase3_12d_r21_environment_audit_summary.json"
AUDIT_MD="reports/phase3_12d_r21_environment_audit_report.md"

write_scope_confirmation() {
  cat > reports/phase3_12d_r21_no_phase4_confirmation.md <<EOF
# Phase3.12d-r2.1 No Phase4 / No CPS Confirmation

- Timestamp: \`$(date -Is)\`
- r2.1 corrects the paired-horizon no-action environment comparator.
- It does not change physical task parameters or patch the submodule.
- Candidate execution is allowed only after the r2.1 audit has no FAIL.
- No model training was run.
- No future DDPM was trained.
- No Phase4 was run.
- No CPS was run.
- Best realized DDPM and goal-geometry candidates are diagnostic oracles only.
EOF
}
trap write_scope_confirmation EXIT

for gate in \
  PHASE3_ALLOW_PAIRED_HORIZON_AUDIT \
  PHASE3_PAIRED_HORIZON_AUDIT_CONFIRMED \
  PHASE3_ALLOW_QUERY_LOCAL_SNAPSHOT \
  PHASE3_QUERY_LOCAL_SNAPSHOT_CONFIRMED
do
  if [[ "${!gate:-0}" != "1" ]]; then
    echo "[Phase3.12d-r2.1][BLOCKED] $gate must be 1"
    exit 1
  fi
done

cat > reports/phase3_12d_r21_runtime_config.md <<EOF
# Phase3.12d-r2.1 Paired-Horizon Audit + Query-Local Snapshot Runtime Config

- Timestamp: \`$(date -Is)\`
- Python: \`$(which python)\`
- Conda env: \`${CONDA_DEFAULT_ENV:-}\`
- Main HEAD: \`$(git rev-parse HEAD)\`
- Submodule: \`$(git submodule status external/deformable-ravens)\`
- Conditions: \`$CONDITIONS\`
- Visible seeds: \`$SEEDS\`
- Query steps: \`$STEPS\`
- Raw DDPM samples/query: \`$RAW_SAMPLES\`
- Expected full queries: \`48\`
- Expected full candidate rows: \`720\`

Audit correction:
- historical r2: hidden(t=N) versus free(t=0)
- r2.1: hidden(t=N) versus free(t=N)
- horizons: \`${PHASE3_12D_R21_EVALUATION_STEPS:-0 1 5 20}\`
- free replicate per seed/horizon: enabled
- hidden unarmed control per seed/horizon: enabled
- absolute no-action drift: diagnostic only

Hard thresholds:
- free reproducibility max/MAE: \`${PHASE3_12D_R21_REPRO_MAX:-1e-7}\` / \`${PHASE3_12D_R21_REPRO_MAE:-1e-8}\`
- condition-independent evolution max/MAE: \`${PHASE3_12D_R21_COMMON_MAX:-1e-6}\` / \`${PHASE3_12D_R21_COMMON_MAE:-1e-7}\`
- paired visible max/MAE: \`${PHASE3_12D_R21_PAIRED_MAX:-1e-4}\` / \`${PHASE3_12D_R21_PAIRED_MAE:-1e-5}\`
- fraction difference: \`${PHASE3_12D_R21_FRACTION_THRESHOLD:-1e-9}\`
- curve difference: \`${PHASE3_12D_R21_CURVE_THRESHOLD:-1e-5}\`

Scope:
- no physical-environment patch
- no data generation
- no training
- no Phase4
- no CPS
EOF

cat > reports/phase3_12d_r2_comparator_correction.md <<'EOF'
# Phase3.12d-r2 Comparator Correction

The historical r2 environment report compared hidden armed geometry after N
no-action physics steps with the free geometry at zero steps. That test measures
absolute post-reset motion, not condition-specific visible leakage.

Phase3.12d-r2.1 retains the historical report unchanged and replaces only the
hard-gate comparator with same-horizon controls:

- free replicate(N) versus free(N)
- hidden unarmed(N) versus free(N)
- hidden armed(N) versus free(N)
- armed motion(N) versus free motion(N)

No task or physics parameter is changed by r2.1.
EOF

# Static checks. r2.1 must not rerun the old r2 runner because it hardcodes the
# historical comparator.
python -m py_compile \
  scripts/phase3_12d_r21_preflight.py \
  scripts/phase3_12d_r21_paired_horizon_audit.py \
  scripts/phase3_12d_r21_static_selftest.py \
  scripts/phase3_12d_r2_common.py \
  scripts/phase3_12d_r2_environment_audit.py \
  scripts/phase3_12d_r2_query_local_snapshot.py \
  scripts/phase3_12d_r2_analyze.py \
  scripts/phase3_12d_r2_postprocess.py \
  external/deformable-ravens/ravens/environment.py \
  external/deformable-ravens/ravens/tasks/ccda_hidden_contact_cable.py

bash -n scripts/phase3_12d_r21_run.sh
git diff --check
python scripts/phase3_12d_r21_static_selftest.py

# Existing representation and query-local implementation preflight.
python scripts/phase3_12d_r2_preflight.py \
  --root "$ROOT" \
  --windows data/phase3_state_diff_windows/phase3_windows.npz \
  --out-json reports/phase3_12d_r21_base_preflight_summary.json \
  --out-md reports/phase3_12d_r21_base_preflight_report.md

# Comparator-specific preflight.
python scripts/phase3_12d_r21_preflight.py \
  --root "$ROOT" \
  --r2-summary reports/phase3_12d_r2_environment_audit_summary.json \
  --out-json reports/phase3_12d_r21_preflight_summary.json \
  --out-md reports/phase3_12d_r21_preflight_report.md

# Non-bypassable paired-horizon environment gate.
python scripts/phase3_12d_r21_paired_horizon_audit.py \
  --root "$ROOT" \
  --seeds $SEEDS \
  --goal-seeds 312000 312500 \
  --evaluation-steps ${PHASE3_12D_R21_EVALUATION_STEPS:-0 1 5 20} \
  --min-settle-steps "${PHASE3_12D_R21_MIN_SETTLE_STEPS:-540}" \
  --max-settle-steps "${PHASE3_12D_R21_MAX_SETTLE_STEPS:-2400}" \
  --static-checks-required "${PHASE3_12D_R21_STATIC_CHECKS_REQUIRED:-8}" \
  --static-check-interval "${PHASE3_12D_R21_STATIC_CHECK_INTERVAL:-10}" \
  --motion-timeout "${PHASE3_12D_R21_MOTION_TIMEOUT:-15.0}" \
  --force-x "${PHASE3_12D_R21_FORCE_X:-15.0}" \
  --force-y "${PHASE3_12D_R21_FORCE_Y:-0.0}" \
  --force-steps "${PHASE3_12D_R21_FORCE_STEPS:-240}" \
  --pre-arm-max "${PHASE3_12D_R21_PRE_ARM_MAX:-1e-7}" \
  --pre-arm-mae "${PHASE3_12D_R21_PRE_ARM_MAE:-1e-8}" \
  --immediate-max "${PHASE3_12D_R21_IMMEDIATE_MAX:-1e-6}" \
  --immediate-mae "${PHASE3_12D_R21_IMMEDIATE_MAE:-1e-7}" \
  --repro-max "${PHASE3_12D_R21_REPRO_MAX:-1e-7}" \
  --repro-mae "${PHASE3_12D_R21_REPRO_MAE:-1e-8}" \
  --common-max "${PHASE3_12D_R21_COMMON_MAX:-1e-6}" \
  --common-mae "${PHASE3_12D_R21_COMMON_MAE:-1e-7}" \
  --paired-max "${PHASE3_12D_R21_PAIRED_MAX:-1e-4}" \
  --paired-mae "${PHASE3_12D_R21_PAIRED_MAE:-1e-5}" \
  --fraction-threshold "${PHASE3_12D_R21_FRACTION_THRESHOLD:-1e-9}" \
  --curve-threshold "${PHASE3_12D_R21_CURVE_THRESHOLD:-1e-5}" \
  --anchor-threshold "${PHASE3_12D_R21_ANCHOR_THRESHOLD:-1e-7}" \
  --velocity-warn-max "${PHASE3_12D_R21_VELOCITY_WARN_MAX:-1e-3}" \
  --causal-divergence-threshold "${PHASE3_12D_R21_CAUSAL_DIVERGENCE:-0.003}" \
  --goal-actionable-threshold "${PHASE3_12D_R21_ACTIONABLE_THRESHOLD:-0.003}" \
  --out-json "$AUDIT_JSON" \
  --out-md "$AUDIT_MD"

run_matrix() {
  local label="$1"
  local conditions="$2"
  local seeds="$3"
  local steps="$4"
  local stop_on_failure="$5"

  local query_csv="reports/phase3_12d_r21_${label}_queries.csv"
  local candidate_csv="reports/phase3_12d_r21_${label}_candidate_effects.csv"
  local progress_json="reports/phase3_12d_r21_${label}_progress.json"
  local raw_json="reports/phase3_12d_r21_${label}_raw_summary.json"
  local workers="reports/phase3_12d_r21_${label}_workers"
  local stop_arg=()

  if [[ "$stop_on_failure" == "1" ]]; then
    stop_arg+=(--stop-on-query-failure)
  fi

  rm -rf "$workers"
  rm -f "$query_csv" "$candidate_csv" "$progress_json" "$raw_json"

  python scripts/phase3_12d_r2_query_local_snapshot.py \
    --root "$ROOT" \
    --windows data/phase3_state_diff_windows/phase3_windows.npz \
    --old-checkpoint-root checkpoints/phase3 \
    --phase39b-raw reports/phase3_9b_ablation_raw_summary.json \
    --best-ablation "${PHASE3_12D_R21_BEST_ABLATION:-xy_only_high_weight}" \
    --conditions $conditions \
    --visible-seeds $seeds \
    --query-steps $steps \
    --raw-samples "$RAW_SAMPLES" \
    --top-k "${PHASE3_12D_R21_TOP_K:-64}" \
    --motion-timeout "${PHASE3_12D_R21_MOTION_TIMEOUT:-15.0}" \
    --action-clip-std "${PHASE3_12D_R21_ACTION_CLIP_STD:-3.0}" \
    --score-proxy-weight "${PHASE3_12D_R21_SCORE_PROXY_WEIGHT:-1.0}" \
    --score-ood-weight "${PHASE3_12D_R21_SCORE_OOD_WEIGHT:-0.15}" \
    --score-clip-weight "${PHASE3_12D_R21_SCORE_CLIP_WEIGHT:-2.0}" \
    --score-action-mae-weight "${PHASE3_12D_R21_SCORE_ACTION_MAE_WEIGHT:-4.0}" \
    --score-pull-diff-weight "${PHASE3_12D_R21_SCORE_PULL_DIFF_WEIGHT:-2.0}" \
    --score-small-pull-weight "${PHASE3_12D_R21_SCORE_SMALL_PULL_WEIGHT:-2.0}" \
    --min-pull "${PHASE3_12D_R21_MIN_PULL:-0.08}" \
    --min-settle-steps "${PHASE3_12D_R21_MIN_SETTLE_STEPS:-540}" \
    --max-settle-steps "${PHASE3_12D_R21_MAX_SETTLE_STEPS:-2400}" \
    --static-checks-required "${PHASE3_12D_R21_STATIC_CHECKS_REQUIRED:-8}" \
    --static-check-interval "${PHASE3_12D_R21_STATIC_CHECK_INTERVAL:-10}" \
    --restore-bead-xy-max-abs "${PHASE3_12D_R21_RESTORE_BEAD_XY_MAX_ABS:-1e-7}" \
    --restore-bead-xy-mae "${PHASE3_12D_R21_RESTORE_BEAD_XY_MAE:-1e-8}" \
    --restore-robot-q-max-abs "${PHASE3_12D_R21_RESTORE_ROBOT_Q_MAX_ABS:-1e-7}" \
    --restore-fraction-threshold "${PHASE3_12D_R21_RESTORE_FRACTION:-1e-12}" \
    --restore-curve-threshold "${PHASE3_12D_R21_RESTORE_CURVE:-1e-9}" \
    --query-timeout-sec "${PHASE3_12D_R21_QUERY_TIMEOUT_SEC:-2400}" \
    --total-timeout-sec "${PHASE3_12D_R21_TOTAL_TIMEOUT_SEC:-86400}" \
    "${stop_arg[@]}" \
    --out-query-csv "$query_csv" \
    --out-candidate-csv "$candidate_csv" \
    --progress-json "$progress_json" \
    --out-json "$raw_json" \
    --worker-dir "$workers"

  python scripts/phase3_12d_r2_analyze.py \
    --root "$ROOT" \
    --query-csv "$query_csv" \
    --candidate-csv "$candidate_csv" \
    --progress-json "$progress_json" \
    --raw-json "$raw_json" \
    --environment-audit "$AUDIT_JSON" \
    --conditions $conditions \
    --visible-seeds $seeds \
    --query-steps $steps \
    --raw-samples "$RAW_SAMPLES" \
    --primary-condition "${PHASE3_12D_R21_PRIMARY_CONDITION:-hidden_breakaway_pin}" \
    --actionable-threshold "${PHASE3_12D_R21_ACTIONABLE_THRESHOLD:-0.003}" \
    --headroom-threshold "${PHASE3_12D_R21_HEADROOM_THRESHOLD:-0.003}" \
    --min-positive-query-fraction "${PHASE3_12D_R21_MIN_POSITIVE_QUERY_FRACTION:-0.5}" \
    --goal-actionable-rate-threshold "${PHASE3_12D_R21_GOAL_ACTIONABLE_RATE:-0.5}" \
    --bootstrap-samples "${PHASE3_12D_R21_BOOTSTRAP_SAMPLES:-10000}" \
    --out-json "reports/phase3_12d_r21_${label}_summary.json" \
    --out-md "reports/phase3_12d_r21_${label}_report.md" \
    --out-query-summary-csv "reports/phase3_12d_r21_${label}_query_headroom.csv" \
    --out-family-summary-csv "reports/phase3_12d_r21_${label}_family_summary.csv"

  python scripts/phase3_12d_r2_postprocess.py \
    --summary-json "reports/phase3_12d_r21_${label}_summary.json" \
    --report-md "reports/phase3_12d_r21_${label}_report.md" \
    --environment-audit "$AUDIT_JSON" \
    --label "r21_${label}"
}

# Stage 1: the dynamic-prefix query that failed in the old cross-process design.
run_matrix "single_query_smoke" "free" "312000" "4" "1"

# Stage 2: primary condition across both historical seed cohorts.
run_matrix "primary_smoke" "hidden_breakaway_pin" "312000 312500" "0 4" "1"

# Stage 3: full 48-query / 720-candidate matrix.
run_matrix "full" "$CONDITIONS" "$SEEDS" "$STEPS" "0"

cp reports/phase3_12d_r21_full_summary.json reports/phase3_12d_r21_summary.json
cp reports/phase3_12d_r21_full_report.md reports/phase3_12d_r21_report.md
cp reports/phase3_12d_r21_full_query_headroom.csv reports/phase3_12d_r21_query_headroom.csv
cp reports/phase3_12d_r21_full_family_summary.csv reports/phase3_12d_r21_family_summary.csv

echo "[Phase3.12d-r2.1] completed"
