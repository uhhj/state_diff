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

CONDITIONS="${PHASE3_12D_R2_CONDITIONS:-free hidden_breakaway_pin hidden_high_friction}"
SEEDS="${PHASE3_12D_R2_VISIBLE_SEEDS:-312000 312001 312002 312003 312500 312501 312502 312503}"
STEPS="${PHASE3_12D_R2_QUERY_STEPS:-0 4}"
RAW_SAMPLES="${PHASE3_12D_R2_RAW_SAMPLES:-8}"

write_scope_confirmation() {
  cat > reports/phase3_12d_r2_no_phase4_confirmation.md <<EOF
# Phase3.12d-r2 No Phase4 / No CPS Confirmation

- Timestamp: \`$(date -Is)\`
- Phase3.12d-r2 performed paired-visible latent arming repair, environment audit, and query-local candidate diagnostics only.
- No model training was run.
- No future DDPM was trained.
- No Phase4 was run.
- No CPS was run.
- Candidate execution is permitted only after the paired-visible environment audit passes.
- Best realized DDPM sample and goal-geometry action are diagnostic oracles only.
EOF
}

for gate in \
  PHASE3_ALLOW_PAIRED_VISIBLE_ARMING_REPAIR \
  PHASE3_PAIRED_VISIBLE_ARMING_REPAIR_CONFIRMED \
  PHASE3_ALLOW_QUERY_LOCAL_SNAPSHOT \
  PHASE3_QUERY_LOCAL_SNAPSHOT_CONFIRMED
do
  if [[ "${!gate:-0}" != "1" ]]; then
    echo "[Phase3.12d-r2][BLOCKED] $gate must be 1"
    write_scope_confirmation
    exit 1
  fi
done

cat > reports/phase3_12d_r2_runtime_config.md <<EOF
# Phase3.12d-r2 Paired-Visible Arming + Query-Local Snapshot Runtime Config

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

Latent arming:
- hidden contact deferred during common visible settle: \`True\`
- post-arm wall-clock settle: \`0\`
- breakaway damping: \`0.0\`
- zero-offset anchor at current bead pose: \`True\`
- velocity canonicalization before arming: \`True\`

Hard visible-parity thresholds:
- pre-arm bead XY max abs: \`${PHASE3_12D_R2_PRE_ARM_MAX:-1e-7}\`
- pre-arm bead XY MAE: \`${PHASE3_12D_R2_PRE_ARM_MAE:-1e-8}\`
- immediate post-arm max abs: \`${PHASE3_12D_R2_IMMEDIATE_MAX:-1e-6}\`
- immediate post-arm MAE: \`${PHASE3_12D_R2_IMMEDIATE_MAE:-1e-7}\`
- 20-step post-arm max abs: \`${PHASE3_12D_R2_DRIFT_MAX:-1e-4}\`
- 20-step post-arm MAE: \`${PHASE3_12D_R2_DRIFT_MAE:-1e-5}\`

Scope:
- no training
- no Phase4
- no CPS
EOF

# 1. Apply the strict, idempotent submodule task patch.
python scripts/phase3_12d_r2_patch_submodule.py --root "$ROOT"

# 2. Generate r2 query/analyzer from the already audited r1 implementation.
python scripts/phase3_12d_r2_prepare_query_local.py --root "$ROOT"

# 3. Static checks before any PyBullet runtime.
python -m py_compile \
  scripts/phase3_12d_r2_patch_submodule.py \
  scripts/phase3_12d_r2_prepare_query_local.py \
  scripts/phase3_12d_r2_static_selftest.py \
  scripts/phase3_12d_r2_common.py \
  scripts/phase3_12d_r2_preflight.py \
  scripts/phase3_12d_r2_environment_audit.py \
  scripts/phase3_12d_r2_query_local_snapshot.py \
  scripts/phase3_12d_r2_analyze.py \
  scripts/phase3_12d_r2_postprocess.py \
  external/deformable-ravens/ravens/environment.py \
  external/deformable-ravens/ravens/tasks/ccda_hidden_contact_cable.py

bash -n scripts/phase3_12d_r2_run.sh
git diff --check
python scripts/phase3_12d_r2_static_selftest.py

# 4. Code and representation preflight.
python scripts/phase3_12d_r2_preflight.py \
  --root "$ROOT" \
  --windows data/phase3_state_diff_windows/phase3_windows.npz \
  --out-json reports/phase3_12d_r2_preflight_summary.json \
  --out-md reports/phase3_12d_r2_preflight_report.md

# 5. Environment audit is a non-bypassable hard gate.
python scripts/phase3_12d_r2_environment_audit.py \
  --root "$ROOT" \
  --seeds $SEEDS \
  --goal-seeds 312000 312500 \
  --drift-steps 1 5 20 \
  --min-settle-steps "${PHASE3_12D_R2_MIN_SETTLE_STEPS:-540}" \
  --max-settle-steps "${PHASE3_12D_R2_MAX_SETTLE_STEPS:-2400}" \
  --static-checks-required "${PHASE3_12D_R2_STATIC_CHECKS_REQUIRED:-8}" \
  --static-check-interval "${PHASE3_12D_R2_STATIC_CHECK_INTERVAL:-10}" \
  --motion-timeout "${PHASE3_12D_R2_MOTION_TIMEOUT:-15.0}" \
  --force-x "${PHASE3_12D_R2_FORCE_X:-15.0}" \
  --force-y "${PHASE3_12D_R2_FORCE_Y:-0.0}" \
  --force-steps "${PHASE3_12D_R2_FORCE_STEPS:-240}" \
  --pre-arm-max "${PHASE3_12D_R2_PRE_ARM_MAX:-1e-7}" \
  --pre-arm-mae "${PHASE3_12D_R2_PRE_ARM_MAE:-1e-8}" \
  --immediate-max "${PHASE3_12D_R2_IMMEDIATE_MAX:-1e-6}" \
  --immediate-mae "${PHASE3_12D_R2_IMMEDIATE_MAE:-1e-7}" \
  --drift-max "${PHASE3_12D_R2_DRIFT_MAX:-1e-4}" \
  --drift-mae "${PHASE3_12D_R2_DRIFT_MAE:-1e-5}" \
  --fraction-threshold "${PHASE3_12D_R2_FRACTION_THRESHOLD:-1e-9}" \
  --curve-threshold "${PHASE3_12D_R2_CURVE_THRESHOLD:-1e-5}" \
  --anchor-threshold "${PHASE3_12D_R2_ANCHOR_THRESHOLD:-1e-7}" \
  --causal-divergence-threshold "${PHASE3_12D_R2_CAUSAL_DIVERGENCE:-0.003}" \
  --goal-actionable-threshold "${PHASE3_12D_R2_ACTIONABLE_THRESHOLD:-0.003}" \
  --out-json reports/phase3_12d_r2_environment_audit_summary.json \
  --out-md reports/phase3_12d_r2_environment_audit_report.md

run_matrix() {
  local label="$1"
  local conditions="$2"
  local seeds="$3"
  local steps="$4"
  local stop_on_failure="$5"
  local query_csv="reports/phase3_12d_r2_${label}_queries.csv"
  local candidate_csv="reports/phase3_12d_r2_${label}_candidate_effects.csv"
  local progress_json="reports/phase3_12d_r2_${label}_progress.json"
  local raw_json="reports/phase3_12d_r2_${label}_raw_summary.json"
  local workers="reports/phase3_12d_r2_${label}_workers"
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
    --best-ablation "${PHASE3_12D_R2_BEST_ABLATION:-xy_only_high_weight}" \
    --conditions $conditions \
    --visible-seeds $seeds \
    --query-steps $steps \
    --raw-samples "$RAW_SAMPLES" \
    --top-k "${PHASE3_12D_R2_TOP_K:-64}" \
    --motion-timeout "${PHASE3_12D_R2_MOTION_TIMEOUT:-15.0}" \
    --action-clip-std "${PHASE3_12D_R2_ACTION_CLIP_STD:-3.0}" \
    --score-proxy-weight "${PHASE3_12D_R2_SCORE_PROXY_WEIGHT:-1.0}" \
    --score-ood-weight "${PHASE3_12D_R2_SCORE_OOD_WEIGHT:-0.15}" \
    --score-clip-weight "${PHASE3_12D_R2_SCORE_CLIP_WEIGHT:-2.0}" \
    --score-action-mae-weight "${PHASE3_12D_R2_SCORE_ACTION_MAE_WEIGHT:-4.0}" \
    --score-pull-diff-weight "${PHASE3_12D_R2_SCORE_PULL_DIFF_WEIGHT:-2.0}" \
    --score-small-pull-weight "${PHASE3_12D_R2_SCORE_SMALL_PULL_WEIGHT:-2.0}" \
    --min-pull "${PHASE3_12D_R2_MIN_PULL:-0.08}" \
    --min-settle-steps "${PHASE3_12D_R2_MIN_SETTLE_STEPS:-540}" \
    --max-settle-steps "${PHASE3_12D_R2_MAX_SETTLE_STEPS:-2400}" \
    --static-checks-required "${PHASE3_12D_R2_STATIC_CHECKS_REQUIRED:-8}" \
    --static-check-interval "${PHASE3_12D_R2_STATIC_CHECK_INTERVAL:-10}" \
    --restore-bead-xy-max-abs "${PHASE3_12D_R2_RESTORE_BEAD_XY_MAX_ABS:-1e-7}" \
    --restore-bead-xy-mae "${PHASE3_12D_R2_RESTORE_BEAD_XY_MAE:-1e-8}" \
    --restore-robot-q-max-abs "${PHASE3_12D_R2_RESTORE_ROBOT_Q_MAX_ABS:-1e-7}" \
    --restore-fraction-threshold "${PHASE3_12D_R2_RESTORE_FRACTION:-1e-12}" \
    --restore-curve-threshold "${PHASE3_12D_R2_RESTORE_CURVE:-1e-9}" \
    --query-timeout-sec "${PHASE3_12D_R2_QUERY_TIMEOUT_SEC:-2400}" \
    --total-timeout-sec "${PHASE3_12D_R2_TOTAL_TIMEOUT_SEC:-86400}" \
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
    --environment-audit reports/phase3_12d_r2_environment_audit_summary.json \
    --conditions $conditions \
    --visible-seeds $seeds \
    --query-steps $steps \
    --raw-samples "$RAW_SAMPLES" \
    --primary-condition "${PHASE3_12D_R2_PRIMARY_CONDITION:-hidden_breakaway_pin}" \
    --actionable-threshold "${PHASE3_12D_R2_ACTIONABLE_THRESHOLD:-0.003}" \
    --headroom-threshold "${PHASE3_12D_R2_HEADROOM_THRESHOLD:-0.003}" \
    --min-positive-query-fraction "${PHASE3_12D_R2_MIN_POSITIVE_QUERY_FRACTION:-0.5}" \
    --goal-actionable-rate-threshold "${PHASE3_12D_R2_GOAL_ACTIONABLE_RATE:-0.5}" \
    --bootstrap-samples "${PHASE3_12D_R2_BOOTSTRAP_SAMPLES:-10000}" \
    --out-json "reports/phase3_12d_r2_${label}_summary.json" \
    --out-md "reports/phase3_12d_r2_${label}_report.md" \
    --out-query-summary-csv "reports/phase3_12d_r2_${label}_query_headroom.csv" \
    --out-family-summary-csv "reports/phase3_12d_r2_${label}_family_summary.csv"

  python scripts/phase3_12d_r2_postprocess.py \
    --summary-json "reports/phase3_12d_r2_${label}_summary.json" \
    --report-md "reports/phase3_12d_r2_${label}_report.md" \
    --environment-audit reports/phase3_12d_r2_environment_audit_summary.json \
    --label "$label"
}

# Stage 1: dynamic-prefix query that failed in the old cross-process design.
run_matrix "single_query_smoke" "free" "312000" "4" "1"

# Stage 2: primary condition across both historical seed cohorts.
run_matrix "primary_smoke" "hidden_breakaway_pin" "312000 312500" "0 4" "1"

# Stage 3: complete 48-query / 720-candidate matrix.
run_matrix "full" "$CONDITIONS" "$SEEDS" "$STEPS" "0"

cp reports/phase3_12d_r2_full_summary.json reports/phase3_12d_r2_summary.json
cp reports/phase3_12d_r2_full_report.md reports/phase3_12d_r2_report.md
cp reports/phase3_12d_r2_full_query_headroom.csv reports/phase3_12d_r2_query_headroom.csv
cp reports/phase3_12d_r2_full_family_summary.csv reports/phase3_12d_r2_family_summary.csv

write_scope_confirmation
echo "[Phase3.12d-r2] completed"
