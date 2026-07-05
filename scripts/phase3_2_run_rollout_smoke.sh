#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "[Phase3.2] root=$ROOT"
echo "[Phase3.2] branch=$(git branch --show-current)"
echo "[Phase3.2] head=$(git rev-parse HEAD)"

mkdir -p reports

export PHASE3_CONDITIONS="${PHASE3_CONDITIONS:-free hidden_pin hidden_high_friction hidden_breakaway_pin}"
export PHASE3_PRIMARY_HIDDEN_CONDITION="${PHASE3_PRIMARY_HIDDEN_CONDITION:-hidden_breakaway_pin}"
export PHASE3_DIAGNOSTIC_HIDDEN_CONDITION="${PHASE3_DIAGNOSTIC_HIDDEN_CONDITION:-hidden_pin}"

export CCDA_BREAKAWAY_FORCE="${CCDA_BREAKAWAY_FORCE:-2.6}"
export CCDA_BREAKAWAY_DISP="${CCDA_BREAKAWAY_DISP:-0.045}"
export CCDA_BREAKAWAY_BEAD_RATIO="${CCDA_BREAKAWAY_BEAD_RATIO:-0.45}"
export CCDA_ORACLE_BREAKAWAY_PULL_DIST="${CCDA_ORACLE_BREAKAWAY_PULL_DIST:-0.36}"

python scripts/phase3_2_rollout_code_hazard_audit.py \
  --root "$ROOT" \
  --out_json reports/phase3_2_rollout_code_hazard_summary.json \
  --out_md reports/phase3_2_rollout_code_hazard_report.md

python scripts/phase3_2_rollout_runtime_probe.py \
  --root "$ROOT" \
  --windows data/phase3_state_diff_windows/phase3_windows.npz \
  --checkpoint_root checkpoints/phase3 \
  --out_json reports/phase3_2_rollout_runtime_probe_summary.json \
  --out_md reports/phase3_2_rollout_runtime_probe_report.md

if [[ "${PHASE3_ALLOW_ROLLOUT:-0}" != "1" ]]; then
  echo "[Phase3.2] Audit/probe completed. Rollout not run because PHASE3_ALLOW_ROLLOUT != 1."
  exit 0
fi

if [[ "${PHASE3_ROLLOUT_CONFIRMED:-0}" != "1" ]]; then
  echo "[Phase3.2] Audit/probe completed. Rollout not run because PHASE3_ROLLOUT_CONFIRMED != 1."
  exit 0
fi

python scripts/phase3_policy_rollout.py \
  --conditions $PHASE3_CONDITIONS \
  --primary_hidden_condition "$PHASE3_PRIMARY_HIDDEN_CONDITION" \
  --diagnostic_hidden_condition "$PHASE3_DIAGNOSTIC_HIDDEN_CONDITION" \
  --selected_recoverable_config "breakaway_force_2p6_disp_0p045_pull_0p36" \
  --max_episodes_per_condition "${MAX_EPISODES_PER_CONDITION:-4}" \
  --max_steps "${MAX_STEPS:-16}" \
  --rollout_models ${ROLLOUT_MODELS:-paper_state state_action} \
  --checkpoint_root checkpoints/phase3 \
  --windows data/phase3_state_diff_windows/phase3_windows.npz \
  --out_csv reports/phase3_2_rollout_smoke_trials.csv \
  --out_json reports/phase3_2_rollout_smoke_summary.json \
  --out_md reports/phase3_2_rollout_smoke_report.md

python scripts/phase3_2_summarize_rollout_smoke.py \
  --root "$ROOT" \
  --trials_csv reports/phase3_2_rollout_smoke_trials.csv \
  --out_json reports/phase3_2_rollout_smoke_summary_checked.json \
  --out_md reports/phase3_2_rollout_smoke_summary_checked.md

echo "[Phase3.2] done"
