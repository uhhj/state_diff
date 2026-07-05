#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFRAVENS_ROOT="$ROOT/external/deformable-ravens"

NUM_SEEDS="${NUM_SEEDS:-8}"
SEED_START="${SEED_START:-310000}"
MAX_STEPS="${MAX_STEPS:-16}"
RANDOM_TRIALS="${RANDOM_TRIALS:-24}"
WORKERS="${WORKERS:-6}"
FRESH="${FRESH:-1}"
POLICIES="${POLICIES:-nominal oracle_breakaway_then_place oracle_partial_release_then_place oracle_pull oracle_regrasp guided_search}"
CONDITIONS="${CONDITIONS:-free hidden_pin hidden_high_friction hidden_breakaway_pin hidden_partial_pin}"
CONFIGS="${CONFIGS:-breakaway_default breakaway_force_1p0_disp_0p025_pull_0p20 breakaway_force_1p2_disp_0p025_pull_0p22 breakaway_force_1p5_disp_0p030_pull_0p22 breakaway_force_1p8_disp_0p035_pull_0p25 breakaway_bead_0p35_force_1p5_disp_0p030 breakaway_bead_0p55_force_1p5_disp_0p030 partial_force_1p2_span_1 partial_force_1p8_span_1 breakaway_force_2p2_disp_0p040_pull_0p32 breakaway_force_2p6_disp_0p045_pull_0p36 breakaway_force_3p0_disp_0p050_pull_0p40}"

OUT_BASE="$DEFRAVENS_ROOT/data/phase2_5b_recoverability"
REPORT_BASE="$ROOT/reports/phase2_5b_grid"
if [[ "$FRESH" == "1" ]]; then
  rm -rf "$OUT_BASE" "$REPORT_BASE"
fi
mkdir -p "$OUT_BASE" "$REPORT_BASE"

run_one () {
  local label="$1"
  shift
  local skip=1
  for wanted in $CONFIGS; do
    if [[ "$wanted" == "$label" ]]; then
      skip=0
    fi
  done
  if [[ "$skip" == "1" ]]; then
    return 0
  fi

  echo "[Phase2.5b] sweep=$label env=$*"

  (
    export CCDA_SETTLE_SECONDS="${CCDA_SETTLE_SECONDS:-0.02}"
    export CCDA_AUDIT_T_LIM="${CCDA_AUDIT_T_LIM:-15.0}"
    export "$@"
    cd "$DEFRAVENS_ROOT"
    python ccda_recoverability_audit.py \
      --task hidden-contact-cable-line \
      --conditions $CONDITIONS \
      --policies $POLICIES \
      --num_seeds "$NUM_SEEDS" \
      --seed_start "$SEED_START" \
      --output_root "$OUT_BASE/$label" \
      --max_steps "$MAX_STEPS" \
      --random_trials "$RANDOM_TRIALS" \
      --workers "$WORKERS" \
      --fresh
  )

  cd "$ROOT"
  python scripts/phase2_5_recoverability_audit.py \
    --root "$ROOT" \
    --trials_csv "$OUT_BASE/$label/recoverability_trials.csv" \
    --summary_json "$OUT_BASE/$label/recoverability_summary.json" \
    --out_json "$REPORT_BASE/${label}_summary.json" \
    --out_md "$REPORT_BASE/${label}_report.md" || true
}

run_one breakaway_default \
  CCDA_BREAKAWAY_FORCE=1.2 \
  CCDA_BREAKAWAY_DISP=0.035 \
  CCDA_BREAKAWAY_BEAD_RATIO=0.45 \
  CCDA_ORACLE_BREAKAWAY_PULL_DIST=0.16

run_one breakaway_force_1p0_disp_0p025_pull_0p20 \
  CCDA_BREAKAWAY_FORCE=1.0 \
  CCDA_BREAKAWAY_DISP=0.025 \
  CCDA_BREAKAWAY_BEAD_RATIO=0.45 \
  CCDA_ORACLE_BREAKAWAY_PULL_DIST=0.20

run_one breakaway_force_1p2_disp_0p025_pull_0p22 \
  CCDA_BREAKAWAY_FORCE=1.2 \
  CCDA_BREAKAWAY_DISP=0.025 \
  CCDA_BREAKAWAY_BEAD_RATIO=0.45 \
  CCDA_ORACLE_BREAKAWAY_PULL_DIST=0.22

run_one breakaway_force_1p5_disp_0p030_pull_0p22 \
  CCDA_BREAKAWAY_FORCE=1.5 \
  CCDA_BREAKAWAY_DISP=0.030 \
  CCDA_BREAKAWAY_BEAD_RATIO=0.45 \
  CCDA_ORACLE_BREAKAWAY_PULL_DIST=0.22

run_one breakaway_force_1p8_disp_0p035_pull_0p25 \
  CCDA_BREAKAWAY_FORCE=1.8 \
  CCDA_BREAKAWAY_DISP=0.035 \
  CCDA_BREAKAWAY_BEAD_RATIO=0.45 \
  CCDA_ORACLE_BREAKAWAY_PULL_DIST=0.25

run_one breakaway_bead_0p35_force_1p5_disp_0p030 \
  CCDA_BREAKAWAY_FORCE=1.5 \
  CCDA_BREAKAWAY_DISP=0.030 \
  CCDA_BREAKAWAY_BEAD_RATIO=0.35 \
  CCDA_ORACLE_BREAKAWAY_PULL_DIST=0.22

run_one breakaway_bead_0p55_force_1p5_disp_0p030 \
  CCDA_BREAKAWAY_FORCE=1.5 \
  CCDA_BREAKAWAY_DISP=0.030 \
  CCDA_BREAKAWAY_BEAD_RATIO=0.55 \
  CCDA_ORACLE_BREAKAWAY_PULL_DIST=0.22

run_one breakaway_force_2p2_disp_0p040_pull_0p32 \
  CCDA_BREAKAWAY_FORCE=2.2 \
  CCDA_BREAKAWAY_DISP=0.040 \
  CCDA_BREAKAWAY_BEAD_RATIO=0.45 \
  CCDA_ORACLE_BREAKAWAY_PULL_DIST=0.32

run_one breakaway_force_2p6_disp_0p045_pull_0p36 \
  CCDA_BREAKAWAY_FORCE=2.6 \
  CCDA_BREAKAWAY_DISP=0.045 \
  CCDA_BREAKAWAY_BEAD_RATIO=0.45 \
  CCDA_ORACLE_BREAKAWAY_PULL_DIST=0.36

run_one breakaway_force_3p0_disp_0p050_pull_0p40 \
  CCDA_BREAKAWAY_FORCE=3.0 \
  CCDA_BREAKAWAY_DISP=0.050 \
  CCDA_BREAKAWAY_BEAD_RATIO=0.45 \
  CCDA_ORACLE_BREAKAWAY_PULL_DIST=0.40

run_one partial_force_1p2_span_1 \
  CCDA_PARTIAL_PIN_FORCE=1.2 \
  CCDA_PARTIAL_PIN_SPAN=1 \
  CCDA_PARTIAL_PIN_BEAD_RATIO=0.45

run_one partial_force_1p8_span_1 \
  CCDA_PARTIAL_PIN_FORCE=1.8 \
  CCDA_PARTIAL_PIN_SPAN=1 \
  CCDA_PARTIAL_PIN_BEAD_RATIO=0.45

cd "$ROOT"
python scripts/phase2_5b_select_recoverable_config.py \
  --grid_dir "$REPORT_BASE" \
  --out_json "$ROOT/reports/phase2_5b_recoverability_summary.json" \
  --out_md "$ROOT/reports/phase2_5b_recoverability_report.md"

echo "[Phase2.5b] done. See reports/phase2_5b_recoverability_report.md"
