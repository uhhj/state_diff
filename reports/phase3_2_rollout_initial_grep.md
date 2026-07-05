# Phase3.2 Initial Rollout Grep

```text
scripts/phase3_policy_rollout.py:60:        item["free_vs_hidden_pin_success_gap"] = None if free is None or pin is None else float(free - pin)
scripts/phase3_policy_rollout.py:73:    ap.add_argument("--conditions", nargs="+", default=["free", "hidden_pin", "hidden_high_friction"])
scripts/phase3_policy_rollout.py:116:        for condition in args.conditions:
scripts/phase3_run_all.sh:10:PHASE3_ALLOW_ROLLOUT="${PHASE3_ALLOW_ROLLOUT:-0}"
scripts/phase3_run_all.sh:12:PHASE3_CONDITIONS="${PHASE3_CONDITIONS:-free hidden_pin hidden_high_friction hidden_breakaway_pin}"
scripts/phase3_run_all.sh:13:PHASE3_PRIMARY_HIDDEN_CONDITION="${PHASE3_PRIMARY_HIDDEN_CONDITION:-hidden_breakaway_pin}"
scripts/phase3_run_all.sh:72:condition_args=(--conditions $PHASE3_CONDITIONS --primary_hidden_condition "$PHASE3_PRIMARY_HIDDEN_CONDITION" --diagnostic_hidden_condition "$PHASE3_DIAGNOSTIC_HIDDEN_CONDITION")
scripts/phase3_run_all.sh:75:echo "[Phase3] conditions=$PHASE3_CONDITIONS"
scripts/phase3_run_all.sh:76:echo "[Phase3] primary_hidden_condition=$PHASE3_PRIMARY_HIDDEN_CONDITION diagnostic_hidden_condition=$PHASE3_DIAGNOSTIC_HIDDEN_CONDITION"
scripts/phase3_run_all.sh:161:  if [[ "$PHASE3_ALLOW_ROLLOUT" != "1" ]]; then
scripts/phase3_run_all.sh:162:    echo "[Phase3][ERROR] rollout is disabled by default. Set PHASE3_ALLOW_ROLLOUT=1 to run it explicitly."
scripts/phase3_run_all.sh:196:  echo "[Phase3] Step sanity diagnostics. Expected env: coord_bimanual"
```
