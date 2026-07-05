# Phase3.1 Code Hazard Audit

## Verdict

- Verdict: `WARN`
- Branch: `Experiment1`
- HEAD: `13d9c6c0373a6d110451cb7bb9fb020dc05cc1e9`
- origin/Experiment1: `13d9c6c0373a6d110451cb7bb9fb020dc05cc1e9`
- Local HEAD equals origin/Experiment1: `True`
- Stale report marked: `True`

## Findings

| Level | Name | File | Line | Detail |
|---|---|---|---:|---|
| `INFO` | `rollout_gate` | `scripts/phase3_run_all.sh` |  | run_rollout - Rollout code exists; it must be gated by PHASE3_ALLOW_ROLLOUT=1. |
| `INFO` | `eval_hidden_pin_reference` | `scripts/phase3_eval_baselines.py` | 153 |                 has_pin_ref = has_diagnostic_ref if diagnostic_hidden == "hidden_pin" else ("hidden_pin" in ref) - Allowed only when hidden_pin is diagnostic and primary_hidden_condition is dynamic. |
| `WARN` | `rollout_hidden_pin_primary_metric` | `scripts/phase3_policy_rollout.py` |  | free_vs_hidden_pin_success_gap - Rollout summary still reports the old hidden_pin success gap. Patch or explicitly reinterpret this before rollout smoke. |
| `WARN` | `rollout_three_condition_default` | `scripts/phase3_policy_rollout.py` |  | default=["free", "hidden_pin", "hidden_high_friction"] - Rollout defaults omit hidden_breakaway_pin. Use explicit conditions or patch defaults before rollout smoke. |

## Required String Checks

| File | Required string | Present |
|---|---|---:|
| `scripts/phase3_eval_baselines.py` | `primary_hidden_condition` | `True` |
| `scripts/phase3_eval_baselines.py` | `diagnostic_hidden_condition` | `True` |
| `scripts/phase3_eval_baselines.py` | `hidden_breakaway_pin` | `True` |
| `scripts/phase3_check_input_leakage.py` | `primary_hidden_condition` | `True` |
| `scripts/phase3_check_input_leakage.py` | `diagnostic_hidden_condition` | `True` |
| `scripts/phase3_run_all.sh` | `PHASE3_PRIMARY_HIDDEN_CONDITION` | `True` |
| `scripts/phase3_run_all.sh` | `PHASE3_DIAGNOSTIC_HIDDEN_CONDITION` | `True` |
| `scripts/phase3_run_all.sh` | `PHASE3_CONDITIONS` | `True` |
| `scripts/phase3_run_all.sh` | `PHASE3_ALLOW_ROLLOUT` | `True` |
| `ccda_phase3/data_io.py` | `hidden_breakaway_pin` | `True` |
| `ccda_phase3/data_io.py` | `FORBIDDEN_METADATA_NOT_IN_X` | `True` |

## Interpretation

- `FAIL` blocks rollout and Phase4.
- `WARN` requires documentation, but may not block Phase3.1 if unrelated to leakage or primary-pair correctness.
- `INFO` records diagnostic-only hidden_pin references or explicit rollout gates.
