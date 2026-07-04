# Phase2.5 Hidden-Contact Recoverability Audit

## Verdict

- Verdict: `FAIL`
- Selected recoverable condition: `None`
- Recommendation: No recoverable candidate found. Do not run Phase3 medium; tune soft/breakaway/friction parameters and rerun Phase2.5.

## Condition Summary

| Condition | Nominal Success | Oracle Success | Search Best Success | Gap | Class |
|---|---:|---:|---:|---:|---|
| free | 1.000 | 1.000 | 0.000 | 0.000 | weak_easy_control |
| hidden_breakaway_pin | 0.375 | 0.500 | 0.000 | 0.125 | ambiguous_needs_tuning |
| hidden_friction_patch | 1.000 | 1.000 | 0.000 | 0.000 | weak_easy_control |
| hidden_high_friction | 1.000 | 1.000 | 0.000 | 0.000 | weak_easy_control |
| hidden_partial_pin | 0.250 | 0.250 | 0.000 | 0.000 | ambiguous_needs_tuning |
| hidden_pin | 0.000 | 0.000 | 0.000 | 0.000 | impossible_diagnostic |
| hidden_soft_pin | 0.875 | 1.000 | 0.000 | 0.125 | weak_easy_control |

## Parameter Sweep

| Sweep | Condition | Nominal Success | Oracle Success | Search Best Success | Gap | Class |
|---|---|---:|---:|---:|---:|---|
| breakaway_force_0p4 | hidden_breakaway_pin | 0.875 | 1.000 | 0.000 | 0.125 | weak_easy_control |
| breakaway_force_0p8 | hidden_breakaway_pin | 0.875 | 1.000 | 0.000 | 0.125 | weak_easy_control |
| partial_force_0p8 | hidden_partial_pin | 1.000 | 0.875 | 0.000 | -0.125 | weak_easy_control |
| soft_force_0p4 | hidden_soft_pin | 1.000 | 1.000 | 0.000 | 0.000 | weak_easy_control |

## Branch Roles

- Hard diagnostic branch: `hidden_pin`
- Weak control branch: `hidden_high_friction`
- Selected recoverable CPS branch: `None`

## Interpretation

- Success rates are computed from `final_fraction >= 0.95`, not from internal episode bookkeeping flags.
- `hidden_pin` is retained even if impossible; it should not be the success-improvement target.
- A recoverable branch requires low nominal success and high hidden-condition-aware oracle/search success.
- Do not run Phase3 medium unless this report selects a recoverable condition.
