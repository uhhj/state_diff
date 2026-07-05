# Phase2.5c Recoverability Confirmation Report

## Verdict

- Verdict: `PASS`
- Selected condition if PASS: `hidden_breakaway_pin`
- Selected config if PASS: `breakaway_force_2p6_disp_0p045_pull_0p36`
- Confirmation scale: `confirmation_n = 20, weaker than preferred n=30`

## Strict Checks

| Check | Pass |
|---|---:|
| `free_nominal_ok` | `True` |
| `free_guided_ok` | `True` |
| `hidden_pin_oracle_best_low` | `True` |
| `selected_nominal_low` | `True` |
| `selected_oracle_breakaway_or_best_high` | `True` |
| `selected_gap_high` | `True` |
| `selected_breakaway_released_high` | `True` |

## Condition Summary with Counts

| Condition | Nominal | Guided Search | Oracle Mean | Oracle Best | Oracle Breakaway | Breakaway Released |
|---|---:|---:|---:|---:|---:|---:|
| `free` | 20/20 = 1.000 | 20/20 = 1.000 | 80/80 = 1.000 | 20/20 = 1.000 | 20/20 = 1.000 | 0/120 = 0.000 |
| `hidden_pin` | 0/20 = 0.000 | 1/20 = 0.050 | 0/80 = 0.000 | 0/20 = 0.000 | 0/20 = 0.000 | 0/120 = 0.000 |
| `hidden_high_friction` | 0/20 = 0.000 | 20/20 = 1.000 | 79/80 = 0.988 | 20/20 = 1.000 | 20/20 = 1.000 | 0/120 = 0.000 |
| `hidden_breakaway_pin` | 0/20 = 0.000 | 20/20 = 1.000 | 77/80 = 0.963 | 20/20 = 1.000 | 18/20 = 0.900 | 100/120 = 0.833 |

## Per-Policy Counts

### `free`

| Policy | Success | Total | Rate | Breakaway Released |
|---|---:|---:|---:|---:|
| `guided_search` | 20 | 20 | 1.000 | 0/20 = 0.000 |
| `nominal` | 20 | 20 | 1.000 | 0/20 = 0.000 |
| `oracle_breakaway_then_place` | 20 | 20 | 1.000 | 0/20 = 0.000 |
| `oracle_pull` | 20 | 20 | 1.000 | 0/20 = 0.000 |
| `oracle_regrasp` | 20 | 20 | 1.000 | 0/20 = 0.000 |
| `oracle_wiggle` | 20 | 20 | 1.000 | 0/20 = 0.000 |

### `hidden_pin`

| Policy | Success | Total | Rate | Breakaway Released |
|---|---:|---:|---:|---:|
| `guided_search` | 1 | 20 | 0.050 | 0/20 = 0.000 |
| `nominal` | 0 | 20 | 0.000 | 0/20 = 0.000 |
| `oracle_breakaway_then_place` | 0 | 20 | 0.000 | 0/20 = 0.000 |
| `oracle_pull` | 0 | 20 | 0.000 | 0/20 = 0.000 |
| `oracle_regrasp` | 0 | 20 | 0.000 | 0/20 = 0.000 |
| `oracle_wiggle` | 0 | 20 | 0.000 | 0/20 = 0.000 |

### `hidden_high_friction`

| Policy | Success | Total | Rate | Breakaway Released |
|---|---:|---:|---:|---:|
| `guided_search` | 20 | 20 | 1.000 | 0/20 = 0.000 |
| `nominal` | 0 | 20 | 0.000 | 0/20 = 0.000 |
| `oracle_breakaway_then_place` | 20 | 20 | 1.000 | 0/20 = 0.000 |
| `oracle_pull` | 20 | 20 | 1.000 | 0/20 = 0.000 |
| `oracle_regrasp` | 20 | 20 | 1.000 | 0/20 = 0.000 |
| `oracle_wiggle` | 19 | 20 | 0.950 | 0/20 = 0.000 |

### `hidden_breakaway_pin`

| Policy | Success | Total | Rate | Breakaway Released |
|---|---:|---:|---:|---:|
| `guided_search` | 20 | 20 | 1.000 | 20/20 = 1.000 |
| `nominal` | 0 | 20 | 0.000 | 0/20 = 0.000 |
| `oracle_breakaway_then_place` | 18 | 20 | 0.900 | 20/20 = 1.000 |
| `oracle_pull` | 20 | 20 | 1.000 | 20/20 = 1.000 |
| `oracle_regrasp` | 20 | 20 | 1.000 | 20/20 = 1.000 |
| `oracle_wiggle` | 19 | 20 | 0.950 | 20/20 = 1.000 |

## Interpretation

- This report uses count/total statistics to avoid over-interpreting small-sample fractions.
- `oracle_mean_success` is not used as the main recoverability criterion.
- Recoverability is confirmed only if `oracle_best_success` or `oracle_breakaway_then_place_success` is at least 0.60.
- `hidden_pin` remains hard diagnostic and is not the CPS success-improvement target.
