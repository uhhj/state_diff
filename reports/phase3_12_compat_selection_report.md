# Phase3.12 Future Target Locality / Compatibility-Aware Selection Report

## Verdict

- Verdict: `WARN`
- Root cause: `phase312_simple_locality_selection_supported`
- Episode rows: `54`
- Step rows: `864`
- Progress status: `completed`

## Per-Condition Selector Comparison

| Condition | DDPM mean | Input NN | Condition NN | Compat global | Compat condition | Compat action-geom | Input Δ | Condition Δ | Compat global Δ | Compat condition Δ | Compat action-geom Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `free` | 0.0278 | 0.0000 | 0.0000 | 0.0556 | 0.0000 | 0.1250 | -0.0278 | -0.0278 | 0.0278 | -0.0278 | 0.0972 |
| `hidden_breakaway_pin` | 0.1111 | 0.0000 | 0.2361 | 0.0417 | 0.0000 | 0.0417 | -0.1111 | 0.1250 | -0.0694 | -0.1111 | -0.0694 |
| `hidden_high_friction` | 0.0694 | 0.0556 | 0.0000 | 0.2500 | 0.1528 | 0.0417 | -0.0139 | -0.0694 | 0.1806 | 0.0833 | -0.0278 |

## Pull / Match / Context Diagnostics

| Condition | DDPM pull | Compat global pull | Compat condition pull | Action-geom pull | DDPM match | Condition match | Compat condition match | Compat global context L2 | Compat condition context L2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `free` | 0.1408 | 0.2686 | 0.0987 | 0.1038 | 0.042 | 1.000 | 1.000 | 0.0952 | 0.0680 |
| `hidden_breakaway_pin` | 0.0648 | 0.1737 | 0.1342 | 0.1286 | 0.375 | 1.000 | 1.000 | 0.0716 | 0.0987 |
| `hidden_high_friction` | 0.1109 | 0.3858 | 0.0893 | 0.1131 | 0.583 | 1.000 | 1.000 | 0.1112 | 0.0696 |

## Episode Summary

| Selector | Condition | Rows | OK | Timeout | Uses condition | Final | Δ | Future match | Context L2 | Score | Pull | OOD | Clip | Action MAE | Pull diff | Failures |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `ddpm_mean` | `free` | 3 | 3 | 0 | 0 | 0.0278 | 0.0278 | 0.042 | nan | nan | 0.1408 | 0.369 | 0.000 | nan | nan | 0 |
| `ddpm_mean` | `hidden_breakaway_pin` | 3 | 3 | 0 | 0 | 0.1111 | 0.1111 | 0.375 | nan | nan | 0.0648 | 0.240 | 0.000 | nan | nan | 0 |
| `ddpm_mean` | `hidden_high_friction` | 3 | 3 | 0 | 0 | 0.0694 | 0.0556 | 0.583 | nan | nan | 0.1109 | 0.311 | 0.000 | nan | nan | 0 |
| `input_nearest` | `free` | 3 | 3 | 0 | 0 | 0.0000 | 0.0000 | 0.354 | 0.0818 | nan | 0.2137 | 0.642 | 0.006 | nan | nan | 0 |
| `input_nearest` | `hidden_breakaway_pin` | 3 | 3 | 0 | 0 | 0.0000 | 0.0000 | 0.417 | 0.0800 | nan | 0.2209 | 0.643 | 0.000 | nan | nan | 0 |
| `input_nearest` | `hidden_high_friction` | 3 | 3 | 0 | 0 | 0.0556 | 0.0556 | 0.062 | 0.0552 | nan | 0.1257 | 0.443 | 0.000 | nan | nan | 0 |
| `condition_nearest` | `free` | 3 | 3 | 0 | 1 | 0.0000 | 0.0000 | 1.000 | 0.0870 | nan | 0.2053 | 0.701 | 0.006 | nan | nan | 0 |
| `condition_nearest` | `hidden_breakaway_pin` | 3 | 3 | 0 | 1 | 0.2361 | 0.1806 | 1.000 | 0.0531 | nan | 0.0762 | 0.419 | 0.000 | nan | nan | 0 |
| `condition_nearest` | `hidden_high_friction` | 3 | 3 | 0 | 1 | 0.0000 | -0.0556 | 1.000 | 0.0565 | nan | 0.1238 | 0.468 | 0.000 | nan | nan | 0 |
| `compat_global_topk` | `free` | 3 | 3 | 0 | 0 | 0.0556 | 0.0556 | 0.104 | 0.0952 | 0.2940 | 0.2686 | 0.599 | 0.000 | 0.0095 | 0.0353 | 0 |
| `compat_global_topk` | `hidden_breakaway_pin` | 3 | 3 | 0 | 0 | 0.0417 | 0.0417 | 0.250 | 0.0716 | 0.2059 | 0.1737 | 0.411 | 0.000 | 0.0079 | 0.0144 | 0 |
| `compat_global_topk` | `hidden_high_friction` | 3 | 3 | 0 | 0 | 0.2500 | 0.2500 | 0.146 | 0.1112 | 0.3124 | 0.3858 | 0.717 | 0.000 | 0.0147 | 0.0131 | 0 |
| `compat_condition_topk` | `free` | 3 | 3 | 0 | 1 | 0.0000 | 0.0000 | 1.000 | 0.0680 | 0.2604 | 0.0987 | 0.428 | 0.000 | 0.0076 | 0.0329 | 0 |
| `compat_condition_topk` | `hidden_breakaway_pin` | 3 | 3 | 0 | 1 | 0.0000 | 0.0000 | 1.000 | 0.0987 | 0.2662 | 0.1342 | 0.452 | 0.000 | 0.0081 | 0.0277 | 0 |
| `compat_condition_topk` | `hidden_high_friction` | 3 | 3 | 0 | 1 | 0.1528 | 0.1528 | 1.000 | 0.0696 | 0.2052 | 0.0893 | 0.345 | 0.000 | 0.0067 | 0.0084 | 0 |
| `compat_condition_action_geom` | `free` | 3 | 3 | 0 | 1 | 0.1250 | 0.1250 | 1.000 | 0.0985 | 0.2954 | 0.1038 | 0.564 | 0.000 | 0.0108 | 0.0074 | 0 |
| `compat_condition_action_geom` | `hidden_breakaway_pin` | 3 | 3 | 0 | 1 | 0.0417 | 0.0417 | 1.000 | 0.1236 | 0.3102 | 0.1286 | 0.666 | 0.000 | 0.0086 | 0.0094 | 0 |
| `compat_condition_action_geom` | `hidden_high_friction` | 3 | 3 | 0 | 1 | 0.0417 | 0.0417 | 1.000 | 0.0903 | 0.2400 | 0.1131 | 0.442 | 0.000 | 0.0068 | 0.0074 | 0 |

## Issues

| Level | Name | Detail |
|---|---|---|
| `WARN` | `condition_nearest_improves_primary_upper_bound` | {'gain': 0.125} |

## Interpretation

- This is a compatibility-aware future target selection diagnostic only.
- No model training was run.
- No future DDPM was trained.
- No Phase4 or CPS was run.
- Condition-aware selectors are diagnostic upper bounds and not deployable policy evidence.
- If only condition-aware compatibility works, the next step is a contact/locality proxy diagnostic, not full CPS.
