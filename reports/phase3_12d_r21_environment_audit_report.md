# Phase3.12d-r2.1 Paired-Horizon No-Action Drift Audit

## Verdict

- Verdict: `FAIL`
- Root cause: `phase312d_r21_latent_constraint_no_action_visible_leak_supported`
- Query-local snapshot allowed: `False`
- Horizon rows: `32`

## Correction

- r2 compared `hidden(t=N)` against `free(t=0)`.
- r2.1 compares `hidden(t=N)` against `free(t=N)`.
- Absolute no-action drift remains diagnostic and is not a condition-specific hard gate.

## Paired-horizon geometry

| Seed | Steps | Free drift max | Repro max | Common max | Paired max | Paired MAE | Excess max | Fraction diff | Curve diff |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 312000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 312000 | 1 | 2.2277236e-06 | 0 | 0 | 3.516674e-06 | 4.3865293e-07 | 3.516674e-06 | 0 | 3.2842901e-07 |
| 312000 | 5 | 1.0073185e-05 | 0 | 0 | 6.4969063e-06 | 1.480182e-06 | 6.4969063e-06 | 0 | 5.4936566e-07 |
| 312000 | 20 | 4.1604042e-05 | 0 | 0 | 2.4855137e-05 | 3.3569522e-06 | 2.4855137e-05 | 0 | 1.9703552e-06 |
| 312001 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 312001 | 1 | 2.2649765e-06 | 0 | 0 | 1.1622906e-06 | 2.4649004e-07 | 1.1622906e-06 | 0 | 3.479671e-08 |
| 312001 | 5 | 1.1920929e-05 | 0 | 0 | 5.7816505e-06 | 1.2113402e-06 | 5.7816505e-06 | 0 | 1.0152289e-06 |
| 312001 | 20 | 4.8488379e-05 | 0 | 0 | 1.4841557e-05 | 3.17581e-06 | 1.4841557e-05 | 0 | 7.0153518e-07 |
| 312002 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 312002 | 1 | 5.5506825e-06 | 0 | 0 | 3.6656857e-06 | 7.7237686e-07 | 3.6656857e-06 | 0 | 3.2273698e-07 |
| 312002 | 5 | 2.5741756e-05 | 0 | 0 | 1.9036233e-05 | 3.5373184e-06 | 1.9036233e-05 | 0 | 7.2207622e-07 |
| 312002 | 20 | 0.00012720376 | 0 | 0 | 4.0709972e-05 | 6.6858095e-06 | 4.0709972e-05 | 0 | 2.8253205e-07 |
| 312003 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 312003 | 1 | 1.3560057e-05 | 0 | 0 | 6.9141388e-06 | 5.1657359e-07 | 6.9141388e-06 | 0 | 9.6086973e-07 |
| 312003 | 5 | 3.5554171e-05 | 0 | 0 | 2.0503998e-05 | 2.9895455e-06 | 2.0503998e-05 | 0 | 3.9784952e-06 |
| 312003 | 20 | 0.00011515617 | 0 | 0 | 7.724762e-05 | 1.0835007e-05 | 7.724762e-05 | 0 | 8.5239957e-06 |
| 312500 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 312500 | 1 | 6.7949295e-06 | 0 | 0 | 3.6358833e-06 | 5.128483e-07 | 3.6358833e-06 | 0 | 7.8092814e-08 |
| 312500 | 5 | 3.9473176e-05 | 0 | 0 | 1.6689301e-05 | 2.0911296e-06 | 1.6689301e-05 | 0 | 1.2074444e-06 |
| 312500 | 20 | 0.00018879771 | 0 | 0 | 9.663403e-05 | 1.2227955e-05 | 9.663403e-05 | 0 | 8.3100037e-06 |
| 312501 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 312501 | 1 | 9.1716647e-06 | 0 | 0 | 8.2701445e-06 | 1.0337681e-06 | 8.2701445e-06 | 0 | 7.5854041e-07 |
| 312501 | 5 | 6.3575804e-05 | 0 | 0 | 3.7007034e-05 | 4.8964284e-06 | 3.7007034e-05 | 0 | 3.6725276e-07 |
| 312501 | 20 | 0.00025819242 | 0 | 0 | 6.9364905e-05 | 1.1970599e-05 | 6.9364905e-05 | 0 | 6.1836941e-06 |
| 312502 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 312502 | 1 | 5.6028366e-06 | 0 | 0 | 3.9041042e-06 | 7.3139866e-07 | 3.9041042e-06 | 0 | 4.6896977e-07 |
| 312502 | 5 | 2.43783e-05 | 0 | 0 | 1.2993813e-05 | 2.4624169e-06 | 1.2993813e-05 | 0 | 1.8754743e-08 |
| 312502 | 20 | 9.9599361e-05 | 0 | 0 | 4.9352646e-05 | 8.2391004e-06 | 4.9352646e-05 | 0 | 4.9105042e-06 |
| 312503 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 312503 | 1 | 6.6161156e-06 | 0 | 0 | 3.5762787e-06 | 5.3240607e-07 | 3.5762787e-06 | 0 | 4.1797623e-07 |
| 312503 | 5 | 3.8266182e-05 | 0 | 0 | 1.7285347e-05 | 2.1805366e-06 | 1.7285347e-05 | 0 | 1.3931372e-06 |
| 312503 | 20 | 0.00014811754 | 0 | 0 | 0.00017020106 | 1.7751629e-05 | 0.00017020106 | 0 | 1.1997191e-05 |

## Maximum diagnostics

| Metric | Value |
|---|---:|
| `free_reproducibility_max_abs` | 0 |
| `common_evolution_max_abs` | 0 |
| `paired_visible_max_abs` | 0.00017020106 |
| `excess_motion_max_abs` | 0.00017020106 |
| `paired_velocity_max_abs` | 0.0074034616 |

## Controlled causal probe

| Seed | Max divergence | Hidden released | Release physics step | Hook errors |
|---:|---:|---:|---:|---|
| 312000 | 0.064197 | True | 27 | None / None |
| 312500 | 0.053780 | True | 29 | None / None |

## Free goal-actionability probe

| Seed | Dense gain | Ordered gain | Fraction gain |
|---:|---:|---:|---:|
| 312000 | 0.074637 | 0.166565 | 0.500000 |
| 312500 | 0.271735 | 0.333874 | 0.166667 |

## Issues

| Level | Name | Detail |
|---|---|---|
| `WARN` | `paired_velocity_difference_large` | {'seed': 312002, 'steps': 1, 'max_abs': 0.0010200097458437085, 'mae': 0.00015941857168242122, 'rmse': 0.00028847028730411505} |
| `WARN` | `paired_velocity_difference_large` | {'seed': 312002, 'steps': 5, 'max_abs': 0.001763632993970532, 'mae': 0.0003344023000442478, 'rmse': 0.0004860257440361971} |
| `WARN` | `paired_velocity_difference_large` | {'seed': 312002, 'steps': 20, 'max_abs': 0.001344734919257462, 'mae': 0.00023963653127603537, 'rmse': 0.00036892753704969173} |
| `WARN` | `paired_velocity_difference_large` | {'seed': 312003, 'steps': 1, 'max_abs': 0.0016595984052401036, 'mae': 9.621127370549074e-05, 'rmse': 0.0002458406171058927} |
| `WARN` | `paired_velocity_difference_large` | {'seed': 312003, 'steps': 5, 'max_abs': 0.0013513116355170496, 'mae': 0.00024525759880273453, 'rmse': 0.00041906239729721866} |
| `FAIL` | `latent_constraint_no_action_visible_leak` | {'seed': 312003, 'steps': 20, 'same_horizon': {'max_abs': 7.724761962890625e-05, 'mae': 1.0835006833076477e-05, 'rmse': 2.159991506016806e-05, 'fraction_diff': 0.0, 'curve_diff': 8.523995723827907e-06}, 'excess_motion': {'max_abs': 7.724761962890625e-05, 'mae': 1.0835006833076477e-05, 'rmse': 2.159991506016806e-05}} |
| `WARN` | `paired_velocity_difference_large` | {'seed': 312003, 'steps': 20, 'max_abs': 0.0016085869283415377, 'mae': 0.00028581971272460355, 'rmse': 0.00046171873058782766} |
| `FAIL` | `latent_constraint_no_action_visible_leak` | {'seed': 312500, 'steps': 20, 'same_horizon': {'max_abs': 9.663403034210205e-05, 'mae': 1.2227954963843027e-05, 'rmse': 2.4289071486875617e-05, 'fraction_diff': 0.0, 'curve_diff': 8.31000370750748e-06}, 'excess_motion': {'max_abs': 9.663403034210205e-05, 'mae': 1.2227954963843027e-05, 'rmse': 2.4289071486875617e-05}} |
| `WARN` | `paired_velocity_difference_large` | {'seed': 312500, 'steps': 20, 'max_abs': 0.0030155202402966097, 'mae': 0.0003194382310185675, 'rmse': 0.0005811336930787409} |
| `WARN` | `paired_velocity_difference_large` | {'seed': 312501, 'steps': 1, 'max_abs': 0.0019857222796417773, 'mae': 0.00020260083657003887, 'rmse': 0.0004005860036722668} |
| `WARN` | `paired_velocity_difference_large` | {'seed': 312501, 'steps': 5, 'max_abs': 0.005683271505404264, 'mae': 0.0007121634352253005, 'rmse': 0.0014556658408934622} |
| `FAIL` | `latent_constraint_no_action_visible_leak` | {'seed': 312501, 'steps': 20, 'same_horizon': {'max_abs': 6.936490535736084e-05, 'mae': 1.1970599492390951e-05, 'rmse': 2.1966457359432857e-05, 'fraction_diff': 0.0, 'curve_diff': 6.1836940652894995e-06}, 'excess_motion': {'max_abs': 6.936490535736084e-05, 'mae': 1.1970599492390951e-05, 'rmse': 2.1966457359432857e-05}} |
| `WARN` | `paired_velocity_difference_large` | {'seed': 312502, 'steps': 20, 'max_abs': 0.001522718113847077, 'mae': 0.0002892831406820938, 'rmse': 0.00043516911953491577} |
| `WARN` | `paired_velocity_difference_large` | {'seed': 312503, 'steps': 5, 'max_abs': 0.0019132795569021255, 'mae': 0.00020820975538211114, 'rmse': 0.0003995852290540786} |
| `FAIL` | `latent_constraint_no_action_visible_leak` | {'seed': 312503, 'steps': 20, 'same_horizon': {'max_abs': 0.00017020106315612793, 'mae': 1.7751629153887432e-05, 'rmse': 3.776247622561682e-05, 'fraction_diff': 0.0, 'curve_diff': 1.1997190791186371e-05}, 'excess_motion': {'max_abs': 0.00017020106315612793, 'mae': 1.7751629153887432e-05, 'rmse': 3.776247622561682e-05}} |
| `WARN` | `paired_velocity_difference_large` | {'seed': 312503, 'steps': 20, 'max_abs': 0.007403461582725868, 'mae': 0.0007413028697137571, 'rmse': 0.0015212666683854495} |

## Decision

- A reproducibility FAIL means the threshold is below simulator repeatability.
- A common-evolution FAIL means condition-independent reset paths diverge.
- A paired-visible FAIL means the armed latent constraint leaks into no-action visible geometry.
- Only a report with no FAIL permits query-local candidate execution.
- No model training, Phase4, or CPS was run.
